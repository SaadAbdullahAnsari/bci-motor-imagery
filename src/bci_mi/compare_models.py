from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from bci_mi.config import (
    ALL_SUBJECTS,
    ALL_SUBJECTS_MODEL_COMPARISON_RESULTS_PATH,
    CSP_COMPONENT_OPTIONS,
    EEGNET_BATCH_SIZE,
    EEGNET_LEARNING_RATE,
    EEGNET_MAX_EPOCHS,
    EEGNET_PATIENCE,
    EEGNET_SEEDS,
    EEGNET_WEIGHT_DECAY,
    MODEL_COMPARISON_RESULTS_PATH,
    RANDOM_STATE,
    REPORT_DIR,
)
from bci_mi.data import load_left_right_data
from bci_mi.train import make_train_test_split


@dataclass
class EEGNetResult:
    accuracy: float
    balanced_accuracy: float
    best_val_accuracy: float
    epochs_trained: int


def set_reproducibility(seed: int = RANDOM_STATE) -> None:
    """
    Make random behaviour more reproducible.

    This does not make deep learning perfectly deterministic on every machine,
    but it reduces avoidable randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_csp_lda_pipeline(n_components: int) -> Pipeline:
    """
    Build CSP + LDA with a configurable number of CSP components.
    """
    csp = CSP(
        n_components=n_components,
        reg="ledoit_wolf",
        log=True,
        norm_trace=False,
    )

    lda = LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
    )

    return Pipeline(
        steps=[
            ("csp", csp),
            ("lda", lda),
        ]
    )


class EEGNetSmall(nn.Module):
    """
    Compact EEGNet-style convolutional neural network.

    Input shape expected:
        (batch, 1, n_channels, n_times)

    This uses:
    - temporal convolution
    - depthwise spatial convolution
    - separable temporal convolution
    - linear classifier
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        f1: int = 8,
        depth_multiplier: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        f2 = f1 * depth_multiplier

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, 64),
                padding=(0, 32),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                in_channels=f1,
                out_channels=f2,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                in_channels=f2,
                out_channels=f2,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=f2,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=f2,
                out_channels=f2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            n_features = self.features(dummy).reshape(1, -1).shape[1]

        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


def standardise_eeg(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardise EEG using only the training data statistics.

    This avoids leaking test-set information into preprocessing.
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std == 0, 1.0, std)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled


def encode_labels(
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Convert string labels into integer labels for PyTorch.
    """
    labels = sorted(set(y_train))
    label_to_index = {label: index for index, label in enumerate(labels)}

    y_train_encoded = np.array([label_to_index[label] for label in y_train])
    y_test_encoded = np.array([label_to_index[label] for label in y_test])

    return y_train_encoded, y_test_encoded, label_to_index


def make_tensor_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Convert EEG arrays into a PyTorch DataLoader.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_eegnet(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int = RANDOM_STATE,
) -> EEGNetResult:
    """
    Train and evaluate EEGNetSmall on one subject.
    """
    set_reproducibility(seed)

    X_train_scaled, X_test_scaled = standardise_eeg(X_train, X_test)
    y_train_encoded, y_test_encoded, _ = encode_labels(y_train, y_test)

    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train_scaled,
        y_train_encoded,
        test_size=0.2,
        random_state=seed,
        stratify=y_train_encoded,
    )

    train_loader = make_tensor_loader(
        X_train_part,
        y_train_part,
        batch_size=EEGNET_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = make_tensor_loader(
        X_val,
        y_val,
        batch_size=EEGNET_BATCH_SIZE,
        shuffle=False,
    )
    test_loader = make_tensor_loader(
        X_test_scaled,
        y_test_encoded,
        batch_size=EEGNET_BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_channels = X_train.shape[1]
    n_times = X_train.shape[2]
    n_classes = len(set(y_train_encoded))

    model = EEGNetSmall(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=EEGNET_LEARNING_RATE,
        weight_decay=EEGNET_WEIGHT_DECAY,
    )

    best_val_accuracy = 0.0
    best_state = None
    epochs_without_improvement = 0
    epochs_trained = 0

    for epoch in range(1, EEGNET_MAX_EPOCHS + 1):
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimiser.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimiser.step()

        val_accuracy = evaluate_torch_model(model, val_loader, device).accuracy

        print(f"Epoch {epoch:03d} | validation accuracy: {val_accuracy:.3f}")

        epochs_trained = epoch

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EEGNET_PATIENCE:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_torch_model(model, test_loader, device)

    return EEGNetResult(
        accuracy=test_metrics.accuracy,
        balanced_accuracy=test_metrics.balanced_accuracy,
        best_val_accuracy=best_val_accuracy,
        epochs_trained=epochs_trained,
    )


def evaluate_torch_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> EEGNetResult:
    """
    Evaluate a PyTorch classifier.
    """
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()

            predictions.extend(batch_predictions)
            targets.extend(y_batch.numpy())

    accuracy = accuracy_score(targets, predictions)
    balanced_accuracy = balanced_accuracy_score(targets, predictions)

    return EEGNetResult(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        best_val_accuracy=0.0,
        epochs_trained=0,
    )


def compare_subject(subject: int) -> list[dict]:
    """
    Compare CSP component settings and EEGNet for one subject.
    """
    print()
    print("=" * 72)
    print(f"Comparing models for subject {subject}")
    print("=" * 72)

    X, y, metadata = load_left_right_data(subjects=[subject])
    X_train, X_test, y_train, y_test = make_train_test_split(X, y, metadata)

    results = []

    for n_components in CSP_COMPONENT_OPTIONS:
        print()
        print(f"Training CSP + LDA with {n_components} components...")

        clf = build_csp_lda_pipeline(n_components=n_components)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

        print(f"CSP {n_components} accuracy: {accuracy:.3f}")
        print(f"CSP {n_components} balanced accuracy: {balanced_accuracy:.3f}")

        results.append(
            {
                "subject": subject,
                "model": "CSP + LDA",
                "n_csp_components": n_components,
                "seed": np.nan,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "best_val_accuracy": np.nan,
                "epochs_trained": np.nan,
            }
        )

    print()
    print("Training EEGNetSmall across multiple seeds...")

    for seed in EEGNET_SEEDS:
        print()
        print(f"Training EEGNetSmall with seed {seed}...")

        eegnet_result = evaluate_eegnet(
            X_train,
            X_test,
            y_train,
            y_test,
            seed=seed,
        )

        print(f"EEGNet seed {seed} accuracy: {eegnet_result.accuracy:.3f}")
        print(
            f"EEGNet seed {seed} balanced accuracy: "
            f"{eegnet_result.balanced_accuracy:.3f}"
        )

        results.append(
            {
                "subject": subject,
                "model": "EEGNetSmall",
                "n_csp_components": np.nan,
                "seed": seed,
                "accuracy": eegnet_result.accuracy,
                "balanced_accuracy": eegnet_result.balanced_accuracy,
                "best_val_accuracy": eegnet_result.best_val_accuracy,
                "epochs_trained": eegnet_result.epochs_trained,
            }
        )

    return results


def run_comparison(
    subjects: list[int] | None = None,
    output_path: Path = MODEL_COMPARISON_RESULTS_PATH,
) -> pd.DataFrame:
    """
    Run model comparison for selected subjects.

    For each subject, this compares:
    - CSP + LDA with different numbers of CSP components
    - EEGNetSmall across multiple random seeds
    """
    if subjects is None:
        subjects = [1]

    REPORT_DIR.mkdir(exist_ok=True)

    print()
    print("=" * 72)
    print("Running model comparison")
    print("=" * 72)
    print(f"Subjects: {subjects}")
    print(f"CSP component options: {CSP_COMPONENT_OPTIONS}")
    print(f"EEGNet seeds: {EEGNET_SEEDS}")
    print(f"Output path: {output_path}")

    all_results = []

    for subject in subjects:
        subject_results = compare_subject(subject)
        all_results.extend(subject_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)

    summary_df = (
        results_df.groupby(["model", "n_csp_components"], dropna=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            n_runs=("accuracy", "count"),
        )
        .reset_index()
    )

    subject_summary_df = (
        results_df.groupby(["subject", "model", "n_csp_components"], dropna=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            n_runs=("accuracy", "count"),
        )
        .reset_index()
    )

    print()
    print("=" * 72)
    print("Raw model comparison results")
    print("=" * 72)
    print(results_df)

    print()
    print("=" * 72)
    print("Summary by model")
    print("=" * 72)
    print(summary_df)

    print()
    print("=" * 72)
    print("Summary by subject and model")
    print("=" * 72)
    print(subject_summary_df)

    print()
    print(f"Saved comparison results to: {output_path}")

    return results_df


if __name__ == "__main__":
    run_comparison(
        subjects=ALL_SUBJECTS,
        output_path=ALL_SUBJECTS_MODEL_COMPARISON_RESULTS_PATH,
    )
