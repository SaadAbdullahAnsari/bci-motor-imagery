from __future__ import annotations

import joblib
import pandas as pd
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from bci_mi.config import (
    MODEL_DIR,
    MODEL_PATH,
    N_CSP_COMPONENTS,
    RANDOM_STATE,
    REPORT_DIR,
    RESULTS_PATH,
)
from bci_mi.data import load_left_right_data


def build_pipeline() -> Pipeline:
    """
    Build a CSP + LDA motor imagery classifier.

    CSP extracts spatial EEG features that separate left-hand and right-hand
    imagery based on class-specific variance patterns.

    LDA then classifies those features.
    """
    csp = CSP(
        n_components=N_CSP_COMPONENTS,
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


def make_train_test_split(X, y, metadata):
    """
    Split EEG data into train and test sets.

    If the dataset has multiple sessions, use the last session as the test set.
    This is more realistic than randomly mixing all trials together.

    If only one session exists, fall back to a stratified random split.
    """
    if "session" in metadata.columns and metadata["session"].nunique() > 1:
        sessions = sorted(metadata["session"].unique())
        test_session = sessions[-1]

        test_mask = metadata["session"] == test_session
        train_mask = ~test_mask

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        print("Using session-based split.")
        print(f"Train sessions: {sorted(metadata.loc[train_mask, 'session'].unique())}")
        print(f"Test session: {test_session}")

        return X_train, X_test, y_train, y_test

    print("Only one session found. Using stratified random split.")

    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_model():
    """
    Load data, train the CSP + LDA classifier, evaluate it, and save outputs.
    """
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    print("Loading left-vs-right motor imagery data...")
    X, y, metadata = load_left_right_data()

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"metadata shape: {metadata.shape}")
    print(f"Labels: {sorted(set(y))}")

    X_train, X_test, y_train, y_test = make_train_test_split(X, y, metadata)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    clf = build_pipeline()

    print("Training CSP + LDA classifier...")
    clf.fit(X_train, y_train)

    print("Evaluating classifier...")
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    labels = sorted(set(y_test))

    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels))

    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print()
    print(f"Accuracy: {accuracy:.3f}")

    results = pd.DataFrame(
        [
            {
                "model": "CSP + LDA",
                "subject": 1,
                "accuracy": accuracy,
                "n_train_trials": len(y_train),
                "n_test_trials": len(y_test),
                "n_csp_components": N_CSP_COMPONENTS,
            }
        ]
    )

    results.to_csv(RESULTS_PATH, index=False)
    joblib.dump(clf, MODEL_PATH)

    print()
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved results to: {RESULTS_PATH}")

    return clf, results


if __name__ == "__main__":
    train_model()
