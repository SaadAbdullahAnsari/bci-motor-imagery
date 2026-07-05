from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from bci_mi.config import ALL_SUBJECTS, ALL_SUBJECTS_RESULTS_PATH, REPORT_DIR
from bci_mi.data import load_left_right_data
from bci_mi.train import build_pipeline, make_train_test_split


def evaluate_subject(subject: int) -> dict:
    """
    Train and evaluate one subject-specific CSP + LDA classifier.

    The model is trained on that subject's training session and tested on
    that subject's held-out evaluation session.
    """
    print()
    print("=" * 72)
    print(f"Evaluating subject {subject}")
    print("=" * 72)

    X, y, metadata = load_left_right_data(subjects=[subject])

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

    labels = sorted(set(y_test))
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = accuracy_score(y_test, y_pred)

    print()
    print("Confusion matrix:")
    print(matrix)

    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print()
    print(f"Subject {subject} accuracy: {accuracy:.3f}")

    sessions = sorted(metadata["session"].unique())
    test_session = sessions[-1]
    train_sessions = sorted(
        metadata.loc[metadata["session"] != test_session, "session"].unique()
    )

    return {
        "subject": subject,
        "model": "CSP + LDA",
        "accuracy": accuracy,
        "n_train_trials": len(y_train),
        "n_test_trials": len(y_test),
        "train_sessions": ",".join(train_sessions),
        "test_session": test_session,
    }


def run_all_subjects() -> pd.DataFrame:
    """
    Run subject-specific training and evaluation for all configured subjects.
    """
    REPORT_DIR.mkdir(exist_ok=True)

    results = []

    for subject in ALL_SUBJECTS:
        subject_result = evaluate_subject(subject)
        results.append(subject_result)

    results_df = pd.DataFrame(results)

    mean_accuracy = results_df["accuracy"].mean()
    std_accuracy = results_df["accuracy"].std()

    print()
    print("=" * 72)
    print("All-subject summary")
    print("=" * 72)
    print(results_df)
    print()
    print(f"Mean accuracy: {mean_accuracy:.3f}")
    print(f"Std accuracy:  {std_accuracy:.3f}")

    results_df.to_csv(ALL_SUBJECTS_RESULTS_PATH, index=False)

    print()
    print(f"Saved all-subject results to: {ALL_SUBJECTS_RESULTS_PATH}")

    return results_df


if __name__ == "__main__":
    run_all_subjects()
