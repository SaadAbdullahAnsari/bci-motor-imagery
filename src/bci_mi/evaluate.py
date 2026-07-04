from __future__ import annotations

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from bci_mi.config import MODEL_PATH
from bci_mi.data import load_left_right_data
from bci_mi.train import make_train_test_split


def evaluate_saved_model() -> None:
    """
    Load a saved model and evaluate it on the same held-out test split.
    """
    print(f"Loading saved model from: {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)

    print("Loading data...")
    X, y, metadata = load_left_right_data()

    _, X_test, _, y_test = make_train_test_split(X, y, metadata)

    print("Running predictions...")
    y_pred = clf.predict(X_test)

    labels = sorted(set(y_test))
    accuracy = accuracy_score(y_test, y_pred)

    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels))

    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print()
    print(f"Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    evaluate_saved_model()
