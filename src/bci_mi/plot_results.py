from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from bci_mi.config import MODEL_PATH, REPORT_DIR
from bci_mi.data import load_left_right_data
from bci_mi.train import make_train_test_split

CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix_subject1.png"


def plot_confusion_matrix() -> None:
    """
    Load the saved CSP + LDA model and save a confusion matrix plot.
    """
    REPORT_DIR.mkdir(exist_ok=True)

    print(f"Loading saved model from: {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)

    print("Loading data...")
    X, y, metadata = load_left_right_data()

    _, X_test, _, y_test = make_train_test_split(X, y, metadata)

    print("Running predictions...")
    y_pred = clf.predict(X_test)

    labels = np.array(sorted(set(y_test)))
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = accuracy_score(y_test, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )

    display.plot(values_format="d")
    plt.title(f"CSP + LDA Confusion Matrix | Accuracy = {accuracy:.3f}")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()

    print(f"Saved confusion matrix plot to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    plot_confusion_matrix()
