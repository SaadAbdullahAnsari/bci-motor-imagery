from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from bci_mi.config import ALL_SUBJECTS_RESULTS_PATH, REPORT_DIR


ALL_SUBJECTS_ACCURACY_PLOT_PATH = REPORT_DIR / "all_subjects_accuracy.png"


def plot_all_subject_accuracies() -> None:
    """
    Plot subject-specific CSP + LDA accuracies from the benchmark CSV.
    """
    if not ALL_SUBJECTS_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find results file: {ALL_SUBJECTS_RESULTS_PATH}. "
            "Run `python -m bci_mi.benchmark` first."
        )

    results = pd.read_csv(ALL_SUBJECTS_RESULTS_PATH)

    mean_accuracy = results["accuracy"].mean()

    plt.figure(figsize=(8, 5))
    plt.bar(results["subject"].astype(str), results["accuracy"])
    plt.axhline(mean_accuracy, linestyle="--", label=f"Mean = {mean_accuracy:.3f}")
    plt.axhline(0.5, linestyle=":", label="Chance = 0.500")

    plt.xlabel("Subject")
    plt.ylabel("Accuracy")
    plt.title("CSP + LDA Accuracy Across Subjects")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    REPORT_DIR.mkdir(exist_ok=True)
    plt.savefig(ALL_SUBJECTS_ACCURACY_PLOT_PATH, dpi=200)
    plt.close()

    print(f"Saved all-subject accuracy plot to: {ALL_SUBJECTS_ACCURACY_PLOT_PATH}")


if __name__ == "__main__":
    plot_all_subject_accuracies()