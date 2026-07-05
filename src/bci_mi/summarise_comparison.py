from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bci_mi.config import (
    ALL_SUBJECTS_MODEL_COMPARISON_RESULTS_PATH,
    MODEL_COMPARISON_ALL_SUBJECTS_PLOT_PATH,
    MODEL_COMPARISON_SUBJECT_SUMMARY_PATH,
    MODEL_COMPARISON_SUMMARY_PATH,
    REPORT_DIR,
)


def load_full_comparison_results() -> pd.DataFrame:
    """
    Load the all-subject model comparison CSV.
    """
    if not ALL_SUBJECTS_MODEL_COMPARISON_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {ALL_SUBJECTS_MODEL_COMPARISON_RESULTS_PATH}. "
            "Run `python -m bci_mi.compare_models` first."
        )

    return pd.read_csv(ALL_SUBJECTS_MODEL_COMPARISON_RESULTS_PATH)


def build_model_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Build overall summary by model and CSP component setting.

    For CSP, each row is one subject/component result.
    For EEGNet, rows include all seeds across all subjects.
    """
    summary = (
        results.groupby(["model", "n_csp_components"], dropna=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            n_runs=("accuracy", "count"),
        )
        .reset_index()
    )

    return summary


def build_subject_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Build subject-level summary comparing best CSP against EEGNet mean/best seed.
    """
    csp_results = results[results["model"] == "CSP + LDA"].copy()
    eegnet_results = results[results["model"] == "EEGNetSmall"].copy()

    best_csp = csp_results.loc[
        csp_results.groupby("subject")["accuracy"].idxmax()
    ].copy()

    best_csp = best_csp[
        ["subject", "n_csp_components", "accuracy", "balanced_accuracy"]
    ].rename(
        columns={
            "n_csp_components": "best_csp_components",
            "accuracy": "best_csp_accuracy",
            "balanced_accuracy": "best_csp_balanced_accuracy",
        }
    )

    eegnet_summary = (
        eegnet_results.groupby("subject")
        .agg(
            eegnet_mean_accuracy=("accuracy", "mean"),
            eegnet_std_accuracy=("accuracy", "std"),
            eegnet_best_accuracy=("accuracy", "max"),
            eegnet_worst_accuracy=("accuracy", "min"),
            eegnet_mean_balanced_accuracy=("balanced_accuracy", "mean"),
        )
        .reset_index()
    )

    subject_summary = best_csp.merge(eegnet_summary, on="subject")

    subject_summary["best_method_by_mean"] = np.where(
        subject_summary["eegnet_mean_accuracy"] > subject_summary["best_csp_accuracy"],
        "EEGNetSmall mean",
        "Best CSP + LDA",
    )

    subject_summary["best_method_by_best_seed"] = np.where(
        subject_summary["eegnet_best_accuracy"] > subject_summary["best_csp_accuracy"],
        "EEGNetSmall best seed",
        "Best CSP + LDA",
    )

    return subject_summary


def plot_full_comparison(subject_summary: pd.DataFrame) -> None:
    """
    Plot best CSP accuracy vs EEGNet mean and best seed for each subject.
    """
    x_positions = np.arange(len(subject_summary))
    width = 0.25

    plt.figure(figsize=(11, 5))

    plt.bar(
        x_positions - width,
        subject_summary["best_csp_accuracy"],
        width=width,
        label="Best CSP + LDA",
    )

    plt.bar(
        x_positions,
        subject_summary["eegnet_mean_accuracy"],
        width=width,
        yerr=subject_summary["eegnet_std_accuracy"],
        capsize=4,
        label="EEGNetSmall mean ± std",
    )

    plt.bar(
        x_positions + width,
        subject_summary["eegnet_best_accuracy"],
        width=width,
        label="EEGNetSmall best seed",
    )

    plt.axhline(0.5, linestyle=":", label="Chance = 0.500")

    plt.xticks(x_positions, subject_summary["subject"].astype(str))
    plt.xlabel("Subject")
    plt.ylabel("Accuracy")
    plt.title("All-Subject Model Comparison")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(MODEL_COMPARISON_ALL_SUBJECTS_PLOT_PATH, dpi=200)
    plt.close()


def main() -> None:
    """
    Summarise and plot the full model comparison.
    """
    REPORT_DIR.mkdir(exist_ok=True)

    results = load_full_comparison_results()

    model_summary = build_model_summary(results)
    subject_summary = build_subject_summary(results)

    model_summary.to_csv(MODEL_COMPARISON_SUMMARY_PATH, index=False)
    subject_summary.to_csv(MODEL_COMPARISON_SUBJECT_SUMMARY_PATH, index=False)

    plot_full_comparison(subject_summary)

    print()
    print("=" * 72)
    print("Overall model summary")
    print("=" * 72)
    print(model_summary)

    print()
    print("=" * 72)
    print("Subject-level summary")
    print("=" * 72)
    print(subject_summary)

    print()
    print(f"Saved model summary to: {MODEL_COMPARISON_SUMMARY_PATH}")
    print(f"Saved subject summary to: {MODEL_COMPARISON_SUBJECT_SUMMARY_PATH}")
    print(f"Saved plot to: {MODEL_COMPARISON_ALL_SUBJECTS_PLOT_PATH}")


if __name__ == "__main__":
    main()
