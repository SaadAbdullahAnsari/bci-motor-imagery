from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bci_mi.config import MODEL_COMPARISON_RESULTS_PATH, REPORT_DIR

MODEL_COMPARISON_PLOT_PATH = REPORT_DIR / "model_comparison_subject1.png"


def summarise_model_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise CSP component results and EEGNet multi-seed results.

    CSP rows have one result per component setting.
    EEGNet rows have one result per seed, so we aggregate them.
    """
    csp_results = results[results["model"] == "CSP + LDA"].copy()
    csp_results["label"] = (
        "CSP " + csp_results["n_csp_components"].astype(int).astype(str) + " components"
    )
    csp_summary = csp_results[["label", "accuracy", "balanced_accuracy"]].rename(
        columns={
            "accuracy": "mean_accuracy",
            "balanced_accuracy": "mean_balanced_accuracy",
        }
    )
    csp_summary["std_accuracy"] = 0.0

    eegnet_results = results[results["model"] == "EEGNetSmall"].copy()
    eegnet_summary = pd.DataFrame(
        [
            {
                "label": "EEGNetSmall\nmean over seeds",
                "mean_accuracy": eegnet_results["accuracy"].mean(),
                "mean_balanced_accuracy": eegnet_results["balanced_accuracy"].mean(),
                "std_accuracy": eegnet_results["accuracy"].std(),
            }
        ]
    )

    return pd.concat([csp_summary, eegnet_summary], ignore_index=True)


def plot_model_comparison() -> None:
    """
    Plot CSP component settings against EEGNetSmall multi-seed performance.
    """
    if not MODEL_COMPARISON_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MODEL_COMPARISON_RESULTS_PATH}. "
            "Run `python -m bci_mi.compare_models` first."
        )

    REPORT_DIR.mkdir(exist_ok=True)

    results = pd.read_csv(MODEL_COMPARISON_RESULTS_PATH)
    summary = summarise_model_comparison(results)

    x_positions = np.arange(len(summary))

    plt.figure(figsize=(9, 5))
    plt.bar(
        x_positions,
        summary["mean_accuracy"],
        yerr=summary["std_accuracy"],
        capsize=5,
    )
    plt.axhline(0.5, linestyle=":", label="Chance = 0.500")

    plt.xticks(x_positions, summary["label"])
    plt.ylabel("Accuracy")
    plt.title("Subject 1 Model Comparison")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(MODEL_COMPARISON_PLOT_PATH, dpi=200)
    plt.close()

    print("Model comparison summary:")
    print(summary)
    print()
    print(f"Saved plot to: {MODEL_COMPARISON_PLOT_PATH}")


if __name__ == "__main__":
    plot_model_comparison()
