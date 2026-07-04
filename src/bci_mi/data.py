from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery

from bci_mi.config import FMAX, FMIN, SUBJECTS


def load_left_right_data(
    subjects: list[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load left-vs-right motor imagery EEG epochs.

    Parameters
    ----------
    subjects:
        List of subject IDs to load. If None, uses SUBJECTS from config.py.

    Returns
    -------
    X:
        EEG data with shape:
        (n_trials, n_channels, n_times)

    y:
        Labels for each trial, usually left_hand/right_hand.

    metadata:
        Table describing each trial, including subject/session/run information.
    """
    if subjects is None:
        subjects = SUBJECTS

    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=FMIN, fmax=FMAX)

    X, y, metadata = paradigm.get_data(
        dataset=dataset,
        subjects=subjects,
    )

    return X, y, metadata


def describe_data(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame) -> None:
    """
    Print a basic summary of the loaded EEG data.
    """
    print("Data loaded successfully.")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"metadata shape: {metadata.shape}")
    print()
    print("Unique labels:")
    print(sorted(set(y)))
    print()
    print("Metadata columns:")
    print(list(metadata.columns))
    print()
    print("First few metadata rows:")
    print(metadata.head())


if __name__ == "__main__":
    X_loaded, y_loaded, metadata_loaded = load_left_right_data()
    describe_data(X_loaded, y_loaded, metadata_loaded)
