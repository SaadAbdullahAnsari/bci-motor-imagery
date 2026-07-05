# Left-vs-Right Motor Imagery BCI Classifier

This repository implements and compares baseline models for EEG-based left-vs-right motor imagery classification.

The project uses classical EEG decoding methods and a compact EEGNet-style neural network to classify motor imagery trials from the `BNCI2014_001` dataset through MOABB.


---

## Overview

The project currently compares:

1. **CSP + LDA**
   - Common Spatial Patterns for spatial EEG feature extraction.
   - Linear Discriminant Analysis for classification.
   - Tested with 2, 4, and 6 CSP components.

2. **EEGNetSmall**
   - A compact EEGNet-style convolutional neural network.
   - Trained subject-specifically.
   - Evaluated across 5 random seeds per subject.

The classification task is binary:

```text
left_hand vs right_hand
```

The evaluation is subject-specific:

```text
train on 0train session
test on 1test session
```

This avoids randomly mixing trials across sessions, which can make EEG results look better than they actually are.

---

## Dataset

The project uses:

```text
BNCI2014_001
```

Accessed through:

```text
MOABB
```

For each subject, the left/right motor imagery task provides:

```text
288 trials
22 EEG channels
1001 time samples per trial
```

For the current setup:

```text
144 training trials
144 held-out test trials
```

Subjects evaluated:

```text
1 to 9
```

Downloaded EEG data is not committed to GitHub. It is fetched locally through MOABB/MNE.

---

## Pipeline

The main classical pipeline is:

```text
EEG epochs
→ band-pass filtering, 8–30 Hz
→ CSP spatial filtering
→ LDA classification
→ held-out session evaluation
```

The EEGNet-style pipeline is:

```text
EEG epochs
→ standardisation using training data only
→ compact CNN
→ validation-based early stopping
→ held-out session evaluation
```

---

## Current Results

### All-subject model comparison

| Model | CSP components | Runs | Mean accuracy | Std accuracy |
|---|---:|---:|---:|---:|
| CSP + LDA | 2 | 9 | 0.715 | 0.172 |
| CSP + LDA | 4 | 9 | 0.729 | 0.177 |
| CSP + LDA | 6 | 9 | **0.747** | 0.144 |
| EEGNetSmall | N/A | 45 | 0.652 | 0.175 |

Current strongest average model:

```text
CSP + LDA with 6 components
Mean accuracy: 0.747
```

EEGNetSmall was less reliable overall:

```text
Mean accuracy: 0.652 across 45 runs
5 seeds × 9 subjects
```

EEGNetSmall performed strongly for some subjects but was unstable across seeds and weaker on average than CSP + LDA.

---

## Subject-Level Summary

| Subject | Best CSP components | Best CSP accuracy | EEGNet mean accuracy | EEGNet best seed accuracy |
|---:|---:|---:|---:|---:|
| 1 | 4 | 0.826 | 0.679 | 0.771 |
| 2 | 6 | 0.611 | 0.501 | 0.542 |
| 3 | 2 | 0.910 | **0.915** | **0.944** |
| 4 | 2 / 4 | 0.708 | 0.560 | 0.597 |
| 5 | 2 | 0.569 | 0.485 | 0.514 |
| 6 | 4 | 0.722 | 0.525 | 0.632 |
| 7 | 6 | 0.681 | 0.549 | 0.604 |
| 8 | 4 / 6 | 0.951 | 0.794 | 0.931 |
| 9 | 2 / 4 | 0.910 | 0.860 | **0.917** |

Main interpretation:

```text
CSP + LDA is the more reliable baseline.
EEGNetSmall can work well on some subjects, but is seed-sensitive and less stable.
The current best practical baseline is CSP + LDA, not EEGNetSmall.
```

This is not a failure of EEGNet as a method. It means this initial compact EEGNet-style implementation, trained subject-specifically on small data, is not yet competitive with CSP + LDA across all subjects.

---

## Repository Structure

```text
bci-motor-imagery/
├── src/
│   └── bci_mi/
│       ├── __init__.py
│       ├── benchmark.py
│       ├── compare_models.py
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── plot_benchmark.py
│       ├── plot_model_comparison.py
│       ├── plot_results.py
│       ├── summarise_comparison.py
│       ├── train.py
│       └── utils.py
│
├── tests/
│   └── test_imports.py
│
├── models/
│   └── .gitkeep
│
├── reports/
│   └── .gitkeep
│
├── notebooks/
│
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Create the environment

```bash
conda create -n bci-mi python=3.11 -y
conda activate bci-mi
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install the project in editable mode

```bash
pip install -e .
```

Editable mode allows imports like:

```python
from bci_mi.data import load_left_right_data
```

while still letting local code changes take effect immediately.

---

## Core Dependencies

The main packages used are:

| Package | Purpose |
|---|---|
| MNE-Python | EEG handling, signal processing, CSP |
| MOABB | Public BCI dataset access |
| scikit-learn | LDA, metrics, pipelines |
| PyTorch | EEGNetSmall neural network |
| NumPy | Numerical arrays |
| pandas | Results tables |
| matplotlib | Plotting |
| joblib | Saving/loading trained models |
| pytest | Tests |
| black | Formatting |
| ruff | Linting |

---

## Running the Project

### Run tests

```bash
pytest
```

### Load and inspect data

```bash
python -m bci_mi.data
```

Expected output includes:

```text
X shape: (288, 22, 1001)
y shape: (288,)
metadata shape: (288, 3)
Unique labels: ['left_hand', 'right_hand']
```

### Train the initial CSP + LDA model for subject 1

```bash
python -m bci_mi.train
```

This trains the default CSP + LDA model and saves local outputs to:

```text
models/
reports/
```

### Evaluate saved subject 1 model

```bash
python -m bci_mi.evaluate
```

### Plot subject 1 confusion matrix

```bash
python -m bci_mi.plot_results
```

---

## Benchmarking

### Run CSP + LDA across all subjects

```bash
python -m bci_mi.benchmark
```

This trains one subject-specific CSP + LDA model per subject and saves:

```text
reports/all_subjects_results.csv
```

### Plot all-subject CSP benchmark

```bash
python -m bci_mi.plot_benchmark
```

This creates:

```text
reports/all_subjects_accuracy.png
```

---

## Model Comparison

### Compare CSP settings and EEGNetSmall

```bash
python -m bci_mi.compare_models
```

This compares:

```text
CSP + LDA with 2 components
CSP + LDA with 4 components
CSP + LDA with 6 components
EEGNetSmall with seeds 0, 1, 2, 3, 4
```

Across:

```text
subjects 1 to 9
```

It saves:

```text
reports/model_comparison_all_subjects_results.csv
```

Expected number of rows:

```text
9 subjects × (3 CSP settings + 5 EEGNet seeds) = 72 rows
```

### Summarise and plot full comparison

```bash
python -m bci_mi.summarise_comparison
```

This creates:

```text
reports/model_comparison_summary.csv
reports/model_comparison_subject_summary.csv
reports/model_comparison_all_subjects.png
```

These files are generated locally and ignored by Git.

---

## Generated Files

The following outputs are intentionally ignored by Git:

```text
models/*.joblib
models/*.pkl
reports/*.csv
reports/*.png
*.mat
mne_data/
MNE-bnci-data/
C-/
```

---

## Reproducibility Notes

The CSP + LDA models are deterministic for the current setup.

EEGNetSmall is evaluated across multiple seeds because neural network performance can vary due to:

```text
random weight initialisation
batch order
validation split
early stopping behaviour
```

Current EEGNet seeds:

```text
0, 1, 2, 3, 4
```

The EEGNet results should therefore be interpreted using mean and standard deviation, not one single best fit cherry-picked run. 

---

## Current Interpretation

The current results suggest:

1. **CSP + LDA is the strongest and most reliable baseline.**
2. **6 CSP components gives the best average performance across subjects.**
3. **EEGNetSmall can perform well on some subjects, especially subjects 3, 8, and 9.**
4. **EEGNetSmall is more seed-sensitive and less stable than CSP + LDA.**
5. **For this small subject-specific dataset setup, classical decoding remains more dependable than the current neural model.**


---

## Limitations

Current limitations:

- Only left-vs-right motor imagery is used.
- Models are subject-specific.
- EEGNetSmall is a compact initial implementation, not a fully tuned deep-learning benchmark.
- No real-time EEG streaming is implemented.
- No cross-subject transfer learning is implemented.
- No filter-bank CSP is implemented yet.
- No Riemannian geometry classifier is implemented yet.

---

## Next Steps

Planned modelling improvements:

1. Add **Filter Bank CSP**.
2. Add **Riemannian covariance classifiers**.
3. Tune EEGNetSmall hyperparameters more systematically.
4. Add per-subject confusion matrices for the model comparison.
5. Compare within-session vs cross-session evaluation.
6. Add support for more MOABB motor imagery datasets.
7. Explore cross-subject and transfer-learning evaluation.
8. Eventually add a real-time inference loop using streaming EEG.

The sensible next modelling step is likely:

```text
Filter Bank CSP vs standard CSP
```
