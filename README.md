# Left-vs-Right Motor Imagery BCI Classifier

This project implements a baseline EEG motor imagery classifier for left-hand vs right-hand imagined movement.

The first model uses a classical BCI pipeline:

EEG epochs → band-pass filtering → CSP spatial filtering → LDA classification → evaluation on held-out session

## Dataset

The project uses the `BNCI2014_001` dataset through MOABB.

For the first baseline, the project uses subject 1 and restricts the task to binary left-hand vs right-hand motor imagery.

## Current baseline

Model:

```text
CSP + LDA
```

Current subject 1 held-out session performance:

```text
Accuracy: 0.806
```

All-subject mean accuracy:
```text
Accuracy: 0.747
```


Confusion matrix:

```text
[[72  0]
 [28 44]]
```

Interpretation:

- The model correctly classified all left-hand trials.
- The model missed some right-hand trials.
- The baseline is useful, but class performance is uneven.

## Setup

Create and activate the environment:

```bash
conda create -n bci-mi python=3.11 -y
conda activate bci-mi
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

## Run tests

```bash
pytest
```

## Load data

```bash
python -m bci_mi.data
```

## Train model

```bash
python -m bci_mi.train
```

## Evaluate saved model

```bash
python -m bci_mi.evaluate
```
## Evaluate all subjects

```bash
python -m bci_mi.benchmark
```

This trains and evaluates one subject-specific CSP + LDA model per subject, then saves a summary table to:

```text
reports/all_subjects_results.csv
```

## Plot all-subject benchmark results

```bash
python -m bci_mi.plot_benchmark
```

This creates:

```text
reports/all_subjects_accuracy.png
```

Current all-subject CSP + LDA benchmark:

```text
Mean accuracy: 0.747
Best subject: 8, accuracy 0.951
Worst subject: 5, accuracy 0.563
```

## Plot confusion matrix

```bash
python -m bci_mi.plot_results
```

## Project structure

```text
bci-motor-imagery/
├── src/
│   └── bci_mi/
│       ├── config.py
│       ├── data.py
│       ├── train.py
│       ├── evaluate.py
│       └── plot_results.py
├── tests/
├── models/
├── reports/
├── notebooks/
├── requirements.txt
└── pyproject.toml
```

## Notes

Generated model files, downloaded EEG data, and local result outputs are intentionally ignored by Git.

This keeps the repository lightweight and reproducible instead of turning GitHub into a dumping ground for cache files and scientific debris.

## Next steps

Planned improvements:

- Compare subject-specific performance across all 9 subjects.
- Add filter-bank CSP.
- Add Riemannian geometry classifier.
- Compare against deeper models only after strong classical baselines.