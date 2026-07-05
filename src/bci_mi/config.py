from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
ALL_SUBJECTS_RESULTS_PATH = REPORT_DIR / "all_subjects_results.csv"

RANDOM_STATE = 42

# Motor imagery rhythm range.
# 8-30 Hz roughly covers mu and beta rhythms commonly used in MI-BCI.
FMIN = 8
FMAX = 30

# Start with one subject only.
SUBJECTS = [1]
ALL_SUBJECTS = list(range(1, 10))

DATASET_NAME = "BNCI2014_001"

# CSP settings.
N_CSP_COMPONENTS = 6

MODEL_PATH = MODEL_DIR / "csp_lda_subject1.joblib"
RESULTS_PATH = REPORT_DIR / "results_subject1.csv"
