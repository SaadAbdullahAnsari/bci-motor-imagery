from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42

# Motor imagery rhythm range.
# This captures much of the mu and beta rhythm range used in MI-BCI work.
FMIN = 8
FMAX = 30

# Start with one subject only.
# Do not immediately use all subjects unless you enjoy confusing yourself.
SUBJECTS = [1]

# Dataset name we are starting with.
DATASET_NAME = "BNCI2014_001"
