from pathlib import Path

# This file lives at <project root>/src/abnormal_activity_detection/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

NORMAL_SUBDIR = "adl_noerror_clean"
ERROR_SUBDIR = "adl_error_clean"


def get_dataset_dirs():
    """Return (cleaned no-error data folder, cleaned error data folder)."""
    return PROCESSED_DATA_DIR / NORMAL_SUBDIR, PROCESSED_DATA_DIR / ERROR_SUBDIR


def get_results_dir(*parts: str) -> Path:
    """Return results/<parts...>, creating it if needed.

    Example: get_results_dir("activity_classification", "features", "engineered_features")
    """
    output_dir = RESULTS_DIR.joinpath(*parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

