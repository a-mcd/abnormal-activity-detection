"""Rank features by how well they separate NORMAL from ERRORED sessions within one activity.

Labels no-error sessions 0 and error sessions 1, then scores each engineered
feature with mutual information against that label.

Settings: TARGET_ACTIVITY_CODE in abnormal_activity_detection/anomaly/configs/features.py
Reads:    results/abnormal_detection/features/<activity>/engineered_features/
          (from scripts/abnormal_detection/features/build_features.py)
Writes:   results/abnormal_detection/features/<activity>/mutual_info/mi_scores_<activity>.csv

Not the same as scripts/activity_classification/features/mutual_info.py, which
ranks features by how well they tell the five activities apart.
"""

import pandas as pd
import re
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path
from abnormal_activity_detection.anomaly.configs.features import TARGET_ACTIVITY_CODE
from abnormal_activity_detection.paths import RESULTS_DIR

# =========================================================
# CONFIG
# =========================================================
TOP_K = 40              # Number of top features to display


BASE_DIR = RESULTS_DIR / "abnormal_detection" / "features"

if TARGET_ACTIVITY_CODE is not None:
    features_dir = BASE_DIR / TARGET_ACTIVITY_CODE / "engineered_features"
    error_path = features_dir / f"{TARGET_ACTIVITY_CODE}_error_all_features.csv"
    noerror_path = features_dir / f"{TARGET_ACTIVITY_CODE}_noerror_all_features.csv"
    output_dir = BASE_DIR / TARGET_ACTIVITY_CODE / "mutual_info"
    output_path = output_dir / f"mi_scores_{TARGET_ACTIVITY_CODE}.csv"
else:
    features_dir = BASE_DIR / "all_activities" / "engineered_features"
    error_path = features_dir / "all_activities_error_all_features.csv"
    noerror_path = features_dir / "all_activities_noerror_all_features.csv"
    output_dir = BASE_DIR / "all_activities" / "mutual_info"
    output_path = output_dir / "mi_scores_all_activities.csv"


# =========================================================
# HELPERS
# =========================================================
def extract_activity(filename):
    """
    Extract activity code (t1–t5) from filename.
    Example: p01.t3.csv -> t3
    """
    match = re.search(r"(t[1-5])", str(filename), re.IGNORECASE)
    return match.group(1).lower() if match else None


# =========================================================
# LOAD DATA
# =========================================================
df_error = pd.read_csv(error_path)
df_noerror = pd.read_csv(noerror_path)

# Add labels
df_error["label"] = 1      # abnormal
df_noerror["label"] = 0    # normal

# Combine
df = pd.concat([df_error, df_noerror], ignore_index=True)

# =========================================================
# ADD ACTIVITY COLUMN
# =========================================================
df["activity"] = df["filename"].apply(extract_activity)

# Drop rows where activity couldn't be parsed
df = df.dropna(subset=["activity"])

# =========================================================
# FILTER BY ACTIVITY
# =========================================================
if TARGET_ACTIVITY_CODE is not None:
    df = df[df["activity"] == TARGET_ACTIVITY_CODE].copy()

print(f"\nDataset size after filtering: {len(df)}")
print(df["label"].value_counts())

# =========================================================
# PREPARE FEATURES
# =========================================================
# Drop non-feature columns
non_feature_cols = ["filename", "participant_id", "activity_code", "activity_index", "dataset", "label"]


X = df.drop(columns=[c for c in non_feature_cols if c in df.columns])
y = df["label"]

# Handle missing values
X = X.fillna(0)

# Optional: remove constant columns (important!)
X = X.loc[:, X.nunique() > 1]

print(f"\nNumber of features used: {X.shape[1]}")

# =========================================================
# MUTUAL INFORMATION
# =========================================================
mi = mutual_info_classif(X, y, random_state=42)

mi_scores = pd.Series(mi, index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

# =========================================================
# OUTPUT RESULTS
# =========================================================
print("\nTop features (Mutual Information):\n")
print(mi_scores.head(TOP_K))

# Save results
output_dir.mkdir(parents=True, exist_ok=True)
mi_scores.to_csv(output_path)

print(f"\nSaved MI scores to: {output_path}")