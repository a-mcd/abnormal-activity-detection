#!/usr/bin/env python3
"""Rank features by how well they tell the ACTIVITIES apart (t1-t5).

Scores each engineered feature with mutual information against the activity
label. MODE chooses the sessions: "noerror", "error" or "combined".

Reads:  results/activity_classification/features/engineered_features/
        (from scripts/activity_classification/features/build_features.py)
Writes: results/activity_classification/features/mutual_info/mi_activity_<MODE>.csv

Not the same as scripts/abnormal_detection/features/mutual_info.py, which
ranks features by how well they separate normal from errored sessions
within one activity.
"""

import pandas as pd
from pathlib import Path

from abnormal_activity_detection.paths import RESULTS_DIR
from sklearn.feature_selection import mutual_info_classif

# =========================================================
# CONFIG
# =========================================================
TOP_K = 60

BASE_DIR = RESULTS_DIR / "activity_classification" / "features"

FEATURES_DIR = BASE_DIR / "engineered_features"

NOERROR_PATH = FEATURES_DIR / "noerror_all_features.csv"
ERROR_PATH   = FEATURES_DIR / "error_all_features.csv"

OUTPUT_DIR = BASE_DIR / "mutual_info"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose one:
#   "noerror"  -> rank features for classifying t1–t5 using normal episodes only
#   "error"    -> rank features for classifying t1–t5 using error episodes only
#   "combined" -> rank features for classifying t1–t5 using both together
MODE = "error"

# Use activity_code or activity_index as target
TARGET_COL = "activity_code"

# Optional: keep only a subset of activities, e.g. ["t1", "t2", "t3"]
ACTIVITIES_TO_KEEP = None

# If True, remove one-hot bucket groups that are constant after filtering
REMOVE_CONSTANT_COLUMNS = True

RANDOM_STATE = 42


# =========================================================
# HELPERS
# =========================================================
def load_dataset(mode: str) -> pd.DataFrame:
    if mode == "noerror":
        df = pd.read_csv(NOERROR_PATH).copy()
    elif mode == "error":
        df = pd.read_csv(ERROR_PATH).copy()
    elif mode == "combined":
        df_noerror = pd.read_csv(NOERROR_PATH).copy()
        df_error = pd.read_csv(ERROR_PATH).copy()
        df = pd.concat([df_noerror, df_error], ignore_index=True)
    else:
        raise ValueError("MODE must be one of: 'noerror', 'error', 'combined'")
    return df


def prepare_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    y = df[target_col].copy()

    if target_col == "activity_code":
        y = y.astype(str)
        y = pd.Categorical(y, categories=sorted(y.dropna().unique()))
        return pd.Series(y.codes, index=df.index, name=target_col)

    return y


# =========================================================
# LOAD DATA
# =========================================================
df = load_dataset(MODE)

print(f"\nLoaded mode: {MODE}")
print(f"Initial dataset size: {len(df)}")

if ACTIVITIES_TO_KEEP is not None:
    df = df[df["activity_code"].isin(ACTIVITIES_TO_KEEP)].copy()
    print(f"Dataset size after activity filter: {len(df)}")

if TARGET_COL not in df.columns:
    raise ValueError(f"'{TARGET_COL}' not found in CSV columns.")

# Drop rows with missing target
df = df.dropna(subset=[TARGET_COL]).copy()

print("\nActivity counts:")
print(df["activity_code"].value_counts().sort_index())

# =========================================================
# PREPARE X / y
# =========================================================
y = prepare_target(df, TARGET_COL)

# Drop non-feature columns
# Important:
# - remove activity_code/activity_index because they are the target/leakage
# - remove dataset because it is metadata, not a true sensor-derived feature
# - remove filename / participant_id because they are identifiers
non_feature_cols = [
    "filename",
    "participant_id",
    "activity_code",
    "activity_index",
    "dataset",
    "label",
]

X = df.drop(columns=[c for c in non_feature_cols if c in df.columns]).copy()

# Fill missing values
X = X.fillna(0)

# Keep only numeric columns
X = X.select_dtypes(include=["number", "bool"]).copy()

# Convert bool to int
for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(int)

# Remove constant columns
if REMOVE_CONSTANT_COLUMNS:
    X = X.loc[:, X.nunique(dropna=False) > 1].copy()

print(f"\nNumber of features used: {X.shape[1]}")

if X.shape[1] == 0:
    raise RuntimeError("No usable feature columns remained after filtering.")

# =========================================================
# MUTUAL INFORMATION
# =========================================================
mi = mutual_info_classif(
    X,
    y,
    random_state=RANDOM_STATE,
)

mi_scores = pd.Series(mi, index=X.columns, name="mi_score").sort_values(ascending=False)

mi_df = mi_scores.reset_index()
mi_df.columns = ["feature", "mi_score"]

# =========================================================
# OUTPUT
# =========================================================
print(f"\nTop {TOP_K} features for activity classification ({MODE}):\n")
print(mi_df.head(TOP_K).to_string(index=False))

output_path = OUTPUT_DIR / f"mi_activity_{MODE}.csv"
mi_df.to_csv(output_path, index=False)

print(f"\nSaved MI scores to: {output_path}")