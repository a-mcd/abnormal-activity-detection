import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from abnormal_activity_detection.paths import RESULTS_DIR

# =========================================================
# CONFIG
# =========================================================
TARGET_ACTIVITY = "t2"   # Set to None for all activities


# Engineered features come from scripts/abnormal_detection/features/build_features.py
feature_label = TARGET_ACTIVITY or "all_activities"
features_dir = (
    RESULTS_DIR / "abnormal_detection" / "features" / feature_label / "engineered_features"
)
error_path = features_dir / f"{feature_label}_error_all_features.csv"
noerror_path = features_dir / f"{feature_label}_noerror_all_features.csv"

output_base_dir = (
    RESULTS_DIR / "abnormal_detection" / "isolation_forest"
    / (TARGET_ACTIVITY or "t-all") / "feature_importance"
)


TOP_K = 40
N_REPEATS = 15
RANDOM_STATE = 42

IFOREST_CONFIG = {
    "n_estimators": 300,
    "max_samples": "auto",
    "contamination": "auto",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}

# =========================================================
# HELPERS
# =========================================================
def extract_activity(filename: str):
    """
    Extract activity code t1..t5 from source_file.
    Example: p01.t3.csv -> t3
    """
    m = re.search(r"(t[1-5])", str(filename), re.IGNORECASE)
    return m.group(1).lower() if m else None


def safe_fill_and_filter(X: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values, keep numeric columns, remove constant columns.
    """
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    numeric_cols = X.select_dtypes(include=[np.number, bool]).columns.tolist()
    X = X[numeric_cols].copy()

    if not X.empty:
        X = X.loc[:, X.nunique(dropna=False) > 1]

    return X


def print_top_features(title: str, scores: pd.Series, top_k: int = 20):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    if scores.empty:
        print("No feature importances available.")
    else:
        print(scores.head(top_k).to_string())


def anomaly_score_from_model(estimator, X):
    """
    Convert Isolation Forest decision_function output so that:
    higher = more anomalous
    """
    return -estimator.decision_function(X)


def anomaly_score_similarity_scorer(estimator, X, y_true):
    """
    Custom scoring function for permutation_importance.

    y_true here is the reference anomaly score vector.
    We compare the model's anomaly scores on the permuted X
    against the original anomaly scores using negative MSE.
    Higher score = better match.
    """
    y_pred = anomaly_score_from_model(estimator, X)
    return -mean_squared_error(y_true, y_pred)


def compute_permutation_importance(
    model,
    X_array,
    y_target,
    feature_names,
    n_repeats=10,
    random_state=42,
):
    """
    Compute permutation importance using a custom scorer.
    """
    if len(feature_names) == 0 or len(y_target) == 0 or len(X_array) == 0:
        return pd.Series(dtype=float)

    result = permutation_importance(
        estimator=model,
        X=X_array,
        y=y_target,
        scoring=anomaly_score_similarity_scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )

    scores = pd.Series(result.importances_mean, index=feature_names)
    scores = scores.sort_values(ascending=False)
    return scores


# =========================================================
# LOAD DATA
# =========================================================
df_error = pd.read_csv(error_path)
df_noerror = pd.read_csv(noerror_path)

df_error["label"] = 1      # abnormal
df_noerror["label"] = 0    # normal

df = pd.concat([df_error, df_noerror], ignore_index=True)

if "filename" not in df.columns:
    raise ValueError("Expected a 'filename' column in the input CSVs.")

df["activity"] = df["filename"].apply(extract_activity)
df = df.dropna(subset=["activity"]).copy()

if TARGET_ACTIVITY is not None:
    df = df[df["activity"] == TARGET_ACTIVITY].copy()

if df.empty:
    raise ValueError(f"No rows found for TARGET_ACTIVITY={TARGET_ACTIVITY!r}")

print("\nDataset after filtering")
print(df["label"].value_counts(dropna=False))
print(f"Rows: {len(df)}")

# =========================================================
# PREPARE FEATURES
# =========================================================
non_feature_cols = [
    "source_file",
    "filename",
    "participant_id",
    "activity_code",
    "activity_index",
    "dataset",
    "label",
    "activity",
]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

X = df[feature_cols].copy()
y = df["label"].astype(int).copy()

X = safe_fill_and_filter(X)

if X.empty:
    raise ValueError("No usable numeric feature columns remain after filtering.")

feature_names = X.columns.tolist()

print(f"\nNumber of usable numeric features: {len(feature_names)}")

# Split into normal / error
X_normal = X[y == 0].copy()
X_error = X[y == 1].copy()

if X_normal.empty:
    raise ValueError("No normal rows found. Isolation Forest must be trained on normal rows.")
if X_error.empty:
    print("\nWarning: No error rows found. Only normal/global importance will be computed.")

print(f"Normal rows: {len(X_normal)}")
print(f"Error rows : {len(X_error)}")

# =========================================================
# SCALE
# =========================================================
scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)
X_all_scaled = scaler.transform(X)

X_error_scaled = None
if not X_error.empty:
    X_error_scaled = scaler.transform(X_error)

# =========================================================
# TRAIN ISOLATION FOREST ON NORMAL ONLY
# =========================================================
model = IsolationForest(**IFOREST_CONFIG)
model.fit(X_normal_scaled)

# =========================================================
# ANOMALY SCORES
# =========================================================
all_anomaly_scores = anomaly_score_from_model(model, X_all_scaled)
normal_anomaly_scores = anomaly_score_from_model(model, X_normal_scaled)

error_anomaly_scores = None
if X_error_scaled is not None:
    error_anomaly_scores = anomaly_score_from_model(model, X_error_scaled)

# Save scores with metadata
score_df = pd.DataFrame({
    "filename": df["filename"].values,
    "activity": df["activity"].values,
    "label": y.values,
    "anomaly_score": all_anomaly_scores,
}).sort_values("anomaly_score", ascending=False)

print("\nTop 10 most anomalous episodes:")
print(score_df.head(10).to_string(index=False))

# =========================================================
# PERMUTATION IMPORTANCE
# =========================================================
print("\nCalculating GLOBAL permutation importance...")
global_importance = compute_permutation_importance(
    model=model,
    X_array=X_all_scaled,
    y_target=all_anomaly_scores,
    feature_names=feature_names,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
)

print("Calculating NORMAL-ONLY permutation importance...")
normal_importance = compute_permutation_importance(
    model=model,
    X_array=X_normal_scaled,
    y_target=normal_anomaly_scores,
    feature_names=feature_names,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
)

error_importance = pd.Series(dtype=float)
if X_error_scaled is not None and len(X_error) > 1:
    print("Calculating ERROR-ONLY permutation importance...")
    error_importance = compute_permutation_importance(
        model=model,
        X_array=X_error_scaled,
        y_target=error_anomaly_scores,
        feature_names=feature_names,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
else:
    print("Skipping ERROR-ONLY permutation importance (not enough error rows).")

# =========================================================
# COMBINE RESULTS
# =========================================================
importance_df = pd.DataFrame({
    "feature": feature_names
})

importance_df["global_importance"] = importance_df["feature"].map(global_importance).fillna(0.0)
importance_df["normal_importance"] = importance_df["feature"].map(normal_importance).fillna(0.0)
importance_df["error_importance"] = importance_df["feature"].map(error_importance).fillna(0.0)

importance_df["mean_importance"] = importance_df[
    ["global_importance", "normal_importance", "error_importance"]
].mean(axis=1)

importance_df["max_importance"] = importance_df[
    ["global_importance", "normal_importance", "error_importance"]
].max(axis=1)

importance_df = importance_df.sort_values(
    ["mean_importance", "max_importance"],
    ascending=False
).reset_index(drop=True)

# =========================================================
# PRINT RESULTS
# =========================================================
print_top_features("TOP GLOBAL FEATURE IMPORTANCE", global_importance, TOP_K)
print_top_features("TOP NORMAL-ONLY FEATURE IMPORTANCE", normal_importance, TOP_K)
print_top_features("TOP ERROR-ONLY FEATURE IMPORTANCE", error_importance, TOP_K)

print(f"\n{'=' * 70}")
print("TOP COMBINED FEATURE IMPORTANCE")
print(f"{'=' * 70}")
print(
    importance_df[
        ["feature", "global_importance", "normal_importance", "error_importance", "mean_importance"]
    ]
    .head(TOP_K)
    .to_string(index=False)
)

# =========================================================
# SAVE OUTPUTS
# =========================================================
activity_suffix = TARGET_ACTIVITY if TARGET_ACTIVITY is not None else "all"
output_base_dir.mkdir(parents=True, exist_ok=True)

out_scores = output_base_dir / f"iforest_anomaly_scores_{activity_suffix}.csv"
out_global = output_base_dir / f"iforest_global_importance_{activity_suffix}.csv"
out_normal = output_base_dir / f"iforest_normal_importance_{activity_suffix}.csv"
out_error = output_base_dir / f"iforest_error_importance_{activity_suffix}.csv"
out_combined = output_base_dir / f"iforest_combined_importance_{activity_suffix}.csv"

score_df.to_csv(out_scores, index=False)
global_importance.rename("importance").to_csv(out_global, header=True)
normal_importance.rename("importance").to_csv(out_normal, header=True)
error_importance.rename("importance").to_csv(out_error, header=True)
importance_df.to_csv(out_combined, index=False)

print("\nSaved files:")
print(f"  {out_scores}")
print(f"  {out_global}")
print(f"  {out_normal}")
print(f"  {out_error}")
print(f"  {out_combined}")