#!/usr/bin/env python3
"""Build engineered features for ACTIVITY CLASSIFICATION (which activity is this, t1-t5?).

Turns every cleaned session, from all five activities, into one row of features
using abnormal_activity_detection.features.activity.

Reads:  data/processed/adl_noerror_clean, data/processed/adl_error_clean
Writes: results/activity_classification/features/engineered_features/
        noerror_all_features.csv and error_all_features.csv

Not the same as scripts/abnormal_detection/features/build_features.py, which
builds a different feature set for one activity at a time, compares each
session against normal reference sessions, and is used for normal-vs-error
detection.
"""

from pathlib import Path

from abnormal_activity_detection.data import load_data
from abnormal_activity_detection.features.activity import (
    build_feature_matrix,
    feature_columns,
)
from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir

FEATURE_CONFIG = {}
GENERATE_ALL_FEATURES = True


def process_dataset(
    sessions,
    meta,
    dataset_label: str,
    output_dir: Path,
    cols,
):
    if not sessions:
        print(f"No sessions found for dataset={dataset_label}")
        return

    print(f"\nLoaded {dataset_label} sessions: {len(sessions)}")
    print("Activity filter: NONE (using all activities)")

    # Debug sensors
    for df, m in zip(sessions, meta):
        sensors = sorted(df["sensor"].astype(str).str.upper().unique())
        print(m["filename"], m["activity_code"], sensors)

    # -------------------------
    # Feature matrix
    # -------------------------
    features_df = build_feature_matrix(
        session_dfs=sessions,
        meta=meta,
        cols=cols,
    )

    # -------------------------
    # Output
    # -------------------------
    suffix = "all_features" if GENERATE_ALL_FEATURES else "configured_features"
    out_fp = output_dir / f"{dataset_label}_{suffix}.csv"

    features_df.to_csv(out_fp, index=False)

    print(f"\nSaved engineered features: {out_fp}")
    print(f"Rows: {len(features_df)}")
    print(f"Feature columns: {len(cols)}")


def main():
    normal_dir, error_dir = get_dataset_dirs()

    out_dir = get_results_dir("activity_classification", "features", "engineered_features")

    # -------------------------
    # Load ALL activities
    # -------------------------
    normal_sessions, normal_meta, error_sessions, error_meta = load_data(
        target_activity_code="t-all",   
        normal_dir=normal_dir,
        error_dir=error_dir,
        n_splits=0,
    )

    # -------------------------
    # Feature columns
    # -------------------------
    if GENERATE_ALL_FEATURES:
        all_sessions = normal_sessions + error_sessions
        all_meta = normal_meta + error_meta

        cols = feature_columns(
            session_dfs=all_sessions,
            meta=all_meta,
            all_features=True,
        )
    else:
        cols = feature_columns(
            target_activity="t-all",   # optional if your config supports it
            feature_config=FEATURE_CONFIG,
        )

    # -------------------------
    # Process noerror
    # -------------------------
    process_dataset(
        sessions=normal_sessions,
        meta=normal_meta,
        dataset_label="noerror",
        output_dir=out_dir,
        cols=cols,
    )

    # -------------------------
    # Process error
    # -------------------------
    process_dataset(
        sessions=error_sessions,
        meta=error_meta,
        dataset_label="error",
        output_dir=out_dir,
        cols=cols,
    )


if __name__ == "__main__":
    main()