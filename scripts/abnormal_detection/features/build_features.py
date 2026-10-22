#!/usr/bin/env python3
"""Build engineered features for ABNORMAL DETECTION (is this session normal or errored?).

Turns the cleaned sessions of one activity into one row of features each, using
abnormal_activity_detection.features.anomaly. Reference statistics (event and
n-gram vocabularies) come from the no-error sessions; each no-error row uses
leave-one-out references, so a session is never compared against itself.

Settings: TARGET_ACTIVITY_CODE in abnormal_activity_detection/anomaly/configs/features.py
          ("t1".."t5", or None for all activities)
Reads:    data/processed/adl_noerror_clean, data/processed/adl_error_clean
Writes:   results/abnormal_detection/features/<activity>/engineered_features/
          <activity>_noerror_all_features.csv and <activity>_error_all_features.csv

Not the same as scripts/activity_classification/features/build_features.py,
which builds a different feature set across all activities for telling the
activities apart.
"""

from pathlib import Path
from abnormal_activity_detection.data import (
    load_data
)
from abnormal_activity_detection.features.anomaly import (
    build_feature_matrix,
    build_reference_stats,
    feature_columns,
    build_feature_matrix_train_loo_reference,
)
from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir
from abnormal_activity_detection.anomaly.configs.features import (
    TARGET_ACTIVITY_CODE,
    FEATURE_CONFIG,
)

GENERATE_ALL_FEATURES = True

def process_dataset(
    sessions,
    meta,
    dataset_label: str,
    activity_label: str,
    output_dir: Path,
    cols,
    reference_stats,
):
    if not sessions:
        print(f"No sessions found for dataset={dataset_label}, activity={TARGET_ACTIVITY_CODE}")
        return

    print(f"\nLoaded {dataset_label} sessions: {len(sessions)}")
    print("Activity filter:", TARGET_ACTIVITY_CODE or "ALL")

    # Debug sensors
    for df, m in zip(sessions, meta):
        sensors = sorted(df["sensor"].astype(str).str.upper().unique())
        print(m["filename"], m["activity_code"], sensors)

    # -------------------------
    # Feature matrix
    # -------------------------
    if dataset_label == "noerror":
        features_df = build_feature_matrix_train_loo_reference(
            session_dfs=sessions,
            meta=meta,
            cols=cols,
        )
    else:
        features_df = build_feature_matrix(
            session_dfs=sessions,
            meta=meta,
            cols=cols,
            reference_stats=reference_stats,
        )

    # -------------------------
    # Output
    # -------------------------
    suffix = "all_features" if GENERATE_ALL_FEATURES else "configured_features"
    out_fp = output_dir / f"{activity_label}_{dataset_label}_{suffix}.csv"

    features_df.to_csv(out_fp, index=False)

    print(f"\nSaved engineered features: {out_fp}")
    print(f"Rows: {len(features_df)}")
    print(f"Feature columns: {len(cols)}")


def main():
    
    normal_dir, error_dir = get_dataset_dirs()

    activity_label = TARGET_ACTIVITY_CODE if TARGET_ACTIVITY_CODE else "all_activities"
    out_dir = get_results_dir("abnormal_detection", "features", activity_label, "engineered_features")

    # -------------------------
    # Load noerror and error first
    # -------------------------
    normal_sessions, normal_meta, error_sessions, error_meta = load_data(target_activity_code=TARGET_ACTIVITY_CODE, normal_dir=normal_dir, error_dir=error_dir, n_splits=0)

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
        if TARGET_ACTIVITY_CODE is None:
            raise ValueError(
                "TARGET_ACTIVITY_CODE must be set to a specific activity when GENERATE_ALL_FEATURES=False"
            )
        cols = feature_columns(
            target_activity=TARGET_ACTIVITY_CODE,
            feature_config=FEATURE_CONFIG,
        )

    # -------------------------
    # Reference stats
    # -------------------------
    # Build ONLY from noerror sessions
    reference_stats = build_reference_stats(normal_sessions)

    print("\nReference stats built from noerror:")
    print("event vocab:", len(reference_stats["event_vocab"]))
    print("bigram vocab:", len(reference_stats["bigram_vocab"]))
    print("trigram vocab:", len(reference_stats["trigram_vocab"]))

    # -------------------------
    # Process noerror
    # -------------------------
    process_dataset(
        sessions=normal_sessions,
        meta=normal_meta,
        dataset_label="noerror",
        activity_label=activity_label,
        output_dir=out_dir,
        cols=cols,
        reference_stats=reference_stats,
    )

    # -------------------------
    # Process error
    # -------------------------
    process_dataset(
        sessions=error_sessions,
        meta=error_meta,
        dataset_label="error",
        activity_label=activity_label,
        output_dir=out_dir,
        cols=cols,
        reference_stats=reference_stats,
    )


if __name__ == "__main__":
    main()