"""Session-level extraction and feature-matrix builders: turn sessions into
one feature row each, using all the feature families.
"""

from typing import Optional
import numpy as np
import pandas as pd

from .composite import create_engineered_composite_features
from .constants import ALL_AD1_SENSORS, COLLAPSE_SENSORS, META_COLS
from .helpers import build_timestamp, safe_sensor_name
from .phone import create_phone_features
from .reference import build_reference_stats, create_reference_sequence_features, get_reference_stats_for_activity
from .tap_burner import analogue_stage1_feature_dict, analogue_stage2_feature_dict, analogue_stage3_feature_dict, apply_stage3, apply_stage4_direction_labels, apply_stage5_build_event_table, stage1_tap_window_feature_dict
from .timing import compute_global_episode_features, create_event_count_diversity_features, create_temporal_dynamic_features
from .transitions import create_event_encoding_features, create_stage5_repeat_features, create_transition_behaviour_features


def default_value_for_missing_feature(col: str):
    """
    Return the correct fallback when a feature key was not created.

    Rules:
    - Sparse count / presence / occurrence features -> 0
    - Continuous summary / timing / ratio / dispersion features -> np.nan
    """
    if col is None:
        return np.nan

    # Event encoding / sparse count features
    if (
        col.startswith("sensor::")
        or col.startswith("msg::")
        or col.startswith("evt::")
        or col.startswith("state::")
        or col.startswith("type::")
        or col.startswith("trans::")
    ):
        return 0

    # Explicit count-like / indicator features
    zero_default_cols = {
        "NUM_EVENTS",
        "N_BIGRAMS",
        "N_TRIGRAMS",
        "UNIQUE_MESSAGES",
        "UNIQUE_SENSOR_MESSAGE_COUNT",
        "UNIQUE_SENSOR_COUNT",
        "HAS_ANY_PHONE_CALL",
        "SPIKE_COUNT",
        "AD1_B_LAST_IS_DOWN",
        "AD1_C_LAST_IS_DOWN",
        "AD1_BC_TOTAL_INTERACTIONS",
        "UNEXPECTED_SENSOR_DURING_TAP_WINDOW",
        "AD1_BC_ACTIVE_LAST_8S",
        "PHONE_COMPLETED_CALL_COUNT",
        "PHONE_INCOMPLETE_CALL_COUNT",
    }

    if col in zero_default_cols:
        return 0

    # Analogue count / indicator / interaction style features
    if (
        col.endswith("_count")
        or col.endswith("_num_changes")
        or col.endswith("_spike_count")
        or col.endswith("_interactions")
        or col.endswith("::ACTIVE")
    ):
        return 0

    #if col == "AD1_BC_ACTIVE_LAST_TO_END":
    #    return -1.0

    return np.nan

# =========================================================
# SESSION-LEVEL FEATURE EXTRACTION
# =========================================================
def extract_features_for_session(
    df: pd.DataFrame,
    source_file: str = "",
    required_cols: Optional[list[str]] = None,
    reference_stats: Optional[dict] = None,
) -> dict:
    """
    Build the same features and stages as the standalone script.
    If required_cols is provided, output is filtered to those columns.
    """
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]

    required_base = {"sensor", "message"}
    missing = required_base - set(out.columns)
    if missing:
        raise ValueError(f"{source_file or 'session'} is missing required columns: {sorted(missing)}")

    out["sensor"] = out["sensor"].astype(str).str.strip().str.upper()

    if "timestamp" not in out.columns:
        out["timestamp"] = build_timestamp(out)
    else:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    if out["timestamp"].notna().any():
        out = out.sort_values("timestamp").reset_index(drop=True)

    out["message_num"] = pd.to_numeric(out["message"], errors="coerce")
    analogue_rows = out[out["message_num"].notna()].copy()

    row = {
        "source_file": source_file,
    }

    row.update(compute_global_episode_features(out))

    if "timestamp" in out.columns and out["timestamp"].notna().any():
        episode_start_time = out["timestamp"].min()
        episode_end_time = out["timestamp"].max()
    else:
        episode_start_time = pd.NaT
        episode_end_time = pd.NaT

    # -----------------------------------------------------
    # Stage 1 raw session-level tap features
    # -----------------------------------------------------
    row.update(
        stage1_tap_window_feature_dict(
            session_df=out,
            episode_end_time=episode_end_time,
            window_sec=8,
            tap_sensors=("AD1-B", "AD1-C"),
            allowed_sensors=("AD1-B", "AD1-C", "M17", "M18"),
        )
    )

    # -----------------------------------------------------
    # Stage 1 per-sensor analogue summaries
    # -----------------------------------------------------
    for sensor in sorted(ALL_AD1_SENSORS):
        sensor_df = analogue_rows[analogue_rows["sensor"] == sensor].copy()
        row.update(
            analogue_stage1_feature_dict(
                sensor,
                sensor_df,
                episode_start_time=episode_start_time,
                episode_end_time=episode_end_time,
            )
        )

    stage4_tables = []

    for sensor in COLLAPSE_SENSORS:
        sensor_df = analogue_rows[analogue_rows["sensor"] == sensor].copy()

        clean_name = safe_sensor_name(sensor)
        prefix = f"analogue_{clean_name}"

        if not sensor_df.empty:
            sensor_df = sensor_df.sort_values("timestamp").reset_index(drop=True)

            stage2_features, stage2_df = analogue_stage2_feature_dict(sensor, sensor_df)
            row.update(stage2_features)

            stage3_df = apply_stage3(stage2_df, source_file)
            row.update(analogue_stage3_feature_dict(sensor, stage3_df))

            stage4_df = apply_stage4_direction_labels(stage3_df)
            stage4_tables.append(stage4_df)
        else:
            row[f"{prefix}_num_changes"] = 0
            row[f"{prefix}_spike_count"] = 0
            row[f"{prefix}_interactions"] = 0

    stage5_event_df = apply_stage5_build_event_table(out, stage4_tables)

    row.update(create_stage5_repeat_features(stage5_event_df))
    row.update(create_event_count_diversity_features(stage5_event_df))
    row.update(create_temporal_dynamic_features(stage5_event_df))
    row.update(create_event_encoding_features(stage5_event_df))
    row.update(create_transition_behaviour_features(stage5_event_df))
    row.update(create_reference_sequence_features(stage5_event_df, reference_stats=reference_stats))
    row.update(create_engineered_composite_features(row))
    row.update(create_phone_features(stage5_event_df))

    if required_cols is not None:
        return {col: row.get(col, default_value_for_missing_feature(col)) for col in required_cols}

    return row

def infer_all_feature_columns(
    session_dfs: list[pd.DataFrame],
    meta: Optional[list[dict]] = None,
) -> list[str]:
    """
    Infer the full observed feature schema by extracting all rows and taking the union
    of all non-meta feature keys.
    """
    if meta is None:
        meta = [{} for _ in session_dfs]

    cols = set()

    for df, m in zip(session_dfs, meta):
        source_file = str(m.get("filename", ""))
        row = extract_features_for_session(
            df=df,
            source_file=source_file,
            required_cols=None,
            reference_stats=None,
        )
        cols.update(row.keys())

    cols.discard("source_file")
    cols -= set(META_COLS)
    return sorted(cols)


def feature_columns(
    target_activity=None,
    feature_config=None,
    all_features=False,
    session_dfs=None,
    meta=None,
):
    if all_features:
        if session_dfs is None:
            raise ValueError("session_dfs required when all_features=True")
        return infer_all_feature_columns(session_dfs=session_dfs, meta=meta)

    if feature_config is None:
        raise ValueError("feature_config must be provided")

    if target_activity not in feature_config:
        raise KeyError(f"{target_activity} not found in FEATURE_CONFIG")

    cols = list(feature_config[target_activity])

    if target_activity == "t-all":
        for c in ["activity_t1", "activity_t2", "activity_t3", "activity_t4", "activity_t5"]:
            if c not in cols:
                cols.append(c)

    return cols


def build_feature_matrix(
    
    session_dfs: list[pd.DataFrame],
    meta: list[dict],
    cols: Optional[list[str]] = None,
    reference_stats: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build a feature matrix with one row per session.

    If cols is None, all observed feature columns across the provided sessions are used.
    """
    if cols is None:
        cols = infer_all_feature_columns(session_dfs=session_dfs, meta=meta)

    rows = []

    for df, m in zip(session_dfs, meta):
        source_file = str(m.get("filename", ""))
        row_reference_stats = get_reference_stats_for_activity(
            reference_stats=reference_stats,
            activity_code=m.get("activity_code"),
        )
        feat_row = extract_features_for_session(
            df=df,
            source_file=source_file,
            required_cols=cols,
            reference_stats=row_reference_stats,
        )

        activity_code = str(m.get("activity_code", "")).strip().lower()
        for code in ["t1", "t2", "t3", "t4", "t5"]:
            feat_row[f"activity_{code}"] = 1.0 if activity_code == code else 0.0

        feat_row.update(m)
        rows.append(feat_row)

    out = pd.DataFrame(rows)

    ordered_cols = META_COLS + cols
    ordered_cols = [c for c in ordered_cols if c in out.columns]
    return out.reindex(columns=ordered_cols)


def build_feature_matrix_train_loo_reference(
    session_dfs: list[pd.DataFrame],
    meta: list[dict],
    cols: list[str],
) -> pd.DataFrame:
    rows = []

    all_activity_codes = {
        str(m.get("activity_code", "")).strip()
        for m in meta
        if str(m.get("activity_code", "")).strip()
    }
    is_t_all_mode = len(all_activity_codes) > 1

    for i, (df, m) in enumerate(zip(session_dfs, meta)):
        activity_code_i = str(m.get("activity_code", "")).strip()

        if is_t_all_mode:
            ref_dfs = [
                session_dfs[j]
                for j in range(len(session_dfs))
                if j != i and str(meta[j].get("activity_code", "")).strip() == activity_code_i
            ]
        else:
            ref_dfs = [session_dfs[j] for j in range(len(session_dfs)) if j != i]

        reference_stats_i = build_reference_stats(ref_dfs)

        source_file = str(m.get("filename", ""))
        feat_row = extract_features_for_session(
            df=df,
            source_file=source_file,
            required_cols=cols,
            reference_stats=reference_stats_i,
        )

        activity_code = str(m.get("activity_code", "")).strip().lower()
        for code in ["t1", "t2", "t3", "t4", "t5"]:
            feat_row[f"activity_{code}"] = 1.0 if activity_code == code else 0.0

        feat_row.update(m)
        rows.append(feat_row)

    out = pd.DataFrame(rows)
    ordered_cols = META_COLS + cols
    ordered_cols = [c for c in ordered_cols if c in out.columns]
    return out.reindex(columns=ordered_cols)
