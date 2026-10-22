"""Tap and burner (analogue AD1-A/B/C) sensor pipeline, Stages 1-5.

AD1-A is the burner; AD1-B and AD1-C are the hot and cold taps. Stages 1-4
summarise and clean these signals, and Stage 5 builds the final event table
used by the other feature families.
"""

import numpy as np
import pandas as pd

from .constants import ALL_AD1_SENSORS, COLLAPSE_SENSORS, MICRO_ADJUSTMENT_WINDOW_SEC, TAIL_N
from .helpers import build_timestamp, is_active_message_raw, safe_sensor_name


def compute_tail_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)
    y = values.astype(float)
    return float(np.polyfit(x, y, 1)[0])


def collapse_consecutive_identical_rows(sensor_df: pd.DataFrame) -> pd.DataFrame:
    if sensor_df.empty:
        return sensor_df.copy()

    out = sensor_df.copy().reset_index(drop=True)
    out["value_changed"] = out["message_num"].ne(out["message_num"].shift(1))
    out["run_id"] = out["value_changed"].cumsum()
    out = out.groupby("run_id", as_index=False).first()
    out = out.drop(columns=["value_changed", "run_id"], errors="ignore")
    return out


def _merge_group_keep_first_time_last_value(df: pd.DataFrame, idxs: list[int]) -> dict:
    first_row = df.loc[idxs[0]].copy()
    last_row = df.loc[idxs[-1]]
    row_dict = first_row.to_dict()

    if "message" in df.columns:
        row_dict["message"] = last_row["message"]
    if "message_num" in df.columns:
        row_dict["message_num"] = last_row["message_num"]

    return row_dict


def compute_num_changes(values: np.ndarray) -> int:
    if len(values) < 2:
        return 0
    return int(np.sum(np.diff(values) != 0))


def compute_spike_count(values: np.ndarray) -> int:
    if len(values) < 3:
        return 0

    count = 0
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            count += 1
    return int(count)


def stage1_tap_window_feature_dict(
    session_df: pd.DataFrame,
    episode_end_time,
    window_sec: int = 8,
    tap_sensors=("AD1-B", "AD1-C"),
    allowed_sensors=("AD1-B", "AD1-C", "M17", "M18"),
) -> dict:
    feats = {
        "UNEXPECTED_SENSOR_DURING_TAP_WINDOW": 0.0,
        f"AD1_BC_ACTIVE_LAST_{int(window_sec)}S": 0.0,
    }

    if session_df is None or len(session_df) == 0:
        return feats

    if "sensor" not in session_df.columns or "message" not in session_df.columns:
        return feats

    if "timestamp" not in session_df.columns:
        return feats

    df = session_df.copy()
    df["sensor_norm"] = df["sensor"].map(lambda x: str(x).strip().upper())
    #df["msg_norm"] = df["message"].map(norm_msg)
    df["msg_norm"] = df["message"]

    tap_sensors = {str(s).strip().upper() for s in tap_sensors}
    allowed_sensors = {str(s).strip().upper() for s in allowed_sensors}

    tap_df = df[df["sensor_norm"].isin(tap_sensors)].copy()

    if len(tap_df) > 0 and pd.notna(episode_end_time):
        secs_to_end = (episode_end_time - tap_df["timestamp"]).dt.total_seconds()
        in_last_window = (secs_to_end >= 0) & (secs_to_end <= float(window_sec))
        active_mask = tap_df["message"].map(is_active_message_raw)

        if bool((in_last_window & active_mask).any()):
            feats[f"AD1_BC_ACTIVE_LAST_{int(window_sec)}S"] = 1.0

    if len(tap_df) > 0:
        tap_start = tap_df["timestamp"].min()
        tap_end = tap_df["timestamp"].max()

        window_df = df[
            (df["timestamp"] >= tap_start) &
            (df["timestamp"] <= tap_end)
        ].copy()

        window_df = window_df[window_df["msg_norm"] != "OFF"]
        unexpected_mask = ~window_df["sensor_norm"].isin(allowed_sensors)

        if bool(unexpected_mask.any()):
            feats["UNEXPECTED_SENSOR_DURING_TAP_WINDOW"] = 1.0

    return feats

# =========================================================
# STAGE 1 FEATURES
# =========================================================
def analogue_stage1_feature_dict(
    sensor_name: str,
    sensor_df: pd.DataFrame,
    episode_start_time,
    episode_end_time,
) -> dict:
    clean_name = safe_sensor_name(sensor_name)
    prefix = f"analogue_{clean_name}"

    vals = (
        pd.to_numeric(sensor_df["message_num"], errors="coerce")
        .dropna()
        .astype(float)
        .to_numpy()
    )

    nan_features = {
        f"{prefix}_count": 0.0,
        f"{prefix}_mean": -1.0,
        f"{prefix}_std": -1.0,
        f"{prefix}_min": -1.0,
        f"{prefix}_max": -1.0,
        f"{prefix}_median": -1.0,
        f"{prefix}_p25": -1.0,
        f"{prefix}_p75": -1.0,
        f"{prefix}_p90": -1.0,
        f"{prefix}_range": -1.0,
        f"{prefix}_iqr": -1.0,
        f"{prefix}_cv": -1.0,
        f"{prefix}_pct_zero": -1.0,
        f"{prefix}_last_value": -1.0,
        f"{prefix}_tail_mean": -1.0,
        f"{prefix}_tail_slope": -1.0,
        f"{prefix}::ACTIVE": 0.0,
        f"{prefix}_dur": 0.0,
        f"{prefix}_secs_to_episode_end": -1.0,
        f"{prefix}_ratio_to_episode_end": -1.0,
    }

    if len(vals) == 0:
        return nan_features

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=0))
    min_val = float(np.min(vals))
    max_val = float(np.max(vals))
    p25 = float(np.percentile(vals, 25))
    p75 = float(np.percentile(vals, 75))

    tail_vals = vals[-TAIL_N:] if len(vals) >= TAIL_N else vals
    active = 1.0 if np.any(vals > 0) else 0.0

    if "timestamp" in sensor_df.columns and sensor_df["timestamp"].notna().any():
        t_min = sensor_df["timestamp"].min()
        t_max = sensor_df["timestamp"].max()
        dur = (
            float((t_max - t_min).total_seconds())
            if pd.notna(t_min) and pd.notna(t_max)
            else np.nan
        )
        last_sensor_time = t_max
    else:
        dur = np.nan
        last_sensor_time = pd.NaT

    if pd.notna(last_sensor_time) and pd.notna(episode_end_time):
        secs_to_episode_end = float((episode_end_time - last_sensor_time).total_seconds())
        secs_to_episode_end = max(0.0, secs_to_episode_end)
    else:
        secs_to_episode_end = np.nan

    if pd.notna(episode_start_time) and pd.notna(episode_end_time):
        episode_duration = float((episode_end_time - episode_start_time).total_seconds())
        episode_duration = max(0.0, episode_duration)
    else:
        episode_duration = np.nan

    ratio_to_episode_end = (
        float(secs_to_episode_end / episode_duration)
        if pd.notna(secs_to_episode_end) and pd.notna(episode_duration) and episode_duration > 0
        else np.nan
    )

    return {
        f"{prefix}_count": int(len(vals)),
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_min": min_val,
        f"{prefix}_max": max_val,
        f"{prefix}_median": float(np.median(vals)),
        f"{prefix}_p25": p25,
        f"{prefix}_p75": p75,
        f"{prefix}_p90": float(np.percentile(vals, 90)),
        f"{prefix}_range": float(max_val - min_val),
        f"{prefix}_iqr": float(p75 - p25),
        f"{prefix}_cv": float(std_val / mean_val) if mean_val != 0 else np.nan,
        f"{prefix}_pct_zero": float(np.mean(vals == 0) * 100.0),
        f"{prefix}_last_value": float(vals[-1]),
        f"{prefix}_tail_mean": float(np.mean(tail_vals)),
        f"{prefix}_tail_slope": compute_tail_slope(tail_vals),
        f"{prefix}::ACTIVE": active,
        f"{prefix}_dur": dur,
        f"{prefix}_secs_to_episode_end": secs_to_episode_end,
        f"{prefix}_ratio_to_episode_end": ratio_to_episode_end,
    }


# =========================================================
# STAGE 2 FEATURES
# =========================================================
def analogue_stage2_feature_dict(sensor_name: str, sensor_df: pd.DataFrame):
    clean_name = safe_sensor_name(sensor_name)
    prefix = f"analogue_{clean_name}"

    if sensor_df.empty:
        return (
            {
                f"{prefix}_num_changes": 0,
                f"{prefix}_spike_count": 0,
            },
            sensor_df.copy(),
        )

    stage2 = collapse_consecutive_identical_rows(sensor_df)
    vals = stage2["message_num"].dropna().astype(float).to_numpy()

    if len(vals) == 0:
        feature_dict = {
            f"{prefix}_num_changes": 0,
            f"{prefix}_spike_count": 0,
        }
    else:
        feature_dict = {
            f"{prefix}_num_changes": compute_num_changes(vals),
            f"{prefix}_spike_count": compute_spike_count(vals),
        }

    return feature_dict, stage2


# =========================================================
# STAGE 3
# =========================================================
def apply_stage3(sensor_df_stage2: pd.DataFrame, csv_name: str = "") -> pd.DataFrame:
    if sensor_df_stage2.empty:
        return sensor_df_stage2.copy()

    out = sensor_df_stage2.copy().reset_index(drop=True)

    if "timestamp" not in out.columns or out["timestamp"].isna().all():
        return out

    merged_rows = []
    current_group = [0]

    for i in range(1, len(out)):
        prev_ts = out.loc[i - 1, "timestamp"]
        curr_ts = out.loc[i, "timestamp"]

        if pd.isna(prev_ts) or pd.isna(curr_ts):
            time_diff = np.inf
        else:
            time_diff = (curr_ts - prev_ts).total_seconds()

        if time_diff < MICRO_ADJUSTMENT_WINDOW_SEC:
            current_group.append(i)
        else:
            merged_rows.append(_merge_group_keep_first_time_last_value(out, current_group))
            current_group = [i]

    merged_rows.append(_merge_group_keep_first_time_last_value(out, current_group))
    return pd.DataFrame(merged_rows)


def analogue_stage3_feature_dict(sensor_name: str, stage3_df: pd.DataFrame) -> dict:
    clean_name = safe_sensor_name(sensor_name)
    prefix = f"analogue_{clean_name}"

    feature_dict = {
        f"{prefix}_interactions": 0,
    }

    # Add default binary end-state features for the sink sensors
    if sensor_name == "AD1-B":
        feature_dict["AD1_B_LAST_IS_DOWN"] = 0.0
    elif sensor_name == "AD1-C":
        feature_dict["AD1_C_LAST_IS_DOWN"] = 0.0

    if stage3_df.empty:
        return feature_dict

    feature_dict[f"{prefix}_interactions"] = int(len(stage3_df))

    # Treat last value == 0 as DOWN/OFF
    if "message_num" in stage3_df.columns:
        vals = pd.to_numeric(stage3_df["message_num"], errors="coerce").dropna()

        if len(vals) > 0:
            last_val = float(vals.iloc[-1])

            if sensor_name == "AD1-B":
                feature_dict["AD1_B_LAST_IS_DOWN"] = 1.0 if last_val == 0.0 else 0.0
            elif sensor_name == "AD1-C":
                feature_dict["AD1_C_LAST_IS_DOWN"] = 1.0 if last_val == 0.0 else 0.0

    return feature_dict


# =========================================================
# STAGE 4
# =========================================================
def apply_stage4_direction_labels(sensor_df_stage3: pd.DataFrame) -> pd.DataFrame:
    if sensor_df_stage3.empty:
        return sensor_df_stage3.copy()

    out = sensor_df_stage3.copy().reset_index(drop=True)

    if "message_num" not in out.columns:
        return out

    vals = pd.to_numeric(out["message_num"], errors="coerce")
    direction_labels = []
    prev_val = None

    for curr_val in vals:
        if pd.isna(curr_val):
            direction_labels.append("START" if prev_val is None else None)
            prev_val = curr_val
            continue

        if prev_val is None or pd.isna(prev_val):
            direction_labels.append("START")
        elif curr_val > prev_val:
            direction_labels.append("UP")
        elif curr_val < prev_val:
            direction_labels.append("DOWN")
        else:
            direction_labels.append("SAME")

        prev_val = curr_val

    out["message"] = direction_labels
    return out


# =========================================================
# STAGE 5
# =========================================================
def apply_stage5_build_event_table(
    original_df: pd.DataFrame,
    stage4_tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Stage 5:
    - remove all raw AD1 rows from the original table
    - append Stage 4 AD1-B / AD1-C rows
    - AD1-A is therefore excluded
    """
    base = original_df.copy()

    if "sensor" in base.columns:
        sensor_norm = base["sensor"].astype(str).str.strip().str.upper()
        base = base.loc[~sensor_norm.isin(ALL_AD1_SENSORS)].copy()

    if stage4_tables:
        ad1_bc_stage4 = pd.concat(stage4_tables, ignore_index=True)
        out = pd.concat([base, ad1_bc_stage4], ignore_index=True, sort=False)
    else:
        out = base

    if "timestamp" in out.columns and out["timestamp"].notna().any():
        out = out.sort_values("timestamp").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    return out


def build_stage5_event_table(df: pd.DataFrame, source_file: str = "") -> pd.DataFrame:
    """
    Run the session through the pipeline up to the end of Stage 5
    and return the Stage 5 event table.
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

    stage4_tables = []

    for sensor in COLLAPSE_SENSORS:
        sensor_df = analogue_rows[analogue_rows["sensor"] == sensor].copy()

        if sensor_df.empty:
            continue

        sensor_df = sensor_df.sort_values("timestamp").reset_index(drop=True)

        _, stage2_df = analogue_stage2_feature_dict(sensor, sensor_df)
        stage3_df = apply_stage3(stage2_df, source_file)
        stage4_df = apply_stage4_direction_labels(stage3_df)
        stage4_tables.append(stage4_df)

    stage5_event_df = apply_stage5_build_event_table(out, stage4_tables)
    return stage5_event_df
