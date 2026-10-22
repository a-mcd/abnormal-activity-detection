"""Episode-level, event-count/diversity and temporal-dynamic (timing) features."""

import numpy as np
import pandas as pd


def compute_global_episode_features(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "UNIQUE_SENSOR_COUNT": 0,
            "EPISODE_DURATION": np.nan,
        }

    if "sensor" in df.columns:
        unique_sensor_count = int(
            df["sensor"].dropna().astype(str).str.strip().nunique()
        )
    else:
        unique_sensor_count = 0

    if "timestamp" in df.columns and df["timestamp"].notna().any():
        t_min = df["timestamp"].min()
        t_max = df["timestamp"].max()
        episode_duration = (
            float((t_max - t_min).total_seconds())
            if pd.notna(t_min) and pd.notna(t_max)
            else np.nan
        )
    else:
        episode_duration = np.nan

    return {
        "UNIQUE_SENSOR_COUNT": unique_sensor_count,
        "EPISODE_DURATION": episode_duration,
    }


# =========================================================
# EVENT COUNT & DIVERSITY FEATURES
# =========================================================
def create_event_count_diversity_features(event_df: pd.DataFrame) -> dict:
    if event_df.empty:
        return {
            "NUM_EVENTS": 0,
            "N_BIGRAMS": 0,
            "N_TRIGRAMS": 0,
            "UNIQUE_MESSAGES": 0,
            "UNIQUE_SENSOR_MESSAGE_COUNT": 0,
        }

    out = event_df.copy()

    if "sensor" not in out.columns:
        out["sensor"] = ""
    if "message" not in out.columns:
        out["message"] = ""

    out["sensor"] = out["sensor"].astype(str).str.strip().str.upper()
    out["message"] = out["message"].astype(str).str.strip().str.upper()

    num_events = int(len(out))
    n_bigrams = max(0, num_events - 1)
    n_trigrams = max(0, num_events - 2)
    unique_messages = int(out["message"].nunique())

    sensor_message_pairs = out[["sensor", "message"]].drop_duplicates()
    unique_sensor_message_count = int(len(sensor_message_pairs))

    return {
        "NUM_EVENTS": num_events,
        "N_BIGRAMS": n_bigrams,
        "N_TRIGRAMS": n_trigrams,
        "UNIQUE_MESSAGES": unique_messages,
        "UNIQUE_SENSOR_MESSAGE_COUNT": unique_sensor_message_count,
    }


# =========================================================
# TEMPORAL DYNAMIC FEATURES
# =========================================================
def create_temporal_dynamic_features(event_df: pd.DataFrame) -> dict:
    default = {
        "MEAN_DT": np.nan,
        "STD_DT": np.nan,
        "DT_MEDIAN": np.nan,
        "P90": np.nan,
        "MAX_DT": np.nan,
        "PCT_DT_ZERO": np.nan,
        "LOGDT_MEAN": np.nan,
        "LOGDT_STD": np.nan,
        "EVENT_RATE": np.nan,
        "EVENTS_PER_MIN": np.nan,
        "CV_DT": np.nan,
        "BURSTINESS": np.nan,
        "BURSTINESS_X_P90": np.nan,
    }

    if event_df.empty or "timestamp" not in event_df.columns:
        return default

    out = event_df.copy()
    out = out.loc[out["timestamp"].notna()].copy()

    if out.empty:
        return default

    out = out.sort_values("timestamp").reset_index(drop=True)

    num_events = int(len(out))

    if num_events >= 2:
        dt = out["timestamp"].diff().dt.total_seconds().fillna(0.0).to_numpy()
    else:
        dt = np.array([0.0], dtype=float)

    dt = np.clip(dt, a_min=0.0, a_max=None)
    dt_pos = dt[dt > 0]

    pct_dt_zero = float(np.mean(dt == 0) * 100.0) if len(dt) > 0 else np.nan

    if len(dt_pos) > 0:
        mean_dt = float(np.mean(dt_pos))
        std_dt = float(np.std(dt_pos, ddof=0))
        dt_median = float(np.median(dt_pos))
        p90 = float(np.percentile(dt_pos, 90))
        max_dt = float(np.max(dt_pos))

        log_dt = np.log1p(dt_pos)
        logdt_mean = float(np.mean(log_dt))
        logdt_std = float(np.std(log_dt, ddof=0))

        cv_dt = float(std_dt / mean_dt) if mean_dt != 0 else np.nan

        denom = std_dt + mean_dt
        burstiness = float((std_dt - mean_dt) / denom) if denom != 0 else np.nan
        burstiness_x_p90 = float(burstiness * p90) if pd.notna(burstiness) and pd.notna(p90) else np.nan
    else:
        mean_dt = np.nan
        std_dt = np.nan
        dt_median = np.nan
        p90 = np.nan
        max_dt = np.nan
        logdt_mean = np.nan
        logdt_std = np.nan
        cv_dt = np.nan
        burstiness = np.nan
        burstiness_x_p90 = np.nan

    if num_events >= 2:
        duration = float((out["timestamp"].max() - out["timestamp"].min()).total_seconds())
        duration = max(0.0, duration)
    else:
        duration = np.nan

    event_rate = float(num_events / duration) if pd.notna(duration) and duration > 0 else np.nan
    events_per_min = float(num_events / (duration / 60.0)) if pd.notna(duration) and duration > 0 else np.nan

    return {
        "MEAN_DT": mean_dt,
        "STD_DT": std_dt,
        "DT_MEDIAN": dt_median,
        "P90": p90,
        "MAX_DT": max_dt,
        "PCT_DT_ZERO": pct_dt_zero,
        "LOGDT_MEAN": logdt_mean,
        "LOGDT_STD": logdt_std,
        "EVENT_RATE": event_rate,
        "EVENTS_PER_MIN": events_per_min,
        "CV_DT": cv_dt,
        "BURSTINESS": burstiness,
        "BURSTINESS_X_P90": burstiness_x_p90,
    }
