import re
from typing import Optional

import numpy as np
import pandas as pd


# =========================================================
# CONFIG / CONSTANTS
# =========================================================
ALL_AD1_SENSORS = {"AD1-A", "AD1-B", "AD1-C"}

PHONE_MESSAGE_SET = {"START", "END"}
PHONE_SENSOR_SET = {"ASTERISK"}
TARGET_PHONE_ANCHOR_SENSOR = "I08"

META_COLS = [
    "filename",
    "participant_id",
    "activity_code",
    "activity_index",
    "dataset",
]


# =========================================================
# HELPERS
# =========================================================
def safe_sensor_name(sensor: str) -> str:
    sensor = str(sensor).strip().lower()
    sensor = sensor.replace("-", "_")
    sensor = re.sub(r"[^a-z0-9_]+", "_", sensor)
    sensor = re.sub(r"_+", "_", sensor).strip("_")
    return sensor


def find_time_column(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}

    if "timestamp" in cols:
        return ("single", cols["timestamp"])
    if "datetime" in cols:
        return ("single", cols["datetime"])
    if "date" in cols and "time" in cols:
        return ("split", (cols["date"], cols["time"]))

    return (None, None)


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    mode, col_info = find_time_column(df)

    def parse_series(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip()

        ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")

        missing = ts.isna()
        if missing.any():
            ts.loc[missing] = pd.to_datetime(
                s.loc[missing],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )

        missing = ts.isna()
        if missing.any():
            ts.loc[missing] = pd.to_datetime(
                s.loc[missing],
                errors="coerce",
            )

        return ts

    if mode == "single":
        return parse_series(df[col_info])

    if mode == "split":
        date_col, time_col = col_info
        combined = (
            df[date_col].astype(str).str.strip()
            + " "
            + df[time_col].astype(str).str.strip()
        )
        return parse_series(combined)

    return pd.Series(pd.NaT, index=df.index)


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


def default_value_for_missing_feature(col: str):
    if col is None:
        return np.nan

    if (
        col.startswith("sensor::")
        or col.startswith("msg::")
        or col.startswith("evt::")
        or col.startswith("evt_bin::")
        or col.startswith("trans::")
    ):
        return 0

    zero_default_cols = {
        "NUM_EVENTS",
        "N_BIGRAMS",
        "N_TRIGRAMS",
        "UNIQUE_MESSAGES",
        "UNIQUE_SENSOR_MESSAGE_COUNT",
        "UNIQUE_SENSOR_COUNT",
        "HAS_ANY_PHONE_CALL",
        "PHONE_COMPLETED_CALL_COUNT",
        "PHONE_INCOMPLETE_CALL_COUNT",
        "CONSECUTIVE_REPEAT_FRAC",
    }

    if col in zero_default_cols:
        return 0

    if col.endswith("_count") or col.endswith("::ACTIVE"):
        return 0

    return np.nan


# =========================================================
# RAW EVENT TABLE
# =========================================================
def build_activity_event_table(df: pd.DataFrame, source_file: str = "") -> pd.DataFrame:
    """
    Build the event table directly from the raw/lightly cleaned session data.

    No Stage 2–5 processing:
    - no collapsing repeated analogue values
    - no micro-adjustment merging
    - no direction labels
    - no rebuilt Stage-5 event table
    """
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]

    required_base = {"sensor", "message"}
    missing = required_base - set(out.columns)
    if missing:
        raise ValueError(
            f"{source_file or 'session'} is missing required columns: {sorted(missing)}"
        )

    out["sensor"] = out["sensor"].astype(str).str.strip().str.upper()
    out["message"] = out["message"].astype(str).str.strip().str.upper()

    if "timestamp" not in out.columns:
        out["timestamp"] = build_timestamp(out)
    else:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    if out["timestamp"].notna().any():
        out = out.sort_values("timestamp").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    return out


# =========================================================
# SIMPLE ANALOGUE FEATURES
# =========================================================
def analogue_summary_feature_dict(
    sensor_name: str,
    sensor_df: pd.DataFrame,
) -> dict:
    """
    Reduced analogue features for activity classification.

    Keep only a small set that helps distinguish activities:
    - count
    - mean
    - std
    - max
    - ACTIVE
    - dur
    """
    clean_name = safe_sensor_name(sensor_name)
    prefix = f"analogue_{clean_name}"

    vals = (
        pd.to_numeric(sensor_df["message"], errors="coerce")
        .dropna()
        .astype(float)
        .to_numpy()
    )

    default = {
        f"{prefix}_count": 0.0,
        f"{prefix}_mean": -1.0,
        f"{prefix}_std": -1.0,
        f"{prefix}_max": -1.0,
        f"{prefix}::ACTIVE": 0.0,
        f"{prefix}_dur": 0.0,
    }

    if len(vals) == 0:
        return default

    mean_val = float(np.mean(vals))
    std_val = float(np.std(vals, ddof=0))
    max_val = float(np.max(vals))
    active = 1.0 if np.any(vals > 0) else 0.0

    if "timestamp" in sensor_df.columns and sensor_df["timestamp"].notna().any():
        t_min = sensor_df["timestamp"].min()
        t_max = sensor_df["timestamp"].max()
        dur = (
            float((t_max - t_min).total_seconds())
            if pd.notna(t_min) and pd.notna(t_max)
            else np.nan
        )
    else:
        dur = np.nan

    return {
        f"{prefix}_count": int(len(vals)),
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_max": max_val,
        f"{prefix}::ACTIVE": active,
        f"{prefix}_dur": dur,
    }

def create_tap_duration_bucket_features(out: pd.DataFrame) -> dict:
    """
    Create duration bucket features for AD1-B and AD1-C (taps).
    """

    features = {}

    for sensor in ["AD1-B", "AD1-C"]:
        clean = safe_sensor_name(sensor)  # ad1_b / ad1_c

        sensor_df = out[
            (out["sensor"] == sensor)
            & pd.to_numeric(out["message"], errors="coerce").notna()
        ].copy()

        # Default: no bucket active
        buckets = [
            "DUR_0_60",
            "DUR_61_120",
            "DUR_121_200",
            "DUR_201_320",
            "DUR_321_PLUS",
        ]

        for b in buckets:
            features[f"{clean}_dur_bucket::{b}"] = 0

        if sensor_df.empty or "timestamp" not in sensor_df.columns:
            continue

        sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"], errors="coerce")
        sensor_df = sensor_df.dropna(subset=["timestamp"])

        if sensor_df.empty:
            continue

        t_min = sensor_df["timestamp"].min()
        t_max = sensor_df["timestamp"].max()

        if pd.notna(t_min) and pd.notna(t_max):
            dur = float((t_max - t_min).total_seconds())

            bucket = duration_bucket_label(dur)
            features[f"{clean}_dur_bucket::{bucket}"] = 1

    return features


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
        burstiness_x_p90 = (
            float(burstiness * p90)
            if pd.notna(burstiness) and pd.notna(p90)
            else np.nan
        )
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
    events_per_min = (
        float(num_events / (duration / 60.0))
        if pd.notna(duration) and duration > 0
        else np.nan
    )

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


# =========================================================
# EVENT ENCODING FEATURES
# =========================================================
def create_event_encoding_features(event_df: pd.DataFrame) -> dict:
    if event_df.empty:
        return {}

    out = event_df.copy()

    if "sensor" not in out.columns:
        return {}

    out["sensor"] = out["sensor"].astype(str).str.strip().str.upper()

    if "message" in out.columns:
        out["message"] = out["message"].astype(str).str.strip().str.upper()
    else:
        out["message"] = ""

    out = out[~out["sensor"].isin(ALL_AD1_SENSORS)].copy()
    feats = {}

    # -------------------------
    # sensor:: counts
    # -------------------------
    for sensor, count in out["sensor"].value_counts().items():
        feats[f"sensor::{sensor}"] = int(count)

    # -------------------------
    # msg:: counts
    # -------------------------
    for msg, count in out["message"].value_counts().items():
        feats[f"msg::{msg}"] = int(count)

    # -------------------------
    # evt:: counts
    # evt_bin:: binary presence
    # -------------------------
    evt_counts = out.groupby(["sensor", "message"]).size()
    for (sensor, msg), count in evt_counts.items():
        feats[f"evt::{sensor}::{msg}"] = int(count)
        feats[f"evt_bin::{sensor}::{msg}"] = 1

    # -------------------------
    # trans:: counts
    # -------------------------
    sensors = out["sensor"].tolist()
    for a, b in zip(sensors[:-1], sensors[1:]):
        key = f"trans::{a}::{b}"
        feats[key] = feats.get(key, 0) + 1

    return feats

# =========================================================
# TRANSITION / REPEAT FEATURES
# =========================================================
def _build_event_tokens(event_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if event_df.empty:
        return [], []

    out = event_df.copy()

    if "sensor" not in out.columns:
        out["sensor"] = ""
    if "message" not in out.columns:
        out["message"] = ""

    out["sensor"] = out["sensor"].astype(str).str.strip().str.upper()
    out["message"] = out["message"].astype(str).str.strip().str.upper()

    event_tokens = (out["sensor"] + "::" + out["message"]).tolist()
    sensor_tokens = out["sensor"].tolist()

    return event_tokens, sensor_tokens


def _make_ngrams(tokens: list[str], n: int) -> list[tuple]:
    if len(tokens) < n or n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def create_transition_behaviour_features(event_df: pd.DataFrame) -> dict:
    default = {
        "SWITCH_RATE": np.nan,
        "BIGRAM_MATCH_RATIO": np.nan,
        "TRIGRAM_MATCH_RATIO": np.nan,
        "SENSOR_BIGRAM_MATCH_RATIO": np.nan,
    }

    if event_df.empty:
        return default

    event_tokens, sensor_tokens = _build_event_tokens(event_df)

    num_events = len(event_tokens)
    if num_events == 0:
        return default

    if num_events >= 2:
        n_adjacent = num_events - 1
        sensor_switches = sum(
            1 for a, b in zip(sensor_tokens[:-1], sensor_tokens[1:]) if a != b
        )
        switch_rate = float(sensor_switches / n_adjacent)
    else:
        switch_rate = np.nan

    event_bigrams = _make_ngrams(event_tokens, 2)
    event_trigrams = _make_ngrams(event_tokens, 3)
    sensor_bigrams = _make_ngrams(sensor_tokens, 2)

    if len(event_bigrams) > 0:
        bigram_match_ratio = float(1.0 - (len(set(event_bigrams)) / len(event_bigrams)))
    else:
        bigram_match_ratio = np.nan

    if len(event_trigrams) > 0:
        trigram_match_ratio = float(1.0 - (len(set(event_trigrams)) / len(event_trigrams)))
    else:
        trigram_match_ratio = np.nan

    if len(sensor_bigrams) > 0:
        sensor_bigram_match_ratio = float(
            1.0 - (len(set(sensor_bigrams)) / len(sensor_bigrams))
        )
    else:
        sensor_bigram_match_ratio = np.nan

    return {
        "SWITCH_RATE": switch_rate,
        "BIGRAM_MATCH_RATIO": bigram_match_ratio,
        "TRIGRAM_MATCH_RATIO": trigram_match_ratio,
        "SENSOR_BIGRAM_MATCH_RATIO": sensor_bigram_match_ratio,
    }


def create_repeat_features(event_df: pd.DataFrame) -> dict:
    default = {
        "CONSECUTIVE_REPEAT_FRAC": 0.0,
    }

    if event_df is None or event_df.empty:
        return default

    out = event_df.copy()

    if "sensor" not in out.columns:
        out["sensor"] = ""
    if "message" not in out.columns:
        out["message"] = ""

    out["sensor"] = out["sensor"].astype(str).str.strip().str.upper()
    out["message"] = out["message"].astype(str).str.strip().str.upper()

    n_events = len(out)
    if n_events <= 1:
        return default

    same_as_prev = (
        out["sensor"].eq(out["sensor"].shift(1)) &
        out["message"].eq(out["message"].shift(1))
    )

    n_repeat_consecutive = int(same_as_prev.iloc[1:].sum())
    consecutive_repeat_frac = float(n_repeat_consecutive / (n_events - 1))

    return {
        "CONSECUTIVE_REPEAT_FRAC": consecutive_repeat_frac,
    }


# =========================================================
# PHONE FEATURES
# =========================================================
def create_phone_features(df: pd.DataFrame) -> dict:
    default = {
        "HAS_ANY_PHONE_CALL": 0,
        "PHONE_TOTAL_CALL_TIME": -1.0,
    }

    if df.empty:
        return default

    out = df.copy()

    if "sensor" not in out.columns:
        out["sensor"] = ""
    if "message" not in out.columns:
        out["message"] = ""

    out["sensor"] = out["sensor"].astype(str).str.strip().str.upper()
    out["message"] = out["message"].astype(str).str.strip().str.upper()

    if "timestamp" not in out.columns:
        return default

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()

    if out.empty:
        return default

    out = out.sort_values("timestamp").reset_index(drop=True)

    phone_df = out[
        out["sensor"].isin(PHONE_SENSOR_SET)
        & out["message"].isin(PHONE_MESSAGE_SET)
    ].copy()

    # -----------------------------------------------------
    # HAS_ANY_PHONE_CALL
    # -----------------------------------------------------
    has_any_phone_call = int(len(phone_df) > 0)

    # -----------------------------------------------------
    # PHONE_TOTAL_CALL_TIME
    # -----------------------------------------------------
    total_call_time = 0.0
    open_start_time = pd.NaT

    for _, row in phone_df.iterrows():
        msg = row["message"]
        ts = row["timestamp"]

        if msg == "START":
            open_start_time = ts

        elif msg == "END" and pd.notna(open_start_time):
            dur = float((ts - open_start_time).total_seconds())
            total_call_time += max(0.0, dur)
            open_start_time = pd.NaT

    # If no valid completed calls
    if total_call_time == 0.0:
        total_call_time = -1.0

    return {
        "HAS_ANY_PHONE_CALL": has_any_phone_call,
        "PHONE_TOTAL_CALL_TIME": total_call_time,
    }


# =========================================================
# SESSION-LEVEL FEATURE EXTRACTION
# =========================================================
def extract_features_for_session(
    df: pd.DataFrame,
    source_file: str = "",
    required_cols: Optional[list[str]] = None,
) -> dict:
    out = build_activity_event_table(df, source_file=source_file)

    row = {
        "source_file": source_file,
    }

    row.update(compute_global_episode_features(out))

    analogue_rows = out[
        out["sensor"].isin(ALL_AD1_SENSORS)
        & pd.to_numeric(out["message"], errors="coerce").notna()
    ].copy()

    for sensor in sorted(ALL_AD1_SENSORS):
        sensor_df = analogue_rows[analogue_rows["sensor"] == sensor].copy()
        row.update(analogue_summary_feature_dict(sensor, sensor_df))

    row.update(create_event_count_diversity_features(out))
    row.update(create_temporal_dynamic_features(out))
    row.update(create_event_encoding_features(out))
    row.update(create_transition_behaviour_features(out))
    row.update(create_repeat_features(out))
    row.update(create_phone_features(out))
    row.update(create_tap_duration_bucket_features(out))

    if required_cols is not None:
        return {col: row.get(col, default_value_for_missing_feature(col)) for col in required_cols}

    return row


def infer_all_feature_columns(
    session_dfs: list[pd.DataFrame],
    meta: Optional[list[dict]] = None,
) -> list[str]:
    if meta is None:
        meta = [{} for _ in session_dfs]

    cols = set()

    for df, m in zip(session_dfs, meta):
        source_file = str(m.get("filename", ""))
        row = extract_features_for_session(
            df=df,
            source_file=source_file,
            required_cols=None,
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
    return cols

def duration_bucket_label(duration_s: float) -> str:
    if duration_s <= 60:
        return "DUR_0_60"
    elif duration_s <= 120:
        return "DUR_61_120"
    elif duration_s <= 200:
        return "DUR_121_200"
    elif duration_s <= 320:
        return "DUR_201_320"
    else:
        return "DUR_321_PLUS"


def build_feature_matrix(
    session_dfs: list[pd.DataFrame],
    meta: list[dict],
    cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    if cols is None:
        cols = infer_all_feature_columns(session_dfs=session_dfs, meta=meta)

    rows = []

    for df, m in zip(session_dfs, meta):
        source_file = str(m.get("filename", ""))
        feat_row = extract_features_for_session(
            df=df,
            source_file=source_file,
            required_cols=cols,
        )

        feat_row.update(m)
        rows.append(feat_row)

    out = pd.DataFrame(rows)

    ordered_cols = META_COLS + cols
    ordered_cols = [c for c in ordered_cols if c in out.columns]
    return out.reindex(columns=ordered_cols)