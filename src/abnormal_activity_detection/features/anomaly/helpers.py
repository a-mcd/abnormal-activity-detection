"""Small helpers shared by several feature families (timestamps, sensor names, missing values)."""

import re
import numpy as np
import pandas as pd


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

def is_active_message_raw(msg) -> bool:
    """
    Stage-1 active logic for raw messages.
    - numeric analogue value > 0 => active
    - ON / OPEN / START / PRESENT => active
    - everything else => inactive
    """
    if pd.isna(msg):
        return False

    #msg_str = norm_msg(msg)
    msg_str = msg

    try:
        return float(msg_str) > 0.0
    except Exception:
        return msg_str in {"ON", "OPEN", "START", "PRESENT"}


# =========================================================
# ENGINEERED / COMPOSITE FEATURES
# =========================================================
def clean_missing(x):
    if pd.isna(x):
        return np.nan
    try:
        x = float(x)
        return np.nan if x == -1 else x
    except Exception:
        return np.nan
