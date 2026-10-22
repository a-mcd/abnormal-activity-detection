"""Helpers for reading session CSVs in the activity classifiers.

These match the classifiers' original copies exactly. Note that they differ
from the similarly named helpers in abnormal_activity_detection.data:
get_sensor_type uses norm_sensor rather than clean_sensor_name, and
VALID_MESSAGES here does not include "UP" or "DOWN".
"""

import pandas as pd

from abnormal_activity_detection.data import norm_msg, norm_sensor
from abnormal_activity_detection.features.activity import duration_bucket_label

VALID_MESSAGES = {"ON", "OFF", "START", "END", "OPEN", "CLOSE", "PRESENT", "ABSENT"}

__all__ = [
    "VALID_MESSAGES",
    "norm_sensor",
    "norm_msg",
    "get_sensor_type",
    "is_analogue_sensor",
    "is_known_message",
    "build_timestamp",
    "duration_bucket_label",
]


def get_sensor_type(sensor: str) -> str:
    s = norm_sensor(sensor)
    if s.startswith("m"):
        return "motion"
    if s.startswith("i"):
        return "item"
    if s.startswith("d"):
        return "door"
    if s == "asterisk":
        return "asterisk"
    if s.startswith("ad1"):
        return "analogue"
    return "unknown"


def is_analogue_sensor(sensor: str) -> bool:
    return get_sensor_type(sensor) == "analogue"


def is_known_message(sensor: str, msg: str) -> bool:
    st = get_sensor_type(sensor)
    msg = norm_msg(msg)

    if st == "analogue":
        try:
            float(msg)
            return True
        except Exception:
            return False

    if st in ("motion", "item", "door", "asterisk"):
        return msg in VALID_MESSAGES

    return False


def build_timestamp(df: pd.DataFrame) -> pd.Series:
    """Combine the date and time columns into one timestamp (with or without fractional seconds)."""
    d = pd.to_datetime(df["date"], errors="coerce")
    t = pd.to_datetime(df["time"], format="%H:%M:%S.%f", errors="coerce")
    t2 = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
    t = t.fillna(t2)
    return d + (t - t.dt.normalize())
