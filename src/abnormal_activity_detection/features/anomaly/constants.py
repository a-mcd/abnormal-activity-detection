"""Shared constants and metadata column names."""


# =========================================================
# CONFIG / CONSTANTS
# =========================================================
TAIL_N = 5
COLLAPSE_SENSORS = {"AD1-B", "AD1-C"}
ALL_AD1_SENSORS = {"AD1-A", "AD1-B", "AD1-C"}
MICRO_ADJUSTMENT_WINDOW_SEC = 2

PHONE_MESSAGE_SET = {
    "START",
    "END",
}

PHONE_SENSOR_SET = {
    "ASTERISK",
}

TARGET_PHONE_ANCHOR_SENSOR = "I08"

META_COLS = [
    "filename",
    "participant_id",
    "activity_code",
    "activity_index",
    "dataset",
]
