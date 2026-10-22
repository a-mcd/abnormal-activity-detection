"""Phone-use features (phone book I08 and the VOIP phone sensor)."""

import numpy as np
import pandas as pd

from .constants import PHONE_MESSAGE_SET, PHONE_SENSOR_SET, TARGET_PHONE_ANCHOR_SENSOR


# =========================================================
# PHONE FEATURES
# =========================================================
def create_phone_features(df: pd.DataFrame) -> dict:
    default = {
        "HAS_ANY_PHONE_CALL": 0,
        "FIRST_PHONEBOOK_TO_LAST_PHONE_DELAY": -1.0,

        "PHONE_COMPLETED_CALL_COUNT": 0,
        "PHONE_INCOMPLETE_CALL_COUNT": 0,
        "FIRST_PHONEBOOK_TO_FIRST_PHONE_START_DELAY": -1.0,
        "PHONE_TOTAL_CALL_TIME": -1.0,
        "PHONE_LONGEST_CALL_TIME": -1.0,
        "PHONE_SHORTEST_CALL_TIME": -1.0,
        "LAST_CALL_END_TO_EPISODE_END": -1.0,
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

    # -----------------------------------------------------
    # Identify phone rows: only ASTERISK with START / END
    # -----------------------------------------------------
    phone_df = out[
        out["sensor"].isin(PHONE_SENSOR_SET)
        & out["message"].isin(PHONE_MESSAGE_SET)
    ].copy()

    has_any_phone_call = int(len(phone_df) > 0)

    # -----------------------------------------------------
    # Existing feature:
    # time from first phone book interaction to last phone event
    # -----------------------------------------------------
    i08_mask = out["sensor"].eq(TARGET_PHONE_ANCHOR_SENSOR)
    first_i08_time = out.loc[i08_mask, "timestamp"].min() if i08_mask.any() else pd.NaT
    last_phone_time = phone_df["timestamp"].max() if len(phone_df) > 0 else pd.NaT

    if pd.notna(first_i08_time) and pd.notna(last_phone_time):
        first_phonebook_to_last_phone_delay = float(
            (last_phone_time - first_i08_time).total_seconds()
        )
        first_phonebook_to_last_phone_delay = max(0.0, first_phonebook_to_last_phone_delay)
    else:
        first_phonebook_to_last_phone_delay = -1.0

    # -----------------------------------------------------
    # New call-level features
    # -----------------------------------------------------
    completed_call_count = 0
    incomplete_call_count = 0
    call_durations = []
    first_phone_start_time = pd.NaT
    last_call_end_time = pd.NaT

    open_start_time = pd.NaT

    for _, row in phone_df.iterrows():
        msg = row["message"]
        ts = row["timestamp"]

        if msg == "START":
            if pd.isna(first_phone_start_time):
                first_phone_start_time = ts

            if pd.notna(open_start_time):
                incomplete_call_count += 1

            open_start_time = ts

        elif msg == "END":
            if pd.notna(open_start_time):
                dur = float((ts - open_start_time).total_seconds())
                dur = max(0.0, dur)

                call_durations.append(dur)
                completed_call_count += 1
                last_call_end_time = ts
                open_start_time = pd.NaT

    if pd.notna(open_start_time):
        incomplete_call_count += 1

    # -----------------------------------------------------
    # Delay from first phone book interaction to first phone START
    # -----------------------------------------------------
    if pd.notna(first_i08_time) and pd.notna(first_phone_start_time):
        first_phonebook_to_first_phone_start_delay = float(
            (first_phone_start_time - first_i08_time).total_seconds()
        )
        first_phonebook_to_first_phone_start_delay = max(
            0.0, first_phonebook_to_first_phone_start_delay
        )
    else:
        first_phonebook_to_first_phone_start_delay = -1.0

    # -----------------------------------------------------
    # Aggregate call durations
    # -----------------------------------------------------
    if len(call_durations) > 0:
        phone_total_call_time = float(np.sum(call_durations))
        phone_longest_call_time = float(np.max(call_durations))
        phone_shortest_call_time = float(np.min(call_durations))
    else:
        phone_total_call_time = -1.0
        phone_longest_call_time = -1.0
        phone_shortest_call_time = -1.0

    # -----------------------------------------------------
    # Time from end of last completed call to end of episode
    # -----------------------------------------------------
    episode_end_time = out["timestamp"].max()

    if pd.notna(last_call_end_time) and pd.notna(episode_end_time):
        last_call_end_to_episode_end = float(
            (episode_end_time - last_call_end_time).total_seconds()
        )
        last_call_end_to_episode_end = max(0.0, last_call_end_to_episode_end)
    else:
        last_call_end_to_episode_end = -1.0

    return {
        "HAS_ANY_PHONE_CALL": has_any_phone_call,
        "FIRST_PHONEBOOK_TO_LAST_PHONE_DELAY": first_phonebook_to_last_phone_delay,
        "PHONE_COMPLETED_CALL_COUNT": int(completed_call_count),
        "PHONE_INCOMPLETE_CALL_COUNT": int(incomplete_call_count),
        "FIRST_PHONEBOOK_TO_FIRST_PHONE_START_DELAY": first_phonebook_to_first_phone_start_delay,
        "PHONE_TOTAL_CALL_TIME": phone_total_call_time,
        "PHONE_LONGEST_CALL_TIME": phone_longest_call_time,
        "PHONE_SHORTEST_CALL_TIME": phone_shortest_call_time,
        "LAST_CALL_END_TO_EPISODE_END": last_call_end_to_episode_end,
    }
