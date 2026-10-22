"""Engineered composite features combining other feature families."""

import numpy as np
import pandas as pd

from .helpers import clean_missing


    
def create_engineered_composite_features(row: dict) -> dict:
    p90 = row.get("P90", np.nan)
    p50 = row.get("DT_MEDIAN", np.nan)
    switch_rate = row.get("SWITCH_RATE", np.nan)
    max_dt = row.get("MAX_DT", np.nan)

    p90 = float(p90) if pd.notna(p90) else np.nan
    p50 = float(p50) if pd.notna(p50) else np.nan
    switch_rate = float(switch_rate) if pd.notna(switch_rate) else np.nan
    max_dt = float(max_dt) if pd.notna(max_dt) else np.nan

    spike_count = int(
        (row.get("analogue_ad1_b_spike_count", 0) or 0) +
        (row.get("analogue_ad1_c_spike_count", 0) or 0)
    )

    ad1_bc_total_interactions = int(
        (row.get("analogue_ad1_b_interactions", 0) or 0) +
        (row.get("analogue_ad1_c_interactions", 0) or 0)
    )

    b_secs_to_end = clean_missing(row.get("analogue_ad1_b_secs_to_episode_end", np.nan))
    c_secs_to_end = clean_missing(row.get("analogue_ad1_c_secs_to_episode_end", np.nan))

    b_last_value = clean_missing(row.get("analogue_ad1_b_last_value", np.nan))
    c_last_value = clean_missing(row.get("analogue_ad1_c_last_value", np.nan))
    b_secs_to_end = float(b_secs_to_end) if pd.notna(b_secs_to_end) else np.nan
    c_secs_to_end = float(c_secs_to_end) if pd.notna(c_secs_to_end) else np.nan
    b_last_value = float(b_last_value) if pd.notna(b_last_value) else np.nan
    c_last_value = float(c_last_value) if pd.notna(c_last_value) else np.nan

    if pd.notna(b_secs_to_end) and pd.notna(c_secs_to_end):
        last_tap_value = b_last_value if b_secs_to_end <= c_secs_to_end else c_last_value
    elif pd.notna(b_secs_to_end):
        last_tap_value = b_last_value
    elif pd.notna(c_secs_to_end):
        last_tap_value = c_last_value
    else:
        last_tap_value = -1.0

    if pd.notna(last_tap_value) and last_tap_value > 0.25:
        last_tap_value_lt_2_5 = 1
    else:
        last_tap_value_lt_2_5 = 0


    return {
        "P90_MINUS_P50": float(p90 - p50) if pd.notna(p90) and pd.notna(p50) else np.nan,
        "P90_X_SWITCH": float(p90 * switch_rate) if pd.notna(p90) and pd.notna(switch_rate) else np.nan,
        "SWITCH_OVER_P90": float(switch_rate / p90) if pd.notna(switch_rate) and pd.notna(p90) and p90 != 0 else np.nan,
        "MAX_DT_OVER_P90": float(max_dt / p90) if pd.notna(max_dt) and pd.notna(p90) and p90 != 0 else np.nan,
        "SPIKE_COUNT": spike_count,
        "AD1_BC_TOTAL_INTERACTIONS": ad1_bc_total_interactions,
        "LAST_TAP_VALUE": last_tap_value,
        "LAST_TAP_VALUE_LT_2_5": last_tap_value_lt_2_5,
    }
