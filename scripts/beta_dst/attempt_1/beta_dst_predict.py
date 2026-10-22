#!/usr/bin/env python3

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import os
import re
import numpy as np

from abnormal_activity_detection.paths import RAW_DATA_DIR, RESULTS_DIR

ACTIVITIES = ["t1", "t2", "t3", "t4", "t5"]
# Belief values are read from, and results written to, results/beta_dst/attempt_N/
BASE_DIR = RESULTS_DIR / "beta_dst" / Path(__file__).resolve().parent.name
FILE_TEMPLATE = "{}_sensor_belief_values.csv"

NORMAL_DIR = RAW_DATA_DIR / "adl_noerror"

BELIEF_COLS_LOWER = {"sensor", "message", "belief", "disbelief", "uncertainty"}

Mass = Tuple[float, float, float]  # (belief, disbelief, uncertainty)
Lookup = Dict[str, Dict[Tuple[str, str], Mass]]

EDGES_WATER = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)          # AD1-B / AD1-C
EDGES_BURNER = (2.85, 2.95, 3.05, 3.15, 3.25, 3.35) 
VALID_MESSAGES = {"ON", "OFF", "START", "END", "OPEN", "CLOSE", "PRESENT", "ABSENT", "ACTIVE"}             # AD1-A


def build_belief_lookup():
    """
    Returns nested dictionary:

    lookup[activity][(sensor, message)] = (belief, disbelief, uncertainty)
    """

    lookup = {}

    for act in ACTIVITIES:
        file_path = BASE_DIR / FILE_TEMPLATE.format(act)

        df = pd.read_csv(file_path)

        df.columns = df.columns.astype(str).str.strip()
        required = {"sensor", "message", "Belief", "Disbelief", "Uncertainty"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{file_path.name} missing {missing}; found {df.columns.tolist()}")


        activity_dict = {}

        for _, row in df.iterrows():
            #print(row["sensor"], row["message"], row["Belief"], row["Disbelief"], row["Uncertainty"])
            key = (row["sensor"], row["message"])

            activity_dict[key] = (
                float(row["Belief"]),
                float(row["Disbelief"]),
                float(row["Uncertainty"]),
            )
        
        lookup[act] = activity_dict

    return lookup



def get_mass(lookup, activity, sensor, message):
    """
    Safe lookup.

    Returns neutral mass if sensor not found.
    """
    key = (sensor.strip(), message.strip())

    lookup_val = lookup.get(activity, {}).get(key)

    if lookup_val is None:
        raise ValueError(f"Lookup missing for activity={activity}, sensor={sensor}, message={message}")

    return lookup_val


def get_sensor_type(sensor: str) -> str:
    s = sensor
    if s.startswith("M"):
        return "motion"
    elif s.startswith("I"):
        return "item"
    elif s.startswith("D"):
        return "door"
    elif s == "asterisk":
        return "asterisk"
    elif s.startswith("AD1"):
        return "analogue"
    return "unknown"


def is_known_message(sensor: str, msg: str) -> bool:
    st = get_sensor_type(sensor)
    
    if st == "analogue":
        try:
            float(msg)
            return True
        except:
            return False

    if st in ("motion", "item", "door", "asterisk"):
        return msg in VALID_MESSAGES

    return False


def read_unique_events(file_path):
    """
    Reads the CASAS file and returns a list of unique (sensor, message) pairs.
    """
    df = pd.read_csv(file_path)

    df = bin_analogue_rows(df)

    # Ensure columns are in lowercase and stripped
    cols = {c.lower(): c for c in df.columns}

    sensor_col = cols["sensor"]
    msg_col = cols["message"]

    # If date/time exist, sort so "first seen" matches the episode order
    if "date" in cols and "time" in cols:
        date_col = cols["date"]
        time_col = cols["time"]
        ts = pd.to_datetime(df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
                            errors="coerce")
        df = df.assign(_ts=ts).sort_values("_ts", kind="mergesort").drop(columns=["_ts"])


    df_u = df.drop_duplicates(subset=[sensor_col, msg_col], keep="first").copy()
    df_u = df_u.rename(columns={sensor_col: "sensor", msg_col: "message"})
    df_u = df_u[["sensor", "message"]]
    return df_u

def extract_activity_code(filename: str) -> str:
    m = re.search(r'\.(t[1-5])\.csv$', filename)
    return m.group(1) if m else None
    
def conflict_k(m1: Mass, m2: Mass) -> float:
    b1, d1, _u1 = m1
    b2, d2, _u2 = m2

    return b1 * d2 + d1 * b2


def dst_fuse(m1: Mass, m2: Mass):
    """
    Dempster's rule for binary frame {a, ~a} with Theta (uncertainty).
    Returns (fused_mass, K_for_this_step).
    """
    b1, d1, u1 = m1
    b2, d2, u2 = m2

    K = conflict_k(m1, m2)
    denom = 1.0 - K

    # Total conflict fallback
    if denom <= 1e-15:
        return (0.0, 0.0, 1.0), K

    b = (b1 * b2 + b1 * u2 + u1 * b2) / denom
    d = (d1 * d2 + d1 * u2 + u1 * d2) / denom
    u = (u1 * u2) / denom

    return (b, d, u), K


def fuse_table_for_activity(lookup: Lookup, events_df: pd.DataFrame, activity: str) -> pd.DataFrame:
    rows = []
    fused_b = fused_d = fused_u = None

    for i, r in enumerate(events_df.itertuples(index=False), start=1):
        sensor, message = r.sensor, r.message
        b, d, u = get_mass(lookup, activity, sensor, message)

        if i == 1:
            fused_b, fused_d, fused_u = b, d, u
            K = 0.0
        else:
            (fused_b, fused_d, fused_u), K = dst_fuse((fused_b, fused_d, fused_u), (b, d, u))

        rows.append({
            "step": i,
            "sensor": sensor,
            "message": message,
            "Belief": b,
            "Disbelief": d,
            "Uncertainty": u,
            "Fused_Belief": fused_b,
            "Fused_Disbelief": fused_d,
            "Fused_Uncertainty": fused_u,
            "K": K,
        })

    out = pd.DataFrame(rows)
    return out


def bin_analogue_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert analogue sensor rows into BIN_k messages:
      AD1-A -> EDGES_BURNER
      AD1-B / AD1-C -> EDGES_WATER

    Assumes analogue values are in df["message"] as numeric strings.
    Non-numeric analogue rows are dropped.
    """
    df = df.copy()

    # Process each analogue sensor separately
    for sensor in ["AD1-A", "AD1-B", "AD1-C"]:
        mask = df["sensor"] == sensor
        if not mask.any():
            continue

        vals = pd.to_numeric(df.loc[mask, "message"], errors="coerce")
        # drop non-numeric analogue rows
        bad = vals.isna()
        if bad.any():
            df = df.drop(df.loc[mask].index[bad])

        # recompute mask after drops
        mask = df["sensor"] == sensor
        if not mask.any():
            continue

        vals = pd.to_numeric(df.loc[mask, "message"], errors="coerce").to_numpy(dtype=float)
        edges = np.array(EDGES_BURNER if sensor == "AD1-A" else EDGES_WATER, dtype=float)

        # bin index in [0..len(edges)] using right=True for <= upper bound
        bin_idx = np.digitize(vals, bins=edges, right=True)
        df.loc[mask, "message"] = [f"BIN_{k}" for k in bin_idx.tolist()]

    return df


if __name__ == "__main__":
    lookup = build_belief_lookup()

    DEBUG_DIR = BASE_DIR / "debug"
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    results_rows = []

    pd.set_option("display.float_format", lambda x: f"{x:.12g}")

    for filename in sorted(os.listdir(NORMAL_DIR)):

        if not filename.endswith(".csv"):
            continue

        casas_file_path = os.path.join(NORMAL_DIR, filename)
        events = read_unique_events(casas_file_path)

        # Filter unknown/bad rows
        good = events.apply(lambda r: get_sensor_type(r["sensor"]) != "unknown"
                                and is_known_message(r["sensor"], r["message"]), axis=1)
        events = events[good].copy()

        summary_rows = []
        for act in ACTIVITIES:
            tbl = fuse_table_for_activity(lookup, events, act)

            final = tbl.tail(1).iloc[0]
            fused_b = float(final["Fused_Belief"])
            fused_d = float(final["Fused_Disbelief"])
            fused_u = float(final["Fused_Uncertainty"])

            print(f"\nFinal fused ({act}):")
            print(pd.DataFrame([{
                "Fused_Belief": fused_b,
                "Fused_Disbelief": fused_d,
                "Fused_Uncertainty": fused_u,
            }]).to_string(index=False))

            summary_rows.append({
                "file": filename,
                "activity": act,
                "Fused_Belief": fused_b,
                "Fused_Disbelief": fused_d,
                "Fused_Uncertainty": fused_u,
                "Score_B_minus_D": fused_b - fused_d,
            })

            debug_path = DEBUG_DIR / f"debug_fusion_{act}_{filename}"
            tbl.to_csv(debug_path, index=False)
            print(f"Wrote step-by-step debug CSV: {debug_path}")

        # Build TB1 summary once all activities are done
        tb1 = pd.DataFrame(summary_rows)
        best_row = tb1.sort_values(["Fused_Belief", "Score_B_minus_D"], ascending=False).iloc[0]
        best_act = best_row["activity"]

        # Store only what you asked for
        results_rows.append({
            "csv_file": filename,
            "actual_activity": extract_activity_code(filename),
            "predicted_activity": best_act
        })

    results_df = pd.DataFrame(results_rows)
    out_path = DEBUG_DIR / "best_activity_per_file.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nWrote summary CSV: {out_path}")