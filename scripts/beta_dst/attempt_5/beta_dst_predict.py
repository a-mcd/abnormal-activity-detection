#!/usr/bin/env python3

from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import os
import re
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from abnormal_activity_detection.paths import RAW_DATA_DIR, RESULTS_DIR

ACTIVITIES = ["t1", "t2", "t3", "t4", "t5"]
# Belief values are read from, and results written to, results/beta_dst/attempt_N/
BASE_DIR = RESULTS_DIR / "beta_dst" / Path(__file__).resolve().parent.name
FILE_TEMPLATE = "{}_sensor_belief_values.csv"

NORMAL_DIR = RAW_DATA_DIR / "adl_noerror"
ERROR_DIR = RAW_DATA_DIR / "adl_error"


BELIEF_COLS_LOWER = {"sensor", "message", "belief", "disbelief", "uncertainty","discounted_belief", "discounted_disbelief", "discounted_uncertainty"}

Mass = Tuple[float, float, float]  # (belief, disbelief, uncertainty)
Lookup = Dict[str, Dict[Tuple[str, str], Mass]]

VALID_MESSAGES = {"ON", "OFF", "START", "END", "OPEN", "CLOSE", "PRESENT", "ABSENT", "ACTIVE"}


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
        missing = BELIEF_COLS_LOWER - set(df.columns)
        if missing:
            raise ValueError(f"{file_path.name} missing {missing}; found {df.columns.tolist()}")


        activity_dict = {}

        for _, row in df.iterrows():
            #print(row["sensor"], row["message"], row["Belief"], row["Disbelief"], row["Uncertainty"])
            key = (row["sensor"], row["message"])

            activity_dict[key] = (
                float(row["belief"]),
                float(row["disbelief"]),
                float(row["uncertainty"]),
                float(row["discounted_belief"]),
                float(row["discounted_disbelief"]),
                float(row["discounted_uncertainty"]),
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


def analogue_duration_bucket(seconds: int) -> str:
    if seconds <= 5:
        return "DUR_0_5"
    elif seconds <= 15:
        return "DUR_6_15"
    elif seconds <= 45:
        return "DUR_16_45"
    elif seconds <= 100:
        return "DUR_46_100"
    else:
        return "DUR_101_PLUS"


def parse_episode_timestamps(df: pd.DataFrame) -> pd.Series:
    """
    Parse timestamps for an episode.
    Supports CASAS style: separate 'date' and 'time' columns.
    Also supports a single timestamp-like column if present.
    """
    cols = {c.lower(): c for c in df.columns}


    s = df[cols["date"]].astype(str).str.strip() + " " + df[cols["time"]].astype(str).str.strip()
    return pd.to_datetime(s, errors="raise", format="mixed")


def drop_singleton_ad1a_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    If an episode contains exactly ONE AD1-A row, drop that AD1-A row.
    Otherwise leave AD1-A untouched.
    """
    df = df.copy()

    # Normalize sensor column for robust matching
    cols = {c.lower(): c for c in df.columns}
    if "sensor" not in cols:
        return df

    sensor_col = cols["sensor"]
    s = df[sensor_col].astype(str).str.strip()

    ad1a_mask = s.eq("AD1-A")
    if int(ad1a_mask.sum()) == 1:
        df = df.loc[~ad1a_mask].copy()

    return df


def preprocess_analogue_with_ad1b_duration(df: pd.DataFrame) -> pd.DataFrame:
    """
    - AD1-A: convert numeric readings -> message 'ACTIVE' (as before)
    - AD1-B: replace ALL AD1-B rows with ONE row: (sensor='AD1-B', message='DUR_...')
    - AD1-C: replace ALL AD1-C rows with ONE row: (sensor='AD1-C', message='DUR_...')
    """
    df = df.copy()

    # Normalize strings (helps matching)
    if "sensor" in df.columns:
        df["sensor"] = df["sensor"].astype(str).str.strip()
    if "message" in df.columns:
        df["message"] = df["message"].astype(str).str.strip()

    # ---- AD1-A -> ACTIVE (only if message is numeric) ----
    m_a = df["sensor"].eq("AD1-A")
    if m_a.any():
        numeric = pd.to_numeric(df.loc[m_a, "message"], errors="coerce")
        df.loc[m_a & numeric.notna(), "message"] = "ACTIVE"

    # Helper: collapse sensor rows to one DUR bucket row
    def collapse_sensor_to_duration(sensor_name: str) -> pd.DataFrame:
        nonlocal df
        m = df["sensor"].eq(sensor_name)
        if not m.any():
            return df

        ts = parse_episode_timestamps(df.loc[m])
        ts = pd.to_datetime(ts, errors="coerce")
        if ts.notna().sum() == 0:
            # If timestamps are unusable, leave sensor as-is (safer)
            return df

        dur_sec = int((ts.max() - ts.min()).total_seconds())
        bucket = analogue_duration_bucket(dur_sec)

        # Drop all rows for that sensor
        df = df.loc[~m].copy()

        # Append one synthetic duration row
        synth = {"sensor": sensor_name, "message": bucket}
        df = pd.concat([df, pd.DataFrame([synth])], ignore_index=True)
        return df

    # ---- AD1-B -> one duration bucket row ----
    collapse_sensor_to_duration("AD1-B")

    # ---- AD1-C -> one duration bucket row ----
    collapse_sensor_to_duration("AD1-C")

    return df


def drop_ad1a_if_short_span(df: pd.DataFrame, min_span_seconds: int = 120) -> pd.DataFrame:
    """
    Drop ALL AD1-A rows if the time span between the first and last AD1-A
    event in the episode is < min_span_seconds.
    """
    df = df.copy()

    # Normalize column names + sensor values
    cols = {c.lower(): c for c in df.columns}
    if "sensor" not in cols:
        return df

    sensor_col = cols["sensor"]
    sensor_vals = df[sensor_col].astype(str).str.strip()

    m_a = sensor_vals.eq("AD1-A")
    if not m_a.any():
        return df  # no AD1-A rows

    # Parse timestamps only for AD1-A rows
    ts = parse_episode_timestamps(df.loc[m_a])

    # If timestamps failed for all AD1-A rows, don't drop (safer)
    ts = pd.to_datetime(ts, errors="coerce")
    if ts.notna().sum() == 0:
        return df

    span_sec = (ts.max() - ts.min()).total_seconds()

    if span_sec < float(min_span_seconds):
        df = df.loc[~m_a].copy()

    return df


def read_unique_events(file_path):
    """
    Reads the CASAS file and returns a list of unique (sensor, message) pairs.
    """
    df = pd.read_csv(file_path)
    #df = add_activity_duration_event(df)
    df = drop_ad1a_if_short_span(df, min_span_seconds=120)
    df = preprocess_analogue_with_ad1b_duration(df)


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
        b, d, u, db, dd, du = get_mass(lookup, activity, sensor, message)

        if i == 1:
            fused_b, fused_d, fused_u = db, dd, du
            K = 0.0
        else:
            (fused_b, fused_d, fused_u), K = dst_fuse((fused_b, fused_d, fused_u), (db, dd, du))

        # Plausibility (binary frame)
        pl = b + u                 # == 1 - d
        dpl = db + du              # discounted plausibility
        fused_pl = fused_b + fused_u  # == 1 - fused_d

        rows.append({
            "step": i,
            "sensor": sensor,
            "message": message,
            "Belief": b,

            "Disbelief": d,
            "Uncertainty": u,
            "Plausibility": pl,

            "Discounted_Belief": db,
            "Discounted_Disbelief": dd,
            "Discounted_Uncertainty": du,
            "Discounted_Plausibility": dpl,

            "Fused_Belief": fused_b,
            "Fused_Disbelief": fused_d,
            "Fused_Uncertainty": fused_u,
            "Fused_Plausibility": fused_pl,

            "K": K,
        })

    out = pd.DataFrame(rows)
    return out



def get_sensor_type(sensor: str) -> str:
    s = sensor
    if s == "ACT_DUR":
        return "episode"
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
        if msg == "ACTIVE":
            return True
        
        if sensor.strip().upper() in ("AD1-B", "AD1-C") and msg.startswith("DUR_"):
            return True
        
        try:
            float(msg)
            return True
        except:
            return False

    if st in ("motion", "item", "door", "asterisk"):
        return msg in VALID_MESSAGES
    
    if st == "episode":
        return msg.startswith("DUR_")

    return False

def activity_duration_bucket(seconds: int) -> str:
    if seconds <= 60:
        return "DUR_0_60"
    elif seconds <= 120:
        return "DUR_61_120"
    elif seconds <= 200:
        return "DUR_121_200"
    elif seconds <= 320:
        return "DUR_201_320"
    else:
        return "DUR_321_PLUS"


def add_activity_duration_event(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "time" not in cols:
        # no timestamps available -> don't add ACT_DUR
        return df

    ts = parse_episode_timestamps(df)               # uses date+time
    ts = pd.to_datetime(ts, errors="coerce")
    if ts.notna().sum() == 0:
        return df

    dur_sec = int((ts.max() - ts.min()).total_seconds())
    bucket = activity_duration_bucket(dur_sec)

    synth = {"sensor": "ACT_DUR", "message": bucket}
    # keep same schema: if your df has more cols, pandas will fill NaN for them
    df = pd.concat([df, pd.DataFrame([synth])], ignore_index=True)
    return df


def run_dataset(lookup, input_dir: Path, output_dir: Path, dataset_label: str):
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results_rows = []

    pd.set_option("display.float_format", lambda x: f"{x:.12g}")

    for filename in sorted(os.listdir(input_dir)):

        if not filename.endswith(".csv"):
            continue

        casas_file_path = os.path.join(input_dir, filename)
        events = read_unique_events(casas_file_path)

        # Filter unknown/bad rows
        good = events.apply(lambda r: get_sensor_type(r["sensor"]) != "unknown"
                                and is_known_message(r["sensor"], r["message"]), axis=1)
        events = events[good].copy()

        file_row = {
            "csv_file": filename,
            "actual_activity": extract_activity_code(filename),
        }

        summary_rows = []
        for act in ACTIVITIES:
            tbl = fuse_table_for_activity(lookup, events, act)

            final = tbl.tail(1).iloc[0]
            fused_b = float(final["Fused_Belief"])
            fused_d = float(final["Fused_Disbelief"])
            fused_u = float(final["Fused_Uncertainty"])
            fused_pl = fused_b + fused_u  # plausibility of the activity (== 1 - fused_d)

            # keep your per-activity print if you want
            print(f"\nFinal fused ({act}):")
            print(pd.DataFrame([{
                "Fused_Belief": fused_b,
                "Fused_Disbelief": fused_d,
                "Fused_Uncertainty": fused_u,
                "Fused_Plausibility": fused_pl,
            }]).to_string(index=False))

            # store WIDE columns per activity (this is what you asked for)
            file_row[f"{act}_Fused_Belief"] = fused_b
            file_row[f"{act}_Fused_Disbelief"] = fused_d
            file_row[f"{act}_Fused_Plausibility"] = fused_pl

            # keep a small summary table for choosing best activity
            summary_rows.append({
                "activity": act,
                "Fused_Belief": fused_b,
                "Fused_Disbelief": fused_d,
                "Fused_Uncertainty": fused_u,
                "Fused_Plausibility": fused_pl,
                "Score": fused_b - fused_d,  # or switch to BetP if you prefer
            })

            debug_path = output_dir / f"debug_fusion_{act}_{filename}"
            tbl.to_csv(debug_path, index=False)
            print(f"Wrote step-by-step debug CSV: {debug_path}")

        tb1 = pd.DataFrame(summary_rows)

        # --- Best + 2nd best by Fused_Belief ---
        ranked = tb1.sort_values("Fused_Belief", ascending=False, kind="mergesort").reset_index(drop=True)

        best_act = ranked.loc[0, "activity"]
        file_row["predicted_activity"] = best_act

        second_best_act = ranked.loc[1, "activity"] if len(ranked) >= 2 else np.nan
        file_row["second_best_activity"] = second_best_act

        # --- belief gap = max - second max (use the ranked list, avoids edge cases) ---
        if len(ranked) >= 2:
            belief_gap = float(ranked.loc[0, "Fused_Belief"] - ranked.loc[1, "Fused_Belief"])
        elif len(ranked) == 1:
            belief_gap = float(ranked.loc[0, "Fused_Belief"])
        else:
            belief_gap = np.nan

        file_row["belief_gap_max_minus_second"] = belief_gap

        # --- plausibility gap = best plausibility - 2nd best plausibility ---
        if len(ranked) >= 2:
            plaus_gap = float(ranked.loc[0, "Fused_Plausibility"] - ranked.loc[1, "Fused_Plausibility"])
        elif len(ranked) == 1:
            plaus_gap = float(ranked.loc[0, "Fused_Plausibility"])  # or 0.0 if you prefer
        else:
            plaus_gap = np.nan

        file_row["plausibility_gap_best_minus_second"] = plaus_gap

        results_rows.append(file_row)


    results_df = pd.DataFrame(results_rows)
    # ---- reorder columns ----
    front_cols = ["csv_file", "actual_activity", "predicted_activity", "second_best_activity", "belief_gap_max_minus_second", "plausibility_gap_best_minus_second"]
    other_cols = [c for c in results_df.columns if c not in front_cols]
    results_df = results_df[front_cols + other_cols]

    out_path = output_dir / f"best_activity_per_file_{dataset_label}.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nWrote summary CSV: {out_path}")

    # -------------------------
    # Confusion Matrix (end)
    # -------------------------
    labels = ACTIVITIES  # enforce t1..t5 order

    # Drop any rows where actual_activity couldn't be extracted
    eval_df = results_df.dropna(subset=["actual_activity"]).copy()

    y_true = eval_df["actual_activity"].astype(str).tolist()
    y_pred = eval_df["predicted_activity"].astype(str).tolist()

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"actual_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])

    print("\nConfusion Matrix (rows=actual, cols=pred):")
    print(cm_df.to_string())

    # Optional: also save it
    cm_path = output_dir / f"confusion_matrix_{dataset_label}.csv"
    cm_df.to_csv(cm_path, index=True)
    print(f"\nWrote confusion matrix CSV: {cm_path}")

    print("\nClassification report:")
    report_text = classification_report(y_true, y_pred, labels=labels)
    print(report_text)
    report_path = output_dir / f"classification_report_{dataset_label}.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nWrote classification report: {report_path}")
   

if __name__ == "__main__":
    lookup = build_belief_lookup()

    # Separate output directories
    OUT_NOERROR = BASE_DIR / "debug_noerror"
    OUT_ERROR = BASE_DIR / "debug_error"

    run_dataset(lookup, NORMAL_DIR, OUT_NOERROR, "noerror")
    run_dataset(lookup, ERROR_DIR, OUT_ERROR, "error")