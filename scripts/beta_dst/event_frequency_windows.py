#!/usr/bin/env python3
"""
Window-based sensor/message counts for CASAS Kyoto CSVs (30-second windows).

Changes vs episode-based:
- Each CSV is divided into fixed windows of WINDOW_SECONDS (default 30s)
- Within each window, each (sensor,message) is counted at most once
- Output counts = number of windows (across all CSVs of that activity) that contained the (sensor,message)
  Example: if "D01,CLOSE" occurs in 3 separate 30s windows across t1 files => t1_count = 3
  (even if it fires multiple times inside any one window)

analogue sensors are binned into BIN_k messages using provided edges.
"""

from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import os

from abnormal_activity_detection.paths import RAW_DATA_DIR, RESULTS_DIR



# -------------------------
# Bin edges (provided)
# -------------------------
EDGES_WATER = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)          # AD1-B / AD1-C
EDGES_BURNER = (2.85, 2.95, 3.05, 3.15, 3.25, 3.35)              # AD1-A

ACTIVITIES = ["t1", "t2", "t3", "t4", "t5"]


def extract_activity_code(filename: str) -> str | None:
    """
    Extract activity code from filename.
    Supports: p03.t1.csv, p03_t1.csv, something-t2-extra.csv, etc.
    """
    m = re.search(r'(?i)(?:^|[._-])(t[1-5])(?:[._-]|$)', filename)
    return m.group(1).lower() if m else None


def read_casas_csv(csv_path: Path) -> pd.DataFrame:
    """
    Reads a CASAS-style CSV, returning columns:
      - ts (datetime64[ns])
      - sensor (str)
      - message (str)

    Supports:
    Headered files with columns including date/time/timestamp + sensor/message
    """

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    combined = (
        df["date"].astype(str).str.strip() + " " +
        df["time"].astype(str).str.strip()
    )

    ts = pd.to_datetime(combined, format="mixed", errors="raise")

    # Sensor/message
    if "sensor" not in df.columns or "message" not in df.columns:
        # sometimes 'msg'
        if "sensor" in df.columns and "msg" in df.columns:
            df = df.rename(columns={"msg": "message"})
        else:
            raise ValueError("No sensor/message columns found")

    out = pd.DataFrame({
        "ts": ts,
        "sensor": df["sensor"].astype(str).str.strip(),
        "message": df["message"].astype(str).str.strip(),
    })
    out = out[out["ts"].notna()].copy()
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


def normalize_pair(sensor: str, message: str) -> tuple[str, str]:
    s = str(sensor).strip()
    m = str(message).strip()
    if not m.startswith("BIN_"):
        m = m.upper()
    return s, m


def build_window_counts(input_dir: Path, window_seconds: int = 30) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      sensor, message, t1_count..t5_count
    where each count = number of 30s windows (across all files of that activity)
    that contain (sensor,message) at least once.
    """
    counts = defaultdict(lambda: {a: 0 for a in ACTIVITIES})

    csv_files = sorted([p for p in input_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in: {input_dir}")

    for csv_path in csv_files:
        act = extract_activity_code(csv_path.name)
        if act not in ACTIVITIES:
            continue

        df = read_casas_csv(csv_path)
        if df.empty:
            continue

        df = bin_analogue_rows(df)

        # Create 30s window buckets relative to file start:
        # window_id = floor((ts - min_ts) / window_seconds)
        t0 = df["ts"].min()
        dt_sec = (df["ts"] - t0).dt.total_seconds()
        df["window_id"] = (dt_sec // window_seconds).astype(int)

        # For each window: count each pair once
        # Approach: group by window_id, take unique pairs, increment global counts
        for _, wdf in df.groupby("window_id", sort=False):
            pairs = set(normalize_pair(s, m) for s, m in zip(wdf["sensor"], wdf["message"]))
            for pair in pairs:
                counts[pair][act] += 1

    rows = []
    for (sensor, message), per_act in counts.items():
        rows.append({
            "sensor": sensor,
            "message": message,
            "t1_count": per_act["t1"],
            "t2_count": per_act["t2"],
            "t3_count": per_act["t3"],
            "t4_count": per_act["t4"],
            "t5_count": per_act["t5"],
        })

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise RuntimeError(
            f"No window counts produced. Check filenames contain t1..t5 and are in {input_dir}"
        )

    out_df = out_df.sort_values(["sensor", "message"]).reset_index(drop=True)
    return out_df


from math import ceil
from collections import Counter

def build_window_event_log(input_dir: Path, window_seconds: int = 30) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (file, window_id), including:
      - filename
      - activity
      - window_id
      - window_start
      - window_end
      - events_in_window (e.g. "M01|ON:2;D01|OPEN:1")
      - n_unique_pairs
      - n_total_events

    Includes windows with no events between min(ts) and max(ts).
    """
    rows = []

    csv_files = sorted([p for p in input_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in: {input_dir}")

    for csv_path in csv_files:
        act = extract_activity_code(csv_path.name)
        if act not in ACTIVITIES:
            continue

        df = read_casas_csv(csv_path)
        if df.empty:
            continue

        # Apply analogue binning so the "events within each window" matches your counting logic
        df = bin_analogue_rows(df)

        # Normalize pairs (same as counting)
        df["sensor_n"] = df["sensor"].astype(str).str.strip()
        df["message_n"] = df["message"].astype(str).str.strip()
        df.loc[~df["message_n"].str.startswith("BIN_", na=False), "message_n"] = df.loc[
            ~df["message_n"].str.startswith("BIN_", na=False), "message_n"
        ].str.upper()

        t0 = df["ts"].min()
        t_last = df["ts"].max()

        dt_sec = (df["ts"] - t0).dt.total_seconds()
        df["window_id"] = (dt_sec // window_seconds).astype(int)

        # Determine full window range (include empty windows)
        total_seconds = (t_last - t0).total_seconds()
        n_windows = int(ceil((total_seconds + 1e-9) / window_seconds))  # +eps for exact boundaries
        max_window_id = max(0, n_windows - 1)

        # Pre-group events by window_id for quick lookup
        grouped = {wid: wdf for wid, wdf in df.groupby("window_id", sort=False)}

        for wid in range(0, max_window_id + 1):
            w_start = t0 + pd.to_timedelta(wid * window_seconds, unit="s")
            w_end_nominal = w_start + pd.to_timedelta(window_seconds, unit="s")
            # Clip final window end to last timestamp (optional, but usually clearer)
            w_end = min(w_end_nominal, t_last)

            if wid in grouped:
                wdf = grouped[wid]
                pair_strings = (wdf["sensor_n"] + "|" + wdf["message_n"]).tolist()
                counts = Counter(pair_strings)

                # Sort for stable output
                parts = [f"{k}:{counts[k]}" for k in sorted(counts.keys())]
                events_str = ";".join(parts)

                n_unique = len(counts)
                n_total = sum(counts.values())
            else:
                events_str = ""
                n_unique = 0
                n_total = 0

            rows.append({
                "filename": csv_path.name,
                "activity": act,
                "window_id": wid,
                "window_start": w_start,
                "window_end": w_end,
                "events_in_window": events_str,
                "n_unique_pairs": n_unique,
                "n_total_events": n_total,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No window event log produced. Check filenames contain t1..t5 and are in {input_dir}")

    # Nice ordering
    out = out.sort_values(["filename", "window_id"]).reset_index(drop=True)
    return out


def main():

    window_seconds = 30

    input_dir = RAW_DATA_DIR / "adl_noerror"

    out_dir = RESULTS_DIR / "beta_dst"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv_counts = out_dir / "event_stats_30s_windows.csv"
    out_csv_windows = out_dir / "window_event_log_30s.csv"

    df_out = build_window_counts(input_dir, window_seconds)
    df_out.to_csv(out_csv_counts, index=False)

    print(f"Wrote: {out_csv_counts} ({len(df_out)} rows)")
    print("Columns:", ", ".join(df_out.columns))

    df_log = build_window_event_log(input_dir, window_seconds)
    df_log.to_csv(out_csv_windows, index=False)
    print(f"Wrote: {out_csv_windows} ({len(df_log)} rows)")
    print("Columns:", ", ".join(df_log.columns))


if __name__ == "__main__":
    main()
