#!/usr/bin/env python3
"""
Episode-based sensor/message counts for CASAS Kyoto CSVs.

What it does
1) Reads all CSV files in a directory
2) Infers activity code from filename (e.g., ".t1." or "_t1_" etc.)
3) For each (sensor, message) pair, counts in how many *files/episodes* it appears per activity
   - Example: if "D01,CLOSE" appears in 3 different t1 CSV files => t1_count = 3
   - Multiple appearances within the same file count only once
4) analogue sensors are binned:
   - AD1-A uses EDGES_BURNER => message becomes BIN_0..BIN_6
   - AD1-B / AD1-C use EDGES_WATER => message becomes BIN_0..BIN_7
5) Writes output CSV: sensor, message, t1_count..t5_count
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

from abnormal_activity_detection.paths import RESULTS_DIR

# -------------------------
# Bin edges (provided)
# -------------------------
EDGES_WATER = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)          # AD1-B / AD1-C
EDGES_BURNER = (2.85, 2.95, 3.05, 3.15, 3.25, 3.35)              # AD1-A

ACTIVITIES = ["t1", "t2", "t3", "t4", "t5"]


# -------------------------
# Helpers
# -------------------------
def extract_activity_code(filename: str) -> str | None:
    """
    Extracts activity code from filename.
    Supports patterns like:
      p03.t1.csv
      p03_t1.csv
      something.t2.extra.csv
    """
    m = re.search(r'(?i)(?:^|[._-])(t[1-5])(?:[._-]|$)', filename)
    return m.group(1).lower() if m else None


def read_event_csv(csv_path: Path) -> pd.DataFrame:
    """
    Reads a CASAS-style CSV assumed to have at least:
      date,time,sensor,message
    Some files might have headers or extra columns.
    We normalize to columns: sensor, message, (optional) value.
    """
    # Try headered first; if that fails, read as headerless
    df = None
    try:
        df = pd.read_csv(csv_path)
        # if it doesn't have sensor/message, fall back
        if not {"sensor", "message"}.issubset(set(c.lower() for c in df.columns)):
            raise ValueError("missing sensor/message headers")
        # normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"msg": "message"})
        df = df[["sensor", "message"]].copy()
        df["sensor"] = df["sensor"].astype(str).str.strip()
        df["message"] = df["message"].astype(str).str.strip()
        return df
    except Exception:
        # headerless: expect at least 4 columns (date, time, sensor, message)
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] < 4:
            raise ValueError(f"{csv_path.name}: expected >=4 columns (date,time,sensor,message)")
        df = df.iloc[:, :4].copy()
        df.columns = ["date", "time", "sensor", "message"]
        df["sensor"] = df["sensor"].astype(str).str.strip()
        df["message"] = df["message"].astype(str).str.strip()
        return df[["sensor", "message"]]


def bin_analogue_messages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts analogue sensor rows into (sensor, BIN_k) rows.
    Assumes analogue rows have sensor in {"AD1-A","AD1-B","AD1-C"} and message is numeric.
    Non-numeric rows for these sensors are ignored.
    """
    out_rows = []

    # Keep non-analogue rows as-is
    non_analogue = df[~df["sensor"].isin(["AD1-A", "AD1-B", "AD1-C"])].copy()
    out_rows.append(non_analogue)

    # Handle each analogue sensor separately
    for sensor in ["AD1-A", "AD1-B", "AD1-C"]:
        sdf = df[df["sensor"] == sensor].copy()
        if sdf.empty:
            continue

        # message holds numeric values in your data (e.g., "0.0322661" or "2.84989")
        vals = pd.to_numeric(sdf["message"], errors="coerce")
        sdf = sdf[vals.notna()].copy()
        if sdf.empty:
            continue
        vals = vals.loc[sdf.index].to_numpy(dtype=float)

        edges = EDGES_BURNER if sensor == "AD1-A" else EDGES_WATER

        # bins: 0..len(edges) where:
        # 0 => <= edges[0]
        # k => (edges[k-1], edges[k]] for k=1..len(edges)-1
        # len(edges) => > edges[-1]
        # We implement using numpy digitize with right=True to match <= on upper bound.
        bin_idx = np.digitize(vals, bins=np.array(edges, dtype=float), right=True)  # returns 0..len(edges)

        sdf["message"] = [f"BIN_{k}" for k in bin_idx.tolist()]
        out_rows.append(sdf[["sensor", "message"]])

    return pd.concat(out_rows, ignore_index=True)


def normalize_pair(sensor: str, message: str) -> tuple[str, str]:
    """
    Normalizes sensor IDs and messages.
    - Sensor: strip, keep as-is (but remove accidental whitespace)
    - Message: upper-case typical CASAS tokens (ON/OFF/OPEN/CLOSE/PRESENT/ABSENT/START_INSTRUCT/STOP_INSTRUCT)
    - BIN_x left as-is
    """
    s = str(sensor).strip()
    m = str(message).strip()
    # keep BIN_* as is
    if not m.startswith("BIN_"):
        m = m.upper()
    return s, m


# -------------------------
# Main computation
# -------------------------
def build_episode_counts(input_dir: Path) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      sensor, message, t1_count..t5_count
    where each count is number of files (episodes) of that activity that contained the (sensor,message).
    """
    # counts[(sensor,message)][activity] = count
    counts = defaultdict(lambda: {a: 0 for a in ACTIVITIES})

    csv_files = sorted([p for p in input_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in: {input_dir}")

    for csv_path in csv_files:
        act = extract_activity_code(csv_path.name)
        if act not in ACTIVITIES:
            # Skip files without recognizable activity tag
            continue

        df = read_event_csv(csv_path)
        df = bin_analogue_messages(df)

        # Episode-based: unique pairs per file
        pairs = set()
        for s, m in zip(df["sensor"], df["message"]):
            pairs.add(normalize_pair(s, m))

        # Increment counts once per pair for this file's activity
        for pair in pairs:
            counts[pair][act] += 1

    # Build output table
    rows = []
    for (sensor, message), per_act in counts.items():
        row = {
            "sensor": sensor,
            "message": message,
            "t1_count": per_act["t1"],
            "t2_count": per_act["t2"],
            "t3_count": per_act["t3"],
            "t4_count": per_act["t4"],
            "t5_count": per_act["t5"],
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise RuntimeError(
            f"No episode counts produced. Check that filenames contain t1..t5 and are in {input_dir}"
        )

    # Nice ordering
    out_df = out_df.sort_values(["sensor", "message"]).reset_index(drop=True)
    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing CASAS CSV files")
    ap.add_argument("--out_csv", default=str(RESULTS_DIR / "beta_dst" / "event_stats.csv"), help="Output CSV path")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()

    df_out = build_episode_counts(input_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} ({len(df_out)} rows)")
    print("Columns:", ", ".join(df_out.columns))


if __name__ == "__main__":
    main()
