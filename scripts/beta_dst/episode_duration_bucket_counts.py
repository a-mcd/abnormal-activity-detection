#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd

from abnormal_activity_detection.paths import RESULTS_DIR


# -------------------------------------------------
# Activity extraction from filename
# -------------------------------------------------
def extract_activity_code(filename: str) -> str | None:
    m = re.search(r"(?i)(?:^|[._-])(t[1-5])(?:[._-]|$)", filename)
    return m.group(1).lower() if m else None


# -------------------------------------------------
# Timestamp helpers (same logic as your original)
# -------------------------------------------------
def find_timestamp_column(df: pd.DataFrame) -> str | None:
    candidates = ["timestamp", "datetime", "ts", "time_stamp", "date_time"]
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    return None


def parse_timestamps(df: pd.DataFrame) -> pd.Series:
    cols_lower = {c.lower(): c for c in df.columns}

    # Case 1: separate date + time
    if "date" in cols_lower and "time" in cols_lower:
        date_col = cols_lower["date"]
        time_col = cols_lower["time"]
        s = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        return pd.to_datetime(s, errors="raise", format="mixed")

    # Case 2: single timestamp column
    ts_col = find_timestamp_column(df)
    if ts_col is not None:
        return pd.to_datetime(df[ts_col].astype(str).str.strip(), errors="raise", format="mixed")

    raise ValueError("No usable timestamp columns found.")


# -------------------------------------------------
# Duration bucket function (EDIT IF NEEDED)
# -------------------------------------------------
def assign_bucket(seconds: float) -> str:
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


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(input_dir: Path, output_csv: Path) -> None:
    rows = []

    for csv_path in sorted(input_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        activity = extract_activity_code(csv_path.name)

        try:
            ts = parse_timestamps(df).dropna()
        except Exception:
            continue

        if ts.empty:
            continue

        duration_seconds = int((ts.max() - ts.min()).total_seconds())
        bucket = assign_bucket(duration_seconds)

        rows.append(
            {
                "filename": csv_path.name,
                "activity": activity,
                "duration_seconds": duration_seconds,
                "duration_bucket": bucket,
            }
        )

    if not rows:
        print("No valid files processed.")
        return

    df_all = pd.DataFrame(rows)

    # -------------------------------------------------
    # Count per activity per bucket
    # -------------------------------------------------
    summary = (
        df_all.groupby(["duration_bucket", "activity"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("duration_bucket")
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(f"Wrote bucket summary to: {output_csv}")


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Count how many of each activity fall into each duration bucket (from raw CSV directory)."
    )
    ap.add_argument("input_dir", type=Path, help="Directory containing CSV files")
    ap.add_argument(
        "--out",
        type=Path,
        default=RESULTS_DIR / "beta_dst" / "duration_bucket_activity_counts.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    main(args.input_dir, args.out)