#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd

from abnormal_activity_detection.paths import RESULTS_DIR

TARGET_SENSOR = "AD1-C"
ACTIVITIES = ["t1", "t2", "t3", "t4", "t5"]

# --------------------------------------------------
# Bucket function
# --------------------------------------------------
def duration_bucket(seconds: int) -> str:
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


def extract_activity_code(filename: str) -> str | None:
    m = re.search(r"(?i)(?:^|[._-])(t[1-5])(?:[._-]|$)", filename)
    return m.group(1).lower() if m else None


def find_timestamp_column(df: pd.DataFrame) -> str | None:
    candidates = ["timestamp", "datetime", "ts", "time_stamp", "date_time"]
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower_map:
            return lower_map[name]
    return None


def parse_timestamps(df: pd.DataFrame) -> pd.Series:
    cols_lower = {c.lower(): c for c in df.columns}

    if "date" in cols_lower and "time" in cols_lower:
        date_col = cols_lower["date"]
        time_col = cols_lower["time"]
        s = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        return pd.to_datetime(s, errors="raise", format="mixed")

    ts_col = find_timestamp_column(df)
    if ts_col is not None:
        return pd.to_datetime(df[ts_col].astype(str).str.strip(), errors="raise", format="mixed")

    raise ValueError("No usable timestamp columns found.")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main(input_dir: Path, output_csv: Path) -> None:
    # initialise count table
    counts = {
        bucket: {a: 0 for a in ACTIVITIES}
        for bucket in [
            "DUR_0_5",
            "DUR_6_15",
            "DUR_16_45",
            "DUR_46_100",
            "DUR_101_PLUS",
        ]
    }

    for csv_path in sorted(input_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        activity = extract_activity_code(csv_path.name)
        if activity not in ACTIVITIES:
            continue

        cols_lower = {c.lower(): c for c in df.columns}
        if "sensor" not in cols_lower:
            continue

        sensor_col = cols_lower["sensor"]
        mask = df[sensor_col].astype(str).str.strip().eq(TARGET_SENSOR)
        if not mask.any():
            continue

        df_hit = df.loc[mask].copy()

        try:
            df_hit["_ts"] = parse_timestamps(df_hit)
        except Exception:
            continue

        duration_sec = int((df_hit["_ts"].max() - df_hit["_ts"].min()).total_seconds())

        bucket = duration_bucket(duration_sec)

        counts[bucket][activity] += 1

    # --------------------------------------------------
    # Build output table
    # --------------------------------------------------
    out_rows = []

    bucket_order = [
        "DUR_0_5",
        "DUR_6_15",
        "DUR_16_45",
        "DUR_46_100",
        "DUR_101_PLUS",
    ]

    for bucket in bucket_order:
        row = {
            "sensor": TARGET_SENSOR,
            "message": bucket,
        }
        for a in ACTIVITIES:
            row[f"{a}_count"] = counts[bucket][a]
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    print(f"Wrote duration bucket counts to: {output_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AD1-B duration bucket counter.")
    ap.add_argument("input_dir", type=Path, help="Directory containing CSV files")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "beta_dst" / "ad1b_duration_buckets.csv")
    args = ap.parse_args()

    main(args.input_dir, args.out)
