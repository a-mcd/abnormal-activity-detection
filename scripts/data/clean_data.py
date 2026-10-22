import pandas as pd
import os
import re
from pathlib import Path

from abnormal_activity_detection.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR


"""
=====================================================================
DATA CLEANING SCRIPT – CASAS ADL DATASET
=====================================================================

This script processes the raw ADL datasets and removes invalid
sensor–message rows, producing cleaned versions of the datasets.

INPUT DIRECTORIES:
    - adl_noerror
    - adl_error

OUTPUT DIRECTORIES (created automatically):
    - adl_noerror_clean
    - adl_error_clean

---------------------------------------------------------------------
WHAT THIS SCRIPT DOES
---------------------------------------------------------------------
For each CSV file:
1. Normalises data:
    - sensor → lowercase
    - message → uppercase

2. Removes invalid rows where:
    - Sensor type is unknown
    - Message is not valid for the sensor type
    - analogue sensors (AD1*) do not contain numeric values

3. Applies jitter smoothing:
    - AD1-B uses an activity-specific base threshold
    - AD1-C uses an activity-specific base threshold
    - Each sensor has its own threshold scaling factor for higher values
    - Values within the effective threshold are grouped into runs and
      replaced by the run average

4. Preserves all other valid rows and original structure

5. Saves cleaned CSV with the same filename to the corresponding
   clean directory

---------------------------------------------------------------------
SUPPORTED SENSOR TYPES
---------------------------------------------------------------------
- Motion   (m*)
- Item     (i*)
- Door     (d*)
- Asterisk (phone events)
- analogue   (ad1*)

---------------------------------------------------------------------
VALID MESSAGES
---------------------------------------------------------------------
For non-analogue sensors:
    {ON, OFF, START, END, OPEN, CLOSE, PRESENT, ABSENT}

For analogue sensors:
    Any numeric value (float-convertible)

---------------------------------------------------------------------
OPTIONAL FILTERING
---------------------------------------------------------------------
- TARGET_ACTIVITY_CODE can be set to:
    "t1", "t2", "t3", "t4", "t5"
  to process a single activity only

---------------------------------------------------------------------
NOTES
---------------------------------------------------------------------
- This script removes invalid sensor-message pairs
- It smooths AD1-B and AD1-C analogue jitter using per-activity thresholds
  plus sensor-specific scaling for higher values
- It does NOT modify timestamps, dt values, or feature engineering
- Designed as a preprocessing step before modelling

=====================================================================
"""

# ============================================================
# CONFIG
# ============================================================
VALID_MESSAGES = {"ON", "OFF", "START", "END", "OPEN", "CLOSE", "PRESENT", "ABSENT"}

DIR_ERROR_IN = RAW_DATA_DIR / "adl_error"
DIR_NORMAL_IN = RAW_DATA_DIR / "adl_noerror"

DIR_ERROR_OUT = PROCESSED_DATA_DIR / "adl_error_clean"
DIR_NORMAL_OUT = PROCESSED_DATA_DIR / "adl_noerror_clean"

# Optional: set to "t1".."t5" to clean only one activity, or None for all
TARGET_ACTIVITY_CODE = None

ALL_ACTIVITY_CODES = {"t1", "t2", "t3", "t4", "t5"}

# Sensor names
AD1_B_SENSOR_NAME = "ad1-b"
AD1_C_SENSOR_NAME = "ad1-c"

# Per-activity base jitter thresholds
AD1_B_JITTER_THRESHOLD_BY_ACTIVITY = {
    "t1": 0.05,
    "t2": 0.05,
    "t3": 0.05,
    "t4": 0.05,
    "t5": 0.05,
}

AD1_C_JITTER_THRESHOLD_BY_ACTIVITY = {
    "t1": 0.03,
    "t2": 0.03,
    "t3": 0.03,
    "t4": 0.03,
    "t5": 0.03,
}

# Separate threshold scaling fractions for higher values
# effective_threshold = max(base_threshold, scale_fraction * run_avg)
AD1_B_THRESHOLD_SCALE_FRACTION = 0.20
AD1_C_THRESHOLD_SCALE_FRACTION = 0.10


# ============================================================
# HELPERS
# ============================================================
def norm_sensor(s: str) -> str:
    return str(s).strip().lower()


def norm_msg(m: str) -> str:
    return str(m).strip().upper()


def extract_activity_code(filename: str) -> str | None:
    m = re.search(r"\.(t[1-5])\.csv$", filename.lower())
    return m.group(1) if m else None


def get_sensor_type(sensor: str) -> str:
    sensor = norm_sensor(sensor)

    if sensor.startswith("m"):
        return "motion"
    elif sensor.startswith("i"):
        return "item"
    elif sensor.startswith("d"):
        return "door"
    elif sensor == "asterisk":
        return "asterisk"
    elif sensor.startswith("ad1"):
        return "analogue"
    else:
        return "unknown"


def is_known_message(sensor: str, msg: str) -> bool:
    sensor_type = get_sensor_type(sensor)
    msg = norm_msg(msg)

    if sensor_type == "analogue":
        try:
            float(msg)
            return True
        except Exception:
            return False

    if sensor_type in ("motion", "item", "door", "asterisk"):
        return msg in VALID_MESSAGES

    return False


def is_valid_row(sensor: str, message: str) -> bool:
    return get_sensor_type(sensor) != "unknown" and is_known_message(sensor, message)


def format_numeric_like_original(value: float, original_text: str) -> str:
    """
    Keep numeric output tidy while preserving reasonable precision.
    """
    original_text = str(original_text).strip()

    if "e" in original_text.lower():
        return f"{value:.10g}"

    if "." in original_text:
        decimals = len(original_text.split(".")[1])
        decimals = min(decimals, 10)
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")

    return f"{value:.10g}"


def get_activity_threshold(threshold_map: dict[str, float], activity_code: str, sensor_name: str) -> float:
    if activity_code not in threshold_map:
        raise ValueError(f"No threshold configured for {sensor_name} and activity {activity_code}")
    return threshold_map[activity_code]


def get_scaled_threshold(base_threshold: float, run_avg: float, scale_fraction: float) -> float:
    """
    Scale threshold upward for higher analogue values.
    """
    return max(base_threshold, scale_fraction * abs(run_avg))


def smooth_analogue_sensor_jitter(
    df: pd.DataFrame,
    sensor_name: str,
    base_threshold: float,
    scale_fraction: float,
) -> tuple[pd.DataFrame, int]:

    df = df.copy()
    changed_count = 0

    run_indices = []
    run_values = []

    def flush_run():
        nonlocal changed_count, run_indices, run_values

        if not run_indices:
            return

        run_avg = sum(run_values) / len(run_values)

        for idx in run_indices:
            raw_msg = df.at[idx, "message"]
            curr_value = float(raw_msg)

            if abs(curr_value - run_avg) > 1e-12:
                df.at[idx, "message"] = format_numeric_like_original(run_avg, raw_msg)
                changed_count += 1

        run_indices = []
        run_values = []

    for idx in df.index:
        if df.at[idx, "sensor"] != sensor_name:
            continue

        raw_msg = df.at[idx, "message"]

        try:
            curr_value = float(raw_msg)
        except Exception:
            continue

        if not run_values:
            run_indices = [idx]
            run_values = [curr_value]
            continue

        run_avg = sum(run_values) / len(run_values)
        effective_threshold = get_scaled_threshold(
            base_threshold=base_threshold,
            run_avg=run_avg,
            scale_fraction=scale_fraction,
        )

        if abs(curr_value - run_avg) <= effective_threshold:
            run_indices.append(idx)
            run_values.append(curr_value)
        else:
            flush_run()
            run_indices = [idx]
            run_values = [curr_value]

    flush_run()

    return df, changed_count


def clean_csv_file(input_path: Path, output_path: Path, activity_code: str) -> dict:
    df = pd.read_csv(input_path)

    required_cols = {"sensor", "message"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"{input_path.name} is missing required columns: {sorted(missing_cols)}")

    original_rows = len(df)

    # Normalize before validating
    df["sensor"] = df["sensor"].map(norm_sensor)
    df["message"] = df["message"].map(norm_msg)

    # Remove invalid rows first
    bad_mask = ~df.apply(lambda r: is_valid_row(r["sensor"], r["message"]), axis=1)
    removed_rows = int(bad_mask.sum())

    bad_rows = df.loc[bad_mask].copy()
    df_clean = df.loc[~bad_mask].copy()

    # Activity-specific base thresholds
    ad1_b_base_threshold = get_activity_threshold(
        AD1_B_JITTER_THRESHOLD_BY_ACTIVITY,
        activity_code,
        AD1_B_SENSOR_NAME,
    )
    ad1_c_base_threshold = get_activity_threshold(
        AD1_C_JITTER_THRESHOLD_BY_ACTIVITY,
        activity_code,
        AD1_C_SENSOR_NAME,
    )

    # Apply smoothing after validation
    df_clean, ad1_b_smoothed_count = smooth_analogue_sensor_jitter(
        df_clean,
        sensor_name=AD1_B_SENSOR_NAME,
        base_threshold=ad1_b_base_threshold,
        scale_fraction=AD1_B_THRESHOLD_SCALE_FRACTION,
    )

    df_clean, ad1_c_smoothed_count = smooth_analogue_sensor_jitter(
        df_clean,
        sensor_name=AD1_C_SENSOR_NAME,
        base_threshold=ad1_c_base_threshold,
        scale_fraction=AD1_C_THRESHOLD_SCALE_FRACTION,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    return {
        "file": input_path.name,
        "activity_code": activity_code,
        "input_rows": original_rows,
        "removed_rows": removed_rows,
        "output_rows": len(df_clean),
        "bad_rows": bad_rows,
        "ad1_b_base_threshold": ad1_b_base_threshold,
        "ad1_c_base_threshold": ad1_c_base_threshold,
        "ad1_b_scale_fraction": AD1_B_THRESHOLD_SCALE_FRACTION,
        "ad1_c_scale_fraction": AD1_C_THRESHOLD_SCALE_FRACTION,
        "ad1_b_smoothed_count": ad1_b_smoothed_count,
        "ad1_c_smoothed_count": ad1_c_smoothed_count,
    }


def clean_directory(input_dir: Path, output_dir: Path, target_activity_code: str | None = None) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for filename in sorted(os.listdir(input_dir)):
        if not filename.lower().endswith(".csv"):
            continue

        activity_code = extract_activity_code(filename)
        if activity_code is None:
            print(f"Skipping {filename} (could not detect activity code)")
            continue

        if target_activity_code is not None and activity_code != target_activity_code:
            continue

        input_path = input_dir / filename
        output_path = output_dir / filename

        try:
            result = clean_csv_file(input_path, output_path, activity_code=activity_code)
            results.append(result)

            if (
                result["removed_rows"] > 0
                or result["ad1_b_smoothed_count"] > 0
                or result["ad1_c_smoothed_count"] > 0
            ):
                print(f"\nProcessed: {filename}")
                print(f"  Activity:              {result['activity_code']}")
                print(f"  Input rows:            {result['input_rows']}")
                print(f"  Removed rows:          {result['removed_rows']}")
                print(f"  AD1-B base threshold:  {result['ad1_b_base_threshold']}")
                print(f"  AD1-B scale fraction:  {result['ad1_b_scale_fraction']}")
                print(f"  AD1-B smoothed:        {result['ad1_b_smoothed_count']}")
                print(f"  AD1-C base threshold:  {result['ad1_c_base_threshold']}")
                print(f"  AD1-C scale fraction:  {result['ad1_c_scale_fraction']}")
                print(f"  AD1-C smoothed:        {result['ad1_c_smoothed_count']}")
                print(f"  Output rows:           {result['output_rows']}")

                if result["removed_rows"] > 0:
                    cols_to_show = [c for c in ["date", "time", "sensor", "message"] if c in result["bad_rows"].columns]
                    print("  Removed invalid rows:")
                    print(result["bad_rows"][cols_to_show].to_string(index=False))

        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    return results


def print_invalid_row_summary(results: list[dict], label: str):
    total_files = len(results)
    total_input_rows = sum(r["input_rows"] for r in results)
    total_removed_rows = sum(r["removed_rows"] for r in results)
    total_output_rows = sum(r["output_rows"] for r in results)

    print("\n" + "=" * 70)
    print(f"INVALID ROW REMOVAL SUMMARY: {label}")
    print("=" * 70)
    print(f"Files processed:   {total_files}")
    print(f"Input rows:        {total_input_rows}")
    print(f"Removed rows:      {total_removed_rows}")
    print(f"Output rows:       {total_output_rows}")

    removed_files = [r for r in results if r["removed_rows"] > 0]
    print(f"Files with invalid row removals: {len(removed_files)}")

    if removed_files:
        summary_df = pd.DataFrame(
            [
                {
                    "file": r["file"],
                    "activity_code": r["activity_code"],
                    "input_rows": r["input_rows"],
                    "removed_rows": r["removed_rows"],
                    "output_rows": r["output_rows"],
                }
                for r in removed_files
            ]
        ).sort_values(["removed_rows", "file"], ascending=[False, True])

        print("\nFiles with invalid rows removed:")
        print(summary_df.to_string(index=False))


def print_smoothing_summary(results: list[dict], label: str):
    total_files = len(results)
    total_ad1_b_smoothed = sum(r["ad1_b_smoothed_count"] for r in results)
    total_ad1_c_smoothed = sum(r["ad1_c_smoothed_count"] for r in results)

    print("\n" + "=" * 70)
    print(f"SMOOTHING SUMMARY: {label}")
    print("=" * 70)
    print(f"Files processed:       {total_files}")
    print(f"AD1-B values smoothed: {total_ad1_b_smoothed}")
    print(f"AD1-C values smoothed: {total_ad1_c_smoothed}")

    smoothed_files = [
        r for r in results
        if r["ad1_b_smoothed_count"] > 0 or r["ad1_c_smoothed_count"] > 0
    ]
    print(f"Files with smoothing applied: {len(smoothed_files)}")

    if smoothed_files:
        summary_df = pd.DataFrame(
            [
                {
                    "file": r["file"],
                    "activity_code": r["activity_code"],
                    "ad1_b_base_threshold": r["ad1_b_base_threshold"],
                    "ad1_b_scale_fraction": r["ad1_b_scale_fraction"],
                    "ad1_b_smoothed_count": r["ad1_b_smoothed_count"],
                    "ad1_c_base_threshold": r["ad1_c_base_threshold"],
                    "ad1_c_scale_fraction": r["ad1_c_scale_fraction"],
                    "ad1_c_smoothed_count": r["ad1_c_smoothed_count"],
                }
                for r in smoothed_files
            ]
        ).sort_values(
            ["ad1_b_smoothed_count", "ad1_c_smoothed_count", "file"],
            ascending=[False, False, True]
        )

        print("\nFiles with smoothing applied:")
        print(summary_df.to_string(index=False))


def main():
    if TARGET_ACTIVITY_CODE is not None and TARGET_ACTIVITY_CODE not in ALL_ACTIVITY_CODES:
        raise ValueError(f"TARGET_ACTIVITY_CODE must be one of {sorted(ALL_ACTIVITY_CODES)} or None")

    print("Cleaning normal dataset...")
    print(f"Input:  {DIR_NORMAL_IN}")
    print(f"Output: {DIR_NORMAL_OUT}")
    normal_results = clean_directory(
        input_dir=DIR_NORMAL_IN,
        output_dir=DIR_NORMAL_OUT,
        target_activity_code=TARGET_ACTIVITY_CODE,
    )

    print("\nCleaning error dataset...")
    print(f"Input:  {DIR_ERROR_IN}")
    print(f"Output: {DIR_ERROR_OUT}")
    error_results = clean_directory(
        input_dir=DIR_ERROR_IN,
        output_dir=DIR_ERROR_OUT,
        target_activity_code=TARGET_ACTIVITY_CODE,
    )

    print_invalid_row_summary(normal_results, "adl_noerror_clean")
    print_smoothing_summary(normal_results, "adl_noerror_clean")

    print_invalid_row_summary(error_results, "adl_error_clean")
    print_smoothing_summary(error_results, "adl_error_clean")

    print("\nDone.")
    print(f"Clean normal CSVs saved to: {DIR_NORMAL_OUT}")
    print(f"Clean error CSVs saved to:  {DIR_ERROR_OUT}")


if __name__ == "__main__":
    main()