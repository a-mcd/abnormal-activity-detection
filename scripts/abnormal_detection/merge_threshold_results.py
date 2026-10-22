from pathlib import Path
import pandas as pd

from abnormal_activity_detection.paths import RESULTS_DIR

def combine_activity_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (activity, threshold) pair, combine the 'noerror' and 'error' rows
    into a single row, then remove the result_type column.
    """
    # Remove leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    required_columns = [
        "activity",
        "threshold",
        "Pred_Normal-Actual_Normal",
        "Pred_Anomaly-Actual_Normal",
        "Pred_Normal-Actual_Anomaly",
        "Pred_Anomaly-Actual_Anomaly",
        "Actual_Normal Percent",
        "Actual_Anomaly Percent",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop result_type before final output, but use it if present only for grouping logic
    group_cols = ["activity", "threshold"]

    def first_non_null(series):
        non_null = series.dropna()
        return non_null.iloc[0] if not non_null.empty else pd.NA

    merged = (
        df.groupby(group_cols, as_index=False)
        .agg({
            "Pred_Normal-Actual_Normal": "sum",
            "Pred_Anomaly-Actual_Normal": "sum",
            "Pred_Normal-Actual_Anomaly": "sum",
            "Pred_Anomaly-Actual_Anomaly": "sum",
            "Actual_Normal Percent": first_non_null,
            "Actual_Anomaly Percent": first_non_null,
        })
    )

    return merged


def merge_csv_files(input_dir: str, output_file: str = "merged.csv") -> None:
    """
    Read all CSV files in a directory, combine matching activity rows inside each file,
    add the source filename, and merge them into one CSV.
    """
    input_path = Path(input_dir)
    all_dataframes = []

    for file_path in sorted(input_path.glob("*.csv")):
        try:
            df = pd.read_csv(file_path)

            # Combine noerror/error rows into one row per activity+threshold
            df = combine_activity_rows(df)

            # Add source filename
            df["source_file"] = file_path.name

            all_dataframes.append(df)

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    if not all_dataframes:
        print("No CSV files found.")
        return

    merged_df = pd.concat(all_dataframes, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"Merged file saved as: {output_file}")


if __name__ == "__main__":

    directory_path = RESULTS_DIR / "abnormal_detection" / "autoencoder" / "t-all" / "autoencoder_tuning_all_in_one"
    output_path = directory_path / "merged.csv"
    merge_csv_files(directory_path, output_path)