import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
    )
from sklearn.preprocessing import StandardScaler
import math


PREDICTION_BASE_COLS = [
    "filename",
    "participant_id",
    "activity_code",
    "activity_index",
    "dataset",
]


# -------------------------
# Model / evaluation helpers
# -------------------------

def print_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\n{title}")
    print("Rows = Actual, Columns = Predicted\n")
    print(f"{'':>20} {'Pred Normal':>15} {'Pred Anomaly':>15}")
    print(f"{'Actual Normal':>20} {cm[0, 0]:>15} {cm[0, 1]:>15}")
    print(f"{'Actual Anomaly':>20} {cm[1, 0]:>15} {cm[1, 1]:>15}")


def compute_basic_metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = cm.sum()
    accuracy = (tn + tp) / total if total else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    balanced_accuracy = (specificity + recall) / 2.0

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "recall_anomaly": float(recall),
        "precision_anomaly": float(precision),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "balanced_accuracy": float(balanced_accuracy),
    }


def build_confusion_matrix_df(
    y_true,
    y_pred,
    index=("Actual_Normal", "Actual_Anomaly"),
    columns=("Pred_Normal", "Pred_Anomaly"),
):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=list(index), columns=list(columns))
    return cm, cm_df


def save_confusion_outputs(cm_df, csv_fp, txt_fp, title):
    
    cm_df.to_csv(csv_fp)
    with open(txt_fp, "w") as f:
        f.write(f"{title}\n\n")
        f.write(cm_df.to_string())


def make_prediction_df(
    df,
    y_true,
    y_pred,
    decision_scores=None,
    fold=None,
    true_col="y_true",
    pred_col="y_pred",
    add_pred_label=True,
):
    out = df[PREDICTION_BASE_COLS].copy()
    if fold is not None:
        out["fold"] = fold
    out[true_col] = y_true
    out[pred_col] = y_pred
    if add_pred_label:
        out["pred_label"] = np.where(out[pred_col] == 1, "anomaly", "normal")
    if decision_scores is not None:
        out["decision_score"] = decision_scores
    return out


def build_classification_report(y_true, y_pred):
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    )


def save_confusion_matrix_files(cm_df, csv_fp, txt_fp, title):
    cm_df.to_csv(csv_fp)

    with open(txt_fp, "w") as f:
        f.write(f"{title}\n")
        f.write("Rows = Actual, Columns = Predicted\n\n")
        f.write(cm_df.to_string())


def save_confusion_matrices_by_activity(
    preds_df: pd.DataFrame,
    output_dir,
    prefix: str,
):
    """
    Save one confusion matrix per activity.
    Expects preds_df to contain:
        activity_code, y_true, y_pred
    """
    if preds_df is None or preds_df.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for activity_code, g in preds_df.groupby("activity_code", dropna=False):
        activity_code = str(activity_code)

        y_true = g["y_true"].to_numpy(dtype=int)
        y_pred = g["y_pred"].to_numpy(dtype=int)

        _, cm_df = build_confusion_matrix_df(y_true, y_pred)

        csv_fp = output_dir / f"{prefix}_{activity_code}_confusion_matrix.csv"
        txt_fp = output_dir / f"{prefix}_{activity_code}_confusion_matrix.txt"

        save_confusion_matrix_files(
            cm_df,
            csv_fp,
            txt_fp,
            title=f"{prefix} - {activity_code} Confusion Matrix",
        )


def save_confusion_matrices_by_activity_one_file(
    preds_df: pd.DataFrame,
    csv_fp,
    txt_fp=None,
    title_prefix: str = "",
):
    """
    Save all per-activity confusion matrices into one CSV and optionally one TXT file.

    Expects preds_df to contain:
        activity_code, y_true, y_pred
    """
    if preds_df is None or preds_df.empty:
        return

    rows = []
    txt_blocks = []

    for activity_code, g in preds_df.groupby("activity_code", dropna=False):
        activity_code = str(activity_code)

        y_true = g["y_true"].to_numpy(dtype=int)
        y_pred = g["y_pred"].to_numpy(dtype=int)

        _, cm_df = build_confusion_matrix_df(y_true, y_pred)

        print(cm_df)

        # CSV rows
        for actual_label, row in cm_df.iterrows():
            rows.append(
                {
                    "activity_code": activity_code,
                    "actual_label": actual_label,
                    "pred_normal": int(row["Pred_Normal"]),
                    "pred_anomaly": int(row["Pred_Anomaly"]),
                }
            )

        # TXT block
        block_title = f"{title_prefix} {activity_code} Confusion Matrix".strip()
        txt_blocks.append(block_title)
        txt_blocks.append("Rows = Actual, Columns = Predicted\n")
        txt_blocks.append(cm_df.to_string())
        txt_blocks.append("\n")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(csv_fp, index=False)

    if txt_fp is not None:
        with open(txt_fp, "w") as f:
            f.write("\n\n".join(txt_blocks))


def build_activity_confusion_summary_table(
    preds_df: pd.DataFrame,
    threshold_col: str = "threshold",
) -> pd.DataFrame:
    """
    Build one summary row per activity.

    Expects preds_df to contain:
        activity_code, y_true, y_pred
    Optionally:
        threshold

    Output columns:
        activity
        threshold
        Pred_Normal-Actual_Normal
        Pred_Anomaly-Actual_Normal
        Pred_Normal-Actual_Anomaly
        Pred_Anomaly-Actual_Anomaly
        Actual_Normal Percent
        Actual_Anomaly Percent
    """
    if preds_df is None or preds_df.empty:
        return pd.DataFrame(columns=[
            "activity",
            "threshold",
            "Pred_Normal-Actual_Normal",
            "Pred_Anomaly-Actual_Normal",
            "Pred_Normal-Actual_Anomaly",
            "Pred_Anomaly-Actual_Anomaly",
            "Actual_Normal Percent",
            "Actual_Anomaly Percent",
        ])

    rows = []

    for activity_code, g in preds_df.groupby("activity_code", dropna=False):
        activity_code = str(activity_code)

        y_true = g["y_true"].to_numpy(dtype=int)
        y_pred = g["y_pred"].to_numpy(dtype=int)

        # confusion entries
        pred_normal_actual_normal = int(((y_true == 0) & (y_pred == 0)).sum())
        pred_anomaly_actual_normal = int(((y_true == 0) & (y_pred == 1)).sum())
        pred_normal_actual_anomaly = int(((y_true == 1) & (y_pred == 0)).sum())
        pred_anomaly_actual_anomaly = int(((y_true == 1) & (y_pred == 1)).sum())

        n_actual_normal = int((y_true == 0).sum())
        n_actual_anomaly = int((y_true == 1).sum())

        actual_normal_percent = (
            float(pred_normal_actual_normal / n_actual_normal)
            if n_actual_normal > 0 else np.nan
        )
        actual_anomaly_percent = (
            float(pred_anomaly_actual_anomaly / n_actual_anomaly)
            if n_actual_anomaly > 0 else np.nan
        )

        if threshold_col in g.columns and g[threshold_col].notna().any():
            unique_thresholds = sorted(g[threshold_col].dropna().unique().tolist())
            if len(unique_thresholds) == 1:
                threshold_value = unique_thresholds[0]
            else:
                threshold_value = ",".join(str(x) for x in unique_thresholds)
        else:
            threshold_value = np.nan

        rows.append({
            "activity": activity_code,
            "threshold": threshold_value,
            "Pred_Normal-Actual_Normal": pred_normal_actual_normal,
            "Pred_Anomaly-Actual_Normal": pred_anomaly_actual_normal,
            "Pred_Normal-Actual_Anomaly": pred_normal_actual_anomaly,
            "Pred_Anomaly-Actual_Anomaly": pred_anomaly_actual_anomaly,
            "Actual_Normal Percent": actual_normal_percent,
            "Actual_Anomaly Percent": actual_anomaly_percent,
        })

    out_df = pd.DataFrame(rows)

    activity_order = ["t1", "t2", "t3", "t4", "t5"]
    out_df["activity"] = pd.Categorical(
        out_df["activity"],
        categories=activity_order,
        ordered=True,
    )
    out_df = out_df.sort_values("activity").reset_index(drop=True)
    out_df["activity"] = out_df["activity"].astype(str)

    return out_df


# ---------------------------
# Activity-specific scaling helpers
# ---------------------------

ACTIVITY_ONEHOT_COLS = [
    "activity_t1",
    "activity_t2",
    "activity_t3",
    "activity_t4",
    "activity_t5",
]

def get_scale_and_passthrough_cols(cols):
    passthrough_cols = [c for c in cols if c in ACTIVITY_ONEHOT_COLS]
    scale_cols = [c for c in cols if c not in passthrough_cols]
    return scale_cols, passthrough_cols


def fit_activity_scalers(
    feature_df: pd.DataFrame,
    scale_cols: list[str],
    activity_col: str = "activity_code",
) -> dict:
    """
    Fit one StandardScaler per activity using only rows from that activity.
    Returns: {activity_code: fitted_scaler}
    """
    scalers = {}

    for activity_code, group_df in feature_df.groupby(activity_col):
        X_group = group_df[scale_cols].copy().astype(float)

        scaler = StandardScaler()
        scaler.fit(X_group)

        scalers[str(activity_code)] = scaler

    return scalers


def apply_activity_scalers(
    feature_df: pd.DataFrame,
    cols: list[str],
    scalers: dict,
    activity_col: str = "activity_code",
) -> pd.DataFrame:
    """
    Apply the matching scaler to each activity subset.
    One-hot activity columns are passed through unchanged.
    """
    scale_cols, passthrough_cols = get_scale_and_passthrough_cols(cols)

    out = feature_df.copy()
    
    for activity_code, idx in out.groupby(activity_col).groups.items():
        activity_code = str(activity_code)

        if activity_code not in scalers:
            raise KeyError(f"No scaler found for activity_code={activity_code}")

        scaler = scalers[activity_code]

        scaled_block = scaler.transform(
            out.loc[idx, scale_cols].copy().astype(float)
        )

        out[scale_cols] = out[scale_cols].astype(float)

    # final model matrix in the original column order
    X = out[cols].copy()

    # make sure passthrough columns stay numeric 0/1
    for c in passthrough_cols:
        if c in X.columns:
            X[c] = X[c].astype(float)

    return X

# --------------------------------
# For activity-specific thresholds
# --------------------------------

def get_thresholds_by_activity(scores, activity_codes, percentiles_by_activity):
    thresholds = {}
    scores = np.asarray(scores)
    activity_codes = pd.Series(activity_codes)

    for activity_code, percentile in percentiles_by_activity.items():
        mask = activity_codes == activity_code

        if not mask.any():
            thresholds[activity_code] = None
            continue

        if percentile is None:
            thresholds[activity_code] = None
            continue

        thresholds[activity_code] = float(
            np.percentile(scores[mask.to_numpy()], float(percentile))
        )

    return thresholds



def classification_metrics_from_preds(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    total = len(y_true)
    accuracy = float((y_true == y_pred).mean()) if total > 0 else math.nan

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }



def accuracy_pct(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if len(y_true) == 0:
        return math.nan
    return float((y_true == y_pred).mean() * 100.0)


def accuracy_by_activity_pct(preds_df: pd.DataFrame) -> dict:
    if preds_df is None or preds_df.empty:
        return {}

    out = {}
    grouped = preds_df.groupby("activity_code", dropna=False)

    for activity_code, g in grouped:
        out[str(activity_code)] = accuracy_pct(
            g["y_true"].to_numpy(dtype=int),
            g["y_pred"].to_numpy(dtype=int),
        )
    return out


def mean_accuracy_across_activities(acc_dict: dict) -> float:
    vals = [v for v in acc_dict.values() if pd.notna(v)]
    if len(vals) == 0:
        return math.nan
    return float(np.mean(vals))


def compute_multiclass_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }



