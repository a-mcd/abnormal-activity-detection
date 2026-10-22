#!/usr/bin/env python3

import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from abnormal_activity_detection.data import (
    extract_activity_code,
    extract_participant_id,
    process_csv
)
from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir
from abnormal_activity_detection.activity.sessions import norm_sensor, norm_msg, get_sensor_type, is_analogue_sensor, is_known_message, build_timestamp, duration_bucket_label

# Map t1..t5 to integer labels 0..4
ACTIVITY_INDEX = {"t1": 0, "t2": 1, "t3": 2, "t4": 3, "t5": 4}
INDEX_ACTIVITY = {v: k for k, v in ACTIVITY_INDEX.items()}

FILENAME_RE = re.compile(r"p(\d+)\.(t[1-5])\.csv$", re.IGNORECASE)
SEED = 42
NORMAL_DIR, ERROR_DIR = get_dataset_dirs()
OUTPUT_DIR = get_results_dir("activity_classification", "logistic_regression")
MODEL_PATH = os.path.join(OUTPUT_DIR, "activity_lr.joblib")
PRED_CSV = os.path.join(OUTPUT_DIR, "adl_error_activity_predictions.csv")

# Output files for feature importance
FEATURE_IMPORTANCE_LONG_CSV = os.path.join(OUTPUT_DIR, "lr_feature_importance_long.csv")
FEATURE_IMPORTANCE_GLOBAL_CSV = os.path.join(OUTPUT_DIR, "lr_feature_importance_global.csv")
FEATURE_IMPORTANCE_TOP_CSV = os.path.join(OUTPUT_DIR, "lr_feature_importance_top10_per_class.csv")

# confusion matrix output directory
CONFUSION_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")
os.makedirs(CONFUSION_DIR, exist_ok=True)


def parse_person_activity(path: str):
    name = os.path.basename(path)
    m = FILENAME_RE.search(name)
    if not m:
        raise ValueError(f"Filename does not match p<id>.t<1-5>.csv: {name}")
    person = int(m.group(1))
    activity = m.group(2).lower()
    return person, activity


def try_parse_ts(date_str: str, time_str: str):
    s = f"{date_str} {time_str}"
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def session_features_from_csv(path: str, dt_clip: float = 60.0) -> dict:
    df = process_csv(path)

    df["sensor"] = df["sensor"].map(norm_sensor)
    df["message"] = df["message"].map(norm_msg)

    good = df.apply(
        lambda r: get_sensor_type(r["sensor"]) != "unknown"
        and is_known_message(r["sensor"], r["message"]),
        axis=1
    )
    df = df[good].copy()

    feats = {}

    if len(df) == 0:
        return feats

    ad1a_active = False
    ad1b_bucket = None
    ad1c_bucket = None

    if "date" in df.columns and "time" in df.columns:
        df["timestamp"] = build_timestamp(df)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        if len(df) == 0:
            return feats

        # Apply AD1-A duration rule to all datasets:
        # if ad1-a exists but spans <= 70 seconds, remove those rows
        ad1a_rows = df[df["sensor"] == "ad1-a"]
        if len(ad1a_rows) >= 1:
            ad1a_duration = (
                ad1a_rows["timestamp"].iloc[-1] - ad1a_rows["timestamp"].iloc[0]
            ).total_seconds()

            if ad1a_duration > 70:
                ad1a_active = True
            else:
                df = df[df["sensor"] != "ad1-a"].copy().reset_index(drop=True)
                if len(df) == 0:
                    return feats

        # Duration buckets for ad1-b
        ad1b_rows = df[df["sensor"] == "ad1-b"]
        if len(ad1b_rows) >= 1:
            ad1b_duration = (
                ad1b_rows["timestamp"].iloc[-1] - ad1b_rows["timestamp"].iloc[0]
            ).total_seconds()
            ad1b_bucket = duration_bucket_label(ad1b_duration)

        # Duration buckets for ad1-c
        ad1c_rows = df[df["sensor"] == "ad1-c"]
        if len(ad1c_rows) >= 1:
            ad1c_duration = (
                ad1c_rows["timestamp"].iloc[-1] - ad1c_rows["timestamp"].iloc[0]
            ).total_seconds()
            ad1c_bucket = duration_bucket_label(ad1c_duration)

    sensors = df["sensor"].tolist()
    msgs = df["message"].tolist()

    # Keep only evt::<sensor>::<message> for NON-analogue sensors
    pair_counts = {}
    for s, m in zip(sensors, msgs):
        if not is_analogue_sensor(s):
            key = f"evt::{s}::{m}"
            pair_counts[key] = pair_counts.get(key, 0) + 1

    feats.update(pair_counts)

    # AD1-A binary feature
    if ad1a_active:
        feats["analogue_ad1-a::ACTIVE"] = 1

    # AD1-B duration bucket
    if ad1b_bucket is not None:
        feats[f"analogue_ad1-b::{ad1b_bucket}"] = 1

    # AD1-C duration bucket
    if ad1c_bucket is not None:
        feats[f"analogue_ad1-c::{ad1c_bucket}"] = 1

    return feats


def load_dataset(data_dir: str):
    X_dicts, y, groups, meta = [], [], [], []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".csv"):
            continue

        activity_code = extract_activity_code(filename)
        if activity_code is None:
            continue
        activity_index = ACTIVITY_INDEX[activity_code]

        participant_id = extract_participant_id(filename)
        if participant_id is None:
            continue

        x = session_features_from_csv(os.path.join(data_dir, filename))
        X_dicts.append(x)
        y.append(activity_index)
        groups.append(participant_id)
        meta.append({
            "filename": filename,
            "activity_index": activity_index,
            "participant_id": participant_id
        })

    return X_dicts, np.array(y), np.array(groups), meta


def train_final_model_on_all_normal(normal_dir: str):
    X_dicts, y, groups, meta = load_dataset(normal_dir)

    clf = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("lr", LogisticRegression(
            max_iter=3000,
            C=0.1,
            n_jobs=None,
            random_state=SEED
        )),
    ])

    clf.fit(X_dicts, y)
    return clf


def predict_folder(model, data_dir: str, out_csv: str):
    rows = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".csv"):
            continue

        path = os.path.join(data_dir, filename)
        feats = session_features_from_csv(path)

        if not feats:
            rows.append({
                "filename": filename,
                "pred_activity": None,
                "pred_activity_index": None,
                "pred_confidence": None,
                "note": "empty_after_filtering"
            })
            continue

        proba = model.predict_proba([feats])[0]
        pred_idx = int(np.argmax(proba))
        pred_label = INDEX_ACTIVITY[pred_idx]
        conf = float(np.max(proba))

        row = {
            "filename": filename,
            "pred_activity": pred_label,
            "pred_activity_index": pred_idx,
            "pred_confidence": conf,
        }

        for i in range(5):
            row[f"p_{INDEX_ACTIVITY[i]}"] = float(proba[i])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions for {len(df)} files to: {out_csv}")


def export_features_to_csv(X_dicts, y, groups, meta, filename: str, base_dir: str):
    vec = DictVectorizer(sparse=False)
    X_array = vec.fit_transform(X_dicts)
    feature_names = vec.get_feature_names_out()

    rows = []
    for i, (x_dict, y_val, group_val, meta_val) in enumerate(zip(X_dicts, y, groups, meta)):
        row_dict = dict(zip(feature_names, X_array[i]))
        row_dict["activity_label"] = INDEX_ACTIVITY[y_val]
        row_dict["activity_index"] = int(y_val)
        row_dict["participant_id"] = group_val
        row_dict["filename"] = meta_val.get("filename", "")
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    out_path = os.path.join(base_dir, filename)
    df.to_csv(out_path, index=False)
    print(f"✓ Exported {len(df)} rows to: {filename}")
    return out_path


def export_confusion_matrix_csv(y_true, y_pred, out_csv: str, title: str = None):
    labels = list(range(5))
    class_names = [INDEX_ACTIVITY[i] for i in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    df_cm = pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names],
    )

    df_cm["row_total"] = df_cm.sum(axis=1)

    total_row = df_cm.sum(axis=0)
    total_row.name = "col_total"
    df_cm = pd.concat([df_cm, total_row.to_frame().T])

    df_cm.to_csv(out_csv)
    print(f"✓ Exported confusion matrix to: {out_csv}")

    if title:
        print(title)
    print("      " + "  ".join([f"{c:>7}" for c in class_names]))
    for i, c in enumerate(class_names):
        row_str = "  ".join([f"{cm[i, j]:>7}" for j in range(len(class_names))])
        print(f"{c:>7}  {row_str}")

    return df_cm


def export_feature_importance_tables(model, base_dir: str, top_n: int = 10):
    vec = model.named_steps["vec"]
    lr = model.named_steps["lr"]

    feature_names = vec.get_feature_names_out()
    coef = lr.coef_

    long_rows = []
    for class_idx in range(coef.shape[0]):
        activity_label = INDEX_ACTIVITY[class_idx]
        class_coefs = coef[class_idx]

        for feat, c in zip(feature_names, class_coefs):
            long_rows.append({
                "activity": activity_label,
                "activity_index": class_idx,
                "feature": feat,
                "coefficient": float(c),
                "abs_coefficient": float(abs(c)),
                "direction": "positive" if c > 0 else ("negative" if c < 0 else "neutral")
            })

    df_long = pd.DataFrame(long_rows)
    long_path = os.path.join(OUTPUT_DIR, "lr_feature_importance_long.csv")
    df_long.to_csv(long_path, index=False)
    print(f"✓ Exported feature importance long table: {long_path}")

    global_importance = np.mean(np.abs(coef), axis=0)
    df_global = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_coefficient": global_importance
    }).sort_values("mean_abs_coefficient", ascending=False)

    global_path = os.path.join(OUTPUT_DIR, "lr_feature_importance_global.csv")
    df_global.to_csv(global_path, index=False)
    print(f"✓ Exported global feature importance table: {global_path}")

    top_rows = []

    for class_idx in range(coef.shape[0]):
        activity_label = INDEX_ACTIVITY[class_idx]
        df_class = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coef[class_idx],
        })
        df_class["abs_coefficient"] = df_class["coefficient"].abs()

        df_pos = df_class.sort_values("coefficient", ascending=False).head(top_n).copy()
        df_pos["activity"] = activity_label
        df_pos["activity_index"] = class_idx
        df_pos["rank_within_activity"] = range(1, len(df_pos) + 1)
        df_pos["importance_type"] = "top_positive"

        df_neg = df_class.sort_values("coefficient", ascending=True).head(top_n).copy()
        df_neg["activity"] = activity_label
        df_neg["activity_index"] = class_idx
        df_neg["rank_within_activity"] = range(1, len(df_neg) + 1)
        df_neg["importance_type"] = "top_negative"

        top_rows.append(df_pos)
        top_rows.append(df_neg)

    df_top = pd.concat(top_rows, ignore_index=True)
    df_top = df_top[
        ["activity", "activity_index", "importance_type", "rank_within_activity",
         "feature", "coefficient", "abs_coefficient"]
    ].sort_values(["activity_index", "importance_type", "rank_within_activity"])

    top_path = os.path.join(OUTPUT_DIR, "lr_feature_importance_top10_per_class.csv")
    df_top.to_csv(top_path, index=False)
    print(f"✓ Exported top feature importance table: {top_path}")

    return long_path, global_path, top_path


def evaluate_abnormal_data(model, abnormal_dir):
    print(f"\n" + "=" * 30)
    print("EVALUATING ABNORMAL DATA")
    print("=" * 30)

    X_dicts_err, y_true_err, _, _ = load_dataset(abnormal_dir)

    if not X_dicts_err:
        print("No valid abnormal data found for evaluation.")
        return None, None, None, None

    y_pred_err = model.predict(X_dicts_err)

    acc = accuracy_score(y_true_err, y_pred_err)
    prec = precision_score(y_true_err, y_pred_err, average="macro", zero_division=0)
    rec = recall_score(y_true_err, y_pred_err, average="macro", zero_division=0)
    f1 = f1_score(y_true_err, y_pred_err, average="macro", zero_division=0)

    print(f"Accuracy on Abnormal Data: {acc:.4f}")
    print(f"Macro Precision:          {prec:.4f}")
    print(f"Macro Recall:             {rec:.4f}")
    print(f"Macro F1:                 {f1:.4f}")

    print("\nClassification Report (Abnormal Data):")
    print(classification_report(
        y_true_err, y_pred_err,
        target_names=[INDEX_ACTIVITY[i] for i in range(5)],
        digits=4,
        zero_division=0
    ))

    out_csv = os.path.join(CONFUSION_DIR, "confusion_matrix_abnormal_data.csv")
    export_confusion_matrix_csv(
        y_true_err,
        y_pred_err,
        out_csv,
        title="Confusion Matrix (Rows=True Abnormal Class, Cols=Predicted Class):"
    )

    return acc, prec, rec, f1


def main():
    X_dicts, y, groups, meta = load_dataset(NORMAL_DIR)

    clf = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("lr", LogisticRegression(
            max_iter=3000,
            C=0.1,
            n_jobs=None,
            random_state=SEED
        )),
    ])

    gkf = GroupKFold(n_splits=5)

    all_true = []
    all_pred = []
    fold_acc = []
    fold_prec = []
    fold_rec = []
    fold_f1 = []

    print(f"Loaded {len(y)} sessions from {NORMAL_DIR}")
    print(f"People (groups): {len(set(groups))}")
    print("Activity counts:", {INDEX_ACTIVITY[i]: int(np.sum(y == i)) for i in sorted(INDEX_ACTIVITY)})

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_dicts, y, groups), start=1):
        X_train_fold = [X_dicts[i] for i in train_idx]
        y_train_fold = y[train_idx]
        groups_train_fold = groups[train_idx]
        meta_train_fold = [meta[i] for i in train_idx]

        X_test_fold = [X_dicts[i] for i in test_idx]
        y_test_fold = y[test_idx]
        groups_test_fold = groups[test_idx]
        meta_test_fold = [meta[i] for i in test_idx]

        export_features_to_csv(
            X_train_fold, y_train_fold, groups_train_fold, meta_train_fold,
            f"features_fold{fold}_train.csv", OUTPUT_DIR
        )

        export_features_to_csv(
            X_test_fold, y_test_fold, groups_test_fold, meta_test_fold,
            f"features_fold{fold}_test.csv", OUTPUT_DIR
        )

        X_train = [X_dicts[i] for i in train_idx]
        y_train = y[train_idx]
        X_test = [X_dicts[i] for i in test_idx]
        y_test = y[test_idx]

        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)

        acc = accuracy_score(y_test, y_hat)
        prec = precision_score(y_test, y_hat, average="macro", zero_division=0)
        rec = recall_score(y_test, y_hat, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_hat, average="macro", zero_division=0)

        fold_acc.append(acc)
        fold_prec.append(prec)
        fold_rec.append(rec)
        fold_f1.append(f1)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_hat.tolist())

        test_people = sorted(set(groups[test_idx].tolist()))
        print("\n" + "=" * 70)
        print(f"Fold {fold}/5")
        print(f"  Test people: {test_people}")
        print(f"  Accuracy:         {acc:.4f}")
        print(f"  Macro Precision:  {prec:.4f}")
        print(f"  Macro Recall:     {rec:.4f}")
        print(f"  Macro F1:         {f1:.4f}")
        print(classification_report(
            y_test, y_hat,
            target_names=[INDEX_ACTIVITY[i] for i in range(5)],
            digits=4,
            zero_division=0
        ))

        fold_cm_csv = os.path.join(CONFUSION_DIR, f"confusion_matrix_fold{fold}.csv")
        export_confusion_matrix_csv(
            y_test,
            y_hat,
            fold_cm_csv,
            title=f"Fold {fold} confusion matrix (rows=true, cols=pred):"
        )

    print("\n" + "#" * 70)
    print("Overall (pooled across folds)")
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    overall_acc = accuracy_score(all_true, all_pred)
    overall_prec = precision_score(all_true, all_pred, average="macro", zero_division=0)
    overall_rec = recall_score(all_true, all_pred, average="macro", zero_division=0)
    overall_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)

    print(f"Mean fold Accuracy:        {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
    print(f"Mean fold Macro Precision: {np.mean(fold_prec):.4f} ± {np.std(fold_prec):.4f}")
    print(f"Mean fold Macro Recall:    {np.mean(fold_rec):.4f} ± {np.std(fold_rec):.4f}")
    print(f"Mean fold Macro F1:        {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")
    print(f"Overall Accuracy:          {overall_acc:.4f}")
    print(f"Overall Macro Precision:   {overall_prec:.4f}")
    print(f"Overall Macro Recall:      {overall_rec:.4f}")
    print(f"Overall Macro F1:          {overall_f1:.4f}")

    overall_cm_csv = os.path.join(CONFUSION_DIR, "confusion_matrix_overall_cv.csv")
    export_confusion_matrix_csv(
        all_true,
        all_pred,
        overall_cm_csv,
        title="Overall confusion matrix (rows=true, cols=pred):"
    )

    print("\n" + "=" * 70)
    print("EXPORTING FEATURE-ENGINEERED DATA")
    print("=" * 70)

    export_features_to_csv(
        X_dicts, y, groups, meta,
        "features_final_model_training_data.csv", OUTPUT_DIR
    )

    X_dicts_err, y_err, groups_err, meta_err = load_dataset(ERROR_DIR)
    if X_dicts_err:
        export_features_to_csv(
            X_dicts_err, y_err, groups_err, meta_err,
            "features_adl_error_data.csv", OUTPUT_DIR
        )
    else:
        print("✗ No error data found to export")

    final_model = train_final_model_on_all_normal(NORMAL_DIR)
    joblib.dump(final_model, MODEL_PATH)
    print(f"\nSaved final activity model to: {MODEL_PATH}")

    print("\n" + "=" * 70)
    print("EXPORTING FEATURE IMPORTANCE TABLES")
    print("=" * 70)
    export_feature_importance_tables(final_model, OUTPUT_DIR, top_n=10)

    print(f"\nRunning inference on error folder: {ERROR_DIR}")
    error_acc, error_prec, error_rec, error_f1 = evaluate_abnormal_data(final_model, ERROR_DIR)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(
        f"Normal dataset (GroupKFold CV) -> "
        f"Accuracy: {overall_acc:.4f} | "
        f"Macro Precision: {overall_prec:.4f} | "
        f"Macro Recall: {overall_rec:.4f} | "
        f"Macro F1: {overall_f1:.4f}"
    )
    if error_acc is not None:
        print(
            f"Error dataset                 -> "
            f"Accuracy: {error_acc:.4f} | "
            f"Macro Precision: {error_prec:.4f} | "
            f"Macro Recall: {error_rec:.4f} | "
            f"Macro F1: {error_f1:.4f}"
        )


if __name__ == "__main__":
    main()