#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from abnormal_activity_detection.data import (
    extract_activity_code,
    extract_participant_id,
)
from abnormal_activity_detection.data import process_csv
from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir
from abnormal_activity_detection.evaluation import compute_multiclass_metrics
from abnormal_activity_detection.activity.sessions import norm_sensor, norm_msg, get_sensor_type, is_analogue_sensor, is_known_message, build_timestamp, duration_bucket_label
from abnormal_activity_detection.activity.exports import export_features_to_csv_with_vec
# -------------------------
# Config
# -------------------------
ACTIVITY_INDEX = {"t1": 0, "t2": 1, "t3": 2, "t4": 3, "t5": 4}
INDEX_ACTIVITY = {v: k for k, v in ACTIVITY_INDEX.items()}
N_CLASSES = 5


SEED = 42
DT_CLIP = 60.0
AD1A_MIN_DURATION_S = 70.0

NORMAL_DIR, ERROR_DIR = get_dataset_dirs()
OUTPUT_DIR = get_results_dir("activity_classification", "random_forest")

MODEL_PATH = os.path.join(OUTPUT_DIR, "activity_rf.joblib")
PRED_CSV = os.path.join(OUTPUT_DIR, "adl_error_activity_rf_predictions.csv")

FEATURE_OUT_DIR = os.path.join(OUTPUT_DIR, "features")
FEATURE_IMPORTANCE_DIR = os.path.join(OUTPUT_DIR, "feature_importance")
CONFUSION_MATRIX_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")
for _d in (FEATURE_OUT_DIR, FEATURE_IMPORTANCE_DIR, CONFUSION_MATRIX_DIR):
    os.makedirs(_d, exist_ok=True)


# -------------------------
# Normalisation + sensor typing
# -------------------------


# -------------------------
# Time parsing
# -------------------------


# -------------------------
# Duration buckets
# -------------------------


# -------------------------
# Confusion matrix export
# -------------------------
def export_confusion_matrix_csv(y_true, y_pred, out_csv: str, labels=None):
    if labels is None:
        labels = list(range(N_CLASSES))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    class_names = [INDEX_ACTIVITY[i] for i in labels]

    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.index.name = "true_label"
    df_cm.columns.name = "pred_label"
    df_cm.to_csv(out_csv)

    print(f"✓ Exported confusion matrix -> {out_csv}")
    return df_cm


# -------------------------
# Feature extraction
# -------------------------
def session_features_from_csv(path: str, dt_clip: float = DT_CLIP) -> dict:
    df = process_csv(path)

    if not {"sensor", "message"}.issubset(df.columns):
        return {}

    df["sensor"] = df["sensor"].map(norm_sensor)
    df["message"] = df["message"].map(norm_msg)

    good = df.apply(
        lambda r: get_sensor_type(r["sensor"]) != "unknown"
        and is_known_message(r["sensor"], r["message"]),
        axis=1
    )
    df = df[good].copy()

    if len(df) == 0:
        return {}

    ad1a_active = False

    if {"date", "time"}.issubset(df.columns):
        df["timestamp"] = build_timestamp(df)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        if len(df) == 0:
            return {}

        ad1a_rows = df[df["sensor"] == "ad1-a"]
        if len(ad1a_rows) > 0:
            ad1a_duration_s = (
                ad1a_rows["timestamp"].iloc[-1] - ad1a_rows["timestamp"].iloc[0]
            ).total_seconds()

            if ad1a_duration_s < AD1A_MIN_DURATION_S:
                df = df[df["sensor"] != "ad1-a"].copy().reset_index(drop=True)
            else:
                ad1a_active = True

        if len(df) == 0:
            return {}
    else:
        return {}

    feats = {}

    sensors = df["sensor"].tolist()
    msgs = df["message"].tolist()

    evt_features = []

    for s, m in zip(sensors, msgs):
        if s == "ad1-a":
            continue
        elif is_analogue_sensor(s):
            # Remove all analogue b-value features for ad1-b / ad1-c
            continue
        else:
            evt_features.append(f"evt::{s}::{m}")

    evt_counter = Counter(evt_features)
    for feat, cnt in evt_counter.items():
        feats[feat] = cnt

    for sensor_name in ["ad1-b", "ad1-c"]:
        sensor_rows = df[df["sensor"] == sensor_name]
        if len(sensor_rows) > 0:
            duration_s = (
                sensor_rows["timestamp"].iloc[-1] - sensor_rows["timestamp"].iloc[0]
            ).total_seconds()
            bucket = duration_bucket_label(duration_s)
            feats[f"analogue_{sensor_name}::{bucket}"] = 1

    if ad1a_active:
        feats["analogue_ad1-a::ACTIVE"] = 1

    return feats

# -------------------------
# Dataset loader
# -------------------------
def load_dataset(data_dir: str):
    X_dicts, y, groups, meta = [], [], [], []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".csv"):
            continue

        activity_code = extract_activity_code(filename)
        participant_id = extract_participant_id(filename)
        if activity_code is None or participant_id is None:
            continue

        x = session_features_from_csv(os.path.join(data_dir, filename))
        if not x:
            continue

        X_dicts.append(x)
        y.append(ACTIVITY_INDEX[activity_code])
        groups.append(participant_id)
        meta.append({
            "filename": filename,
            "activity_index": ACTIVITY_INDEX[activity_code],
            "activity_label": activity_code,
            "participant_id": participant_id,
        })

    return X_dicts, np.array(y, dtype=int), np.array(groups), meta


# -------------------------
# Feature export
# -------------------------


# -------------------------
# RF feature importance export
# -------------------------
def export_rf_feature_importance(clf: Pipeline, out_csv: str, top_n_print: int = 20):
    vec = clf.named_steps["vec"]
    rf = clf.named_steps["rf"]

    feature_names = vec.get_feature_names_out()
    importances = rf.feature_importances_

    df_imp = pd.DataFrame({
        "feature_index": np.arange(len(feature_names)),
        "feature": feature_names,
        "importance": importances,
    })

    total_importance = df_imp["importance"].sum()
    if total_importance > 0:
        df_imp["importance_pct"] = 100.0 * df_imp["importance"] / total_importance
    else:
        df_imp["importance_pct"] = 0.0

    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    df_imp["rank"] = np.arange(1, len(df_imp) + 1)

    df_imp = df_imp[[
        "rank",
        "feature_index",
        "feature",
        "importance",
        "importance_pct",
    ]]

    df_imp.to_csv(out_csv, index=False)
    print(f"✓ Exported RF feature importance -> {out_csv}")

    print(f"\nTop {top_n_print} RF features:")
    print(df_imp.head(top_n_print).to_string(index=False))

    return df_imp


# -------------------------
# Prediction / evaluation on folder
# -------------------------
def predict_folder(model, data_dir: str, out_csv: str,
                   confusion_matrix_csv: str | None = None):
    rows = []
    y_true, y_pred = [], []

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".csv"):
            continue

        true_code = extract_activity_code(filename)
        true_idx = ACTIVITY_INDEX[true_code] if true_code in ACTIVITY_INDEX else None

        feats = session_features_from_csv(os.path.join(data_dir, filename))
        if not feats:
            rows.append({
                "filename": filename,
                "true_activity": true_code,
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
            "true_activity": true_code,
            "true_activity_index": true_idx,
            "pred_activity": pred_label,
            "pred_activity_index": pred_idx,
            "pred_confidence": conf,
        }
        for i in range(5):
            row[f"p_{INDEX_ACTIVITY[i]}"] = float(proba[i])
        rows.append(row)

        if true_idx is not None:
            y_true.append(true_idx)
            y_pred.append(pred_idx)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions for {len(df)} files to: {out_csv}")

    metrics = None

    if y_true:
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)

        metrics = compute_multiclass_metrics(y_true, y_pred)

        print("\nEvaluation on error folder:")
        print(f"Accuracy:        {metrics['accuracy']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
        print(f"Macro F1:        {metrics['macro_f1']:.4f}")

        print(classification_report(
            y_true, y_pred,
            target_names=[INDEX_ACTIVITY[i] for i in range(5)],
            digits=4,
            zero_division=0
        ))

        cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
        print("Confusion matrix (rows=true, cols=pred):")
        print("      " + "  ".join([f"{INDEX_ACTIVITY[i]:>3}" for i in range(5)]))
        for i in range(5):
            row_str = "  ".join([f"{cm[i, j]:>3}" for j in range(5)])
            print(f"{INDEX_ACTIVITY[i]:>3}  {row_str}")

        if confusion_matrix_csv is not None:
            export_confusion_matrix_csv(y_true, y_pred, confusion_matrix_csv)

    return metrics


# -------------------------
# Main
# -------------------------
def main():
    X_dicts, y, groups, meta = load_dataset(NORMAL_DIR)

    print(f"Loaded {len(y)} sessions from {NORMAL_DIR}")
    print(f"People (groups): {len(set(groups))}")
    print("Activity counts:", {INDEX_ACTIVITY[i]: int(np.sum(y == i)) for i in sorted(INDEX_ACTIVITY)})

    if len(y) == 0:
        raise ValueError("No usable sessions found after filtering.")
    if len(set(groups)) < 5:
        raise ValueError(f"Need at least 5 participant groups for GroupKFold, but found {len(set(groups))}.")

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1
    )

    clf = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("rf", rf),
    ])

    gkf = GroupKFold(n_splits=5)

    all_true, all_pred = [], []
    fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_dicts, y, groups), start=1):
        X_train = [X_dicts[i] for i in train_idx]
        y_train = y[train_idx]
        groups_train = groups[train_idx]
        meta_train = [meta[i] for i in train_idx]

        X_test = [X_dicts[i] for i in test_idx]
        y_test = y[test_idx]
        groups_test = groups[test_idx]
        meta_test = [meta[i] for i in test_idx]

        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)

        metrics = compute_multiclass_metrics(y_test, y_hat)

        fold_acc.append(metrics["accuracy"])
        fold_prec.append(metrics["macro_precision"])
        fold_rec.append(metrics["macro_recall"])
        fold_f1.append(metrics["macro_f1"])

        all_true.extend(y_test.tolist())
        all_pred.extend(y_hat.tolist())

        test_people = sorted(set(groups_test.tolist()))
        print("\n" + "=" * 70)
        print(f"Fold {fold}/5")
        print(f"  Test people:      {test_people}")
        print(f"  Accuracy:         {metrics['accuracy']:.4f}")
        print(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
        print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
        print(classification_report(
            y_test, y_hat,
            target_names=[INDEX_ACTIVITY[i] for i in range(5)],
            digits=4,
            zero_division=0
        ))

        cm_fold = confusion_matrix(y_test, y_hat, labels=list(range(5)))
        print("Confusion matrix (rows=true, cols=pred):")
        print("      " + "  ".join([f"{INDEX_ACTIVITY[i]:>3}" for i in range(5)]))
        for i in range(5):
            row_str = "  ".join([f"{cm_fold[i, j]:>3}" for j in range(5)])
            print(f"{INDEX_ACTIVITY[i]:>3}  {row_str}")

        out_cm_fold = os.path.join(CONFUSION_MATRIX_DIR, f"confusion_matrix_fold{fold}.csv")
        export_confusion_matrix_csv(y_test, y_hat, out_cm_fold)

        vec_fold = clf.named_steps["vec"]
        out_train = os.path.join(FEATURE_OUT_DIR, f"features_fold{fold}_train.csv")
        out_test = os.path.join(FEATURE_OUT_DIR, f"features_fold{fold}_test.csv")
        export_features_to_csv_with_vec(vec_fold, X_train, y_train, groups_train, meta_train, out_train)
        export_features_to_csv_with_vec(vec_fold, X_test, y_test, groups_test, meta_test, out_test)

        out_imp_fold = os.path.join(FEATURE_IMPORTANCE_DIR, f"rf_feature_importance_fold{fold}.csv")
        export_rf_feature_importance(clf, out_imp_fold, top_n_print=10)

    print("\n" + "#" * 70)
    print("Overall (pooled across folds)")
    all_true = np.array(all_true, dtype=int)
    all_pred = np.array(all_pred, dtype=int)

    normal_metrics = compute_multiclass_metrics(all_true, all_pred)

    print(f"Mean fold Accuracy:        {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
    print(f"Mean fold Macro Precision: {np.mean(fold_prec):.4f} ± {np.std(fold_prec):.4f}")
    print(f"Mean fold Macro Recall:    {np.mean(fold_rec):.4f} ± {np.std(fold_rec):.4f}")
    print(f"Mean fold Macro F1:        {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")

    print(f"Overall Accuracy:          {normal_metrics['accuracy']:.4f}")
    print(f"Overall Macro Precision:   {normal_metrics['macro_precision']:.4f}")
    print(f"Overall Macro Recall:      {normal_metrics['macro_recall']:.4f}")
    print(f"Overall Macro F1:          {normal_metrics['macro_f1']:.4f}")

    cm = confusion_matrix(all_true, all_pred, labels=list(range(5)))
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("      " + "  ".join([f"{INDEX_ACTIVITY[i]:>3}" for i in range(5)]))
    for i in range(5):
        row_str = "  ".join([f"{cm[i, j]:>3}" for j in range(5)])
        print(f"{INDEX_ACTIVITY[i]:>3}  {row_str}")

    out_cm_normal = os.path.join(CONFUSION_MATRIX_DIR, "confusion_matrix_adl_noerror_overall.csv")
    export_confusion_matrix_csv(all_true, all_pred, out_cm_normal)

    clf.fit(X_dicts, y)
    joblib.dump({"model": clf}, MODEL_PATH)
    print(f"\nSaved final RF activity model to: {MODEL_PATH}")

    vec_final = clf.named_steps["vec"]
    out_final_train = os.path.join(FEATURE_OUT_DIR, "features_final_model_training_data.csv")
    export_features_to_csv_with_vec(vec_final, X_dicts, y, groups, meta, out_final_train)

    out_imp_final = os.path.join(FEATURE_IMPORTANCE_DIR, "rf_feature_importance_final_model.csv")
    export_rf_feature_importance(clf, out_imp_final, top_n_print=20)

    if os.path.isdir(ERROR_DIR):
        X_err, y_err, groups_err, meta_err = load_dataset(ERROR_DIR)
        if X_err:
            out_err_feats = os.path.join(FEATURE_OUT_DIR, "features_adl_error_data.csv")
            export_features_to_csv_with_vec(vec_final, X_err, y_err, groups_err, meta_err, out_err_feats)
        else:
            print("No valid error sessions found for feature export.")
    else:
        print(f"ERROR_DIR not found: {ERROR_DIR}")

    error_metrics = None
    if os.path.isdir(ERROR_DIR):
        print(f"\nRunning inference on error folder: {ERROR_DIR}")
        out_cm_error = os.path.join(CONFUSION_MATRIX_DIR, "confusion_matrix_adl_error.csv")
        error_metrics = predict_folder(
            clf,
            ERROR_DIR,
            PRED_CSV,
            confusion_matrix_csv=out_cm_error
        )
    else:
        print(f"ERROR_DIR not found, skipping inference: {ERROR_DIR}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(
        f"adl_noerror -> Accuracy: {normal_metrics['accuracy']:.4f} | "
        f"Macro Precision: {normal_metrics['macro_precision']:.4f} | "
        f"Macro Recall: {normal_metrics['macro_recall']:.4f} | "
        f"Macro F1: {normal_metrics['macro_f1']:.4f}"
    )

    if error_metrics is not None:
        print(
            f"adl_error   -> Accuracy: {error_metrics['accuracy']:.4f} | "
            f"Macro Precision: {error_metrics['macro_precision']:.4f} | "
            f"Macro Recall: {error_metrics['macro_recall']:.4f} | "
            f"Macro F1: {error_metrics['macro_f1']:.4f}"
        )
    else:
        print("adl_error   -> No evaluation available")


if __name__ == "__main__":
    main()