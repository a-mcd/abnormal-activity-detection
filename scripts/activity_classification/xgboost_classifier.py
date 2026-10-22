#!/usr/bin/env python3
"""
5-fold GroupKFold CV for CASAS-style sessions (pXX.tY.csv) using:
DictVectorizer + XGBoost (multiclass softprob)

Based directly on your LR script:
- Same feature engineering (including analogue binning + per-analogue-sensor stats)
- Same GroupKFold split by participant_id (p01, p26, ...)
- Exports engineered feature CSVs per fold (train/test), plus final training + error-set features
- Trains final model on all normal data, saves pipeline, evaluates on error set
- Exports XGBoost feature importance (gain, weight, cover)
"""

import os
import re
from collections import Counter

from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from abnormal_activity_detection.activity.sessions import norm_sensor, norm_msg, get_sensor_type, is_analogue_sensor, is_known_message, build_timestamp, duration_bucket_label
from abnormal_activity_detection.data import extract_activity_code, extract_participant_id
from abnormal_activity_detection.activity.exports import export_features_to_csv_with_vec

# --- XGBoost ---
try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError(
        "xgboost is not installed. Install with: pip install xgboost"
    ) from e


# -------------------------
# Config
# -------------------------
ACTIVITY_INDEX = {"t1": 0, "t2": 1, "t3": 2, "t4": 3, "t5": 4}
INDEX_ACTIVITY = {v: k for k, v in ACTIVITY_INDEX.items()}
N_CLASSES = 5


SEED = 42
DT_CLIP = 60.0
AD1A_ACTIVE_MIN_DURATION_S = 70

NORMAL_DIR, ERROR_DIR = get_dataset_dirs()
OUTPUT_DIR = get_results_dir("activity_classification", "xgboost")

MODEL_PATH = os.path.join(OUTPUT_DIR, "activity_xgb.joblib")
PRED_CSV = os.path.join(OUTPUT_DIR, "adl_error_activity_xgb_predictions.csv")

FEATURE_OUT_DIR = os.path.join(OUTPUT_DIR, "features")
os.makedirs(FEATURE_OUT_DIR, exist_ok=True)

FEATURE_IMPORTANCE_DIR = os.path.join(OUTPUT_DIR, "feature_importance")
os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)

CONFUSION_MATRIX_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")
os.makedirs(CONFUSION_MATRIX_DIR, exist_ok=True)


def export_confusion_matrix_csv(y_true, y_pred, out_csv: str, labels=None):
    if labels is None:
        labels = list(range(N_CLASSES))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    label_names = [INDEX_ACTIVITY[i] for i in labels]

    df_cm = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in label_names],
        columns=[f"pred_{name}" for name in label_names],
    )
    df_cm.to_csv(out_csv, index=True)
    print(f"✓ Exported confusion matrix -> {out_csv}")

    return df_cm


def format_metric_summary(y_true, y_pred) -> str:
    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return (
        f"Accuracy: {acc:.4f} | "
        f"Macro Precision: {macro_prec:.4f} | "
        f"Macro Recall: {macro_rec:.4f} | "
        f"Macro F1: {macro_f1:.4f}"
    )


# -------------------------
# Filename helpers
# -------------------------


# -------------------------
# Normalisation + sensor typing
# -------------------------


# -------------------------
# Time parsing
# -------------------------


# -------------------------
# analogue binning
# -------------------------
def analogue_bin_from_str(sensor: str, m: str) -> str:
    try:
        v = float(str(m).strip())
    except Exception:
        return "nan"

    s = norm_sensor(sensor)

    if s.startswith("ad1-b") or s.startswith("ad1-c"):
        edges = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
    elif s.startswith("ad1-a"):
        edges = (2.85, 2.95, 3.05, 3.15, 3.25, 3.35)
    else:
        edges = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)

    for i, e in enumerate(edges):
        if v < e:
            return f"b{i}"
    return f"b{len(edges)}"


# -------------------------
# Feature extraction
# -------------------------
def session_features_from_csv(path: str, dt_clip: float = DT_CLIP) -> dict:
    df = pd.read_csv(path)

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

    has_timestamp = False
    if "date" in df.columns and "time" in df.columns:
        df["timestamp"] = build_timestamp(df)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        if len(df) == 0:
            return {}

        has_timestamp = True

    feats = {}

    # Build evt:: features manually so ad1-b / ad1-c analogue value bins are excluded
    evt_counter = Counter()
    for s, m in zip(df["sensor"], df["message"]):
        if is_analogue_sensor(s):
            continue
            # Keep ad1-a analogue event bins
            #if s == "ad1-a":
                #evt_counter[f"evt::{s}::ANALOGUE::{analogue_bin_from_str(s, m)}"] += 1
            #    pass

            # Remove current bin features for ad1-b and ad1-c
            #elif s in ("ad1-b", "ad1-c"):
            #    pass

            # Fallback for any other analogue sensor
            #else:
                #evt_counter[f"evt::{s}::ANALOGUE::{analogue_bin_from_str(s, m)}"] += 1
        else:
            evt_counter[f"evt::{s}::{m}"] += 1

    feats.update(evt_counter)

    if has_timestamp:
        # ad1-a ACTIVE only if duration > 70 seconds
        ad1a_df = df[df["sensor"] == "ad1-a"].copy()
        if not ad1a_df.empty:
            ad1a_duration_s = (
                ad1a_df["timestamp"].iloc[-1] - ad1a_df["timestamp"].iloc[0]
            ).total_seconds()

            if ad1a_duration_s > AD1A_ACTIVE_MIN_DURATION_S:
                feats["analogue_ad1-a::ACTIVE"] = 1

        # ad1-b duration bucket
        ad1b_df = df[df["sensor"] == "ad1-b"].copy()
        if not ad1b_df.empty:
            ad1b_duration_s = (
                ad1b_df["timestamp"].iloc[-1] - ad1b_df["timestamp"].iloc[0]
            ).total_seconds()
            feats[f"analogue_ad1-b::{duration_bucket_label(ad1b_duration_s)}"] = 1

        # ad1-c duration bucket
        ad1c_df = df[df["sensor"] == "ad1-c"].copy()
        if not ad1c_df.empty:
            ad1c_duration_s = (
                ad1c_df["timestamp"].iloc[-1] - ad1c_df["timestamp"].iloc[0]
            ).total_seconds()
            feats[f"analogue_ad1-c::{duration_bucket_label(ad1c_duration_s)}"] = 1

    return dict(feats)


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
# Feature export utilities
# -------------------------


# -------------------------
# XGBoost feature importance export
# -------------------------
def export_xgb_feature_importance(clf: Pipeline, out_csv: str, top_n_print: int = 20):
    vec = clf.named_steps["vec"]
    model = clf.named_steps["xgb"]

    feature_names = vec.get_feature_names_out()
    booster = model.get_booster()

    gain_dict = booster.get_score(importance_type="gain")
    weight_dict = booster.get_score(importance_type="weight")
    cover_dict = booster.get_score(importance_type="cover")
    total_gain_dict = booster.get_score(importance_type="total_gain")
    total_cover_dict = booster.get_score(importance_type="total_cover")

    sklearn_importance = model.feature_importances_

    rows = []
    for i, feat_name in enumerate(feature_names):
        fkey = f"f{i}"
        rows.append({
            "feature_index": i,
            "feature": feat_name,
            "sklearn_feature_importance": float(sklearn_importance[i]),
            "gain": float(gain_dict.get(fkey, 0.0)),
            "weight": float(weight_dict.get(fkey, 0.0)),
            "cover": float(cover_dict.get(fkey, 0.0)),
            "total_gain": float(total_gain_dict.get(fkey, 0.0)),
            "total_cover": float(total_cover_dict.get(fkey, 0.0)),
        })

    df_imp = pd.DataFrame(rows)

    df_imp = df_imp.sort_values(
        ["gain", "sklearn_feature_importance", "weight"],
        ascending=False
    ).reset_index(drop=True)

    df_imp.to_csv(out_csv, index=False)
    print(f"✓ Exported feature importance -> {out_csv}")

    print(f"\nTop {top_n_print} features by gain:")
    print(df_imp[["feature", "gain", "weight", "cover"]].head(top_n_print).to_string(index=False))

    return df_imp


# -------------------------
# Prediction / evaluation on folder
# -------------------------
def predict_folder(model, data_dir: str, out_csv: str, cm_out_csv: str | None = None):
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

    if y_true:
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)

        print("\nEvaluation on error folder:")
        print(format_metric_summary(y_true, y_pred))

        print(classification_report(
            y_true, y_pred,
            target_names=[INDEX_ACTIVITY[i] for i in range(5)],
            digits=4
        ))
        cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
        if cm_out_csv is not None:
            export_confusion_matrix_csv(
                y_true, y_pred,
                out_csv=cm_out_csv,
                labels=list(range(N_CLASSES))
            )
        print("Confusion matrix (rows=true, cols=pred):")
        print("      " + "  ".join([f"{INDEX_ACTIVITY[i]:>3}" for i in range(5)]))
        for i in range(5):
            row_str = "  ".join([f"{cm[i, j]:>3}" for j in range(5)])
            print(f"{INDEX_ACTIVITY[i]:>3}  {row_str}")

        return y_true, y_pred

    return None, None


# -------------------------
# Main
# -------------------------
def main():
    X_dicts, y, groups, meta = load_dataset(NORMAL_DIR)

    print(f"Loaded {len(y)} sessions from {NORMAL_DIR}")
    print(f"People (groups): {len(set(groups))}")
    print("Activity counts:", {INDEX_ACTIVITY[i]: int(np.sum(y == i)) for i in sorted(INDEX_ACTIVITY)})

    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=N_CLASSES,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        reg_alpha=0.5,
        min_child_weight=10.0,
        gamma=0.2,
        random_state=SEED,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
    )


    clf = Pipeline([
        ("vec", DictVectorizer(sparse=True)),
        ("xgb", xgb_clf),
    ])

    gkf = GroupKFold(n_splits=5)

    all_true, all_pred = [], []
    fold_acc, fold_f1 = [], []

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

        acc = accuracy_score(y_test, y_hat)
        f1 = f1_score(y_test, y_hat, average="macro")

        fold_acc.append(acc)
        fold_f1.append(f1)
        all_true.extend(y_test.tolist())
        all_pred.extend(y_hat.tolist())

        test_people = sorted(set(groups_test.tolist()))
        print("\n" + "=" * 70)
        print(f"Fold {fold}/5")
        print(f"  Test people: {test_people}")
        print(f"  Accuracy:   {acc:.4f}")
        print(f"  Macro F1:   {f1:.4f}")
        print(classification_report(
            y_test, y_hat,
            target_names=[INDEX_ACTIVITY[i] for i in range(5)],
            digits=4
        ))

        vec_fold = clf.named_steps["vec"]
        out_train = os.path.join(FEATURE_OUT_DIR, f"features_fold{fold}_train.csv")
        out_test  = os.path.join(FEATURE_OUT_DIR, f"features_fold{fold}_test.csv")
        export_features_to_csv_with_vec(vec_fold, X_train, y_train, groups_train, meta_train, out_train)
        export_features_to_csv_with_vec(vec_fold, X_test, y_test, groups_test, meta_test, out_test)

        out_imp_fold = os.path.join(FEATURE_IMPORTANCE_DIR, f"xgb_feature_importance_fold{fold}.csv")
        export_xgb_feature_importance(clf, out_imp_fold, top_n_print=10)

        out_cm_fold = os.path.join(CONFUSION_MATRIX_DIR, f"xgb_confusion_matrix_fold{fold}.csv")
        export_confusion_matrix_csv(
            y_test, y_hat,
            out_csv=out_cm_fold,
            labels=list(range(N_CLASSES))
        )

    print("\n" + "#" * 70)
    print("Overall (pooled across folds)")
    all_true = np.array(all_true, dtype=int)
    all_pred = np.array(all_pred, dtype=int)

    print(f"Mean fold Accuracy: {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
    print(f"Mean fold Macro F1: {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")

    print("\nFinal summary on adl_noerror (pooled GroupKFold predictions):")
    print(format_metric_summary(all_true, all_pred))

    cm = confusion_matrix(all_true, all_pred, labels=list(range(5)))
    out_cm_noerror = os.path.join(CONFUSION_MATRIX_DIR, "xgb_confusion_matrix_adl_noerror.csv")
    export_confusion_matrix_csv(
        all_true, all_pred,
        out_csv=out_cm_noerror,
        labels=list(range(N_CLASSES))
    )
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("      " + "  ".join([f"{INDEX_ACTIVITY[i]:>3}" for i in range(5)]))
    for i in range(5):
        row_str = "  ".join([f"{cm[i, j]:>3}" for j in range(5)])
        print(f"{INDEX_ACTIVITY[i]:>3}  {row_str}")

    clf.fit(X_dicts, y)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved final XGB activity model to: {MODEL_PATH}")

    vec_final = clf.named_steps["vec"]
    out_final_train = os.path.join(FEATURE_OUT_DIR, "features_final_model_training_data.csv")
    export_features_to_csv_with_vec(vec_final, X_dicts, y, groups, meta, out_final_train)

    out_imp_final = os.path.join(FEATURE_IMPORTANCE_DIR, "xgb_feature_importance_final_model.csv")
    export_xgb_feature_importance(clf, out_imp_final, top_n_print=20)

    if os.path.isdir(ERROR_DIR):
        X_err, y_err, groups_err, meta_err = load_dataset(ERROR_DIR)
        if X_err:
            out_err_feats = os.path.join(FEATURE_OUT_DIR, "features_adl_error_data.csv")
            export_features_to_csv_with_vec(vec_final, X_err, y_err, groups_err, meta_err, out_err_feats)
        else:
            print("No valid error sessions found for feature export.")
    else:
        print(f"ERROR_DIR not found: {ERROR_DIR}")

    err_true, err_pred = None, None
    if os.path.isdir(ERROR_DIR):
        print(f"\nRunning inference on error folder: {ERROR_DIR}")
        out_cm_error = os.path.join(CONFUSION_MATRIX_DIR, "xgb_confusion_matrix_adl_error.csv")
        err_true, err_pred = predict_folder(clf, ERROR_DIR, PRED_CSV, cm_out_csv=out_cm_error)
    else:
        print(f"ERROR_DIR not found, skipping inference: {ERROR_DIR}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"adl_noerror -> {format_metric_summary(all_true, all_pred)}")
    if err_true is not None and err_pred is not None:
        print(f"adl_error   -> {format_metric_summary(err_true, err_pred)}")


if __name__ == "__main__":
    main()