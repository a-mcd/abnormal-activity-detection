#!/usr/bin/env python3
"""
All-in-one One-Class SVM tuning script.

For each hyperparameter configuration, this script:
1. Runs 5-fold GroupKFold CV on NORMAL sessions only
2. Trains a final model on ALL normal sessions
3. Runs inference on ERROR sessions
4. Saves:
   - one summary row per configuration
   - per-fold normal CV predictions
   - final normal train predictions
   - final error predictions

Important:
- Model fitting is always done on NO-ERROR data only.
- ERROR data is used only for reporting after final training.
"""

import json
import math
from copy import deepcopy
from itertools import product
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from abnormal_activity_detection.anomaly.ocsvm import (
    predict_ocsvm_by_activity,
    predict_ocsvm,
    make_ocsvm,
    get_threshold,
    save_ocsvm_model_bundle,
    config_to_tag,
)
from abnormal_activity_detection.anomaly.scoring import compute_normal_only_rank_score
from abnormal_activity_detection.evaluation import (
    apply_activity_scalers,
    fit_activity_scalers,
    get_thresholds_by_activity,
    save_confusion_matrices_by_activity_one_file,
    save_confusion_matrix_files,
    build_activity_confusion_summary_table,
    build_confusion_matrix_df,
    classification_metrics_from_preds,
    accuracy_pct,
    accuracy_by_activity_pct,
    mean_accuracy_across_activities,
)
from abnormal_activity_detection.anomaly.configs.ocsvm import (
    TARGET_ACTIVITY_CODE,
    N_SPLITS,
    FEATURE_CONFIG,
)
from abnormal_activity_detection.data import load_data
from abnormal_activity_detection.features.anomaly import (
    build_feature_matrix,
    build_reference_stats,
    feature_columns,
    build_feature_matrix_train_loo_reference,
    build_reference_stats_by_activity

)
from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir


# =========================================================
# CONFIG
# =========================================================

# Set this to True if you want the script to rank configs by normal-CV only.
# This preserves the strict "normal-only tuning" principle.
RANK_USING_NORMAL_ONLY = True

# Target false positive rate on normal CV folds
TARGET_NORMAL_FP_RATE = 0.10

# Hyperparameter grid

if TARGET_ACTIVITY_CODE == "t-all":
    PARAM_GRID = {
        "kernel": ["linear", "rbf"],
        "gamma": ["scale"],
        "nu": [0.05, 0.1],
        "threshold_percentiles_by_activity": [
            {"t1": 20.0, "t2": 27.0,  "t3": 18.0, "t4": 10.0, "t5": 1.0},
        ],
    }
else:
    PARAM_GRID = {
        "kernel": ["linear"], #,
        "gamma": ["scale"], 
        "nu": [0.06, 0.07, 0.08],
        "threshold_percentile": [None, 1, 2],
    }

# Optional:
# If True, save the fitted final model bundle for EVERY config
SAVE_MODEL_BUNDLE_PER_CONFIG = False

# If True, save per-fold train/val feature matrices per config
SAVE_FEATURE_MATRICES = False

# =========================================================
# PATHS
# =========================================================

NORMAL_DIR, ERROR_DIR = get_dataset_dirs()
OUTPUT_DIR = get_results_dir("abnormal_detection", "one_class_svm", TARGET_ACTIVITY_CODE, "ocsvm_tuning_all_in_one")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CV_PREDS_DIR = OUTPUT_DIR / "cv_predictions"
TRAIN_PREDS_DIR = OUTPUT_DIR / "train_predictions"
ERROR_PREDS_DIR = OUTPUT_DIR / "error_predictions"
MODEL_DIR = OUTPUT_DIR / "model_bundles"

CM_DIR = OUTPUT_DIR / "confusion_matrices"
CV_CM_DIR = CM_DIR / "cv_noerror"
TRAIN_CM_DIR = CM_DIR / "train_noerror"
ERROR_CM_DIR = CM_DIR / "error"

for d in [CV_CM_DIR, TRAIN_CM_DIR, ERROR_CM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

for d in [CV_PREDS_DIR, TRAIN_PREDS_DIR, ERROR_PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if SAVE_MODEL_BUNDLE_PER_CONFIG:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

META_COLS = [
    "filename",
    "participant_id",
    "activity_code",
    "activity_index",
    "dataset",
]


# =========================================================
# MAIN EVALUATION PER CONFIG
# =========================================================

def evaluate_config(
    config_id: int,
    config: dict,
    normal_sessions,
    normal_meta,
    error_sessions,
    error_meta,
    cols,
):
    config_tag = config_to_tag(config_id, config)
    print(f"\n{'=' * 90}")
    print(f"CONFIG {config_id}: {config_tag}")
    print(json.dumps(config, indent=2, default=str))

    # -------------------------
    # NORMAL-ONLY CV
    # -------------------------
    normal_index_df = pd.DataFrame(normal_meta)
    groups = normal_index_df["participant_id"].values
    dummy_X = np.zeros((len(normal_index_df), 1))

    if TARGET_ACTIVITY_CODE == "t-all":
        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=42,
        )
        y_strat = normal_index_df["activity_code"].values
        split_iter = cv.split(dummy_X, y_strat, groups=groups)
    else:
        cv = GroupKFold(n_splits=N_SPLITS)
        split_iter = cv.split(dummy_X, groups=groups)


    fold_rows = []
    all_cv_preds = []

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
        train_normal_dfs = [normal_sessions[i] for i in train_idx]
        train_normal_meta = [normal_meta[i] for i in train_idx]

        val_normal_dfs = [normal_sessions[i] for i in val_idx]
        val_normal_meta = [normal_meta[i] for i in val_idx]

        if TARGET_ACTIVITY_CODE == "t-all":
            reference_stats = build_reference_stats_by_activity(
                train_normal_dfs,
                train_normal_meta,
            )
        else:
            reference_stats = build_reference_stats(train_normal_dfs)

        train_df = build_feature_matrix_train_loo_reference (
            session_dfs=train_normal_dfs,
            meta=train_normal_meta,
            cols=cols,
        )
        val_df = build_feature_matrix(
            session_dfs=val_normal_dfs,
            meta=val_normal_meta,
            cols=cols,
            reference_stats=reference_stats,
        )

        X_train = train_df[cols].copy().astype(float)
        X_val = val_df.reindex(columns=cols, fill_value=0)[cols].copy().astype(float)

        if X_train.empty:
            raise RuntimeError(f"{config_tag} fold {fold_idx}: empty training matrix")
        if X_val.empty:
            raise RuntimeError(f"{config_tag} fold {fold_idx}: empty validation matrix")

        if TARGET_ACTIVITY_CODE == "t-all":
            scalers = fit_activity_scalers(
                feature_df=train_df,
                scale_cols=[c for c in cols if not c.startswith("activity_")],
                activity_col="activity_code",
            )
            X_train_scaled = apply_activity_scalers(
                feature_df=train_df,
                cols=cols,
                scalers=scalers,
                activity_col="activity_code",
            )
            X_val_scaled = apply_activity_scalers(
                feature_df=val_df,
                cols=cols,
                scalers=scalers,
                activity_col="activity_code",
            )
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

        model = make_ocsvm(config)
        model.fit(X_train_scaled)

        train_scores = model.decision_function(X_train_scaled).ravel()
        threshold_percentile = config.get("threshold_percentile")
        threshold_percentiles_by_activity = config.get("threshold_percentiles_by_activity")

        thresholds_by_activity = None

        if TARGET_ACTIVITY_CODE == "t-all":
            if threshold_percentiles_by_activity is not None:
                thresholds_by_activity = get_thresholds_by_activity(
                    scores=train_scores,
                    activity_codes=train_df["activity_code"],
                    percentiles_by_activity=threshold_percentiles_by_activity,
                )
                pred_mode = "percentile_threshold_by_activity"
        else:
            threshold, pred_mode = get_threshold(
                train_scores,
                threshold_percentile,
            )

        y_true_val = np.zeros(len(val_df), dtype=int)
        if TARGET_ACTIVITY_CODE == "t-all":
            if thresholds_by_activity is None:
                y_pred_val, val_scores, native_pred_val, _ = predict_ocsvm(
                    model=model,
                    X_scaled=X_val_scaled,
                    threshold=None,
                )
            else:
                missing = set(val_df["activity_code"]) - set(thresholds_by_activity.keys())
                if missing:
                    raise ValueError(f"Missing thresholds for activities in CV validation: {sorted(missing)}")
        
                y_pred_val, val_scores, native_pred_val, row_thresholds = predict_ocsvm_by_activity(
                    model=model,
                    X=X_val_scaled,
                    activity_codes=val_df["activity_code"],
                    thresholds_by_activity=thresholds_by_activity,
                )
        else:
            y_pred_val, val_scores, native_pred_val, _ = predict_ocsvm(
                model=model,
                X_scaled=X_val_scaled,
                threshold=threshold,
            )

        fold_metrics = classification_metrics_from_preds(y_true_val, y_pred_val)
        fold_metrics.update(
            {
                "config_id": config_id,
                "config_tag": config_tag,
                "fold": fold_idx,
                "threshold": threshold,
                "prediction_mode": pred_mode,
                "n_train_normal": int(len(train_df)),
                "n_val_normal": int(len(val_df)),
            }
        )
        fold_rows.append(fold_metrics)

        preds_df = val_df[META_COLS].copy()
        preds_df["config_id"] = config_id
        preds_df["config_tag"] = config_tag
        preds_df["fold"] = fold_idx
        preds_df["y_true"] = y_true_val
        preds_df["y_pred"] = y_pred_val
        preds_df["true_label"] = "normal"
        preds_df["pred_label"] = np.where(y_pred_val == 1, "anomaly", "normal")
        preds_df["decision_score"] = val_scores
        preds_df["native_pred"] = native_pred_val
        preds_df["native_pred_label"] = np.where(native_pred_val == 1, "anomaly", "normal")
        if TARGET_ACTIVITY_CODE == "t-all":
            if thresholds_by_activity is None:
                preds_df["threshold"] = np.nan
            else:
                preds_df["threshold"] = preds_df["activity_code"].map(thresholds_by_activity)
        else:
            preds_df["threshold"] = threshold
        if TARGET_ACTIVITY_CODE == "t-all":
            config_thresholds = config["threshold_percentiles_by_activity"]
            preds_df["config_threshold"] = preds_df["activity_code"].map(config_thresholds)
        else:
            preds_df["config_threshold"] = config.get("threshold_percentile")
        all_cv_preds.append(preds_df)

        if SAVE_FEATURE_MATRICES:
            feature_dir = OUTPUT_DIR / "feature_matrices" / config_tag
            feature_dir.mkdir(parents=True, exist_ok=True)
            train_df.to_csv(feature_dir / f"fold_{fold_idx}_train_features.csv", index=False)
            val_df.to_csv(feature_dir / f"fold_{fold_idx}_val_features.csv", index=False)

    cv_preds_df = pd.concat(all_cv_preds, ignore_index=True)
    cv_preds_fp = CV_PREDS_DIR / f"{config_tag}__cv_predictions.csv"
    cv_preds_df.to_csv(cv_preds_fp, index=False)

    fold_metrics_df = pd.DataFrame(fold_rows)
    fold_metrics_fp = CV_PREDS_DIR / f"{config_tag}__cv_fold_metrics.csv"
    fold_metrics_df.to_csv(fold_metrics_fp, index=False)

    cv_y_true = cv_preds_df["y_true"].to_numpy(dtype=int)
    cv_y_pred = cv_preds_df["y_pred"].to_numpy(dtype=int)
    cv_metrics = classification_metrics_from_preds(cv_y_true, cv_y_pred)
    cv_accuracy_pct_total = accuracy_pct(cv_y_true, cv_y_pred)
    cv_accuracy_by_activity = accuracy_by_activity_pct(cv_preds_df)
    cv_accuracy_pct_mean_across_activities = mean_accuracy_across_activities(
        cv_accuracy_by_activity
    )

    _, cv_cm_df = build_confusion_matrix_df(cv_y_true, cv_y_pred)
    save_confusion_matrix_files(
        cv_cm_df,
        CV_CM_DIR / f"{config_tag}__cv_noerror_confusion_matrix.csv",
        CV_CM_DIR / f"{config_tag}__cv_noerror_confusion_matrix.txt",
        title=f"{config_tag} - CV No-Error Confusion Matrix",
    )

    if TARGET_ACTIVITY_CODE == "t-all":
        save_confusion_matrices_by_activity_one_file(
            preds_df=cv_preds_df,
            csv_fp=CV_CM_DIR / f"{config_tag}__cv_noerror_confusion_matrices_by_activity.csv",
            txt_fp=CV_CM_DIR / f"{config_tag}__cv_noerror_confusion_matrices_by_activity.txt",
            title_prefix=f"{config_tag} - CV No-Error",
        )


    mean_cv_fp = float(fold_metrics_df["false_positive_rate"].mean())
    std_cv_fp = float(fold_metrics_df["false_positive_rate"].std(ddof=0))

    # -------------------------
    # FINAL TRAIN ON ALL NORMAL
    # -------------------------
    if TARGET_ACTIVITY_CODE == "t-all":
        final_reference_stats = build_reference_stats_by_activity(
            normal_sessions,
            normal_meta,
        )
    else:
        final_reference_stats = build_reference_stats(normal_sessions)

    final_normal_df = build_feature_matrix(
        session_dfs=normal_sessions,
        meta=normal_meta,
        cols=cols,
        reference_stats=final_reference_stats,
    )

    X_final_normal = final_normal_df[cols].copy().astype(float)

    if TARGET_ACTIVITY_CODE == "t-all":
        final_scaler = fit_activity_scalers(
            feature_df=final_normal_df,
            scale_cols=[c for c in cols if not c.startswith("activity_")],
            activity_col="activity_code",
        )
        X_final_normal_scaled = apply_activity_scalers(
            feature_df=final_normal_df,
            cols=cols,
            scalers=final_scaler,
            activity_col="activity_code",
        )
    else:
        final_scaler = StandardScaler()
        X_final_normal_scaled = final_scaler.fit_transform(X_final_normal)

    final_model = make_ocsvm(config)
    final_model.fit(X_final_normal_scaled)

    final_train_scores = final_model.decision_function(X_final_normal_scaled).ravel()
    final_threshold = get_threshold(final_train_scores, config.get("threshold_percentile"))

    threshold_percentile = config.get("threshold_percentile")
    threshold_percentiles_by_activity = config.get("threshold_percentiles_by_activity")

    final_threshold = None
    final_thresholds_by_activity = None
    pred_mode_train = "native_ocsvm"

    if TARGET_ACTIVITY_CODE == "t-all":
        if threshold_percentiles_by_activity is not None:
            final_thresholds_by_activity = get_thresholds_by_activity(
                scores=final_train_scores,
                activity_codes=final_normal_df["activity_code"],
                percentiles_by_activity=threshold_percentiles_by_activity,
            )
            pred_mode_train = "percentile_threshold_by_activity"
    else:
        if threshold_percentile is not None:
            final_threshold = float(np.percentile(final_train_scores, float(threshold_percentile)))
            pred_mode_train = "percentile_threshold"

    y_true_train = np.zeros(len(final_normal_df), dtype=int)

    if TARGET_ACTIVITY_CODE == "t-all":
        if final_thresholds_by_activity is None:
            y_pred_train, train_scores, native_pred_train, _ = predict_ocsvm(
                model=final_model,
                X_scaled=X_final_normal_scaled,
                threshold=None,
            )
        else:
            
            missing = set(final_normal_df["activity_code"]) - set(final_thresholds_by_activity.keys())
            if missing:
                raise ValueError(f"Missing thresholds for activities in final train: {sorted(missing)}")
        
            y_pred_train, train_scores, native_pred_train, row_thresholds = predict_ocsvm_by_activity(
                model=final_model,
                X=X_final_normal_scaled,
                activity_codes=final_normal_df["activity_code"],
                thresholds_by_activity=final_thresholds_by_activity,
            )
    else:
        y_pred_train, train_scores, native_pred_train, _ = predict_ocsvm(
            model=final_model,
            X_scaled=X_final_normal_scaled,
            threshold=final_threshold,
        )

    final_train_metrics = classification_metrics_from_preds(y_true_train, y_pred_train)

    _, train_cm_df = build_confusion_matrix_df(y_true_train, y_pred_train)
    save_confusion_matrix_files(
        train_cm_df,
        TRAIN_CM_DIR / f"{config_tag}__train_noerror_confusion_matrix.csv",
        TRAIN_CM_DIR / f"{config_tag}__train_noerror_confusion_matrix.txt",
        title=f"{config_tag} - Final Train No-Error Confusion Matrix",
    )

    train_preds_df = final_normal_df[META_COLS].copy()
    train_preds_df["config_id"] = config_id
    train_preds_df["config_tag"] = config_tag
    train_preds_df["y_true"] = y_true_train
    train_preds_df["y_pred"] = y_pred_train
    train_preds_df["true_label"] = "normal"
    train_preds_df["pred_label"] = np.where(y_pred_train == 1, "anomaly", "normal")
    train_preds_df["decision_score"] = train_scores
    train_preds_df["native_pred"] = native_pred_train
    train_preds_df["native_pred_label"] = np.where(native_pred_train == 1, "anomaly", "normal")
    if TARGET_ACTIVITY_CODE == "t-all":
        if final_thresholds_by_activity is None:
            train_preds_df["threshold"] = np.nan
        else:
            train_preds_df["threshold"] = train_preds_df["activity_code"].map(final_thresholds_by_activity)
    else:
        train_preds_df["threshold"] = final_threshold
    train_preds_df["prediction_mode"] = pred_mode_train

    train_preds_fp = TRAIN_PREDS_DIR / f"{config_tag}__train_predictions.csv"
    train_preds_df.to_csv(train_preds_fp, index=False)

    if SAVE_MODEL_BUNDLE_PER_CONFIG:
        save_ocsvm_model_bundle(
            out_dir=MODEL_DIR / config_tag,
            target_activity_code=TARGET_ACTIVITY_CODE,
            model=final_model,
            scaler=final_scaler,
            cols=cols,
            reference_stats=final_reference_stats,
            threshold=final_threshold,
            thresholds_by_activity=final_thresholds_by_activity,
            prediction_mode=pred_mode_train,
            config=config,
        )

    # -------------------------
    # FINAL ERROR INFERENCE
    # -------------------------
    if len(error_sessions) > 0:
        error_df = build_feature_matrix(
            session_dfs=error_sessions,
            meta=error_meta,
            cols=cols,
            reference_stats=final_reference_stats,
        )

        pd.DataFrame(
            [[0, 0], [0, 0]],
            index=["Actual Normal", "Actual Anomaly"],
            columns=["Pred Normal", "Pred Anomaly"],
        ).to_csv(ERROR_CM_DIR / f"{config_tag}__error_confusion_matrix.csv")

        X_error = error_df.reindex(columns=cols, fill_value=0)[cols].copy().astype(float)
        if TARGET_ACTIVITY_CODE == "t-all":
            X_error_scaled = apply_activity_scalers(
                feature_df=error_df,
                cols=cols,
                scalers=final_scaler,
                activity_col="activity_code",
            )
        else:
            X_error_scaled = final_scaler.transform(X_error)

        y_true_error = np.ones(len(error_df), dtype=int)
        if TARGET_ACTIVITY_CODE == "t-all":
            if final_thresholds_by_activity is None:
                y_pred_error, error_scores, native_pred_error, _ = predict_ocsvm(
                    model=final_model,
                    X_scaled=X_error_scaled,
                    threshold=None,
                )
                pred_mode_error = "native_ocsvm"
            else:
                missing = set(error_df["activity_code"]) - set(final_thresholds_by_activity.keys())
                if missing:
                    raise ValueError(f"Missing thresholds for activities in final error: {sorted(missing)}")
        
                y_pred_error, error_scores, native_pred_error, row_thresholds = predict_ocsvm_by_activity(
                    model=final_model,
                    X=X_error_scaled,
                    activity_codes=error_df["activity_code"],
                    thresholds_by_activity=final_thresholds_by_activity,
                )
                pred_mode_error = "percentile_threshold_by_activity"
        else:
            y_pred_error, error_scores, native_pred_error, pred_mode_error = predict_ocsvm(
                model=final_model,
                X_scaled=X_error_scaled,
                threshold=final_threshold,
            )
        error_metrics = classification_metrics_from_preds(y_true_error, y_pred_error)
        error_accuracy_pct_total = accuracy_pct(y_true_error, y_pred_error)

        _, error_cm_df = build_confusion_matrix_df(y_true_error, y_pred_error)
        save_confusion_matrix_files(
            error_cm_df,
            ERROR_CM_DIR / f"{config_tag}__error_confusion_matrix.csv",
            ERROR_CM_DIR / f"{config_tag}__error_confusion_matrix.txt",
            title=f"{config_tag} - Final Error Confusion Matrix",
        )

        error_preds_df = error_df[META_COLS].copy()
        error_preds_df["config_id"] = config_id
        error_preds_df["config_tag"] = config_tag
        error_preds_df["y_true"] = y_true_error
        error_preds_df["y_pred"] = y_pred_error
        error_preds_df["true_label"] = "anomaly"
        error_preds_df["pred_label"] = np.where(y_pred_error == 1, "anomaly", "normal")
        error_preds_df["decision_score"] = error_scores
        error_preds_df["native_pred"] = native_pred_error
        error_preds_df["native_pred_label"] = np.where(native_pred_error == 1, "anomaly", "normal")
        if TARGET_ACTIVITY_CODE == "t-all":
            if final_thresholds_by_activity is None:
                error_preds_df["threshold"] = np.nan
            else:
                error_preds_df["threshold"] = error_preds_df["activity_code"].map(final_thresholds_by_activity)
        else:
            error_preds_df["threshold"] = final_threshold
        if TARGET_ACTIVITY_CODE == "t-all":
            config_thresholds = config["threshold_percentiles_by_activity"]
            error_preds_df["config_threshold"] = error_preds_df["activity_code"].map(config_thresholds)
        else:
            error_preds_df["config_threshold"] = config.get("threshold_percentile")

        error_preds_fp = ERROR_PREDS_DIR / f"{config_tag}__error_predictions.csv"
        error_preds_df.to_csv(error_preds_fp, index=False)

        error_accuracy_by_activity = accuracy_by_activity_pct(error_preds_df)
        error_accuracy_pct_mean_across_activities = mean_accuracy_across_activities(
            error_accuracy_by_activity
        )

        if TARGET_ACTIVITY_CODE == "t-all":
            save_confusion_matrices_by_activity_one_file(
                preds_df=error_preds_df,
                csv_fp=ERROR_CM_DIR / f"{config_tag}__error_confusion_matrices_by_activity.csv",
                txt_fp=ERROR_CM_DIR / f"{config_tag}__error_confusion_matrices_by_activity.txt",
                title_prefix=f"{config_tag} - Final Error",
            )

            cv_summary = build_activity_confusion_summary_table(
                preds_df=cv_preds_df,
                threshold_col="config_threshold",
            )
            cv_summary["result_type"] = "noerror"

            error_summary = build_activity_confusion_summary_table(
                preds_df=error_preds_df,
                threshold_col="config_threshold",
            )
            error_summary["result_type"] = "error"

            combined_summary = pd.concat(
                [cv_summary, error_summary],
                ignore_index=True,
            )

            combined_fp = OUTPUT_DIR / f"{config_tag}__combined_confusion_summary_by_activity.csv"
            combined_summary.to_csv(combined_fp, index=False)
    else:
        error_metrics = {
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "accuracy": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "false_positive_rate": math.nan,
            "false_negative_rate": math.nan,
        }

        error_accuracy_pct_total = math.nan
        error_accuracy_by_activity = {}
        error_accuracy_pct_mean_across_activities = math.nan


    # -------------------------
    # SUMMARY ROW
    # -------------------------
    summary_row = {
        "config_id": config_id,
        "config_tag": config_tag,
        "activity": TARGET_ACTIVITY_CODE,
        "n_features": len(cols),
        "feature_columns": json.dumps(cols),

        "kernel": config["kernel"],
        "gamma": config["gamma"],
        "nu": config["nu"],
        "threshold_percentile": config.get("threshold_percentile"),
        "cv_normal_tn": cv_metrics["tn"],
        "cv_normal_fp": cv_metrics["fp"],
        "cv_normal_fn": cv_metrics["fn"],
        "cv_normal_tp": cv_metrics["tp"],
        "cv_normal_accuracy": cv_metrics["accuracy"],
        "cv_normal_accuracy_pct_total": cv_accuracy_pct_total,
        "cv_normal_accuracy_pct_mean_across_activities": cv_accuracy_pct_mean_across_activities,
        "cv_normal_fp_rate": cv_metrics["false_positive_rate"],
        "cv_normal_precision": cv_metrics["precision"],
        "cv_normal_recall": cv_metrics["recall"],
        "cv_mean_fold_fp_rate": mean_cv_fp,
        "cv_std_fold_fp_rate": std_cv_fp,

        "final_train_normal_tn": final_train_metrics["tn"],
        "final_train_normal_fp": final_train_metrics["fp"],
        "final_train_normal_accuracy": final_train_metrics["accuracy"],
        "final_train_normal_fp_rate": final_train_metrics["false_positive_rate"],

        "final_error_tn": error_metrics["tn"],
        "final_error_fp": error_metrics["fp"],
        "final_error_fn": error_metrics["fn"],
        "final_error_tp": error_metrics["tp"],
        "final_error_accuracy": error_metrics["accuracy"],
        "final_error_accuracy_pct_total": error_accuracy_pct_total,
        "final_error_accuracy_pct_mean_across_activities": error_accuracy_pct_mean_across_activities,
        "final_error_precision": error_metrics["precision"],
        "final_error_recall": error_metrics["recall"],
        "final_error_fn_rate": error_metrics["false_negative_rate"],

        "final_threshold": final_threshold,
    }
 
    if TARGET_ACTIVITY_CODE == "t-all":
        summary_row["threshold_percentiles_by_activity"] = json.dumps(
            config["threshold_percentiles_by_activity"]
        )
        summary_row["final_thresholds_by_activity"] = json.dumps(
            final_thresholds_by_activity
        )
    else:
        summary_row["threshold_percentile"] = config["threshold_percentile"]
        summary_row["final_threshold"] = final_threshold    

    for activity_code, acc_val in sorted(cv_accuracy_by_activity.items()):
        safe_key = str(activity_code).replace("-", "_")
        summary_row[f"cv_normal_accuracy_pct_{safe_key}"] = acc_val

    for activity_code, acc_val in sorted(error_accuracy_by_activity.items()):
        safe_key = str(activity_code).replace("-", "_")
        summary_row[f"final_error_accuracy_pct_{safe_key}"] = acc_val

    # Ranking score
    if RANK_USING_NORMAL_ONLY:
        summary_row["rank_score"] = compute_normal_only_rank_score(
            mean_fp=mean_cv_fp,
            std_fp=std_cv_fp,
            target_fp=TARGET_NORMAL_FP_RATE,
        )
    else:
        summary_row["rank_score"] = (
            (1.0 - error_metrics["recall"]) +
            0.5 * mean_cv_fp +
            0.25 * std_cv_fp
        )

    print("\nSummary")
    print(f"  CV normal FP rate       : {cv_metrics['false_positive_rate']:.4f}")
    print(f"  CV mean fold FP rate    : {mean_cv_fp:.4f}")
    print(f"  CV std fold FP rate     : {std_cv_fp:.4f}")
    print(f"  Final train FP rate     : {final_train_metrics['false_positive_rate']:.4f}")
    print(f"  Final error recall      : {error_metrics['recall']:.4f}")
    print(f"  Final threshold         : {final_threshold}")

    return summary_row


# =========================================================
# MAIN
# =========================================================

def main():
    print(f"Target activity: {TARGET_ACTIVITY_CODE}")
    print(f"Normal dir:      {NORMAL_DIR}")
    print(f"Error dir:       {ERROR_DIR}")
    print(f"Output dir:      {OUTPUT_DIR}")

    cols = feature_columns(target_activity=TARGET_ACTIVITY_CODE, feature_config=FEATURE_CONFIG)
    print(f"Feature count:   {len(cols)}")
    print(f"Features:        {cols}")

    normal_sessions, normal_meta, error_sessions, error_meta = load_data(target_activity_code=TARGET_ACTIVITY_CODE, normal_dir=NORMAL_DIR, error_dir=ERROR_DIR, n_splits=N_SPLITS)

    print(f"Normal sessions: {len(normal_sessions)}")
    print(f"Error sessions:  {len(error_sessions)}")

    param_names = list(PARAM_GRID.keys())
    param_values = [PARAM_GRID[name] for name in param_names]

    all_configs = []
    for values in product(*param_values):
        cfg = dict(zip(param_names, values))
        all_configs.append(cfg)

    print(f"Configurations to test: {len(all_configs)}")

    summary_rows = []

    for config_id, config in enumerate(all_configs, start=1):
        try:
            row = evaluate_config(
                config_id=config_id,
                config=deepcopy(config),
                normal_sessions=normal_sessions,
                normal_meta=normal_meta,
                error_sessions=error_sessions,
                error_meta=error_meta,
                cols=cols,
            )
            summary_rows.append(row)
        except Exception as e:
            print(f"\nFAILED config {config_id}: {e}")
            summary_rows.append(
                {
                    "config_id": config_id,
                    "config_tag": f"cfg_{config_id:03d}",
                    "activity": TARGET_ACTIVITY_CODE,
                    "n_features": len(cols),
                    "feature_columns": json.dumps(cols),
                    "kernel": config.get("kernel"),
                    "gamma": config.get("gamma"),
                    "nu": config.get("nu"),
                    "threshold_percentile": config.get("threshold_percentile"),
                    "threshold_percentiles_by_activity": config.get("threshold_percentiles_by_activity"),
                    "rank_score": math.inf,
                    "error_message": str(e),
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    sort_cols = ["rank_score", "final_error_recall", "cv_mean_fold_fp_rate"]
    existing_sort_cols = [c for c in sort_cols if c in summary_df.columns]
    summary_df = summary_df.sort_values(
        existing_sort_cols,
        ascending=[True, False, True][:len(existing_sort_cols)]
    )

    summary_csv_fp = OUTPUT_DIR / f"tuning_summary_{TARGET_ACTIVITY_CODE}.csv"
    summary_json_fp = OUTPUT_DIR / f"tuning_summary_{TARGET_ACTIVITY_CODE}.json"
    summary_df.to_csv(summary_csv_fp, index=False)

    with open(summary_json_fp, "w") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2, default=str)

    print(f"\n{'=' * 90}")
    print("TOP CONFIGS")
    display_cols = [
        "config_id",
        "rank_score",
        "kernel",
        "gamma",
        "nu",
        "threshold_percentile",
        "cv_mean_fold_fp_rate",
        "cv_std_fold_fp_rate",
        "final_train_normal_fp_rate",
        "final_error_recall",
        "final_error_precision",
        "final_threshold",
    ]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[display_cols].head(20).to_string(index=False))

    print("\nSaved outputs")
    print(f"  Summary CSV            -> {summary_csv_fp}")
    print(f"  Summary JSON           -> {summary_json_fp}")
    print(f"  CV predictions dir     -> {CV_PREDS_DIR}")
    print(f"  Train predictions dir  -> {TRAIN_PREDS_DIR}")
    print(f"  Error predictions dir  -> {ERROR_PREDS_DIR}")
    if SAVE_MODEL_BUNDLE_PER_CONFIG:
        print(f"  Model bundles dir      -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
