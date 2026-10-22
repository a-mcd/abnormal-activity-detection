#!/usr/bin/env python3
"""
All-in-one One-Class SVM feature-combination tuning script.

For each configuration, this script:
1. Runs 5-fold CV on NORMAL sessions only
2. Trains a final model on ALL normal sessions
3. Runs inference on ERROR sessions
4. Saves one summary row per configuration

Important:
- Model fitting is always done on NO-ERROR data only.
- ERROR data is used only for reporting after final training.
"""

import json
import math
from copy import deepcopy
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from abnormal_activity_detection.anomaly.ocsvm import (
    predict_ocsvm_by_activity,
    config_to_tag,
    save_ocsvm_model_bundle,
    get_threshold,
    predict_ocsvm,
)
from abnormal_activity_detection.anomaly.configs.ocsvm import (
    TARGET_ACTIVITY_CODE,
    N_SPLITS,
    META_COLS,
)
from abnormal_activity_detection.data import load_data
from abnormal_activity_detection.features.anomaly import (
    build_feature_matrix,
    build_feature_matrix_train_loo_reference,
    build_reference_stats,
    build_reference_stats_by_activity,
)
from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir
from abnormal_activity_detection.evaluation import (
    apply_activity_scalers,
    fit_activity_scalers,
    get_thresholds_by_activity,
    save_confusion_matrix_files,
    build_activity_confusion_summary_table,
    classification_metrics_from_preds
)

# =========================================================
# CONFIG
# =========================================================

RANK_USING_NORMAL_ONLY = False
TARGET_NORMAL_FP_RATE = 0.10
SAVE_MODEL_BUNDLE_PER_CONFIG = False
SAVE_FEATURE_MATRICES = False

# -------------------------
# Feature sets to test
# -------------------------
FEATURE_SET_GRID = [
    [
        "EPISODE_DURATION",
        "NUM_EVENTS",
        "UNIQUE_SENSOR_MESSAGE_COUNT",
        "P90",
        "MEAN_TRANSITION_LOG_LIK",
        "P90_MINUS_P50",
        "MEAN_DT",
        "analogue_ad1_a_last_value",
        "analogue_ad1_b::ACTIVE",
        "SWITCH_RATE",
        "MISSING_EVENT_RATIO",
        "AD1_BC_ACTIVE_LAST_8S",
        "EVENT_RATE",
        "LAST_TAP_VALUE_LT_2_5"
    ],
    [
        "EPISODE_DURATION",
        "NUM_EVENTS",
        "UNIQUE_SENSOR_MESSAGE_COUNT",
        "P90",
        "MEAN_DT",
    ],
    [
        "EPISODE_DURATION",
        "NUM_EVENTS",
        "UNIQUE_SENSOR_MESSAGE_COUNT",
        "P90",
        "P90_MINUS_P50",
    ],
    [
        "EPISODE_DURATION",
        "NUM_EVENTS",
        "UNIQUE_SENSOR_MESSAGE_COUNT",
        "P90",
        "MEAN_TRANSITION_LOG_LIK",
    ],
    [
        "EPISODE_DURATION",
        "NUM_EVENTS",
        "UNIQUE_SENSOR_MESSAGE_COUNT",
        "P90",
        "MEAN_DT",
        "P90_MINUS_P50",
        "MEAN_TRANSITION_LOG_LIK",
    ],
    [
        "EPISODE_DURATION",
        "NUM_EVENTS",
        "UNIQUE_SENSOR_MESSAGE_COUNT",
        "P90",
        "MEAN_DT",
        "P90_MINUS_P50",
        "MEAN_TRANSITION_LOG_LIK",
        "MISSING_EVENT_RATIO",
    ],
]

# -------------------------
# Model parameter grid
# -------------------------
if TARGET_ACTIVITY_CODE == "t-all":
    PARAM_GRID = {
        "kernel": ["linear", "rbf"],
        "gamma": ["scale"],
        "nu": [0.05, 0.1],
        "threshold_percentiles_by_activity": [
            {"t1": 20.0, "t2": 27.0, "t3": 18.0, "t4": 10.0, "t5": 1.0},
        ],
    }
else:
    PARAM_GRID = {
        "kernel": ["linear"],
        "gamma": ["scale"],
        "nu": [0.01, 0.02, 0.03, 0.05],
        "threshold_percentile": [1],
    }

# =========================================================
# PATHS
# =========================================================

NORMAL_DIR, ERROR_DIR = get_dataset_dirs()
OUTPUT_DIR = get_results_dir("abnormal_detection", "one_class_svm", TARGET_ACTIVITY_CODE, "ocsvm_feature_combo_tuning")
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




def make_ocsvm(config: dict) -> OneClassSVM:
    return OneClassSVM(
        kernel=config["kernel"],
        gamma=config["gamma"],
        nu=config["nu"],
    )

def evaluate_config(
    config_id: int,
    config: dict,
    normal_sessions,
    normal_meta,
    error_sessions,
    error_meta,
    cols,
):
    config_tag = config_to_tag(config_id, config, cols)
    print(f"\n{'=' * 90}")
    print(f"CONFIG {config_id}: {config_tag}")
    print("Features:", cols)
    print(json.dumps(config, indent=2, default=str))

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
            reference_stats = build_reference_stats_by_activity(train_normal_dfs, train_normal_meta)
        else:
            reference_stats = build_reference_stats(train_normal_dfs)

        train_df = build_feature_matrix_train_loo_reference(
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

        threshold = None
        thresholds_by_activity = None
        pred_mode = "native_ocsvm"

        if TARGET_ACTIVITY_CODE == "t-all":
            if threshold_percentiles_by_activity is not None:
                thresholds_by_activity = get_thresholds_by_activity(
                    scores=train_scores,
                    activity_codes=train_df["activity_code"],
                    percentiles_by_activity=threshold_percentiles_by_activity,
                )
                pred_mode = "percentile_threshold_by_activity"
        else:
            threshold, pred_mode = get_threshold(train_scores, threshold_percentile)

        y_true_val = np.zeros(len(val_df), dtype=int)

        if TARGET_ACTIVITY_CODE == "t-all":
            if thresholds_by_activity is None:
                y_pred_val, val_scores, native_pred_val, _ = predict_ocsvm(
                    model=model,
                    X_scaled=X_val_scaled,
                    threshold=None,
                )
            else:
                y_pred_val, val_scores, native_pred_val, _ = predict_ocsvm_by_activity(
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

        if TARGET_ACTIVITY_CODE == "t-all":
            if thresholds_by_activity is None:
                preds_df["threshold"] = np.nan
            else:
                preds_df["threshold"] = preds_df["activity_code"].map(thresholds_by_activity)
            preds_df["config_threshold"] = preds_df["activity_code"].map(config["threshold_percentiles_by_activity"])
        else:
            preds_df["threshold"] = threshold
            preds_df["config_threshold"] = config.get("threshold_percentile")

        preds_df["prediction_mode"] = pred_mode
        all_cv_preds.append(preds_df)

    cv_preds_df = pd.concat(all_cv_preds, ignore_index=True)
    cv_preds_df.to_csv(CV_PREDS_DIR / f"{config_tag}__cv_predictions.csv", index=False)

    fold_metrics_df = pd.DataFrame(fold_rows)
    fold_metrics_df.to_csv(CV_PREDS_DIR / f"{config_tag}__cv_fold_metrics.csv", index=False)

    cv_y_true = cv_preds_df["y_true"].to_numpy(dtype=int)
    cv_y_pred = cv_preds_df["y_pred"].to_numpy(dtype=int)
    cv_metrics = classification_metrics_from_preds(cv_y_true, cv_y_pred)
    cv_accuracy_pct_total = accuracy_pct(cv_y_true, cv_y_pred)
    cv_accuracy_by_activity = accuracy_by_activity_pct(cv_preds_df)
    cv_accuracy_pct_mean_across_activities = mean_accuracy_across_activities(cv_accuracy_by_activity)

    _, cv_cm_df = build_confusion_matrix_df(cv_y_true, cv_y_pred)
    save_confusion_matrix_files(
        cv_cm_df,
        CV_CM_DIR / f"{config_tag}__cv_noerror_confusion_matrix.csv",
        CV_CM_DIR / f"{config_tag}__cv_noerror_confusion_matrix.txt",
        title=f"{config_tag} - CV No-Error Confusion Matrix",
    )

    mean_cv_fp = float(fold_metrics_df["false_positive_rate"].mean())
    std_cv_fp = float(fold_metrics_df["false_positive_rate"].std(ddof=0))

    if TARGET_ACTIVITY_CODE == "t-all":
        final_reference_stats = build_reference_stats_by_activity(normal_sessions, normal_meta)
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
    final_threshold = None
    final_thresholds_by_activity = None
    pred_mode_train = "native_ocsvm"

    if TARGET_ACTIVITY_CODE == "t-all":
        if config.get("threshold_percentiles_by_activity") is not None:
            final_thresholds_by_activity = get_thresholds_by_activity(
                scores=final_train_scores,
                activity_codes=final_normal_df["activity_code"],
                percentiles_by_activity=config["threshold_percentiles_by_activity"],
            )
            pred_mode_train = "percentile_threshold_by_activity"
    else:
        threshold, pred_mode = get_threshold(train_scores, threshold_percentile)

    y_true_train = np.zeros(len(final_normal_df), dtype=int)

    if TARGET_ACTIVITY_CODE == "t-all":
        if final_thresholds_by_activity is None:
            y_pred_train, train_scores, native_pred_train, _ = predict_ocsvm(
                model=final_model,
                X_scaled=X_final_normal_scaled,
                threshold=None,
            )
        else:
            y_pred_train, train_scores, native_pred_train, _ = predict_ocsvm_by_activity(
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

    train_preds_df = final_normal_df[META_COLS].copy()
    train_preds_df["config_id"] = config_id
    train_preds_df["config_tag"] = config_tag
    train_preds_df["y_true"] = y_true_train
    train_preds_df["y_pred"] = y_pred_train
    train_preds_df["decision_score"] = train_scores
    train_preds_df["native_pred"] = native_pred_train
    train_preds_df.to_csv(TRAIN_PREDS_DIR / f"{config_tag}__train_predictions.csv", index=False)

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

    if len(error_sessions) > 0:
        error_df = build_feature_matrix(
            session_dfs=error_sessions,
            meta=error_meta,
            cols=cols,
            reference_stats=final_reference_stats,
        )

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
                y_pred_error, error_scores, native_pred_error, _ = predict_ocsvm_by_activity(
                    model=final_model,
                    X=X_error_scaled,
                    activity_codes=error_df["activity_code"],
                    thresholds_by_activity=final_thresholds_by_activity,
                )
                pred_mode_error = "percentile_threshold_by_activity"
        else:
            y_pred_error, error_scores, native_pred_error, _ = predict_ocsvm(
                model=final_model,
                X_scaled=X_error_scaled,
                threshold=final_threshold,
            )

        error_metrics = classification_metrics_from_preds(y_true_error, y_pred_error)
        error_accuracy_pct_total = accuracy_pct(y_true_error, y_pred_error)

        error_preds_df = error_df[META_COLS].copy()
        error_preds_df["config_id"] = config_id
        error_preds_df["config_tag"] = config_tag
        error_preds_df["y_true"] = y_true_error
        error_preds_df["y_pred"] = y_pred_error
        error_preds_df["decision_score"] = error_scores
        error_preds_df["native_pred"] = native_pred_error

        if TARGET_ACTIVITY_CODE == "t-all":
            error_preds_df["config_threshold"] = error_preds_df["activity_code"].map(config["threshold_percentiles_by_activity"])
        else:
            error_preds_df["config_threshold"] = config.get("threshold_percentile")

        error_preds_df.to_csv(ERROR_PREDS_DIR / f"{config_tag}__error_predictions.csv", index=False)

        error_accuracy_by_activity = accuracy_by_activity_pct(error_preds_df)
        error_accuracy_pct_mean_across_activities = mean_accuracy_across_activities(error_accuracy_by_activity)

        if TARGET_ACTIVITY_CODE == "t-all":
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

            combined_summary = pd.concat([cv_summary, error_summary], ignore_index=True)
            combined_summary.to_csv(
                OUTPUT_DIR / f"{config_tag}__combined_confusion_summary_by_activity.csv",
                index=False,
            )
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

    summary_row = {
        "config_id": config_id,
        "config_tag": config_tag,
        "activity": TARGET_ACTIVITY_CODE,
        "feature_columns": json.dumps(cols),
        "n_features": len(cols),
        "kernel": config["kernel"],
        "gamma": config["gamma"],
        "nu": config["nu"],
        "threshold_percentile": config.get("threshold_percentile"),
        "cv_normal_accuracy": cv_metrics["accuracy"],
        "cv_normal_accuracy_pct_total": cv_accuracy_pct_total,
        "cv_normal_accuracy_pct_mean_across_activities": cv_accuracy_pct_mean_across_activities,
        "cv_normal_fp_rate": cv_metrics["false_positive_rate"],
        "cv_mean_fold_fp_rate": mean_cv_fp,
        "cv_std_fold_fp_rate": std_cv_fp,
        "final_train_normal_fp_rate": final_train_metrics["false_positive_rate"],
        "final_error_accuracy": error_metrics["accuracy"],
        "final_error_accuracy_pct_total": error_accuracy_pct_total,
        "final_error_accuracy_pct_mean_across_activities": error_accuracy_pct_mean_across_activities,
        "final_error_recall": error_metrics["recall"],
        "final_error_precision": error_metrics["precision"],
        "rank_score": (
            compute_normal_only_rank_score(mean_cv_fp, std_cv_fp, TARGET_NORMAL_FP_RATE)
            if RANK_USING_NORMAL_ONLY
            else ((1.0 - error_metrics["recall"]) + 0.5 * mean_cv_fp + 0.25 * std_cv_fp)
        ),
    }

    for activity_code, acc_val in sorted(cv_accuracy_by_activity.items()):
        summary_row[f"cv_normal_accuracy_pct_{str(activity_code).replace('-', '_')}"] = acc_val

    for activity_code, acc_val in sorted(error_accuracy_by_activity.items()):
        summary_row[f"final_error_accuracy_pct_{str(activity_code).replace('-', '_')}"] = acc_val

    return summary_row


def main():
    print(f"Target activity: {TARGET_ACTIVITY_CODE}")
    print(f"Normal dir:      {NORMAL_DIR}")
    print(f"Error dir:       {ERROR_DIR}")
    print(f"Output dir:      {OUTPUT_DIR}")

    normal_sessions, normal_meta, error_sessions, error_meta = load_data()

    param_names = list(PARAM_GRID.keys())
    param_values = [PARAM_GRID[name] for name in param_names]

    all_configs = []
    config_id = 0
    for feature_set in FEATURE_SET_GRID:
        for values in product(*param_values):
            config_id += 1
            cfg = dict(zip(param_names, values))
            all_configs.append((config_id, feature_set, cfg))

    print(f"Configurations to test: {len(all_configs)}")

    summary_rows = []

    for config_id, cols, config in all_configs:
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
                    "activity": TARGET_ACTIVITY_CODE,
                    "feature_columns": json.dumps(cols),
                    "n_features": len(cols),
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
    summary_df = summary_df.sort_values(
        ["rank_score", "final_error_recall", "cv_mean_fold_fp_rate"],
        ascending=[True, False, True],
    )

    summary_csv_fp = OUTPUT_DIR / f"feature_combo_tuning_summary_{TARGET_ACTIVITY_CODE}.csv"
    summary_json_fp = OUTPUT_DIR / f"feature_combo_tuning_summary_{TARGET_ACTIVITY_CODE}.json"

    summary_df.to_csv(summary_csv_fp, index=False)
    with open(summary_json_fp, "w") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2, default=str)

    print("\nTOP CONFIGS")
    display_cols = [
        "config_id",
        "n_features",
        "feature_columns",
        "kernel",
        "gamma",
        "nu",
        "threshold_percentile",
        "cv_mean_fold_fp_rate",
        "final_train_normal_fp_rate",
        "final_error_recall",
        "final_error_precision",
        "rank_score",
    ]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[display_cols].head(20).to_string(index=False))

    print("\nSaved outputs")
    print(f"  Summary CSV -> {summary_csv_fp}")
    print(f"  Summary JSON -> {summary_json_fp}")


if __name__ == "__main__":
    main()