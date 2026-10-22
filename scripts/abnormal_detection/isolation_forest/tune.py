#!/usr/bin/env python3
"""
All-in-one Isolation Forest tuning script.

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
from abnormal_activity_detection.anomaly.configs.iforest import (
    TARGET_ACTIVITY_CODE,
    N_SPLITS,
    FEATURE_CONFIG,
    META_COLS,
)
from abnormal_activity_detection.anomaly.iforest import (
    predict_iforest_by_activity,
    predict_iforest,
    get_threshold,
    save_model_bundle,
    make_iforest,
    get_model_input_for_fit_and_infer,
    get_model_input_for_final_train,
    apply_saved_scaling_to_error,
    config_to_tag,
)

from abnormal_activity_detection.features.anomaly import (
    build_feature_matrix,
    build_reference_stats,
    build_reference_stats_by_activity,
    feature_columns,
    build_feature_matrix_train_loo_reference,
)
from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir
from abnormal_activity_detection.evaluation import (
    get_thresholds_by_activity,
    save_confusion_matrix_files,
    save_confusion_matrices_by_activity_one_file,
    build_activity_confusion_summary_table,
    classification_metrics_from_preds,
    build_confusion_matrix_df,
    mean_accuracy_across_activities,
    accuracy_by_activity_pct
)

from abnormal_activity_detection.data import load_data

# =========================================================
# CONFIG
# =========================================================

RANK_USING_NORMAL_ONLY = True
TARGET_NORMAL_FP_RATE = 0.10

if TARGET_ACTIVITY_CODE == "t-all":
    PARAM_GRID = {
        "n_estimators": [400],
        "max_samples": [1.0],
        "max_features": [0.4],
        "contamination": [0.01],
        "threshold_percentiles_by_activity": [
            {"t1": 20, "t2": 18, "t3": 6, "t4": 10, "t5": 15},
            {"t1": 20, "t2": 18, "t3": 7, "t4": 10, "t5": 15},
        ],
        "bootstrap": [False],
    }


else:
    PARAM_GRID = {
        "n_estimators": [400],
        "max_samples": [1.0],
        "max_features": [0.4],
        "contamination": [0.01],
        "threshold_percentile": [18],
        "bootstrap": [False],
    }

SAVE_MODEL_BUNDLE_PER_CONFIG = False
SAVE_FEATURE_MATRICES = False


# =========================================================
# PATHS
# =========================================================

NORMAL_DIR, ERROR_DIR = get_dataset_dirs()
OUTPUT_DIR = get_results_dir("abnormal_detection", "isolation_forest", TARGET_ACTIVITY_CODE, "iforest_tuning_all_in_one")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CV_PREDS_DIR = OUTPUT_DIR / "cv_predictions"
ERROR_PREDS_DIR = OUTPUT_DIR / "error_predictions"
MODEL_DIR = OUTPUT_DIR / "model_bundles"

CM_DIR = OUTPUT_DIR / "confusion_matrices"
CV_CM_DIR = CM_DIR / "cv_noerror"
ERROR_CM_DIR = CM_DIR / "error"

for d in [CV_CM_DIR, ERROR_CM_DIR]:
    d.mkdir(parents=True, exist_ok=True)

for d in [CV_PREDS_DIR, ERROR_PREDS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if SAVE_MODEL_BUNDLE_PER_CONFIG:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

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
    
    if TARGET_ACTIVITY_CODE == "t-all":
        if "threshold_percentiles_by_activity" not in config:
            raise ValueError("t-all requires threshold_percentiles_by_activity")
    else:
        if "threshold_percentile" not in config:
            raise ValueError("non t-all requires threshold_percentile")
    
    
    config_tag = config_to_tag(config_id, config)
    print(f"\n{'=' * 90}")
    print(f"CONFIG {config_id}: {config_tag}")
    print(json.dumps(config, indent=2))

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
        if TARGET_ACTIVITY_CODE == "t-all":
            train_counts = pd.Series(
                [normal_meta[i]["activity_code"] for i in train_idx]
            ).value_counts().sort_index()

            val_counts = pd.Series(
                [normal_meta[i]["activity_code"] for i in val_idx]
            ).value_counts().sort_index()

            print(f"\nFold {fold_idx} train activity counts:")
            print(train_counts)

            print(f"\nFold {fold_idx} val activity counts:")
            print(val_counts)

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

        X_train_model, X_val_model, _ = get_model_input_for_fit_and_infer(
            train_df=train_df,
            infer_df=val_df,
            cols=cols,
            target_activity_code=TARGET_ACTIVITY_CODE,
        )

        if X_train_model.empty:
            raise RuntimeError(f"{config_tag} fold {fold_idx}: empty training matrix")
        if X_val_model.empty:
            raise RuntimeError(f"{config_tag} fold {fold_idx}: empty validation matrix")

        model = make_iforest(config)
        model.fit(X_train_model)

        train_scores = model.decision_function(X_train_model).ravel()
        y_true_val = np.zeros(len(val_df), dtype=int)

        if TARGET_ACTIVITY_CODE == "t-all":
            percentiles_by_activity = config.get("threshold_percentiles_by_activity")

            if not percentiles_by_activity:
                raise ValueError("Missing threshold_percentiles_by_activity for t-all")

            thresholds_by_activity = get_thresholds_by_activity(
                scores=train_scores,
                activity_codes=train_df["activity_code"],
                percentiles_by_activity=percentiles_by_activity,
            )

            threshold = None

            y_pred_val, val_scores, native_pred_val, pred_mode = predict_iforest_by_activity(
                model=model,
                X=X_val_model,
                activity_codes=val_df["activity_code"],
                thresholds_by_activity=thresholds_by_activity,
            )
        else:
            threshold_percentile = config.get("threshold_percentile")
            if threshold_percentile is None:
                raise ValueError("Missing threshold_percentile for non t-all")

            threshold = get_threshold(train_scores, threshold_percentile)
            thresholds_by_activity = None

            
            y_pred_val, val_scores, native_pred_val, pred_mode = predict_iforest(
                model=model,
                X_scaled=X_val_model,
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
            preds_df["threshold"] = preds_df["activity_code"].map(percentiles_by_activity)
        else:
            preds_df["threshold"] = threshold
        preds_df["prediction_mode"] = pred_mode
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
            csv_fp=CV_CM_DIR / f"{config_tag}__cv_confusion_matrices_by_activity.csv",
            txt_fp=CV_CM_DIR / f"{config_tag}__cv_confusion_matrices_by_activity.txt",
            title_prefix=f"{config_tag} CV",
        )

    mean_cv_fp = float(fold_metrics_df["false_positive_rate"].mean())
    std_cv_fp = float(fold_metrics_df["false_positive_rate"].std(ddof=0))

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

    X_final_normal_model, final_scalers = get_model_input_for_final_train(
        final_normal_df=final_normal_df,
        cols=cols,
    )

    final_model = make_iforest(config)
    final_model.fit(X_final_normal_model)

    final_train_scores = final_model.decision_function(X_final_normal_model).ravel()

    if TARGET_ACTIVITY_CODE == "t-all":
        percentiles_by_activity = config.get("threshold_percentiles_by_activity")

        final_thresholds_by_activity = get_thresholds_by_activity(
            scores=final_train_scores,
            activity_codes=final_normal_df["activity_code"],
            percentiles_by_activity=percentiles_by_activity,
        )

        final_threshold = None

    else:
        final_threshold = get_threshold(final_train_scores, config.get("threshold_percentile"))
        final_thresholds_by_activity = None



    if SAVE_MODEL_BUNDLE_PER_CONFIG:
        out_dir = MODEL_DIR / config_tag
        save_model_bundle(
            model=final_model,
            scaler=final_scalers,
            cols=cols,
            reference_stats=final_reference_stats,
            threshold=final_threshold,
            config=config,
            target_activity_code=TARGET_ACTIVITY_CODE,
            out_dir=out_dir,
            thresholds_by_activity=final_thresholds_by_activity,
            prediction_mode=(
                "percentile_threshold_by_activity"
                if final_thresholds_by_activity is not None
                else ("native_iforest" if final_threshold is None else "percentile_threshold")
            ),
        )

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

        X_error_model = apply_saved_scaling_to_error(
            error_df=error_df,
            cols=cols,
            scalers=final_scalers,
            target_activity_code=TARGET_ACTIVITY_CODE
        )

        y_true_error = np.ones(len(error_df), dtype=int)

        if TARGET_ACTIVITY_CODE == "t-all":
            y_pred_error, error_scores, native_pred_error, pred_mode_error = predict_iforest_by_activity(
                model=final_model,
                X=X_error_model,
                activity_codes=error_df["activity_code"],
                thresholds_by_activity=final_thresholds_by_activity,
            )
        else:
            y_pred_error, error_scores, native_pred_error, pred_mode_error = predict_iforest(
                model=final_model,
                X_scaled=X_error_model,
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
            error_preds_df["threshold"] = error_preds_df["activity_code"].map(percentiles_by_activity)
        else:
            error_preds_df["threshold"] = final_threshold
        error_preds_df["prediction_mode"] = pred_mode_error

        error_preds_fp = ERROR_PREDS_DIR / f"{config_tag}__error_predictions.csv"
        error_preds_df.to_csv(error_preds_fp, index=False)

        if TARGET_ACTIVITY_CODE == "t-all":
            save_confusion_matrices_by_activity_one_file(
                preds_df=error_preds_df,
                csv_fp=ERROR_CM_DIR / f"{config_tag}__error_confusion_matrices_by_activity.csv",
                txt_fp=ERROR_CM_DIR / f"{config_tag}__error_confusion_matrices_by_activity.txt",
                title_prefix=f"{config_tag} Error",
            )

            cv_summary = build_activity_confusion_summary_table(
                preds_df=cv_preds_df,
                threshold_col="threshold",
            )
            cv_summary["result_type"] = "noerror"

            error_summary = build_activity_confusion_summary_table(
                preds_df=error_preds_df,
                threshold_col="threshold",
            )
            error_summary["result_type"] = "error"

            combined_summary = pd.concat(
                [cv_summary, error_summary],
                ignore_index=True,
            )

            combined_fp = OUTPUT_DIR / f"{config_tag}__combined_confusion_summary_by_activity.csv"
            combined_summary.to_csv(combined_fp, index=False)



        error_accuracy_by_activity = accuracy_by_activity_pct(error_preds_df)
        error_accuracy_pct_mean_across_activities = mean_accuracy_across_activities(
            error_accuracy_by_activity
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
        "n_features": len(cols),
        "feature_columns": json.dumps(cols),

        "n_estimators": config["n_estimators"],
        "max_samples": config["max_samples"],
        "max_features": config["max_features"],
        "contamination": config["contamination"],
        "bootstrap": config["bootstrap"],
        "random_state": config["random_state"],

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
    print(f"  CV normal accuracy % total              : {cv_accuracy_pct_total:.2f}")
    print(f"  CV normal accuracy % mean across acts   : {cv_accuracy_pct_mean_across_activities:.2f}")
    print(f"  CV normal FP rate                       : {cv_metrics['false_positive_rate']:.4f}")
    print(f"  CV mean fold FP rate                    : {mean_cv_fp:.4f}")
    print(f"  CV std fold FP rate                     : {std_cv_fp:.4f}")
    print(f"  Final error accuracy % total            : {error_accuracy_pct_total:.2f}")
    print(f"  Final error accuracy % mean across acts : {error_accuracy_pct_mean_across_activities:.2f}")
    print(f"  Final error recall                      : {error_metrics['recall']:.4f}")
    print(f"  Final threshold                         : {final_threshold}")

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
        cfg["random_state"] = 42
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
                    "n_estimators": config.get("n_estimators"),
                    "max_samples": config.get("max_samples"),
                    "max_features": config.get("max_features"),
                    "contamination": config.get("contamination"),
                    "bootstrap": config.get("bootstrap"),
                    "random_state": config.get("random_state"),
                    "threshold_percentile": config.get("threshold_percentile"),
                    "rank_score": math.inf,
                    "error_message": str(e),
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    sort_cols = ["rank_score", "final_error_recall", "cv_mean_fold_fp_rate"]
    existing_sort_cols = [c for c in sort_cols if c in summary_df.columns]
    ascending_flags = [True, False, True][:len(existing_sort_cols)]
    summary_df = summary_df.sort_values(existing_sort_cols, ascending=ascending_flags)

    summary_csv_fp = OUTPUT_DIR / f"tuning_summary_{TARGET_ACTIVITY_CODE}.csv"
    summary_json_fp = OUTPUT_DIR / f"tuning_summary_{TARGET_ACTIVITY_CODE}.json"
    summary_df.to_csv(summary_csv_fp, index=False)

    with open(summary_json_fp, "w") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    print(f"\n{'=' * 90}")
    print("TOP CONFIGS")
    display_cols = [
        "config_id",
        "rank_score",
        "n_estimators",
        "max_samples",
        "max_features",
        "contamination",
        "threshold_percentile",
        "cv_normal_accuracy_pct_total",
        "cv_normal_accuracy_pct_mean_across_activities",
        "cv_mean_fold_fp_rate",
        "cv_std_fold_fp_rate",
        "final_train_normal_fp_rate",
        "final_error_accuracy_pct_total",
        "final_error_accuracy_pct_mean_across_activities",
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
    print(f"  Error predictions dir  -> {ERROR_PREDS_DIR}")
    if SAVE_MODEL_BUNDLE_PER_CONFIG:
        print(f"  Model bundles dir      -> {MODEL_DIR}")


if __name__ == "__main__":
    main()