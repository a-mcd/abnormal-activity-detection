#!/usr/bin/env python3
"""
Isolation Forest PER-ACTIVITY with 5-Fold Cross-Validation on NORMAL data only.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold 
from sklearn.metrics import classification_report

from abnormal_activity_detection.anomaly.configs.iforest import (
    TARGET_ACTIVITY_CODE,
    N_SPLITS,
    IFOREST_CONFIG,
    FEATURE_CONFIG,
)
from abnormal_activity_detection.anomaly.iforest import (
    predict_iforest_by_activity,
    predict_iforest,
    make_iforest,
    get_threshold,
)
from abnormal_activity_detection.data import load_sessions_for_activity
from abnormal_activity_detection.features.anomaly import (
    build_feature_matrix,
    build_reference_stats,
    build_reference_stats_by_activity,
    feature_columns,
    build_feature_matrix_train_loo_reference,
)
from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir
from abnormal_activity_detection.evaluation import get_thresholds_by_activity, print_confusion_matrix, save_confusion_matrices_by_activity_one_file
from abnormal_activity_detection.evaluation import (
    fit_activity_scalers,
    apply_activity_scalers,
    save_confusion_matrix_files,
    build_confusion_matrix_df
)

iforest_config = IFOREST_CONFIG[TARGET_ACTIVITY_CODE]

N_ESTIMATORS = iforest_config["n_estimators"]
MAX_SAMPLES = iforest_config["max_samples"]
CONTAMINATION = iforest_config["contamination"]
MAX_FEATURES = iforest_config["max_features"]
BOOTSTRAP = iforest_config["bootstrap"]
RANDOM_STATE = iforest_config["random_state"]
THRESHOLD_PERCENTILE = iforest_config.get("threshold_percentile")
THRESHOLD_PERCENTILES_BY_ACTIVITY = iforest_config.get("threshold_percentiles_by_activity")

print("TARGET_ACTIVITY_CODE:", TARGET_ACTIVITY_CODE)
print("FEATURE_CONFIG[t1]:", FEATURE_CONFIG["t1"])
print("IFOREST_CONFIG[t1]:", IFOREST_CONFIG["t1"])
print("feature_utils file:", build_feature_matrix.__module__)
print("eval file:", __file__)


NORMAL_DIR, _ = get_dataset_dirs()
OUTPUT_DIR = get_results_dir("abnormal_detection", "isolation_forest", TARGET_ACTIVITY_CODE, "isolation_forest_normal_cv")



def debug_feature_shift(X_train, X_val, fold_idx, top_n=15):
    print(f"\n=== Fold {fold_idx} Feature Debug ===")

    print("Train shape:", X_train.shape)
    print("Val shape:  ", X_val.shape)
    print("Same columns/order:", X_train.columns.equals(X_val.columns))
    print("Num feature columns:", len(X_train.columns))
    print("Any duplicate columns in train:", X_train.columns.duplicated().any())
    print("Any duplicate columns in val:  ", X_val.columns.duplicated().any())

    print("Train NaN count:", int(X_train.isna().sum().sum()))
    print("Val NaN count:  ", int(X_val.isna().sum().sum()))
    print("Train inf count:", int(np.isinf(X_train.to_numpy(dtype=float)).sum()))
    print("Val inf count:  ", int(np.isinf(X_val.to_numpy(dtype=float)).sum()))

    train_nonzero = (X_train != 0).sum(axis=1)
    val_nonzero = (X_val != 0).sum(axis=1)
    print("Train nonzero features per row:")
    print(train_nonzero.describe())
    print("Val nonzero features per row:")
    print(val_nonzero.describe())

    train_means = X_train.mean(axis=0)
    val_means = X_val.mean(axis=0)
    mean_diff = (val_means - train_means).abs().sort_values(ascending=False)

    print(f"\nTop {top_n} raw mean shifts:")
    for c in mean_diff.head(top_n).index:
        print(
            f"{c}: train_mean={train_means[c]:.6f}, "
            f"val_mean={val_means[c]:.6f}, "
            f"abs_diff={mean_diff[c]:.6f}"
        )


def main():
    normal_session_dfs, normal_meta = load_sessions_for_activity(
        NORMAL_DIR,
        TARGET_ACTIVITY_CODE,
        dataset_label="normal",
    )

    if len(normal_session_dfs) == 0:
        raise RuntimeError(
            f"No normal sessions loaded for {TARGET_ACTIVITY_CODE} from {NORMAL_DIR}"
        )

    normal_participants = sorted({m["participant_id"] for m in normal_meta})
    splitter_name = "StratifiedGroupKFold" if TARGET_ACTIVITY_CODE == "t-all" else "GroupKFold"
    if len(normal_participants) < N_SPLITS:
        raise RuntimeError(
            f"Need at least {N_SPLITS} unique normal participants for {splitter_name}, "
            f"but found {len(normal_participants)}"
        )

    print(f"Target activity: {TARGET_ACTIVITY_CODE}")
    print(f"Normal sessions: {len(normal_session_dfs)}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Threshold percentile: {THRESHOLD_PERCENTILE}")

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

    all_y_true = []
    all_y_pred = []
    all_predictions_dfs = []
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        split_iter, start=1
    ):
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


        train_normal_dfs = [normal_session_dfs[i] for i in train_idx]
        train_normal_meta = [normal_meta[i] for i in train_idx]

        val_normal_dfs = [normal_session_dfs[i] for i in val_idx]
        val_normal_meta = [normal_meta[i] for i in val_idx]

        cols = feature_columns(target_activity=TARGET_ACTIVITY_CODE, feature_config=FEATURE_CONFIG)

        if TARGET_ACTIVITY_CODE == "t-all":
            reference_stats = build_reference_stats_by_activity(
                train_normal_dfs,
                train_normal_meta,
            )

        else:
            reference_stats = build_reference_stats(train_normal_dfs)


        train_df = build_feature_matrix_train_loo_reference(
            train_normal_dfs,
            train_normal_meta,
            cols=cols,
        )

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(train_df)

        val_normal_df = build_feature_matrix(
            val_normal_dfs,
            val_normal_meta,
            cols=cols,
            reference_stats=reference_stats,
        )
        #print(val_normal_df)

        #Scale each activity differently if using t-all, otherwise just use raw features (since each fold is a single activity)
        if TARGET_ACTIVITY_CODE == "t-all":
            scalers = fit_activity_scalers(
                feature_df=train_df,
                scale_cols=[c for c in cols if not c.startswith("activity_")],
                activity_col="activity_code",
            )

            X_train = apply_activity_scalers(
                feature_df=train_df,
                cols=cols,
                scalers=scalers,
                activity_col="activity_code",
            )

            X_val = apply_activity_scalers(
                feature_df=val_normal_df,
                cols=cols,
                scalers=scalers,
                activity_col="activity_code",
            )

            #for act in sorted(train_df["activity_code"].unique()):
            #    idx = train_df["activity_code"] == act
            #    print(act, X_train.loc[idx, "EPISODE_DURATION"].mean(), X_train.loc[idx, "EPISODE_DURATION"].std())
            
            #print(X_train[[c for c in X_train.columns if c.startswith("activity_")]].drop_duplicates())

        else:
            X_train = train_df[cols].copy().astype(float)
            X_val = val_normal_df[cols].copy().astype(float)


        print(X_train.describe())

        if X_train.empty:
            raise RuntimeError(f"Fold {fold_idx}: training feature matrix is empty.")
        if X_train.isna().sum().sum() > 0:
            raise RuntimeError(f"Fold {fold_idx}: training feature matrix contains NaN values.")
        if np.isinf(X_train.to_numpy(dtype=float)).sum() > 0:
            raise RuntimeError(f"Fold {fold_idx}: training feature matrix contains infinite values.")

        model = make_iforest(iforest_config)
        model.fit(X_train)

        train_scores = model.decision_function(X_train).ravel()

        if TARGET_ACTIVITY_CODE == "t-all":
            if not THRESHOLD_PERCENTILES_BY_ACTIVITY:
                raise ValueError("Missing threshold_percentiles_by_activity for t-all")

            thresholds_by_activity = get_thresholds_by_activity(
                scores=train_scores,
                activity_codes=train_df["activity_code"],
                percentiles_by_activity=THRESHOLD_PERCENTILES_BY_ACTIVITY,
            )
            threshold = None
            prediction_mode = "percentile_threshold_by_activity"
        else:
            thresholds_by_activity = None
            threshold = get_threshold(train_scores, THRESHOLD_PERCENTILE)
            prediction_mode = "native_iforest" if threshold is None else "percentile_threshold"

        
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
        print(X_val.describe())

        if X_val.isna().sum().sum() > 0:
            raise RuntimeError(f"Fold {fold_idx}: validation feature matrix contains NaN values.")
        if np.isinf(X_val.to_numpy(dtype=float)).sum() > 0:
            raise RuntimeError(f"Fold {fold_idx}: validation feature matrix contains infinite values.")

        debug_feature_shift(
            X_train=X_train,
            X_val=X_val,
            fold_idx=fold_idx,
            top_n=15,
        )



        y_true_normal = np.zeros(len(val_normal_df), dtype=int)

        if TARGET_ACTIVITY_CODE == "t-all":
            y_pred_normal, decision_scores, native_pred, _ = predict_iforest_by_activity(
                model=model,
                X=X_val,
                activity_codes=val_normal_df["activity_code"],
                thresholds_by_activity=thresholds_by_activity,
            )
        else: 
            y_pred_normal, decision_scores, native_pred, _ = predict_iforest(model, X_val, threshold=threshold)


        print_confusion_matrix(
            y_true_normal,
            y_pred_normal,
            title=f"Fold {fold_idx} Confusion Matrix",
        )

        fold_report = classification_report(
            y_true_normal,
            y_pred_normal,
            labels=[0, 1],
            target_names=["normal", "anomaly"],
            zero_division=0,
        )
        print(f"\nFold {fold_idx} Classification Report")
        print(fold_report)

        fold_train_features_fp = OUTPUT_DIR / f"fold_{fold_idx}_train_features_{TARGET_ACTIVITY_CODE}.csv"
        fold_val_features_fp = OUTPUT_DIR / f"fold_{fold_idx}_val_features_{TARGET_ACTIVITY_CODE}.csv"

        train_df.to_csv(fold_train_features_fp, index=False)
        val_normal_df.to_csv(fold_val_features_fp, index=False)

        fold_predictions_df = val_normal_df[
            ["filename", "participant_id", "activity_code", "activity_index", "dataset"]
        ].copy()
        fold_predictions_df["fold"] = fold_idx
        fold_predictions_df["y_true"] = y_true_normal
        fold_predictions_df["y_pred"] = y_pred_normal
        fold_predictions_df["true_label"] = "normal"
        fold_predictions_df["pred_label"] = np.where(y_pred_normal == 1, "anomaly", "normal")
        fold_predictions_df["decision_score"] = decision_scores
        if TARGET_ACTIVITY_CODE == "t-all":
            fold_predictions_df["threshold"] = fold_predictions_df["activity_code"].map(thresholds_by_activity)
        else:
            fold_predictions_df["threshold"] = threshold
        fold_predictions_df["prediction_mode"] = prediction_mode
        fold_predictions_df["native_pred"] = native_pred

        fold_preds_fp = OUTPUT_DIR / f"fold_{fold_idx}_predictions_{TARGET_ACTIVITY_CODE}.csv"
        fold_predictions_df.to_csv(fold_preds_fp, index=False)
        pca_fold_df = val_normal_df.merge(
            fold_predictions_df[
                [
                    "filename",
                    "participant_id",
                    "activity_code",
                    "activity_index",
                    "dataset",
                    "fold",
                    "y_true",
                    "y_pred",
                    "true_label",
                    "pred_label",
                    "decision_score",
                    "threshold",
                    "prediction_mode",
                    "native_pred",
                ]
            ],
            on=["filename", "participant_id", "activity_code", "activity_index", "dataset"],
            how="left",
        )

        pca_fold_fp = OUTPUT_DIR / f"fold_{fold_idx}_pca_input_{TARGET_ACTIVITY_CODE}.csv"
        pca_fold_df.to_csv(pca_fold_fp, index=False)

        _, fold_cm_df = build_confusion_matrix_df(y_true_normal, y_pred_normal)
        fold_cm_csv_fp = OUTPUT_DIR / f"fold_{fold_idx}_confusion_matrix_{TARGET_ACTIVITY_CODE}.csv"
        fold_cm_txt_fp = OUTPUT_DIR / f"fold_{fold_idx}_confusion_matrix_{TARGET_ACTIVITY_CODE}.txt"
        save_confusion_matrix_files(
            fold_cm_df,
            fold_cm_csv_fp,
            fold_cm_txt_fp,
            title=f"Fold {fold_idx} Confusion Matrix",
        )


        fold_report_fp = OUTPUT_DIR / f"fold_{fold_idx}_classification_report_{TARGET_ACTIVITY_CODE}.txt"
        with open(fold_report_fp, "w") as f:
            f.write(fold_report)

        fold_metrics.append(
            {
                "fold": fold_idx,
                "prediction_mode": prediction_mode,
                "n_train_normal": int(len(train_df)),
                "n_val_normal": int(len(val_normal_df)),
                "n_features": int(len(cols)),
                "threshold_percentile": THRESHOLD_PERCENTILE,
                "threshold": threshold,
                "n_predicted_normal": int((y_pred_normal == 0).sum()),
                "n_predicted_anomaly": int((y_pred_normal == 1).sum()),
                "n_native_predicted_normal": int((native_pred == 0).sum()),
                "n_native_predicted_anomaly": int((native_pred == 1).sum()),
            }
        )

        all_y_true.extend(y_true_normal.tolist())
        all_y_pred.extend(y_pred_normal.tolist())
        all_predictions_dfs.append(fold_predictions_df)

    y_true_all = np.array(all_y_true, dtype=int)
    y_pred_all = np.array(all_y_pred, dtype=int)

    print_confusion_matrix(
        y_true_all,
        y_pred_all,
        title="Overall Confusion Matrix",
    )

    overall_report = classification_report(
        y_true_all,
        y_pred_all,
        labels=[0, 1],
        target_names=["normal", "anomaly"],
        zero_division=0,
    )
    print("\nOverall Classification Report")
    print(overall_report)

    all_predictions_df = pd.concat(all_predictions_dfs, ignore_index=True)
    all_preds_fp = OUTPUT_DIR / f"all_fold_predictions_{TARGET_ACTIVITY_CODE}.csv"
    all_predictions_df.to_csv(all_preds_fp, index=False)

    _, overall_cm_df = build_confusion_matrix_df(y_true_all, y_pred_all)
    overall_cm_csv_fp = OUTPUT_DIR / f"overall_confusion_matrix_{TARGET_ACTIVITY_CODE}.csv"
    overall_cm_txt_fp = OUTPUT_DIR / f"overall_confusion_matrix_{TARGET_ACTIVITY_CODE}.txt"
    save_confusion_matrix_files(
        overall_cm_df,
        overall_cm_csv_fp,
        overall_cm_txt_fp,
        title="Overall Confusion Matrix",
    )

    if TARGET_ACTIVITY_CODE == "t-all":
        save_confusion_matrices_by_activity_one_file(
            preds_df=all_predictions_df,
            csv_fp=OUTPUT_DIR / "overall_confusion_matrices_by_activity.csv",
            txt_fp=OUTPUT_DIR / "overall_confusion_matrices_by_activity.txt",
            title_prefix="Overall CV",
        )

    overall_report_fp = OUTPUT_DIR / f"overall_classification_report_{TARGET_ACTIVITY_CODE}.txt"
    with open(overall_report_fp, "w") as f:
        f.write(overall_report)

    metrics_payload = {
        "target_activity_code": TARGET_ACTIVITY_CODE,
        "model_type": "IsolationForest",
        "n_estimators": N_ESTIMATORS,
        "max_samples": MAX_SAMPLES,
        "contamination": CONTAMINATION,
        "max_features": MAX_FEATURES,
        "bootstrap": BOOTSTRAP,
        "random_state": RANDOM_STATE,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "prediction_mode": "native_iforest" if THRESHOLD_PERCENTILE is None else "percentile_threshold",
        "n_splits": N_SPLITS,
        "n_total_normal_sessions": int(len(normal_session_dfs)),
        "n_total_predicted_normal": int((y_pred_all == 0).sum()),
        "n_total_predicted_anomaly": int((y_pred_all == 1).sum()),
        "fold_metrics": fold_metrics,
    }
    metrics_fp = OUTPUT_DIR / f"cv_metrics_{TARGET_ACTIVITY_CODE}.json"
    with open(metrics_fp, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\nSaved outputs")
    print(f"  All predictions       -> {all_preds_fp}")
    print(f"  Overall CM CSV        -> {overall_cm_csv_fp}")
    print(f"  Overall CM TXT        -> {overall_cm_txt_fp}")
    print(f"  Overall report        -> {overall_report_fp}")
    print(f"  CV metrics            -> {metrics_fp}")


if __name__ == "__main__":
    main()
