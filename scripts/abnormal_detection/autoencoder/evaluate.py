#!/usr/bin/env python3
"""
Standard Autoencoder PER-ACTIVITY with 5-Fold Cross-Validation on NORMAL data only.
"""

import json
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from abnormal_activity_detection.anomaly.configs.autoencoder import (
    TARGET_ACTIVITY_CODE,
    N_SPLITS,
    AUTOENCODER_CONFIG,
    FEATURE_CONFIG,
    USE_ACTIVITY_AWARE_LOSS,
)
from abnormal_activity_detection.data import load_sessions_for_activity
from abnormal_activity_detection.features.anomaly import (
    build_feature_matrix,
    build_reference_stats,
    build_reference_stats_by_activity,
    feature_columns,
    write_fold_trigram_debug_file,
    build_feature_matrix_train_loo_reference,
)
from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir
from abnormal_activity_detection.evaluation import (
    apply_activity_scalers,
    fit_activity_scalers,
    get_thresholds_by_activity,
    print_confusion_matrix,
    save_confusion_matrices_by_activity_one_file,
    build_confusion_matrix_df,
    save_confusion_matrix_files
)
from abnormal_activity_detection.anomaly.autoencoder import (
    predict_autoencoder_by_activity,
    reconstruction_errors,
    set_seed,
    build_autoencoder,
    get_activity_col_map,
    reconstruction_errors_by_activity,
    predict_autoencoder,
)


ae_config = AUTOENCODER_CONFIG[TARGET_ACTIVITY_CODE]
USE_PER_ACTIVITY_LOSS = (
    TARGET_ACTIVITY_CODE == "t-all" and USE_ACTIVITY_AWARE_LOSS
)

LOSS_NAME = ae_config.get("loss_name")
HUBER_DELTA = ae_config.get("huber_delta")

LOSS_NAME_BY_ACTIVITY = ae_config.get("loss_name_by_activity")
HUBER_DELTA_BY_ACTIVITY = ae_config.get("huber_delta_by_activity")


THRESHOLD_PERCENTILE = ae_config.get("threshold_percentile")
THRESHOLD_PERCENTILES_BY_ACTIVITY = ae_config.get("threshold_percentiles_by_activity")

NORMAL_DIR, _ = get_dataset_dirs()
OUTPUT_DIR = get_results_dir("abnormal_detection", "autoencoder", TARGET_ACTIVITY_CODE, "autoencoder_normal_cv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    set_seed(ae_config["random_state"])

    if USE_PER_ACTIVITY_LOSS and not LOSS_NAME_BY_ACTIVITY:
        raise RuntimeError(
            "USE_ACTIVITY_AWARE_LOSS=True but loss_name_by_activity is missing"
        )
    
    if not USE_PER_ACTIVITY_LOSS and "loss_name" not in ae_config:
        raise RuntimeError(
            "USE_ACTIVITY_AWARE_LOSS=False but loss_name is missing from ae_config"
        )

    if not USE_PER_ACTIVITY_LOSS and LOSS_NAME is None:
        raise RuntimeError(
            "USE_ACTIVITY_AWARE_LOSS=False but loss_name is missing"
        )

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

    print(f"USE_ACTIVITY_AWARE_LOSS: {USE_ACTIVITY_AWARE_LOSS}")

    if USE_PER_ACTIVITY_LOSS:
        print(f"Loss by activity: {LOSS_NAME_BY_ACTIVITY}")
        print(f"Huber delta by activity: {HUBER_DELTA_BY_ACTIVITY}")
    else:
        print(f"Loss name: {LOSS_NAME}")
        print(f"Huber delta: {HUBER_DELTA}")

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

        train_normal_dfs = [normal_session_dfs[i] for i in train_idx]
        train_normal_meta = [normal_meta[i] for i in train_idx]

        val_normal_dfs = [normal_session_dfs[i] for i in val_idx]
        val_normal_meta = [normal_meta[i] for i in val_idx]

        cols = feature_columns(target_activity=TARGET_ACTIVITY_CODE, feature_config=FEATURE_CONFIG)
        activity_col_map = get_activity_col_map(cols) if TARGET_ACTIVITY_CODE == "t-all" else None

        if TARGET_ACTIVITY_CODE == "t-all":
            reference_stats = build_reference_stats_by_activity(
                train_normal_dfs,
                train_normal_meta,
            )
        else:
            reference_stats = build_reference_stats(train_normal_dfs)

            fold_trigram_debug_fp = OUTPUT_DIR / f"fold_{fold_idx}_trigram_debug_{TARGET_ACTIVITY_CODE}.txt"
            write_fold_trigram_debug_file(
                output_fp=fold_trigram_debug_fp,
                fold_idx=fold_idx,
                train_reference_stats=reference_stats,
                val_session_dfs=val_normal_dfs,
                val_meta=val_normal_meta,
            )

        train_df = build_feature_matrix_train_loo_reference(
            train_normal_dfs,
            train_normal_meta,
            cols=cols,
        )

        val_normal_df = build_feature_matrix(
            val_normal_dfs,
            val_normal_meta,
            cols=cols,
            reference_stats=reference_stats,
        )

        X_train = train_df[cols].copy().astype(float)
        X_val = val_normal_df[cols].copy().astype(float)

        if X_train.empty:
            raise RuntimeError(f"Fold {fold_idx}: training feature matrix is empty.")
        if X_train.isna().sum().sum() > 0:
            raise RuntimeError(f"Fold {fold_idx}: training feature matrix contains NaN values.")
        if np.isinf(X_train.to_numpy(dtype=float)).sum() > 0:
            raise RuntimeError(f"Fold {fold_idx}: training feature matrix contains infinite values.")

        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

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
                feature_df=val_normal_df,
                cols=cols,
                scalers=scalers,
                activity_col="activity_code",
            )
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

        model = build_autoencoder(
            input_dim=X_train_scaled.shape[1],
            config=ae_config,
            activity_col_map=activity_col_map,
        )
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=ae_config["patience"],
                restore_best_weights=True,
                verbose=0,
            )
        ]
        model.fit(
            X_train_scaled,
            X_train_scaled,
            epochs=ae_config["epochs"],
            batch_size=ae_config["batch_size"],
            validation_split=ae_config["validation_split"],
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
        )

        if USE_PER_ACTIVITY_LOSS:
            train_errors = reconstruction_errors_by_activity(
                model=model,
                X=X_train_scaled,
                activity_codes=train_df["activity_code"],
                loss_name_by_activity=LOSS_NAME_BY_ACTIVITY,
                huber_delta_by_activity=HUBER_DELTA_BY_ACTIVITY,
            )
        else:
            train_errors = reconstruction_errors(model, X_train_scaled)

        thresholds_by_activity = None

        if TARGET_ACTIVITY_CODE == "t-all":
            if THRESHOLD_PERCENTILES_BY_ACTIVITY is None:
                thresholds_by_activity = None
                threshold = None
                prediction_mode = "native_autoencoder"
            else:
                thresholds_by_activity = get_thresholds_by_activity(
                    scores=train_errors,
                    activity_codes=train_df["activity_code"],
                    percentiles_by_activity=THRESHOLD_PERCENTILES_BY_ACTIVITY,
                )
                threshold = None
                prediction_mode = "percentile_threshold_by_activity"
        else:
            if THRESHOLD_PERCENTILE is None:
                threshold = None
                prediction_mode = "native_autoencoder"
            else:
                threshold = float(np.percentile(train_errors, float(THRESHOLD_PERCENTILE)))
                prediction_mode = "percentile_threshold"

        y_true_normal = np.zeros(len(val_normal_df), dtype=int)

        if TARGET_ACTIVITY_CODE == "t-all":
            if thresholds_by_activity is None:
                y_pred_normal, reconstruction_error, native_pred, _ = predict_autoencoder(
                    model,
                    X_val_scaled,
                    threshold=None,
                )
            else:
                if USE_PER_ACTIVITY_LOSS:
                    y_pred_normal, reconstruction_error, native_pred, row_thresholds = predict_autoencoder_by_activity(
                        model=model,
                        X=X_val_scaled,
                        activity_codes=val_normal_df["activity_code"],
                        thresholds_by_activity=thresholds_by_activity,
                        loss_name_by_activity=LOSS_NAME_BY_ACTIVITY,
                        huber_delta_by_activity=HUBER_DELTA_BY_ACTIVITY,
                    )
                else:
                    reconstruction_error = reconstruction_errors(model, X_val_scaled)
                    row_thresholds = val_normal_df["activity_code"].map(thresholds_by_activity).to_numpy(dtype=float)
                    y_pred_normal = (reconstruction_error > row_thresholds).astype(int)
                    native_pred = y_pred_normal.copy()

                debug_df = pd.DataFrame({
                    "activity": val_normal_df["activity_code"].values,
                    "score": reconstruction_error,
                    "threshold": row_thresholds,
                    "native_pred": native_pred,
                    "final_pred": y_pred_normal,
                })
                print(debug_df.head(20))
        else:
            y_pred_normal, reconstruction_error, native_pred, _ = predict_autoencoder(
                model,
                X_val_scaled,
                threshold=threshold,
            )


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
        fold_predictions_df["reconstruction_error"] = reconstruction_error
        if TARGET_ACTIVITY_CODE == "t-all":
            if thresholds_by_activity is None:
                fold_predictions_df["threshold"] = np.nan
            else:
                fold_predictions_df["threshold"] = (
                    fold_predictions_df["activity_code"].map(thresholds_by_activity)
                )
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
                    "reconstruction_error",
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
                "use_activity_aware_loss": USE_ACTIVITY_AWARE_LOSS,
                "loss_name": LOSS_NAME if not USE_PER_ACTIVITY_LOSS else None,
                "huber_delta": HUBER_DELTA if not USE_PER_ACTIVITY_LOSS else None,
                "loss_name_by_activity": LOSS_NAME_BY_ACTIVITY if USE_PER_ACTIVITY_LOSS else None,
                "huber_delta_by_activity": HUBER_DELTA_BY_ACTIVITY if USE_PER_ACTIVITY_LOSS else None,
                "threshold_percentile": THRESHOLD_PERCENTILE,
                "threshold_percentiles_by_activity": THRESHOLD_PERCENTILES_BY_ACTIVITY if TARGET_ACTIVITY_CODE == "t-all" else None,
                "threshold": threshold,
                "thresholds_by_activity": thresholds_by_activity if TARGET_ACTIVITY_CODE == "t-all" else None,
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
        "model_type": "Autoencoder",
        "use_activity_aware_loss": USE_ACTIVITY_AWARE_LOSS,
        "loss_name": LOSS_NAME if not USE_PER_ACTIVITY_LOSS else None,
        "huber_delta": HUBER_DELTA if not USE_PER_ACTIVITY_LOSS else None,
        "loss_name_by_activity": LOSS_NAME_BY_ACTIVITY if USE_PER_ACTIVITY_LOSS else None,
        "huber_delta_by_activity": HUBER_DELTA_BY_ACTIVITY if USE_PER_ACTIVITY_LOSS else None,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold_percentiles_by_activity": THRESHOLD_PERCENTILES_BY_ACTIVITY if TARGET_ACTIVITY_CODE == "t-all" else None,
        "prediction_mode": prediction_mode,
        "n_splits": N_SPLITS,
        "n_total_normal_sessions": int(len(normal_session_dfs)),
        "n_total_predicted_normal": int((y_pred_all == 0).sum()),
        "n_total_predicted_anomaly": int((y_pred_all == 1).sum()),
        "fold_metrics": fold_metrics,
    }
    metrics_fp = OUTPUT_DIR / f"overall_metrics_{TARGET_ACTIVITY_CODE}.json"
    with open(metrics_fp, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    metrics_fp = OUTPUT_DIR / f"cv_metrics_{TARGET_ACTIVITY_CODE}.json"
    with open(metrics_fp, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_fp = OUTPUT_DIR / f"fold_metrics_{TARGET_ACTIVITY_CODE}.csv"
    fold_metrics_df.to_csv(fold_metrics_fp, index=False)


if __name__ == "__main__":
    main()