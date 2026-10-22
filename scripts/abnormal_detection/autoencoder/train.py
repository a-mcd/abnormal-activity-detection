#!/usr/bin/env python3
import json
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

from abnormal_activity_detection.anomaly.configs.autoencoder import (
    AUTOENCODER_CONFIG,
    TARGET_ACTIVITY_CODE,
    FEATURE_CONFIG,
    USE_ACTIVITY_AWARE_LOSS,
)
from abnormal_activity_detection.data import load_sessions_for_activity
from abnormal_activity_detection.features.anomaly import (
    build_feature_matrix,
    build_reference_stats,
    build_reference_stats_by_activity,
    feature_columns,
)
from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir
from abnormal_activity_detection.evaluation import (
    apply_activity_scalers,
    fit_activity_scalers,
    get_thresholds_by_activity,
)
from abnormal_activity_detection.reference_stats import serialize_reference_stats
from abnormal_activity_detection.anomaly.autoencoder import (
    predict_autoencoder_by_activity,
    reconstruction_errors,
    set_seed,
    build_autoencoder,
    get_activity_col_map,
    reconstruction_errors_by_activity,
    predict_autoencoder,
)

META_COLS = [
    "filename",
    "participant_id",
    "activity_code",
    "activity_index",
    "dataset",
]


def main():
    activity = TARGET_ACTIVITY_CODE
    config = AUTOENCODER_CONFIG[activity]

    USE_PER_ACTIVITY_LOSS = (
        activity == "t-all" and USE_ACTIVITY_AWARE_LOSS
    )

    LOSS_NAME = config.get("loss_name")
    HUBER_DELTA = config.get("huber_delta")

    LOSS_NAME_BY_ACTIVITY = config.get("loss_name_by_activity")
    HUBER_DELTA_BY_ACTIVITY = config.get("huber_delta_by_activity")

    set_seed(config["random_state"])

    if USE_PER_ACTIVITY_LOSS and not LOSS_NAME_BY_ACTIVITY:
        raise RuntimeError(
            "USE_ACTIVITY_AWARE_LOSS=True but loss_name_by_activity is missing"
        )

    if not USE_PER_ACTIVITY_LOSS and "loss_name" not in config:
        raise RuntimeError(
            "USE_ACTIVITY_AWARE_LOSS=False but loss_name is missing from config"
        )

    normal_dir, _ = get_dataset_dirs()

    normal_sessions, normal_meta = load_sessions_for_activity(
        directory=normal_dir,
        target_activity=activity,
        dataset_label="noerror",
    )

    if not normal_sessions:
        raise ValueError(f"No normal sessions found for activity={activity} in {normal_dir}")

    cols = feature_columns(target_activity=activity, feature_config=FEATURE_CONFIG)
    activity_col_map = get_activity_col_map(cols) if activity == "t-all" else None

    if activity == "t-all":
        reference_stats = build_reference_stats_by_activity(
            normal_sessions,
            normal_meta,
        )
    else:
        reference_stats = build_reference_stats(normal_sessions)

    feature_df = build_feature_matrix(
        session_dfs=normal_sessions,
        meta=normal_meta,
        cols=cols,
        reference_stats=reference_stats,
    )
    print(feature_df)

    out_dir = get_results_dir("abnormal_detection", "autoencoder", activity, "autoencoder_train")
    out_dir.mkdir(parents=True, exist_ok=True)

    X = feature_df[cols].copy().astype(float)

    if X.empty:
        raise RuntimeError("Training feature matrix is empty.")
    if X.isna().sum().sum() > 0:
        raise RuntimeError("Training feature matrix contains NaN values.")
    if np.isinf(X.to_numpy(dtype=float)).sum() > 0:
        raise RuntimeError("Training feature matrix contains infinite values.")

    
    if activity == "t-all":
        scalers = fit_activity_scalers(
            feature_df=feature_df,
            scale_cols=[c for c in cols if not c.startswith("activity_")],
            activity_col="activity_code",
        )
        X_scaled = apply_activity_scalers(
            feature_df=feature_df,
            cols=cols,
            scalers=scalers,
            activity_col="activity_code",
        )
        scaler_to_save = scalers
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scaler_to_save = scaler


    model = build_autoencoder(
        input_dim=X_scaled.shape[1],
        config=config,
        activity_col_map=activity_col_map,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = model.fit(
        X_scaled,
        X_scaled,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_split=config["validation_split"],
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    if USE_PER_ACTIVITY_LOSS:
        reconstruction_error = reconstruction_errors_by_activity(
            model=model,
            X=X_scaled,
            activity_codes=feature_df["activity_code"],
            loss_name_by_activity=LOSS_NAME_BY_ACTIVITY,
            huber_delta_by_activity=HUBER_DELTA_BY_ACTIVITY,
        )
    else:
        reconstruction_error = reconstruction_errors(model, X_scaled)

    threshold_percentile = config.get("threshold_percentile")
    threshold_percentiles_by_activity = config.get("threshold_percentiles_by_activity")

    threshold = None
    thresholds_by_activity = None
    prediction_mode = "native_autoencoder"

    if activity == "t-all":
        if threshold_percentiles_by_activity is not None:
            thresholds_by_activity = get_thresholds_by_activity(
                scores=reconstruction_error,
                activity_codes=feature_df["activity_code"],
                percentiles_by_activity=threshold_percentiles_by_activity,
            )
            prediction_mode = "percentile_threshold_by_activity"
    else:
        if threshold_percentile is not None:
            threshold = float(np.percentile(reconstruction_error, float(threshold_percentile)))
            prediction_mode = "percentile_threshold"

    
    if activity == "t-all":
        if thresholds_by_activity is None:
            y_pred, reconstruction_error, native_pred, prediction_mode = predict_autoencoder(
                model=model,
                X_scaled=X_scaled,
                threshold=None,
            )
        else:
            if USE_PER_ACTIVITY_LOSS:
                y_pred, reconstruction_error, native_pred, row_thresholds = predict_autoencoder_by_activity(
                    model=model,
                    X=X_scaled,
                    activity_codes=feature_df["activity_code"],
                    thresholds_by_activity=thresholds_by_activity,
                    loss_name_by_activity=LOSS_NAME_BY_ACTIVITY,
                    huber_delta_by_activity=HUBER_DELTA_BY_ACTIVITY,
                )
            else:
                reconstruction_error = reconstruction_errors(model, X_scaled)
                row_thresholds = feature_df["activity_code"].map(thresholds_by_activity).to_numpy(dtype=float)
                y_pred = (reconstruction_error > row_thresholds).astype(int)
                native_pred = y_pred.copy()
                prediction_mode = "percentile_threshold_by_activity"
    else:
        y_pred, reconstruction_error, native_pred, prediction_mode = predict_autoencoder(
            model=model,
            X_scaled=X_scaled,
            threshold=threshold,
        )

    feature_df.to_csv(out_dir / f"{activity}_train_features.csv", index=False)

    diag_df = feature_df[META_COLS].copy()
    diag_df["reconstruction_error"] = reconstruction_error
    diag_df["y_pred"] = y_pred
    diag_df["pred_label"] = np.where(y_pred == 1, "anomaly", "normal")
    diag_df["prediction_mode"] = prediction_mode


    if activity == "t-all":
        if USE_PER_ACTIVITY_LOSS:
            diag_df["loss_name"] = diag_df["activity_code"].map(LOSS_NAME_BY_ACTIVITY)
            diag_df["huber_delta"] = diag_df["activity_code"].map(HUBER_DELTA_BY_ACTIVITY or {})
        else:
            diag_df["loss_name"] = LOSS_NAME
            diag_df["huber_delta"] = HUBER_DELTA

        if thresholds_by_activity is None:
            diag_df["threshold"] = np.nan
        else:
            diag_df["threshold"] = diag_df["activity_code"].map(thresholds_by_activity)
    else:
        diag_df["loss_name"] = LOSS_NAME
        diag_df["huber_delta"] = HUBER_DELTA
        diag_df["threshold"] = threshold

    diag_df.to_csv(out_dir / f"{activity}_train_predictions.csv", index=False)

    pca_train_df = feature_df.merge(
        diag_df,
        on=["filename", "participant_id", "activity_code", "activity_index", "dataset"],
        how="left",
    )
    pca_train_df["source_split"] = "noerror"
    pca_train_df.to_csv(out_dir / f"{activity}_pca_input_noerror.csv", index=False)

    model_to_save = model.base_model if hasattr(model, "base_model") else model
    model_to_save.save(out_dir / f"{activity}_autoencoder.keras")
    joblib.dump(scaler_to_save, out_dir / f"{activity}_scaler.joblib")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(out_dir / f"{activity}_training_history.csv", index=False)

    assets = {
        "activity": activity,
        "feature_columns": cols,
        "threshold_percentile": threshold_percentile,
        "threshold_percentiles_by_activity": threshold_percentiles_by_activity if activity == "t-all" else None,
        "threshold": threshold,
        "thresholds_by_activity": thresholds_by_activity if activity == "t-all" else None,
        "prediction_mode": prediction_mode,
        "reference_stats": serialize_reference_stats(reference_stats),
        "autoencoder_config": config,
        "use_activity_aware_loss": USE_ACTIVITY_AWARE_LOSS,
        "loss_name": LOSS_NAME if not USE_PER_ACTIVITY_LOSS else None,
        "huber_delta": HUBER_DELTA if not USE_PER_ACTIVITY_LOSS else None,
        "loss_name_by_activity": LOSS_NAME_BY_ACTIVITY if USE_PER_ACTIVITY_LOSS else None,
        "huber_delta_by_activity": HUBER_DELTA_BY_ACTIVITY if USE_PER_ACTIVITY_LOSS else None,
    }

    with open(out_dir / f"{activity}_training_assets.json", "w") as f:
        json.dump(assets, f, indent=2)

    print(f"Saved model bundle to: {out_dir}")
    print(f"Training rows: {len(feature_df)}")
    print(f"Training feature columns: {len(cols)}")
    print(f"Prediction mode: {prediction_mode}")
    print(f"USE_ACTIVITY_AWARE_LOSS: {USE_ACTIVITY_AWARE_LOSS}")

    if USE_PER_ACTIVITY_LOSS:
        print(f"Loss by activity: {LOSS_NAME_BY_ACTIVITY}")
        print(f"Huber delta by activity: {HUBER_DELTA_BY_ACTIVITY}")
    else:
        print(f"Loss name: {LOSS_NAME}")
        print(f"Huber delta: {HUBER_DELTA}")


if __name__ == "__main__":
    main()