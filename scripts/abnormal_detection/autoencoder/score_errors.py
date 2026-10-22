#!/usr/bin/env python3
"""
Run inference on ERROR sessions using a previously trained final Autoencoder model.
"""

import json
import joblib
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report

from abnormal_activity_detection.anomaly.configs.autoencoder import TARGET_ACTIVITY_CODE
from abnormal_activity_detection.data import load_sessions_for_activity
from abnormal_activity_detection.features.anomaly import build_feature_matrix
from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir
from abnormal_activity_detection.reference_stats import deserialize_reference_stats

from abnormal_activity_detection.anomaly.autoencoder import (
    reconstruction_errors,
    predict_autoencoder_by_activity,
    predict_autoencoder,
)
from abnormal_activity_detection.evaluation import (
    apply_activity_scalers,
    save_confusion_matrices_by_activity_one_file,
    print_confusion_matrix,
    build_confusion_matrix_df,
    save_confusion_matrix_files

)


_, ERROR_DIR = get_dataset_dirs()
MODEL_DIR = get_results_dir("abnormal_detection", "autoencoder", TARGET_ACTIVITY_CODE, "autoencoder_train")
OUTPUT_DIR = get_results_dir("abnormal_detection", "autoencoder", TARGET_ACTIVITY_CODE, "autoencoder_error_inference")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



def main():
    model_fp = MODEL_DIR / f"{TARGET_ACTIVITY_CODE}_autoencoder.keras"
    scaler_fp = MODEL_DIR / f"{TARGET_ACTIVITY_CODE}_scaler.joblib"
    assets_fp = MODEL_DIR / f"{TARGET_ACTIVITY_CODE}_training_assets.json"

    if not model_fp.exists():
        raise FileNotFoundError(f"Model file not found: {model_fp}")
    if not scaler_fp.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_fp}")
    if not assets_fp.exists():
        raise FileNotFoundError(f"Assets file not found: {assets_fp}")

    print(f"Loading model:  {model_fp}")
    print(f"Loading scaler: {scaler_fp}")
    print(f"Loading assets: {assets_fp}")


    model = tf.keras.models.load_model(
        model_fp,
        compile=False,
    )
    scaler = joblib.load(scaler_fp)

    with open(assets_fp, "r") as f:
        assets = json.load(f)

    use_activity_aware_loss = assets.get("use_activity_aware_loss", False)
    use_per_activity_loss = (
        TARGET_ACTIVITY_CODE == "t-all" and use_activity_aware_loss
    )

    cols = list(assets["feature_columns"])
    threshold = assets.get("threshold", None)
    thresholds_by_activity = assets.get("thresholds_by_activity", None)
    reference_stats = deserialize_reference_stats(assets.get("reference_stats", {}))
    loss_name = assets.get("loss_name", assets.get("autoencoder_config", {}).get("loss_name", "mse"))
    huber_delta = assets.get("huber_delta", assets.get("autoencoder_config", {}).get("huber_delta"))
    loss_name_by_activity = assets.get(
        "loss_name_by_activity",
        assets.get("autoencoder_config", {}).get("loss_name_by_activity"),
    )
    huber_delta_by_activity = assets.get(
        "huber_delta_by_activity",
        assets.get("autoencoder_config", {}).get("huber_delta_by_activity"),
    )

    print(f"Target activity: {TARGET_ACTIVITY_CODE}")
    print(f"Feature count:   {len(cols)}")
    print(f"USE_ACTIVITY_AWARE_LOSS: {use_activity_aware_loss}")

    if use_per_activity_loss:
        print(f"Loss by activity: {loss_name_by_activity}")
        print(f"Huber delta by activity: {huber_delta_by_activity}")
    elif TARGET_ACTIVITY_CODE == "t-all":
        print(f"Loss name:       {loss_name}")
        print(f"Huber delta:     {huber_delta}")
        print("Saved thresholds by activity:", thresholds_by_activity)
    else:
        print(f"Loss name:       {loss_name}")
        print(f"Huber delta:     {huber_delta}")
        print(f"Saved threshold: {threshold}")

    error_session_dfs, error_meta = load_sessions_for_activity(
        ERROR_DIR,
        TARGET_ACTIVITY_CODE,
        dataset_label="error",
    )

    if len(error_session_dfs) == 0:
        raise RuntimeError(
            f"No error sessions loaded for {TARGET_ACTIVITY_CODE} from {ERROR_DIR}"
        )

    print(f"Error sessions:  {len(error_session_dfs)}")
    print(f"Output dir:      {OUTPUT_DIR}")

    error_df = build_feature_matrix(
        error_session_dfs,
        error_meta,
        cols=cols,
        reference_stats=reference_stats,
    )
    print(error_df)

    features_fp = OUTPUT_DIR / f"error_session_features_{TARGET_ACTIVITY_CODE}.csv"
    error_df.to_csv(features_fp, index=False)

    X_error = error_df.reindex(columns=cols, fill_value=0).copy()

    if X_error.empty:
        raise RuntimeError("Error feature matrix is empty.")
    if X_error.isna().sum().sum() > 0:
        raise RuntimeError("Error feature matrix contains NaN values.")
    if np.isinf(X_error.to_numpy(dtype=float)).sum() > 0:
        raise RuntimeError("Error feature matrix contains infinite values.")

    if TARGET_ACTIVITY_CODE == "t-all":
        X_error_scaled = apply_activity_scalers(
            feature_df=error_df,
            cols=cols,
            scalers=scaler,
            activity_col="activity_code",
        )
    else:
        X_error_scaled = scaler.transform(X_error)

    y_true = np.ones(len(error_df), dtype=int)

    if TARGET_ACTIVITY_CODE == "t-all":
        if thresholds_by_activity is None:
            y_pred, reconstruction_error, _, prediction_mode = predict_autoencoder(
                model,
                X_error_scaled,
                threshold=None,
            )
        else:
            if use_per_activity_loss:
                y_pred, reconstruction_error, _, row_thresholds = predict_autoencoder_by_activity(
                    model=model,
                    X=X_error_scaled,
                    activity_codes=error_df["activity_code"],
                    thresholds_by_activity=thresholds_by_activity,
                    loss_name_by_activity=loss_name_by_activity,
                    huber_delta_by_activity=huber_delta_by_activity,
                )
            else:
                # 🔥 THIS IS THE FIX
                reconstruction_error = reconstruction_errors(model, X_error_scaled)
                row_thresholds = error_df["activity_code"].map(thresholds_by_activity).to_numpy(dtype=float)

                y_pred = (reconstruction_error > row_thresholds).astype(int)
                native_pred = y_pred.copy()

            prediction_mode = "percentile_threshold_by_activity"
    else:
        y_pred, reconstruction_error, native_pred, prediction_mode = predict_autoencoder(
            model,
            X_error_scaled,
            threshold=threshold,
        )

    print("\nPrediction summary")
    print("Predicted normal count: ", int((y_pred == 0).sum()))
    print("Predicted anomaly count:", int((y_pred == 1).sum()))
    print("Predicted anomaly rate: ", float(np.mean(y_pred == 1)))

    print_confusion_matrix(
        y_true,
        y_pred,
        title="Error Inference Confusion Matrix",
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["normal", "anomaly"],
        zero_division=0,
    )
    print("\nClassification Report")
    print(report)

    predictions_df = error_df[
        ["filename", "participant_id", "activity_code", "activity_index", "dataset"]
    ].copy()
    predictions_df["y_true"] = y_true
    predictions_df["y_pred"] = y_pred
    predictions_df["true_label"] = "anomaly"
    predictions_df["pred_label"] = np.where(y_pred == 1, "anomaly", "normal")
    predictions_df["reconstruction_error"] = reconstruction_error
    if TARGET_ACTIVITY_CODE == "t-all":
        if thresholds_by_activity is None:
            predictions_df["threshold"] = np.nan
        else:
            predictions_df["threshold"] = predictions_df["activity_code"].map(thresholds_by_activity)
    else:
        predictions_df["threshold"] = threshold
    predictions_df["prediction_mode"] = prediction_mode

    if TARGET_ACTIVITY_CODE == "t-all":
        if use_per_activity_loss:
            predictions_df["loss_name"] = predictions_df["activity_code"].map(loss_name_by_activity or {})
            predictions_df["huber_delta"] = predictions_df["activity_code"].map(huber_delta_by_activity or {})
        else:
            predictions_df["loss_name"] = loss_name
            predictions_df["huber_delta"] = huber_delta
    else:
        predictions_df["loss_name"] = loss_name
        predictions_df["huber_delta"] = huber_delta

    preds_fp = OUTPUT_DIR / f"error_predictions_{TARGET_ACTIVITY_CODE}.csv"
    predictions_df.to_csv(preds_fp, index=False)

    if TARGET_ACTIVITY_CODE == "t-all":
        save_confusion_matrices_by_activity_one_file(
            preds_df=predictions_df,
            csv_fp=OUTPUT_DIR / "error_confusion_matrices_by_activity.csv",
            txt_fp=OUTPUT_DIR / "error_confusion_matrices_by_activity.txt",
            title_prefix="Error Inference",
        )

    pca_error_df = error_df.merge(
        predictions_df,
        on=["filename", "participant_id", "activity_code", "activity_index", "dataset"],
        how="left",
    )
    pca_error_df["source_split"] = "error"
    pca_error_df.to_csv(
        OUTPUT_DIR / f"{TARGET_ACTIVITY_CODE}_pca_input_error.csv",
        index=False,
    )
    pca_error_df.to_csv(
        OUTPUT_DIR / f"error_pca_input_{TARGET_ACTIVITY_CODE}.csv",
        index=False,
    )

    _, cm_df = build_confusion_matrix_df(y_true, y_pred)
    cm_csv_fp = OUTPUT_DIR / f"error_confusion_matrix_{TARGET_ACTIVITY_CODE}.csv"
    cm_txt_fp = OUTPUT_DIR / f"error_confusion_matrix_{TARGET_ACTIVITY_CODE}.txt"
    save_confusion_matrix_files(
        cm_df,
        cm_csv_fp,
        cm_txt_fp,
        title="Error Inference Confusion Matrix",
    )

    report_fp = OUTPUT_DIR / f"error_classification_report_{TARGET_ACTIVITY_CODE}.txt"
    with open(report_fp, "w") as f:
        f.write(report)

    metrics_payload = {
        "model_type": "Autoencoder",
        "target_activity_code": TARGET_ACTIVITY_CODE,
        "use_activity_aware_loss": use_activity_aware_loss,
        "loss_name": loss_name if not use_per_activity_loss else None,
        "huber_delta": huber_delta if not use_per_activity_loss else None,
        "loss_name_by_activity": loss_name_by_activity if use_per_activity_loss else None,
        "huber_delta_by_activity": huber_delta_by_activity if use_per_activity_loss else None,
        "n_error_sessions": int(len(error_df)),
        "threshold": threshold,
        "prediction_mode": prediction_mode,
        "thresholds_by_activity": thresholds_by_activity if TARGET_ACTIVITY_CODE == "t-all" else None,
        "n_predicted_normal": int((y_pred == 0).sum()),
        "n_predicted_anomaly": int((y_pred == 1).sum()),
        "predicted_anomaly_rate": float(np.mean(y_pred == 1)),
    }

    metrics_fp = OUTPUT_DIR / f"error_inference_metrics_{TARGET_ACTIVITY_CODE}.json"
    with open(metrics_fp, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\nSaved outputs")
    print(f"  Feature matrix   -> {features_fp}")
    print(f"  Predictions      -> {preds_fp}")
    print(f"  Confusion CSV    -> {cm_csv_fp}")
    print(f"  Confusion TXT    -> {cm_txt_fp}")
    print(f"  Report           -> {report_fp}")
    print(f"  Metrics          -> {metrics_fp}")


if __name__ == "__main__":
    main()