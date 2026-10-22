"""One-Class SVM: model creation, thresholds, prediction and saving model bundles."""

import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
import json
import joblib
from abnormal_activity_detection.reference_stats import serialize_reference_stats
from abnormal_activity_detection.anomaly.scoring import percentile_threshold

def predict_ocsvm_by_activity(model, X, activity_codes, thresholds_by_activity):
    activity_codes = pd.Series(activity_codes).reset_index(drop=True)

    missing = [
        activity for activity in activity_codes.unique()
        if activity not in thresholds_by_activity
    ]
    if missing:
        raise ValueError(f"Missing thresholds for activities: {missing}")

    scores = model.decision_function(X)
    raw_pred = model.predict(X)

    y_pred = np.zeros(len(activity_codes), dtype=int)
    row_thresholds = []

    for activity in activity_codes.unique():
        mask = activity_codes == activity
        threshold = thresholds_by_activity[activity]

        if threshold is None:
            y_pred[mask.to_numpy()] = np.where(
                raw_pred[mask.to_numpy()] == 1, 0, 1
            )
        else:
            y_pred[mask.to_numpy()] = (
                scores[mask.to_numpy()] < threshold
            ).astype(int)

    # per-row threshold tracking
    for activity in activity_codes:
        row_thresholds.append(thresholds_by_activity[activity])

    return y_pred, scores, raw_pred, row_thresholds


def get_linear_ocsvm_feature_weights(model, feature_cols):
    if not hasattr(model, "coef_"):
        raise ValueError("This only works for linear OCSVM models.")

    w = model.coef_.ravel()
    out = pd.DataFrame({
        "feature": feature_cols,
        "weight": w,
        "abs_weight": np.abs(w),
    }).sort_values("abs_weight", ascending=False)

    return out


def predict_ocsvm(model, X_scaled, threshold=None):
    """
    Predict with a fitted One-Class SVM.

    Returns
    -------
    y_pred : np.ndarray
        Final binary predictions where 0=normal, 1=anomaly.
    decision_scores : np.ndarray
        decision_function scores from the model.
    native_pred : np.ndarray
        Native OCSVM predictions converted to 0=normal, 1=anomaly.
    prediction_mode : str
        "native_ocsvm" or "percentile_threshold"
    """
    decision_scores = model.decision_function(X_scaled).ravel()
    raw_pred = model.predict(X_scaled)  # 1=normal, -1=anomaly
    native_pred = np.where(raw_pred == 1, 0, 1)

    if threshold is None:
        y_pred = native_pred.copy()
        prediction_mode = "native_ocsvm"
    else:
        y_pred = (decision_scores < threshold).astype(int)
        prediction_mode = "percentile_threshold"

    return y_pred, decision_scores, native_pred, prediction_mode


def make_ocsvm(config: dict) -> OneClassSVM:
    return OneClassSVM(
        kernel=config["kernel"],
        gamma=config["gamma"],
        nu=config["nu"],
    )

def get_threshold(train_scores: np.ndarray, threshold_percentile):
    """
    Compute threshold and prediction mode for OCSVM.

    Returns
    -------
    threshold : float | None
    prediction_mode : str
    """
    threshold = percentile_threshold(train_scores, threshold_percentile)
    if threshold is None:
        return None, "native_ocsvm"
    return threshold, "percentile_threshold"



def save_ocsvm_model_bundle(
    out_dir,
    target_activity_code,
    model,
    scaler,
    cols,
    reference_stats,
    threshold,
    thresholds_by_activity,
    prediction_mode,
    config,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / f"{target_activity_code}_ocsvm.joblib")
    joblib.dump(scaler, out_dir / f"{target_activity_code}_scaler.joblib")

    assets = {
        "activity": target_activity_code,
        "feature_columns": cols,
        "threshold_percentile": config.get("threshold_percentile"),
        "threshold_percentiles_by_activity": config.get("threshold_percentiles_by_activity"),
        "threshold": threshold,
        "thresholds_by_activity": thresholds_by_activity,
        "prediction_mode": prediction_mode,
        "reference_stats": serialize_reference_stats(reference_stats),
        "ocsvm_config": {
            "kernel": config.get("kernel", "rbf"),
            "gamma": config.get("gamma", "scale"),
            "nu": config.get("nu", 0.1),
            "threshold_percentile": config.get("threshold_percentile"),
            "threshold_percentiles_by_activity": config.get("threshold_percentiles_by_activity"),
        },
    }

    with open(out_dir / f"{target_activity_code}_training_assets.json", "w") as f:
        json.dump(assets, f, indent=2)


def config_to_tag(config_id: int, config: dict) -> str:
    kernel_str = str(config["kernel"]).replace(".", "p")
    gamma_str = str(config["gamma"]).replace(".", "p")
    nu_str = str(config["nu"]).replace(".", "p")

    if "threshold_percentiles_by_activity" in config and config["threshold_percentiles_by_activity"] is not None:
        thr_map = config["threshold_percentiles_by_activity"]
        thr_str = "__".join(f"{k}-{str(v).replace('.', 'p')}" for k, v in sorted(thr_map.items()))
    else:
        thr_val = config.get("threshold_percentile")
        thr_str = "none" if thr_val is None else str(thr_val).replace(".", "p")

    return (
        f"cfg_{config_id:03d}"
        f"__ker_{kernel_str}"
        f"__gam_{gamma_str}"
        f"__nu_{nu_str}"
        f"__thr_{thr_str}"
    )
