"""Isolation Forest: model creation, thresholds, prediction, scaling and saving model bundles."""

import numpy as np
import pandas as pd
from abnormal_activity_detection.evaluation import classification_metrics_from_preds
import joblib
from abnormal_activity_detection.reference_stats import serialize_reference_stats
from abnormal_activity_detection.anomaly.scoring import percentile_threshold
import json
from sklearn.ensemble import IsolationForest
from abnormal_activity_detection.evaluation import (
    fit_activity_scalers,
    apply_activity_scalers,
)

def permutation_importance_iforest(
    model,
    X,
    y_true,
    activity_codes,
    thresholds_by_activity,
    feature_names,
):
    # Ensure DataFrame with correct column names/order
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)

    base_pred, _, _, _ = predict_iforest_by_activity(
        model=model,
        X=X,
        activity_codes=activity_codes,
        thresholds_by_activity=thresholds_by_activity,
    )

    base_metrics = classification_metrics_from_preds(y_true, base_pred)
    base_recall = base_metrics["recall"]

    importances = []

    for i, col in enumerate(feature_names):
        X_permuted = X.copy()
        X_permuted.iloc[:, i] = np.random.permutation(X_permuted.iloc[:, i].to_numpy())

        perm_pred, _, _, _ = predict_iforest_by_activity(
            model=model,
            X=X_permuted,
            activity_codes=activity_codes,
            thresholds_by_activity=thresholds_by_activity,
        )

        perm_metrics = classification_metrics_from_preds(y_true, perm_pred)
        perm_recall = perm_metrics["recall"]

        importance = base_recall - perm_recall
        importances.append((col, importance))

    return sorted(importances, key=lambda x: x[1], reverse=True)



def predict_iforest_by_activity(model, X, activity_codes, thresholds_by_activity):
    decision_scores = model.decision_function(X).ravel()

    raw_pred = model.predict(X)
    native_pred = np.where(raw_pred == 1, 0, 1)

    activity_codes = pd.Series(activity_codes).astype(str)
    y_pred = np.zeros(len(activity_codes), dtype=int)

    for i, act in enumerate(activity_codes):
        if act not in thresholds_by_activity:
            raise KeyError(f"No threshold found for activity {act}")
        y_pred[i] = int(decision_scores[i] < thresholds_by_activity[act])

    return y_pred, decision_scores, native_pred, "percentile_threshold_by_activity"



def get_threshold(train_scores: np.ndarray, threshold_percentile):
    """Isolation Forest threshold: the training-score percentile, or None."""
    return percentile_threshold(train_scores, threshold_percentile)


def predict_iforest(model, X, threshold=None):
    """
    Predict with a fitted Isolation Forest.

    Returns
    -------
    y_pred : np.ndarray
        Final binary predictions where 0=normal, 1=anomaly.
    decision_scores : np.ndarray
        decision_function scores from the model.
    native_pred : np.ndarray
        Native Isolation Forest predictions converted to 0=normal, 1=anomaly.
    prediction_mode : str
        "native_iforest" or "percentile_threshold"
    """
    decision_scores = model.decision_function(X).ravel()

    raw_pred = model.predict(X)  # 1=normal, -1=anomaly
    native_pred = np.where(raw_pred == 1, 0, 1)

    if threshold is None:
        y_pred = native_pred.copy()
        prediction_mode = "native_iforest"
    else:
        y_pred = np.where(decision_scores < threshold, 1, 0)
        prediction_mode = "percentile_threshold"

    return y_pred, decision_scores, native_pred, prediction_mode



def save_model_bundle(
    model,
    scaler,
    cols,
    reference_stats,
    threshold,
    config,
    target_activity_code,
    out_dir,
    thresholds_by_activity=None,
    prediction_mode=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / f"{target_activity_code}_iforest.joblib")
    joblib.dump(scaler, out_dir / f"{target_activity_code}_activity_scalers.joblib")

    if prediction_mode is None:
        if thresholds_by_activity is not None:
            prediction_mode = "percentile_threshold_by_activity"
        else:
            prediction_mode = "native_iforest" if threshold is None else "percentile_threshold"

    assets = {
        "activity": target_activity_code,
        "feature_columns": cols,
        "threshold_percentile": config.get("threshold_percentile"),
        "threshold_percentiles_by_activity": config.get("threshold_percentiles_by_activity"),
        "threshold": threshold,
        "thresholds_by_activity": thresholds_by_activity,
        "prediction_mode": prediction_mode,
        "reference_stats": serialize_reference_stats(reference_stats),
        "iforest_config": config,
    }

    with open(out_dir / f"{target_activity_code}_training_assets.json", "w") as f:
        json.dump(assets, f, indent=2)


def make_iforest(config: dict) -> IsolationForest:
    return IsolationForest(
        n_estimators=config["n_estimators"],
        max_samples=config["max_samples"],
        contamination=config["contamination"],
        max_features=config["max_features"],
        bootstrap=config["bootstrap"],
        random_state=config["random_state"],
    )


def config_to_tag(config_id: int, config: dict) -> str:
    base = (
        f"cfg_{config_id:03d}"
        f"__nest_{config['n_estimators']}"
        f"__ms_{str(config['max_samples']).replace('.', 'p')}"
        f"__mf_{str(config['max_features']).replace('.', 'p')}"
        f"__cont_{str(config['contamination']).replace('.', 'p')}"
        f"__boot_{int(bool(config['bootstrap']))}"
    )

    if "threshold_percentiles_by_activity" in config:
        thr = config["threshold_percentiles_by_activity"]
        thr_tag = "__thracts_" + "_".join(
            f"{act}_{str(thr[act]).replace('.', 'p')}"
            for act in sorted(thr.keys())
        )
        return base + thr_tag

    elif "threshold_percentile" in config:
        return base + f"__thr_{str(config['threshold_percentile']).replace('.', 'p')}"




def get_model_input_for_fit_and_infer(train_df, infer_df, cols, target_activity_code):
    """
    Match scaling behaviour to the other scripts:
    - t-all: per-activity scaling
    - single activity: no scaling
    """
    if target_activity_code == "t-all":
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

        X_infer = apply_activity_scalers(
            feature_df=infer_df,
            cols=cols,
            scalers=scalers,
            activity_col="activity_code",
        )
    else:
        scalers = None
        X_train = train_df[cols].copy().astype(float)
        X_infer = infer_df.reindex(columns=cols, fill_value=0)[cols].copy().astype(float)

    return X_train, X_infer, scalers


def get_model_input_for_final_train(final_normal_df, cols, target_activity_code):
    """
    Match scaling behaviour to the other scripts:
    - t-all: per-activity scaling
    - single activity: no scaling
    """
    if target_activity_code == "t-all":
        scalers = fit_activity_scalers(
            feature_df=final_normal_df,
            scale_cols=[c for c in cols if not c.startswith("activity_")],
            activity_col="activity_code",
        )

        X_final = apply_activity_scalers(
            feature_df=final_normal_df,
            cols=cols,
            scalers=scalers,
            activity_col="activity_code",
        )
    else:
        scalers = None
        X_final = final_normal_df[cols].copy().astype(float)

    return X_final, scalers


def apply_saved_scaling_to_error(error_df, cols, scalers, target_activity_code):
    """
    Match scaling behaviour to the other scripts:
    - t-all: apply saved per-activity scalers
    - single activity: no scaling
    """
    if target_activity_code == "t-all":
        return apply_activity_scalers(
            feature_df=error_df,
            cols=cols,
            scalers=scalers,
            activity_col="activity_code",
        )

    return error_df.reindex(columns=cols, fill_value=0)[cols].copy().astype(float)


