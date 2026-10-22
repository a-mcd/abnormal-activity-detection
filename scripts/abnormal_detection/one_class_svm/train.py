#!/usr/bin/env python3
import numpy as np
from sklearn.preprocessing import StandardScaler

from abnormal_activity_detection.anomaly.configs.ocsvm import (
    TARGET_ACTIVITY_CODE,
    OCSVM_CONFIG,
    FEATURE_CONFIG,
)
from abnormal_activity_detection.data import load_sessions_for_activity
from abnormal_activity_detection.features.anomaly import (
    build_feature_matrix,
    build_reference_stats,
    build_reference_stats_by_activity,
    feature_columns,
)
from abnormal_activity_detection.evaluation import (
    apply_activity_scalers,
    fit_activity_scalers,
    get_thresholds_by_activity,
)
from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir
from abnormal_activity_detection.anomaly.ocsvm import (
    predict_ocsvm_by_activity,
    predict_ocsvm,
    make_ocsvm,
    get_threshold,
    save_ocsvm_model_bundle,
)

META_COLS = [
    "filename",
    "participant_id",
    "activity_code",
    "activity_index",
    "dataset",
]



def main():

    normal_dir, _ = get_dataset_dirs()

    normal_sessions, normal_meta = load_sessions_for_activity(
        directory=normal_dir,
        target_activity=TARGET_ACTIVITY_CODE,
        dataset_label="noerror",
    )

    if not normal_sessions:
        raise ValueError(f"No normal sessions found for activity={TARGET_ACTIVITY_CODE} in {normal_dir}")

    cols = feature_columns(target_activity=TARGET_ACTIVITY_CODE, feature_config=FEATURE_CONFIG)
    if TARGET_ACTIVITY_CODE == "t-all":
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

    out_dir = get_results_dir("abnormal_detection", "one_class_svm", TARGET_ACTIVITY_CODE, "ocsvm_train")

    X = feature_df[cols].copy().astype(float)

    if X.empty:
        raise RuntimeError("Training feature matrix is empty.")
    if X.isna().sum().sum() > 0:
        raise RuntimeError("Training feature matrix contains NaN values.")
    if np.isinf(X.to_numpy(dtype=float)).sum() > 0:
        raise RuntimeError("Training feature matrix contains infinite values.")

    if TARGET_ACTIVITY_CODE == "t-all":
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

    config = OCSVM_CONFIG[TARGET_ACTIVITY_CODE]
    model = make_ocsvm(config)
    model.fit(X_scaled)

    decision_scores = model.decision_function(X_scaled).ravel()
    threshold_percentile = config.get("threshold_percentile")
    threshold_percentiles_by_activity = config.get("threshold_percentiles_by_activity")

    threshold = None
    thresholds_by_activity = None
    prediction_mode = "native_ocsvm"

    if TARGET_ACTIVITY_CODE == "t-all":
        if threshold_percentiles_by_activity is not None:
            thresholds_by_activity = get_thresholds_by_activity(
                scores=decision_scores,
                activity_codes=feature_df["activity_code"],
                percentiles_by_activity=threshold_percentiles_by_activity,
            )
            prediction_mode = "percentile_threshold_by_activity"
    else:
        threshold, prediction_mode = get_threshold(
            decision_scores,
            threshold_percentile,
        )


    if TARGET_ACTIVITY_CODE == "t-all":
        if thresholds_by_activity is None:
            raw_pred = model.predict(X_scaled)
            native_pred = np.where(raw_pred == 1, 0, 1)
            y_pred = native_pred.copy()
        else:
            y_pred, decision_scores, native_pred, _ = predict_ocsvm_by_activity(
                model=model,
                X=X_scaled,
                activity_codes=feature_df["activity_code"],
                thresholds_by_activity=thresholds_by_activity,
            )
    else:
        y_pred, decision_scores, native_pred, prediction_mode = predict_ocsvm(
            model=model,
            X_scaled=X_scaled,
            threshold=threshold,
        )



    feature_df.to_csv(out_dir / f"{TARGET_ACTIVITY_CODE}_train_features.csv", index=False)

    diag_df = feature_df[META_COLS].copy()
    diag_df["decision_score"] = decision_scores
    diag_df["y_pred"] = y_pred
    diag_df["pred_label"] = np.where(y_pred == 1, "anomaly", "normal")
    diag_df["prediction_mode"] = prediction_mode
    if TARGET_ACTIVITY_CODE == "t-all":
        if thresholds_by_activity is None:
            diag_df["threshold"] = np.nan
        else:
            diag_df["threshold"] = diag_df["activity_code"].map(thresholds_by_activity)
    else:
        diag_df["threshold"] = threshold
    diag_df.to_csv(out_dir / f"{TARGET_ACTIVITY_CODE}_train_predictions.csv", index=False)

    save_ocsvm_model_bundle(
        out_dir=out_dir,
        target_activity_code=TARGET_ACTIVITY_CODE,
        model=model,
        scaler=scaler_to_save,
        cols=cols,
        reference_stats=reference_stats,
        threshold=threshold,
        thresholds_by_activity=thresholds_by_activity,
        prediction_mode=prediction_mode,
        config=config,
    )

    print(f"Saved model bundle to: {out_dir}")
    print(f"Training rows: {len(feature_df)}")
    print(f"Training feature columns: {len(cols)}")
    print(f"Prediction mode: {prediction_mode}")
    print(f"Threshold: {threshold}")


if __name__ == "__main__":
    main()
