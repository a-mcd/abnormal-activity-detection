#!/usr/bin/env python3
import numpy as np
from abnormal_activity_detection.anomaly.configs.iforest import (
    IFOREST_CONFIG,
    TARGET_ACTIVITY_CODE,
    FEATURE_CONFIG,
    META_COLS,
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
    fit_activity_scalers,
    apply_activity_scalers,
    get_thresholds_by_activity
)
from abnormal_activity_detection.anomaly.iforest import (
    make_iforest,
    get_threshold,
    predict_iforest,
    predict_iforest_by_activity,
    save_model_bundle,
)




def main():
    activity = TARGET_ACTIVITY_CODE

    normal_dir, _ = get_dataset_dirs()

    normal_sessions, normal_meta = load_sessions_for_activity(
        directory=normal_dir,
        target_activity=activity,
        dataset_label="noerror",
    )

    if not normal_sessions:
        raise ValueError(f"No normal sessions found for activity={activity} in {normal_dir}")


    cols = feature_columns(target_activity=activity, feature_config=FEATURE_CONFIG)

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

    
    if activity == "t-all":
        scalers = fit_activity_scalers(
            feature_df=feature_df,
            scale_cols=[c for c in cols if not c.startswith("activity_")],
            activity_col="activity_code",
        )

        X = apply_activity_scalers(
            feature_df=feature_df,
            cols=cols,
            scalers=scalers,
            activity_col="activity_code",
        )
    else:
        scalers = None
        X = feature_df[cols].copy().astype(float)


    out_dir = get_results_dir("abnormal_detection", "isolation_forest", activity, "iforest_train")

    config = IFOREST_CONFIG[TARGET_ACTIVITY_CODE]
    model = make_iforest(config)
    model.fit(X)

    decision_scores = model.decision_function(X).ravel()

    if activity == "t-all":
        percentiles_by_activity = config.get("threshold_percentiles_by_activity")
        if not percentiles_by_activity:
            raise ValueError("Missing threshold_percentiles_by_activity for t-all")

        thresholds_by_activity = get_thresholds_by_activity(
            scores=decision_scores,
            activity_codes=feature_df["activity_code"],
            percentiles_by_activity=percentiles_by_activity,
        )

        threshold = None
        threshold_percentile = None

        y_pred, decision_scores, native_pred, prediction_mode = predict_iforest_by_activity(
            model=model,
            X=X,
            activity_codes=feature_df["activity_code"],
            thresholds_by_activity=thresholds_by_activity,
        )
    else:
        threshold_percentile = config.get("threshold_percentile")
        thresholds_by_activity = None

        threshold = get_threshold(decision_scores, threshold_percentile)

        y_pred, decision_scores, native_pred, prediction_mode = predict_iforest(
            model=model,
            X=X,
            threshold=threshold,
        )
    

    feature_df.to_csv(out_dir / f"{activity}_train_features.csv", index=False)

    diag_df = feature_df[META_COLS].copy()
    diag_df["decision_score"] = decision_scores
    diag_df["y_pred"] = y_pred
    diag_df["pred_label"] = np.where(y_pred == 1, "anomaly", "normal")
    diag_df["prediction_mode"] = prediction_mode
    diag_df["native_pred"] = native_pred
    diag_df["native_pred_label"] = np.where(native_pred == 1, "anomaly", "normal")
    if activity == "t-all":
        diag_df["threshold"] = feature_df["activity_code"].map(thresholds_by_activity)
    else:
        diag_df["threshold"] = threshold

    diag_df.to_csv(out_dir / f"{activity}_train_predictions.csv", index=False)

    pca_train_df = feature_df.merge(
        diag_df,
        on=["filename", "participant_id", "activity_code", "activity_index", "dataset"],
        how="left",
    )

    pca_train_df["source_split"] = "noerror"

    pca_train_df.to_csv(
        out_dir / f"{activity}_pca_input_noerror.csv",
        index=False,
    )

    save_model_bundle(
        model=model,
        scaler=scalers,
        cols=cols,
        reference_stats=reference_stats,
        threshold=threshold,
        config=config,
        target_activity_code=activity,
        out_dir=out_dir,
        thresholds_by_activity=thresholds_by_activity,
        prediction_mode=prediction_mode,
    )

    print(f"Saved model bundle to: {out_dir}")
    print(f"Training rows: {len(feature_df)}")
    print(f"Training feature columns: {len(cols)}")
    print(f"Prediction mode: {prediction_mode}")
    print(f"Threshold: {threshold}")


if __name__ == "__main__":
    main()
