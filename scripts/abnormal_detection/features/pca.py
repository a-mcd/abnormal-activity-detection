#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from abnormal_activity_detection.anomaly.configs.features import (
    TARGET_ACTIVITY_CODE,
    FEATURE_CONFIG,
)
from abnormal_activity_detection.data import load_sessions_for_activity
from abnormal_activity_detection.features.anomaly import (
    build_reference_stats,
    build_feature_matrix,
    build_feature_matrix_train_loo_reference,
)
from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir


# -------------------------
# Paths
# -------------------------
NORMAL_DIR, ERROR_DIR = get_dataset_dirs()
OUTPUT_DIR = get_results_dir("abnormal_detection", "features", TARGET_ACTIVITY_CODE or "all_activities", "pca_feature_analysis")


def get_activity_feature_config(target_activity_code: str):
    """
    Resolve the feature configuration for the requested activity.

    Supported patterns:
    1. FEATURE_CONFIG = "ALL"
    2. FEATURE_CONFIG = [...]
    3. FEATURE_CONFIG = {"t1": [...], "t2": [...], ...}
    4. FEATURE_CONFIG = {"default": [...], "t5": [...]}
    """
    if isinstance(FEATURE_CONFIG, str):
        return FEATURE_CONFIG

    if isinstance(FEATURE_CONFIG, (list, tuple)):
        return list(FEATURE_CONFIG)

    if isinstance(FEATURE_CONFIG, dict):
        if target_activity_code in FEATURE_CONFIG:
            return FEATURE_CONFIG[target_activity_code]
        if "default" in FEATURE_CONFIG:
            return FEATURE_CONFIG["default"]
        raise ValueError(
            f"No feature config found for activity '{target_activity_code}', "
            f"and no 'default' entry exists in FEATURE_CONFIG."
        )

    raise TypeError(
        "FEATURE_CONFIG must be one of: 'ALL', list, tuple, or dict."
    )


def save_pca_plot(pca_df: pd.DataFrame, variance_ratio, output_fp):
    plt.figure(figsize=(9, 7))

    normal_mask = pca_df["label_name"] == "normal"
    error_mask = pca_df["label_name"] == "error"

    plt.scatter(
        pca_df.loc[normal_mask, "PC1"],
        pca_df.loc[normal_mask, "PC2"],
        label="Normal",
        alpha=0.8,
    )

    plt.scatter(
        pca_df.loc[error_mask, "PC1"],
        pca_df.loc[error_mask, "PC2"],
        label="Error",
        alpha=0.8,
    )

    for _, row in pca_df.iterrows():
        plt.annotate(
            row["filename"],
            (row["PC1"], row["PC2"]),
            fontsize=7,
            alpha=0.7,
        )

    plt.xlabel(f"PC1 ({variance_ratio[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({variance_ratio[1] * 100:.1f}% variance)")
    plt.title(f"PCA of Session Features - {TARGET_ACTIVITY_CODE}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_fp, dpi=300, bbox_inches="tight")
    plt.close()


def build_feature_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = df[col]
        rows.append({
            "feature": col,
            "n_unique": int(s.nunique(dropna=False)),
            "std": float(s.std(ddof=0)) if len(s) > 0 else 0.0,
            "min": float(s.min()) if len(s) > 0 else 0.0,
            "max": float(s.max()) if len(s) > 0 else 0.0,
            "mean": float(s.mean()) if len(s) > 0 else 0.0,
            "all_zero": bool((s == 0).all()),
        })
    return pd.DataFrame(rows).sort_values(["std", "n_unique"], ascending=[True, True])


def build_feature_summary_by_label(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        for label_name in ["normal", "error"]:
            s = df.loc[df["label_name"] == label_name, col]
            rows.append({
                "feature": col,
                "label_name": label_name,
                "n_unique": int(s.nunique(dropna=False)),
                "std": float(s.std(ddof=0)) if len(s) > 0 else 0.0,
                "min": float(s.min()) if len(s) > 0 else 0.0,
                "max": float(s.max()) if len(s) > 0 else 0.0,
                "mean": float(s.mean()) if len(s) > 0 else 0.0,
                "all_zero": bool((s == 0).all()) if len(s) > 0 else True,
            })
    return pd.DataFrame(rows).sort_values(["feature", "label_name"])


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    meta_cols = {
        "filename",
        "source_file",
        "participant_id",
        "activity_code",
        "activity_index",
        "dataset",
        "label",
        "label_name",
    }

    cols = []
    for col in df.columns:
        if col in meta_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)

    return cols


def main():
    # -------------------------
    # Resolve feature config for this activity
    # -------------------------
    activity_feature_config = get_activity_feature_config(TARGET_ACTIVITY_CODE)

    # -------------------------
    # Load sessions
    # -------------------------
    normal_session_dfs, normal_meta = load_sessions_for_activity(
        NORMAL_DIR,
        TARGET_ACTIVITY_CODE,
        dataset_label="normal",
    )

    error_session_dfs, error_meta = load_sessions_for_activity(
        ERROR_DIR,
        TARGET_ACTIVITY_CODE,
        dataset_label="error",
    )

    if len(normal_session_dfs) == 0:
        raise RuntimeError(
            f"No normal sessions loaded for {TARGET_ACTIVITY_CODE} from {NORMAL_DIR}"
        )

    if len(error_session_dfs) == 0:
        raise RuntimeError(
            f"No error sessions loaded for {TARGET_ACTIVITY_CODE} from {ERROR_DIR}"
        )

    # -------------------------
    # Build reference stats from NORMAL only
    # -------------------------
    reference_stats = build_reference_stats(normal_session_dfs)

    # -------------------------
    # Build features
    # -------------------------
    normal_df = build_feature_matrix_train_loo_reference(
        normal_session_dfs,
        normal_meta,
        activity_feature_config,
    )

    error_df = build_feature_matrix(
        error_session_dfs,
        error_meta,
        activity_feature_config,
        reference_stats=reference_stats,
    )

    # -------------------------
    # Align schemas
    # -------------------------
    all_cols = sorted(set(normal_df.columns) | set(error_df.columns))
    normal_df = normal_df.reindex(columns=all_cols, fill_value=0.0)
    error_df = error_df.reindex(columns=all_cols, fill_value=0.0)

    # -------------------------
    # Combine
    # -------------------------
    combined_df = pd.concat([normal_df, error_df], axis=0, ignore_index=True)
    combined_df["label"] = np.where(combined_df["dataset"] == "error", 1, 0)
    combined_df["label_name"] = np.where(combined_df["label"] == 1, "error", "normal")

    # -------------------------
    # Feature columns
    # -------------------------
    cols = get_feature_columns(combined_df)

    if len(cols) == 0:
        raise RuntimeError("No numeric feature columns found for PCA.")

    # -------------------------
    # Feature diagnostics
    # -------------------------
    feature_summary_df = build_feature_summary(combined_df, cols)
    feature_summary_by_label_df = build_feature_summary_by_label(combined_df, cols)

    near_zero_var_df = feature_summary_df[
        (feature_summary_df["std"] == 0.0) | (feature_summary_df["n_unique"] <= 1)
    ].copy()

    # -------------------------
    # PCA
    # -------------------------
    X = combined_df[cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    keep_meta_cols = [
        c for c in [
            "filename",
            "source_file",
            "participant_id",
            "activity_code",
            "activity_index",
            "dataset",
            "label",
            "label_name",
        ]
        if c in combined_df.columns
    ]

    pca_df = combined_df[keep_meta_cols].copy()
    pca_df["PC1"] = X_pca[:, 0]
    pca_df["PC2"] = X_pca[:, 1]

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=cols,
        columns=["PC1_loading", "PC2_loading"],
    ).reset_index(names="feature")

    variance_payload = {
        "target_activity_code": TARGET_ACTIVITY_CODE,
        "feature_config": (
            activity_feature_config
            if isinstance(activity_feature_config, str)
            else list(activity_feature_config)
        ),
        "n_normal_sessions": int(len(normal_df)),
        "n_error_sessions": int(len(error_df)),
        "n_features": int(len(cols)),
        "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "pc2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
        "pc1_pc2_total_explained_variance_ratio": float(
            pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]
        ),
    }

    # -------------------------
    # Save outputs
    # -------------------------
    normal_fp = OUTPUT_DIR / f"normal_features_{TARGET_ACTIVITY_CODE}.csv"
    error_fp = OUTPUT_DIR / f"error_features_{TARGET_ACTIVITY_CODE}.csv"
    combined_fp = OUTPUT_DIR / f"combined_features_{TARGET_ACTIVITY_CODE}.csv"
    pca_fp = OUTPUT_DIR / f"pca_projection_{TARGET_ACTIVITY_CODE}.csv"
    loadings_fp = OUTPUT_DIR / f"pca_loadings_{TARGET_ACTIVITY_CODE}.csv"
    variance_fp = OUTPUT_DIR / f"pca_variance_{TARGET_ACTIVITY_CODE}.json"
    plot_fp = OUTPUT_DIR / f"pca_scatter_{TARGET_ACTIVITY_CODE}.png"

    feature_summary_fp = OUTPUT_DIR / f"feature_summary_{TARGET_ACTIVITY_CODE}.csv"
    feature_summary_by_label_fp = OUTPUT_DIR / f"feature_summary_by_label_{TARGET_ACTIVITY_CODE}.csv"
    near_zero_var_fp = OUTPUT_DIR / f"near_zero_variance_features_{TARGET_ACTIVITY_CODE}.csv"

    normal_df.to_csv(normal_fp, index=False)
    error_df.to_csv(error_fp, index=False)
    combined_df.to_csv(combined_fp, index=False)
    pca_df.to_csv(pca_fp, index=False)
    loadings_df.to_csv(loadings_fp, index=False)
    feature_summary_df.to_csv(feature_summary_fp, index=False)
    feature_summary_by_label_df.to_csv(feature_summary_by_label_fp, index=False)
    near_zero_var_df.to_csv(near_zero_var_fp, index=False)

    with open(variance_fp, "w") as f:
        json.dump(variance_payload, f, indent=2)

    save_pca_plot(pca_df, pca.explained_variance_ratio_, plot_fp)

    print(f"Target activity: {TARGET_ACTIVITY_CODE}")
    print(f"Feature config resolved for activity: {TARGET_ACTIVITY_CODE}")
    print(f"Normal sessions: {len(normal_df)}")
    print(f"Error sessions:  {len(error_df)}")
    print(f"Features:        {len(cols)}")
    print(
        f"Explained variance (PC1+PC2): "
        f"{variance_payload['pc1_pc2_total_explained_variance_ratio']:.4f}"
    )

    print("\nNear-zero / constant features:")
    if near_zero_var_df.empty:
        print("  None")
    else:
        print(
            near_zero_var_df[
                ["feature", "n_unique", "std", "min", "max", "all_zero"]
            ].to_string(index=False)
        )

    print("\nSaved outputs")
    print(f"  Normal features           -> {normal_fp}")
    print(f"  Error features            -> {error_fp}")
    print(f"  Combined                  -> {combined_fp}")
    print(f"  PCA projection            -> {pca_fp}")
    print(f"  PCA loadings              -> {loadings_fp}")
    print(f"  PCA variance              -> {variance_fp}")
    print(f"  PCA plot                  -> {plot_fp}")
    print(f"  Feature summary           -> {feature_summary_fp}")
    print(f"  Feature summary by label  -> {feature_summary_by_label_fp}")
    print(f"  Near-zero variance        -> {near_zero_var_fp}")


if __name__ == "__main__":
    main()