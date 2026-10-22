#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

from abnormal_activity_detection.paths import RESULTS_DIR
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = RESULTS_DIR / "activity_classification" / "features"

FEATURES_DIR = BASE_DIR / "engineered_features"

NOERROR_PATH = FEATURES_DIR / "noerror_all_features.csv"
ERROR_PATH   = FEATURES_DIR / "error_all_features.csv"

OUTPUT_DIR = BASE_DIR / "boruta"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODE = "error"   # "noerror" | "error" | "combined"
TARGET_COL = "activity_code"
RANDOM_STATE = 42

TOP_N_RANK_PLOT = 30
SAVE_PLOTS = False

# Exact feature names
EXACT_FEATURES = [
    "EPISODE_DURATION",
    "NUM_EVENTS",
    "N_BIGRAMS",
    "UNIQUE_MESSAGES",
]

# Prefix groups
FEATURE_PREFIXES = [
    "analogue_ad1_b_dur_bucket::",
    "analogue_ad1_c_dur_bucket::",
    "evt::",
    "evt_bin::",
    # "msg::",
    # "sensor::",
]

# =========================================================
# LOAD DATA
# =========================================================
def load_dataset(mode):
    if mode == "noerror":
        return pd.read_csv(NOERROR_PATH)
    elif mode == "error":
        return pd.read_csv(ERROR_PATH)
    elif mode == "combined":
        df1 = pd.read_csv(NOERROR_PATH)
        df2 = pd.read_csv(ERROR_PATH)
        return pd.concat([df1, df2], ignore_index=True)
    else:
        raise ValueError("Invalid MODE")


def collect_matching_features(columns, exact_features, prefixes):
    selected = []

    # Add exact matches
    for feat in exact_features:
        if feat in columns:
            selected.append(feat)
        else:
            print(f"Warning: exact feature not found: {feat}")

    # Add prefix matches
    for col in columns:
        if any(col.startswith(prefix) for prefix in prefixes):
            selected.append(col)

    # Remove duplicates, preserve order
    seen = set()
    deduped = []
    for col in selected:
        if col not in seen:
            deduped.append(col)
            seen.add(col)

    return deduped


# =========================================================
# PLOTTING
# =========================================================
def save_boruta_summary_plot(result_df: pd.DataFrame, output_dir: Path, mode: str):
    selected_count = int(result_df["selected"].sum())
    tentative_count = int(result_df["tentative"].sum())
    rejected_count = int((~result_df["selected"] & ~result_df["tentative"]).sum())

    plot_df = pd.DataFrame({
        "status": ["selected", "tentative", "rejected"],
        "count": [selected_count, tentative_count, rejected_count],
    })

    plt.figure(figsize=(7, 5))
    plt.bar(plot_df["status"], plot_df["count"])
    plt.ylabel("Number of Features")
    plt.title(f"Boruta Feature Selection Summary ({mode})")
    plt.tight_layout()

    out_fp = output_dir / f"boruta_{mode}_selection_summary.png"
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved graph: {out_fp}")


def save_boruta_rank_plot(result_df: pd.DataFrame, output_dir: Path, mode: str, top_n: int = 30):
    plot_df = result_df.sort_values(["ranking", "feature"]).head(top_n).copy()

    if plot_df.empty:
        print("Skipping rank plot: no rows to plot.")
        return

    plot_df = plot_df.sort_values(["ranking", "feature"], ascending=[False, True])

    plt.figure(figsize=(12, max(6, len(plot_df) * 0.35)))
    plt.barh(plot_df["feature"], plot_df["ranking"])
    plt.xlabel("Boruta Rank (1 = best)")
    plt.ylabel("Feature")
    plt.title(f"Top {len(plot_df)} Boruta Feature Rankings ({mode})")
    plt.tight_layout()

    out_fp = output_dir / f"boruta_{mode}_top_{top_n}_ranking.png"
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved graph: {out_fp}")


def save_boruta_selected_plot(result_df: pd.DataFrame, output_dir: Path, mode: str):
    selected_df = result_df[result_df["selected"]].copy()

    if selected_df.empty:
        print("Skipping selected-features plot: no selected features.")
        return

    selected_df = selected_df.sort_values(["ranking", "feature"], ascending=[False, True])

    plt.figure(figsize=(12, max(6, len(selected_df) * 0.35)))
    plt.barh(selected_df["feature"], selected_df["ranking"])
    plt.xlabel("Boruta Rank (1 = best)")
    plt.ylabel("Selected Feature")
    plt.title(f"Boruta Selected Features ({mode})")
    plt.tight_layout()

    out_fp = output_dir / f"boruta_{mode}_selected_features.png"
    plt.savefig(out_fp, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved graph: {out_fp}")


# =========================================================
# MAIN
# =========================================================
def main():
    df = load_dataset(MODE)

    print(f"\nLoaded mode: {MODE}")
    print(f"Dataset size: {len(df)}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    # =====================================================
    # PREPARE TARGET
    # =====================================================
    if TARGET_COL == "activity_code":
        y = df[TARGET_COL].astype(str)
        y = pd.Categorical(y, categories=sorted(y.unique()))
        y = y.codes
    else:
        y = df[TARGET_COL].values

    # =====================================================
    # PREPARE FEATURES
    # =====================================================
    FEATURES_TO_USE = collect_matching_features(
        columns=df.columns,
        exact_features=EXACT_FEATURES,
        prefixes=FEATURE_PREFIXES,
    )

    if not FEATURES_TO_USE:
        raise ValueError("No matching features found.")

    X = df[FEATURES_TO_USE].copy()

    # Handle missing values
    X = X.fillna(0)

    # Keep numeric/bool only
    X = X.select_dtypes(include=["number", "bool"]).astype(float)

    if X.shape[1] == 0:
        raise ValueError("No numeric features left after filtering.")

    print(f"\nUsing {X.shape[1]} features")
    print("\nSelected feature columns:")
    for col in X.columns:
        print(col)

    # =====================================================
    # BORUTA
    # =====================================================
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    boruta = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        verbose=2,
        random_state=RANDOM_STATE
    )

    boruta.fit(X.values, y)

    # =====================================================
    # RESULTS
    # =====================================================
    selected = X.columns[boruta.support_]
    tentative = X.columns[boruta.support_weak_]
    ranking = boruta.ranking_

    print("\n==============================")
    print("SELECTED FEATURES")
    print("==============================")
    print(list(selected))

    print("\n==============================")
    print("TENTATIVE FEATURES")
    print("==============================")
    print(list(tentative))

    # =====================================================
    # SAVE RESULTS
    # =====================================================
    result_df = pd.DataFrame({
        "feature": X.columns,
        "ranking": ranking,
        "selected": boruta.support_,
        "tentative": boruta.support_weak_
    }).sort_values(["ranking", "feature"]).reset_index(drop=True)

    output_csv = OUTPUT_DIR / f"boruta_{MODE}.csv"
    result_df.to_csv(output_csv, index=False)

    print(f"\nSaved results to: {output_csv}")

    # =====================================================
    # SAVE GRAPHS
    # =====================================================
    save_boruta_summary_plot(
        result_df=result_df,
        output_dir=OUTPUT_DIR,
        mode=MODE,
    )

    

    if SAVE_PLOTS:

        save_boruta_rank_plot(
            result_df=result_df,
            output_dir=OUTPUT_DIR,
            mode=MODE,
            top_n=TOP_N_RANK_PLOT,
        )
        
        save_boruta_selected_plot(
            result_df=result_df,
            output_dir=OUTPUT_DIR,
            mode=MODE,
        )


if __name__ == "__main__":
    main()