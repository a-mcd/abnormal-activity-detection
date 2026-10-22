"""Scoring helpers shared by the anomaly detectors (Isolation Forest, One-Class SVM, autoencoder)."""

import numpy as np


def percentile_threshold(train_scores: np.ndarray, threshold_percentile):
    """Return the given percentile of the training scores, or None if no percentile is set."""
    if threshold_percentile is None:
        return None
    return float(np.percentile(train_scores, float(threshold_percentile)))


def compute_normal_only_rank_score(mean_fp, std_fp, target_fp=0.10):
    """Rank a tuning config by how close its false-positive rate on normal data is to the target.

    Lower is better: distance from the target rate plus half the spread across folds.
    """
    return abs(mean_fp - target_fp) + 0.5 * std_fp
