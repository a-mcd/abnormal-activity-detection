"""Normal-reference vocabularies and the features that compare a session's
events and n-grams against them.
"""

from typing import Optional
import numpy as np
import pandas as pd

from .tap_burner import build_stage5_event_table
from .transitions import _build_event_tokens, _make_ngrams


# =========================================================
# REFERENCE / NORMAL-VOCAB FEATURES
# =========================================================
def create_reference_sequence_features(
    event_df: pd.DataFrame,
    reference_stats: Optional[dict] = None,
) -> dict:
    default = {
        "EXTRA_BIGRAM_RATIO": np.nan,
        "EXTRA_EVENT_RATIO": np.nan,
        "EXTRA_TRIGRAM_RATIO": np.nan,
        "MISSING_BIGRAM_RATIO": np.nan,
        "MISSING_EVENT_RATIO": np.nan,
        "TRANSITION_LOG_LIK": np.nan,
        "MEAN_TRANSITION_LOG_LIK": np.nan,
    }

    if event_df.empty:
        return default

    event_tokens, _ = _build_event_tokens(event_df)
    if len(event_tokens) == 0:
        return default

    event_vocab_ref = set((reference_stats or {}).get("event_vocab", set()))
    bigram_vocab_ref = set((reference_stats or {}).get("bigram_vocab", set()))
    trigram_vocab_ref = set((reference_stats or {}).get("trigram_vocab", set()))
    transition_prob = dict((reference_stats or {}).get("transition_prob", {}))

    session_event_set = set(event_tokens)
    session_bigrams = _make_ngrams(event_tokens, 2)
    session_trigrams = _make_ngrams(event_tokens, 3)

    if len(event_tokens) > 0 and len(event_vocab_ref) > 0:
        unseen_events = sum(1 for tok in event_tokens if tok not in event_vocab_ref)
        extra_event_ratio = float(unseen_events / len(event_tokens))
    else:
        extra_event_ratio = np.nan

    if len(session_bigrams) > 0 and len(bigram_vocab_ref) > 0:
        unseen_bigrams = sum(1 for bg in session_bigrams if bg not in bigram_vocab_ref)
        extra_bigram_ratio = float(unseen_bigrams / len(session_bigrams))
    else:
        extra_bigram_ratio = np.nan

    if len(session_trigrams) > 0 and len(trigram_vocab_ref) > 0:
        extra_trigrams = sum(1 for tg in session_trigrams if tg not in trigram_vocab_ref)
        extra_trigram_ratio = float(extra_trigrams / len(session_trigrams))
    else:
        extra_trigram_ratio = np.nan

    if len(event_vocab_ref) > 0:
        missing_events = len(event_vocab_ref - session_event_set)
        missing_event_ratio = float(missing_events / len(event_vocab_ref))
    else:
        missing_event_ratio = np.nan

    if len(bigram_vocab_ref) > 0:
        session_bigram_set = set(session_bigrams)
        missing_bigrams = len(bigram_vocab_ref - session_bigram_set)
        missing_bigram_ratio = float(missing_bigrams / len(bigram_vocab_ref))
    else:
        missing_bigram_ratio = np.nan

    # ---------------------------------------------------------
    # Transition likelihood features
    # ---------------------------------------------------------
    unseen_log_prob = -20.0

    if len(session_bigrams) > 0 and len(transition_prob) > 0:
        log_probs = [
            float(np.log(transition_prob[bg])) if bg in transition_prob else unseen_log_prob
            for bg in session_bigrams
        ]
        transition_log_lik = float(np.sum(log_probs))
        mean_transition_log_lik = float(np.mean(log_probs))
    else:
        transition_log_lik = np.nan
        mean_transition_log_lik = np.nan

    return {
        "EXTRA_BIGRAM_RATIO": extra_bigram_ratio,
        "EXTRA_EVENT_RATIO": extra_event_ratio,
        "EXTRA_TRIGRAM_RATIO": extra_trigram_ratio,
        "MISSING_BIGRAM_RATIO": missing_bigram_ratio,
        "MISSING_EVENT_RATIO": missing_event_ratio,
        "TRANSITION_LOG_LIK": transition_log_lik,
        "MEAN_TRANSITION_LOG_LIK": mean_transition_log_lik,
    }


def get_reference_stats_for_activity(reference_stats: dict, activity_code: str):
    """
    Accept either:
    1) one pooled reference-stats dict, or
    2) a dict of {activity_code: reference_stats_dict}

    Returns the correct reference stats for the given activity.
    """
    if not reference_stats:
        return {}

    # Single pooled reference stats dict
    if "event_vocab" in reference_stats:
        return reference_stats

    # Per-activity reference stats dict
    activity_code = str(activity_code).strip()
    return reference_stats.get(activity_code, {})


def build_reference_stats(normal_session_dfs: list[pd.DataFrame]) -> dict:
    event_vocab = set()
    bigram_vocab = set()
    trigram_vocab = set()

    # For transition likelihood features
    transition_counts = {}
    total_bigrams = 0

    for i, df in enumerate(normal_session_dfs):
        stage5_event_df = build_stage5_event_table(df, source_file=f"normal_ref_{i}")
        event_tokens, _ = _build_event_tokens(stage5_event_df)

        if len(event_tokens) == 0:
            continue

        event_vocab.update(event_tokens)

        session_bigrams = _make_ngrams(event_tokens, 2)
        session_trigrams = _make_ngrams(event_tokens, 3)

        bigram_vocab.update(session_bigrams)
        trigram_vocab.update(session_trigrams)

        for bg in session_bigrams:
            transition_counts[bg] = transition_counts.get(bg, 0) + 1
            total_bigrams += 1

    if total_bigrams > 0:
        transition_prob = {
            bg: count / total_bigrams
            for bg, count in transition_counts.items()
        }
    else:
        transition_prob = {}

    return {
        "event_vocab": event_vocab,
        "bigram_vocab": bigram_vocab,
        "trigram_vocab": trigram_vocab,
        "transition_prob": transition_prob,
    }


def build_reference_stats_by_activity(
    session_dfs: list[pd.DataFrame],
    meta: list[dict],
) -> dict[str, dict]:
    """
    Build reference stats separately for each activity.

    Returns
    -------
    dict
        {
            "t1": {...reference stats...},
            "t2": {...reference stats...},
            ...
        }
    """
    grouped_session_dfs: dict[str, list[pd.DataFrame]] = {}

    for df, m in zip(session_dfs, meta):
        activity_code = str(m.get("activity_code", "")).strip()
        if not activity_code:
            continue
        grouped_session_dfs.setdefault(activity_code, []).append(df)

    reference_stats_by_activity = {}
    for activity_code, activity_session_dfs in grouped_session_dfs.items():
        reference_stats_by_activity[activity_code] = build_reference_stats(
            activity_session_dfs
        )

    return reference_stats_by_activity
