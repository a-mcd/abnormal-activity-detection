"""Event encoding, n-gram, transition-behaviour and repeat features built from
the Stage 5 event table, plus the per-fold trigram debug helpers.
"""

from typing import Optional
import numpy as np
import pandas as pd

from .tap_burner import build_stage5_event_table


# =========================================================
# EVENT ENCODING FEATURES
# =========================================================
def create_event_encoding_features(event_df: pd.DataFrame) -> dict:
    if event_df.empty:
        return {}

    out = event_df.copy()

    if "sensor" not in out.columns:
        return {}

    out["sensor"] = out["sensor"].astype(str).str.strip().str.upper()

    if "message" in out.columns:
        out["message"] = out["message"].astype(str).str.strip().str.upper()
    else:
        out["message"] = ""

    #out["sensor_type"] = out["sensor"].apply(get_sensor_type)
    out["sensor_type"] = out["sensor"]

    feats = {}

    for sensor, count in out["sensor"].value_counts().items():
        feats[f"sensor::{sensor}"] = int(count)

    for msg, count in out["message"].value_counts().items():
        feats[f"msg::{msg}"] = int(count)

    evt_counts = out.groupby(["sensor", "message"]).size()
    for (sensor, msg), count in evt_counts.items():
        feats[f"evt::{sensor}::{msg}"] = int(count)

    state_counts = out.groupby(["sensor", "message"]).size()
    for (sensor, msg), count in state_counts.items():
        feats[f"state::{sensor}::{msg}"] = int(count)

    for sensor_type, count in out["sensor_type"].value_counts().items():
        feats[f"type::{sensor_type}"] = int(count)

    sensors = out["sensor"].tolist()
    for a, b in zip(sensors[:-1], sensors[1:]):
        key = f"trans::{a}::{b}"
        feats[key] = feats.get(key, 0) + 1

    return feats


# =========================================================
# TRANSITION BEHAVIOUR FEATURES
# =========================================================
def _build_event_tokens(event_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if event_df.empty:
        return [], []

    out = event_df.copy()

    if "sensor" not in out.columns:
        out["sensor"] = ""
    if "message" not in out.columns:
        out["message"] = ""

    out["sensor"] = out["sensor"].astype(str).str.strip().str.upper()
    out["message"] = out["message"].astype(str).str.strip().str.upper()

    event_tokens = (out["sensor"] + "::" + out["message"]).tolist()
    sensor_tokens = out["sensor"].tolist()

    return event_tokens, sensor_tokens


def _make_ngrams(tokens: list[str], n: int) -> list[tuple]:
    if len(tokens) < n or n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

def trigram_debug_summary(
    event_df: pd.DataFrame,
    reference_stats: Optional[dict] = None,
) -> dict:
    """
    Build trigram debug information for one session.
    """
    event_tokens, _ = _build_event_tokens(event_df)
    session_trigrams = _make_ngrams(event_tokens, 3)
    trigram_vocab_ref = set((reference_stats or {}).get("trigram_vocab", set()))

    missing_trigrams = [tg for tg in session_trigrams if tg not in trigram_vocab_ref]

    return {
        "event_tokens": event_tokens,
        "session_trigrams": session_trigrams,
        "missing_trigrams": missing_trigrams,
        "unique_session_trigrams": sorted(set(session_trigrams)),
        "unique_missing_trigrams": sorted(set(missing_trigrams)),
        "n_session_trigrams": len(session_trigrams),
        "n_missing_trigrams": len(missing_trigrams),
        "n_unique_session_trigrams": len(set(session_trigrams)),
        "n_unique_missing_trigrams": len(set(missing_trigrams)),
    }


def write_fold_trigram_debug_file(
    output_fp,
    fold_idx: int,
    train_reference_stats: dict,
    val_session_dfs: list[pd.DataFrame],
    val_meta: list[dict],
) -> None:
    """
    Write a fold-level trigram debug report to a text file.
    Includes:
    - all train reference trigrams
    - all validation session trigrams
    - which validation trigrams were missing from train reference
    """
    trigram_vocab_ref = sorted(set(train_reference_stats.get("trigram_vocab", set())))

    with open(output_fp, "w") as f:
        f.write("=" * 100 + "\n")
        f.write(f"FOLD {fold_idx} TRIGRAM DEBUG\n")
        f.write("=" * 100 + "\n\n")

        f.write("TRAIN REFERENCE TRIGRAM VOCAB\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total reference trigrams: {len(trigram_vocab_ref)}\n\n")

        if trigram_vocab_ref:
            for i, tg in enumerate(trigram_vocab_ref, start=1):
                f.write(f"{i:03d}. {tg}\n")
        else:
            f.write("None\n")

        f.write("\n\n")
        f.write("=" * 100 + "\n")
        f.write("VALIDATION SESSION DETAILS\n")
        f.write("=" * 100 + "\n")

        for df, meta_row in zip(val_session_dfs, val_meta):
            filename = str(meta_row.get("filename", "unknown"))
            participant_id = str(meta_row.get("participant_id", "unknown"))
            dataset = str(meta_row.get("dataset", "unknown"))
            activity_code = str(meta_row.get("activity_code", "unknown"))
            activity_index = str(meta_row.get("activity_index", "unknown"))

            stage5_event_df = build_stage5_event_table(df, source_file=filename)
            dbg = trigram_debug_summary(
                stage5_event_df,
                reference_stats=train_reference_stats,
            )

            missing_trigram_set = set(dbg["missing_trigrams"])

            f.write("\n" + "-" * 100 + "\n")
            f.write(f"filename        : {filename}\n")
            f.write(f"participant_id  : {participant_id}\n")
            f.write(f"dataset         : {dataset}\n")
            f.write(f"activity_code   : {activity_code}\n")
            f.write(f"activity_index  : {activity_index}\n")
            f.write(f"n_session_trigrams        : {dbg['n_session_trigrams']}\n")
            f.write(f"n_missing_trigrams        : {dbg['n_missing_trigrams']}\n")
            f.write(f"n_unique_session_trigrams : {dbg['n_unique_session_trigrams']}\n")
            f.write(f"n_unique_missing_trigrams : {dbg['n_unique_missing_trigrams']}\n")

            f.write("\nEVENT TOKENS\n")
            f.write("-" * 100 + "\n")
            if dbg["event_tokens"]:
                for i, tok in enumerate(dbg["event_tokens"], start=1):
                    f.write(f"{i:03d}. {tok}\n")
            else:
                f.write("None\n")

            f.write("\nSESSION TRIGRAMS IN ORDER\n")
            f.write("-" * 100 + "\n")
            if dbg["session_trigrams"]:
                for i, tg in enumerate(dbg["session_trigrams"], start=1):
                    status = "MISSING" if tg in missing_trigram_set else "SEEN"
                    f.write(f"{i:03d}. {tg}    [{status}]\n")
            else:
                f.write("None\n")

            f.write("\nMISSING TRIGRAMS IN ORDER\n")
            f.write("-" * 100 + "\n")
            if dbg["missing_trigrams"]:
                for i, tg in enumerate(dbg["missing_trigrams"], start=1):
                    f.write(f"{i:03d}. {tg}\n")
            else:
                f.write("None\n")

            f.write("\nUNIQUE MISSING TRIGRAMS\n")
            f.write("-" * 100 + "\n")
            if dbg["unique_missing_trigrams"]:
                for i, tg in enumerate(dbg["unique_missing_trigrams"], start=1):
                    f.write(f"{i:03d}. {tg}\n")
            else:
                f.write("None\n")


def print_fold_trigram_debug(
    fold_idx: int,
    train_reference_stats: dict,
    val_session_dfs: list[pd.DataFrame],
    val_meta: list[dict],
) -> None:
    """
    Print trigram vocab learned from train-normal and the missing trigram info
    for each validation session in the fold.
    """
    trigram_vocab_ref = sorted(set(train_reference_stats.get("trigram_vocab", set())))

    print("\n" + "=" * 80)
    print(f"FOLD {fold_idx} TRAIN REFERENCE TRIGRAM VOCAB")
    print("=" * 80)
    print(f"Total reference trigrams: {len(trigram_vocab_ref)}")

    for i, tg in enumerate(trigram_vocab_ref, start=1):
        print(f"{i:03d}. {tg}")

    print("\n" + "=" * 80)
    print(f"FOLD {fold_idx} VALIDATION MISSING TRIGRAMS")
    print("=" * 80)

    for df, meta_row in zip(val_session_dfs, val_meta):
        filename = meta_row.get("filename", "unknown")
        participant_id = meta_row.get("participant_id", "unknown")
        dataset = meta_row.get("dataset", "unknown")

        stage5_event_df = build_stage5_event_table(df, source_file=str(filename))
        dbg = trigram_debug_summary(stage5_event_df, reference_stats=train_reference_stats)

        print("\n" + "-" * 80)
        print(f"Validation file: {filename}")
        print(f"Participant: {participant_id}")
        print(f"Dataset: {dataset}")
        print(f"Session trigram count: {dbg['n_session_trigrams']}")
        print(f"Missing trigram count: {dbg['n_missing_trigrams']}")
        print(f"Unique session trigrams: {dbg['n_unique_session_trigrams']}")
        print(f"Unique missing trigrams: {dbg['n_unique_missing_trigrams']}")

        print("\nAll session trigrams:")
        if dbg["session_trigrams"]:
            for j, tg in enumerate(dbg["session_trigrams"], start=1):
                status = "MISSING" if tg in dbg["missing_trigrams"] else "SEEN"
                print(f"  {j:03d}. {tg}   [{status}]")
        else:
            print("  None")

        print("\nUnique missing trigrams:")
        if dbg["unique_missing_trigrams"]:
            for j, tg in enumerate(dbg["unique_missing_trigrams"], start=1):
                print(f"  {j:03d}. {tg}")
        else:
            print("  None")

def create_transition_behaviour_features(event_df: pd.DataFrame) -> dict:
    default = {
        "SWITCH_RATE": np.nan,
        "REPEAT_RATIO": np.nan,
        "SAME_EVENT_REPEAT_RATIO": np.nan,
        "BIGRAM_MATCH_RATIO": np.nan,
        "TRIGRAM_MATCH_RATIO": np.nan,
        "SENSOR_BIGRAM_MATCH_RATIO": np.nan,
    }

    if event_df.empty:
        return default

    event_tokens, sensor_tokens = _build_event_tokens(event_df)

    num_events = len(event_tokens)
    if num_events == 0:
        return default

    if num_events >= 2:
        n_adjacent = num_events - 1

        sensor_switches = sum(
            1 for a, b in zip(sensor_tokens[:-1], sensor_tokens[1:]) if a != b
        )
        same_event_repeats = sum(
            1 for a, b in zip(event_tokens[:-1], event_tokens[1:]) if a == b
        )

        switch_rate = float(sensor_switches / n_adjacent)
        repeat_ratio = float(same_event_repeats / n_adjacent)
        same_event_repeat_ratio = float(same_event_repeats / n_adjacent)
    else:
        switch_rate = np.nan
        repeat_ratio = np.nan
        same_event_repeat_ratio = np.nan

    event_bigrams = _make_ngrams(event_tokens, 2)
    event_trigrams = _make_ngrams(event_tokens, 3)
    sensor_bigrams = _make_ngrams(sensor_tokens, 2)

    if len(event_bigrams) > 0:
        bigram_match_ratio = float(1.0 - (len(set(event_bigrams)) / len(event_bigrams)))
    else:
        bigram_match_ratio = np.nan

    if len(event_trigrams) > 0:
        trigram_match_ratio = float(1.0 - (len(set(event_trigrams)) / len(event_trigrams)))
    else:
        trigram_match_ratio = np.nan

    if len(sensor_bigrams) > 0:
        sensor_bigram_match_ratio = float(
            1.0 - (len(set(sensor_bigrams)) / len(sensor_bigrams))
        )
    else:
        sensor_bigram_match_ratio = np.nan

    return {
        "SWITCH_RATE": switch_rate,
        "REPEAT_RATIO": repeat_ratio,
        "SAME_EVENT_REPEAT_RATIO": same_event_repeat_ratio,
        "BIGRAM_MATCH_RATIO": bigram_match_ratio,
        "TRIGRAM_MATCH_RATIO": trigram_match_ratio,
        "SENSOR_BIGRAM_MATCH_RATIO": sensor_bigram_match_ratio,
    }


# =========================================================
# STAGE 5 EVENT-SEQUENCE FEATURES
# =========================================================
def create_stage5_repeat_features(event_df: pd.DataFrame) -> dict:
    """
    Compute repeat-based features from the final Stage 5 event table.

    consecutive_repeat_frac:
        Share of adjacent event pairs where the current row is an exact
        duplicate of the previous row in terms of (sensor, message).

    Notes:
    - Computed on the Stage 5 event table only.
    - Since Stage 5 excludes raw AD1 rows and only includes cleaned AD1-B/C
      symbolic events, this avoids raw analogue flooding.
    """
    default = {
        "CONSECUTIVE_REPEAT_FRAC": 0.0,
    }

    if event_df is None or event_df.empty:
        return default

    out = event_df.copy()

    if "sensor" not in out.columns:
        out["sensor"] = ""
    if "message" not in out.columns:
        out["message"] = ""

    out["sensor"] = out["sensor"].astype(str).str.strip().str.upper()
    out["message"] = out["message"].astype(str).str.strip().str.upper()

    n_events = len(out)
    if n_events <= 1:
        return default

    same_as_prev = (
        out["sensor"].eq(out["sensor"].shift(1)) &
        out["message"].eq(out["message"].shift(1))
    )

    n_repeat_consecutive = int(same_as_prev.iloc[1:].sum())
    consecutive_repeat_frac = float(n_repeat_consecutive / (n_events - 1))

    return {
        "CONSECUTIVE_REPEAT_FRAC": consecutive_repeat_frac,
    }
