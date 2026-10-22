#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path

from abnormal_activity_detection.paths import get_dataset_dirs, get_results_dir
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow.keras import layers
from abnormal_activity_detection.activity.sessions import norm_sensor, norm_msg, get_sensor_type, is_known_message, build_timestamp, duration_bucket_label
from abnormal_activity_detection.data import extract_activity_code, extract_participant_id

# -------------------------
# Config
# -------------------------
SEED = 42
# Seeds Python, NumPy and TensorFlow; op determinism stops TensorFlow's
# multi-threaded maths from giving slightly different results on each run.
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

ACTIVITY_INDEX = {"t1": 0, "t2": 1, "t3": 2, "t4": 3, "t5": 4}
INDEX_ACTIVITY = {v: k for k, v in ACTIVITY_INDEX.items()}
N_CLASSES = 5


NORMAL_DIR, ERROR_DIR = get_dataset_dirs()
OUTPUT_DIR = get_results_dir("activity_classification", "lstm")
MODEL_PATH = OUTPUT_DIR / "lstm_activity_classifier.keras"
VOCAB_PATH = OUTPUT_DIR / "lstm_activity_vocab.json"

ERROR_PRED_CSV = OUTPUT_DIR / "adl_error_lstm_predictions.csv"
ERROR_MISCLF_CSV = OUTPUT_DIR / "adl_error_misclassified.csv"

CV_CM_CSV = OUTPUT_DIR / "cv_confusion_matrix.csv"
ERROR_CM_CSV = OUTPUT_DIR / "adl_error_confusion_matrix.csv"

CV_PRED_CSV = OUTPUT_DIR / "cv_predictions_detailed.csv"
CV_MISCLF_CSV = OUTPUT_DIR / "cv_misclassified.csv"

MAX_LEN = 256
EMB_DIM = 64
LSTM_UNITS = 64
DROPOUT = 0.2
EPOCHS = 40
BATCH_SIZE = 16
LR = 5e-4
PATIENCE = 5
VOCAB_MIN_COUNT = 1

AD1A_MIN_DURATION_SECONDS = 120

FILENAME_RE = re.compile(r"^(p\d+)\.(t[1-5])\.csv$", re.IGNORECASE)


# -------------------------
# Helpers: parsing & cleaning
# -------------------------


def remove_short_ad1a_rows(df: pd.DataFrame, min_duration_seconds: int = 120) -> pd.DataFrame:
    """
    If ad1-a exists, calculate the duration between the first and last ad1-a timestamp.
    If that duration is < min_duration_seconds, remove all ad1-a rows.
    """
    ad1a_mask = df["sensor"] == "ad1-a"
    if not ad1a_mask.any():
        return df

    ad1a_times = df.loc[ad1a_mask, "timestamp"].dropna().sort_values()
    if len(ad1a_times) == 0:
        return df

    duration_seconds = (ad1a_times.iloc[-1] - ad1a_times.iloc[0]).total_seconds()

    if duration_seconds < min_duration_seconds:
        df = df.loc[~ad1a_mask].copy()

    return df


# -------------------------
# Confusion matrix helpers
# -------------------------
def save_confusion_matrix_csv(y_true, y_pred, out_path: Path, labels=None):
    if labels is None:
        labels = list(range(N_CLASSES))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    label_names = [INDEX_ACTIVITY[i] for i in labels]

    df_cm = pd.DataFrame(
        cm,
        index=[f"actual_{x}" for x in label_names],
        columns=[f"pred_{x}" for x in label_names]
    )
    df_cm.to_csv(out_path)
    print(f"Saved confusion matrix CSV: {out_path}")
    return cm


def print_confusion_matrix(cm, title="Confusion Matrix"):
    print(f"\n{title} (Rows=True, Cols=Pred):")
    header = "      " + "  ".join([f"{INDEX_ACTIVITY[i]:>3}" for i in range(N_CLASSES)])
    print(header)
    for i in range(N_CLASSES):
        row_str = "  ".join([f"{cm[i, j]:>3}" for j in range(N_CLASSES)])
        print(f"{INDEX_ACTIVITY[i]:>3}  {row_str}")


# -------------------------
# Tokenization
# -------------------------
def analogue_bin_from_str(sensor: str, m: str) -> str:
    if m is None or str(m).strip() == "":
        return None

    try:
        v = float(str(m).strip())
    except Exception:
        return None

    s = norm_sensor(sensor)

    if s.startswith("ad1-a"):
        edges = (2.85, 2.95, 3.05, 3.15, 3.25, 3.35)
    else:
        return None

    for i, e in enumerate(edges):
        if v < e:
            return f"b{i}"

    return f"b{len(edges)}"


def csv_to_event_tokens(csv_path: Path) -> list[str]:
    """
    Converts a CASAS session CSV into a sequence of tokens.

    Rules:
    - Keeps order by timestamp
    - Filters unknown/bad rows
    - If ad1-a duration is under 120s, removes all ad1-a rows
    - ad1-a is represented as event-level analogue-bin tokens
    - ad1-b and ad1-c are represented once each as duration-bucket tokens
    """
    df = pd.read_csv(csv_path)

    if not {"date", "time", "sensor", "message"}.issubset(df.columns):
        return []

    df["sensor"] = df["sensor"].map(norm_sensor)
    df["message"] = df["message"].map(norm_msg)

    good = df.apply(
        lambda r: get_sensor_type(r["sensor"]) != "unknown"
        and is_known_message(r["sensor"], r["message"]),
        axis=1
    )
    df = df[good].copy()
    if len(df) == 0:
        return []

    df["timestamp"] = build_timestamp(df)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) == 0:
        return []

    df = remove_short_ad1a_rows(df, min_duration_seconds=AD1A_MIN_DURATION_SECONDS)
    if len(df) == 0:
        return []

    tokens = []

    # Add one duration-bucket token for ad1-b and ad1-c
    for sensor_name in ["ad1-b", "ad1-c"]:
        sensor_df = df[df["sensor"] == sensor_name].copy()
        if len(sensor_df) > 0:
            duration_s = (sensor_df["timestamp"].max() - sensor_df["timestamp"].min()).total_seconds()
            bucket = duration_bucket_label(duration_s)
            tokens.append(f"{sensor_name}::{bucket}")

    # Add event-level tokens for everything else
    for s, m in zip(df["sensor"].tolist(), df["message"].tolist()):
        st = get_sensor_type(s)

        # ad1-b and ad1-c are now handled only through duration buckets
        if s in {"ad1-b", "ad1-c"}:
            continue

        if st == "analogue":
            b = analogue_bin_from_str(s, m)
            if b is not None:
                tokens.append(f"{s}::ANALOGUE::{b}")
        else:
            tokens.append(f"{s}::{m}")

    return tokens


def seed_ad1_vocab(vocab: dict[str, int]) -> dict[str, int]:
    burner_bins = [f"b{i}" for i in range(0, 6 + 1)]
    duration_bins = [
        "DUR_0_60",
        "DUR_61_120",
        "DUR_121_200",
        "DUR_201_320",
        "DUR_321_PLUS",
    ]

    desired = []

    for b in burner_bins:
        desired.append(f"ad1-a::ANALOGUE::{b}")

    for s in ["ad1-b", "ad1-c"]:
        for d in duration_bins:
            desired.append(f"{s}::{d}")

    for tok in desired:
        if tok not in vocab:
            vocab[tok] = len(vocab)

    return vocab


def build_vocab_from_dir(data_dir: Path, min_count: int = 1) -> dict[str, int]:
    counter = Counter()
    for fn in sorted(os.listdir(data_dir)):
        if not fn.endswith(".csv"):
            continue
        counter.update(csv_to_event_tokens(data_dir / fn))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    vocab = seed_ad1_vocab(vocab)

    for tok, c in counter.most_common():
        if c >= min_count and tok not in vocab:
            vocab[tok] = len(vocab)

    return vocab


def tokens_to_ids(tokens: list[str], vocab: dict[str, int], max_len: int) -> np.ndarray:
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))
    return np.array(ids, dtype=np.int32)


# -------------------------
# Dataset loading
# -------------------------
def load_dataset_sequences(data_dir: Path, vocab: dict[str, int], max_len: int):
    X, y, groups, meta = [], [], [], []

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".csv"):
            continue

        m = FILENAME_RE.match(filename)
        if not m:
            continue

        participant = extract_participant_id(filename)
        activity_code = extract_activity_code(filename)
        if participant is None or activity_code is None:
            continue

        tokens = csv_to_event_tokens(data_dir / filename)
        if len(tokens) == 0:
            continue

        X.append(tokens_to_ids(tokens, vocab, max_len))
        y.append(ACTIVITY_INDEX[activity_code])
        groups.append(participant)
        meta.append({
            "filename": filename,
            "participant": participant,
            "activity": activity_code,
            "orig_len": len(tokens)
        })

    return np.stack(X), np.array(y, dtype=np.int32), np.array(groups), meta


# -------------------------
# Export engineered features to CSV
# -------------------------
def export_sequence_features_csv(X, y, groups, meta, filename: Path, base_dir: str, fold=None, split=None):
    df_meta = pd.DataFrame(meta)
    df_meta["participant"] = groups
    df_meta["activity_index"] = y
    df_meta["activity_label"] = [INDEX_ACTIVITY[int(i)] for i in y]

    if fold is not None:
        df_meta["fold"] = int(fold)
    if split is not None:
        df_meta["dataset_split"] = split

    seq_cols = [f"id_{i}" for i in range(X.shape[1])]
    df_seq = pd.DataFrame(X, columns=seq_cols)

    df = pd.concat([df_meta, df_seq], axis=1)
    out_path = os.path.join(base_dir, filename)
    df.to_csv(out_path, index=False)
    print(f"Exported engineered features: {out_path}  (rows={len(df)}, cols={df.shape[1]})")


# -------------------------
# Model
# -------------------------
def make_model(vocab_size: int, max_len: int) -> tf.keras.Model:
    inp = layers.Input(shape=(max_len,), dtype="int32")
    x = layers.Embedding(vocab_size, EMB_DIM, mask_zero=True)(inp)
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, dropout=DROPOUT, recurrent_dropout=DROPOUT)
    )(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(N_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -------------------------
# Train + CV
# -------------------------
def run_groupkfold_cv(X, y, groups, meta, vocab_size: int):
    gkf = GroupKFold(n_splits=5)

    fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []
    all_true, all_pred = [], []
    cv_prediction_rows = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        export_sequence_features_csv(
            X[tr], y[tr], groups[tr], [meta[i] for i in tr],
            f"features_fold{fold}_train.csv",
            OUTPUT_DIR, fold=fold, split="train"
        )

        export_sequence_features_csv(
            X[te], y[te], groups[te], [meta[i] for i in te],
            f"features_fold{fold}_test.csv",
            OUTPUT_DIR, fold=fold, split="test"
        )

        X_train_full, y_train_full = X[tr], y[tr]
        groups_train_full = groups[tr]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        inner_tr_idx, val_idx = next(gss.split(X_train_full, y_train_full, groups_train_full))

        X_train, y_train = X_train_full[inner_tr_idx], y_train_full[inner_tr_idx]
        X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

        class_weight = {
            0: 1.0,
            1: 1.0,
            2: 1.0,
            3: 1.0,
            4: 1.0,
        }

        model = make_model(vocab_size, X.shape[1])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True
            )
        ]
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=callbacks,
            class_weight=class_weight
        )

        probs = model.predict(X[te], verbose=0)
        y_hat = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)

        acc = accuracy_score(y[te], y_hat)
        prec = precision_score(y[te], y_hat, average="macro", zero_division=0)
        rec = recall_score(y[te], y_hat, average="macro", zero_division=0)
        f1 = f1_score(y[te], y_hat, average="macro", zero_division=0)

        fold_acc.append(acc)
        fold_prec.append(prec)
        fold_rec.append(rec)
        fold_f1.append(f1)
        all_true.extend(y[te].tolist())
        all_pred.extend(y_hat.tolist())

        for local_idx, sample_idx in enumerate(te):
            cv_prediction_rows.append({
                "fold": fold,
                "filename": meta[sample_idx]["filename"],
                "participant": meta[sample_idx]["participant"],
                "true_label": INDEX_ACTIVITY[int(y[sample_idx])],
                "pred_label": INDEX_ACTIVITY[int(y_hat[local_idx])],
                "true_index": int(y[sample_idx]),
                "pred_index": int(y_hat[local_idx]),
                "confidence": float(conf[local_idx]),
                "correct": bool(y[sample_idx] == y_hat[local_idx]),
                "orig_len": int(meta[sample_idx].get("orig_len", 0)),
            })

        test_people = sorted(set(groups[te].tolist()))
        print("\n" + "=" * 70)
        print(f"Fold {fold}/5 | Test people: {test_people}")
        print(
            f"Accuracy: {acc:.4f} | "
            f"Macro Precision: {prec:.4f} | "
            f"Macro Recall: {rec:.4f} | "
            f"Macro F1: {f1:.4f}"
        )
        print(classification_report(
            y[te], y_hat,
            target_names=[INDEX_ACTIVITY[i] for i in range(5)],
            digits=4,
            zero_division=0
        ))

        cm_fold = confusion_matrix(y[te], y_hat, labels=list(range(N_CLASSES)))
        print_confusion_matrix(cm_fold, title=f"Fold {fold} Confusion Matrix")

        fold_misclf = [r for r in cv_prediction_rows if r["fold"] == fold and not r["correct"]]
        if fold_misclf:
            print("\nMisclassified files in this fold:")
            for r in fold_misclf:
                print(
                    f"  {r['filename']} | actual={r['true_label']} | "
                    f"pred={r['pred_label']} | conf={r['confidence']:.4f}"
                )
        else:
            print("\nNo misclassified files in this fold.")

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    overall_acc = accuracy_score(all_true, all_pred)
    overall_prec = precision_score(all_true, all_pred, average="macro", zero_division=0)
    overall_rec = recall_score(all_true, all_pred, average="macro", zero_division=0)
    overall_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)

    print("\n" + "#" * 70)
    print("Overall (pooled across folds)")
    print(f"Mean fold Accuracy:        {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
    print(f"Mean fold Macro Precision: {np.mean(fold_prec):.4f} ± {np.std(fold_prec):.4f}")
    print(f"Mean fold Macro Recall:    {np.mean(fold_rec):.4f} ± {np.std(fold_rec):.4f}")
    print(f"Mean fold Macro F1:        {np.mean(fold_f1):.4f} ± {np.std(fold_f1):.4f}")
    print(f"Overall Accuracy:          {overall_acc:.4f}")
    print(f"Overall Macro Precision:   {overall_prec:.4f}")
    print(f"Overall Macro Recall:      {overall_rec:.4f}")
    print(f"Overall Macro F1:          {overall_f1:.4f}")

    cm = save_confusion_matrix_csv(all_true, all_pred, CV_CM_CSV, labels=list(range(N_CLASSES)))
    print_confusion_matrix(cm, title="Overall CV Confusion Matrix")

    df_cv = pd.DataFrame(cv_prediction_rows).sort_values(
        ["fold", "correct", "filename"], ascending=[True, True, True]
    )
    df_cv.to_csv(CV_PRED_CSV, index=False)
    print(f"Saved CV predictions: {CV_PRED_CSV}")

    df_cv_mis = df_cv[df_cv["correct"] == False].copy()
    df_cv_mis.to_csv(CV_MISCLF_CSV, index=False)
    print(f"Saved CV misclassified files: {CV_MISCLF_CSV}")

    if len(df_cv_mis) > 0:
        print("\nAll CV misclassified files:")
        for _, r in df_cv_mis.iterrows():
            print(
                f"  fold={r['fold']} | {r['filename']} | participant={r['participant']} | "
                f"actual={r['true_label']} | pred={r['pred_label']} | conf={r['confidence']:.4f}"
            )

    return overall_acc, overall_prec, overall_rec, overall_f1


def train_final_and_save(X, y, vocab):
    model = make_model(len(vocab), X.shape[1])

    model.fit(
        X, y,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True
            )
        ]
    )

    model.save(MODEL_PATH)
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f, indent=2)

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved vocab to: {VOCAB_PATH}")


# -------------------------
# Inference on Error Data
# -------------------------
def evaluate_on_error_data():
    if not (MODEL_PATH.exists() and VOCAB_PATH.exists()):
        print("Model or Vocab missing. Train the model first.")
        return None, None, None, None

    with open(VOCAB_PATH, "r") as f:
        vocab = json.load(f)

    model = tf.keras.models.load_model(MODEL_PATH)

    true_labels, pred_labels = [], []

    print(f"\n--- Running Inference on Error Folder: {ERROR_DIR.name} ---")

    rows = []
    seq_rows = []
    pred_rows = []

    for filename in sorted(os.listdir(ERROR_DIR)):
        if not filename.endswith(".csv"):
            continue

        activity_code = extract_activity_code(filename)
        if not activity_code:
            continue

        participant = extract_participant_id(filename)
        tokens = csv_to_event_tokens(ERROR_DIR / filename)
        if not tokens:
            continue

        ids = tokens_to_ids(tokens, vocab, MAX_LEN)
        input_tensor = np.expand_dims(ids, axis=0)

        probs = model.predict(input_tensor, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(np.max(probs))

        true_idx = ACTIVITY_INDEX[activity_code]

        true_labels.append(true_idx)
        pred_labels.append(idx)

        pred_rows.append({
            "filename": filename,
            "participant": participant,
            "true_label": activity_code,
            "pred_label": INDEX_ACTIVITY[idx],
            "true_index": true_idx,
            "pred_index": idx,
            "confidence": conf,
            "correct": bool(true_idx == idx),
            "orig_len": len(tokens),
        })

        rows.append({
            "filename": filename,
            "true_label": activity_code,
            "true_index": true_idx,
            "orig_len": len(tokens),
            "dataset_split": "adl_error_test",
            "source_dir": str(ERROR_DIR),
        })
        seq_rows.append(ids)

    df_preds = pd.DataFrame(pred_rows).sort_values(
        ["correct", "filename"], ascending=[True, True]
    )
    df_preds.to_csv(ERROR_PRED_CSV, index=False)
    print(f"Saved error predictions: {ERROR_PRED_CSV}")

    df_error_mis = df_preds[df_preds["correct"] == False].copy()
    df_error_mis.to_csv(ERROR_MISCLF_CSV, index=False)
    print(f"Saved error misclassified files: {ERROR_MISCLF_CSV}")

    y_true, y_pred = np.array(true_labels), np.array(pred_labels)
    error_acc = accuracy_score(y_true, y_pred)
    error_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    error_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    error_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("\n" + "#" * 30 + "\nERROR DATASET RESULTS\n" + "#" * 30)
    print(f"Total files: {len(y_true)}")
    print(f"Overall Accuracy:        {error_acc:.4f}")
    print(f"Macro Precision:         {error_prec:.4f}")
    print(f"Macro Recall:            {error_rec:.4f}")
    print(f"Macro F1-Score:          {error_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=[INDEX_ACTIVITY[i] for i in range(N_CLASSES)],
        zero_division=0
    ))

    cm = save_confusion_matrix_csv(y_true, y_pred, ERROR_CM_CSV, labels=list(range(N_CLASSES)))
    print_confusion_matrix(cm, title="Error Dataset Confusion Matrix")

    if len(df_error_mis) > 0:
        print("\nMisclassified error files:")
        for _, r in df_error_mis.iterrows():
            print(
                f"  {r['filename']} | participant={r['participant']} | "
                f"actual={r['true_label']} | pred={r['pred_label']} | conf={r['confidence']:.4f}"
            )
    else:
        print("\nNo misclassified files in adl_error.")

    df_meta = pd.DataFrame(rows)
    df_seq = pd.DataFrame(np.stack(seq_rows), columns=[f"id_{i}" for i in range(MAX_LEN)])
    df = pd.concat([df_meta, df_seq], axis=1)
    out_path = OUTPUT_DIR / "adl_error_engineered_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Exported engineered error-test features: {out_path} (rows={len(df)})")

    return error_acc, error_prec, error_rec, error_f1


def inspect_sequence_lengths(data_dir: Path):
    lengths = []

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".csv"):
            continue

        tokens = csv_to_event_tokens(data_dir / filename)
        if len(tokens) > 0:
            lengths.append(len(tokens))

    if len(lengths) == 0:
        print(f"\nNo valid sequences found in: {data_dir}")
        return np.array([])

    lengths = np.array(lengths)

    print("\nSequence Length Statistics")
    print("---------------------------")
    print(f"Total sessions: {len(lengths)}")
    print(f"Min length: {lengths.min()}")
    print(f"Max length: {lengths.max()}")
    print(f"Mean length: {lengths.mean():.1f}")
    print(f"Median length: {np.median(lengths):.1f}")

    print("\nPercentiles")
    for p in [50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(lengths, p):.0f}")

    return lengths


def main():
    if not NORMAL_DIR.exists():
        raise FileNotFoundError(f"NORMAL_DIR not found: {NORMAL_DIR}")

    if not ERROR_DIR.exists():
        raise FileNotFoundError(f"ERROR_DIR not found: {ERROR_DIR}")

    inspect_sequence_lengths(NORMAL_DIR)
    inspect_sequence_lengths(ERROR_DIR)

    vocab = build_vocab_from_dir(NORMAL_DIR, VOCAB_MIN_COUNT)
    print(f"Vocab size (incl PAD/UNK): {len(vocab)}")

    X, y, groups, meta = load_dataset_sequences(NORMAL_DIR, vocab, MAX_LEN)
    print(f"Loaded sessions: {len(y)} | Seq len: {X.shape[1]}")
    print(f"People (groups): {len(set(groups))}")
    print("Activity counts:", {INDEX_ACTIVITY[i]: int(np.sum(y == i)) for i in range(5)})

    normal_acc, normal_prec, normal_rec, normal_f1 = run_groupkfold_cv(
        X, y, groups, meta, vocab_size=len(vocab)
    )

    export_sequence_features_csv(
        X, y, groups, meta,
        "features_final_train.csv",
        OUTPUT_DIR, fold=None, split="final_train_all_normal"
    )
    train_final_and_save(X, y, vocab)

    error_acc, error_prec, error_rec, error_f1 = evaluate_on_error_data()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(
        f"Normal dataset (GroupKFold CV) -> "
        f"Accuracy: {normal_acc:.4f} | "
        f"Macro Precision: {normal_prec:.4f} | "
        f"Macro Recall: {normal_rec:.4f} | "
        f"Macro F1: {normal_f1:.4f}"
    )
    if error_acc is not None and error_f1 is not None:
        print(
            f"Error dataset                 -> "
            f"Accuracy: {error_acc:.4f} | "
            f"Macro Precision: {error_prec:.4f} | "
            f"Macro Recall: {error_rec:.4f} | "
            f"Macro F1: {error_f1:.4f}"
        )


if __name__ == "__main__":
    main()