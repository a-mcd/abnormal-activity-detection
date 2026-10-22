import os
import re
from pathlib import Path
import pandas as pd
VALID_MESSAGES = {"ON", "OFF", "START", "END", "OPEN", "CLOSE", "PRESENT", "ABSENT", "UP", "DOWN"}
ACTIVITY_INDEX = {"t1": 0, "t2": 1, "t3": 2, "t4": 3, "t5": 4}

# -------------------------
# Filename helpers
# -------------------------

def extract_activity_code(filename: str) -> str | None:
    m = re.search(r"\.(t[1-5])\.csv$", filename.lower())
    return m.group(1) if m else None


def extract_participant_id(filename: str) -> str | None:
    m = re.search(r"^(p\d+)\.", filename.lower())
    return m.group(1) if m else None


# -------------------------
# Normalization / typing
# -------------------------
def norm_sensor(s: str) -> str:
    return str(s).strip().lower()


def norm_msg(m: str) -> str:
    return str(m).strip().upper()



def clean_sensor_name(sensor: str) -> str:
    """
    Keeps base AD1 sensor only:
      ad1_c__0_426725 -> ad1_c
    """
    return str(sensor).split("__")[0].strip().lower()


def get_sensor_type(sensor: str) -> str:
    s = clean_sensor_name(sensor)
    if s.startswith("m"):
        return "motion"
    if s.startswith("i"):
        return "item"
    if s.startswith("d"):
        return "door"
    if s == "asterisk":
        return "asterisk"
    if s.startswith("ad1"):
        return "analogue"
    return "unknown"


def is_known_message(sensor: str, msg: str) -> bool:
    sensor_type = get_sensor_type(sensor)
    msg = norm_msg(msg)

    if sensor_type == "analogue":
        try:
            float(msg)
            return True
        except Exception:
            return False

    if sensor_type in ("motion", "item", "door", "asterisk"):
        return msg in VALID_MESSAGES

    return False


# -------------------------
# Load / clean single CSV
# -------------------------
def process_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    df["sensor"] = df["sensor"].map(norm_sensor)
    df["sensor"] = df["sensor"].map(clean_sensor_name)
    df["message"] = df["message"].map(norm_msg)

    bad_mask = ~df.apply(
        lambda r: get_sensor_type(r["sensor"]) != "unknown"
        and is_known_message(r["sensor"], r["message"]),
        axis=1,
    )
    df = df[~bad_mask].copy()

    d = pd.to_datetime(df["date"], errors="coerce")
    t = pd.to_datetime(df["time"], format="%H:%M:%S.%f", errors="coerce")
    t2 = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
    t = t.fillna(t2)

    df["timestamp"] = d + (t - t.dt.normalize())
    df = df.dropna(subset=["timestamp"]).copy()

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["dt"] = df["timestamp"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)

    return df




# -------------------------
# Load sessions
# -------------------------
def load_sessions_for_activity(directory: Path, target_activity: str, dataset_label: str):
    session_dfs, meta = [], []

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".csv"):
            continue
        
        activity_code = extract_activity_code(filename)
        if target_activity != "t-all" and activity_code != target_activity:
            continue

        pid = extract_participant_id(filename)
        if pid is None:
            continue

        fp = os.path.join(directory, filename)
        df = process_csv(fp)

        session_dfs.append(df)
        meta.append(
            {
                "filename": filename,
                "participant_id": pid,
                "activity_code": activity_code,
                "activity_index": ACTIVITY_INDEX[activity_code],
                "dataset": dataset_label,
            }
        )

    return session_dfs, meta


def load_data(target_activity_code, normal_dir, error_dir, n_splits):
    normal_sessions, normal_meta = load_sessions_for_activity(
        directory=normal_dir,
        target_activity=target_activity_code,
        dataset_label="normal",
    )
    error_sessions, error_meta = load_sessions_for_activity(
        directory=error_dir,
        target_activity=target_activity_code,
        dataset_label="error",
    )

    if len(normal_sessions) == 0:
        raise RuntimeError(
            f"No normal sessions loaded for {target_activity_code} from {normal_dir}"
        )
    
    if len(error_sessions) == 0:
        raise RuntimeError(
            f"No error sessions loaded for {target_activity_code} from {error_dir}"
        )

    normal_participants = sorted({m["participant_id"] for m in normal_meta})
    splitter_name = "StratifiedGroupKFold" if target_activity_code == "t-all" else "GroupKFold"

    if len(normal_participants) < n_splits:
        raise RuntimeError(
            f"Need at least {n_splits} unique normal participants for {splitter_name}, "
            f"but found {len(normal_participants)}"
        )

    return normal_sessions, normal_meta, error_sessions, error_meta