import os
import pandas as pd

from abnormal_activity_detection.paths import RAW_DATA_DIR

# ----------------------------
# CONFIG
# ----------------------------
base_dir = RAW_DATA_DIR
dirs = ["adl_error", "adl_noerror"]

# ----------------------------
# COLLECT UNIQUE SENSORS
# ----------------------------
motion_sensors = set()
item_sensors = set()

for d in dirs:
    dir_path = os.path.join(base_dir, d)
    
    for file in os.listdir(dir_path):
        if file.endswith(".csv"):
            fp = os.path.join(dir_path, file)
            
            try:
                df = pd.read_csv(fp)
                df.columns = df.columns.str.lower()
                
                if "sensor" not in df.columns:
                    continue
                
                sensors = df["sensor"].dropna().astype(str).str.upper()
                
                for s in sensors:
                    if s.startswith("M"):
                        motion_sensors.add(s)
                    elif s.startswith("I"):
                        item_sensors.add(s)
            
            except Exception as e:
                print(f"Error reading {fp}: {e}")

# ----------------------------
# RESULTS
# ----------------------------
print(f"Unique Motion Sensors: {len(motion_sensors)}")
print(sorted(motion_sensors))

print(f"\nUnique Item Sensors: {len(item_sensors)}")
print(sorted(item_sensors))