# COM748 Masters Project: Abnormal Activity Detection

Detecting errors in activities of daily living (ADLs) from smart-home sensor data, using the CASAS ADL error dataset. The project has two stages:

1. **Activity classification:** which of the five activities (t1–t5) is a session? (Beta DST, Random Forest, Logistic Regression, XGBoost, LSTM)
2. **Abnormal detection:** within one activity, is a session normal or errored? (Isolation Forest, One-Class SVM, autoencoder)

Both stages are trained on normal (no-error) sessions only. The errored sessions are kept for testing.

Dataset sources:

- https://casas.wsu.edu/datasets/
- https://zenodo.org/records/15712834
- https://pmc.ncbi.nlm.nih.gov/articles/PMC2759863/pdf/nihms131754.pdf

The sensors, activities and errors are described in [Dataset](#dataset) below.

## Repository layout

```
├── data/
│   ├── raw/                      # original CSVs: adl_noerror/, adl_error/
│   └── processed/                # cleaned CSVs: adl_noerror_clean/, adl_error_clean/
├── src/abnormal_activity_detection/   # reusable code (imported by the scripts)
│   ├── paths.py                  # project folders: data/, results/
│   ├── data.py                   # loading sessions
│   ├── evaluation.py             # metrics, thresholds, confusion matrices
│   ├── features/                 # feature engineering: activity.py, anomaly/
│   ├── activity/                 # helpers shared by the activity classifiers
│   └── anomaly/                  # Isolation Forest, One-Class SVM, autoencoder
│       └── configs/              # settings for the abnormal-detection scripts
├── scripts/                      # the steps you run
│   ├── analysis/                 # exploring the data (R plots, sensor listing)
│   ├── data/                     # cleaning
│   ├── activity_classification/  # stage 1
│   ├── beta_dst/                 # Beta DST experiments for activity classification
│   └── abnormal_detection/       # stage 2
├── results/                      # everything the scripts produce
└── docs/                         # presentation and floor-plan images
```

Code in `src/` is never run directly. Scripts in `scripts/` import it, run one step, and save to `results/`. All paths are worked out from the project root, so scripts can be run from any folder.

## 1. Analyse the data

### Python setup

Use Python 3.11 or 3.12. The pinned versions in `requirements.txt` are for an Intel Mac (TensorFlow 2.16.2 is the last release for Intel Macs, and it needs NumPy below 2).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python needs to know where the package lives. Add `src/` to `PYTHONPATH` whenever the environment is activated:

```bash
echo 'export PYTHONPATH="$(dirname "$VIRTUAL_ENV")/src${PYTHONPATH:+:$PYTHONPATH}"' >> .venv/bin/activate
deactivate && source .venv/bin/activate
```

In VS Code, also add a `.env` file at the project root containing `PYTHONPATH=src`, so the run button and debugger find the package.

Check the setup:

```bash
python -c "import abnormal_activity_detection.paths as p; print(p.PROJECT_ROOT)"
```

Run every Python command below from the project root with the environment active. The first run of each script can be slow while Python compiles and caches scikit-learn and TensorFlow.

### R setup

Open `abnormal-activity-detection.Rproj` in RStudio before running any R script. The scripts use the `here` package to build paths from the project root, and opening the project sets that root.

Install the R packages once:

```r
install.packages(c("tidyverse", "lubridate", "scales", "here"))
```

### Exploring the data

These scripts read the raw data in `data/raw/` and need nothing else to run first.

| Script | Shows | Output in `results/analysis/` |
|---|---|---|
| `scripts/analysis/class_balance.R` | Episodes per activity and sessions per participant | `graphs/episodes_per_activity.png`, `graphs/sessions_per_participant.png` |
| `scripts/analysis/sensor_rate_per_activity.R` | How often each sensor message appears in each activity | `graphs/sensor_presence_rate_per_activity_heatmap.png`, `sensor_message_counts_by_activity_long.csv` |
| `scripts/analysis/episode_progression.R` | How an activity's sensor events unfold over an episode (`target_activity` in the script) | `graphs/episode_progress/` |
| `scripts/analysis/burner_sensor.R` | Burner (AD1-A) readings across all activities | `graphs/all_activities_burner_*.png` |
| `scripts/analysis/tap_sensors.R` | Hot and cold tap (AD1-B, AD1-C) readings across all activities | `graphs/all_activities_tap_*.png` |
| `scripts/analysis/plot_ad1_bc.R` | Tap readings for one activity (`target_activity` in the script) | `graphs/tap_plots/` |
| `scripts/analysis/list_sensors.py` | The unique motion and item sensors in the data (printed, not saved) | – |

## 2. Clean the data

```bash
python scripts/data/clean_data.py
```

Reads `data/raw/`, removes invalid sensor rows, smooths tap-sensor jitter, and writes `data/processed/`. Every model uses the cleaned data. It is already committed, so you only need to run this if you change the cleaning. It overwrites `data/processed/`.

## 3. Stage 1: Activity classification

### Feature engineering

```bash
python scripts/activity_classification/features/build_features.py   # engineered features for all activities
python scripts/activity_classification/features/mutual_info.py      # rank features by how well they separate the activities (MODE in script)
python scripts/activity_classification/features/run_boruta.py       # Boruta feature selection
```

Run `build_features.py` first. It writes `results/activity_classification/features/engineered_features/`, which the other two scripts and these R plots read:

- `scripts/activity_classification/features/plot_ad1_features.R`: tap and burner features
- `scripts/activity_classification/features/plot_feature_groups.R`: event, sensor, message and transition features

Plots are saved to `results/activity_classification/features/graphs/`. The classifiers below don't need these steps; they build their own features from the cleaned data.

### Beta DST

Classifies each session by combining evidence from its sensor messages using Beta distributions and Dempster–Shafer theory (DST). The work is in five attempts. Each attempt's prediction script is in `scripts/beta_dst/attempt_N/`, and its workbook, belief values and results are in `results/beta_dst/attempt_N/`.

1. Count in how many episodes of each activity each sensor message occurs:

   ```bash
   python scripts/beta_dst/event_frequency_episodes.py --input_dir data/raw/adl_noerror
   ```

   A further script counts tap-use durations by bucket. Run it once for each tap, setting `TARGET_SENSOR` at the top of the script first, and give each run its own output file (otherwise the second run overwrites the first):

   ```bash
   # TARGET_SENSOR = "AD1-B" (hot tap)
   python scripts/beta_dst/create_duration_buckets.py data/raw/adl_noerror --out results/beta_dst/ad1b_duration_buckets.csv
   # TARGET_SENSOR = "AD1-C" (cold tap)
   python scripts/beta_dst/create_duration_buckets.py data/raw/adl_noerror --out results/beta_dst/ad1c_duration_buckets.csv
   ```

   All outputs go to `results/beta_dst/`.

2. Combine the counts into one table:

   - from `event_frequency_episodes.py`, keep every sensor message except the binned analogue ones (the AD1-A, AD1-B and AD1-C `BIN_` rows), which are dropped
   - add the AD1-B and AD1-C duration-bucket counts from `create_duration_buckets.py` in their place
   - add the AD1-A `ACTIVE` count manually; the burner isn't bucketed

Steps 3–8 are done in the attempt's Excel workbook (`beta.xlsx` or `beta_episodes.xlsx`), starting from the combined counts on the `Activity_Counts` sheet. Each row is one sensor message, with columns `t1_count` to `t5_count`.

3. **"Not" values.** `total` is the sum of `t1_count` to `t5_count`. For each activity, the "not" value counts the episodes of the *other* activities in which the sensor message occurred:

   ```
   not_t1 = total - t1_count        (and likewise not_t2 to not_t5)
   ```

   For example, AD1-A `ACTIVE` has counts 0, 0, 24, 3 and 1, so `total` is 28 and `not_t3` is 28 − 24 = 4.

4. **Distinctiveness score.** How concentrated a sensor message is in one activity: 1 if it only ever occurs in one activity, 0 if it occurs equally in all five. It is one minus the normalised entropy of the activity counts:

   ```
   p_i = t_i_count / total
   Distinctiveness_Score = 1 - ( -Σ p_i ln(p_i) ) / ln(5)        (0 if total is 0; terms with p_i = 0 are skipped)
   ```

   In Excel:

   ```
   =IF(H2=0,0,1-(-(IF(C2=0,0,(C2/H2)*LN(C2/H2))+IF(D2=0,0,(D2/H2)*LN(D2/H2))+IF(E2=0,0,(E2/H2)*LN(E2/H2))+IF(F2=0,0,(F2/H2)*LN(F2/H2))+IF(G2=0,0,(G2/H2)*LN(G2/H2)))/LN(5)))
   ```

5. **Maximum reliability.** A cap on how much each type of sensor can be trusted:

   | Sensor type | `Reliability_Max` |
   |---|---|
   | Burner (AD1-A) | 0.95 |
   | Tap duration buckets (AD1-B, AD1-C) | 0.7 |
   | Item sensors (I01–I09) | 0.95 |
   | Door sensor (D01) | 0.95 |
   | Phone sensor | 0.95 |
   | Motion sensors (M) | 0.2 |

6. **Reliability.** The distinctiveness score scaled to between 0.05 and 0.95, capped at the sensor type's maximum:

   ```
   Reliability = MIN(Reliability_Max, 0.05 + 0.9 * Distinctiveness_Score)
   ```

   In Excel: `=MIN(Q2, 0.05 + 0.9*P2)`.

7. **Belief, disbelief and uncertainty.** Each activity has its own sheet in the workbook, with one row per sensor message. On it, the activity's count is the evidence for it and its "not" value the evidence against it, with a prior weight of 5:

   ```
   belief      = t_count / (t_count + not_t + 5)
   disbelief   = not_t   / (t_count + not_t + 5)
   uncertainty = 5       / (t_count + not_t + 5)
   ```

   The three always add up to 1. In Excel, for t5: `=Activity_Counts!G2/(Activity_Counts!G2+Activity_Counts!N2+5)` for belief, with `N2` and `5` as the numerators for disbelief and uncertainty.

8. **Discounted values.** Belief and disbelief are scaled by the sensor message's reliability, and the rest goes to uncertainty:

   ```
   discounted_belief      = Reliability * belief
   discounted_disbelief   = Reliability * disbelief
   discounted_uncertainty = 1 - discounted_belief - discounted_disbelief
   ```

   These are added to the same activity sheet as step 7.

9. **Export the activity sheets.** Before classifying, save each activity's sheet as a CSV in the attempt's results folder, named `t1_sensor_belief_values.csv` to `t5_sensor_belief_values.csv` (for example `results/beta_dst/attempt_5/t1_sensor_belief_values.csv`). The prediction script reads these files by name, and each needs the columns `sensor`, `message`, `belief`, `disbelief`, `uncertainty`, `discounted_belief`, `discounted_disbelief` and `discounted_uncertainty`.

10. Classify the sessions with the attempt's prediction script, for example attempt 5:

   ```bash
   python scripts/beta_dst/attempt_5/beta_dst_predict.py
   ```

   It reads the `t1`–`t5` sensor belief value CSVs from `results/beta_dst/attempt_5/`, classifies every raw session in `data/raw/adl_noerror` and `data/raw/adl_error`, and writes the results to `results/beta_dst/attempt_5/debug_noerror/` and `debug_error/`: a debug file per session and activity, plus the confusion matrix, classification report and best activity per file. Attempt 1 classifies only the no-error sessions and writes to `debug/`.

Two scripts belong to earlier approaches and aren't used:

- `event_frequency_windows.py` counts per 30-second window instead of per episode.
- `episode_duration_bucket_counts.py` counts how many episodes of each activity fall into each duration bucket. It was used in an earlier version of the DST method and later discarded.

### Random Forest, Logistic Regression, XGBoost and LSTM

```bash
python scripts/activity_classification/random_forest.py
python scripts/activity_classification/logistic_regression.py
python scripts/activity_classification/xgboost_classifier.py
python scripts/activity_classification/lstm.py
```

Each reads the cleaned data, runs 5-fold cross-validation grouped by participant on the normal sessions, trains a final model on all of them, then classifies the error sessions. Like stage 2, they only learn from normal sessions; the errored sessions are used for testing. Settings (seed, model parameters) are near the top of each script. Outputs go to `results/activity_classification/<model>/`.

## 4. Stage 2: Abnormal detection

### Settings

Settings for these scripts live in `src/abnormal_activity_detection/anomaly/configs/`:

| File | Used by |
|---|---|
| `features.py` | `scripts/abnormal_detection/features/` |
| `iforest.py` | `scripts/abnormal_detection/isolation_forest/` |
| `ocsvm.py` | `scripts/abnormal_detection/one_class_svm/` |
| `autoencoder.py` | `scripts/abnormal_detection/autoencoder/` |

Each file sets `TARGET_ACTIVITY_CODE` (`"t1"`–`"t5"`, or `"t-all"` for all activities together), the features to use, and the model settings. Check the target activity before running: the files can be set to different activities.

### Feature engineering

```bash
python scripts/abnormal_detection/features/build_features.py   # engineered features for one activity
python scripts/abnormal_detection/features/mutual_info.py      # rank features by how well they separate normal from errored sessions
python scripts/abnormal_detection/features/pca.py              # PCA of the features (builds its own from the cleaned data)
```

Run `build_features.py` before `mutual_info.py`. It writes `results/abnormal_detection/features/<activity>/engineered_features/`, which `mutual_info.py` reads. `scripts/abnormal_detection/features/plot_all_features.R` plots these features into the same activity folder. The models below don't need these steps; they build their own features from the cleaned data.

### Isolation Forest, One-Class SVM and autoencoder

For each model, run the steps in this order (Isolation Forest shown; `one_class_svm/` and `autoencoder/` have the same scripts):

```bash
python scripts/abnormal_detection/isolation_forest/tune.py          # 1. try each setting in PARAM_GRID
#                                                                      2. copy the chosen settings into configs/iforest.py
python scripts/abnormal_detection/isolation_forest/evaluate.py      # 3. 5-fold CV on normal sessions (false-alarm rate)
python scripts/abnormal_detection/isolation_forest/train.py         # 4. final model on all normal sessions
python scripts/abnormal_detection/isolation_forest/score_errors.py  # 5. test the final model on errored sessions
```

The models only ever learn from normal sessions; errored sessions are used for testing. `score_errors.py` loads the model saved by `train.py`, so run `train.py` first. Outputs go to `results/abnormal_detection/<model>/<activity>/`.

`scripts/abnormal_detection/isolation_forest/feature_importance.py` measures which features matter most to an Isolation Forest. It reads the engineered features, so run the feature engineering step first.

## Reproducibility

Random seeds are fixed in every model. The autoencoder and LSTM also switch on TensorFlow's deterministic mode, so repeated runs on the same machine give the same results. Results can still differ slightly between machines or library versions.

# Dataset

## Activity Labels

CSV files are named according to the activity they represent:

- **t1** = Telephone use
- **t2** = Hand washing
- **t3** = Meal preparation
- **t4** = Eating and medication
- **t5** = Cleaning

Example: `p17.t1.csv` contains sensor data from participant 17 during telephone use.

## Sensors
Each sensor is tagged with a sensor ID, date, time and sensor value. 

### M01 - M026: PIR motion detectors
ON when detected motion starts and OFF when it stops

#### Living Room Area
- M01: Near hallway to stairs/backdoor
- M02
- M03
- M04: Near front window
- M05: Near front window
- M06
- M07
- M08
- M09
- M010
- M011: Near front window
- M012: At front door
- M013
- M014
- M015: Near kitchen

#### Kitchen
- M016: On entry to kitchen
- M017: Between cooker and sink
- M018: Near sink

#### Hallway from kitchen to stairs 
- M019: Hallway
- M020: Cupboard

#### Hallway to Backdoor/Stairs
- M021: On entry from livingroom
- M022
- M023: 
- M024: Backdoor
- M025: Bottom of stairs

#### Stairs
- M026

#### Landing
- M027: Top of stairs
- M028
- M029

#### Main Bedroom
- M031
- M032
- M033
- M034
- M035
- M036

#### Bathroom
- M037: Bathroom door
- M038: Near bathroom sinks
- M039: Near top of bathtub
- M040: Near bottom of bathtub
- M041: Door to toilet

#### Other bedrooms
- M042: At bedroom1 door
- M043: At bedroom2 door

### I01 - I08: Binary Sensor
Items are set on sensors. PRESENT or ABSENT indicating if the item is on the sensor or not.
- I01: Oatmeal: Botton shelf of cupboard
- I02: Raisins: Botton shelf of cupboard
- I03: Brown sugar: Botton shelf of cupboard
- I04: Bowl: Botton shelf of cupboard
- I05: Measuring spoon: Botton shelf of cupboard
- I06: Medicine container: Middle shelf of cupboard
- I07: Pot: On counter
- I08: Phone book
- I09 or asterisk: Phone - VOIP Monitoring: The system utilized Voice over Internet Protocol (VOIP) to capture phone usage data. This allowed the researchers to monitor when calls were made or received within the smart environment.

### D01
Door sensor on kitchen cabinet (OPEN or CLOSE)

### AD1
Values range from 3.34583 (highest) and 2.79887 (lowest) with a higher number representating a higher intensity. 
- AD1-A: burner sensor (value indicates level)

These are water flow sensors attached to the kitchen sink. They measure the level of water flow, with higher values indicating stronger water flow. Values range form
- AB1-A: burner 0.593993 (highest) and 0.0308598 (lowest).
- AD1-B: Hot water for kitchen sink (value indicates level)
- AD1-C: Cold water for kitchen sink (value indicates level)

### T
Temperatire sensors not used in the dataset
- T01: Living room
- T02: Kitchen
- T03: Bedroom

## Activities

### 1) Telephone Use
Specified steps:
    1) Look up a specified number in the phone book.
    2) Call the number.
    3) Write down cooking instructions given in the recorded message.

Specific Error: Dial a wrong phone number then retry and successfully reach the recorded message

Error Simlation: Simulate someone who is having difficulty using the phonebook and recording information about the recipe.

Senors Generally Triggered:
- M01, M07, M08, M09, M13, M14: Living Room Area
- I08: Phone book
- asterisk: Phone

Success Tasks: 24
Errored Tasks: 20


### 2) Hand Washing
1) Wash hands in kitchen sink

Specific Error: Leave the water running after washing hands.

Error Simlation: Simulate someone who gets confused and becomes stuck completing the task.

Senors Generally Triggered:
- M13, M14: Livingroom area
- M15: Livingroom area near kitchen
- M16: Kitchen on entry to kitchen
- M17: Kitchen between cooker and sink
- AD1-B: Hot water for kitchen sink

Success Tasks: 24
Errored Tasks: 20


### 3) Meal Preparation
1) Cook oat meal on stove according to recorded directoions 
2) Add brown sugar and raisens once done

Specific Error: Leave the burner on after cooking the oatmeal

Error Simlation: Simulate someone having difficulty with the timing and order of the steps involved in the task

Senors Generally Triggered:
- M016: On entry to kitchen
- M017: Between cooker and sink
- D01: Door sensor on kitchen cabinet
- I01: Oatmeal: Botton shelf of cupboard
- I02: Raisins: Botton shelf of cupboard
- I03: Brown sugar: Botton shelf of cupboard
- I04: Bowl: Botton shelf of cupboard
- I05: Measuring spoon: Botton shelf of cupboard
- I07: Pot: On counter
- AD1-A: burner sensor 
- AD1-C: Cold water for kitchen sink 

Success Tasks: 24
Errored Tasks: 20


### 4) Eating and Medication Use
1) Eat oat meal together with a glass of water and medicine (a piece of candy)

Specific Error: Forget to take medication with the meal.

Error Simlation: Simulate someone completing the task steps in a slow and inefficient manner.

Senors Generally Triggered:
- M013: Livingroom area
- M014: Livingroom area
- M015: Livingroom area near kitchen
- M016: Kitchen On entry to kitchen
- M017: Kitchen Between cooker and sink
- D01: Door sensor on kitchen cabinet
- I06: Medicine container: Middle shelf of cupboard

Success Tasks: 24
Errored Tasks: 20

### 5) Cleaning
1) Clean and put away the dishes and ingredients.

Specific Error: Wipe of the dishes without using running water to clean.

Error Simlation: Simulate someone who becomes confused and moves off task and is later directed back to the task.

Senors Generally Triggered:
- M013, M014: Livingroom area
- M015: Livingroom area near kitchen
- M016: On entry to kitchen
- M017: Between cooker and sink
- M018: Near sink
- D01: Door sensor on kitchen cabinet
- I03: Brown sugar: Botton shelf of cupboard
- I04: Bowl: Botton shelf of cupboard
- I06: Medicine container: Middle shelf of cupboard
- AD1-B: Hot water for kitchen sink

Success Tasks: 24
Errored Tasks: 20

### Stats
- 48 participants
- 5 activities
    - Error tasks total: 100
    - Error tasks per activity: 20
    - Success tasks total: 120
    - Success tasks per activity: 24
- Total sensor types: 56 used (Not counting temp sensors)
- Total errored sensor events: 5161
- Total success sensor events: 6425













python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo 'export PYTHONPATH="$(dirname "$VIRTUAL_ENV")/src${PYTHONPATH:+:$PYTHONPATH}"' >> .venv/bin/activate


deactivate
source .venv/bin/activate
echo $PYTHONPATH
