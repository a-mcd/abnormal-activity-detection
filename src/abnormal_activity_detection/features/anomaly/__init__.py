"""Feature extraction pipeline for CASAS session data.

OVERVIEW
--------
This file converts one raw session CSV into a single feature row for modelling.
It does this in two main parts:

1) Analogue sensor preprocessing pipeline (Stages 1-5)
   - Focused mainly on AD1-A, AD1-B, and AD1-C.
   - AD1-B and AD1-C are transformed into cleaner event-like representations.
   - AD1-A is kept for analogue summary features, but excluded from the final
     Stage 5 event table.

2) Session/event feature extraction
   - Builds count, timing, transition, sequence, composite, and phone features
     from the final event table.

STAGES
------

Stage 1: Raw analogue summary features
--------------------------------------
Functions:
    analogue_stage1_feature_dict(...)
    stage1_tap_window_feature_dict(...)

Purpose:
    Compute basic per-sensor numeric summaries directly from the raw analogue
    readings for each AD1 sensor, plus raw session-level tap-window features.

What it produces:
    - count
    - mean, std, min, max, median
    - percentiles (p25, p75, p90)
    - range, IQR, coefficient of variation
    - percent of zero values
    - last value
    - tail mean / tail slope
    - ACTIVE flag
    - analogue duration
    - seconds from last analogue reading to episode end
    - ratio from last analogue reading to episode end
    - UNEXPECTED_SENSOR_DURING_TAP_WINDOW
    - AD1_BC_ACTIVE_LAST_8S


Stage 2: Collapse repeated identical analogue values
----------------------------------------------------
Function:
    analogue_stage2_feature_dict(...)

Purpose:
    Remove consecutive duplicate analogue readings so that long runs of the
    same value are treated as one state rather than repeated noise.

What it produces:
    - num_changes
    - spike_count
    - a reduced Stage 2 dataframe

Why:
    Raw analogue sensors can repeat the same numeric value many times. This
    stage makes later interaction counting and direction analysis more robust.

Stage 3: Merge micro-adjustment events
--------------------------------------
Functions:
    apply_stage3(...)
    analogue_stage3_feature_dict(...)

Purpose:
    Merge analogue updates that happen very close together in time
    (< MICRO_ADJUSTMENT_WINDOW_SEC) into one interaction.

How it works:
    - keeps the first timestamp in the short burst
    - keeps the last value in the burst

What it produces:
    - interactions count
    - AD1_B_LAST_IS_DOWN
    - AD1_C_LAST_IS_DOWN
    - a cleaned Stage 3 dataframe

Why:
    Small rapid changes often represent one user action rather than several
    independent interactions.

Stage 4: Convert analogue changes into direction labels
-------------------------------------------------------
Function:
    apply_stage4_direction_labels(...)

Purpose:
    Convert Stage 3 numeric analogue values into event-style messages.

Labels used:
    - START : first value
    - UP    : current value > previous value
    - DOWN  : current value < previous value
    - SAME  : current value == previous value

Why:
    This turns AD1-B and AD1-C from continuous numeric signals into symbolic
    events that can be mixed with the normal sensor/message event stream.

Stage 5: Build final event table
--------------------------------
Functions:
    apply_stage5_build_event_table(...)
    build_stage5_event_table(...)

Purpose:
    Create the final event sequence used for downstream sequence/timing
    features.

How it works:
    - removes all raw AD1 rows from the original table
    - appends Stage 4 event rows for AD1-B and AD1-C
    - excludes AD1-A from the final event table
    - sorts everything by timestamp

Why:
    The final event table is intended to represent meaningful session events
    without raw analogue noise. AD1-B/C are included as cleaned event-like
    signals; AD1-A is left out of the event stream and handled through its
    analogue summary features.

DOWNSTREAM FEATURE GROUPS
-------------------------

After Stage 5, the code creates several families of features:

1) Global episode features
   - overall session properties such as UNIQUE_SENSOR_COUNT and EPISODE_DURATION

2) Event count / diversity features
   - NUM_EVENTS, N_BIGRAMS, N_TRIGRAMS, UNIQUE_MESSAGES,
     UNIQUE_SENSOR_MESSAGE_COUNT

3) Temporal dynamic features
   - inter-event timing summaries such as MEAN_DT, STD_DT, P90, MAX_DT,
     EVENT_RATE, EVENTS_PER_MIN, BURSTINESS

4) Event encoding features
   - sparse counts of:
       sensor::<sensor>
       msg::<message>
       evt::<sensor>::<message>
       state::<sensor>::<message>
       type::<sensor_type>
       trans::<sensor_a>::<sensor_b>

5) Transition behaviour features
   - pattern/repetition statistics such as SWITCH_RATE, REPEAT_RATIO,
     BIGRAM_MATCH_RATIO, TRIGRAM_MATCH_RATIO

6) Reference sequence features
   - compares the session to normal-reference vocabularies built from training
     sessions:
       EXTRA_EVENT_RATIO
       EXTRA_BIGRAM_RATIO
       EXTRA_TRIGRAM_RATIO
       MISSING_EVENT_RATIO
       MISSING_BIGRAM_RATIO
       TRANSITION_LOG_LIK
       MEAN_TRANSITION_LOG_LIK

7) Engineered composite features
   - combinations of previously created features, e.g.
       P90_MINUS_P50
       P90_X_SWITCH
       SWITCH_OVER_P90
       MAX_DT_OVER_P90
       SPIKE_COUNT
       AD1_BC_TOTAL_INTERACTIONS

8) Phone features
   - phone-related indicators such as HAS_ANY_PHONE_CALL and
     FIRST_PHONEBOOK_TO_LAST_PHONE_DELAY

MAIN ENTRY POINTS
-----------------

extract_features_for_session(...)
    Main per-session feature extractor. Runs Stages 1-5 and then builds all
    feature groups.

build_feature_matrix(...)
    Builds a dataframe with one feature row per session.

build_reference_stats(...)
    Builds the normal-reference vocabularies used by the sequence comparison
    features.

build_feature_matrix_train_loo_reference(...)
    Builds training rows using leave-one-out reference stats so each session is
    compared against references built from the other training sessions only.

MODULES
-------
constants    Shared constants and metadata column names.
helpers      Small helpers shared by several feature families (timestamps, sensor names, missing values).
tap_burner   Tap and burner (analogue AD1-A/B/C) sensor pipeline, Stages 1-5.
timing       Episode-level, event-count/diversity and temporal-dynamic (timing) features.
transitions  Event encoding, n-gram, transition-behaviour and repeat features built from
phone        Phone-use features (phone book I08 and the VOIP phone sensor).
composite    Engineered composite features combining other feature families.
reference    Normal-reference vocabularies and the features that compare a session's
pipeline     Session-level extraction and feature-matrix builders: turn sessions into
"""


from .constants import (
    TAIL_N,
    COLLAPSE_SENSORS,
    ALL_AD1_SENSORS,
    MICRO_ADJUSTMENT_WINDOW_SEC,
    PHONE_MESSAGE_SET,
    PHONE_SENSOR_SET,
    TARGET_PHONE_ANCHOR_SENSOR,
    META_COLS,
)

from .helpers import (
    safe_sensor_name,
    find_time_column,
    build_timestamp,
    is_active_message_raw,
    clean_missing,
)

from .tap_burner import (
    compute_tail_slope,
    collapse_consecutive_identical_rows,
    compute_num_changes,
    compute_spike_count,
    stage1_tap_window_feature_dict,
    analogue_stage1_feature_dict,
    analogue_stage2_feature_dict,
    apply_stage3,
    analogue_stage3_feature_dict,
    apply_stage4_direction_labels,
    apply_stage5_build_event_table,
    build_stage5_event_table,
)

from .timing import (
    compute_global_episode_features,
    create_event_count_diversity_features,
    create_temporal_dynamic_features,
)

from .transitions import (
    create_event_encoding_features,
    trigram_debug_summary,
    write_fold_trigram_debug_file,
    print_fold_trigram_debug,
    create_transition_behaviour_features,
    create_stage5_repeat_features,
)

from .phone import (
    create_phone_features,
)

from .composite import (
    create_engineered_composite_features,
)

from .reference import (
    create_reference_sequence_features,
    get_reference_stats_for_activity,
    build_reference_stats,
    build_reference_stats_by_activity,
)

from .pipeline import (
    default_value_for_missing_feature,
    extract_features_for_session,
    infer_all_feature_columns,
    feature_columns,
    build_feature_matrix,
    build_feature_matrix_train_loo_reference,
)

__all__ = [
    "TAIL_N",
    "COLLAPSE_SENSORS",
    "ALL_AD1_SENSORS",
    "MICRO_ADJUSTMENT_WINDOW_SEC",
    "PHONE_MESSAGE_SET",
    "PHONE_SENSOR_SET",
    "TARGET_PHONE_ANCHOR_SENSOR",
    "META_COLS",
    "safe_sensor_name",
    "find_time_column",
    "build_timestamp",
    "is_active_message_raw",
    "clean_missing",
    "compute_tail_slope",
    "collapse_consecutive_identical_rows",
    "compute_num_changes",
    "compute_spike_count",
    "stage1_tap_window_feature_dict",
    "analogue_stage1_feature_dict",
    "analogue_stage2_feature_dict",
    "apply_stage3",
    "analogue_stage3_feature_dict",
    "apply_stage4_direction_labels",
    "apply_stage5_build_event_table",
    "build_stage5_event_table",
    "compute_global_episode_features",
    "create_event_count_diversity_features",
    "create_temporal_dynamic_features",
    "create_event_encoding_features",
    "trigram_debug_summary",
    "write_fold_trigram_debug_file",
    "print_fold_trigram_debug",
    "create_transition_behaviour_features",
    "create_stage5_repeat_features",
    "create_phone_features",
    "create_engineered_composite_features",
    "create_reference_sequence_features",
    "get_reference_stats_for_activity",
    "build_reference_stats",
    "build_reference_stats_by_activity",
    "default_value_for_missing_feature",
    "extract_features_for_session",
    "infer_all_feature_columns",
    "feature_columns",
    "build_feature_matrix",
    "build_feature_matrix_train_loo_reference",
]
