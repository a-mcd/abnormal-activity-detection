"""One-Class SVM experiment settings: target activity, folds, features and hyperparameters."""

TARGET_ACTIVITY_CODE = "t1"   # change to "t1".."t5 or "t-all"
N_SPLITS = 5
META_COLS = [
    "filename",
    "participant_id",
    "activity_code",
    "activity_index",
    "dataset",
]

FEATURE_CONFIG = {
    "t1": [
        "MEAN_DT",
        "P90_MINUS_P50",
        "EVENT_RATE",
        "SWITCH_RATE",
        "MEAN_TRANSITION_LOG_LIK",
    ],
    "t2": [
        "UNEXPECTED_SENSOR_DURING_TAP_WINDOW",
        "AD1_BC_ACTIVE_LAST_8S",
        "LAST_TAP_VALUE_LT_2_5",
    ],
    "t3": [
            "analogue_ad1_a_last_value",
            "EXTRA_EVENT_RATIO",
        ], 
    "t4": [
            "sensor::I06",
            "EPISODE_DURATION",
            "analogue_ad1_a_last_value",
            "UNIQUE_SENSOR_MESSAGE_COUNT",
    ],
    "t5": [ 
            "EPISODE_DURATION", #
            "NUM_EVENTS", #
            "UNIQUE_SENSOR_MESSAGE_COUNT", #
            "P90", #
            "MISSING_EVENT_RATIO", #
        ],
    "t-all": [
        "MEAN_DT",
        "P90_MINUS_P50",
        "EVENT_RATE",
        "SWITCH_RATE",
        "MEAN_TRANSITION_LOG_LIK",
        "UNEXPECTED_SENSOR_DURING_TAP_WINDOW",
        "AD1_BC_ACTIVE_LAST_8S",
        "LAST_TAP_VALUE_LT_2_5",
        "analogue_ad1_a_last_value",
        "EXTRA_EVENT_RATIO",
        "sensor::I06",
        "EPISODE_DURATION",
        "UNIQUE_SENSOR_MESSAGE_COUNT",
        "NUM_EVENTS",
        "P90",
        "MISSING_EVENT_RATIO",
        "analogue_ad1_b::ACTIVE",
    ],
}


OCSVM_CONFIG = {
    "t1": {
        "kernel": "linear",
        "gamma": "scale",#1.0, #1.0, # 0.1, 1.0
        "nu": 0.1, #0.01, , 0.1
        "threshold_percentile": None,
    },
    "t2": {
        "kernel": "rbf",
        "gamma": 0.1,
        "nu": 0.10,
        "threshold_percentile": 5.0,
    },
    "t3": { 
        "kernel": "linear",
        "gamma": "scale",
        "nu": 0.01,
        "threshold_percentile": 1.0,
    },
    "t4": {
        "kernel": "rbf",
        "gamma": 0.01,
        "nu": 0.01,
        "threshold_percentile": 1.0,
    },
    "t5": {
        "kernel": "linear",
        "gamma": "scale",
        "nu": 0.08,
        "threshold_percentile": 1,
    },
    "t-all": {
        "kernel": "linear",
        "gamma": "scale",
        "nu": 0.05,
        "threshold_percentiles_by_activity": {
            "t1": 20,
            "t2": 27,
            "t3": 18,
            "t4": 10,
            "t5": 1,
        },
    }
}