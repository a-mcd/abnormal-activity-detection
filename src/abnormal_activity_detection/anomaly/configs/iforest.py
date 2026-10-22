"""Isolation Forest experiment settings: target activity, folds, features and hyperparameters."""

TARGET_ACTIVITY_CODE = "t1"   # change to "t1".."t5 or t-all for all activities"
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
       "MEAN_TRANSITION_LOG_LIK",
       "P90_MINUS_P50",
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
            "EPISODE_DURATION",
            "NUM_EVENTS",
            "UNIQUE_SENSOR_MESSAGE_COUNT",
            "P90",
            "MISSING_EVENT_RATIO",
            "analogue_ad1_b::ACTIVE",
    ],
        # ALL activities
    "t-all": [
        "MEAN_DT",
        "MEAN_TRANSITION_LOG_LIK",
        "P90_MINUS_P50",
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
    ]
}

IFOREST_CONFIG = {
    "t1": {
        "n_estimators": 400,
        "max_samples": 1.0,
        "max_features": 0.6,
        "contamination": 0.01,
        "threshold_percentile": 22,
        "random_state": 42,
        "bootstrap": False,
    },
    "t2": {
        "n_estimators": 200,
        "max_samples": 0.8,
        "max_features": 1.0,#0.4,
        "contamination": 0.01,
        "threshold_percentile": 25,
        "random_state": 42,
        "bootstrap": False,
    },
    "t3": { 
        "n_estimators": 200, 
        "max_samples": 1.0, 
        "contamination": 0.01, # or e.g. 0.05, 0.1 
        "max_features": 1.0,#0.3, 
        "bootstrap": False, 
        "random_state": 42, 
        "threshold_percentile": 4, # e.g. 5 => mark lowest 5% train scores as anomaly 
    },
    "t4": {
        "n_estimators": 200,
        "max_samples": 1.0,
        "contamination": 0.01,   
        "max_features": 1.0,
        "bootstrap": False,
        "random_state": 42,
        "threshold_percentile": 3,  
    },
    "t5": {
        "n_estimators": 200,
        "max_samples": 1.0,
        "contamination": 0.01,   
        "max_features": 0.6,
        "bootstrap": False,
        "random_state": 42,
        "threshold_percentile": 1,  
    },
    "t-all": {
        "n_estimators": 400,
        "max_samples": 1.0,
        "max_features": 0.4,
        "contamination": 0.01,
        "threshold_percentiles_by_activity": {
            "t1": 20,
            "t2": 18,
            "t3": 6,
            "t4": 10,
            "t5": 15,
        },
        "random_state": 42,
        "bootstrap": False,
    }
}
