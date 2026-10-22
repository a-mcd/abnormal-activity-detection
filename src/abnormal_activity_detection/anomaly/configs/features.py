"""Settings for the abnormal-detection feature scripts (build_features, pca, mutual_info)."""

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

TARGET_ACTIVITY_CODE = "t1" # Set to None → ALL activities for build_engineered_features