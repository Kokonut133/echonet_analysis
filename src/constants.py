from __future__ import annotations

SAMPLE_RATE_HZ = 250

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

DEMOGRAPHIC_FEATURES = ["age_at_ecg", "sex", "race_ethnicity", "location_setting"]
NUMERIC_DEMOGRAPHIC_FEATURES = ["age_at_ecg"]
CATEGORICAL_DEMOGRAPHIC_FEATURES = ["sex", "race_ethnicity", "location_setting"]

TABULAR_ECG_FEATURES = [
    "ventricular_rate",
    "atrial_rate",
    "pr_interval",
    "qrs_duration",
    "qt_corrected",
    "age_at_ecg",
]

TARGET_LABELS = [
    "shd_moderate_or_greater_flag",
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45_flag",
    "tr_max_gte_32_flag",
]

DATASET_SUBDIR = (
    "echonext-a-dataset-for-detecting-echocardiogram-confirmed-structural-heart-disease-from-ecgs-1.1.0"
)
METADATA_FILENAME = "echonext_metadata_100k.csv"
