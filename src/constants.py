from __future__ import annotations

import functools
import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def load_config() -> dict:
    return json.loads((_PROJECT_ROOT / "data" / "project_config.json").read_text())


_cfg = load_config()

SAMPLE_RATE_HZ: int           = _cfg["ecg"]["sample_rate_hz"]
N_LEADS: int                  = _cfg["ecg"]["n_leads"]
SIGNAL_LENGTH_SAMPLES: int    = _cfg["ecg"]["signal_length_samples"]
LEAD_NAMES: list[str]         = _cfg["ecg"]["lead_names"]

DEMOGRAPHIC_FEATURES: list[str]             = _cfg["features"]["demographic"]
NUMERIC_DEMOGRAPHIC_FEATURES: list[str]     = _cfg["features"]["demographic_numeric"]
CATEGORICAL_DEMOGRAPHIC_FEATURES: list[str] = _cfg["features"]["demographic_categorical"]
TABULAR_ECG_FEATURES: list[str]             = _cfg["features"]["tabular_ecg"]

TARGET_LABELS: list[str] = _cfg["targets"]

DATASET_SUBDIR: str    = _cfg["dataset"]["subdir"]
METADATA_FILENAME: str = _cfg["dataset"]["metadata_file"]
