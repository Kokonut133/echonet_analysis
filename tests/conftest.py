from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

from src.constants import (
    CATEGORICAL_DEMOGRAPHIC_FEATURES,
    LEAD_NAMES,
    METADATA_FILENAME,
    N_LEADS,
    NUMERIC_DEMOGRAPHIC_FEATURES,
    SIGNAL_LENGTH_SAMPLES,
    TARGET_LABELS,
)

N_SAMPLES = {"train": 100, "val": 30, "test": 30}
N_TIMEPOINTS = SIGNAL_LENGTH_SAMPLES
N_TABULAR_FEATURES = 7
N_TIME_FEATURES = 9
N_SPECTRAL_FEATURES = 6
N_WAVEFORM_FEATURES = N_LEADS * (N_TIME_FEATURES + N_SPECTRAL_FEATURES)  # 180

_TIME_FEAT_NAMES = ["mean", "std", "min", "max", "rms", "energy", "skewness", "kurtosis", "zcr"]
_SPECTRAL_FEAT_NAMES = [
    "lf_power", "ecg_band_power", "hf_power", "total_power", "dominant_freq", "spectral_entropy"
]
_WF_FEATURE_NAMES = [
    f"{lead}_{feat}"
    for lead in LEAD_NAMES
    for feat in _TIME_FEAT_NAMES + _SPECTRAL_FEAT_NAMES
]


def load_script(relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    module_name = path.stem
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_metadata(rng: np.random.Generator) -> pd.DataFrame:
    splits = sum(([split] * count for split, count in N_SAMPLES.items()), [])
    n_total = sum(N_SAMPLES.values())
    return pd.DataFrame({
        "ecg_key": range(n_total),
        "patient_key": range(n_total),
        "age_at_ecg": rng.integers(30, 90, n_total),
        "sex": rng.choice(["male", "female"], n_total),
        "race_ethnicity": rng.choice(["white", "black", "hispanic", "other"], n_total),
        "location_setting": rng.choice(["inpatient", "outpatient", "emergency"], n_total),
        "acquisition_year": rng.integers(2010, 2022, n_total),
        "most_recent_ecg": rng.integers(0, 2, n_total),
        "ventricular_rate": rng.uniform(50, 130, n_total),
        "atrial_rate": rng.uniform(50, 130, n_total),
        "pr_interval": rng.uniform(100, 250, n_total),
        "qrs_duration": rng.uniform(70, 150, n_total),
        "qt_corrected": rng.uniform(300, 500, n_total),
        **{label: rng.integers(0, 2, n_total) for label in TARGET_LABELS},
        "split": splits,
    })


@pytest.fixture(scope="session")
def fake_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict:
    base: Path = tmp_path_factory.mktemp("fake_echonext")
    features_dir: Path = base / "extracted_features"
    features_dir.mkdir()

    rng = np.random.default_rng(42)

    metadata = _make_metadata(rng)
    metadata.to_csv(base / METADATA_FILENAME, index=False)

    for split, count in N_SAMPLES.items():
        waveforms = rng.standard_normal((count, 1, N_TIMEPOINTS, N_LEADS)).astype(np.float32)
        tabular = rng.standard_normal((count, N_TABULAR_FEATURES)).astype(np.float64)
        wf_feats = rng.standard_normal((count, N_WAVEFORM_FEATURES)).astype(np.float32)

        np.save(base / f"EchoNext_{split}_waveforms.npy", waveforms)
        np.save(base / f"EchoNext_{split}_tabular_features.npy", tabular)
        np.save(features_dir / f"ecg_waveform_features_{split}.npy", wf_feats)

    (features_dir / "ecg_waveform_feature_names.json").write_text(json.dumps(_WF_FEATURE_NAMES))

    return {"base": base, "features_dir": features_dir}
