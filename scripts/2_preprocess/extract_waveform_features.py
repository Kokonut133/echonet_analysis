"""in:  EchoNext_{split}_waveforms.npy       (N, 1, 2500, 12)  250 Hz
out: ecg_waveform_features_{split}.npy    (N, 180)  15 feats × 12 leads
     ecg_waveform_feature_names.json      list[180]

time (9):     mean, std, min, max, rms, energy, skewness, kurtosis, zcr
spectral (6): lf_power, ecg_band_power, hf_power, total_power, dominant_freq, spectral_entropy
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.constants import DATASET_SUBDIR, LEAD_NAMES, SAMPLE_RATE_HZ, load_config

_batch_size_default: int = load_config()["training"]["waveform_extraction"]["batch_size"]


TIME_FEATURE_NAMES = ["mean", "std", "min", "max", "rms", "energy", "skewness", "kurtosis", "zcr"]
SPECTRAL_FEATURE_NAMES = ["lf_power", "ecg_band_power", "hf_power", "total_power", "dominant_freq", "spectral_entropy"]


@dataclass(frozen=True)
class Config:
    dataset_dir: Path
    output_dir: Path
    batch_size: int = _batch_size_default
    splits: tuple[str, ...] = ("train", "val", "test")


def all_feature_names() -> list[str]:
    return [f"{lead}_{feat}" for lead in LEAD_NAMES for feat in TIME_FEATURE_NAMES + SPECTRAL_FEATURE_NAMES]


def extract_time_features(X: np.ndarray) -> np.ndarray:
    # (batch, time, leads) -> (batch, leads, 9)  [mean, std, min, max, rms, energy, skewness, kurtosis, zcr]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    xmin = X.min(axis=1)
    xmax = X.max(axis=1)
    rms = np.sqrt((X ** 2).mean(axis=1))
    energy = (X ** 2).sum(axis=1)

    std_safe = std + 1e-8
    centered = X - mean[:, np.newaxis, :]
    skewness = (centered ** 3).mean(axis=1) / (std_safe ** 3)
    kurtosis = (centered ** 4).mean(axis=1) / (std_safe ** 4) - 3.0

    sign_diff = np.diff(np.sign(X), axis=1)
    zcr = (sign_diff != 0).mean(axis=1)

    return np.stack([mean, std, xmin, xmax, rms, energy, skewness, kurtosis, zcr], axis=-1)


def extract_spectral_features(X: np.ndarray, fs: int = SAMPLE_RATE_HZ) -> np.ndarray:
    # (batch, time, leads) -> (batch, leads, 6)  [lf_power, ecg_band_power, hf_power, total_power, dominant_freq, spectral_entropy]
    n_time = X.shape[1]
    freqs = np.fft.rfftfreq(n_time, d=1.0 / fs)

    fft_coeffs = np.fft.rfft(X, axis=1)
    power = np.abs(fft_coeffs) ** 2

    lf_mask = (freqs >= 0.5) & (freqs < 5.0)
    ecg_mask = (freqs >= 5.0) & (freqs < 40.0)
    hf_mask = (freqs >= 40.0) & (freqs < 100.0)

    total_power = power.sum(axis=1)
    lf_power = power[:, lf_mask, :].sum(axis=1)
    ecg_band_power = power[:, ecg_mask, :].sum(axis=1)
    hf_power = power[:, hf_mask, :].sum(axis=1)

    dominant_freq_idx = power.argmax(axis=1)
    dominant_freq = freqs[dominant_freq_idx]

    p_norm = power / (total_power[:, np.newaxis, :] + 1e-10)
    spectral_entropy = -(p_norm * np.log(p_norm + 1e-10)).sum(axis=1)

    return np.stack(
        [lf_power, ecg_band_power, hf_power, total_power, dominant_freq, spectral_entropy],
        axis=-1,
    )


def extract_features_from_batch(waveform_batch: np.ndarray, fs: int = SAMPLE_RATE_HZ) -> np.ndarray:
    # (batch, 1, 2500, 12) -> (batch, 180)  [12 leads × 15 features, lead-major order]
    X = waveform_batch[:, 0, :, :]

    time_feats = extract_time_features(X)
    spectral_feats = extract_spectral_features(X, fs=fs)

    combined = np.concatenate([time_feats, spectral_feats], axis=-1)
    n_samples, n_leads, n_feats = combined.shape
    return combined.reshape(n_samples, n_leads * n_feats)


def extract_and_save_split_features(waveform_path: Path, config: Config) -> np.ndarray:
    # (N, 1, 2500, 12) -> (N, 180)
    waveforms = np.load(waveform_path, mmap_mode="r")
    n_samples = waveforms.shape[0]
    n_features = len(LEAD_NAMES) * (len(TIME_FEATURE_NAMES) + len(SPECTRAL_FEATURE_NAMES))

    output = np.empty((n_samples, n_features), dtype=np.float32)

    n_batches = (n_samples + config.batch_size - 1) // config.batch_size
    for i in range(n_batches):
        start = i * config.batch_size
        end = min(start + config.batch_size, n_samples)
        batch = waveforms[start:end].astype(np.float32)
        output[start:end] = extract_features_from_batch(batch, fs=SAMPLE_RATE_HZ)
        print(f"  batch {i + 1}/{n_batches}  ({end}/{n_samples} samples)")

    return output


def save_feature_names(output_dir: Path) -> None:
    names = all_feature_names()
    path = output_dir / "ecg_waveform_feature_names.json"
    with path.open("w") as f:
        json.dump(names, f, indent=2)
    print(f"Saved {len(names)} feature names → {path.name}")


def extract_and_cache_waveform_features(config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_feature_names(config.output_dir)

    for split in config.splits:
        waveform_path = config.dataset_dir / f"EchoNext_{split}_waveforms.npy"
        if not waveform_path.exists():
            print(f"Skipping '{split}' — file not found: {waveform_path}")
            continue

        print(f"\n=== {split} ===")
        features = extract_and_save_split_features(waveform_path, config)

        output_path = config.output_dir / f"ecg_waveform_features_{split}.npy"
        np.save(output_path, features)
        print(f"Saved {features.shape} → {output_path.name}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = Config(
        dataset_dir=project_root / "data" / DATASET_SUBDIR,
        output_dir=project_root / "data" / "extracted_features",
    )
    extract_and_cache_waveform_features(config)


if __name__ == "__main__":
    main()
