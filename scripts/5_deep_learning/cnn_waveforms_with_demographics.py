"""Train a 1D CNN on ECG waveforms + demographic features to predict 12 SHD labels.

The CNN backbone processes 12-lead waveforms into a 256-dim embedding.
Demographic features (age, sex, race/ethnicity, care setting) are encoded and
concatenated to the embedding before the classification head.

Input:  EchoNext_{split}_waveforms.npy  + echonext_metadata_100k.csv
Output: reports/cnn_combined_results.csv     (per-label metrics on val set)
        reports/cnn_combined_train_log.csv   (per-epoch loss and AUROC)
        checkpoints/cnn_combined.pt          (best model weights)
        (filenames are prefixed "cnn_combined" to avoid collisions with
        cnn_waveforms_only.py, which writes "cnn_waveforms_*" outputs)

Trains for up to 30 epochs (patience 8), matching the cnn_waveforms_v2 budget.
Also writes checkpoints/cnn_combined_demo_encoder.joblib and
reports/cnn_combined_demo_features.json so scripts/9_ablation can reproduce the
demographic encoding and zero out one feature group at a time.

Smoke-tested by
tests/test_smoke.py::test_cnn_waveforms_with_demographics_saves_checkpoint_results_and_train_log.

Usage:
  python scripts/5_deep_learning/cnn_waveforms_with_demographics.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from src.constants import DATASET_SUBDIR, METADATA_FILENAME, N_LEADS, TARGET_LABELS
from src.dataset import ECGDataset, load_split
from src.preprocessing import build_demographic_preprocessor
from src.metrics import evaluate_loader
from src.models import ECGConvNet
from src.training import TrainConfig, Trainer, build_dataloaders, compute_pos_weights


@dataclass
class RunConfig:
    dataset_dir: Path
    metadata_path: Path
    output_dir: Path
    checkpoint_dir: Path
    train_config: TrainConfig = field(default_factory=TrainConfig)


def build_run_config() -> RunConfig:
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "data" / DATASET_SUBDIR
    return RunConfig(
        dataset_dir=dataset_dir,
        metadata_path=dataset_dir / METADATA_FILENAME,
        output_dir=project_root / "reports",
        checkpoint_dir=project_root / "checkpoints",
        train_config=TrainConfig(
            checkpoint_dir=project_root / "checkpoints",
            # Matches the cnn_waveforms_v2 budget so the two runs are comparable
            # and the run fits in a few hours on one GPU.
            n_epochs=30,
            patience=8,
            num_workers=2,
        ),
    )


def train_cnn_on_waveforms_and_demographics(config: RunConfig) -> None:
    metadata = pd.read_csv(config.metadata_path)
    print(f"Loaded metadata: {len(metadata):,} rows\n")

    demo_encoder = build_demographic_preprocessor()

    print("Loading splits (waveforms + demographics)...")
    train_data, demo_encoder = load_split(
        "train", metadata, config.dataset_dir,
        demo_encoder=demo_encoder, fit_encoder=True,
    )
    val_data, _ = load_split(
        "val", metadata, config.dataset_dir,
        demo_encoder=demo_encoder, fit_encoder=False,
    )

    n_demo = train_data.demo_features.shape[1]
    print(f"\nDemographic feature dimensionality: {n_demo}")

    # Persist the fitted encoder and its column names: the demographic-ablation
    # analysis has to reproduce this exact encoding to zero out one feature
    # group at a time.
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(demo_encoder, config.checkpoint_dir / "cnn_combined_demo_encoder.joblib")
    demo_feature_names = [str(n) for n in demo_encoder.get_feature_names_out()]
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "cnn_combined_demo_features.json").write_text(
        json.dumps(demo_feature_names, indent=2)
    )
    print(f"Demographic feature names: {demo_feature_names}")

    train_labels = train_data.labels
    train_ds = ECGDataset(train_data)
    val_ds = ECGDataset(val_data)
    del train_data, val_data

    train_loader, val_loader = build_dataloaders(train_ds, val_ds, config.train_config)

    model = ECGConvNet(
        n_leads=N_LEADS,
        n_labels=len(TARGET_LABELS),
        n_demo_features=n_demo,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    pos_weights = compute_pos_weights(train_labels)

    trainer = Trainer(
        model=model,
        config=config.train_config,
        pos_weights=pos_weights,
        label_names=TARGET_LABELS,
        checkpoint_name="cnn_combined",
    )

    print()
    train_log = trainer.fit(train_loader, val_loader)

    print("\nEvaluating on val set...")
    results = evaluate_loader(model, val_loader, trainer.device, TARGET_LABELS)
    print(results.to_string(index=False))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output_dir / "cnn_combined_results.csv", index=False)
    train_log.to_csv(config.output_dir / "cnn_combined_train_log.csv", index=False)
    print(f"\nSaved results to {config.output_dir}")


def main() -> None:
    config = build_run_config()
    train_cnn_on_waveforms_and_demographics(config)


if __name__ == "__main__":
    main()
