"""Train a 1D CNN on ECG waveforms + demographic features to predict 12 SHD labels.

The CNN backbone processes 12-lead waveforms into a 256-dim embedding.
Demographic features (age, sex, race/ethnicity, care setting) are encoded and
concatenated to the embedding before the classification head.

Input:  EchoNext_{split}_waveforms.npy  + echonext_metadata_100k.csv
Output: reports/cnn_combined_results.csv     (per-label metrics on val set)
        reports/cnn_combined_train_log.csv   (per-epoch loss and AUROC)
        checkpoints/cnn_combined.pt          (best model weights)

Usage:
  python scripts/train_cnn_combined.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

# make src importable when running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.constants import DATASET_DIRNAME, TARGET_LABELS
from src.dataset import ECGDataset, build_demo_encoder, load_split
from src.metrics import evaluate_loader
from src.model import ECGConvNet
from src.training import TrainConfig, Trainer, compute_pos_weights


@dataclass
class RunConfig:
    dataset_dir: Path
    metadata_path: Path
    output_dir: Path
    checkpoint_dir: Path
    train_config: TrainConfig = field(default_factory=TrainConfig)


def build_run_config() -> RunConfig:
    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / "data" / DATASET_DIRNAME
    return RunConfig(
        dataset_dir=dataset_dir,
        metadata_path=dataset_dir / "echonext_metadata_100k.csv",
        output_dir=project_root / "reports",
        checkpoint_dir=project_root / "checkpoints",
        train_config=TrainConfig(
            n_epochs=50,
            batch_size=64,
            learning_rate=1e-3,
            patience=10,
            checkpoint_dir=project_root / "checkpoints",
        ),
    )


def run(config: RunConfig) -> None:
    metadata = pd.read_csv(config.metadata_path)
    print(f"Loaded metadata: {len(metadata):,} rows\n")

    demo_encoder = build_demo_encoder()

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

    train_ds = ECGDataset(train_data)
    val_ds = ECGDataset(val_data)

    train_loader = DataLoader(
        train_ds, batch_size=config.train_config.batch_size,
        shuffle=True, num_workers=config.train_config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.train_config.batch_size,
        shuffle=False, num_workers=config.train_config.num_workers, pin_memory=True,
    )

    model = ECGConvNet(
        n_leads=12,
        n_labels=len(TARGET_LABELS),
        n_demo_features=n_demo,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    pos_weights = compute_pos_weights(train_data.labels)

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
    run(config)


if __name__ == "__main__":
    main()
