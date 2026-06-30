"""Train a 1D CNN on 12-lead ECG waveforms to predict 12 structural heart disease labels.

Input:  EchoNext_{split}_waveforms.npy  →  (N, 12, 2500) after reshaping
Output: reports/cnn_waveforms_results.csv     (per-label metrics on val set)
        reports/cnn_waveforms_train_log.csv   (per-epoch loss and AUROC)
        checkpoints/cnn_waveforms.pt          (best model weights)

Usage:
  python scripts/train_cnn_waveforms.py
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
from src.dataset import ECGDataset, load_split
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

    print("Loading splits (waveforms only)...")
    train_data, _ = load_split("train", metadata, config.dataset_dir)
    val_data, _ = load_split("val", metadata, config.dataset_dir)

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
        n_demo_features=0,
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    pos_weights = compute_pos_weights(train_data.labels)

    trainer = Trainer(
        model=model,
        config=config.train_config,
        pos_weights=pos_weights,
        label_names=TARGET_LABELS,
        checkpoint_name="cnn_waveforms",
    )

    print()
    train_log = trainer.fit(train_loader, val_loader)

    print("\nEvaluating on val set...")
    results = evaluate_loader(model, val_loader, trainer.device, TARGET_LABELS)
    print(results.to_string(index=False))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output_dir / "cnn_waveforms_results.csv", index=False)
    train_log.to_csv(config.output_dir / "cnn_waveforms_train_log.csv", index=False)
    print(f"\nSaved results to {config.output_dir}")


def main() -> None:
    config = build_run_config()
    run(config)


if __name__ == "__main__":
    main()
