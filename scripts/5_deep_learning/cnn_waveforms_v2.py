"""Train CNN v2 on 12-lead ECG waveforms: per-record normalisation +
augmentation on top of the same architecture as cnn_waveforms_only.py.

Motivation (progress_log.md, step 5 / task 7 brief): the v1 waveform-only
CNN overfits (train loss keeps falling while val AUROC plateaus ~0.80) and
there is a ~10x raw-amplitude difference between waveform files that the
model has to absorb with no explicit normalisation. v2 adds:
  - per-record standardisation (single global-std scalar per record, see
    src/augmentation.py:standardize_per_record) applied to train AND val/test
  - train-time augmentation: random time shift, amplitude scale, baseline
    wander, Gaussian noise (src/augmentation.py:train_augment)

Uses the same Trainer / TrainConfig / ECGConvNet from src (unmodified).

Input:  EchoNext_{split}_waveforms.npy  ->  (N, 12, 2500) after reshaping
Output: reports/cnn_waveforms_v2_results.csv     (per-label metrics on val set)
        reports/cnn_waveforms_v2_train_log.csv   (per-epoch loss and AUROC)
        checkpoints/cnn_waveforms_v2.pt          (best model weights)

Usage:
  python scripts/5_deep_learning/cnn_waveforms_v2.py
  python scripts/5_deep_learning/cnn_waveforms_v2.py --max-batches 5   # dry run
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from torch.utils.data import Subset

from src.augmentation import eval_transform, train_augment
from src.constants import DATASET_SUBDIR, METADATA_FILENAME, N_LEADS, TARGET_LABELS
from src.dataset import load_split
from src.dataset_v2 import AugmentedECGDataset
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
            n_epochs=30,
            patience=8,
            num_workers=2,
        ),
    )


def train_cnn_v2(config: RunConfig, max_batches: int | None = None) -> None:
    metadata = pd.read_csv(config.metadata_path)
    print(f"Loaded metadata: {len(metadata):,} rows\n")

    print("Loading splits (waveforms only)...")
    train_data, _ = load_split("train", metadata, config.dataset_dir)
    val_data, _ = load_split("val", metadata, config.dataset_dir)

    train_labels = train_data.labels
    train_ds = AugmentedECGDataset(train_data, transform=train_augment)
    val_ds = AugmentedECGDataset(val_data, transform=eval_transform)
    del train_data, val_data

    if max_batches is not None:
        n_train = min(len(train_ds), max_batches * config.train_config.batch_size)
        n_val = min(len(val_ds), max_batches * config.train_config.batch_size)
        print(
            f"--max-batches={max_batches}: truncating to {n_train:,} train / "
            f"{n_val:,} val samples for a dry run"
        )
        train_ds = Subset(train_ds, range(n_train))
        val_ds = Subset(val_ds, range(n_val))

    train_loader, val_loader = build_dataloaders(train_ds, val_ds, config.train_config)

    model = ECGConvNet(
        n_leads=N_LEADS,
        n_labels=len(TARGET_LABELS),
        n_demo_features=0,
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    pos_weights = compute_pos_weights(train_labels)

    trainer = Trainer(
        model=model,
        config=config.train_config,
        pos_weights=pos_weights,
        label_names=TARGET_LABELS,
        checkpoint_name="cnn_waveforms_v2",
    )

    print()
    t0 = time.time()
    train_log = trainer.fit(train_loader, val_loader)
    elapsed_min = (time.time() - t0) / 60
    n_epochs_run = len(train_log)
    per_epoch_min = elapsed_min / max(n_epochs_run, 1)
    print(
        f"\nTraining took {elapsed_min:.1f} min for {n_epochs_run} epoch(s) "
        f"({per_epoch_min:.2f} min/epoch)"
    )

    print("\nEvaluating on val set...")
    results = evaluate_loader(model, val_loader, trainer.device, TARGET_LABELS)
    print(results.to_string(index=False))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output_dir / "cnn_waveforms_v2_results.csv", index=False)
    train_log.to_csv(config.output_dir / "cnn_waveforms_v2_train_log.csv", index=False)
    print(f"\nSaved results to {config.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="If set, truncate train/val datasets to this many batches worth "
        "of samples, for a fast dry run (GPU path + per-epoch timing check).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_run_config()
    train_cnn_v2(config, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
