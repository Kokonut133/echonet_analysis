"""5-fold stratified cross-validation of the fused (waveform + demographics)
CNN over the pooled official train+val rows, to get a fold-to-fold variance
estimate for the headline fused-vs-waveform-only AUROC gap.

Every CI in this project comes from bootstrap-resampling the *test set*, so
none of them capture training/seed variance. This script fills that gap:

  1. Pool the official train (72,475) + val (4,626) rows = 77,101 records.
  2. Split the pool into 5 stratified folds on shd_moderate_or_greater_flag
     (src.crossval.make_folds — the official test split is NEVER touched here).
  3. Per fold: fit a fresh demographic encoder on the in-fold TRAINING rows
     only (fitting on the whole pool would leak heldout-fold/test rows'
     statistics into imputation/scaling/one-hot categories), train
     ECGConvNet(n_leads=12, n_labels=12, n_demo_features=<fold-specific>) on
     the in-fold rows, early-stopping on held-out-fold mean AUROC.
  4. Score that fold's model twice: on its own held-out fold, and on the
     official test split (held fixed across folds) — the second number is
     what actually answers the variance question.
  5. After all folds, also score the 5-model ensemble (mean predicted
     probability) on the official test split.

Rows come from two separate memory-mapped waveform files (train, val);
src.dataset_kfold.ConcatSplitDataset indexes across them without copying
either into RAM, mapping a global pool index to (file, row).

Outputs:
  reports/kfold_results.csv    fold,target,split,auroc,auprc,n_positive,n_total
                                split in {heldout_fold, test}; fold in
                                {"0".."k-1", "ensemble"} (ensemble rows are
                                split=="test" only).
  checkpoints/kfold/fold{i}.pt best-val-AUROC checkpoint per fold.
  progress/kfold.md            timestamped per-fold / per-epoch log lines.

Usage:
  python -u scripts/6_evaluate/kfold_cnn.py [--k 5] [--folds 5]
      [--n-epochs 20] [--patience 6] [--batch-size 32] [--num-workers 2]
      [--max-batches N] [--seed 42] [--only-fold N]

  --folds caps how many of the k folds are actually trained (e.g. --folds 1
  for a smoke run, or --folds 3 if the projected runtime is too long).
  --max-batches caps batches/epoch, for a fast smoke run of the data path.

  --only-fold N (added for agent SERIAL's stage runner, 2026-09-23): train
  *only* fold N, leaving every other fold's existing rows in
  reports/kfold_results.csv untouched (rather than the whole file being
  overwritten with just this run's folds). Used to finish a single
  interrupted fold — e.g. fold 4 of a 5-fold run that crashed mid-fold —
  without re-running the folds that already completed. After training and
  evaluating fold N, this also rebuilds the cross-fold "ensemble" row by
  reloading every other fold's checkpoint under checkpoints/kfold/foldK.pt
  (if present) and re-running test-split inference for each — each fold's
  demographic encoder is refit deterministically from `folds[K]`'s train
  indices (same seed, same split), so this reproduces exactly what that
  fold's own run would have written, without needing to persist encoders.
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.constants import DATASET_SUBDIR, METADATA_FILENAME, N_LEADS, TARGET_LABELS
from src.crossval import DEFAULT_STRATIFY_COL, make_folds
from src.dataset import ECGDataset, SplitData, load_split
from src.dataset_kfold import (
    ConcatSplitDataset,
    build_pool_metadata,
    fit_fold_demographic_encoder,
    transform_demo,
)
from src.metrics import compute_per_label_metrics, evaluate_loader
from src.models import ECGConvNet
from src.training import Trainer, TrainConfig, build_dataloaders, compute_pos_weights, resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LimitedLoader:
    """Wraps a DataLoader so an epoch stops after `max_batches` batches.

    Exposes `.dataset` and `__len__` so it drops into Trainer.fit unchanged;
    used only for --max-batches smoke runs, never in the real CV run.
    """

    def __init__(self, loader: DataLoader, max_batches: int):
        self.loader = loader
        self.max_batches = max_batches
        self.dataset = loader.dataset

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.max_batches)

    def __len__(self) -> int:
        return min(self.max_batches, len(self.loader))


def log_line(msg: str, log_path: Path) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} - {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


@torch.no_grad()
def evaluate_and_collect(
    model: ECGConvNet, loader: DataLoader, device: torch.device, label_names: list[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Like src.metrics.evaluate_loader, but also returns the raw (y_true,
    y_prob) arrays — needed here to build the cross-fold ensemble, which
    evaluate_loader doesn't expose."""
    model.eval()
    all_probs, all_true, all_mask = [], [], []
    for waveforms, demo, labels, valid_mask in loader:
        waveforms = waveforms.to(device)
        demo = demo.to(device)
        probs = torch.sigmoid(model(waveforms, demo)).cpu().numpy()
        all_probs.append(probs)
        all_true.append(labels.numpy())
        all_mask.append(valid_mask.numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_true, axis=0).astype(float)
    y_mask = np.concatenate(all_mask, axis=0)
    y_true[~y_mask] = float("nan")

    metrics = compute_per_label_metrics(y_true, y_prob, label_names)
    return metrics, y_true, y_prob


def rows_from_metrics(fold_label: str, split: str, metrics: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in metrics.iterrows():
        rows.append({
            "fold": fold_label,
            "target": r["label"],
            "split": split,
            "auroc": r["auroc"],
            "auprc": r["auprc"],
            "n_positive": r["n_positive"],
            "n_total": r["n_total"],
        })
    return rows


def predict_test_with_fold_checkpoint(
    fold_i: int,
    folds: list[tuple[np.ndarray, np.ndarray]],
    pool_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    test_data: SplitData,
    checkpoint_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Rebuild fold `fold_i`'s demographic encoder (deterministic given the
    same seed/fold split — `fit_fold_demographic_encoder` fits only on
    `folds[fold_i]`'s train indices) and run its saved checkpoint over the
    test split. Used by `--only-fold` to rebuild the cross-fold ensemble row
    without retraining or re-evaluating folds that already completed in an
    earlier run. Returns (y_true, y_prob), or None if that fold's checkpoint
    doesn't exist yet (e.g. it has never completed)."""
    ckpt_path = checkpoint_dir / f"fold{fold_i}.pt"
    if not ckpt_path.exists():
        return None

    train_idx, _ = folds[fold_i]
    encoder = fit_fold_demographic_encoder(pool_meta, train_idx)
    test_demo = transform_demo(encoder, test_meta)
    n_demo = test_demo.shape[1]

    fold_test_split = SplitData(
        waveforms=test_data.waveforms, labels=test_data.labels, demo_features=test_demo,
        label_names=TARGET_LABELS,
    )
    test_ds = ECGDataset(fold_test_split)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = ECGConvNet(n_leads=N_LEADS, n_labels=len(TARGET_LABELS), n_demo_features=n_demo)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    _, y_true, y_prob = evaluate_and_collect(model, test_loader, device, TARGET_LABELS)
    return y_true, y_prob


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=5, help="number of CV folds to split the pool into")
    p.add_argument("--folds", type=int, default=5, help="how many of the k folds to actually train")
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=None, help="cap batches/epoch, for smoke runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--only-fold", type=int, default=None,
        help="train only this one fold index (0..k-1), preserving other folds' rows already in "
             "reports/kfold_results.csv and rebuilding the cross-fold ensemble row from all available checkpoints",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_dir = PROJECT_ROOT / "data" / DATASET_SUBDIR
    metadata_path = dataset_dir / METADATA_FILENAME
    reports_dir = PROJECT_ROOT / "reports"
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / "kfold"
    progress_log = PROJECT_ROOT / "progress" / "kfold.md"
    results_path = reports_dir / "kfold_results.csv"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    metadata = pd.read_csv(metadata_path)
    pool_meta = build_pool_metadata(metadata)
    n_train_rows = int((metadata["split"] == "train").sum())
    n_val_rows = int((metadata["split"] == "val").sum())
    if len(pool_meta) != n_train_rows + n_val_rows:
        raise ValueError(
            f"pool metadata row count mismatch: pool={len(pool_meta)}, "
            f"train+val={n_train_rows + n_val_rows}"
        )

    if args.only_fold is not None and not (0 <= args.only_fold < args.k):
        raise ValueError(f"--only-fold must be in [0, {args.k}), got {args.only_fold}")

    log_line(
        f"Pool: {len(pool_meta):,} rows ({n_train_rows:,} train + {n_val_rows:,} val); "
        f"k={args.k}, "
        + (f"running only fold {args.only_fold}" if args.only_fold is not None else f"running {args.folds} fold(s)")
        + f", max_batches={args.max_batches}",
        progress_log,
    )

    folds = make_folds(pool_meta, k=args.k, stratify_col=DEFAULT_STRATIFY_COL, seed=args.seed)

    print("Loading waveforms (memory-mapped) + labels for train/val/test splits...")
    train_data, _ = load_split("train", metadata, dataset_dir, label_names=TARGET_LABELS, demo_encoder=None)
    val_data, _ = load_split("val", metadata, dataset_dir, label_names=TARGET_LABELS, demo_encoder=None)
    test_data, _ = load_split("test", metadata, dataset_dir, label_names=TARGET_LABELS, demo_encoder=None)
    test_meta = metadata[metadata["split"] == "test"].reset_index(drop=True)

    # Row-count / alignment sanity check, mirroring load_split's own check:
    # the row-count assertions inside load_split already confirm each file's
    # row order matches metadata[metadata.split==<split>]'s order; this just
    # confirms our pool concatenation used those same lengths consistently.
    assert len(train_data.waveforms) == n_train_rows
    assert len(val_data.waveforms) == n_val_rows

    pool_labels = np.concatenate([train_data.labels, val_data.labels], axis=0)

    device = resolve_device("auto")
    log_line(f"Device: {device}", progress_log)

    if args.only_fold is not None and results_path.exists():
        # Preserve every other fold's already-written rows — this run only
        # (re)trains one fold, so the rest of the CSV must survive untouched.
        # Any stale rows for *this* fold (e.g. a checkpoint saved right before
        # a crash, with no matching evaluation rows) or a stale ensemble row
        # (about to be rebuilt below) are dropped first.
        existing_df = pd.read_csv(results_path)
        existing_df = existing_df[
            (existing_df["fold"].astype(str) != str(args.only_fold)) & (existing_df["fold"] != "ensemble")
        ]
        all_rows: list[dict] = existing_df.to_dict("records")
    else:
        all_rows = []
    fold_test_probs: list[np.ndarray] = []
    test_y_true: np.ndarray | None = None
    fold_times: list[float] = []

    fold_indices = [args.only_fold] if args.only_fold is not None else list(range(min(args.folds, args.k)))

    for fold_i in fold_indices:
        t_fold0 = time.time()
        train_idx, heldout_idx = folds[fold_i]

        encoder = fit_fold_demographic_encoder(pool_meta, train_idx)
        pool_demo = transform_demo(encoder, pool_meta)
        n_demo = pool_demo.shape[1]

        fold_train_split = SplitData(
            waveforms=train_data.waveforms,
            labels=train_data.labels,
            demo_features=pool_demo[:n_train_rows],
            label_names=TARGET_LABELS,
        )
        fold_val_split = SplitData(
            waveforms=val_data.waveforms,
            labels=val_data.labels,
            demo_features=pool_demo[n_train_rows:],
            label_names=TARGET_LABELS,
        )
        concat_ds = ConcatSplitDataset(fold_train_split, fold_val_split)

        fold_train_ds = Subset(concat_ds, train_idx.tolist())
        fold_heldout_ds = Subset(concat_ds, heldout_idx.tolist())

        train_config = TrainConfig(
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            num_workers=args.num_workers,
            checkpoint_dir=checkpoint_dir,
        )
        train_loader, heldout_loader = build_dataloaders(fold_train_ds, fold_heldout_ds, train_config)

        if args.max_batches is not None:
            train_loader_for_fit: object = LimitedLoader(train_loader, args.max_batches)
            heldout_loader_for_fit: object = LimitedLoader(heldout_loader, args.max_batches)
        else:
            train_loader_for_fit = train_loader
            heldout_loader_for_fit = heldout_loader

        pos_weights = compute_pos_weights(pool_labels[train_idx])

        model = ECGConvNet(n_leads=N_LEADS, n_labels=len(TARGET_LABELS), n_demo_features=n_demo)
        trainer = Trainer(
            model=model,
            config=train_config,
            pos_weights=pos_weights,
            label_names=TARGET_LABELS,
            checkpoint_name=f"fold{fold_i}",
        )

        log_line(
            f"Fold {fold_i}: train={len(train_idx):,} heldout={len(heldout_idx):,} "
            f"n_demo={n_demo} device={trainer.device}",
            progress_log,
        )

        train_log = trainer.fit(train_loader_for_fit, heldout_loader_for_fit)
        for _, r in train_log.iterrows():
            log_line(
                f"Fold {fold_i} epoch {int(r['epoch'])}: loss={r['train_loss']:.4f} "
                f"val_mean_auroc={r['val_mean_auroc']:.4f} lr={r['lr']:.2e}",
                progress_log,
            )

        # Full (unlimited) held-out-fold evaluation with the best checkpoint.
        heldout_metrics = evaluate_loader(trainer.model, heldout_loader, trainer.device, TARGET_LABELS)
        all_rows.extend(rows_from_metrics(str(fold_i), "heldout_fold", heldout_metrics))

        # This fold's model scored on the official test split (fold-specific
        # demographic encoding, fit only on this fold's in-fold train rows).
        test_demo = transform_demo(encoder, test_meta)
        fold_test_split = SplitData(
            waveforms=test_data.waveforms,
            labels=test_data.labels,
            demo_features=test_demo,
            label_names=TARGET_LABELS,
        )
        test_ds = ECGDataset(fold_test_split)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        test_metrics, y_true_test, y_prob_test = evaluate_and_collect(
            trainer.model, test_loader, trainer.device, TARGET_LABELS
        )
        all_rows.extend(rows_from_metrics(str(fold_i), "test", test_metrics))

        if args.max_batches is None:
            # Only accumulate for the ensemble on real (non-smoke) runs.
            fold_test_probs.append(y_prob_test)
            if test_y_true is None:
                test_y_true = y_true_test

        pd.DataFrame(all_rows).to_csv(results_path, index=False)

        dt = time.time() - t_fold0
        fold_times.append(dt)
        shd_row = test_metrics[test_metrics["label"] == DEFAULT_STRATIFY_COL]
        shd_auroc = shd_row["auroc"].iloc[0] if not shd_row.empty else float("nan")
        n_epochs_run = int(train_log["epoch"].max()) if len(train_log) else 0
        log_line(
            f"Fold {fold_i} done in {dt/60:.1f} min ({n_epochs_run} epochs) | "
            f"test SHD AUROC={shd_auroc:.4f}",
            progress_log,
        )

    if args.only_fold is not None:
        # A single-fold run only has this fold's probs in memory, so the
        # ensemble is rebuilt from every fold's checkpoint on disk instead.
        log_line("Rebuilding cross-fold ensemble from all available fold checkpoints...", progress_log)
        ens_probs: list[np.ndarray] = []
        ens_true: np.ndarray | None = None
        for i in range(args.k):
            pred = predict_test_with_fold_checkpoint(
                i, folds, pool_meta, test_meta, test_data, checkpoint_dir, device, args.batch_size, args.num_workers,
            )
            if pred is None:
                log_line(f"  fold {i}: no checkpoint on disk yet — excluded from the ensemble", progress_log)
                continue
            y_true_i, y_prob_i = pred
            ens_probs.append(y_prob_i)
            if ens_true is None:
                ens_true = y_true_i
            log_line(f"  fold {i}: included in ensemble ({checkpoint_dir / f'fold{i}.pt'})", progress_log)

        if len(ens_probs) >= 2 and ens_true is not None:
            ensemble_prob = np.mean(ens_probs, axis=0)
            ensemble_metrics = compute_per_label_metrics(ens_true, ensemble_prob, TARGET_LABELS)
            all_rows.extend(rows_from_metrics("ensemble", "test", ensemble_metrics))
            pd.DataFrame(all_rows).to_csv(results_path, index=False)
            log_line(
                f"Wrote {len(ens_probs)}-model ensemble row(s) to {results_path.name} (rebuilt via --only-fold)",
                progress_log,
            )
        else:
            log_line(
                f"Only {len(ens_probs)} fold checkpoint(s) available (need >=2) — no ensemble row written.",
                progress_log,
            )
    elif len(fold_test_probs) >= 2 and test_y_true is not None:
        ensemble_prob = np.mean(fold_test_probs, axis=0)
        ensemble_metrics = compute_per_label_metrics(test_y_true, ensemble_prob, TARGET_LABELS)
        all_rows.extend(rows_from_metrics("ensemble", "test", ensemble_metrics))
        pd.DataFrame(all_rows).to_csv(results_path, index=False)
        log_line(
            f"Wrote {len(fold_test_probs)}-model ensemble row(s) to {results_path.name}",
            progress_log,
        )

    total_time = time.time() - t_start
    if fold_times:
        avg = sum(fold_times) / len(fold_times)
        projected_total = avg * args.k
        log_line(
            f"Avg fold time: {avg/60:.1f} min. Projected total for k={args.k} folds: "
            f"{projected_total/3600:.2f} h",
            progress_log,
        )
    log_line(f"Run complete in {total_time/60:.1f} min.", progress_log)


if __name__ == "__main__":
    main()
