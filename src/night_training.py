"""Serial stage-runner primitives (agent SERIAL, 2026-09-23).

The machine crashed earlier today while three heavy jobs (a GPU 5-fold CNN
CV, a CPU Optuna search, and an idle orchestrator) ran concurrently —
measured conditions just before the crash: GPU ~12% utilised, dataloader
workers blocked in WSL2's `p9_client_rpc`, page cache saturated at 13GB with
116MB free, against a 16GB memory-mapped waveform file on a 15GB-RAM box.
Almost certainly memory/I/O exhaustion from concurrency, not any single job.

This module is the non-CLI half of the fix: everything
`scripts/12_overnight/run_overnight.py` needs that is *not* already in
`src/training.py` / `src/dataset.py` / `src/dataset_kfold.py`, grouped into:

  - pure, unit-testable helpers: the warmup+cosine LR schedule, the budget
    checker, the `/proc/meminfo`-based memory guard, and the stage-state JSON
    round-trip (atomic write: temp file + `os.replace`, so a crash mid-write
    can never corrupt the real state file)
  - `ResumableLongrunTrainer`, a from-scratch-or-resumed warmup+cosine
    trainer for the `longrun` stage: checkpoints every N epochs, writes a
    resume checkpoint (model + optimizer + epoch + best-val + patience)
    every single epoch, and calls back after every epoch so the caller can
    flush a CSV row immediately (never buffer a run's worth of logging)
  - the `elimination_sweep` search space + a from-scratch,
    per-epoch-report/prune Optuna trial loop (needs `trial.report` after
    *every* epoch for `SuccessiveHalvingPruner` to fire, which
    `Trainer.fit` does not expose)
  - the `warmstart_tune` search space + a warm-started refinement trial loop
    that reloads a `longrun` checkpoint into a fresh model (dropout has no
    learned parameters, so changing it doesn't break `load_state_dict`) and
    runs a short warmup-free cosine tail

Nothing here launches a subprocess or owns `reports/serial_state.json` —
that orchestration lives in `scripts/12_overnight/run_overnight.py`, which
is the only thing that ever writes the stage-state file (so there is never
more than one writer, even though stages run as separate subprocesses).
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import optuna
import torch
from torch.utils.data import DataLoader, Dataset

from src.metrics import evaluate_loader, mean_auroc
from src.models import ECGConvNet
from src.training import Trainer, TrainConfig, build_dataloaders

# --------------------------------------------------------------------------
# Warmup + cosine LR schedule — pure function, no torch objects required.
# --------------------------------------------------------------------------


def warmup_cosine_lr(
    epoch_idx: int,
    base_lr: float,
    warmup_epochs: int,
    total_epochs: int,
    end_lr: float = 0.0,
) -> float:
    """LR for 0-indexed `epoch_idx` under linear warmup then cosine decay.

    `epoch_idx` in [0, warmup_epochs) ramps linearly from `base_lr /
    warmup_epochs` (epoch 0) up to `base_lr` (reached exactly at
    `epoch_idx == warmup_epochs`, i.e. the first epoch after warmup).
    `epoch_idx` in [warmup_epochs, total_epochs) cosine-decays from
    `base_lr` down to `end_lr`, reaching `end_lr` exactly at the final
    epoch (`epoch_idx == total_epochs - 1`).
    """
    if warmup_epochs <= 0:
        warmup_epochs = 0
    if epoch_idx < warmup_epochs:
        return base_lr * (epoch_idx + 1) / max(warmup_epochs, 1)

    span = max(total_epochs - 1 - warmup_epochs, 1)
    progress = (epoch_idx - warmup_epochs) / span
    progress = min(max(progress, 0.0), 1.0)
    return end_lr + 0.5 * (base_lr - end_lr) * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------
# Budget checking
# --------------------------------------------------------------------------


def elapsed_hours(start_ts: float, now_ts: float | None = None) -> float:
    now_ts = time.time() if now_ts is None else now_ts
    return (now_ts - start_ts) / 3600.0


def budget_exceeded(start_ts: float, budget_hours: float | None, now_ts: float | None = None) -> bool:
    """True once `now_ts` (default: real time.time()) is `budget_hours` past `start_ts`.

    `budget_hours <= 0` or `None` means "no budget" -> never exceeded.
    """
    if budget_hours is None or budget_hours <= 0:
        return False
    return elapsed_hours(start_ts, now_ts) >= budget_hours


def deadline_ts(start_ts: float, budget_hours: float | None) -> float:
    """Absolute epoch-seconds deadline, for passing down to inner training loops."""
    if budget_hours is None or budget_hours <= 0:
        return float("inf")
    return start_ts + budget_hours * 3600.0


def deadline_exceeded(deadline_ts_value: float, now_ts: float | None = None) -> bool:
    """True once `now_ts` (default: real time.time()) is past an absolute deadline."""
    now_ts = time.time() if now_ts is None else now_ts
    return now_ts >= deadline_ts_value


# --------------------------------------------------------------------------
# Memory guard — reads /proc/meminfo's MemAvailable, the same signal the
# crash post-mortem used (13GB page cache, 116MB free just before the box
# went down). Every dependency is injectable so this is testable without a
# real /proc filesystem, a real clock, or a real sleep.
# --------------------------------------------------------------------------

DEFAULT_MEMINFO_PATH = Path("/proc/meminfo")
DEFAULT_MEM_THRESHOLD_GB = 2.5


def read_mem_available_kb(meminfo_path: Path = DEFAULT_MEMINFO_PATH) -> float:
    """Parse `MemAvailable` (kB) out of a /proc/meminfo-shaped text file.

    Raises if the file is missing or the field isn't present — callers decide
    whether that means "fail open" (proceed) or "fail closed" (skip); the
    stage runner treats an unreadable /proc/meminfo as fail-closed, since the
    whole point of this guard is not repeating today's OOM.
    """
    text = Path(meminfo_path).read_text()
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Malformed MemAvailable line in {meminfo_path}: {line!r}")
            return float(parts[1])
    raise ValueError(f"MemAvailable not found in {meminfo_path}")


def wait_for_memory(
    read_fn: Callable[[], float],
    sleep_fn: Callable[[float], None],
    threshold_kb: float,
    retries: int = 5,
    poll_interval_s: float = 30.0,
    log_fn: Callable[[str], None] = print,
) -> bool:
    """True once `read_fn()` (MemAvailable, kB) is >= `threshold_kb`.

    Retries up to `retries` times, sleeping `poll_interval_s` between reads.
    Returns False (caller should skip the stage rather than risk another
    OOM) if it never clears the threshold within the retry budget. A read
    that raises is treated as "still below threshold" (fail closed) rather
    than propagating — a stage should never crash the whole runner because
    /proc/meminfo hiccuped.
    """
    for attempt in range(retries + 1):
        try:
            available = read_fn()
        except Exception as exc:  # noqa: BLE001 - fail closed, keep retrying
            log_fn(f"Memory guard: could not read MemAvailable ({exc}) — treating as below threshold.")
            available = -1.0
        if available >= threshold_kb:
            if attempt > 0:
                log_fn(f"Memory guard: MemAvailable recovered to {available / 1e6:.2f} GB after {attempt} retries.")
            return True
        log_fn(
            f"Memory guard: MemAvailable {available / 1e6:.2f} GB < threshold "
            f"{threshold_kb / 1e6:.2f} GB (attempt {attempt + 1}/{retries + 1})."
        )
        if attempt < retries:
            sleep_fn(poll_interval_s)
    return False


def maybe_set_cuda_memory_fraction(fraction: float = 0.75) -> None:
    """Cap this process's CUDA allocator to `fraction` of GPU memory, if CUDA
    is available — so a single stage subprocess can never claim the whole
    8GB card. Never raises (best-effort; older torch/driver combos can lack
    the call)."""
    if torch.cuda.is_available():
        try:
            torch.cuda.set_per_process_memory_fraction(fraction)
        except Exception:
            pass


def diagnostics_line(epoch: int, t_stage_start: float, meminfo_path: Path = DEFAULT_MEMINFO_PATH) -> str:
    """One-line MemAvailable / GPU memory / elapsed-time snapshot, logged once
    per epoch so the *next* crash is diagnosable (this is literally why the
    earlier run couldn't be debugged — nothing was logging system state)."""
    try:
        mem_kb = read_mem_available_kb(meminfo_path)
        mem_str = f"{mem_kb / 1e6:.2f}GB"
    except Exception:
        mem_str = "n/a"
    gpu_str = "n/a"
    if torch.cuda.is_available():
        try:
            gpu_str = f"{torch.cuda.memory_allocated() / 1e6:.0f}MB alloc / {torch.cuda.memory_reserved() / 1e6:.0f}MB reserved"
        except Exception:
            pass
    elapsed_min = (time.time() - t_stage_start) / 60.0
    return f"[diag] epoch={epoch} MemAvailable={mem_str} GPUmem={gpu_str} stage_elapsed={elapsed_min:.1f}min"


# --------------------------------------------------------------------------
# Stage-state JSON round-trip. Only the orchestrator process
# (scripts/12_overnight/run_overnight.py, not run with --run-stage) ever
# calls save_state — each stage subprocess reports back through its own CSV
# / checkpoint files plus a small per-stage result JSON, never by writing
# this file directly, so there is exactly one writer at all times even
# though stages run in separate processes.
# --------------------------------------------------------------------------

DEFAULT_STATE_PATH = Path("reports/serial_state.json")
STAGE_NAMES: list[str] = ["kfold_fold4", "longrun", "warmstart_tune", "elimination_sweep", "final_eval"]


def default_state(stage_names: list[str] = STAGE_NAMES) -> dict:
    return {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {name: {"status": "pending"} for name in stage_names},
        "checkpoints": {},
        "timings": {},
    }


def load_state(path: Path = DEFAULT_STATE_PATH) -> dict:
    if not path.exists():
        return default_state()
    return json.loads(path.read_text())


def save_state(state: dict, path: Path = DEFAULT_STATE_PATH) -> None:
    """Atomic write: write to a temp file, then `os.replace` it over the real
    path. `os.replace` is atomic on POSIX (rename(2)), so a crash at any
    point up to (and including) the write to the temp file leaves the real
    `path` completely untouched — the temp file is simply orphaned garbage,
    never a half-written state file that --resume would trip over."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, path)


def mark_stage(state: dict, stage: str, status: str, **extra) -> dict:
    entry = state.setdefault("stages", {}).setdefault(stage, {})
    entry["status"] = status
    entry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    entry.update(extra)
    return state


def stage_needs_to_run(state: dict, stage: str) -> bool:
    """Resume semantics: any status starting with "done" (e.g. "done" or
    "done (budget)") is skipped. Everything else — "pending", "running" (a
    crash-interrupted stage), "failed", "skipped", "dry_run_ok" — is treated
    as needing (re)work; each stage's own runner decides whether that means
    resuming from its own checkpoint or starting over."""
    status = state.get("stages", {}).get(stage, {}).get("status", "pending")
    return not status.startswith("done")


# --------------------------------------------------------------------------
# `longrun` stage: ResumableLongrunTrainer
# --------------------------------------------------------------------------


@dataclass
class ResumeInfo:
    epoch: int
    best_val_auroc: float
    patience_counter: int


def save_resume_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_auroc: float,
    patience_counter: int,
) -> None:
    """Atomic (temp + os.replace) resume checkpoint written every epoch —
    model + optimizer + epoch + best-val + patience counter, everything
    needed to continue training as if the process never died."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_auroc": best_val_auroc,
            "patience_counter": patience_counter,
        },
        tmp,
    )
    os.replace(tmp, path)


def load_resume_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device
) -> ResumeInfo:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ResumeInfo(
        epoch=ckpt["epoch"],
        best_val_auroc=ckpt["best_val_auroc"],
        patience_counter=ckpt["patience_counter"],
    )


class ResumableLongrunTrainer(Trainer):
    """`Trainer` subclass for the `longrun` stage: linear warmup + cosine
    decay instead of `ReduceLROnPlateau`, a periodic epoch checkpoint, a
    best-val checkpoint, and a resume checkpoint (model + optimizer + epoch
    + best-val + patience) written every single epoch so a crash loses at
    most one epoch of work, not the whole stage.

    Reuses `Trainer._train_epoch` (the per-epoch train loop) unchanged;
    everything scheduler/checkpoint/logging/resume-related is new here.
    `fit()` is fully overridden (not calling `super().fit()`), same
    rationale as the earlier `WarmupCosineTrainer` it replaces: the parent's
    `ReduceLROnPlateau` step and its own patience/checkpoint bookkeeping is
    exactly what this subclass replaces.
    """

    def __init__(
        self,
        model: ECGConvNet,
        config: TrainConfig,
        pos_weights: torch.Tensor | None = None,
        label_names: list[str] | None = None,
        checkpoint_name: str = "longrun_best",
        warmup_epochs: int = 3,
        end_lr_fraction: float = 0.01,
        checkpoint_every: int = 5,
        epoch_checkpoint_dir: Path | None = None,
        epoch_checkpoint_prefix: str = "longrun_epoch",
        resume_checkpoint_path: Path | None = None,
        on_epoch_end: Callable[[dict], None] | None = None,
        stage_deadline_ts: float = float("inf"),
        track_label: str | None = None,
    ):
        super().__init__(model, config, pos_weights, label_names, checkpoint_name)
        self.warmup_epochs = warmup_epochs
        self.track_label = track_label
        self.end_lr = config.learning_rate * end_lr_fraction
        self.checkpoint_every = checkpoint_every
        self.epoch_checkpoint_dir = epoch_checkpoint_dir or config.checkpoint_dir
        self.epoch_checkpoint_prefix = epoch_checkpoint_prefix
        self.resume_checkpoint_path = resume_checkpoint_path or (self.epoch_checkpoint_dir / "longrun_last.pt")
        self.on_epoch_end = on_epoch_end
        self.stage_deadline_ts = stage_deadline_ts
        self.epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 1
        self.best_val_auroc = -1.0
        self.patience_counter = 0
        self._resumed_from: int | None = None
        if self.resume_checkpoint_path.exists():
            info = load_resume_checkpoint(self.resume_checkpoint_path, self.model, self.optimizer, self.device)
            self.start_epoch = info.epoch + 1
            self.best_val_auroc = info.best_val_auroc
            self.patience_counter = info.patience_counter
            self._resumed_from = info.epoch

    def lr_for_epoch(self, epoch_idx: int) -> float:
        return warmup_cosine_lr(
            epoch_idx, self.config.learning_rate, self.warmup_epochs, self.config.n_epochs, self.end_lr
        )

    def _epoch_checkpoint_path(self, epoch: int) -> Path:
        return self.epoch_checkpoint_dir / f"{self.epoch_checkpoint_prefix}{epoch}.pt"

    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        import pandas as pd

        stopped_reason = "completed_all_epochs"
        t_stage_start = time.time()
        log: list[dict] = []

        if self._resumed_from is not None:
            print(
                f"[longrun] Resuming from epoch {self._resumed_from} "
                f"(best val mean-AUROC so far={self.best_val_auroc:.4f}, patience={self.patience_counter})",
                flush=True,
            )
        print(
            f"[longrun] Training on {self.device} | {len(train_loader.dataset):,} train, "
            f"{len(val_loader.dataset):,} val | warmup={self.warmup_epochs} total={self.config.n_epochs} "
            f"patience={self.config.patience} start_epoch={self.start_epoch}",
            flush=True,
        )

        if self.start_epoch > self.config.n_epochs:
            stopped_reason = "already_complete_on_resume"

        for epoch in range(self.start_epoch, self.config.n_epochs + 1):
            lr = self.lr_for_epoch(epoch - 1)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            train_loss = self._train_epoch(train_loader, epoch)
            val_metrics = evaluate_loader(self.model, val_loader, self.device, self.label_names)
            val_auroc = mean_auroc(val_metrics)

            row = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_mean_auroc": round(val_auroc, 6),
                "lr": lr,
            }
            if self.track_label is not None:
                match = val_metrics[val_metrics["label"] == self.track_label]
                row["val_shd_auroc"] = float(match["auroc"].iloc[0]) if not match.empty else float("nan")

            print(diagnostics_line(epoch, t_stage_start), flush=True)
            row["mem_available_gb"] = None
            try:
                row["mem_available_gb"] = round(read_mem_available_kb() / 1e6, 3)
            except Exception:
                pass
            row["elapsed_s"] = round(time.time() - t_stage_start, 1)

            print(
                f"[longrun] Epoch {epoch:3d}/{self.config.n_epochs} | "
                f"loss={train_loss:.4f} | val mean-AUROC={val_auroc:.4f} | lr={lr:.2e}",
                flush=True,
            )
            log.append(row)
            if self.on_epoch_end is not None:
                self.on_epoch_end(row)

            improved = val_auroc > self.best_val_auroc
            if improved:
                self.best_val_auroc = val_auroc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.patience_counter += 1

            if epoch % self.checkpoint_every == 0:
                torch.save(self.model.state_dict(), self._epoch_checkpoint_path(epoch))
                print(f"[longrun]   periodic checkpoint -> {self._epoch_checkpoint_path(epoch)}", flush=True)

            # Written every epoch, unconditionally — this is what makes the
            # stage resumable mid-training rather than only at 5-epoch
            # boundaries.
            save_resume_checkpoint(
                self.resume_checkpoint_path, self.model, self.optimizer, epoch, self.best_val_auroc,
                self.patience_counter,
            )

            if self.patience_counter >= self.config.patience:
                stopped_reason = f"early_stopping(patience={self.config.patience})"
                print(f"[longrun] Early stopping at epoch {epoch} (best val AUROC={self.best_val_auroc:.4f})", flush=True)
                break

            if time.time() > self.stage_deadline_ts:
                stopped_reason = "budget_exceeded"
                print(f"[longrun] Stage budget exceeded after epoch {epoch} — stopping cleanly.", flush=True)
                break

            try:
                if read_mem_available_kb() < DEFAULT_MEM_THRESHOLD_GB * 1e6:
                    stopped_reason = "memory_guard"
                    print(f"[longrun] MemAvailable below guard threshold after epoch {epoch} — stopping cleanly.", flush=True)
                    break
            except Exception:
                pass

        if self.checkpoint_path.exists():
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

        return pd.DataFrame(log), stopped_reason, self.best_val_auroc


# --------------------------------------------------------------------------
# `elimination_sweep` stage: from-scratch Optuna search space + trial loop
# --------------------------------------------------------------------------

ELIMINATION_SPACE = {
    "learning_rate": {"low": 1e-5, "high": 3e-3, "log": True},
    "weight_decay": {"low": 1e-6, "high": 1e-2, "log": True},
    "dropout": {"low": 0.1, "high": 0.6, "log": False},
    "batch_size": {"choices": [16, 32, 64]},
}


def sample_elimination_params(trial: optuna.trial.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", **ELIMINATION_SPACE["learning_rate"]),
        "weight_decay": trial.suggest_float("weight_decay", **ELIMINATION_SPACE["weight_decay"]),
        "dropout": trial.suggest_float(
            "dropout", ELIMINATION_SPACE["dropout"]["low"], ELIMINATION_SPACE["dropout"]["high"]
        ),
        "batch_size": trial.suggest_categorical("batch_size", ELIMINATION_SPACE["batch_size"]["choices"]),
    }


def train_from_scratch_with_pruning(
    params: dict,
    train_ds: Dataset,
    val_ds: Dataset,
    pos_weights: torch.Tensor,
    label_names: list[str],
    n_leads: int,
    n_demo_features: int,
    n_epochs: int,
    device: str,
    num_workers: int,
    trial: optuna.trial.Trial | None = None,
    on_epoch_end: Callable[[int, float, float], None] | None = None,
    epoch_deadline_ts: float = float("inf"),
) -> float:
    """From-scratch training loop with a per-epoch `trial.report` +
    `should_prune` check, for `elimination_sweep`'s SuccessiveHalvingPruner.
    Built on `Trainer._train_epoch` / `src.metrics.evaluate_loader` directly
    (not `Trainer.fit`, which never reports intermediate values, so pruning
    could never fire).

    Returns the final (or last-seen, if the deadline cuts the run short) val
    mean-AUROC.
    """
    config = TrainConfig(
        n_epochs=n_epochs,
        batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        patience=n_epochs + 1,  # patience is irrelevant here; pruning/deadline decide when to stop
        num_workers=num_workers,
        device=device,
    )
    model = ECGConvNet(
        n_leads=n_leads, n_labels=len(label_names), n_demo_features=n_demo_features, dropout=params["dropout"],
    )
    trainer = Trainer(
        model=model, config=config, pos_weights=pos_weights, label_names=label_names,
        checkpoint_name=f"elimination_scratch_trial{trial.number if trial is not None else 'x'}",
    )
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, config)

    t_stage_start = time.time()
    val_auroc = float("nan")
    for epoch in range(1, n_epochs + 1):
        trainer._train_epoch(train_loader, epoch)
        val_metrics = evaluate_loader(trainer.model, val_loader, trainer.device, label_names)
        val_auroc = mean_auroc(val_metrics)
        print(diagnostics_line(epoch, t_stage_start), flush=True)

        if on_epoch_end is not None:
            on_epoch_end(epoch, val_auroc, config.learning_rate)

        if trial is not None:
            trial.report(val_auroc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if time.time() > epoch_deadline_ts:
            break

    return val_auroc


# --------------------------------------------------------------------------
# `warmstart_tune` stage: warm-started refinement search space + trial loop
# --------------------------------------------------------------------------

WARMSTART_SPACE = {
    "lr_multiplier": {"low": 0.1, "high": 3.0, "log": True},
    "weight_decay": {"low": 1e-6, "high": 1e-2, "log": True},
    "dropout": {"low": 0.1, "high": 0.6, "log": False},
    "end_lr_fraction": {"low": 0.001, "high": 0.5, "log": True},
}


def sample_warmstart_params(trial: optuna.trial.Trial) -> dict:
    return {
        "lr_multiplier": trial.suggest_float("lr_multiplier", **WARMSTART_SPACE["lr_multiplier"]),
        "weight_decay": trial.suggest_float("weight_decay", **WARMSTART_SPACE["weight_decay"]),
        "dropout": trial.suggest_float(
            "dropout", WARMSTART_SPACE["dropout"]["low"], WARMSTART_SPACE["dropout"]["high"]
        ),
        "end_lr_fraction": trial.suggest_float("end_lr_fraction", **WARMSTART_SPACE["end_lr_fraction"]),
    }


def load_warm_start_model(
    checkpoint_path: Path,
    n_leads: int,
    n_labels: int,
    n_demo_features: int,
    dropout: float,
    device: torch.device,
) -> ECGConvNet:
    """A fresh ECGConvNet with a (possibly different) dropout rate, loaded
    with a `longrun` checkpoint's weights. Safe because `nn.Dropout` has no
    learned parameters, so the state_dict's keys/shapes are identical
    regardless of the dropout probability used to construct the module."""
    model = ECGConvNet(n_leads=n_leads, n_labels=n_labels, n_demo_features=n_demo_features, dropout=dropout)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def run_warm_started_epochs(
    model: ECGConvNet,
    base_lr: float,
    weight_decay: float,
    end_lr: float,
    n_epochs: int,
    train_ds: Dataset,
    val_ds: Dataset,
    pos_weights: torch.Tensor,
    label_names: list[str],
    batch_size: int,
    num_workers: int,
    device: str,
    trial: optuna.trial.Trial | None = None,
    on_epoch_end: Callable[[int, float, float], None] | None = None,
    epoch_deadline_ts: float = float("inf"),
    checkpoint_path: Path | None = None,
) -> tuple[float, float]:
    """Continue training `model` (already warm-started) for `n_epochs` with a
    cosine decay from `base_lr` to `end_lr` (no warmup — the run is already
    past its warmup phase). Used for both a `warmstart_tune` trial and its
    control (continuing the default schedule): the caller picks
    `base_lr`/`weight_decay`/`end_lr`/`model.head`-dropout accordingly.

    Returns (final val mean-AUROC, final val SHD-composite AUROC).
    """
    from src.crossval import DEFAULT_STRATIFY_COL

    config = TrainConfig(
        n_epochs=n_epochs, batch_size=batch_size, learning_rate=base_lr, weight_decay=weight_decay,
        patience=n_epochs + 1, num_workers=num_workers, device=device,
    )
    trainer = Trainer(
        model=model, config=config, pos_weights=pos_weights, label_names=label_names,
        checkpoint_name=f"warmstart_trial{trial.number if trial is not None else 'control'}",
    )
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, config)

    t_stage_start = time.time()
    val_auroc = float("nan")
    shd_auroc = float("nan")
    best_auroc = -1.0
    for epoch in range(1, n_epochs + 1):
        lr = warmup_cosine_lr(epoch - 1, base_lr, warmup_epochs=0, total_epochs=n_epochs, end_lr=end_lr)
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = lr

        trainer._train_epoch(train_loader, epoch)
        val_metrics = evaluate_loader(trainer.model, val_loader, trainer.device, label_names)
        val_auroc = mean_auroc(val_metrics)
        shd_row = val_metrics[val_metrics["label"] == DEFAULT_STRATIFY_COL]
        shd_auroc = float(shd_row["auroc"].iloc[0]) if not shd_row.empty else float("nan")
        print(diagnostics_line(epoch, t_stage_start), flush=True)

        if checkpoint_path is not None and val_auroc > best_auroc:
            best_auroc = val_auroc
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(trainer.model.state_dict(), checkpoint_path)

        if on_epoch_end is not None:
            on_epoch_end(epoch, val_auroc, lr)

        if trial is not None:
            trial.report(val_auroc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if time.time() > epoch_deadline_ts:
            break

    return val_auroc, shd_auroc


def nearest_checkpoint_epoch(epochs_run: int, checkpoint_every: int, fraction: float = 0.75) -> int:
    """Nearest multiple of `checkpoint_every` to `fraction * epochs_run`, clamped
    into [checkpoint_every, epochs_run rounded down to a multiple of checkpoint_every]."""
    if epochs_run < checkpoint_every:
        return 0
    target = fraction * epochs_run
    n_steps = round(target / checkpoint_every)
    n_steps = max(1, n_steps)
    max_available = epochs_run // checkpoint_every
    n_steps = min(n_steps, max_available)
    return n_steps * checkpoint_every
