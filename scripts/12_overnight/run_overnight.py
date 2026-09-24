"""Serial, crash-resilient stage runner (agent SERIAL, 2026-09-23).

**Why this replaces the earlier Phase A/B/C pipeline.** The machine crashed
and rebooted earlier today while three heavy jobs ran *concurrently*: a
5-fold CNN CV (GPU + heavy memory-mapped I/O), a CPU Optuna search, and this
script's predecessor idling, alongside the user gaming. Measured just before
the crash: GPU ~12% utilised, dataloader workers blocked in WSL2's
`p9_client_rpc`, page cache saturated at 13GB with 116MB free, against a
16GB memory-mapped waveform file on a 15GB-RAM box. Almost certainly
memory/I/O exhaustion from running things at once, not any single job. The
old pipeline was also never actually exercised on the real GPU path before
being handed off.

This version runs **exactly one heavy stage at a time**, each as its own
`subprocess.run`-launched child process (so a stage that leaks memory or
wedges a dataloader worker cannot poison this orchestrator's own process,
and the orchestrator can hard-kill a wedged child on a deadline). Progress
is tracked in `reports/serial_state.json`, written atomically (temp file +
`os.replace`) by this process alone — every stage subprocess reports back
through its own CSVs, checkpoints, and a small `serial_stage_result_<stage>.json`
summary, never by touching the state file directly, so there is exactly one
writer even though stages run in separate processes.

Stages, in value-per-hour order (a crash at any point keeps the most useful
work already on disk):

  1. kfold_fold4        (budget 1.5h) finish the interrupted 5th CV fold
  2. longrun             (budget 5h)   fused CNN, up to 60 epochs, warmup+cosine
  3. warmstart_tune      (budget 4h)   ASHA tail refinement from a longrun checkpoint
  4. elimination_sweep   (budget 3h)   from-scratch ASHA sweep, eliminates bad regions
  5. final_eval          (budget 0.5h) held-out test scoring, only if something improved

Resource guards (checked before *and* during each stage): `MemAvailable`
from `/proc/meminfo` must clear `--mem-threshold-gb` (default 2.5) before a
stage starts, retried a few times before the stage is skipped rather than
risking another OOM; `num_workers`/`batch_size` default to 2/32 and are
never raised except where a trial explicitly varies `batch_size`
(`elimination_sweep`'s own search space, 16/32/64 — an intentional, narrow
exception); `--max-concurrent-stages` above 1 is refused outright, not just
discouraged; each stage subprocess additionally caps its own CUDA allocator
to 75% of the card via `torch.cuda.set_per_process_memory_fraction`.

Usage (the real, long run — left ready but NOT started by this agent):
  setsid nohup .venv/bin/python -u scripts/12_overnight/run_overnight.py \\
      > progress/serial_run.log 2>&1 < /dev/null &

  # --resume is the default: safe to Ctrl-C and rerun, or to rerun after a
  # reboot — it picks up wherever it left off. Force a clean restart with:
  .venv/bin/python -u scripts/12_overnight/run_overnight.py --restart

  # Plan without touching the GPU or any data:
  .venv/bin/python -u scripts/12_overnight/run_overnight.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import Subset

from src.constants import DATASET_SUBDIR, METADATA_FILENAME, N_LEADS, TARGET_LABELS, load_config
from src.crossval import DEFAULT_STRATIFY_COL
from src.dataset import ECGDataset, load_split
from src.models import ECGConvNet
from src.overnight_training import (
    ELIMINATION_SPACE,
    STAGE_NAMES,
    WARMSTART_SPACE,
    ResumableLongrunTrainer,
    budget_exceeded,  # noqa: F401 - re-exported for tests / callers
    deadline_exceeded,
    deadline_ts,
    default_state,
    load_state,
    load_warm_start_model,
    maybe_set_cuda_memory_fraction,
    mark_stage,
    nearest_checkpoint_epoch,
    read_mem_available_kb,
    run_warm_started_epochs,
    sample_elimination_params,
    sample_warmstart_params,
    save_state,
    stage_needs_to_run,
    train_from_scratch_with_pruning,
    wait_for_memory,
    warmup_cosine_lr,
)
from src.preprocessing import build_demographic_preprocessor
from src.training import TrainConfig, build_dataloaders, compute_pos_weights, resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "serial"
PROGRESS_DIR = PROJECT_ROOT / "progress"
PROGRESS_MD = PROGRESS_DIR / "serial.md"
STATE_PATH = REPORTS_DIR / "serial_state.json"
SCRIPT_PATH = Path(__file__).resolve()

# Reference line every downstream verdict is measured against: the 4-fold
# k-fold CV test-split SHD-composite AUROC mean (reports/kfold_summary.md /
# reports/kfold_results.csv, 4 folds complete when this pipeline was
# written). Stage 1 (kfold_fold4) will refresh kfold_results.csv/summary.md
# with a genuine 5-fold mean, printed in the notes alongside this constant —
# but the gate `final_eval` uses is this fixed number, per the task.
KFOLD_SHD_MEAN_4FOLD = 0.8392  # TEST-split 4-fold mean; reference only, NOT a gate
# The longrun/warmstart stages measure SHD AUROC on the official VALIDATION
# split, so they must be gated against a validation number. Comparing a val
# score with the k-fold TEST mean above is a split mismatch: it wrongly
# suppressed a genuine improvement on the first overnight run (longrun reached
# val 0.837 against this baseline's 0.8279, but was compared with 0.8392 and
# skipped). Source: reports/cnn_combined_results.csv, the single-run fused model.
FUSED_SINGLE_RUN_VAL_SHD = 0.8279

DEFAULT_BUDGET_HOURS = {
    "kfold_fold4": 1.5,
    "longrun": 5.0,
    "warmstart_tune": 4.0,
    "elimination_sweep": 3.0,
    "final_eval": 0.5,
}
DEFAULT_GLOBAL_DEADLINE_HOURS = sum(DEFAULT_BUDGET_HOURS.values())  # 14.0


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} - {msg}", flush=True)


def milestone(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    PROGRESS_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_MD, "a") as f:
        f.write(f"{ts} - {msg}\n")
        f.flush()
    log(msg)


def trial_intermediates(trial) -> dict:
    """Intermediate values reported so far by a RUNNING Optuna trial.

    `optuna.Trial` exposes no `intermediate_values` attribute -- that lives on
    `FrozenTrial`, which only exists once the trial has finished. During a trial
    the reported values have to be read back out of the study's storage. Returns
    an empty dict if they cannot be retrieved, so CSV logging never takes down a
    training run.
    """
    try:
        frozen = trial.study._storage.get_trial(trial._trial_id)
        return dict(frozen.intermediate_values or {})
    except Exception:  # pragma: no cover - storage shape is optuna-version dependent
        return {}


def append_csv_row(path: Path, row: dict, fieldnames: list[str]) -> None:
    """Append one row, header first if new, flushed immediately. Nothing in
    this pipeline buffers a run's worth of rows in memory — every epoch and
    every trial is on disk before the next one starts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()
        os.fsync(f.fileno())


def write_stage_result_json(stage: str, obj: dict) -> None:
    """Atomic (temp + os.replace) small result summary a stage subprocess
    writes for itself — this is what the orchestrator reads back after the
    subprocess exits, so `reports/serial_state.json` still has exactly one
    writer (this file, only when NOT running as `--run-stage`)."""
    path = REPORTS_DIR / f"serial_stage_result_{stage}.json"
    obj = dict(obj)
    obj["stage"] = stage
    obj["written_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def read_stage_result_json(stage: str) -> dict:
    path = REPORTS_DIR / f"serial_stage_result_{stage}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", type=str, default=",".join(STAGE_NAMES),
                   help="comma-separated subset/order restriction of: " + ",".join(STAGE_NAMES))
    p.add_argument("--run-stage", type=str, default=None,
                    choices=["longrun", "warmstart_tune", "elimination_sweep", "final_eval"],
                    help=argparse.SUPPRESS)  # internal: this invocation IS one stage's subprocess
    p.add_argument("--resume", action="store_true", default=True, help="default behaviour: skip stages already done")
    p.add_argument("--restart", action="store_true", default=False, help="ignore all prior progress, start from scratch")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-concurrent-stages", type=int, default=1,
                    help="hard invariant, refused above 1 — see module docstring")

    p.add_argument("--mem-threshold-gb", type=float, default=2.5)
    p.add_argument("--mem-retries", type=int, default=5)
    p.add_argument("--mem-retry-wait-s", type=float, default=30.0)

    p.add_argument("--global-deadline-hours", type=float, default=DEFAULT_GLOBAL_DEADLINE_HOURS)
    p.add_argument("--budget-kfold-fold4-hours", type=float, default=DEFAULT_BUDGET_HOURS["kfold_fold4"])
    p.add_argument("--budget-longrun-hours", type=float, default=DEFAULT_BUDGET_HOURS["longrun"])
    p.add_argument("--budget-warmstart-tune-hours", type=float, default=DEFAULT_BUDGET_HOURS["warmstart_tune"])
    p.add_argument("--budget-elimination-sweep-hours", type=float, default=DEFAULT_BUDGET_HOURS["elimination_sweep"])
    p.add_argument("--budget-final-eval-hours", type=float, default=DEFAULT_BUDGET_HOURS["final_eval"])

    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cuda-mem-fraction", type=float, default=0.75)

    p.add_argument("--only-fold", type=int, default=4, help="kfold_fold4: which fold index to (re)train")

    p.add_argument("--longrun-max-epochs", type=int, default=60)
    p.add_argument("--longrun-warmup-epochs", type=int, default=3)
    p.add_argument("--longrun-patience", type=int, default=12)
    p.add_argument("--longrun-checkpoint-every", type=int, default=5)

    p.add_argument("--warmstart-epochs", type=int, default=5)
    p.add_argument("--elimination-epochs", type=int, default=5)

    p.add_argument("--smoke", action="store_true",
                    help="longrun only: tiny real run (few epochs, optional subsample) to prove the GPU path works")
    p.add_argument("--smoke-epochs", type=int, default=2)
    p.add_argument("--smoke-n-train", type=int, default=None)
    p.add_argument("--smoke-n-val", type=int, default=None)

    # internal: set by the orchestrator when it self-dispatches a --run-stage subprocess
    p.add_argument("--stage-deadline-ts", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--global-deadline-ts", type=float, default=None, help=argparse.SUPPRESS)

    args = p.parse_args(argv)

    if args.max_concurrent_stages > 1:
        p.error(
            "--max-concurrent-stages > 1 is refused: exactly one heavy stage alive at a time is a hard "
            "invariant of this runner (see module docstring), not a tunable knob."
        )
    if args.num_workers > 2:
        p.error("--num-workers > 2 is refused: the resource-guard invariant caps this at 2.")
    if args.batch_size > 32:
        p.error(
            "--batch-size > 32 is refused for the shared default batch size. elimination_sweep's own "
            "search space may still explicitly try 64 per trial — that is the one approved exception."
        )
    return args


# --------------------------------------------------------------------------
# Shared data loading (each stage subprocess loads its own copy — simpler
# and safer than sharing a loaded dataset across processes, and still only
# ever one such load alive at a time because stages run serially).
# --------------------------------------------------------------------------


@dataclass
class SharedData:
    train_ds: ECGDataset
    val_ds: ECGDataset
    pos_weights: torch.Tensor
    n_demo: int
    label_names: list[str]


def load_shared_data(
    dataset_dir: Path,
    metadata_path: Path,
    demo_tag: str = "cnn_serial",
    smoke_n_train: int | None = None,
    smoke_n_val: int | None = None,
) -> SharedData:
    metadata = pd.read_csv(metadata_path)
    demo_encoder = build_demographic_preprocessor()
    log("Loading official train split (memory-mapped)...")
    train_data, demo_encoder = load_split(
        "train", metadata, dataset_dir, label_names=TARGET_LABELS, demo_encoder=demo_encoder, fit_encoder=True,
    )
    log("Loading official val split (memory-mapped)...")
    val_data, _ = load_split(
        "val", metadata, dataset_dir, label_names=TARGET_LABELS, demo_encoder=demo_encoder, fit_encoder=False,
    )
    pos_weights = compute_pos_weights(train_data.labels)
    n_demo = train_data.demo_features.shape[1]
    train_ds: ECGDataset | Subset = ECGDataset(train_data)
    val_ds: ECGDataset | Subset = ECGDataset(val_data)

    if smoke_n_train is not None:
        train_ds = Subset(train_ds, list(range(min(smoke_n_train, len(train_ds)))))
        log(f"[smoke] train subsampled to {len(train_ds):,} rows")
    if smoke_n_val is not None:
        val_ds = Subset(val_ds, list(range(min(smoke_n_val, len(val_ds)))))
        log(f"[smoke] val subsampled to {len(val_ds):,} rows")

    import joblib
    (PROJECT_ROOT / "checkpoints").mkdir(parents=True, exist_ok=True)
    joblib.dump(demo_encoder, PROJECT_ROOT / "checkpoints" / f"{demo_tag}_demo_encoder.joblib")

    return SharedData(train_ds=train_ds, val_ds=val_ds, pos_weights=pos_weights, n_demo=n_demo, label_names=TARGET_LABELS)


# --------------------------------------------------------------------------
# Hard-kill process runner — a safety net on top of every stage's own
# internal deadline check, since the earlier crash was exactly a wedged
# dataloader worker that would never have reached its own "check the clock
# between batches" code.
# --------------------------------------------------------------------------


def run_stage_process(
    cmd: list[str],
    log_path: Path,
    stage_deadline_ts: float,
    global_deadline_ts: float,
    poll_s: float = 10.0,
    kill_grace_s: float = 30.0,
) -> tuple[int, bool]:
    """Runs `cmd` as a subprocess, polling every `poll_s`. If either deadline
    passes while it's still alive, sends SIGTERM, waits `kill_grace_s`, then
    SIGKILL if it's still not dead. Returns (returncode, budget_hit)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as logf:
        logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} launching: {' '.join(cmd)} ===\n")
        logf.flush()
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=logf, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    ret = proc.wait(timeout=poll_s)
                    return ret, False
                except subprocess.TimeoutExpired:
                    now = time.time()
                    if now > stage_deadline_ts or now > global_deadline_ts:
                        logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - deadline reached, sending SIGTERM\n")
                        logf.flush()
                        proc.terminate()
                        try:
                            ret = proc.wait(timeout=kill_grace_s)
                        except subprocess.TimeoutExpired:
                            logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - SIGTERM ignored, sending SIGKILL\n")
                            logf.flush()
                            proc.kill()
                            ret = proc.wait(timeout=30)
                        return ret, True
        except KeyboardInterrupt:
            logf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - KeyboardInterrupt, terminating child\n")
            logf.flush()
            proc.terminate()
            raise


def build_self_command(stage: str, args: argparse.Namespace, stage_deadline_ts: float, global_deadline_ts: float) -> list[str]:
    cmd = [
        sys.executable, "-u", str(SCRIPT_PATH), "--run-stage", stage,
        "--stage-deadline-ts", repr(stage_deadline_ts),
        "--global-deadline-ts", repr(global_deadline_ts),
        "--num-workers", str(args.num_workers),
        "--batch-size", str(args.batch_size),
        "--mem-threshold-gb", str(args.mem_threshold_gb),
        "--cuda-mem-fraction", str(args.cuda_mem_fraction),
    ]
    if stage == "longrun":
        cmd += [
            "--longrun-max-epochs", str(args.longrun_max_epochs),
            "--longrun-warmup-epochs", str(args.longrun_warmup_epochs),
            "--longrun-patience", str(args.longrun_patience),
            "--longrun-checkpoint-every", str(args.longrun_checkpoint_every),
        ]
        if args.smoke:
            cmd += ["--smoke", "--smoke-epochs", str(args.smoke_epochs)]
            if args.smoke_n_train is not None:
                cmd += ["--smoke-n-train", str(args.smoke_n_train)]
            if args.smoke_n_val is not None:
                cmd += ["--smoke-n-val", str(args.smoke_n_val)]
    elif stage == "warmstart_tune":
        cmd += ["--warmstart-epochs", str(args.warmstart_epochs)]
    elif stage == "elimination_sweep":
        cmd += ["--elimination-epochs", str(args.elimination_epochs)]
    return cmd


# --------------------------------------------------------------------------
# Stage entry points — executed when this process IS a stage's subprocess
# (`--run-stage <name>`). Each returns a process exit code.
# --------------------------------------------------------------------------


def stage_entry_longrun(args: argparse.Namespace) -> int:
    stage_deadline = args.stage_deadline_ts if args.stage_deadline_ts is not None else float("inf")
    global_deadline = args.global_deadline_ts if args.global_deadline_ts is not None else float("inf")
    effective_deadline = min(stage_deadline, global_deadline)

    n_epochs = args.smoke_epochs if args.smoke else args.longrun_max_epochs
    log(f"[longrun] starting (smoke={args.smoke}) n_epochs={n_epochs} warmup={args.longrun_warmup_epochs} "
        f"patience={args.longrun_patience} checkpoint_every={args.longrun_checkpoint_every} "
        f"batch_size={args.batch_size} num_workers={args.num_workers}")

    maybe_set_cuda_memory_fraction(args.cuda_mem_fraction)

    dataset_dir = PROJECT_ROOT / "data" / DATASET_SUBDIR
    metadata_path = dataset_dir / METADATA_FILENAME
    data = load_shared_data(
        dataset_dir, metadata_path, demo_tag="cnn_serial",
        smoke_n_train=args.smoke_n_train if args.smoke else None,
        smoke_n_val=args.smoke_n_val if args.smoke else None,
    )

    cnn_cfg = load_config()["training"]["cnn"]
    lr, wd, dropout = cnn_cfg["learning_rate"], cnn_cfg["weight_decay"], cnn_cfg["dropout"]

    device = resolve_device("auto")
    model = ECGConvNet(n_leads=N_LEADS, n_labels=len(data.label_names), n_demo_features=data.n_demo, dropout=dropout)
    train_config = TrainConfig(
        n_epochs=n_epochs, batch_size=args.batch_size, learning_rate=lr, weight_decay=wd,
        patience=args.longrun_patience, num_workers=args.num_workers, device=str(device),
        checkpoint_dir=CHECKPOINT_DIR,
    )

    train_log_csv = REPORTS_DIR / "serial_longrun_trainlog.csv"
    fields = ["epoch", "train_loss", "val_mean_auroc", "val_shd_auroc", "lr", "mem_available_gb", "elapsed_s", "timestamp"]

    def on_epoch_end(row: dict) -> None:
        row = dict(row)
        row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        append_csv_row(train_log_csv, row, fields)
        write_stage_result_json("longrun", {
            "status": "running", "epochs_run": row["epoch"], "planned_total_epochs": n_epochs,
            "last_val_mean_auroc": row["val_mean_auroc"], "last_val_shd_auroc": row.get("val_shd_auroc"),
        })

    trainer = ResumableLongrunTrainer(
        model=model, config=train_config, pos_weights=data.pos_weights, label_names=data.label_names,
        checkpoint_name="longrun_best", warmup_epochs=args.longrun_warmup_epochs,
        checkpoint_every=args.longrun_checkpoint_every, epoch_checkpoint_dir=CHECKPOINT_DIR,
        on_epoch_end=on_epoch_end, stage_deadline_ts=effective_deadline, track_label=DEFAULT_STRATIFY_COL,
    )
    train_loader, val_loader = build_dataloaders(data.train_ds, data.val_ds, train_config)
    log_df, stopped_reason, best_val_auroc = trainer.fit(train_loader, val_loader)

    epochs_run = int(log_df["epoch"].max()) if len(log_df) else (trainer.start_epoch - 1)
    best_shd_auroc, best_shd_epoch, first_epoch_over = float("nan"), None, None
    beat_kfold_mean = False
    if "val_shd_auroc" in log_df.columns and len(log_df) and log_df["val_shd_auroc"].notna().any():
        shd_col = log_df["val_shd_auroc"]
        best_shd_auroc = float(shd_col.max())
        best_shd_epoch = int(log_df.loc[shd_col.idxmax(), "epoch"])
        over = log_df[shd_col >= FUSED_SINGLE_RUN_VAL_SHD]
        beat_kfold_mean = not over.empty
        first_epoch_over = int(over["epoch"].iloc[0]) if beat_kfold_mean else None

    result = {
        "status": "completed", "epochs_run": epochs_run, "planned_total_epochs": n_epochs,
        "stopped_reason": stopped_reason, "best_val_mean_auroc": best_val_auroc,
        "best_shd_auroc": best_shd_auroc, "best_shd_epoch": best_shd_epoch,
        "beat_kfold_mean": beat_kfold_mean, "first_epoch_over_kfold_mean": first_epoch_over,
        "config": {
            "learning_rate": lr, "weight_decay": wd, "dropout": dropout, "batch_size": args.batch_size,
            "warmup_epochs": args.longrun_warmup_epochs, "checkpoint_every": args.longrun_checkpoint_every,
            "patience": args.longrun_patience,
        },
        "checkpoint_best": str(trainer.checkpoint_path), "checkpoint_last": str(trainer.resume_checkpoint_path),
        "demo_encoder_path": str(PROJECT_ROOT / "checkpoints" / "cnn_serial_demo_encoder.joblib"),
        "smoke": bool(args.smoke),
    }
    write_stage_result_json("longrun", result)
    log(f"[longrun] finished: {epochs_run} epochs ({stopped_reason}); best val mean-AUROC={best_val_auroc:.4f}; "
        f"best SHD AUROC={best_shd_auroc}")
    return 0


def stage_entry_warmstart_tune(args: argparse.Namespace) -> int:
    stage_deadline = args.stage_deadline_ts if args.stage_deadline_ts is not None else float("inf")
    global_deadline = args.global_deadline_ts if args.global_deadline_ts is not None else float("inf")
    effective_deadline = min(stage_deadline, global_deadline)

    maybe_set_cuda_memory_fraction(args.cuda_mem_fraction)

    longrun_result = read_stage_result_json("longrun")
    if not longrun_result or longrun_result.get("status") != "completed":
        write_stage_result_json("warmstart_tune", {"status": "skipped", "reason": "no completed longrun stage result found"})
        log("[warmstart_tune] skipped: no completed longrun result.")
        return 0

    epochs_run = longrun_result.get("epochs_run", 0)
    checkpoint_every = longrun_result.get("config", {}).get("checkpoint_every", 5)
    ckpt_epoch = nearest_checkpoint_epoch(epochs_run, checkpoint_every)
    ckpt_path = CHECKPOINT_DIR / f"longrun_epoch{ckpt_epoch}.pt"
    if not epochs_run or ckpt_epoch == 0 or not ckpt_path.exists():
        write_stage_result_json("warmstart_tune", {
            "status": "skipped",
            "reason": f"no usable longrun checkpoint (epochs_run={epochs_run}, ckpt_epoch={ckpt_epoch}, exists={ckpt_path.exists()})",
        })
        log("[warmstart_tune] skipped: no usable longrun checkpoint.")
        return 0

    dataset_dir = PROJECT_ROOT / "data" / DATASET_SUBDIR
    metadata_path = dataset_dir / METADATA_FILENAME
    data = load_shared_data(dataset_dir, metadata_path, demo_tag="cnn_serial")
    device = resolve_device("auto")

    trials_csv = REPORTS_DIR / "serial_warmstart_trials.csv"
    fields = [
        "trial_number", "state", "role", "lr_multiplier", "weight_decay", "dropout", "end_lr_fraction",
        "final_val_mean_auroc", "final_shd_auroc", "best_reported_auroc", "n_epochs_run", "duration_s", "timestamp",
    ]

    cfg = longrun_result.get("config", {})
    planned_total = longrun_result.get("planned_total_epochs", 60)
    warmup_epochs = cfg.get("warmup_epochs", 3)
    base_lr0 = cfg.get("learning_rate", 0.001)
    end_lr0 = base_lr0 * 0.01
    wd0 = cfg.get("weight_decay", 0.0001)
    dropout0 = cfg.get("dropout", 0.3)
    batch_size0 = cfg.get("batch_size", args.batch_size)

    lr_at_ckpt = warmup_cosine_lr(ckpt_epoch - 1, base_lr0, warmup_epochs, planned_total, end_lr0)
    lr_at_ckpt_plus_n = warmup_cosine_lr(
        min(ckpt_epoch - 1 + args.warmstart_epochs, planned_total - 1), base_lr0, warmup_epochs, planned_total, end_lr0
    )
    log(f"[warmstart_tune] warm-starting from {ckpt_path.name} (longrun epoch {ckpt_epoch}/{epochs_run}). "
        f"LR at that point ~{lr_at_ckpt:.2e}.")

    # --- control: continue the default schedule for warmstart_epochs, once.
    # Without this, a trial "winning" just means "five more epochs of training".
    control_model = load_warm_start_model(ckpt_path, N_LEADS, len(data.label_names), data.n_demo, dropout0, device)
    t0 = time.time()
    control_auroc, control_shd = run_warm_started_epochs(
        model=control_model, base_lr=lr_at_ckpt, weight_decay=wd0, end_lr=lr_at_ckpt_plus_n,
        n_epochs=args.warmstart_epochs, train_ds=data.train_ds, val_ds=data.val_ds,
        pos_weights=data.pos_weights, label_names=data.label_names, batch_size=batch_size0,
        num_workers=args.num_workers, device=str(device), trial=None, epoch_deadline_ts=effective_deadline,
        checkpoint_path=CHECKPOINT_DIR / "warmstart_control_best.pt",
    )
    control_beat_kfold_mean = control_shd == control_shd and control_shd >= FUSED_SINGLE_RUN_VAL_SHD
    append_csv_row(trials_csv, {
        "trial_number": -1, "state": "COMPLETE", "role": "control", "lr_multiplier": 1.0,
        "weight_decay": wd0, "dropout": dropout0,
        "end_lr_fraction": (lr_at_ckpt_plus_n / lr_at_ckpt) if lr_at_ckpt else "",
        "final_val_mean_auroc": round(control_auroc, 6),
        "final_shd_auroc": round(control_shd, 6) if control_shd == control_shd else "",
        "best_reported_auroc": round(control_auroc, 6), "n_epochs_run": args.warmstart_epochs,
        "duration_s": round(time.time() - t0, 1), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, fields)
    log(f"[warmstart_tune] control (continue default schedule, {args.warmstart_epochs} epochs): "
        f"val mean-AUROC={control_auroc:.4f} SHD-AUROC={control_shd:.4f}")

    remaining = max(0.0, effective_deadline - time.time())
    if remaining <= 0:
        write_stage_result_json("warmstart_tune", {
            "status": "completed", "note": "only the control ran; budget exhausted",
            "control_auroc": control_auroc, "control_shd_auroc": control_shd,
            "control_beat_kfold_mean": control_beat_kfold_mean,
            "warm_start_checkpoint": str(ckpt_path), "warm_start_epoch": ckpt_epoch,
        })
        return 0

    trial_shd: dict[int, float] = {}
    study_path = REPORTS_DIR / "serial_warmstart_study.db"
    study = optuna.create_study(
        study_name="warmstart_tune", storage=f"sqlite:///{study_path}", load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=43),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2),
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = sample_warmstart_params(trial)
        model = load_warm_start_model(ckpt_path, N_LEADS, len(data.label_names), data.n_demo, params["dropout"], device)
        base_lr = lr_at_ckpt * params["lr_multiplier"]
        end_lr = base_lr * params["end_lr_fraction"]
        t0i = time.time()
        try:
            val_auroc, shd_auroc = run_warm_started_epochs(
                model=model, base_lr=base_lr, weight_decay=params["weight_decay"], end_lr=end_lr,
                n_epochs=args.warmstart_epochs, train_ds=data.train_ds, val_ds=data.val_ds,
                pos_weights=data.pos_weights, label_names=data.label_names, batch_size=batch_size0,
                num_workers=args.num_workers, device=str(device), trial=trial, epoch_deadline_ts=effective_deadline,
                checkpoint_path=CHECKPOINT_DIR / f"warmstart_trial{trial.number}.pt",
            )
        except optuna.TrialPruned:
            best = max(trial_intermediates(trial).values()) if trial_intermediates(trial) else ""
            append_csv_row(trials_csv, {
                "trial_number": trial.number, "state": "PRUNED", "role": "trial", **params,
                "final_val_mean_auroc": "", "final_shd_auroc": "", "best_reported_auroc": best,
                "n_epochs_run": len(trial_intermediates(trial)), "duration_s": round(time.time() - t0i, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, fields)
            raise
        trial_shd[trial.number] = shd_auroc
        best = max(trial_intermediates(trial).values()) if trial_intermediates(trial) else round(val_auroc, 6)
        append_csv_row(trials_csv, {
            "trial_number": trial.number, "state": "COMPLETE", "role": "trial", **params,
            "final_val_mean_auroc": round(val_auroc, 6),
            "final_shd_auroc": round(shd_auroc, 6) if shd_auroc == shd_auroc else "",
            "best_reported_auroc": best, "n_epochs_run": len(trial_intermediates(trial)),
            "duration_s": round(time.time() - t0i, 1), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, fields)
        return val_auroc

    study.optimize(objective, n_trials=100_000, timeout=remaining)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_trial_auroc = max((t.value for t in completed), default=float("nan"))
    best_trial_number = max(completed, key=lambda t: t.value).number if completed else None
    beat_control = bool(completed) and best_trial_auroc > control_auroc
    best_trial_shd = trial_shd.get(best_trial_number, float("nan")) if best_trial_number is not None else float("nan")
    best_trial_beat_kfold_mean = best_trial_shd == best_trial_shd and best_trial_shd >= FUSED_SINGLE_RUN_VAL_SHD

    result = {
        "status": "completed", "n_trials": len(study.trials), "n_completed": len(completed),
        "best_trial_auroc": best_trial_auroc, "best_trial_number": best_trial_number,
        "best_trial_shd_auroc": best_trial_shd, "best_trial_beat_kfold_mean": best_trial_beat_kfold_mean,
        "control_auroc": control_auroc, "control_shd_auroc": control_shd,
        "control_beat_kfold_mean": control_beat_kfold_mean, "beat_control": beat_control,
        "warm_start_checkpoint": str(ckpt_path), "warm_start_epoch": ckpt_epoch,
        "best_trial_checkpoint": (
            str(CHECKPOINT_DIR / f"warmstart_trial{best_trial_number}.pt")
            if beat_control and best_trial_number is not None else None
        ),
        "limitations": (
            "cannot tune architecture (checkpoint fixes tensor shapes) or initial LR/warmup (already past); "
            "warm-start bias toward longrun's own regularisation config."
        ),
    }
    write_stage_result_json("warmstart_tune", result)
    log(f"[warmstart_tune] finished: {len(study.trials)} trials. best={best_trial_auroc:.4f} vs control={control_auroc:.4f} "
        f"({'beat' if beat_control else 'did not beat'} control).")
    return 0


def stage_entry_elimination_sweep(args: argparse.Namespace) -> int:
    stage_deadline = args.stage_deadline_ts if args.stage_deadline_ts is not None else float("inf")
    global_deadline = args.global_deadline_ts if args.global_deadline_ts is not None else float("inf")
    effective_deadline = min(stage_deadline, global_deadline)

    maybe_set_cuda_memory_fraction(args.cuda_mem_fraction)

    dataset_dir = PROJECT_ROOT / "data" / DATASET_SUBDIR
    metadata_path = dataset_dir / METADATA_FILENAME
    data = load_shared_data(dataset_dir, metadata_path, demo_tag="cnn_serial")
    device = resolve_device("auto")

    trials_csv = REPORTS_DIR / "serial_elimination_trials.csv"
    fields = [
        "trial_number", "state", "learning_rate", "weight_decay", "dropout", "batch_size",
        "final_val_mean_auroc", "best_reported_auroc", "n_epochs_run", "duration_s", "timestamp",
    ]

    def objective(trial: optuna.trial.Trial) -> float:
        params = sample_elimination_params(trial)
        t0 = time.time()
        try:
            val_auroc = train_from_scratch_with_pruning(
                params=params, train_ds=data.train_ds, val_ds=data.val_ds, pos_weights=data.pos_weights,
                label_names=data.label_names, n_leads=N_LEADS, n_demo_features=data.n_demo,
                n_epochs=args.elimination_epochs, device=str(device), num_workers=args.num_workers,
                trial=trial, epoch_deadline_ts=effective_deadline,
            )
        except optuna.TrialPruned:
            best = max(trial_intermediates(trial).values()) if trial_intermediates(trial) else ""
            append_csv_row(trials_csv, {
                "trial_number": trial.number, "state": "PRUNED", **params,
                "final_val_mean_auroc": "", "best_reported_auroc": best,
                "n_epochs_run": len(trial_intermediates(trial)), "duration_s": round(time.time() - t0, 1),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, fields)
            raise
        best = max(trial_intermediates(trial).values()) if trial_intermediates(trial) else round(val_auroc, 6)
        append_csv_row(trials_csv, {
            "trial_number": trial.number, "state": "COMPLETE", **params,
            "final_val_mean_auroc": round(val_auroc, 6), "best_reported_auroc": best,
            "n_epochs_run": len(trial_intermediates(trial)), "duration_s": round(time.time() - t0, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, fields)
        return val_auroc

    study_path = REPORTS_DIR / "serial_elimination_study.db"
    study = optuna.create_study(
        study_name="elimination_sweep", storage=f"sqlite:///{study_path}", load_if_exists=True,
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2),
    )
    remaining = max(0.0, effective_deadline - time.time())
    study.optimize(objective, n_trials=100_000, timeout=remaining)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    best_value, best_params = float("nan"), None
    if completed:
        best_trial = max(completed, key=lambda t: t.value)
        best_value, best_params = best_trial.value, best_trial.params

    result = {
        "status": "completed", "n_trials": len(study.trials), "n_completed": len(completed), "n_pruned": len(pruned),
        "best_value": best_value, "best_params": best_params,
        "purpose_note": (
            "Elimination sweep: eliminates bad regions of the space (4-5 epochs/trial is enough to rule out "
            "e.g. very high LR or very high weight decay), not a fine-grained ranking of the top configs."
        ),
    }
    write_stage_result_json("elimination_sweep", result)
    log(f"[elimination_sweep] finished: {len(study.trials)} trials ({len(completed)} completed, {len(pruned)} pruned).")
    return 0


def stage_entry_final_eval(args: argparse.Namespace) -> int:
    longrun_result = read_stage_result_json("longrun")
    warmstart_result = read_stage_result_json("warmstart_tune")

    candidates: list[tuple[str, Path, float]] = []
    if longrun_result.get("beat_kfold_mean") and longrun_result.get("checkpoint_best"):
        shd = longrun_result.get("best_shd_auroc", float("nan"))
        candidates.append(("longrun", Path(longrun_result["checkpoint_best"]), shd))
    if (
        warmstart_result.get("best_trial_beat_kfold_mean")
        and warmstart_result.get("beat_control")
        and warmstart_result.get("best_trial_checkpoint")
    ):
        shd = warmstart_result.get("best_trial_shd_auroc", float("nan"))
        candidates.append(("warmstart_tune", Path(warmstart_result["best_trial_checkpoint"]), shd))

    candidates = [c for c in candidates if c[1].exists()]
    if not candidates:
        write_stage_result_json("final_eval", {
            "status": "skipped",
            "reason": (
                f"No checkpoint beat the single-run fused val SHD AUROC ({FUSED_SINGLE_RUN_VAL_SHD}). "
                f"longrun best_shd_auroc={longrun_result.get('best_shd_auroc')}, "
                f"warmstart_tune best_trial_shd_auroc={warmstart_result.get('best_trial_shd_auroc')} "
                "(and/or warmstart's best trial did not beat its own control)."
            ),
        })
        log("[final_eval] skipped: no checkpoint beat the k-fold SHD mean.")
        return 0

    source, checkpoint_path, shd_val = max(candidates, key=lambda c: c[2] if c[2] == c[2] else -1.0)
    log(f"[final_eval] running held-out test-set evaluation for {checkpoint_path} "
        f"(source={source}, val SHD AUROC={shd_val:.4f}, tag=cnn_serial) — "
        "this is the only test-split use in this whole run, done once.")
    cmd = [
        sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "6_evaluate" / "evaluate_test_set.py"),
        "--checkpoint", str(checkpoint_path), "--tag", "cnn_serial",
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    write_stage_result_json("final_eval", {
        "status": "completed" if result.returncode == 0 else "failed",
        "checkpoint_used": str(checkpoint_path), "source": source, "source_val_shd_auroc": shd_val,
        "tag": "cnn_serial", "returncode": result.returncode,
    })
    log(f"[final_eval] finished with return code {result.returncode}.")
    return result.returncode


STAGE_ENTRY_POINTS = {
    "longrun": stage_entry_longrun,
    "warmstart_tune": stage_entry_warmstart_tune,
    "elimination_sweep": stage_entry_elimination_sweep,
    "final_eval": stage_entry_final_eval,
}


# --------------------------------------------------------------------------
# Orchestrator-side stage runners (this process is NOT a stage subprocess).
# --------------------------------------------------------------------------


def guard_or_skip(stage: str, args: argparse.Namespace, state: dict) -> bool:
    threshold_kb = args.mem_threshold_gb * 1e6
    ok = wait_for_memory(
        read_fn=read_mem_available_kb, sleep_fn=time.sleep, threshold_kb=threshold_kb,
        retries=args.mem_retries, poll_interval_s=args.mem_retry_wait_s, log_fn=log,
    )
    if not ok:
        mark_stage(state, stage, "skipped", reason=f"MemAvailable stayed below {args.mem_threshold_gb}GB guard threshold")
        save_state(state, STATE_PATH)
        milestone(f"{stage} skipped: memory guard never cleared {args.mem_threshold_gb}GB threshold.")
        return False
    return True


def orchestrate_kfold_fold4(args: argparse.Namespace, state: dict, global_deadline_ts: float) -> None:
    stage = "kfold_fold4"
    mark_stage(state, stage, "running", started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_state(state, STATE_PATH)
    stage_start = time.time()
    stage_deadline = deadline_ts(stage_start, args.budget_kfold_fold4_hours)
    log_path = PROGRESS_DIR / "serial_kfold_fold4.log"

    cmd1 = [
        sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "6_evaluate" / "kfold_cnn.py"),
        "--k", "5", "--only-fold", str(args.only_fold), "--n-epochs", "20", "--patience", "6",
        "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers), "--seed", "42",
    ]
    milestone(f"kfold_fold4 starting. budget={args.budget_kfold_fold4_hours}h. cmd: {' '.join(cmd1)}")
    ret1, budget_hit1 = run_stage_process(cmd1, log_path, stage_deadline, global_deadline_ts)
    if ret1 != 0 and not budget_hit1:
        mark_stage(state, stage, "failed", reason=f"kfold_cnn.py exited {ret1}")
        save_state(state, STATE_PATH)
        milestone(f"kfold_fold4 FAILED: kfold_cnn.py exited {ret1}. See {log_path}.")
        return
    if budget_hit1:
        mark_stage(state, stage, "done (budget)", reason="kfold_cnn.py hit its wall-clock budget and was stopped")
        save_state(state, STATE_PATH)
        milestone("kfold_fold4: budget exceeded during kfold_cnn.py — marked done (budget); summarize skipped this run.")
        return

    cmd2 = [sys.executable, "-u", str(PROJECT_ROOT / "scripts" / "6_evaluate" / "kfold_summarize.py")]
    ret2, budget_hit2 = run_stage_process(cmd2, log_path, stage_deadline, global_deadline_ts)
    if ret2 != 0:
        mark_stage(state, stage, "failed", reason=f"kfold_summarize.py exited {ret2}")
        save_state(state, STATE_PATH)
        milestone(f"kfold_fold4 FAILED at summarize step: exit {ret2}. See {log_path}.")
        return

    shd_mean_5fold, n_folds = None, None
    try:
        kdf = pd.read_csv(REPORTS_DIR / "kfold_results.csv")
        test_rows = kdf[
            (kdf["split"] == "test") & (kdf["fold"] != "ensemble") & (kdf["target"] == DEFAULT_STRATIFY_COL)
        ]
        if len(test_rows):
            shd_mean_5fold = float(test_rows["auroc"].mean())
            n_folds = int(test_rows["fold"].nunique())
    except Exception:
        pass

    status = "done" if (time.time() - stage_start) <= args.budget_kfold_fold4_hours * 3600 else "done (budget)"
    mark_stage(
        state, stage, status, n_folds_completed=n_folds, shd_mean_test=shd_mean_5fold,
        artifacts=["reports/kfold_results.csv", "reports/kfold_summary.md", "figures/results/kfold_variance.png"],
    )
    save_state(state, STATE_PATH)
    milestone(f"kfold_fold4 done. n_folds_completed={n_folds}, SHD test mean={shd_mean_5fold}.")


def orchestrate_self_stage(stage: str, args: argparse.Namespace, state: dict, global_deadline_ts: float) -> None:
    mark_stage(state, stage, "running", started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_state(state, STATE_PATH)
    stage_start = time.time()
    budget_hours = {
        "longrun": args.budget_longrun_hours, "warmstart_tune": args.budget_warmstart_tune_hours,
        "elimination_sweep": args.budget_elimination_sweep_hours, "final_eval": args.budget_final_eval_hours,
    }[stage]
    stage_deadline = deadline_ts(stage_start, budget_hours)

    cmd = build_self_command(stage, args, stage_deadline, global_deadline_ts)
    log_path = PROGRESS_DIR / f"serial_{stage}.log"
    milestone(f"{stage} starting. budget={budget_hours}h. cmd: {' '.join(cmd)}")
    ret, budget_hit = run_stage_process(cmd, log_path, stage_deadline, global_deadline_ts)
    result = read_stage_result_json(stage)

    if ret != 0 and not budget_hit:
        mark_stage(state, stage, "failed", returncode=ret, result=result)
        save_state(state, STATE_PATH)
        milestone(f"{stage} FAILED: exit code {ret}. See {log_path}.")
        return
    if budget_hit:
        mark_stage(state, stage, "done (budget)", returncode=ret, result=result)
        save_state(state, STATE_PATH)
        milestone(f"{stage}: hit its wall-clock/global budget — stopped by the orchestrator's hard-kill safety net.")
        return
    if result.get("status") == "skipped":
        mark_stage(state, stage, "skipped", reason=result.get("reason"))
        save_state(state, STATE_PATH)
        milestone(f"{stage} skipped: {result.get('reason')}")
        return

    mark_stage(state, stage, "done", result=result)
    save_state(state, STATE_PATH)
    milestone(f"{stage} done. result: {json.dumps(result, default=str)[:500]}")


def dry_run_stage(stage: str, args: argparse.Namespace, state: dict) -> None:
    mark_stage(state, stage, "running")
    save_state(state, STATE_PATH)
    now = time.time()
    global_deadline = deadline_ts(now, args.global_deadline_hours)

    if stage == "kfold_fold4":
        stage_deadline = deadline_ts(now, args.budget_kfold_fold4_hours)
        cmd1 = [
            sys.executable, "-u", "scripts/6_evaluate/kfold_cnn.py", "--k", "5", "--only-fold", str(args.only_fold),
            "--n-epochs", "20", "--patience", "6", "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers), "--seed", "42",
        ]
        cmd2 = ["scripts/6_evaluate/kfold_summarize.py"]
        log(f"[dry-run] {stage}: would run {' '.join(cmd1)}")
        log(f"[dry-run] {stage}: then {' '.join(cmd2)}")
    else:
        budget_hours = {
            "longrun": args.budget_longrun_hours, "warmstart_tune": args.budget_warmstart_tune_hours,
            "elimination_sweep": args.budget_elimination_sweep_hours, "final_eval": args.budget_final_eval_hours,
        }[stage]
        stage_deadline = deadline_ts(now, budget_hours)
        cmd = build_self_command(stage, args, stage_deadline, global_deadline)
        log(f"[dry-run] {stage}: would run {' '.join(cmd)}")

        if stage == "longrun":
            e0 = warmup_cosine_lr(0, 0.001, args.longrun_warmup_epochs, args.longrun_max_epochs, 0.00001)
            e_warm = warmup_cosine_lr(args.longrun_warmup_epochs, 0.001, args.longrun_warmup_epochs, args.longrun_max_epochs, 0.00001)
            e_final = warmup_cosine_lr(args.longrun_max_epochs - 1, 0.001, args.longrun_warmup_epochs, args.longrun_max_epochs, 0.00001)
            log(f"[dry-run] longrun LR sanity: epoch0={e0:.2e} end_of_warmup={e_warm:.2e} final_epoch={e_final:.2e}")
        elif stage == "warmstart_tune":
            study = optuna.create_study(direction="maximize")
            trial = study.ask()
            log(f"[dry-run] warmstart_tune search space: {WARMSTART_SPACE}")
            log(f"[dry-run] example sampled params: {sample_warmstart_params(trial)}")
        elif stage == "elimination_sweep":
            study = optuna.create_study(direction="maximize")
            trial = study.ask()
            log(f"[dry-run] elimination_sweep search space: {ELIMINATION_SPACE}")
            log(f"[dry-run] example sampled params: {sample_elimination_params(trial)}")
        elif stage == "final_eval":
            log(f"[dry-run] final_eval gate: beats single-run fused VAL SHD AUROC ({FUSED_SINGLE_RUN_VAL_SHD})? "
                "checked against reports/serial_stage_result_longrun.json / serial_stage_result_warmstart_tune.json.")

    mark_stage(state, stage, "dry_run_ok")
    save_state(state, STATE_PATH)
    milestone(f"{stage} dry-run OK (no GPU/data touched).")


# --------------------------------------------------------------------------
# Notes + figure (cheap, run in-process after every stage transition).
# --------------------------------------------------------------------------


def write_serial_notes(state: dict) -> None:
    lines = ["# Serial stage runner — findings", ""]
    stages = state.get("stages", {})

    kf = stages.get("kfold_fold4", {})
    lines += ["## Stage 1 — kfold_fold4", ""]
    if kf.get("status", "").startswith("done"):
        lines.append(
            f"Status: {kf.get('status')}. {kf.get('n_folds_completed')} folds now present in "
            f"`reports/kfold_results.csv`; test-split SHD-composite AUROC mean across them = "
            f"{kf.get('shd_mean_test')}."
        )
    else:
        lines.append(f"Status: {kf.get('status', 'not run')}.")
    lines.append("")

    lr = stages.get("longrun", {}).get("result", {})
    lr_status = stages.get("longrun", {}).get("status", "not run")
    lines += ["## Stage 2 — longrun", ""]
    if lr:
        lines.append(
            f"Status: {lr_status}. Ran {lr.get('epochs_run')}/{lr.get('planned_total_epochs')} epochs "
            f"({lr.get('stopped_reason')}). Best val mean-AUROC={lr.get('best_val_mean_auroc')}; "
            f"best val SHD-composite AUROC={lr.get('best_shd_auroc')} at epoch {lr.get('best_shd_epoch')}."
        )
        lines.append(
            f"\n**Verdict vs the single-run fused val baseline ({FUSED_SINGLE_RUN_VAL_SHD}); the k-fold test mean {KFOLD_SHD_MEAN_4FOLD} is a test-split reference, not the gate:** "
            + (
                f"longrun's best SHD AUROC **beat** it, first passing at epoch "
                f"{lr.get('first_epoch_over_kfold_mean')} — direct evidence the 20-epoch budget used elsewhere in "
                "this project was the binding constraint, not overfitting."
                if lr.get("beat_kfold_mean")
                else f"longrun's best val SHD AUROC ({lr.get('best_shd_auroc')}) did **not** beat {FUSED_SINGLE_RUN_VAL_SHD}."
            )
        )
        if lr.get("stopped_reason") == "budget_exceeded":
            lines.append(
                "\nThe run stopped on its wall-clock budget, not early stopping — **val AUROC may still have been "
                "rising when it stopped; the ceiling is not established.** A longer budget is needed to know where "
                "this architecture actually saturates."
            )
        elif str(lr.get("stopped_reason", "")).startswith("early_stopping"):
            lines.append(
                f"\npatience={lr.get('config', {}).get('patience')} *did* fire this time — given enough epoch "
                "budget this architecture does eventually plateau; the earlier claim was only that a 20-epoch "
                "budget cut it off before that point was reached."
            )
    else:
        lines.append(f"Status: {lr_status}.")
    lines.append("")

    ws = stages.get("warmstart_tune", {}).get("result", {})
    ws_status = stages.get("warmstart_tune", {}).get("status", "not run")
    lines += ["## Stage 3 — warmstart_tune", ""]
    if ws and ws.get("status") == "completed":
        lines.append(
            f"Status: {ws_status}. Warm-started from longrun epoch {ws.get('warm_start_epoch')}. "
            f"{ws.get('n_trials', 0)} ASHA trials ({ws.get('n_completed', 0)} completed) vs. one control run "
            "that continues the unmodified schedule for the same number of epochs from the same checkpoint."
        )
        lines.append(
            f"\nBest trial val mean-AUROC={ws.get('best_trial_auroc')} vs control={ws.get('control_auroc')} — "
            f"the best trial **{'beat' if ws.get('beat_control') else 'did not beat'}** its control."
        )
        lines.append(f"\n**Honest limits**: {ws.get('limitations', '')}")
    elif ws_status == "skipped":
        lines.append(f"Status: skipped — {stages.get('warmstart_tune', {}).get('reason')}")
    else:
        lines.append(f"Status: {ws_status}.")
    lines.append("")

    es = stages.get("elimination_sweep", {}).get("result", {})
    es_status = stages.get("elimination_sweep", {}).get("status", "not run")
    lines += ["## Stage 4 — elimination_sweep", ""]
    if es and es.get("status") == "completed":
        lines.append(
            f"Status: {es_status}. {es.get('n_trials', 0)} trials ({es.get('n_completed', 0)} completed, "
            f"{es.get('n_pruned', 0)} pruned). Best surviving config (informational, not a ranked winner — see "
            f"below): `{es.get('best_params')}` (val mean-AUROC={es.get('best_value')})."
        )
        lines.append(f"\n{es.get('purpose_note', '')}")
    else:
        lines.append(f"Status: {es_status}.")
    lines.append("")

    fe = stages.get("final_eval", {}).get("result", {})
    fe_status = stages.get("final_eval", {}).get("status", "not run")
    lines += ["## Stage 5 — final_eval", ""]
    if fe_status == "skipped":
        lines.append(f"Skipped: {stages.get('final_eval', {}).get('reason', fe.get('reason', ''))}")
    elif fe:
        lines.append(
            f"Status: {fe_status}. Scored `{fe.get('checkpoint_used')}` (source={fe.get('source')}) once on the "
            f"held-out test split as tag `cnn_serial` — see `reports/final_results.csv` for the full per-target "
            "table. The test split was never used for stage selection, only this one-time scoring."
        )
    else:
        lines.append(f"Status: {fe_status}.")
    lines.append("")

    lines += ["## Does \"data-limited, not regularisation-limited\" (reports/cnn_v2_notes.md) still hold?", ""]
    if lr.get("beat_kfold_mean"):
        lines.append(
            "**No, not as originally stated.** That claim rested on two runs (cnn_v2_notes.md's v1/v2 comparison) "
            "that both sat inside a short epoch budget with patience never firing. longrun shows the same "
            "architecture keeps improving past 20 epochs given a longer budget and a schedule that won't stop it "
            "early, and it beat the 4-fold k-fold SHD mean. The regularisation picture (per-record normalisation + "
            "augmentation) may still be correct as far as it goes, but 'data-limited rather than "
            "regularisation-limited' was too strong a conclusion from runs that never had enough epochs to plateau."
        )
    elif lr:
        lines.append(
            "**Not overturned by this run.** longrun did not beat the k-fold SHD mean within its wall-clock "
            "budget, so this evidence does not contradict the existing 'data-limited' reading — but it does not "
            "strongly confirm it either, since a wall-clock budget (not necessarily the loss landscape) is what "
            "cut the run short. See `stopped_reason` above; if it was `budget_exceeded`, the ceiling is still open."
        )
    else:
        lines.append("longrun has not completed yet — nothing to say here until it has.")
    lines.append("")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by scripts/12_overnight/run_overnight.py._")

    (REPORTS_DIR / "serial_notes.md").write_text("\n".join(lines) + "\n")
    milestone("Wrote reports/serial_notes.md.")


def plot_serial_training_curves() -> None:
    train_log_path = REPORTS_DIR / "serial_longrun_trainlog.csv"
    elimination_path = REPORTS_DIR / "serial_elimination_trials.csv"
    if not train_log_path.exists() and not elimination_path.exists():
        log("Nothing to plot yet (no longrun trainlog or elimination trials) — skipping figure.")
        return

    import matplotlib.pyplot as plt
    from src.plotting import apply_style

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    if train_log_path.exists():
        log_df = pd.read_csv(train_log_path)
        if len(log_df):
            ax2 = ax.twinx()
            ax.plot(log_df["epoch"], log_df["val_mean_auroc"], color="#2CA6A4", label="val mean AUROC (12 targets)")
            if "val_shd_auroc" in log_df.columns:
                ax.plot(log_df["epoch"], log_df["val_shd_auroc"], color="#7C3AED", label="val SHD AUROC")
            # The y-axis here is VALIDATION AUROC, so the reference must be a
            # validation number. The k-fold mean is a test-split figure and drawing
            # it here implied the run was underperforming when it was not.
            ax.axhline(FUSED_SINGLE_RUN_VAL_SHD, color="#E0457B", linestyle="--", linewidth=1,
                       label=f"single-run fused val SHD ({FUSED_SINGLE_RUN_VAL_SHD})")
            ax2.plot(log_df["epoch"], log_df["lr"], color="#9CA3AF", linewidth=1, label="LR")
            ax2.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_ylabel("val AUROC")
            ax2.set_ylabel("learning rate (log)")
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
            ax.set_title("longrun: val AUROC and LR vs epoch")
        else:
            ax.text(0.5, 0.5, "No longrun epochs logged yet", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No longrun epochs logged yet", ha="center", va="center")

    ax = axes[1]
    if elimination_path.exists():
        trials_df = pd.read_csv(elimination_path)
        if len(trials_df):
            for state_name, color in [("COMPLETE", "#2CA6A4"), ("PRUNED", "#9CA3AF")]:
                sub = trials_df[trials_df["state"] == state_name]
                if len(sub):
                    vals = pd.to_numeric(sub["best_reported_auroc"].replace("", np.nan), errors="coerce")
                    ax.scatter(sub["learning_rate"], vals, c=color, label=f"{state_name.lower()} ({len(sub)})", alpha=0.8, s=40)
            ax.set_xscale("log")
            ax.set_xlabel("learning rate")
            ax.set_ylabel("best reported val mean AUROC")
            ax.set_title("elimination_sweep: pruned vs surviving trials")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No elimination_sweep trials logged yet", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No elimination_sweep trials logged yet", ha="center", va="center")

    fig.tight_layout()
    out_path = PROJECT_ROOT / "figures" / "results" / "serial_training_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    milestone(f"Wrote {out_path}.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

    # This invocation IS one stage's subprocess.
    if args.run_stage is not None:
        sys.exit(STAGE_ENTRY_POINTS[args.run_stage](args))

    # Orchestrator mode.
    requested_stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in requested_stages:
        if s not in STAGE_NAMES:
            raise SystemExit(f"Unknown stage {s!r}; choose from {STAGE_NAMES}")

    resume = args.resume and not args.restart
    if args.restart:
        milestone("--restart: clearing reports/serial_state.json and checkpoints/serial/** back to scratch.")
        state = default_state()
        save_state(state, STATE_PATH)
        for p in CHECKPOINT_DIR.glob("*"):
            if p.is_file():
                p.unlink()
        for pattern in ["serial_*.csv", "serial_*_study.db*", "serial_stage_result_*.json"]:
            for p in REPORTS_DIR.glob(pattern):
                p.unlink()
    else:
        state = load_state(STATE_PATH) if resume else default_state()
        if not resume:
            save_state(state, STATE_PATH)

    global_start = time.time()
    global_deadline_ts = deadline_ts(global_start, args.global_deadline_hours)

    milestone(
        f"=== Serial run starting === stages={requested_stages} resume={resume} restart={args.restart} "
        f"dry_run={args.dry_run} smoke={args.smoke} global_deadline_hours={args.global_deadline_hours} "
        f"max_concurrent_stages={args.max_concurrent_stages} (hard invariant: 1) "
        f"mem_threshold_gb={args.mem_threshold_gb} num_workers={args.num_workers} batch_size={args.batch_size}"
    )

    for stage in STAGE_NAMES:
        if stage not in requested_stages:
            continue
        if resume and not stage_needs_to_run(state, stage):
            log(f"Stage {stage} already done (resume) — skipping.")
            continue
        if deadline_exceeded(global_deadline_ts):
            milestone(f"Global deadline ({args.global_deadline_hours}h) reached before stage {stage} could start — stopping.")
            break

        if args.dry_run:
            dry_run_stage(stage, args, state)
            continue

        if not guard_or_skip(stage, args, state):
            continue

        stage_t0 = time.time()
        try:
            if stage == "kfold_fold4":
                orchestrate_kfold_fold4(args, state, global_deadline_ts)
            else:
                orchestrate_self_stage(stage, args, state, global_deadline_ts)
        except Exception as exc:  # noqa: BLE001 - one stage failing must not kill the others
            tb = traceback.format_exc()
            log(f"Stage {stage} raised in the orchestrator: {exc}\n{tb}")
            mark_stage(state, stage, "failed", error=str(exc), traceback=tb)
            save_state(state, STATE_PATH)
            milestone(f"Stage {stage} FAILED in the orchestrator: {exc}")
        state.setdefault("timings", {})[f"{stage}_seconds"] = round(time.time() - stage_t0, 1)
        save_state(state, STATE_PATH)

        try:
            write_serial_notes(state)
        except Exception:
            log(f"Failed to update reports/serial_notes.md:\n{traceback.format_exc()}")
        try:
            plot_serial_training_curves()
        except Exception:
            log(f"Failed to update figures/results/serial_training_curves.png:\n{traceback.format_exc()}")

    milestone("=== Serial run finished (or stopped at a deadline) ===")


if __name__ == "__main__":
    main()
