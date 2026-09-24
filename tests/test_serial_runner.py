"""Tests for the serial, crash-resilient stage runner (agent SERIAL,
2026-09-23): `src/night_training.py` primitives + the orchestrator in
`scripts/12_overnight/run_overnight.py`.

Everything here is fast and CPU-only / no real dataset — the one real GPU
exercise (proving the actual training path, checkpoint writes, and resume
against the real waveform data) is the separate live smoke test described in
the task, not part of this suite.
"""
from __future__ import annotations

import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.models import ECGConvNet
from src.night_training import (
    ResumableLongrunTrainer,
    STAGE_NAMES,
    budget_exceeded,
    deadline_exceeded,
    deadline_ts,
    default_state,
    load_state,
    mark_stage,
    nearest_checkpoint_epoch,
    read_mem_available_kb,
    save_state,
    stage_needs_to_run,
    wait_for_memory,
    warmup_cosine_lr,
)
from src.training import TrainConfig, build_dataloaders
from tests.conftest import load_script

# --------------------------------------------------------------------------
# Warmup + cosine LR schedule
# --------------------------------------------------------------------------


def test_warmup_cosine_lr_at_epoch_zero_is_the_first_warmup_step():
    lr = warmup_cosine_lr(epoch_idx=0, base_lr=1e-3, warmup_epochs=3, total_epochs=60, end_lr=1e-5)
    assert lr == pytest.approx(1e-3 / 3)


def test_warmup_cosine_lr_at_end_of_warmup_equals_base_lr():
    # epoch_idx == warmup_epochs is the first epoch *after* warmup completes,
    # where the cosine curve starts (progress == 0 -> lr == base_lr).
    lr = warmup_cosine_lr(epoch_idx=3, base_lr=1e-3, warmup_epochs=3, total_epochs=60, end_lr=1e-5)
    assert lr == pytest.approx(1e-3)


def test_warmup_cosine_lr_at_final_epoch_equals_end_lr():
    lr = warmup_cosine_lr(epoch_idx=59, base_lr=1e-3, warmup_epochs=3, total_epochs=60, end_lr=1e-5)
    assert lr == pytest.approx(1e-5, abs=1e-9)


def test_warmup_cosine_lr_is_monotonic_within_each_phase():
    warmup_lrs = [warmup_cosine_lr(e, 1e-3, 3, 60, 1e-5) for e in range(3)]
    assert warmup_lrs == sorted(warmup_lrs)

    decay_lrs = [warmup_cosine_lr(e, 1e-3, 3, 60, 1e-5) for e in range(3, 60)]
    assert decay_lrs == sorted(decay_lrs, reverse=True)


def test_warmup_cosine_lr_handles_zero_warmup_epochs():
    lr0 = warmup_cosine_lr(epoch_idx=0, base_lr=1e-3, warmup_epochs=0, total_epochs=10, end_lr=0.0)
    assert lr0 == pytest.approx(1e-3)


# --------------------------------------------------------------------------
# Budget checker
# --------------------------------------------------------------------------


def test_budget_exceeded_false_before_the_budget_elapses():
    start = 1_000_000.0
    assert budget_exceeded(start, budget_hours=1.0, now_ts=start + 30 * 60) is False


def test_budget_exceeded_true_once_elapsed_passes_the_budget():
    start = 1_000_000.0
    assert budget_exceeded(start, budget_hours=1.0, now_ts=start + 2 * 3600) is True


def test_budget_exceeded_never_true_for_a_non_positive_budget():
    start = 1_000_000.0
    assert budget_exceeded(start, budget_hours=0, now_ts=start + 999 * 3600) is False
    assert budget_exceeded(start, budget_hours=None, now_ts=start + 999 * 3600) is False


def test_deadline_ts_is_start_plus_budget_seconds():
    start = 1_000_000.0
    assert deadline_ts(start, 2.0) == pytest.approx(start + 7200)
    assert deadline_ts(start, 0) == float("inf")


def test_deadline_exceeded():
    assert deadline_exceeded(100.0, now_ts=101.0) is True
    assert deadline_exceeded(100.0, now_ts=99.0) is False


def test_budget_checker_fires_between_epochs_and_stage_stops_cleanly(tmp_path):
    """Real ResumableLongrunTrainer.fit(), tiny synthetic data, CPU: the
    deadline is already in the past, so training must stop after exactly one
    epoch (not run to completion), write a resume checkpoint, and report
    stopped_reason == "budget_exceeded" — proving the check happens between
    epochs, not only once at stage start."""
    trainer, train_loader, val_loader = _make_tiny_trainer(tmp_path, stage_deadline_ts=time.time() - 1.0, n_epochs=5)
    log_df, stopped_reason, _ = trainer.fit(train_loader, val_loader)
    assert stopped_reason == "budget_exceeded"
    assert len(log_df) == 1
    assert (tmp_path / "tiny_last.pt").exists()


def test_resume_continues_from_the_epoch_after_the_last_checkpoint(tmp_path):
    """A stage marked 'running' when the process died is resumed from its
    own last checkpoint, not restarted — this exercises that at the
    trainer level with real (tiny) torch code."""
    trainer1, train_loader, val_loader = _make_tiny_trainer(tmp_path, stage_deadline_ts=time.time() - 1.0, n_epochs=5)
    log_df1, stopped1, _ = trainer1.fit(train_loader, val_loader)
    assert stopped1 == "budget_exceeded"
    last_epoch = int(log_df1["epoch"].max())

    trainer2, train_loader2, val_loader2 = _make_tiny_trainer(tmp_path, stage_deadline_ts=float("inf"), n_epochs=5)
    assert trainer2._resumed_from == last_epoch
    assert trainer2.start_epoch == last_epoch + 1

    log_df2, stopped2, _ = trainer2.fit(train_loader2, val_loader2)
    assert int(log_df2["epoch"].min()) == last_epoch + 1
    assert stopped2 == "completed_all_epochs"


class _TinyECGDataset(torch.utils.data.Dataset):
    """Minimal (n_leads=2, T=64) synthetic stand-in for ECGDataset, just to
    exercise ResumableLongrunTrainer's real epoch loop fast on CPU."""

    def __init__(self, n: int = 8, n_leads: int = 2, t: int = 64, n_labels: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.waveforms = rng.standard_normal((n, n_leads, t)).astype(np.float32)
        self.demo = np.zeros((n, 0), dtype=np.float32)
        self.labels = rng.integers(0, 2, (n, n_labels)).astype(np.float32)
        self.mask = np.ones((n, n_labels), dtype=bool)

    def __len__(self) -> int:
        return len(self.waveforms)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.waveforms[idx]),
            torch.from_numpy(self.demo[idx]),
            torch.from_numpy(self.labels[idx]),
            torch.from_numpy(self.mask[idx]),
        )


def _make_tiny_trainer(tmp_path, stage_deadline_ts: float, n_epochs: int):
    n_leads, n_labels = 2, 2
    ds = _TinyECGDataset(n_leads=n_leads, n_labels=n_labels)
    cfg = TrainConfig(
        n_epochs=n_epochs, batch_size=2, learning_rate=1e-3, weight_decay=0.0,
        patience=100, num_workers=0, device="cpu", checkpoint_dir=tmp_path,
    )
    train_loader, val_loader = build_dataloaders(ds, ds, cfg)
    model = ECGConvNet(n_leads=n_leads, n_labels=n_labels, n_demo_features=0, dropout=0.1)
    trainer = ResumableLongrunTrainer(
        model=model, config=cfg, pos_weights=torch.ones(n_labels), label_names=["a", "b"],
        checkpoint_name="tiny_best", warmup_epochs=1, checkpoint_every=100,
        epoch_checkpoint_dir=tmp_path, resume_checkpoint_path=tmp_path / "tiny_last.pt",
        stage_deadline_ts=stage_deadline_ts,
    )
    return trainer, train_loader, val_loader


# --------------------------------------------------------------------------
# State file round-trip / atomic write
# --------------------------------------------------------------------------


def test_state_round_trips_through_save_and_load(tmp_path):
    path = tmp_path / "serial_state.json"
    state = default_state()
    mark_stage(state, "longrun", "done", best_shd_auroc=0.841)
    state["checkpoints"]["longrun_best"] = "checkpoints/serial/longrun_best.pt"
    state["timings"]["longrun_seconds"] = 123.4

    save_state(state, path)
    assert path.exists()
    loaded = load_state(path)
    assert loaded == state


def test_load_state_returns_default_state_when_file_is_missing(tmp_path):
    path = tmp_path / "does_not_exist.json"
    loaded = load_state(path)
    assert loaded["stages"] == {name: {"status": "pending"} for name in STAGE_NAMES}


def test_save_state_write_is_atomic_no_leftover_tmp_file(tmp_path):
    path = tmp_path / "state.json"
    save_state(default_state(), path)
    assert not (tmp_path / "state.json.tmp").exists()
    state2 = load_state(path)
    mark_stage(state2, "longrun", "running")
    save_state(state2, path)
    assert load_state(path)["stages"]["longrun"]["status"] == "running"


def test_state_survives_a_simulated_crash_mid_write(tmp_path):
    """A crash between `tmp.write_text(...)` and `os.replace(...)` leaves a
    half-written (or here, simply stale/corrupt) `.tmp` file next to a
    perfectly good real state file. `os.replace` never ran, so the real file
    must be exactly what the last *successful* save_state wrote — the
    leftover tmp file must be ignored entirely."""
    path = tmp_path / "serial_state.json"
    good_state = default_state()
    mark_stage(good_state, "kfold_fold4", "done", shd_mean_test=0.841)
    save_state(good_state, path)

    # Simulate a crash mid-write: a stray, corrupt .tmp file next to the good
    # real file (as if the process died after tmp.write_text but before
    # os.replace on some *subsequent* save attempt).
    (path.with_suffix(path.suffix + ".tmp")).write_text("{not valid json!!")

    loaded = load_state(path)
    assert loaded == good_state
    assert loaded["stages"]["kfold_fold4"]["status"] == "done"


# --------------------------------------------------------------------------
# Resume scheduling semantics
# --------------------------------------------------------------------------


def test_done_stages_are_skipped_on_resume():
    state = default_state()
    mark_stage(state, "kfold_fold4", "done")
    mark_stage(state, "longrun", "done (budget)")
    assert stage_needs_to_run(state, "kfold_fold4") is False
    assert stage_needs_to_run(state, "longrun") is False  # "done (budget)" still starts with "done"


def test_running_stage_is_treated_as_needing_to_run_again_on_resume():
    state = default_state()
    mark_stage(state, "longrun", "running")
    assert stage_needs_to_run(state, "longrun") is True


def test_pending_failed_and_skipped_stages_all_need_to_run():
    state = default_state()
    mark_stage(state, "elimination_sweep", "failed")
    mark_stage(state, "final_eval", "skipped")
    assert stage_needs_to_run(state, "kfold_fold4") is True  # still pending
    assert stage_needs_to_run(state, "elimination_sweep") is True
    assert stage_needs_to_run(state, "final_eval") is True


# --------------------------------------------------------------------------
# Memory guard
# --------------------------------------------------------------------------


def test_read_mem_available_kb_parses_proc_meminfo_shaped_text(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16384000 kB\nMemFree:         1000000 kB\nMemAvailable:    2621440 kB\n")
    assert read_mem_available_kb(meminfo) == pytest.approx(2621440.0)


def test_read_mem_available_kb_raises_when_field_missing(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16384000 kB\n")
    with pytest.raises(ValueError):
        read_mem_available_kb(meminfo)


def test_wait_for_memory_blocks_and_eventually_skips_when_always_below_threshold():
    calls = {"reads": 0, "sleeps": 0}

    def read_fn():
        calls["reads"] += 1
        return 500_000.0  # always well below threshold

    def sleep_fn(_s):
        calls["sleeps"] += 1

    ok = wait_for_memory(
        read_fn=read_fn, sleep_fn=sleep_fn, threshold_kb=2_500_000.0, retries=3, poll_interval_s=1.0, log_fn=lambda m: None,
    )
    assert ok is False
    assert calls["reads"] == 4  # initial + 3 retries
    assert calls["sleeps"] == 3  # sleeps between retries, not after the last one


def test_wait_for_memory_recovers_before_retries_are_exhausted():
    readings = iter([500_000.0, 500_000.0, 3_000_000.0])

    ok = wait_for_memory(
        read_fn=lambda: next(readings), sleep_fn=lambda _s: None, threshold_kb=2_500_000.0,
        retries=5, poll_interval_s=1.0, log_fn=lambda m: None,
    )
    assert ok is True


def test_wait_for_memory_passes_immediately_when_already_above_threshold():
    calls = {"reads": 0}

    def read_fn():
        calls["reads"] += 1
        return 10_000_000.0

    ok = wait_for_memory(read_fn=read_fn, sleep_fn=lambda _s: None, threshold_kb=2_500_000.0, log_fn=lambda m: None)
    assert ok is True
    assert calls["reads"] == 1


# --------------------------------------------------------------------------
# nearest_checkpoint_epoch (warmstart_tune's "nearest 3/4 of epochs run" rule)
# --------------------------------------------------------------------------


def test_nearest_checkpoint_epoch_picks_closest_multiple_of_checkpoint_every():
    assert nearest_checkpoint_epoch(epochs_run=40, checkpoint_every=5, fraction=0.75) == 30
    assert nearest_checkpoint_epoch(epochs_run=20, checkpoint_every=5, fraction=0.75) == 15


def test_nearest_checkpoint_epoch_is_zero_below_one_interval():
    assert nearest_checkpoint_epoch(epochs_run=3, checkpoint_every=5, fraction=0.75) == 0


def test_nearest_checkpoint_epoch_never_exceeds_epochs_run_rounded_down():
    result = nearest_checkpoint_epoch(epochs_run=7, checkpoint_every=5, fraction=0.75)
    assert result <= 7
    assert result % 5 == 0


# --------------------------------------------------------------------------
# CLI: --max-concurrent-stages / --num-workers / --batch-size hard invariants
# --------------------------------------------------------------------------


@pytest.fixture()
def serial_module():
    return load_script("scripts/12_overnight/run_overnight.py")


def test_max_concurrent_stages_above_one_is_refused(serial_module):
    with pytest.raises(SystemExit):
        serial_module.parse_args(["--max-concurrent-stages", "2"])


def test_max_concurrent_stages_of_one_is_accepted(serial_module):
    args = serial_module.parse_args(["--max-concurrent-stages", "1"])
    assert args.max_concurrent_stages == 1


def test_num_workers_above_two_is_refused(serial_module):
    with pytest.raises(SystemExit):
        serial_module.parse_args(["--num-workers", "4"])


def test_batch_size_above_32_is_refused_for_the_shared_default(serial_module):
    with pytest.raises(SystemExit):
        serial_module.parse_args(["--batch-size", "64"])


# --------------------------------------------------------------------------
# --dry-run walks all five stages without touching the GPU or any data
# --------------------------------------------------------------------------


def test_dry_run_walks_all_five_stages_without_gpu_or_data(serial_module, tmp_path, monkeypatch):
    monkeypatch.setattr(serial_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(serial_module, "CHECKPOINT_DIR", tmp_path / "checkpoints" / "serial")
    monkeypatch.setattr(serial_module, "PROGRESS_DIR", tmp_path / "progress")
    monkeypatch.setattr(serial_module, "PROGRESS_MD", tmp_path / "progress" / "serial.md")
    monkeypatch.setattr(serial_module, "STATE_PATH", tmp_path / "reports" / "serial_state.json")

    # A dry run must never load the (16GB memory-mapped) dataset or touch CUDA.
    def _forbidden(*a, **k):
        raise AssertionError("dry-run must never load real data")

    monkeypatch.setattr(serial_module, "load_shared_data", _forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("dry-run touched CUDA")))

    argv = ["--dry-run", "--global-deadline-hours", "0.1"]
    serial_module.main(argv)

    state = json.loads((tmp_path / "reports" / "serial_state.json").read_text())
    for stage in serial_module.STAGE_NAMES:
        assert state["stages"][stage]["status"] == "dry_run_ok", state["stages"][stage]

    md = (tmp_path / "progress" / "serial.md").read_text()
    for stage in serial_module.STAGE_NAMES:
        assert stage in md


def test_dry_run_prints_the_real_launch_commands(serial_module, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(serial_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(serial_module, "CHECKPOINT_DIR", tmp_path / "checkpoints" / "serial")
    monkeypatch.setattr(serial_module, "PROGRESS_DIR", tmp_path / "progress")
    monkeypatch.setattr(serial_module, "PROGRESS_MD", tmp_path / "progress" / "serial.md")
    monkeypatch.setattr(serial_module, "STATE_PATH", tmp_path / "reports" / "serial_state.json")
    monkeypatch.setattr(serial_module, "load_shared_data", lambda *a, **k: (_ for _ in ()).throw(AssertionError))

    serial_module.main(["--dry-run", "--stages", "kfold_fold4,longrun"])
    out = capsys.readouterr().out
    assert "kfold_cnn.py" in out
    assert "--run-stage longrun" in out


# --------------------------------------------------------------------------
# Resume skips a stage already marked done in reports/serial_state.json
# --------------------------------------------------------------------------


def test_orchestrator_resume_skips_a_stage_already_done(serial_module, tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(serial_module, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(serial_module, "CHECKPOINT_DIR", tmp_path / "checkpoints" / "serial")
    monkeypatch.setattr(serial_module, "PROGRESS_DIR", tmp_path / "progress")
    monkeypatch.setattr(serial_module, "PROGRESS_MD", tmp_path / "progress" / "serial.md")
    state_path = reports_dir / "serial_state.json"
    monkeypatch.setattr(serial_module, "STATE_PATH", state_path)

    pre_state = serial_module.default_state()
    serial_module.mark_stage(pre_state, "kfold_fold4", "done")
    serial_module.save_state(pre_state, state_path)

    calls = []
    monkeypatch.setattr(serial_module, "guard_or_skip", lambda *a, **k: calls.append("guard") or True)

    # dry-run so nothing heavy actually launches, but the resume-skip check
    # happens before dry_run_stage is even called.
    serial_module.main(["--dry-run", "--stages", "kfold_fold4"])

    assert calls == []  # kfold_fold4 was already done -> guard/dry-run never invoked for it
    final_state = json.loads(state_path.read_text())
    assert final_state["stages"]["kfold_fold4"]["status"] == "done"
