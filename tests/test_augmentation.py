from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.augmentation import (
    compose,
    eval_transform,
    random_amplitude_scale,
    random_baseline_wander,
    random_gaussian_noise,
    random_time_shift,
    standardize_per_record,
    train_augment,
)
from tests.conftest import load_script

N_LEADS = 12
N_SAMPLES = 2500


def _fake_record(rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    return (rng.standard_normal((N_LEADS, N_SAMPLES)) * scale).astype(np.float32)


# ---- shape preservation ----------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        standardize_per_record,
        random_time_shift,
        random_amplitude_scale,
        random_baseline_wander,
        random_gaussian_noise,
        train_augment,
        eval_transform,
    ],
)
def test_shape_preserved(fn):
    rng = np.random.default_rng(0)
    x = _fake_record(rng)
    out = fn(x, rng=np.random.default_rng(1))
    assert out.shape == x.shape
    assert out.dtype == np.float32


def test_shape_preserved_for_torch_tensor():
    pytest.importorskip("torch")
    import torch

    rng = np.random.default_rng(0)
    x = torch.from_numpy(_fake_record(rng))
    out = train_augment(x, rng=np.random.default_rng(1))
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape


# ---- standardisation --------------------------------------------------------


def test_standardize_per_record_gives_unit_global_std():
    rng = np.random.default_rng(3)
    # amplitude on the order of the ~10x raw-file discrepancy noted in
    # progress_log.md — normalisation should erase it.
    x = _fake_record(rng, scale=11.3) + 50.0
    out = standardize_per_record(x)
    assert out.std() == pytest.approx(1.0, abs=1e-4)


def test_standardize_per_record_preserves_lead_amplitude_ratios():
    rng = np.random.default_rng(4)
    base = _fake_record(rng)
    x = base.copy()
    x[0] *= 5.0  # lead 0 has 5x the amplitude of the others
    out = standardize_per_record(x)
    ratio_before = base[0].std() > 0 and (x[0] - x[0].mean()).std() / (
        x[1] - x[1].mean()
    ).std()
    ratio_after = (out[0] - out[0].mean()).std() / (out[1] - out[1].mean()).std()
    assert ratio_after == pytest.approx(ratio_before, rel=1e-3)


def test_standardize_per_record_eps_guarded_for_constant_input():
    x = np.zeros((N_LEADS, N_SAMPLES), dtype=np.float32)
    out = standardize_per_record(x)
    assert np.isfinite(out).all()


def test_eval_transform_is_standardization_only():
    rng = np.random.default_rng(5)
    x = _fake_record(rng, scale=3.0)
    assert np.allclose(eval_transform(x), standardize_per_record(x))


# ---- determinism -------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        random_time_shift,
        random_amplitude_scale,
        random_baseline_wander,
        random_gaussian_noise,
        train_augment,
    ],
)
def test_deterministic_given_seeded_generator(fn):
    rng_seed = 123
    x = _fake_record(np.random.default_rng(0))

    out1 = fn(x, rng=np.random.default_rng(rng_seed))
    out2 = fn(x, rng=np.random.default_rng(rng_seed))

    np.testing.assert_array_equal(out1, out2)


def test_different_seeds_give_different_augmentation():
    x = _fake_record(np.random.default_rng(0))
    out1 = train_augment(x, rng=np.random.default_rng(1))
    out2 = train_augment(x, rng=np.random.default_rng(2))
    assert not np.allclose(out1, out2)


def test_compose_threads_single_rng_through_all_fns():
    x = _fake_record(np.random.default_rng(0))
    composed = compose(random_amplitude_scale, random_gaussian_noise)

    out1 = composed(x, rng=np.random.default_rng(7))
    out2 = composed(x, rng=np.random.default_rng(7))

    np.testing.assert_array_equal(out1, out2)


# ---- smoke test of the v2 training script -----------------------------------


def test_cnn_waveforms_v2_saves_checkpoint_results_and_train_log(fake_dataset, tmp_path):
    pytest.importorskip("torch")
    from src.constants import METADATA_FILENAME
    from src.training import TrainConfig

    m = load_script("scripts/5_deep_learning/cnn_waveforms_v2.py")

    checkpoints = tmp_path / "checkpoints"
    config = m.RunConfig(
        dataset_dir=fake_dataset["base"],
        metadata_path=fake_dataset["base"] / METADATA_FILENAME,
        output_dir=tmp_path / "reports",
        checkpoint_dir=checkpoints,
        train_config=TrainConfig(
            n_epochs=2,
            batch_size=8,
            patience=2,
            num_workers=0,
            checkpoint_dir=checkpoints,
        ),
    )
    m.train_cnn_v2(config)

    results = pd.read_csv(tmp_path / "reports" / "cnn_waveforms_v2_results.csv")
    train_log = pd.read_csv(tmp_path / "reports" / "cnn_waveforms_v2_train_log.csv")

    assert (checkpoints / "cnn_waveforms_v2.pt").exists()
    assert {"label", "auroc", "auprc", "n_total"}.issubset(results.columns)
    assert {"epoch", "train_loss", "val_mean_auroc"}.issubset(train_log.columns)
    assert len(train_log) <= 2, "Train log should have at most n_epochs rows"
    assert (train_log["train_loss"] >= 0).all(), "Loss values must be non-negative"

    valid_auroc = results["auroc"].dropna()
    assert ((valid_auroc >= 0) & (valid_auroc <= 1)).all()


def test_cnn_waveforms_v2_max_batches_truncates_run(fake_dataset, tmp_path):
    pytest.importorskip("torch")
    from src.constants import METADATA_FILENAME
    from src.training import TrainConfig

    m = load_script("scripts/5_deep_learning/cnn_waveforms_v2.py")

    checkpoints = tmp_path / "checkpoints"
    config = m.RunConfig(
        dataset_dir=fake_dataset["base"],
        metadata_path=fake_dataset["base"] / METADATA_FILENAME,
        output_dir=tmp_path / "reports",
        checkpoint_dir=checkpoints,
        train_config=TrainConfig(
            n_epochs=1,
            batch_size=8,
            patience=1,
            num_workers=0,
            checkpoint_dir=checkpoints,
        ),
    )
    # fake train split has 100 samples; cap to 2 batches of 8 = 16 samples
    m.train_cnn_v2(config, max_batches=2)

    assert (checkpoints / "cnn_waveforms_v2.pt").exists()
