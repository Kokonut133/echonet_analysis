from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.constants import METADATA_FILENAME, TARGET_LABELS
from tests.conftest import N_SAMPLES, N_WAVEFORM_FEATURES, load_script

SMOKE_TARGETS = TARGET_LABELS[:2]


def _assert_metric_series_in_range(series: pd.Series, name: str) -> None:
    valid = series.dropna()
    assert ((valid >= 0) & (valid <= 1)).all(), (
        f"'{name}' has out-of-range values: {valid[~((valid >= 0) & (valid <= 1))]}"
    )




def test_extract_waveform_features_output_shape_and_values(fake_dataset, tmp_path):
    m = load_script("scripts/2_preprocess/extract_waveform_features.py")

    config = m.Config(
        dataset_dir=fake_dataset["base"],
        output_dir=tmp_path,
        batch_size=50,
        splits=("train", "val"),
    )
    m.extract_and_cache_waveform_features(config)

    assert (tmp_path / "ecg_waveform_feature_names.json").exists()

    for split in ("train", "val"):
        feat_path = tmp_path / f"ecg_waveform_features_{split}.npy"
        assert feat_path.exists(), f"Missing: {feat_path.name}"

        arr = np.load(feat_path)
        assert arr.ndim == 2, "Expected 2-D feature matrix"
        assert arr.shape[0] == N_SAMPLES[split], (
            f"Row count mismatch for '{split}': got {arr.shape[0]}, expected {N_SAMPLES[split]}"
        )
        assert arr.shape[1] == N_WAVEFORM_FEATURES, (
            f"Feature count mismatch: got {arr.shape[1]}, expected {N_WAVEFORM_FEATURES}"
        )
        assert np.isfinite(arr).all(), f"Non-finite values in extracted features for '{split}'"




def test_demographic_baseline_results_shape_and_metric_ranges(fake_dataset, tmp_path):
    m = load_script("scripts/3_baselines/demographic_only.py")

    config = m.Config(
        metadata_path=fake_dataset["base"] / METADATA_FILENAME,
        output_path=tmp_path / "demographic_results.csv",
        targets=SMOKE_TARGETS,
        random_seed=0,
    )
    results = m.run_demographic_baseline(config)

    assert (tmp_path / "demographic_results.csv").exists()
    assert isinstance(results, pd.DataFrame) and not results.empty

    required_cols = {"target", "model", "feature_set", "auroc", "auprc", "balanced_acc", "prevalence"}
    assert required_cols.issubset(results.columns), (
        f"Missing columns: {required_cols - set(results.columns)}"
    )
    assert set(results["target"].unique()) == set(SMOKE_TARGETS)
    assert results["feature_set"].eq("demographics").all()

    _assert_metric_series_in_range(results["auroc"], "auroc")
    _assert_metric_series_in_range(results["auprc"], "auprc")
    _assert_metric_series_in_range(results["prevalence"], "prevalence")




def test_ecg_feature_comparison_covers_all_feature_sets_and_models(fake_dataset, tmp_path):
    m = load_script("scripts/4_classical_ml/compare_ecg_feature_sets.py")

    config = m.Config(
        paths=m.Paths(
            dataset_dir=fake_dataset["base"],
            features_dir=fake_dataset["features_dir"],
            metadata_path=fake_dataset["base"] / METADATA_FILENAME,
            output_path=tmp_path / "ecg_feature_results.csv",
        ),
        targets=SMOKE_TARGETS,
        random_seed=0,
    )
    results = m.run_ecg_feature_comparison(config)

    assert (tmp_path / "ecg_feature_results.csv").exists()
    assert isinstance(results, pd.DataFrame) and not results.empty

    assert set(results["feature_set"].unique()) == {"tabular_only", "waveform_only", "combined"}, (
        "Not all three feature sets were evaluated"
    )
    assert {"LogisticRegression", "RandomForest", "GradientBoosting"}.issubset(
        set(results["model"].unique())
    ), "Not all classifiers ran"

    _assert_metric_series_in_range(results["auroc"], "auroc")
    _assert_metric_series_in_range(results["auprc"], "auprc")




def test_cnn_waveforms_only_saves_checkpoint_results_and_train_log(fake_dataset, tmp_path):
    pytest.importorskip("torch")
    from src.training import TrainConfig
    m = load_script("scripts/5_deep_learning/cnn_waveforms_only.py")

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
    m.train_cnn_on_waveforms(config)

    results = pd.read_csv(tmp_path / "reports" / "cnn_waveforms_results.csv")
    train_log = pd.read_csv(tmp_path / "reports" / "cnn_waveforms_train_log.csv")

    assert (checkpoints / "cnn_waveforms.pt").exists()
    assert {"label", "auroc", "auprc", "n_total"}.issubset(results.columns)
    assert {"epoch", "train_loss", "val_mean_auroc"}.issubset(train_log.columns)
    assert len(train_log) <= 2, "Train log should have at most n_epochs rows"
    assert (train_log["train_loss"] >= 0).all(), "Loss values must be non-negative"

    _assert_metric_series_in_range(results["auroc"], "auroc")




def test_cnn_waveforms_with_demographics_saves_checkpoint_results_and_train_log(
    fake_dataset, tmp_path
):
    pytest.importorskip("torch")
    from src.training import TrainConfig
    m = load_script("scripts/5_deep_learning/cnn_waveforms_with_demographics.py")

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
    m.train_cnn_on_waveforms_and_demographics(config)

    results = pd.read_csv(tmp_path / "reports" / "cnn_combined_results.csv")
    train_log = pd.read_csv(tmp_path / "reports" / "cnn_combined_train_log.csv")

    assert (checkpoints / "cnn_combined.pt").exists()
    assert {"label", "auroc", "auprc", "n_total"}.issubset(results.columns)
    assert {"epoch", "train_loss", "val_mean_auroc"}.issubset(train_log.columns)
    assert (train_log["train_loss"] >= 0).all()

    _assert_metric_series_in_range(results["auroc"], "auroc")




def test_compute_binary_metrics_perfect_predictions_give_auroc_one():
    from src.evaluation import compute_binary_metrics
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    metrics = compute_binary_metrics(y_true, y_prob)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)


def test_compute_binary_metrics_inverted_predictions_give_auroc_zero():
    from src.evaluation import compute_binary_metrics
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    metrics = compute_binary_metrics(y_true, y_prob)
    assert metrics["auroc"] == pytest.approx(0.0)


def test_compute_binary_metrics_single_class_returns_nan():
    from src.evaluation import compute_binary_metrics
    y_true = np.array([1, 1, 1, 1])
    y_prob = np.array([0.8, 0.9, 0.7, 0.6])
    metrics = compute_binary_metrics(y_true, y_prob)
    assert math.isnan(metrics["auroc"]), "AUROC should be NaN when only one class is present"
    assert math.isnan(metrics["auprc"]), "AUPRC should be NaN when only one class is present"




def test_compute_pos_weights_balanced_labels_give_weight_one():
    from src.training import compute_pos_weights
    labels = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float32)
    weights = compute_pos_weights(labels)
    assert weights.shape == (2,)
    assert weights[0].item() == pytest.approx(1.0, abs=1e-4)
    assert weights[1].item() == pytest.approx(1.0, abs=1e-4)


def test_compute_pos_weights_rare_positive_gives_high_weight():
    from src.training import compute_pos_weights
    labels = np.array([[1]] + [[0]] * 9, dtype=np.float32)
    weights = compute_pos_weights(labels)
    assert weights[0].item() == pytest.approx(9.0, abs=0.5)


def test_compute_pos_weights_ignores_nan_rows():
    from src.training import compute_pos_weights
    labels = np.array([[1.0], [0.0], [float("nan")], [0.0], [1.0]], dtype=np.float32)
    weights_with_nan = compute_pos_weights(labels)
    labels_no_nan = np.array([[1.0], [0.0], [0.0], [1.0]], dtype=np.float32)
    weights_without_nan = compute_pos_weights(labels_no_nan)
    assert weights_with_nan[0].item() == pytest.approx(weights_without_nan[0].item(), abs=1e-4)




def test_ecg_conv_net_output_shape_without_demographics():
    pytest.importorskip("torch")
    import torch
    from src.models import ECGConvNet

    batch_size, n_leads, n_time, n_labels = 4, 12, 2500, 12
    model = ECGConvNet(n_leads=n_leads, n_labels=n_labels, n_demo_features=0)
    waveforms = torch.randn(batch_size, n_leads, n_time)
    logits = model(waveforms)
    assert logits.shape == (batch_size, n_labels), (
        f"Expected output shape ({batch_size}, {n_labels}), got {tuple(logits.shape)}"
    )


def test_ecg_conv_net_output_shape_with_demographics():
    pytest.importorskip("torch")
    import torch
    from src.models import ECGConvNet

    batch_size, n_leads, n_time, n_labels, n_demo = 4, 12, 2500, 12, 8
    model = ECGConvNet(n_leads=n_leads, n_labels=n_labels, n_demo_features=n_demo)
    waveforms = torch.randn(batch_size, n_leads, n_time)
    demo = torch.randn(batch_size, n_demo)
    logits = model(waveforms, demo)
    assert logits.shape == (batch_size, n_labels)


def test_res_block_halves_time_dimension_at_stride_two():
    pytest.importorskip("torch")
    import torch
    from src.models import ResBlock

    block = ResBlock(in_channels=32, out_channels=64, stride=2)
    x = torch.randn(2, 32, 100)
    out = block(x)
    assert out.shape == (2, 64, 50), (
        f"ResBlock with stride=2 should halve time dim: expected (2, 64, 50), got {tuple(out.shape)}"
    )




def test_compute_per_label_metrics_excludes_nan_labels():
    from src.metrics import compute_per_label_metrics

    rng = np.random.default_rng(0)
    n = 20

    label_a_true = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    label_a_prob = np.where(label_a_true == 1, 0.9, 0.1)

    label_b_true = np.concatenate([np.full(8, float("nan")), np.ones(6), np.zeros(6)])
    label_b_prob = rng.uniform(0, 1, n)

    y_true = np.stack([label_a_true, label_b_true], axis=1)
    y_prob = np.stack([label_a_prob, label_b_prob], axis=1)

    df = compute_per_label_metrics(y_true, y_prob, label_names=["label_a", "label_b"])

    row_a = df[df["label"] == "label_a"].iloc[0]
    row_b = df[df["label"] == "label_b"].iloc[0]

    assert row_a["n_total"] == n, "label_a has no NaN rows; all 20 should be counted"
    assert row_b["n_total"] == 12, "label_b has 8 NaN rows; only 12 should be counted"
    assert row_a["auroc"] == pytest.approx(1.0), "Perfect predictions should give AUROC=1.0"


def test_compute_per_label_metrics_too_few_samples_returns_nan():
    from src.metrics import compute_per_label_metrics

    y_true = np.array([[1.0], [0.0], [1.0]])   # only 3 valid rows (< threshold of 10)
    y_prob = np.array([[0.9], [0.1], [0.8]])
    df = compute_per_label_metrics(y_true, y_prob, label_names=["rare_label"])

    assert math.isnan(df.iloc[0]["auroc"]), (
        "Labels with fewer than 10 valid samples should return NaN for AUROC"
    )
