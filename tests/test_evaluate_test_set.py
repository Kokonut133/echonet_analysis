from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import METADATA_FILENAME, TARGET_LABELS
from tests.conftest import N_SAMPLES, load_script

SMOKE_TARGETS = TARGET_LABELS[:2]


def _assert_unit_interval(series: pd.Series, name: str) -> None:
    valid = series.dropna()
    assert ((valid >= 0) & (valid <= 1)).all(), (
        f"'{name}' has out-of-range values: {valid[~((valid >= 0) & (valid <= 1))]}"
    )


@pytest.fixture(scope="module")
def evaluation_result(fake_dataset, tmp_path_factory):
    pytest.importorskip("torch")
    from src.training import TrainConfig

    tmp_path = tmp_path_factory.mktemp("evaluate_test_set")

    fs = load_script("scripts/4_classical_ml/compare_ecg_feature_sets.py")
    feature_results_path = tmp_path / "ecg_feature_results.csv"
    fs_config = fs.Config(
        paths=fs.Paths(
            dataset_dir=fake_dataset["base"],
            features_dir=fake_dataset["features_dir"],
            metadata_path=fake_dataset["base"] / METADATA_FILENAME,
            output_path=feature_results_path,
        ),
        targets=SMOKE_TARGETS,
        random_seed=0,
    )
    fs.run_ecg_feature_comparison(fs_config)

    cnn = load_script("scripts/5_deep_learning/cnn_waveforms_only.py")
    checkpoints = tmp_path / "checkpoints"
    cnn_config = cnn.RunConfig(
        dataset_dir=fake_dataset["base"],
        metadata_path=fake_dataset["base"] / METADATA_FILENAME,
        output_dir=tmp_path / "cnn_reports",
        checkpoint_dir=checkpoints,
        train_config=TrainConfig(
            n_epochs=1,
            batch_size=8,
            patience=1,
            num_workers=0,
            checkpoint_dir=checkpoints,
        ),
    )
    cnn.train_cnn_on_waveforms(cnn_config)

    m = load_script("scripts/6_evaluate/evaluate_test_set.py")
    reports_dir = tmp_path / "reports"
    predictions_dir = reports_dir / "predictions"
    config = m.Config(
        paths=m.Paths(
            dataset_dir=fake_dataset["base"],
            features_dir=fake_dataset["features_dir"],
            metadata_path=fake_dataset["base"] / METADATA_FILENAME,
            feature_results_path=feature_results_path,
            checkpoint_path=checkpoints / "cnn_waveforms.pt",
            reports_dir=reports_dir,
            predictions_dir=predictions_dir,
        ),
        targets=SMOKE_TARGETS,
        n_bootstrap=20,
        bootstrap_seed=0,
    )
    results = m.run_evaluation(config)

    return {
        "module": m,
        "config": config,
        "results": results,
        "reports_dir": reports_dir,
        "predictions_dir": predictions_dir,
    }


def test_final_results_schema_and_ranges(evaluation_result):
    results = evaluation_result["results"]
    reports_dir = evaluation_result["reports_dir"]

    assert (reports_dir / "final_results.csv").exists()
    assert (reports_dir / "final_results_summary.md").exists()
    assert isinstance(results, pd.DataFrame) and not results.empty

    required_cols = {
        "target", "tier", "model", "split",
        "auroc", "auroc_ci_low", "auroc_ci_high",
        "auprc", "auprc_ci_low", "auprc_ci_high",
        "balanced_acc", "prevalence", "n_positive", "n_total",
    }
    assert required_cols.issubset(results.columns)

    assert set(results["tier"].unique()) == {
        "demographics", "tabular_ecg", "waveform_features", "combined", "cnn_raw_waveform",
    }
    assert (results["split"] == "test").all()
    assert set(results["target"].unique()) == set(SMOKE_TARGETS)

    for col in ("auroc", "auprc", "auroc_ci_low", "auroc_ci_high", "auprc_ci_low", "auprc_ci_high", "balanced_acc", "prevalence"):
        _assert_unit_interval(results[col], col)

    # CI bounds should bracket the point estimate (within rounding).
    assert (results["auroc_ci_low"] <= results["auroc"] + 1e-6).all()
    assert (results["auroc_ci_high"] >= results["auroc"] - 1e-6).all()

    assert (results["n_total"] == N_SAMPLES["test"]).all()
    assert (results["n_positive"] <= results["n_total"]).all()


def test_prediction_npz_files_written_for_every_tier(evaluation_result):
    predictions_dir = evaluation_result["predictions_dir"]

    classical_tiers = ["tabular_ecg", "waveform_features", "combined"]
    for tier in classical_tiers:
        npz_path = predictions_dir / f"{tier}_test.npz"
        assert npz_path.exists(), f"Missing predictions for tier {tier}"
        data = np.load(npz_path, allow_pickle=True)
        assert {"y_true", "y_prob", "label_names", "record_index"}.issubset(data.keys())
        assert data["y_true"].shape == (N_SAMPLES["test"], len(SMOKE_TARGETS))
        assert data["y_prob"].shape == data["y_true"].shape
        assert list(data["label_names"]) == list(SMOKE_TARGETS)
        assert np.array_equal(data["record_index"], np.arange(N_SAMPLES["test"]))
        assert ((data["y_prob"] >= 0) & (data["y_prob"] <= 1)).all()

    # demographics tier also persists model_names (needed to identify the cached model later)
    demo_path = predictions_dir / "demographics_test.npz"
    assert demo_path.exists()
    demo_data = np.load(demo_path, allow_pickle=True)
    assert "model_names" in demo_data
    assert len(demo_data["model_names"]) == len(SMOKE_TARGETS)

    cnn_path = predictions_dir / "cnn_raw_waveform_test.npz"
    assert cnn_path.exists()
    cnn_data = np.load(cnn_path, allow_pickle=True)
    assert cnn_data["y_true"].shape == (N_SAMPLES["test"], len(TARGET_LABELS))
    assert cnn_data["y_prob"].shape == cnn_data["y_true"].shape
    assert ((cnn_data["y_prob"] >= 0) & (cnn_data["y_prob"] <= 1)).all()


def test_second_run_reuses_cached_predictions_without_refitting(evaluation_result, monkeypatch):
    """A second call to run_evaluation with the same predictions_dir must not refit any
    classical/demographics model or reload the CNN — everything should come from cache."""
    m = evaluation_result["module"]
    config = evaluation_result["config"]

    def _boom_suite(*args, **kwargs):
        raise AssertionError("standard_classifier_suite() should not be called when predictions are cached")

    def _boom_cnn(*args, **kwargs):
        raise AssertionError("ECGConvNet should not be instantiated when CNN predictions are cached")

    monkeypatch.setattr(m, "standard_classifier_suite", _boom_suite)
    monkeypatch.setattr(m, "ECGConvNet", _boom_cnn)

    results = m.run_evaluation(config)

    assert isinstance(results, pd.DataFrame) and not results.empty
    assert set(results["tier"].unique()) == {
        "demographics", "tabular_ecg", "waveform_features", "combined", "cnn_raw_waveform",
    }
    _assert_unit_interval(results["auroc"], "auroc")


def test_refit_flag_bypasses_cache(evaluation_result, monkeypatch):
    """--refit (config.refit=True) must ignore the cache and hit standard_classifier_suite again."""
    import dataclasses

    m = evaluation_result["module"]
    config = dataclasses.replace(evaluation_result["config"], refit=True, targets=SMOKE_TARGETS[:1])

    calls = {"n": 0}
    original_suite = m.standard_classifier_suite

    def _counting_suite(*args, **kwargs):
        calls["n"] += 1
        return original_suite(*args, **kwargs)

    monkeypatch.setattr(m, "standard_classifier_suite", _counting_suite)

    m.run_evaluation(config)

    assert calls["n"] > 0, "refit=True should still call standard_classifier_suite()"


def test_best_model_name_picks_highest_val_auroc():
    m = load_script("scripts/6_evaluate/evaluate_test_set.py")
    feature_results = pd.DataFrame([
        {"target": "t1", "feature_set": "tabular_only", "model": "LogisticRegression", "auroc": 0.60},
        {"target": "t1", "feature_set": "tabular_only", "model": "RandomForest", "auroc": 0.75},
        {"target": "t1", "feature_set": "tabular_only", "model": "GradientBoosting", "auroc": 0.70},
    ])
    assert m.best_model_name(feature_results, "t1", "tabular_only") == "RandomForest"
    assert m.best_model_name(feature_results, "missing_target", "tabular_only") is None
