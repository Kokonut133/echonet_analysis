"""Tests for src/calibration.py."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.calibration import (
    apply_isotonic,
    apply_platt,
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    max_calibration_error,
    reliability_curve,
)


def _perfectly_calibrated(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores that are themselves valid probabilities: y ~ Bernoulli(p)."""
    rng = np.random.default_rng(seed)
    y_prob = rng.uniform(0.01, 0.99, size=n)
    y_true = (rng.uniform(size=n) < y_prob).astype(float)
    return y_true, y_prob


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _overconfident(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores pushed toward 0/1 far more than the true probability warrants.

    A latent feature x drives the true probability p = sigmoid(x); labels are
    drawn from that. The "model" score is sigmoid(k * x) with k >> 1, i.e. the
    same monotone ranking (so AUROC is unaffected) but pushed hard toward 0/1
    -- exactly the shape a pos_weight-inflated logit produces: correct
    ordering, badly miscalibrated magnitude.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=1.0, size=n)
    true_p = _sigmoid(x)
    y_true = (rng.uniform(size=n) < true_p).astype(float)
    y_prob = _sigmoid(3.0 * x)
    return y_true, y_prob


class TestBrier:
    def test_analytic_value(self):
        # y_true fixed, y_prob fixed -> brier is exactly mean((p-y)^2)
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.8, 0.2, 0.6, 0.4])
        expected = np.mean((y_prob - y_true) ** 2)
        assert brier_score(y_true, y_prob) == pytest.approx(expected)

    def test_perfect_predictions_zero_brier(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(y_true, y_prob) == pytest.approx(0.0)

    def test_skill_score_positive_for_informative_model(self):
        y_true, y_prob = _perfectly_calibrated(5000, seed=1)
        bss = brier_skill_score(y_true, y_prob)
        assert bss > 0.0

    def test_skill_score_zero_for_climatology(self):
        rng = np.random.default_rng(2)
        y_true = (rng.uniform(size=2000) < 0.3).astype(float)
        # climatology = the SAMPLE prevalence, not the generating probability
        y_prob = np.full(2000, y_true.mean())
        bss = brier_skill_score(y_true, y_prob)
        assert bss == pytest.approx(0.0, abs=1e-6)


class TestCalibrationError:
    def test_perfectly_calibrated_low_ece(self):
        y_true, y_prob = _perfectly_calibrated(20000, seed=3)
        ece = expected_calibration_error(y_true, y_prob, n_bins=10, strategy="quantile")
        assert ece < 0.03

    def test_overconfident_large_ece(self):
        y_true, y_prob = _overconfident(5000, seed=4)
        ece = expected_calibration_error(y_true, y_prob, n_bins=10, strategy="quantile")
        assert ece > 0.1

    def test_mce_at_least_ece(self):
        y_true, y_prob = _overconfident(5000, seed=5)
        ece = expected_calibration_error(y_true, y_prob)
        mce = max_calibration_error(y_true, y_prob)
        assert mce >= ece

    def test_reliability_curve_bin_counts_sum_to_n(self):
        y_true, y_prob = _perfectly_calibrated(1000, seed=6)
        mean_pred, obs_freq, counts = reliability_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        assert counts.sum() == 1000
        assert mean_pred.shape == obs_freq.shape == counts.shape

    def test_nan_labels_dropped(self):
        rng = np.random.default_rng(7)
        y_true, y_prob = _perfectly_calibrated(2000, seed=7)
        y_true_with_nan = y_true.copy()
        nan_mask = rng.uniform(size=2000) < 0.3
        y_true_with_nan[nan_mask] = np.nan

        ece_dropped = expected_calibration_error(y_true_with_nan, y_prob)
        ece_clean = expected_calibration_error(y_true[~nan_mask], y_prob[~nan_mask])
        assert ece_dropped == pytest.approx(ece_clean)

        brier_dropped = brier_score(y_true_with_nan, y_prob)
        brier_clean = brier_score(y_true[~nan_mask], y_prob[~nan_mask])
        assert brier_dropped == pytest.approx(brier_clean)


class TestRecalibrationPreservesRanking:
    def test_platt_preserves_auroc(self):
        y_true, y_prob = _overconfident(4000, seed=8)
        # split so we're not fitting and scoring on the same rows
        half = 2000
        model = fit_platt(y_true[:half], y_prob[:half])
        calibrated = apply_platt(model, y_prob[half:])
        auroc_raw = roc_auc_score(y_true[half:], y_prob[half:])
        auroc_calibrated = roc_auc_score(y_true[half:], calibrated)
        assert auroc_calibrated == pytest.approx(auroc_raw, abs=1e-6)

    def test_isotonic_preserves_auroc(self):
        # Isotonic regression is monotone NON-decreasing (not strictly
        # increasing): fit on held-out data it can map two distinct raw
        # scores to the same flat step, introducing a tie that wasn't there
        # before. AUROC counts ties as half-credit, so a handful of such
        # ties can nudge AUROC by a hair -- this checks it stays within
        # a small tolerance, not bit-exact equality.
        y_true, y_prob = _overconfident(4000, seed=9)
        half = 2000
        model = fit_isotonic(y_true[:half], y_prob[:half])
        calibrated = apply_isotonic(model, y_prob[half:])
        auroc_raw = roc_auc_score(y_true[half:], y_prob[half:])
        auroc_calibrated = roc_auc_score(y_true[half:], calibrated)
        assert auroc_calibrated == pytest.approx(auroc_raw, abs=1e-3)

    def test_platt_reduces_ece_when_overconfident(self):
        y_true, y_prob = _overconfident(6000, seed=10)
        half = 3000
        model = fit_platt(y_true[:half], y_prob[:half])
        calibrated = apply_platt(model, y_prob[half:])
        ece_raw = expected_calibration_error(y_true[half:], y_prob[half:])
        ece_calibrated = expected_calibration_error(y_true[half:], calibrated)
        assert ece_calibrated < ece_raw

    def test_isotonic_reduces_ece_when_overconfident(self):
        y_true, y_prob = _overconfident(6000, seed=11)
        half = 3000
        model = fit_isotonic(y_true[:half], y_prob[:half])
        calibrated = apply_isotonic(model, y_prob[half:])
        ece_raw = expected_calibration_error(y_true[half:], y_prob[half:])
        ece_calibrated = expected_calibration_error(y_true[half:], calibrated)
        assert ece_calibrated < ece_raw

    def test_fit_platt_drops_nan_labels(self):
        y_true, y_prob = _overconfident(2000, seed=12)
        y_true_nan = y_true.copy()
        y_true_nan[:100] = np.nan
        # should not raise and should fit on the remaining 1900 rows
        model = fit_platt(y_true_nan, y_prob)
        calibrated = apply_platt(model, y_prob)
        assert calibrated.shape == y_prob.shape
