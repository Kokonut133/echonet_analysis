from __future__ import annotations

import numpy as np
import optuna
import pytest

from src.tuning import (
    HGB_PARAM_SPACE,
    LR_C_SPACE,
    LR_CLASS_WEIGHT_SPACE,
    LR_L1_RATIO_SPACE,
    LR_PENALTY_SOLVER_SPACE,
    build_hgb_params,
    build_lr_params,
    params_to_pipeline,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _ask() -> optuna.trial.Trial:
    """A live optuna Trial from a throwaway study — used as the 'stubbed trial'
    so search-space builders are exercised through the real suggest_* API."""
    study = optuna.create_study(direction="maximize")
    return study.ask()


# --- search-space bounds -------------------------------------------------------

def test_hgb_search_space_within_declared_bounds():
    trial = _ask()
    params = build_hgb_params(trial)

    assert set(params.keys()) == set(HGB_PARAM_SPACE.keys())
    for name, spec in HGB_PARAM_SPACE.items():
        value = params[name]
        assert spec["low"] <= value <= spec["high"], f"{name}={value} outside [{spec['low']}, {spec['high']}]"
        if spec["type"] == "int":
            assert isinstance(value, int)


def test_hgb_search_space_many_trials_stay_in_bounds():
    study = optuna.create_study(direction="maximize")
    for _ in range(25):
        trial = study.ask()
        params = build_hgb_params(trial)
        for name, spec in HGB_PARAM_SPACE.items():
            assert spec["low"] <= params[name] <= spec["high"]
        study.tell(trial, 0.5)


def test_lr_search_space_within_declared_bounds():
    trial = _ask()
    params = build_lr_params(trial)

    assert LR_C_SPACE["low"] <= params["C"] <= LR_C_SPACE["high"]
    assert params["solver"] in {solver for _, solver in LR_PENALTY_SOLVER_SPACE.values()}
    assert LR_L1_RATIO_SPACE["low"] <= params["l1_ratio"] <= LR_L1_RATIO_SPACE["high"]
    assert params["class_weight"] in LR_CLASS_WEIGHT_SPACE


def test_lr_search_space_many_trials_valid_solver_l1_ratio_pairs():
    # every sampled (solver, l1_ratio) must be one scikit-learn actually accepts
    valid_pairs = set(LR_PENALTY_SOLVER_SPACE.values())
    study = optuna.create_study(direction="maximize")
    for _ in range(25):
        trial = study.ask()
        params = build_lr_params(trial)
        label = trial.params["penalty_solver"]
        fixed_l1_ratio, solver = LR_PENALTY_SOLVER_SPACE[label]
        assert params["solver"] == solver
        if fixed_l1_ratio is not None:
            assert params["l1_ratio"] == fixed_l1_ratio
        else:
            assert 0.0 <= params["l1_ratio"] <= 1.0
        study.tell(trial, 0.5)
    assert valid_pairs  # sanity: space is non-empty


# --- params -> fitted-able pipeline ---------------------------------------------

def _synthetic_data(n=80, d=12, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    # sprinkle some NaNs so the imputer in the pipeline is exercised
    nan_mask = rng.rand(n, d) < 0.05
    X[nan_mask] = np.nan
    y = (rng.rand(n) > 0.5).astype(int)
    return X, y


@pytest.mark.parametrize("model_name", ["HistGradientBoostingClassifier", "LogisticRegression"])
def test_params_to_pipeline_fits_on_synthetic_matrix(model_name):
    trial = _ask()
    params = build_hgb_params(trial) if model_name == "HistGradientBoostingClassifier" else build_lr_params(trial)
    # keep the synthetic fit fast regardless of what got sampled
    if model_name == "HistGradientBoostingClassifier":
        params["max_iter"] = min(params["max_iter"], 20)

    pipeline = params_to_pipeline(model_name, params, seed=0)
    X, y = _synthetic_data()

    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)

    assert proba.shape == (X.shape[0], 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_params_to_pipeline_unknown_model_raises():
    with pytest.raises(ValueError):
        params_to_pipeline("NotAModel", {}, seed=0)
