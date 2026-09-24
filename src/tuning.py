"""Optuna hyperparameter search for the classical (non-CNN) models.

Reusable pieces for `scripts/11_tuning/tune_classical.py`:
  - declared search-space bounds for HistGradientBoostingClassifier and
    LogisticRegression, plus the functions that sample from them (so tests
    can check a stubbed trial's suggestions stay inside the declared bounds)
  - `params_to_pipeline`, which turns a sampled params dict into a fitted-able
    sklearn Pipeline (imputation [+ scaling for LR] -> classifier)
  - `run_study`, which builds an Optuna study (TPE sampler, MedianPruner for
    the model that supports staged scoring) and optimises it against a fixed
    train/val split

CPU-only by construction: nothing here touches a GPU. Every .fit() call is
additionally wrapped in `threadpoolctl.threadpool_limits(MAX_THREADS)` so
total BLAS/OpenMP parallelism stays <= 3 even if a caller forgets to set
OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS before import (the
tuning script sets those too, belt and braces).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import optuna
import threadpoolctl
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import TPESampler
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from src.preprocessing import build_numeric_impute_pipeline, build_numeric_impute_scale_pipeline

# --- search-space declarations ------------------------------------------------
# Declared as data (not buried inside the sampling function) so a test can
# assert a stubbed trial's suggestions never leave these bounds.

HGB_PARAM_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate":     {"type": "float", "low": 0.01,   "high": 0.3,  "log": True},
    "max_iter":          {"type": "int",   "low": 50,     "high": 500},
    "max_leaf_nodes":    {"type": "int",   "low": 7,      "high": 127},
    "min_samples_leaf":  {"type": "int",   "low": 5,      "high": 200},
    "l2_regularization": {"type": "float", "low": 1e-8,   "high": 10.0, "log": True},
    "max_bins":          {"type": "int",   "low": 32,     "high": 255},
}

# scikit-learn 1.9 deprecated the `penalty` constructor argument in favour of
# `l1_ratio` (0.0 == "l2", 1.0 == "l1", in between == "elasticnet"), so the
# penalty/solver search space is expressed directly in l1_ratio terms — this
# also sidesteps the FutureWarning that passing `penalty=` now raises.
# label -> (fixed l1_ratio, or None if it should be tuned, solver)
LR_PENALTY_SOLVER_SPACE: dict[str, tuple[float | None, str]] = {
    "l2_lbfgs":        (0.0, "lbfgs"),
    "l2_liblinear":    (0.0, "liblinear"),
    "l1_liblinear":    (1.0, "liblinear"),
    "l1_saga":         (1.0, "saga"),
    "elasticnet_saga": (None, "saga"),
}
LR_C_SPACE: dict[str, Any] = {"type": "float", "low": 1e-4, "high": 1e2, "log": True}
LR_CLASS_WEIGHT_SPACE: list[Any] = [None, "balanced"]
LR_L1_RATIO_SPACE: dict[str, Any] = {"type": "float", "low": 0.0, "high": 1.0}
LR_FIXED_MAX_ITER = 2000  # not tuned — matches the project's untuned config

MODEL_NAMES: list[str] = ["HistGradientBoostingClassifier", "LogisticRegression"]

# HistGB fitting is staged internally (warm_start lets us add trees
# incrementally), so it gets a real MedianPruner. LogisticRegression has no
# staged-scoring equivalent in this project (no partial_fit path that would
# make early stopping meaningful for saga/liblinear/lbfgs), so it is left
# unpruned per the task's "otherwise leave unpruned" instruction.
PRUNING_STEP = 25  # boosting rounds added per staged-evaluation checkpoint
MAX_THREADS = 3  # hardware rule: keep total parallelism at 3 or below (no GPU, shared box)


def _suggest_from_spec(trial: optuna.trial.Trial, name: str, spec: dict[str, Any]):
    if spec["type"] == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))


def build_hgb_params(trial: optuna.trial.Trial) -> dict[str, Any]:
    """Sample a HistGradientBoostingClassifier param dict from HGB_PARAM_SPACE."""
    return {name: _suggest_from_spec(trial, name, spec) for name, spec in HGB_PARAM_SPACE.items()}


def build_lr_params(trial: optuna.trial.Trial) -> dict[str, Any]:
    """Sample a LogisticRegression param dict: C, penalty/solver (as l1_ratio+solver), class_weight."""
    params: dict[str, Any] = {
        "C": _suggest_from_spec(trial, "C", LR_C_SPACE),
    }
    label = trial.suggest_categorical("penalty_solver", list(LR_PENALTY_SOLVER_SPACE.keys()))
    fixed_l1_ratio, solver = LR_PENALTY_SOLVER_SPACE[label]
    params["solver"] = solver
    params["l1_ratio"] = (
        _suggest_from_spec(trial, "l1_ratio", LR_L1_RATIO_SPACE) if fixed_l1_ratio is None else fixed_l1_ratio
    )
    params["class_weight"] = trial.suggest_categorical("class_weight", LR_CLASS_WEIGHT_SPACE)
    return params


SEARCH_SPACE_BUILDERS: dict[str, Callable[[optuna.trial.Trial], dict[str, Any]]] = {
    "HistGradientBoostingClassifier": build_hgb_params,
    "LogisticRegression": build_lr_params,
}


# --- params -> pipeline --------------------------------------------------------

def params_to_pipeline(model_name: str, params: dict[str, Any], seed: int = 42) -> Pipeline:
    """Turn a sampled (or best) params dict into a fitted-able sklearn Pipeline."""
    if model_name == "HistGradientBoostingClassifier":
        p = dict(params)
        p.setdefault("early_stopping", False)  # match the setting used during tuning
        clf = HistGradientBoostingClassifier(random_state=seed, **p)
        return Pipeline([("pre", build_numeric_impute_pipeline()), ("clf", clf)])

    if model_name == "LogisticRegression":
        p = dict(params)
        p.setdefault("max_iter", LR_FIXED_MAX_ITER)
        clf = LogisticRegression(random_state=seed, **p)
        return Pipeline([("pre", build_numeric_impute_scale_pipeline()), ("clf", clf)])

    raise ValueError(f"Unknown model_name: {model_name!r}")


# --- study construction ---------------------------------------------------------

@dataclass
class PreparedData:
    """Preprocessed (imputed [+ scaled]) train/val matrices for one model family.

    Preprocessing itself is not tuned, so it's fit once outside the trial loop
    rather than refit on every trial.
    """
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray


def _prepare_data(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> PreparedData:
    if model_name == "HistGradientBoostingClassifier":
        pre = build_numeric_impute_pipeline()
    elif model_name == "LogisticRegression":
        pre = build_numeric_impute_scale_pipeline()
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")
    Xt_train = pre.fit_transform(X_train)
    Xt_val = pre.transform(X_val)
    return PreparedData(X_train=Xt_train, y_train=y_train, X_val=Xt_val, y_val=y_val)


def _hgb_objective(data: PreparedData, seed: int) -> Callable[[optuna.trial.Trial], float]:
    def objective(trial: optuna.trial.Trial) -> float:
        params = build_hgb_params(trial)
        max_iter = params.pop("max_iter")
        # early_stopping defaults to 'auto' (True above 10k rows), which would let
        # the estimator silently cap boosting rounds below our sampled max_iter —
        # defeating the point of tuning it. Disable it so max_iter is authoritative.
        clf = HistGradientBoostingClassifier(
            random_state=seed, warm_start=True, early_stopping=False, max_iter=PRUNING_STEP, **params
        )

        current_iter = 0
        auroc = float("nan")
        with warnings.catch_warnings(), threadpoolctl.threadpool_limits(limits=MAX_THREADS):
            warnings.simplefilter("ignore", ConvergenceWarning)
            while current_iter < max_iter:
                current_iter = min(current_iter + PRUNING_STEP, max_iter)
                clf.max_iter = current_iter
                clf.fit(data.X_train, data.y_train)
                y_prob = clf.predict_proba(data.X_val)[:, 1]
                auroc = roc_auc_score(data.y_val, y_prob)
                trial.report(auroc, current_iter)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return auroc

    return objective


def _lr_objective(data: PreparedData, seed: int) -> Callable[[optuna.trial.Trial], float]:
    def objective(trial: optuna.trial.Trial) -> float:
        params = build_lr_params(trial)
        params["max_iter"] = LR_FIXED_MAX_ITER
        clf = LogisticRegression(random_state=seed, **params)
        with warnings.catch_warnings(), threadpoolctl.threadpool_limits(limits=MAX_THREADS):
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(data.X_train, data.y_train)
            y_prob = clf.predict_proba(data.X_val)[:, 1]
        return roc_auc_score(data.y_val, y_prob)

    return objective


_OBJECTIVE_BUILDERS: dict[str, Callable[[PreparedData, int], Callable]] = {
    "HistGradientBoostingClassifier": _hgb_objective,
    "LogisticRegression": _lr_objective,
}


def run_study(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    timeout: float,
    seed: int = 42,
    study_name: str | None = None,
) -> optuna.Study:
    """Run a TPE-sampled Optuna study for `model_name` on a fixed train/val split.

    Stops at whichever of `n_trials` / `timeout` (seconds) comes first.
    HistGradientBoostingClassifier gets a MedianPruner driven by a warm-start
    staged-evaluation loop; LogisticRegression is left unpruned (no staged
    scoring path).
    """
    if model_name not in _OBJECTIVE_BUILDERS:
        raise ValueError(f"Unknown model_name: {model_name!r}")

    data = _prepare_data(model_name, X_train, y_train, X_val, y_val)
    objective = _OBJECTIVE_BUILDERS[model_name](data, seed)

    pruner = MedianPruner(n_warmup_steps=2) if model_name == "HistGradientBoostingClassifier" else NopPruner()
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1, show_progress_bar=False)
    return study


def trial_value_for_reporting(trial: optuna.trial.FrozenTrial) -> float:
    """AUROC to log for a trial, including pruned trials (last reported value)."""
    if trial.value is not None:
        return trial.value
    if trial.intermediate_values:
        return max(trial.intermediate_values.values())
    return float("nan")


def safe_param_importances(study: optuna.Study) -> dict[str, float]:
    """optuna.importance.get_param_importances, tolerant of degenerate studies."""
    try:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) < 2:
            return {}
        return dict(optuna.importance.get_param_importances(study))
    except Exception:
        return {}
