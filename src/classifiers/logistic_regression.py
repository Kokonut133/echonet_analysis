from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing import build_numeric_impute_scale_pipeline
from src.constants import load_config

_cfg = load_config()["training"]["sklearn"]
_lr = _cfg["logistic_regression"]
_seed = _cfg["random_seed"]


def logistic_regression_pipeline(seed: int = _seed) -> Pipeline:
    return Pipeline([
        ("pre", build_numeric_impute_scale_pipeline()),
        ("clf", LogisticRegression(
            max_iter=_lr["max_iter"],
            class_weight="balanced",
            solver=_lr["solver"],
            random_state=seed,
        )),
    ])
