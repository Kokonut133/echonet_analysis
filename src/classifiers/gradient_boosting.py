from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_numeric_impute_pipeline
from src.constants import load_config

_cfg = load_config()["training"]["sklearn"]
_gb = _cfg["gradient_boosting"]
_seed = _cfg["random_seed"]


def gradient_boosting_pipeline(seed: int = _seed) -> Pipeline:
    return Pipeline([
        ("pre", build_numeric_impute_pipeline()),
        ("clf", GradientBoostingClassifier(
            n_estimators=_gb["n_estimators"],
            max_depth=_gb["max_depth"],
            learning_rate=_gb["learning_rate"],
            subsample=_gb["subsample"],
            random_state=seed,
        )),
    ])
