from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.preprocessing import build_numeric_impute_pipeline
from src.constants import load_config

_cfg = load_config()["training"]["sklearn"]
_rf = _cfg["random_forest"]
_seed = _cfg["random_seed"]


def random_forest_pipeline(seed: int = _seed) -> Pipeline:
    return Pipeline([
        ("pre", build_numeric_impute_pipeline()),
        ("clf", RandomForestClassifier(
            n_estimators=_rf["n_estimators"],
            class_weight="balanced",
            min_samples_leaf=_rf["min_samples_leaf"],
            random_state=seed,
            n_jobs=-1,
        )),
    ])
