from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.classifiers.gradient_boosting import gradient_boosting_pipeline
from src.classifiers.logistic_regression import logistic_regression_pipeline
from src.classifiers.random_forest import random_forest_pipeline
from src.constants import load_config

_seed = load_config()["training"]["sklearn"]["random_seed"]


def standard_classifier_suite(seed: int = _seed) -> dict[str, Pipeline]:
    return {
        "LogisticRegression": logistic_regression_pipeline(seed),
        "RandomForest": random_forest_pipeline(seed),
        "GradientBoosting": gradient_boosting_pipeline(seed),
    }
