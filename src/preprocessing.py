from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import CATEGORICAL_DEMOGRAPHIC_FEATURES, NUMERIC_DEMOGRAPHIC_FEATURES


def build_numeric_impute_scale_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])


def build_numeric_impute_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])


def build_demographic_preprocessor() -> ColumnTransformer:
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric",     build_numeric_impute_scale_pipeline(), NUMERIC_DEMOGRAPHIC_FEATURES),
        ("categorical", categorical_pipe,                      CATEGORICAL_DEMOGRAPHIC_FEATURES),
    ])
