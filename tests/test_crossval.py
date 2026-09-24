from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.crossval import make_folds
from src.dataset import SplitData
from src.dataset_kfold import (
    ConcatSplitDataset,
    build_pool_metadata,
    fit_fold_demographic_encoder,
    transform_demo,
)


def _fake_pool(n: int, seed: int = 0, prevalence: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    flags = (rng.random(n) < prevalence).astype(float)
    return pd.DataFrame({
        "ecg_key": np.arange(n),
        "shd_moderate_or_greater_flag": flags,
    })


def test_folds_partition_pool_exactly_once_with_no_overlap():
    pool = _fake_pool(500)
    folds = make_folds(pool, k=5, seed=42)

    assert len(folds) == 5

    all_heldout = []
    for train_idx, heldout_idx in folds:
        # no overlap between train and heldout within a fold
        assert set(train_idx.tolist()).isdisjoint(set(heldout_idx.tolist()))
        # train + heldout covers the whole pool within a fold
        assert set(train_idx.tolist()) | set(heldout_idx.tolist()) == set(range(len(pool)))
        all_heldout.extend(heldout_idx.tolist())

    # every row appears in exactly one fold's heldout set, across all folds
    assert sorted(all_heldout) == list(range(len(pool)))
    assert len(all_heldout) == len(set(all_heldout))


def test_every_fold_has_both_classes_for_the_stratification_target():
    pool = _fake_pool(500, prevalence=0.3)
    folds = make_folds(pool, k=5, seed=42)

    y = pool["shd_moderate_or_greater_flag"].to_numpy()
    for train_idx, heldout_idx in folds:
        assert set(np.unique(y[train_idx]).tolist()) == {0.0, 1.0}
        assert set(np.unique(y[heldout_idx]).tolist()) == {0.0, 1.0}


def test_fold_assignment_is_deterministic_under_fixed_seed():
    pool = _fake_pool(500)
    folds_a = make_folds(pool, k=5, seed=42)
    folds_b = make_folds(pool, k=5, seed=42)

    assert len(folds_a) == len(folds_b)
    for (train_a, held_a), (train_b, held_b) in zip(folds_a, folds_b):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(held_a, held_b)


def test_different_seeds_give_different_fold_assignment():
    pool = _fake_pool(500)
    folds_a = make_folds(pool, k=5, seed=42)
    folds_b = make_folds(pool, k=5, seed=7)

    # at least one fold's heldout set should differ between the two seeds
    differs = any(
        not np.array_equal(np.sort(a[1]), np.sort(b[1]))
        for a, b in zip(folds_a, folds_b)
    )
    assert differs


def test_stratification_keeps_class_balance_close_across_folds():
    pool = _fake_pool(2000, prevalence=0.3)
    folds = make_folds(pool, k=5, seed=42)
    y = pool["shd_moderate_or_greater_flag"].to_numpy()
    overall_prevalence = y.mean()

    for _, heldout_idx in folds:
        fold_prevalence = y[heldout_idx].mean()
        assert abs(fold_prevalence - overall_prevalence) < 0.05


def test_raises_on_stratify_col_with_nan():
    pool = _fake_pool(200)
    pool.loc[0, "shd_moderate_or_greater_flag"] = np.nan
    with pytest.raises(ValueError):
        make_folds(pool, k=5, seed=42)


def test_raises_on_missing_stratify_col():
    pool = _fake_pool(200).drop(columns=["shd_moderate_or_greater_flag"])
    with pytest.raises(KeyError):
        make_folds(pool, k=5, seed=42)


# --- src.dataset_kfold -------------------------------------------------------

def _fake_split(n: int, n_labels: int = 2, offset: float = 0.0) -> SplitData:
    waveforms = np.full((n, 4, 10), offset, dtype=np.float32)
    for i in range(n):
        waveforms[i] += i  # distinguishable per-row content
    labels = np.zeros((n, n_labels), dtype=np.float32)
    demo = np.zeros((n, 3), dtype=np.float32)
    return SplitData(waveforms=waveforms, labels=labels, demo_features=demo, label_names=["a", "b"][:n_labels])


def test_build_pool_metadata_orders_train_then_val():
    metadata = pd.DataFrame({
        "ecg_key": [10, 11, 20, 21, 22, 99],
        "split": ["train", "train", "val", "val", "val", "test"],
    })
    pool = build_pool_metadata(metadata)
    assert pool["ecg_key"].tolist() == [10, 11, 20, 21, 22]


def test_concat_split_dataset_routes_global_index_to_correct_file_and_row():
    first = _fake_split(3, offset=0.0)
    second = _fake_split(2, offset=100.0)
    ds = ConcatSplitDataset(first, second)

    assert len(ds) == 5
    assert ds.file_and_row(0) == ("first", 0)
    assert ds.file_and_row(2) == ("first", 2)
    assert ds.file_and_row(3) == ("second", 0)
    assert ds.file_and_row(4) == ("second", 1)
    with pytest.raises(IndexError):
        ds.file_and_row(5)

    # row content actually comes from the right underlying split
    waveform_2, _, _, _ = ds[2]
    assert waveform_2[0, 0].item() == pytest.approx(2.0)  # first split, row 2
    waveform_3, _, _, _ = ds[3]
    assert waveform_3[0, 0].item() == pytest.approx(100.0)  # second split, row 0


def test_concat_split_dataset_rejects_mismatched_label_names():
    first = _fake_split(3, n_labels=2)
    second = _fake_split(2, n_labels=1)
    with pytest.raises(ValueError):
        ConcatSplitDataset(first, second)


def test_fold_demographic_encoder_is_fit_only_on_in_fold_rows_not_whole_pool():
    # A category only present outside train_idx must not appear in the fitted
    # encoder's output columns — proves the encoder doesn't see held-out rows.
    pool = pd.DataFrame({
        "age_at_ecg": [50.0, 60.0, 70.0, 80.0],
        "sex": ["M", "F", "M", "F"],
        "race_ethnicity": ["A", "A", "A", "ONLY_OUTSIDE_TRAIN"],
        "location_setting": ["inpatient", "inpatient", "inpatient", "inpatient"],
    })
    train_idx = np.array([0, 1, 2])  # excludes row 3, which has the rare category
    encoder = fit_fold_demographic_encoder(pool, train_idx)

    feature_names = list(encoder.get_feature_names_out())
    assert not any("ONLY_OUTSIDE_TRAIN" in name for name in feature_names)

    # transforming the held-out row (with the unseen category) doesn't error,
    # thanks to handle_unknown="ignore", and produces the same column count
    transformed = transform_demo(encoder, pool.iloc[[3]])
    assert transformed.shape == (1, len(feature_names))
