"""Unit tests for the demographic-ablation helpers (src/ablation.py)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.ablation import (
    DEMOGRAPHIC_GROUPS,
    demographic_group_indices,
    zero_demographic_columns,
)

ENCODED_NAMES = [
    "numeric__age_at_ecg",
    "categorical__sex_female",
    "categorical__sex_male",
    "categorical__race_ethnicity_asian",
    "categorical__race_ethnicity_black",
    "categorical__race_ethnicity_white",
    "categorical__location_setting_emergency",
    "categorical__location_setting_inpatient",
]


def test_group_indices_partition_every_column_exactly_once():
    groups = demographic_group_indices(ENCODED_NAMES)
    assert set(groups) == set(DEMOGRAPHIC_GROUPS)
    flat = sorted(i for idx in groups.values() for i in idx)
    assert flat == list(range(len(ENCODED_NAMES)))


def test_group_indices_assign_expected_columns():
    groups = demographic_group_indices(ENCODED_NAMES)
    assert groups["age_at_ecg"] == [0]
    assert groups["sex"] == [1, 2]
    assert groups["race_ethnicity"] == [3, 4, 5]
    assert groups["location_setting"] == [6, 7]


def test_group_indices_rejects_a_missing_group():
    with pytest.raises(ValueError, match="race_ethnicity"):
        demographic_group_indices(["numeric__age_at_ecg", "categorical__sex_male"])


def test_zero_demographic_columns_zeroes_only_the_requested_columns():
    demo = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = zero_demographic_columns(demo, [1, 3])
    assert torch.equal(out[:, 1], torch.zeros(3))
    assert torch.equal(out[:, 3], torch.zeros(3))
    assert torch.equal(out[:, 0], demo[:, 0])
    assert torch.equal(out[:, 2], demo[:, 2])


def test_zero_demographic_columns_does_not_mutate_its_input():
    demo = torch.ones(2, 3)
    zero_demographic_columns(demo, [0, 1, 2])
    assert torch.equal(demo, torch.ones(2, 3))


def test_zero_demographic_columns_with_no_indices_is_identity():
    demo = torch.rand(4, 5)
    assert torch.equal(zero_demographic_columns(demo, []), demo)


def test_zeroing_all_columns_makes_the_head_input_demographics_free():
    """Two records differing only in demographics must score identically once the
    whole demographic block is zeroed."""
    from src.models import ECGConvNet

    torch.manual_seed(0)
    model = ECGConvNet(n_leads=12, n_labels=12, n_demo_features=4).eval()
    waveforms = torch.rand(2, 12, 2500)
    demo_a = torch.tensor([[1.0, 0.0, 2.0, -1.0], [0.0, 1.0, -3.0, 4.0]])
    all_cols = list(range(4))
    with torch.no_grad():
        out_a = model(waveforms, zero_demographic_columns(demo_a, all_cols))
        out_b = model(waveforms, zero_demographic_columns(torch.zeros_like(demo_a), all_cols))
    assert np.allclose(out_a.numpy(), out_b.numpy(), atol=1e-6)
