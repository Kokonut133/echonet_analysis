from __future__ import annotations

import numpy as np
import torch

from src.ablation import (
    aggregate_by_group,
    assign_feature_groups,
    keep_indices_to_zero,
    lead_category,
    split_feature_name,
    zero_leads,
)
from src.constants import LEAD_NAMES
from src.models import ECGConvNet

TABULAR_NAMES = [
    "sex", "ventricular_rate", "atrial_rate", "pr_interval",
    "qrs_duration", "qt_corrected", "age_at_ecg",
]
WAVEFORM_FEATURE_TYPES = [
    "mean", "std", "min", "max", "rms", "energy", "skewness", "kurtosis", "zcr",
    "lf_power", "ecg_band_power", "hf_power", "total_power", "dominant_freq", "spectral_entropy",
]


def test_zero_leads_zeroes_only_requested_channels_and_preserves_shape():
    torch.manual_seed(0)
    model = ECGConvNet(n_leads=12, n_labels=12, n_demo_features=0)
    model.eval()

    x = torch.randn(4, 12, 2500)
    drop = [0, 5, 11]
    zeroed = zero_leads(x, drop)

    # shape preserved
    assert zeroed.shape == x.shape

    # exactly the requested channels are zeroed
    assert torch.all(zeroed[:, drop, :] == 0)
    keep = [i for i in range(12) if i not in drop]
    assert torch.equal(zeroed[:, keep, :], x[:, keep, :])

    # original tensor is untouched (not zeroed in place)
    assert not torch.all(x[:, drop, :] == 0)

    # zeroing no leads is a no-op copy
    unchanged = zero_leads(x, [])
    assert torch.equal(unchanged, x)

    # the result still flows through a randomly-initialised model with the right shape
    with torch.no_grad():
        out = model(zeroed)
    assert out.shape == (4, 12)
    assert torch.isfinite(out).all()


def test_keep_indices_to_zero():
    assert keep_indices_to_zero(12, [1, 2]) == [0, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert keep_indices_to_zero(12, list(range(12))) == []
    assert keep_indices_to_zero(12, []) == list(range(12))


def test_lead_category():
    assert lead_category("V1") == "chest"
    assert lead_category("V6") == "chest"
    assert lead_category("II") == "limb"
    assert lead_category("aVR") == "limb"
    assert lead_category("tabular") == "tabular"


def test_split_feature_name_waveform_and_tabular():
    assert split_feature_name("V1_mean", LEAD_NAMES, TABULAR_NAMES) == ("V1", "mean")
    assert split_feature_name("aVR_spectral_entropy", LEAD_NAMES, TABULAR_NAMES) == (
        "aVR", "spectral_entropy",
    )
    assert split_feature_name("age_at_ecg", LEAD_NAMES, TABULAR_NAMES) == ("tabular", "age_at_ecg")


def test_assign_feature_groups_and_aggregation_sum_correctly_on_toy_187_vector():
    feature_names = [
        f"{lead}_{feat}" for lead in LEAD_NAMES for feat in WAVEFORM_FEATURE_TYPES
    ] + TABULAR_NAMES
    assert len(feature_names) == 187

    groups = assign_feature_groups(feature_names, LEAD_NAMES, TABULAR_NAMES)
    assert list(groups.columns) == ["feature", "group_lead", "group_type"]
    assert len(groups) == 187

    values = np.ones(187)

    by_lead = aggregate_by_group(values, groups["group_lead"])
    # each of the 12 leads gets exactly 15 waveform features
    for lead in LEAD_NAMES:
        assert by_lead[lead] == 15
    # all 7 tabular features collapse into a single "tabular" group
    assert by_lead["tabular"] == 7
    assert by_lead.sum() == 187

    by_type = aggregate_by_group(values, groups["group_type"])
    # each of the 15 waveform feature types appears once per lead (12 leads)
    for feat in WAVEFORM_FEATURE_TYPES:
        assert by_type[feat] == 12
    # tabular features are not summed with anything else
    for name in TABULAR_NAMES:
        assert by_type[name] == 1
    assert by_type.sum() == 187

    # a non-trivial weighting aggregates correctly too
    weighted = np.arange(187, dtype=float)
    by_lead_weighted = aggregate_by_group(weighted, groups["group_lead"])
    expected_first_lead = weighted[:15].sum()
    assert by_lead_weighted[LEAD_NAMES[0]] == expected_first_lead
