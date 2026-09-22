"""Shared helpers for lead-ablation and feature-importance analyses (Task 6).

Two independent uses:
  - `zero_leads` / `keep_indices_to_zero`: CNN lead-ablation sensitivity analysis
    (scripts/9_ablation/lead_ablation.py). The model is NOT retrained on missing
    leads; zeroing a lead is a sensitivity probe, not a "trained on fewer leads"
    result.
  - `split_feature_name` / `assign_feature_groups` / `aggregate_by_group`:
    grouping the 187 combined (tabular + waveform) feature importances by lead
    and by feature type (scripts/9_ablation/feature_importance.py).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

CHEST_LEADS: tuple[str, ...] = ("V1", "V2", "V3", "V4", "V5", "V6")
LIMB_LEADS: tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF")

# Shared colour scheme across the lead-ablation and feature-importance figures.
CHEST_COLOR: str = "#DD8452"
LIMB_COLOR: str = "#4C72B0"
TABULAR_COLOR: str = "#7F7F7F"


def group_color(group_lead: str) -> str:
    """Colour for a group_lead value ('tabular' or a lead name), consistent
    across figures/ablation/lead_ablation.png and figures/ablation/feature_importance.png."""
    if group_lead == "tabular":
        return TABULAR_COLOR
    return CHEST_COLOR if lead_category(group_lead) == "chest" else LIMB_COLOR


def lead_category(lead: str) -> str:
    """Classify a lead name as 'chest', 'limb', or 'tabular' (metadata group)."""
    if lead in CHEST_LEADS:
        return "chest"
    if lead in LIMB_LEADS:
        return "limb"
    return "tabular"


def zero_leads(waveforms: torch.Tensor, lead_indices: Sequence[int]) -> torch.Tensor:
    """Return a copy of a (B, n_leads, T) waveform batch with the given lead
    channels (dim=1) replaced by zeros. Does not modify `waveforms` in place."""
    out = waveforms.clone()
    if len(lead_indices) > 0:
        idx = torch.as_tensor(list(lead_indices), dtype=torch.long, device=waveforms.device)
        out.index_fill_(1, idx, 0.0)
    return out


def keep_indices_to_zero(n_leads: int, keep: Sequence[int]) -> list[int]:
    """Indices to zero out so that only the leads in `keep` (by index) survive."""
    keep_set = set(keep)
    return [i for i in range(n_leads) if i not in keep_set]


def split_feature_name(
    name: str, lead_names: Sequence[str], tabular_names: Sequence[str]
) -> tuple[str, str]:
    """Split a feature name into (group_lead, group_type).

    Tabular metadata features map to ("tabular", <name>). Waveform features are
    named "{lead}_{feature_type}" (e.g. "V1_mean") and map to (<lead>, <feature_type>).
    """
    if name in tabular_names:
        return "tabular", name
    lead, sep, feat_type = name.partition("_")
    if not sep or lead not in lead_names:
        raise ValueError(
            f"Cannot parse feature name {name!r} into lead/type "
            f"(known leads={list(lead_names)}, tabular={list(tabular_names)})"
        )
    return lead, feat_type


def assign_feature_groups(
    feature_names: Sequence[str], lead_names: Sequence[str], tabular_names: Sequence[str]
) -> pd.DataFrame:
    """Tag each feature name with its (group_lead, group_type)."""
    rows = []
    for name in feature_names:
        group_lead, group_type = split_feature_name(name, lead_names, tabular_names)
        rows.append({"feature": name, "group_lead": group_lead, "group_type": group_type})
    return pd.DataFrame(rows, columns=["feature", "group_lead", "group_type"])


def aggregate_by_group(values: np.ndarray | Sequence[float], groups: Sequence[str]) -> pd.Series:
    """Sum `values` within each distinct label in `groups` (order of first appearance)."""
    values = np.asarray(values, dtype=float)
    groups = list(groups)
    if len(values) != len(groups):
        raise ValueError("values and groups must have the same length")
    order = list(dict.fromkeys(groups))
    series = pd.Series(values, index=groups).groupby(level=0).sum()
    return series.reindex(order)
