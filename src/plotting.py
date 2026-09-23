from __future__ import annotations

import matplotlib as mpl

# --- shared vocabulary for results reporting + figures -----------------------

TIER_ORDER: list[str] = [
    "demographics",
    "tabular_ecg",
    "waveform_features",
    "combined",
    "cnn_raw_waveform",
    "cnn_ecg_and_demographics",
]

# Variants that are model iterations rather than rungs of the information
# ladder: they are reported in tables but left out of the ladder figures.
TIER_VARIANTS: dict[str, str] = {
    "cnn_raw_waveform_v2": "CNN v2 (normalised + augmented)",
}

TIER_LABELS: dict[str, str] = {
    "demographics": "Demographics",
    "tabular_ecg": "+ ECG metadata",
    "waveform_features": "Waveform features",
    "combined": "Combined",
    "cnn_raw_waveform": "CNN (raw ECG)",
    "cnn_ecg_and_demographics": "CNN (raw ECG + demographics)",
    **{k: v for k, v in TIER_VARIANTS.items()},
}

TIER_DESCRIPTIONS: dict[str, str] = {
    "demographics": "age, sex, race, care setting",
    "tabular_ecg": "+ heart rate, PR/QRS/QTc intervals",
    "waveform_features": "hand-crafted signal features, no demographics",
    "combined": "ECG metadata + waveform features",
    "cnn_raw_waveform": "deep learning directly on the raw 12-lead signal",
    "cnn_ecg_and_demographics": "raw 12-lead signal fused with age, sex, race and care setting",
    "cnn_raw_waveform_v2": "same architecture, per-record normalisation + augmentation",
}

# Palette keyed by tier, ordered along the information ladder; the two CNN rungs
# carry the accent colours so they read as the top of the ladder.
TIER_COLORS: dict[str, str] = {
    "demographics": "#9CA3AF",
    "tabular_ecg": "#60A5FA",
    "waveform_features": "#2CA6A4",
    "combined": "#F4A340",
    "cnn_raw_waveform": "#E0457B",
    "cnn_ecg_and_demographics": "#7C3AED",
    "cnn_raw_waveform_v2": "#B91C5C",
}

TARGET_SHORT_NAMES: dict[str, str] = {
    "shd_moderate_or_greater_flag": "Structural heart disease (any)",
    "lvef_lte_45_flag": "LVEF ≤ 45%",
    "lvwt_gte_13_flag": "LV wall thickness ≥ 13mm",
    "aortic_stenosis_moderate_or_greater_flag": "Aortic stenosis",
    "aortic_regurgitation_moderate_or_greater_flag": "Aortic regurgitation",
    "mitral_regurgitation_moderate_or_greater_flag": "Mitral regurgitation",
    "tricuspid_regurgitation_moderate_or_greater_flag": "Tricuspid regurgitation",
    "pulmonary_regurgitation_moderate_or_greater_flag": "Pulmonary regurgitation",
    "rv_systolic_dysfunction_moderate_or_greater_flag": "RV systolic dysfunction",
    "pericardial_effusion_moderate_large_flag": "Pericardial effusion",
    "pasp_gte_45_flag": "Pulmonary artery pressure ≥ 45 mmHg",
    "tr_max_gte_32_flag": "TR max velocity ≥ 3.2 m/s",
}


def apply_style() -> None:
    """Shared matplotlib theme for all results figures: clean sans font, no
    top/right spines, dpi 200, readable at README width."""
    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelcolor": "#111827",
        "axes.edgecolor": "#6B7280",
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "text.color": "#111827",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10.5,
        "lines.linewidth": 1.8,
        "patch.linewidth": 0.8,
    })
