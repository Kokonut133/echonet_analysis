"""Builds docs/index.html — a self-contained, interactive "information ladder" results page.

The page shows AUROC/AUPRC per structural-heart-disease target across five
modeling tiers (demographics -> tabular ECG -> waveform features -> combined ->
raw-waveform CNN), using Plotly.js loaded from a CDN with the data embedded
inline as JSON (no server, no fetch — the published docs/index.html works
standalone on GitHub Pages).

Preferred data source: reports/final_results.csv (held-out TEST split, with
bootstrap CIs), produced by a separate results-summary script. If that file
does not exist yet, this script falls back to building the same schema from
the existing VALIDATION-split result CSVs already in reports/:

  reports/demographic_baseline_results.csv  -> tier "demographics"
  reports/ecg_feature_model_results.csv     -> tiers "tabular_ecg" / "waveform_features" / "combined"
                                                (best model per target x feature_set by AUROC)
  reports/cnn_waveforms_results.csv         -> tier "cnn_raw_waveform"

The page clearly labels which case it is in ("held-out test results" vs.
"validation results (provisional)").

Usage:
    .venv/bin/python scripts/10_site/build_site.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

FINAL_RESULTS_PATH = REPORTS_DIR / "final_results.csv"
DEMOGRAPHIC_RESULTS_PATH = REPORTS_DIR / "demographic_baseline_results.csv"
ECG_FEATURE_RESULTS_PATH = REPORTS_DIR / "ecg_feature_model_results.csv"
CNN_RESULTS_PATH = REPORTS_DIR / "cnn_waveforms_results.csv"

FINAL_RESULTS_COLUMNS = [
    "target", "tier", "model", "split",
    "auroc", "auroc_ci_low", "auroc_ci_high",
    "auprc", "auprc_ci_low", "auprc_ci_high",
    "balanced_acc", "prevalence", "n_positive", "n_total",
]

TIER_ORDER = [
    "demographics",
    "tabular_ecg",
    "waveform_features",
    "combined",
    "cnn_raw_waveform",
    "cnn_ecg_and_demographics",
]

TIER_LABELS = {
    "demographics": "Demographics only",
    "tabular_ecg": "Tabular ECG metrics",
    "waveform_features": "Hand-crafted waveform features",
    "combined": "Tabular + waveform features",
    "cnn_raw_waveform": "1D CNN on raw waveform",
    "cnn_ecg_and_demographics": "1D CNN on raw waveform + demographics",
    "cnn_raw_waveform_v2": "1D CNN, normalised + augmented",
}

# muted grey -> blue -> teal -> purple -> accent orange (for the CNN tier)
TIER_COLORS_LIGHT = {
    "demographics": "#8a8d93",
    "tabular_ecg": "#2a78d6",
    "waveform_features": "#1baf7a",
    "combined": "#4a3aa7",
    "cnn_raw_waveform": "#eb6834",
    "cnn_ecg_and_demographics": "#b4308f",
    "cnn_raw_waveform_v2": "#b8501f",
}
TIER_COLORS_DARK = {
    "demographics": "#a3a7ad",
    "tabular_ecg": "#3987e5",
    "waveform_features": "#199e70",
    "combined": "#9085e9",
    "cnn_raw_waveform": "#d95926",
    "cnn_ecg_and_demographics": "#c964ad",
    "cnn_raw_waveform_v2": "#c2703f",
}

# Preserves clinical-importance ordering used throughout the project (see progress_log.md).
TARGET_ORDER = [
    "shd_moderate_or_greater_flag",
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pasp_gte_45_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "tr_max_gte_32_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
]

TARGET_LABELS = {
    "shd_moderate_or_greater_flag": "Structural heart disease (broad)",
    "lvef_lte_45_flag": "LVEF ≤ 45%",
    "lvwt_gte_13_flag": "LV wall thickness ≥ 13mm",
    "aortic_stenosis_moderate_or_greater_flag": "Aortic stenosis (moderate+)",
    "aortic_regurgitation_moderate_or_greater_flag": "Aortic regurgitation (moderate+)",
    "mitral_regurgitation_moderate_or_greater_flag": "Mitral regurgitation (moderate+)",
    "tricuspid_regurgitation_moderate_or_greater_flag": "Tricuspid regurgitation (moderate+)",
    "pulmonary_regurgitation_moderate_or_greater_flag": "Pulmonary regurgitation (moderate+)",
    "rv_systolic_dysfunction_moderate_or_greater_flag": "RV systolic dysfunction (moderate+)",
    "pericardial_effusion_moderate_large_flag": "Pericardial effusion (moderate/large)",
    "pasp_gte_45_flag": "PASP ≥ 45 mmHg",
    "tr_max_gte_32_flag": "TR max velocity ≥ 3.2 m/s",
}


@dataclass(frozen=True)
class SiteData:
    df: pd.DataFrame
    is_final: bool  # True: held-out test results from reports/final_results.csv. False: fallback.


def _empty_ci_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("auroc_ci_low", "auroc_ci_high", "auprc_ci_low", "auprc_ci_high"):
        df[col] = pd.NA
    return df


def _best_per_group(df: pd.DataFrame, group_cols: list[str], metric: str = "auroc") -> pd.DataFrame:
    idx = df.groupby(group_cols)[metric].idxmax()
    return df.loc[idx].reset_index(drop=True)


def build_fallback_results() -> pd.DataFrame:
    """Reassembles the final_results.csv schema from existing validation-split reports."""
    frames = []

    demo = pd.read_csv(DEMOGRAPHIC_RESULTS_PATH)
    demo_best = _best_per_group(demo, ["target"])
    demo_best = demo_best.assign(tier="demographics", split="val")
    frames.append(demo_best[["target", "tier", "model", "split", "auroc", "auprc", "balanced_acc", "prevalence", "n_positive", "n_total"]])

    feat = pd.read_csv(ECG_FEATURE_RESULTS_PATH)
    feature_set_to_tier = {
        "tabular_only": "tabular_ecg",
        "waveform_only": "waveform_features",
        "combined": "combined",
    }
    feat_best = _best_per_group(feat, ["target", "feature_set"])
    feat_best = feat_best.assign(
        tier=feat_best["feature_set"].map(feature_set_to_tier),
        split="val",
    )
    frames.append(feat_best[["target", "tier", "model", "split", "auroc", "auprc", "balanced_acc", "prevalence", "n_positive", "n_total"]])

    cnn = pd.read_csv(CNN_RESULTS_PATH)
    cnn = cnn.rename(columns={"label": "target"})
    cnn = cnn.assign(tier="cnn_raw_waveform", model="ECGConvNet", split="val")
    frames.append(cnn[["target", "tier", "model", "split", "auroc", "auprc", "balanced_acc", "prevalence", "n_positive", "n_total"]])

    combined = pd.concat(frames, ignore_index=True)
    combined = _empty_ci_cols(combined)
    return combined[FINAL_RESULTS_COLUMNS]


def load_site_data() -> SiteData:
    if FINAL_RESULTS_PATH.exists():
        df = pd.read_csv(FINAL_RESULTS_PATH)
        missing = set(FINAL_RESULTS_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{FINAL_RESULTS_PATH} is missing expected columns: {sorted(missing)}")
        return SiteData(df=df[FINAL_RESULTS_COLUMNS], is_final=True)

    df = build_fallback_results()
    return SiteData(df=df, is_final=False)


def _tier_sort_key(tier: str) -> int:
    return TIER_ORDER.index(tier) if tier in TIER_ORDER else len(TIER_ORDER)


def _target_sort_key(target: str) -> int:
    return TARGET_ORDER.index(target) if target in TARGET_ORDER else len(TARGET_ORDER)


def records_from_df(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["_tier_order"] = df["tier"].map(_tier_sort_key)
    df["_target_order"] = df["target"].map(_target_sort_key)
    df = df.sort_values(["_target_order", "_tier_order"]).drop(columns=["_tier_order", "_target_order"])

    records = []
    for row in df.to_dict(orient="records"):
        rec = {}
        for key, value in row.items():
            if pd.isna(value):
                rec[key] = None
            elif isinstance(value, float):
                rec[key] = round(value, 4)
            else:
                rec[key] = value
        rec["target_label"] = TARGET_LABELS.get(rec["target"], rec["target"])
        rec["tier_label"] = TIER_LABELS.get(rec["tier"], rec["tier"])
        records.append(rec)
    return records


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EchoNext ECG Results</title>
<meta name="description" content="Interactive AUROC/AUPRC comparison across ECG-based structural heart disease models, from demographics-only baselines to a raw-waveform 1D CNN.">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {
    color-scheme: light;
    --surface-0: #ffffff;
    --surface-1: #f7f7f6;
    --surface-2: #eeeeec;
    --border: #dcdcd9;
    --text-primary: #14140f;
    --text-secondary: #52514e;
    --text-muted: #7a7973;
    --accent: #2a78d6;
    --link: #1c5cab;
    __TIER_VARS_LIGHT__
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface-0: #131312;
      --surface-1: #1a1a19;
      --surface-2: #232322;
      --border: #37372f;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #928f83;
      --accent: #3987e5;
      --link: #86b6ef;
      __TIER_VARS_DARK__
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-0: #131312;
    --surface-1: #1a1a19;
    --surface-2: #232322;
    --border: #37372f;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #928f83;
    --accent: #3987e5;
    --link: #86b6ef;
    __TIER_VARS_DARK__
  }

  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    padding: 0;
    background: var(--surface-1);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  body {
    max-width: 1100px;
    margin: 0 auto;
    padding: 24px 16px 64px;
  }
  header { margin-bottom: 20px; }
  h1 {
    font-size: 1.6rem;
    margin: 0 0 6px;
    letter-spacing: -0.01em;
  }
  .subtitle {
    color: var(--text-secondary);
    font-size: 0.95rem;
    margin: 0 0 14px;
  }
  .badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 3px 9px;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    background: var(--surface-2);
    margin-bottom: 12px;
  }
  .badge.final { color: #0ca30c; border-color: #0ca30c66; }
  .badge.provisional { color: #c98500; border-color: #c9850066; }

  .intro {
    background: var(--surface-0);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    font-size: 0.95rem;
    line-height: 1.55;
    color: var(--text-secondary);
    margin-bottom: 22px;
  }
  .intro strong { color: var(--text-primary); }

  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: center;
    margin-bottom: 10px;
  }
  .control-group {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--surface-0);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 10px;
  }
  .control-group span.control-label {
    font-size: 0.78rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }
  button.toggle-btn {
    font: inherit;
    font-size: 0.85rem;
    padding: 5px 12px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--surface-1);
    color: var(--text-secondary);
    cursor: pointer;
  }
  button.toggle-btn.active {
    background: var(--accent);
    color: #ffffff;
    border-color: var(--accent);
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 14px;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.83rem;
    padding: 5px 10px 5px 8px;
    border-radius: 7px;
    border: 1px solid var(--border);
    background: var(--surface-0);
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
  }
  .legend-item.off {
    opacity: 0.4;
  }
  .legend-swatch {
    width: 11px;
    height: 11px;
    border-radius: 3px;
    flex: none;
  }

  #chart {
    width: 100%;
    background: var(--surface-0);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 6px;
    margin-bottom: 26px;
  }

  h2 {
    font-size: 1.1rem;
    margin: 0 0 10px;
  }

  .table-wrap {
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface-0);
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
    min-width: 640px;
  }
  thead th {
    text-align: left;
    padding: 9px 10px;
    border-bottom: 1px solid var(--border);
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    cursor: pointer;
    white-space: nowrap;
  }
  thead th:hover { color: var(--text-primary); }
  thead th.sorted::after { content: " \25BE"; }
  thead th.sorted-asc::after { content: " \25B4"; }
  tbody td {
    padding: 7px 10px;
    border-bottom: 1px solid var(--border);
    color: var(--text-secondary);
    white-space: nowrap;
  }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:hover { background: var(--surface-1); }
  td.num { font-variant-numeric: tabular-nums; }
  .tier-dot {
    display: inline-block;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    margin-right: 6px;
  }

  footer {
    margin-top: 32px;
    padding-top: 14px;
    border-top: 1px solid var(--border);
    font-size: 0.82rem;
    color: var(--text-muted);
  }
  footer a { color: var(--link); }

  @media (max-width: 560px) {
    h1 { font-size: 1.35rem; }
    .controls { gap: 10px; }
  }
</style>
</head>
<body>

<header>
  <div class="badge __STATUS_CLASS__">__STATUS_LABEL__</div>
  <h1>EchoNext ECG &rarr; Structural Heart Disease: Model Results</h1>
  <p class="subtitle">Generated __GENERATED_AT__ from <code>__SOURCE_LABEL__</code></p>
</header>

<div class="intro">
  <p>
    This page compares how much predictive signal for echocardiogram-confirmed
    structural heart disease (SHD) can be pulled from a 12-lead ECG, as more
    information is added: starting from <strong>demographics alone</strong>
    (age, sex, race/ethnicity, care setting), then adding
    <strong>standard ECG measurements</strong> (heart rate, PR/QRS/QT intervals),
    then <strong>hand-crafted waveform features</strong>, their
    <strong>combination</strong>, and finally a <strong>1D convolutional neural
    network</strong> trained directly on the raw waveform. Each dot is one
    disease target; AUROC (area under the ROC curve) measures how well a model
    ranks people with the condition above people without it &mdash; 0.5 is a
    coin flip, 1.0 is perfect separation. __PROVISIONAL_NOTE__
  </p>
</div>

<div class="controls">
  <div class="control-group">
    <span class="control-label">Metric</span>
    <button class="toggle-btn active" data-metric="auroc" id="metric-auroc">AUROC</button>
    <button class="toggle-btn" data-metric="auprc" id="metric-auprc">AUPRC</button>
  </div>
</div>

<div class="legend" id="legend"></div>

<div id="chart"></div>

<h2>Results table</h2>
<div class="table-wrap">
  <table id="results-table">
    <thead>
      <tr>
        <th data-key="target_label">Target</th>
        <th data-key="tier_label">Tier</th>
        <th data-key="model">Model</th>
        <th data-key="split">Split</th>
        <th data-key="auroc" class="num sorted">AUROC (&plusmn; ½ CI)</th>
        <th data-key="auprc" class="num">AUPRC</th>
        <th data-key="balanced_acc" class="num">Bal. acc.</th>
        <th data-key="prevalence" class="num">Prevalence</th>
        <th data-key="n_total" class="num">N</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>
</div>

<footer>
  Source code and full analysis:
  <a href="https://github.com/Kokonut133/echonet_analysis" target="_blank" rel="noopener">github.com/Kokonut133/echonet_analysis</a>.
  Research / portfolio project &mdash; not a medical device. See
  <a href="https://github.com/Kokonut133/echonet_analysis/blob/master/MODEL_CARD.md" target="_blank" rel="noopener">MODEL_CARD.md</a>
  for intended use and limitations.
</footer>

<script>
const RESULTS = __DATA_JSON__;
const TIER_ORDER = __TIER_ORDER_JSON__;
const TIER_LABELS = __TIER_LABELS_JSON__;
const TIER_COLORS = __TIER_COLORS_JSON__;
const IS_FINAL = __IS_FINAL_JSON__;

const state = {
  metric: "auroc",
  hiddenTiers: new Set(),
  sortKey: "auroc",
  sortDir: "desc",
};

function isDark() {
  const stamp = document.documentElement.getAttribute("data-theme");
  if (stamp === "dark") return true;
  if (stamp === "light") return false;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function tierColor(tier) {
  const mode = isDark() ? "dark" : "light";
  return TIER_COLORS[tier][mode];
}

function buildLegend() {
  const el = document.getElementById("legend");
  el.innerHTML = "";
  TIER_ORDER.forEach((tier) => {
    const item = document.createElement("div");
    item.className = "legend-item" + (state.hiddenTiers.has(tier) ? " off" : "");
    item.dataset.tier = tier;
    const sw = document.createElement("span");
    sw.className = "legend-swatch";
    sw.style.background = tierColor(tier);
    item.appendChild(sw);
    item.appendChild(document.createTextNode(TIER_LABELS[tier]));
    item.addEventListener("click", () => {
      if (state.hiddenTiers.has(tier)) {
        state.hiddenTiers.delete(tier);
      } else {
        state.hiddenTiers.add(tier);
      }
      render();
    });
    el.appendChild(item);
  });
}

function metricRange(metric) {
  return metric === "auroc" ? [0.5, 1.0] : [0, 1.0];
}

function buildChart() {
  const metric = state.metric;
  const ciLow = metric + "_ci_low";
  const ciHigh = metric + "_ci_high";

  const targets = [];
  RESULTS.forEach((r) => {
    if (!targets.includes(r.target_label)) targets.push(r.target_label);
  });
  // Plotly categorical y-axis renders top-to-bottom in reverse list order.
  const yOrder = targets.slice().reverse();

  const traces = TIER_ORDER.filter((t) => !state.hiddenTiers.has(t)).map((tier) => {
    const rows = RESULTS.filter((r) => r.tier === tier);
    const hasCI = rows.some((r) => r[ciLow] !== null && r[ciLow] !== undefined);
    const errorArray = hasCI ? rows.map((r) => (r[ciHigh] != null && r[metric] != null) ? (r[ciHigh] - r[metric]) : 0) : null;
    const errorArrayMinus = hasCI ? rows.map((r) => (r[ciLow] != null && r[metric] != null) ? (r[metric] - r[ciLow]) : 0) : null;

    return {
      type: "scatter",
      mode: "markers",
      name: TIER_LABELS[tier],
      x: rows.map((r) => r[metric]),
      y: rows.map((r) => r.target_label),
      marker: { color: tierColor(tier), size: 10, line: { color: "rgba(0,0,0,0.15)", width: 1 } },
      error_x: hasCI ? {
        type: "data",
        symmetric: false,
        array: errorArray,
        arrayminus: errorArrayMinus,
        color: tierColor(tier),
        thickness: 1.2,
        width: 3,
      } : undefined,
      customdata: rows.map((r) => [r.model, r[metric === "auroc" ? "auprc" : "auroc"], r.prevalence, r.n_total, r.split]),
      hovertemplate:
        "<b>%{y}</b><br>" +
        TIER_LABELS[tier] + "<br>" +
        "Model: %{customdata[0]}<br>" +
        (metric === "auroc" ? "AUROC: %{x:.3f}<br>AUPRC: %{customdata[1]:.3f}<br>" : "AUPRC: %{x:.3f}<br>AUROC: %{customdata[1]:.3f}<br>") +
        "Prevalence: %{customdata[2]:.1%}<br>" +
        "N: %{customdata[3]}<br>" +
        "Split: %{customdata[4]}" +
        "<extra></extra>",
    };
  });

  const textColor = getComputedStyle(document.body).getPropertyValue("--text-secondary").trim();
  const gridColor = getComputedStyle(document.body).getPropertyValue("--border").trim();
  const bg = getComputedStyle(document.body).getPropertyValue("--surface-0").trim();

  const layout = {
    paper_bgcolor: bg,
    plot_bgcolor: bg,
    font: { color: textColor, family: "inherit", size: 12 },
    margin: { l: 240, r: 24, t: 20, b: 46 },
    xaxis: {
      range: metricRange(metric),
      title: metric.toUpperCase(),
      gridcolor: gridColor,
      zeroline: false,
      color: textColor,
    },
    yaxis: {
      categoryorder: "array",
      categoryarray: yOrder,
      gridcolor: gridColor,
      color: textColor,
      automargin: true,
    },
    legend: { orientation: "h" },
    showlegend: false,
    height: Math.max(420, targets.length * 46 + 80),
  };

  Plotly.react("chart", traces, layout, { responsive: true, displayModeBar: false });
}

function fmtPct(v) {
  return v === null || v === undefined ? "" : (v * 100).toFixed(1) + "%";
}
function fmtNum(v, digits) {
  return v === null || v === undefined ? "" : Number(v).toFixed(digits);
}

// "0.828 \u00b1 0.011" \u2014 margin is half the bootstrap CI width, which is what the
// \u00b1 notation implies; the exact (slightly asymmetric) bounds stay in the CSV.
function fmtPm(v, lo, hi) {
  if (v === null || v === undefined) return "";
  const base = Number(v).toFixed(3);
  if (lo === null || lo === undefined || hi === null || hi === undefined) return base;
  return base + " \u00b1 " + ((Number(hi) - Number(lo)) / 2).toFixed(3);
}

function buildTable() {
  const tbody = document.querySelector("#results-table tbody");
  const rows = RESULTS.filter((r) => !state.hiddenTiers.has(r.tier));
  const dir = state.sortDir === "asc" ? 1 : -1;
  rows.sort((a, b) => {
    let av = a[state.sortKey];
    let bv = b[state.sortKey];
    if (av === null || av === undefined) av = -Infinity;
    if (bv === null || bv === undefined) bv = -Infinity;
    if (typeof av === "string") return dir * av.localeCompare(bv);
    return dir * (av - bv);
  });

  tbody.innerHTML = "";
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      "<td>" + r.target_label + "</td>" +
      "<td><span class='tier-dot' style='background:" + tierColor(r.tier) + "'></span>" + r.tier_label + "</td>" +
      "<td>" + r.model + "</td>" +
      "<td>" + r.split + "</td>" +
      "<td class='num'>" + fmtPm(r.auroc, r.auroc_ci_low, r.auroc_ci_high) + "</td>" +
      "<td class='num'>" + fmtNum(r.auprc, 3) + "</td>" +
      "<td class='num'>" + fmtNum(r.balanced_acc, 3) + "</td>" +
      "<td class='num'>" + fmtPct(r.prevalence) + "</td>" +
      "<td class='num'>" + (r.n_total ?? "") + "</td>";
    tbody.appendChild(tr);
  });

  document.querySelectorAll("#results-table thead th").forEach((th) => {
    th.classList.remove("sorted", "sorted-asc");
    if (th.dataset.key === state.sortKey) {
      th.classList.add(state.sortDir === "desc" ? "sorted" : "sorted-asc");
    }
  });
}

function render() {
  buildLegend();
  buildChart();
  buildTable();
}

document.querySelectorAll(".toggle-btn[data-metric]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".toggle-btn[data-metric]").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.metric = btn.dataset.metric;
    if (state.sortKey === "auroc" || state.sortKey === "auprc") {
      state.sortKey = state.metric;
    }
    render();
  });
});

document.querySelectorAll("#results-table thead th").forEach((th) => {
  th.addEventListener("click", () => {
    const key = th.dataset.key;
    if (state.sortKey === key) {
      state.sortDir = state.sortDir === "desc" ? "asc" : "desc";
    } else {
      state.sortKey = key;
      state.sortDir = "desc";
    }
    render();
  });
});

if (window.matchMedia) {
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
}

render();
</script>
</body>
</html>
"""


def render_html(site: SiteData) -> str:
    records = records_from_df(site.df)

    tier_vars_light = "\n    ".join(f"--tier-{t}: {TIER_COLORS_LIGHT[t]};" for t in TIER_ORDER)
    tier_vars_dark = "\n    ".join(f"--tier-{t}: {TIER_COLORS_DARK[t]};" for t in TIER_ORDER)

    tier_colors_json = {
        t: {"light": TIER_COLORS_LIGHT[t], "dark": TIER_COLORS_DARK[t]} for t in TIER_ORDER
    }

    if site.is_final:
        status_class = "final"
        status_label = "Held-out test results"
        source_label = "reports/final_results.csv"
        provisional_note = (
            "These figures are the official <strong>held-out test split</strong> results, "
            "with bootstrap confidence intervals shown as error bars where available."
        )
    else:
        status_class = "provisional"
        status_label = "Validation results (provisional)"
        source_label = "reports/{demographic_baseline,ecg_feature_model,cnn_waveforms}_results.csv (fallback)"
        provisional_note = (
            "<strong>These are provisional validation-split results</strong>, assembled "
            "automatically because <code>reports/final_results.csv</code> (the held-out "
            "test-split summary with bootstrap confidence intervals) has not been produced "
            "yet. Re-run <code>scripts/10_site/build_site.py</code> once it exists to switch "
            "this page to official test-split numbers."
        )

    html = HTML_TEMPLATE
    html = html.replace("__TIER_VARS_LIGHT__", tier_vars_light)
    html = html.replace("__TIER_VARS_DARK__", tier_vars_dark)
    html = html.replace("__STATUS_CLASS__", status_class)
    html = html.replace("__STATUS_LABEL__", status_label)
    html = html.replace("__SOURCE_LABEL__", source_label)
    html = html.replace("__PROVISIONAL_NOTE__", provisional_note)
    html = html.replace("__GENERATED_AT__", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    html = html.replace("__DATA_JSON__", json.dumps(records))
    html = html.replace("__TIER_ORDER_JSON__", json.dumps(TIER_ORDER))
    html = html.replace("__TIER_LABELS_JSON__", json.dumps(TIER_LABELS))
    html = html.replace("__TIER_COLORS_JSON__", json.dumps(tier_colors_json))
    html = html.replace("__IS_FINAL_JSON__", json.dumps(site.is_final))
    return html


def write_docs_readme() -> None:
    readme = (
        "# docs/\n\n"
        "This folder holds the published GitHub Pages site: a single self-contained "
        "`index.html` with an interactive AUROC/AUPRC comparison across modeling tiers "
        "for the EchoNext structural-heart-disease models.\n\n"
        "Regenerate it with `.venv/bin/python scripts/10_site/build_site.py` from the "
        "project root (it reads from `reports/` and rewrites `docs/index.html` in place).\n\n"
        "To enable Pages: repo **Settings -> Pages -> Deploy from branch -> `master` -> `/docs`**.\n"
    )
    (DOCS_DIR / "README.md").write_text(readme)


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    site = load_site_data()
    html = render_html(site)
    (DOCS_DIR / "index.html").write_text(html)
    write_docs_readme()

    print(f"Wrote {DOCS_DIR / 'index.html'} ({len(html):,} bytes)")
    print(f"Data source: {'reports/final_results.csv (held-out test)' if site.is_final else 'fallback (validation split)'}")
    print(f"Rows embedded: {len(site.df)}")


if __name__ == "__main__":
    main()
