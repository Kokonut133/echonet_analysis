# Model Card: EchoNext ECG Structural Heart Disease Models

This card covers the models trained in this repository on the EchoNext PhysioNet
dataset to predict echocardiogram-confirmed structural heart disease (SHD) from
12-lead ECGs. It follows the standard model-card structure (Mitchell et al., 2019).

## Model Details

Three model families are trained, in increasing order of information available to
the model:

| Tier | Model | Inputs |
|---|---|---|
| `demographics` | Logistic Regression / Random Forest / Gradient Boosting | age, sex, race/ethnicity, care setting |
| `tabular_ecg` / `waveform_features` / `combined` | Logistic Regression / Random Forest / Gradient Boosting | 7 tabular ECG measurements (ventricular/atrial rate, PR, QRS, QTc, age) and/or 180 hand-crafted waveform features (12 leads × 15 time/frequency-domain summary statistics per lead) |
| `cnn_raw_waveform` | **ECGConvNet** | raw 12-lead, 2,500-sample (250 Hz, 10 s) waveform, optionally concatenated with demographics before the classification head |

**ECGConvNet architecture** (`src/models/cnn.py`): a 1D residual CNN —
`stem (Conv1d k=7 s=2) → ResBlock(32→64,s=2) → ResBlock(64→128,s=2) →
ResBlock(128→256,s=2) → ResBlock(256→256,s=2) → AdaptiveAvgPool1d(1) →
[concat demographics if used] → Linear(256→128) → ReLU → Dropout(0.3) →
Linear(128→12 logits)`. Each `ResBlock` is a standard two-conv residual unit with a
projection shortcut when shape changes. Measured parameter count (waveform-only
variant, `n_demo_features=0`): **≈0.93M parameters** — smaller than the ~3.7M
figure that circulated during planning; the number here was counted directly from
`sum(p.numel() for p in model.parameters())` on the instantiated model and should
be treated as the accurate figure.

All 12 targets are predicted jointly as a multi-label problem: one shared trunk,
12 output logits, trained with masked `BCEWithLogitsLoss` (per-element mask so
rows with a missing/NaN label for a given target do not contribute to that
target's loss) and per-label `pos_weight` (inverse class frequency, computed by
`src/training.py::compute_pos_weights`) to counteract label imbalance.

**Training setup**: AdamW optimizer, `ReduceLROnPlateau` learning-rate scheduler
(halves LR on validation-AUROC plateau), early stopping on mean validation AUROC
across the 12 targets (patience configured in `data/project_config.json`,
default 10 epochs). Classical baselines use scikit-learn Logistic Regression,
Random Forest, and Gradient Boosting, one model per (target × feature-set)
combination, configured in `data/project_config.json`.

- **Developed by**: Christian Stur, as a personal ML portfolio project.
- **Model date**: trained 2026 (see `progress_log.md` for the dated build-out).
- **Model type**: multi-label binary classifiers (tabular sklearn estimators
  and a 1D residual CNN in PyTorch).
- **License / repository**: see repository root
  (https://github.com/Kokonut133/echonet_analysis).

## Intended Use

**Intended use**: research and educational demonstration of an end-to-end ECG →
structural-heart-disease ML pipeline (data exploration, feature engineering,
classical baselines, deep learning, evaluation). Intended audience is technical/ML
reviewers evaluating the author's modeling work.

**Out of scope**: this is **not a medical device** and must not be used for
clinical diagnosis, triage, or any patient-facing decision. It has not been
validated, cleared, or approved for clinical use, has not been tested on any
population outside the single EchoNext cohort described below, and has no
mechanism for clinical-grade quality assurance, monitoring, or human-factors
review.

## Factors

- **Demographics available in the data**: age (18–90, mean ≈61), sex (male/female
  as recorded), race/ethnicity (4 groups), and care setting (4 categories:
  inpatient/outpatient/emergency/other). These are the only demographic factors
  currently evaluated; subgroup-stratified performance (by sex, age band,
  race/ethnicity, or care setting) has **not yet been computed** — see Caveats.
- **Targets**: 12 binary structural-heart-disease flags derived from
  echocardiography (broad SHD, LVEF ≤45%, LV wall thickness ≥13mm, aortic
  stenosis, aortic regurgitation, mitral regurgitation, tricuspid regurgitation,
  pulmonary regurgitation, RV systolic dysfunction, pericardial effusion,
  PASP ≥45mmHg, TR velocity ≥3.2 m/s). Prevalence ranges from 0.8% (pulmonary
  regurgitation) to 52% (broad SHD) — see `progress_log.md` for the full table.
- **Signal factors**: 12-lead, 250 Hz, 10-second recordings. Signal amplitude
  scale differs ~10x between the unsplit portion of the dataset and the
  train/val/test splits (noted during exploration) — waveforms are normalized
  before model input to address this.

## Metrics

Primary metrics: **AUROC** (ranking ability, 0.5 = chance) and **AUPRC**
(precision-recall, more informative under class imbalance), plus balanced
accuracy at the default 0.5 probability threshold. All numbers below are
**validation-split results** (n=4,626), taken from the CSVs currently in
`reports/` (`demographic_baseline_results.csv`, `ecg_feature_model_results.csv`,
`cnn_waveforms_results.csv`). They are **not held-out test results** and carry no
confidence intervals.

> Held-out **test**-split results with bootstrap confidence intervals are being
> produced separately and will live in `reports/final_results_summary.md`
> (not yet present at the time this card was written — see that file directly
> for the authoritative numbers once available).

### CNN raw-waveform model (`cnn_raw_waveform`, validation split, n=4,626)

| target | AUROC | AUPRC | bal. acc | prevalence |
|---|---:|---:|---:|---:|
| shd_moderate_or_greater_flag | 0.819 | 0.791 | 0.738 | 0.430 |
| lvef_lte_45_flag | 0.883 | 0.704 | 0.802 | 0.187 |
| lvwt_gte_13_flag | 0.728 | 0.375 | 0.664 | 0.190 |
| aortic_stenosis_moderate_or_greater_flag | 0.821 | 0.246 | 0.756 | 0.055 |
| aortic_regurgitation_moderate_or_greater_flag | 0.780 | 0.052 | 0.720 | 0.013 |
| mitral_regurgitation_moderate_or_greater_flag | 0.814 | 0.237 | 0.744 | 0.061 |
| tricuspid_regurgitation_moderate_or_greater_flag | 0.841 | 0.280 | 0.764 | 0.066 |
| pulmonary_regurgitation_moderate_or_greater_flag | 0.742 | 0.018 | 0.684 | 0.005 |
| rv_systolic_dysfunction_moderate_or_greater_flag | 0.885 | 0.484 | 0.804 | 0.080 |
| pericardial_effusion_moderate_large_flag | 0.766 | 0.031 | 0.689 | 0.011 |
| pasp_gte_45_flag | 0.774 | 0.348 | 0.703 | 0.126 |
| tr_max_gte_32_flag | 0.790 | 0.178 | 0.728 | 0.058 |

Best-per-target classical model (across `demographics` / `tabular_only` /
`waveform_only` / `combined` feature sets) generally trails the CNN — e.g. broad
SHD: 0.809 AUROC (combined, GradientBoosting) vs. 0.819 (CNN); LVEF≤45%: 0.861 vs.
0.883. Full per-target/per-feature-set/per-model tables are in
`reports/ecg_feature_model_results.csv` and `reports/demographic_baseline_results.csv`.

**Threshold sensitivity note**: because training uses `pos_weight` to correct for
class imbalance, the model's raw output logits/probabilities are recalibrated
away from the natural prevalence — a 0.5 probability threshold no longer
corresponds to the decision boundary an un-weighted model would produce. Balanced
accuracy figures above are therefore threshold-*and*-weighting sensitive; AUROC/
AUPRC (threshold-independent) are the more reliable comparison metric.

## Training Data

- **Dataset**: EchoNext v1.1.0 (PhysioNet) — 100,000 de-identified 12-lead ECGs
  paired with structural-heart-disease labels derived from paired
  echocardiograms, from a single institution/health system.
- **Official splits used**: train 72,475 / validation 4,626 / test 5,442
  (17,457 additional records are in an unsplit portion of the dataset and are
  not used for train/val/test in this project).
- **Preprocessing**: waveforms loaded at 250 Hz / 2,500 samples per lead;
  180 hand-crafted waveform features (12 leads × 9 time-domain + 6 spectral-domain
  statistics) cached separately for classical models; 7 tabular ECG measurements
  used as-is (with the ~10% missingness in PR interval noted in the data).

## Evaluation Data

Same EchoNext v1.1.0 cohort, official validation split (n=4,626) for the numbers
in this card; the official test split (n=5,442) is reserved for the separate,
held-out evaluation referenced above (`reports/final_results_summary.md`).
No external, out-of-institution, or prospectively collected evaluation data has
been used.

## Ethical Considerations

- **Label provenance / noise**: SHD labels are derived from echocardiogram
  reports using threshold-based flags (e.g. LVEF ≤45%, PASP ≥45mmHg). These
  thresholds are clinically motivated but are still a proxy for the underlying
  pathology, and echo reads themselves carry inter-reader variability — label
  noise should be assumed and is not separately quantified here.
- **Single-institution data**: all records come from one dataset/institution.
  Performance has not been checked against any other population, ECG device
  vendor, or health system, so generalization is unknown.
- **No external validation**: results in this repository are internal
  train/validation/test splits of the same source dataset, not independent
  external validation.
- **Subgroup performance not established**: no stratified AUROC/AUPRC by sex,
  age band, race/ethnicity, or care setting has been computed yet, despite these
  fields being available. Given known disparities in echo access and ECG
  interpretation across demographic groups in the literature, this is a material
  gap before any use beyond a portfolio demonstration.
- **Imbalanced/rare targets**: several targets (e.g. pulmonary regurgitation,
  aortic regurgitation, pericardial effusion) have <2% prevalence; AUPRC is low
  even when AUROC looks reasonable, meaning most true positives would still be
  missed at practical operating points.

## Caveats and Recommendations

- Do not use this model, or any output derived from it, to inform real patient
  care decisions.
- Treat validation-split numbers in this card as provisional; prefer the
  held-out test-split numbers with bootstrap confidence intervals in
  `reports/final_results_summary.md` once available.
- Before any use beyond portfolio/research demonstration, at minimum: compute
  subgroup-stratified metrics, calibrate/report metrics at clinically relevant
  operating points (not just the default 0.5 threshold, given `pos_weight`
  recalibration), and validate on external data.
- Rare-label performance (AUPRC) should be weighted more heavily than AUROC when
  judging clinical usefulness of any individual target.
