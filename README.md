# EchoNext: ECG-Based Structural Heart Disease Detection

Predicting echocardiogram-confirmed structural heart disease from 12-lead ECGs — comparing demographics, ECG metadata, hand-crafted waveform features, and a 1D CNN on the raw signal.

[![CI](https://github.com/Kokonut133/echonet_analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Kokonut133/echonet_analysis/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

**[Interactive results](https://kokonut133.github.io/echonet_analysis/)** · **[Model card](MODEL_CARD.md)** · **[Label definitions](docs/labels.md)**

![Information ladder](figures/results/hero_information_ladder.png)

## TL;DR

This project predicted 12 echocardiogram-confirmed structural heart disease outcomes — from broad "any moderate+ disease" down to specific valve and chamber findings — using 100,000 12-lead ECGs from the EchoNext PhysioNet dataset. Five progressively richer inputs were compared on the same targets and splits: demographics alone, demographics plus standard ECG measurements, 180 hand-crafted waveform features, both combined, and a 1D CNN on the raw waveform. Each added layer of signal improved discrimination on held-out test data, and a CNN reading the raw waveform alongside demographics came out on top (0.836 AUROC for any moderate-or-greater disease, 0.888 for reduced ejection fraction) — but a transparent 180-feature gradient-boosting model landed within 0.02 of the waveform-only CNN, and beat it outright on aortic stenosis. Chasing down *why* that one target inverted is the most interesting part of the project. Per-target numbers with confidence intervals, interpretability, lead and demographic ablations, and subgroup breakdowns are below.

## Key results

Held-out **test** split (5,442 ECGs, never used for training or model selection). AUROC ± half the width of a 95% bootstrap confidence interval. Full 12-target table: [`reports/final_results_summary.md`](reports/final_results_summary.md).

| Input | SHD (any moderate+) | LVEF ≤ 45% | Aortic stenosis |
|---|---|---|---|
| Demographics only (age, sex, race, setting) | 0.696 ± 0.013 | 0.684 ± 0.018 | 0.844 ± 0.021 |
| + standard ECG measurements | 0.763 ± 0.013 | 0.794 ± 0.016 | 0.843 ± 0.020 |
| 180 hand-crafted waveform features | 0.796 ± 0.012 | 0.850 ± 0.012 | 0.757 ± 0.025 |
| Combined tabular + waveform | 0.815 ± 0.012 | 0.863 ± 0.013 | 0.870 ± 0.018 |
| 1D CNN on raw 12-lead waveforms | 0.828 ± 0.011 | 0.880 ± 0.012 | 0.829 ± 0.023 |
| **1D CNN, raw waveforms + demographics** | **0.836 ± 0.011** | **0.888 ± 0.011** | **0.879 ± 0.017** |

Three things this table is meant to show:

1. **Each rung of the ladder earns its place.** Going from who the patient is to what their ECG looks like is worth ~0.14 AUROC on the broad SHD target, and mean AUROC across all 12 targets rises monotonically up the ladder: 0.658 → 0.722 → 0.770 → 0.796 → 0.811 → 0.828.
2. **Deep learning wins, but modestly.** The waveform-only CNN beats the transparent feature-based model by 0.013–0.023 AUROC, and on several targets their confidence intervals overlap. A 180-feature gradient-boosting model you can inspect gets most of the way there.
3. **One target exposed why — and the fix confirmed it.** Aortic stenosis was the one place the ladder inverted: demographics alone (0.844) beat raw waveform features (0.757), and the interpretable model (0.870) beat the CNN (0.829), because patient age carries that diagnosis and a waveform-only CNN never sees age. Fusing demographics into the CNN's head recovers +0.050 on exactly that target and puts it back on top (0.879). The [demographic ablation](reports/ablation_notes.md) closes the loop: removing age alone costs 0.145 AUROC there, against at most 0.02 on any other target.

## What the model sees

![ECG example, positive vs. negative](figures/results/ecg_example_positive_vs_negative.png)

Side-by-side 12-lead traces for a structural-heart-disease-positive and a negative record. The differences that separate these classes are subtle and spread across leads — which is why hand-crafted per-lead statistics get most of the way to CNN-level performance.

## Pipeline

| Stage | What it does | Script |
|---|---|---|
| 1. Explore | Dataset inventory, label prevalence, example ECGs | `scripts/1_explore/generate_overview.py` |
| 2. Feature extraction | 180 per-lead time/frequency statistics, cached as `.npy` | `scripts/2_preprocess/extract_waveform_features.py` |
| 3. Baseline | Demographics-only models | `scripts/3_baselines/demographic_only.py` |
| 4. Classical ML | LogReg / RandomForest / GradientBoosting on tabular, waveform and combined features | `scripts/4_classical_ml/compare_ecg_feature_sets.py` |
| 5. Deep learning | 1D ResNet on raw 12-lead waveforms; variants with demographics fused in, and with normalization + augmentation | `scripts/5_deep_learning/cnn_waveforms_only.py`, `cnn_waveforms_with_demographics.py`, `cnn_waveforms_v2.py` |
| 6. Evaluation | Held-out test set, bootstrap CIs, all result figures | `scripts/6_evaluate/`, `scripts/7_figures/` |
| 7. Analysis | Saliency, lead and demographic ablation, feature importance, interactive site | `scripts/8_interpretability/`, `scripts/9_ablation/`, `scripts/10_site/` |

## Further analyses

### Interpretability
![Saliency examples](figures/interpretability/saliency_examples.png)

SmoothGrad input-gradient saliency and Grad-CAM (on the CNN's final residual stage) highlight which parts of the waveform drive each prediction. Saliency concentrates around the QRS complex and T-wave, consistent with known ECG correlates of structural disease.

### Lead ablation
![Lead ablation](figures/ablation/lead_ablation.png)

Zeroing one lead at a time at inference (without retraining) measures how much the trained CNN leans on each of the 12 leads. This is a sensitivity analysis, not a "trained on fewer leads" result — see `scripts/9_ablation/lead_ablation.py` for the distinction.

### Which demographics does the fused model use?
![Demographic ablation](figures/ablation/demographic_ablation.png)

Zeroing each demographic input of the fused CNN at inference shows that demographics are worth 0.027 mean AUROC and **age is 69% of it** — and that age is not a diffuse effect but almost entirely one diagnosis (−0.145 on aortic stenosis, against at most 0.02 elsewhere), matching the clinical picture of age-related valve calcification. Two inputs that could have been problematic are not load-bearing: zeroing race/ethnicity changes mean AUROC by −0.002, and care setting — the input most at risk of proxying "this patient is already known to be sick" — by −0.003, both with paired bootstrap intervals spanning zero on the headline target. Details in [`reports/ablation_notes.md`](reports/ablation_notes.md).

### Subgroup fairness
![Subgroup AUROC](figures/results/subgroup_auroc.png)

AUROC broken out by sex, age band, race/ethnicity and care setting. Performance is even across sex and race/ethnicity (differences within overlapping confidence intervals), but drops for the oldest patients on the broad SHD target — 0.764 for ≥80 against 0.845 for 50–65 — the kind of gap an aggregate number hides. Notably, **giving the model age as an input does not close that gap**: the fused CNN shows the same ≥80 deficit as the waveform-only model, which makes sense once you notice that within the oldest band nearly everyone is at risk (64% prevalence), so age has little left to discriminate on. Higher average accuracy and fairer accuracy are not the same objective. Numbers in [`reports/subgroup_results.csv`](reports/subgroup_results.csv).

### Operating points
![Operating points](figures/results/operating_points.png)

What a screening threshold actually costs. To catch 90% of true structural-heart-disease cases, the CNN also flags about 51% of people without it; at 80% sensitivity that falls to 32%. Reporting a single accuracy number would hide this trade-off entirely.

## Dataset

[EchoNext](https://physionet.org/) (PhysioNet, v1.1.0): 100,000 de-identified 12-lead ECGs (250 Hz, 10 s), each paired with structural heart disease labels derived from a corresponding echocardiogram. Demographics (age, sex, race/ethnicity, care setting) and 7 standard ECG measurements (heart rate, PR/QRS/QTc intervals) are included alongside the raw waveform. Data is **not included in this repository** — download it directly from PhysioNet and place it under `data/<dataset-folder>/` per `data/project_config.json`.

12 targets, prevalence in the full dataset:

| Target | Prevalence |
|---|---:|
| SHD, any moderate-or-greater | 52.2% |
| LV wall thickness ≥13mm | 24.2% |
| LVEF ≤45% | 23.9% |
| PASP ≥45 mmHg | 19.0% |
| RV systolic dysfunction, moderate+ | 13.2% |
| Tricuspid regurgitation, moderate+ | 10.7% |
| TR velocity ≥3.2 m/s | 10.2% |
| Mitral regurgitation, moderate+ | 8.5% |
| Aortic stenosis, moderate+ | 4.1% |
| Pericardial effusion, moderate/large | 3.0% |
| Aortic regurgitation, moderate+ | 1.3% |
| Pulmonary regurgitation, moderate+ | 0.8% |

Full label definitions: [`docs/labels.md`](docs/labels.md).

## Reproduce

```bash
# install (Python 3.11+)
pip install -e .
pip install -r requirements.txt

# place the downloaded EchoNext files under data/<dataset-folder>/
# per data/project_config.json

python scripts/1_explore/generate_overview.py
python scripts/2_preprocess/extract_waveform_features.py
python scripts/3_baselines/demographic_only.py
python scripts/4_classical_ml/compare_ecg_feature_sets.py
python scripts/5_deep_learning/cnn_waveforms_only.py
python scripts/5_deep_learning/cnn_waveforms_with_demographics.py  # best model
python scripts/5_deep_learning/cnn_waveforms_v2.py      # normalized + augmented variant
python scripts/6_evaluate/evaluate_test_set.py

pytest tests -q   # 54 smoke + unit tests, ~3.5 min, no full dataset required
```

## Design decisions

- **Memory-mapped waveforms** (`np.load(..., mmap_mode="r")`): the raw train waveform array is ~16 GB; nothing in the pipeline loads it fully into RAM.
- **Per-lead hand-crafted features before deep learning**: 180 compact time/frequency-domain statistics (12 leads × 15 features) reduce the 16 GB waveform array to ~50 MB, making classical-ML iteration fast and giving a transparent baseline to compare the CNN against.
- **Masked BCE with per-label `pos_weight`**: labels have missingness (up to ~9-55% on echo sub-measurements) and range from 0.8% to 52% prevalence; masking excludes missing rows from each label's loss term, and `pos_weight` counteracts class imbalance without discarding data.
- **Multi-label, single shared trunk**: all 12 targets are predicted jointly from one model (sklearn: one estimator per target/feature-set; CNN: one shared backbone, 12 output logits) rather than training 12 separate pipelines.
- **Early stopping on mean validation AUROC**: optimizing a single scalar (mean AUROC across the 12 targets) for early stopping and learning-rate reduction, rather than per-target checkpoints.
- **Official dataset splits, untouched test set**: train/val/test splits from the dataset's own `split` column are used throughout; the test split is touched only once, at final evaluation (`scripts/6_evaluate/evaluate_test_set.py`).
- **Bootstrap confidence intervals on the held-out test set**: final reported numbers are accompanied by bootstrap CIs rather than point estimates alone (`reports/final_results_summary.md`).
- **Lead ablation as sensitivity analysis, not retraining**: lead importance is measured by zeroing channels on the already-trained CNN, keeping it cheap and directly comparable across leads.

## Limitations & next steps

- Labels are echo-derived, not adjudicated by a separate clinical panel — echo reading variability is an unmeasured source of label noise.
- Rare targets (pulmonary regurgitation 0.8%, aortic regurgitation 1.3%) have low AUPRC despite reasonable AUROC — the models miss most positives at any practical operating threshold.
- **Performance appears to be dataset-limited, not regularization-limited.** A second CNN with per-record normalization and waveform augmentation (time shift, amplitude scaling, baseline wander, noise) did delay overfitting — the best epoch moved from 10 to 17 — but bought no generalization: mean test AUROC was 0.804 against 0.811 for the unaugmented model, with every per-target confidence interval overlapping. Two models with the same architecture and very different regularization landing in the same place points at echo-derived label noise and the limits of what 10 seconds of ECG encodes, rather than at something more tuning would fix ([`reports/cnn_v2_notes.md`](reports/cnn_v2_notes.md)).
- Demographic subgroup performance (`figures/results/subgroup_auroc.png`) has not been used to recalibrate or reweight training — it is reported, not yet acted on.
- No external validation cohort; all results are internal to the EchoNext splits.
