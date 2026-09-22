# EchoNext: ECG-Based Structural Heart Disease Detection

Predicting echocardiogram-confirmed structural heart disease from 12-lead ECGs — comparing demographics, ECG metadata, hand-crafted waveform features, and a 1D CNN on the raw signal.

[![CI](https://github.com/Kokonut133/echonet_analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Kokonut133/echonet_analysis/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

**[Interactive results](https://kokonut133.github.io/echonet_analysis/)** · **[Model card](MODEL_CARD.md)** · **[Label definitions](docs/labels.md)**

![Information ladder](figures/results/hero_information_ladder.png)

## TL;DR

This project predicted 12 echocardiogram-confirmed structural heart disease outcomes — from broad "any moderate+ disease" down to specific valve and chamber findings — using 100,000 12-lead ECGs from the EchoNext PhysioNet dataset. Five progressively richer inputs were compared on the same targets and splits: demographics alone, demographics plus standard ECG measurements, 180 hand-crafted waveform features, both combined, and a 1D CNN on the raw waveform. Each added layer of signal improved discrimination on held-out test data, and the CNN came out on top (0.828 AUROC for any moderate-or-greater disease, 0.880 for reduced ejection fraction) — but a transparent 180-feature gradient-boosting model landed within 0.02 of it, and on one target beat it outright. Per-target numbers with confidence intervals, interpretability, lead ablation and subgroup breakdowns are below.

## Key results

Held-out **test** split (5,442 ECGs, never used for training or model selection), AUROC with 95% bootstrap confidence intervals. Full 12-target table: [`reports/final_results_summary.md`](reports/final_results_summary.md).

| Input | SHD (any moderate+) | LVEF ≤ 45% | RV dysfunction |
|---|---|---|---|
| Demographics only (age, sex, race, setting) | 0.696 [0.683–0.709] | 0.684 [0.665–0.702] | 0.663 [0.636–0.687] |
| + standard ECG measurements | 0.763 [0.751–0.776] | 0.794 [0.779–0.810] | 0.796 [0.774–0.817] |
| 180 hand-crafted waveform features | 0.796 [0.785–0.809] | 0.850 [0.838–0.862] | 0.845 [0.826–0.865] |
| Combined tabular + waveform | 0.815 [0.804–0.827] | 0.863 [0.851–0.876] | 0.855 [0.836–0.874] |
| **1D CNN on raw 12-lead waveforms** | **0.828 [0.818–0.839]** | **0.880 [0.868–0.892]** | **0.878 [0.861–0.894]** |

Three things this table is meant to show:

1. **Each rung of the ladder earns its place.** Going from who the patient is to what their ECG looks like is worth ~0.13 AUROC on the broad SHD target; the ordering holds on 10 of the 12 targets.
2. **Deep learning wins, but modestly.** The CNN beats the transparent feature-based model by 0.013–0.023 AUROC, and on several targets their confidence intervals overlap. A 180-feature gradient-boosting model you can inspect gets most of the way there.
3. **One target inverts the ladder.** For aortic stenosis, demographics alone (0.844) beat raw waveform features (0.757) and the combined model (0.870) beats the CNN (0.829) — age carries that label, and the CNN never sees age. Worth knowing before assuming a single architecture should win everywhere.

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
| 5. Deep learning | 1D ResNet on raw 12-lead waveforms, plus a normalized + augmented variant | `scripts/5_deep_learning/cnn_waveforms_only.py`, `cnn_waveforms_v2.py` |
| 6. Evaluation | Held-out test set, bootstrap CIs, all result figures | `scripts/6_evaluate/`, `scripts/7_figures/` |
| 7. Analysis | Saliency, lead ablation, feature importance, interactive site | `scripts/8_interpretability/`, `scripts/9_ablation/`, `scripts/10_site/` |

## Further analyses

### Interpretability
![Saliency examples](figures/interpretability/saliency_examples.png)

SmoothGrad input-gradient saliency and Grad-CAM (on the CNN's final residual stage) highlight which parts of the waveform drive each prediction. Saliency concentrates around the QRS complex and T-wave, consistent with known ECG correlates of structural disease.

### Lead ablation
![Lead ablation](figures/ablation/lead_ablation.png)

Zeroing one lead at a time at inference (without retraining) measures how much the trained CNN leans on each of the 12 leads. This is a sensitivity analysis, not a "trained on fewer leads" result — see `scripts/9_ablation/lead_ablation.py` for the distinction.

### Subgroup fairness
![Subgroup AUROC](figures/results/subgroup_auroc.png)

AUROC broken out by sex, age band, race/ethnicity and care setting. Performance is even across sex and race/ethnicity (differences within overlapping confidence intervals), but drops for the oldest patients on the broad SHD target — 0.77 for ≥80 against 0.84 for under-50s — which is the kind of gap an aggregate number hides. Numbers in [`reports/subgroup_results.csv`](reports/subgroup_results.csv).

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
