## 2026-06-22 — 1. data exploration

Why this step: understand what the data looks like before touching a model — splits, missingness, label prevalences, and signal characteristics.

- 100k ECG records, 39 columns; official splits: 72,475 train / 4,626 val / 5,442 test / 17,457 unsplit
- Signal: 10-second 12-lead ECG at 250 Hz → 2,500 samples/lead; 7 pre-standardised tabular ECG features also provided
- Demographics: age 18–90 (mean 61), male/female, 4 care settings, 4 race/ethnicity groups
- Missing: PR interval 10.4%, atrial rate 0.6% — both ECG measurements; demographics complete
- Observation: waveform amplitude differs ~10× between the `no_split` set (std ≈ 0.11) and val/test (std ≈ 1.0); normalise before training

sorted by clinical importance

| target               | prevalence | n_pos  |
|----------------------|-----------:|-------:|
| SHD broad            |      52.2% | 52,188 |
| LVEF ≤45%            |      23.9% | 23,892 |
| LV wall thick.       |      24.2% | 24,220 |
| Aortic stenosis      |       4.1% |  4,054 |
| Aortic regurg.       |       1.3% |  1,264 |
| Mitral regurg.       |       8.5% |  8,451 |
| Tricuspid regurg.    |      10.7% | 10,651 |
| Pulm. regurg.        |       0.8% |    821 |
| RV dysfunction       |      13.2% | 13,243 |
| Pericardial eff.     |       3.0% |  3,023 |
| PASP ≥45 mmHg        |      19.0% | 18,993 |
| TR vel. ≥3.2 m/s     |      10.2% | 10,212 |

---

## 2026-06-24 — 2. demographic baseline

Core metric = AUROC: how well the model ranks sick above healthy across all thresholds. 0.5 = coin flip, 1.0 = perfect. Above 0.70 is useful for screening.

- RandomForest: best on 11/12 targets, selected going forward
- GradientBoosting: collapses to majority guess at default threshold (bal. acc ≈ 0.50)
- best result: aortic stenosis (0.86) — age dominates this label
- weakest: broad SHD flag (0.70) — needs ECG features to improve
- rare labels (<2% prevalence): ok AUROC but low AUPRC, model misses most positives

sorted by clinical importance (mortality risk and urgency of intervention)

| target               | AUROC | AUPRC | bal. acc | prevalence |
|----------------------|------:|------:|---------:|-----------:|
| SHD broad            | 0.695 | 0.694 |    0.642 |      0.522 |
| LVEF ≤45%            | 0.691 | 0.402 |    0.638 |      0.239 |
| Aortic stenosis      | 0.859 | 0.281 |    0.786 |      0.041 |
| Pericardial effusion | 0.788 | 0.186 |    0.722 |      0.030 |
| RV dysfunction       | 0.722 | 0.310 |    0.662 |      0.132 |
| PASP ≥45 mmHg        | 0.682 | 0.329 |    0.635 |      0.190 |
| LV wall thick.       | 0.679 | 0.393 |    0.629 |      0.242 |
| Mitral regurgitation | 0.716 | 0.220 |    0.661 |      0.085 |
| Aortic regurgitation | 0.763 | 0.069 |    0.720 |      0.013 |
| Tricuspid regurg.    | 0.716 | 0.262 |    0.662 |      0.107 |
| TR velocity ≥3.2 m/s | 0.726 | 0.239 |    0.666 |      0.102 |
| Pulm. regurgitation  | 0.863 | 0.164 |    0.829 |      0.008 |

---

## 2026-06-30 — 3. waveform feature extraction

Why this step: raw waveforms (16 GB train) are too large to iterate over with sklearn. Extract compact per-lead statistics once, cache as .npy, train in seconds.

- 180 features per record: 12 leads × 15 feats (9 time-domain + 6 spectral)
- time: mean, std, min, max, rms, energy, skewness, kurtosis, zcr
- spectral: power in 3 bands (0.5–5 Hz / 5–40 Hz / 40–100 Hz), total power, dominant freq, spectral entropy
- output: train 72,475 × 180 ≈ 50 MB vs 16 GB raw — 320× size reduction
- batched at 2,000 records to stay within RAM; all three splits processed

No model results. Features feed into step 4 (classical ML comparison).

---

## 2026-06-30 — 4. classical ML comparison

Why this step: the demographic-only baseline (step 2) tops out at 0.70–0.69 AUROC on the headline targets. This step adds ECG metadata (rate, intervals) and the 180 hand-crafted waveform features (step 3), and compares LogisticRegression / RandomForest / GradientBoosting across three feature sets — `tabular_only`, `waveform_only`, `combined` — for all 12 targets, on the official train/val split.

- GradientBoosting wins on `combined` for both headline targets and is the best or near-best model on almost every target/feature-set combination
- adding ECG metadata alone (tabular_only vs. demographics-only) lifts SHD from 0.70 → 0.74 and LVEF ≤45% from 0.69 → 0.76
- hand-crafted waveform features add another large jump, especially for LVEF (0.76 → 0.85) — the waveform carries most of the signal for systolic dysfunction
- combining tabular + waveform features gives a further, smaller gain (SHD 0.79 → 0.81, LVEF 0.85 → 0.86) — diminishing returns once the waveform features are in
- full per-target, per-feature-set, per-model results: `reports/ecg_feature_model_results.csv`

best AUROC per feature set (GradientBoosting, headline targets)

| target    | tabular_only | waveform_only | combined |
|-----------|-------------:|---------------:|---------:|
| SHD broad |        0.745 |          0.789 |    0.809 |
| LVEF ≤45% |        0.764 |          0.853 |    0.861 |

---

## 2026-07-09 — 5. 1D CNN on raw waveforms

Why this step: classical models plateau once hand-crafted features are exhausted. A 1D CNN trained directly on the raw 12-lead waveforms can learn morphology the hand-crafted features don't capture — this step checks how much further that buys.

- `ECGConvNet` trained on raw `(12, 2500)` waveforms only (no demographics), masked BCE with per-label pos_weight, Adam + ReduceLROnPlateau, early stopping on mean val AUROC
- ran for 20 epochs (patience not yet hit); val mean-AUROC rose sharply through epoch ~10 (0.762 → 0.804), then plateaued/oscillated around 0.795–0.80 through epoch 20 while LR was halved at epoch 16 after 5 epochs without improvement
- train loss kept falling every epoch through 20 (1.07 → 0.68) while val AUROC stayed flat — classic overfitting past the plateau; the saved checkpoint is the best-val-AUROC epoch, not the final epoch
- beats the best classical `combined` model on 10/12 targets; biggest jumps on RV dysfunction (0.856 → 0.885) and LVEF ≤45% (0.861 → 0.883)
- full per-epoch log: `reports/cnn_waveforms_train_log.csv`; full per-target results: `reports/cnn_waveforms_results.csv`

sorted by clinical importance

| target                | AUROC | AUPRC | bal. acc | prevalence |
|-----------------------|------:|------:|---------:|-----------:|
| SHD broad             | 0.819 | 0.791 |    0.738 |      0.430 |
| LVEF ≤45%             | 0.883 | 0.704 |    0.802 |      0.187 |
| LV wall thick.        | 0.728 | 0.375 |    0.664 |      0.190 |
| Aortic stenosis       | 0.821 | 0.246 |    0.756 |      0.055 |
| Aortic regurgitation  | 0.780 | 0.052 |    0.720 |      0.013 |
| Mitral regurgitation  | 0.814 | 0.237 |    0.744 |      0.061 |
| Tricuspid regurg.     | 0.841 | 0.280 |    0.764 |      0.066 |
| Pulm. regurgitation   | 0.742 | 0.018 |    0.684 |      0.005 |
| RV dysfunction        | 0.885 | 0.484 |    0.804 |      0.080 |
| Pericardial eff.      | 0.766 | 0.031 |    0.689 |      0.011 |
| PASP ≥45 mmHg         | 0.774 | 0.348 |    0.703 |      0.126 |
| TR vel. ≥3.2 m/s      | 0.790 | 0.178 |    0.728 |      0.058 |

Note: `cnn_waveforms_with_demographics.py` (waveforms + demographics) exists and is covered by the smoke tests but has not been trained to completion yet — no results to report for it.
