## 2026-06-30 — 1. data exploration

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

## 2026-06-30 — 2. waveform feature extraction

Why this step: raw waveforms (16 GB train) are too large to iterate over with sklearn. Extract compact per-lead statistics once, cache as .npy, train in seconds.

- 180 features per record: 12 leads × 15 feats (9 time-domain + 6 spectral)
- time: mean, std, min, max, rms, energy, skewness, kurtosis, zcr
- spectral: power in 3 bands (0.5–5 Hz / 5–40 Hz / 40–100 Hz), total power, dominant freq, spectral entropy
- output: train 72,475 × 180 ≈ 50 MB vs 16 GB raw — 320× size reduction
- batched at 2,000 records to stay within RAM; all three splits processed

No model results. Features feed into step 4 (classical ML comparison).

---

## 2026-06-24 — 3. demographic baseline
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
