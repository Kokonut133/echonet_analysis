## 2026-06-24
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
