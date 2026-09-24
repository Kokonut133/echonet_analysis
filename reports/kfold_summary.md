# 5-fold CV variance estimate: fused CNN vs. waveform-only CNN

Every confidence interval elsewhere in this project comes from bootstrap-resampling the (fixed, single-run) test set — none of them capture training/seed variance. This report fills that gap: the fused architecture (`ECGConvNet` with waveform + demographics) is retrained 5 times on 5 stratified folds of the pooled official train+val rows (77,101 records, stratified on `shd_moderate_or_greater_flag`), early-stopped on held-out-fold mean AUROC, and each fold's model is scored on the untouched official test split. The demographic encoder is refit per fold on the in-fold training rows only.

## Per-target: fold-to-fold mean ± std vs. the single-run number

| target | single-run fused AUROC | 5-fold mean ± std (test) | n folds | 5-fold ensemble | single-run waveform-only AUROC |
|---|---:|---:|---:|---:|---:|
| Structural heart disease (any) | 0.8355 | 0.8386 ± 0.0025 | 5 | 0.8473 | 0.8282 |
| LVEF ≤ 45% | 0.8878 | 0.8886 ± 0.0024 | 5 | 0.8995 | 0.8799 |
| RV systolic dysfunction | 0.8828 | 0.8862 ± 0.0050 | 5 | 0.8971 | 0.8778 |
| Aortic stenosis | 0.8791 | 0.8768 ± 0.0030 | 5 | 0.8850 | 0.8288 |
| LV wall thickness ≥ 13mm | 0.7564 | 0.7621 ± 0.0017 | 5 | 0.7714 | 0.7458 |
| Aortic regurgitation | 0.7952 | 0.7543 ± 0.0238 | 5 | 0.7748 | 0.7466 |
| Mitral regurgitation | 0.8236 | 0.8277 ± 0.0025 | 5 | 0.8378 | 0.8071 |
| Tricuspid regurgitation | 0.8574 | 0.8509 ± 0.0037 | 5 | 0.8632 | 0.8470 |
| Pulmonary regurgitation | 0.8363 | 0.7841 ± 0.0395 | 5 | 0.8201 | 0.8227 |
| Pericardial effusion | 0.7786 | 0.7701 ± 0.0227 | 5 | 0.7982 | 0.7875 |
| Pulmonary artery pressure ≥ 45 mmHg | 0.8033 | 0.8023 ± 0.0053 | 5 | 0.8132 | 0.7835 |
| TR max velocity ≥ 3.2 m/s | 0.7937 | 0.7946 ± 0.0061 | 5 | 0.8050 | 0.7734 |

## The deliverable: does the fused-vs-waveform-only gap survive fold noise?

- **Structural heart disease (any)**: single-run gap (fused 0.8355 − waveform-only 0.8282) = +0.0073. Fold-to-fold std of the fused model's test AUROC = 0.0025 (fold mean 0.8386). The gap is **LARGER than** one fold-to-fold standard deviation; the 5-fold ensemble reaches 0.8473.
- **LVEF ≤ 45%**: single-run gap (fused 0.8878 − waveform-only 0.8799) = +0.0079. Fold-to-fold std of the fused model's test AUROC = 0.0024 (fold mean 0.8886). The gap is **LARGER than** one fold-to-fold standard deviation; the 5-fold ensemble reaches 0.8995.
- **RV systolic dysfunction**: single-run gap (fused 0.8828 − waveform-only 0.8778) = +0.0050. Fold-to-fold std of the fused model's test AUROC = 0.0050 (fold mean 0.8862). The gap is **SMALLER than (i.e. within noise of)** one fold-to-fold standard deviation; the 5-fold ensemble reaches 0.8971.
- **Aortic stenosis**: single-run gap (fused 0.8791 − waveform-only 0.8288) = +0.0503. Fold-to-fold std of the fused model's test AUROC = 0.0030 (fold mean 0.8768). The gap is **LARGER than** one fold-to-fold standard deviation; the 5-fold ensemble reaches 0.8850.

## Notes

- `split` in `reports/kfold_results.csv` is `heldout_fold` (in-CV held-out fold) or `test` (official test split, scored per fold model). `fold` is `0`-`4` for individual folds or `ensemble` for the mean of the 5 models' predicted probabilities on the test split.
- The official test split was never used for fold selection, early stopping, or the demographic encoder fit — only for this final per-fold scoring.

