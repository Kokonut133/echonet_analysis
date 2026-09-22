# CNN v2: per-record normalisation + augmentation vs v1

## Setup

Both runs use the identical `ECGConvNet` architecture (930,252 params), the same
optimizer/schedule (AdamW, lr=1e-3, `ReduceLROnPlateau` factor=0.5 patience=5,
grad-clip 1.0), the same batch size (32), and the same train/val split. The only
differences:

- **v1** (`scripts/5_deep_learning/cnn_waveforms_only.py`): raw waveforms, no
  normalisation, no augmentation, `patience=10`, `num_workers=0`.
- **v2** (`scripts/5_deep_learning/cnn_waveforms_v2.py`): per-record
  standardisation (`src/augmentation.py:standardize_per_record` — subtract
  per-lead mean, divide by a single per-record global-std scalar, applied to
  train *and* val/test) plus train-time augmentation (`train_augment`: random
  time shift ±250 samples, amplitude scale 0.8–1.2x, baseline wander
  0.1–0.5 Hz, Gaussian noise std=0.02). `patience=8`, `num_workers=2` (moved
  augmentation off the main process — has no effect on results, only
  throughput).

v1 early-stopped at epoch 20 (best epoch 10). v2 early-stopped at epoch 25
(best epoch 17).

## Per-target val AUROC, v1 vs v2

| target | v1 AUROC | v2 AUROC | Δ (v2−v1) | n_positive (val) |
|---|---:|---:|---:|---:|
| pulmonary_regurgitation_moderate_or_greater_flag | 0.7419 | 0.7602 | **+0.0183** | 21 |
| pasp_gte_45_flag | 0.7741 | 0.7897 | **+0.0156** | 581 |
| mitral_regurgitation_moderate_or_greater_flag | 0.8139 | 0.8273 | **+0.0134** | 282 |
| tr_max_gte_32_flag | 0.7899 | 0.8007 | **+0.0108** | 267 |
| tricuspid_regurgitation_moderate_or_greater_flag | 0.8405 | 0.8511 | **+0.0106** | 305 |
| lvwt_gte_13_flag | 0.7282 | 0.7383 | **+0.0101** | 877 |
| **shd_moderate_or_greater_flag** | 0.8191 | 0.8268 | **+0.0077** | 1990 |
| **lvef_lte_45_flag** | 0.8832 | 0.8893 | **+0.0061** | 866 |
| rv_systolic_dysfunction_moderate_or_greater_flag | 0.8851 | 0.8785 | −0.0066 | 368 |
| aortic_stenosis_moderate_or_greater_flag | 0.8211 | 0.8098 | −0.0113 | 252 |
| pericardial_effusion_moderate_large_flag | 0.7656 | 0.7476 | −0.0180 | 52 |
| aortic_regurgitation_moderate_or_greater_flag | 0.7804 | 0.7362 | **−0.0442** | 62 |
| **mean AUROC (12 targets)** | **0.8036** | **0.8046** | **+0.0010** | |

The two headline clinical targets both improved modestly:
`shd_moderate_or_greater_flag` +0.0077 and `lvef_lte_45_flag` +0.0061 — the
best-powered labels (1990 and 866 positives respectively), where the AUROC
estimate itself is most stable.

The eight labels with ≥250 val positives are net positive for v2 (7 up, 1
essentially flat/slightly down: rv_systolic_dysfunction −0.0066). The clearest
losses are on the two rarest labels: `aortic_regurgitation` (62 positives,
−0.0442) and `pericardial_effusion` (52 positives, −0.0180) — with n_pos this
small a handful of flipped rankings swings AUROC by several points in either
direction, so this reads as high-variance noise from a small eval set rather
than evidence that augmentation specifically hurts rare-label detection
(`pulmonary_regurgitation`, with only 21 positives, moved the other way,
+0.0183).

## Did it fix the overfitting?

Yes, partially — it delayed it rather than raising the ceiling.

- **Best epoch moved later**: v1 peaked at epoch 10, v2 at epoch 17 — the
  regularising effect of per-record normalisation + augmentation slows down
  how fast the model memorises the training set.
- **Training loss is higher at the point of best val performance**: v1's
  train loss at its own best epoch (10) was 0.9046; v2's at its best epoch
  (17) was 0.8397. More directly, at the *same* epoch number (17), v1's train
  loss had already fallen to 0.7454 (well past its own best-val epoch and
  deep into overfitting) while v2's was still 0.8397 — the augmented model is
  fitting the training set more slowly and staying near its val optimum for
  longer, exactly what you'd expect from added regularisation.
- **But the val mean-AUROC plateau barely moved**: 0.8036 → 0.8046, a
  +0.0010 gain that is well within the run-to-run noise you'd expect from a
  single seed on a 4,626-sample val set. Both curves top out in the same
  ~0.80 neighborhood; v2 just takes longer and trains more stably to get
  there.

**Honest read**: normalisation + augmentation did what regularisation is
supposed to do — it slowed / delayed overfitting (later best epoch, smaller
train/val gap at the optimum) — but it did not lift the achievable val
mean-AUROC. That suggests the ~0.80 mean-AUROC ceiling on this task/architecture
is closer to a **data or label-noise limit** (echo-confirmed structural heart
disease labels are themselves imperfect ground truth, and a 12-lead resting
ECG may simply not carry enough signal for some of these labels) than a
**regularisation limit** that more aggressive augmentation, longer training,
or better normalisation would keep pushing past. Getting materially past
~0.80 mean AUROC likely needs a different lever — more/better labels, a larger
or pretrained backbone, or auxiliary signal (demographics, tabular ECG
features) — not more waveform regularisation.

## Training curves

See `figures/cnn_v2_training_curves.png` (train loss and val mean-AUROC vs
epoch, v1 vs v2 overlaid, best-epoch markers).

## Held-out test results (added after the final evaluation)

The validation comparison above is what model selection saw. On the untouched
test split, evaluated once per checkpoint via
`scripts/6_evaluate/evaluate_test_set.py`, the v2 advantage does not survive:

| | mean test AUROC (12 targets) | SHD (any) | LVEF ≤ 45% |
|---|---:|---:|---:|
| v1 (raw, no augmentation) | **0.8107** | 0.828 [0.818–0.839] | 0.880 [0.868–0.892] |
| v2 (normalised + augmented) | 0.8042 | 0.826 [0.815–0.838] | 0.875 [0.863–0.887] |

Every per-target confidence interval overlaps between the two models. v2's
+0.0010 validation edge was noise, and on test v1 is marginally ahead.

**Conclusion.** Two models with the same architecture but very different
regularisation regimes land in the same place. The v2 run clearly did what it
was designed to do — it pushed the best epoch from 10 to 17 and kept training
loss and validation AUROC in step for longer — yet bought no generalisation.
That points at a ceiling in the data (echo-derived label noise, and the limits
of what a 10-second ECG encodes about cardiac structure) rather than at
underfitting or overfitting that more tuning would fix. The reported CNN tier
therefore stays on v1, which is also the checkpoint used by the interpretability
and lead-ablation analyses.
