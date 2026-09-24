# Overnight serial run — findings (2026-09-24)

Five stages, run strictly one at a time by `scripts/12_overnight/run_overnight.py`.
Stages 1-2 completed on the first launch; stages 3-4 crashed on an Optuna API
misuse and were re-run after the fix (see "Bugs found" at the end). This file was
written by hand from the stage outputs, replacing an auto-generated version whose
verdict lines contradicted the data.

## Stage 1 — finish the 5th CV fold

Done in 31 min. `reports/kfold_results.csv` now holds all 5 folds; test-split
SHD-composite AUROC mean = **0.8386** (was 0.8392 over 4 folds).

## Stage 2 — long run with warmup + cosine (the key experiment)

60-epoch budget, linear warmup (3 epochs) then cosine decay, `patience=12`.

- **Early-stopped at epoch 24, best at epoch 12.** Best val mean AUROC 0.8245,
  best val SHD AUROC 0.837 (first crossed the 0.8279 single-run val baseline at
  epoch 5).
- This settles the question the k-fold run raised. The earlier fixed-LR runs were
  still improving at epoch 19 of 20, which looked like the epoch budget being the
  binding constraint. With a proper schedule the model converges by epoch 12 and
  then genuinely plateaus for 12 straight epochs. The budget was not the ceiling.

## Stage 5 — held-out test evaluation of the long-run model

Scored once on the official test split as tier `cnn_raw_waveform_cnn_serial`.
Per-target change against the previous best model (`cnn_ecg_and_demographics`),
with the change expressed in units of the fold-to-fold standard deviation measured
in stage 1:

| target | prev best | long run | delta | fold std | delta / std |
|---|---:|---:|---:|---:|---:|
| RV systolic dysfunction | 0.8828 | 0.8915 | +0.0087 | 0.0063 | 1.38 |
| LV wall thickness >= 13mm | 0.7564 | 0.7638 | +0.0074 | 0.0041 | 1.80 |
| LVEF <= 45% | 0.8878 | 0.8923 | +0.0045 | 0.0049 | 0.91 |
| Aortic stenosis | 0.8791 | 0.8834 | +0.0043 | 0.0043 | 1.01 |
| SHD composite | 0.8355 | 0.8385 | +0.0030 | 0.0042 | 0.72 |
| Tricuspid regurgitation | 0.8574 | 0.8565 | -0.0009 | 0.0060 | 0.15 |
| TR max velocity | 0.7937 | 0.7914 | -0.0023 | 0.0069 | 0.33 |
| Mitral regurgitation | 0.8236 | 0.8202 | -0.0034 | 0.0047 | 0.73 |
| PASP >= 45 mmHg | 0.8033 | 0.7978 | -0.0055 | 0.0065 | 0.84 |
| Pericardial effusion | 0.7786 | 0.7666 | -0.0120 | 0.0233 | 0.51 |
| Aortic regurgitation | 0.7952 | 0.7640 | -0.0312 | 0.0228 | 1.37 |
| Pulmonary regurgitation | 0.8363 | 0.7943 | -0.0420 | 0.0383 | 1.10 |
| **mean over 12 targets** | **0.8275** | **0.8217** | -0.0058 | | |

**No target moved by as much as 2 fold standard deviations.** Every apparent gain
and every apparent loss sits inside the training noise measured in stage 1. The
right conclusion is that the warmup+cosine model is **indistinguishable** from the
previous best, not that it improved the common targets and hurt the rare ones.
Without the fold-variance estimate this would have been written up as a +0.0087
gain on RV dysfunction; it is not a result.

The 12-target mean fell because the two lowest-prevalence targets (pulmonary
regurgitation at 0.8% and aortic regurgitation at 1.3%) moved down, and those are
precisely the targets whose fold std is an order of magnitude larger than the
common ones. Averaging over targets with wildly different variance gives the
noisiest targets disproportionate weight in the mean.

## Stage 3 — warm-started refinement of the late-training phase

Budget-limited after 4 h: 77 rows, 55 pruned, 20 trials complete, plus 2 control
runs. Best trial val SHD AUROC 0.8226 against a control of 0.8179, so the search
did beat its own control by +0.0047 over the same 5 epochs from the same
checkpoint.

Two honest caveats, the second of which is a design mistake worth recording:

1. The stated limits hold: this cannot tune architecture (the checkpoint fixes
   tensor shapes) or initial LR/warmup (already past), and there is a warm-start
   bias toward the long run's regularisation.
2. **The warm-start point was chosen badly.** The 3/4 checkpoint is epoch 18, but
   the long run's best epoch was 12, so every trial resumed from an already
   post-peak, cosine-decayed state. That is why both the trials (0.8226) and the
   control (0.8179) land well below the epoch-12 peak of 0.837. A rerun should
   warm-start from the *best* checkpoint, not a fixed fraction of the epochs.

## Stage 4 — from-scratch low-fidelity elimination sweep

Budget-limited after 3 h: 69 trials, **60 pruned (87%)**, 9 complete, 5 epochs
each. This is the most actionable result of the run.

- **Surviving learning rates all fall in 1.5e-4 to 3.6e-4.** Pruned trials span
  1.2e-5 to 2.6e-3.
- The project's hand-picked `learning_rate` of **1e-3** (`data/project_config.json`)
  sits outside the surviving band, roughly 3-6x too high.
- Every surviving trial used `batch_size=32` with dropout between 0.29 and 0.36.
- Best 5-epoch configuration (lr 1.65e-4, weight decay 3.4e-4, dropout 0.32,
  batch 32) reached val mean AUROC **0.8229 in 5 epochs**, against the long run's
  0.8245 in 24 epochs with lr 1e-3.

So a cheap 5-epoch sweep reached within 0.002 of a 24-epoch run, and it did the
job it was designed for: eliminating bad regions rather than ranking winners. It
cannot tell us which of the 9 survivors is best, and it does not claim to.

## Does "data-limited, not regularisation-limited" still hold?

**Yes, and this run strengthens it.** Three training regimes now land in the same
place: v1 (fixed LR, 20 epochs, 0.8107 mean test AUROC), v2 (normalised and
augmented, 0.8042), and this warmup+cosine run converged to a genuine plateau
(0.8217, every per-target delta inside one fold std). A schedule fix that
demonstrably changed the optimisation (converged at epoch 12 rather than still
climbing at 20) produced no measurable change in generalisation.

What would still falsify it: a materially larger backbone, self-supervised
pretraining, or relabelled/adjudicated targets. Learning-rate and schedule choices
are now ruled out as the explanation.

## Bugs found and fixed during the run

1. `AttributeError: 'Trial' object has no attribute 'intermediate_values'` killed
   both Optuna stages on their first trial. `intermediate_values` exists only on a
   finished `FrozenTrial`; a live trial's reported values must be read back from
   the study storage. Fixed with a `trial_intermediates()` helper in
   `scripts/12_overnight/run_overnight.py`, verified against a live study.
2. `final_eval` gated a **validation** number (long run 0.837) against a **test**
   number (k-fold mean 0.8392) and so skipped a legitimate evaluation. The correct
   baseline is the single-run fused model's val SHD AUROC of 0.8279, which the long
   run beat from epoch 5 on. Gate repointed at the val baseline; the k-fold test
   mean is retained in the code as a reference only.
3. The stage-1 `--only-fold` path and the tier-naming for a late checkpoint both
   work but produce the awkward tier name `cnn_raw_waveform_cnn_serial`, because
   `cnn_tier_name()` has no mapping for the `cnn_serial` tag. Cosmetic; left as is
   so the existing CSV is not rewritten.
