# Task 6 — Lead ablation and feature importance: notes

Two independent analyses of "which ECG leads / features carry the signal":

- **6a — CNN lead ablation** (`scripts/9_ablation/lead_ablation.py`): the trained
  `ECGConvNet` (checkpoints/cnn_waveforms.pt) evaluated on val with leads zeroed
  out one at a time (sensitivity analysis; the model was **not** retrained on
  missing leads). See `reports/lead_ablation_results.csv`,
  `figures/ablation/lead_ablation.png`, `figures/ablation/reduced_lead_sets.png`.
- **6b — GBM permutation importance** (`scripts/9_ablation/feature_importance.py`):
  a GradientBoosting model retrained per headline target on the `combined`
  (tabular[7] + waveform[180] = 187) feature set, with `sklearn.inspection.permutation_importance`
  on val (n_repeats=5). See `reports/feature_importance.csv`,
  `figures/ablation/feature_importance.png`.

## Do the CNN lead ablation and the GBM lead importance agree?

1. **Partial agreement on chest leads.** Both methods put V1 and V2 among the
   most informative leads for the headline targets — CNN: zeroing V1 costs
   -0.025 mean ΔAUROC (largest of any lead); GBM: V1 and V2 are its 2nd/3rd
   highest-importance leads (Σimportance 0.038 / 0.036), behind only aVR.
2. **Disagreement on which limb lead matters most.** CNN ranks lead I highest
   among limb leads (-0.012 ΔAUROC when dropped, 2nd overall); GBM ranks aVR
   highest (Σimportance 0.050, its single most important lead). Both agree
   *a* limb lead matters alongside the chest leads, but not which one.
3. **The comparison isn't apples-to-apples.** The GBM's `combined` set includes
   7 tabular metadata features (rate, intervals, age, sex) the CNN never sees,
   and these dominate for some targets — tabular features account for 83% of
   total GBM importance on aortic stenosis (almost entirely `ventricular_rate`)
   and 46% on SHD, so GBM "lead importance" there is really "what's left after
   a strong heart-rate effect."
4. **Where the tabular effect is weaker, the two methods converge.** For LVEF
   and RV dysfunction (tabular share only 22% / 30%), both methods point to
   V1/V2/V3 (chest) plus aVR/I (limb) as the load-bearing leads — consistent
   with the anteroseptal / RV-facing anatomy those leads capture.
5. **Reduced-lead-set take-away.** Single-lead monitoring is a poor substitute
   for full 12-lead: lead II alone retains only 52-77% of full-12-lead AUROC
   across the headline targets (worst on RV dysfunction: 0.46 vs. 0.885, at
   chance; best on aortic stenosis: 77%; on LVEF ≤45% specifically, 60%: 0.53
   vs. 0.88). The clinically-motivated 4-lead set (I, II, V1, V5) does far
   better, retaining 86-92% of full AUROC, on par with the limb-only (90-96%)
   and chest-only (89-94%) 6-lead subsets — so if electrode count must be
   cut, the 4-lead reduced set is a much safer fallback than any single lead.

---

## Demographic ablation — the ECG+demographics CNN

`scripts/9_ablation/demographic_ablation.py` zeroes the 13 encoded demographic
inputs of the `cnn_combined` checkpoint at inference — the whole block, then one
group at a time — on the held-out test split. Zero is the neutral value for both
encodings here (a one-hot block of zeros carries no category; the age column is
standardised, so zero is the training mean). As with the lead ablation, the model
is not retrained.

| Zeroed input | Mean ΔAUROC (12 targets) | Aortic stenosis |
|---|---:|---:|
| All demographics | −0.0273 | −0.147 |
| **Age** | **−0.0189** | **−0.145** |
| Care setting | −0.0034 | −0.004 |
| Sex | −0.0026 | +0.001 |
| Race / ethnicity | −0.0021 | +0.003 |

1. **Demographics are worth 0.027 AUROC, and age is 69% of it.** Removing age
   alone costs almost as much as removing everything. Sex, care setting and
   race/ethnicity together account for under 0.009.
2. **Age is not a diffuse effect — it is almost entirely one diagnosis.** Zeroing
   age costs 0.145 AUROC on aortic stenosis, an order of magnitude more than on
   any other target (next largest: 0.02). This is the same target where
   demographics-only beat the waveform-only CNN, and where fusing demographics in
   recovered +0.050 on test. Age-related valve calcification is doing the work,
   and the model is using it exactly where clinical knowledge says it should.
3. **Race/ethnicity is not a material input.** Zeroing it changes mean AUROC by
   −0.0021, and on the headline SHD target the paired bootstrap interval spans
   zero (−0.0024 to +0.0009), as it does on aortic stenosis (−0.0044 to +0.0101).
   The model's performance therefore does not depend on the patient's recorded
   race/ethnicity — reassuring for a clinical model, and the reason no dedicated
   figure is included for it.
4. **The care-setting leakage concern did not materialise either.** `location_setting`
   (emergency / inpatient / outpatient / procedural) was the input most at risk of
   being a proxy for "this patient is already known to be sick", but zeroing it
   costs only 0.0034 mean AUROC with an interval spanning zero on the headline
   target (−0.0019 to +0.0022). The fused model is reading physiology and age, not
   care context.
