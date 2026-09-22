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
