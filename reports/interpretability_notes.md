# Interpretability notes — waveform CNN (`cnn_waveforms.pt`)

Method: SmoothGrad input-gradient saliency (20 noisy copies, noise std = 10% of
each record's own signal std) and Grad-CAM on `model.stage4` (the last ResBlock),
evaluated on the **test** split only (5,442 records; never train/val).

- **Lead I and the right precordial leads V1/V2 dominate every headline target.**
  Mean saliency share: V1 ≈ 0.16-0.19, Lead I ≈ 0.11-0.13, vs. a uniform baseline
  of 0.083 per lead (12 leads). Limb leads III and aVF contribute the least
  (≈ 0.04 each) — see `reports/lead_importance.csv` and `figures/interpretability/lead_importance.png`.
- This pattern holds for SHD (broad), LVEF ≤45%, RV systolic dysfunction, and
  aortic stenosis alike — the model leans on the same anterior/right-sided leads
  regardless of target, suggesting it has learned a shared "abnormal QRS
  morphology" representation rather than target-specific lead selection.
- **Grad-CAM importance peaks sharply at the QRS complex** (the R-peak) for all
  four targets, with a secondary, broader rise through the ST-T segment
  (roughly +100 to +300 ms post-R) — see `figures/interpretability/gradcam_beat_average.png`.
  Importance is comparatively flat over the P wave and the pre-QRS baseline.
- Per-example saliency overlays (`figures/interpretability/saliency_examples.png`)
  confirm this at the individual-beat level: for high-confidence true positives
  on LVEF ≤45% and RV systolic dysfunction, the brightest saliency consistently
  sits on the QRS deflection in V1/V2, with occasional flares on the T wave.

**Caveat:** gradient-based saliency (even with SmoothGrad) is a local, first-order
approximation of the model's behavior around the input — it can be noisy, is
sensitive to the implicit "zero" baseline, and does not establish that the
highlighted region is causally necessary for the prediction (unlike an
ablation/occlusion study). Treat these maps as a qualitative read on *where the
model is looking*, not a certified explanation of *why* it predicts what it does.
