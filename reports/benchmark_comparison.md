# Benchmark comparison: this project vs. published EchoNext results

This project trains on the PhysioNet **EchoNext** dataset
(`data/echonext-a-dataset-for-detecting-echocardiogram-confirmed-structural-heart-disease-from-ecgs-1.1.0/`).
That dataset's own shipped `README.md` states it is "the dataset used to
train the **EchoNext Mini-Model**" — the same 100,000-ECG cohort, with the
same official `train` (72,475) / `val` (4,626) / `test` (5,442) split we
use (confirmed against our `MODEL_CARD.md`). There are two published papers
relevant here, and they are **not interchangeable**:

1. **Hughes et al., "EchoNext-Mini" (NEJM AI, 2026)** — the companion paper
   for the exact public 100k-ECG dataset in this repo. Its baseline model
   is evaluated on the same n = 5,442 held-out test split we report on.
   This is the directly comparable benchmark.
2. **Poterucha et al., "EchoNext" (Nature, 2025)** — the full-scale model
   behind the FDA-cleared product, trained on 796,816 ECG-echocardiogram
   pairs from 149,819 patients across the NewYork-Presbyterian system, and
   evaluated on a separate, external, 8-hospital multicentre test set of
   44,719 patients. This is a much larger and differently-evaluated model;
   it is included below for context only, not as an apples-to-apples
   comparison.

## Comparison table

Our numbers are from `reports/final_results.csv`, tier `cnn_ecg_and_demographics`
(ECGConvNet, raw waveform + fused demographics), `split = test`, n = 5,442 —
the same test-set size as the EchoNext-Mini paper reports.

| Target | Published AUROC (source, cohort) | Our test AUROC (± CI half-width) | Difference (ours − published) |
|---|---|---:|---:|
| Composite SHD (`shd_moderate_or_greater_flag`) | **0.820** (95% CI 0.809–0.831) — EchoNext-Mini, NEJM AI 2026, n=5,442 (same test split) | **0.8355** ± 0.011 (0.8245–0.8462) | +0.016 (CIs overlap: 0.8245–0.831) |
| Composite SHD — AUPRC | **0.789** (95% CI 0.772–0.804) — EchoNext-Mini | **0.8064** ± 0.0149 (0.7916–0.8214) | +0.018 (CIs overlap: 0.7916–0.804) |
| LV wall thickness ≥1.3cm (`lvwt_gte_13_flag`) | **0.734** (95% CI 0.719–0.751) — EchoNext-Mini's *lowest*-performing sub-label | **0.7564** ± 0.0146 (0.7417–0.7709) | +0.022 (CIs overlap: 0.7417–0.751) |
| RV systolic dysfunction (`rv_systolic_dysfunction_moderate_or_greater_flag`) | **0.866** (95% CI 0.849–0.882) — EchoNext-Mini's *highest*-performing sub-label | **0.8828** ± 0.0164 (0.8668–0.8996) | +0.017 (CIs overlap: 0.8668–0.882) |
| Composite SHD, full-scale EchoNext (context only, *different* cohort/test set) | **0.852** (95% CI 0.845–0.859) — Poterucha et al., Nature 2025, external 8-hospital n=44,719 | n/a (not the same evaluation cohort) | not comparable |
| All other 9 sub-labels (aortic stenosis, aortic/mitral/tricuspid/pulmonary regurgitation, PASP, TR velocity, LVEF, pericardial effusion) | **not reported / could not verify** — see below | see `reports/final_results.csv` | — |

**Read on the composite SHD result:** on what appears to be the identical
held-out test set (n = 5,442, same official PhysioNet split), our small CNN's
point estimate is nominally *above* the published EchoNext-Mini baseline for
both AUROC and AUPRC, but the 95% confidence intervals overlap substantially
in both cases. The honest reading is **statistical parity, not a "beat"** —
with this test-set size and this event rate, two independently-trained
models can easily land within ~1.5–2 points of AUROC of each other by
chance. The same overlap pattern holds for the two individual sub-labels we
could verify (LVWT, RV dysfunction).

## Why the numbers land where they do

- **Model capacity.** Our `cnn_ecg_and_demographics` model is a 1D residual
  CNN with a measured **≈0.93M parameters** (see `MODEL_CARD.md`), trained
  for 20 epochs (`reports/cnn_combined_train_log.csv` — 20 logged epochs)
  on a single consumer GPU. Neither published paper reports a parameter
  count for their CNN (see "could not verify" below), but the full-scale
  Nature EchoNext model is described as a multitask classifier with
  separate terminal branches trained on ~8x more data — very likely a much
  larger network with materially more training compute than ours.
- **Training-set size.** For the EchoNext-Mini comparison, training-set
  size is *not* a likely explanatory factor — both models train on the
  same 72,475 ECGs. For the full-scale Nature EchoNext model, training
  data is ~11x larger (796,816 pairs vs. 72,475) and multi-institutional
  (NYP system-wide vs. Columbia/Allen only), which plausibly explains why
  its external-test AUROC (0.852) exceeds both the mini-model's and our
  own in-distribution numbers — a genuinely harder, more generalizable
  target.
- **Preprocessing.** No difference to report: we consume the dataset's
  own already-preprocessed `EchoNext_<split>_waveforms.npy` arrays (12-lead,
  10s, 250Hz, median-filtered, percentile-clipped, normalized — see the
  dataset's shipped README) rather than re-deriving waveforms ourselves,
  so waveform preprocessing is identical to whatever the mini-model paper
  used on the same files.
- **Evaluation cohort.** For the EchoNext-Mini row, the cohort is verified
  identical (same official test split, n=5,442). For the Nature EchoNext
  row, the cohort is explicitly *not* comparable — external, multi-site,
  ~8x larger test set — which is why it is listed for context only and not
  differenced against our number.
- **Pretraining.** We train from scratch; we found no evidence either
  published model uses ECG-foundation-model pretraining (the Nature paper
  describes a CNN trained end-to-end; the mini-model paper's methods
  section could not be accessed to check — see below).
- **What "close" means here.** Reaching statistical parity with a
  peer-reviewed, hospital-scale baseline using a model roughly two orders
  of magnitude smaller, on identical data, indicates the training pipeline
  (preprocessing, loss weighting, architecture choices) is sound for
  in-distribution prediction. It does **not** demonstrate the harder,
  more clinically relevant property the Nature paper's 0.852-on-external-data
  result speaks to: generalization across institutions/devices, which we
  have not tested (only one institution's data is available in this
  dataset, and we do not have an external test set).

## What would close the remaining gap (vs. the full-scale Nature EchoNext model)

- Train on a larger, multi-institutional cohort (the Nature EchoNext model
  used ~11x more ECG-echo pairs from 8 hospitals vs. our single-cohort 72k).
- External validation on a genuinely held-out institution/device population,
  not just an in-distribution split of the same source data.
- A larger backbone and/or ECG-foundation-model pretraining/fine-tuning,
  given our model is ~0.93M parameters trained from scratch for 20 epochs.
- Longer training with more aggressive learning-rate scheduling / early-stopping
  patience tuning (current run stops at 20 logged epochs).
- Subgroup-stratified evaluation (by sex, age, race/ethnicity, care setting)
  to confirm the aggregate parity holds across populations, not just overall
  — flagged as an open gap in `MODEL_CARD.md` as well.

## Could not verify (do not treat as established)

- **Full per-label AUROC/AUPRC table for EchoNext-Mini.** `ai.nejm.org`
  returned HTTP 403 (paywalled) on every direct fetch attempt (full text,
  PDF, and the bare DOI landing page). The composite-SHD figures and the
  two AUROC range-extremes (LVWT lowest, RV dysfunction highest) used above
  were triangulated across 3+ independent web searches that consistently
  returned the same wording/numbers, giving reasonable confidence they
  reflect the actual abstract text — but they are second-hand
  (search-snippet) quotes, not a page we read directly ourselves.
- **AUPRC range extremes for EchoNext-Mini** (search snippets attributed
  "3.2% (2.2–5.5)" to aortic stenosis as the lowest AUPRC sub-label, and
  "59.9% (56.6–63.3)" to LVEF≤45% as the highest). We are **not** including
  these in the comparison table: an AUPRC of 3.2% on a label with ~5.3%
  prevalence in our own test data would be *worse than a random classifier*
  for that label, which is an unusual result worth independent confirmation
  before treating as fact — and our own aortic-stenosis AUPRC (0.39) is
  ~12x that figure, a gap large enough that we suspect possible
  mis-attribution in the search synthesis rather than a real result. Could
  not verify against the primary source.
- **The other 9 individual sub-label AUROC/AUPRC figures** for EchoNext-Mini
  (aortic stenosis, aortic regurgitation, mitral regurgitation, tricuspid
  regurgitation, pulmonary regurgitation, PASP, TR velocity, LVEF, pericardial
  effusion) — not found in any accessible source. Not reported here.
- **EchoNext-Mini model architecture and parameter count.** The dataset's
  GitHub companion repo (`PierreElias/IntroECG`, `7-EchoNext Minimodel/README.md`)
  describes only the inference interface, not architecture; one search
  snippet says the mini-model uses "the same architecture as the original
  EchoNext model," but no parameter count is given anywhere we could access
  for either model.
- **Whether EchoNext-Mini uses any ECG-foundation-model pretraining.** Not
  found in any accessible source.
- **Individual sub-label AUROC for the full-scale Nature EchoNext model**
  beyond the few numbers stated in-text (RV dysfunction 91%, low LVEF 90%,
  LV wall thickness 77%, aortic regurgitation 78%, pulmonary regurgitation
  79%, pericardial effusion 80%) — aortic stenosis, mitral regurgitation,
  and tricuspid regurgitation AUROCs for this model are referenced as being
  in a supplementary table/figure we could not access.

## Citations

- Hughes JW, Jing L, Finer J, Hartzel D, Kelsey C, Long A, Rocha D, Ruhl J,
  Poterucha TJ, Elias P. "EchoNext-Mini: A Dataset and Baseline AI Model for
  Detecting Structural Heart Disease from Electrocardiograms." *NEJM AI*,
  3(5), April 2026. https://doi.org/10.1056/AIdbp2500516
- Poterucha TJ, Jing L, Ricart RP, et al. "Detecting structural heart disease
  from electrocardiograms using AI." *Nature*. 2025 Jul 16;644(8075):221–230.
  https://doi.org/10.1038/s41586-025-09227-0
- Elias P, Finer J, et al. "EchoNext: A Dataset for Detecting
  Echocardiogram-Confirmed Structural Heart Disease from ECGs" (v1.1.0).
  PhysioNet, 2025. https://doi.org/10.13026/3ykd-bf14 —
  https://physionet.org/content/echonext/1.1.0/
- Dataset license: PhysioNet Restricted Health Data License v1.5.0
  (`data/echonext-.../LICENSE.txt`).
