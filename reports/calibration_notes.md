# Calibration notes

The CNN's raw output is a good *ranking* score (AUROC 0.83-0.88 on the headline
targets) but a poor *probability*. On `cnn_ecg_and_demographics`, raw Brier
skill score (vs. always predicting prevalence) is +0.33 for the SHD composite
but **-0.84 for RV dysfunction and -2.41 for aortic stenosis** — the raw
probabilities are worse than just reporting the base rate. Raw ECE is 0.042
(SHD) up to 0.19-0.27 (LVEF, RV dysfunction, aortic stenosis).

Direction: **over-confident, and worse for rarer targets.** Mean predicted
probability exceeds true prevalence at every rung — 0.36 vs. 0.18 (LVEF),
0.30 vs. 0.08 (RV), 0.32 vs. 0.05 (aortic stenosis) — because `pos_weight =
(1-p)/p` (`src/training.py::compute_pos_weights`) scales the loss penalty on
missed positives by that same factor, and it grows fast as prevalence shrinks
(~1.3x for SHD at 43% prevalence, ~18x for aortic stenosis at 5%). The model
is trained to call positives generously, not to report calibrated risk.

Platt scaling and isotonic regression (fit on a held-out stratified half of
test, scored on the other half — see `calibration_results.csv`) both pull
this back sharply: ECE drops to ~0.01-0.03 and Brier skill turns positive
everywhere it was tested (e.g. aortic stenosis: -2.41 -> +0.22). Both are
monotone maps, so **AUROC is unchanged** (Platt: exact to 1e-6; isotonic:
within ~0.02 from tie-breaking on small fit samples — see the `[note]` lines
in the script's stdout — never a real ranking loss).

Practical consequence: **do not read the raw output as a probability, and do
not use 0.5 as the operating threshold.** A raw score of 0.5 corresponds to a
recalibrated probability near 0.5 only for SHD; for rarer targets a raw score
of 0.5 is nowhere near a 50% true risk (e.g. a raw score of ~0.97 is needed
before aortic stenosis risk actually crosses 50%). A screening deployment
should pick its operating threshold from the recalibrated probability (or
directly from the ROC/PR curve at the desired sensitivity), never from the
raw 0.5 cutoff used elsewhere in this repo's balanced-accuracy tables.
