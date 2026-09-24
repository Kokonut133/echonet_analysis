# Future work

This is a roadmap, not a wish list: every item below is tied to a specific number or
finding already in this repository, and every item says what it would cost to do on
the hardware this project was actually built on.

## Hardware this project runs on

Everything here was built and trained on one machine: 8 CPU cores, 15 GB RAM, a
single NVIDIA RTX 4060 Ti with 8 GB VRAM, with the 29 GB dataset on a
Windows-mounted drive under WSL2 (`/mnt/c`), which makes disk I/O the training
bottleneck rather than the GPU. A single epoch over the 72,475-record training
split of the 930,252-parameter CNN (`MODEL_CARD.md`) costs about 2-3 minutes, and
every CNN run in this repo used a 20-30 epoch budget as a result. That constraint
shapes almost every item below, and where it does, the item says exactly which
resource is binding and what it would take to remove it.

## Tagging scheme

Every item carries exactly one tag:

- `[feasible here]`: could be done on the machine described above, even if it is
  slow.
- `[hardware-limited]`: blocked or impractical on 8 GB VRAM, 15 GB RAM, or the
  current disk I/O. The item says specifically which of those is the binding
  constraint and what resource change would remove it.
- `[needs external data]`: blocked by data access, not by compute.

Each item also says, explicitly, whether its main payoff would be a better number
on the results tables, or more confidence that the numbers already reported hold
up. Most project write-ups only list the first kind. A reviewer usually cares more
about the second, so this file keeps the two separate rather than letting
"future work" quietly mean "things that would raise AUROC."

---

## Modelling

#### CNN hyperparameter search `[feasible here]`

The classical models (logistic regression, random forest, gradient boosting) are
being tuned by a parallel effort on this project; the CNN never got the same
treatment, and `data/project_config.json` fixes one hand-picked setting
(`learning_rate: 0.001`, `weight_decay: 0.0001`, `dropout: 0.3`, `batch_size: 32`)
that every CNN variant in this repo, waveform-only, fused, v2-augmented, was
trained with. No sweep over learning rate, weight decay, dropout, or channel width
has been run. A random search of 15-25 trials over those four axes, with an
aggressive early-kill rule (stop a trial by epoch 5 if it is clearly behind), is
the realistic budget: at 2-3 minutes/epoch that is on the order of a day of
single-GPU wall clock, not blocked by 8 GB VRAM since the current model leaves
most of it unused. This is a performance item, and the targets most likely to move
are the ones furthest from data-limited, since the fixed hyperparameters were
never adapted to any single target.

#### Longer training with a warmup and cosine learning-rate schedule `[feasible here]`

Training currently uses only `ReduceLROnPlateau` (factor 0.5, patience 5,
`data/project_config.json`), with no learning-rate warmup and no cosine decay.
Both CNN runs early-stopped well inside their budget: v1 peaked at epoch 10 of a
20-epoch run, v2 (normalised and augmented) peaked at epoch 17 of a 25-epoch run
(`reports/cnn_v2_notes.md`). A short warmup followed by cosine decay over a longer
fixed budget, say 60-80 epochs, is the standard fix for a plateau-triggered
schedule stopping too early rather than the model having actually converged.
Effort is the same per-epoch cost as any existing run, so a full 60-80 epoch run
is 2-4 hours, feasible overnight on the 4060 Ti. Framed as performance, but
tempered: `cnn_v2_notes.md`'s own finding is that training 7 epochs further (17
vs. 10) did not raise held-out test AUROC (0.8107 vs. 0.8042 mean), so this is
worth trying but not guaranteed to move the ceiling.

#### Deeper or wider 1D CNN variants `[feasible here]`

The current backbone is four residual stages (32 to 64 to 128 to 256 to 256
channels, `src/models/cnn.py`) totaling 930,252 parameters. At batch size 32
(`project_config.json`) the activations at every stage are small (sequence length
halves at each stride-2 stage, starting from 2500 samples), so there is
considerable headroom in 8 GB of VRAM before a wider or deeper convolutional
backbone becomes memory-bound. A ResNet-34-style widen, or an extra residual
stage, is a reasonable next step to try before reaching for a different
architecture family. Effort: comparable to any other CNN training run, plus
whatever extra time the larger model needs per epoch (modest, since convolutions
scale with parameter count, not sequence length squared). Performance item.

#### Sequence models: 1D transformer, conformer, or structured state-space model `[hardware-limited]`

Binding constraint: 8 GB VRAM against a 12 x 2500 input. Full self-attention over
a 2500-step sequence needs an attention score matrix that scales with the square
of sequence length; at batch size 32 (the value used throughout this project) even
a modest 8-head transformer layer would need several gigabytes just for attention
scores in fp32, before counting the rest of the model or the optimizer state. That
either forces a much smaller batch size, sequence patching down from 2500 to a few
hundred tokens, mixed precision, and gradient accumulation, all real engineering
on top of the model itself, or a bigger card: something in the 24-40 GB range
would let a transformer or conformer train at a reasonable batch size without
patching. Structured state-space models (S4, Mamba-style) scale linearly in
sequence length rather than quadratically and are the more plausible sequence
architecture to attempt on this hardware, but their CUDA kernels are less mature
than a plain CNN's and would be new engineering, not a drop-in replacement.
Performance item, and the one most gated by hardware on this list.

#### Multi-task loss weighting and per-label operating thresholds `[feasible here]`

Prevalence across the 12 targets ranges from 0.8% (pulmonary regurgitation) to
52.2% (broad SHD) (README target table). `src/training.py::compute_pos_weights`
already rebalances each label's own positive/negative ratio inside the loss
(`pos_weight = (1-p)/p`), but all 12 per-label losses are then summed with equal
weight into one backward pass, so a target with more or cleaner signal can still
dominate the shared trunk's gradient. Two changes are cheap: tune per-target loss
weights (uncertainty weighting or GradNorm rather than a flat sum), and select a
decision threshold per target instead of the default 0.5, since `MODEL_CARD.md`
already flags that `pos_weight` rebalancing makes 0.5 a poor default threshold.
Loss reweighting needs a retrain (hours); per-label thresholds are minutes of work
against the predictions already cached in `reports/predictions/`. The retrain is a
performance item; the per-label thresholds are a confidence item, since they fix
how an already-trained model's output is read rather than what it learns.

#### Label-noise modelling `[feasible here]`

`reports/cnn_v2_notes.md`'s own conclusion, after comparing the unaugmented CNN to
a per-record-normalised, augmented version, is that the gap looks data-limited
rather than regularisation-limited: v2 delayed overfitting (best epoch moved from
10 to 17) but landed at a lower held-out mean test AUROC than v1 (0.8042 vs.
0.8107), with every per-target confidence interval overlapping. If that read is
right, a training objective that models label noise directly (a noise-tolerant loss, co-teaching-style
filtering of high-loss examples, or label smoothing sized to plausible echo-reader
disagreement rates) targets the actual bottleneck more directly than further
regularisation does. Effort: one CNN training run, same order of cost as any other
variant in this repo. Performance item, motivated by a result that is really about
confidence in where the ceiling comes from.

#### Group-balanced loss or per-group thresholds for the age-band gap `[feasible here]`

`reports/subgroup_results.csv` shows the fused CNN at 0.7642 AUROC on broad SHD
for patients 80 and older, against 0.8451 for the 50-65 band; the waveform-only
CNN shows the same pattern (0.7703 vs. 0.8401). README's own reading, given that
64% of the 80+ band is positive, is that little separating signal is left in that
group regardless of what the model is given, and giving the fused model age as an
input did not close the gap (`reports/ablation_notes.md` shows zeroing age costs
only up to 0.02 AUROC outside aortic stenosis, so age is not what is missing
here). Nothing in the repo has been recalibrated or reweighted in response to this
so far (README Limitations says so directly). A group-balanced loss term
(upweighting the 80+ band's contribution to the BCE loss) is worth trying, but
should be judged against the cheaper alternative of separate per-age-band decision
thresholds, since a ceiling caused by low residual class separation is not
obviously fixed by reweighting the loss. Loss reweighting needs a retrain;
per-group thresholds are minutes against cached predictions. Both are performance
items for the affected subgroup, though the honest expectation, given the
prevalence argument above, is a modest gain at best.

#### Calibration-aware training `[feasible here]`

A separate effort in this repository (`src/calibration.py`, currently in
progress) covers post-hoc calibration: Brier score, reliability curves, and
Platt/isotonic recalibration applied after a model is already trained. What
belongs here instead is changing what happens during training. `tests/
test_calibration.py`'s own test fixture for an "overconfident" model describes
"scores pushed toward 0/1 far more than the true positive rate warrants ... like a
pos_weight-inflated CNN," which is a direct description of what `pos_weight =
(1-p)/p` (`src/training.py`) does to this project's own CNN outputs. Training-time
options include a learned temperature-scaling parameter jointly optimised with the
classifier, focal loss or label smoothing in place of `pos_weight`, or a
calibration-aware imbalance correction that does not distort the probability
scale the way `pos_weight` does. Effort: comparable to any other CNN retrain; a
temperature parameter adds negligible cost. Confidence item, and complementary to
the post-hoc work rather than overlapping with it.

---

## Data use

#### Self-supervised pretraining on the unused `no_split` records `[feasible here]`

17,457 of the 100,000 EchoNext records sit in the `no_split` portion of the
dataset and are not used anywhere in this project's pipeline (`MODEL_CARD.md`
Training Data, `progress_log.md` step 1). A contrastive objective built on the
augmentations already implemented for the v2 CNN (`src/augmentation.py`: time
shift, amplitude scale, baseline wander, noise) or a masked-reconstruction
objective over waveform patches could pretrain the CNN backbone on these plus the
72,475 train records (89,932 waveform records in total) before fine-tuning on
labels. Per-epoch cost is the same order as supervised training since it reads the
same files at the same size; pretraining typically needs more epochs to pay off,
so budget several times a single supervised run's wall clock. One thing worth
fixing first: `no_split`'s waveform amplitude differs by roughly 10x from val/test
(see below), and that gap should be understood before treating `no_split` as drawn
from the same distribution for pretraining. Performance item.

#### Self-supervised pretraining or fine-tuning on external ECG corpora `[needs external data]`

Pretraining a backbone on a public 12-lead corpus such as PTB-XL, CODE-15, or
MIMIC-IV-ECG, then fine-tuning on EchoNext's labels, is a standard way to help a
small labeled set. What blocks it here is access and integration, not compute:
none of these datasets appear anywhere in `data/project_config.json` or this
repository, each has a separate access process (PTB-XL downloads directly from
PhysioNet, CODE-15 and MIMIC-IV-ECG both require their own credentialed-access
requests), and each has its own lead conventions, sample rate, and preprocessing
that would need reconciling with this project's fixed 250 Hz / 2500-sample /
12-lead format (`data/project_config.json`) before `src/dataset.py`'s loading code
could be reused as-is. Effort is mostly a new loader and resampling/alignment work
once access is granted; the pretraining compute itself is the same order as the
`no_split` item above, just over a larger corpus. Performance item.

#### Using the full EchoNext cohort rather than the 100,000-record subset `[needs external data]`

This project uses the 100,000-record EchoNext release published on PhysioNet
(README, `MODEL_CARD.md`); that release is PhysioNet's own fixed-size public
subset of the underlying study cohort, and no larger figure for that cohort
appears anywhere in this repository. Whether a bigger release ever becomes public
is a question for the dataset's maintainers, not something this project can act
on directly. If it did, `src/dataset.py`'s split logic would need no changes,
since splits are already read from the metadata's own `split` column, but
per-epoch time would grow roughly with record count, worsening the I/O bottleneck
already described above. This is an access question, not a redesign. Performance
item, contingent on access that may never materialise.

#### External validation on a different institution's ECGs `[needs external data]`

Every number in this repository comes from EchoNext's own train/val/test splits
of one institution's data; both README's Limitations section and
`MODEL_CARD.md`'s Ethical Considerations say this directly, and no retuning inside
this repo changes it. What is needed is a second labeled ECG-plus-echo cohort from
a different health system, ideally covering an overlapping set of the 12 targets.
This is arguably the single highest-value item in this file for confidence rather
than performance: every other item here can move the numbers, but only this one
tells a reviewer whether the headline 0.836 AUROC for broad SHD, or the 0.879 for
aortic stenosis (`reports/final_results_summary.md`), generalises past one
hospital's ECG machines, population mix, and echo-reading conventions. Effort is
dominated by finding and negotiating access to a second dataset; once available,
evaluation reuses `scripts/6_evaluate/evaluate_test_set.py` unchanged.

#### Resolve the unexplained roughly 10x waveform amplitude gap in `no_split` `[feasible here]`

`progress_log.md` step 1 records that signal amplitude differs by about 10x
between the `no_split` set (std approximately 0.11) and val/test (std
approximately 1.0), and `MODEL_CARD.md` repeats it as a known signal factor that
per-record normalisation was added to paper over, without the underlying cause
ever being chased down. A short investigation, loading `no_split` waveforms
directly through the existing mmap path (`src/dataset.py::load_split`) and
comparing per-lead statistics against val/test, would distinguish a straightforward
explanation (a scaling or unit difference introduced during that split's export)
from something that would matter more (a systematic device or gain difference in
the underlying records). This is an afternoon of array inspection, not a training
run. Left unresolved, it is a live risk for the self-supervised pretraining item
above: pretraining on `no_split` data with an unexplained scale difference could
teach the backbone a normalisation artifact rather than real waveform structure.
Purely a confidence item: it changes no reported metric, but protects whatever is
built on that file next.

---

## Evaluation

#### Multiple-comparison-aware reporting `[feasible here]`

Every comparison in `reports/final_results.csv` carries a bootstrap confidence
interval, which is good practice, but with 12 targets compared across up to 7
tiers, some of the "tier X beats tier Y" claims in this project are effectively
multiple comparisons without correction. A concrete example already in the data:
on broad SHD, the raw-waveform CNN's test CI is [0.8176, 0.8393] and the
demographics-fused CNN's is [0.8245, 0.8462] (`reports/final_results.csv`); the
two overlap even though the fused model's point estimate is 0.0073 higher. Running
a Holm-Bonferroni or similar correction specifically over the headline tier-vs-tier
comparisons would tell a reviewer which individual per-target deltas survive
correction and which look real only until multiple testing is accounted for.
Effort: arithmetic over numbers already computed, on the order of an hour.
Confidence item.

#### Repeated-seed variance for the CNN `[feasible here]`

Every CNN result reported here, waveform-only, fused, and v2, comes from a single
training seed. The 0.828-to-0.836 gap between the waveform-only and fused CNN on
broad SHD (`reports/final_results_summary.md`) has a bootstrap CI over the test
set, but no estimate of how much that gap would move if training were simply
re-run with a different seed, which is a different source of noise than sampling
noise in the test set. Training the same configuration 3-5 times and reporting
the spread across seeds would separate "this architecture choice reliably wins"
from "this run happened to land well." Effort: 3-5 repeats of a roughly one-hour
CNN training run, so about half a day of unattended wall clock, fully feasible on
the existing GPU since repeats run sequentially rather than needing more memory.
Confidence item, and likely the cheapest one on this list relative to what it
would settle.

#### Per-label operating-point tables beyond the one worked example `[feasible here]`

README's operating-points section works through one target: at 90% sensitivity
the CNN also flags about 51% of people without broad SHD, and at 80% sensitivity
that drops to 32%. The other 11 targets do not get the same treatment, and the
rare ones need it most: pulmonary regurgitation (0.8% prevalence) has an AUPRC of
only 0.0209 at the combined tier and 0.0037 at demographics-only
(`reports/final_results.csv`), meaning AUROC alone hides how few flagged cases
would actually be positive at a usable threshold. Effort: this reuses the plotting
code already built for the SHD example (`scripts/7_figures`) against the other 11
targets, on the order of an hour or two. Confidence item: it does not change any
model, only what a reader can see about the same numbers already reported.

---

## Interpretability

#### Occlusion-based validation of the saliency and Grad-CAM findings `[feasible here]`

`reports/interpretability_notes.md` states its own limitation plainly: SmoothGrad
and Grad-CAM are local, first-order approximations, noisy and sensitive to the
implicit zero baseline, and do not establish that the highlighted region is
causally necessary for a prediction the way an ablation or occlusion study would.
The lead-ablation work (`scripts/9_ablation/lead_ablation.py`) already runs
exactly that kind of causal check, but only across whole leads, never across time.
Extending the same zero-out-and-remeasure approach to temporal segments (the QRS
window vs. the ST-T window vs. the P wave) would turn the saliency finding that
importance peaks at the QRS complex with a secondary rise through the ST-T segment
around +100 to +300 ms post-R (`reports/interpretability_notes.md`) into a causal
claim rather than a gradient-based one. Effort: a variant of an existing script,
an afternoon, no retraining needed. Confidence item.

#### Systematic error analysis on cached predictions `[feasible here]`

`reports/predictions/` already caches per-tier test predictions, which is what
lets a new checkpoint be scored in minutes (README design decisions). What has not
been done is a structured look at the CNN's largest false positives and false
negatives, cut by target and by demographic subgroup. The subgroup analysis
reports aggregate AUROC by group (the 0.7642-vs-0.8451 age gap above, for
instance) but does not look at what the specific missed or over-flagged cases have
in common. Effort: a few hours of analysis against files that already exist,
no new predictions required. Confidence item: it explains an already-reported
number rather than producing a new one.

---

## Deployment and engineering

#### ONNX export `[feasible here]`

Exporting the trained `ECGConvNet` (930,252 parameters, `MODEL_CARD.md`) from
`checkpoints/cnn_waveforms.pt` to ONNX via `torch.onnx.export` against the model
definition in `src/models/cnn.py` is small and mechanical, and removes the
PyTorch/CUDA runtime dependency for anything that only needs to run inference.
Effort: about an hour, including a numerical-equivalence check against the
original checkpoint. Deployment item, prerequisite for the next two.

#### CPU latency and throughput benchmarking `[feasible here]`

No latency or throughput number exists anywhere in this repository; every
reported number so far is accuracy-only. A 930,252-parameter model is small enough
that CPU inference is realistic, and this project already has 8 CPU cores
available on the development machine itself to benchmark single-record latency
and batched throughput against, no GPU required for inference at this size.
Effort: a benchmarking script, an hour or two. Deployment item, and a reasonable
prerequisite before claiming the model is practical to serve anywhere.

#### A minimal inference API `[feasible here]`

A thin API (FastAPI or similar) that loads the exported checkpoint and serves
single-record predictions, ideally returning the calibrated probabilities from the
separate post-hoc calibration effort (`src/calibration.py`) once that work lands,
would demonstrate the pipeline end-to-end instead of stopping at offline
evaluation scripts. Effort: about a day, mostly plumbing around preprocessing that
already exists in `src/dataset.py` and the forward pass in `src/models/cnn.py`.
Deployment item.

#### Drift monitoring, designed against existing data `[feasible here]`

With no production deployment there is no live traffic to monitor yet, but the
design and a prototype can be built now: tracking input feature distributions (the
180 waveform features already computed in `scripts/2_preprocess`) against the
training distribution with a population-stability-index or KS-test alarm, and
specifically flagging the kind of drift the `no_split` file already shows in this
project's own data, a roughly 10x amplitude shift (`progress_log.md`) that would
have tripped exactly this kind of monitor had it existed. Effort: a design
document plus a prototype script run against static val/test data as a stand-in
for live traffic, feasible without any running system. Deployment item, grounded
in a concrete failure mode this project has already seen once.

#### Cache the training waveform array fully in RAM `[hardware-limited]`

Binding constraint: 15 GB total RAM against a training array that is itself about
16 GB on disk (README design decisions: "The raw training array is about 16 GB,
and nothing in the pipeline loads it fully into RAM"), which is exactly why
`src/dataset.py` opens it with `np.load(..., mmap_mode="r")` instead of caching
it. Memory-mapping avoids an out-of-memory crash but pays a read cost on every
epoch instead of paying it once. Getting to somewhere around 32-64 GB of RAM, so
the OS page cache can hold the array after the first epoch, would remove that
repeated cost. This is a hardware upgrade, not new engineering: the mmap code
already in `src/dataset.py` would benefit without any changes.

#### Move the dataset onto local NVMe storage `[hardware-limited]`

Binding constraint: the 29 GB dataset lives on a Windows-mounted drive under
WSL2 (`/mnt/c`), and that cross-filesystem I/O, not the GPU and not the
930,252-parameter model, is the reason a single epoch over 72,475 records costs
2-3 minutes and every CNN run in this repository capped out at a 20-30 epoch
budget. Copying the dataset onto local NVMe storage inside the WSL2 filesystem
(roughly 30-40 GB of free space needed for the dataset plus checkpoints), or
converting the raw `.npy` arrays to a more sequential-read-friendly chunked format
such as sharded HDF5 or WebDataset-style shards, would remove the cross-filesystem
penalty. This is arguably the single most impactful infrastructure change in
this file, since the hyperparameter search, longer-schedule training, and
repeated-seed items above are all bottlenecked by the same 2-3 minutes/epoch
figure. It needs local fast storage, not a bigger GPU.

---

## Summary by section

| Section | Items | feasible here | hardware-limited | needs external data |
|---|---:|---:|---:|---:|
| Modelling | 8 | 7 | 1 | 0 |
| Data use | 4 | 2 | 0 | 2 |
| Evaluation | 4 | 3 | 0 | 1 |
| Interpretability | 2 | 2 | 0 | 0 |
| Deployment and engineering | 6 | 4 | 2 | 0 |
| **Total** | **24** | **18** | **3** | **3** |
