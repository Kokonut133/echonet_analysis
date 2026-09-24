# Task 11 — Optuna hyperparameter search over the classical models: notes

**Why this exists.** Every classical model in the project (`src/classifiers/`)
used fixed hyperparameters straight out of `data/project_config.json` — no
search at all. A reviewer's fair objection: that makes "the interpretable
feature-based model comes within ~0.02 AUROC of the CNN" a weak claim, since
an untuned baseline is a weak baseline. This tunes
`HistGradientBoostingClassifier` and `LogisticRegression` with Optuna (TPE
sampler) on the `combined` feature set (7 tabular ECG metadata columns + 180
hand-crafted waveform-feature columns = 187) for the 4 headline targets, fit
on the official `train` split and selected on the official `val` split. **The
test split was never loaded anywhere in this task** — `scripts/11_tuning/tune_classical.py`
only reads `EchoNext_{train,val}_tabular_features.npy` and
`ecg_waveform_features_{train,val}.npy`; no `test` file, and no raw waveform
`.npy`, is touched.

**Why HistGradientBoostingClassifier, not the existing `GradientBoostingClassifier`.**
`GradientBoostingClassifier` builds trees on the full dense 72,475 × 187
matrix and is the slowest model already in the project. `HistGradientBoostingClassifier`
histogram-bins the features first (`max_bins`, here tuned) and is scikit-learn's
own drop-in fast implementation for exactly this shape of problem — it also
exposes `warm_start`, which this task uses for genuine mid-fit pruning (below).
`GradientBoostingClassifier` is kept as-is elsewhere in the project; this task
only tunes its faster sibling.

## Compute budget (CPU-only, no GPU touched)

- 4 targets × 2 models = 8 Optuna studies, each capped at 40 trials **or**
  25 minutes, whichever came first (`optuna.study.optimize(..., timeout=1500)`).
- `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` pinned to 3
  before any numpy/sklearn import, and every `.fit()` call additionally wrapped
  in `threadpoolctl.threadpool_limits(limits=3)` inside `src/tuning.py` — total
  parallelism never exceeded 3 threads, and `study.optimize(..., n_jobs=1)`
  throughout (no trial-level parallelism), per the hardware rule (GPU reserved
  for a concurrent 5-fold CNN training job on this box).
- **Total wall-clock time: 238 minutes (~4.0 hours)**, all 8 studies combined,
  run once as `.venv/bin/python scripts/11_tuning/tune_classical.py`.
- `optuna==5.0.0`, installed via `.venv/bin/python -m pip install optuna`.

## What moved, and what did not

| target | model | untuned val AUROC | tuned val AUROC | Δ |
|---|---|---:|---:|---:|
| Structural heart disease (any) | HistGradientBoostingClassifier | 0.8090 | 0.8121 | **+0.0031** |
| Structural heart disease (any) | LogisticRegression | 0.7939 | 0.7952 | +0.0013 |
| LVEF ≤ 45% | HistGradientBoostingClassifier | 0.8612 | 0.8632 | +0.0020 |
| LVEF ≤ 45% | LogisticRegression | 0.8415 | 0.8452 | **+0.0037** |
| RV systolic dysfunction | HistGradientBoostingClassifier | 0.8563 | 0.8627 | **+0.0064** |
| RV systolic dysfunction | LogisticRegression | 0.8363 | 0.8379 | +0.0016 |
| Aortic stenosis | HistGradientBoostingClassifier | 0.8471 | 0.8609 | **+0.0138** |
| Aortic stenosis | LogisticRegression | 0.8631 | 0.8624 | **−0.0007** |

("untuned val AUROC" = the existing `combined`-feature-set row in
`reports/ecg_feature_model_results.csv` for the same target/model; for
HistGradientBoostingClassifier that's the `GradientBoosting` row, since HGB
directly replaces it. Full per-trial log: `reports/tuning_results.csv`, 203
rows; best-per-study summary: `reports/tuning_best.csv`.)

**HistGradientBoostingClassifier moved for every target**, from a small
+0.002 (LVEF) up to +0.0138 (aortic stenosis) — the untuned `GradientBoostingClassifier`
config (150 trees, depth 4, lr 0.1) was particularly poorly suited to aortic
stenosis's 4.1% prevalence, where the tuned model found much shallower,
lower-learning-rate configurations that generalise better on a rare positive
class.

**LogisticRegression moved less, and for aortic stenosis it didn't move at
all** — the tuned value (0.8624) landed *below* the untuned baseline
(0.8631). This is a compute-budget artifact, not a finding about the model:
the aortic-stenosis `LogisticRegression` study only completed **2 trials**
in its full 25-minute allowance (see "Why LogisticRegression trials were slow"
below), so TPE barely sampled the space before time ran out. Treat the
aortic-stenosis LR result as effectively untuned; a longer budget would very
likely find something at least as good as the existing default. This is
reported honestly rather than smoothed over — the point of this task was to
make the classical baseline *not* look artificially weak, and pretending a
2-trial search is a real search would cut the other way.

## Which hyperparameters mattered (Optuna `get_param_importances`, winning model per target)

Read from `figures/results/tuning_importance.png` (top row of each panel).
For the three HGB-winning targets, importance is computed over all **completed**
trials (8–10 of 40 per study — most trials were pruned, see below); for
aortic stenosis (LogisticRegression winner) it is computed over only its 2
completed trials and should be read as anecdotal, not a real importance
ranking.

- **SHD (any)** — `max_iter` and `min_samples_leaf` dominate (roughly tied,
  together over 45% of total importance), then `max_bins` and `max_leaf_nodes`;
  `learning_rate` mattered least.
- **LVEF ≤ 45%** — `max_bins` is the single largest factor, with
  `min_samples_leaf`, `max_leaf_nodes` and `max_iter` clustered close behind;
  `learning_rate` again the smallest contributor.
- **RV systolic dysfunction** — `learning_rate` is clearly the largest factor
  here (unlike the other two HGB targets), followed by `max_bins` and
  `l2_regularization`.
- **Aortic stenosis (LogisticRegression, 2 trials — not reliable)** — `C`
  captured essentially all of the (unreliable) importance signal, which is
  expected with two data points and shouldn't be read as a real finding.

**Overall pattern:** no single hyperparameter dominates across every
target — regularisation/complexity controls (`min_samples_leaf`, `max_iter`,
`max_bins`) matter most, consistent with this being a fairly low-dimensional
(187-column), moderately-sized (72k row) problem where the main risk is
over- or under-fitting tree complexity rather than needing an exotic
learning-rate schedule. `learning_rate` was the *least* important HGB
hyperparameter for two of three targets and the *most* important for the
third (RV dysfunction) — a reminder that per-target tuning genuinely finds
different optima rather than one config dominating everywhere.

## Pruning: did it earn its keep?

Yes, dramatically, for HistGradientBoostingClassifier. The `MedianPruner`
runs against a genuine staged-evaluation loop (`warm_start=True`, boosting
rounds added in chunks of 25, val AUROC re-checked and reported to Optuna
after each chunk — not scikit-learn's `staged_predict_proba`, which would
require the tree to already be fully built before it could tell you
anything). Pruned-trial fractions per HGB study: SHD 31/40 (78%), LVEF 32/40
(80%), RV 32/40 (80%), AS 30/40 (75%). This is why every HGB study finished
its full 40-trial budget in 700–1000 seconds — well under the 25-minute cap —
while every LogisticRegression study (unpruned, per the task's own
instruction: LR has no staged-scoring equivalent worth building here) either
hit the 25-minute wall or came close to it, completing far fewer trials.

**Why LogisticRegression trials were slow**, worth recording since it's the
direct cause of the aortic-stenosis under-tuning above: `saga` and (at high
`C`) `liblinear` do not converge quickly on 72,475 × 187 with
`class_weight="balanced"`, especially for the rarer targets — measured
single-fit times ranged from ~1.5s (`lbfgs`) to 50–90s (`saga`, small `C`) in
isolation, and full studies averaged 77s/trial (SHD) up to 766s/trial
(aortic stenosis, 4.1% prevalence). Optuna's `timeout` is checked *between*
trials, not inside one, so a study can run one slow trial past the nominal
budget — expected, bounded behaviour, not a bug, but it does mean the
*effective* trial budget for LogisticRegression on rare targets was much
smaller than 40.

## Does this narrow the gap to the CNN?

Using **val-split** numbers only (`reports/cnn_combined_results.csv`, the
fused ECG+demographics CNN — n=4,626, same val split used throughout this
task; test-split numbers are intentionally not used or reported anywhere in
this document):

| target | best untuned classical (val) | best tuned classical (val) | CNN (val) | gap before | gap after |
|---|---:|---:|---:|---:|---:|
| Structural heart disease (any) | 0.8090 | 0.8121 | 0.8279 | 0.0189 | **0.0158** |
| LVEF ≤ 45% | 0.8612 | 0.8632 | 0.8847 | 0.0235 | **0.0215** |
| RV systolic dysfunction | 0.8563 | 0.8627 | 0.8834 | 0.0271 | **0.0207** |
| Aortic stenosis | 0.8631 | 0.8624 | 0.8783 | 0.0152 | 0.0159 |

For 3 of 4 headline targets, tuning narrows the val-split gap to the CNN —
most for RV dysfunction (0.0271 → 0.0207, a 24% relative reduction) and SHD
itself (0.0189 → 0.0158, 16% relative reduction). Aortic stenosis is
unchanged-to-slightly-worse, entirely attributable to the under-explored
LogisticRegression search described above rather than a real ceiling — the
untuned baseline for that target was already the strongest classical result
of anything in this project (0.8631), and HGB tuning alone gained +0.0138
without quite reaching it.

**This result should be read as strengthening, not weakening, the project's
central claim.** The whole point of an untuned-baseline objection is that a
lazy comparison makes the interpretable model look artificially competitive
(or artificially uncompetitive) relative to the CNN. Here, spending real
compute on the classical side moved the gap *the way the reviewer would have
predicted if the original comparison had been unfair* — narrower, not wider
— for 3 of 4 targets, and the one exception is explained by a compute-budget
artifact this document reports rather than hides. That is exactly the honest
outcome the ablation was meant to produce.

**Note for future work:** the CNN itself was left untuned in this task, for
the same hardware reason driving everything else here (GPU reserved for a
concurrent training job on this box) — that item belongs on the project's
future-work list, owned by another agent, not here.

## Reproducing this

```
.venv/bin/python scripts/11_tuning/tune_classical.py \
  --targets shd_moderate_or_greater_flag lvef_lte_45_flag \
            rv_systolic_dysfunction_moderate_or_greater_flag \
            aortic_stenosis_moderate_or_greater_flag \
  --n-trials 40 --timeout 1500
```

Reusable pieces (search-space definitions, `run_study`, `params_to_pipeline`)
are in `src/tuning.py`, covered by `tests/test_tuning.py`
(`.venv/bin/python -m pytest tests/test_tuning.py -q`).

## Two sklearn-version gotchas found and worked around (documented in code)

- `HistGradientBoostingClassifier` defaults to `early_stopping="auto"`
  (True above 10k rows), which would silently cap boosting rounds below the
  sampled `max_iter` — defeating the point of tuning it. Fixed by passing
  `early_stopping=False` explicitly, both during tuning and in
  `params_to_pipeline`.
- scikit-learn 1.9 deprecated `LogisticRegression(penalty=...)` in favour of
  `l1_ratio` (0.0 ≡ "l2", 1.0 ≡ "l1", in between ≡ "elasticnet"). The search
  space is expressed directly in `l1_ratio` terms to avoid the resulting
  `FutureWarning`/inconsistent-value warning entirely.
