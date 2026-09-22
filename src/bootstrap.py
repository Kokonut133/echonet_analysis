from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_N_RESAMPLES = 1000
DEFAULT_SEED = 42


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `metric_fn(y_true, y_prob)`, resampled over records."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    samples: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        samples.append(metric_fn(yt, yp))

    if not samples:
        return float("nan"), float("nan")

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(samples, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def bootstrap_auroc_auprc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    ci: float = 0.95,
) -> dict[str, float]:
    """95% bootstrap CIs for AUROC and AUPRC in a single resampling loop."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    auroc_samples: list[float] = []
    auprc_samples: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        auroc_samples.append(roc_auc_score(yt, yp))
        auprc_samples.append(average_precision_score(yt, yp))

    alpha = (1.0 - ci) / 2.0

    def _percentiles(samples: list[float]) -> tuple[float, float]:
        if not samples:
            return float("nan"), float("nan")
        lo, hi = np.percentile(samples, [100 * alpha, 100 * (1 - alpha)])
        return float(lo), float(hi)

    auroc_lo, auroc_hi = _percentiles(auroc_samples)
    auprc_lo, auprc_hi = _percentiles(auprc_samples)

    return {
        "auroc_ci_low": auroc_lo,
        "auroc_ci_high": auroc_hi,
        "auprc_ci_low": auprc_lo,
        "auprc_ci_high": auprc_hi,
    }
