"""Waveform normalisation and augmentation for the ECG CNN v2 pipeline.

All functions operate on a single record of shape (n_leads, n_samples) —
e.g. (12, 2500) — accepted as either a numpy array or a torch tensor, and
return the same type/shape they were given. None of them mutate the input
in place.

Randomness is injected explicitly via a `numpy.random.Generator` (the `rng`
keyword). Every function is deterministic given the same input and the same
Generator *state* (i.e. seed a fresh `np.random.default_rng(seed)` before
each call to reproduce a result). `rng=None` falls back to a freshly
seeded, OS-entropy-backed generator, which is intentionally non-deterministic
and safe to use across forked DataLoader worker processes.

Motivation (see progress_log.md, step 1 / step 5): raw waveform amplitude
differs by roughly 10x between the `no_split`/train files and the val/test
files. `standardize_per_record` removes this per-record scale factor with a
*single scalar* divisor (global std of the lead-mean-centered record) so
that the relative amplitude ratios between leads — which carry clinical
signal — are preserved. It should be applied to every record, train and
eval alike; the other functions are train-time-only augmentations.
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dependency in this repo
    torch = None  # type: ignore


def _is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _to_numpy(x):
    """Return (numpy_array float32, was_torch, device, dtype)."""
    if _is_torch(x):
        return x.detach().cpu().numpy().astype(np.float32), True, x.device, x.dtype
    return np.asarray(x, dtype=np.float32), False, None, None


def _from_numpy(arr: np.ndarray, was_torch: bool, device, dtype):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if was_torch:
        t = torch.from_numpy(arr)
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        return t
    return arr


def _rng_or_default(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def standardize_per_record(
    x, rng: np.random.Generator | None = None, eps: float = 1e-8
):
    """Subtract per-lead mean, divide by a single per-record global std.

    Preserves inter-lead amplitude ratios (unlike per-lead standardisation,
    which would erase them). `rng` is accepted-but-unused so this function
    has the same call signature as the augmentation functions and can be
    used inside `compose`.
    """
    arr, was_torch, device, dtype = _to_numpy(x)
    lead_mean = arr.mean(axis=-1, keepdims=True)  # (n_leads, 1)
    centered = arr - lead_mean
    global_std = float(centered.std())
    global_std = max(global_std, eps)
    out = centered / global_std
    return _from_numpy(out, was_torch, device, dtype)


def random_time_shift(
    x,
    rng: np.random.Generator | None = None,
    max_shift: int = 250,
    mode: str = "circular",
):
    """Shift the record along the time axis by a random offset in
    [-max_shift, max_shift]. `mode="circular"` wraps around; any other mode
    zero-pads the vacated region.
    """
    rng = _rng_or_default(rng)
    arr, was_torch, device, dtype = _to_numpy(x)
    n = arr.shape[-1]
    shift = int(rng.integers(-max_shift, max_shift + 1))

    if shift == 0:
        out = arr.copy()
    elif mode == "circular":
        out = np.roll(arr, shift, axis=-1)
    else:
        out = np.zeros_like(arr)
        if shift > 0:
            out[..., shift:] = arr[..., : n - shift]
        else:
            out[..., : n + shift] = arr[..., -shift:]

    return _from_numpy(out, was_torch, device, dtype)


def random_amplitude_scale(
    x,
    rng: np.random.Generator | None = None,
    low: float = 0.8,
    high: float = 1.2,
):
    """Multiply the whole record by a single random scalar in [low, high]."""
    rng = _rng_or_default(rng)
    arr, was_torch, device, dtype = _to_numpy(x)
    scale = float(rng.uniform(low, high))
    out = arr * scale
    return _from_numpy(out, was_torch, device, dtype)


def random_baseline_wander(
    x,
    rng: np.random.Generator | None = None,
    max_amp: float = 0.1,
    fs: float = 250,
    freq_range: Tuple[float, float] = (0.1, 0.5),
):
    """Add a low-frequency sinusoid (respiration-like baseline wander).

    Each lead gets an independent random frequency, phase and amplitude
    (amplitude drawn in [0, max_amp]) so leads wander somewhat independently,
    as they do in real recordings.
    """
    rng = _rng_or_default(rng)
    arr, was_torch, device, dtype = _to_numpy(x)
    n_leads, n_samples = arr.shape
    t = np.arange(n_samples, dtype=np.float32) / fs

    freqs = rng.uniform(freq_range[0], freq_range[1], size=n_leads)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_leads)
    amps = rng.uniform(0.0, max_amp, size=n_leads)

    wander = amps[:, None] * np.sin(
        2 * np.pi * freqs[:, None] * t[None, :] + phases[:, None]
    )
    out = arr + wander.astype(np.float32)
    return _from_numpy(out, was_torch, device, dtype)


def random_gaussian_noise(
    x, rng: np.random.Generator | None = None, std: float = 0.02
):
    """Add i.i.d. Gaussian noise with the given std to every sample."""
    rng = _rng_or_default(rng)
    arr, was_torch, device, dtype = _to_numpy(x)
    noise = rng.normal(0.0, std, size=arr.shape).astype(np.float32)
    out = arr + noise
    return _from_numpy(out, was_torch, device, dtype)


def compose(*fns: Callable) -> Callable:
    """Chain transform functions, each called as `fn(x, rng=rng)`.

    A single Generator is created (if none is supplied to the composed
    call) and threaded through every function so the whole pipeline is
    reproducible from one seed.
    """

    def composed(x, rng: np.random.Generator | None = None):
        rng = _rng_or_default(rng)
        for fn in fns:
            x = fn(x, rng=rng)
        return x

    return composed


# Standardise first (removes the ~10x raw-amplitude discrepancy between
# files), then augment with absolute magnitudes that are meaningful
# relative to a unit-std signal.
train_augment: Callable = compose(
    standardize_per_record,
    partial(random_time_shift, max_shift=250, mode="circular"),
    partial(random_amplitude_scale, low=0.8, high=1.2),
    partial(random_baseline_wander, max_amp=0.1, fs=250),
    partial(random_gaussian_noise, std=0.02),
)


def eval_transform(x, rng: np.random.Generator | None = None):
    """Same normalisation as train_augment, minus the random augmentation."""
    return standardize_per_record(x, rng=rng)
