"""Gradient-based interpretability tools for the ECGConvNet waveform CNN.

All functions are pure: torch model + torch tensor(s) in, numpy array(s) out.
No printing, no I/O. Intended to be driven from scripts/8_interpretability/.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models import ECGConvNet


def _to_batch(waveforms: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Add a batch dim if `waveforms` is a single (n_leads, T) record."""
    if waveforms.dim() == 2:
        return waveforms.unsqueeze(0), True
    return waveforms, False


def input_gradient_saliency(
    model: ECGConvNet,
    waveforms: torch.Tensor,
    label_idx: int,
    n_smooth: int = 20,
    noise_frac: float = 0.1,
    device: torch.device | None = None,
) -> np.ndarray:
    """SmoothGrad input-gradient saliency: |d logit[label_idx] / d input|, averaged
    over `n_smooth` noisy copies of the input (Gaussian noise, std = noise_frac *
    per-record signal std) to reduce speckle.

    Args:
        model: trained ECGConvNet (eval mode is enforced internally).
        waveforms: (n_leads, T) or (B, n_leads, T) float tensor, raw amplitude.
        label_idx: index into the model's output logits.
        n_smooth: number of noisy copies averaged (SmoothGrad).
        noise_frac: Gaussian noise std as a fraction of each record's own std.
        device: device to run on; defaults to the model's current device.

    Returns:
        np.ndarray of shape (n_leads, T) if input was unbatched, else (B, n_leads, T).
        Non-negative (absolute gradient magnitude).
    """
    x, single = _to_batch(waveforms)
    device = device or next(model.parameters()).device
    model = model.to(device)
    model.eval()
    x = x.to(device).float()

    # per-record scalar std -> noise scale
    sigma = noise_frac * x.reshape(x.shape[0], -1).std(dim=1).clamp(min=1e-8)
    sigma = sigma.view(-1, 1, 1)

    grad_accum = torch.zeros_like(x)
    for _ in range(n_smooth):
        noise = torch.randn_like(x) * sigma
        xi = (x + noise).detach().clone().requires_grad_(True)
        logits = model(xi)
        target = logits[:, label_idx].sum()
        (grad,) = torch.autograd.grad(target, xi)
        grad_accum += grad.abs()

    saliency = (grad_accum / n_smooth).detach().cpu().numpy()
    if single:
        saliency = saliency[0]
    return saliency


def grad_cam_1d(
    model: ECGConvNet,
    waveforms: torch.Tensor,
    label_idx: int,
    target_layer: nn.Module | None = None,
) -> np.ndarray:
    """Grad-CAM on a 1D conv feature map, upsampled to the input length.

    Produces a single importance curve shared across leads (Grad-CAM pools over
    channels, and the target layer's activations already mix all 12 input leads).

    Args:
        model: trained ECGConvNet.
        waveforms: (n_leads, T) or (B, n_leads, T) float tensor.
        label_idx: index into the model's output logits.
        target_layer: module to hook (defaults to `model.stage4`, the last ResBlock).

    Returns:
        np.ndarray of shape (T,) if input was unbatched, else (B, T). Non-negative
        (ReLU'd), upsampled by linear interpolation to match the input time length.
    """
    x, single = _to_batch(waveforms)
    device = next(model.parameters()).device
    model.eval()
    x = x.to(device).float()

    if target_layer is None:
        target_layer = model.stage4

    activations: dict[str, torch.Tensor] = {}

    def _hook(_module: nn.Module, _inp, out: torch.Tensor) -> None:
        activations["value"] = out

    handle = target_layer.register_forward_hook(_hook)
    try:
        logits = model(x)
    finally:
        handle.remove()

    target = logits[:, label_idx].sum()
    acts = activations["value"]  # (B, C, L)
    (grads,) = torch.autograd.grad(target, acts, retain_graph=False)

    weights = grads.mean(dim=2, keepdim=True)  # global-avg-pool over time -> (B, C, 1)
    cam = torch.relu((weights * acts).sum(dim=1))  # (B, L)

    cam_max = cam.amax(dim=1, keepdim=True).clamp(min=1e-8)
    cam = cam / cam_max

    T = x.shape[-1]
    cam_up = torch.nn.functional.interpolate(
        cam.unsqueeze(1), size=T, mode="linear", align_corners=False
    ).squeeze(1)

    result = cam_up.detach().cpu().numpy()
    if single:
        result = result[0]
    return result


def lead_importance(saliency: np.ndarray) -> np.ndarray:
    """Normalised per-lead share of total saliency.

    Args:
        saliency: (n_leads, T) array (e.g. from `input_gradient_saliency`), or
            (B, n_leads, T) in which case it is averaged over the batch first.

    Returns:
        np.ndarray of shape (n_leads,) that sums to 1 (uniform if saliency is all-zero).
    """
    sal = np.asarray(saliency)
    if sal.ndim == 3:
        sal = sal.mean(axis=0)

    per_lead = sal.sum(axis=1)  # (n_leads,)
    total = per_lead.sum()
    if total <= 0:
        return np.full(per_lead.shape, 1.0 / per_lead.shape[0], dtype=np.float64)
    return per_lead / total
