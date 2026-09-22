from __future__ import annotations

import numpy as np
import pytest
import torch

from src.interpretability import grad_cam_1d, input_gradient_saliency, lead_importance
from src.models import ECGConvNet


@pytest.fixture()
def tiny_model() -> ECGConvNet:
    torch.manual_seed(0)
    model = ECGConvNet(n_leads=12, n_labels=12, n_demo_features=0)
    model.eval()
    return model


@pytest.fixture()
def random_waveform() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(12, 2500)


def test_saliency_shape_and_nonnegative(tiny_model, random_waveform):
    saliency = input_gradient_saliency(tiny_model, random_waveform, label_idx=0, n_smooth=3)
    assert saliency.shape == (12, 2500)
    assert isinstance(saliency, np.ndarray)
    assert np.all(saliency >= 0)


def test_saliency_batched(tiny_model):
    torch.manual_seed(2)
    batch = torch.randn(4, 12, 2500)
    saliency = input_gradient_saliency(tiny_model, batch, label_idx=1, n_smooth=2)
    assert saliency.shape == (4, 12, 2500)
    assert np.all(saliency >= 0)


def test_gradcam_shape_and_nonnegative(tiny_model, random_waveform):
    cam = grad_cam_1d(tiny_model, random_waveform, label_idx=0)
    assert cam.shape == (2500,)
    assert isinstance(cam, np.ndarray)
    assert np.all(cam >= 0)


def test_gradcam_batched(tiny_model):
    torch.manual_seed(3)
    batch = torch.randn(3, 12, 2500)
    cam = grad_cam_1d(tiny_model, batch, label_idx=2)
    assert cam.shape == (3, 2500)


def test_lead_importance_sums_to_one(tiny_model, random_waveform):
    saliency = input_gradient_saliency(tiny_model, random_waveform, label_idx=0, n_smooth=2)
    shares = lead_importance(saliency)
    assert shares.shape == (12,)
    assert np.all(shares >= 0)
    assert np.isclose(shares.sum(), 1.0, atol=1e-6)


def test_lead_importance_batched_input():
    rng = np.random.default_rng(0)
    saliency = rng.random((5, 12, 2500))
    shares = lead_importance(saliency)
    assert shares.shape == (12,)
    assert np.isclose(shares.sum(), 1.0, atol=1e-6)


def test_lead_importance_zero_saliency():
    shares = lead_importance(np.zeros((12, 2500)))
    assert shares.shape == (12,)
    assert np.isclose(shares.sum(), 1.0, atol=1e-6)
    assert np.allclose(shares, 1.0 / 12)
