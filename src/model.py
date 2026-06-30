"""1D CNN model for multi-label ECG classification.

The backbone processes 12-lead waveforms with stacked residual Conv1d blocks,
compressing (batch, 12, 2500) down to a 256-dim embedding via adaptive average pooling.
Demographic features are concatenated after the backbone when n_demo_features > 0.
The head maps the joint embedding to n_labels independent binary logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Two-layer residual block with optional downsampling shortcut.

    Uses stride on the first conv to halve the time dimension when stride > 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ECGConvNet(nn.Module):
    """Multi-label 1D CNN for 12-lead ECG classification.

    Architecture:
      Stem   : Conv1d(12 → 32, k=7, stride=2) + BN + ReLU
      Stage 1: ResBlock(32 → 64,  stride=2)
      Stage 2: ResBlock(64 → 128, stride=2)
      Stage 3: ResBlock(128 → 256, stride=2)
      Stage 4: ResBlock(256 → 256, stride=2)
      Pool   : AdaptiveAvgPool1d(1)  → (batch, 256)
      [if n_demo_features > 0: concat → (batch, 256 + n_demo_features)]
      Head   : Linear → ReLU → Dropout → Linear → n_labels logits

    Args:
        n_leads: number of ECG leads (channels in the input waveform).
        n_labels: number of binary output labels.
        n_demo_features: size of the demographic feature vector; 0 disables the demo path.
        dropout: dropout probability applied in the classification head.
    """

    def __init__(
        self,
        n_leads: int = 12,
        n_labels: int = 12,
        n_demo_features: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_demo_features = n_demo_features

        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.stage1 = ResBlock(32, 64, stride=2)
        self.stage2 = ResBlock(64, 128, stride=2)
        self.stage3 = ResBlock(128, 256, stride=2)
        self.stage4 = ResBlock(256, 256, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

        backbone_dim = 256 + n_demo_features
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_labels),
        )

    def forward(self, waveforms: torch.Tensor, demo: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            waveforms: (batch, n_leads, time)
            demo: (batch, n_demo_features) or None / empty tensor

        Returns:
            logits: (batch, n_labels) — raw scores before sigmoid
        """
        x = self.stem(waveforms)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).squeeze(-1)  # (batch, 256)

        if self.n_demo_features > 0 and demo is not None and demo.shape[1] > 0:
            x = torch.cat([x, demo], dim=1)

        return self.head(x)
