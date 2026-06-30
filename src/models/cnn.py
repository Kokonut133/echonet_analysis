from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
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
    """
    Stem → ResBlock(32→64) → ResBlock(64→128) → ResBlock(128→256) → ResBlock(256→256)
    → AdaptiveAvgPool1d(1) → [cat demo if n_demo_features > 0] → Linear head → n_labels logits
    in: (batch, n_leads, 2500)  out: (batch, n_labels)
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
        x = self.stem(waveforms)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).squeeze(-1)

        if self.n_demo_features > 0 and demo is not None and demo.shape[1] > 0:
            x = torch.cat([x, demo], dim=1)

        return self.head(x)
