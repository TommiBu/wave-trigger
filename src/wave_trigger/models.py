from __future__ import annotations

import torch
import torch.nn as nn


class WaveMLP(nn.Module):
    """Flattened window. Smallest baseline, no notion of time."""

    def __init__(self, n_steps: int, n_features: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_steps * n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class WaveCNN(nn.Module):
    """1D convolution over time. Default; best accuracy per parameter here."""

    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)


class WaveGRU(nn.Module):
    """Recurrent variant. Handles variable length; overfits easily on small sets."""

    def __init__(self, n_features: int, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


def build_model(arch: str, n_steps: int, n_features: int) -> nn.Module:
    if arch == "mlp":
        return WaveMLP(n_steps, n_features)
    if arch == "cnn":
        return WaveCNN(n_features)
    if arch == "gru":
        return WaveGRU(n_features)
    raise ValueError(f"unknown arch: {arch!r} (mlp | cnn | gru)")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
