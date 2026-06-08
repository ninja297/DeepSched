"""CNN-LSTM-Attention workload prediction module for DeepSched."""

from __future__ import annotations

import torch
from torch import nn


class WPM(nn.Module):
    """Hybrid CNN-LSTM-Attention forecaster.

    The 1-D CNN extracts local utilization patterns, the LSTM models temporal
    dynamics, and multi-head attention lets the model reweight relevant past
    hidden states before forecasting the horizon.
    """

    def __init__(
        self,
        input_size: int = 1,
        cnn_channels: int = 64,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        n_heads: int = 4,
        horizon: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, horizon * input_size)
        self.horizon = horizon
        self.input_size = input_size
        self.attn_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        hidden, _ = self.lstm(conv)
        attended, weights = self.attn(hidden, hidden, hidden)
        self.attn_weights = weights.detach()
        fused = self.norm(hidden + attended)
        pred = self.fc(fused[:, -1, :])
        return pred.view(-1, self.horizon, self.input_size)
