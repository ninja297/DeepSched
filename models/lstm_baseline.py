"""Baseline LSTM workload predictor for DeepSched."""

from __future__ import annotations

import torch
from torch import nn


class LSTMBaseline(nn.Module):
    """Many-to-many CPU utilization forecaster.

    Input shape:
        (batch, lookback, features)

    Output shape:
        (batch, horizon, features)
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        horizon: int = 10,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(hidden_size, horizon * input_size)
        self.horizon = horizon
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :])
        return pred.view(-1, self.horizon, self.input_size)
