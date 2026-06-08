"""Evaluate the saved LSTM baseline checkpoint and rebuild its forecast plot."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.lstm_baseline import LSTMBaseline
from train import evaluate_model, get_device, plot_forecast


def main() -> int:
    model = LSTMBaseline(hidden_size=64)
    checkpoint = "models/LSTM_baseline_best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=get_device(), weights_only=True))
    evaluate_model(model, "LSTM_baseline")
    plot_forecast(model, "LSTM_baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
