"""Evaluate the saved CNN-LSTM-Attention checkpoint and rebuild its figures."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.wpm import WPM
from train import evaluate_model, get_device, plot_attention, plot_forecast


def main() -> int:
    model = WPM(cnn_channels=32, hidden_size=64)
    checkpoint = "models/CNN_LSTM_Attn_best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=get_device(), weights_only=True))
    evaluate_model(model, "CNN_LSTM_Attn")
    plot_forecast(model, "CNN_LSTM_Attn")
    plot_attention(model, "CNN_LSTM_Attn")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
