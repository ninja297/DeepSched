"""C-04.5 forecast quality analysis.

This evaluates forecasting behavior beyond aggregate MAE/RMSE:

- error as a function of prediction horizon
- burst-event detection accuracy
- peak-event detection accuracy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_mpl_config_dir = ROOT / ".cache" / "matplotlib"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.lstm_baseline import LSTMBaseline
from models.wpm import WPM
from train import get_device, load_split


def load_models() -> dict[str, torch.nn.Module]:
    device = get_device()
    models: dict[str, torch.nn.Module] = {
        "LSTM_baseline": LSTMBaseline(hidden_size=64),
        "CNN_LSTM_Attn": WPM(cnn_channels=32, hidden_size=64),
    }
    checkpoints = {
        "LSTM_baseline": ROOT / "models" / "LSTM_baseline_best.pt",
        "CNN_LSTM_Attn": ROOT / "models" / "CNN_LSTM_Attn_best.pt",
    }
    for name, model in models.items():
        model.load_state_dict(
            torch.load(checkpoints[name], map_location=device, weights_only=True)
        )
        models[name] = model.to(device).eval()
    return models


def predict(model: torch.nn.Module) -> tuple[np.ndarray, np.ndarray]:
    data = load_split("test", ROOT / "data" / "processed")
    device = get_device()
    with torch.no_grad():
        pred = model(data["X"].to(device)).cpu().numpy()
    return data["y"].numpy(), pred


def horizon_errors(model_name: str, true: np.ndarray, pred: np.ndarray) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    horizon = true.shape[1]
    for step in range(horizon):
        err = true[:, step, :] - pred[:, step, :]
        rows.append(
            {
                "model": model_name,
                "horizon_step": step + 1,
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err**2))),
            }
        )
    return rows


def event_metrics(
    model_name: str,
    true: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    event_name: str,
) -> dict[str, float | str]:
    true_event = true.max(axis=1).reshape(-1) >= threshold
    pred_event = pred.max(axis=1).reshape(-1) >= threshold
    tp = float(np.sum(true_event & pred_event))
    fp = float(np.sum(~true_event & pred_event))
    fn = float(np.sum(true_event & ~pred_event))
    tn = float(np.sum(~true_event & ~pred_event))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "model": model_name,
        "event": event_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "support": float(np.sum(true_event)),
    }


def plot_horizon_errors(df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 3.5))
    for model_name, group in df.groupby("model"):
        plt.plot(group["horizon_step"], group["RMSE"], marker="o", label=model_name)
    plt.xlabel("Forecast horizon step")
    plt.ylabel("RMSE")
    plt.title("Forecast Error vs Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "forecast_error_vs_horizon.pdf", bbox_inches="tight")
    plt.close()


def plot_event_metrics(df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    event_df = df[df["event"].isin(["burst", "peak"])].copy()
    labels = [f"{row.model}\n{row.event}" for row in event_df.itertuples()]
    x = np.arange(len(event_df))
    width = 0.25
    plt.figure(figsize=(8, 3.8))
    plt.bar(x - width, event_df["precision"], width, label="Precision")
    plt.bar(x, event_df["recall"], width, label="Recall")
    plt.bar(x + width, event_df["f1"], width, label="F1")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Burst and Peak Detection Quality")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "burst_peak_detection.pdf", bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--burst-threshold", type=float, default=0.75)
    parser.add_argument("--peak-threshold", type=float, default=0.90)
    args = parser.parse_args()

    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)

    horizon_rows = []
    event_rows = []
    for model_name, model in load_models().items():
        true, pred = predict(model)
        horizon_rows.extend(horizon_errors(model_name, true, pred))
        event_rows.append(
            event_metrics(model_name, true, pred, args.burst_threshold, "burst")
        )
        event_rows.append(
            event_metrics(model_name, true, pred, args.peak_threshold, "peak")
        )

    horizon_df = pd.DataFrame(horizon_rows)
    event_df = pd.DataFrame(event_rows)
    horizon_df.to_csv(results_dir / "forecast_horizon_metrics.csv", index=False)
    event_df.to_csv(results_dir / "forecast_event_metrics.csv", index=False)
    plot_horizon_errors(horizon_df, figures_dir)
    plot_event_metrics(event_df, figures_dir)
    print(horizon_df)
    print(event_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
