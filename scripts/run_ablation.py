"""Run C-04 WPM ablation experiments."""

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
import pandas as pd

from models.ablations import CNNLSTM, LSTMAttention
from models.lstm_baseline import LSTMBaseline
from models.wpm import WPM
from train import evaluate_model, plot_attention, plot_forecast, train_model


def architecture_experiments(args: argparse.Namespace) -> list[str]:
    experiments = {
        "LSTM_only": LSTMBaseline(hidden_size=args.hidden_size, horizon=10),
        "CNN_LSTM": CNNLSTM(
            cnn_channels=args.cnn_channels,
            hidden_size=args.hidden_size,
            horizon=10,
        ),
        "LSTM_Attention": LSTMAttention(hidden_size=args.hidden_size, horizon=10),
        "CNN_LSTM_Attn": WPM(
            cnn_channels=args.cnn_channels,
            hidden_size=args.hidden_size,
            horizon=10,
        ),
    }
    completed = []
    for name, model in experiments.items():
        train_model(model, name, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        evaluate_model(model, name)
        plot_forecast(model, name)
        plot_attention(model, name)
        completed.append(name)
    return completed


def horizon_experiments(args: argparse.Namespace) -> list[str]:
    completed = []
    for horizon in [1, 5, 10]:
        name = f"WPM_H{horizon}"
        model = WPM(
            cnn_channels=args.cnn_channels,
            hidden_size=args.hidden_size,
            horizon=horizon,
        )
        train_model(model, name, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
        evaluate_model(model, name)
        completed.append(name)
    return completed


def plot_ablation_summary(results_path: Path, figures_dir: Path) -> None:
    df = pd.read_csv(results_path)
    arch = df[df["model"].isin(["LSTM_only", "CNN_LSTM", "LSTM_Attention", "CNN_LSTM_Attn"])]
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not arch.empty:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
            ax.bar(arch["model"], arch[metric], color="#4C78A8")
            ax.set_title(metric)
            ax.tick_params(axis="x", rotation=25, labelsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "ablation_bar.pdf", bbox_inches="tight")
        plt.close(fig)

    horizon = df[df["model"].str.match(r"WPM_H\d+")].copy()
    if not horizon.empty:
        horizon["H"] = horizon["model"].str.extract(r"H(\d+)").astype(int)
        horizon = horizon.sort_values("H")
        plt.figure(figsize=(5.5, 3.4))
        plt.plot(horizon["H"], horizon["RMSE"], marker="o")
        plt.xlabel("Prediction horizon H")
        plt.ylabel("RMSE")
        plt.title("WPM RMSE vs Forecast Horizon")
        plt.tight_layout()
        plt.savefig(figures_dir / "horizon_line.pdf", bbox_inches="tight")
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--cnn-channels", type=int, default=32)
    parser.add_argument("--skip-architecture", action="store_true")
    parser.add_argument("--skip-horizon", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    completed = []
    if not args.skip_architecture:
        completed.extend(architecture_experiments(args))
    if not args.skip_horizon:
        completed.extend(horizon_experiments(args))
    plot_ablation_summary(ROOT / "results" / "wpm_metrics.csv", ROOT / "figures")
    print("Completed ablations:", ", ".join(completed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
