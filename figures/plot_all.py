"""C-08 rebuild paper summary figures from saved result CSV files."""

from __future__ import annotations

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


FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def save(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name, bbox_inches="tight")
    plt.close()


def plot_ablation_bar() -> None:
    df = pd.read_csv(RESULTS / "wpm_metrics.csv")
    arch = df[df["model"].isin(["LSTM_only", "CNN_LSTM", "LSTM_Attention", "CNN_LSTM_Attn"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, metric in zip(axes, ["MAE", "RMSE", "MAPE"]):
        ax.bar(arch["model"], arch[metric], color="#4C78A8")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    save("ablation_bar.pdf")


def plot_horizon_line() -> None:
    df = pd.read_csv(RESULTS / "wpm_metrics.csv")
    h = df[df["model"].str.match(r"WPM_H\d+")].copy()
    h["H"] = h["model"].str.extract(r"H(\d+)").astype(int)
    h = h.sort_values("H")
    plt.figure(figsize=(5.5, 3.4))
    plt.plot(h["H"], h["RMSE"], marker="o")
    plt.xlabel("Prediction horizon H")
    plt.ylabel("RMSE")
    plt.title("WPM RMSE vs Forecast Horizon")
    save("horizon_line.pdf")


def plot_forecast_quality() -> None:
    df = pd.read_csv(RESULTS / "forecast_horizon_metrics.csv")
    plt.figure(figsize=(7, 3.5))
    for model, group in df.groupby("model"):
        plt.plot(group["horizon_step"], group["RMSE"], marker="o", label=model)
    plt.xlabel("Forecast horizon step")
    plt.ylabel("RMSE")
    plt.title("Forecast Error vs Horizon")
    plt.legend()
    save("forecast_error_vs_horizon.pdf")


def plot_event_quality() -> None:
    df = pd.read_csv(RESULTS / "forecast_event_metrics.csv")
    labels = [f"{row.model}\n{row.event}" for row in df.itertuples()]
    x = range(len(df))
    width = 0.25
    plt.figure(figsize=(8, 3.8))
    plt.bar([i - width for i in x], df["precision"], width, label="Precision")
    plt.bar(list(x), df["recall"], width, label="Recall")
    plt.bar([i + width for i in x], df["f1"], width, label="F1")
    plt.xticks(list(x), labels, rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Burst and Peak Detection Quality")
    plt.legend()
    save("burst_peak_detection.pdf")


def plot_scheduling() -> None:
    path = RESULTS / "ppo_metrics.csv"
    if path.exists():
        df = pd.read_csv(path).dropna(subset=["AWT"])
    else:
        df = pd.read_csv(RESULTS / "scheduling_metrics.csv").dropna(subset=["AWT"])
    plot_df = df.sort_values("AWT", ascending=True)
    plt.figure(figsize=(8, 4.2))
    colors = [
        "#2563EB" if "Rule" in s else "#DC2626" if "PPO" in s else "#94A3B8"
        for s in plot_df["scheduler"]
    ]
    plt.barh(plot_df["scheduler"], plot_df["AWT"], color=colors)
    plt.xlabel("Average waiting time")
    plt.title("Scheduling Comparison")
    save("ppo_vs_baselines.pdf")


def plot_ppo_curve() -> None:
    path = RESULTS / "ppo_training_curve.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    plt.figure(figsize=(7, 3.4))
    for agent, group in df.groupby("agent"):
        plt.plot(group["timesteps"], group["mean_step_reward"], marker="o", label=agent)
    plt.xlabel("Training timesteps")
    plt.ylabel("Mean rollout step reward")
    plt.title("PPO Stage 1 Training Curve")
    plt.legend()
    save("ppo_reward_curve.pdf")


def main() -> int:
    plot_ablation_bar()
    plot_horizon_line()
    plot_forecast_quality()
    plot_event_quality()
    plot_scheduling()
    plot_ppo_curve()
    print("C-08 figures rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
