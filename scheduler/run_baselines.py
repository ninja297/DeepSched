"""Run C-06 scheduling baselines and the WPM-guided rule advisor."""

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
import torch

from models.wpm import WPM
from scheduler.classical import run_fcfs, run_rr, run_sjf
from scheduler.rule_advisor import run_rule_advisor
from scheduler.simulation import generate_processes
from train import get_device, load_split


def build_forecasts() -> pd.DataFrame:
    device = get_device()
    model = WPM(cnn_channels=32, hidden_size=64)
    model.load_state_dict(
        torch.load(ROOT / "models" / "cnn_lstm_attn_best.pt", map_location=device, weights_only=True)
    )
    model = model.to(device).eval()
    test = load_split("test", ROOT / "data" / "processed")
    with torch.no_grad():
        forecast = model(test["X"].to(device)).cpu().numpy()
    return forecast


def plot_metrics(df: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.sort_values("AWT", ascending=True)
    plt.figure(figsize=(7, 3.8))
    colors = ["#2563EB" if row == "Rule_WPM" else "#94A3B8" for row in plot_df["scheduler"]]
    plt.barh(plot_df["scheduler"], plot_df["AWT"], color=colors)
    plt.xlabel("Average waiting time")
    plt.title("Scheduling Baselines vs Forecast-Aware Rule Advisor")
    plt.tight_layout()
    plt.savefig(figures_dir / "baseline_comparison.pdf", bbox_inches="tight")
    plt.close()


def main() -> int:
    processes = generate_processes(n=500, seed=42)
    forecasts = build_forecasts()
    rows = [
        {"scheduler": "FCFS", **run_fcfs(processes)},
        {"scheduler": "SJF", **run_sjf(processes)},
        {"scheduler": "RR_20", **run_rr(processes, quantum=20.0)},
        {"scheduler": "Rule_WPM", **run_rule_advisor(processes, forecasts, quantum=20.0)},
    ]
    df = pd.DataFrame(rows)
    results_dir = ROOT / "results"
    figures_dir = ROOT / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "scheduling_metrics.csv", index=False)
    plot_metrics(df, figures_dir)
    print(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
