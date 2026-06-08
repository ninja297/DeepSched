"""C-01 data preprocessing for DeepSched.

Input:
    data/raw/alibaba/machine_usage_days_1_to_8_grouped_300_seconds.csv
    or data/raw/google/task_usage/*.csv.gz

Output:
    data/processed/train.pt
    data/processed/val.pt
    data/processed/test.pt
    data/processed/scaler.pkl
    figures/data_eda.pdf
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib

_mpl_config_dir = Path(".cache/matplotlib").resolve()
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


GOOGLE_TASK_USAGE_COLUMNS = [
    "start_time",
    "end_time",
    "job_id",
    "task_index",
    "machine_id",
    "cpu_rate",
    "canonical_memory_usage",
    "assigned_memory_usage",
    "unmapped_page_cache",
    "total_page_cache",
    "maximum_memory_usage",
    "disk_io_time",
    "local_disk_space_usage",
    "maximum_cpu_rate",
    "maximum_disk_io_time",
    "cycles_per_instruction",
    "memory_accesses_per_instruction",
    "sample_portion",
    "aggregation_type",
    "sampled_cpu_usage",
]


def load_google_cpu_usage(raw_dir: Path, max_rows: int | None) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv.gz")) + sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No Google task_usage shards found in {raw_dir}. "
            "Run `python scripts/download_data.py` first."
        )

    chunks: list[pd.DataFrame] = []
    remaining = max_rows
    for path in files:
        use_rows = remaining if remaining is not None else None
        df = pd.read_csv(
            path,
            header=None,
            names=GOOGLE_TASK_USAGE_COLUMNS,
            usecols=["start_time", "machine_id", "cpu_rate"],
            nrows=use_rows,
        ).dropna()
        chunks.append(df)
        if remaining is not None:
            remaining -= len(df)
            if remaining <= 0:
                break

    usage = pd.concat(chunks, ignore_index=True)
    usage["time"] = pd.to_datetime(usage["start_time"], unit="us")
    usage["machine_id"] = usage["machine_id"].astype("int64")
    usage["cpu_rate"] = usage["cpu_rate"].clip(lower=0.0)
    return usage[["time", "machine_id", "cpu_rate"]]


def load_alibaba_series(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Alibaba sample not found at {path}. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "cpu_util_percent" not in df.columns:
        raise ValueError(f"Expected cpu_util_percent in {path}; found {list(df.columns)}")
    cpu = df["cpu_util_percent"].astype("float32").clip(0.0, 100.0) / 100.0
    return pd.DataFrame({"cpu_util": cpu})


def build_machine_series(usage: pd.DataFrame, top_machines: int) -> pd.DataFrame:
    active = (
        usage.groupby("machine_id")["cpu_rate"]
        .count()
        .sort_values(ascending=False)
        .head(top_machines)
        .index
    )
    usage = usage[usage["machine_id"].isin(active)]
    per_machine = (
        usage.groupby(["time", "machine_id"], as_index=False)["cpu_rate"]
        .sum()
        .assign(cpu_rate=lambda df: df["cpu_rate"].clip(0.0, 1.0))
    )
    pivot = per_machine.pivot(index="time", columns="machine_id", values="cpu_rate")
    pivot = pivot.sort_index().resample("5min").mean().ffill().bfill()
    return pivot.dropna(axis=1, how="all")


def make_windows(series: np.ndarray, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = [], []
    for i in range(len(series) - lookback - horizon + 1):
        x_values.append(series[i : i + lookback])
        y_values.append(series[i + lookback : i + lookback + horizon])
    if not x_values:
        raise ValueError(
            f"Not enough time steps ({len(series)}) for lookback={lookback}, horizon={horizon}."
        )
    return np.asarray(x_values, dtype=np.float32), np.asarray(y_values, dtype=np.float32)


def save_splits(x_values: np.ndarray, y_values: np.ndarray, output_dir: Path) -> None:
    n = len(x_values)
    splits = {
        "train": (0, int(0.70 * n)),
        "val": (int(0.70 * n), int(0.85 * n)),
        "test": (int(0.85 * n), n),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (start, end) in splits.items():
        torch.save(
            {
                "X": torch.from_numpy(x_values[start:end]),
                "y": torch.from_numpy(y_values[start:end]),
            },
            output_dir / f"{name}.pt",
        )


def plot_eda(series: np.ndarray, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sample = series[: min(500, len(series)), 0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].plot(sample, linewidth=1.0)
    axes[0].set_title("CPU Trace")
    axes[0].set_xlabel("5-min step")
    axes[0].set_ylabel("Normalized utilization")

    axes[1].hist(series[:, 0], bins=30, color="#4C78A8", alpha=0.85)
    axes[1].set_title("Utilization Histogram")
    axes[1].set_xlabel("Normalized utilization")

    centered = sample - sample.mean()
    denom = np.dot(centered, centered)
    lags = np.arange(1, min(31, len(sample)))
    acf = [np.dot(centered[:-lag], centered[lag:]) / denom if denom else 0.0 for lag in lags]
    axes[2].bar(lags, acf, color="#59A14F")
    axes[2].set_title("Autocorrelation")
    axes[2].set_xlabel("Lag")

    fig.tight_layout()
    fig.savefig(figures_dir / "data_eda.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["auto", "alibaba", "google"], default="auto")
    parser.add_argument(
        "--alibaba-path",
        type=Path,
        default=Path("data/raw/alibaba/machine_usage_days_1_to_8_grouped_300_seconds.csv"),
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/google/task_usage"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--top-machines", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=250_000)
    args = parser.parse_args()

    use_alibaba = args.source == "alibaba" or (
        args.source == "auto" and args.alibaba_path.exists()
    )
    if use_alibaba:
        pivot = load_alibaba_series(args.alibaba_path)
        source = "alibaba_2018_machine_usage_zenodo_300s"
    else:
        usage = load_google_cpu_usage(args.raw_dir, args.max_rows)
        pivot = build_machine_series(usage, args.top_machines)
        source = "google_clusterdata_2011_2_task_usage"

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(pivot.to_numpy(dtype=np.float32))

    x_values, y_values = make_windows(scaled, args.lookback, args.horizon)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "series_columns": [str(c) for c in pivot.columns],
            "lookback": args.lookback,
            "horizon": args.horizon,
            "source": source,
        },
        args.output_dir / "scaler.pkl",
    )
    save_splits(x_values, y_values, args.output_dir)
    plot_eda(scaled, args.figures_dir)

    print("Saved train / val / test tensors.")
    print(f"Pivot shape: {pivot.shape}")
    print(f"X shape: {x_values.shape}; y shape: {y_values.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
