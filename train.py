"""Shared training and evaluation utilities for DeepSched workload predictors."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

_mpl_config_dir = Path(".cache/matplotlib").resolve()
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.lstm_baseline import LSTMBaseline
from models.wpm import WPM


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(split: str, data_dir: Path = Path("data/processed")) -> dict[str, torch.Tensor]:
    path = data_dir / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `python data/preprocess.py` first.")
    return torch.load(path, weights_only=False)


def train_model(
    model: nn.Module,
    model_name: str,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    data_dir: Path = Path("data/processed"),
    models_dir: Path = Path("models"),
) -> float:
    device = get_device()
    tr = load_split("train", data_dir)
    vl = load_split("val", data_dir)

    train_loader = DataLoader(
        TensorDataset(tr["X"], tr["y"]),
        batch_size=batch_size,
        shuffle=True,
    )
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / f"{model_name}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_x = vl["X"].to(device)
            val_y = vl["y"].to(device)
            val_loss = loss_fn(model(val_x), val_y).item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            mean_train = float(np.mean(train_losses))
            print(
                f"{model_name} epoch={epoch:03d} "
                f"train_mse={mean_train:.6f} val_mse={val_loss:.6f}"
            )

    print(f"{model_name} best_val_mse={best_val:.6f}")
    return best_val


def evaluate_model(
    model: nn.Module,
    model_name: str,
    split: str = "test",
    data_dir: Path = Path("data/processed"),
    models_dir: Path = Path("models"),
    results_dir: Path = Path("results"),
) -> dict[str, float | str]:
    device = get_device()
    checkpoint = models_dir / f"{model_name}_best.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model = model.to(device).eval()

    data = load_split(split, data_dir)
    with torch.no_grad():
        pred = model(data["X"].to(device)).cpu().numpy()
    true = data["y"].numpy()

    row: dict[str, float | str] = {
        "model": model_name,
        "split": split,
        "MAE": float(mean_absolute_error(true.reshape(-1), pred.reshape(-1))),
        "RMSE": float(np.sqrt(mean_squared_error(true.reshape(-1), pred.reshape(-1)))),
        "MAPE": float(np.mean(np.abs((true - pred) / (true + 1e-8))) * 100.0),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "wpm_metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        df = df[~((df["model"] == model_name) & (df["split"] == split))]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(metrics_path, index=False)
    print(row)
    return row


def plot_forecast(
    model: nn.Module,
    model_name: str,
    data_dir: Path = Path("data/processed"),
    figures_dir: Path = Path("figures"),
) -> Path:
    device = get_device()
    model = model.to(device).eval()
    data = load_split("test", data_dir)
    with torch.no_grad():
        pred = model(data["X"].to(device)).cpu().numpy()
    true = data["y"].numpy()

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"{model_name.lower()}_forecast.pdf"
    n = min(200, len(true))
    plt.figure(figsize=(10, 3))
    plt.plot(true[:n, 0, 0], label="Actual", linewidth=1.4)
    plt.plot(pred[:n, 0, 0], label="Predicted", linewidth=1.4)
    plt.title(f"{model_name}: 1-step-ahead forecast vs actual")
    plt.xlabel("Test window")
    plt.ylabel("Normalized CPU utilization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return out_path


def plot_attention(
    model: nn.Module,
    model_name: str,
    data_dir: Path = Path("data/processed"),
    figures_dir: Path = Path("figures"),
) -> Path | None:
    if not hasattr(model, "attn_weights"):
        return None

    device = get_device()
    model = model.to(device).eval()
    data = load_split("test", data_dir)
    with torch.no_grad():
        _ = model(data["X"][:1].to(device))
    weights = getattr(model, "attn_weights", None)
    if weights is None:
        return None

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"{model_name.lower()}_attention_weights.pdf"
    plt.figure(figsize=(7, 4))
    plt.imshow(weights[0].cpu().numpy(), aspect="auto", cmap="Blues")
    plt.colorbar(label="Attention weight")
    plt.title(f"{model_name}: attention weights")
    plt.xlabel("Key time step")
    plt.ylabel("Query time step")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    return out_path


def run_lstm_baseline(args: argparse.Namespace) -> None:
    model = LSTMBaseline(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        horizon=args.horizon,
        dropout=args.dropout,
    )
    train_model(
        model,
        "LSTM_baseline",
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    evaluate_model(model, "LSTM_baseline")
    checkpoint = Path("models/LSTM_baseline_best.pt")
    model.load_state_dict(torch.load(checkpoint, map_location=get_device(), weights_only=True))
    plot_forecast(model, "LSTM_baseline")


def run_wpm(args: argparse.Namespace) -> None:
    model = WPM(
        input_size=args.input_size,
        cnn_channels=args.cnn_channels,
        hidden_size=args.hidden_size,
        lstm_layers=args.num_layers,
        n_heads=args.n_heads,
        horizon=args.horizon,
        dropout=args.dropout,
    )
    train_model(
        model,
        "cnn_lstm_attn",
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    evaluate_model(model, "cnn_lstm_attn")
    checkpoint = Path("models/cnn_lstm_attn_best.pt")
    model.load_state_dict(torch.load(checkpoint, map_location=get_device(), weights_only=True))
    plot_forecast(model, "cnn_lstm_attn")
    plot_attention(model, "cnn_lstm_attn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "wpm"], default="lstm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--cnn-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model == "lstm":
        run_lstm_baseline(args)
    elif args.model == "wpm":
        run_wpm(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
