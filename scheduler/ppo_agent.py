"""C-07 PPO agents for the DeepSched scheduling environment."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_mpl_config_dir = ROOT / ".cache" / "matplotlib"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from env.cpu_sched_env import CPUSchedEnv
from models.wpm import WPM
from train import get_device


class RewardLogger(BaseCallback):
    """Log approximate rollout reward during PPO training."""

    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label
        self.rows: list[dict[str, float | str]] = []

    def _on_rollout_end(self) -> None:
        rewards = self.locals.get("rewards")
        if rewards is None:
            return
        self.rows.append(
            {
                "agent": self.label,
                "timesteps": float(self.num_timesteps),
                "mean_step_reward": float(np.mean(rewards)),
            }
        )

    def _on_step(self) -> bool:
        return True


def load_wpm_forecaster() -> Callable[[torch.Tensor], torch.Tensor]:
    model = WPM(cnn_channels=32, hidden_size=64)
    checkpoint = ROOT / "models" / "CNN_LSTM_Attn_best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return model.forward


def make_env(use_forecast: bool, seed: int) -> Callable[[], CPUSchedEnv]:
    forecast_fn = load_wpm_forecaster() if use_forecast else None

    def _factory() -> CPUSchedEnv:
        return CPUSchedEnv(forecast_fn=forecast_fn, seed=seed)

    return _factory


def train_agent(
    label: str,
    use_forecast: bool,
    total_timesteps: int,
    n_envs: int,
    seed: int,
) -> tuple[PPO, RewardLogger]:
    env = make_vec_env(make_env(use_forecast, seed), n_envs=n_envs, seed=seed)
    logger = RewardLogger(label)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=1,
        seed=seed,
        device=device,
    )
    agent.learn(total_timesteps=total_timesteps, callback=logger)
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    agent.save(models_dir / f"{label}_stage1")
    env.close()
    return agent, logger


def evaluate_agent(
    agent: PPO,
    label: str,
    use_forecast: bool,
    seeds: list[int],
) -> dict[str, float | str]:
    rows = []
    for seed in seeds:
        env = CPUSchedEnv(
            forecast_fn=load_wpm_forecaster() if use_forecast else None,
            seed=seed,
        )
        obs, info = env.reset(seed=seed)
        rewards = []
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            rewards.append(float(reward))
            done = terminated or truncated
        rows.append(
            {
                "scheduler": label,
                "seed": seed,
                "mean_reward": float(np.mean(rewards)),
                "total_reward": float(np.sum(rewards)),
                "AWT": float(info["AWT"]),
                "ATT": float(info["ATT"]),
                "throughput": float(info["throughput"]),
                "cpu_utilization": float(info["cpu_utilization"]),
                "energy_proxy": float(info["energy_proxy"]),
                "completed": float(info["completed"]),
                "invalid_actions": float(info["invalid_actions"]),
            }
        )

    df = pd.DataFrame(rows)
    summary: dict[str, float | str] = {"scheduler": label}
    for col in [
        "mean_reward",
        "total_reward",
        "AWT",
        "ATT",
        "throughput",
        "cpu_utilization",
        "energy_proxy",
        "completed",
        "invalid_actions",
    ]:
        summary[col] = float(df[col].mean())
    return summary


def update_scheduling_metrics(rows: list[dict[str, float | str]]) -> None:
    out_path = ROOT / "results" / "scheduling_metrics.csv"
    if out_path.exists():
        df = pd.read_csv(out_path)
        df = df[~df["scheduler"].isin([str(row["scheduler"]) for row in rows])]
    else:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(out_path, index=False)


def save_training_curve(loggers: list[RewardLogger]) -> None:
    rows: list[dict[str, float | str]] = []
    for logger in loggers:
        rows.extend(logger.rows)
    out_path = ROOT / "results" / "ppo_training_curve.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["agent", "timesteps", "mean_step_reward"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[100, 101])
    parser.add_argument("--skip-no-forecast", action="store_true")
    parser.add_argument("--skip-forecast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    agents: list[tuple[str, bool, PPO]] = []
    loggers: list[RewardLogger] = []

    if not args.skip_no_forecast:
        agent, logger = train_agent(
            "PPO_no_forecast",
            use_forecast=False,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
        )
        agents.append(("PPO_no_forecast", False, agent))
        loggers.append(logger)

    if not args.skip_forecast:
        agent, logger = train_agent(
            "PPO_DeepSched",
            use_forecast=True,
            total_timesteps=args.total_timesteps,
            n_envs=args.n_envs,
            seed=args.seed + 1,
        )
        agents.append(("PPO_DeepSched", True, agent))
        loggers.append(logger)

    rows = [
        evaluate_agent(agent, label, use_forecast, args.eval_seeds)
        for label, use_forecast, agent in agents
    ]
    update_scheduling_metrics(rows)
    save_training_curve(loggers)
    print(pd.DataFrame(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
