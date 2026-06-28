"""Evaluate C-07 RL policies and environment baselines on CPUSchedEnv."""

from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_mpl_config_dir = ROOT / ".cache" / "matplotlib"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from env.cpu_sched_env import CPUSchedEnv
from scheduler.confidence import should_trust_forecast
from scheduler.ppo_agent import load_wpm_forecaster


def valid_queue_len(obs: np.ndarray, max_queue: int) -> int:
    bursts = obs[:max_queue]
    waits = obs[max_queue : 2 * max_queue]
    return int(np.count_nonzero((bursts > 0) | (waits > 0)))


def fcfs_policy(obs: np.ndarray, max_queue: int) -> int:
    return 0


def sjf_policy(obs: np.ndarray, max_queue: int) -> int:
    n = valid_queue_len(obs, max_queue)
    if n == 0:
        return 0
    bursts = obs[:n]
    return int(np.argmin(bursts))


def rule_wpm_policy(obs: np.ndarray, max_queue: int) -> int:
    n = valid_queue_len(obs, max_queue)
    if n == 0:
        return 0
    forecast = obs[2 * max_queue : 3 * max_queue]
    if not should_trust_forecast(forecast):
        return 0
    u = float(np.mean(forecast))
    slope = float(forecast[-1] - forecast[0])
    if u >= 0.35 or slope >= 0.01:
        return sjf_policy(obs, max_queue)
    return 0


def run_policy(label: str, seeds: list[int], policy_fn, use_forecast: bool = False) -> dict[str, float | str]:
    rows = []
    forecast_fn = load_wpm_forecaster() if use_forecast else None
    for seed in seeds:
        env = CPUSchedEnv(forecast_fn=forecast_fn, seed=seed)
        obs, info = env.reset(seed=seed)
        rewards = []
        done = False
        while not done:
            action = policy_fn(obs, env.max_queue)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(float(reward))
            done = terminated or truncated
        rows.append({**info, "mean_reward": float(np.mean(rewards)), "total_reward": float(np.sum(rewards))})
    return summarize(label, rows)


def run_ppo(label: str, seeds: list[int], use_forecast: bool) -> dict[str, float | str]:
    agent = PPO.load(ROOT / "models" / f"{label}_stage1.zip")
    rows = []
    forecast_fn = load_wpm_forecaster() if use_forecast else None
    for seed in seeds:
        env = CPUSchedEnv(forecast_fn=forecast_fn, seed=seed)
        obs, info = env.reset(seed=seed)
        rewards = []
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            rewards.append(float(reward))
            done = terminated or truncated
        rows.append({**info, "mean_reward": float(np.mean(rewards)), "total_reward": float(np.sum(rewards))})
    return summarize(label, rows)


def summarize(label: str, rows: list[dict[str, float]]) -> dict[str, float | str]:
    df = pd.DataFrame(rows)
    out: dict[str, float | str] = {"scheduler": label}
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
        out[col] = float(df[col].mean())
    return out


def main() -> int:
    seeds = [100, 101]
    rows = [
        run_policy("Random", seeds, lambda obs, max_q: int(np.random.default_rng().integers(max_q))),
        run_policy("FCFS_policy", seeds, fcfs_policy),
        run_policy("SJF_policy", seeds, sjf_policy),
        run_policy("Confidence_Rule_WPM", seeds, rule_wpm_policy, use_forecast=True),
        run_ppo("PPO_no_forecast", seeds, use_forecast=False),
        run_ppo("PPO_DeepSched", seeds, use_forecast=True),
    ]
    out_path = ROOT / "results" / "ppo_metrics.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(pd.DataFrame(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
