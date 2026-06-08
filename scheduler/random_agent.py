"""Random-agent sanity baseline for CPUSchedEnv."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from env.cpu_sched_env import CPUSchedEnv


def run_episode(seed: int) -> dict[str, float]:
    env = CPUSchedEnv(seed=seed)
    obs, info = env.reset(seed=seed)
    rewards = []
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return {
        "seed": float(seed),
        "mean_reward": float(np.mean(rewards)),
        "total_reward": float(np.sum(rewards)),
        "steps": float(len(rewards)),
        "AWT": float(info["AWT"]),
        "ATT": float(info["ATT"]),
        "throughput": float(info["throughput"]),
        "cpu_utilization": float(info["cpu_utilization"]),
        "energy_proxy": float(info["energy_proxy"]),
        "completed": float(info["completed"]),
        "invalid_actions": float(info["invalid_actions"]),
    }


def main() -> int:
    rows = [run_episode(seed) for seed in range(5)]
    out_path = ROOT / "results" / "random_agent.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path}")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
