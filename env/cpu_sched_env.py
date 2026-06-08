"""C-05 Gymnasium CPU scheduling environment.

The environment simulates a single CPU with a bounded ready queue. At each step,
the agent selects a ready-queue slot to run for one time quantum. The observation
contains normalized remaining bursts, current waiting times, and an optional
forecast vector from the workload prediction module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


@dataclass
class Proc:
    pid: int
    arrival: float
    burst: float
    remaining: float


class CPUSchedEnv(gym.Env):
    """Single-core process scheduling simulator."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        forecast_fn: Any | None = None,
        max_queue: int = 10,
        quantum: float = 20.0,
        episode_time: float = 50_000.0,
        arrival_lambda: float = 2.5,
        burst_mean: float = 2.5,
        burst_sigma: float = 0.8,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.forecast_fn = forecast_fn
        self.max_queue = max_queue
        self.quantum = quantum
        self.episode_time = episode_time
        self.arrival_lambda = arrival_lambda
        self.burst_mean = burst_mean
        self.burst_sigma = burst_sigma
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.max_queue * 3,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.max_queue)

        self.queue: list[Proc] = []
        self.t = 0.0
        self.next_pid = 0
        self.history: list[float] = []
        self.completed: list[dict[str, float]] = []
        self.total_busy = 0.0
        self.invalid_actions = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.queue = []
        self.t = 0.0
        self.next_pid = 0
        self.history = []
        self.completed = []
        self.total_busy = 0.0
        self.invalid_actions = 0
        self._arrive(force=True)
        return self._obs(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.queue:
            self.t += self.quantum
            self._arrive(force=True)
            return self._obs(), -0.01, self.t >= self.episode_time, False, self._info()

        action = int(action)
        if action >= len(self.queue):
            self.invalid_actions += 1
            action = 0

        proc = self.queue.pop(action)
        wait_before = self.t - proc.arrival
        run_for = min(self.quantum, proc.remaining)
        proc.remaining -= run_for
        self.t += run_for
        self.total_busy += run_for
        self._arrive()

        if proc.remaining <= 1e-8:
            turnaround = self.t - proc.arrival
            self.completed.append(
                {
                    "pid": float(proc.pid),
                    "arrival": proc.arrival,
                    "burst": proc.burst,
                    "finish": self.t,
                    "waiting": turnaround - proc.burst,
                    "turnaround": turnaround,
                }
            )
        else:
            self.queue.append(proc)

        queue_pressure = min(len(self.queue) / self.max_queue, 1.0)
        self.history.append(queue_pressure)
        energy = run_for * 0.05
        reward = -((wait_before / 500.0) + (0.1 * energy) + (0.05 * queue_pressure))
        terminated = self.t >= self.episode_time
        return self._obs(), float(reward), terminated, False, self._info()

    def _arrive(self, force: bool = False) -> None:
        arrivals = int(self.rng.poisson(self.arrival_lambda))
        if force and arrivals == 0:
            arrivals = 1
        for _ in range(arrivals):
            if len(self.queue) >= self.max_queue:
                break
            burst = float(self.rng.lognormal(self.burst_mean, self.burst_sigma))
            self.queue.append(
                Proc(
                    pid=self.next_pid,
                    arrival=self.t,
                    burst=burst,
                    remaining=burst,
                )
            )
            self.next_pid += 1

    def _forecast(self) -> np.ndarray:
        if self.forecast_fn is None or len(self.history) < 60:
            return np.zeros(self.max_queue, dtype=np.float32)
        hist = np.asarray(self.history[-60:], dtype=np.float32).reshape(1, 60, 1)
        with torch.no_grad():
            forecast = self.forecast_fn(torch.from_numpy(hist)).detach().cpu().numpy().reshape(-1)
        out = np.zeros(self.max_queue, dtype=np.float32)
        n = min(len(forecast), self.max_queue)
        out[:n] = np.clip(forecast[:n], 0.0, 1.0)
        return out

    def _obs(self) -> np.ndarray:
        q = self.queue[: self.max_queue]
        pad = self.max_queue - len(q)
        bursts = [min(p.remaining / 200.0, 1.0) for p in q] + [0.0] * pad
        waits = [min((self.t - p.arrival) / 500.0, 1.0) for p in q] + [0.0] * pad
        forecast = self._forecast().tolist()
        return np.asarray(bursts + waits + forecast, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        if self.completed:
            awt = float(np.mean([p["waiting"] for p in self.completed]))
            att = float(np.mean([p["turnaround"] for p in self.completed]))
        else:
            awt = 0.0
            att = 0.0
        return {
            "time": float(self.t),
            "queue_len": len(self.queue),
            "completed": len(self.completed),
            "AWT": awt,
            "ATT": att,
            "throughput": float(len(self.completed) / self.t) if self.t > 0 else 0.0,
            "cpu_utilization": float(self.total_busy / self.t) if self.t > 0 else 0.0,
            "energy_proxy": float(self.total_busy * 0.05),
            "invalid_actions": self.invalid_actions,
        }
