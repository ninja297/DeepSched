"""Shared process-trace generation and scheduling metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Process:
    pid: int
    arrival: float
    burst: float


@dataclass
class RuntimeProcess:
    pid: int
    arrival: float
    burst: float
    remaining: float


def generate_processes(
    n: int = 500,
    seed: int = 42,
    arrival_scale: float = 30.0,
    burst_mean: float = 2.5,
    burst_sigma: float = 0.8,
) -> list[Process]:
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(arrival_scale, n))
    bursts = rng.lognormal(burst_mean, burst_sigma, n)
    return [
        Process(pid=i, arrival=float(arrival), burst=float(burst))
        for i, (arrival, burst) in enumerate(zip(arrivals, bursts))
    ]


def clone_processes(processes: list[Process]) -> list[RuntimeProcess]:
    return [
        RuntimeProcess(p.pid, p.arrival, p.burst, p.burst)
        for p in processes
    ]


def metrics(completed: list[dict[str, float]], makespan: float) -> dict[str, float]:
    waits = [p["finish"] - p["arrival"] - p["burst"] for p in completed]
    turnarounds = [p["finish"] - p["arrival"] for p in completed]
    total_burst = sum(p["burst"] for p in completed)
    return {
        "AWT": float(np.mean(waits)),
        "ATT": float(np.mean(turnarounds)),
        "throughput": float(len(completed) / makespan) if makespan > 0 else 0.0,
        "cpu_utilization": float(total_burst / makespan) if makespan > 0 else 0.0,
        "energy_proxy": float(total_burst * 0.05),
        "completed": float(len(completed)),
        "makespan": float(makespan),
    }
