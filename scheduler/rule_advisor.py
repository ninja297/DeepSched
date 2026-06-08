"""Forecast-aware rule advisor for CPU scheduling."""

from __future__ import annotations

import numpy as np

from scheduler.classical import _arrive, _finish
from scheduler.simulation import Process, RuntimeProcess, clone_processes, metrics


def rule_advisor(
    ready: list[RuntimeProcess],
    forecast: np.ndarray,
    tau_hi: float = 0.35,
    slope_hi: float = 0.01,
) -> int:
    """Pick a ready-queue index using the predicted utilization horizon."""
    if not ready:
        return 0
    u = float(np.mean(forecast))
    slope = float(forecast[-1] - forecast[0])
    if u >= tau_hi or slope >= slope_hi:
        return min(range(len(ready)), key=lambda i: ready[i].remaining)
    return 0


def run_rule_advisor(
    processes: list[Process],
    forecast_series: np.ndarray,
    quantum: float = 20.0,
) -> dict[str, float]:
    jobs = clone_processes(processes)
    ready: list[RuntimeProcess] = []
    completed: list[dict[str, float]] = []
    now = 0.0
    idx = 0
    forecast_idx = 0

    while len(completed) < len(jobs):
        idx = _arrive(jobs, ready, idx, now)
        if not ready:
            now = jobs[idx].arrival
            continue

        forecast = forecast_series[min(forecast_idx, len(forecast_series) - 1)]
        choice = rule_advisor(ready, forecast)
        proc = ready.pop(choice)
        run_for = min(quantum, proc.remaining)
        now += run_for
        proc.remaining -= run_for
        forecast_idx += 1
        idx = _arrive(jobs, ready, idx, now)

        if proc.remaining <= 1e-8:
            completed.append(_finish(proc, now))
        else:
            ready.append(proc)

    out = metrics(completed, now)
    out["forecast_steps_used"] = float(forecast_idx)
    return out
