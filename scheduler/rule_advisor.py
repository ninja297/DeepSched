"""Confidence-aware forecast advisor for CPU scheduling."""

from __future__ import annotations

import numpy as np

from scheduler.classical import _arrive, _finish
from scheduler.confidence import forecast_confidence, should_trust_forecast
from scheduler.simulation import Process, RuntimeProcess, clone_processes, metrics


def rule_advisor(
    ready: list[RuntimeProcess],
    forecast: np.ndarray,
    tau_hi: float = 0.35,
    slope_hi: float = 0.01,
    confidence_floor: float = 0.70,
) -> int:
    """Pick a ready-queue index using confidence-gated predictions.

    If forecast confidence is low, the advisor falls back to FCFS instead of
    blindly trusting a potentially unstable model output.
    """
    if not ready:
        return 0
    if not should_trust_forecast(forecast, confidence_floor=confidence_floor):
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
    confidence_values: list[float] = []
    trusted_forecasts = 0

    while len(completed) < len(jobs):
        idx = _arrive(jobs, ready, idx, now)
        if not ready:
            now = jobs[idx].arrival
            continue

        forecast = forecast_series[min(forecast_idx, len(forecast_series) - 1)]
        confidence = forecast_confidence(forecast)
        confidence_values.append(confidence)
        if should_trust_forecast(forecast):
            trusted_forecasts += 1
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
    out["mean_confidence"] = float(np.mean(confidence_values)) if confidence_values else 0.0
    out["trusted_forecast_rate"] = (
        float(trusted_forecasts / len(confidence_values)) if confidence_values else 0.0
    )
    return out
