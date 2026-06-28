"""Confidence estimation for forecast-aware scheduling.

Stage 1 uses a lightweight confidence estimate based on horizon stability:
forecasts whose horizon values vary less are treated as more trustworthy. Stage
2 can replace this with calibrated residual uncertainty, ensembles, or
conformal intervals.
"""

from __future__ import annotations

import numpy as np


def forecast_confidence(forecast: np.ndarray, uncertainty_scale: float = 0.05) -> float:
    """Return a confidence score in [0, 1] for a forecast horizon."""
    values = np.asarray(forecast, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    horizon_std = float(np.std(values))
    confidence = 1.0 - (horizon_std / max(uncertainty_scale, 1e-8))
    return float(np.clip(confidence, 0.0, 1.0))


def should_trust_forecast(
    forecast: np.ndarray,
    confidence_floor: float = 0.70,
    uncertainty_scale: float = 0.05,
) -> bool:
    """Decide whether scheduler should use the forecast or fall back."""
    return forecast_confidence(forecast, uncertainty_scale) >= confidence_floor
