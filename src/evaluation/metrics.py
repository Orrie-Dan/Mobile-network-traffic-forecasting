"""
Generic forecast error metrics, independent of any specific model.

MAE and RMSE are always computed over every provided observation.
MAPE retains every observation by flooring the absolute denominator; it does
not drop zeros. Inverse-transform neural predictions to the original traffic
scale before calling these functions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import MAPE_DENOMINATOR_FLOOR, MAPE_NEAR_ZERO_THRESHOLD


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error over all observations."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error over all observations."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(
    y_true,
    y_pred,
    *,
    denominator_floor: float = MAPE_DENOMINATOR_FLOOR,
) -> float:
    """
    Mean Absolute Percentage Error in percent.

    Zero / near-zero rule
    ---------------------
    The denominator is ``max(|y_true|, denominator_floor)``. Every observation
    is retained. Observations are **not** dropped when actual traffic is zero
    or near zero. MAE and RMSE do not use this floor and remain on the full set.

    The floor is a numerical safeguard (default ``1e-8``). On the Phase 5
    forecasting squares, inspected training/validation/test traffic is bounded
    well above 1, so the floor does not change those MAPE values.
    """
    if denominator_floor <= 0:
        raise ValueError("denominator_floor must be positive")
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denom = np.maximum(np.abs(y_true), float(denominator_floor))
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def mape_diagnostics(
    y_true,
    *,
    denominator_floor: float = MAPE_DENOMINATOR_FLOOR,
    near_zero_threshold: float = MAPE_NEAR_ZERO_THRESHOLD,
) -> dict[str, Any]:
    """Counts for documenting MAPE stability; does not drop observations."""
    y_true = np.asarray(y_true, dtype=float)
    abs_true = np.abs(y_true)
    n = int(y_true.size)
    n_zero = int(np.sum(y_true == 0))
    n_below_floor = int(np.sum(abs_true < float(denominator_floor)))
    n_near_zero = int(np.sum(abs_true < float(near_zero_threshold)))
    return {
        "n_observations": n,
        "n_zero": n_zero,
        "n_below_denominator_floor": n_below_floor,
        "n_below_near_zero_threshold": n_near_zero,
        "denominator_floor": float(denominator_floor),
        "near_zero_threshold": float(near_zero_threshold),
        "min_abs_actual": float(abs_true.min()) if n else None,
        "observations_dropped": 0,
    }


def regression_metrics(
    y_true,
    y_pred,
    *,
    denominator_floor: float = MAPE_DENOMINATOR_FLOOR,
) -> dict[str, Any]:
    """MAE, RMSE, MAPE plus MAPE diagnostics. Call on original-scale values."""
    diag = mape_diagnostics(y_true, denominator_floor=denominator_floor)
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred, denominator_floor=denominator_floor),
        "mape_diagnostics": diag,
    }


def inverse_transform_then_metrics(
    y_true_original,
    y_pred_scaled,
    *,
    inverse_transform,
    denominator_floor: float = MAPE_DENOMINATOR_FLOOR,
) -> dict[str, Any]:
    """
    Inverse-transform scaled predictions, then score on the original scale.

    ``inverse_transform`` must map a 1-d array of scaled values to original units.
    ``y_true_original`` must already be on the original traffic scale.
    """
    y_true = np.asarray(y_true_original, dtype=float)
    y_pred = np.asarray(inverse_transform(np.asarray(y_pred_scaled, dtype=float)), dtype=float)
    out = regression_metrics(y_true, y_pred, denominator_floor=denominator_floor)
    out["metrics_scale"] = "original"
    return out
