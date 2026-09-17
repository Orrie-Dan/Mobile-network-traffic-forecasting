"""
Naive persistence baseline: ŷ(t+1) = y(t).

This is an evaluation reference only — not one of the three research models
(SARIMA, LSTM, TCN).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import INTERVAL_MINUTES, NAIVE_BASELINE_NAME
from src.data.forecasting import naive_persistence_forecast
from src.models.base import TimingResult, Timer, assert_one_step_alignment


class NaivePersistenceModel:
    """
    Persistence forecaster.

    For each target timestamp t+1, the prediction equals the observed value at t.
    No parameters are estimated.
    """

    name: str = NAIVE_BASELINE_NAME
    is_research_model: bool = False

    def __init__(self) -> None:
        self.timing = TimingResult()
        self._fitted = False
        self._series: Optional[pd.Series] = None

    def get_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_research_model": self.is_research_model,
            "rule": "y_hat(t+1) = y(t)",
            "horizon_minutes": INTERVAL_MINUTES,
        }

    def fit(self, series: pd.Series) -> "NaivePersistenceModel":
        """
        Store the chronological series used for persistence lookups.

        No statistical estimation occurs. Timing is recorded for interface
        consistency and is not comparable to SARIMA/LSTM/TCN training cost.
        """
        timer = Timer().start()
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("series index must be a DatetimeIndex of target timestamps")
        self._series = series.astype("float64").sort_index()
        self._fitted = True
        elapsed = timer.stop()
        self.timing.fit_seconds = elapsed
        self.timing.fit_started_at = timer.started_at
        self.timing.fit_ended_at = timer.ended_at
        return self

    def predict(
        self,
        target_timestamps: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        """
        Return persistence predictions aligned to target timestamps.

        If ``target_timestamps`` is None, predictions are produced for every
        timestamp after the first observation in the fitted series.
        """
        if not self._fitted or self._series is None:
            raise RuntimeError("Call fit() before predict()")

        timer = Timer().start()
        preds = naive_persistence_forecast(self._series)
        if target_timestamps is not None:
            targets = pd.DatetimeIndex(target_timestamps)
            missing = targets.difference(preds.index)
            if len(missing):
                raise KeyError(
                    f"Persistence cannot predict {len(missing)} timestamps "
                    f"(example: {missing[0]}). Need y(t) for each target t+1."
                )
            out = preds.loc[targets]
            # Alignment: prediction at t+1 equals series value at t = t+1 - 10min
            prev = targets - pd.Timedelta(minutes=INTERVAL_MINUTES)
            expected = self._series.reindex(prev).to_numpy()
            got = out.to_numpy(dtype="float64")
            if not np.allclose(got, expected, equal_nan=True):
                raise AssertionError("Naive persistence alignment check failed")
            assert_one_step_alignment(prev, targets)
        else:
            out = preds.iloc[1:]
        elapsed = timer.stop()
        self.timing.predict_seconds = elapsed
        self.timing.predict_started_at = timer.started_at
        self.timing.predict_ended_at = timer.ended_at
        return out.astype("float64")

    def predict_from_last_values(self, last_values: np.ndarray) -> np.ndarray:
        """Vectorised persistence: each prediction is the last input value."""
        arr = np.asarray(last_values, dtype="float64")
        return arr.copy()
