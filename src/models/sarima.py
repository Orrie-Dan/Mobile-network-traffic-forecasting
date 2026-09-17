"""
SARIMA one-step-ahead forecaster for unscaled Internet traffic.

Uses statsmodels SARIMAX with seasonal period s=144 (daily cycle at 10-minute
resolution). Does not use MinMax-scaled values. Does not touch the final test
set during fitting. Parameter selection over the small candidate set belongs to
Phase 6B (train/validation only).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.config import (
    SARIMA_DEFAULT_ORDER,
    SARIMA_DEFAULT_SEASONAL_ORDER,
    SARIMA_FIT_METHOD,
    SARIMA_MAXITER,
    SARIMA_ORDER_CANDIDATES,
    SARIMA_SEASONAL_ORDER_CANDIDATES,
    SARIMA_SEASONAL_PERIOD,
)
from src.models.base import TimingResult, Timer, assert_no_test_in_fit_window


@dataclass(frozen=True)
class SARIMAConfig:
    """
    Configurable SARIMA orders.

    Non-seasonal order (p, d, q):
        p — autoregressive lags of the non-seasonal process
        d — non-seasonal differencing order
        q — moving-average lags of the non-seasonal process

    Seasonal order (P, D, Q, s):
        P — seasonal AR lags
        D — seasonal differencing order
        Q — seasonal MA lags
        s — seasonal period in steps (fixed at 144 = 24h / 10min)

    Fitting method:
        Default ``lbfgs`` via statsmodels SARIMAX.fit (MLE).
    """

    order: tuple[int, int, int] = SARIMA_DEFAULT_ORDER
    seasonal_order: tuple[int, int, int, int] = SARIMA_DEFAULT_SEASONAL_ORDER
    trend: Optional[str] = None
    method: str = SARIMA_FIT_METHOD
    maxiter: int = SARIMA_MAXITER
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True

    def __post_init__(self) -> None:
        if len(self.order) != 3:
            raise ValueError("order must be (p, d, q)")
        if len(self.seasonal_order) != 4:
            raise ValueError("seasonal_order must be (P, D, Q, s)")
        if int(self.seasonal_order[3]) != SARIMA_SEASONAL_PERIOD:
            raise ValueError(
                f"Primary SARIMA seasonal period must be s={SARIMA_SEASONAL_PERIOD}; "
                f"got s={self.seasonal_order[3]}. Do not use s=1008 in the primary model."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": list(self.order),
            "seasonal_order": list(self.seasonal_order),
            "seasonal_period_s": int(self.seasonal_order[3]),
            "trend": self.trend,
            "method": self.method,
            "maxiter": self.maxiter,
            "enforce_stationarity": self.enforce_stationarity,
            "enforce_invertibility": self.enforce_invertibility,
            "order_candidates_for_phase6b": [list(o) for o in SARIMA_ORDER_CANDIDATES],
            "seasonal_order_candidates_for_phase6b": [
                list(o) for o in SARIMA_SEASONAL_ORDER_CANDIDATES
            ],
            "input_scale": "original_unscaled",
        }


class SARIMAModel:
    """
    Univariate SARIMA wrapper around statsmodels SARIMAX.

    Fit on the chronological training series only (original traffic units).
    One-step predictions use the Kalman filter / apply API so that each forecast
    conditions on observations available through time t.
    """

    name: str = "SARIMA"
    is_research_model: bool = True

    def __init__(self, config: Optional[SARIMAConfig] = None) -> None:
        self.config = config or SARIMAConfig()
        self.timing = TimingResult()
        self.result_: Any = None
        self.train_series_: Optional[pd.Series] = None
        self.fit_success_: bool = False
        self.fit_error_: Optional[str] = None
        self.fit_warning_: Optional[str] = None

    def get_config(self) -> dict[str, Any]:
        return self.config.to_dict()

    def fit(self, train_series: pd.Series) -> "SARIMAModel":
        """
        Fit SARIMA on the unscaled training series.

        Raises a clean RuntimeError if estimation fails (does not silently
        continue with an incomplete model).
        """
        if not isinstance(train_series.index, pd.DatetimeIndex):
            raise TypeError("train_series must be indexed by timestamps")
        series = train_series.astype("float64").sort_index()
        assert_no_test_in_fit_window(series.index, context="SARIMA.fit")
        if series.isna().any():
            raise ValueError("SARIMA training series contains NaN; no imputation applied")

        self.train_series_ = series
        self.fit_success_ = False
        self.fit_error_ = None
        self.fit_warning_ = None

        timer = Timer().start()
        try:
            model = SARIMAX(
                series,
                order=self.config.order,
                seasonal_order=self.config.seasonal_order,
                trend=self.config.trend,
                enforce_stationarity=self.config.enforce_stationarity,
                enforce_invertibility=self.config.enforce_invertibility,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                # cov_type='none' skips expensive covariance estimation (fit params still obtained)
                self.result_ = model.fit(
                    method=self.config.method,
                    maxiter=self.config.maxiter,
                    disp=False,
                    cov_type="none",
                )
                if caught:
                    self.fit_warning_ = "; ".join(str(w.message) for w in caught[:5])
            self.fit_success_ = True
        except Exception as exc:  # noqa: BLE001 - surface clean failure to caller
            self.fit_error_ = f"{type(exc).__name__}: {exc}"
            self.result_ = None
            self.fit_success_ = False
            elapsed = timer.stop()
            self.timing.fit_seconds = elapsed
            self.timing.fit_started_at = timer.started_at
            self.timing.fit_ended_at = timer.ended_at
            raise RuntimeError(
                f"SARIMA fit failed for order={self.config.order}, "
                f"seasonal_order={self.config.seasonal_order}: {self.fit_error_}"
            ) from exc

        elapsed = timer.stop()
        self.timing.fit_seconds = elapsed
        self.timing.fit_started_at = timer.started_at
        self.timing.fit_ended_at = timer.ended_at
        if self.result_ is not None:
            try:
                self.timing.extras["aic"] = float(self.result_.aic)
                self.timing.extras["bic"] = float(self.result_.bic)
            except Exception:  # noqa: BLE001
                self.timing.extras["aic"] = None
                self.timing.extras["bic"] = None
        self.timing.extras["fit_warning"] = self.fit_warning_
        return self

    def predict_one_step(
        self,
        history: pd.Series,
        *,
        target_timestamps: Optional[Sequence[pd.Timestamp]] = None,
    ) -> pd.Series:
        """
        One-step-ahead predictions on ``history`` using the fitted parameters.

        ``history`` must include all observations available through each
        forecast origin. Predictions are returned in original traffic units.
        This method does not compute evaluation metrics.

        Memory note: prefers ``extend`` of new observations onto the fitted
        training result (avoids re-filtering the full history with a dense
        (state, state, T) buffer, which is costly for s=144).
        """
        if not self.fit_success_ or self.result_ is None:
            raise RuntimeError("SARIMA model is not fitted successfully")
        if self.train_series_ is None:
            raise RuntimeError("SARIMA model has no stored training series")

        hist = history.astype("float64").sort_index()
        if hist.isna().any():
            raise ValueError("history contains NaN; no imputation applied")

        timer = Timer().start()
        train = self.train_series_.astype("float64").sort_index()
        # Observations beyond the fitting sample (typically validation).
        new = hist.loc[hist.index > train.index[-1]]
        overlap = hist.loc[hist.index <= train.index[-1]]
        if len(overlap) and not np.allclose(
            overlap.to_numpy(), train.reindex(overlap.index).to_numpy(), equal_nan=True
        ):
            # History disagrees with the fitting sample — fall back carefully.
            new = hist.iloc[0:0]
            overlap = hist

        preds_train = pd.Series(
            np.asarray(self.result_.fittedvalues, dtype="float64"),
            index=train.index,
            name="predicted_mean",
        )

        if len(new):
            # Memory-safe path (statsmodels >=0.15):
            # ``extend`` filters only the new endog that follows the training
            # sample, keeping fitted parameters fixed. There is no ``refit``
            # argument in this API; extend does not re-estimate parameters.
            extended = self.result_.extend(np.asarray(new.to_numpy(), dtype="float64"))
            # extend() returns results for the *new* observations only.
            fv = np.asarray(extended.fittedvalues, dtype="float64").reshape(-1)
            if len(fv) != len(new):
                # Defensive: some builds may return full-series fitted values.
                fv = fv[-len(new) :]
            preds_new = pd.Series(fv, index=new.index, name="predicted_mean")
            preds = pd.concat([preds_train, preds_new]).sort_index()
            del extended
        else:
            # Re-filter full history only when necessary (small series / smoke tests).
            # Prefer low_memory filtering when available to avoid huge state tensors.
            try:
                applied = self.result_.apply(hist, fit_kwargs={"low_memory": True})
            except TypeError:
                applied = self.result_.apply(hist)
            if hasattr(applied, "get_prediction"):
                preds = applied.get_prediction().predicted_mean.astype("float64")
            else:
                preds = pd.Series(
                    np.asarray(applied.fittedvalues, dtype="float64"),
                    index=hist.index,
                    name="predicted_mean",
                )
            del applied

        if target_timestamps is not None:
            targets = pd.DatetimeIndex(target_timestamps)
            missing = targets.difference(preds.index)
            if len(missing):
                raise KeyError(
                    f"SARIMA cannot produce {len(missing)} requested timestamps "
                    f"(example: {missing[0]})"
                )
            preds = preds.loc[targets]
        elapsed = timer.stop()
        self.timing.predict_seconds = elapsed
        self.timing.predict_started_at = timer.started_at
        self.timing.predict_ended_at = timer.ended_at
        return preds.astype("float64")

    def forecast_next(self, steps: int = 1) -> np.ndarray:
        """Out-of-sample forecast from the end of the training series."""
        if not self.fit_success_ or self.result_ is None:
            raise RuntimeError("SARIMA model is not fitted successfully")
        timer = Timer().start()
        fc = self.result_.forecast(steps=steps)
        elapsed = timer.stop()
        self.timing.predict_seconds = elapsed
        self.timing.predict_started_at = timer.started_at
        self.timing.predict_ended_at = timer.ended_at
        return np.asarray(fc, dtype="float64")
