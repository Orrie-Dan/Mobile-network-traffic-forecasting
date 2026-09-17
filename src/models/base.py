"""
Shared interfaces and timing helpers for forecasting models.

Phase 6A implements models only. Final Dec 16–22 evaluation is out of scope.
Validation may be used later for configuration selection; the test split must
remain untouched for fitting and hyperparameter decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.config import INTERVAL_MINUTES, PRIMARY_SEQUENCE_LENGTH, TEST_START, utc_timestamp


@dataclass
class TimingResult:
    """Consistent wall-clock timing for fit and predict."""

    fit_seconds: Optional[float] = None
    predict_seconds: Optional[float] = None
    fit_started_at: Optional[str] = None
    fit_ended_at: Optional[str] = None
    predict_started_at: Optional[str] = None
    predict_ended_at: Optional[str] = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "fit_seconds": self.fit_seconds,
            "predict_seconds": self.predict_seconds,
            "fit_started_at": self.fit_started_at,
            "fit_ended_at": self.fit_ended_at,
            "predict_started_at": self.predict_started_at,
            "predict_ended_at": self.predict_ended_at,
        }
        out.update(self.extras)
        return out


class Timer:
    """Simple wall-clock timer used consistently across all models."""

    def __init__(self) -> None:
        self._t0: Optional[float] = None
        self.started_at: Optional[str] = None
        self.ended_at: Optional[str] = None
        self.elapsed_seconds: Optional[float] = None

    def start(self) -> "Timer":
        self._t0 = time.perf_counter()
        self.started_at = pd.Timestamp.utcnow().isoformat()
        self.ended_at = None
        self.elapsed_seconds = None
        return self

    def stop(self) -> float:
        if self._t0 is None:
            raise RuntimeError("Timer was not started")
        self.elapsed_seconds = float(time.perf_counter() - self._t0)
        self.ended_at = pd.Timestamp.utcnow().isoformat()
        self._t0 = None
        return self.elapsed_seconds


@dataclass
class NeuralTrainHistory:
    """Training diagnostics for LSTM/TCN. Test loss must never appear here."""

    epochs_completed: int = 0
    best_epoch: Optional[int] = None
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs_completed": self.epochs_completed,
            "best_epoch": self.best_epoch,
            "final_train_loss": self.train_loss[-1] if self.train_loss else None,
            "final_validation_loss": (
                self.validation_loss[-1] if self.validation_loss else None
            ),
            "best_validation_loss": (
                float(min(self.validation_loss)) if self.validation_loss else None
            ),
            "stopped_early": self.stopped_early,
            "train_loss": list(self.train_loss),
            "validation_loss": list(self.validation_loss),
            "monitored_split": "validation_only_if_provided",
            "test_loss_used": False,
        }


def assert_no_test_in_fit_window(
    timestamps: pd.DatetimeIndex | np.ndarray | list,
    *,
    context: str = "fit",
) -> None:
    """Raise if any timestamp falls inside the reserved final test period."""
    test_start = utc_timestamp(TEST_START)
    ts = pd.DatetimeIndex(timestamps)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    n_test = int((ts >= test_start).sum())
    if n_test:
        raise ValueError(
            f"{context}: {n_test} timestamps fall on/after the reserved test start "
            f"{test_start}. Test must remain untouched during fitting and tuning."
        )


def assert_one_step_alignment(
    input_end_timestamps: pd.DatetimeIndex | np.ndarray,
    target_timestamps: pd.DatetimeIndex | np.ndarray,
    *,
    step_minutes: int = INTERVAL_MINUTES,
) -> None:
    """Require each target to be exactly one 10-minute step after the input end."""
    ends = pd.DatetimeIndex(input_end_timestamps)
    targets = pd.DatetimeIndex(target_timestamps)
    if len(ends) != len(targets):
        raise AssertionError("input_end and target timestamp lengths differ")
    step = pd.Timedelta(minutes=step_minutes)
    bad = targets - ends != step
    if bad.any():
        idx = int(np.flatnonzero(bad)[0])
        raise AssertionError(
            f"One-step alignment failed at index {idx}: "
            f"input_end={ends[idx]} target={targets[idx]} "
            f"(expected delta {step})"
        )


def assert_boundary_alignment_example(
    input_end: pd.Timestamp,
    target: pd.Timestamp,
) -> None:
    """
    Explicit Phase 5 boundary check:

    input ends 2013-12-15 23:50  →  target 2013-12-16 00:00
    """
    expected_end = utc_timestamp("2013-12-15 23:50")
    expected_target = utc_timestamp("2013-12-16 00:00")
    end = pd.Timestamp(input_end)
    tgt = pd.Timestamp(target)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    if tgt.tzinfo is None:
        tgt = tgt.tz_localize("UTC")
    if end != expected_end or tgt != expected_target:
        raise AssertionError(
            f"Boundary alignment failed: got input_end={end}, target={tgt}; "
            f"expected {expected_end} → {expected_target}"
        )


def reshape_sequences_for_keras(
    X: np.ndarray,
    *,
    sequence_length: int = PRIMARY_SEQUENCE_LENGTH,
) -> np.ndarray:
    """Ensure neural inputs have shape (samples, L, 1)."""
    arr = np.asarray(X, dtype="float32")
    if arr.ndim == 2:
        if arr.shape[1] != sequence_length:
            raise ValueError(
                f"Expected sequence length {sequence_length}, got {arr.shape[1]}"
            )
        arr = arr[..., np.newaxis]
    elif arr.ndim == 3:
        if arr.shape[1] != sequence_length or arr.shape[2] != 1:
            raise ValueError(
                f"Expected shape (n, {sequence_length}, 1), got {arr.shape}"
            )
    else:
        raise ValueError(f"X must be 2-D or 3-D, got shape {arr.shape}")
    return arr
