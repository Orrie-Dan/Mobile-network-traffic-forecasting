"""
Phase 5 forecasting experiment preparation.

Builds chronological splits, integrity reports, neural-network sequences,
train-only scalers, SARIMA univariate series, and the naive persistence
baseline. Does **not** fit SARIMA, LSTM, or TCN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.analysis.eda import load_target_timeseries
from src.config import (
    ALIGNMENT_EXAMPLE_FILE,
    BINS_PER_DAY,
    BINS_PER_WEEK,
    EDA_SQUARES,
    EXPECTED_ROWS,
    EXPERIMENT_CONFIG_SNAPSHOT_FILE,
    FORECASTING_DATASET_FILE,
    FORECASTING_MODELS,
    FREQUENCY,
    INTERVAL_MINUTES,
    MAPE_DENOMINATOR_FLOOR,
    MAPE_NEAR_ZERO_THRESHOLD,
    NAIVE_BASELINE_NAME,
    PRIMARY_SEQUENCE_LENGTH,
    PROCESSED_SOURCE_FILE,
    SCALER_FEATURE_RANGE,
    SCALER_PARAMS_FILE,
    SCALER_TYPE,
    SEQUENCE_LENGTH_ANALYSIS_FILE,
    SEQUENCE_LENGTH_CANDIDATES,
    SPLIT_VALIDATION_FILE,
    TARGET_COLUMN,
    TARGET_SQUARES,
    TEST_END,
    TEST_START,
    ZERO_INSPECTION_FILE,
    expected_index,
    split_bounds,
    utc_timestamp,
)

SPLIT_ORDER: tuple[str, ...] = ("train", "validation", "test")


class ForecastDataIntegrityError(RuntimeError):
    """Raised when integrity checks fail. Data are not silently repaired."""

    def __init__(self, issues: list[str], summary: Optional[pd.DataFrame] = None):
        self.issues = issues
        self.summary = summary
        numbered = "\n".join(f"  - {item}" for item in issues)
        super().__init__(
            "Forecast data integrity checks failed. "
            "No silent repair (zero-fill, interpolation, or forward-fill) was applied.\n"
            + numbered
        )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(path: Path | str, *, root: Optional[Path] = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or _project_root()) / p


def load_source_target_timeseries(
    path: Optional[Path | str] = None,
    *,
    root: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the Phase 3B 10-minute extract. Does not reprocess raw ZIPs."""
    csv_path = _resolve(path or PROCESSED_SOURCE_FILE, root=root)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Processed target-square series not found: {csv_path}. "
            "Run Phase 3B extraction first; do not reprocess the full raw dataset here."
        )
    return load_target_timeseries(csv_path)


def assign_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a ``split`` column. Timestamps outside the experiment remain NaN."""
    out = frame.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    split = pd.Series(pd.NA, index=out.index, dtype="object")
    for name, (start, end) in split_bounds().items():
        split.loc[(ts >= start) & (ts <= end)] = name
    out["split"] = split
    return out


def build_forecasting_dataset(
    source: pd.DataFrame,
    *,
    squares: Sequence[int] = TARGET_SQUARES,
) -> pd.DataFrame:
    """
    Tidy forecasting table: timestamp, square_id, internet_traffic, split.

    Rows outside train/validation/test are dropped (not zero-filled).
    """
    required = {"timestamp", "square_id", TARGET_COLUMN}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Source is missing columns: {sorted(missing)}")

    work = source.loc[source["square_id"].isin(list(squares))].copy()
    unexpected = sorted(set(work["square_id"].unique()) - set(squares))
    if unexpected:
        raise ForecastDataIntegrityError(
            [f"Unexpected square_id values in source subset: {unexpected}"]
        )

    work = assign_splits(work)
    work = work.loc[work["split"].notna()].copy()
    work["square_id"] = work["square_id"].astype("int32")
    work["split"] = pd.Categorical(work["split"], categories=list(SPLIT_ORDER), ordered=True)
    work = work[["timestamp", "square_id", TARGET_COLUMN, "split"]]
    work = work.sort_values(["square_id", "timestamp"]).reset_index(drop=True)
    return work


def inspect_zero_traffic(
    frame: pd.DataFrame,
    *,
    squares: Sequence[int] = TARGET_SQUARES,
) -> pd.DataFrame:
    """Inspect zeros and near-zeros per square and split. Does not drop rows."""
    rows: list[dict[str, Any]] = []
    for sid in squares:
        sub_all = frame.loc[frame["square_id"] == sid]
        scopes = [("all_experiment", sub_all)]
        for split in SPLIT_ORDER:
            scopes.append((split, sub_all.loc[sub_all["split"] == split]))
        for scope, sub in scopes:
            vals = sub[TARGET_COLUMN].astype("float64")
            rows.append(
                {
                    "square_id": int(sid),
                    "scope": scope,
                    "n_observations": int(len(vals)),
                    "n_nan": int(vals.isna().sum()),
                    "n_zero": int((vals == 0).sum()),
                    "n_below_mape_floor": int((vals.abs() < MAPE_DENOMINATOR_FLOOR).sum()),
                    "n_below_near_zero_threshold": int(
                        (vals.abs() < MAPE_NEAR_ZERO_THRESHOLD).sum()
                    ),
                    "min_internet_traffic": float(vals.min()) if len(vals) else None,
                    "max_internet_traffic": float(vals.max()) if len(vals) else None,
                    "observations_dropped": 0,
                }
            )
    return pd.DataFrame(rows)


def _square_issues(
    sub: pd.DataFrame,
    *,
    square_id: int,
    allowed_squares: Sequence[int],
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    sid = int(square_id)
    prefix = f"square {sid}"

    unexpected = sorted(set(sub["square_id"].unique()) - set(allowed_squares))
    if unexpected:
        issues.append(f"{prefix}: unexpected square_id values {unexpected}")

    if sub.empty:
        issues.append(f"{prefix}: no rows after split assignment")
        return issues, {}

    if not sub["timestamp"].is_monotonic_increasing:
        issues.append(f"{prefix}: timestamps are not in chronological order")

    n_dup = int(sub["timestamp"].duplicated().sum())
    if n_dup:
        issues.append(f"{prefix}: {n_dup} duplicate timestamps")

    n_nan_ts = int(sub["timestamp"].isna().sum())
    if n_nan_ts:
        issues.append(f"{prefix}: {n_nan_ts} missing timestamp values")

    n_nan_y = int(sub[TARGET_COLUMN].isna().sum())
    if n_nan_y:
        issues.append(f"{prefix}: {n_nan_y} missing {TARGET_COLUMN} values")

    diffs = sub["timestamp"].diff().dropna()
    bad_freq = diffs[diffs != pd.Timedelta(minutes=INTERVAL_MINUTES)]
    if len(bad_freq):
        issues.append(
            f"{prefix}: {len(bad_freq)} adjacent gaps are not {INTERVAL_MINUTES} minutes "
            f"(examples: {bad_freq.head(3).tolist()})"
        )

    counts = {name: int((sub["split"] == name).sum()) for name in SPLIT_ORDER}
    missing_counts = {name: 0 for name in SPLIT_ORDER}
    for name in SPLIT_ORDER:
        expected_ts = expected_index(name)
        observed = pd.DatetimeIndex(sub.loc[sub["split"] == name, "timestamp"].unique())
        missing = expected_ts.difference(observed)
        extra = observed.difference(expected_ts)
        missing_counts[name] = int(len(missing))
        if missing_counts[name]:
            issues.append(
                f"{prefix}: {missing_counts[name]} missing timestamps in {name} "
                f"(first: {missing.min() if len(missing) else None})"
            )
        if len(extra):
            issues.append(
                f"{prefix}: {len(extra)} timestamps labelled {name} are outside the "
                f"{name} bounds"
            )
        if counts[name] != EXPECTED_ROWS[name]:
            issues.append(
                f"{prefix}: {name} has {counts[name]} rows; expected {EXPECTED_ROWS[name]}"
            )

    train_ts = set(sub.loc[sub["split"] == "train", "timestamp"])
    val_ts = set(sub.loc[sub["split"] == "validation", "timestamp"])
    test_ts = set(sub.loc[sub["split"] == "test", "timestamp"])
    if train_ts & test_ts:
        issues.append(f"{prefix}: train/test timestamp overlap ({len(train_ts & test_ts)})")
    if val_ts & test_ts:
        issues.append(
            f"{prefix}: validation/test timestamp overlap ({len(val_ts & test_ts)})"
        )
    if train_ts & val_ts:
        issues.append(
            f"{prefix}: train/validation timestamp overlap ({len(train_ts & val_ts)})"
        )

    test_start = utc_timestamp(TEST_START)
    test_end = utc_timestamp(TEST_END)
    train_in_test = sub.loc[
        (sub["split"] == "train")
        & (sub["timestamp"] >= test_start)
        & (sub["timestamp"] <= test_end)
    ]
    val_in_test = sub.loc[
        (sub["split"] == "validation")
        & (sub["timestamp"] >= test_start)
        & (sub["timestamp"] <= test_end)
    ]
    if len(train_in_test):
        issues.append(
            f"{prefix}: {len(train_in_test)} training rows fall inside the test period"
        )
    if len(val_in_test):
        issues.append(
            f"{prefix}: {len(val_in_test)} validation rows fall inside the test period"
        )

    summary = {
        "square_id": sid,
        "train_rows": counts["train"],
        "validation_rows": counts["validation"],
        "test_rows": counts["test"],
        "missing_train": missing_counts["train"],
        "missing_validation": missing_counts["validation"],
        "missing_test": missing_counts["test"],
        "duplicate_timestamps": n_dup,
        "start_timestamp": str(sub["timestamp"].min()),
        "end_timestamp": str(sub["timestamp"].max()),
    }
    return issues, summary


def validate_forecasting_dataset(
    frame: pd.DataFrame,
    *,
    squares: Sequence[int] = TARGET_SQUARES,
    raise_on_issue: bool = True,
) -> pd.DataFrame:
    """
    Per-square integrity checks. Raises ``ForecastDataIntegrityError`` on failure.

    Does not interpolate, forward-fill, or zero-fill.
    """
    issues: list[str] = []
    summaries: list[dict[str, Any]] = []
    present = sorted(int(s) for s in frame["square_id"].unique())
    expected = [int(s) for s in squares]
    if present != sorted(expected):
        issues.append(f"square_id set {present} does not match TARGET_SQUARES {expected}")

    extra = sorted(set(present) - set(expected))
    if extra:
        issues.append(f"unexpected square_id values in forecasting dataset: {extra}")

    for sid in expected:
        sub = (
            frame.loc[frame["square_id"] == sid]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        square_issues, summary = _square_issues(
            sub, square_id=sid, allowed_squares=expected
        )
        issues.extend(square_issues)
        if summary:
            summaries.append(summary)

    report = pd.DataFrame(summaries)
    if issues and raise_on_issue:
        raise ForecastDataIntegrityError(issues, summary=report)
    report.attrs["issues"] = issues
    return report


def univariate_series(
    frame: pd.DataFrame,
    square_id: int,
    *,
    split: Optional[str] = None,
) -> pd.Series:
    """Chronological univariate Internet traffic for SARIMA or diagnostics."""
    sub = frame.loc[frame["square_id"] == square_id]
    if split is not None:
        sub = sub.loc[sub["split"] == split]
    series = (
        sub.sort_values("timestamp")
        .set_index("timestamp")[TARGET_COLUMN]
        .astype("float64")
    )
    if series.index.has_duplicates:
        raise ForecastDataIntegrityError(
            [f"square {square_id}: duplicate timestamps in univariate series"]
        )
    as_freq = series.asfreq(FREQUENCY)
    if as_freq.isna().any():
        n_miss = int(as_freq.isna().sum())
        raise ForecastDataIntegrityError(
            [
                f"square {square_id}: asfreq('{FREQUENCY}') introduced {n_miss} missing "
                "values. No interpolation or zero-fill was applied."
            ]
        )
    return as_freq


def naive_persistence_forecast(series: pd.Series) -> pd.Series:
    """
    Naive persistence baseline: prediction(t+1) = y(t).

    This is **not** one of the three research models. The returned series is
    aligned to the target timestamp t+1 (the value equals the previous observation).
    The first timestamp has no prediction.
    """
    return series.shift(1)


@dataclass
class SquareScaler:
    """MinMax (default) or standard scaler fitted on training data only."""

    square_id: int
    scaler_type: str = SCALER_TYPE
    scaler: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=SCALER_FEATURE_RANGE)
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type}")

    def fit_train(self, train_values: np.ndarray) -> "SquareScaler":
        values = np.asarray(train_values, dtype="float64").reshape(-1, 1)
        if values.size == 0:
            raise ValueError(f"square {self.square_id}: cannot fit scaler on empty train set")
        self.scaler.fit(values)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype="float64").reshape(-1, 1)
        return self.scaler.transform(arr).ravel()

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype="float64").reshape(-1, 1)
        return self.scaler.inverse_transform(arr).ravel()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "square_id": int(self.square_id),
            "scaler_type": self.scaler_type,
            "n_samples_seen": int(self.scaler.n_samples_seen_),
        }
        if hasattr(self.scaler, "data_min_"):
            payload["data_min"] = float(self.scaler.data_min_[0])
            payload["data_max"] = float(self.scaler.data_max_[0])
            payload["feature_range"] = list(self.scaler.feature_range)
        if hasattr(self.scaler, "mean_"):
            payload["mean"] = float(self.scaler.mean_[0])
            payload["scale"] = float(self.scaler.scale_[0])
        return payload


def fit_square_scalers(
    frame: pd.DataFrame,
    *,
    squares: Sequence[int] = TARGET_SQUARES,
    scaler_type: str = SCALER_TYPE,
) -> dict[int, SquareScaler]:
    """Fit one scaler per square on the training split only."""
    scalers: dict[int, SquareScaler] = {}
    for sid in squares:
        train_vals = frame.loc[
            (frame["square_id"] == sid) & (frame["split"] == "train"), TARGET_COLUMN
        ].to_numpy(dtype="float64")
        val_vals = frame.loc[
            (frame["square_id"] == sid) & (frame["split"] == "validation"), TARGET_COLUMN
        ]
        test_vals = frame.loc[
            (frame["square_id"] == sid) & (frame["split"] == "test"), TARGET_COLUMN
        ]
        scaler = SquareScaler(square_id=int(sid), scaler_type=scaler_type).fit_train(
            train_vals
        )
        # Leakage guard: train-only fit must match the training min/max (MinMax).
        if scaler_type == "minmax":
            train_min = float(np.min(train_vals))
            train_max = float(np.max(train_vals))
            if not np.isclose(scaler.scaler.data_min_[0], train_min):
                raise ForecastDataIntegrityError(
                    [f"square {sid}: scaler data_min_ does not match training minimum"]
                )
            if not np.isclose(scaler.scaler.data_max_[0], train_max):
                raise ForecastDataIntegrityError(
                    [f"square {sid}: scaler data_max_ does not match training maximum"]
                )
            if len(val_vals) and (
                float(val_vals.min()) < train_min - 1e-12
                or float(val_vals.max()) > train_max + 1e-12
            ):
                # Allowed: val/test may fall outside the train range after transform.
                pass
            # Confirm the scaler was not fit on val/test extrema if they differ.
            combined_min = float(
                np.min(
                    np.concatenate(
                        [train_vals, val_vals.to_numpy(), test_vals.to_numpy()]
                    )
                )
            )
            combined_max = float(
                np.max(
                    np.concatenate(
                        [train_vals, val_vals.to_numpy(), test_vals.to_numpy()]
                    )
                )
            )
            if not np.isclose(combined_min, train_min) or not np.isclose(
                combined_max, train_max
            ):
                if np.isclose(scaler.scaler.data_min_[0], combined_min) and not np.isclose(
                    combined_min, train_min
                ):
                    raise ForecastDataIntegrityError(
                        [f"square {sid}: scaler appears to have used non-training minima"]
                    )
                if np.isclose(scaler.scaler.data_max_[0], combined_max) and not np.isclose(
                    combined_max, train_max
                ):
                    raise ForecastDataIntegrityError(
                        [f"square {sid}: scaler appears to have used non-training maxima"]
                    )
        if int(scaler.scaler.n_samples_seen_) != int(len(train_vals)):
            raise ForecastDataIntegrityError(
                [
                    f"square {sid}: scaler n_samples_seen_={scaler.scaler.n_samples_seen_} "
                    f"!= train rows {len(train_vals)}"
                ]
            )
        scalers[int(sid)] = scaler
    return scalers


def transform_column(
    frame: pd.DataFrame,
    scalers: dict[int, SquareScaler],
    *,
    column: str = TARGET_COLUMN,
) -> pd.Series:
    """Transform an already-split frame; scaler identity is per square."""
    out = np.empty(len(frame), dtype="float64")
    for sid, scaler in scalers.items():
        mask = frame["square_id"] == sid
        out[mask.to_numpy()] = scaler.transform(frame.loc[mask, column].to_numpy())
    return pd.Series(out, index=frame.index, name=f"{column}_scaled")


def make_supervised_sequences(
    frame: pd.DataFrame,
    *,
    square_id: int,
    sequence_length: int,
    target_split: Optional[str] = None,
    values_column: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """
    Sliding-window one-step-ahead samples.

    X[i] = [y(t-L+1), ..., y(t)]
    y[i] = y(t+1)

    Sample split is the split of the **target** timestamp. Inputs may come from
    earlier splits so that validation/test windows can use history that would
    have been available at time t. Inputs never include the target or any later
    observation.

    Training samples therefore cannot contain test (or validation) observations.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")

    series_df = (
        frame.loc[frame["square_id"] == square_id]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    values = series_df[values_column].to_numpy(dtype="float64")
    timestamps = pd.DatetimeIndex(series_df["timestamp"])
    splits = series_df["split"].astype(str).to_numpy()
    n = len(values)
    if n <= sequence_length:
        raise ValueError(
            f"square {square_id}: need more than {sequence_length} observations"
        )

    step = pd.Timedelta(minutes=INTERVAL_MINUTES)
    test_start = utc_timestamp(TEST_START)
    target_idx = np.arange(sequence_length, n)
    input_end_idx = target_idx - 1
    target_times = timestamps[target_idx]
    input_end_times = timestamps[input_end_idx]
    target_splits = splits[target_idx]

    keep = np.ones(len(target_idx), dtype=bool)
    if target_split is not None:
        keep &= target_splits == target_split

    alignment_ok = (target_times - input_end_times) == step
    leakage_flags: list[str] = []
    if not alignment_ok.all():
        bad = np.flatnonzero(~alignment_ok)
        for j in bad[:20]:
            leakage_flags.append(
                f"target {target_times[j]} is not {INTERVAL_MINUTES} minutes after "
                f"input end {input_end_times[j]}"
            )

    windows = np.lib.stride_tricks.sliding_window_view(values, sequence_length + 1)
    X_all = windows[:, :-1]
    y_all = windows[:, -1]
    # Inputs for target index i are splits[i-L:i] (length L, excluding the target).
    input_split_windows = np.lib.stride_tricks.sliding_window_view(
        splits[:-1], sequence_length
    )

    train_mask = target_splits == "train"
    val_mask = target_splits == "validation"
    input_has_non_train = (input_split_windows != "train").any(axis=1)
    input_has_test = (input_split_windows == "test").any(axis=1)
    train_target_in_test_period = np.asarray(target_times >= test_start)

    bad_train_inputs = train_mask & input_has_non_train
    bad_test_in_history = (train_mask | val_mask) & input_has_test
    bad_train_period = train_mask & train_target_in_test_period

    if bad_train_inputs.any() or bad_test_in_history.any() or bad_train_period.any():
        n_bad = (
            int(bad_train_inputs.sum())
            + int(bad_test_in_history.sum())
            + int(bad_train_period.sum())
        )
        leakage_flags.append(
            f"square {square_id}: {n_bad} sequences violate split leakage rules"
        )

    if leakage_flags:
        raise ForecastDataIntegrityError(
            [f"square {square_id} sequence leakage/alignment:"] + leakage_flags[:20]
        )

    keep &= alignment_ok
    X = np.ascontiguousarray(X_all[keep])
    y = np.ascontiguousarray(y_all[keep])
    return {
        "square_id": int(square_id),
        "sequence_length": int(sequence_length),
        "X": X,
        "y": y,
        "target_timestamp": pd.DatetimeIndex(target_times[keep]),
        "input_end_timestamp": pd.DatetimeIndex(input_end_times[keep]),
        "split": target_splits[keep],
        "n_samples": int(len(y)),
    }


def alignment_example(
    frame: pd.DataFrame,
    *,
    square_id: int = TARGET_SQUARES[0],
    sequence_length: int = PRIMARY_SEQUENCE_LENGTH,
) -> pd.DataFrame:
    """
    Explicit one-step boundary:

    input ends 2013-12-15 23:50  →  target 2013-12-16 00:00
    """
    seq = make_supervised_sequences(
        frame,
        square_id=square_id,
        sequence_length=sequence_length,
        target_split="test",
    )
    target = utc_timestamp("2013-12-16 00:00")
    mask = seq["target_timestamp"] == target
    if not mask.any():
        raise ForecastDataIntegrityError(
            [
                f"square {square_id}: no sequence with target {target}. "
                "Check that validation history is available to the first test point."
            ]
        )
    idx = int(np.flatnonzero(mask)[0])
    input_end = seq["input_end_timestamp"][idx]
    expected_end = utc_timestamp("2013-12-15 23:50")
    if input_end != expected_end:
        raise ForecastDataIntegrityError(
            [
                f"square {square_id}: expected input end {expected_end}, got {input_end}"
            ]
        )
    x_row = seq["X"][idx]
    return pd.DataFrame(
        {
            "square_id": [int(square_id)],
            "sequence_length": [int(sequence_length)],
            "input_start_timestamp": [
                str(input_end - pd.Timedelta(minutes=INTERVAL_MINUTES * (sequence_length - 1)))
            ],
            "input_end_timestamp": [str(input_end)],
            "target_timestamp": [str(seq["target_timestamp"][idx])],
            "target_split": [seq["split"][idx]],
            "input_last_value": [float(x_row[-1])],
            "target_value": [float(seq["y"][idx])],
            "step_minutes": [INTERVAL_MINUTES],
            "naive_persistence_prediction": [float(x_row[-1])],
        }
    )


def analyse_sequence_lengths(
    frame: pd.DataFrame,
    *,
    squares: Sequence[int] = TARGET_SQUARES,
    candidates: Sequence[int] = SEQUENCE_LENGTH_CANDIDATES,
) -> pd.DataFrame:
    """
    Count samples and memory for candidate window lengths. Does not train models.
    """
    rows: list[dict[str, Any]] = []
    bytes_per_value = 4  # float32 planned storage for neural tensors
    for L in candidates:
        hours = L * INTERVAL_MINUTES / 60.0
        days = hours / 24.0
        for sid in squares:
            seq = make_supervised_sequences(
                frame, square_id=int(sid), sequence_length=int(L)
            )
            n_by_split = {
                name: int(np.sum(seq["split"] == name)) for name in SPLIT_ORDER
            }
            n_train = n_by_split["train"]
            x_mb = (n_train * L * bytes_per_value) / (1024 ** 2)
            rows.append(
                {
                    "square_id": int(sid),
                    "sequence_length": int(L),
                    "window_hours": hours,
                    "window_days": days,
                    "captures_daily_lag_144": bool(L >= BINS_PER_DAY),
                    "captures_weekly_lag_1008": bool(L >= BINS_PER_WEEK),
                    "train_samples": n_train,
                    "validation_samples": n_by_split["validation"],
                    "test_samples": n_by_split["test"],
                    "train_X_float32_mb": round(x_mb, 4),
                    "lstm_relative_unroll_cost_vs_144": round(L / float(BINS_PER_DAY), 4),
                    "tcn_relative_input_length_vs_144": round(L / float(BINS_PER_DAY), 4),
                    "selected_primary": bool(L == PRIMARY_SEQUENCE_LENGTH),
                }
            )
    return pd.DataFrame(rows)


def assess_sarima_seasonal_period() -> dict[str, Any]:
    """
    Documented assessment of SARIMA seasonal periods. No model is fitted.
    """
    return {
        "data_frequency": FREQUENCY,
        "candidate_s_daily": BINS_PER_DAY,
        "candidate_s_weekly": BINS_PER_WEEK,
        "recommended_seasonal_period": BINS_PER_DAY,
        "do_not_use_weekly_in_single_sarima": True,
        "reasons": {
            "s_144_statistical": (
                "Phase 3B ACF on square 5161 is high at lag 144 (~0.88), matching "
                "the 24-hour cycle at 10-minute resolution."
            ),
            "s_144_practical": (
                "A seasonal period of 144 is large for classical SARIMA but remains "
                "the smallest period that encodes the dominant daily cycle without "
                "changing the 10-minute resolution. Seasonal orders should be kept "
                "small (to be chosen in the training phase)."
            ),
            "s_1008_statistical": (
                "ACF at lag 1008 is also high (~0.84), so a weekly cycle is present."
            ),
            "s_1008_impractical": (
                "SARIMA(p,d,q)(P,D,Q)s with s=1008 is not a practical specification "
                "here: seasonal differencing of order 1008 removes a full week; "
                "training contains only about 5.4 weekly cycles (38 days); and "
                "likelihood estimation with lag 1008 is computationally heavy and "
                "statistically thin."
            ),
            "double_seasonal": (
                "A single SARIMA model cannot host two seasonal periods (144 and 1008) "
                "without leaving the standard SARIMA class. Dual seasonality is "
                "therefore not used."
            ),
        },
        "sarima_input": (
            "Univariate chronological training series at 10-minute frequency. "
            "No sliding-window matrix. Validation is reserved for later "
            "configuration selection; the test period is unused until the final experiment."
        ),
        "models_fitted": False,
    }


def config_snapshot() -> dict[str, Any]:
    bounds = {k: [str(a), str(b)] for k, (a, b) in split_bounds().items()}
    return {
        "forecasting_models": list(FORECASTING_MODELS),
        "naive_baseline": NAIVE_BASELINE_NAME,
        "target_squares": list(TARGET_SQUARES),
        "eda_squares_not_in_forecasting_experiment": [
            s for s in EDA_SQUARES if s not in TARGET_SQUARES
        ],
        "target_column": TARGET_COLUMN,
        "frequency": FREQUENCY,
        "one_step_ahead_minutes": INTERVAL_MINUTES,
        "split_bounds_utc_inclusive": bounds,
        "expected_rows": dict(EXPECTED_ROWS),
        "primary_sequence_length": PRIMARY_SEQUENCE_LENGTH,
        "sequence_length_candidates": list(SEQUENCE_LENGTH_CANDIDATES),
        "scaler_type": SCALER_TYPE,
        "scaler_fitted_on": "train_only",
        "mape_denominator_floor": MAPE_DENOMINATOR_FLOOR,
        "mape_near_zero_threshold_diagnostic": MAPE_NEAR_ZERO_THRESHOLD,
        "models_trained_in_phase_5": False,
    }


def prepare_phase5_experiment(
    *,
    root: Optional[Path] = None,
    source_path: Optional[Path | str] = None,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """
    Run the Phase 5 preparation pipeline. Stops on integrity failure.
    Does not train SARIMA, LSTM, or TCN.
    """
    root = root or _project_root()
    source = load_source_target_timeseries(source_path, root=root)
    dataset = build_forecasting_dataset(source, squares=TARGET_SQUARES)
    validation = validate_forecasting_dataset(dataset, squares=TARGET_SQUARES)
    zeros = inspect_zero_traffic(dataset, squares=TARGET_SQUARES)
    scalers = fit_square_scalers(dataset, squares=TARGET_SQUARES)
    seq_analysis = analyse_sequence_lengths(dataset, squares=TARGET_SQUARES)
    alignment_frames = [
        alignment_example(dataset, square_id=sid) for sid in TARGET_SQUARES
    ]
    alignment = pd.concat(alignment_frames, ignore_index=True)
    sarima_note = assess_sarima_seasonal_period()
    snapshot = config_snapshot()
    scaler_params = {str(sid): sc.to_dict() for sid, sc in scalers.items()}

    leftover = source.loc[source["square_id"].isin(TARGET_SQUARES)].copy()
    leftover = assign_splits(leftover)
    n_unused = int(leftover["split"].isna().sum())

    outputs: dict[str, str] = {}
    if write_outputs:
        ds_path = _resolve(FORECASTING_DATASET_FILE, root=root)
        ds_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(ds_path, index=False)
        outputs["forecasting_dataset"] = str(ds_path)

        val_path = _resolve(SPLIT_VALIDATION_FILE, root=root)
        val_path.parent.mkdir(parents=True, exist_ok=True)
        validation.to_csv(val_path, index=False)
        outputs["split_validation"] = str(val_path)

        seq_path = _resolve(SEQUENCE_LENGTH_ANALYSIS_FILE, root=root)
        seq_analysis.to_csv(seq_path, index=False)
        outputs["sequence_length_analysis"] = str(seq_path)

        zero_path = _resolve(ZERO_INSPECTION_FILE, root=root)
        zeros.to_csv(zero_path, index=False)
        outputs["zero_inspection"] = str(zero_path)

        align_path = _resolve(ALIGNMENT_EXAMPLE_FILE, root=root)
        alignment.to_csv(align_path, index=False)
        outputs["alignment_example"] = str(align_path)

        scaler_path = _resolve(SCALER_PARAMS_FILE, root=root)
        with open(scaler_path, "w", encoding="utf-8") as f:
            json.dump(scaler_params, f, indent=2)
        outputs["scaler_params"] = str(scaler_path)

        cfg_path = _resolve(EXPERIMENT_CONFIG_SNAPSHOT_FILE, root=root)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        outputs["experiment_config"] = str(cfg_path)

    return {
        "dataset": dataset,
        "validation": validation,
        "zero_inspection": zeros,
        "scalers": scalers,
        "sequence_length_analysis": seq_analysis,
        "alignment_example": alignment,
        "sarima": sarima_note,
        "config": snapshot,
        "scaler_params": scaler_params,
        "n_unused_source_rows": n_unused,
        "outputs": outputs,
        "models_trained": False,
    }
