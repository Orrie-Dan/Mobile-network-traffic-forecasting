"""
Central experiment configuration for one-step-ahead forecasting.

Phase 5 uses these values as the single source of truth. Do not copy split
dates, target squares, or frequency constants into ad-hoc scripts.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Locked research models (do not change)
# ---------------------------------------------------------------------------
FORECASTING_MODELS: tuple[str, ...] = ("SARIMA", "LSTM", "TCN")
NAIVE_BASELINE_NAME: str = "Naive persistence baseline"

# ---------------------------------------------------------------------------
# Forecasting areas and target
# ---------------------------------------------------------------------------
# Highest-total-traffic squares from Phase 3A. Assignment squares 4159 and 4556
# remain in the EDA extract but are not part of the forecasting experiment.
TARGET_SQUARES: list[int] = [5161, 5059, 5259]
EDA_SQUARES: list[int] = [5161, 5059, 5259, 4159, 4556]
TARGET_COLUMN: str = "internet_traffic"

# ---------------------------------------------------------------------------
# Temporal resolution
# ---------------------------------------------------------------------------
FREQUENCY: str = "10min"
INTERVAL_MINUTES: int = 10
INTERVAL_MS: int = 600_000
BINS_PER_DAY: int = 144
BINS_PER_WEEK: int = 1008
STEP_AHEAD: int = 1  # one 10-minute step

# ---------------------------------------------------------------------------
# Chronological split (UTC timestamps, inclusive)
# ---------------------------------------------------------------------------
# Applied to the UTC ``timestamp`` column, not the filename ``date`` column.
TRAIN_START: str = "2013-11-01 00:00"
TRAIN_END: str = "2013-12-08 23:50"
VALIDATION_START: str = "2013-12-09 00:00"
VALIDATION_END: str = "2013-12-15 23:50"
TEST_START: str = "2013-12-16 00:00"
TEST_END: str = "2013-12-22 23:50"

EXPECTED_ROWS: dict[str, int] = {
    "train": 38 * BINS_PER_DAY,  # 2013-11-01 .. 2013-12-08 → 5472
    "validation": 7 * BINS_PER_DAY,  # 1008
    "test": 7 * BINS_PER_DAY,  # 1008
}

# ---------------------------------------------------------------------------
# LSTM / TCN sequence length
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH_CANDIDATES: tuple[int, ...] = (144, 288, 1008)
# Primary length selected in Phase 5 (see docs/methodology.md). Same for LSTM and TCN.
PRIMARY_SEQUENCE_LENGTH: int = 144

# ---------------------------------------------------------------------------
# Normalization (neural models only)
# ---------------------------------------------------------------------------
SCALER_TYPE: str = "minmax"
SCALER_FEATURE_RANGE: tuple[float, float] = (0.0, 1.0)

# ---------------------------------------------------------------------------
# MAPE zero / near-zero handling
# ---------------------------------------------------------------------------
# Denominator floor retains every observation. MAE/RMSE never use this floor.
MAPE_DENOMINATOR_FLOOR: float = 1e-8
MAPE_NEAR_ZERO_THRESHOLD: float = 1.0  # diagnostic only; does not drop rows

# ---------------------------------------------------------------------------
# Paths (relative to repository root)
# ---------------------------------------------------------------------------
PROCESSED_SOURCE_FILE: str = "data/processed/target_square_internet_timeseries.csv"
FORECASTING_DATASET_FILE: str = "data/processed/forecasting_target_squares.csv"
SPLIT_VALIDATION_FILE: str = "results/metrics/phase5_split_validation.csv"
SEQUENCE_LENGTH_ANALYSIS_FILE: str = "results/metrics/phase5_sequence_length_analysis.csv"
ZERO_INSPECTION_FILE: str = "results/metrics/phase5_zero_traffic_inspection.csv"
SCALER_PARAMS_FILE: str = "results/metrics/phase5_scaler_params.json"
EXPERIMENT_CONFIG_SNAPSHOT_FILE: str = "results/metrics/phase5_experiment_config.json"
ALIGNMENT_EXAMPLE_FILE: str = "results/metrics/phase5_alignment_example.csv"

# ---------------------------------------------------------------------------
# Phase 6A — model defaults (configuration selection is Phase 6B; test unused)
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# SARIMA: seasonal period fixed at 144 (daily cycle at 10-minute resolution).
SARIMA_SEASONAL_PERIOD: int = BINS_PER_DAY  # 144
SARIMA_DEFAULT_ORDER: tuple[int, int, int] = (1, 0, 1)  # (p, d, q)
SARIMA_DEFAULT_SEASONAL_ORDER: tuple[int, int, int, int] = (1, 0, 1, SARIMA_SEASONAL_PERIOD)
SARIMA_FIT_METHOD: str = "lbfgs"
SARIMA_MAXITER: int = 50

# Legacy flat lists retained for documentation/backward compatibility.
# Phase 6B uses the explicit named candidate dicts below (not a Cartesian product).
SARIMA_ORDER_CANDIDATES: tuple[tuple[int, int, int], ...] = (
    (1, 0, 1),
    (1, 0, 0),
    (0, 0, 1),
)
SARIMA_SEASONAL_ORDER_CANDIDATES: tuple[tuple[int, int, int, int], ...] = (
    (1, 0, 1, SARIMA_SEASONAL_PERIOD),
    (0, 0, 1, SARIMA_SEASONAL_PERIOD),
)

# LSTM defaults — intentionally compact one-step architecture.
LSTM_UNITS: int = 32
LSTM_LAYERS: int = 1
LSTM_DROPOUT: float = 0.1
LSTM_LEARNING_RATE: float = 1e-3
LSTM_BATCH_SIZE: int = 64
LSTM_EPOCHS: int = 20
LSTM_EARLY_STOPPING_PATIENCE: int = 5

# TCN defaults — compact causal dilated residual network.
TCN_FILTERS: int = 32
TCN_KERNEL_SIZE: int = 3
TCN_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)
TCN_DROPOUT: float = 0.1
TCN_LEARNING_RATE: float = 1e-3
TCN_BATCH_SIZE: int = 64
TCN_EPOCHS: int = 20
TCN_EARLY_STOPPING_PATIENCE: int = 5

# Temporary checkpoints (gitignored directories only; never commit binaries).
MODEL_CHECKPOINT_DIR: str = "tmp/model_checkpoints"

# ---------------------------------------------------------------------------
# Phase 6B — small validation-based candidate sets (NOT locked configs)
# ---------------------------------------------------------------------------
# Selection uses train for fitting and validation for scoring only.
# The Dec 16–22 test period is never used here.
#
# Normalization for cross-square aggregation (training statistics only):
#   normalized_MAE  = validation_MAE  / training_mean_traffic
#   normalized_RMSE = validation_RMSE / training_mean_traffic

PHASE6B_SELECTION_PRIMARY: str = "mean_normalized_validation_mae"
PHASE6B_SELECTION_SECONDARY: str = "mean_normalized_validation_rmse"

PHASE6B_SARIMA_CANDIDATES: dict[str, dict] = {
    # No d=1 / D=1 in this set: seasonal period 144 already makes estimation
    # expensive; differencing would further enlarge the state and was not
    # required for the Phase 5 one-step design review.
    "A": {
        "order": (1, 0, 1),
        "seasonal_order": (1, 0, 1, SARIMA_SEASONAL_PERIOD),
        "method": SARIMA_FIT_METHOD,
    },
    "B": {
        "order": (1, 0, 0),
        "seasonal_order": (1, 0, 1, SARIMA_SEASONAL_PERIOD),
        "method": SARIMA_FIT_METHOD,
    },
    "C": {
        "order": (0, 0, 1),
        "seasonal_order": (1, 0, 1, SARIMA_SEASONAL_PERIOD),
        "method": SARIMA_FIT_METHOD,
    },
    "D": {
        "order": (1, 0, 1),
        "seasonal_order": (0, 0, 1, SARIMA_SEASONAL_PERIOD),
        "method": SARIMA_FIT_METHOD,
    },
}
# Full-length SARIMAX(s=144) MLE on all 5472 train points is computationally
# heavy. For Phase 6B *order selection only*, fit on the most recent
# PHASE6B_SARIMA_SELECTION_TRAIN_DAYS of the training split (still strictly
# before validation/test). Phase 6C should refit the locked order on the full
# training series before final testing.
PHASE6B_SARIMA_SELECTION_TRAIN_DAYS: int = 7
PHASE6B_SARIMA_SELECTION_MAXITER: int = 15


PHASE6B_LSTM_CANDIDATES: dict[str, dict] = {
    "A": {
        "units": 32,
        "n_layers": 1,
        "dropout": 0.1,
        "learning_rate": 1e-3,
    },
    "B": {
        "units": 64,
        "n_layers": 1,
        "dropout": 0.1,
        "learning_rate": 1e-3,
    },
    "C": {
        "units": 32,
        "n_layers": 2,
        "dropout": 0.1,
        "learning_rate": 1e-3,
    },
    "D": {
        "units": 64,
        "n_layers": 1,
        "dropout": 0.2,
        "learning_rate": 5e-4,
    },
}

PHASE6B_TCN_CANDIDATES: dict[str, dict] = {
    "A": {
        "filters": 32,
        "kernel_size": 3,
        "dilations": (1, 2, 4, 8),
        "dropout": 0.1,
        "learning_rate": 1e-3,
    },
    "B": {
        "filters": 32,
        "kernel_size": 3,
        "dilations": (1, 2, 4, 8, 16, 32),
        "dropout": 0.1,
        "learning_rate": 1e-3,
    },
    "C": {
        "filters": 64,
        "kernel_size": 3,
        "dilations": (1, 2, 4, 8, 16, 32),
        "dropout": 0.1,
        "learning_rate": 1e-3,
    },
    "D": {
        "filters": 32,
        "kernel_size": 5,
        "dilations": (1, 2, 4, 8, 16),
        "dropout": 0.1,
        "learning_rate": 5e-4,
    },
}

# Output paths for Phase 6B artefacts
PHASE6B_CANDIDATE_RESULTS_FILE: str = "results/metrics/phase6b_candidate_results.csv"
PHASE6B_SELECTION_SUMMARY_FILE: str = "results/metrics/phase6b_selection_summary.csv"
PHASE6B_SQUARE_RESULTS_FILE: str = "results/metrics/phase6b_square_results.csv"
PHASE6B_CONFIGURATIONS_FILE: str = "results/metrics/phase6b_configurations.json"
PHASE6B_EXPERIMENT_METADATA_FILE: str = "results/metrics/phase6b_experiment_metadata.json"
PHASE6B_VALIDATION_REPORT_FILE: str = "results/metrics/phase6b_validation_report.md"

# ---------------------------------------------------------------------------
# Phase 6C — final test evaluation of locked configs
# ---------------------------------------------------------------------------
# Consumes Phase 6B locked configurations. Refits each locked model on the
# full training split (SARIMA: full 5472 rows + SARIMA_MAXITER; neural: train
# fit with validation early stopping). Scores the reserved Dec 16–22 test week
# once. No hyperparameter search and no cross-family selection on the test set.
#
# Fallback locked picks (Colab Phase 6B run) used only when no locked JSON is
# found; prefer PHASE6B_CONFIGURATIONS_FILE or Drive locked_configs.json.

PHASE6C_DEFAULT_LOCKED: dict[str, dict] = {
    "SARIMA": {
        "candidate_id": "A",
        "order": (1, 0, 1),
        "seasonal_order": (1, 0, 1, SARIMA_SEASONAL_PERIOD),
        "method": SARIMA_FIT_METHOD,
        "maxiter": SARIMA_MAXITER,
    },
    "LSTM": {
        "candidate_id": "B",
        "units": 64,
        "n_layers": 1,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "batch_size": LSTM_BATCH_SIZE,
        "epochs": LSTM_EPOCHS,
        "early_stopping_patience": LSTM_EARLY_STOPPING_PATIENCE,
        "sequence_length": PRIMARY_SEQUENCE_LENGTH,
        "seed": RANDOM_SEED,
    },
    "TCN": {
        "candidate_id": "D",
        "filters": 32,
        "kernel_size": 5,
        "dilations": (1, 2, 4, 8, 16),
        "dropout": 0.1,
        "learning_rate": 5e-4,
        "batch_size": TCN_BATCH_SIZE,
        "epochs": TCN_EPOCHS,
        "early_stopping_patience": TCN_EARLY_STOPPING_PATIENCE,
        "sequence_length": PRIMARY_SEQUENCE_LENGTH,
        "seed": RANDOM_SEED,
    },
}

PHASE6C_SQUARE_RESULTS_FILE: str = "results/metrics/phase6c_square_results.csv"
PHASE6C_SUMMARY_FILE: str = "results/metrics/phase6c_summary.csv"
PHASE6C_COMPARISON_FILE: str = "results/metrics/phase6c_comparison.json"
PHASE6C_EXPERIMENT_METADATA_FILE: str = "results/metrics/phase6c_experiment_metadata.json"
PHASE6C_REPORT_FILE: str = "results/metrics/phase6c_test_report.md"
PHASE6C_PREDICTIONS_DIR: str = "results/metrics/phase6c_predictions"


def utc_timestamp(value: str) -> pd.Timestamp:
    """Parse a config timestamp and localise to UTC if naive."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def split_bounds() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Inclusive UTC start/end for each chronological split."""
    return {
        "train": (utc_timestamp(TRAIN_START), utc_timestamp(TRAIN_END)),
        "validation": (utc_timestamp(VALIDATION_START), utc_timestamp(VALIDATION_END)),
        "test": (utc_timestamp(TEST_START), utc_timestamp(TEST_END)),
    }


def expected_index(split: str) -> pd.DatetimeIndex:
    """Complete 10-minute UTC grid for a named split."""
    start, end = split_bounds()[split]
    return pd.date_range(start=start, end=end, freq=FREQUENCY, tz="UTC")


def label_split(timestamp: pd.Timestamp) -> Optional[str]:
    """Return train/validation/test or None if the timestamp is outside the experiment."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    for name, (start, end) in split_bounds().items():
        if start <= ts <= end:
            return name
    return None
