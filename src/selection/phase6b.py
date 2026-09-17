"""
Phase 6B — validation-based configuration selection utilities.

Selects ONE locked configuration per model family using train fitting and
validation scoring only. The reserved Dec 16–22 test period is never used.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from src import config as cfg
from src.data.forecasting import (
    SquareScaler,
    fit_square_scalers,
    make_supervised_sequences,
    univariate_series,
)
from src.evaluation.metrics import mae, mape, rmse
from src.models.base import assert_no_test_in_fit_window, assert_one_step_alignment
from src.models.lstm import LSTMConfig, LSTMModel
from src.models.naive import NaivePersistenceModel
from src.models.sarima import SARIMAConfig, SARIMAModel
from src.models.tcn import TCNConfig, TCNModel
from src.models.training import set_global_seeds
import gc


def _clear_keras_session() -> None:
    """Release graph/memory between successive neural candidate fits."""
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_forecasting_frame(root: Optional[Path] = None) -> pd.DataFrame:
    root = root or project_root()
    path = root / cfg.FORECASTING_DATASET_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Phase 5 forecasting dataset: {path}. "
            "Run Phase 5 preparation before Phase 6B."
        )
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    df["square_id"] = df["square_id"].astype(int)
    # Hard refuse if any row is labelled test but sneaks into selection helpers later.
    if "split" not in df.columns:
        raise ValueError("forecasting dataset missing split column")
    return df.sort_values(["square_id", "timestamp"]).reset_index(drop=True)


def assert_phase5_protocol(df: pd.DataFrame, squares: Sequence[int]) -> None:
    """Confirm Phase 5 splits / squares are unchanged and unused for test selection."""
    present = sorted(int(s) for s in df["square_id"].unique())
    expected = sorted(int(s) for s in squares)
    if present != expected:
        raise AssertionError(f"Unexpected squares {present}; expected {expected}")
    for sid in expected:
        sub = df.loc[df["square_id"] == sid]
        counts = sub["split"].value_counts().to_dict()
        if int(counts.get("train", 0)) != cfg.EXPECTED_ROWS["train"]:
            raise AssertionError(f"square {sid}: unexpected train rows {counts.get('train')}")
        if int(counts.get("validation", 0)) != cfg.EXPECTED_ROWS["validation"]:
            raise AssertionError(
                f"square {sid}: unexpected validation rows {counts.get('validation')}"
            )
        if int(counts.get("test", 0)) != cfg.EXPECTED_ROWS["test"]:
            raise AssertionError(f"square {sid}: unexpected test rows {counts.get('test')}")
        train_ts = sub.loc[sub["split"] == "train", "timestamp"]
        val_ts = sub.loc[sub["split"] == "validation", "timestamp"]
        test_start = cfg.utc_timestamp(cfg.TEST_START)
        if (train_ts >= test_start).any() or (val_ts >= test_start).any():
            raise AssertionError(f"square {sid}: train/validation overlaps test period")
    if cfg.PRIMARY_SEQUENCE_LENGTH != 144:
        raise AssertionError("Sequence length must remain 144 in Phase 6B")


def training_mean_traffic(df: pd.DataFrame, square_id: int) -> float:
    """Training-only mean used as the normalization scale for selection."""
    vals = df.loc[
        (df["square_id"] == square_id) & (df["split"] == "train"),
        cfg.TARGET_COLUMN,
    ].astype("float64")
    if vals.empty:
        raise ValueError(f"No training rows for square {square_id}")
    mean = float(vals.mean())
    if mean <= 0:
        raise ValueError(f"Non-positive training mean for square {square_id}: {mean}")
    return mean


def planned_experiment_size(
    *,
    models: Sequence[str],
    squares: Sequence[int],
) -> dict[str, Any]:
    n_sarima = len(cfg.PHASE6B_SARIMA_CANDIDATES) if "SARIMA" in models else 0
    n_lstm = len(cfg.PHASE6B_LSTM_CANDIDATES) if "LSTM" in models else 0
    n_tcn = len(cfg.PHASE6B_TCN_CANDIDATES) if "TCN" in models else 0
    n_naive = 1 if "NAIVE" in models or "Naive" in models else 0
    # Always evaluate naive as reference when running the full experiment
    fits = {
        "SARIMA_fits": n_sarima * len(squares),
        "LSTM_fits": n_lstm * len(squares),
        "TCN_fits": n_tcn * len(squares),
        "naive_evals": len(squares),
        "total_research_candidate_square_runs": (n_sarima + n_lstm + n_tcn) * len(squares),
    }
    return {
        "squares": list(squares),
        "n_squares": len(squares),
        "sarima_candidates": n_sarima,
        "lstm_candidates": n_lstm,
        "tcn_candidates": n_tcn,
        **fits,
        "note": "Test split is not included in any fit or selection metric.",
    }


def _failed_row(base: dict[str, Any], error: Exception) -> dict[str, Any]:
    out = dict(base)
    out.update(
        {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "fit_time_seconds": base.get("fit_time_seconds"),
            "prediction_time_seconds": None,
            "validation_mae": None,
            "validation_rmse": None,
            "validation_mape": None,
            "normalized_validation_mae": None,
            "normalized_validation_rmse": None,
        }
    )
    return out


def _metric_fields(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    training_mean: float,
) -> dict[str, float]:
    v_mae = mae(y_true, y_pred)
    v_rmse = rmse(y_true, y_pred)
    v_mape = mape(y_true, y_pred)
    return {
        "validation_mae": v_mae,
        "validation_rmse": v_rmse,
        "validation_mape": v_mape,
        "training_mean": training_mean,
        "normalized_validation_mae": v_mae / training_mean,
        "normalized_validation_rmse": v_rmse / training_mean,
    }


def evaluate_naive_validation(
    df: pd.DataFrame,
    *,
    squares: Sequence[int] = cfg.TARGET_SQUARES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid in squares:
        train = univariate_series(df, sid, split="train")
        val = univariate_series(df, sid, split="validation")
        # History available for persistence through validation (no test).
        history = pd.concat([train, val]).sort_index()
        assert_no_test_in_fit_window(history.index, context="naive validation history")
        model = NaivePersistenceModel()
        model.fit(history)
        preds = model.predict(target_timestamps=val.index)
        y_true = val.to_numpy(dtype="float64")
        y_pred = preds.to_numpy(dtype="float64")
        # Alignment: first val target uses last train observation
        assert preds.index[0] == cfg.utc_timestamp(cfg.VALIDATION_START)
        assert history.index[history.index.get_loc(preds.index[0]) - 1] == cfg.utc_timestamp(
            cfg.TRAIN_END
        )
        tmean = training_mean_traffic(df, sid)
        row = {
            "model": "Naive",
            "candidate_id": "persistence",
            "square_id": int(sid),
            "status": "ok",
            "fit_time_seconds": model.timing.fit_seconds,
            "prediction_time_seconds": model.timing.predict_seconds,
            "is_research_model": False,
            **_metric_fields(y_true, y_pred, training_mean=tmean),
        }
        rows.append(row)
    return rows


def evaluate_sarima_candidates(
    df: pd.DataFrame,
    *,
    squares: Sequence[int] = cfg.TARGET_SQUARES,
    candidates: Optional[dict[str, dict]] = None,
) -> list[dict[str, Any]]:
    candidates = candidates or cfg.PHASE6B_SARIMA_CANDIDATES
    rows: list[dict[str, Any]] = []
    for cand_id, params in candidates.items():
        order = tuple(params["order"])
        seasonal_order = tuple(params["seasonal_order"])
        maxiter = int(params.get("maxiter", cfg.PHASE6B_SARIMA_SELECTION_MAXITER))
        method = str(params.get("method", cfg.SARIMA_FIT_METHOD))
        for sid in squares:
            base = {
                "model": "SARIMA",
                "candidate_id": cand_id,
                "square_id": int(sid),
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "method": method,
                "maxiter": maxiter,
                "is_research_model": True,
                "training_mean": training_mean_traffic(df, sid),
            }
            print(
                f"  SARIMA candidate {cand_id} square {sid} "
                f"order={order} seasonal={seasonal_order} "
                f"selection_days={cfg.PHASE6B_SARIMA_SELECTION_TRAIN_DAYS} ...",
                flush=True,
            )
            try:
                train_full = univariate_series(df, sid, split="train")
                val = univariate_series(df, sid, split="validation")
                n_sel = int(cfg.PHASE6B_SARIMA_SELECTION_TRAIN_DAYS) * cfg.BINS_PER_DAY
                train = train_full.iloc[-n_sel:] if len(train_full) > n_sel else train_full
                assert_no_test_in_fit_window(train.index, context="SARIMA.fit")
                config = SARIMAConfig(
                    order=order,  # type: ignore[arg-type]
                    seasonal_order=seasonal_order,  # type: ignore[arg-type]
                    method=method,
                    maxiter=maxiter,
                )
                model = SARIMAModel(config)
                model.fit(train)
                # Extend fitted train with validation observations only (memory-safe).
                history = pd.concat([train, val]).sort_index()
                assert_no_test_in_fit_window(history.index, context="SARIMA.predict history")
                preds = model.predict_one_step(history, target_timestamps=val.index)
                assert preds.index[0] == cfg.utc_timestamp(cfg.VALIDATION_START)
                assert train_full.index[-1] == cfg.utc_timestamp(cfg.TRAIN_END)
                y_true = val.to_numpy(dtype="float64")
                y_pred = preds.to_numpy(dtype="float64")
                metrics = _metric_fields(y_true, y_pred, training_mean=base["training_mean"])
                row = {
                    **base,
                    "status": "ok",
                    "convergence_status": "success",
                    "fit_warning": model.fit_warning_,
                    "fit_time_seconds": model.timing.fit_seconds,
                    "prediction_time_seconds": model.timing.predict_seconds,
                    "selection_train_rows": int(len(train)),
                    "selection_train_days": cfg.PHASE6B_SARIMA_SELECTION_TRAIN_DAYS,
                    "selection_note": (
                        "Fitted on last "
                        f"{cfg.PHASE6B_SARIMA_SELECTION_TRAIN_DAYS} training days "
                        "for order selection only; full-train refit deferred to Phase 6C."
                    ),
                    **metrics,
                }
                rows.append(row)
                del model, train, val, history, preds
                gc.collect()
            except Exception as exc:  # noqa: BLE001 - record failed candidates
                failed = _failed_row(base, exc)
                failed["convergence_status"] = "failed"
                failed["fit_warning"] = str(exc)
                rows.append(failed)
                print(f"    FAILED: {exc}", flush=True)
                gc.collect()
    return rows


def _scaled_frame(df: pd.DataFrame, scalers: dict[int, SquareScaler]) -> pd.DataFrame:
    out = df.copy()
    scaled = np.empty(len(out), dtype="float64")
    scaled[:] = np.nan
    for sid, scaler in scalers.items():
        mask = out["square_id"] == sid
        scaled[mask.to_numpy()] = scaler.transform(
            out.loc[mask, cfg.TARGET_COLUMN].to_numpy(dtype="float64")
        )
    out[cfg.TARGET_COLUMN] = scaled
    return out


def _neural_sequences_for_square(
    df: pd.DataFrame,
    scaled_df: pd.DataFrame,
    square_id: int,
) -> dict[str, Any]:
    train_seq = make_supervised_sequences(
        scaled_df,
        square_id=square_id,
        sequence_length=cfg.PRIMARY_SEQUENCE_LENGTH,
        target_split="train",
    )
    val_seq = make_supervised_sequences(
        scaled_df,
        square_id=square_id,
        sequence_length=cfg.PRIMARY_SEQUENCE_LENGTH,
        target_split="validation",
    )
    assert_no_test_in_fit_window(train_seq["target_timestamp"], context="neural train targets")
    assert_no_test_in_fit_window(val_seq["target_timestamp"], context="neural val targets")
    assert_one_step_alignment(val_seq["input_end_timestamp"], val_seq["target_timestamp"])
    # First validation target must be validation start; input end is 10 min earlier.
    assert val_seq["target_timestamp"][0] == cfg.utc_timestamp(cfg.VALIDATION_START)
    assert val_seq["input_end_timestamp"][0] == cfg.utc_timestamp(cfg.TRAIN_END)
    # Original-scale validation targets for metrics
    val_orig = make_supervised_sequences(
        df,
        square_id=square_id,
        sequence_length=cfg.PRIMARY_SEQUENCE_LENGTH,
        target_split="validation",
    )
    return {
        "X_train": train_seq["X"],
        "y_train": train_seq["y"],
        "train_targets": train_seq["target_timestamp"],
        "X_val": val_seq["X"],
        "y_val_scaled": val_seq["y"],
        "val_targets": val_seq["target_timestamp"],
        "y_val_original": val_orig["y"],
    }


def evaluate_lstm_candidates(
    df: pd.DataFrame,
    scalers: dict[int, SquareScaler],
    *,
    squares: Sequence[int] = cfg.TARGET_SQUARES,
    candidates: Optional[dict[str, dict]] = None,
    seed: int = cfg.RANDOM_SEED,
) -> list[dict[str, Any]]:
    candidates = candidates or cfg.PHASE6B_LSTM_CANDIDATES
    scaled_df = _scaled_frame(df, scalers)
    rows: list[dict[str, Any]] = []
    for cand_id, params in candidates.items():
        for sid in squares:
            tmean = training_mean_traffic(df, sid)
            base = {
                "model": "LSTM",
                "candidate_id": cand_id,
                "square_id": int(sid),
                "units": int(params["units"]),
                "n_layers": int(params["n_layers"]),
                "dropout": float(params["dropout"]),
                "learning_rate": float(params["learning_rate"]),
                "batch_size": cfg.LSTM_BATCH_SIZE,
                "max_epochs": cfg.LSTM_EPOCHS,
                "sequence_length": cfg.PRIMARY_SEQUENCE_LENGTH,
                "seed": seed,
                "is_research_model": True,
                "training_mean": tmean,
            }
            print(
                f"  LSTM candidate {cand_id} square {sid} "
                f"units={params['units']} layers={params['n_layers']} ...",
                flush=True,
            )
            try:
                set_global_seeds(seed)
                data = _neural_sequences_for_square(df, scaled_df, int(sid))
                config = LSTMConfig(
                    units=int(params["units"]),
                    n_layers=int(params["n_layers"]),
                    dropout=float(params["dropout"]),
                    learning_rate=float(params["learning_rate"]),
                    batch_size=cfg.LSTM_BATCH_SIZE,
                    epochs=cfg.LSTM_EPOCHS,
                    early_stopping_patience=cfg.LSTM_EARLY_STOPPING_PATIENCE,
                    seed=seed,
                )
                model = LSTMModel(config, scaler=scalers[int(sid)])
                model.fit(
                    data["X_train"],
                    data["y_train"],
                    X_val=data["X_val"],
                    y_val=data["y_val_scaled"],
                    train_target_timestamps=data["train_targets"],
                    val_target_timestamps=data["val_targets"],
                    verbose=0,
                )
                y_pred = model.predict(data["X_val"])
                metrics = _metric_fields(
                    data["y_val_original"], y_pred, training_mean=tmean
                )
                best_val_loss = (
                    float(min(model.history_.validation_loss))
                    if model.history_.validation_loss
                    else None
                )
                rows.append(
                    {
                        **base,
                        "status": "ok",
                        "fit_time_seconds": model.timing.fit_seconds,
                        "prediction_time_seconds": model.timing.predict_seconds,
                        "actual_epochs": model.history_.epochs_completed,
                        "best_epoch": model.history_.best_epoch,
                        "best_validation_loss": best_val_loss,
                        **metrics,
                    }
                )
                del model
            except Exception as exc:  # noqa: BLE001
                rows.append(_failed_row(base, exc))
                print(f"    FAILED: {exc}", flush=True)
            finally:
                _clear_keras_session()
    return rows


def evaluate_tcn_candidates(
    df: pd.DataFrame,
    scalers: dict[int, SquareScaler],
    *,
    squares: Sequence[int] = cfg.TARGET_SQUARES,
    candidates: Optional[dict[str, dict]] = None,
    seed: int = cfg.RANDOM_SEED,
) -> list[dict[str, Any]]:
    candidates = candidates or cfg.PHASE6B_TCN_CANDIDATES
    scaled_df = _scaled_frame(df, scalers)
    rows: list[dict[str, Any]] = []
    for cand_id, params in candidates.items():
        dilations = tuple(params["dilations"])
        tmp_cfg = TCNConfig(
            filters=int(params["filters"]),
            kernel_size=int(params["kernel_size"]),
            dilations=dilations,
            dropout=float(params["dropout"]),
            learning_rate=float(params["learning_rate"]),
        )
        rf_steps = tmp_cfg.receptive_field()
        rf_minutes = rf_steps * cfg.INTERVAL_MINUTES
        for sid in squares:
            tmean = training_mean_traffic(df, sid)
            base = {
                "model": "TCN",
                "candidate_id": cand_id,
                "square_id": int(sid),
                "filters": int(params["filters"]),
                "kernel_size": int(params["kernel_size"]),
                "dilations": str(dilations),
                "dropout": float(params["dropout"]),
                "learning_rate": float(params["learning_rate"]),
                "batch_size": cfg.TCN_BATCH_SIZE,
                "max_epochs": cfg.TCN_EPOCHS,
                "sequence_length": cfg.PRIMARY_SEQUENCE_LENGTH,
                "receptive_field_steps": rf_steps,
                "receptive_field_minutes": rf_minutes,
                "seed": seed,
                "is_research_model": True,
                "training_mean": tmean,
            }
            print(
                f"  TCN candidate {cand_id} square {sid} "
                f"filters={params['filters']} RF={rf_steps} ...",
                flush=True,
            )
            try:
                set_global_seeds(seed)
                data = _neural_sequences_for_square(df, scaled_df, int(sid))
                config = TCNConfig(
                    filters=int(params["filters"]),
                    kernel_size=int(params["kernel_size"]),
                    dilations=dilations,
                    dropout=float(params["dropout"]),
                    learning_rate=float(params["learning_rate"]),
                    batch_size=cfg.TCN_BATCH_SIZE,
                    epochs=cfg.TCN_EPOCHS,
                    early_stopping_patience=cfg.TCN_EARLY_STOPPING_PATIENCE,
                    seed=seed,
                )
                model = TCNModel(config, scaler=scalers[int(sid)])
                model.fit(
                    data["X_train"],
                    data["y_train"],
                    X_val=data["X_val"],
                    y_val=data["y_val_scaled"],
                    train_target_timestamps=data["train_targets"],
                    val_target_timestamps=data["val_targets"],
                    verbose=0,
                )
                y_pred = model.predict(data["X_val"])
                metrics = _metric_fields(
                    data["y_val_original"], y_pred, training_mean=tmean
                )
                best_val_loss = (
                    float(min(model.history_.validation_loss))
                    if model.history_.validation_loss
                    else None
                )
                rows.append(
                    {
                        **base,
                        "status": "ok",
                        "fit_time_seconds": model.timing.fit_seconds,
                        "prediction_time_seconds": model.timing.predict_seconds,
                        "actual_epochs": model.history_.epochs_completed,
                        "best_epoch": model.history_.best_epoch,
                        "best_validation_loss": best_val_loss,
                        **metrics,
                    }
                )
                del model
            except Exception as exc:  # noqa: BLE001
                rows.append(_failed_row(base, exc))
                print(f"    FAILED: {exc}", flush=True)
            finally:
                _clear_keras_session()
    return rows


def _candidate_complexity(model: str, candidate_id: str) -> tuple:
    """Lower is simpler — used as final tie-breaker."""
    if model == "SARIMA":
        params = cfg.PHASE6B_SARIMA_CANDIDATES[candidate_id]
        order = params["order"]
        seas = params["seasonal_order"]
        return (sum(order) + sum(seas[:3]), candidate_id)
    if model == "LSTM":
        params = cfg.PHASE6B_LSTM_CANDIDATES[candidate_id]
        return (
            int(params["n_layers"]),
            int(params["units"]),
            float(params["dropout"]),
            candidate_id,
        )
    if model == "TCN":
        params = cfg.PHASE6B_TCN_CANDIDATES[candidate_id]
        return (
            len(params["dilations"]),
            int(params["filters"]),
            int(params["kernel_size"]),
            candidate_id,
        )
    return (candidate_id,)


def select_best_candidates(candidate_results: pd.DataFrame) -> pd.DataFrame:
    """
    Select one candidate per research model family.

    Primary: mean normalized validation MAE across squares (ok rows only).
    Secondary: mean normalized validation RMSE.
    Tie-breakers: lower mean raw MAE, lower total fit time, simpler config.
    """
    research = candidate_results.loc[
        candidate_results["model"].isin(["SARIMA", "LSTM", "TCN"])
    ].copy()
    summaries: list[dict[str, Any]] = []
    for model in ("SARIMA", "LSTM", "TCN"):
        sub = research.loc[research["model"] == model]
        if sub.empty:
            continue
        ranked: list[tuple] = []
        for cand_id, grp in sub.groupby("candidate_id"):
            ok = grp.loc[grp["status"] == "ok"]
            if len(ok) == 0:
                continue
            # Require successful results on all evaluated squares for selection
            if ok["square_id"].nunique() < sub["square_id"].nunique():
                # Still allow selection if some squares failed for all candidates;
                # prefer candidates with complete square coverage.
                coverage = int(ok["square_id"].nunique())
            else:
                coverage = int(ok["square_id"].nunique())
            mean_n_mae = float(ok["normalized_validation_mae"].mean())
            mean_n_rmse = float(ok["normalized_validation_rmse"].mean())
            mean_mae = float(ok["validation_mae"].mean())
            mean_rmse = float(ok["validation_rmse"].mean())
            mean_mape = float(ok["validation_mape"].mean())
            total_fit = float(ok["fit_time_seconds"].sum())
            ranked.append(
                (
                    -coverage,  # more coverage first (more negative sort key later → use reverse)
                    mean_n_mae,
                    mean_n_rmse,
                    mean_mae,
                    total_fit,
                    _candidate_complexity(model, str(cand_id)),
                    str(cand_id),
                    {
                        "model": model,
                        "selected_candidate_id": str(cand_id),
                        "selection_metric": cfg.PHASE6B_SELECTION_PRIMARY,
                        "n_squares_ok": coverage,
                        "mean_normalized_validation_mae": mean_n_mae,
                        "mean_normalized_validation_rmse": mean_n_rmse,
                        "mean_raw_validation_mae": mean_mae,
                        "mean_raw_validation_rmse": mean_rmse,
                        "mean_raw_validation_mape": mean_mape,
                        "total_fit_time_seconds": total_fit,
                    },
                )
            )
        if not ranked:
            summaries.append(
                {
                    "model": model,
                    "selected_candidate_id": None,
                    "selection_metric": cfg.PHASE6B_SELECTION_PRIMARY,
                    "n_squares_ok": 0,
                    "mean_normalized_validation_mae": None,
                    "mean_normalized_validation_rmse": None,
                    "mean_raw_validation_mae": None,
                    "mean_raw_validation_rmse": None,
                    "mean_raw_validation_mape": None,
                    "total_fit_time_seconds": None,
                    "note": "No successful candidates",
                }
            )
            continue
        # Sort: higher coverage first, then lower metrics / time / complexity
        ranked.sort(
            key=lambda t: (t[0], t[1], t[2], t[3], t[4], t[5], t[6])
        )
        best = ranked[0][-1]
        summaries.append(best)
    return pd.DataFrame(summaries)


def locked_configurations(selection_summary: pd.DataFrame) -> dict[str, Any]:
    locked: dict[str, Any] = {
        "phase": "6B",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": {
            "primary": cfg.PHASE6B_SELECTION_PRIMARY,
            "secondary": cfg.PHASE6B_SELECTION_SECONDARY,
            "tie_breakers": [
                "lower_mean_raw_validation_mae",
                "lower_total_fit_time_seconds",
                "simpler_configuration",
            ],
            "normalization": (
                "validation_error / training_mean_traffic "
                "(training statistics only; per square)"
            ),
            "aggregation": "mean across target squares",
            "test_period_used": False,
        },
        "models": {},
    }
    for _, row in selection_summary.iterrows():
        model = row["model"]
        cand = row["selected_candidate_id"]
        if cand is None or (isinstance(cand, float) and np.isnan(cand)):
            locked["models"][model] = {"status": "unselected", "candidate_id": None}
            continue
        if model == "SARIMA":
            params = dict(cfg.PHASE6B_SARIMA_CANDIDATES[str(cand)])
            params["order"] = list(params["order"])
            params["seasonal_order"] = list(params["seasonal_order"])
        elif model == "LSTM":
            params = dict(cfg.PHASE6B_LSTM_CANDIDATES[str(cand)])
            params.update(
                {
                    "sequence_length": cfg.PRIMARY_SEQUENCE_LENGTH,
                    "batch_size": cfg.LSTM_BATCH_SIZE,
                    "epochs": cfg.LSTM_EPOCHS,
                    "early_stopping_patience": cfg.LSTM_EARLY_STOPPING_PATIENCE,
                    "seed": cfg.RANDOM_SEED,
                }
            )
        elif model == "TCN":
            params = dict(cfg.PHASE6B_TCN_CANDIDATES[str(cand)])
            params["dilations"] = list(params["dilations"])
            tmp = TCNConfig(
                filters=int(params["filters"]),
                kernel_size=int(params["kernel_size"]),
                dilations=tuple(params["dilations"]),
                dropout=float(params["dropout"]),
                learning_rate=float(params["learning_rate"]),
            )
            params.update(
                {
                    "sequence_length": cfg.PRIMARY_SEQUENCE_LENGTH,
                    "batch_size": cfg.TCN_BATCH_SIZE,
                    "epochs": cfg.TCN_EPOCHS,
                    "early_stopping_patience": cfg.TCN_EARLY_STOPPING_PATIENCE,
                    "seed": cfg.RANDOM_SEED,
                    "receptive_field_steps": tmp.receptive_field(),
                    "receptive_field_minutes": tmp.receptive_field() * cfg.INTERVAL_MINUTES,
                }
            )
        else:
            params = {}
        locked["models"][model] = {
            "candidate_id": str(cand),
            "status": "locked",
            "config": params,
            "validation_summary": {
                "mean_normalized_validation_mae": row["mean_normalized_validation_mae"],
                "mean_normalized_validation_rmse": row["mean_normalized_validation_rmse"],
                "mean_raw_validation_mae": row["mean_raw_validation_mae"],
                "mean_raw_validation_rmse": row["mean_raw_validation_rmse"],
                "mean_raw_validation_mape": row["mean_raw_validation_mape"],
                "total_fit_time_seconds": row["total_fit_time_seconds"],
            },
        }
    return locked


def square_results_for_selected(
    candidate_results: pd.DataFrame,
    selection_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, sel in selection_summary.iterrows():
        model = sel["model"]
        cand = sel["selected_candidate_id"]
        if cand is None or (isinstance(cand, float) and np.isnan(cand)):
            continue
        sub = candidate_results.loc[
            (candidate_results["model"] == model)
            & (candidate_results["candidate_id"] == cand)
        ]
        for _, r in sub.iterrows():
            rows.append(
                {
                    "model": model,
                    "square_id": int(r["square_id"]),
                    "candidate_id": cand,
                    "status": r["status"],
                    "validation_mae": r.get("validation_mae"),
                    "validation_rmse": r.get("validation_rmse"),
                    "validation_mape": r.get("validation_mape"),
                    "normalized_validation_mae": r.get("normalized_validation_mae"),
                    "normalized_validation_rmse": r.get("normalized_validation_rmse"),
                    "fit_time_seconds": r.get("fit_time_seconds"),
                    "prediction_time_seconds": r.get("prediction_time_seconds"),
                }
            )
    return pd.DataFrame(rows)


def collect_environment_metadata(
    *,
    planned: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    tf_version = None
    gpu_info: list[str] = []
    try:
        import tensorflow as tf

        tf_version = tf.__version__
        gpu_info = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:  # noqa: BLE001
        pass
    try:
        import statsmodels

        sm_version = statsmodels.__version__
    except Exception:  # noqa: BLE001
        sm_version = None

    def _cand_json(d: dict[str, dict]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in d.items():
            item = {}
            for kk, vv in v.items():
                if isinstance(vv, tuple):
                    item[kk] = list(vv)
                else:
                    item[kk] = vv
            out[k] = item
        return out

    return {
        "phase": "6B",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "tensorflow_version": tf_version,
        "statsmodels_version": sm_version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "random_seed": seed,
        "gpu_devices": gpu_info,
        "sequence_length": cfg.PRIMARY_SEQUENCE_LENGTH,
        "target_squares": list(cfg.TARGET_SQUARES),
        "split_bounds_utc_inclusive": {
            k: [str(a), str(b)] for k, (a, b) in cfg.split_bounds().items()
        },
        "normalization_definition": {
            "normalized_MAE": "validation_MAE / training_mean_traffic",
            "normalized_RMSE": "validation_RMSE / training_mean_traffic",
            "training_mean_source": "train split only, per square",
            "validation_or_test_used_for_scale": False,
        },
        "selection_rule": {
            "primary": cfg.PHASE6B_SELECTION_PRIMARY,
            "secondary": cfg.PHASE6B_SELECTION_SECONDARY,
            "tie_breakers": [
                "lower_mean_raw_validation_mae",
                "lower_total_fit_time_seconds",
                "simpler_configuration",
            ],
        },
        "candidates": {
            "SARIMA": _cand_json(cfg.PHASE6B_SARIMA_CANDIDATES),
            "LSTM": _cand_json(cfg.PHASE6B_LSTM_CANDIDATES),
            "TCN": _cand_json(cfg.PHASE6B_TCN_CANDIDATES),
        },
        "tcn_receptive_fields": {
            cid: {
                "receptive_field_steps": TCNConfig(
                    filters=int(p["filters"]),
                    kernel_size=int(p["kernel_size"]),
                    dilations=tuple(p["dilations"]),
                    dropout=float(p["dropout"]),
                    learning_rate=float(p["learning_rate"]),
                ).receptive_field(),
                "formula": "1 + 2*(kernel_size-1)*sum(dilations)",
                "note": "Two causal convolutions per residual block",
            }
            for cid, p in cfg.PHASE6B_TCN_CANDIDATES.items()
        },
        "planned_experiment_size": planned,
        "final_test_period": {
            "start": cfg.TEST_START,
            "end": cfg.TEST_END,
            "used_in_phase_6b": False,
        },
    }


def write_validation_report(
    path: Path,
    *,
    selection_summary: pd.DataFrame,
    square_results: pd.DataFrame,
    candidate_results: pd.DataFrame,
    naive_results: pd.DataFrame,
    locked: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Phase 6B — Validation-Based Configuration Selection Report\n")
    lines.append("## 1. Purpose\n")
    lines.append(
        "Select **one locked configuration per model family** (SARIMA, LSTM, TCN) "
        "using the Phase 5 training period for fitting and the Phase 5 validation "
        "period for scoring. This is not a final model ranking across families.\n"
    )
    lines.append("## 2. Data used\n")
    lines.append(
        f"- Target squares: `{cfg.TARGET_SQUARES}`\n"
        f"- Train: `{cfg.TRAIN_START}` → `{cfg.TRAIN_END}` "
        f"({cfg.EXPECTED_ROWS['train']} rows/square)\n"
        f"- Validation: `{cfg.VALIDATION_START}` → `{cfg.VALIDATION_END}` "
        f"({cfg.EXPECTED_ROWS['validation']} rows/square)\n"
        f"- Sequence length: `{cfg.PRIMARY_SEQUENCE_LENGTH}`\n"
        f"- Test period `{cfg.TEST_START}` → `{cfg.TEST_END}`: **held out**\n"
    )
    lines.append("## 3. Leakage controls\n")
    lines.append(
        "- Scalers fitted on training data only (Phase 5).\n"
        "- Neural early stopping monitors validation loss only.\n"
        "- SARIMA parameters estimated on the training series only.\n"
        "- Selection metrics use validation predictions only.\n"
        "- Test targets are never loaded into the selection logic.\n"
        "- Normalization denominators use **training** mean traffic only.\n"
    )
    lines.append("## 4. Candidate configurations\n")
    lines.append("### SARIMA\n")
    for cid, p in cfg.PHASE6B_SARIMA_CANDIDATES.items():
        lines.append(f"- **{cid}**: order={p['order']}, seasonal_order={p['seasonal_order']}\n")
    lines.append("\n### LSTM\n")
    for cid, p in cfg.PHASE6B_LSTM_CANDIDATES.items():
        lines.append(
            f"- **{cid}**: units={p['units']}, layers={p['n_layers']}, "
            f"dropout={p['dropout']}, lr={p['learning_rate']}\n"
        )
    lines.append("\n### TCN\n")
    for cid, p in cfg.PHASE6B_TCN_CANDIDATES.items():
        rf = metadata["tcn_receptive_fields"][cid]["receptive_field_steps"]
        lines.append(
            f"- **{cid}**: filters={p['filters']}, kernel={p['kernel_size']}, "
            f"dilations={p['dilations']}, dropout={p['dropout']}, lr={p['learning_rate']}, "
            f"RF={rf} steps ({rf * cfg.INTERVAL_MINUTES} min)\n"
        )
    lines.append("\n## 5. Selection criterion\n")
    lines.append(
        "1. Primary: mean normalized validation MAE across squares  \n"
        "   (`validation_MAE / training_mean_traffic`)  \n"
        "2. Secondary: mean normalized validation RMSE  \n"
        "3. Tie-breakers: lower mean raw MAE, lower total fit time, simpler configuration  \n"
    )
    lines.append("## 6. Validation results\n")
    lines.append("### Selection summary\n")
    lines.append("```\n")
    lines.append(selection_summary.to_string(index=False))
    lines.append("\n```\n")
    lines.append("\n### Selected candidate by square\n")
    lines.append("```\n")
    lines.append(square_results.to_string(index=False))
    lines.append("\n```\n")
    lines.append("\n## 7. Selected configuration for each model\n")
    for model, payload in locked.get("models", {}).items():
        lines.append(f"### {model}\n")
        lines.append(f"- Candidate: `{payload.get('candidate_id')}`\n")
        lines.append(f"- Config: `{json.dumps(payload.get('config', {}), sort_keys=True)}`\n")
    lines.append("\n## 8. Naive baseline validation result\n")
    lines.append("```\n")
    lines.append(naive_results.to_string(index=False))
    lines.append("\n```\n")
    lines.append(
        "\nThe naive baseline is a reference only and was not tuned.\n"
    )
    lines.append("\n## 9. TCN receptive-field analysis\n")
    lines.append(
        "Receptive field uses `1 + 2*(kernel_size-1)*sum(dilations)` because each "
        "residual block contains two causal dilated convolutions.\n\n"
    )
    for cid, info in metadata["tcn_receptive_fields"].items():
        rf = info["receptive_field_steps"]
        lines.append(
            f"- Candidate {cid}: RF = {rf} steps "
            f"({rf * cfg.INTERVAL_MINUTES} minutes); "
            f"covers full L=144 window: {rf >= cfg.PRIMARY_SEQUENCE_LENGTH}\n"
        )
    lines.append(
        "\nA wider receptive field is **not** assumed to be better; selection is empirical.\n"
    )
    lines.append("\n## 10. Timing\n")
    ok = candidate_results.loc[candidate_results["status"] == "ok"]
    if len(ok):
        timing = (
            ok.groupby("model")["fit_time_seconds"]
            .agg(["count", "sum", "mean", "max"])
            .reset_index()
        )
        lines.append("```\n")
        lines.append(timing.to_string(index=False))
        lines.append("\n```\n")
    failed = candidate_results.loc[candidate_results["status"] == "failed"]
    lines.append(f"\nFailed candidate×square rows: **{len(failed)}**\n")
    if len(failed):
        lines.append("```\n")
        cols = [c for c in ["model", "candidate_id", "square_id", "error"] if c in failed.columns]
        lines.append(failed[cols].to_string(index=False))
        lines.append("\n```\n")
    lines.append("\n## 11. Reproducibility information\n")
    lines.append(
        f"- Random seed: `{metadata.get('random_seed')}`\n"
        f"- Python: `{metadata.get('python_version', '')[:80]}`\n"
        f"- TensorFlow: `{metadata.get('tensorflow_version')}`\n"
        f"- statsmodels: `{metadata.get('statsmodels_version')}`\n"
        f"- NumPy: `{metadata.get('numpy_version')}`\n"
        f"- pandas: `{metadata.get('pandas_version')}`\n"
        f"- Platform: `{metadata.get('platform')}`\n"
        f"- GPU devices: `{metadata.get('gpu_devices')}`\n"
    )
    lines.append("\n## 12. Statement on the final test period\n")
    lines.append(
        "**The Dec 16–22 test period was NOT used** for fitting, early stopping, "
        "normalization scales, hyperparameter selection, or model ranking in Phase 6B.\n"
    )
    lines.append(
        "\nPhase 6B does **not** declare a winning model family. Cross-family "
        "comparison on the held-out test week belongs to a later phase.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")
