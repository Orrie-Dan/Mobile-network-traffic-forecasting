"""
Phase 6C — final Dec 16–22 test evaluation of locked Phase 6B configs.

Refits each locked model on the full training split (SARIMA: full train +
default maxiter; LSTM/TCN: train fit with validation early stopping), then
scores one-step-ahead predictions on the reserved test week only.

No hyperparameter search. Cross-family ranking uses test metrics for reporting
only; configs themselves were locked in Phase 6B on validation.
"""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

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
from src.selection.phase6b import (
    _clear_keras_session,
    _scaled_frame,
    assert_phase5_protocol,
    load_forecasting_frame,
    project_root,
    training_mean_traffic,
)


def _test_metric_fields(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    training_mean: float,
) -> dict[str, float]:
    t_mae = mae(y_true, y_pred)
    t_rmse = rmse(y_true, y_pred)
    t_mape = mape(y_true, y_pred)
    return {
        "test_mae": t_mae,
        "test_rmse": t_rmse,
        "test_mape": t_mape,
        "training_mean": training_mean,
        "normalized_test_mae": t_mae / training_mean,
        "normalized_test_rmse": t_rmse / training_mean,
    }


def _failed_row(base: dict[str, Any], error: Exception) -> dict[str, Any]:
    out = dict(base)
    out.update(
        {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "fit_time_seconds": base.get("fit_time_seconds"),
            "prediction_time_seconds": None,
            "test_mae": None,
            "test_rmse": None,
            "test_mape": None,
            "normalized_test_mae": None,
            "normalized_test_rmse": None,
        }
    )
    return out


def _as_tuple(value: Any) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError(f"Expected list/tuple, got {type(value)!r}")


def normalize_locked_configs(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """
    Accept either Phase 6B script JSON (``models`` key) or Colab flat JSON
    (``SARIMA`` / ``LSTM`` / ``TCN`` top-level keys). Returns a uniform map:

        {family: {candidate_id, ...params}}
    """
    out: dict[str, dict[str, Any]] = {}

    models = raw.get("models")
    if isinstance(models, dict) and models:
        for family, block in models.items():
            if not isinstance(block, dict):
                continue
            if block.get("status") == "unselected":
                continue
            cand = block.get("candidate_id") or block.get("label")
            cfg_block = dict(block.get("config") or {})
            # Merge top-level order fields if present without nested config
            for k, v in block.items():
                if k in ("candidate_id", "label", "status", "config", "validation_summary"):
                    continue
                cfg_block.setdefault(k, v)
            entry = {"candidate_id": str(cand) if cand is not None else None, **cfg_block}
            out[str(family).upper()] = entry
        return out

    # Colab flat format
    for family in ("SARIMA", "LSTM", "TCN"):
        if family not in raw:
            continue
        block = dict(raw[family])
        cand = block.pop("label", block.pop("candidate_id", None))
        # Colab uses "lr" / "batch" shorthand
        if "lr" in block and "learning_rate" not in block:
            block["learning_rate"] = block.pop("lr")
        if "batch" in block and "batch_size" not in block:
            block["batch_size"] = block.pop("batch")
        block.pop("rf", None)
        out[family] = {"candidate_id": str(cand) if cand is not None else None, **block}

    return out


def default_locked_configs() -> dict[str, dict[str, Any]]:
    """Copy of ``PHASE6C_DEFAULT_LOCKED`` with JSON-friendly list tuples expanded."""
    out: dict[str, dict[str, Any]] = {}
    for family, params in cfg.PHASE6C_DEFAULT_LOCKED.items():
        entry = dict(params)
        if "order" in entry:
            entry["order"] = list(entry["order"])
        if "seasonal_order" in entry:
            entry["seasonal_order"] = list(entry["seasonal_order"])
        if "dilations" in entry:
            entry["dilations"] = list(entry["dilations"])
        out[family] = entry
    return out


def load_locked_configs(
    path: Optional[Path] = None,
    *,
    root: Optional[Path] = None,
    allow_default: bool = True,
) -> tuple[dict[str, dict[str, Any]], Path | None]:
    """
    Load locked configs from ``path``, or search common Phase 6B artefact paths.

    Returns ``(normalized_configs, source_path_or_None_if_default)``.
    """
    root = root or project_root()
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        candidates.extend(
            [
                root / cfg.PHASE6B_CONFIGURATIONS_FILE,
                root / "results" / "metrics" / "locked_configs.json",
                root / "results" / "phase6b" / "locked_configs.json",
            ]
        )

    for cand in candidates:
        if cand.is_file():
            raw = json.loads(cand.read_text(encoding="utf-8-sig"))
            locked = normalize_locked_configs(raw)
            if not locked:
                raise ValueError(f"No model configs found in {cand}")
            return locked, cand

    if not allow_default:
        raise FileNotFoundError(
            "No Phase 6B locked configuration file found. "
            f"Expected one of: {[str(c) for c in candidates]}"
        )
    return default_locked_configs(), None


def _neural_sequences_for_test(
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
    test_seq = make_supervised_sequences(
        scaled_df,
        square_id=square_id,
        sequence_length=cfg.PRIMARY_SEQUENCE_LENGTH,
        target_split="test",
    )
    assert_no_test_in_fit_window(train_seq["target_timestamp"], context="6C neural train")
    assert_no_test_in_fit_window(val_seq["target_timestamp"], context="6C neural val")
    assert_one_step_alignment(test_seq["input_end_timestamp"], test_seq["target_timestamp"])
    assert test_seq["target_timestamp"][0] == cfg.utc_timestamp(cfg.TEST_START)
    assert test_seq["input_end_timestamp"][0] == cfg.utc_timestamp(cfg.VALIDATION_END)

    test_orig = make_supervised_sequences(
        df,
        square_id=square_id,
        sequence_length=cfg.PRIMARY_SEQUENCE_LENGTH,
        target_split="test",
    )
    return {
        "X_train": train_seq["X"],
        "y_train": train_seq["y"],
        "train_targets": train_seq["target_timestamp"],
        "X_val": val_seq["X"],
        "y_val_scaled": val_seq["y"],
        "val_targets": val_seq["target_timestamp"],
        "X_test": test_seq["X"],
        "test_targets": test_seq["target_timestamp"],
        "y_test_original": test_orig["y"],
    }


def evaluate_naive_test(
    df: pd.DataFrame,
    *,
    squares: Sequence[int] = cfg.TARGET_SQUARES,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sid in squares:
        train = univariate_series(df, sid, split="train")
        val = univariate_series(df, sid, split="validation")
        test = univariate_series(df, sid, split="test")
        # Persistence needs observed lags through the test window (one-step with
        # actual history — not recursive multi-step).
        history = pd.concat([train, val, test]).sort_index()
        tmean = training_mean_traffic(df, sid)
        base = {
            "model": "Naive",
            "candidate_id": "persistence",
            "square_id": int(sid),
            "is_research_model": False,
            "training_mean": tmean,
        }
        print(f"  Naive square {sid} ...", flush=True)
        try:
            model = NaivePersistenceModel()
            model.fit(history)
            preds = model.predict(target_timestamps=test.index)
            assert preds.index[0] == cfg.utc_timestamp(cfg.TEST_START)
            metrics = _test_metric_fields(
                test.to_numpy(dtype="float64"),
                preds.to_numpy(dtype="float64"),
                training_mean=tmean,
            )
            rows.append(
                {
                    **base,
                    "status": "ok",
                    "fit_time_seconds": model.timing.fit_seconds,
                    "prediction_time_seconds": model.timing.predict_seconds,
                    "n_test": int(len(test)),
                    **metrics,
                    "y_true": test.to_numpy(dtype="float64"),
                    "y_pred": preds.to_numpy(dtype="float64"),
                    "timestamps": [str(t) for t in test.index],
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(_failed_row(base, exc))
            print(f"    FAILED: {exc}", flush=True)
    return rows


def evaluate_sarima_locked(
    df: pd.DataFrame,
    locked: dict[str, Any],
    *,
    squares: Sequence[int] = cfg.TARGET_SQUARES,
) -> list[dict[str, Any]]:
    order = _as_tuple(locked["order"])
    seasonal_order = _as_tuple(locked["seasonal_order"])
    maxiter = int(locked.get("maxiter", cfg.SARIMA_MAXITER))
    method = str(locked.get("method", cfg.SARIMA_FIT_METHOD))
    cand = locked.get("candidate_id", "?")
    rows: list[dict[str, Any]] = []

    for sid in squares:
        tmean = training_mean_traffic(df, sid)
        base = {
            "model": "SARIMA",
            "candidate_id": str(cand),
            "square_id": int(sid),
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "method": method,
            "maxiter": maxiter,
            "is_research_model": True,
            "training_mean": tmean,
        }
        print(
            f"  SARIMA-{cand} square {sid} full-train fit "
            f"order={order} seasonal={seasonal_order} maxiter={maxiter} ...",
            flush=True,
        )
        try:
            train = univariate_series(df, sid, split="train")
            val = univariate_series(df, sid, split="validation")
            test = univariate_series(df, sid, split="test")
            assert len(train) == cfg.EXPECTED_ROWS["train"]
            assert_no_test_in_fit_window(train.index, context="SARIMA.fit Phase 6C")
            config = SARIMAConfig(
                order=order,  # type: ignore[arg-type]
                seasonal_order=seasonal_order,  # type: ignore[arg-type]
                method=method,
                maxiter=maxiter,
            )
            model = SARIMAModel(config)
            model.fit(train)
            # One-step with observed history through test (extend path).
            history = pd.concat([train, val, test]).sort_index()
            preds = model.predict_one_step(history, target_timestamps=test.index)
            assert preds.index[0] == cfg.utc_timestamp(cfg.TEST_START)
            metrics = _test_metric_fields(
                test.to_numpy(dtype="float64"),
                preds.to_numpy(dtype="float64"),
                training_mean=tmean,
            )
            rows.append(
                {
                    **base,
                    "status": "ok",
                    "convergence_status": "success",
                    "fit_warning": model.fit_warning_,
                    "fit_time_seconds": model.timing.fit_seconds,
                    "prediction_time_seconds": model.timing.predict_seconds,
                    "fit_train_rows": int(len(train)),
                    "n_test": int(len(test)),
                    **metrics,
                    "y_true": test.to_numpy(dtype="float64"),
                    "y_pred": preds.to_numpy(dtype="float64"),
                    "timestamps": [str(t) for t in test.index],
                }
            )
            del model, train, val, test, history, preds
            gc.collect()
        except Exception as exc:  # noqa: BLE001
            failed = _failed_row(base, exc)
            failed["convergence_status"] = "failed"
            rows.append(failed)
            print(f"    FAILED: {exc}", flush=True)
            gc.collect()
    return rows


def evaluate_lstm_locked(
    df: pd.DataFrame,
    scalers: dict[int, SquareScaler],
    locked: dict[str, Any],
    *,
    squares: Sequence[int] = cfg.TARGET_SQUARES,
    seed: int = cfg.RANDOM_SEED,
) -> list[dict[str, Any]]:
    cand = locked.get("candidate_id", "?")
    scaled_df = _scaled_frame(df, scalers)
    rows: list[dict[str, Any]] = []

    for sid in squares:
        tmean = training_mean_traffic(df, sid)
        base = {
            "model": "LSTM",
            "candidate_id": str(cand),
            "square_id": int(sid),
            "units": int(locked["units"]),
            "n_layers": int(locked["n_layers"]),
            "dropout": float(locked["dropout"]),
            "learning_rate": float(locked["learning_rate"]),
            "batch_size": int(locked.get("batch_size", cfg.LSTM_BATCH_SIZE)),
            "max_epochs": int(locked.get("epochs", cfg.LSTM_EPOCHS)),
            "sequence_length": cfg.PRIMARY_SEQUENCE_LENGTH,
            "seed": seed,
            "is_research_model": True,
            "training_mean": tmean,
        }
        print(
            f"  LSTM-{cand} square {sid} "
            f"units={locked['units']} layers={locked['n_layers']} ...",
            flush=True,
        )
        try:
            set_global_seeds(seed)
            data = _neural_sequences_for_test(df, scaled_df, int(sid))
            config = LSTMConfig(
                units=int(locked["units"]),
                n_layers=int(locked["n_layers"]),
                dropout=float(locked["dropout"]),
                learning_rate=float(locked["learning_rate"]),
                batch_size=int(locked.get("batch_size", cfg.LSTM_BATCH_SIZE)),
                epochs=int(locked.get("epochs", cfg.LSTM_EPOCHS)),
                early_stopping_patience=int(
                    locked.get("early_stopping_patience", cfg.LSTM_EARLY_STOPPING_PATIENCE)
                ),
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
            y_pred = model.predict(data["X_test"])
            metrics = _test_metric_fields(
                data["y_test_original"], y_pred, training_mean=tmean
            )
            rows.append(
                {
                    **base,
                    "status": "ok",
                    "fit_time_seconds": model.timing.fit_seconds,
                    "prediction_time_seconds": model.timing.predict_seconds,
                    "n_test": int(len(y_pred)),
                    **metrics,
                    "y_true": np.asarray(data["y_test_original"], dtype="float64"),
                    "y_pred": np.asarray(y_pred, dtype="float64"),
                    "timestamps": [str(t) for t in data["test_targets"]],
                }
            )
            del model, data
            _clear_keras_session()
        except Exception as exc:  # noqa: BLE001
            rows.append(_failed_row(base, exc))
            print(f"    FAILED: {exc}", flush=True)
            _clear_keras_session()
    return rows


def evaluate_tcn_locked(
    df: pd.DataFrame,
    scalers: dict[int, SquareScaler],
    locked: dict[str, Any],
    *,
    squares: Sequence[int] = cfg.TARGET_SQUARES,
    seed: int = cfg.RANDOM_SEED,
) -> list[dict[str, Any]]:
    cand = locked.get("candidate_id", "?")
    dilations = _as_tuple(locked["dilations"])
    scaled_df = _scaled_frame(df, scalers)
    rows: list[dict[str, Any]] = []

    for sid in squares:
        tmean = training_mean_traffic(df, sid)
        tmp = TCNConfig(
            filters=int(locked["filters"]),
            kernel_size=int(locked["kernel_size"]),
            dilations=dilations,  # type: ignore[arg-type]
            dropout=float(locked["dropout"]),
            learning_rate=float(locked["learning_rate"]),
        )
        base = {
            "model": "TCN",
            "candidate_id": str(cand),
            "square_id": int(sid),
            "filters": int(locked["filters"]),
            "kernel_size": int(locked["kernel_size"]),
            "dilations": str(dilations),
            "dropout": float(locked["dropout"]),
            "learning_rate": float(locked["learning_rate"]),
            "batch_size": int(locked.get("batch_size", cfg.TCN_BATCH_SIZE)),
            "max_epochs": int(locked.get("epochs", cfg.TCN_EPOCHS)),
            "receptive_field_steps": tmp.receptive_field(),
            "sequence_length": cfg.PRIMARY_SEQUENCE_LENGTH,
            "seed": seed,
            "is_research_model": True,
            "training_mean": tmean,
        }
        print(
            f"  TCN-{cand} square {sid} "
            f"f={locked['filters']} k={locked['kernel_size']} dil={dilations} ...",
            flush=True,
        )
        try:
            set_global_seeds(seed)
            data = _neural_sequences_for_test(df, scaled_df, int(sid))
            config = TCNConfig(
                filters=int(locked["filters"]),
                kernel_size=int(locked["kernel_size"]),
                dilations=dilations,  # type: ignore[arg-type]
                dropout=float(locked["dropout"]),
                learning_rate=float(locked["learning_rate"]),
                batch_size=int(locked.get("batch_size", cfg.TCN_BATCH_SIZE)),
                epochs=int(locked.get("epochs", cfg.TCN_EPOCHS)),
                early_stopping_patience=int(
                    locked.get("early_stopping_patience", cfg.TCN_EARLY_STOPPING_PATIENCE)
                ),
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
            y_pred = model.predict(data["X_test"])
            metrics = _test_metric_fields(
                data["y_test_original"], y_pred, training_mean=tmean
            )
            rows.append(
                {
                    **base,
                    "status": "ok",
                    "fit_time_seconds": model.timing.fit_seconds,
                    "prediction_time_seconds": model.timing.predict_seconds,
                    "n_test": int(len(y_pred)),
                    **metrics,
                    "y_true": np.asarray(data["y_test_original"], dtype="float64"),
                    "y_pred": np.asarray(y_pred, dtype="float64"),
                    "timestamps": [str(t) for t in data["test_targets"]],
                }
            )
            del model, data
            _clear_keras_session()
        except Exception as exc:  # noqa: BLE001
            rows.append(_failed_row(base, exc))
            print(f"    FAILED: {exc}", flush=True)
            _clear_keras_session()
    return rows


def square_results_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten evaluation rows (drop large prediction arrays)."""
    drop = {"y_true", "y_pred", "timestamps"}
    records = [{k: v for k, v in r.items() if k not in drop} for r in rows]
    return pd.DataFrame(records)


def aggregate_test_summary(square_df: pd.DataFrame) -> pd.DataFrame:
    """Mean test metrics across squares per model family."""
    ok = square_df.loc[square_df["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for model, grp in ok.groupby("model", sort=False):
        rows.append(
            {
                "model": model,
                "candidate_id": grp["candidate_id"].iloc[0],
                "is_research_model": bool(grp["is_research_model"].iloc[0]),
                "n_squares": int(grp["square_id"].nunique()),
                "mean_test_mae": float(grp["test_mae"].mean()),
                "mean_test_rmse": float(grp["test_rmse"].mean()),
                "mean_test_mape": float(grp["test_mape"].mean()),
                "mean_normalized_test_mae": float(grp["normalized_test_mae"].mean()),
                "mean_normalized_test_rmse": float(grp["normalized_test_rmse"].mean()),
                "total_fit_time_seconds": float(grp["fit_time_seconds"].fillna(0).sum()),
                "total_prediction_time_seconds": float(
                    grp["prediction_time_seconds"].fillna(0).sum()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    # Rank research models by mean normalized MAE (reporting only — not selection)
    research = summary.loc[summary["is_research_model"]].copy()
    if not research.empty:
        research = research.sort_values(
            ["mean_normalized_test_mae", "mean_normalized_test_rmse", "mean_test_mae"]
        )
        summary = pd.concat(
            [research, summary.loc[~summary["is_research_model"]]],
            ignore_index=True,
        )
    return summary


def comparison_payload(
    summary: pd.DataFrame,
    locked: dict[str, dict[str, Any]],
    *,
    source_path: Path | None,
) -> dict[str, Any]:
    ranking = []
    for _, row in summary.iterrows():
        ranking.append(
            {
                "model": row["model"],
                "candidate_id": row["candidate_id"],
                "is_research_model": bool(row["is_research_model"]),
                "mean_normalized_test_mae": row["mean_normalized_test_mae"],
                "mean_normalized_test_rmse": row["mean_normalized_test_rmse"],
                "mean_test_mae": row["mean_test_mae"],
                "mean_test_rmse": row["mean_test_rmse"],
                "mean_test_mape": row["mean_test_mape"],
            }
        )
    research = [r for r in ranking if r["is_research_model"]]
    best = research[0]["model"] if research else None
    return {
        "phase": "6C",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_week": {
            "start": cfg.TEST_START,
            "end": cfg.TEST_END,
            "expected_rows_per_square": cfg.EXPECTED_ROWS["test"],
        },
        "locked_config_source": str(source_path) if source_path else "PHASE6C_DEFAULT_LOCKED",
        "locked_configs": locked,
        "protocol": {
            "sarima_fit": "full training split, original scale, SARIMA_MAXITER",
            "neural_fit": "train sequences; early stopping on validation loss",
            "scaler": "train-only MinMax; inverse-transform before metrics",
            "forecast": "one-step-ahead with observed lags (not recursive multi-step)",
            "selection_on_test": False,
            "note": (
                "Configs were locked in Phase 6B on validation. "
                "Test ranking is for reporting only."
            ),
        },
        "ranking_by_mean_normalized_test_mae": ranking,
        "best_research_model_by_mean_nmae": best,
    }


def write_predictions(
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for r in rows:
        if r.get("status") != "ok" or "y_true" not in r:
            continue
        path = out_dir / f"{r['model'].lower()}_sq{r['square_id']}_test.csv"
        pd.DataFrame(
            {
                "timestamp": r["timestamps"],
                "y_true": r["y_true"],
                "y_pred": r["y_pred"],
            }
        ).to_csv(path, index=False)
        written.append(path)
    return written


def write_test_report(
    summary: pd.DataFrame,
    square_df: pd.DataFrame,
    comparison: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# Phase 6C — Final Test Evaluation Report\n",
        "\n",
        f"- Evaluated at (UTC): `{comparison['evaluated_at_utc']}`\n",
        f"- Locked config source: `{comparison['locked_config_source']}`\n",
        f"- Test week: `{cfg.TEST_START}` → `{cfg.TEST_END}`\n",
        f"- Best research model (mean nMAE): **{comparison.get('best_research_model_by_mean_nmae')}**\n",
        "\n",
        "## Aggregate test metrics\n\n",
    ]
    if not summary.empty:
        try:
            lines.append(summary.to_markdown(index=False))
        except ImportError:
            lines.append("```\n" + summary.to_string(index=False) + "\n```")
        lines.append("\n\n")
    lines.append("## Per-square test metrics\n\n")
    cols = [
        c
        for c in (
            "model",
            "candidate_id",
            "square_id",
            "status",
            "test_mae",
            "test_rmse",
            "test_mape",
            "normalized_test_mae",
            "normalized_test_rmse",
            "fit_time_seconds",
        )
        if c in square_df.columns
    ]
    if cols:
        try:
            lines.append(square_df[cols].to_markdown(index=False))
        except ImportError:
            lines.append("```\n" + square_df[cols].to_string(index=False) + "\n```")
        lines.append("\n")
    lines.append(
        "\n## Protocol note\n\n"
        "Hyperparameters were locked in Phase 6B on validation only. "
        "This phase refits locked configs and scores Dec 16–22 once.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def run_phase6c(
    *,
    root: Optional[Path] = None,
    locked_path: Optional[Path] = None,
    squares: Optional[Sequence[int]] = None,
    models: Optional[Sequence[str]] = None,
    seed: int = cfg.RANDOM_SEED,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """
    End-to-end Phase 6C evaluation.

    Returns a dict with square_results, summary, comparison, and output paths.
    """
    root = root or project_root()
    squares = list(squares or cfg.TARGET_SQUARES)
    want = {m.upper() for m in (models or ("NAIVE", "SARIMA", "LSTM", "TCN"))}

    df = load_forecasting_frame(root)
    assert_phase5_protocol(df, squares)
    locked, source = load_locked_configs(locked_path, root=root, allow_default=True)

    all_rows: list[dict[str, Any]] = []
    if "NAIVE" in want:
        print("=== Naive (test) ===", flush=True)
        all_rows.extend(evaluate_naive_test(df, squares=squares))

    if "SARIMA" in want:
        if "SARIMA" not in locked:
            raise KeyError("Locked configs missing SARIMA")
        print("=== SARIMA locked (full-train refit → test) ===", flush=True)
        all_rows.extend(evaluate_sarima_locked(df, locked["SARIMA"], squares=squares))

    scalers: dict[int, SquareScaler] | None = None
    if want & {"LSTM", "TCN"}:
        scalers = fit_square_scalers(df, squares=squares)

    if "LSTM" in want:
        if "LSTM" not in locked:
            raise KeyError("Locked configs missing LSTM")
        assert scalers is not None
        print("=== LSTM locked (train fit / val early-stop → test) ===", flush=True)
        all_rows.extend(
            evaluate_lstm_locked(df, scalers, locked["LSTM"], squares=squares, seed=seed)
        )

    if "TCN" in want:
        if "TCN" not in locked:
            raise KeyError("Locked configs missing TCN")
        assert scalers is not None
        print("=== TCN locked (train fit / val early-stop → test) ===", flush=True)
        all_rows.extend(
            evaluate_tcn_locked(df, scalers, locked["TCN"], squares=squares, seed=seed)
        )

    square_df = square_results_table(all_rows)
    summary = aggregate_test_summary(square_df)
    comparison = comparison_payload(summary, locked, source_path=source)

    outputs: dict[str, str] = {}
    if write_outputs:
        sq_path = root / cfg.PHASE6C_SQUARE_RESULTS_FILE
        sum_path = root / cfg.PHASE6C_SUMMARY_FILE
        cmp_path = root / cfg.PHASE6C_COMPARISON_FILE
        meta_path = root / cfg.PHASE6C_EXPERIMENT_METADATA_FILE
        report_path = root / cfg.PHASE6C_REPORT_FILE
        pred_dir = root / cfg.PHASE6C_PREDICTIONS_DIR

        sq_path.parent.mkdir(parents=True, exist_ok=True)
        square_df.to_csv(sq_path, index=False)
        summary.to_csv(sum_path, index=False)
        cmp_path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
        meta = {
            "phase": "6C",
            "squares": squares,
            "models": sorted(want),
            "seed": seed,
            "locked_source": str(source) if source else "default",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        write_test_report(summary, square_df, comparison, report_path)
        written_preds = write_predictions(all_rows, pred_dir)
        outputs = {
            "square_results": str(sq_path),
            "summary": str(sum_path),
            "comparison": str(cmp_path),
            "metadata": str(meta_path),
            "report": str(report_path),
            "predictions_dir": str(pred_dir),
            "n_prediction_files": str(len(written_preds)),
        }
        print("\nPhase 6C artefacts:", flush=True)
        for k, v in outputs.items():
            print(f"  {k}: {v}", flush=True)

    return {
        "square_results": square_df,
        "summary": summary,
        "comparison": comparison,
        "locked": locked,
        "locked_source": source,
        "outputs": outputs,
        "rows": all_rows,
    }
