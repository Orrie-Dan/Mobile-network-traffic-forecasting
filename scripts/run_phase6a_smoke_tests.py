"""
Phase 6A smoke tests: fit each model on a SMALL train subset and check shapes.

Does NOT evaluate the reserved Dec 16–22 test period.
Does NOT claim relative model performance.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.data.forecasting import (  # noqa: E402
    SquareScaler,
    alignment_example,
    fit_square_scalers,
    load_source_target_timeseries,
    make_supervised_sequences,
    prepare_phase5_experiment,
    univariate_series,
)
from src.models import (  # noqa: E402
    LSTMConfig,
    LSTMModel,
    NaivePersistenceModel,
    SARIMAConfig,
    SARIMAModel,
    TCNConfig,
    TCNModel,
)
from src.models.base import (  # noqa: E402
    assert_boundary_alignment_example,
    assert_one_step_alignment,
    reshape_sequences_for_keras,
)


def _load_forecast_df() -> pd.DataFrame:
    path = ROOT / cfg.FORECASTING_DATASET_FILE
    if path.is_file():
        df = pd.read_csv(path, parse_dates=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        return df
    result = prepare_phase5_experiment(root=ROOT, write_outputs=True)
    return result["dataset"]


def test_alignment(forecast_df: pd.DataFrame) -> dict:
    align = alignment_example(
        forecast_df, square_id=5161, sequence_length=cfg.PRIMARY_SEQUENCE_LENGTH
    )
    row = align.iloc[0]
    assert_boundary_alignment_example(row["input_end_timestamp"], row["target_timestamp"])
    seq = make_supervised_sequences(
        forecast_df,
        square_id=5161,
        sequence_length=cfg.PRIMARY_SEQUENCE_LENGTH,
        target_split="test",
    )
    assert_one_step_alignment(seq["input_end_timestamp"], seq["target_timestamp"])
    assert seq["X"].shape[1] == cfg.PRIMARY_SEQUENCE_LENGTH
    return {
        "boundary_ok": True,
        "test_sequences": int(seq["n_samples"]),
        "note": "Alignment verified; test sequences were NOT used for fitting.",
    }


def smoke_naive(forecast_df: pd.DataFrame) -> dict:
    series = univariate_series(forecast_df, 5161, split="train")
    # Use train+validation history only for a short persistence check (no test).
    train_val = pd.concat(
        [
            univariate_series(forecast_df, 5161, split="train"),
            univariate_series(forecast_df, 5161, split="validation"),
        ]
    ).sort_index()
    model = NaivePersistenceModel()
    model.fit(train_val)
    targets = univariate_series(forecast_df, 5161, split="validation").index[:48]
    preds = model.predict(target_timestamps=targets)
    assert len(preds) == 48
    assert preds.isna().sum() == 0
    return {
        "model": model.name,
        "is_research_model": model.is_research_model,
        "n_predictions": int(len(preds)),
        "timing": model.timing.to_dict(),
        "ok": True,
    }


def smoke_sarima(forecast_df: pd.DataFrame) -> dict:
    # Small subset only — full seasonal MLE over the full train set is later.
    # s=144 state-space estimation is expensive; 2 days + few iterations suffice
    # to verify the implementation path without becoming a final experiment.
    full_train = univariate_series(forecast_df, 5161, split="train")
    subset = full_train.iloc[-(2 * cfg.BINS_PER_DAY) :]
    config = SARIMAConfig(
        order=(1, 0, 0),
        seasonal_order=(0, 1, 1, cfg.SARIMA_SEASONAL_PERIOD),
        maxiter=5,
    )
    print(
        f"  SARIMA smoke: n={len(subset)}, order={config.order}, "
        f"seasonal={config.seasonal_order}, maxiter={config.maxiter}",
        flush=True,
    )
    model = SARIMAModel(config)
    model.fit(subset)
    hist = subset
    targets = hist.index[-12:]
    preds = model.predict_one_step(hist, target_timestamps=targets)
    assert len(preds) == 12
    assert np.isfinite(preds.to_numpy()).all()
    nxt = model.forecast_next(1)
    assert nxt.shape == (1,)
    return {
        "model": model.name,
        "config": model.get_config(),
        "train_subset_len": int(len(subset)),
        "n_predictions": int(len(preds)),
        "fit_success": model.fit_success_,
        "timing": model.timing.to_dict(),
        "ok": True,
    }


def _scaled_train_val_sequences(
    forecast_df: pd.DataFrame,
    scaler: SquareScaler,
    *,
    n_train: int = 256,
    n_val: int = 64,
) -> dict:
    scaled_df = forecast_df.copy()
    scaled_df["internet_traffic"] = np.nan
    for split in ("train", "validation"):
        mask = (scaled_df["square_id"] == scaler.square_id) & (scaled_df["split"] == split)
        vals = forecast_df.loc[mask, "internet_traffic"].to_numpy()
        scaled_df.loc[mask, "internet_traffic"] = scaler.transform(vals)

    train_seq = make_supervised_sequences(
        scaled_df,
        square_id=scaler.square_id,
        sequence_length=cfg.PRIMARY_SEQUENCE_LENGTH,
        target_split="train",
    )
    val_seq = make_supervised_sequences(
        scaled_df,
        square_id=scaler.square_id,
        sequence_length=cfg.PRIMARY_SEQUENCE_LENGTH,
        target_split="validation",
    )
    # Smoke: take the last n_train training windows (still entirely before test)
    X_tr = train_seq["X"][-n_train:]
    y_tr = train_seq["y"][-n_train:]
    t_tr = train_seq["target_timestamp"][-n_train:]
    X_va = val_seq["X"][:n_val]
    y_va = val_seq["y"][:n_val]
    t_va = val_seq["target_timestamp"][:n_val]
    return {
        "X_train": X_tr,
        "y_train": y_tr,
        "train_targets": t_tr,
        "X_val": X_va,
        "y_val": y_va,
        "val_targets": t_va,
    }


def smoke_lstm(forecast_df: pd.DataFrame, scaler: SquareScaler) -> dict:
    data = _scaled_train_val_sequences(forecast_df, scaler)
    config = LSTMConfig(epochs=2, batch_size=32, early_stopping_patience=2)
    model = LSTMModel(config, scaler=scaler)
    model.build()
    shapes_before = model.input_output_shapes()
    X_batch = reshape_sequences_for_keras(data["X_train"][:8])
    assert X_batch.shape == (8, cfg.PRIMARY_SEQUENCE_LENGTH, 1)

    model.fit(
        data["X_train"],
        data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        train_target_timestamps=data["train_targets"],
        val_target_timestamps=data["val_targets"],
        verbose=0,
    )
    y_hat = model.predict(data["X_val"][:16])
    assert y_hat.shape == (16,)
    assert np.isfinite(y_hat).all()
    # Scaled predict shape check
    y_scaled = model.predict_scaled(data["X_val"][:16])
    assert y_scaled.shape == (16,)
    return {
        "model": model.name,
        "config": model.get_config(),
        "shapes": shapes_before,
        "keras_input_shape": list(model.model_.input_shape),
        "keras_output_shape": list(model.model_.output_shape),
        "batch_input_shape": list(X_batch.shape),
        "batch_output_shape": [16, 1],
        "n_train_smoke": int(len(data["X_train"])),
        "n_val_smoke": int(len(data["X_val"])),
        "epochs_completed": model.history_.epochs_completed,
        "timing": model.timing.to_dict(),
        "summary": model.summary_text(),
        "ok": True,
    }


def smoke_tcn(forecast_df: pd.DataFrame, scaler: SquareScaler) -> dict:
    data = _scaled_train_val_sequences(forecast_df, scaler)
    config = TCNConfig(
        epochs=2,
        batch_size=32,
        early_stopping_patience=2,
        dilations=(1, 2, 4),
        filters=16,
    )
    model = TCNModel(config, scaler=scaler)
    model.build()
    shapes_before = model.input_output_shapes()
    X_batch = reshape_sequences_for_keras(data["X_train"][:8])
    assert X_batch.shape == (8, cfg.PRIMARY_SEQUENCE_LENGTH, 1)

    model.fit(
        data["X_train"],
        data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        train_target_timestamps=data["train_targets"],
        val_target_timestamps=data["val_targets"],
        verbose=0,
    )
    y_hat = model.predict(data["X_val"][:16])
    assert y_hat.shape == (16,)
    assert np.isfinite(y_hat).all()
    return {
        "model": model.name,
        "config": model.get_config(),
        "shapes": shapes_before,
        "keras_input_shape": list(model.model_.input_shape),
        "keras_output_shape": list(model.model_.output_shape),
        "batch_input_shape": list(X_batch.shape),
        "batch_output_shape": [16, 1],
        "receptive_field": config.receptive_field(),
        "n_train_smoke": int(len(data["X_train"])),
        "n_val_smoke": int(len(data["X_val"])),
        "epochs_completed": model.history_.epochs_completed,
        "timing": model.timing.to_dict(),
        "summary": model.summary_text(),
        "ok": True,
    }


def main() -> int:
    print("Phase 6A — Model Implementation smoke tests", flush=True)
    print("Test period Dec 16–22 is NOT used for fitting or metrics.", flush=True)
    forecast_df = _load_forecast_df()
    print(f"Loaded forecasting rows={len(forecast_df)}", flush=True)
    scalers = fit_square_scalers(forecast_df, squares=cfg.TARGET_SQUARES)
    scaler = scalers[5161]

    print("1) Alignment...", flush=True)
    alignment = test_alignment(forecast_df)
    print("2) Naive...", flush=True)
    naive = smoke_naive(forecast_df)
    print("3) SARIMA...", flush=True)
    sarima = smoke_sarima(forecast_df)
    print("4) LSTM...", flush=True)
    lstm = smoke_lstm(forecast_df, scaler)
    print("5) TCN...", flush=True)
    tcn = smoke_tcn(forecast_df, scaler)

    report: dict = {
        "phase": "6A",
        "final_test_evaluation_performed": False,
        "performance_claims": False,
        "alignment": alignment,
        "naive": naive,
        "sarima": sarima,
        "lstm": lstm,
        "tcn": tcn,
    }

    out = ROOT / "results" / "metrics" / "phase6a_smoke_tests.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    printable = {}
    for k, v in report.items():
        if isinstance(v, dict):
            printable[k] = {kk: vv for kk, vv in v.items() if kk != "summary"}
        else:
            printable[k] = v
    print(json.dumps(printable, indent=2, default=str), flush=True)
    print(f"\nWrote {out}", flush=True)
    print("All smoke tests passed. No Dec 16–22 evaluation performed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
