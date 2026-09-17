"""
Small Phase 6B SARIMA smoke test: one square, one candidate.

Does NOT run the full 12-fit experiment. Does NOT touch Dec 16–22 test data.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.data.forecasting import univariate_series  # noqa: E402
from src.evaluation.metrics import mae, mape, rmse  # noqa: E402
from src.models.base import assert_no_test_in_fit_window  # noqa: E402
from src.models.sarima import SARIMAConfig, SARIMAModel  # noqa: E402
from src.selection.phase6b import load_forecasting_frame, training_mean_traffic  # noqa: E402


def main() -> int:
    # Lightest seasonal MA-only candidate for a quick API/alignment check.
    cand_id = "D"
    params = cfg.PHASE6B_SARIMA_CANDIDATES[cand_id]
    square_id = 5161

    print("=" * 72, flush=True)
    print("Phase 6B SARIMA smoke test (1 square × 1 candidate)", flush=True)
    print("=" * 72, flush=True)
    print(f"installed statsmodels version: {statsmodels.__version__}", flush=True)
    print(
        "validation forecasting method: "
        "SARIMAXResults.extend(new_endog)  "
        "[filters new observations only; parameters stay fixed; no refit kwarg]",
        flush=True,
    )
    print(f"validation start: {cfg.VALIDATION_START}", flush=True)
    print(f"validation end:   {cfg.VALIDATION_END}", flush=True)
    print(f"expected validation predictions: {cfg.EXPECTED_ROWS['validation']}", flush=True)
    print(f"candidate: {cand_id} order={params['order']} seasonal={params['seasonal_order']}", flush=True)
    print(f"square_id: {square_id}", flush=True)
    print(f"selection_train_days: {cfg.PHASE6B_SARIMA_SELECTION_TRAIN_DAYS}", flush=True)
    print("test period used: False", flush=True)

    df = load_forecasting_frame(ROOT)
    df = df.loc[df["square_id"] == square_id].copy()
    # Refuse test leakage into this smoke path
    assert not (
        (df["split"] == "validation")
        & (df["timestamp"] >= cfg.utc_timestamp(cfg.TEST_START))
    ).any()

    train_full = univariate_series(df, square_id, split="train")
    val = univariate_series(df, square_id, split="validation")
    assert len(val) == cfg.EXPECTED_ROWS["validation"]
    assert val.index[0] == cfg.utc_timestamp(cfg.VALIDATION_START)
    assert val.index[-1] == cfg.utc_timestamp(cfg.VALIDATION_END)

    n_sel = cfg.PHASE6B_SARIMA_SELECTION_TRAIN_DAYS * cfg.BINS_PER_DAY
    train = train_full.iloc[-n_sel:]
    assert_no_test_in_fit_window(train.index, context="smoke SARIMA.fit")
    assert train.index[-1] == cfg.utc_timestamp(cfg.TRAIN_END)

    config = SARIMAConfig(
        order=tuple(params["order"]),  # type: ignore[arg-type]
        seasonal_order=tuple(params["seasonal_order"]),  # type: ignore[arg-type]
        method=str(params.get("method", cfg.SARIMA_FIT_METHOD)),
        maxiter=int(params.get("maxiter", cfg.PHASE6B_SARIMA_SELECTION_MAXITER)),
    )
    model = SARIMAModel(config)
    print("Fitting...", flush=True)
    model.fit(train)
    print(f"fit_seconds={model.timing.fit_seconds:.2f}", flush=True)

    history = pd.concat([train, val]).sort_index()
    assert_no_test_in_fit_window(history.index, context="smoke SARIMA.predict")
    print("Predicting validation via extend()...", flush=True)
    preds = model.predict_one_step(history, target_timestamps=val.index)

    assert len(preds) == cfg.EXPECTED_ROWS["validation"], len(preds)
    assert preds.index[0] == cfg.utc_timestamp(cfg.VALIDATION_START)
    assert preds.index[-1] == cfg.utc_timestamp(cfg.VALIDATION_END)
    # One-step alignment: first validation target follows last training stamp
    assert train.index[-1] == cfg.utc_timestamp("2013-12-08 23:50")
    assert preds.index[0] == cfg.utc_timestamp("2013-12-09 00:00")
    assert (preds.index[0] - train.index[-1]) == pd.Timedelta(minutes=10)
    assert np.isfinite(preds.to_numpy()).all()

    y_true = val.to_numpy(dtype="float64")
    y_pred = preds.to_numpy(dtype="float64")
    tmean = training_mean_traffic(df, square_id)
    print(f"n_validation_predictions: {len(preds)}", flush=True)
    print(
        f"alignment OK: {train.index[-1]} -> {preds.index[0]}",
        flush=True,
    )
    print(
        f"validation MAE={mae(y_true, y_pred):.4f} "
        f"RMSE={rmse(y_true, y_pred):.4f} "
        f"MAPE={mape(y_true, y_pred):.4f} "
        f"norm_MAE={mae(y_true, y_pred)/tmean:.6f}",
        flush=True,
    )
    del model, preds, history
    gc.collect()
    print("SMOKE TEST PASSED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
