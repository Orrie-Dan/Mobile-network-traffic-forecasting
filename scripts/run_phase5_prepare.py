"""Run Phase 5 forecasting experiment preparation (no model training)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    FORECASTING_MODELS,
    NAIVE_BASELINE_NAME,
    PRIMARY_SEQUENCE_LENGTH,
    TARGET_SQUARES,
)
from src.data.forecasting import ForecastDataIntegrityError, prepare_phase5_experiment  # noqa: E402


def main() -> int:
    print("Phase 5 — Forecasting Experiment Design & Data Preparation")
    print("Models locked (not trained):", ", ".join(FORECASTING_MODELS))
    print("Baseline (not a research model):", NAIVE_BASELINE_NAME)
    print("Forecasting squares:", TARGET_SQUARES)
    print("Primary sequence length:", PRIMARY_SEQUENCE_LENGTH)
    try:
        result = prepare_phase5_experiment(root=ROOT, write_outputs=True)
    except ForecastDataIntegrityError as exc:
        print("INTEGRITY FAILURE — stopping without repair.")
        print(exc)
        if exc.summary is not None and len(exc.summary):
            print(exc.summary.to_string(index=False))
        return 1

    print("\nSplit validation:")
    print(result["validation"].to_string(index=False))
    print("\nZero / near-zero inspection:")
    print(result["zero_inspection"].to_string(index=False))
    print("\nSequence-length analysis:")
    print(result["sequence_length_analysis"].to_string(index=False))
    print("\nOne-step alignment example:")
    print(result["alignment_example"].to_string(index=False))
    print("\nSARIMA seasonal-period recommendation:")
    print(json.dumps(result["sarima"], indent=2))
    print("\nScaler params (train only):")
    print(json.dumps(result["scaler_params"], indent=2))
    print("\nUnused source rows (outside train/validation/test):", result["n_unused_source_rows"])
    print("Outputs:")
    for key, path in result["outputs"].items():
        print(f"  {key}: {path}")
    print("\nmodels_trained:", result["models_trained"])
    print("Phase 5 preparation complete. Do not train SARIMA, LSTM, or TCN until reviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
