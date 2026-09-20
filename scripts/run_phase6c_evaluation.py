"""
Phase 6C — final Dec 16–22 test evaluation of locked Phase 6B configs.

Refits locked SARIMA / LSTM / TCN on the training protocol, scores the
reserved test week, and writes comparison artefacts under results/metrics/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.selection.phase6c import run_phase6c  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6C final test evaluation")
    p.add_argument(
        "--models",
        default="naive,sarima,lstm,tcn",
        help="Comma-separated subset: naive,sarima,lstm,tcn",
    )
    p.add_argument(
        "--squares",
        default=",".join(str(s) for s in cfg.TARGET_SQUARES),
        help="Comma-separated square IDs (default: Phase 5 targets)",
    )
    p.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    p.add_argument(
        "--locked-config",
        type=Path,
        default=None,
        help=(
            "Path to Phase 6B locked JSON (script or Colab format). "
            "Default: search results/metrics/phase6b_configurations.json, "
            "else PHASE6C_DEFAULT_LOCKED."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data + locked configs and print plan without fitting",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    models = [m.strip().upper() for m in args.models.split(",") if m.strip()]
    squares = [int(s) for s in args.squares.split(",") if s.strip()]
    if set(squares) - set(cfg.TARGET_SQUARES):
        raise SystemExit(f"Squares must be a subset of {cfg.TARGET_SQUARES}; got {squares}")

    if args.dry_run:
        from src.selection.phase6c import load_forecasting_frame, load_locked_configs
        from src.selection.phase6b import assert_phase5_protocol

        df = load_forecasting_frame(ROOT)
        assert_phase5_protocol(df, squares)
        locked, source = load_locked_configs(args.locked_config, root=ROOT, allow_default=True)
        print("Phase 6C dry-run OK")
        print(f"  squares: {squares}")
        print(f"  models:  {models}")
        print(f"  locked source: {source or 'PHASE6C_DEFAULT_LOCKED'}")
        print(json.dumps(locked, indent=2, default=str))
        return 0

    result = run_phase6c(
        root=ROOT,
        locked_path=args.locked_config,
        squares=squares,
        models=models,
        seed=args.seed,
        write_outputs=True,
    )
    print("\n=== Aggregate summary ===")
    print(result["summary"].to_string(index=False))
    best = result["comparison"].get("best_research_model_by_mean_nmae")
    print(f"\nBest research model by mean nMAE: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
