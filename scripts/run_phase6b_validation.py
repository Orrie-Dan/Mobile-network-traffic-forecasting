"""
Phase 6B — validation-based configuration selection.

Fits candidates on the Phase 5 training period and scores them on the
Phase 5 validation period only. The reserved Dec 16–22 test week is never used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Limit TF thread fan-out before TensorFlow is imported by model modules.
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src.data.forecasting import fit_square_scalers  # noqa: E402
from src.selection.phase6b import (  # noqa: E402
    assert_phase5_protocol,
    collect_environment_metadata,
    evaluate_lstm_candidates,
    evaluate_naive_validation,
    evaluate_sarima_candidates,
    evaluate_tcn_candidates,
    load_forecasting_frame,
    locked_configurations,
    planned_experiment_size,
    select_best_candidates,
    square_results_for_selected,
    write_validation_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6B validation-based configuration selection")
    p.add_argument(
        "--models",
        default="naive,sarima,lstm,tcn",
        help="Comma-separated subset: naive,sarima,lstm,tcn",
    )
    p.add_argument(
        "--squares",
        default=",".join(str(s) for s in cfg.TARGET_SQUARES),
        help="Comma-separated square IDs (default: all Phase 5 target squares)",
    )
    p.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    p.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on candidates per model family (A,B,C,... order)",
    )
    p.add_argument(
        "--skip-confirm",
        action="store_true",
        help="Do not require interactive confirmation of experiment size",
    )
    p.add_argument(
        "--append-results",
        action="store_true",
        help=(
            "Merge with an existing phase6b_candidate_results.csv "
            "(replace rows for the models run in this invocation)."
        ),
    )
    return p.parse_args()


def _limit_candidates(d: dict, max_candidates: int | None) -> dict:
    if max_candidates is None:
        return d
    keys = sorted(d.keys())[: max(0, int(max_candidates))]
    return {k: d[k] for k in keys}


def main() -> int:
    args = parse_args()
    models = {m.strip().lower() for m in args.models.split(",") if m.strip()}
    squares = [int(s) for s in args.squares.split(",") if s.strip()]
    if sorted(squares) != sorted(cfg.TARGET_SQUARES) and set(squares) - set(cfg.TARGET_SQUARES):
        raise SystemExit(f"Squares must be a subset of {cfg.TARGET_SQUARES}; got {squares}")

    sarima_cands = _limit_candidates(cfg.PHASE6B_SARIMA_CANDIDATES, args.max_candidates)
    lstm_cands = _limit_candidates(cfg.PHASE6B_LSTM_CANDIDATES, args.max_candidates)
    tcn_cands = _limit_candidates(cfg.PHASE6B_TCN_CANDIDATES, args.max_candidates)

    planned_models = []
    if "sarima" in models:
        planned_models.append("SARIMA")
    if "lstm" in models:
        planned_models.append("LSTM")
    if "tcn" in models:
        planned_models.append("TCN")

    # Temporarily monkey-patch counts via explicit planned dict
    planned = {
        "squares": squares,
        "n_squares": len(squares),
        "sarima_candidates": len(sarima_cands) if "sarima" in models else 0,
        "lstm_candidates": len(lstm_cands) if "lstm" in models else 0,
        "tcn_candidates": len(tcn_cands) if "tcn" in models else 0,
        "SARIMA_fits": (len(sarima_cands) * len(squares)) if "sarima" in models else 0,
        "LSTM_fits": (len(lstm_cands) * len(squares)) if "lstm" in models else 0,
        "TCN_fits": (len(tcn_cands) * len(squares)) if "tcn" in models else 0,
        "naive_evals": len(squares) if "naive" in models else 0,
        "total_research_candidate_square_runs": 0,
        "note": "Test split is not included in any fit or selection metric.",
        "seed": args.seed,
    }
    planned["total_research_candidate_square_runs"] = (
        planned["SARIMA_fits"] + planned["LSTM_fits"] + planned["TCN_fits"]
    )

    print("=" * 72, flush=True)
    print("Phase 6B — Validation-based configuration selection", flush=True)
    print("=" * 72, flush=True)
    print("TEST PERIOD IS HELD OUT (Dec 16–22). No test metrics will be computed.", flush=True)
    print(json.dumps(planned, indent=2), flush=True)

    # Safety: refuse unexpectedly large searches
    if planned["total_research_candidate_square_runs"] > 40:
        raise SystemExit(
            f"Refusing unexpectedly large search: "
            f"{planned['total_research_candidate_square_runs']} candidate×square runs. "
            "Reduce candidates or pass --max-candidates."
        )
    if planned["sarima_candidates"] > 6 or planned["lstm_candidates"] > 8 or planned["tcn_candidates"] > 8:
        raise SystemExit("Candidate counts exceed Phase 6B safety limits.")

    if not args.skip_confirm and sys.stdin.isatty():
        ans = input("Proceed with this experiment size? [y/N] ").strip().lower()
        if ans not in {"y", "yes"}:
            print("Aborted.", flush=True)
            return 1

    df = load_forecasting_frame(ROOT)
    # Restrict to requested squares without dropping protocol checks on the full file
    assert_phase5_protocol(
        df.loc[df["square_id"].isin(cfg.TARGET_SQUARES)].copy(),
        cfg.TARGET_SQUARES,
    )
    df = df.loc[df["square_id"].isin(squares)].copy()
    # Ensure test rows exist in the file but are never passed to evaluators as targets
    n_test = int((df["split"] == "test").sum())
    print(f"Loaded forecasting rows={len(df)} (including {n_test} held-out test rows).", flush=True)

    scalers = fit_square_scalers(df, squares=squares)
    for sid, sc in scalers.items():
        assert int(sc.scaler.n_samples_seen_) == cfg.EXPECTED_ROWS["train"]

    all_rows: list[dict] = []
    naive_rows: list[dict] = []

    if "naive" in models:
        print("\n[1/4] Naive persistence (validation only)...", flush=True)
        naive_rows = evaluate_naive_validation(df, squares=squares)
        all_rows.extend(naive_rows)

    if "sarima" in models:
        print("\n[2/4] SARIMA candidates...", flush=True)
        all_rows.extend(
            evaluate_sarima_candidates(df, squares=squares, candidates=sarima_cands)
        )

    if "lstm" in models:
        print("\n[3/4] LSTM candidates...", flush=True)
        all_rows.extend(
            evaluate_lstm_candidates(
                df, scalers, squares=squares, candidates=lstm_cands, seed=args.seed
            )
        )

    if "tcn" in models:
        print("\n[4/4] TCN candidates...", flush=True)
        all_rows.extend(
            evaluate_tcn_candidates(
                df, scalers, squares=squares, candidates=tcn_cands, seed=args.seed
            )
        )

    if not all_rows:
        raise SystemExit("No results produced. Check --models.")

    candidate_results = pd.DataFrame(all_rows)
    out_cand = ROOT / cfg.PHASE6B_CANDIDATE_RESULTS_FILE
    if args.append_results and out_cand.is_file():
        prev = pd.read_csv(out_cand)
        models_run = set(candidate_results["model"].unique())
        prev = prev.loc[~prev["model"].isin(models_run)]
        candidate_results = pd.concat([prev, candidate_results], ignore_index=True)
        print(
            f"Appended/replaced models {sorted(models_run)} into existing candidate results.",
            flush=True,
        )

    naive_df = candidate_results.loc[candidate_results["model"] == "Naive"].copy()
    research_df = candidate_results.loc[
        candidate_results["model"].isin(["SARIMA", "LSTM", "TCN"])
    ].copy()

    selection_summary = select_best_candidates(research_df)
    square_results = square_results_for_selected(research_df, selection_summary)
    locked = locked_configurations(selection_summary)
    metadata = collect_environment_metadata(planned=planned, seed=args.seed)

    # Write outputs
    out_sel = ROOT / cfg.PHASE6B_SELECTION_SUMMARY_FILE
    out_sq = ROOT / cfg.PHASE6B_SQUARE_RESULTS_FILE
    out_cfg = ROOT / cfg.PHASE6B_CONFIGURATIONS_FILE
    out_meta = ROOT / cfg.PHASE6B_EXPERIMENT_METADATA_FILE
    out_report = ROOT / cfg.PHASE6B_VALIDATION_REPORT_FILE
    for path in (out_cand, out_sel, out_sq, out_cfg, out_meta, out_report):
        path.parent.mkdir(parents=True, exist_ok=True)

    candidate_results.to_csv(out_cand, index=False)
    selection_summary.to_csv(out_sel, index=False)
    square_results.to_csv(out_sq, index=False)
    with open(out_cfg, "w", encoding="utf-8") as f:
        json.dump(locked, f, indent=2)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    write_validation_report(
        out_report,
        selection_summary=selection_summary,
        square_results=square_results,
        candidate_results=research_df,
        naive_results=naive_df,
        locked=locked,
        metadata=metadata,
    )

    print("\n" + "=" * 72, flush=True)
    print("SELECTION SUMMARY", flush=True)
    print(selection_summary.to_string(index=False), flush=True)
    print("\nLocked configurations:", flush=True)
    print(json.dumps(locked["models"], indent=2), flush=True)
    n_fail = int((research_df["status"] == "failed").sum()) if len(research_df) else 0
    print(f"\nFailed candidate×square rows: {n_fail}", flush=True)
    print("Dec 16–22 test evaluation performed: False", flush=True)
    print("Outputs:", flush=True)
    for label, path in [
        ("candidate_results", out_cand),
        ("selection_summary", out_sel),
        ("square_results", out_sq),
        ("configurations", out_cfg),
        ("metadata", out_meta),
        ("validation_report", out_report),
    ]:
        print(f"  {label}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
