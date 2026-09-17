"""Run Phase 3B EDA from compact processed files (after extraction)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.eda import (
    TARGET_SQUARES,
    TOP_SQUARES,
    analyse_missingness,
    descriptive_statistics,
    load_square_totals,
    load_target_timeseries,
    save_daily_target_plot,
    save_first_two_weeks_plots,
    save_missingness_figure,
    save_square_acf,
    save_square_seasonality,
    save_top_areas_bar,
    save_traffic_distribution,
    save_unusual_period_plot,
)

FIGURES = ROOT / "results" / "figures"
METRICS = ROOT / "results" / "metrics"
PROCESSED = ROOT / "data" / "processed"
FIGURES.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    totals = load_square_totals(PROCESSED / "square_internet_totals.csv")
    # Verify top-3 match Phase 3A
    top3 = totals.head(3)["square_id"].astype(int).tolist()
    assert top3 == TOP_SQUARES, f"Top-3 mismatch: {top3} vs {TOP_SQUARES}"

    ts = load_target_timeseries(PROCESSED / "target_square_internet_timeseries.csv")
    present = sorted(ts["square_id"].unique().tolist())
    assert present == sorted(TARGET_SQUARES), f"Unexpected squares: {present}"
    assert ts["timestamp"].is_monotonic_increasing or all(
        g["timestamp"].is_monotonic_increasing
        for _, g in ts.groupby("square_id")
    )

    missingness = analyse_missingness(ts, target_squares=TARGET_SQUARES)
    with open(METRICS / "phase3b_missingness.json", "w", encoding="utf-8") as f:
        json.dump(missingness, f, indent=2)

    stats = descriptive_statistics(
        ts, target_squares=TARGET_SQUARES, missingness=missingness
    )
    stats.to_csv(METRICS / "target_square_descriptive_statistics.csv", index=False)

    save_traffic_distribution(
        totals, FIGURES / "eda_traffic_distribution.png", top_squares=TOP_SQUARES
    )
    save_top_areas_bar(
        totals, FIGURES / "eda_top_areas.png", n=15, top_squares=TOP_SQUARES
    )
    save_first_two_weeks_plots(ts, FIGURES, target_squares=TARGET_SQUARES)
    save_daily_target_plot(
        ts, FIGURES / "eda_target_squares_daily.png", target_squares=TARGET_SQUARES
    )
    save_missingness_figure(missingness, FIGURES / "eda_target_squares_missingness.png")

    acf_info = save_square_acf(
        ts, FIGURES / "eda_square_5161_acf.png", square_id=5161, nlags=1100
    )
    seasonality = save_square_seasonality(
        ts, FIGURES / "eda_square_5161_seasonality.png", square_id=5161
    )
    unusual = save_unusual_period_plot(
        ts, FIGURES / "eda_square_5161_unusual_period.png", square_id=5161
    )

    summary = {
        "top_3_squares": TOP_SQUARES,
        "target_squares": TARGET_SQUARES,
        "rows_by_square": {
            int(k): int(v) for k, v in ts.groupby("square_id").size().items()
        },
        "timestamp_min": str(ts["timestamp"].min()),
        "timestamp_max": str(ts["timestamp"].max()),
        "first_two_weeks_dates": ["2013-11-01", "2013-11-14"],
        "descriptive_statistics": stats.to_dict(orient="records"),
        "missingness": missingness,
        "acf_square_5161": acf_info,
        "seasonality_square_5161": {
            k: seasonality[k]
            for k in (
                "analysis",
                "peak_hour_local",
                "low_hour_local",
                "peak_weekday",
                "low_weekday",
            )
        },
        "unusual_period_square_5161": unusual,
        "figures": sorted(p.name for p in FIGURES.glob("eda_*.png")),
    }
    with open(METRICS / "phase3b_eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Phase 3B EDA complete")
    print("rows_by_square:", summary["rows_by_square"])
    print("missingness:", {k: v["missing_percentage"] for k, v in missingness["by_square"].items()})
    print("acf:", acf_info["acf_at_key_lags"])
    print("unusual:", unusual["event_timestamp_utc"], unusual["event_internet_traffic"])
    print("figures:", len(summary["figures"]))


if __name__ == "__main__":
    main()
