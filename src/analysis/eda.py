"""
Exploratory data analysis helpers for mobile network traffic (Phase 3B).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOP_SQUARES = [5161, 5059, 5259]
TARGET_SQUARES = [5161, 5059, 5259, 4159, 4556]
INTERVAL_MINUTES = 10
BINS_PER_DAY = 24 * 60 // INTERVAL_MINUTES  # 144


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_square_totals(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values("total_internet_traffic", ascending=False).reset_index(
        drop=True
    )


def load_target_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["square_id"] = df["square_id"].astype(int)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df.sort_values(["square_id", "timestamp"]).reset_index(drop=True)


def expected_timestamps(ts_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Build a complete 10-minute UTC grid from min to max inclusive."""
    start = ts_index.min()
    end = ts_index.max()
    return pd.date_range(start=start, end=end, freq="10min", tz="UTC")


def analyse_missingness(
    ts: pd.DataFrame,
    *,
    target_squares: Sequence[int] = TARGET_SQUARES,
) -> dict[str, Any]:
    """
    Compare observed timestamps to the full 10-minute grid spanning the series.

    Does not fill missing values. Per-day counts use the filename ``date`` column
    (calendar day of the source file), not UTC calendar-day labels of timestamps.
    """
    all_ts = expected_timestamps(ts["timestamp"])
    expected_n = int(len(all_ts))
    # Filename dates present anywhere in the extracted target series
    expected_dates = sorted(ts["date"].astype(str).unique())
    by_square: dict[str, Any] = {}

    for sid in target_squares:
        sub = ts.loc[ts["square_id"] == sid]
        observed = pd.DatetimeIndex(sub["timestamp"].unique()).sort_values()
        observed_n = int(len(observed))
        missing = all_ts.difference(observed)
        missing_n = int(len(missing))

        if "date" in sub.columns and len(sub):
            daily_counts = (
                sub.groupby("date")["timestamp"]
                .nunique()
                .reindex(expected_dates, fill_value=0)
                .astype(int)
            )
        else:
            daily_counts = (
                sub.set_index("timestamp")
                .groupby(pd.Grouper(freq="D"))["internet_traffic"]
                .count()
            )

        by_square[str(sid)] = {
            "square_id": int(sid),
            "expected_timestamps": expected_n,
            "observed_timestamps": observed_n,
            "missing_timestamps": missing_n,
            "missing_percentage": round(100.0 * missing_n / expected_n, 6)
            if expected_n
            else None,
            "min_observations_per_day": int(daily_counts.min())
            if len(daily_counts)
            else 0,
            "max_observations_per_day": int(daily_counts.max())
            if len(daily_counts)
            else 0,
            "avg_observations_per_day": float(daily_counts.mean())
            if len(daily_counts)
            else 0.0,
            "n_days_with_any_observation": int((daily_counts > 0).sum())
            if len(daily_counts)
            else 0,
            "n_filename_dates": len(expected_dates),
        }

    return {
        "interval_minutes": INTERVAL_MINUTES,
        "expected_timestamps_global": expected_n,
        "grid_start": str(all_ts.min()) if expected_n else None,
        "grid_end": str(all_ts.max()) if expected_n else None,
        "filename_dates": expected_dates,
        "filled_with_zero": False,
        "by_square": by_square,
    }


def descriptive_statistics(
    ts: pd.DataFrame,
    *,
    target_squares: Sequence[int] = TARGET_SQUARES,
    missingness: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    rows = []
    for sid in target_squares:
        vals = ts.loc[ts["square_id"] == sid, "internet_traffic"].astype("float64")
        mean = float(vals.mean()) if len(vals) else float("nan")
        std = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        cv = float(std / mean) if mean and np.isfinite(mean) and mean != 0 else float(
            "nan"
        )
        miss = None
        if missingness is not None:
            miss = missingness["by_square"][str(sid)]["missing_timestamps"]
        rows.append(
            {
                "square_id": int(sid),
                "n_observations": int(len(vals)),
                "n_missing_timestamps": miss,
                "mean": mean,
                "median": float(vals.median()) if len(vals) else float("nan"),
                "std": std,
                "min": float(vals.min()) if len(vals) else float("nan"),
                "max": float(vals.max()) if len(vals) else float("nan"),
                "coefficient_of_variation": cv,
            }
        )
    return pd.DataFrame(rows)


def save_traffic_distribution(
    totals: pd.DataFrame,
    path: Path,
    *,
    top_squares: Sequence[int] = TOP_SQUARES,
) -> None:
    _ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    x = totals["total_internet_traffic"].astype(float)
    axes[0].hist(x, bins=60, color="#4C78A8", edgecolor="white", linewidth=0.4)
    axes[0].set_title("Distribution of total Internet traffic (all squares)")
    axes[0].set_xlabel("Total Internet traffic (activity units)")
    axes[0].set_ylabel("Number of squares")

    axes[1].hist(
        x, bins=60, color="#F58518", edgecolor="white", linewidth=0.4, log=False
    )
    axes[1].set_xscale("log")
    axes[1].set_title("Same distribution (log-scaled x-axis)")
    axes[1].set_xlabel("Total Internet traffic (log scale)")
    axes[1].set_ylabel("Number of squares")

    for ax in axes:
        for sid, color in zip(top_squares, ["#E45756", "#54A24B", "#B279A2"]):
            val = float(
                totals.loc[totals["square_id"] == sid, "total_internet_traffic"].iloc[0]
            )
            ax.axvline(val, color=color, linestyle="--", linewidth=1.2, label=f"{sid}")
        ax.legend(title="Top 3", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_top_areas_bar(
    totals: pd.DataFrame,
    path: Path,
    *,
    n: int = 15,
    top_squares: Sequence[int] = TOP_SQUARES,
) -> None:
    _ensure_dir(path.parent)
    top = totals.head(n).copy()
    top["label"] = top["square_id"].astype(str)
    colors = [
        "#E45756" if sid in top_squares else "#4C78A8" for sid in top["square_id"]
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(
        top["label"][::-1],
        top["total_internet_traffic"][::-1],
        color=colors[::-1],
    )
    ax.set_xlabel("Total Internet traffic (activity units)")
    ax.set_ylabel("square_id")
    ax.set_title(f"Top {n} squares by total Internet traffic")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def first_two_weeks_mask(ts: pd.DataFrame) -> pd.Series:
    """Filename calendar dates 2013-11-01 through 2013-11-14."""
    return (ts["date"] >= "2013-11-01") & (ts["date"] <= "2013-11-14")


def save_first_two_weeks_plots(
    ts: pd.DataFrame,
    figures_dir: Path,
    *,
    target_squares: Sequence[int] = TARGET_SQUARES,
) -> list[Path]:
    _ensure_dir(figures_dir)
    subset = ts.loc[first_two_weeks_mask(ts)].copy()
    paths: list[Path] = []

    for sid in target_squares:
        sub = subset.loc[subset["square_id"] == sid].sort_values("timestamp")
        path = figures_dir / f"eda_square_{sid}_first_2_weeks.png"
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.plot(
            sub["timestamp"],
            sub["internet_traffic"],
            color="#4C78A8",
            linewidth=0.8,
        )
        ax.set_title(f"Square {sid}: Internet traffic, first two weeks (2013-11-01–14)")
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel("Internet traffic (activity units)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # Readable comparison: 5 panels
    cmp_path = figures_dir / "eda_target_squares_first_2_weeks_comparison.png"
    fig, axes = plt.subplots(len(target_squares), 1, figsize=(11, 10), sharex=True)
    if len(target_squares) == 1:
        axes = [axes]
    for ax, sid in zip(axes, target_squares):
        sub = subset.loc[subset["square_id"] == sid].sort_values("timestamp")
        ax.plot(sub["timestamp"], sub["internet_traffic"], linewidth=0.7, color="#4C78A8")
        ax.set_ylabel(f"{sid}")
        ax.grid(True, alpha=0.25)
    axes[0].set_title(
        "Target squares — first two weeks Internet traffic (shared time axis)"
    )
    axes[-1].set_xlabel("Timestamp (UTC)")
    fig.tight_layout()
    fig.savefig(cmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(cmp_path)
    return paths


def save_daily_target_plot(
    ts: pd.DataFrame,
    path: Path,
    *,
    target_squares: Sequence[int] = TARGET_SQUARES,
) -> pd.DataFrame:
    """Daily totals (sum of 10-minute Internet activity) by square."""
    _ensure_dir(path.parent)
    daily = (
        ts.groupby(["square_id", "date"], as_index=False)["internet_traffic"]
        .sum()
        .rename(columns={"internet_traffic": "daily_internet_traffic"})
    )
    daily["date"] = pd.to_datetime(daily["date"])

    fig, ax = plt.subplots(figsize=(12, 5))
    for sid in target_squares:
        sub = daily.loc[daily["square_id"] == sid].sort_values("date")
        ax.plot(
            sub["date"],
            sub["daily_internet_traffic"],
            marker="o",
            markersize=2.5,
            linewidth=1.1,
            label=str(sid),
        )
    ax.set_title("Daily total Internet traffic by target square")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily total Internet traffic (sum of 10-min activity)")
    ax.legend(title="square_id", ncol=5, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return daily


def compute_acf_values(
    series: pd.Series,
    *,
    nlags: int = 1100,
) -> np.ndarray:
    from statsmodels.tsa.stattools import acf

    values = series.astype("float64").dropna().to_numpy()
    return acf(values, nlags=nlags, fft=True)


def save_square_acf(
    ts: pd.DataFrame,
    path: Path,
    *,
    square_id: int = 5161,
    nlags: int = 1100,
) -> dict[str, Any]:
    _ensure_dir(path.parent)
    series = (
        ts.loc[ts["square_id"] == square_id]
        .sort_values("timestamp")["internet_traffic"]
        .astype("float64")
    )
    acf_vals = compute_acf_values(series, nlags=nlags)
    lags_of_interest = {
        "10_minutes": 1,
        "1_hour": 6,
        "6_hours": 36,
        "12_hours": 72,
        "1_day": 144,
        "2_days": 288,
        "3_days": 432,
        "1_week": 1008,
    }
    selected = {
        name: float(acf_vals[lag]) if lag < len(acf_vals) else None
        for name, lag in lags_of_interest.items()
    }

    fig, ax = plt.subplots(figsize=(11, 4.2))
    lags = np.arange(len(acf_vals))
    ax.vlines(lags, 0, acf_vals, color="#4C78A8", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    conf = 1.96 / np.sqrt(len(series))
    ax.axhline(conf, color="gray", linestyle="--", linewidth=0.8)
    ax.axhline(-conf, color="gray", linestyle="--", linewidth=0.8)
    for name, lag in lags_of_interest.items():
        if lag < len(acf_vals):
            ax.axvline(lag, color="#E45756", alpha=0.35, linestyle=":")
            ax.text(lag, ax.get_ylim()[1] * 0.92, name.replace("_", "\n"), fontsize=7, ha="center")
    ax.set_xlim(0, nlags)
    ax.set_title(f"Square {square_id}: autocorrelation (10-min Internet traffic)")
    ax.set_xlabel("Lag (number of 10-minute steps)")
    ax.set_ylabel("ACF")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "square_id": square_id,
        "n_observations": int(len(series)),
        "nlags": nlags,
        "acf_at_key_lags": selected,
        "approx_ci_95": float(conf),
    }


def save_square_seasonality(
    ts: pd.DataFrame,
    path: Path,
    *,
    square_id: int = 5161,
) -> dict[str, Any]:
    """Hour-of-day and day-of-week average profiles for Square 5161."""
    _ensure_dir(path.parent)
    sub = ts.loc[ts["square_id"] == square_id].copy()
    sub = sub.sort_values("timestamp")
    # Convert to Europe/Rome for local clock patterns (dataset is Milan)
    local = sub["timestamp"].dt.tz_convert("Europe/Rome")
    sub["hour"] = local.dt.hour
    sub["day_name"] = local.dt.day_name()
    sub["dow"] = local.dt.dayofweek

    by_hour = sub.groupby("hour")["internet_traffic"].mean()
    by_dow = (
        sub.groupby(["dow", "day_name"])["internet_traffic"]
        .mean()
        .reset_index()
        .sort_values("dow")
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(by_hour.index, by_hour.values, marker="o", color="#4C78A8")
    axes[0].set_title(f"Square {square_id}: mean traffic by hour (Europe/Rome)")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Mean Internet traffic")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].grid(True, alpha=0.25)

    axes[1].bar(by_dow["day_name"], by_dow["internet_traffic"], color="#F58518")
    axes[1].set_title(f"Square {square_id}: mean traffic by weekday (Europe/Rome)")
    axes[1].set_xlabel("Day of week")
    axes[1].set_ylabel("Mean Internet traffic")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    peak_hour = int(by_hour.idxmax())
    low_hour = int(by_hour.idxmin())
    peak_day = str(by_dow.loc[by_dow["internet_traffic"].idxmax(), "day_name"])
    low_day = str(by_dow.loc[by_dow["internet_traffic"].idxmin(), "day_name"])
    return {
        "square_id": square_id,
        "analysis": "seasonality_hour_and_weekday",
        "peak_hour_local": peak_hour,
        "low_hour_local": low_hour,
        "peak_weekday": peak_day,
        "low_weekday": low_day,
        "mean_by_hour": {str(int(k)): float(v) for k, v in by_hour.items()},
        "mean_by_weekday": {
            str(r.day_name): float(r.internet_traffic) for r in by_dow.itertuples()
        },
    }


def save_unusual_period_plot(
    ts: pd.DataFrame,
    path: Path,
    *,
    square_id: int = 5161,
) -> dict[str, Any]:
    """
    Flag unusual high activity using a transparent rolling robust threshold.

    Method: 1-day rolling median and MAD; flag points where
    (x - median) / (1.4826*MAD) > 6, then focus plot around the strongest event.
    """
    _ensure_dir(path.parent)
    sub = (
        ts.loc[ts["square_id"] == square_id]
        .sort_values("timestamp")
        .set_index("timestamp")[["internet_traffic"]]
        .astype("float64")
    )
    window = BINS_PER_DAY  # 1 day
    roll_med = sub["internet_traffic"].rolling(window, center=True, min_periods=36).median()
    mad = (
        (sub["internet_traffic"] - roll_med)
        .abs()
        .rolling(window, center=True, min_periods=36)
        .median()
    )
    robust_z = (sub["internet_traffic"] - roll_med) / (1.4826 * mad.replace(0, np.nan))
    flags = robust_z > 6
    flagged = sub.loc[flags.fillna(False)].copy()
    flagged["robust_z"] = robust_z.loc[flagged.index]

    if len(flagged):
        event_ts = flagged["robust_z"].idxmax()
        event_val = float(sub.loc[event_ts, "internet_traffic"])
        event_z = float(flagged.loc[event_ts, "robust_z"])
    else:
        # Fallback: global maximum
        event_ts = sub["internet_traffic"].idxmax()
        event_val = float(sub.loc[event_ts, "internet_traffic"])
        event_z = None

    pad = pd.Timedelta(hours=36)
    focus = sub.loc[event_ts - pad : event_ts + pad]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(focus.index, focus["internet_traffic"], color="#4C78A8", linewidth=0.9)
    ax.axvline(event_ts, color="#E45756", linestyle="--", linewidth=1.2)
    ax.scatter([event_ts], [event_val], color="#E45756", zorder=5)
    ax.set_title(
        f"Square {square_id}: unusual high-traffic period around {event_ts}"
    )
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("Internet traffic (activity units)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "square_id": square_id,
        "method": "rolling_1day_median_mad_robust_z_gt_6",
        "event_timestamp_utc": str(event_ts),
        "event_internet_traffic": event_val,
        "event_robust_z": event_z,
        "n_flagged_points": int(flags.fillna(False).sum()),
        "note": (
            "High robust-z relative to a 1-day rolling median/MAD. "
            "The dataset alone cannot establish a real-world cause."
        ),
    }


def save_missingness_figure(
    missingness: dict[str, Any],
    path: Path,
) -> None:
    _ensure_dir(path.parent)
    rows = [missingness["by_square"][k] for k in missingness["by_square"]]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        df["square_id"].astype(str),
        df["missing_percentage"],
        color="#72B7B2",
    )
    ax.set_title("Missing 10-minute bins vs full timestamp grid (target squares)")
    ax.set_xlabel("square_id")
    ax.set_ylabel("Missing percentage")
    for i, row in df.iterrows():
        ax.text(
            i,
            row["missing_percentage"] + 0.02,
            f"{row['missing_timestamps']}",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
