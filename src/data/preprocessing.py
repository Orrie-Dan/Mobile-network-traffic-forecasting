"""
Preprocessing utilities for mobile network traffic data.

Phase 2: country-code handling and Internet aggregation to square × time.
Phase 3A: sequential full-dataset summaries (per-square and daily totals)
without materialising the full square × timestamp panel.
"""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.data.loading import (
    INTERNET_TASK_COLUMNS,
    INTERNET_TASK_DTYPES,
    PathLike,
    ZipMemberInfo,
    discover_unique_daily_members,
    extract_zip_member,
    get_process_rss_mb,
    parse_member_date,
    read_traffic_file,
    remove_path,
    resolve_zip_paths,
)

AggregationStrategy = Literal["sum_all_countries", "country_39_only"]


def aggregate_internet_traffic(
    df: pd.DataFrame,
    *,
    strategy: AggregationStrategy = "sum_all_countries",
    country_code: int = 39,
) -> pd.DataFrame:
    """
    Aggregate Internet traffic to one value per (square_id, time_interval).

    Strategies
    ----------
    sum_all_countries
        Sum ``internet_traffic`` across country codes for each square/time.
        Empty/NaN contributions are skipped by pandas ``sum`` (skipna=True).
    country_39_only
        Keep rows with ``country_code == country_code`` (default 39 = Italy),
        then sum within square/time (usually one row, but sum is safe).

    Returns columns: square_id, time_interval, internet_traffic
    """
    required = {"square_id", "time_interval", "internet_traffic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df
    if strategy == "country_39_only":
        if "country_code" not in df.columns:
            raise ValueError("country_39_only requires a country_code column")
        work = df.loc[df["country_code"] == country_code]
    elif strategy != "sum_all_countries":
        raise ValueError(f"Unknown strategy: {strategy}")

    grouped = (
        work.groupby(["square_id", "time_interval"], sort=False)["internet_traffic"]
        .sum(min_count=1)
        .reset_index()
    )
    # min_count=1 → NaN if no non-null values; leave as NaN (no silent zero-fill)
    return grouped


def merge_chunk_aggregates(
    aggregates: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine per-chunk (square_id, time_interval) sums into a day-level table.

    Required when the same square/time appears in multiple chunks or when
    country-level rows for one cell span chunk boundaries (rare but possible).
    """
    if not aggregates:
        return pd.DataFrame(columns=["square_id", "time_interval", "internet_traffic"])
    combined = pd.concat(aggregates, ignore_index=True)
    return (
        combined.groupby(["square_id", "time_interval"], sort=False)["internet_traffic"]
        .sum(min_count=1)
        .reset_index()
    )


def summarize_country_internet(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise Internet traffic by country_code.

    Returns counts of rows, non-null Internet rows, and summed Internet traffic
    with share of total non-null Internet sum.
    """
    if "country_code" not in df.columns or "internet_traffic" not in df.columns:
        raise ValueError("Expected country_code and internet_traffic columns")

    summary = (
        df.groupby("country_code", dropna=False)
        .agg(
            n_rows=("country_code", "size"),
            n_internet_non_null=("internet_traffic", lambda s: int(s.notna().sum())),
            internet_sum=("internet_traffic", "sum"),
        )
        .reset_index()
        .sort_values("internet_sum", ascending=False)
    )
    total = float(summary["internet_sum"].sum())
    summary["internet_share"] = (
        summary["internet_sum"] / total if total else float("nan")
    )
    return summary


def process_daily_internet_file(
    file_path: PathLike,
    *,
    chunksize: int = 500_000,
    strategy: AggregationStrategy = "sum_all_countries",
) -> dict[str, Any]:
    """
    Chunk-process one daily TXT into square×time Internet aggregates.

    Does not fill missing square×timestamp combinations with zeros.
    Accumulates country codes and basic validation counters.
    """
    chunk_aggs: list[pd.DataFrame] = []
    raw_rows = 0
    missing_or_invalid_rows = 0
    country_codes: set[int] = set()
    unique_squares: set[int] = set()
    unique_timestamps: set[int] = set()
    schema_ok = True
    schema_notes: list[str] = []

    reader = read_traffic_file(
        file_path,
        chunksize=chunksize,
        usecols=INTERNET_TASK_COLUMNS,
        dtype=INTERNET_TASK_DTYPES,
    )

    for chunk in reader:
        expected_cols = set(INTERNET_TASK_COLUMNS)
        if set(chunk.columns) != expected_cols:
            schema_ok = False
            schema_notes.append(
                f"Unexpected columns in chunk: {sorted(chunk.columns)}"
            )

        raw_rows += len(chunk)

        invalid_mask = (
            chunk["square_id"].isna()
            | chunk["time_interval"].isna()
            | chunk["country_code"].isna()
        )
        n_invalid = int(invalid_mask.sum())
        missing_or_invalid_rows += n_invalid
        if n_invalid:
            chunk = chunk.loc[~invalid_mask]

        # Track country codes (valid integer codes only)
        country_codes.update(int(c) for c in chunk["country_code"].unique().tolist())
        unique_squares.update(int(s) for s in chunk["square_id"].unique().tolist())
        unique_timestamps.update(
            int(t) for t in chunk["time_interval"].unique().tolist()
        )

        # Aggregate in float64 for more stable day totals
        work = chunk.copy()
        work["internet_traffic"] = work["internet_traffic"].astype("float64")
        chunk_aggs.append(aggregate_internet_traffic(work, strategy=strategy))
        del chunk, work

    day_agg = merge_chunk_aggregates(chunk_aggs)
    del chunk_aggs
    gc.collect()

    # Drop aggregated rows with no non-null Internet contribution
    day_agg["internet_traffic"] = day_agg["internet_traffic"].astype("float64")
    contributing = day_agg["internet_traffic"].notna()
    day_agg_valid = day_agg.loc[contributing].copy()

    ts_sorted = sorted(unique_timestamps)
    diffs = np.diff(ts_sorted) if len(ts_sorted) > 1 else np.array([], dtype=np.int64)
    interval_ok = bool(len(diffs) == 0 or (diffs == 600_000).all())

    daily_by_square = (
        day_agg_valid.groupby("square_id", sort=False)["internet_traffic"]
        .agg(daily_internet_traffic="sum", observation_count="count")
        .reset_index()
    )

    return {
        "raw_rows": raw_rows,
        "aggregated_rows": int(len(day_agg_valid)),
        "unique_squares": unique_squares,
        "unique_timestamps": unique_timestamps,
        "country_codes": country_codes,
        "internet_traffic_sum": float(day_agg_valid["internet_traffic"].sum()),
        "missing_or_invalid_rows": missing_or_invalid_rows,
        "schema_ok": schema_ok,
        "schema_notes": schema_notes,
        "timestamps_all_10_minutes": interval_ok,
        "timestamp_diff_counts": {
            int(k): int(v) for k, v in zip(*np.unique(diffs, return_counts=True))
        }
        if len(diffs)
        else {},
        "daily_by_square": daily_by_square,
        "day_square_time": day_agg_valid,
    }


def build_square_traffic_summary(
    *,
    data_dir: Optional[PathLike] = None,
    zip_paths: Optional[Sequence[PathLike]] = None,
    output_dir: Optional[PathLike] = None,
    metrics_dir: Optional[PathLike] = None,
    temp_dir: Optional[PathLike] = None,
    chunksize: int = 500_000,
    strategy: AggregationStrategy = "sum_all_countries",
    expected_n_days: int = 62,
) -> dict[str, Any]:
    """
    Phase 3A: process all unique daily files sequentially.

    Writes:
    - ``data/processed/square_internet_totals.csv``
    - ``data/processed/daily_square_internet_totals.csv``
    - ``results/metrics/phase3_dataset_summary.json``
    - ``results/metrics/phase3_processing_log.csv``

    Does not materialise the full square × 10-minute panel for all squares.
    Does not fill missing square×timestamp cells with zeros.
    """
    root = Path(__file__).resolve().parents[2]
    out_dir = Path(output_dir) if output_dir is not None else root / "data" / "processed"
    met_dir = Path(metrics_dir) if metrics_dir is not None else root / "results" / "metrics"
    tmp_dir = Path(temp_dir) if temp_dir is not None else root / "tmp" / "phase3a_extract"
    out_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    rss_start = get_process_rss_mb()
    peak_rss = rss_start

    discovery = discover_unique_daily_members(zip_paths, data_dir=data_dir)
    members: list[ZipMemberInfo] = discovery["members"]

    validation: dict[str, Any] = {
        "n_unique_daily_files": discovery["n_unique_daily_files"],
        "expected_n_days": expected_n_days,
        "exact_62_files": discovery["n_unique_daily_files"] == expected_n_days,
        "duplicate_dates": discovery["duplicate_dates"],
        "missing_dates": discovery["missing_dates"],
        "dates_continuous": discovery["dates_continuous"],
        "non_daily_members": discovery["non_daily_members"],
        "zip_open_failures": [],
        "file_failures": [],
        "incomplete": False,
        "stopped_after_date": None,
    }

    if discovery["duplicate_dates"]:
        validation["warnings"] = validation.get("warnings", [])
        validation["warnings"].append(
            "Duplicate daily filenames found across ZIPs; first occurrence used."
        )

    square_traffic: dict[int, float] = defaultdict(float)
    square_obs: dict[int, int] = defaultdict(int)
    all_country_codes: set[int] = set()
    all_timestamps: set[int] = set()
    all_squares: set[int] = set()
    daily_frames: list[pd.DataFrame] = []
    log_rows: list[dict[str, Any]] = []

    total_raw_rows = 0
    total_aggregated_rows = 0
    total_missing_or_invalid = 0
    files_processed = 0

    for member in members:
        date_str = parse_member_date(member.member_name)
        day_t0 = time.perf_counter()
        txt_path: Optional[Path] = None
        try:
            try:
                txt_path = extract_zip_member(
                    member.zip_path, member.member_name, tmp_dir
                )
            except Exception as exc:  # noqa: BLE001 - report and decide continuity
                validation["zip_open_failures"].append(
                    {
                        "zip_file": member.zip_path.name,
                        "filename": member.member_name,
                        "error": str(exc),
                    }
                )
                validation["incomplete"] = True
                validation["stopped_after_date"] = date_str
                # Cannot skip a day without compromising completeness of totals
                break

            day_result = process_daily_internet_file(
                txt_path,
                chunksize=chunksize,
                strategy=strategy,
            )
        except Exception as exc:  # noqa: BLE001
            validation["file_failures"].append(
                {
                    "date": date_str,
                    "zip_file": member.zip_path.name,
                    "filename": member.member_name,
                    "error": str(exc),
                }
            )
            validation["incomplete"] = True
            validation["stopped_after_date"] = date_str
            if txt_path is not None:
                remove_path(txt_path)
            break
        finally:
            if txt_path is not None:
                remove_path(txt_path)

        # Update cumulative structures
        daily = day_result["daily_by_square"]
        daily = daily.copy()
        daily.insert(0, "date", date_str)
        daily_frames.append(daily)

        for row in daily.itertuples(index=False):
            sid = int(row.square_id)
            square_traffic[sid] += float(row.daily_internet_traffic)
            square_obs[sid] += int(row.observation_count)

        all_country_codes.update(day_result["country_codes"])
        all_timestamps.update(day_result["unique_timestamps"])
        all_squares.update(day_result["unique_squares"])
        total_raw_rows += day_result["raw_rows"]
        total_aggregated_rows += day_result["aggregated_rows"]
        total_missing_or_invalid += day_result["missing_or_invalid_rows"]
        files_processed += 1

        day_seconds = time.perf_counter() - day_t0
        peak_rss = max(peak_rss, get_process_rss_mb())

        log_rows.append(
            {
                "date": date_str,
                "zip_file": member.zip_path.name,
                "filename": member.member_name,
                "raw_rows": day_result["raw_rows"],
                "aggregated_rows": day_result["aggregated_rows"],
                "unique_squares": len(day_result["unique_squares"]),
                "unique_timestamps": len(day_result["unique_timestamps"]),
                "internet_traffic_sum": day_result["internet_traffic_sum"],
                "processing_time_seconds": round(day_seconds, 3),
                "schema_ok": day_result["schema_ok"],
                "timestamps_all_10_minutes": day_result["timestamps_all_10_minutes"],
                "missing_or_invalid_rows": day_result["missing_or_invalid_rows"],
            }
        )

        if not day_result["schema_ok"] or not day_result["timestamps_all_10_minutes"]:
            validation.setdefault("day_quality_flags", []).append(
                {
                    "date": date_str,
                    "schema_ok": day_result["schema_ok"],
                    "schema_notes": day_result["schema_notes"],
                    "timestamps_all_10_minutes": day_result[
                        "timestamps_all_10_minutes"
                    ],
                    "timestamp_diff_counts": day_result["timestamp_diff_counts"],
                }
            )

        del day_result, daily
        gc.collect()
        peak_rss = max(peak_rss, get_process_rss_mb())
        print(
            f"[{files_processed}/{len(members)}] {date_str} "
            f"raw={log_rows[-1]['raw_rows']} agg={log_rows[-1]['aggregated_rows']} "
            f"t={log_rows[-1]['processing_time_seconds']}s "
            f"rss={get_process_rss_mb():.1f}MB",
            flush=True,
        )

    processing_time = time.perf_counter() - t_start
    rss_end = get_process_rss_mb()

    # Build output tables
    square_totals = (
        pd.DataFrame(
            {
                "square_id": list(square_traffic.keys()),
                "total_internet_traffic": list(square_traffic.values()),
                "observation_count": [square_obs[s] for s in square_traffic.keys()],
            }
        )
        .sort_values("total_internet_traffic", ascending=False)
        .reset_index(drop=True)
    )

    if daily_frames:
        daily_totals = (
            pd.concat(daily_frames, ignore_index=True)
            .sort_values(["date", "square_id"])
            .reset_index(drop=True)
        )
    else:
        daily_totals = pd.DataFrame(
            columns=[
                "date",
                "square_id",
                "daily_internet_traffic",
                "observation_count",
            ]
        )

    # Validation: cumulative square totals vs sum of daily totals
    daily_sum = float(daily_totals["daily_internet_traffic"].sum()) if len(daily_totals) else 0.0
    square_sum = float(square_totals["total_internet_traffic"].sum()) if len(square_totals) else 0.0
    validation["cumulative_equals_daily_sum"] = bool(
        np.isclose(daily_sum, square_sum, rtol=0.0, atol=1e-3)
        or (daily_sum == 0.0 and square_sum == 0.0)
    )
    validation["daily_sum_total_internet"] = daily_sum
    validation["square_sum_total_internet"] = square_sum
    validation["abs_diff_daily_vs_square_sum"] = abs(daily_sum - square_sum)
    validation["files_processed"] = files_processed
    validation["files_expected"] = len(members)
    validation["all_files_processed"] = files_processed == len(members) and not validation[
        "incomplete"
    ]

    ts_sorted = sorted(all_timestamps)
    if len(ts_sorted) > 1:
        global_diffs = np.diff(ts_sorted)
        # Global unique timestamps across days are not all adjacent by 10 min
        # (day boundaries / gaps). Report modal positive diff among consecutive
        # unique timestamps that equal 600000 where present.
        validation["global_adjacent_diff_counts"] = {
            int(k): int(v)
            for k, v in zip(*np.unique(global_diffs, return_counts=True))
        }
        validation["fraction_global_diffs_10_minutes"] = float(
            (global_diffs == 600_000).mean()
        )
    else:
        validation["global_adjacent_diff_counts"] = {}
        validation["fraction_global_diffs_10_minutes"] = None

    top10 = square_totals.head(10).copy()
    top10.insert(0, "rank", range(1, len(top10) + 1))
    top10_records = [
        {
            "rank": int(r.rank),
            "square_id": int(r.square_id),
            "total_internet_traffic": float(r.total_internet_traffic),
            "observation_count": int(r.observation_count),
        }
        for r in top10.itertuples(index=False)
    ]

    # Write outputs
    square_path = out_dir / "square_internet_totals.csv"
    daily_path = out_dir / "daily_square_internet_totals.csv"
    log_path = met_dir / "phase3_processing_log.csv"
    summary_path = met_dir / "phase3_dataset_summary.json"

    square_totals.to_csv(square_path, index=False)
    daily_totals.to_csv(daily_path, index=False)
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    summary = {
        "number_of_zip_files": discovery["n_zip_files"],
        "zip_files": discovery["zip_files"],
        "number_of_daily_files": files_processed,
        "number_of_daily_files_discovered": discovery["n_unique_daily_files"],
        "first_date": discovery["first_date"],
        "last_date": discovery["last_date"],
        "number_of_unique_squares": len(all_squares),
        "total_raw_rows_processed": total_raw_rows,
        "total_aggregated_square_time_rows": total_aggregated_rows,
        "number_of_unique_timestamps": len(all_timestamps),
        "timestamp_interval_minutes": 10,
        "aggregation_method": strategy,
        "number_of_country_codes": len(all_country_codes),
        "total_internet_traffic": square_sum,
        "missing_or_invalid_rows": total_missing_or_invalid,
        "processing_time_seconds": round(processing_time, 3),
        "peak_memory_mb": round(peak_rss, 3),
        "rss_start_mb": round(rss_start, 3),
        "rss_end_mb": round(rss_end, 3),
        "chunksize": chunksize,
        "top_10_squares_by_total_internet_traffic": top10_records,
        "top_3_squares_by_total_internet_traffic": top10_records[:3],
        "output_files": {
            "square_internet_totals": str(square_path),
            "daily_square_internet_totals": str(daily_path),
            "processing_log": str(log_path),
            "dataset_summary": str(summary_path),
        },
        "validation": validation,
        "notes": {
            "missing_square_timestamp_policy": (
                "Absent square×timestamp combinations were NOT filled with zeros."
            ),
            "full_panel_not_materialised": (
                "Only per-square and daily-per-square summaries were written."
            ),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# Alias requested in the Phase 3A brief
process_full_dataset = build_square_traffic_summary

# Phase 3B target areas: top-3 by total Internet + assignment-required squares.
TOP_SQUARES: list[int] = [5161, 5059, 5259]
ASSIGNMENT_SQUARES: list[int] = [4159, 4556]
TARGET_SQUARES: list[int] = TOP_SQUARES + ASSIGNMENT_SQUARES


def extract_target_square_timeseries(
    *,
    data_dir: Optional[PathLike] = None,
    zip_paths: Optional[Sequence[PathLike]] = None,
    target_squares: Optional[Sequence[int]] = None,
    output_dir: Optional[PathLike] = None,
    temp_dir: Optional[PathLike] = None,
    chunksize: int = 500_000,
    strategy: AggregationStrategy = "sum_all_countries",
    write_parquet: bool = True,
) -> pd.DataFrame:
    """
    Phase 3B: extract 10-minute Internet series for selected squares only.

    Processes one daily ZIP member at a time, filters to ``target_squares``
    inside each chunk, aggregates with ``sum_all_countries`` by default, and
    does **not** fill missing square×timestamp bins with zeros.

    Writes:
    - ``data/processed/target_square_internet_timeseries.csv``
    - ``data/processed/target_square_internet_timeseries.parquet`` (if pyarrow available)
    """
    squares = [int(s) for s in (target_squares or TARGET_SQUARES)]
    square_set = set(squares)

    root = Path(__file__).resolve().parents[2]
    out_dir = Path(output_dir) if output_dir is not None else root / "data" / "processed"
    tmp_dir = (
        Path(temp_dir) if temp_dir is not None else root / "tmp" / "phase3b_extract"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    discovery = discover_unique_daily_members(zip_paths, data_dir=data_dir)
    members: list[ZipMemberInfo] = discovery["members"]
    if not members:
        raise FileNotFoundError("No daily dataset members discovered for extraction.")

    day_frames: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    peak_rss = get_process_rss_mb()

    for i, member in enumerate(members, start=1):
        date_str = parse_member_date(member.member_name)
        txt_path: Optional[Path] = None
        try:
            txt_path = extract_zip_member(
                member.zip_path, member.member_name, tmp_dir
            )
            chunk_aggs: list[pd.DataFrame] = []
            for chunk in read_traffic_file(
                txt_path,
                chunksize=chunksize,
                usecols=INTERNET_TASK_COLUMNS,
                dtype=INTERNET_TASK_DTYPES,
            ):
                filtered = chunk.loc[chunk["square_id"].isin(square_set)].copy()
                del chunk
                if filtered.empty:
                    continue
                filtered["internet_traffic"] = filtered["internet_traffic"].astype(
                    "float64"
                )
                # Count country-level non-null Internet rows contributing to each bin
                contrib = (
                    filtered.loc[filtered["internet_traffic"].notna()]
                    .groupby(["square_id", "time_interval"], sort=False)
                    .size()
                    .rename("observation_count")
                    .reset_index()
                )
                agg = aggregate_internet_traffic(filtered, strategy=strategy)
                merged = agg.merge(
                    contrib, on=["square_id", "time_interval"], how="left"
                )
                merged["observation_count"] = (
                    merged["observation_count"].fillna(0).astype("int32")
                )
                chunk_aggs.append(merged)
                del filtered, agg, contrib, merged

            if chunk_aggs:
                day_agg = merge_chunk_aggregates(
                    [c.drop(columns=["observation_count"]) for c in chunk_aggs]
                )
                # Re-sum observation counts across chunks for the same keys
                obs = (
                    pd.concat(
                        [
                            c[["square_id", "time_interval", "observation_count"]]
                            for c in chunk_aggs
                        ],
                        ignore_index=True,
                    )
                    .groupby(["square_id", "time_interval"], sort=False)[
                        "observation_count"
                    ]
                    .sum()
                    .reset_index()
                )
                day_agg = day_agg.merge(
                    obs, on=["square_id", "time_interval"], how="left"
                )
                day_agg["observation_count"] = (
                    day_agg["observation_count"].fillna(0).astype("int32")
                )
                day_agg = day_agg.loc[day_agg["internet_traffic"].notna()].copy()
                day_agg["date"] = date_str
                day_frames.append(day_agg)
                del day_agg, obs
            del chunk_aggs
        finally:
            if txt_path is not None:
                remove_path(txt_path)
            gc.collect()
            peak_rss = max(peak_rss, get_process_rss_mb())
            print(
                f"[extract {i}/{len(members)}] {date_str} "
                f"rss={get_process_rss_mb():.1f}MB",
                flush=True,
            )

    if not day_frames:
        raise RuntimeError("Target-square extraction produced no rows.")

    result = pd.concat(day_frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(
        result["time_interval"], unit="ms", utc=True
    )
    # Keep a calendar date column aligned to filename date (assignment windows)
    result = result[
        ["timestamp", "date", "square_id", "internet_traffic", "observation_count"]
    ].copy()
    result["square_id"] = result["square_id"].astype("int32")
    result = result.sort_values(["square_id", "timestamp"]).reset_index(drop=True)

    # Sanity checks
    unexpected = sorted(set(result["square_id"].unique()) - square_set)
    if unexpected:
        raise ValueError(f"Unexpected square_ids in extraction output: {unexpected}")
    missing_targets = sorted(square_set - set(result["square_id"].unique()))
    if missing_targets:
        # Not fatal if a required square never appears, but report clearly
        print(f"WARNING: no rows for squares: {missing_targets}", flush=True)

    csv_path = out_dir / "target_square_internet_timeseries.csv"
    result.to_csv(csv_path, index=False)

    parquet_path = out_dir / "target_square_internet_timeseries.parquet"
    if write_parquet:
        try:
            result.to_parquet(parquet_path, index=False)
        except Exception as exc:  # noqa: BLE001
            print(f"Parquet write skipped: {exc}", flush=True)
            parquet_path = None

    elapsed = time.perf_counter() - t0
    meta = {
        "target_squares": squares,
        "n_rows": int(len(result)),
        "rows_by_square": {
            int(k): int(v)
            for k, v in result.groupby("square_id").size().items()
        },
        "first_timestamp": str(result["timestamp"].min()),
        "last_timestamp": str(result["timestamp"].max()),
        "aggregation_method": strategy,
        "processing_time_seconds": round(elapsed, 3),
        "peak_memory_mb": round(peak_rss, 3),
        "csv_path": str(csv_path),
        "parquet_path": str(parquet_path) if parquet_path else None,
        "missing_filled_with_zero": False,
        "n_daily_files": len(members),
        "date_range": [discovery["first_date"], discovery["last_date"]],
    }
    meta_path = out_dir.parent.parent / "results" / "metrics" / "phase3b_extraction_meta.json"
    # out_dir is data/processed → parent.parent is project root only if structure matches
    # Prefer metrics next to other phase metrics
    metrics_dir = root / "results" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    meta_path = metrics_dir / "phase3b_extraction_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Wrote {csv_path} rows={len(result)} "
        f"time={elapsed:.1f}s peak_rss={peak_rss:.1f}MB",
        flush=True,
    )
    return result
