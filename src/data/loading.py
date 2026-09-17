"""
Memory-efficient loading and inspection utilities for the Milan
telecommunications dataset (Harvard Dataverse).

Raw ZIP archives are expected to live outside the repository. Paths are
resolved from function arguments, then environment variables, then optional
discovery under a configured data directory — not from personal hard-coded
paths.
"""

from __future__ import annotations

import os
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import pandas as pd

PathLike = Union[str, Path]

COLUMN_NAMES: tuple[str, ...] = (
    "square_id",
    "time_interval",
    "country_code",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet_traffic",
)

# Backward-compatible alias used in Phase 1 docs/code.
PROPOSED_COLUMN_NAMES = COLUMN_NAMES

# Columns required for Internet traffic forecasting after aggregation.
INTERNET_TASK_COLUMNS: tuple[str, ...] = (
    "square_id",
    "time_interval",
    "country_code",
    "internet_traffic",
)

# Explicit dtypes for the Internet-task subset. Rationale is documented in
# docs/methodology.md (validated against float64 on the Phase 2 prototype day).
INTERNET_TASK_DTYPES: dict[str, str] = {
    "square_id": "int32",
    "time_interval": "int64",
    "country_code": "int32",
    "internet_traffic": "float32",
}

FULL_TABLE_DTYPES: dict[str, str] = {
    "square_id": "int32",
    "time_interval": "int64",
    "country_code": "int32",
    "sms_in": "float32",
    "sms_out": "float32",
    "call_in": "float32",
    "call_out": "float32",
    "internet_traffic": "float32",
}


@dataclass(frozen=True)
class ZipMemberInfo:
    zip_path: Path
    member_name: str
    compressed_size: int
    uncompressed_size: int


def _as_path(path: PathLike) -> Path:
    return Path(path)


def get_process_rss_mb() -> float:
    """Return current process resident set size in megabytes."""
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _env_zip_paths() -> list[Path]:
    """Parse MILAN_ZIP_PATHS (os.pathsep-separated) if set."""
    raw = os.environ.get("MILAN_ZIP_PATHS", "").strip()
    if not raw:
        return []
    return [_as_path(part.strip()) for part in raw.split(os.pathsep) if part.strip()]


def _env_data_dir() -> Optional[Path]:
    for key in ("MILAN_DATA_DIR", "DATA_DIR"):
        value = os.environ.get(key, "").strip()
        if value:
            return _as_path(value)
    return None


def discover_zip_archives(data_dir: PathLike) -> list[Path]:
    """
    Discover ``dataverse_files*.zip`` archives under ``data_dir`` (non-recursive).

    Names may vary across machines / Colab Drive mounts; only the glob pattern
    is assumed, not a fixed personal path.
    """
    root = _as_path(data_dir)
    if not root.is_dir():
        return []
    return sorted(root.glob("dataverse_files*.zip"))


def resolve_zip_paths(
    zip_paths: Optional[Sequence[PathLike]] = None,
    *,
    data_dir: Optional[PathLike] = None,
    missing_ok: bool = False,
) -> list[Path]:
    """
    Resolve existing ZIP paths.

    Priority:
    1. Explicit ``zip_paths`` argument
    2. ``MILAN_ZIP_PATHS`` environment variable
    3. Discovery under ``data_dir`` / ``MILAN_DATA_DIR`` / ``DATA_DIR``
    """
    candidates: list[Path] = []
    if zip_paths is not None:
        candidates.extend(_as_path(p) for p in zip_paths)
    else:
        candidates.extend(_env_zip_paths())
        root = _as_path(data_dir) if data_dir is not None else _env_data_dir()
        if root is not None:
            candidates.extend(discover_zip_archives(root))

    # De-duplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)

    existing = [p for p in unique if p.is_file()]
    if not existing and not missing_ok:
        hint = (
            "Pass zip_paths=..., set MILAN_ZIP_PATHS, or set MILAN_DATA_DIR / "
            "DATA_DIR to a folder containing dataverse_files*.zip archives."
        )
        raise FileNotFoundError(f"No dataset ZIP files found. {hint}")
    return existing


def list_dataset_files(
    zip_paths: Optional[Sequence[PathLike]] = None,
    *,
    data_dir: Optional[PathLike] = None,
) -> list[ZipMemberInfo]:
    """List TXT members inside the dataset ZIP archives (no extraction)."""
    members: list[ZipMemberInfo] = []
    for zp in resolve_zip_paths(zip_paths, data_dir=data_dir):
        with zipfile.ZipFile(zp, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                members.append(
                    ZipMemberInfo(
                        zip_path=zp,
                        member_name=info.filename,
                        compressed_size=info.compress_size,
                        uncompressed_size=info.file_size,
                    )
                )
    members.sort(key=lambda m: m.member_name)
    return members


def inspect_zip_contents(
    zip_paths: Optional[Sequence[PathLike]] = None,
    *,
    data_dir: Optional[PathLike] = None,
) -> list[dict]:
    """Summarize each ZIP: path, size, member count, and member names."""
    summaries: list[dict] = []
    for zp in resolve_zip_paths(zip_paths, data_dir=data_dir):
        with zipfile.ZipFile(zp, "r") as zf:
            names = sorted(i.filename for i in zf.infolist() if not i.is_dir())
            uncompressed = sum(i.file_size for i in zf.infolist())
        summaries.append(
            {
                "zip_path": str(zp),
                "zip_name": zp.name,
                "compressed_bytes": zp.stat().st_size,
                "uncompressed_bytes": uncompressed,
                "n_members": len(names),
                "member_names": names,
            }
        )
    return summaries


def iter_text_lines_from_zip(
    zip_path: PathLike,
    member_name: str,
    *,
    max_bytes: int = 65_536,
    max_lines: Optional[int] = None,
) -> Iterator[str]:
    """
    Stream decoded text lines from a ZIP member without full extraction.

    Reads at most ``max_bytes`` from the start of the compressed member.
    Incomplete trailing lines are discarded.
    """
    zp = _as_path(zip_path)
    with zipfile.ZipFile(zp, "r") as zf:
        with zf.open(member_name) as handle:
            raw = handle.read(max_bytes)
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if raw and not text.endswith(("\n", "\r")):
        lines = lines[:-1]
    if max_lines is not None:
        lines = lines[:max_lines]
    yield from lines


def preview_text_file(
    zip_path: PathLike,
    member_name: str,
    *,
    n_lines: int = 20,
    max_bytes: int = 65_536,
) -> list[str]:
    """Return the first ``n_lines`` of a TXT member inside a ZIP."""
    return list(
        iter_text_lines_from_zip(
            zip_path,
            member_name,
            max_bytes=max_bytes,
            max_lines=n_lines,
        )
    )


def _split_fields(line: str) -> list[str]:
    return line.split("\t")


def inspect_sample_schema(
    zip_path: PathLike,
    member_name: str,
    *,
    n_lines: int = 200,
    max_bytes: int = 200_000,
) -> dict:
    """
    Inspect delimiter, column count, dtypes, and missingness on a small sample.

    Does not load the full file into memory or into a DataFrame.
    """
    lines = list(
        iter_text_lines_from_zip(
            zip_path,
            member_name,
            max_bytes=max_bytes,
            max_lines=n_lines,
        )
    )
    field_counts: Counter[int] = Counter()
    empty_by_col: Counter[int] = Counter()
    type_by_col: dict[int, Counter[str]] = {}
    tab_lines = 0

    for line in lines:
        if "\t" in line:
            tab_lines += 1
        parts = _split_fields(line)
        field_counts[len(parts)] += 1
        for idx, value in enumerate(parts):
            if value == "":
                empty_by_col[idx] += 1
                continue
            bucket = type_by_col.setdefault(idx, Counter())
            try:
                if "." in value or "e" in value.lower():
                    float(value)
                    bucket["float"] += 1
                else:
                    int(value)
                    bucket["int"] += 1
            except ValueError:
                bucket["other"] += 1

    n = len(lines)
    return {
        "zip_path": str(zip_path),
        "member_name": member_name,
        "n_lines_sampled": n,
        "tab_separated_fraction": (tab_lines / n) if n else None,
        "field_count_distribution": dict(field_counts),
        "appears_headerless": n > 0
        and all(_looks_numeric_row(_split_fields(line)) for line in lines[:5]),
        "empty_counts_by_column": dict(empty_by_col),
        "observed_types_by_column": {
            idx: dict(counter) for idx, counter in sorted(type_by_col.items())
        },
        "preview_rows": [_split_fields(line) for line in lines[:5]],
    }


def _looks_numeric_row(parts: Sequence[str]) -> bool:
    if len(parts) < 3:
        return False
    try:
        int(parts[0])
        int(parts[1])
        int(parts[2])
        return True
    except ValueError:
        return False


def check_timestamp_intervals(
    zip_path: PathLike,
    member_name: str,
    *,
    n_lines: int = 5_000,
    max_bytes: int = 500_000,
    square_id: Optional[int] = 1,
    timestamp_column: int = 1,
    square_column: int = 0,
) -> dict:
    """
    Check adjacent unique timestamp differences for a square (or all rows).

    Expected interval for this dataset is 600_000 ms (10 minutes).
    """
    lines = list(
        iter_text_lines_from_zip(
            zip_path,
            member_name,
            max_bytes=max_bytes,
            max_lines=n_lines,
        )
    )
    timestamps: list[int] = []
    for line in lines:
        parts = _split_fields(line)
        if len(parts) <= max(timestamp_column, square_column):
            continue
        if square_id is not None and parts[square_column] != str(square_id):
            continue
        try:
            timestamps.append(int(parts[timestamp_column]))
        except ValueError:
            continue

    unique_ts = sorted(set(timestamps))
    diffs = [unique_ts[i + 1] - unique_ts[i] for i in range(len(unique_ts) - 1)]
    diff_counts = Counter(diffs)

    first_utc = (
        datetime.fromtimestamp(unique_ts[0] / 1000, tz=timezone.utc).isoformat()
        if unique_ts
        else None
    )
    last_utc = (
        datetime.fromtimestamp(unique_ts[-1] / 1000, tz=timezone.utc).isoformat()
        if unique_ts
        else None
    )

    return {
        "zip_path": str(zip_path),
        "member_name": member_name,
        "square_id_filter": square_id,
        "n_rows_considered": len(timestamps),
        "n_unique_timestamps": len(unique_ts),
        "first_timestamp_ms": unique_ts[0] if unique_ts else None,
        "last_timestamp_ms": unique_ts[-1] if unique_ts else None,
        "first_timestamp_utc": first_utc,
        "last_timestamp_utc": last_utc,
        "diff_counts_ms": dict(sorted(diff_counts.items())),
        "all_diffs_are_10_minutes": bool(diffs) and all(d == 600_000 for d in diffs),
        "fraction_10_minute_diffs": (
            sum(1 for d in diffs if d == 600_000) / len(diffs) if diffs else None
        ),
    }


def find_member(
    member_name: str,
    zip_paths: Optional[Sequence[PathLike]] = None,
    *,
    data_dir: Optional[PathLike] = None,
) -> ZipMemberInfo:
    """Locate a TXT member across configured ZIP archives."""
    for member in list_dataset_files(zip_paths, data_dir=data_dir):
        if member.member_name == member_name:
            return member
    raise FileNotFoundError(f"Member not found in configured ZIPs: {member_name}")


def parse_member_date(member_name: str) -> Optional[str]:
    """
    Parse ``YYYY-MM-DD`` from ``sms-call-internet-mi-YYYY-MM-DD.txt``.

    Returns None if the name does not match the expected pattern.
    """
    stem = Path(member_name).stem
    marker = "-mi-"
    if marker not in stem:
        return None
    date_str = stem.split(marker, 1)[1]
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None
    return date_str


def summarize_filename_date_coverage(
    zip_paths: Optional[Sequence[PathLike]] = None,
    *,
    data_dir: Optional[PathLike] = None,
) -> dict:
    """Derive the date span implied by ``sms-call-internet-mi-YYYY-MM-DD.txt`` names."""
    members = list_dataset_files(zip_paths, data_dir=data_dir)
    dates: list[str] = []
    for m in members:
        date_str = parse_member_date(m.member_name)
        if date_str is not None:
            dates.append(date_str)
    dates = sorted(set(dates))
    return {
        "n_data_files": len(members),
        "n_unique_dates": len(dates),
        "earliest_date": dates[0] if dates else None,
        "latest_date": dates[-1] if dates else None,
        "dates": dates,
    }


def discover_unique_daily_members(
    zip_paths: Optional[Sequence[PathLike]] = None,
    *,
    data_dir: Optional[PathLike] = None,
) -> dict:
    """
    Discover daily TXT members across ZIPs, deduplicated by filename date.

    Returns a dict with sorted unique members, duplicate reports, and
    missing-date checks between the earliest and latest filename dates.
    """
    from datetime import date, timedelta

    members = list_dataset_files(zip_paths, data_dir=data_dir)
    by_date: dict[str, list[ZipMemberInfo]] = {}
    non_daily: list[str] = []
    for member in members:
        date_str = parse_member_date(member.member_name)
        if date_str is None:
            non_daily.append(member.member_name)
            continue
        by_date.setdefault(date_str, []).append(member)

    duplicates = {
        d: [m.member_name for m in ms]
        for d, ms in by_date.items()
        if len(ms) > 1
    }
    # Prefer the first occurrence when duplicates exist.
    unique_members = [by_date[d][0] for d in sorted(by_date)]
    dates = sorted(by_date)
    missing_dates: list[str] = []
    if dates:
        start = date.fromisoformat(dates[0])
        end = date.fromisoformat(dates[-1])
        expected: list[str] = []
        cur = start
        while cur <= end:
            expected.append(cur.isoformat())
            cur += timedelta(days=1)
        missing_dates = [d for d in expected if d not in by_date]

    zip_names = sorted({m.zip_path.name for m in unique_members})
    return {
        "members": unique_members,
        "n_zip_files": len(zip_names),
        "zip_files": zip_names,
        "n_unique_daily_files": len(unique_members),
        "dates": dates,
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "duplicate_dates": duplicates,
        "missing_dates": missing_dates,
        "non_daily_members": non_daily,
        "dates_continuous": len(missing_dates) == 0 and len(unique_members) > 0,
    }


def extract_zip_member(
    zip_path: PathLike,
    member_name: str,
    destination_dir: PathLike,
) -> Path:
    """
    Extract a single ZIP member into ``destination_dir``.

    Only the requested member is written. Caller should delete the file when
    finished (see ``remove_path``).
    """
    zp = _as_path(zip_path)
    out_dir = _as_path(destination_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / Path(member_name).name
    with zipfile.ZipFile(zp, "r") as zf:
        with zf.open(member_name) as src, open(target, "wb") as dst:
            while True:
                block = src.read(1024 * 1024)
                if not block:
                    break
                dst.write(block)
    return target


def remove_path(path: PathLike) -> None:
    """Delete a file if it exists; ignore missing paths."""
    p = _as_path(path)
    if p.is_file():
        p.unlink()


def read_traffic_file(
    file_path: PathLike,
    *,
    chunksize: Optional[int] = 500_000,
    usecols: Optional[Sequence[str]] = None,
    dtype: Optional[dict] = None,
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """
    Read a Milan traffic TXT file with explicit schema.

    - Tab-separated, headerless
    - Empty fields become NaN
    - Column names assigned from ``COLUMN_NAMES``
    - When ``chunksize`` is set, returns an iterator of DataFrames
    - When ``chunksize`` is None, returns a single DataFrame (baseline use)
    """
    path = _as_path(file_path)
    selected = list(usecols) if usecols is not None else list(COLUMN_NAMES)
    unknown = set(selected) - set(COLUMN_NAMES)
    if unknown:
        raise ValueError(f"Unknown columns requested: {sorted(unknown)}")

    name_to_index = {name: idx for idx, name in enumerate(COLUMN_NAMES)}
    col_indices = [name_to_index[name] for name in selected]
    resolved_dtype = None
    if dtype is not None:
        resolved_dtype = {name: dtype[name] for name in selected if name in dtype}

    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=list(COLUMN_NAMES),
        usecols=col_indices,
        dtype=resolved_dtype,
        na_values=[""],
        keep_default_na=True,
        chunksize=chunksize,
        engine="c",
    )


def add_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``timestamp`` from ``time_interval`` (Unix ms → UTC datetime64[ns]).

    Does not resample or interpolate.
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["time_interval"], unit="ms", utc=True)
    return out


def validate_float32_precision(
    series_f64: pd.Series,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> dict:
    """Compare float64 values against float32 cast for unexpected loss."""
    valid = series_f64.dropna()
    if valid.empty:
        return {
            "n_compared": 0,
            "max_abs_error": None,
            "max_rel_error": None,
            "within_tolerance": True,
        }
    as_f32 = valid.astype("float32").astype("float64")
    abs_err = (valid.astype("float64") - as_f32).abs()
    rel_err = abs_err / valid.astype("float64").abs().clip(lower=atol)
    max_abs = float(abs_err.max())
    max_rel = float(rel_err.max())
    ok = bool(((abs_err <= atol) | (rel_err <= rtol)).all())
    return {
        "n_compared": int(len(valid)),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "within_tolerance": ok,
    }
