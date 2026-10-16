"""Reference/current window splitting for scored-predictions rows."""

from __future__ import annotations

from datetime import datetime


def parse_ts(text: str) -> datetime:
    """Parse an ISO-8601 timestamp; raises ValueError with a clear message."""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        raise ValueError(f"unparseable timestamp: {text!r} (expected ISO-8601)")


def window_range(rows: list[dict[str, str]], ts_col: str) -> tuple[str, str]:
    """(min, max) timestamp of a window as ISO strings."""
    stamps = [r[ts_col] for r in rows]
    return min(stamps), max(stamps)


def split_windows(rows: list[dict[str, str]],
                  ts_col: str = "timestamp",
                  ref_rows: int | None = None,
                  cur_rows: int | None = None,
                  split_at: str | None = None,
                  ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split rows into (reference, current) windows ordered by timestamp.

    Modes:
    - ``split_at``: rows with timestamp < split_at go to reference, the rest
      to current.
    - ``ref_rows`` / ``cur_rows``: first N rows are reference, last M rows are
      current (after timestamp sort).
    - default: first half reference, second half current.
    """
    if ts_col not in rows[0]:
        raise ValueError(f"timestamp column {ts_col!r} not found in CSV")

    bad = sum(1 for r in rows if not _parseable(r.get(ts_col, "")))
    if bad:
        raise ValueError(
            f"{bad} row(s) have unparseable {ts_col!r} values (expected ISO-8601)")

    ordered = sorted(rows, key=lambda r: parse_ts(r[ts_col]))

    if split_at is not None:
        cut = parse_ts(split_at)
        ref = [r for r in ordered if _safe_lt(parse_ts(r[ts_col]), cut)]
        cur = [r for r in ordered if not _safe_lt(parse_ts(r[ts_col]), cut)]
    elif ref_rows is not None or cur_rows is not None:
        n_ref = ref_rows if ref_rows is not None else len(ordered) // 2
        n_cur = cur_rows if cur_rows is not None else len(ordered) // 2
        if n_ref < 1 or n_cur < 1:
            raise ValueError("--ref-rows and --cur-rows must each be >= 1")
        if n_ref + n_cur > len(ordered):
            raise ValueError(
                f"ref_rows ({n_ref}) + cur_rows ({n_cur}) exceeds "
                f"{len(ordered)} data rows")
        ref = ordered[:n_ref]
        cur = ordered[len(ordered) - n_cur:]
    else:
        half = len(ordered) // 2
        ref, cur = ordered[:half], ordered[half:]

    if len(ref) < 2 or len(cur) < 2:
        raise ValueError(
            f"need at least 2 rows per window, got {len(ref)} reference "
            f"and {len(cur)} current")
    return ref, cur


def _parseable(text: str) -> bool:
    try:
        parse_ts(text)
        return True
    except ValueError:
        return False


def _safe_lt(a: datetime, b: datetime) -> bool:
    try:
        return a < b
    except TypeError:
        raise ValueError(
            "cannot compare timestamps: mixing timezone-aware and "
            "timezone-naive values")
