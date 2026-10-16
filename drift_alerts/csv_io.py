"""CSV loading for scored-predictions files."""

from __future__ import annotations

import csv


def read_csv(path: str) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV into (columns, rows). Raises ValueError on a bad file."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: no header row found")
        columns = list(reader.fieldnames)
        rows = [{k: (v if v is not None else "") for k, v in row.items()}
                for row in reader]
    if not rows:
        raise ValueError(f"{path}: no data rows found")
    return columns, rows


def is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except (ValueError, TypeError):
        return False


def infer_kind(values: list[str]) -> str:
    """'numeric' if every non-missing value parses as float, else 'categorical'."""
    present = [v for v in values if v != ""]
    if not present:
        return "categorical"
    return "numeric" if all(is_float(v) for v in present) else "categorical"


def present_values(values: list[str]) -> tuple[list[str], int]:
    """Split raw column values into (non-missing values, missing count)."""
    present = [v for v in values if v != ""]
    return present, len(values) - len(present)
