from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_fvecs(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(4)
    if len(raw) != 4:
        raise ValueError(f"empty fvecs file: {path}")
    dimension = int(np.frombuffer(raw, dtype="<i4")[0])
    row_bytes = 4 * (dimension + 1)
    if dimension <= 0 or path.stat().st_size % row_bytes:
        raise ValueError(f"invalid fvecs layout: {path}")
    rows = path.stat().st_size // row_bytes
    data = np.memmap(path, dtype="<i4", mode="r", shape=(rows, dimension + 1))
    if not np.all(data[:, 0] == dimension):
        raise ValueError(f"inconsistent fvecs dimensions: {path}")
    return data[:, 1:].view("<f4")


def read_ivecs(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(4)
    if len(raw) != 4:
        raise ValueError(f"empty ivecs file: {path}")
    width = int(np.frombuffer(raw, dtype="<i4")[0])
    row_bytes = 4 * (width + 1)
    if width <= 0 or path.stat().st_size % row_bytes:
        raise ValueError(f"invalid ivecs layout: {path}")
    rows = path.stat().st_size // row_bytes
    data = np.memmap(path, dtype="<i4", mode="r", shape=(rows, width + 1))
    if not np.all(data[:, 0] == width):
        raise ValueError(f"inconsistent ivecs widths: {path}")
    return np.asarray(data[:, 1:])


def read_attribute_rows(path: str | Path, value_type: str) -> list[list[int | float]]:
    caster = float if value_type in {"float", "double"} else int
    rows: list[list[int | float]] = []
    with Path(path).open(newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or any(not value.strip() for value in row):
                raise ValueError(f"{path}:{line_number}: empty attribute value")
            rows.append([caster(value) for value in row])
    return rows
