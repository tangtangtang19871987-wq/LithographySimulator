"""Minimal NPZ-like helpers without third-party dependencies.

This demo stores arrays as JSON in a .npz (ZIP) container for portability.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Iterable


def save_array(path: str | Path, key: str, array2d: list[list[float]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        payload = {"key": key, "array": array2d}
        zf.writestr("data.json", json.dumps(payload))


def load_array(path: str | Path) -> tuple[str, list[list[float]]]:
    p = Path(path)
    with zipfile.ZipFile(p, "r") as zf:
        raw = zf.read("data.json").decode("utf-8")
    payload = json.loads(raw)
    return str(payload.get("key", "aerial")), payload["array"]


def flatten(array2d: list[list[float]]) -> list[float]:
    return [float(v) for row in array2d for v in row]


def shape(array2d: list[list[float]]) -> list[int]:
    rows = len(array2d)
    cols = len(array2d[0]) if rows else 0
    return [rows, cols]


def has_nan(values: Iterable[float]) -> bool:
    return any(math.isnan(v) for v in values)


def has_inf(values: Iterable[float]) -> bool:
    return any(math.isinf(v) for v in values)
