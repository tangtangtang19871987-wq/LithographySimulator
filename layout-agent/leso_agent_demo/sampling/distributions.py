from __future__ import annotations


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
