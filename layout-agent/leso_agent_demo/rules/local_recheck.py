from __future__ import annotations

from .checks import run_basic_mrc


def recheck_local_region(shapes, constraints, bbox=None):
    return run_basic_mrc(shapes, constraints)
