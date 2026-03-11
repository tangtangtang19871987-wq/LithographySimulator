"""Result analysis and external validation helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from urllib import request

from npz_utils import flatten, has_inf, has_nan, load_array, shape


def _safe_stats(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    finite = [v for v in values if math.isfinite(v)] or [0.0]
    mean = sum(finite) / len(finite)
    var = sum((v - mean) ** 2 for v in finite) / len(finite)
    return min(finite), max(finite), mean, math.sqrt(var)


def _calc_center_cd_px(binary: list[list[bool]]) -> int:
    if not binary:
        return 0
    row = binary[len(binary) // 2]
    return sum(1 for x in row if x)


def extract_litho_summary(npz_path: str | Path) -> dict[str, Any]:
    """Extract deterministic summary statistics from a lithography NPZ output."""
    path = Path(npz_path)
    _, arr = load_array(path)
    vals = flatten(arr)
    vmin, vmax, mean, std = _safe_stats(vals)
    threshold = 0.5 * vmax
    hotspot_t = 0.9 * vmax

    binary = [[float(v) >= threshold for v in row] for row in arr]
    hotspot = [[float(v) >= hotspot_t for v in row] for row in arr]
    total = len(vals) if vals else 1
    hot_count = sum(1 for row in hotspot for x in row if x)
    bin_count = sum(1 for row in binary for x in row if x)

    return {
        "ok": True,
        "source": str(path),
        "shape": shape(arr),
        "min": float(vmin),
        "max": float(vmax),
        "mean": float(mean),
        "std": float(std),
        "has_nan": has_nan(vals),
        "has_inf": has_inf(vals),
        "center_cd_px": _calc_center_cd_px(binary),
        "hotspot_ratio": float(hot_count / total),
        "binary_area_ratio": float(bin_count / total),
    }


def call_validation_api(
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Call external validation API, with deterministic mock mode.

    Uses requests when available, otherwise urllib fallback.
    """
    if endpoint.startswith("mock://validate"):
        summary = payload.get("summary", {})
        ok = (
            not summary.get("has_nan", False)
            and not summary.get("has_inf", False)
            and summary.get("binary_area_ratio", 0.0) > 0.01
        )
        return {
            "ok": True,
            "endpoint": endpoint,
            "mode": "mock",
            "verdict": "pass" if ok else "fail",
            "score": 0.95 if ok else 0.2,
            "checks": {
                "finite": not summary.get("has_nan", False) and not summary.get("has_inf", False),
                "non_empty_pattern": summary.get("binary_area_ratio", 0.0) > 0.01,
            },
        }

    body = json.dumps(payload).encode("utf-8")
    try:
        import requests  # type: ignore

        response = requests.post(endpoint, json=payload, timeout=timeout_s)
        response.raise_for_status()
        return {"ok": True, "endpoint": endpoint, "mode": "http", "response": response.json()}
    except Exception:
        req = request.Request(endpoint, data=body, headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=timeout_s) as resp:  # nosec - demo code
            response_body = json.loads(resp.read().decode("utf-8"))
        return {"ok": True, "endpoint": endpoint, "mode": "http", "response": response_body}


def build_validation_report(
    summary: dict[str, Any],
    api_validation: dict[str, Any],
    job_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine local + external checks into a final report."""
    local_ok = not summary.get("has_nan", True) and not summary.get("has_inf", True)
    if summary.get("max", 0.0) <= summary.get("min", 0.0):
        local_ok = False

    external_ok = api_validation.get("verdict") == "pass"
    if api_validation.get("mode") == "http":
        external_ok = bool(api_validation.get("response", {}).get("ok", False))

    job_ok = True if job_status is None else job_status.get("state") == "finished"

    reasons: list[str] = []
    if not job_ok:
        reasons.append("job_not_finished")
    if not local_ok:
        reasons.append("local_summary_failed")
    if not external_ok:
        reasons.append("external_validation_failed")

    return {
        "ok": True,
        "final_verdict": "pass" if (job_ok and local_ok and external_ok) else "fail",
        "checks": {"job_ok": job_ok, "local_ok": local_ok, "external_ok": external_ok},
        "reasons": reasons,
        "summary": summary,
        "external": api_validation,
        "job_status": job_status,
    }
