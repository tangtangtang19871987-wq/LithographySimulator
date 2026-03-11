"""Deterministic subprocess job launcher, monitor, and validator."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from npz_utils import flatten, has_inf, has_nan, load_array

VALID_STATES = {"pending", "running", "finished", "failed", "invalid", "timeout"}


def _scan_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"exists": False, "lines": 0, "has_error": False, "has_warning": False}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    return {
        "exists": True,
        "lines": len(text.splitlines()),
        "has_error": "error" in lowered,
        "has_warning": "warning" in lowered,
        "tail": text.splitlines()[-5:],
    }


def _validate_npz(npz_path: Path) -> dict[str, Any]:
    checks = {
        "exists": npz_path.exists(),
        "size_bytes": npz_path.stat().st_size if npz_path.exists() else 0,
        "readable": False,
        "has_numeric_array": False,
        "finite": False,
        "non_constant": False,
    }
    if not checks["exists"] or checks["size_bytes"] <= 0:
        return checks

    try:
        _, arr = load_array(npz_path)
        vals = flatten(arr)
        checks["readable"] = True
        checks["has_numeric_array"] = len(vals) > 0
        checks["finite"] = not has_nan(vals) and not has_inf(vals)
        checks["non_constant"] = bool(max(vals) > min(vals)) if vals else False
    except Exception:
        return checks
    return checks


def run_simulation_job(
    recipe_path: str | Path,
    layout_path: str | Path,
    out_dir: str | Path,
    fail_mode: str = "none",
    timeout_s: float = 10.0,
    poll_interval_s: float = 0.2,
) -> dict[str, Any]:
    """Run the external simulator stub with deterministic polling and validation."""
    recipe = Path(recipe_path)
    layout = Path(layout_path)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    result_npz = out_root / "result.npz"
    log_path = out_root / "sim.log"

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("simulate_stub.py")),
        "--recipe",
        str(recipe),
        "--layout",
        str(layout),
        "--out",
        str(result_npz),
        "--log",
        str(log_path),
        "--fail-mode",
        fail_mode,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    state = "pending"
    started = time.monotonic()
    history: list[dict[str, Any]] = [{"t": 0.0, "state": state}]

    while True:
        elapsed = time.monotonic() - started
        ret = proc.poll()

        if elapsed >= timeout_s and ret is None:
            proc.kill()
            state = "timeout"
            history.append({"t": round(elapsed, 3), "state": state})
            break

        if ret is None:
            next_state = "running"
            if state != next_state:
                state = next_state
                history.append({"t": round(elapsed, 3), "state": state})
            time.sleep(poll_interval_s)
            continue

        state = "finished" if ret == 0 else "failed"
        history.append({"t": round(elapsed, 3), "state": state, "return_code": ret})
        break

    stdout, stderr = proc.communicate(timeout=1)
    log_scan = _scan_log(log_path)
    npz_checks = _validate_npz(result_npz)

    if state == "finished":
        if log_scan.get("has_error"):
            state = "invalid"
        if not (npz_checks["readable"] and npz_checks["has_numeric_array"]):
            state = "invalid"
        if not (npz_checks["finite"] and npz_checks["non_constant"]):
            state = "invalid"

    final = {
        "ok": True,
        "state": state,
        "valid_states": sorted(VALID_STATES),
        "command": cmd,
        "return_code": proc.returncode,
        "timing": {
            "elapsed_s": round(time.monotonic() - started, 3),
            "timeout_s": timeout_s,
            "poll_interval_s": poll_interval_s,
        },
        "paths": {"out_dir": str(out_root), "result_npz": str(result_npz), "log": str(log_path)},
        "history": history,
        "validator": {"log": log_scan, "npz": npz_checks},
        "stdout": stdout,
        "stderr": stderr,
    }
    return final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fail-mode", default="none", choices=["none", "error", "invalid"])
    args = parser.parse_args()

    print(json.dumps(run_simulation_job(args.recipe, args.layout, args.out_dir, fail_mode=args.fail_mode), indent=2))
