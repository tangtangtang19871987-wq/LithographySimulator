"""External simulator stub used by subprocess-based workflow tests."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from npz_utils import save_array


def _generate_pattern(size: int, seed: int = 7) -> list[list[float]]:
    random.seed(seed)
    data: list[list[float]] = []
    for i in range(size):
        row: list[float] = []
        y = -1.0 + 2.0 * i / (size - 1)
        for j in range(size):
            x = -1.0 + 2.0 * j / (size - 1)
            r = math.sqrt(x * x + y * y)
            gauss = math.exp(-((x * x + y * y) / 0.15))
            rings = 0.3 * (math.sin(12.0 * r) + 1.0) / 2.0
            noise = 0.02 * (random.random() - 0.5)
            row.append(max(0.0, gauss + rings + noise))
        data.append(row)
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--fail-mode", choices=["none", "error", "invalid"], default="none")
    parser.add_argument("--sleep-ms", type=int, default=200)
    args = parser.parse_args()

    out_path = Path(args.out)
    log_path = Path(args.log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log:
        log.write("[stub] startup\n")
        log.write(f"[stub] recipe={args.recipe} layout={args.layout}\n")
        log.flush()
        time.sleep(args.sleep_ms / 1000)

        if args.fail_mode == "error":
            log.write("[stub] ERROR: synthetic simulator crash requested\n")
            log.flush()
            return 2

        if args.fail_mode == "invalid":
            invalid = [[float("nan"), float("inf")], [float("inf"), float("nan")]]
            save_array(out_path, "aerial", invalid)
            log.write("[stub] WARNING: generated invalid tensor values\n")
            log.write("[stub] finished_invalid\n")
            log.flush()
            return 0

        arr = _generate_pattern(size=128)
        flat = [v for r in arr for v in r]
        save_array(out_path, "aerial", arr)
        metrics = {"min": min(flat), "max": max(flat), "mean": sum(flat) / len(flat)}
        log.write(f"[stub] metrics={json.dumps(metrics)}\n")
        log.write("[stub] finished_ok\n")
        log.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
