"""命令行工具。"""

import argparse
import json
import sys
from pathlib import Path

from ..runner import EDARunner
from ..config import load_config


def get_runner(backend: str, config: dict) -> EDARunner:
    return EDARunner(backend=backend, **config.get("backends", {}).get(backend, {}))


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA Tool Runner")
    parser.add_argument("--config", default="~/.eda_runner.json")
    subparsers = parser.add_subparsers(dest="command")

    submit = subparsers.add_parser("submit")
    submit.add_argument("cmd")
    submit.add_argument("--id", required=True)
    submit.add_argument("--backend", default="local")

    status = subparsers.add_parser("status")
    status.add_argument("task_id", nargs="?")

    result = subparsers.add_parser("result")
    result.add_argument("task_id")

    kill = subparsers.add_parser("kill")
    kill.add_argument("task_id")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "submit":
        r = get_runner(args.backend, config)
        res = r.submit(args.cmd, args.id)
        if res.ok:
            print(f"✓ 任务 {args.id} 已提交")
        else:
            print(f"✗ 任务 {args.id} 提交失败: {res.error}")
            sys.exit(1)
    elif args.command == "status":
        r = get_runner(config.get("default_backend", "local"), config)
        if args.task_id:
            print(json.dumps(r.status(args.task_id).to_dict(), ensure_ascii=False, indent=2))
        else:
            for tid in r.list_tasks():
                print(tid, r.status(tid).status.value)
    elif args.command == "result":
        r = get_runner(config.get("default_backend", "local"), config)
        print(json.dumps(r.result(args.task_id).to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "kill":
        r = get_runner(config.get("default_backend", "local"), config)
        ok = r.kill(args.task_id)
        print("✓" if ok else "✗")
        if not ok:
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
