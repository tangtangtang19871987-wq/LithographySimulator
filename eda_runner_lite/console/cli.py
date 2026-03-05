"""轻量 CLI。"""

import argparse
import json
import sys

from ..runner import LiteRunner


def main() -> None:
    parser = argparse.ArgumentParser("eda-runner-lite")
    parser.add_argument("--log-dir", default="~/.eda_runner_lite")
    sub = parser.add_subparsers(dest="cmd")

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("task_id")
    p_submit.add_argument("command")

    p_status = sub.add_parser("status")
    p_status.add_argument("task_id")

    p_result = sub.add_parser("result")
    p_result.add_argument("task_id")

    args = parser.parse_args()
    runner = LiteRunner(log_dir=args.log_dir)

    if args.cmd == "submit":
        info = runner.submit(args.command, args.task_id)
        print(json.dumps(info.__dict__, ensure_ascii=False))
    elif args.cmd == "status":
        info = runner.status(args.task_id)
        payload = info.__dict__.copy()
        payload["state"] = info.state.value
        print(json.dumps(payload, ensure_ascii=False))
    elif args.cmd == "result":
        print(json.dumps(runner.result(args.task_id), ensure_ascii=False))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
