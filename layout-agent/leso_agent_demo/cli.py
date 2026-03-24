from __future__ import annotations

import argparse
import json
import sys

from leso_agent_demo.agent.orchestrator import run_batch
from leso_agent_demo.agent.skills.compile_spec import validate_spec_schema
from leso_agent_demo.agent.skills.plan_sampling import assemble_sampling_plan
from leso_agent_demo.agent.skills.update_policy import write_next_sampling_plan
from leso_agent_demo.utils import dump_yaml, load_yaml


def cmd_compile_spec(args):
    spec = validate_spec_schema(load_yaml(args.spec))
    print(spec.model_dump_json(indent=2))


def cmd_run_batch(args):
    spec = validate_spec_schema(load_yaml(args.spec))
    if args.plan:
        plan = assemble_sampling_plan(spec, plan_id=load_yaml(args.plan).get("plan_id", "plan_001"))
    else:
        plan = assemble_sampling_plan(spec)
    batch = run_batch(spec, plan, args.out)
    print(json.dumps(batch, indent=2))


def cmd_summarize(args):
    with open(f"{args.batch_dir}/batch_policy_report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    print(json.dumps(report, indent=2))


def cmd_update_policy(args):
    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)
    spec = validate_spec_schema(load_yaml(args.spec))
    plan = assemble_sampling_plan(spec)
    next_plan, proposals, selected = write_next_sampling_plan(plan, report)
    _ = proposals, selected
    sys.stdout.write(dump_yaml(next_plan))


def main():
    parser = argparse.ArgumentParser(description="LESO demo CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("compile-spec")
    p1.add_argument("spec")
    p1.set_defaults(func=cmd_compile_spec)

    p2 = sub.add_parser("run-batch")
    p2.add_argument("spec")
    p2.add_argument("--plan")
    p2.add_argument("--out", default="out/batch_001")
    p2.set_defaults(func=cmd_run_batch)

    p3 = sub.add_parser("summarize")
    p3.add_argument("batch_dir")
    p3.set_defaults(func=cmd_summarize)

    p4 = sub.add_parser("update-policy")
    p4.add_argument("report")
    p4.add_argument("--spec", default="configs/default_spec.yaml")
    p4.set_defaults(func=cmd_update_policy)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
