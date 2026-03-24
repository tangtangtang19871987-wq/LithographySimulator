from __future__ import annotations

from copy import deepcopy

from leso_agent_demo.reports.policy_actions import propose_policy_actions


def select_policy_actions(proposals):
    return proposals[:2]


def write_next_sampling_plan(plan, batch_report):
    proposals = propose_policy_actions(batch_report)
    selected = select_policy_actions(proposals)
    next_plan = deepcopy(plan.model_dump())
    next_plan["plan_id"] = f"{plan.plan_id}_next"
    for action in selected:
        if action["type"] == "shrink_parameter_range" and action["target"] == "line_end_cluster.stagger_prob":
            next_plan["family_rules"].setdefault("line_end_cluster", {})["stagger_prob"] = action["to"]
    return next_plan, proposals, selected
