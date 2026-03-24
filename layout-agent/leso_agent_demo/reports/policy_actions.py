from __future__ import annotations


def propose_policy_actions(batch_report: dict) -> list[dict]:
    actions = []
    for fam, stats in batch_report.get("family_stats", {}).items():
        if stats.get("fail_rate", 0) > 0.15 and fam == "line_end_cluster":
            actions.append(
                {
                    "action_id": "A1",
                    "type": "shrink_parameter_range",
                    "target": "line_end_cluster.stagger_prob",
                    "from": [0.06, 0.16],
                    "to": [0.06, 0.11],
                    "reason": "high fail rate in dense pitch bins",
                }
            )
    return actions
