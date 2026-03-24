from leso_agent_demo.reports.policy_actions import propose_policy_actions


def test_policy_proposal_for_high_fail_line_end_cluster():
    report = {"family_stats": {"line_end_cluster": {"count": 10, "fail_rate": 0.2}}}
    actions = propose_policy_actions(report)
    assert actions and actions[0]["type"] == "shrink_parameter_range"
