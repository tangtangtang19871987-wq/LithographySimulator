from __future__ import annotations

from leso_agent_demo.layout.schema import SamplingPlanDSL


def allocate_family_counts(spec, previous_batch_stats=None):
    _ = previous_batch_stats
    return {k: int(round(v * spec.count)) for k, v in spec.families.items()}


def choose_region_templates(spec):
    _ = spec
    return {
        "single_dense": 0.25,
        "left_dense_right_sparse": 0.20,
        "center_cluster_sparse_boundary": 0.18,
        "top_dense_bottom_transition": 0.17,
        "two_hotspot_islands": 0.10,
        "staggered_transition_band": 0.10,
    }


def build_parameter_priors(spec, risk_bins=None):
    _ = spec, risk_bins
    return {
        "line_end_cluster": {"dense_pitch_threshold_nm": 44, "stagger_prob": [0.06, 0.16], "cluster_size": [2, 6]},
        "broken_bundle": {"break_probability": [0.03, 0.12], "max_breaks_per_region": 4},
        "tip_gap_stress": {"target_fraction_near_rule": 0.60},
    }


def assemble_sampling_plan(spec, plan_id="plan_001", previous_batch_stats=None):
    alloc = allocate_family_counts(spec, previous_batch_stats)
    templates = choose_region_templates(spec)
    priors = build_parameter_priors(spec)
    return SamplingPlanDSL(
        plan_id=plan_id,
        spec_id=spec.spec_id,
        family_allocation=alloc,
        region_templates=templates,
        family_rules=priors,
        repair_policy={"local_fix_first": True, "max_local_repairs": 3, "reject_after_failed_repairs": True},
    )
