from __future__ import annotations


def build_sample_record(sample_id, spec_id, plan_id, seed, meta, trace):
    return {
        "sample_id": sample_id,
        "spec_id": spec_id,
        "plan_id": plan_id,
        "seed": seed,
        "family": meta["family"],
        "region_template": meta["region_template"],
        "generator_trace": trace,
    }
