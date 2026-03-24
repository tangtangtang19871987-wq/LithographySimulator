from __future__ import annotations


def build_batch_policy_report(batch_id, samples, diagnoses):
    n = len(samples)
    valid = sum(1 for d in diagnoses if d["status"] == "valid")
    repairable = sum(1 for d in diagnoses if d["status"] == "repairable_fail")
    family_stats = {}
    for s, d in zip(samples, diagnoses):
        fam = s["family"]
        stat = family_stats.setdefault(fam, {"count": 0, "fail": 0})
        stat["count"] += 1
        if d["status"] != "valid":
            stat["fail"] += 1
    family_stats_out = {
        k: {"count": v["count"], "fail_rate": (v["fail"] / v["count"] if v["count"] else 0.0)}
        for k, v in family_stats.items()
    }
    return {
        "batch_id": batch_id,
        "num_samples": n,
        "valid_rate": valid / n if n else 0.0,
        "repair_rate": repairable / n if n else 0.0,
        "reject_rate": 0.0,
        "family_stats": family_stats_out,
        "parameter_risk_bins": [],
        "coverage_gaps": [],
    }
