from __future__ import annotations


def build_sample_diagnosis(sample_id, density, target_range, event_counts):
    total_events = sum(event_counts.values())
    status = "valid" if total_events == 0 and target_range[0] <= density <= target_range[1] else "repairable_fail"
    dominant_issue = max(event_counts, key=event_counts.get) if event_counts else "none"
    return {
        "sample_id": sample_id,
        "status": status,
        "valid_after_repair": False,
        "density": density,
        "target_density_range": list(target_range),
        "event_counts": event_counts,
        "dominant_issue": dominant_issue,
        "dominant_regions": ["R1"],
        "blame_candidates": [{"generator": "unknown", "condition": "n/a"}] if event_counts else [],
    }
