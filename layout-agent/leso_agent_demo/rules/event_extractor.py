from __future__ import annotations


def extract_local_events(sample_id: str, region_id: str, raw_events: list[dict], family: str):
    records = []
    for i, ev in enumerate(raw_events):
        records.append(
            {
                "event_id": f"E_{sample_id}_{i:04d}",
                "sample_id": sample_id,
                "type": ev["type"],
                "severity": 1.0 if ev["required_nm"] == 0 else max(0.0, min(1.0, (ev["required_nm"] - ev["measured_nm"]) / ev["required_nm"])),
                "bbox": [0, 0, 0, 0],
                "measured_nm": ev["measured_nm"],
                "required_nm": ev["required_nm"],
                "region_id": region_id,
                "generator_refs": [family],
                "semantic_tags": ["demo"],
                "repairable_by": ["retract_line_end", "resample_local_motif"],
            }
        )
    return records
