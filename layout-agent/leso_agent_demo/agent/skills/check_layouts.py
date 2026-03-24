from __future__ import annotations

from leso_agent_demo.rules.checks import run_basic_mrc
from leso_agent_demo.rules.event_extractor import extract_local_events


def compute_density_metrics(shapes, canvas):
    total_area = canvas.width_nm * canvas.height_nm
    metal = sum(s.area for s in shapes)
    return metal / total_area if total_area else 0.0


def compute_family_metrics(record):
    return {"trace_steps": len(record.get("generator_trace", [])), "family": record.get("family")}


def check_layout(sample_id, shapes, record, spec):
    raw_events, counts = run_basic_mrc(shapes, spec.constraints)
    events = extract_local_events(sample_id, "R1", raw_events, record["family"])
    density = compute_density_metrics(shapes, spec.canvas)
    return events, counts, density
