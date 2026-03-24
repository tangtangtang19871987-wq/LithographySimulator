from __future__ import annotations

from pathlib import Path

from leso_agent_demo.agent.skills.aggregate_reports import build_batch_policy, build_sample_diagnosis_record
from leso_agent_demo.agent.skills.check_layouts import check_layout
from leso_agent_demo.agent.skills.export_dataset import write_markdown_summary, write_metadata_json
from leso_agent_demo.agent.skills.generate_tiles import generate_tiles


def run_batch(spec, plan, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated = generate_tiles(spec, plan, seed=1234)
    sample_records = []
    diagnoses = []

    for sample_id, shapes, record in generated:
        events, event_counts, density = check_layout(sample_id, shapes, record, spec)
        diagnosis = build_sample_diagnosis_record(sample_id, density, spec.objectives.density_range, event_counts)

        write_metadata_json(out / f"{sample_id}.sample_record.json", record)
        write_metadata_json(out / f"{sample_id}.events.json", events)
        write_metadata_json(out / f"{sample_id}.diagnosis.json", diagnosis)

        sample_records.append(record)
        diagnoses.append(diagnosis)

    batch = build_batch_policy(sample_records, diagnoses, batch_id="B001")
    write_metadata_json(out / "batch_policy_report.json", batch)
    write_markdown_summary(
        out / "SUMMARY.md",
        [
            f"# Batch {batch['batch_id']}",
            f"- num_samples: {batch['num_samples']}",
            f"- valid_rate: {batch['valid_rate']:.3f}",
            f"- repair_rate: {batch['repair_rate']:.3f}",
        ],
    )
    return batch
