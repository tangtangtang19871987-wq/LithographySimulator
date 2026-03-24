from __future__ import annotations

from leso_agent_demo.reports.batch_policy_report import build_batch_policy_report
from leso_agent_demo.reports.sample_diagnosis import build_sample_diagnosis


def build_sample_diagnosis_record(sample_id, density, target_density_range, event_counts):
    return build_sample_diagnosis(sample_id, density, target_density_range, event_counts)


def build_batch_policy(samples, diagnoses, batch_id="B001"):
    return build_batch_policy_report(batch_id, samples, diagnoses)
