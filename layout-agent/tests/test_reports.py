from leso_agent_demo.reports.sample_diagnosis import build_sample_diagnosis


def test_sample_diagnosis_valid_when_no_events_and_density_in_range():
    out = build_sample_diagnosis("S1", 0.3, (0.2, 0.4), {})
    assert out["status"] == "valid"
