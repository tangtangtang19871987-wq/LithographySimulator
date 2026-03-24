from leso_agent_demo.layout.schema import SpecDSL


def test_spec_model_accepts_default_spec():
    data = {
        "spec_id": "x",
        "count": 1,
        "canvas": {"width_nm": 100, "height_nm": 100, "grid_nm": 1},
        "objectives": {"density_range": [0.2, 0.4], "valid_rate_target": 0.9, "near_rule_fraction": 0.1, "hard_case_fraction": 0.1},
        "families": {"line_bundle": 1.0},
        "distributions": {},
        "constraints": {"min_width_nm": 18, "min_space_nm": 18, "min_tip_to_tip_nm": 31, "min_tip_to_side_nm": 25, "min_fragment_length_nm": 24},
    }
    spec = SpecDSL.model_validate(data)
    assert spec.spec_id == "x"
