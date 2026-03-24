from leso_agent_demo.layout.primitives import RectShape
from leso_agent_demo.layout.schema import ConstraintSpec
from leso_agent_demo.rules.checks import run_basic_mrc


def test_basic_mrc_reports_width_violation():
    shapes = [RectShape(0, 0, 10, 100)]
    cons = ConstraintSpec(min_width_nm=18, min_space_nm=18, min_tip_to_tip_nm=31, min_tip_to_side_nm=25, min_fragment_length_nm=24)
    events, _ = run_basic_mrc(shapes, cons)
    assert any(e["type"] == "min_width_violation" for e in events)
