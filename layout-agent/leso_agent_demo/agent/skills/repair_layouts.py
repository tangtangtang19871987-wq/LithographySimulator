from __future__ import annotations

from leso_agent_demo.rules.local_recheck import recheck_local_region
from leso_agent_demo.rules.repair_ops import retract_line_end


def repair_layout(shapes, spec):
    ops = retract_line_end(shapes, amount_nm=4)
    raw_events, counts = recheck_local_region(shapes, spec.constraints)
    return ops, raw_events, counts
