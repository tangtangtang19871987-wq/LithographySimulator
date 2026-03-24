from __future__ import annotations

from collections import defaultdict


def run_basic_mrc(shapes, constraints):
    events = []
    for idx, s in enumerate(shapes):
        if s.width < constraints.min_width_nm:
            events.append({"type": "min_width_violation", "shape_idx": idx, "measured_nm": s.width, "required_nm": constraints.min_width_nm})
        if s.length < constraints.min_fragment_length_nm:
            events.append({"type": "fragment_length_violation", "shape_idx": idx, "measured_nm": s.length, "required_nm": constraints.min_fragment_length_nm})

    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            a, b = shapes[i], shapes[j]
            # simplified horizontal spacing check
            overlap_y = not (a.y2 <= b.y1 or b.y2 <= a.y1)
            if overlap_y:
                gap = max(b.x1 - a.x2, a.x1 - b.x2)
                if gap >= 0 and gap < constraints.min_space_nm:
                    events.append({"type": "min_space_violation", "shape_idx": i, "shape_jdx": j, "measured_nm": gap, "required_nm": constraints.min_space_nm})
    by_type = defaultdict(int)
    for ev in events:
        by_type[ev["type"]] += 1
    return events, dict(by_type)
