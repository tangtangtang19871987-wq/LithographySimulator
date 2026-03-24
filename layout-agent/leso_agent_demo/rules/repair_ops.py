from __future__ import annotations


def retract_line_end(shapes, amount_nm: int = 4):
    repaired = []
    for s in shapes:
        t = s.tags.get("line_end")
        if t == "right":
            s.x2 = max(s.x1, s.x2 - amount_nm)
            repaired.append("retract_line_end")
    return repaired
