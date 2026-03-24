from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RectShape:
    x1: int
    y1: int
    x2: int
    y2: int
    layer: str = "M1"
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return min(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def length(self) -> int:
        return max(self.x2 - self.x1, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


@dataclass
class PrimitiveRef:
    primitive_id: str
    primitive_type: str
    shape_ids: list[str]
    tags: dict[str, str] = field(default_factory=dict)
