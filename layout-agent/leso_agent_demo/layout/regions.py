from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Region:
    region_id: str
    role: str
    bbox: tuple[int, int, int, int]


@dataclass
class RegionLayout:
    template_name: str
    regions: list[Region]


def make_region_layout(template: str, width: int, height: int) -> RegionLayout:
    if template == "single_dense":
        return RegionLayout(template, [Region("R1", "dense_main", (0, 0, width, height))])
    if template == "left_dense_right_sparse":
        mid = width // 2
        return RegionLayout(
            template,
            [
                Region("R1", "dense_main", (0, 0, mid, height)),
                Region("R2", "sparse_main", (mid, 0, width, height)),
            ],
        )
    band = (height // 3, 2 * height // 3)
    return RegionLayout(
        template,
        [
            Region("R1", "sparse_main", (0, 0, width, band[0])),
            Region("R2", "transition_band", (0, band[0], width, band[1])),
            Region("R3", "dense_main", (0, band[1], width, height)),
        ],
    )
