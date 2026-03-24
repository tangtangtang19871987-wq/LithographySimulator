from __future__ import annotations

import random
from dataclasses import dataclass

from .primitives import PrimitiveRef, RectShape


class MotifGenerator:
    name: str

    def sample_params(self, rng: random.Random, region, priors):
        raise NotImplementedError

    def build(self, rng: random.Random, region, params):
        raise NotImplementedError


@dataclass
class LineBundleGenerator(MotifGenerator):
    name: str = "line_bundle"

    def sample_params(self, rng, region, priors):
        pitch = rng.choice([36, 40, 44, 48])
        width = rng.choice([18, 20])
        count = rng.randint(3, 9)
        return {"pitch_nm": pitch, "width_nm": width, "count": count}

    def build(self, rng, region, params):
        x1, y1, x2, y2 = region.bbox
        shapes, refs = [], []
        base_y = y1 + params["pitch_nm"]
        for i in range(params["count"]):
            cy = base_y + i * params["pitch_nm"]
            if cy + params["width_nm"] > y2:
                break
            sid = f"S_{region.region_id}_{i}"
            shapes.append(RectShape(x1 + 10, cy, x2 - 10, cy + params["width_nm"], tags={"generator": self.name}))
            refs.append(PrimitiveRef(primitive_id=f"P_{region.region_id}_{i}", primitive_type="track_line", shape_ids=[sid]))
        return shapes, refs


@dataclass
class LineEndClusterGenerator(MotifGenerator):
    name: str = "line_end_cluster"

    def sample_params(self, rng, region, priors):
        return {"stagger_prob": rng.uniform(0.06, 0.16), "cluster_size": rng.randint(2, 6)}

    def build(self, rng, region, params):
        x1, y1, x2, y2 = region.bbox
        shapes, refs = [], []
        for i in range(params["cluster_size"]):
            w = 18
            seg_len = rng.randint(80, 260)
            sy = y1 + 30 + i * 40
            if sy + w > y2:
                break
            ex = x1 + 30 + seg_len
            shapes.append(RectShape(x1 + 30, sy, min(ex, x2 - 10), sy + w, tags={"generator": self.name, "line_end": "right"}))
            refs.append(PrimitiveRef(primitive_id=f"LE_{region.region_id}_{i}", primitive_type="line_end", shape_ids=[f"LE_S_{i}"]))
        return shapes, refs


@dataclass
class BrokenBundleGenerator(MotifGenerator):
    name: str = "broken_bundle"

    def sample_params(self, rng, region, priors):
        return {"break_probability": rng.uniform(0.03, 0.12), "lines": rng.randint(2, 5)}

    def build(self, rng, region, params):
        x1, y1, x2, y2 = region.bbox
        shapes, refs = [], []
        pitch, width = 44, 18
        for i in range(params["lines"]):
            yy = y1 + 20 + i * pitch
            if yy + width >= y2:
                break
            brk = rng.randint(x1 + 80, x2 - 80)
            gap = rng.randint(20, 60)
            shapes.append(RectShape(x1 + 10, yy, brk, yy + width, tags={"generator": self.name}))
            shapes.append(RectShape(brk + gap, yy, x2 - 10, yy + width, tags={"generator": self.name}))
            refs.append(PrimitiveRef(primitive_id=f"BB_{region.region_id}_{i}", primitive_type="cut_break", shape_ids=[f"B1_{i}", f"B2_{i}"]))
        return shapes, refs
