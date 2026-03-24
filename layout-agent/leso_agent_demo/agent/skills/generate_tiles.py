from __future__ import annotations

import random

from leso_agent_demo.layout.motifs import BrokenBundleGenerator, LineBundleGenerator, LineEndClusterGenerator
from leso_agent_demo.reports.sample_record import build_sample_record
from leso_agent_demo.sampling.family_sampler import FamilySampler
from leso_agent_demo.sampling.hierarchical_sampler import HierarchicalSampler, MotifRegistry
from leso_agent_demo.sampling.region_sampler import RegionSampler


def generate_tiles(spec, plan, seed=1234):
    rng = random.Random(seed)
    sampler = HierarchicalSampler(
        family_sampler=FamilySampler(),
        region_sampler=RegionSampler(),
        motif_registry=MotifRegistry([LineBundleGenerator(), LineEndClusterGenerator(), BrokenBundleGenerator()]),
    )
    out = []
    for i in range(spec.count):
        shapes, trace, meta = sampler.sample_tile(spec, plan, rng)
        sample_id = f"S_{i:07d}"
        rec = build_sample_record(sample_id, spec.spec_id, plan.plan_id, rng.randint(0, 2**31 - 1), meta, trace)
        out.append((sample_id, shapes, rec))
    return out
