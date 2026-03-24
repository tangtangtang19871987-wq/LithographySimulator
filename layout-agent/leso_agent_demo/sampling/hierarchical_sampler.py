from __future__ import annotations


class MotifRegistry:
    def __init__(self, generators):
        self.generators = {g.name: g for g in generators}

    def pick_generators(self, family, region, plan, rng):
        if family in self.generators:
            return [self.generators[family]]
        return [self.generators["line_bundle"]]


class HierarchicalSampler:
    def __init__(self, family_sampler, region_sampler, motif_registry):
        self.family_sampler = family_sampler
        self.region_sampler = region_sampler
        self.motif_registry = motif_registry

    def sample_tile(self, spec, plan, rng):
        family = self.family_sampler.sample(spec, plan, rng)
        region_layout = self.region_sampler.sample(spec, plan, family, rng)
        all_shapes, trace = [], []
        for region in region_layout.regions:
            generators = self.motif_registry.pick_generators(family, region, plan, rng)
            for gen in generators:
                params = gen.sample_params(rng, region, plan.family_rules.get(gen.name, {}))
                shapes, primitive_refs = gen.build(rng, region, params)
                all_shapes.extend(shapes)
                trace.append(
                    {
                        "generator": gen.name,
                        "region_id": region.region_id,
                        "params": params,
                        "primitive_ids": [p.primitive_id for p in primitive_refs],
                    }
                )
        return all_shapes, trace, {"family": family, "region_template": region_layout.template_name}
