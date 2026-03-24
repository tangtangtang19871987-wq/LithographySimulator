from __future__ import annotations

from leso_agent_demo.layout.regions import make_region_layout


class RegionSampler:
    def sample(self, spec, plan, family, rng):
        _ = family
        names = list(plan.region_templates.keys())
        weights = [plan.region_templates[n] for n in names]
        name = rng.choices(names, weights=weights, k=1)[0]
        return make_region_layout(name, spec.canvas.width_nm, spec.canvas.height_nm)
