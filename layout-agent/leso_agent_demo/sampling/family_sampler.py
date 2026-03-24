from __future__ import annotations


class FamilySampler:
    def sample(self, spec, plan, rng):
        families = list(plan.family_allocation.keys())
        weights = [max(1, plan.family_allocation[f]) for f in families]
        return rng.choices(families, weights=weights, k=1)[0]
