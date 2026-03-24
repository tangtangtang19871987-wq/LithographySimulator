import random

from leso_agent_demo.layout.motifs import LineBundleGenerator
from leso_agent_demo.layout.regions import Region


def test_line_bundle_build_non_empty():
    gen = LineBundleGenerator()
    region = Region("R1", "dense_main", (0, 0, 256, 256))
    rng = random.Random(1)
    params = gen.sample_params(rng, region, {})
    shapes, _ = gen.build(rng, region, params)
    assert shapes
