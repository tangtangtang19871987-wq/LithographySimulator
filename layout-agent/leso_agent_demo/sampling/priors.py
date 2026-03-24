from __future__ import annotations


def build_default_priors() -> dict:
    return {"line_end_cluster": {"stagger_prob": [0.06, 0.16]}, "broken_bundle": {"break_probability": [0.03, 0.12]}}
