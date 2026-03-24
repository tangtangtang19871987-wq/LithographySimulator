from __future__ import annotations

from leso_agent_demo.layout.schema import ProcessStyle, SpecDSL


def parse_user_request(text: str) -> dict:
    return {"raw": text, "count": 2000 if "2000" in text else 1000}


def normalize_process_assumptions(semantic_request: dict) -> dict:
    _ = semantic_request
    from dataclasses import asdict
    return asdict(ProcessStyle())


def validate_spec_schema(draft_spec: dict) -> SpecDSL:
    return SpecDSL.model_validate(draft_spec)
