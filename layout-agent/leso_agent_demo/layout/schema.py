from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CanvasSpec:
    width_nm: int
    height_nm: int
    grid_nm: int = 1


@dataclass
class ConstraintSpec:
    min_width_nm: int
    min_space_nm: int
    min_tip_to_tip_nm: int
    min_tip_to_side_nm: int
    min_fragment_length_nm: int


@dataclass
class ProcessStyle:
    node: str = "7nm_class"
    lithography: str = "immersion_duv_multipatterned"
    layer_style: str = "lower_metal_leso"
    preferred_direction: str = "horizontal"
    manhattan_only: bool = True


@dataclass
class ObjectiveSpec:
    density_range: tuple[float, float]
    valid_rate_target: float
    near_rule_fraction: float
    hard_case_fraction: float


@dataclass
class ReportingSpec:
    save_layout_png: bool = True
    save_gds: bool = True
    save_json_reports: bool = True


@dataclass
class SpecDSL:
    spec_id: str
    count: int
    canvas: CanvasSpec
    objectives: ObjectiveSpec
    families: dict[str, float]
    distributions: dict[str, dict[str, Any]]
    constraints: ConstraintSpec
    process_style: ProcessStyle = field(default_factory=ProcessStyle)
    reporting: ReportingSpec = field(default_factory=ReportingSpec)

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "SpecDSL":
        canvas = CanvasSpec(**data["canvas"])
        constraints = ConstraintSpec(**data["constraints"])
        objectives_raw = data["objectives"]
        objectives = ObjectiveSpec(
            density_range=tuple(objectives_raw["density_range"]),
            valid_rate_target=objectives_raw["valid_rate_target"],
            near_rule_fraction=objectives_raw["near_rule_fraction"],
            hard_case_fraction=objectives_raw["hard_case_fraction"],
        )
        process_style = ProcessStyle(**data.get("process_style", {}))
        reporting = ReportingSpec(**data.get("reporting", {}))

        model = cls(
            spec_id=data["spec_id"],
            count=int(data["count"]),
            canvas=canvas,
            objectives=objectives,
            families=data["families"],
            distributions=data.get("distributions", {}),
            constraints=constraints,
            process_style=process_style,
            reporting=reporting,
        )
        total = sum(model.families.values())
        if not (0.98 <= total <= 1.02):
            raise ValueError(f"families must sum to ~1.0, got {total:.3f}")
        return model

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def model_dump_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent)


@dataclass
class SamplingPlanDSL:
    plan_id: str
    spec_id: str
    family_allocation: dict[str, int]
    region_templates: dict[str, float]
    family_rules: dict[str, dict[str, Any]] = field(default_factory=dict)
    repair_policy: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SampleRecord:
    sample_id: str
    spec_id: str
    plan_id: str
    seed: int
    family: str
    region_template: str
    generator_trace: list[dict[str, Any]]


@dataclass
class EventRecord:
    event_id: str
    sample_id: str
    type: str
    severity: float
    bbox: tuple[int, int, int, int]
    measured_nm: float
    required_nm: float
    region_id: str
    generator_refs: list[str]
    semantic_tags: list[str] = field(default_factory=list)
    repairable_by: list[str] = field(default_factory=list)


@dataclass
class SampleDiagnosis:
    sample_id: str
    status: str
    valid_after_repair: bool
    density: float
    target_density_range: tuple[float, float]
    event_counts: dict[str, int]
    dominant_issue: str
    dominant_regions: list[str]
    blame_candidates: list[dict[str, str]]


@dataclass
class BatchPolicyReport:
    batch_id: str
    num_samples: int
    valid_rate: float
    repair_rate: float
    reject_rate: float
    family_stats: dict[str, dict[str, float]]
    parameter_risk_bins: list[dict[str, Any]]
    coverage_gaps: list[dict[str, float | str]]
