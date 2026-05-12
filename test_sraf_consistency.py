"""Synthetic regression tests for SRAF ray-casting consistency analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pytest

from sraf_ray_consistency import (
    Polygon,
    RayCastingConfig,
    SRAFConsistencyAnalyzer,
    UnitCell,
    Vec2,
)


def _contact(center: Vec2) -> Polygon:
    return Polygon.rectangle(center, 60.0, 60.0, layer="contact")


def _srafs(center: Vec2, width_scale: float = 1.0) -> List[Polygon]:
    # Four cardinal SRAFs around one contact.  Rectangles are intentionally
    # longer than wide so the PCA skeleton extractor has a stable major axis.
    w = 24.0 * width_scale
    return [
        Polygon.rectangle(center + Vec2(125.0, 0.0), 100.0, w, layer="sraf"),
        Polygon.rectangle(center + Vec2(-125.0, 0.0), 100.0, w, layer="sraf"),
        Polygon.rectangle(center + Vec2(0.0, 125.0), w, 100.0, layer="sraf"),
        Polygon.rectangle(center + Vec2(0.0, -125.0), w, 100.0, layer="sraf"),
    ]


def _cluster_srafs() -> List[Polygon]:
    srafs = _srafs(Vec2(-45.0, 0.0)) + _srafs(Vec2(45.0, 0.0))
    # A shared central assist bar makes inter-contact rays meaningful.
    srafs.append(Polygon.rectangle(Vec2(0.0, 0.0), 22.0, 90.0, layer="sraf"))
    return srafs


def _make_grid(
    rows: int,
    cols: int,
    anomalies: Dict[int, str] | None = None,
    cluster: bool = False,
) -> SRAFConsistencyAnalyzer:
    anomalies = anomalies or {}
    config = RayCastingConfig(
        num_angular_rays=36,
        num_edge_samples=4,
        max_ray_distance=600.0,
        rdf_max_radius=500.0,
        score_scale=0.25,
    )
    analyzer = SRAFConsistencyAnalyzer(config)
    pitch = 500.0
    cell_id = 0
    for row in range(rows):
        for col in range(cols):
            origin = Vec2(col * pitch, row * pitch)
            if cluster:
                contacts = [_contact(origin + Vec2(-45.0, 0.0)), _contact(origin + Vec2(45.0, 0.0))]
                local_srafs = _cluster_srafs()
            else:
                contacts = [_contact(origin)]
                local_srafs = _srafs(Vec2(0.0, 0.0))
            anomaly = anomalies.get(cell_id)
            if anomaly == "shift_right":
                local_srafs[0] = local_srafs[0].translated(Vec2(25.0, 0.0))
            elif anomaly == "missing_top":
                local_srafs = [poly for poly in local_srafs if poly.centroid().y < 100.0]
            elif anomaly == "wide":
                if cluster:
                    local_srafs = [Polygon.rectangle(poly.centroid(), 120.0, 36.0, layer="sraf") for poly in local_srafs]
                else:
                    local_srafs = _srafs(Vec2(0.0, 0.0), width_scale=1.5)
            elif anomaly == "asym_cluster":
                local_srafs[-1] = local_srafs[-1].translated(Vec2(28.0, 0.0))
            srafs = [poly.translated(origin) for poly in local_srafs]
            analyzer.add_unit_cell(
                UnitCell(
                    cell_id=cell_id,
                    cell_type="cluster" if cluster else "single",
                    origin=origin,
                    contacts=contacts,
                    srafs=srafs,
                    grid_position=(row, col),
                )
            )
            cell_id += 1
    return analyzer


def _assert_detected(analyzer: SRAFConsistencyAnalyzer, expected: Sequence[int]) -> None:
    scores = analyzer.analyze_all()
    detected = {cell_id for cell_id, _ in analyzer.get_anomalous_cells(threshold=0.3)}
    assert detected == set(expected)
    for cell_id, score in scores.items():
        if cell_id in expected:
            assert score.overall_score >= 0.3
            assert score.anomaly_details
        else:
            assert score.overall_score < 0.3


def test_all_consistent_baseline() -> None:
    _assert_detected(_make_grid(4, 4), [])


def test_single_sraf_shifted_25nm() -> None:
    _assert_detected(_make_grid(4, 4, {9: "shift_right"}), [9])


def test_missing_top_sraf() -> None:
    _assert_detected(_make_grid(4, 4, {6: "missing_top"}), [6])


def test_all_srafs_wider() -> None:
    _assert_detected(_make_grid(4, 4, {3: "wide"}), [3])


def test_two_contact_cluster_asymmetric_sraf() -> None:
    _assert_detected(_make_grid(3, 3, {4: "asym_cluster"}, cluster=True), [4])


def test_multiple_anomaly_types(tmp_path: Path) -> None:
    analyzer = _make_grid(5, 5, {8: "shift_right", 16: "missing_top", 24: "wide"})
    _assert_detected(analyzer, [8, 16, 24])
    results = {
        str(cell_id): {
            "overall_score": score.overall_score,
            "details": score.anomaly_details,
        }
        for cell_id, score in analyzer.scores.items()
    }
    output = tmp_path / "results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    assert output.exists()
