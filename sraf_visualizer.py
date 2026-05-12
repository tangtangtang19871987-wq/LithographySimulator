"""SVG visualization utilities for SRAF consistency analysis.

The visualizer deliberately writes dependency-free SVG files instead of requiring
matplotlib or Pillow.  The generated figures cover the same diagnostic views as
the analysis report: ray overlays, radial/angular profiles, score heatmaps, grid
overviews, and fingerprint distance matrices.
"""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from sraf_ray_consistency import SRAFConsistencyAnalyzer, UnitCell, Vec2

Color = Tuple[int, int, int]


def _rgb(color: Color) -> str:
    return f"rgb({color[0]},{color[1]},{color[2]})"


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _heat_color(value: float) -> Color:
    value = max(0.0, min(1.0, value))
    return (int(_lerp(35, 220, value)), int(_lerp(145, 45, value)), int(_lerp(70, 35, value)))


class SVGCanvas:
    """Small helper for writing self-contained SVG diagnostics."""

    def __init__(self, width: int = 800, height: int = 800, margin: int = 30):
        self.width = width
        self.height = height
        self.margin = margin
        self.items: List[str] = []
        self.bounds = (-1.0, -1.0, 1.0, 1.0)

    def set_world_bounds(self, bounds: Tuple[float, float, float, float]) -> None:
        min_x, min_y, max_x, max_y = bounds
        pad_x = max((max_x - min_x) * 0.05, 1.0)
        pad_y = max((max_y - min_y) * 0.05, 1.0)
        self.bounds = (min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y)

    def world(self, point: Vec2) -> Tuple[float, float]:
        min_x, min_y, max_x, max_y = self.bounds
        sx = (self.width - 2 * self.margin) / max(max_x - min_x, 1.0)
        sy = (self.height - 2 * self.margin) / max(max_y - min_y, 1.0)
        scale = min(sx, sy)
        x = self.margin + (point.x - min_x) * scale
        y = self.height - self.margin - (point.y - min_y) * scale
        return x, y

    def polygon(self, points: Sequence[Vec2], fill: str, stroke: str, stroke_width: float = 1.0, opacity: float = 1.0) -> None:
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in (self.world(p) for p in points))
        self.items.append(
            f'<polygon points="{coords}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}" />'
        )

    def line(self, start: Vec2, end: Vec2, stroke: str, stroke_width: float = 1.0, opacity: float = 1.0) -> None:
        x1, y1 = self.world(start)
        x2, y2 = self.world(end)
        self.items.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}" />'
        )

    def circle(self, center: Vec2, radius: float, fill: str, stroke: str = "none") -> None:
        x, y = self.world(center)
        self.items.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{fill}" stroke="{stroke}" />')

    def text(self, x: float, y: float, value: str, size: int = 14, anchor: str = "middle") -> None:
        self.items.append(f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}">{escape(value)}</text>')

    def rect_px(self, x: float, y: float, width: float, height: float, fill: str, stroke: str = "none") -> None:
        self.items.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{fill}" stroke="{stroke}" />')

    def save(self, path: str | Path) -> None:
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
            '<rect width="100%" height="100%" fill="white" />',
            *self.items,
            '</svg>',
        ]
        Path(path).write_text("\n".join(svg), encoding="utf-8")


class SRAFVisualizer:
    """Generate diagnostic SVG artifacts from an analyzed dataset."""

    RAY_COLORS = {
        "angular": "#d62728",
        "edge_normal": "#ff7f0e",
        "vertex_bisector": "#9467bd",
        "inter_contact": "#17becf",
    }

    def __init__(self, analyzer: SRAFConsistencyAnalyzer):
        self.analyzer = analyzer

    def save_ray_layout(self, cell_id: int, path: str | Path) -> None:
        cell = self._cell(cell_id)
        canvas = SVGCanvas(900, 900)
        canvas.set_world_bounds(self._cell_bounds(cell))
        for poly in cell.srafs:
            canvas.polygon(poly.vertices, fill="#9ecae1", stroke="#3182bd", opacity=0.65)
        for poly in cell.contacts:
            canvas.polygon(poly.vertices, fill="#bdbdbd", stroke="#252525", stroke_width=1.4)
        score = self.analyzer.scores.get(cell_id)
        ray_z = []
        if score is not None:
            ray_z = [max(score.per_ray_zscore[i : i + 5], default=0.0) for i in range(0, len(score.per_ray_zscore), 5)]
        for idx, ray in enumerate(self.analyzer.rays_by_cell.get(cell_id, [])):
            origin = ray.origin + cell.origin
            end = origin + ray.direction * min(self.analyzer.config.max_ray_distance, 250.0)
            z_value = ray_z[idx] if idx < len(ray_z) else 0.0
            if z_value > 3.0:
                canvas.line(origin, end, "#ff0000", stroke_width=2.2, opacity=0.85)
            else:
                canvas.line(origin, end, self.RAY_COLORS[ray.ray_type.value], stroke_width=0.8, opacity=0.30)
        for hit_list in self.analyzer.hits_by_cell.get(cell_id, []):
            for hit in hit_list[:1]:
                canvas.circle(hit.closest_point + cell.origin, radius=3.0, fill="#e41a1c")
        canvas.text(450, 24, f"Cell {cell_id} ray/SRAF intersections", size=18)
        canvas.save(path)

    def save_feature_profiles(self, cell_id: int, path: str | Path) -> None:
        fp = self.analyzer.fingerprints[cell_id]
        canvas = SVGCanvas(900, 520, margin=40)
        canvas.text(450, 24, f"Cell {cell_id} fingerprints", size=18)
        self._draw_bar_profile(canvas, fp.rdf, 60, 80, 340, 320, "Radial distribution")
        self._draw_polar_profile(canvas, fp.angular_profile, 660, 250, 150, "Angular profile")
        canvas.save(path)

    def save_score_heatmap(self, path: str | Path) -> None:
        positions = [cell.grid_position for cell in self.analyzer.cells if cell.grid_position is not None]
        if not positions:
            raise ValueError("score heatmap requires UnitCell.grid_position values")
        rows = max(r for r, _ in positions) + 1
        cols = max(c for _, c in positions) + 1
        canvas = SVGCanvas(80 * cols + 80, 80 * rows + 80)
        for cell in self.analyzer.cells:
            if cell.grid_position is None:
                continue
            row, col = cell.grid_position
            score = self.analyzer.scores[cell.cell_id].overall_score
            x = 50 + col * 80
            y = 50 + row * 80
            canvas.rect_px(x, y, 70, 70, _rgb(_heat_color(score)), stroke="#222")
            canvas.text(x + 35, y + 30, str(cell.cell_id), size=14)
            canvas.text(x + 35, y + 52, f"{score:.2f}", size=12)
            if not self.analyzer.scores[cell.cell_id].topology_match:
                canvas.items.append(f'<line x1="{x+12}" y1="{y+12}" x2="{x+58}" y2="{y+58}" stroke="black" stroke-width="3" />')
                canvas.items.append(f'<line x1="{x+58}" y1="{y+12}" x2="{x+12}" y2="{y+58}" stroke="black" stroke-width="3" />')
        canvas.text((80 * cols + 80) / 2, 24, "SRAF consistency score heatmap", size=18)
        canvas.save(path)

    def save_grid_overview(self, path: str | Path) -> None:
        canvas = SVGCanvas(1000, 1000)
        bounds = self._bounds_for_cells(self.analyzer.cells)
        canvas.set_world_bounds(bounds)
        for cell in self.analyzer.cells:
            score = self.analyzer.scores[cell.cell_id].overall_score
            stroke = _rgb(_heat_color(score))
            for poly in cell.srafs:
                canvas.polygon(poly.vertices, fill="#c6dbef", stroke="#6baed6", opacity=0.55)
            for poly in cell.contacts:
                canvas.polygon(poly.vertices, fill="#969696", stroke="#252525")
            min_x, min_y, max_x, max_y = self._cell_bounds(cell)
            corners = [Vec2(min_x, min_y), Vec2(max_x, min_y), Vec2(max_x, max_y), Vec2(min_x, max_y)]
            canvas.polygon(corners, fill="none", stroke=stroke, stroke_width=3.0, opacity=0.9)
        canvas.text(500, 24, "Grid overview with anomaly-colored cell borders", size=18)
        canvas.save(path)

    def save_fingerprint_matrix(self, path: str | Path) -> None:
        matrix = self.analyzer.fingerprint_distance_matrix()
        n = len(matrix)
        canvas = SVGCanvas(40 * n + 120, 40 * n + 120)
        max_value = max((value for row in matrix for value in row), default=1.0) or 1.0
        for r, row in enumerate(matrix):
            for c, value in enumerate(row):
                t = value / max_value
                color = (int(_lerp(245, 30, t)), int(_lerp(245, 80, t)), int(_lerp(245, 160, t)))
                canvas.rect_px(70 + c * 40, 60 + r * 40, 38, 38, _rgb(color))
        canvas.text((40 * n + 120) / 2, 24, "Fingerprint distance matrix", size=18)
        canvas.save(path)

    def _cell(self, cell_id: int) -> UnitCell:
        for cell in self.analyzer.cells:
            if cell.cell_id == cell_id:
                return cell
        raise KeyError(cell_id)

    @staticmethod
    def _cell_bounds(cell: UnitCell) -> Tuple[float, float, float, float]:
        polygons = cell.contacts + cell.srafs
        min_x = min(poly.bounds()[0] for poly in polygons)
        min_y = min(poly.bounds()[1] for poly in polygons)
        max_x = max(poly.bounds()[2] for poly in polygons)
        max_y = max(poly.bounds()[3] for poly in polygons)
        return min_x, min_y, max_x, max_y

    @staticmethod
    def _bounds_for_cells(cells: Iterable[UnitCell]) -> Tuple[float, float, float, float]:
        bounds = [SRAFVisualizer._cell_bounds(cell) for cell in cells]
        return min(b[0] for b in bounds), min(b[1] for b in bounds), max(b[2] for b in bounds), max(b[3] for b in bounds)

    @staticmethod
    def _draw_bar_profile(canvas: SVGCanvas, values: Sequence[float], x: float, y: float, width: float, height: float, title: str) -> None:
        max_value = max(values, default=1.0) or 1.0
        bar_w = width / max(len(values), 1)
        canvas.text(x + width / 2, y - 18, title, size=15)
        canvas.rect_px(x, y, width, height, "#f7f7f7", stroke="#333")
        for idx, value in enumerate(values):
            h = height * value / max_value
            canvas.rect_px(x + idx * bar_w, y + height - h, max(bar_w - 1, 1), h, "#3182bd")

    @staticmethod
    def _draw_polar_profile(canvas: SVGCanvas, values: Sequence[float], cx: float, cy: float, radius: float, title: str) -> None:
        canvas.text(cx, cy - radius - 20, title, size=15)
        for frac in (0.25, 0.5, 0.75, 1.0):
            canvas.items.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius*frac:.2f}" fill="none" stroke="#ddd" />')
        if not values:
            return
        max_value = max(values) or 1.0
        points = []
        for idx, value in enumerate(values):
            angle = 2 * math.pi * idx / len(values)
            r = radius * value / max_value
            points.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        canvas.items.append(f'<polygon points="{coords}" fill="#fb6a4a" fill-opacity="0.35" stroke="#cb181d" stroke-width="2" />')
