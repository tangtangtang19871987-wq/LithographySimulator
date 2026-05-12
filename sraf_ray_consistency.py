"""SRAF ray-casting consistency analysis.

This module implements a compact, dependency-light pipeline for comparing
sub-resolution assist feature (SRAF) placement around symmetry-equivalent
contact/via unit cells.  The public API is intentionally small: construct
``UnitCell`` objects with contact and SRAF polygons, add them to an
``SRAFConsistencyAnalyzer``, and call ``analyze_all``.

The implementation follows the six-stage workflow described in the repository
issue: skeleton extraction, canonical ray generation, ray/SRAF intersection,
feature extraction, fingerprinting, and robust consistency scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

EPS = 1.0e-9
NO_HIT_DISTANCE = -1.0


@dataclass(frozen=True)
class Vec2:
    """Simple immutable 2-D vector used by the geometry layer."""

    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scale: float) -> "Vec2":
        return Vec2(self.x * scale, self.y * scale)

    __rmul__ = __mul__

    def __truediv__(self, scale: float) -> "Vec2":
        return Vec2(self.x / scale, self.y / scale)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2") -> float:
        return self.x * other.y - self.y * other.x

    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        length = self.norm()
        if length < EPS:
            return Vec2(0.0, 0.0)
        return self / length

    def rotate(self, angle_rad: float) -> "Vec2":
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return Vec2(c * self.x - s * self.y, s * self.x + c * self.y)

    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @staticmethod
    def from_pair(pair: Sequence[float]) -> "Vec2":
        return Vec2(float(pair[0]), float(pair[1]))


@dataclass
class Polygon:
    """Polygon represented by ordered vertices."""

    vertices: List[Vec2]
    layer: str = ""

    @staticmethod
    def rectangle(
        center: Vec2,
        width: float,
        height: float,
        layer: str = "",
        angle_deg: float = 0.0,
    ) -> "Polygon":
        half_w = width / 2.0
        half_h = height / 2.0
        pts = [
            Vec2(-half_w, -half_h),
            Vec2(half_w, -half_h),
            Vec2(half_w, half_h),
            Vec2(-half_w, half_h),
        ]
        angle = math.radians(angle_deg)
        return Polygon([center + p.rotate(angle) for p in pts], layer=layer)

    def translated(self, delta: Vec2) -> "Polygon":
        return Polygon([v + delta for v in self.vertices], layer=self.layer)

    def transformed(self, op: "SymmetryOp") -> "Polygon":
        return Polygon([op.apply(v) for v in self.vertices], layer=self.layer)

    def centroid(self) -> Vec2:
        area_twice = 0.0
        cx = 0.0
        cy = 0.0
        verts = self.vertices
        for a, b in self.edges():
            cross = a.cross(b)
            area_twice += cross
            cx += (a.x + b.x) * cross
            cy += (a.y + b.y) * cross
        if abs(area_twice) < EPS:
            return Vec2(sum(v.x for v in verts) / len(verts), sum(v.y for v in verts) / len(verts))
        return Vec2(cx / (3.0 * area_twice), cy / (3.0 * area_twice))

    def area(self) -> float:
        return abs(sum(a.cross(b) for a, b in self.edges())) / 2.0

    def edges(self) -> List[Tuple[Vec2, Vec2]]:
        return list(zip(self.vertices, self.vertices[1:] + self.vertices[:1]))

    def bounds(self) -> Tuple[float, float, float, float]:
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        return min(xs), min(ys), max(xs), max(ys)


    def signed_area(self) -> float:
        return sum(a.cross(b) for a, b in self.edges()) / 2.0

    def is_ccw(self) -> bool:
        return self.signed_area() >= 0.0

    def normalized_orientation(self) -> "Polygon":
        if self.is_ccw():
            return self
        return Polygon(list(reversed(self.vertices)), layer=self.layer)

    def contains_point(self, point: Vec2, eps: float = EPS) -> bool:
        inside = False
        for a, b in self.edges():
            ab = b - a
            ap = point - a
            if abs(ab.cross(ap)) <= eps and min(a.x, b.x) - eps <= point.x <= max(a.x, b.x) + eps and min(a.y, b.y) - eps <= point.y <= max(a.y, b.y) + eps:
                return True
            if (a.y > point.y) != (b.y > point.y):
                x_cross = (b.x - a.x) * (point.y - a.y) / (b.y - a.y + EPS) + a.x
                if point.x < x_cross:
                    inside = not inside
        return inside

    def has_self_intersection(self, eps: float = EPS) -> bool:
        edges = self.edges()
        for i, (a, b) in enumerate(edges):
            for j, (c, d) in enumerate(edges):
                if abs(i - j) <= 1 or {i, j} == {0, len(edges) - 1}:
                    continue
                if _segments_intersect(a, b, c, d, eps):
                    return True
        return False


def _orientation(a: Vec2, b: Vec2, c: Vec2) -> float:
    return (b - a).cross(c - a)


def _segments_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2, eps: float = EPS) -> bool:
    def on_segment(p: Vec2, q: Vec2, r: Vec2) -> bool:
        return (
            min(p.x, r.x) - eps <= q.x <= max(p.x, r.x) + eps
            and min(p.y, r.y) - eps <= q.y <= max(p.y, r.y) + eps
            and abs(_orientation(p, q, r)) <= eps
        )

    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 * o2 < -eps and o3 * o4 < -eps:
        return True
    return on_segment(a, c, b) or on_segment(a, d, b) or on_segment(c, a, d) or on_segment(c, b, d)


def _rdp(points: List[Vec2], tolerance: float) -> List[Vec2]:
    if len(points) <= 2 or tolerance <= 0:
        return points
    start, end = points[0], points[-1]
    seg = end - start
    seg_len = seg.norm()
    max_dist = -1.0
    max_index = 0
    for i, point in enumerate(points[1:-1], start=1):
        if seg_len < EPS:
            dist = (point - start).norm()
        else:
            dist = abs(seg.cross(point - start)) / seg_len
        if dist > max_dist:
            max_dist = dist
            max_index = i
    if max_dist > tolerance:
        left = _rdp(points[: max_index + 1], tolerance)
        right = _rdp(points[max_index:], tolerance)
        return left[:-1] + right
    return [start, end]


class RayType(str, Enum):
    ANGULAR = "angular"
    EDGE_NORMAL = "edge_normal"
    VERTEX_BISECTOR = "vertex_bisector"
    INTER_CONTACT = "inter_contact"


@dataclass(frozen=True)
class SymmetryOp:
    """Rigid/reflection operation for canonical unit-cell alignment."""

    rotation_deg: float = 0.0
    mirror_x: bool = False
    translation: Vec2 = Vec2(0.0, 0.0)

    def apply(self, point: Vec2) -> Vec2:
        p = Vec2(-point.x, point.y) if self.mirror_x else point
        return p.rotate(math.radians(self.rotation_deg)) + self.translation

    def apply_direction(self, direction: Vec2) -> Vec2:
        d = Vec2(-direction.x, direction.y) if self.mirror_x else direction
        return d.rotate(math.radians(self.rotation_deg)).normalized()

    def inverse(self) -> "SymmetryOp":
        # Implemented as a callable matrix inverse in canonicalize helpers below.
        return SymmetryOp(-self.rotation_deg, self.mirror_x, Vec2(0.0, 0.0))

    def to_canonical_point(self, point: Vec2) -> Vec2:
        p = point - self.translation
        p = p.rotate(math.radians(-self.rotation_deg))
        if self.mirror_x:
            p = Vec2(-p.x, p.y)
        return p

    def to_canonical_direction(self, direction: Vec2) -> Vec2:
        d = direction.rotate(math.radians(-self.rotation_deg))
        if self.mirror_x:
            d = Vec2(-d.x, d.y)
        return d.normalized()

    def from_canonical_point(self, point: Vec2) -> Vec2:
        return self.apply(point)

    def from_canonical_direction(self, direction: Vec2) -> Vec2:
        return self.apply_direction(direction)


@dataclass
class Ray:
    origin: Vec2
    direction: Vec2
    ray_type: RayType
    contact_index: int
    ordinal: int
    angle: float
    valid: bool = True
    invalid_reason: str = ""


@dataclass
class SRAFSkeleton:
    start: Vec2
    end: Vec2
    half_width: float
    polygon_index: int

    @property
    def angle(self) -> float:
        return (self.end - self.start).angle()

    @property
    def length(self) -> float:
        return (self.end - self.start).norm()


@dataclass
class RayHit:
    skeleton_index: int
    distance: float
    entry_distance: float
    exit_distance: float
    projected_width: float
    skeleton_angle: float
    closest_point: Vec2


@dataclass
class RayFeature:
    hit_count: int
    nearest_distance: float
    projected_width: float
    skeleton_angle: float
    total_coverage: float
    valid: bool = True
    miss: bool = False
    truncated: bool = False

    def as_tuple(self) -> Tuple[float, float, float, float, float]:
        return (
            float(self.hit_count),
            self.nearest_distance,
            self.projected_width,
            self.skeleton_angle,
            self.total_coverage,
        )


@dataclass
class CellFingerprint:
    coarse_hash: str
    rdf: List[float]
    angular_profile: List[float]
    hu_moments: List[float]
    topology_signature: str
    feature_vector: List[float]


@dataclass
class ConsistencyScore:
    cell_id: int
    cell_type: str
    overall_score: float
    radial_deviation: float
    angular_deviation: float
    moment_deviation: float
    topology_match: bool
    anomaly_details: List[str] = field(default_factory=list)
    per_ray_zscore: List[float] = field(default_factory=list)
    insufficient_data: bool = False


@dataclass
class UnitCell:
    cell_id: int
    cell_type: str
    origin: Vec2
    contacts: List[Polygon]
    srafs: List[Polygon]
    symmetry: SymmetryOp = field(default_factory=SymmetryOp)
    grid_position: Optional[Tuple[int, int]] = None

    def local_contacts(self) -> List[Polygon]:
        return [p.translated(Vec2(-self.origin.x, -self.origin.y)) for p in self.contacts]

    def local_srafs(self) -> List[Polygon]:
        return [p.translated(Vec2(-self.origin.x, -self.origin.y)) for p in self.srafs]


@dataclass
class RayCastingConfig:
    num_angular_rays: int = 36
    num_edge_samples: int = 5
    max_ray_distance: float = 2000.0
    distance_quantization: float = 5.0
    width_quantization: float = 2.0
    angular_profile_bins: int = 72
    rdf_bins: int = 32
    rdf_max_radius: float = 800.0
    topology_sectors: int = 16
    distance_noise_floor: float = 2.0
    width_noise_floor: float = 1.0
    score_scale: float = 0.30
    include_halo: bool = False
    halo_radius: float = 0.0
    polygon_area_epsilon: float = 1.0e-6
    short_edge_tolerance: float = 1.0
    ray_origin_nudge: float = 1.0e-3
    parallel_tolerance: float = 1.0e-9
    missing_rate_exclusion: float = 0.5
    min_reference_cells: int = 10
    exclude_top_fraction: float = 0.05
    use_local_baseline: bool = False
    local_baseline_radius: int = 2
    use_spatial_index: bool = True
    spatial_bin_size: float = 250.0


class SRAFConsistencyAnalyzer:
    """Analyze SRAF consistency across equivalent unit cells."""

    def __init__(self, config: Optional[RayCastingConfig] = None):
        self.config = config or RayCastingConfig()
        self.cells: List[UnitCell] = []
        self.rays_by_cell: Dict[int, List[Ray]] = {}
        self.hits_by_cell: Dict[int, List[List[RayHit]]] = {}
        self.features_by_cell: Dict[int, List[RayFeature]] = {}
        self.fingerprints: Dict[int, CellFingerprint] = {}
        self.scores: Dict[int, ConsistencyScore] = {}
        self.warnings: Dict[int, List[str]] = {}
        self.invalid_rays_by_cell: Dict[int, List[Ray]] = {}

    def add_unit_cell(self, cell: UnitCell) -> None:
        self.cells.append(cell)

    def analyze_all(self) -> Dict[int, ConsistencyScore]:
        for cell in self.cells:
            self._analyze_cell(cell)
        self._score_cells()
        return self.scores

    def get_anomalous_cells(self, threshold: float = 0.3) -> List[Tuple[int, ConsistencyScore]]:
        return sorted(
            [(cid, score) for cid, score in self.scores.items() if score.overall_score >= threshold],
            key=lambda item: (-item[1].overall_score, item[0]),
        )

    def get_hash_buckets(self) -> Dict[str, List[int]]:
        buckets: Dict[str, List[int]] = {}
        for cell_id, fp in self.fingerprints.items():
            buckets.setdefault(fp.coarse_hash, []).append(cell_id)
        return buckets

    def fingerprint_distance_matrix(self) -> List[List[float]]:
        ids = [cell.cell_id for cell in self.cells]
        matrix = [[0.0 for _ in ids] for _ in ids]
        for i, a_id in enumerate(ids):
            for j, b_id in enumerate(ids):
                if i < j:
                    d = self._fingerprint_distance(self.fingerprints[a_id], self.fingerprints[b_id])
                    matrix[i][j] = matrix[j][i] = d
        return matrix

    def export_ray_residuals_csv(self, path: str) -> None:
        """Write per-ray features and robust residuals for reproducible debugging."""
        lines = [
            "cell_id,ray_ordinal,ray_type,contact_index,hit_count,nearest_distance,projected_width,total_coverage,miss,truncated,max_feature_zscore"
        ]
        for cell in sorted(self.cells, key=lambda c: c.cell_id):
            features = self.features_by_cell.get(cell.cell_id, [])
            rays = self.rays_by_cell.get(cell.cell_id, [])
            z = self.scores.get(cell.cell_id, ConsistencyScore(cell.cell_id, cell.cell_type, 0, 0, 0, 0, True)).per_ray_zscore
            for idx, (ray, feature) in enumerate(zip(rays, features)):
                z_slice = z[idx * 5 : idx * 5 + 5]
                lines.append(
                    f"{cell.cell_id},{ray.ordinal},{ray.ray_type.value},{ray.contact_index},{feature.hit_count},"
                    f"{feature.nearest_distance:.9g},{feature.projected_width:.9g},{feature.total_coverage:.9g},"
                    f"{int(feature.miss)},{int(feature.truncated)},{(max(z_slice) if z_slice else 0.0):.9g}"
                )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def _analyze_cell(self, cell: UnitCell) -> None:
        warnings: List[str] = []
        contacts = [
            self._to_canonical_polygon(p, cell.symmetry)
            for p in self._preprocess_polygons(cell.local_contacts(), "contact", warnings)
        ]
        raw_srafs = self._preprocess_polygons(self._srafs_with_optional_halo(cell), "sraf", warnings)
        srafs = [self._to_canonical_polygon(p, cell.symmetry) for p in raw_srafs]
        if contacts:
            filtered_srafs = []
            for poly in srafs:
                center = poly.centroid()
                if any(contact.contains_point(center, self.config.short_edge_tolerance) for contact in contacts):
                    warnings.append("ignored SRAF with centroid overlapping a contact polygon")
                    continue
                filtered_srafs.append(poly)
            srafs = filtered_srafs
        rays = self._generate_canonical_rays(contacts)
        valid_rays = [ray for ray in rays if ray.valid]
        invalid_rays = [ray for ray in rays if not ray.valid]
        skeletons = [self._extract_skeleton(poly, i) for i, poly in enumerate(srafs)]
        index = self._build_skeleton_index(skeletons)
        hits = [self._intersect_ray_skeletons(ray, skeletons, index) for ray in valid_rays]
        features = [self._features_for_hits(hit_list, ray.valid) for hit_list, ray in zip(hits, valid_rays)]
        fingerprint = self._build_fingerprint(features, valid_rays, contacts, srafs)
        self.rays_by_cell[cell.cell_id] = valid_rays
        self.invalid_rays_by_cell[cell.cell_id] = invalid_rays
        self.hits_by_cell[cell.cell_id] = hits
        self.features_by_cell[cell.cell_id] = features
        self.fingerprints[cell.cell_id] = fingerprint
        self.warnings[cell.cell_id] = warnings + [ray.invalid_reason for ray in invalid_rays if ray.invalid_reason]

    def _srafs_with_optional_halo(self, cell: UnitCell) -> List[Polygon]:
        own = cell.local_srafs()
        if not self.config.include_halo or self.config.halo_radius <= 0:
            return own
        result = list(own)
        for other in self.cells:
            if other.cell_id == cell.cell_id:
                continue
            delta = other.origin - cell.origin
            if delta.norm() <= self.config.halo_radius:
                result.extend([p.translated(delta) for p in other.local_srafs()])
        return result

    def _to_canonical_polygon(self, polygon: Polygon, symmetry: SymmetryOp) -> Polygon:
        """Map a cell-local polygon into the canonical orientation.

        The analyzer stores rays in this canonical frame so ray ordinal ``k`` has
        the same geometric meaning for all symmetry-equivalent cells.
        """
        return Polygon([symmetry.to_canonical_point(v) for v in polygon.vertices], layer=polygon.layer)

    def _preprocess_polygons(self, polygons: List[Polygon], role: str, warnings: List[str]) -> List[Polygon]:
        processed: List[Polygon] = []
        for idx, polygon in enumerate(polygons):
            clean = self._clean_polygon(polygon, role, idx, warnings)
            if clean is not None:
                processed.append(clean)
        return processed

    def _clean_polygon(self, polygon: Polygon, role: str, index: int, warnings: List[str]) -> Optional[Polygon]:
        if len(polygon.vertices) < 3:
            warnings.append(f"skipped {role} polygon {index}: fewer than three vertices")
            return None
        vertices: List[Vec2] = []
        for vertex in polygon.vertices:
            if not math.isfinite(vertex.x) or not math.isfinite(vertex.y):
                warnings.append(f"skipped {role} polygon {index}: non-finite coordinate")
                return None
            if not vertices or (vertex - vertices[-1]).norm() > self.config.short_edge_tolerance:
                vertices.append(vertex)
        if len(vertices) > 1 and (vertices[0] - vertices[-1]).norm() <= self.config.short_edge_tolerance:
            vertices.pop()
        if len(vertices) < 3:
            warnings.append(f"skipped {role} polygon {index}: collapsed after short-edge merge")
            return None
        closed = vertices + [vertices[0]]
        simplified = _rdp(closed, self.config.short_edge_tolerance)[:-1]
        if len(simplified) >= 3:
            vertices = simplified
        cleaned = Polygon(vertices, layer=polygon.layer).normalized_orientation()
        if cleaned.area() <= self.config.polygon_area_epsilon:
            warnings.append(f"skipped {role} polygon {index}: area below tolerance")
            return None
        if cleaned.has_self_intersection(self.config.parallel_tolerance):
            warnings.append(f"skipped {role} polygon {index}: self-intersection detected")
            return None
        return cleaned

    def _generate_canonical_rays(self, contacts: List[Polygon]) -> List[Ray]:
        rays: List[Ray] = []
        ordered_contacts = sorted(enumerate(contacts), key=lambda item: (item[1].centroid().x, item[1].centroid().y))
        ordinal = 0
        for contact_rank, (original_index, contact) in enumerate(ordered_contacts):
            center = contact.centroid()
            for i in range(self.config.num_angular_rays):
                angle = 2.0 * math.pi * i / self.config.num_angular_rays
                ray = self._make_ray(
                    center,
                    Vec2(math.cos(angle), math.sin(angle)),
                    RayType.ANGULAR,
                    contact_rank,
                    ordinal,
                )
                rays.append(ray)
                ordinal += 1
            for edge_index, (a, b) in enumerate(contact.edges()):
                edge = b - a
                if edge.norm() <= self.config.short_edge_tolerance:
                    rays.append(
                        Ray(a, Vec2(0.0, 0.0), RayType.EDGE_NORMAL, contact_rank, ordinal, 0.0, False, "skipped zero-length contact edge")
                    )
                    ordinal += 1
                    continue
                normal = Vec2(-edge.y, edge.x).normalized()
                midpoint = (a + b) * 0.5
                if normal.dot(midpoint - center) < 0:
                    normal = normal * -1.0
                for sample in range(self.config.num_edge_samples):
                    t = (sample + 0.5) / self.config.num_edge_samples
                    origin = a * (1.0 - t) + b * t
                    rays.append(self._make_ray(origin, normal, RayType.EDGE_NORMAL, contact_rank, ordinal))
                    ordinal += 1
            verts = contact.vertices
            n = len(verts)
            orientation = math.copysign(1.0, sum(a.cross(b) for a, b in contact.edges()) or 1.0)
            for i, vertex in enumerate(verts):
                prev_v = verts[(i - 1) % n]
                next_v = verts[(i + 1) % n]
                e_prev = (vertex - prev_v).normalized()
                e_next = (next_v - vertex).normalized()
                inward = (Vec2(-e_prev.y, e_prev.x) * orientation + Vec2(-e_next.y, e_next.x) * orientation).normalized()
                direction = (inward * -1.0).normalized()
                if direction.norm() < EPS:
                    direction = (vertex - center).normalized()
                rays.append(self._make_ray(vertex, direction, RayType.VERTEX_BISECTOR, contact_rank, ordinal))
                ordinal += 1
        if len(ordered_contacts) > 1:
            centroids = [poly.centroid() for _, poly in ordered_contacts]
            for i, origin in enumerate(centroids):
                for j, target in enumerate(centroids):
                    if i == j:
                        continue
                    direction = target - origin
                    if direction.norm() <= self.config.short_edge_tolerance:
                        rays.append(
                            Ray(origin, Vec2(0.0, 0.0), RayType.INTER_CONTACT, i, ordinal, 0.0, False, "skipped coincident inter-contact ray")
                        )
                    else:
                        rays.append(self._make_ray(origin, direction, RayType.INTER_CONTACT, i, ordinal))
                    ordinal += 1
        return rays

    def _make_ray(self, origin: Vec2, direction: Vec2, ray_type: RayType, contact_index: int, ordinal: int) -> Ray:
        unit = direction.normalized()
        if unit.norm() < EPS:
            return Ray(origin, unit, ray_type, contact_index, ordinal, 0.0, False, "skipped zero-length ray direction")
        nudged_origin = origin + unit * self.config.ray_origin_nudge
        return Ray(nudged_origin, unit, ray_type, contact_index, ordinal, unit.angle())

    def _extract_skeleton(self, polygon: Polygon, index: int) -> SRAFSkeleton:
        pts = [(v.x, v.y) for v in polygon.vertices]
        if len(pts) < 2:
            p = polygon.centroid()
            return SRAFSkeleton(p, p, 0.0, index)
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(y for _, y in pts) / len(pts)
        sxx = sum((x - cx) ** 2 for x, _ in pts) / len(pts)
        syy = sum((y - cy) ** 2 for _, y in pts) / len(pts)
        sxy = sum((x - cx) * (y - cy) for x, y in pts) / len(pts)
        if abs(sxy) < EPS and abs(sxx - syy) < EPS:
            major = (1.0, 0.0)
        else:
            theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
            major = (math.cos(theta), math.sin(theta))
        minor = (-major[1], major[0])
        major_proj = [(x - cx) * major[0] + (y - cy) * major[1] for x, y in pts]
        minor_proj = [(x - cx) * minor[0] + (y - cy) * minor[1] for x, y in pts]
        start = Vec2(cx + major[0] * min(major_proj), cy + major[1] * min(major_proj))
        end = Vec2(cx + major[0] * max(major_proj), cy + major[1] * max(major_proj))
        half_width = max((max(minor_proj) - min(minor_proj)) / 2.0, math.sqrt(max(polygon.area(), 0.0)) * 0.05)
        return SRAFSkeleton(start, end, half_width, index)

    def _build_skeleton_index(self, skeletons: List[SRAFSkeleton]) -> Optional[Dict[Tuple[int, int], List[int]]]:
        if not self.config.use_spatial_index or not skeletons:
            return None
        bin_size = max(self.config.spatial_bin_size, EPS)
        index: Dict[Tuple[int, int], List[int]] = {}
        for i, skeleton in enumerate(skeletons):
            min_x = min(skeleton.start.x, skeleton.end.x) - skeleton.half_width
            max_x = max(skeleton.start.x, skeleton.end.x) + skeleton.half_width
            min_y = min(skeleton.start.y, skeleton.end.y) - skeleton.half_width
            max_y = max(skeleton.start.y, skeleton.end.y) + skeleton.half_width
            for bx in range(math.floor(min_x / bin_size), math.floor(max_x / bin_size) + 1):
                for by in range(math.floor(min_y / bin_size), math.floor(max_y / bin_size) + 1):
                    index.setdefault((bx, by), []).append(i)
        return index

    def _candidate_skeleton_indices(
        self,
        ray: Ray,
        skeletons: List[SRAFSkeleton],
        index: Optional[Dict[Tuple[int, int], List[int]]],
    ) -> List[int]:
        if index is None:
            return list(range(len(skeletons)))
        bin_size = max(self.config.spatial_bin_size, EPS)
        step = max(bin_size / 2.0, 1.0)
        seen: Set[int] = set()
        t = 0.0
        while t <= self.config.max_ray_distance:
            p = ray.origin + ray.direction * t
            bx = math.floor(p.x / bin_size)
            by = math.floor(p.y / bin_size)
            for nx in (bx - 1, bx, bx + 1):
                for ny in (by - 1, by, by + 1):
                    seen.update(index.get((nx, ny), []))
            t += step
        return sorted(seen)

    def _intersect_ray_skeletons(self, ray: Ray, skeletons: List[SRAFSkeleton], index: Optional[Dict[Tuple[int, int], List[int]]] = None) -> List[RayHit]:
        hits: List[RayHit] = []
        candidate_indices = self._candidate_skeleton_indices(ray, skeletons, index)
        for i in candidate_indices:
            skeleton = skeletons[i]
            hit = self._closest_ray_segment_hit(ray, skeleton, i)
            if hit is not None and hit.entry_distance <= self.config.max_ray_distance:
                hits.append(hit)
        hits.sort(key=lambda h: (round(h.distance / 1.0e-6), h.skeleton_index))
        return hits

    def _closest_ray_segment_hit(self, ray: Ray, skeleton: SRAFSkeleton, skeleton_index: int) -> Optional[RayHit]:
        p = ray.origin
        d = ray.direction.normalized()
        a = skeleton.start
        v = skeleton.end - skeleton.start
        vv = v.dot(v)
        if vv < EPS:
            u = 0.0
            t = max(0.0, (a - p).dot(d))
        else:
            # Minimize |p + t*d - (a + u*v)|^2.  First solve the
            # unconstrained two-variable system, then clamp segment u and
            # recompute the ray projection t.
            r = a - p
            dv = d.dot(v)
            dr = d.dot(r)
            vr = v.dot(r)
            det = vv - dv * dv
            if abs(det) < EPS:
                u = max(0.0, min(1.0, vr / vv))
            else:
                u = max(0.0, min(1.0, (dv * dr - vr) / (dv * dv - vv)))
            closest_on_seg = a + v * u
            t = max(0.0, (closest_on_seg - p).dot(d))
        closest_ray = p + d * t
        closest_seg = a + v * u
        separation = (closest_ray - closest_seg).norm()
        angle_delta = abs(math.sin(ray.direction.angle() - skeleton.angle))
        if angle_delta <= self.config.parallel_tolerance and separation <= skeleton.half_width + self.config.parallel_tolerance:
            return None
        if separation > skeleton.half_width + EPS or t <= self.config.ray_origin_nudge * 0.1:
            return None
        projected_width = (2.0 * skeleton.half_width) / max(angle_delta, 0.20)
        chord_half = math.sqrt(max(skeleton.half_width**2 - separation**2, 0.0)) / max(angle_delta, 0.20)
        entry = max(0.0, t - chord_half)
        exit_distance = min(self.config.max_ray_distance, t + chord_half)
        return RayHit(
            skeleton_index=skeleton_index,
            distance=max(0.0, t),
            entry_distance=entry,
            exit_distance=exit_distance,
            projected_width=projected_width,
            skeleton_angle=skeleton.angle,
            closest_point=closest_seg,
        )

    def _features_for_hits(self, hits: List[RayHit], valid: bool = True) -> RayFeature:
        if not valid:
            return RayFeature(0, NO_HIT_DISTANCE, 0.0, 0.0, 0.0, valid=False, miss=True)
        if not hits:
            return RayFeature(0, NO_HIT_DISTANCE, 0.0, 0.0, 0.0, valid=True, miss=True)
        nearest = hits[0]
        coverage = sum(max(0.0, h.exit_distance - h.entry_distance) for h in hits)
        truncated = any(h.exit_distance >= self.config.max_ray_distance - EPS for h in hits)
        return RayFeature(len(hits), nearest.distance, nearest.projected_width, nearest.skeleton_angle, coverage, valid=True, miss=False, truncated=truncated)

    def _build_fingerprint(
        self,
        features: List[RayFeature],
        rays: List[Ray],
        contacts: List[Polygon],
        srafs: List[Polygon],
    ) -> CellFingerprint:
        feature_vector = [value for f in features for value in f.as_tuple()]
        quantized: List[Tuple[int, int, int, int, int]] = []
        for feature in features:
            quantized.append(
                (
                    feature.hit_count,
                    int(round(feature.nearest_distance / self.config.distance_quantization)),
                    int(round(feature.projected_width / self.config.width_quantization)),
                    int(round(((feature.skeleton_angle + math.pi) % (2 * math.pi)) / (math.pi / 18.0))),
                    int(round(feature.total_coverage / self.config.width_quantization)),
                )
            )
        coarse_hash = hashlib.sha256(json.dumps(quantized, separators=(",", ":")).encode()).hexdigest()
        rdf = self._radial_distribution(contacts, srafs)
        angular_profile = self._angular_profile(contacts, srafs)
        hu_moments = self._hu_moments(srafs)
        topology_signature = self._topology_signature(rays, features)
        return CellFingerprint(coarse_hash, rdf, angular_profile, hu_moments, topology_signature, feature_vector)

    def _radial_distribution(self, contacts: List[Polygon], srafs: List[Polygon]) -> List[float]:
        hist = [0.0 for _ in range(self.config.rdf_bins)]
        if not contacts:
            return hist
        centers = [poly.centroid() for poly in contacts]
        bin_width = self.config.rdf_max_radius / self.config.rdf_bins
        for sraf in srafs:
            sc = sraf.centroid()
            area = sraf.area()
            for cc in centers:
                idx = int((sc - cc).norm() / bin_width)
                if 0 <= idx < self.config.rdf_bins:
                    hist[idx] += area
        contact_count = max(len(contacts), 1)
        return [value / contact_count for value in hist]

    def _angular_profile(self, contacts: List[Polygon], srafs: List[Polygon]) -> List[float]:
        profile = [self.config.max_ray_distance for _ in range(self.config.angular_profile_bins)]
        if not contacts:
            return profile
        centroids = [p.centroid() for p in contacts]
        center = Vec2(sum(c.x for c in centroids) / len(centroids), sum(c.y for c in centroids) / len(centroids))
        for sraf in srafs:
            delta = sraf.centroid() - center
            r = delta.norm()
            if r < EPS:
                continue
            angle = delta.angle() % (2 * math.pi)
            idx = int(round(angle / (2 * math.pi) * self.config.angular_profile_bins)) % self.config.angular_profile_bins
            profile[idx] = min(profile[idx], r)
            for off in (-1, 1):
                profile[(idx + off) % self.config.angular_profile_bins] = min(
                    profile[(idx + off) % self.config.angular_profile_bins], r * 1.05
                )
        return [value / self.config.max_ray_distance for value in profile]

    def _hu_moments(self, srafs: List[Polygon]) -> List[float]:
        samples: List[Tuple[float, float, float]] = []
        for poly in srafs:
            area = max(poly.area(), EPS)
            for v in poly.vertices:
                samples.append((v.x, v.y, area / len(poly.vertices)))
        if not samples:
            return [0.0] * 7
        total = sum(w for _, _, w in samples)
        cx = sum(x * w for x, _, w in samples) / total
        cy = sum(y * w for _, y, w in samples) / total

        def mu(power_x: int, power_y: int) -> float:
            return sum(((x - cx) ** power_x) * ((y - cy) ** power_y) * w for x, y, w in samples)

        def eta(power_x: int, power_y: int) -> float:
            return mu(power_x, power_y) / (total ** (1.0 + (power_x + power_y) / 2.0) + EPS)

        n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
        n30, n12, n21, n03 = eta(3, 0), eta(1, 2), eta(2, 1), eta(0, 3)
        hu = [
            n20 + n02,
            (n20 - n02) ** 2 + 4 * n11**2,
            (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
            (n30 + n12) ** 2 + (n21 + n03) ** 2,
            (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
            (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2)
            + 4 * n11 * (n30 + n12) * (n21 + n03),
            (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
            - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
        ]
        return [(1.0 if value >= 0 else -1.0) * math.log10(abs(value) + EPS) for value in hu]

    def _topology_signature(self, rays: List[Ray], features: List[RayFeature]) -> str:
        sectors = [0] * self.config.topology_sectors
        for ray, feature in zip(rays, features):
            if feature.hit_count <= 0:
                continue
            idx = int((ray.angle % (2 * math.pi)) / (2 * math.pi) * self.config.topology_sectors)
            sectors[idx] += min(feature.hit_count, 9)
        return hashlib.md5("".join(str(min(v, 9)) for v in sectors).encode()).hexdigest()

    def _score_cells(self) -> None:
        by_type: Dict[str, List[UnitCell]] = {}
        for cell in self.cells:
            by_type.setdefault(cell.cell_type, []).append(cell)
        self.scores = {}
        for cell_type, cells in by_type.items():
            baseline_cells = self._select_baseline_cells(cells)
            global_refs = self._reference_bundle(baseline_cells)
            for cell in cells:
                reference_cells = baseline_cells
                if self.config.use_local_baseline and cell.grid_position is not None:
                    local = [
                        candidate
                        for candidate in baseline_cells
                        if candidate.grid_position is not None
                        and max(
                            abs(candidate.grid_position[0] - cell.grid_position[0]),
                            abs(candidate.grid_position[1] - cell.grid_position[1]),
                        )
                        <= self.config.local_baseline_radius
                    ]
                    if len(local) >= min(self.config.min_reference_cells, len(baseline_cells)):
                        reference_cells = local
                refs = self._reference_bundle(reference_cells) if reference_cells is not baseline_cells else global_refs
                self._score_one_cell(cell, cell_type, refs, len(reference_cells) < self.config.min_reference_cells)

    def _select_baseline_cells(self, cells: List[UnitCell]) -> List[UnitCell]:
        if len(cells) < self.config.min_reference_cells or self.config.exclude_top_fraction <= 0:
            return cells
        refs = self._reference_bundle(cells)
        ranked = []
        for cell in cells:
            fp = self.fingerprints[cell.cell_id]
            radial = self._normalized_l1(fp.rdf, refs["rdf"], 0.02)
            angular = self._normalized_l1(fp.angular_profile, refs["angular"], 0.01)
            moment = self._normalized_l1(fp.hu_moments, refs["moment"], 0.05)
            ranked.append((0.30 * radial + 0.30 * angular + 0.20 * moment, cell))
        drop_count = min(len(cells) - self.config.min_reference_cells, max(1, int(len(cells) * self.config.exclude_top_fraction)))
        if drop_count <= 0:
            return cells
        dropped = {cell.cell_id for _, cell in sorted(ranked, key=lambda item: item[0], reverse=True)[:drop_count]}
        return [cell for cell in cells if cell.cell_id not in dropped]

    def _reference_bundle(self, cells: List[UnitCell]) -> Dict[str, object]:
        fps = [self.fingerprints[cell.cell_id] for cell in cells]
        feature_rows = [fp.feature_vector for fp in fps]
        feature_ref = self._column_median(feature_rows)
        abs_dev_rows = [[abs(value - ref) for value, ref in zip(row, feature_ref)] for row in feature_rows]
        feature_mad = self._column_median(abs_dev_rows)
        floors = self._feature_noise_vector(len(feature_ref))
        feature_scale = [max(mad * 1.4826, floor) for mad, floor in zip(feature_mad, floors)]
        missing_rates = self._feature_missing_rates(feature_rows)
        return {
            "rdf": self._column_median([fp.rdf for fp in fps]),
            "angular": self._column_median([fp.angular_profile for fp in fps]),
            "moment": self._column_median([fp.hu_moments for fp in fps]),
            "feature_ref": feature_ref,
            "feature_scale": feature_scale,
            "missing_rates": missing_rates,
            "topology": self._majority([fp.topology_signature for fp in fps]) if fps else "",
        }

    def _score_one_cell(self, cell: UnitCell, cell_type: str, refs: Dict[str, object], insufficient_data: bool) -> None:
        fp = self.fingerprints[cell.cell_id]
        radial = self._normalized_l1(fp.rdf, refs["rdf"], 0.02)  # type: ignore[arg-type]
        angular = self._normalized_l1(fp.angular_profile, refs["angular"], 0.01)  # type: ignore[arg-type]
        moment = self._normalized_l1(fp.hu_moments, refs["moment"], 0.05)  # type: ignore[arg-type]
        topology_match = fp.topology_signature == refs["topology"]
        raw = 0.30 * radial + 0.30 * angular + 0.20 * moment + (0.0 if topology_match else 0.20)
        if insufficient_data:
            raw *= 0.5
        overall = min(1.0, raw / max(self.config.score_scale, EPS))
        z = self._per_feature_zscore(
            fp.feature_vector,
            refs["feature_ref"],  # type: ignore[arg-type]
            refs["feature_scale"],  # type: ignore[arg-type]
            refs["missing_rates"],  # type: ignore[arg-type]
        )
        details = self._anomaly_details(radial, angular, moment, topology_match, z)
        if insufficient_data:
            details.append("reference sample below recommended minimum; review manually")
        details.extend(self.warnings.get(cell.cell_id, []))
        self.scores[cell.cell_id] = ConsistencyScore(
            cell.cell_id,
            cell_type,
            overall,
            radial,
            angular,
            moment,
            topology_match,
            details,
            z,
            insufficient_data,
        )

    def _feature_missing_rates(self, feature_rows: List[List[float]]) -> List[float]:
        if not feature_rows:
            return []
        rates: List[float] = []
        for col in range(len(feature_rows[0])):
            if col % 5 == 1:
                rates.append(sum(1 for row in feature_rows if row[col] == NO_HIT_DISTANCE) / len(feature_rows))
            else:
                rates.append(0.0)
        return rates

    def _per_feature_zscore(
        self,
        values: List[float],
        refs: List[float],
        scales: List[float],
        missing_rates: List[float],
    ) -> List[float]:
        z: List[float] = []
        for idx, (value, ref, scale) in enumerate(zip(values, refs, scales)):
            if idx % 5 == 1 and missing_rates[idx] > self.config.missing_rate_exclusion:
                z.append(0.0)
            elif idx % 5 == 1 and value == NO_HIT_DISTANCE and ref == NO_HIT_DISTANCE:
                z.append(0.0)
            else:
                z.append(abs((value - ref) / max(scale, EPS)))
        return z

    def _feature_noise_vector(self, size: int) -> List[float]:
        floors = [0.5, self.config.distance_noise_floor, self.config.width_noise_floor, 0.02, self.config.width_noise_floor]
        return [floors[i % len(floors)] for i in range(size)]

    def _anomaly_details(
        self,
        radial: float,
        angular: float,
        moment: float,
        topology_match: bool,
        z: List[float],
    ) -> List[str]:
        details: List[str] = []
        if not topology_match:
            details.append("topology signature differs from the reference pattern")
        if radial > 0.10:
            details.append(f"radial SRAF density deviation {radial:.3f}")
        if angular > 0.10:
            details.append(f"angular distance-profile deviation {angular:.3f}")
        if moment > 0.10:
            details.append(f"shape moment deviation {moment:.3f}")
        if z:
            max_z = max(z)
            if max_z > 3.0:
                details.append(f"largest per-ray robust z-score {max_z:.1f}")
        return details

    @staticmethod
    def _majority(items: List[str]) -> str:
        counts: Dict[str, int] = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    @staticmethod
    def _column_median(rows: List[List[float]]) -> List[float]:
        if not rows:
            return []
        return [SRAFConsistencyAnalyzer._median([row[i] for row in rows]) for i in range(len(rows[0]))]

    @staticmethod
    def _median(values: List[float]) -> float:
        ordered = sorted(values)
        n = len(ordered)
        mid = n // 2
        if n % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2.0

    @staticmethod
    def _normalized_l1(values: List[float], reference: List[float], floor: float) -> float:
        active = [(value, ref) for value, ref in zip(values, reference) if abs(value) > floor or abs(ref) > floor]
        if not active:
            return 0.0
        total = 0.0
        for value, ref in active:
            total += abs(value - ref) / max(abs(ref), floor)
        return total / len(active)

    @staticmethod
    def _fingerprint_distance(a: CellFingerprint, b: CellFingerprint) -> float:
        def norm_delta(x: List[float], y: List[float]) -> float:
            return math.sqrt(sum((vx - vy) ** 2 for vx, vy in zip(x, y)))

        rdf = norm_delta(a.rdf, b.rdf)
        angular = norm_delta(a.angular_profile, b.angular_profile)
        moments = norm_delta(a.hu_moments, b.hu_moments) / max(len(a.hu_moments), 1)
        topology = 0.0 if a.topology_signature == b.topology_signature else 1.0
        return 0.30 * rdf + 0.30 * angular + 0.20 * moments + 0.20 * topology


__all__ = [
    "CellFingerprint",
    "ConsistencyScore",
    "Polygon",
    "Ray",
    "RayCastingConfig",
    "RayFeature",
    "RayHit",
    "RayType",
    "SRAFConsistencyAnalyzer",
    "SRAFSkeleton",
    "SymmetryOp",
    "UnitCell",
    "Vec2",
]
