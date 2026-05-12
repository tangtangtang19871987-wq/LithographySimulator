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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

EPS = 1.0e-9


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

    def _analyze_cell(self, cell: UnitCell) -> None:
        contacts = [self._to_canonical_polygon(p, cell.symmetry) for p in cell.local_contacts()]
        srafs = [self._to_canonical_polygon(p, cell.symmetry) for p in self._srafs_with_optional_halo(cell)]
        rays = self._generate_canonical_rays(contacts)
        skeletons = [self._extract_skeleton(poly, i) for i, poly in enumerate(srafs)]
        hits = [self._intersect_ray_skeletons(ray, skeletons) for ray in rays]
        features = [self._features_for_hits(hit_list) for hit_list in hits]
        fingerprint = self._build_fingerprint(features, rays, contacts, srafs)
        self.rays_by_cell[cell.cell_id] = rays
        self.hits_by_cell[cell.cell_id] = hits
        self.features_by_cell[cell.cell_id] = features
        self.fingerprints[cell.cell_id] = fingerprint

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

    def _generate_canonical_rays(self, contacts: List[Polygon]) -> List[Ray]:
        rays: List[Ray] = []
        ordered_contacts = sorted(enumerate(contacts), key=lambda item: (item[1].centroid().x, item[1].centroid().y))
        ordinal = 0
        for contact_rank, (original_index, contact) in enumerate(ordered_contacts):
            center = contact.centroid()
            for i in range(self.config.num_angular_rays):
                angle = 2.0 * math.pi * i / self.config.num_angular_rays
                rays.append(Ray(center, Vec2(math.cos(angle), math.sin(angle)), RayType.ANGULAR, contact_rank, ordinal, angle))
                ordinal += 1
            for edge_index, (a, b) in enumerate(contact.edges()):
                edge = b - a
                normal = Vec2(-edge.y, edge.x).normalized()
                midpoint = (a + b) * 0.5
                if normal.dot(midpoint - center) < 0:
                    normal = normal * -1.0
                for sample in range(self.config.num_edge_samples):
                    t = (sample + 0.5) / self.config.num_edge_samples
                    origin = a * (1.0 - t) + b * t
                    rays.append(Ray(origin, normal, RayType.EDGE_NORMAL, contact_rank, ordinal, normal.angle()))
                    ordinal += 1
            verts = contact.vertices
            n = len(verts)
            orientation = math.copysign(1.0, sum(a.cross(b) for a, b in contact.edges()) or 1.0)
            for i, vertex in enumerate(verts):
                prev_v = verts[(i - 1) % n]
                next_v = verts[(i + 1) % n]
                e_prev = (vertex - prev_v).normalized()
                e_next = (next_v - vertex).normalized()
                # Exterior bisector: opposite of normalized inward bisector.
                inward = (Vec2(-e_prev.y, e_prev.x) * orientation + Vec2(-e_next.y, e_next.x) * orientation).normalized()
                direction = (inward * -1.0).normalized()
                if direction.norm() < EPS:
                    direction = (vertex - center).normalized()
                rays.append(Ray(vertex, direction, RayType.VERTEX_BISECTOR, contact_rank, ordinal, direction.angle()))
                ordinal += 1
        if len(ordered_contacts) > 1:
            centroids = [poly.centroid() for _, poly in ordered_contacts]
            for i, origin in enumerate(centroids):
                for j, target in enumerate(centroids):
                    if i == j:
                        continue
                    direction = (target - origin).normalized()
                    rays.append(Ray(origin, direction, RayType.INTER_CONTACT, i, ordinal, direction.angle()))
                    ordinal += 1
        return rays

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

    def _intersect_ray_skeletons(self, ray: Ray, skeletons: List[SRAFSkeleton]) -> List[RayHit]:
        hits: List[RayHit] = []
        for i, skeleton in enumerate(skeletons):
            hit = self._closest_ray_segment_hit(ray, skeleton, i)
            if hit is not None and hit.entry_distance <= self.config.max_ray_distance:
                hits.append(hit)
        hits.sort(key=lambda h: h.distance)
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
        if separation > skeleton.half_width + EPS or t < -EPS:
            return None
        angle_delta = abs(math.sin(ray.direction.angle() - skeleton.angle))
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

    def _features_for_hits(self, hits: List[RayHit]) -> RayFeature:
        if not hits:
            return RayFeature(0, self.config.max_ray_distance, 0.0, 0.0, 0.0)
        nearest = hits[0]
        coverage = sum(max(0.0, h.exit_distance - h.entry_distance) for h in hits)
        return RayFeature(len(hits), nearest.distance, nearest.projected_width, nearest.skeleton_angle, coverage)

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
            fps = [self.fingerprints[cell.cell_id] for cell in cells]
            rdf_ref = self._column_median([fp.rdf for fp in fps])
            angular_ref = self._column_median([fp.angular_profile for fp in fps])
            moment_ref = self._column_median([fp.hu_moments for fp in fps])
            feature_rows = [fp.feature_vector for fp in fps]
            feature_ref = self._column_median(feature_rows)
            abs_dev_rows = [[abs(value - ref) for value, ref in zip(row, feature_ref)] for row in feature_rows]
            feature_mad = self._column_median(abs_dev_rows)
            floors = self._feature_noise_vector(len(feature_ref))
            feature_scale = [max(mad * 1.4826, floor) for mad, floor in zip(feature_mad, floors)]
            topology_ref = self._majority([fp.topology_signature for fp in fps])
            for cell in cells:
                fp = self.fingerprints[cell.cell_id]
                radial = self._normalized_l1(fp.rdf, rdf_ref, 0.02)
                angular = self._normalized_l1(fp.angular_profile, angular_ref, 0.01)
                moment = self._normalized_l1(fp.hu_moments, moment_ref, 0.05)
                topology_match = fp.topology_signature == topology_ref
                raw = 0.30 * radial + 0.30 * angular + 0.20 * moment + (0.0 if topology_match else 0.20)
                overall = min(1.0, raw / max(self.config.score_scale, EPS))
                z = [abs((value - ref) / scale) for value, ref, scale in zip(fp.feature_vector, feature_ref, feature_scale)]
                details = self._anomaly_details(radial, angular, moment, topology_match, z)
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
                )

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
