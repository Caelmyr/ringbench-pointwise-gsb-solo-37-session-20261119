"""Scattering vectors and separations of already measured detector spots."""
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from .detector_io import csv_bytes
from .errors import WorkbenchError
from .values import items, label, mapping, selection, unique

MAX_SELECTED = 18


@dataclass(frozen=True)
class VectorSettings:
    labels: tuple
    grouping: str
    reference: str | None

    @classmethod
    def read(cls, v):
        doc = mapping(v, "斑点向量设置")
        names = tuple(unique(
            [label(name, "斑点名称", 48) for name in items(
                doc.get("labels"), "所选斑点", minimum=2, maximum=MAX_SELECTED,
            )],
            "所选斑点",
        ))
        grouping = selection(doc.get("grouping", "pairs"), ("pairs", "reference"), "斑点组合")
        chosen = doc.get("reference")
        if grouping == "reference":
            chosen = label(chosen, "基准斑点", 48)
            if chosen not in names:
                raise WorkbenchError("基准斑点必须包含在当前选择中。")
        return cls(names, grouping, chosen)


@dataclass(frozen=True)
class Vector:
    label: str
    x: float
    y: float
    xq: float
    yq: float
    zq: float

    @property
    def coordinates(self):
        return np.array([self.xq, self.yq, self.zq])

    @property
    def magnitude(self):
        return math.hypot(self.xq, self.yq, self.zq)

    @property
    def d_spacing(self):
        length = self.magnitude
        return 2 * math.pi / length if length > 0 else None

    def document(self):
        return {
            "label": self.label,
            "detector_x_px": self.x,
            "detector_y_px": self.y,
            "qx": self.xq,
            "qy": self.yq,
            "qz": self.zq,
            "q_magnitude": self.magnitude,
            "d_a": self.d_spacing,
        }


@dataclass(frozen=True)
class VectorDifference:
    first: Vector
    second: Vector
    delta_xq: float
    delta_yq: float
    delta_zq: float
    angle: float | None
    image_dist: float
    image_angle: float | None

    @property
    def delta_norm(self):
        return math.hypot(self.delta_xq, self.delta_yq, self.delta_zq)

    @property
    def in_plane(self):
        return math.hypot(self.delta_xq, self.delta_yq)

    def document(self):
        return {
            "first": self.first.label,
            "second": self.second.label,
            "delta_qx": self.delta_xq,
            "delta_qy": self.delta_yq,
            "delta_qz": self.delta_zq,
            "delta_q": self.delta_norm,
            "delta_in_plane": self.in_plane,
            "angle_between_q_deg": self.angle,
            "detector_distance_px": self.image_dist,
            "detector_bearing_deg": self.image_angle,
        }


@dataclass(frozen=True)
class VectorMeasurement:
    vectors: list
    pairs: list
    spec: VectorSettings

    def document(self):
        nonzero = [p.delta_norm for p in self.pairs if p.delta_norm > 0]
        return {
            "units": {"vectors": "A^-1", "detector": "px", "angles": "deg"},
            "axes": {
                "qx": "detector right",
                "qy": "detector down",
                "qz": "outgoing projection minus incident wavevector along beam",
            },
            "grouping": self.spec.grouping,
            "reference": self.spec.reference,
            "vectors": [v.document() for v in self.vectors],
            "pairs": [pair.document() for pair in self.pairs],
            "summary": {
                "spots": len(self.vectors),
                "comparisons": len(self.pairs),
                "smallest_positive_separation": min(nonzero, default=None),
                "greatest_separation": max((p.delta_norm for p in self.pairs), default=None),
            },
        }

    def csv(self, kind):
        if kind == "vectors":
            headings = [
                "label", "detector_x_px", "detector_y_px", "qx", "qy", "qz", "q_magnitude", "d_a",
            ]
            rows = [vector.document() for vector in self.vectors]
        else:
            headings = [
                "first", "second", "delta_qx", "delta_qy", "delta_qz", "delta_q", "delta_in_plane",
                "angle_between_q_deg", "detector_distance_px", "detector_bearing_deg",
            ]
            rows = [pair.document() for pair in self.pairs]
        return csv_bytes(headings, (tuple(_csv_value(row[key]) for key in headings) for row in rows))


def _csv_value(value):
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _scatter(spot, geometry):
    dx = (spot.x - geometry.cx) * geometry.pixel_mm
    dy = (spot.y - geometry.cy) * geometry.pixel_mm
    a = geometry.distance_mm
    ray = math.hypot(dx, dy, a)
    k = 2 * math.pi / geometry.wavelength_a
    qx = k * dx / ray
    qy = k * dy / ray
    # Stable form of k * (a / ray - 1) near the direct beam.
    qz = -k * (dx * dx + dy * dy) / (ray * (ray + a))
    return Vector(spot.label, spot.x, spot.y, qx, qy, qz)


def _compare(first, second, geometry):
    q0, q1 = first.coordinates, second.coordinates
    dx, dy, dz = q1 - q0
    if first.magnitude > 0 and second.magnitude > 0:
        q_angle = math.degrees(math.atan2(float(np.linalg.norm(np.cross(q0, q1))), float(q0 @ q1)))
    else:
        q_angle = None
    detector_x = second.x - first.x
    detector_y = second.y - first.y
    radius = math.hypot(detector_x, detector_y)
    bearing = (
        (math.degrees(math.atan2(detector_y, detector_x)) - geometry.azimuth_zero) % 360
        if radius > 0 else None
    )
    return VectorDifference(first, second, float(dx), float(dy), float(dz), q_angle, radius, bearing)


def measure_vectors(spots, geometry, spec):
    independent = {spot.label: spot for spot in spots}
    if any(entry not in independent for entry in spec.labels):
        raise WorkbenchError("所选斑点的标记已变化，请重新选择。")
    vectors = [_scatter(independent[entry], geometry) for entry in spec.labels]
    if spec.grouping == "reference":
        source = next(v for v in vectors if v.label == spec.reference)
        bases = [(source, value) for value in vectors if value.label != source.label]
    else:
        bases = combinations(vectors, 2)
    return VectorMeasurement(vectors, [_compare(a, b, geometry) for a, b in bases], spec)
