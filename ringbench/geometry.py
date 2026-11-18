"""Flat-detector coordinates, diffraction units and reference-ring calibration."""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .errors import WorkbenchError
from .values import MAX_SIDE, items, mapping, number, pair, selection, xy_bounds

RADIAL_UNITS = ("px", "mm", "2theta", "q")
AXIS_LABELS = {"px": "半径 (px)", "mm": "半径 (mm)", "2theta": "2θ (°)", "q": "q (Å⁻¹)"}
MIN_FIT_SPAN = 30.0


@dataclass(frozen=True)
class Geometry:
    cx: float
    cy: float
    pixel_mm: float = 0.172
    distance_mm: float = 150.0
    wavelength_a: float = 1.54
    azimuth_zero: float = 0.0

    @classmethod
    def default(cls, width, height):
        return cls((width - 1) / 2, (height - 1) / 2)

    @classmethod
    def read(cls, value, width, height):
        v = mapping(value, "探测器几何")
        cx, cy = pair(
            [v.get("cx"), v.get("cy")], "环心",
            xy_bounds(width, height, max(width, height)),
        )
        return cls(
            cx,
            cy,
            number(v.get("pixel_mm"), "像素尺寸 (mm)", 0.0001, 20),
            number(v.get("distance_mm"), "探测器距离 (mm)", 0.2, 10_000),
            number(v.get("wavelength_a"), "入射波长 (Å)", 0.02, 100),
            number(v.get("azimuth_zero", 0), "角度零位", -360, 360),
        )

    def document(self):
        return asdict(self)

    def radial(self, radius, unit):
        selection(unit, RADIAL_UNITS, "径向单位")
        r = np.asarray(radius)
        if unit == "px":
            return r
        mm = r * self.pixel_mm
        if unit == "mm":
            return mm
        two_theta = np.arctan2(mm, self.distance_mm)
        if unit == "2theta":
            return np.degrees(two_theta)
        return 4.0 * np.pi / self.wavelength_a * np.sin(two_theta / 2.0)

    def radius_from_unit(self, radial, unit):
        value = np.asarray(radial)
        selection(unit, RADIAL_UNITS, "径向单位")
        if unit == "px":
            return value
        if unit == "mm":
            return value / self.pixel_mm
        if unit == "2theta":
            radians = np.radians(value)
        else:
            arcs = value * self.wavelength_a / (4 * np.pi)
            if np.any(np.abs(arcs) >= math.sqrt(0.5)):
                raise WorkbenchError("q 超过当前平面几何能够表示的角度范围。")
            radians = 2 * np.arcsin(arcs)
        if np.any(radians >= math.pi / 2):
            raise WorkbenchError("2θ 必须位于探测器正向半空间。")
        return self.distance_mm * np.tan(radians) / self.pixel_mm

    def values_at(self, x, y):
        dx = np.asarray(x) - self.cx
        dy = np.asarray(y) - self.cy
        rr = np.hypot(dx, dy)
        az = (np.degrees(np.arctan2(dy, dx)) - self.azimuth_zero) % 360.0
        return rr, az

    def position(self, radius, azimuth):
        angle = np.radians(np.asarray(azimuth) + self.azimuth_zero)
        return self.cx + radius * np.cos(angle), self.cy + radius * np.sin(angle)

    def radius_limit(self, width, height):
        xs = np.array([-0.5, width - 0.5])
        ys = np.array([-0.5, height - 0.5])
        distances = np.hypot(xs[None, :] - self.cx, ys[:, None] - self.cy)
        return float(np.max(distances))

    def detector_grid(self, shape, row_start=0, row_end=None):
        height, width = shape
        end = height if row_end is None else row_end
        return self.values_at(
            np.arange(width, dtype=np.float64)[None, :],
            np.arange(row_start, end, dtype=np.float64)[:, None],
        )

    def spot_coordinates(self, x, y):
        distance, azimuth = self.values_at(x, y)
        return {
            "radius_px": float(distance),
            "radius_mm": float(self.radial(distance, "mm")),
            "two_theta_deg": float(self.radial(distance, "2theta")),
            "q_inverse_a": float(self.radial(distance, "q")),
            "azimuth_deg": float(azimuth),
        }


@dataclass(frozen=True)
class Arc:
    start: float
    span: float

    @classmethod
    def read(cls, start=0, span=360):
        return cls(
            number(start, "起始方位", -36000, 36000) % 360.0,
            number(span, "方位宽度", 0.1, 360),
        )

    def relative(self, azimuth):
        return (azimuth - self.start) % 360.0

    def inside(self, azimuth):
        return self.relative(azimuth) < self.span

    def documents(self):
        return {"start": self.start, "span": self.span}

    def sample_angles(self, cells):
        return (self.start + np.linspace(0, self.span, cells, endpoint=False) + self.span / cells / 2) % 360


def ring_marks(value, width, height, *, minimum=0):
    raw = items(value, "环线标点", minimum=minimum, maximum=256)
    bounds = xy_bounds(width, height, 0.5)
    marks = [pair(point, "环线标点", bounds) for point in raw]
    if len(set(marks)) != len(marks):
        raise WorkbenchError("环线标点中有重复值。")
    return marks


def angular_span(x, y, cx, cy):
    phi = np.sort(np.degrees(np.arctan2(y - cy, x - cx)) % 360.0)
    gaps = np.diff(np.concatenate((phi, phi[:1] + 360.0)))
    return float(360.0 - np.max(gaps))


def fit_circle(points, width, height):
    """Fit one reference ring from user-supplied marks in detector pixels."""
    p = np.asarray(ring_marks(points, width, height, minimum=3), dtype=np.float64)
    mean = np.mean(p, axis=0)
    local = p - mean
    x, y = local.T
    matrix = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    solution, _, rank, _ = np.linalg.lstsq(matrix, x * x + y * y, rcond=None)
    if rank != 3:
        raise WorkbenchError("环线标点接近同一直线，无法拟合圆。")
    center = mean + solution[:2]
    cx, cy = center
    distances = np.hypot(p[:, 0] - cx, p[:, 1] - cy)
    radius = float(np.mean(distances))
    span = angular_span(p[:, 0], p[:, 1], cx, cy)
    if span < MIN_FIT_SPAN or not 2 < radius <= 2 * max(width, height):
        raise WorkbenchError("可用环弧太短或半径不合理，请加选更分散的环线标点。")
    pair([cx, cy], "拟合环心", xy_bounds(width, height, max(width, height)))
    residual = distances - radius
    return {
        "cx": float(cx),
        "cy": float(cy),
        "radius_px": radius,
        "rms_px": float(np.sqrt(np.mean(residual ** 2))),
        "max_residual_px": float(np.max(np.abs(residual))),
        "span_deg": span,
        "count": len(p),
        "residual_px": residual.tolist(),
    }


def reference_distance(geometry, picked_radius_px, reference_d):
    radius = number(picked_radius_px, "参考环半径", 0.5, MAX_SIDE * 3)
    d = number(reference_d, "晶面间距", 0.001, 5000)
    sine = geometry.wavelength_a / (2 * d)
    if not 0 < sine < math.sqrt(0.5):
        raise WorkbenchError("这组晶面间距和波长无法在当前平面正向探测器上标定。")
    angle = 2 * math.asin(sine)
    distance = radius * geometry.pixel_mm / math.tan(angle)
    distance = number(distance, "拟合探测器距离", 0.2, 10_000)
    return {"distance_mm": distance, "two_theta_deg": math.degrees(angle), "d_a": d}


def reference_rings(geometry, entries, width, height):
    result = []
    limit = geometry.radius_limit(width, height)
    for value in items(entries, "参考环", maximum=30):
        d = number(value, "晶面间距", 0.001, 5000)
        sine = geometry.wavelength_a / (2 * d)
        if not 0 < sine < math.sqrt(0.5):
            continue
        two_theta = 2 * math.asin(sine)
        r = geometry.distance_mm * math.tan(two_theta) / geometry.pixel_mm
        if 0.5 <= r <= limit:
            result.append({"d_a": d, "radius_px": r, "angle_deg": math.degrees(two_theta)})
    return result
