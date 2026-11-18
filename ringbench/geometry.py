"""Flat-detector coordinates, diffraction units and reference-ring calibration."""
from dataclasses import asdict, dataclass
import itertools
import math

import numpy as np

from .errors import WorkbenchError
from .values import MAX_SIDE, items, mapping, number, pair, selection, xy_bounds

RADIAL_UNITS = ("px", "mm", "2theta", "q")
AXIS_LABELS = {"px": "半径 (px)", "mm": "半径 (mm)", "2theta": "2θ (°)", "q": "q (Å⁻¹)"}
MIN_FIT_SPAN = 30.0
INLIER_TOLERANCE_PX = 3.0
MARK_SIGMA_PX = 1.0
CENTER_SIGMA_LIMIT_PX = 6.0
EXHAUSTIVE_MARKS = 20
CANDIDATE_TRIPLES = 1500
CANDIDATE_SEED = 20240617


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


def _circle_through(a, b, c):
    """Circumcircle of three marks, or None if they are (nearly) collinear."""
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    determinant = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    scale = max(abs(x1), abs(y1), abs(x2), abs(y2), abs(x3), abs(y3), 1.0)
    if abs(determinant) <= 1e-9 * scale * scale:
        return None
    s1 = x1 * x1 + y1 * y1
    s2 = x2 * x2 + y2 * y2
    s3 = x3 * x3 + y3 * y3
    cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / determinant
    cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / determinant
    radius = math.hypot(x1 - cx, y1 - cy)
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(radius) and radius > 1e-9):
        return None
    return np.array([cx, cy], dtype=np.float64), radius


def _algebraic_circle(points):
    """Least-squares circle through every supplied mark (algebraic start value)."""
    mean = np.mean(points, axis=0)
    local = points - mean
    x, y = local.T
    matrix = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    solution, _, rank, _ = np.linalg.lstsq(matrix, x * x + y * y, rcond=None)
    if rank != 3:
        return None
    center = mean + solution[:2]
    radius = float(np.mean(np.hypot(points[:, 0] - center[0], points[:, 1] - center[1])))
    if not (np.all(np.isfinite(center)) and math.isfinite(radius) and radius > 1e-9):
        return None
    return center, radius


def _refine_circle(points, center, radius):
    """Polish a circle by minimizing true radial distances (Gauss-Newton)."""
    center = np.asarray(center, dtype=np.float64)
    radius = float(radius)
    for _ in range(20):
        dx = points[:, 0] - center[0]
        dy = points[:, 1] - center[1]
        distance = np.hypot(dx, dy)
        if np.any(distance < 1e-9):
            break
        jacobian = np.column_stack([-dx / distance, -dy / distance, -np.ones_like(dx)])
        step, *_ = np.linalg.lstsq(jacobian, -(distance - radius), rcond=None)
        center = center + step[:2]
        radius += float(step[2])
        if np.max(np.abs(step)) < 1e-10:
            break
        if not (np.all(np.isfinite(center)) and math.isfinite(radius) and radius > 1e-9):
            return None
    return center, radius


def _radial_residuals(points, center, radius):
    return np.hypot(points[:, 0] - center[0], points[:, 1] - center[1]) - radius


def _circle_candidates(points):
    """Starting circles from all triples (small sets) or a seeded sample (large sets)."""
    start = _algebraic_circle(points)
    if start is not None:
        yield start
    count = len(points)
    if count <= EXHAUSTIVE_MARKS:
        triples = itertools.combinations(range(count), 3)
    else:
        rng = np.random.default_rng(CANDIDATE_SEED)
        triples = (
            tuple(sorted(int(index) for index in rng.choice(count, 3, replace=False)))
            for _ in range(CANDIDATE_TRIPLES)
        )
    for i, j, k in triples:
        circle = _circle_through(points[i], points[j], points[k])
        if circle is not None:
            yield circle


def _consensus_circle(points):
    """Circle supported by the largest group of marks, rejecting isolated mis-clicks."""
    best = None
    tied = []
    for center, radius in _circle_candidates(points):
        residual = np.abs(_radial_residuals(points, center, radius))
        used = residual <= INLIER_TOLERANCE_PX
        count = int(np.count_nonzero(used))
        if count < 3:
            continue
        rms = float(np.sqrt(np.mean(residual[used] ** 2)))
        if best is not None and count < best[0]:
            continue
        if best is not None and count == best[0]:
            tied.append((center, radius))
        if best is None or count > best[0]:
            tied = []
            best = (count, -rms, center, radius, used)
    if best is None:
        return None, []
    _, _, center, radius, used = best
    for _ in range(6):
        refined = _refine_circle(points[used], center, radius)
        if refined is None:
            break
        center, radius = refined
        updated = np.abs(_radial_residuals(points, center, radius)) <= INLIER_TOLERANCE_PX
        if np.count_nonzero(updated) < 3:
            break
        converged = np.array_equal(updated, used)
        used = updated
        if converged:
            break
    refined = _refine_circle(points[used], center, radius)
    if refined is None:
        return None, []
    center, radius = refined
    # An equally supported circle built from a different set of marks is a rival ring.
    rivals = [
        (candidate, candidate_radius)
        for candidate, candidate_radius in tied
        if not np.array_equal(
            np.abs(_radial_residuals(points, candidate, candidate_radius)) <= INLIER_TOLERANCE_PX,
            used,
        )
    ]
    return (center, radius, used), rivals


def _center_sigma(points, center):
    """One-sigma center uncertainty for the given mark geometry, in detector pixels."""
    dx = points[:, 0] - center[0]
    dy = points[:, 1] - center[1]
    distance = np.hypot(dx, dy)
    if np.any(distance < 1e-9):
        return math.inf
    jacobian = np.column_stack([dx / distance, dy / distance, -np.ones(len(points))])
    covariance = MARK_SIGMA_PX ** 2 * np.linalg.pinv(jacobian.T @ jacobian)
    values = np.linalg.eigvalsh(covariance[:2, :2])
    return math.sqrt(max(float(np.max(values)), 0.0))


def _ambiguous(model, message):
    model.update({
        "status": "ambiguous", "message": message,
        "cx": None, "cy": None, "radius_px": None, "rms_px": None, "max_residual_px": None,
    })
    return model


def fit_circle(points, width, height):
    """Fit one reference ring from user marks, rejecting isolated mis-clicks."""
    p = np.asarray(ring_marks(points, width, height, minimum=3), dtype=np.float64)
    empty = {
        "cx": None, "cy": None, "radius_px": None, "rms_px": None,
        "max_residual_px": None, "span_deg": None, "count": len(p),
        "used_count": 0, "used": None, "residual_px": [None] * len(p), "rejected": [],
        "center_sigma_px": None, "center_sigma_limit_px": CENTER_SIGMA_LIMIT_PX,
    }
    consensus, rivals = _consensus_circle(p)
    if consensus is None:
        return _ambiguous(
            empty,
            "这些标点没有落在同一条参考环上，无法确定环心。请检查误点，或沿参考环补选分散的标点。",
        )
    center, radius, used = consensus
    if rivals:
        return _ambiguous(
            {**empty, "count": len(p)},
            "这些标点可以分成几组、各自拟合成不同的圆，无法确定哪一条是参考环。"
            "请清空标点后只沿同一条参考环布点；现有环心保持不变。",
        )
    residual = _radial_residuals(p, center, radius)
    used_residual = residual[used]
    used_count = int(np.count_nonzero(used))
    span = angular_span(p[used, 0], p[used, 1], center[0], center[1])
    sigma = _center_sigma(p[used], center)
    low_x, high_x, low_y, high_y = xy_bounds(width, height, max(width, height))
    model = {
        **empty,
        "cx": float(center[0]),
        "cy": float(center[1]),
        "radius_px": float(radius),
        "rms_px": float(np.sqrt(np.mean(used_residual ** 2))),
        "max_residual_px": float(np.max(np.abs(used_residual))),
        "span_deg": span,
        "used_count": used_count,
        "used": used.tolist(),
        "residual_px": residual.tolist(),
        "rejected": [
            {"index": index, "x": float(p[index, 0]), "y": float(p[index, 1]),
             "residual_px": float(residual[index])}
            for index in range(len(p)) if not used[index]
        ],
        "center_sigma_px": sigma,
    }
    if 2 * used_count <= len(p):
        return _ambiguous(
            model,
            f"多数组标点不在同一条参考环上（仅采用 {used_count}/{len(p)} 点）。"
            "请检查误点，或重新沿参考环布点。",
        )
    if (
        not 2 < radius <= 2 * max(width, height)
        or not low_x <= center[0] <= high_x or not low_y <= center[1] <= high_y
    ):
        return _ambiguous(model, "拟合半径或环心位置不合理，标点可能混入了其他参考环或斑点。请补选分散的标点。")
    if span < MIN_FIT_SPAN or sigma > CENTER_SIGMA_LIMIT_PX:
        spread = f"（约 ±{min(sigma, 999.0):.1f} px）" if math.isfinite(sigma) else ""
        return _ambiguous(
            model,
            f"采用 {used_count} 个标点只覆盖约 {span:.0f}°，环心位置还不唯一{spread}。"
            "请沿同一条参考环补选更分散的标点；现有环心保持不变。",
        )
    model.update({"status": "ok", "message": ""})
    return model


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
