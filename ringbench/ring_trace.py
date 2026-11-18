"""Trace an expected diffraction ring without interpolating across detector gaps."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import find_peaks

from .detector_io import csv_bytes
from .errors import WorkbenchError
from .geometry import Arc
from .reduction import Accumulator, Axis
from .values import integer, mapping, number

MIN_RESOLUTION = 24
MAX_RESOLUTION = 360


@dataclass(frozen=True)
class TraceSettings:
    radius: float
    tolerance: float
    angular_bins: int
    minimum_pixels: int
    minimum_prominence: float
    arc: Arc

    @classmethod
    def read(cls, value):
        v = mapping(value, "环线误差参数")
        radius = number(v.get("radius"), "目标半径 px", 2, 6000)
        tol = number(v.get("tolerance", 8), "径向搜索半宽 px", 1, 100)
        if tol >= radius:
            raise WorkbenchError("环线搜索区域必须离开直束中心。")
        return cls(
            radius,
            tol,
            integer(v.get("angular_bins", 72), "环线扇区", MIN_RESOLUTION, MAX_RESOLUTION),
            integer(v.get("minimum_pixels", 2), "每格像素下限", 1, 64),
            number(v.get("minimum_prominence", 12), "环峰突出度下限", 0, 1e16),
            Arc.read(v.get("arc_start", 0), v.get("arc_span", 360)),
        )


@dataclass(frozen=True)
class TracePoint:
    azimuth: float
    radius: float | None
    x: float | None
    y: float | None
    height: float | None
    prominence: float | None
    pixels: int
    reason: str

    def document(self, expected):
        return {
            "angle_deg": self.azimuth,
            "radius_px": self.radius,
            "deviation_px": None if self.radius is None else self.radius - expected,
            "x_px": self.x,
            "y_px": self.y,
            "height_counts": self.height,
            "prominence_counts": self.prominence,
            "peak_pixels": self.pixels,
            "status": self.reason,
        }


@dataclass(frozen=True)
class RingTrace:
    settings: TraceSettings
    points: list

    def summary(self):
        values = np.asarray([
            point.radius - self.settings.radius for point in self.points if point.radius is not None
        ])
        if values.size == 0:
            center, rms, amplitude, lower, upper = None, None, None, None, None
        else:
            center = float(np.mean(values))
            rms = float(np.sqrt(np.mean(values**2)))
            amplitude = float(np.ptp(values))
            lower, upper = (float(v) for v in np.percentile(values, [5, 95]))
        return {
            "expected_radius_px": self.settings.radius,
            "search_tolerance_px": self.settings.tolerance,
            "observed_sectors": len(values),
            "candidate_sectors": len(self.points),
            "mean_deviation_px": center,
            "deviation_rms_px": rms,
            "deviation_span_px": amplitude,
            "percentile_5_px": lower,
            "percentile_95_px": upper,
        }

    def document(self):
        rows = [point.document(self.settings.radius) for point in self.points]
        return {
            "summary": self.summary(),
            "points": rows,
            "series": {
                "kind": "ring_trace",
                "axis": {"label": "方位角 (°)", "unit": "deg", "centers": [p.azimuth for p in self.points]},
                "values": {
                    "mean": [p["deviation_px"] for p in rows],
                    "count": [p["peak_pixels"] for p in rows],
                },
            },
        }

    def csv(self):
        fields = [
            "angle_deg", "radius_px", "deviation_px", "x_px", "y_px",
            "height_counts", "prominence_counts", "peak_pixels", "status",
        ]
        return csv_bytes(
            fields,
            (
                tuple(doc[field] for field in fields)
                for doc in (p.document(self.settings.radius) for p in self.points)
            ),
        )


def _curved_peak(x0, step, left, middle, right):
    second = left - 2 * middle + right
    if second >= -1e-9:
        return x0, middle
    offset = min(0.5, max(-0.5, 0.5 * (left - right) / second))
    peak = middle + 0.5 * (right - left) * offset + 0.5 * second * offset**2
    return x0 + step * offset, peak


def _pick_sector(center, vals, counts, theta, geometry, settings):
    present = counts >= settings.minimum_pixels
    if np.count_nonzero(present) < 3:
        return TracePoint(theta, None, None, None, None, None, 0, "insufficient_pixels")
    limits = np.flatnonzero(np.diff(np.r_[False, present, False]))
    closest = None
    for begin, end in limits.reshape(-1, 2):
        if end - begin < 3:
            continue
        section = vals[begin:end]
        indices, metrics = find_peaks(section, prominence=max(1e-10, settings.minimum_prominence))
        for index, height in zip(indices + begin, metrics["prominences"]):
            peak, peak_counts = _curved_peak(
                center[index], center[1] - center[0], vals[index-1], vals[index], vals[index+1],
            )
            dist = abs(peak - settings.radius)
            candidate = (dist, -height, int(index), peak, peak_counts)
            if closest is None or candidate < closest:
                closest = candidate
    if closest is None:
        return TracePoint(theta, None, None, None, None, None, 0, "no_eligible_peak")
    _, reverse_prominence, index, peak, height = closest
    x, y = geometry.position(peak, theta)
    return TracePoint(
        theta, float(peak), float(x), float(y), float(height), float(-reverse_prominence),
        int(counts[index]), "observed",
    )


def trace_ring(data, valid, geometry, spec):
    lo = spec.radius - spec.tolerance
    hi = spec.radius + spec.tolerance
    r_bins = int(max(9, math.ceil(hi - lo)))
    arc = spec.arc
    r_axis = Axis(np.linspace(lo, hi, r_bins + 1), "px", "半径 (px)")
    a_axis = Axis(
        np.linspace(arc.start, arc.start + arc.span, spec.angular_bins + 1),
        "deg", "方位角 (°)",
    )
    cells = Accumulator(r_bins * spec.angular_bins)
    for start in range(0, data.shape[0], 128):
        stop = min(start + 128, data.shape[0])
        r, a = geometry.detector_grid(data.shape, start, stop)
        relative = arc.relative(a)
        eligible = r_axis.inside(r) & (relative < arc.span)
        ri = r_axis.index(r[eligible])
        ai = a_axis.index(relative[eligible] + arc.start)
        cells.add(ai * r_bins + ri, data[start:stop][eligible], valid[start:stop][eligible])
    m = cells.finish((spec.angular_bins, r_bins))
    return RingTrace(
        spec,
        [
            _pick_sector(r_axis.centers, m.mean[i], m.counts[i], angle, geometry, spec)
            for i, angle in enumerate(a_axis.centers)
        ],
    )
