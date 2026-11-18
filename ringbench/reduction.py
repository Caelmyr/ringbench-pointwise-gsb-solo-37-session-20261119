"""Rebin finite detector pixel centers into radial and azimuthal output cells."""
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths

from .detector_io import csv_bytes, numbers_for_json
from .errors import WorkbenchError
from .geometry import AXIS_LABELS

BLOCK_ROWS = 128
MAX_ANGULAR_SPAN = 360.0


@dataclass(frozen=True)
class Axis:
    edges: np.ndarray
    unit: str
    label: str

    @property
    def centers(self):
        return (self.edges[:-1] + self.edges[1:]) / 2

    @property
    def cells(self):
        return self.edges.size - 1

    @property
    def low(self):
        return self.edges[0]

    @property
    def high(self):
        return self.edges[-1]

    def index(self, data):
        values = (data - self.low) * (self.cells / (self.high - self.low))
        return np.clip(np.floor(values).astype(np.intp), 0, self.cells - 1)

    def inside(self, data):
        return (data >= self.low) & (data <= self.high)

    def document(self):
        return {
            "label": self.label,
            "unit": self.unit,
            "edges": self.edges.tolist(),
            "centers": self.centers.tolist(),
        }


def radial_axis(shape, geometry, unit, count, interval):
    if interval is None:
        far = geometry.radius_limit(shape[1], shape[0])
        low, high = 0, float(geometry.radial(far, unit))
    else:
        low, high = interval
        if float(geometry.radius_from_unit(high, unit)) <= 0:
            raise WorkbenchError("径向范围没有落在可表达的探测器半径。")
    return Axis(np.linspace(low, high, count + 1), unit, AXIS_LABELS[unit])


def angular_axis(arc, count):
    return Axis(np.linspace(arc.start, arc.start + arc.span, count + 1), "deg", "方位角 (°)")


@dataclass(frozen=True)
class Moments:
    counts: np.ndarray
    eligible: np.ndarray
    total: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    @property
    def fraction(self):
        return np.divide(
            self.counts,
            self.eligible,
            out=np.full(self.counts.shape, np.nan, dtype=np.float64),
            where=self.eligible > 0,
        )

    def summary(self):
        valid = self.counts > 0
        return {
            "included_pixels": int(self.counts.sum()),
            "geometric_pixels": int(self.eligible.sum()),
            "nonempty_cells": int(valid.sum()),
            "total_cells": int(valid.size),
            "masked_cells": int(np.sum((self.eligible > 0) & ~valid)),
            "off_detector_cells": int(np.sum(self.eligible == 0)),
        }

    def vector_document(self):
        return {
            "count": self.counts.tolist(),
            "eligible": self.eligible.tolist(),
            "sum": numbers_for_json(self.total),
            "mean": numbers_for_json(self.mean),
            "std": numbers_for_json(self.std),
            "min": numbers_for_json(self.minimum),
            "max": numbers_for_json(self.maximum),
            "fraction": numbers_for_json(self.fraction),
        }


class Accumulator:
    """Separate geometric support, finite unmasked support, and value moments."""

    def __init__(self, cells):
        self.cells = cells
        self.eligible = np.zeros(cells, dtype=np.int64)
        self.count = np.zeros(cells, dtype=np.int64)
        self.total = np.zeros(cells, dtype=np.float64)
        self.square = np.zeros(cells, dtype=np.float64)
        self.minimum = np.full(cells, np.inf, dtype=np.float64)
        self.maximum = np.full(cells, -np.inf, dtype=np.float64)

    def add(self, index, data, valid):
        self.eligible += np.bincount(index, minlength=self.cells)
        ix = index[valid]
        if ix.size == 0:
            return
        v = data[valid].astype(np.float64, copy=False)
        self.count += np.bincount(ix, minlength=self.cells)
        self.total += np.bincount(ix, weights=v, minlength=self.cells)
        self.square += np.bincount(ix, weights=v * v, minlength=self.cells)
        np.minimum.at(self.minimum, ix, v)
        np.maximum.at(self.maximum, ix, v)

    def finish(self, shape):
        exists = self.count > 0
        mean = np.divide(
            self.total,
            self.count,
            out=np.full(self.cells, np.nan, dtype=np.float64),
            where=exists,
        )
        mean_square = np.divide(
            self.square, self.count,
            out=np.full(self.cells, np.nan, dtype=np.float64),
            where=exists,
        )
        std = np.sqrt(np.maximum(0.0, mean_square - mean * mean))
        result = (
            self.count, self.eligible, np.where(exists, self.total, np.nan), mean, std,
            np.where(exists, self.minimum, np.nan), np.where(exists, self.maximum, np.nan),
        )
        return Moments(*(array.reshape(shape) for array in result))


@dataclass(frozen=True)
class Curve:
    axis: Axis
    moments: Moments
    kind: str
    parameters: dict

    def document(self, peak_parameters=None):
        obj = {
            "kind": self.kind,
            "axis": self.axis.document(),
            "values": self.moments.vector_document(),
            "summary": self.moments.summary(),
            "parameters": self.parameters,
        }
        if self.kind == "profile":
            obj["peaks"] = detect_profile_peaks(self, **(peak_parameters or {}))
        return obj

    def csv(self):
        moments = self.moments
        rows = zip(
            self.axis.edges[:-1],
            self.axis.edges[1:],
            self.axis.centers,
            moments.counts,
            moments.eligible,
            numbers_for_json(moments.mean),
            numbers_for_json(moments.total),
            numbers_for_json(moments.std),
            numbers_for_json(moments.minimum),
            numbers_for_json(moments.maximum),
            numbers_for_json(moments.fraction),
        )
        return csv_bytes(
            [
                f"start_{self.axis.unit}", f"end_{self.axis.unit}", f"center_{self.axis.unit}",
                "valid_pixels", "candidate_pixels", "mean_count", "sum_count",
                "std_count", "min_count", "max_count", "usable_fraction",
            ],
            rows,
        )


@dataclass(frozen=True)
class Cake:
    radial: Axis
    angular: Axis
    moments: Moments

    def document(self):
        rows = []
        support = self.moments.counts
        mean = self.moments.mean
        for index in range(self.angular.cells):
            rows.append({
                "intensity": numbers_for_json(mean[index]),
                "valid_pixels": support[index].tolist(),
                "candidate_pixels": self.moments.eligible[index].tolist(),
            })
        return {
            "radial": self.radial.document(),
            "angular": self.angular.document(),
            "rows": rows,
            "summary": self.moments.summary(),
        }

    def csv(self):
        def rows():
            radii = self.radial.centers
            angles = self.angular.centers
            for row, az in enumerate(angles):
                for col, radius in enumerate(radii):
                    intensity = self.moments.mean[row, col]
                    yield (
                        radius, az, int(self.moments.counts[row, col]),
                        int(self.moments.eligible[row, col]),
                        float(intensity) if np.isfinite(intensity) else None,
                    )
        return csv_bytes(
            [f"radius_{self.radial.unit}", "azimuth_deg", "valid_pixels", "candidate_pixels", "mean_count"],
            rows(),
        )


def _regions(data, geometry, arc):
    for top in range(0, data.shape[0], BLOCK_ROWS):
        stop = min(top + BLOCK_ROWS, data.shape[0])
        radius, azimuth = geometry.detector_grid(data.shape, top, stop)
        yield slice(top, stop), radius, arc.relative(azimuth)


def integrate_profile(data, valid, geometry, spec):
    axis = radial_axis(data.shape, geometry, spec.unit, spec.bins, spec.radial)
    acc = Accumulator(axis.cells)
    arc = spec.arc
    for region, radius, azimuth in _regions(data, geometry, arc):
        coordinate = geometry.radial(radius, spec.unit)
        inside = (azimuth < arc.span) & axis.inside(coordinate)
        index = axis.index(coordinate[inside])
        acc.add(index, data[region][inside], valid[region][inside])
    return Curve(
        axis, acc.finish((axis.cells,)), "profile",
        {"arc": arc.documents(), "radial_unit": spec.unit, "assignment": "pixel_center"},
    )


def integrate_azimuth(data, valid, geometry, spec):
    arc = spec.arc
    axis = angular_axis(arc, spec.bins)
    acc = Accumulator(axis.cells)
    for region, radius, azimuth in _regions(data, geometry, arc):
        inside = (radius >= spec.inner_px) & (radius < spec.outer_px) & (azimuth < arc.span)
        index = axis.index(azimuth[inside] + arc.start)
        acc.add(index, data[region][inside], valid[region][inside])
    return Curve(
        axis, acc.finish((axis.cells,)), "azimuth",
        {"arc": arc.documents(), "annulus_px": [spec.inner_px, spec.outer_px], "assignment": "pixel_center"},
    )


def integrate_cake(data, valid, geometry, spec):
    arc = spec.arc
    r_axis = radial_axis(data.shape, geometry, spec.unit, spec.radial_bins, spec.radial)
    a_axis = angular_axis(arc, spec.angular_bins)
    acc = Accumulator(r_axis.cells * a_axis.cells)
    for region, radius, azimuth in _regions(data, geometry, arc):
        vals = geometry.radial(radius, spec.unit)
        inside = r_axis.inside(vals) & (azimuth < arc.span)
        values = a_axis.index(azimuth[inside] + arc.start) * r_axis.cells + r_axis.index(vals[inside])
        acc.add(values, data[region][inside], valid[region][inside])
    return Cake(r_axis, a_axis, acc.finish((a_axis.cells, r_axis.cells)))


def _runs(valid):
    switches = np.diff(np.r_[False, valid, False].astype(np.int8))
    begins = np.flatnonzero(switches == 1)
    ends = np.flatnonzero(switches == -1)
    return zip(begins, ends)


def detect_profile_peaks(curve, prominence=None, maximum=18):
    means = curve.moments.mean
    valid = np.isfinite(means)
    if np.count_nonzero(valid) < 3:
        return []
    if prominence is None:
        low, high = np.percentile(means[valid], [25, 95])
        prominence = max(np.finfo(float).eps, float(high - low) * 0.14)
    floor = max(float(prominence), np.finfo(float).eps)
    centers = curve.axis.centers
    candidates = []
    for begin, end in _runs(valid):
        if end - begin < 3:
            continue
        section = means[begin:end]
        ix, properties = find_peaks(section, prominence=floor)
        if len(ix) == 0:
            continue
        _, heights, left, right = peak_widths(section, ix, rel_height=0.5)
        x_indices = np.arange(len(means))
        widths = np.interp(right + begin, x_indices, centers) - np.interp(left + begin, x_indices, centers)
        for j, index in enumerate(ix + begin):
            candidates.append({
                "cell": int(index),
                "coordinate": float(centers[index]),
                "mean": float(means[index]),
                "prominence": float(properties["prominences"][j]),
                "width_at_half_prominence": float(widths[j]),
                "evaluation_level": float(heights[j]),
                "valid_pixels": int(curve.moments.counts[index]),
            })
    candidates.sort(key=lambda item: (-item["prominence"], item["coordinate"]))
    return sorted(candidates[:maximum], key=lambda item: item["coordinate"])


def probe_cake(cake, radius_index, angle_index):
    if radius_index < 0 or radius_index >= cake.radial.cells or angle_index < 0 or angle_index >= cake.angular.cells:
        raise WorkbenchError("极坐标取样点不在输出范围内。")
    m = cake.moments
    val = m.mean[angle_index, radius_index]
    return {
        "radial": [
            float(cake.radial.edges[radius_index]),
            float(cake.radial.edges[radius_index + 1]),
        ],
        "angle": [
            float(cake.angular.edges[angle_index]),
            float(cake.angular.edges[angle_index + 1]),
        ],
        "unit": cake.radial.unit,
        "mean": float(val) if np.isfinite(val) else None,
        "valid_pixels": int(m.counts[angle_index, radius_index]),
        "candidate_pixels": int(m.eligible[angle_index, radius_index]),
    }
