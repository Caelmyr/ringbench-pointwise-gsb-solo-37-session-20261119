"""Rebin finite detector pixel centers into radial and azimuthal output cells."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths

from .detector_io import csv_bytes, numbers_for_json
from .errors import WorkbenchError
from .geometry import AXIS_LABELS

BLOCK_ROWS = 128
MAX_ANGULAR_SPAN = 360.0
HALF_HEIGHT_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))
SHOULDER_SIGMA_FLOOR = 0.25
SHOULDER_MIN_RATIO = 0.10
SHOULDER_MIN_NOISE = 3.0
SHOULDER_MIN_SIGNIFICANCE = 8.0
SHOULDER_SEPARATION = 1.5
SHOULDER_MERGE_MARGIN = 25.0
SHOULDER_MIN_BINS = 8
SHOULDER_EDGE_SAMPLES = 2
SHOULDER_NEIGHBOR_RATIO = 0.25


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
            detected = detect_profile_peaks(self, **(peak_parameters or {}))
            obj["peaks"] = resolve_profile_peaks(self, detected)
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


def _gauss(t, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def _one_peak(p, t, background):
    return background(t) + _gauss(t, p[0], p[1], p[2])


def _one_peak_jacobian(p, t, background):
    g = np.exp(-0.5 * ((t - p[1]) / p[2]) ** 2)
    return np.column_stack([
        g,
        p[0] * g * (t - p[1]) / p[2] ** 2,
        p[0] * g * (t - p[1]) ** 2 / p[2] ** 3,
    ])


def _skewed_peak(p, t, background):
    sigma = np.where(t < p[1], p[2], p[3])
    return background(t) + _gauss(t, p[0], p[1], sigma)


def _skewed_peak_jacobian(p, t, background):
    left = t < p[1]
    sigma = np.where(left, p[2], p[3])
    g = np.exp(-0.5 * ((t - p[1]) / sigma) ** 2)
    tail = p[0] * g * (t - p[1]) ** 2
    return np.column_stack([
        g,
        p[0] * g * (t - p[1]) / sigma ** 2,
        np.where(left, tail / p[2] ** 3, 0.0),
        np.where(left, 0.0, tail / p[3] ** 3),
    ])


def _two_peaks(p, t, background):
    return background(t) + _gauss(t, p[0], p[1], p[2]) + _gauss(t, p[3], p[4], p[5])


def _two_peaks_jacobian(p, t, background):
    first = np.exp(-0.5 * ((t - p[1]) / p[2]) ** 2)
    second = np.exp(-0.5 * ((t - p[4]) / p[5]) ** 2)
    return np.column_stack([
        first,
        p[0] * first * (t - p[1]) / p[2] ** 2,
        p[0] * first * (t - p[1]) ** 2 / p[2] ** 3,
        second,
        p[3] * second * (t - p[4]) / p[5] ** 2,
        p[3] * second * (t - p[4]) ** 2 / p[5] ** 3,
    ])


def _bic(residuals, parameters):
    size = residuals.size
    chi = float(np.sum(residuals ** 2))
    return size * math.log(max(chi, 1e-12) / size) + parameters * math.log(size)


def _solve(function, jacobian, start, lower, upper, t, values, background):
    return least_squares(
        lambda p: function(p, t, background) - values,
        start,
        jac=lambda p: jacobian(p, t, background),
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=200,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
    )


def _fit_components(x, values):
    origin = float(x[0])
    scale = float(x[-1] - x[0])
    t = (x - origin) / scale
    step = float(np.median(np.diff(x))) / scale
    left_level = float(np.median(values[:SHOULDER_EDGE_SAMPLES]))
    right_level = float(np.median(values[-SHOULDER_EDGE_SAMPLES:]))

    def background(positions):
        return left_level + (right_level - left_level) * positions

    amplitude = max(float(np.max(values)) - min(left_level, right_level), 1e-9)
    low_sigma = max(SHOULDER_SIGMA_FLOOR * step, 1e-3)
    sigma = max(step, 0.02)
    center = float(t[int(np.argmax(values))])
    lower = [0.0, 0.0, low_sigma]
    upper = [5.0 * amplitude, 1.0, 1.0]
    single = _solve(_one_peak, _one_peak_jacobian, [amplitude, center, sigma],
                    lower, upper, t, values, background)
    skewed = _solve(_skewed_peak, _skewed_peak_jacobian,
                    [amplitude, center, sigma, sigma],
                    lower + [low_sigma], upper + [1.0], t, values, background)
    simple = min((single, _bic(single.fun, 3)), (skewed, _bic(skewed.fun, 4)),
                 key=lambda item: item[1])[0]
    residual = np.convolve(-simple.fun, np.ones(3) / 3.0, mode="same")
    order = int(np.argmax(residual))
    peaks = []
    for index in (order, int(np.argmax(residual[:order + 1])),
                  order + int(np.argmax(residual[order:]))):
        start = [amplitude, center, sigma,
                 max(float(residual[index]), 0.05 * amplitude),
                 float(np.clip(t[index], low_sigma, 1.0 - low_sigma)), sigma]
        peaks.append(_solve(_two_peaks, _two_peaks_jacobian, start,
                            lower + [0.0, 0.0, low_sigma],
                            upper + [5.0 * amplitude, 1.0, 1.0], t, values, background))
    double = min(peaks, key=lambda result: float(np.sum(result.fun ** 2)))
    return {
        "origin": origin,
        "scale": scale,
        "single": single,
        "skewed": skewed,
        "double": double,
        "single_bic": _bic(single.fun, 3),
        "skewed_bic": _bic(skewed.fun, 4),
        "double_bic": _bic(double.fun, 6),
    }


def _unresolved(candidate, status, reason, **detail):
    return {
        **candidate,
        "fwhm": None,
        "components": 1,
        "prominence_source": "local_maximum",
        "width_source": "half_prominence",
        "resolution": {"status": status, "reason": reason, **detail},
    }


def _component_evidence(fit, step, values):
    p = fit["double"].x
    first_center = fit["origin"] + p[1] * fit["scale"]
    second_center = fit["origin"] + p[4] * fit["scale"]
    first_width = HALF_HEIGHT_FWHM * p[2] * fit["scale"]
    second_width = HALF_HEIGHT_FWHM * p[5] * fit["scale"]
    amplitudes = (float(p[0]), float(p[3]))
    separation = abs(second_center - first_center)
    noise = math.sqrt(float(np.sum(fit["single"].fun ** 2)) / max(1, values.size - 3))
    jacobian = fit["double"].jac
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * (
        float(np.sum(fit["double"].fun ** 2)) / max(1, values.size - 6)
    )
    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    significance = min(amplitudes[0] / max(float(errors[0]), 1e-12),
                       amplitudes[1] / max(float(errors[3]), 1e-12))
    strength = float(np.percentile(values, 95) - np.percentile(values, 5))
    margin = step / fit["scale"]
    positions = (p[1], p[4])
    return {
        "centers": (first_center, second_center),
        "widths": (first_width, second_width),
        "amplitudes": amplitudes,
        "separation": separation,
        "weaker": min(amplitudes),
        "threshold": max(0.02 * strength, SHOULDER_MIN_NOISE * noise),
        "significance": float(significance),
        "interior": (min(positions) >= margin and max(positions) <= 1.0 - margin),
        "merge_margin": SHOULDER_MERGE_MARGIN,
        "minimum_separation": max(SHOULDER_SEPARATION * step, 0.5 * min(first_width, second_width)),
        "ratio": min(amplitudes) / max(max(amplitudes), 1e-12),
    }


def resolve_profile_peaks(curve, peaks):
    """Split one detected maximum into two components when the profile supports it."""
    means = curve.moments.mean
    centers = curve.axis.centers
    cells = centers.size
    if cells < 3:
        return list(peaks)
    step = float(np.median(np.diff(centers)))
    positions = [peak["cell"] for peak in peaks]
    strengths = [float(peak["prominence"]) for peak in peaks]
    limits = []
    for index in range(len(peaks)):
        floor = SHOULDER_NEIGHBOR_RATIO * strengths[index]
        previous = next(
            (positions[i] for i in range(index - 1, -1, -1) if strengths[i] >= floor), None,
        )
        following = next(
            (positions[i] for i in range(index + 1, len(peaks)) if strengths[i] >= floor), None,
        )
        limits.append((previous, following))
    output = []
    covered = []
    for index, peak in enumerate(peaks):
        cell = peak["cell"]
        if any(low <= cell <= high for low, high in covered):
            continue
        width = max(float(peak["width_at_half_prominence"]), step)
        reach = int(math.ceil(max(5.0 * step, 1.5 * width) / step))
        core = int(math.ceil(max(2.0 * step, width) / step))
        left = max(0, cell - reach)
        right = min(cells - 1, cell + reach)
        previous, following = limits[index]
        if previous is not None:
            left = max(left, (previous + cell) // 2)
        if following is not None:
            right = min(right, (cell + following) // 2)
        separated = [
            other for other in positions if other != cell and abs(other - cell) <= core
        ]
        if separated:
            output.append(_unresolved(peak, "single", "separate_maxima",
                                      note="相邻极大已作为独立峰列出，不再重复分峰。"))
            continue
        core_window = slice(max(left, cell - core), min(right, cell + core) + 1)
        missing = int(np.count_nonzero(~np.isfinite(means[core_window])))
        if missing:
            output.append(_unresolved(peak, "insufficient_evidence", "missing_bins",
                                      missing_bins=missing,
                                      note="证据不足：肩部区间缺少有效分箱，保留现有单峰读数。"))
            continue
        while left < cell and not np.isfinite(means[left]):
            left += 1
        while right > cell and not np.isfinite(means[right]):
            right -= 1
        gaps = int(np.count_nonzero(~np.isfinite(means[left:right + 1])))
        if gaps:
            output.append(_unresolved(peak, "insufficient_evidence", "missing_bins",
                                      missing_bins=gaps,
                                      note="证据不足：肩部区间缺少有效分箱，保留现有单峰读数。"))
            continue
        if right - left + 1 < SHOULDER_MIN_BINS:
            output.append(_unresolved(peak, "insufficient_evidence", "too_few_bins",
                                      bins=right - left + 1,
                                      note="证据不足：峰区有效分箱太少，保留现有单峰读数。"))
            continue
        if left <= 0 or right >= cells - 1:
            output.append(_unresolved(peak, "insufficient_evidence", "partial_profile",
                                      note="证据不足：峰侧翼落在有效分箱范围外，保留现有单峰读数。"))
            continue
        if width < 1.5 * step:
            output.append(_unresolved(peak, "single", "narrow_maximum",
                                      note="峰宽接近单个分箱，未做肩部分峰。"))
            continue
        x = centers[left:right + 1]
        values = means[left:right + 1]
        if x.size < 4 or not np.isfinite(values).all() or x[-1] <= x[0]:
            output.append(_unresolved(peak, "insufficient_evidence", "too_few_bins",
                                      bins=int(x.size),
                                      note="证据不足：峰区有效分箱太少，保留现有单峰读数。"))
            continue
        fit = _fit_components(x, values)
        evidence = _component_evidence(fit, step, values)
        best_single = min(fit["single_bic"], fit["skewed_bic"])
        if (
            fit["double_bic"] <= best_single - evidence["merge_margin"]
            and evidence["separation"] >= evidence["minimum_separation"]
            and evidence["ratio"] >= SHOULDER_MIN_RATIO
            and evidence["weaker"] >= evidence["threshold"]
            and evidence["significance"] >= SHOULDER_MIN_SIGNIFICANCE
            and evidence["interior"]
        ):
            first, second = evidence["centers"]
            pair = sorted(zip(evidence["centers"], evidence["widths"], evidence["amplitudes"]))
            placed = []
            for order, (coordinate, fwhm, amplitude) in enumerate(pair, 1):
                near = int(np.argmin(np.abs(centers - coordinate)))
                placed.append(near)
                output.append({
                    **peak,
                    "cell": near,
                    "coordinate": float(coordinate),
                    "mean": float(means[near]),
                    "prominence": float(amplitude),
                    "width_at_half_prominence": float(fwhm),
                    "fwhm": float(fwhm),
                    "components": 2,
                    "component": order,
                    "prominence_source": "component_fit",
                    "width_source": "component_fit",
                    "resolution": {
                        "status": "resolved",
                        "component": order,
                        "separation": float(abs(second - first)),
                        "delta_bic": float(fit["double_bic"] - best_single),
                        "note": f"拟合分成两个峰，间距 {abs(second - first):.3g}。",
                    },
                })
            covered.append((min(placed), max(placed)))
            continue
        model = "asymmetric" if fit["skewed_bic"] < fit["single_bic"] else "gaussian"
        output.append({
            **_unresolved(peak, "single", "single_maximum"),
            "resolution": {
                "status": "single",
                "reason": "single_maximum",
                "model": model,
                "delta_bic": float(fit["double_bic"] - best_single),
                "note": "拟合不支持肩部独立成峰，保留单峰读数。",
            },
        })
    return sorted(output, key=lambda item: item["coordinate"])


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
