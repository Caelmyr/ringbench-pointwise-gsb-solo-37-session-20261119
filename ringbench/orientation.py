"""Fit interpretable angular harmonics to observed annular intensities."""
from dataclasses import dataclass
import math

import numpy as np

from .detector_io import csv_bytes, numbers_for_json
from .errors import WorkbenchError
from .values import integer, mapping, number, selection

ORDER_MAX = 8
CONDITION_MAX = 1e5


@dataclass(frozen=True)
class HarmonicSettings:
    order: int
    minimum_pixels: int
    minimum_fraction: float
    weights: str

    @classmethod
    def read(cls, value):
        v = mapping(value, "取向拟合参数")
        return cls(
            integer(v.get("order", 2), "谐波阶数", 1, ORDER_MAX),
            integer(v.get("minimum_pixels", 2), "最少有效像素", 1, 4096),
            number(v.get("minimum_fraction", 0.25), "最小可用比例", 0, 1),
            selection(v.get("weights", "equal"), ("equal", "pixels"), "方位拟合权重"),
        )


@dataclass(frozen=True)
class Harmonic:
    order: int
    cosine: float
    sine: float
    amplitude: float
    direction: float | None
    relative: float | None
    sigma_amplitude: float | None

    def document(self):
        period = 360 / self.order
        directions = [] if self.direction is None else [self.direction + i * period for i in range(self.order)]
        return {
            "order": self.order,
            "cosine": self.cosine,
            "sine": self.sine,
            "amplitude": self.amplitude,
            "direction_deg": self.direction,
            "period_deg": period,
            "directions_deg": directions,
            "relative_amplitude": self.relative,
            "amplitude_standard_error": self.sigma_amplitude,
        }


@dataclass(frozen=True)
class Orientation:
    curve: object
    used: np.ndarray
    prediction: np.ndarray
    error: np.ndarray
    coefficients: list
    mean: float
    stats: dict
    settings: HarmonicSettings

    @property
    def principal(self):
        return self.coefficients[self.settings.order - 1]

    def document(self):
        return {
            "kind": "orientation",
            "axis": self.curve.axis.document(),
            "values": {
                **self.curve.moments.vector_document(),
                "fitted": numbers_for_json(self.prediction),
                "residual": numbers_for_json(self.error),
                "fit_used": self.used.tolist(),
            },
            "equation": {"mean": self.mean, "components": [h.document() for h in self.coefficients]},
            "principal": self.principal.document(),
            "summary": self.stats,
        }

    def csv(self):
        return csv_bytes(
            ["azimuth_deg", "mean_count", "valid_pixels", "candidate_pixels",
             "included_in_fit", "fitted_count", "residual_count"],
            zip(
                self.curve.axis.centers,
                numbers_for_json(self.curve.moments.mean),
                self.curve.moments.counts,
                self.curve.moments.eligible,
                self.used,
                numbers_for_json(self.prediction),
                numbers_for_json(self.error),
            ),
        )


def _matrix(centers, order):
    radians = np.radians(centers)
    basis = [np.ones(len(radians), dtype=np.float64)]
    for i in range(1, order + 1):
        basis.extend([np.cos(i * radians), np.sin(i * radians)])
    return np.column_stack(basis)


def _circular_gap(degrees):
    values = np.sort(degrees % 360)
    enlarged = np.r_[values, values[0] + 360]
    return float(np.max(np.diff(enlarged)))


def _measure_harmonics(params, covariance):
    mean = float(params[0])
    n = (len(params) - 1) // 2
    result = []
    for i in range(1, n + 1):
        left, right = 2 * i - 1, 2 * i
        a, b = (float(v) for v in params[[left, right]])
        magnitude = math.hypot(a, b)
        if magnitude <= 1e-12:
            direction, uncertainty = None, None
        else:
            direction = (math.degrees(math.atan2(b, a)) / i) % (360 / i)
            if covariance is None:
                uncertainty = None
            else:
                diff = np.array([a, b]) / magnitude
                local = covariance[left:right + 1, left:right + 1]
                uncertainty = float(np.sqrt(max(0.0, diff @ local @ diff)))
        ratio = magnitude / abs(mean) if abs(mean) > 1e-12 else None
        result.append(Harmonic(i, a, b, magnitude, direction, ratio, uncertainty))
    return result


def fit_orientation(curve, settings):
    m = curve.moments
    centers = curve.axis.centers
    valid = (
        (m.counts >= settings.minimum_pixels) &
        (m.fraction >= settings.minimum_fraction) &
        np.isfinite(m.mean)
    )
    dimension = 1 + 2 * settings.order
    used = int(valid.sum())
    if used < dimension + 2:
        raise WorkbenchError("可用方位箱太少。请扩大角度范围、加宽环带或降低拟合阶数。")
    design = _matrix(centers, settings.order)
    weights = np.sqrt(m.counts[valid]) if settings.weights == "pixels" else np.ones(used)
    weights /= weights.mean()
    X = design[valid] * weights[:, None]
    y = m.mean[valid] * weights
    params, _, rank, singular = np.linalg.lstsq(X, y, rcond=None)
    if rank < dimension or singular[-1] < singular[0] / CONDITION_MAX:
        raise WorkbenchError("可用方位不足以区分这些取向项，请扩大覆盖角或降低拟合阶数。")
    prediction = design @ params
    residual = m.mean - prediction
    weighted = residual[valid] * weights
    y0 = float(np.average(m.mean[valid], weights=weights ** 2))
    rss = float(np.dot(weighted, weighted))
    null = (m.mean[valid] - y0) * weights
    tss = float(np.dot(null, null))
    variance = rss / (used - dimension)
    pinv = np.linalg.pinv(X)
    covariance = variance * (pinv @ pinv.T)
    return Orientation(
        curve, valid, prediction, residual,
        _measure_harmonics(params, covariance),
        float(params[0]),
        {
            "sample_bins": used,
            "total_bins": int(len(centers)),
            "largest_circular_gap_deg": _circular_gap(centers[valid]),
            "weighted_rms": math.sqrt(rss / float(np.dot(weights, weights))),
            "condition": float(singular[0] / singular[-1]),
            "r_squared": None if tss <= 0 else 1 - rss / tss,
        },
        settings,
    )
