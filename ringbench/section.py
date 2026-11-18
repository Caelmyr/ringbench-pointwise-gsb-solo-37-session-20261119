"""Sample image cross sections without creating counts in unobserved support."""
from dataclasses import dataclass
import math

import numpy as np

from .detector_io import csv_bytes, numbers_for_json
from .errors import WorkbenchError

MAX_LINE_SAMPLES = 8192
MAX_RECTANGLE_SAMPLES = 131_072


@dataclass(frozen=True)
class Sampling:
    values: np.ndarray
    coverage: np.ndarray
    on_detector: np.ndarray


def interpolate(detector, valid, xs, ys):
    """Bilinear sampling normalizes only finite, unmasked neighbor support."""
    h, w = detector.shape
    x_floor = np.floor(xs).astype(np.intp)
    y_floor = np.floor(ys).astype(np.intp)
    u, v = xs - x_floor, ys - y_floor
    expected = np.zeros(xs.shape, dtype=np.float64)
    accepted = np.zeros(xs.shape, dtype=np.float64)
    result = np.zeros(xs.shape, dtype=np.float64)
    for dx, dy, weight in (
        (0, 0, (1 - u) * (1 - v)),
        (1, 0, u * (1 - v)),
        (0, 1, (1 - u) * v),
        (1, 1, u * v),
    ):
        ix, iy = x_floor + dx, y_floor + dy
        on = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h) & (weight > 0)
        expected[on] += weight[on]
        ix_safe = ix.clip(0, w - 1)
        iy_safe = iy.clip(0, h - 1)
        support = on & valid[iy_safe, ix_safe]
        accepted[support] += weight[support]
        result[support] += weight[support] * detector[iy_safe[support], ix_safe[support]]
    values = np.divide(
        result, accepted, out=np.full(result.shape, np.nan), where=accepted > 1e-10,
    )
    return Sampling(values, accepted, expected)


@dataclass(frozen=True)
class CrossSection:
    x: np.ndarray
    y: np.ndarray
    distance: np.ndarray
    mean: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    support: np.ndarray
    aperture: np.ndarray
    half_width: float
    expected_aperture: int
    chosen_step: float

    def summary(self):
        return {
            "rows": int(self.x.size),
            "observed_rows": int(np.sum(self.support > 0)),
            "off_detector_rows": int(np.sum(self.aperture <= 0)),
            "length_px": float(self.distance[-1]),
            "half_width_px": self.half_width,
            "samples_across": self.expected_aperture,
            "step_px": self.chosen_step,
        }

    def document(self):
        return {
            "kind": "line",
            "axis": {"label": "沿线距离 (px)", "unit": "px", "centers": self.distance.tolist()},
            "values": {
                "x": self.x.tolist(),
                "y": self.y.tolist(),
                "mean": numbers_for_json(self.mean),
                "min": numbers_for_json(self.minimum),
                "max": numbers_for_json(self.maximum),
                "count": self.support.tolist(),
                "eligible": self.aperture.tolist(),
                "fraction": self.support.tolist(),
            },
            "summary": self.summary(),
        }

    def csv(self):
        return csv_bytes(
            [
                "distance_px", "x", "y", "mean_count", "min_count", "max_count",
                "usable_fraction", "on_detector_fraction",
            ],
            zip(
                self.distance, self.x, self.y, numbers_for_json(self.mean),
                numbers_for_json(self.minimum), numbers_for_json(self.maximum), self.support, self.aperture,
            ),
        )


def cross_section(data, valid, spec):
    a, b = np.array(spec.a), np.array(spec.b)
    delta = b - a
    length = float(np.linalg.norm(delta))
    divisions = max(1, math.ceil(length / spec.step))
    samples = divisions + 1
    if samples > MAX_LINE_SAMPLES:
        raise WorkbenchError("横截面采样过密，增加步距后再试。")
    if spec.half_width == 0:
        across = np.array([0.0])
    else:
        count = 2 * math.ceil(spec.half_width) + 1
        across = np.linspace(-spec.half_width, spec.half_width, count)
    if samples * across.size > MAX_RECTANGLE_SAMPLES:
        raise WorkbenchError("剖面宽度与纵向步距的组合产生过多采样点。")
    distance = np.linspace(0.0, length, samples)
    direction = delta / length
    normal = np.array([-direction[1], direction[0]])
    xy = a + distance[:, None] * direction
    positions = xy[:, None, :] + across[None, :, None] * normal
    v = interpolate(data, valid, positions[..., 0], positions[..., 1])
    weights = v.coverage.sum(axis=1)
    numerator = np.sum(np.where(np.isfinite(v.values), v.values * v.coverage, 0), axis=1)
    mean = np.divide(numerator, weights, out=np.full(samples, np.nan), where=weights > 1e-10)
    min_v = np.min(np.where(np.isfinite(v.values), v.values, np.inf), axis=1)
    max_v = np.max(np.where(np.isfinite(v.values), v.values, -np.inf), axis=1)
    return CrossSection(
        xy[:, 0], xy[:, 1], distance, mean,
        np.where(np.isfinite(min_v), min_v, np.nan),
        np.where(np.isfinite(max_v), max_v, np.nan),
        weights / across.size,
        v.on_detector.mean(axis=1),
        spec.half_width,
        across.size,
        length / divisions,
    )
