"""Manual measurement of a radial peak against chosen quiet side bands."""
from dataclasses import dataclass

import numpy as np

from .detector_io import csv_bytes, numbers_for_json
from .errors import WorkbenchError
from .values import mapping, ordered, selection

BASELINES = ("constant", "gradient")


@dataclass(frozen=True)
class MeasureSettings:
    peak: tuple
    left: tuple
    right: tuple
    baseline: str

    @classmethod
    def read(cls, value):
        v = mapping(value, "峰面积参数")
        peak = ordered(v.get("peak"), "径向峰窗口", 0)
        left = ordered(v.get("left"), "峰前背景窗口", 0)
        right = ordered(v.get("right"), "峰后背景窗口", 0)
        if left[1] > peak[0] or peak[1] > right[0]:
            raise WorkbenchError("背景窗口须分别位于峰窗口前后，窗口不能重叠。")
        return cls(peak, left, right, selection(v.get("baseline", "gradient"), BASELINES, "峰背景"))


@dataclass(frozen=True)
class AreaMeasurement:
    curve: object
    settings: MeasureSettings
    baseline: np.ndarray
    residual: np.ndarray
    area_weights: np.ndarray
    left_mean: float
    right_mean: float
    counts_area: float
    baseline_area: float
    positive_area: float
    net_area: float
    centroid: float | None
    maximum: int
    flags: list

    @property
    def covered(self):
        return float(np.sum(self.area_weights))

    def document(self):
        stated_width = self.settings.peak[1] - self.settings.peak[0]
        pos = self.curve.axis.centers[self.maximum]
        height = self.residual[self.maximum]
        return {
            "unit": self.curve.axis.unit,
            "mean_counts_area": self.counts_area,
            "background_area": self.baseline_area,
            "positive_area": self.positive_area,
            "net_area": self.net_area,
            "net_positive_centroid": self.centroid,
            "max_position": float(pos),
            "max_above_baseline": float(height),
            "left_background": self.left_mean,
            "right_background": self.right_mean,
            "available_width": self.covered,
            "requested_width": stated_width,
            "fraction_observed": self.covered / stated_width,
            "warnings": self.flags,
            "axis": self.curve.axis.document(),
            "values": {
                **self.curve.moments.vector_document(),
                "baseline": numbers_for_json(self.baseline),
                "net": numbers_for_json(self.residual),
                "integration_width": self.area_weights.tolist(),
            },
        }

    def csv(self):
        return csv_bytes(
            [
                f"center_{self.curve.axis.unit}", "candidate_pixels", "valid_pixels", "mean_count",
                "baseline_count", "net_count", "included_width",
            ],
            zip(
                self.curve.axis.centers,
                self.curve.moments.eligible,
                self.curve.moments.counts,
                numbers_for_json(self.curve.moments.mean),
                numbers_for_json(self.baseline),
                numbers_for_json(self.residual),
                self.area_weights,
            ),
        )


def _weights(x, bounds):
    return np.maximum(0, np.minimum(x[1:], bounds[1]) - np.maximum(x[:-1], bounds[0]))


def _moment(axis, data, band):
    weight = _weights(axis.edges, band)
    support = (weight > 0) & np.isfinite(data)
    if np.count_nonzero(support) < 2:
        raise WorkbenchError("每个背景区需要至少两个有计数的径向箱。")
    total = float(np.sum(weight[support]))
    x = float(np.dot(axis.centers[support], weight[support]) / total)
    y = float(np.dot(data[support], weight[support]) / total)
    return x, y, total


def measure_area(curve, settings):
    x = curve.axis
    y = curve.moments.mean
    if settings.left[0] < x.low or settings.right[1] > x.high:
        raise WorkbenchError("峰和背景窗口必须都在当前径向积分的横轴范围内。")
    left_x, left_y, left_width = _moment(x, y, settings.left)
    right_x, right_y, right_width = _moment(x, y, settings.right)
    widths = _weights(x.edges, settings.peak)
    usable = (widths > 0) & np.isfinite(y)
    widths[~usable] = 0
    if np.count_nonzero(usable) < 2:
        raise WorkbenchError("峰内有效径向箱太少，无法衡量峰面积。")
    if settings.baseline == "constant":
        middle = (left_width * left_y + right_width * right_y) / (left_width + right_width)
        baseline = np.full(len(y), middle)
    else:
        baseline = np.interp(x.centers, [left_x, right_x], [left_y, right_y])
    residual = y - baseline
    widths_used = widths[usable]
    gross = float(np.dot(y[usable], widths_used))
    background = float(np.dot(baseline[usable], widths_used))
    positive = np.maximum(residual[usable], 0) * widths_used
    evidence = float(positive.sum())
    part = float(widths.sum()) / (settings.peak[1] - settings.peak[0])
    max_index = int(np.flatnonzero(usable)[np.argmax(residual[usable])])
    warnings = []
    if part < 0.99:
        warnings.append("峰窗口存在数据空段，面积只计算有效部分。")
    if evidence == 0:
        warnings.append("没有高于当前估计背景的像素区间。")
    centroid = float(np.dot(x.centers[usable], positive) / evidence) if evidence > 0 else None
    return AreaMeasurement(
        curve, settings, baseline, residual, widths, left_y, right_y, gross, background,
        evidence, gross - background, centroid, max_index, warnings,
    )
