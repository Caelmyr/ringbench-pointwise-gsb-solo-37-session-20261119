"""Estimate detector distance when several measured radii have known d spacings."""
from dataclasses import dataclass
import math

import numpy as np

from .detector_io import csv_bytes
from .errors import WorkbenchError
from .values import items, label, mapping, number, selection


@dataclass(frozen=True)
class ReferenceObservation:
    name: str
    radius: float
    spacing: float
    sigma: float

    @classmethod
    def read(cls, obj, index):
        value = mapping(obj, "标定环观测")
        return cls(
            label(value.get("label", f"参考环 {index+1}"), "参考环名称", 50),
            number(value.get("radius_px"), "探测器环半径", 0.5, 18_000),
            number(value.get("d_a"), "参考 d 值", 0.001, 5000),
            number(value.get("sigma_px", 1), "环半径误差", 0.01, 250),
        )

    def factor(self, geometry):
        s = geometry.wavelength_a / (2 * self.spacing)
        if s <= 0 or s >= math.sqrt(0.5):
            raise WorkbenchError(f"{self.name}在当前波长下没有探测器正向投影。")
        return math.tan(2 * math.asin(s)) / geometry.pixel_mm


@dataclass(frozen=True)
class DistanceFit:
    distance_mm: float
    uncertainty_mm: float
    rms_px: float
    method: str
    measurements: list

    def document(self):
        return {
            "distance_mm": self.distance_mm,
            "formal_sigma_mm": self.uncertainty_mm,
            "rms_residual_px": self.rms_px,
            "method": self.method,
            "used_rings": self.measurements,
        }

    def csv(self):
        keys = [
            "reference", "d_a", "measured_px", "fit_px", "residual_px",
            "radial_sigma_px", "individual_distance_mm", "fitted_distance_mm",
        ]
        return csv_bytes(
            keys,
            (
                tuple(row[key] if key != "fitted_distance_mm" else self.distance_mm for key in keys)
                for row in self.measurements
            ),
        )


def fit_distances(geometry, obj):
    data = mapping(obj, "已知参考环")
    readings = [
        ReferenceObservation.read(v, i)
        for i, v in enumerate(items(data.get("observations"), "标定环观测", minimum=2, maximum=24))
    ]
    labels = {reading.name for reading in readings}
    if len(labels) != len(readings):
        raise WorkbenchError("请使用不同名称区分标定环。")
    if len({r.spacing for r in readings}) < 2:
        raise WorkbenchError("多环标定至少需要两个不同的参考晶面间距。")
    method = selection(data.get("method", "weighted"), ("weighted", "equal"), "参考环权重")
    signal = np.array([r.radius for r in readings])
    source = np.array([r.factor(geometry) for r in readings])
    sigmas = np.array([r.sigma for r in readings])
    weighting = np.ones(len(readings)) if method == "equal" else 1 / sigmas**2
    denominator = np.dot(source * weighting, source)
    optimum = float(np.dot(source * weighting, signal) / denominator)
    optimum = number(optimum, "估计探测器距离", 0.2, 10_000)
    expected = source * optimum
    residual = signal - expected
    variance = float(np.dot(weighting, residual ** 2) / max(1, len(signal) - 1))
    uncertainty = math.sqrt(variance / denominator)
    result = []
    for row, model, offset, factor in zip(readings, expected, residual, source):
        title = row.name
        if title.startswith(("=", "+", "-", "@")):
            title = "'" + title
        result.append({
            "reference": title,
            "d_a": row.spacing,
            "measured_px": row.radius,
            "fit_px": float(model),
            "residual_px": float(offset),
            "radial_sigma_px": row.sigma,
            "individual_distance_mm": float(row.radius / factor),
        })
    return DistanceFit(
        optimum, uncertainty, float(np.sqrt(np.mean(residual**2))), method, result,
    )
