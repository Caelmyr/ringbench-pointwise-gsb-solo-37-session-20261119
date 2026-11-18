"""Optional corrections for projected solid angle and incident polarization."""
from dataclasses import dataclass

import numpy as np

from .detector_io import array_bytes, csv_bytes, finite_summary
from .errors import WorkbenchError
from .values import boolean, mapping, number, selection

REASONS = {
    0: "像素可用",
    1: "探测器遮罩或缺失计数",
    2: "相对响应过低",
    3: "输出超过可表示范围",
}


@dataclass(frozen=True)
class GeometricResponseSpec:
    solid_angle: bool
    polarization: str
    fraction: float
    direction: float
    minimum: float

    @classmethod
    def read(cls, obj):
        v = mapping(obj, "几何响应设置")
        angle = boolean(v.get("solid_angle", True), "按立体角归一")
        pol = selection(v.get("polarization", "none"), ("none", "mixed"), "偏振修正模式")
        if not angle and pol == "none":
            raise WorkbenchError("请选择立体角或偏振校正。")
        return cls(
            angle,
            pol,
            number(v.get("fraction", 0.5), "主方向偏振分数", 0, 1),
            number(v.get("direction", 0), "主偏振方向", -360, 360),
            number(v.get("minimum", 0.08), "最小可接受相对响应", 0.00001, 0.95),
        )


@dataclass(frozen=True)
class GeometricResponse:
    raw: np.ndarray
    factors: np.ndarray
    values: np.ndarray
    reasons: np.ndarray
    spec: GeometricResponseSpec

    @property
    def valid(self):
        return self.reasons == 0

    def document(self):
        support = self.valid
        return {
            "geometry_model": "flat_orthogonal_detector",
            "solid_angle_corrected": self.spec.solid_angle,
            "polarization": self.spec.polarization,
            "min_response": self.spec.minimum,
            "source_counts": finite_summary(self.raw, support),
            "relative_response": finite_summary(self.factors, support),
            "output_counts": finite_summary(self.values, support),
            "rows": self.summary_rows(),
        }

    def summary_rows(self):
        totals = np.bincount(self.reasons.ravel(), minlength=len(REASONS))
        return [
            {"id": code, "reason": meaning, "pixels": int(totals[code])}
            for code, meaning in REASONS.items()
        ]

    def csv(self):
        return csv_bytes(
            ["code", "description", "pixels"],
            [(row["id"], row["reason"], row["pixels"]) for row in self.summary_rows()],
        )

    def matrix(self, kind):
        if kind == "response":
            return array_bytes(self.factors)
        return array_bytes(self.values)


def _geometric_row(geometry, spec, shape, top, bottom):
    r, az = geometry.detector_grid(shape, top, bottom)
    scaled = r * geometry.pixel_mm / geometry.distance_mm
    denom = 1 + scaled ** 2
    if spec.solid_angle:
        solid = 1 / (denom * np.sqrt(denom))
    else:
        solid = np.ones(r.shape, dtype=np.float64)
    if spec.polarization == "mixed":
        cosine = np.cos(np.radians(az - spec.direction))
        cosine_sq = cosine ** 2
        projection = spec.fraction * cosine_sq + (1 - spec.fraction) * (1 - cosine_sq)
        solid *= 1 - scaled ** 2 / denom * projection
    return solid


def geometric_response(data, valid, geometry, spec):
    h, _ = data.shape
    factors = np.empty(data.shape, dtype=np.float32)
    counts = np.full(data.shape, np.nan, dtype=np.float32)
    reasons = np.where(valid, 0, 1).astype(np.uint8)
    limit = np.finfo(np.float32).max
    for top in range(0, h, 128):
        end = min(top + 128, h)
        response = _geometric_row(geometry, spec, data.shape, top, end)
        flags = reasons[top:end]
        good = flags == 0
        low = response < spec.minimum
        flags[good & low] = 2
        corrected = data[top:end].astype(np.float64) / response
        overflow = (np.abs(corrected) > limit) & (flags == 0)
        flags[overflow] = 3
        okay = flags == 0
        factors[top:end] = np.where(okay, response, np.nan)
        counts[top:end] = np.where(okay, corrected, np.nan)
    if not np.any(reasons == 0):
        raise WorkbenchError("当前阈值和遮罩没有留下可校正探测器像素。")
    return GeometricResponse(data, factors, counts, reasons, spec)
