"""Same-pixel comparison of a signal frame against a reference frame."""
from dataclasses import dataclass

import numpy as np

from .detector_io import array_bytes, finite_summary
from .errors import WorkbenchError

MIN_FIT_OVERLAP = 48
SAMPLE_LIMIT = 75_000


@dataclass(frozen=True)
class Normalization:
    scale: float
    offset: float
    method: str
    score: dict

    def document(self):
        return {"scale": self.scale, "offset": self.offset, "method": self.method, **self.score}


@dataclass(frozen=True)
class Difference:
    data: np.ndarray
    overlap: np.ndarray
    normalization: Normalization
    signal_valid: int
    reference_valid: int

    def document(self):
        return {
            "normalization": self.normalization.document(),
            "overlap": int(self.overlap.sum()),
            "signal_valid": self.signal_valid,
            "reference_valid": self.reference_valid,
            "residuals": finite_summary(self.data, self.overlap),
        }

    def npy(self):
        return array_bytes(self.data)


def _sample_overlap(signal, reference, overlap):
    coords = np.flatnonzero(overlap.ravel())
    if coords.size < MIN_FIT_OVERLAP:
        raise WorkbenchError("两帧重合的可用计数像素太少，无法拟合强度。")
    count = coords.size
    if count > SAMPLE_LIMIT:
        order = np.linspace(0, count - 1, SAMPLE_LIMIT).astype(np.intp)
        coords = coords[order]
    r = reference.ravel()[coords].astype(np.float64)
    s = signal.ravel()[coords].astype(np.float64)
    return s, r


def _weighted_solution(u, s, w):
    weight_sum = float(w.sum())
    u0 = float(np.dot(w, u) / weight_sum)
    s0 = float(np.dot(w, s) / weight_sum)
    du = u - u0
    denom = float(np.dot(w, du * du))
    if denom <= weight_sum * 1e-26:
        raise WorkbenchError("参考帧的重合取样近乎恒定，无法拟合缩放。")
    k = float(np.dot(w, du * (s - s0)) / denom)
    return k, s0 - k * u0


def exposure_fit(signal, reference, overlap):
    s, r = _sample_overlap(signal, reference, overlap)
    origin = float(np.median(r))
    width = float(np.ptp(r))
    if width <= max(1e-12, abs(origin) * 5e-8):
        raise WorkbenchError("参考帧在有效重合区没有足够取值范围，请手动填写比例。")
    x = (r - origin) / width
    weights = np.ones(len(x), dtype=np.float64)
    slope, offset = _weighted_solution(x, s, weights)
    for _ in range(6):
        residual = s - slope * x - offset
        middle = float(np.median(residual))
        sigma = max(float(np.median(np.abs(residual - middle))) * 1.4826, 1e-7)
        weights = np.minimum(1.0, (1.8 * sigma) / np.maximum(np.abs(residual - middle), 1e-15))
        slope, offset = _weighted_solution(x, s, weights)
    k = slope / width
    b = offset - k * origin
    if not 0.001 <= k <= 1000 or not np.isfinite(b):
        raise WorkbenchError("拟合没有得到可用的正向曝光比例，请检查参考帧。")
    residual = s - (r * k + b)
    return Normalization(
        k, b, "fit",
        {
            "sampled_pixels": len(x),
            "median_absolute_error": float(np.median(np.abs(residual))),
            "downweighted_pixels": int(np.count_nonzero(weights < 0.99)),
        },
    )


def compare(signal, signal_ok, reference, reference_ok, spec):
    if signal.shape != reference.shape:
        raise WorkbenchError("差图需要两张具有相同原始探测器尺寸的帧。")
    overlap = signal_ok & reference_ok
    if not np.any(overlap):
        raise WorkbenchError("两帧在当前遮罩下没有重合的可用像素。")
    if spec.normalization == "fit":
        norm = exposure_fit(signal, reference, overlap)
    else:
        norm = Normalization(spec.scale, spec.offset, "scale", {})
    data = np.full(signal.shape, np.nan, dtype=np.float32)
    with np.errstate(over="ignore"):
        data[overlap] = signal[overlap].astype(np.float64) - (
            norm.scale * reference[overlap].astype(np.float64) + norm.offset
        )
    if not np.all(np.isfinite(data[overlap])):
        raise WorkbenchError("差值超出可导出数值范围。")
    return Difference(
        data, overlap, norm, int(np.count_nonzero(signal_ok)), int(np.count_nonzero(reference_ok)),
    )
