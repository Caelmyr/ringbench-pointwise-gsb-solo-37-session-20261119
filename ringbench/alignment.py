"""Estimate a small reference-frame translation from the diffraction rings.

The reference counts are compared with the main exposure only on pixels that
exist in both frames.  The estimate is intentionally conservative: when the
overlapping valid area is tiny, when the frame has no usable ring structure,
or when several translations explain the rings equally well, no shift is
applied and the caller keeps the ordinary same-pixel difference.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndi_shift

from .errors import WorkbenchError

SAMPLE_LIMIT = 60_000
MIN_SCORE_PIXELS = 256
MIN_OVERLAP_FRACTION = 0.10
BACKGROUND_SIGMA = 3.0
CLIP_PERCENTILE = 99.5
MIN_CORRELATION = 0.20
MIN_PEAK_MARGIN = 0.02
MIN_RELATIVE_MARGIN = 0.55
MAX_MEANINGFUL_SHIFT = 0.25
RIVAL_DISTANCE = 1.5
BOUNDARY_MARGIN = 0.51

MESSAGES = {
    "insufficient_overlap": (
        "两帧在当前遮罩下可用于对齐的重合像素太少，无法判断参照图平移；"
        "差图仍按未平移计算，请继续用原图手动作图比较。"
    ),
    "flat_reference": (
        "重合区的计数几乎没有环结构，无法判断参照图平移；"
        "差图仍按未平移计算，请继续用原图手动作图比较。"
    ),
    "low_correlation": (
        "两帧的环结构相关性太低，无法可靠确定平移；"
        "差图仍按未平移计算，请继续用原图手动作图比较。"
    ),
    "ambiguous_shift": (
        "存在多个相近的平移候选，像重复纹理，无法分出唯一平移；"
        "差图仍按未平移计算，请继续用原图手动作图比较。"
    ),
    "search_limit": (
        "最佳平移落在搜索范围边缘，不能确认参照图是否移得更远；"
        "差图仍按未平移计算，请继续用原图手动作图比较。"
    ),
}


@dataclass(frozen=True)
class Alignment:
    """Rigid reference shift found on the detector grid.

    The aligned reference value at destination ``(x, y)`` is sampled from the
    source at ``(x + shift_x, y + shift_y)``, so the reference content is
    displaced by ``(-shift_x, -shift_y)`` relative to the main exposure.  The
    applied shift is zero whenever ``applied`` is false; ``candidate_*`` keeps
    the best estimate so rejected attempts can still be inspected per pixel.
    """

    reliable: bool
    applied: bool
    reason: str | None
    shift_x: float
    shift_y: float
    candidate_x: float
    candidate_y: float
    correlation: float | None
    margin: float | None
    sampled_pixels: int
    search_radius_px: float
    message: str | None

    def document(self):
        return {
            "reliable": self.reliable,
            "applied": self.applied,
            "reason": self.reason,
            "message": self.message,
            "shift_x": self.shift_x,
            "shift_y": self.shift_y,
            "candidate_shift_x": self.candidate_x,
            "candidate_shift_y": self.candidate_y,
            "correlation": self.correlation,
            "candidate_margin": self.margin,
            "sampled_pixels": self.sampled_pixels,
            "search_radius_px": self.search_radius_px,
        }


def _rejected(reason, *, candidate=(0.0, 0.0), correlation=None, margin=None,
              sampled=0, radius=0.0):
    return Alignment(
        False, False, reason, 0.0, 0.0, float(candidate[0]), float(candidate[1]),
        correlation, margin, int(sampled), float(radius), MESSAGES[reason],
    )


def _sample_coordinates(ok, limit=SAMPLE_LIMIT):
    ys, xs = np.nonzero(ok)
    if ys.size > limit:
        order = np.linspace(0, ys.size - 1, limit).astype(np.intp)
        ys, xs = ys[order], xs[order]
    return ys, xs


def _high_pass(data, ok, sigma=BACKGROUND_SIGMA):
    """Remove the slowly varying diffuse background, leaving rings and edges."""
    weight = ok.astype(np.float32, copy=False)
    mass = gaussian_filter(weight, sigma, mode="constant")
    del weight
    filled = np.where(ok, data, 0.0).astype(np.float32, copy=False)
    background = gaussian_filter(filled, sigma, mode="constant")
    np.divide(background, mass, out=background, where=mass > 1e-6)
    background[mass <= 1e-6] = 0.0
    del mass
    filled -= background
    return filled


def _clip_limit(values):
    if values.size == 0:
        return 0.0
    if values.size > 200_000:
        order = np.linspace(0, values.size - 1, 200_000).astype(np.intp)
        values = values[order]
    limit = float(np.percentile(np.abs(values), CLIP_PERCENTILE))
    return limit if limit > 0 else 0.0


def _correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if not math.isfinite(denom) or denom <= 1e-20:
        return float("-inf")
    return float(np.dot(a, b) / denom)


def _curvature_offset(left, middle, right):
    denominator = left - 2.0 * middle + right
    if not math.isfinite(denominator) or denominator >= -1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def _refine(scores, iy, ix, radius):
    """Parabolic sub-pixel refinement around the best integer candidate."""
    height, width = scores.shape
    dy = dx = 0.0
    if 0 < iy < height - 1:
        dy = _curvature_offset(scores[iy - 1, ix], scores[iy, ix], scores[iy + 1, ix])
    if 0 < ix < width - 1:
        dx = _curvature_offset(scores[iy, ix - 1], scores[iy, ix], scores[iy, ix + 1])
    return dx, dy, float(ix - radius + dx), float(iy - radius + dy)


def estimate_alignment(signal, signal_ok, reference, reference_ok, max_shift):
    """Search integer shifts then refine the best ring-correlation peak."""
    if signal.shape != reference.shape:
        raise WorkbenchError("自动对齐需要两张具有相同原始探测器尺寸的帧。")
    signal_ok = signal_ok & np.isfinite(signal)
    reference_ok = reference_ok & np.isfinite(reference)
    signal_count = int(np.count_nonzero(signal_ok))
    reference_count = int(np.count_nonzero(reference_ok))
    if min(signal_count, reference_count) < MIN_SCORE_PIXELS:
        return _rejected("insufficient_overlap", sampled=min(signal_count, reference_count),
                         radius=max_shift)

    overlap_count = int(np.count_nonzero(signal_ok & reference_ok))
    if overlap_count < MIN_SCORE_PIXELS:
        return _rejected("insufficient_overlap", sampled=overlap_count, radius=max_shift)
    if overlap_count < MIN_OVERLAP_FRACTION * min(signal_count, reference_count):
        return _rejected("insufficient_overlap", sampled=overlap_count, radius=max_shift)

    ys, xs = _sample_coordinates(signal_ok & reference_ok)
    sampled = int(ys.size)
    if sampled < MIN_SCORE_PIXELS:
        return _rejected("insufficient_overlap", sampled=sampled, radius=max_shift)

    signal_hp = _high_pass(signal, signal_ok)
    reference_hp = _high_pass(reference, reference_ok)
    signal_limit = _clip_limit(signal_hp[signal_ok])
    reference_limit = _clip_limit(reference_hp[reference_ok])
    floor = 1e-5 * max(
        float(np.median(np.abs(signal[signal_ok]))),
        float(np.median(np.abs(reference[reference_ok]))),
        1.0,
    )
    if signal_limit <= floor or reference_limit <= floor:
        return _rejected("flat_reference", sampled=sampled, radius=max_shift)
    np.clip(signal_hp, -signal_limit, signal_limit, out=signal_hp)
    np.clip(reference_hp, -reference_limit, reference_limit, out=reference_hp)

    radius = int(math.floor(max_shift + 1e-9))
    radius = max(1, radius)
    size = 2 * radius + 1
    scores = np.full((size, size), float("-inf"), dtype=np.float64)
    height, width = signal.shape
    required = max(MIN_SCORE_PIXELS, int(math.ceil(MIN_OVERLAP_FRACTION * sampled)))
    signal_values = signal_hp[ys, xs]
    for index_y, shift_y in enumerate(range(-radius, radius + 1)):
        py = ys + shift_y
        inside_y = (py >= 0) & (py < height)
        for index_x, shift_x in enumerate(range(-radius, radius + 1)):
            if shift_x * shift_x + shift_y * shift_y > max_shift * max_shift + 1e-9:
                continue
            px = xs + shift_x
            inside = inside_y & (px >= 0) & (px < width)
            inside[inside] = reference_ok[py[inside], px[inside]]
            if np.count_nonzero(inside) < required:
                continue
            value = _correlation(signal_values[inside], reference_hp[py[inside], px[inside]])
            if math.isfinite(value):
                scores[index_y, index_x] = value

    if not np.any(np.isfinite(scores)):
        return _rejected("insufficient_overlap", sampled=sampled, radius=max_shift)
    flat = int(np.argmax(scores))
    iy, ix = divmod(flat, size)
    best = float(scores[iy, ix])
    _, _, candidate_x, candidate_y = _refine(scores, iy, ix, radius)

    grid_y, grid_x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    apart = np.hypot(grid_x - (ix - radius), grid_y - (iy - radius)) > RIVAL_DISTANCE
    rivals = scores[apart & np.isfinite(scores)]
    rival = float(np.max(rivals)) if rivals.size else float("-inf")
    margin = best - rival if math.isfinite(rival) else None

    if best < MIN_CORRELATION:
        return _rejected("low_correlation", candidate=(candidate_x, candidate_y),
                         correlation=best, margin=margin, sampled=sampled, radius=max_shift)
    if math.hypot(candidate_x, candidate_y) >= max_shift - BOUNDARY_MARGIN:
        return _rejected("search_limit", candidate=(candidate_x, candidate_y),
                         correlation=best, margin=margin, sampled=sampled, radius=max_shift)
    if margin is not None and margin < max(MIN_PEAK_MARGIN, MIN_RELATIVE_MARGIN * (1.0 - best)):
        return _rejected("ambiguous_shift", candidate=(candidate_x, candidate_y),
                         correlation=best, margin=margin, sampled=sampled, radius=max_shift)
    if math.hypot(candidate_x, candidate_y) < MAX_MEANINGFUL_SHIFT:
        return Alignment(
            True, False, None, 0.0, 0.0, float(candidate_x), float(candidate_y),
            best, margin, sampled, float(max_shift), None,
        )
    return Alignment(
        True, True, None, float(candidate_x), float(candidate_y),
        float(candidate_x), float(candidate_y), best, margin, sampled,
        float(max_shift), None,
    )


def translate_reference(reference, valid, shift_x, shift_y):
    """Sample the reference at ``(x + shift_x, y + shift_y)``.

    A destination pixel is usable only when every source pixel contributing to
    its interpolation is usable.  Counts shifted out of the frame or covered by
    a source mask therefore never enter the comparison.
    """
    height, width = reference.shape
    if abs(shift_x) < 1e-9 and abs(shift_y) < 1e-9:
        return reference, valid
    integer_x, integer_y = int(round(shift_x)), int(round(shift_y))
    exact = abs(shift_x - integer_x) < 1e-9 and abs(shift_y - integer_y) < 1e-9
    if exact:
        output = np.full(reference.shape, np.nan, dtype=np.float32)
        accepted = np.zeros(reference.shape, dtype=bool)
        top = max(0, -integer_y)
        bottom = min(height, height - integer_y)
        left = max(0, -integer_x)
        right = min(width, width - integer_x)
        if bottom > top and right > left:
            output[top:bottom, left:right] = reference[
                top + integer_y:bottom + integer_y, left + integer_x:right + integer_x,
            ]
            accepted[top:bottom, left:right] = valid[
                top + integer_y:bottom + integer_y, left + integer_x:right + integer_x,
            ]
        return output, accepted

    filled = np.where(valid, reference, 0.0).astype(np.float32, copy=False)
    weight = valid.astype(np.float32, copy=False)
    moved = ndi_shift(
        filled, (-shift_y, -shift_x), order=1, mode="constant", cval=0.0, prefilter=False,
    )
    support = ndi_shift(
        weight, (-shift_y, -shift_x), order=1, mode="constant", cval=0.0, prefilter=False,
    )
    accepted = support >= 1.0 - 1e-5
    output = np.full(reference.shape, np.nan, dtype=np.float32)
    np.divide(moved, support, out=output, where=accepted)
    return output, accepted
