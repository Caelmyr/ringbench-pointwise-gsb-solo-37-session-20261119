"""Audit support of a full geometric ring rather than only recorded pixels."""
from dataclasses import dataclass
import math

import numpy as np

from .detector_io import csv_bytes
from .geometry import Arc
from .values import integer, mapping, number

RADIAL_SAMPLES = 64
ANGULAR_SAMPLES_PER_BIN = 24


@dataclass(frozen=True)
class CoverageSpec:
    inner: float
    outer: float
    bins: int
    arc: Arc

    @classmethod
    def read(cls, body, shape):
        document = mapping(body, "整环覆盖设置")
        maximum = math.hypot(*shape) * 3
        lower = number(document.get("inner"), "内半径", 0, maximum)
        upper = number(document.get("outer"), "外半径", lower + 0.01, maximum + 0.01)
        bins = integer(document.get("bins", 72), "覆盖分箱", 12, 360)
        return cls(
            lower, upper, bins,
            Arc.read(document.get("start", 0), document.get("span", 360)),
        )


@dataclass(frozen=True)
class AngularCoverage:
    settings: CoverageSpec
    candidate_per_bin: int
    on_detector: np.ndarray
    available: np.ndarray
    masked: np.ndarray
    missing: np.ndarray

    @property
    def edges(self):
        arc = self.settings.arc
        return np.linspace(arc.start, arc.start + arc.span, self.settings.bins + 1)

    def bands(self):
        n = self.candidate_per_bin
        possible = self.on_detector
        valid = self.available
        return {
            "mean": (valid / n).tolist(),
            "on_detector": (possible / n).tolist(),
            "count": valid.tolist(),
            "missing": self.missing.tolist(),
            "masked": self.masked.tolist(),
            "eligible": possible.tolist(),
        }

    def summary(self):
        all_samples = self.candidate_per_bin * self.settings.bins
        support = int(self.available.sum())
        detector = int(self.on_detector.sum())
        blank = self.available == 0
        return {
            "total_evaluation_samples": all_samples,
            "detector_fraction": detector / all_samples,
            "supported_fraction": support / all_samples,
            "masked_fraction": float(np.sum(self.masked) / all_samples),
            "missing_fraction": float(np.sum(self.missing) / all_samples),
            "sections_without_support": int(blank.sum()),
            "longest_unsupported_arc_deg": self._longest_blank(blank),
        }

    def _longest_blank(self, blank):
        N = self.settings.bins
        full = math.isclose(self.settings.arc.span, 360)
        target = np.tile(blank, 2) if full else blank
        boundaries = np.flatnonzero(np.diff(np.r_[False, target, False]))
        lengths = boundaries[1::2] - boundaries[::2]
        count = min(N, int(lengths.max())) if lengths.size else 0
        return count * (self.settings.arc.span / N)

    def document(self):
        edges = self.edges
        return {
            "kind": "coverage",
            "axis": {"label": "方位角 (°)", "unit": "deg", "edges": edges.tolist(),
                     "centers": ((edges[1:] + edges[:-1]) / 2).tolist()},
            "values": self.bands(),
            "summary": self.summary(),
            "inner_px": self.settings.inner,
            "outer_px": self.settings.outer,
            "method": "uniform_area_polar_quadrature",
        }

    def csv(self):
        total = self.candidate_per_bin
        return csv_bytes(
            [
                "angle_start_deg", "angle_end_deg", "total_area_samples",
                "on_detector_samples", "usable_samples", "masked_samples", "missing_samples",
            ],
            zip(
                self.edges[:-1], self.edges[1:], [total] * self.settings.bins,
                self.on_detector, self.available, self.masked, self.missing,
            ),
        )


def _radial_nodes(spec):
    progress = (np.arange(RADIAL_SAMPLES) + 0.5) / RADIAL_SAMPLES
    return np.sqrt(spec.inner ** 2 + progress * (spec.outer ** 2 - spec.inner ** 2))


def _angle_nodes(spec):
    count = spec.bins * ANGULAR_SAMPLES_PER_BIN
    steps = (np.arange(count) + 0.5) / count
    return (spec.arc.start + steps * spec.arc.span).reshape((spec.bins, ANGULAR_SAMPLES_PER_BIN))


def estimate_coverage(mask, geometry, spec):
    r = _radial_nodes(spec)[None, :, None]
    a = _angle_nodes(spec)[:, None, :]
    x, y = geometry.position(r, a)
    ix, iy = np.floor(x + 0.5).astype(np.intp), np.floor(y + 0.5).astype(np.intp)
    h, w = mask.codes.shape
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    sampled = np.full(ix.shape, 255, dtype=np.uint8)
    sampled[inside] = mask.codes[iy[inside], ix[inside]]
    view = lambda condition: np.count_nonzero(condition, axis=(1, 2))
    return AngularCoverage(
        spec, RADIAL_SAMPLES * ANGULAR_SAMPLES_PER_BIN,
        view(inside), view(sampled == 0), view((sampled > 1) & inside), view(sampled == 1),
    )
