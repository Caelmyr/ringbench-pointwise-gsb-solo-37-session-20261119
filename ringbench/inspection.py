"""Detector ROI summaries, row/column checks, and repeat-exposure stability."""
from dataclasses import dataclass

import numpy as np

from .detector_io import csv_bytes, finite_summary, numbers_for_json
from .errors import WorkbenchError
from .values import integer, mapping, pair


@dataclass(frozen=True)
class Region:
    start: tuple
    stop: tuple

    @classmethod
    def read(cls, value, width, height):
        if value is None:
            return cls((0, 0), (width, height))
        v = mapping(value, "计数检查区域")
        bounds = (0, width, 0, height)
        a = pair(v.get("start"), "区域起点", bounds)
        b = pair(v.get("stop"), "区域终点", bounds)
        left, right = sorted((int(a[0]), int(b[0])))
        top, bottom = sorted((int(a[1]), int(b[1])))
        if right <= left or bottom <= top:
            raise WorkbenchError("区域需要覆盖至少一个完整探测器像素。")
        return cls((left, top), (right, bottom))

    @property
    def width(self):
        return self.stop[0] - self.start[0]

    @property
    def height(self):
        return self.stop[1] - self.start[1]

    def pixels(self):
        return np.s_[self.start[1]:self.stop[1], self.start[0]:self.stop[0]]

    def document(self):
        return {"start": list(self.start), "stop": list(self.stop), "width": self.width, "height": self.height}


@dataclass(frozen=True)
class Histogram:
    edges: np.ndarray
    bins: np.ndarray
    below: int
    above: int

    def document(self):
        return {
            "edges": self.edges.tolist(),
            "count": self.bins.tolist(),
            "underflow": self.below,
            "overflow": self.above,
        }

    def csv(self):
        return csv_bytes(
            ["count_low", "count_high", "valid_pixels"],
            zip(self.edges[:-1], self.edges[1:], self.bins),
        )


def histogram(data, valid, bins, interval=None):
    sample = data[valid].astype(np.float64, copy=False)
    if interval is None:
        if sample.size == 0:
            low, high = 0, 1
        else:
            low, high = np.percentile(sample, [0.5, 99.5])
            if high <= low:
                low, high = sample.min(), sample.max()
            if high <= low:
                eps = max(1.0, abs(float(low)) * 1e-6)
                low, high = float(low) - eps / 2, float(high) + eps / 2
    else:
        low, high = interval
    edges = np.linspace(low, high, bins + 1)
    counts, _ = np.histogram(sample, bins=edges)
    return Histogram(edges, counts, int(np.count_nonzero(sample < low)), int(np.count_nonzero(sample > high)))


@dataclass(frozen=True)
class StripeProfile:
    coordinate: np.ndarray
    support: np.ndarray
    maximum_support: int
    mean: np.ndarray
    standard: np.ndarray
    dimension: str

    def summary(self):
        return {
            "axis": self.dimension,
            "index_count": int(self.coordinate.size),
            "no_support": int(np.count_nonzero(self.support == 0)),
            "complete_support": int(np.count_nonzero(self.support == self.maximum_support)),
            "median_mean": float(np.median(self.mean[np.isfinite(self.mean)])) if np.any(self.support) else None,
        }

    def document(self):
        return {
            "kind": "stripe",
            "axis": {
                "label": "探测器列 (px)" if self.dimension == "column" else "探测器行 (px)",
                "unit": "px",
                "centers": self.coordinate.tolist(),
            },
            "values": {
                "mean": numbers_for_json(self.mean),
                "std": numbers_for_json(self.standard),
                "count": self.support.tolist(),
                "eligible": [self.maximum_support] * self.coordinate.size,
                "fraction": (self.support / self.maximum_support).tolist(),
            },
            "summary": self.summary(),
        }

    def csv(self):
        return csv_bytes(
            [f"detector_{self.dimension}_px", "valid_pixels", "eligible_pixels", "mean_count", "std_count"],
            zip(
                self.coordinate, self.support, [self.maximum_support] * len(self.coordinate),
                numbers_for_json(self.mean), numbers_for_json(self.standard),
            ),
        )


def stripe_profile(data, valid, origin, dimension):
    collapse_axis = 0 if dimension == "column" else 1
    pixels = data.shape[collapse_axis]
    support = valid.sum(axis=collapse_axis)
    d = np.where(valid, data, 0).astype(np.float64)
    average = np.divide(
        d.sum(axis=collapse_axis),
        support,
        out=np.full(support.shape, np.nan),
        where=support > 0,
    )
    expanded = average[None, :] if collapse_axis == 0 else average[:, None]
    errors = np.where(valid, (d - expanded) ** 2, 0)
    variance = np.divide(
        errors.sum(axis=collapse_axis),
        support,
        out=np.full(support.shape, np.nan),
        where=support > 0,
    )
    coords = np.arange(origin, origin + len(average))
    return StripeProfile(coords, support, pixels, average, np.sqrt(variance), dimension)


@dataclass(frozen=True)
class Inspection:
    roi: Region
    summary: dict
    hist: Histogram
    columns: StripeProfile
    rows: StripeProfile

    def document(self):
        return {
            "region": self.roi.document(),
            "statistics": self.summary,
            "histogram": self.hist.document(),
            "columns": self.columns.document(),
            "rows": self.rows.document(),
        }


def inspect_region(data, valid, region, bins, interval=None):
    pix = region.pixels()
    window = data[pix]
    mask = valid[pix]
    summary = finite_summary(window, mask)
    summary["candidate"] = int(window.size)
    summary["valid_fraction"] = float(mask.mean())
    return Inspection(
        region, summary, histogram(window, mask, bins, interval),
        stripe_profile(window, mask, region.start[0], "column"),
        stripe_profile(window, mask, region.start[1], "row"),
    )


def read_pixel(data, mask, geometry, point):
    h, w = data.shape
    x = integer(point[0], "列", 0, w - 1)
    y = integer(point[1], "行", 0, h - 1)
    value = data[y, x]
    local = Region((max(0, x - 2), max(0, y - 2)), (min(w, x + 3), min(h, y + 3)))
    neighborhood = finite_summary(data[local.pixels()], mask.codes[local.pixels()] == 0)
    return {
        "x": x,
        "y": y,
        "counts": float(value) if np.isfinite(value) else None,
        "valid": bool(mask.codes[y, x] == 0),
        "reason": mask.label_at(x, y),
        "nearby_valid": neighborhood["count"],
        "nearby_mean": neighborhood["mean"],
        **geometry.spot_coordinates(x, y),
    }


@dataclass(frozen=True)
class Stability:
    mean: np.ndarray
    deviation: np.ndarray
    observations: np.ndarray
    selected: list
    minimum: int

    @property
    def sufficient(self):
        return self.observations >= self.minimum

    def document(self):
        usable = self.sufficient
        return {
            "selected_frames": self.selected,
            "minimum_observations": self.minimum,
            "detector_pixels": int(usable.size),
            "sufficient_pixels": int(np.count_nonzero(usable)),
            "observation_histogram": np.bincount(
                self.observations.ravel(), minlength=len(self.selected) + 1,
            ).tolist(),
            "mean_counts": finite_summary(self.mean, usable),
            "sample_deviation": finite_summary(self.deviation, usable),
        }

    def outliers(self, level, max_rows):
        found = np.flatnonzero(self.sufficient & (self.deviation >= level))
        if len(found) == 0:
            return []
        value = self.deviation.ravel()[found]
        selection = np.lexsort((found, -value))[:max_rows]
        result = []
        for flat in found[selection]:
            y, x = divmod(int(flat), self.mean.shape[1])
            result.append({
                "x": x,
                "y": y,
                "observations": int(self.observations[y, x]),
                "mean": float(self.mean[y, x]),
                "std": float(self.deviation[y, x]),
            })
        return result

    def outlier_csv(self, level, max_rows):
        rows = self.outliers(level, max_rows)
        return csv_bytes(
            ["detector_x_px", "detector_y_px", "observations", "mean_count", "sample_std_count"],
            ((r["x"], r["y"], r["observations"], r["mean"], r["std"]) for r in rows),
        )


def stability(frames, shape, minimum):
    """Track per-pixel observed repeats with an online stable sample variance."""
    center = np.zeros(shape, dtype=np.float64)
    squared = np.zeros(shape, dtype=np.float64)
    observed = np.zeros(shape, dtype=np.uint8)
    names = []
    for frame, data, valid in frames:
        if data.shape != shape or valid.shape != shape:
            raise WorkbenchError("重复曝光统计只支持同尺寸的探测器帧。")
        observed[valid] += 1
        value = data[valid].astype(np.float64)
        before = center[valid]
        delta = value - before
        adjusted = before + delta / observed[valid]
        center[valid] = adjusted
        squared[valid] += delta * (value - adjusted)
        names.append({"id": frame.id, "title": frame.title})
    if len(names) < minimum:
        raise WorkbenchError("选择的重复曝光数量少于最低有效观测次数。")
    sufficient = observed >= minimum
    dev = np.sqrt(
        np.divide(squared, observed.astype(np.int16) - 1,
                  out=np.full(shape, np.nan), where=sufficient),
    )
    return Stability(np.where(sufficient, center, np.nan), dev, observed, names, minimum)
