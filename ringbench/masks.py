"""Detector-space exclusion layers with explicit provenance."""
from collections import Counter
from dataclasses import dataclass
import math

import numpy as np

from .detector_io import read_array
from .errors import WorkbenchError
from .geometry import Arc
from .values import MAX_MASKS, items, label, mapping, number, pair, selection, xy_bounds

SHAPES = ("rectangle", "disk", "polygon", "intensity", "annulus")


@dataclass(frozen=True)
class MaskLayer:
    kind: str
    title: str
    enabled: bool
    args: dict

    def document(self):
        return {
            "kind": self.kind,
            "title": self.title,
            "enabled": self.enabled,
            **self.args,
        }

    @classmethod
    def read(cls, value, width, height):
        value = mapping(value, "遮罩层")
        kind = selection(value.get("kind"), SHAPES, "遮罩形状")
        title = label(value.get("title"), "遮罩名", 60)
        enabled = value.get("enabled", True)
        if type(enabled) is not bool:
            raise WorkbenchError("遮罩启用状态无效。")
        if kind == "intensity":
            args = cls._intensity(value)
        elif kind == "annulus":
            args = cls._annulus(value, width, height)
        else:
            args = cls._shape(value, kind, width, height)
        return cls(kind, title, enabled, args)

    @staticmethod
    def _intensity(value):
        low = number(value.get("low"), "下限计数", allow_none=True)
        high = number(value.get("high"), "上限计数", allow_none=True)
        if low is None and high is None:
            raise WorkbenchError("计数屏蔽需要至少一个阈值。")
        if low is not None and high is not None and low >= high:
            raise WorkbenchError("计数上限应大于计数下限。")
        return {"low": low, "high": high}

    @staticmethod
    def _annulus(value, width, height):
        cap = math.hypot(width, height) * 3
        inner = number(value.get("inner"), "内半径", 0, cap)
        outer = number(value.get("outer"), "外半径", 0.01, cap)
        if inner >= outer:
            raise WorkbenchError("外半径必须大于内半径。")
        arc = Arc.read(value.get("start", 0), value.get("span", 360))
        return {"inner": inner, "outer": outer, **arc.documents()}

    @staticmethod
    def _shape(value, kind, width, height):
        bounds = xy_bounds(width, height, max(width, height))
        if kind == "rectangle":
            a = pair(value.get("a"), "矩形角点", bounds)
            b = pair(value.get("b"), "矩形角点", bounds)
            return {
                "a": [min(a[0], b[0]), min(a[1], b[1])],
                "b": [max(a[0], b[0]), max(a[1], b[1])],
            }
        if kind == "disk":
            cent = pair(value.get("center"), "圆形遮罩中心", bounds)
            r = number(value.get("radius"), "圆半径", 0.1, math.hypot(width, height) * 2)
            return {"center": list(cent), "radius": r}
        points = items(value.get("points"), "多边形顶点", minimum=3, maximum=32)
        points = [list(pair(p, "多边形顶点", bounds)) for p in points]
        if len({tuple(p) for p in points}) != len(points):
            raise WorkbenchError("多边形遮罩存在重合顶点。")
        shifted = points[1:] + points[:1]
        area = sum(ax * by - bx * ay for (ax, ay), (bx, by) in zip(points, shifted))
        if abs(area) < 0.04:
            raise WorkbenchError("请圈出有面积的多边形。")
        return {"points": points}

    def raster(self, data, geometry):
        h, w = data.shape
        if self.kind == "intensity":
            lower, upper = self.args["low"], self.args["high"]
            result = np.zeros(data.shape, dtype=bool)
            if lower is not None:
                result |= data <= lower
            if upper is not None:
                result |= data >= upper
            return result
        if self.kind == "rectangle":
            (x0, y0), (x1, y1) = self.args["a"], self.args["b"]
            out = np.zeros(data.shape, dtype=bool)
            left = max(0, math.ceil(x0))
            right = min(w, math.floor(x1) + 1)
            top = max(0, math.ceil(y0))
            bottom = min(h, math.floor(y1) + 1)
            if right > left and bottom > top:
                out[top:bottom, left:right] = True
            return out
        if self.kind == "disk":
            x, y = self.args["center"]
            dx = np.arange(w, dtype=np.float64) - x
            dy = np.arange(h, dtype=np.float64) - y
            return dx[None, :] ** 2 + dy[:, None] ** 2 <= self.args["radius"] ** 2
        if self.kind == "annulus":
            inner, outer = self.args["inner"], self.args["outer"]
            arc = Arc(self.args["start"], self.args["span"])
            out = np.empty(data.shape, dtype=bool)
            for row in range(0, h, 128):
                end = min(row + 128, h)
                r, azimuth = geometry.detector_grid(data.shape, row, end)
                out[row:end] = (r >= inner) & (r <= outer) & arc.inside(azimuth)
            return out
        return _polygon_mask(data.shape, self.args["points"])


def _polygon_mask(shape, points):
    height, width = shape
    out = np.zeros(shape, dtype=bool)
    top = max(0, math.ceil(min(y for x, y in points)))
    bottom = min(height, math.floor(max(y for x, y in points)) + 1)
    left = max(0, math.ceil(min(x for x, y in points)))
    right = min(width, math.floor(max(x for x, y in points)) + 1)
    if top >= bottom or left >= right:
        return out
    xx = np.arange(left, right, dtype=np.float64)[None, :]
    yy = np.arange(top, bottom, dtype=np.float64)[:, None]
    section = out[top:bottom, left:right]
    boundary = np.zeros_like(section)
    for (ax, ay), (bx, by) in zip(points, points[1:] + points[:1]):
        dx, dy = bx - ax, by - ay
        if abs(dy) > 1e-8:
            on_y = (ay > yy) != (by > yy)
            cross_x = ax + (yy - ay) * dx / dy
            section ^= on_y & (xx < cross_x)
        collinear = np.abs((xx - ax) * dy - (yy - ay) * dx) < 1e-7
        on_segment = (
            (xx >= min(ax, bx)) & (xx <= max(ax, bx))
            & (yy >= min(ay, by)) & (yy <= max(ay, by))
        )
        boundary |= collinear & on_segment
    section |= boundary
    return out


def mask_layers(value, width, height):
    return [
        MaskLayer.read(layer, width, height)
        for layer in items(value, "工程遮罩", maximum=MAX_MASKS)
    ]


@dataclass(frozen=True)
class MaskResult:
    codes: np.ndarray
    layers: list
    imported: bool

    @property
    def valid(self):
        return self.codes == 0

    def label_at(self, x, y):
        code = int(self.codes[y, x])
        if code == 0:
            return None
        if code == 1:
            return "原始计数缺失"
        if code == 2:
            return "导入的探测器遮罩"
        return self.layers[code - 3].title

    def summary(self):
        counts = np.bincount(self.codes.ravel(), minlength=len(self.layers) + 3)
        excluded = int(self.codes.size - counts[0])
        return {
            "total": int(self.codes.size),
            "valid": int(counts[0]),
            "excluded": excluded,
            "fraction_valid": float(counts[0] / self.codes.size),
            "missing": int(counts[1]),
            "imported_count": int(counts[2]),
            "imported": self.imported,
            "layers": [
                {
                    "index": idx,
                    "title": layer.title,
                    "enabled": layer.enabled,
                    "excluded_first": int(counts[idx + 3]),
                }
                for idx, layer in enumerate(self.layers)
            ],
        }


def evaluate(data, geometry, layers, imported=None):
    codes = np.zeros(data.shape, dtype=np.uint8)
    codes[~np.isfinite(data)] = 1
    if imported is not None:
        if imported.shape != data.shape:
            raise WorkbenchError("导入遮罩和探测器的行列必须一致。")
        codes[imported & (codes == 0)] = 2
    for index, spec in enumerate(layers, 3):
        if spec.enabled:
            covered = spec.raster(data, geometry)
            codes[covered & (codes == 0)] = index
    return MaskResult(codes, layers, imported is not None)


def read_imported_mask(payload, shape):
    source = read_array(payload)
    if source.shape != shape:
        raise WorkbenchError("二值遮罩尺寸必须与探测器相同。")
    if source.dtype.kind != "b":
        if source.dtype.kind not in "iu" or not np.all((source == 0) | (source == 1)):
            raise WorkbenchError("二值遮罩文件必须由布尔值或 0、1 整数组成，1 表示排除。")
    return np.asarray(source, dtype=bool, order="C")


def reason_histogram(result, points):
    reasons = Counter(result.label_at(x, y) for x, y in points)
    return [
        {"reason": "可用" if reason is None else reason, "pixels": value}
        for reason, value in reasons.most_common()
    ]
