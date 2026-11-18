"""Subpixel detector intensity boundaries with finite-cell support."""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .detector_io import csv_bytes
from .errors import WorkbenchError
from .values import integer, mapping, number, selection

MAX_SEGMENTS = 65_000
MAX_PREVIEW_VERTICES = 40_000
SIDES = ((0, 1), (1, 2), (3, 2), (0, 3))
COORDINATES = ((0, 0), (1, 0), (1, 1), (0, 1))
ONE_PAIR = {
    1: (0, 3), 2: (0, 1), 3: (1, 3), 4: (1, 2), 6: (0, 2), 7: (2, 3),
    8: (2, 3), 9: (0, 2), 11: (1, 2), 12: (1, 3), 13: (0, 1), 14: (0, 3),
}


@dataclass(frozen=True)
class ContourSettings:
    level: float | None
    mode: str
    percentile: float
    minimum: float
    maximum: int

    @classmethod
    def read(cls, obj):
        doc = mapping(obj, "等强线参数")
        choice = selection(doc.get("mode", "percentile"), ("fixed", "percentile"), "等强线取值")
        level = number(doc.get("level"), "等强线计数", allow_none=choice != "fixed")
        return cls(
            level,
            choice,
            number(doc.get("percentile", 97), "计数百分位", 0.1, 99.99),
            number(doc.get("minimum", 12), "最短等强线长度", 0, 40_000),
            integer(doc.get("maximum", 32), "等强线上限", 1, 160),
        )


@dataclass(frozen=True)
class ContourPath:
    points: np.ndarray
    closed: bool
    length: float

    def bounds(self):
        left, top = self.points.min(axis=0)
        right, bottom = self.points.max(axis=0)
        return [float(left), float(top), float(right), float(bottom)]

    def centroid(self):
        if self.closed and len(self.points) > 3:
            x, y = self.points[:-1].T
            next_x, next_y = self.points[1:].T
            terms = x * next_y - next_x * y
            denominator = 3 * terms.sum()
            if abs(denominator) > 1e-8:
                return [float(np.dot(terms, x + next_x) / denominator),
                        float(np.dot(terms, y + next_y) / denominator)]
        center = np.mean(self.points, axis=0)
        return center.tolist()

    @property
    def area(self):
        if not self.closed:
            return None
        p = self.points
        return float(abs(np.sum(p[:-1,0] * p[1:,1] - p[:-1,1] * p[1:,0])) / 2)

    def preview(self, maximum):
        if len(self.points) <= maximum:
            return self.points.tolist()
        ix = np.linspace(0, len(self.points) - 1, maximum, dtype=np.intp)
        return self.points[ix].tolist()

    def document(self, maximum):
        return {
            "closed": self.closed,
            "length_px": self.length,
            "area_px2": self.area,
            "center_xy": self.centroid(),
            "bounds_xy": self.bounds(),
            "vertex_count": len(self.points),
            "xy": self.preview(maximum),
        }


@dataclass(frozen=True)
class Contours:
    level: float
    paths: list
    usable_cells: int
    crossed_cells: int
    source_total: int
    omitted: int

    def summary(self):
        return {
            "isovalue": self.level,
            "supported_cells": self.usable_cells,
            "crossed_cells": self.crossed_cells,
            "traced_paths": self.source_total,
            "displayed_paths": len(self.paths),
            "closed_paths": sum(p.closed for p in self.paths),
            "paths_below_size_or_budget": self.omitted,
        }

    def document(self):
        budget = MAX_PREVIEW_VERTICES // max(1, len(self.paths))
        return {
            "summary": self.summary(),
            "paths": [p.document(budget) for p in self.paths],
        }

    def csv(self):
        return csv_bytes(
            ["path", "vertex", "detector_x_px", "detector_y_px", "closed", "isovalue"],
            (
                (path_index, point_index, x, y, path.closed, self.level)
                for path_index, path in enumerate(self.paths, 1)
                for point_index, (x, y) in enumerate(path.points, 1)
            ),
        )


def _edge_node(row, col, side):
    if side == 0:
        return (row, col, 0)
    if side == 2:
        return (row + 1, col, 0)
    if side == 1:
        return (row, col + 1, 1)
    return (row, col, 1)


def _edge_point(row, col, side, corner):
    i, j = SIDES[side]
    a, b = corner[i], corner[j]
    ratio = -a / (b - a)
    x0, y0 = COORDINATES[i]
    x1, y1 = COORDINATES[j]
    return col + x0 + ratio * (x1 - x0), row + y0 + ratio * (y1 - y0)


def _pairs(case, corner):
    if case in ONE_PAIR:
        return (ONE_PAIR[case],)
    # Compare the opposite-corner products of the bilinear centered surface.
    saddle = corner[0] * corner[2] - corner[1] * corner[3]
    if saddle >= 0:
        return ((0, 1), (2, 3))
    return ((0, 3), (1, 2))


def _draft(data, valid, level):
    usable = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, 1:] & valid[1:, :-1]
    steps = data > level
    case = (
        steps[:-1, :-1].astype(np.uint8) |
        steps[:-1, 1:].astype(np.uint8) * 2 |
        steps[1:, 1:].astype(np.uint8) * 4 |
        steps[1:, :-1].astype(np.uint8) * 8
    )
    signed = data.astype(np.float64) - level
    selected = np.argwhere(usable & (case != 0) & (case != 15))
    if len(selected) > MAX_SEGMENTS:
        raise WorkbenchError("此计数值产生过多边界，请提高阈值或先排除无关区域。")
    edges = []
    coordinates = {}
    for row, col in selected:
        idx = int(case[row, col])
        corner = (
            signed[row, col], signed[row, col + 1],
            signed[row + 1, col + 1], signed[row + 1, col],
        )
        for side_a, side_b in _pairs(idx, corner):
            a = _edge_node(row, col, side_a)
            b = _edge_node(row, col, side_b)
            if a not in coordinates:
                coordinates[a] = _edge_point(row, col, side_a, corner)
            if b not in coordinates:
                coordinates[b] = _edge_point(row, col, side_b, corner)
            edges.append((a, b))
    return edges, coordinates, int(usable.sum()), len(selected)


def _walk(start, graph, traversed):
    path = [start]
    node = start
    while node in graph:
        available = [(next_node, k) for next_node, k in graph[node] if k not in traversed]
        if not available:
            break
        advance, index = available[0]
        traversed.add(index)
        path.append(advance)
        if advance == start:
            return path, True
        node = advance
    return path, False


def _assemble(edges, coordinates):
    graph = defaultdict(list)
    for edge_index, (a, b) in enumerate(edges):
        graph[a].append((b, edge_index))
        graph[b].append((a, edge_index))
    visited = set()

    def paths_from(nodes):
        for node in nodes:
            while any(index not in visited for _, index in graph[node]):
                path, closed = _walk(node, graph, visited)
                points = np.array([coordinates[idx] for idx in path])
                dist = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
                if dist > 0:
                    yield ContourPath(points, closed, dist)

    boundaries = sorted(node for node in graph if len(graph[node]) == 1)
    yield from paths_from(boundaries)
    yield from paths_from(graph)


def extract_contours(data, valid, settings):
    if not np.any(valid):
        raise WorkbenchError("没有可用探测器像素，无法寻找等强线。")
    if settings.mode == "fixed":
        level = settings.level
    else:
        level = float(np.percentile(data[valid], settings.percentile))
    edges, coords, support, crossed = _draft(data, valid, level)
    resolved = list(_assemble(edges, coords))
    resolved.sort(key=lambda path: -path.length)
    winners = [p for p in resolved if p.length >= settings.minimum][:settings.maximum]
    return Contours(level, winners, support, crossed, len(resolved), len(resolved) - len(winners))
