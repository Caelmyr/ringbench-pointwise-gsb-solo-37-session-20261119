"""Locate and measure isolated bright diffraction spots in detector pixels."""
from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage

from .detector_io import csv_bytes
from .errors import WorkbenchError

FIND_BUDGET = 750
MIN_BACKGROUND = 8


@dataclass(frozen=True)
class Counts:
    gross: float
    net: float | None
    background: float | None
    background_mad: float | None
    peak: float
    pixels: int
    expected_pixels: int
    background_pixels: int
    centroid_x: float | None
    centroid_y: float | None

    def document(self):
        return {
            "gross": self.gross,
            "net": self.net,
            "background": self.background,
            "background_mad": self.background_mad,
            "max_pixel": self.peak,
            "valid_pixels": self.pixels,
            "full_pixels": self.expected_pixels,
            "valid_fraction": self.pixels / self.expected_pixels,
            "background_pixels": self.background_pixels,
            "centroid": [self.centroid_x, self.centroid_y] if self.centroid_x is not None else None,
        }


def _cutout(data, valid, x, y, outer):
    left, right = math.ceil(x - outer), math.floor(x + outer) + 1
    top, bottom = math.ceil(y - outer), math.floor(y + outer) + 1
    fx = np.arange(left, right, dtype=np.float64)
    fy = np.arange(top, bottom, dtype=np.float64)
    r_square = (fy[:, None] - y) ** 2 + (fx[None, :] - x) ** 2
    h, w = data.shape
    in_x = (fx >= 0) & (fx < w)
    in_y = (fy >= 0) & (fy < h)
    ok = np.zeros(r_square.shape, dtype=bool)
    values = np.zeros(r_square.shape, dtype=np.float64)
    where = np.ix_(in_y, in_x)
    view = np.s_[max(0, top):max(0, min(h, bottom)), max(0, left):max(0, min(w, right))]
    values[where] = data[view]
    ok[where] = valid[view]
    return r_square, fx, fy, values, ok


def aperture_counts(data, valid, x, y, aperture):
    r2, xx, yy, values, accepted = _cutout(data, valid, x, y, aperture.background_outer)
    expected = r2 <= aperture.radius ** 2
    spot_support = expected & accepted
    pixels = int(spot_support.sum())
    if not pixels:
        raise WorkbenchError("斑点窗内没有可用像素。")
    annulus = (r2 >= aperture.background_inner ** 2) & (r2 <= aperture.background_outer ** 2)
    samples = values[annulus & accepted]
    gross = float(np.sum(values[spot_support], dtype=np.float64))
    peak = float(np.max(values[spot_support]))
    net, median, mad, cx, cy = None, None, None, None, None
    if samples.size >= MIN_BACKGROUND:
        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        net = gross - median * pixels
        weights = np.where(spot_support, np.maximum(values - median, 0), 0)
        weight_sum = float(weights.sum())
        if weight_sum > 0:
            cx = float(np.dot(weights.sum(axis=0), xx) / weight_sum)
            cy = float(np.dot(weights.sum(axis=1), yy) / weight_sum)
    return Counts(
        gross, net, median, mad, peak, pixels, int(expected.sum()), int(samples.size), cx, cy,
    )


def measure(data, valid, geometry, spots, aperture):
    rows = []
    for spot in spots:
        row = {
            "label": spot.label,
            "x": spot.x,
            "y": spot.y,
            **geometry.spot_coordinates(spot.x, spot.y),
        }
        try:
            counts = aperture_counts(data, valid, spot.x, spot.y, aperture)
            rows.append({**row, "measured": True, "counts": counts.document()})
        except WorkbenchError as exc:
            rows.append({**row, "measured": False, "error": str(exc), "counts": None})
    return rows


def catalog_csv(rows):
    header = [
        "name", "x_px", "y_px", "radius_px", "azimuth_deg", "two_theta_deg", "q_inverse_a",
        "gross", "net", "background", "background_pixels", "valid_pixels", "full_pixels",
        "centroid_x_px", "centroid_y_px", "status",
    ]

    def table():
        for row in rows:
            counts = row.get("counts") or {}
            centroid = counts.get("centroid") or [None, None]
            label = row["label"]
            if label.startswith(("=", "+", "-", "@")):
                label = "'" + label
            yield (
                label, row["x"], row["y"], row["radius_px"], row["azimuth_deg"],
                row["two_theta_deg"], row["q_inverse_a"], counts.get("gross"), counts.get("net"),
                counts.get("background"), counts.get("background_pixels"), counts.get("valid_pixels"),
                counts.get("full_pixels"), *centroid, "measured" if row["measured"] else row["error"],
            )

    return csv_bytes(header, table())


def find_candidates(data, valid, geometry, aperture, threshold, separation, maximum):
    if not np.any(valid):
        return []
    available = data[valid]
    if threshold is None:
        middle, high, top = np.percentile(available, [50, 98, 99.8])
        threshold = float(max(middle + 8 * np.median(np.abs(available - middle)), high))
        if threshold > top:
            threshold = float(top)
    detector = np.where(valid, data, -np.inf)
    size = int(2 * math.ceil(separation) + 1)
    nearby_max = ndimage.maximum_filter(detector, size=size, mode="constant", cval=-np.inf)
    coordinates = np.flatnonzero((detector == nearby_max) & valid & (data > threshold))
    if coordinates.size > FIND_BUDGET:
        eligible = data.ravel()[coordinates]
        order = np.lexsort((coordinates, -eligible))[:FIND_BUDGET]
        coordinates = coordinates[order]
    order = sorted(coordinates.tolist(), key=lambda ix: (-float(data.ravel()[ix]), ix))
    accepted = []
    w = data.shape[1]
    for index in order:
        y, x = divmod(index, w)
        if any((x - p["x"]) ** 2 + (y - p["y"]) ** 2 < separation ** 2 for p in accepted):
            continue
        counts = aperture_counts(data, valid, x, y, aperture)
        if counts.background is None or counts.peak <= counts.background:
            continue
        candidate = {
            "x": float(x), "y": float(y), "peak": counts.peak,
            "net": counts.net, "background": counts.background,
            **geometry.spot_coordinates(x, y),
        }
        accepted.append(candidate)
        if len(accepted) >= maximum:
            break
    return accepted
