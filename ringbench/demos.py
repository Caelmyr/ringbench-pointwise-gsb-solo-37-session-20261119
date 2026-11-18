"""Deterministic teaching images, with no experimental source claims."""
from dataclasses import asdict
import math

import numpy as np

from .detector_io import capture_from_array
from .errors import WorkbenchError
from .geometry import Geometry
from .state import (
    Aperture, AzimuthSettings, CakeSettings, DifferenceSettings, LineSettings, ProfileSettings, default_state,
)

DATASETS = {
    "powder": {
        "title": "铜合金粉末 · 合成",
        "description": "多条完整和局部遮挡的粉末环，可比较曝光、径向峰和环线标定。",
        "size": (600, 480),
    },
    "texture": {
        "title": "压延晶粒 · 合成",
        "description": "散射强度沿方位成组变化，可观察强取向、离散斑点和探测器间隙。",
        "size": (720, 520),
    },
    "edge": {
        "title": "偏心探测器 · 合成",
        "description": "环带被探测器边缘截断，可观察遮罩、短弧和输出边界。",
        "size": (480, 418),
    },
}


def choices():
    return [{"key": key, "title": item["title"], "description": item["description"]}
            for key, item in DATASETS.items()]


def _wrap_delta(value, position):
    return (value - position + 180.0) % 360.0 - 180.0


def _rings(radius, az, entries):
    result = np.zeros(radius.shape, dtype=np.float64)
    for target, width, count, texture in entries:
        if texture:
            brightness = 0.14 + sum(
                strength * np.exp(-0.5 * (_wrap_delta(az, angle) / spread) ** 2)
                for angle, spread, strength in texture
            )
        else:
            brightness = 1.0 + 0.09 * np.cos(np.radians(az * 2 + target))
        if isinstance(width, tuple):
            sigma = np.where(radius < target, width[0], width[1])
        else:
            sigma = width
        result += count * np.exp(-0.5 * ((radius - target) / sigma) ** 2) * brightness
    return result


def _geometry(key, width, height):
    if key == "edge":
        return Geometry(60.6, height * 0.56 + 0.2, 0.14, 102.0, 1.54, 6.5)
    return Geometry(width * 0.485 + 0.43, height * 0.52 - 0.27, 0.172, 158.0, 1.54, 0)


def _ring_specs(key):
    if key == "texture":
        arcs = [(37, 11, 1.0), (215, 25, 0.84), (301, 8, 0.4)]
        return [(65, 1.8, 315, []), (110, 2.6, 780, arcs), (162, 3.4, 465, arcs),
                (207, 1.9, 420, [(101, 17, 1.1)]), (273, 2.4, 270, arcs)]
    if key == "edge":
        return [(48, 2.1, 490, []), (91, 3.1, 640, []), (152, 1.8, 390, []),
                (230, 1.7, 490, []), (282, 2.4, 400, []), (341, 1.55, 290, [])]
    return [(43, 1.4, 440, []), (80, 2.3, 800, []), (128, 2.2, 615, []), (132, 1.9, 277, []),
            (163, 2.65, 360, []), (193, (1.7, 4.0), 540, []), (235, 2.1, 395, []),
            (267, 1.9, 220, [])]


def _beam_stop(radius, azimuth, width):
    beamstop = radius < (20.0 if width > 500 else 15.0)
    support = (np.abs(_wrap_delta(azimuth, 270.0)) * np.pi / 180.0 * radius) < 3.2
    return beamstop | (support & (radius > 8) & (radius < 105))


def _masks(key, geometry, w, h):
    shapes = [
        {"kind": "disk", "title": "直射束挡片", "enabled": True,
         "center": [geometry.cx, geometry.cy], "radius": 21 if w > 500 else 16},
        {"kind": "rectangle", "title": "竖向模块间隙", "enabled": True,
         "a": [math.floor(w * 0.67), 0], "b": [math.floor(w * 0.67) + 4, h - 1]},
        {"kind": "intensity", "title": "饱和与坏点", "enabled": True, "low": None, "high": 4094},
    ]
    if key == "texture":
        shapes.append({
            "kind": "polygon", "title": "右上支架投影", "enabled": True,
            "points": [[w * 0.63, 0], [w - 1, 0], [w - 1, h * 0.15], [w * 0.75, h * 0.085]],
        })
    return shapes


def _reference_d(geometry, targets):
    results = []
    for r in targets:
        angle = math.atan2(r * geometry.pixel_mm, geometry.distance_mm)
        results.append(geometry.wavelength_a / (2 * math.sin(angle / 2)))
    return results


def _sample_marks(key, geometry, radius, width, height):
    if key == "edge":
        angles = [-42, -15, 14, 42]
    else:
        angles = [8, 53, 115, 169, 228, 309]
    offsets = (0.2, -0.4, 0.3, 0.08, -0.2, 0.32)
    results = []
    for angle, offset in zip(angles, offsets):
        x, y = geometry.position(radius + offset, angle)
        if 0 <= x < width and 0 <= y < height:
            results.append([round(float(x), 2), round(float(y), 2)])
    return results


def make_demo(key):
    if key not in DATASETS:
        raise WorkbenchError("未找到该合成示范。")
    spec = DATASETS[key]
    w, h = spec["size"]
    shape = (h, w)
    g = _geometry(key, w, h)
    radius, az = g.detector_grid(shape)
    yy, xx = np.indices(shape)
    rng = np.random.default_rng({"powder": 42, "texture": 271, "edge": 667}[key])
    diffuse = 56 + 0.10 * xx + 0.03 * yy + 20 * np.exp(-radius / 140)
    base = diffuse + _rings(radius, az, _ring_specs(key))
    spot_map = np.zeros(shape, dtype=np.float64)
    anchors = []
    candidates = [(74, 39), (116, 148), (150, 285), (195, 67), (217, 325), (246, 17), (247, 109)]
    for index, (r, a) in enumerate(candidates):
        sx, sy = g.position(r, a)
        if 8 < sx < w - 8 and 8 < sy < h - 8:
            peak = (810 + 70 * index) if key == "texture" else (490 + 45 * index)
            spot_map += peak * np.exp(-0.5 * (((xx - sx) / 1.7) ** 2 + ((yy - sy) / 2.4) ** 2))
            if len(anchors) < 4:
                anchors.append({"x": round(float(sx), 2), "y": round(float(sy), 2),
                                "label": f"斑点 {len(anchors) + 1}"})
    current = base + spot_map + rng.normal(0, 5.8, shape)
    comparison = (base + spot_map * 0.37 - 6.4) / 1.21 + rng.normal(0, 4.6, shape)
    beam = _beam_stop(radius, az, w)
    current[beam] *= 0.10
    comparison[beam] *= 0.10
    lane = math.floor(w * 0.67)
    current[:, lane:lane + 5] = np.nan
    comparison[:, lane - 1:lane + 5] = np.nan
    if key == "texture":
        current[(xx > w * 0.87) & (yy < h * 0.117)] *= 0.26
    for y, x in rng.integers([2, 2], [h - 2, w - 2], (26, 2)):
        current[y, x] = 4095
    ring_values = [row[0] for row in _ring_specs(key)]
    ring_radius = ring_values[1]
    marks = _sample_marks(key, g, ring_radius, w, h)
    doc = default_state(w, h)
    first = "1" * 32
    second = "2" * 32
    settings = {
        **doc, "geometry": g.document(), "masks": _masks(key, g, w, h),
        "marks": marks, "spots": anchors,
        "reference_d": _reference_d(g, ring_values[:4]),
        "profile": asdict(ProfileSettings(
            bins=350 if key == "powder" else 176, radial=(0, 315) if key == "powder" else None,
        )),
        "azimuth": asdict(AzimuthSettings(ring_radius - 5, ring_radius + 5)),
        "cake": asdict(CakeSettings(radial_bins=164, angular_bins=144)),
        "line": asdict(LineSettings((12, round(g.cy)), (w - 14, round(g.cy)), 1.5, 1.0)),
        "aperture": asdict(Aperture(4, 7, 11)),
        "difference": asdict(DifferenceSettings(reference=second, normalization="fit")),
        "active_frame": first,
    }
    captures = [
        capture_from_array(current, "主曝光", source_name=f"{key}-signal-synthetic.npy", source_format="synthetic"),
        capture_from_array(comparison, "参照曝光", source_name=f"{key}-reference-synthetic.npy", source_format="synthetic"),
    ]
    detector_mask = (beam & (radius < 110))
    return spec["title"], captures, settings, [first, second], detector_mask
