"""Render shared detector and reduction previews without changing source counts."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .detector_io import png_bytes
from .errors import WorkbenchError

LUT_STOPS = {
    "ice": (
        (0.0, (10, 18, 34)), (0.26, (28, 45, 95)), (0.49, (35, 102, 139)),
        (0.73, (85, 167, 154)), (0.9, (195, 222, 141)), (1, (255, 248, 197)),
    ),
    "sun": (
        (0, (16, 8, 37)), (0.24, (74, 33, 89)), (0.49, (165, 55, 96)),
        (0.73, (229, 105, 65)), (0.91, (251, 184, 107)), (1, (255, 250, 211)),
    ),
    "gray": ((0, (13, 16, 22)), (1, (248, 248, 248))),
    "signed": (
        (0, (45, 124, 244)), (0.29, (33, 63, 122)), (0.5, (16, 24, 36)),
        (0.7, (132, 76, 51)), (1, (255, 187, 88)),
    ),
    "support": ((0, (31, 36, 48)), (0.5, (79, 90, 142)), (1, (115, 228, 172))),
}
MISSING_COLOR = np.array([18, 23, 34], dtype=np.uint8)
EXCLUDED_COLOR = np.array([139, 77, 132], dtype=np.uint8)
GUIDE_COLOR = (105, 243, 216, 230)
MAX_PREVIEW = 1152


@dataclass(frozen=True)
class Window:
    low: float
    high: float

    def scale(self, data):
        if self.high <= self.low:
            raise WorkbenchError("显示窗口需要正宽度。")
        return np.clip((data.astype(np.float64, copy=False) - self.low) / (self.high - self.low), 0, 1)

    def document(self):
        return {"lower": self.low, "upper": self.high}


def _lut(name):
    stops = LUT_STOPS[name]
    places, rgb = zip(*stops)
    queries = np.linspace(0, 1, 256)
    return np.array(
        [np.interp(queries, places, values) for values in zip(*rgb)],
        dtype=np.uint8,
    ).T


def choose_window(data, clean, display):
    if display.range_mode == "manual":
        return Window(display.lower, display.upper)
    accepted = data[clean]
    if accepted.size == 0:
        return Window(0, 1)
    if accepted.size > 160_000:
        order = np.linspace(0, accepted.size - 1, 160_000).astype(np.intp)
        accepted = accepted[order]
    bottom, top = (float(v) for v in np.percentile(accepted, [1, 99.4]))
    if top <= bottom:
        bottom, top = float(accepted.min()), float(accepted.max())
    if top <= bottom:
        epsilon = max(1, abs(bottom) * 1e-6)
        return Window(bottom - epsilon / 2, top + epsilon / 2)
    return Window(bottom, top)


def colors(data, window, name, stretch="linear"):
    finite = np.isfinite(data)
    scaled = window.scale(np.where(finite, data, window.low))
    if stretch == "sqrt":
        scaled = np.sqrt(scaled)
    if stretch == "asinh":
        scaled = np.arcsinh(7 * scaled) / np.arcsinh(7)
    rgb = _lut(name)[np.rint(scaled * 255).astype(np.uint8)]
    rgb[~finite] = MISSING_COLOR
    return rgb


def _tint_mask(img, codes):
    flagged = codes > 1
    striped = ((np.arange(img.shape[0])[:, None] + np.arange(img.shape[1])[None, :]) % 10) < 3
    under = flagged & striped
    rows = img[under].astype(np.float32) * 0.44 + EXCLUDED_COLOR * 0.56
    img[under] = np.rint(rows).astype(np.uint8)
    remaining = flagged & ~striped
    img[remaining] = np.rint(img[remaining] * 0.72).astype(np.uint8)


def _thumbnail(rgb, max_side):
    height, width = rgb.shape[:2]
    if max(height, width) <= max_side:
        return rgb
    scale = max_side / max(height, width)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.array(Image.fromarray(rgb).resize(target, Image.Resampling.BOX))


def detector_view(data, mask, display, *, max_side=MAX_PREVIEW):
    window = choose_window(data, mask.valid, display)
    rgb = colors(data, window, display.palette, display.stretch)
    if display.show_mask:
        _tint_mask(rgb, mask.codes)
    lo, hi = window.low, window.high
    finite = mask.valid & np.isfinite(data)
    summary = {
        **window.document(),
        "low_clipped": int(np.count_nonzero(finite & (data < lo))),
        "high_clipped": int(np.count_nonzero(finite & (data > hi))),
    }
    return png_bytes(_thumbnail(rgb, max_side)), summary


def mask_view(mask):
    out = np.zeros((*mask.codes.shape, 3), dtype=np.uint8)
    out[:] = [27, 67, 67]
    out[mask.codes == 1] = [26, 26, 40]
    out[mask.codes == 2] = [199, 90, 133]
    custom = mask.codes > 2
    if np.any(custom):
        indices = mask.codes[custom] - 3
        palette = np.array([[244, 145, 77], [166, 129, 239], [77, 157, 237], [228, 106, 97]])
        out[custom] = palette[indices % len(palette)]
    return png_bytes(_thumbnail(out, MAX_PREVIEW))


def reduction_view(cake, display, *, support=False, size=(920, 610)):
    if support:
        frac = cake.moments.fraction
        rgb = colors(frac, Window(0, 1), "support")
    else:
        data = cake.moments.mean
        window = choose_window(data, np.isfinite(data), display)
        rgb = colors(data, window, display.palette, display.stretch)
    return png_bytes(np.array(Image.fromarray(rgb[::-1]).resize(size, Image.Resampling.NEAREST)))


def difference_view(diff, spec, *, max_side=MAX_PREVIEW):
    det = diff.data
    if spec.window is not None:
        limit = spec.window
    else:
        finite = np.abs(det[diff.overlap])
        limit = max(1e-6, float(np.percentile(finite, 98.2)))
    rgb = colors(det, Window(-limit, limit), "signed")
    return png_bytes(_thumbnail(rgb, max_side)), limit


def _draw_marker(draw, x, y, half, color, width=1):
    draw.line((x - half, y, x + half, y), fill=color, width=width)
    draw.line((x, y - half, x, y + half), fill=color, width=width)


def _circle(draw, x, y, rx, ry, color, width=1):
    draw.ellipse((x-rx, y-ry, x+rx, y+ry), outline=color, width=width)


def annotated_detector(data, mask, settings, geometry, references):
    image = colors(data, choose_window(data, mask.valid, settings["display"]),
                   settings["display"].palette, settings["display"].stretch)
    if settings["display"].show_mask:
        _tint_mask(image, mask.codes)
    down = _thumbnail(image, MAX_PREVIEW)
    h, w = down.shape[:2]
    sy, sx = h / data.shape[0], w / data.shape[1]
    p = Image.fromarray(down)
    drawing = ImageDraw.Draw(p, "RGBA")
    x, y = (geometry.cx + 0.5) * sx, (geometry.cy + 0.5) * sy
    _draw_marker(drawing, x, y, 8, GUIDE_COLOR, 2)
    for ring in references:
        radius = ring["radius_px"]
        _circle(drawing, x, y, radius * sx, radius * sy, GUIDE_COLOR)
    for px, py in settings["marks"]:
        dx, dy = (px + 0.5) * sx, (py + 0.5) * sy
        _circle(drawing, dx, dy, 3, 3, (254, 204, 105, 242), 2)
    for spot in settings["spots"]:
        dx, dy = (spot["x"] + 0.5) * sx, (spot["y"] + 0.5) * sy
        radius = settings["aperture"]["radius"]
        _circle(drawing, dx, dy, max(3, radius * sx), max(3, radius * sy), (141, 248, 238, 230))
    return png_bytes(np.asarray(p))


def draw_curve_png(curve, *, bar="mean", width=880, height=420):
    ys = curve.moments.mean if bar == "mean" else curve.moments.counts.astype(float)
    xs = curve.axis.centers
    image = Image.new("RGB", (width, height), "#101828")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 68, width - 28, 30, height - 52
    valid = np.isfinite(ys)
    low = float(np.min(ys[valid])) if np.any(valid) else 0.0
    high = float(np.max(ys[valid])) if np.any(valid) else 1.0
    if high == low:
        high = low + max(1.0, abs(low) * 1e-6)
    axis_min, axis_max = curve.axis.low, curve.axis.high
    for amount in np.linspace(0, 1, 6):
        y = round(bottom - amount * (bottom - top))
        draw.line((left, y, right, y), fill="#293448")
        draw.text((9, y - 6), f"{low + amount * (high - low):.4g}", fill="#b5c3d6")
        x = round(left + amount * (right - left))
        draw.line((x, top, x, bottom), fill="#293448")
        draw.text((x - 14, bottom + 8), f"{axis_min + amount * (axis_max - axis_min):.4g}", fill="#b5c3d6")
    draw.text((left, 9), "MEAN COUNTS" if bar == "mean" else "VALID PIXELS", fill="#b5c3d6")
    draw.text((right - 88, height - 22), f"Axis: {curve.axis.unit}", fill="#b5c3d6")
    past = None
    for x, y, ok in zip(xs, ys, valid):
        if not ok:
            past = None
            continue
        p = (
            left + (x - axis_min) / (axis_max - axis_min) * (right - left),
            bottom - (y - low) / (high - low) * (bottom - top),
        )
        if past is not None:
            draw.line((*past, *p), fill="#61dec2", width=2)
        else:
            draw.point(p, fill="#61dec2")
        past = p
    return png_bytes(np.asarray(image))
