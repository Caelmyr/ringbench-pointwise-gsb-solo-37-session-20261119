"""Comparable radial measurements across aligned detector exposures."""
from dataclasses import dataclass

import numpy as np

from .detector_io import csv_bytes, numbers_for_json
from .errors import WorkbenchError
from .reduction import integrate_profile
from .values import items, mapping, optional_ordered, selection, token, unique

COMPARISON_MODES = ("raw", "area")
SUPPORT_MODES = ("common", "per_frame")


@dataclass(frozen=True)
class SeriesSettings:
    frames: tuple
    support: str
    normalization: str
    anchor: tuple | None

    @classmethod
    def read(cls, value):
        doc = mapping(value, "径向谱比较")
        ids = tuple(unique(
            [token(v, "比较曝光编号") for v in items(doc.get("frames"), "曝光", minimum=2, maximum=6)],
            "比较曝光",
        ))
        support = selection(doc.get("support", "common"), SUPPORT_MODES, "比较像素支撑")
        norm = selection(doc.get("normalization", "raw"), COMPARISON_MODES, "径向谱缩放")
        anchor = optional_ordered(doc.get("anchor"), "谱面积对照区间", 0)
        if norm == "area" and anchor is None:
            raise WorkbenchError("比较曲线形状时请选择谱面积归一的径向区间。")
        return cls(ids, support, norm, anchor)


@dataclass(frozen=True)
class Compared:
    frame: object
    curve: object
    values: np.ndarray
    coefficient: float
    anchor_area: float | None

    def document(self):
        return {
            "frame": self.frame.document(),
            "values": numbers_for_json(self.values),
            "pixel_support": self.curve.moments.counts.tolist(),
            "coefficient": self.coefficient,
            "anchor_area": self.anchor_area,
        }


@dataclass(frozen=True)
class ProfileSeries:
    axis: object
    frames: list
    support: np.ndarray
    spec: SeriesSettings
    reference: str

    def differences(self):
        ref = self.frames[0].values
        result = []
        for measured in self.frames[1:]:
            valid = np.isfinite(ref) & np.isfinite(measured.values)
            difference = np.where(valid, measured.values - ref, np.nan)
            if np.count_nonzero(valid) > 0:
                clean = difference[valid]
                rms = float(np.sqrt(np.mean(clean ** 2)))
                bias = float(np.mean(clean))
                ixs = np.flatnonzero(valid)
                peak_index = int(ixs[np.argmax(np.abs(clean))])
                peak = {
                    "position": float(self.axis.centers[peak_index]),
                    "difference": float(difference[peak_index]),
                }
            else:
                rms, bias, peak = None, None, None
            result.append({
                "frame": measured.frame.id,
                "title": measured.frame.title,
                "shared_bins": int(valid.sum()),
                "signed_mean_difference": bias,
                "rms_difference": rms,
                "max_difference": peak,
            })
        return result

    def document(self):
        return {
            "axis": self.axis.document(),
            "frames": [p.document() for p in self.frames],
            "common_pixels": self.support.tolist(),
            "reference": self.reference,
            "support_mode": self.spec.support,
            "normalization": self.spec.normalization,
            "anchor": self.spec.anchor,
            "metrics": self.differences(),
        }

    def csv(self):
        return csv_bytes(
            [
                "frame_id", f"radial_low_{self.axis.unit}", f"radial_high_{self.axis.unit}",
                "original_mean", "compared_mean", "pixel_support", "common_pixel_support", "multiplier",
            ],
            self._rows(),
        )

    def _rows(self):
        for p in self.frames:
            yield from zip(
                [p.frame.id] * self.axis.cells,
                self.axis.edges[:-1],
                self.axis.edges[1:],
                numbers_for_json(p.curve.moments.mean),
                numbers_for_json(p.values),
                p.curve.moments.counts,
                self.support,
                [p.coefficient] * self.axis.cells,
            )


def _overlap_lengths(edges, anchor):
    return np.maximum(np.minimum(edges[1:], anchor[1]) - np.maximum(edges[:-1], anchor[0]), 0)


def _normalization(curves, spec):
    if spec.normalization == "raw":
        return [(1.0, None)] * len(curves)
    axis = curves[0].axis
    windows = _overlap_lengths(axis.edges, spec.anchor)
    if not np.any(windows > 0):
        raise WorkbenchError("谱面积对照区间未覆盖当前径向曲线。")
    shared = windows > 0
    for item in curves:
        shared &= np.isfinite(item.moments.mean)
    if np.count_nonzero(shared) < 3:
        raise WorkbenchError("归一范围中至少需要三个所有曝光都能测得的径向箱。")
    areas = [float(np.dot(item.moments.mean[shared], windows[shared])) for item in curves]
    if any(area <= 0 for area in areas):
        raise WorkbenchError("对照区的积分面积必须为正；请选择有明确散射峰的计数范围。")
    return [(areas[0] / area, area) for area in areas]


def compare_profile_series(related, common, valid_sets, g, profile, settings):
    if not np.any(common) and settings.support == "common":
        raise WorkbenchError("所选曝光不存在共同可用的探测器像素。")
    curves = [
        integrate_profile(data, common if settings.support == "common" else valid, g, profile)
        for (_, data), valid in zip(related, valid_sets)
    ]
    if settings.support == "common":
        common_support = curves[0].moments.counts
    else:
        common_support = integrate_profile(related[0][1], common, g, profile).moments.counts
    weights = _normalization(curves, settings)
    compared = [
        Compared(frame, curve, curve.moments.mean * scale, scale, area)
        for (frame, _), curve, (scale, area) in zip(related, curves, weights)
    ]
    return ProfileSeries(curves[0].axis, compared, common_support, settings, related[0][0].id)
