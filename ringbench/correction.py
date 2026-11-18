"""Relative detector dark and flat response correction on one detector grid."""
from dataclasses import dataclass

import numpy as np

from .detector_io import array_bytes, csv_bytes, finite_summary
from .errors import WorkbenchError
from .values import boolean, mapping, number, token

INVALID_SIGNAL = 1
INVALID_DARK = 2
INVALID_FLAT = 4
WEAK_RESPONSE = 8
UNSTABLE_GAIN = 16
RANGE_OVERFLOW = 32
NO_DETECTOR = 64
FLAG_LABELS = {
    INVALID_SIGNAL: "主曝光计数缺失",
    INVALID_DARK: "暗场计数缺失",
    INVALID_FLAT: "平场计数缺失",
    WEAK_RESPONSE: "平场减暗后的计数不够",
    UNSTABLE_GAIN: "校正增益过高",
    RANGE_OVERFLOW: "校正结果超出数值范围",
    NO_DETECTOR: "探测器遮罩排除",
}


@dataclass(frozen=True)
class ResponseSettings:
    dark: str | None
    flat: str | None
    minimum_signal: float
    maximum_gain: float
    keep_negative: bool

    @classmethod
    def read(cls, v):
        doc = mapping(v, "响应校正设置")
        dark = doc.get("dark")
        flat = doc.get("flat")
        if dark is None and flat is None:
            raise WorkbenchError("请至少选择一帧暗场或平场。")
        dark_id = token(dark, "暗场编号") if dark is not None else None
        flat_id = token(flat, "平场编号") if flat is not None else None
        if dark_id is not None and dark_id == flat_id:
            raise WorkbenchError("暗帧和平场不能使用相同曝光。")
        return cls(
            dark_id,
            flat_id,
            number(doc.get("minimum_signal", 1), "平场净计数下限", 0.00001, 1e30),
            number(doc.get("maximum_gain", 6), "增益上限", 1, 1000),
            boolean(doc.get("keep_negative", True), "保留负校正计数"),
        )

    @property
    def method(self):
        if self.dark and self.flat:
            return "dark_flat"
        if self.dark:
            return "dark_only"
        return "flat_only"


@dataclass(frozen=True)
class Correction:
    counts: np.ndarray
    gain: np.ndarray | None
    flags: np.ndarray
    pivot: float | None
    mode: str
    clipped_negatives: int

    @property
    def valid(self):
        return self.flags == 0

    def document(self):
        return {
            "method": self.mode,
            "normalization_reference": self.pivot,
            "usable": int(self.valid.sum()),
            "candidate": int(self.flags.size),
            "negative_values_clamped": self.clipped_negatives,
            "counts": finite_summary(self.counts, self.valid),
            "gain": finite_summary(self.gain, self.valid) if self.gain is not None else None,
            "flags": [
                {"id": key, "reason": reason, "pixels": int(np.count_nonzero(self.flags & key))}
                for key, reason in FLAG_LABELS.items()
            ],
        }

    def npy(self):
        return array_bytes(self.counts)

    def gain_npy(self):
        if self.gain is None:
            raise WorkbenchError("仅有暗场减法时不会生成平场增益。")
        return array_bytes(self.gain)

    def flags_csv(self):
        return csv_bytes(
            ["code", "meaning", "pixel_count"],
            (
                (key, reason, int(np.count_nonzero(self.flags & key)))
                for key, reason in FLAG_LABELS.items()
            ),
        )


def _correct_inputs(signal, dark, flat, signal_support, reference_support):
    dimensions = signal.shape
    for part in (dark, flat, signal_support, reference_support):
        if part is not None and part.shape != dimensions:
            raise WorkbenchError("校正使用的原始帧和遮罩必须具有相同的探测器尺寸。")
    flags = np.zeros(dimensions, dtype=np.uint8)
    flags[~np.isfinite(signal)] |= INVALID_SIGNAL
    if dark is not None:
        flags[~np.isfinite(dark)] |= INVALID_DARK
    if flat is not None:
        flags[~np.isfinite(flat)] |= INVALID_FLAT
    flags[~signal_support] |= NO_DETECTOR
    if reference_support is not None:
        flags[~reference_support] |= NO_DETECTOR
    return flags


def _flat_gain(flat, dark, flags, spec):
    net = flat.astype(np.float64)
    if dark is not None:
        net -= dark
    reference_bad = (flags & (INVALID_DARK | INVALID_FLAT | NO_DETECTOR)) > 0
    workable = ~reference_bad & np.isfinite(net)
    strong = workable & (net >= spec.minimum_signal)
    flags[workable & ~strong] |= WEAK_RESPONSE
    if np.count_nonzero(strong) < 12:
        raise WorkbenchError("平场参考的有效净计数太少，无法得到相对灵敏度。")
    pivot = float(np.median(net[strong]))
    gain = np.divide(pivot, net, out=np.full(net.shape, np.nan), where=strong)
    poor = strong & (gain > spec.maximum_gain)
    flags[poor] |= UNSTABLE_GAIN
    return gain, pivot


def correct(signal, dark, flat, signal_support, reference_support, spec):
    flags = _correct_inputs(signal, dark, flat, signal_support, reference_support)
    corrected = signal.astype(np.float64)
    if dark is not None:
        corrected -= dark
    gain, pivot = None, None
    if flat is not None:
        gain, pivot = _flat_gain(flat, dark, flags, spec)
        corrected *= gain
    valid = flags == 0
    overflow = valid & ((np.abs(corrected) > np.finfo(np.float32).max) | ~np.isfinite(corrected))
    flags[overflow] |= RANGE_OVERFLOW
    valid = flags == 0
    if not np.any(valid):
        raise WorkbenchError("校正后没有可用的探测器计数。")
    clamped = 0
    if not spec.keep_negative:
        negatives = valid & (corrected < 0)
        clamped = int(np.count_nonzero(negatives))
        corrected[negatives] = 0
    corrected[~valid] = np.nan
    if gain is not None:
        gain = np.where(valid, gain, np.nan).astype(np.float32)
    return Correction(corrected.astype(np.float32), gain, flags, pivot, spec.method, clamped)
