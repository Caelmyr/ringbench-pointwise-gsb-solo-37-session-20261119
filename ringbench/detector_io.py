"""Read numeric detector captures and write interoperable numeric products."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import Path
import csv
import json
import math

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import WorkbenchError
from .values import MAX_IMAGE_BODY, image_size, label

PNG_MODES = {"1", "L", "I", "I;16", "I;16B", "I;16L", "F"}
MAX_NPY_HEADER = 2048


@dataclass(frozen=True)
class Capture:
    data: np.ndarray
    title: str
    source_name: str
    source_format: str
    source_dtype: str
    digest: str

    @property
    def shape(self):
        return self.data.shape

    def metadata(self):
        h, w = self.shape
        return {
            "title": self.title,
            "source_name": self.source_name,
            "source_format": self.source_format,
            "source_dtype": self.source_dtype,
            "digest": self.digest,
            "height": h,
            "width": w,
        }


def digest(data):
    return sha256(data).hexdigest()


def bounded_bytes(stream, maximum):
    payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise WorkbenchError("文件大于当前工作台允许的上传容量。", status=413)
    if not payload:
        raise WorkbenchError("文件为空。")
    return payload


def array_bytes(data):
    dest = BytesIO()
    np.save(dest, np.asarray(data, dtype="<f4"), allow_pickle=False)
    return dest.getvalue()


def mask_bytes(mask):
    dest = BytesIO()
    np.save(dest, np.asarray(mask, dtype=np.bool_), allow_pickle=False)
    return dest.getvalue()


def _numeric_dtype(dtype):
    return dtype.fields is None and dtype.kind in "biuf" and dtype.itemsize <= 8


def _npy_header(src):
    version = np.lib.format.read_magic(src)
    if version == (1, 0):
        reader = np.lib.format.read_array_header_1_0
    elif version == (2, 0):
        reader = np.lib.format.read_array_header_2_0
    else:
        raise WorkbenchError("仅支持 NPY v1 和 v2 数值文件。")
    shape, _, dtype = reader(src, max_header_size=MAX_NPY_HEADER)
    height, width = image_size(shape)
    if not _numeric_dtype(dtype):
        raise WorkbenchError("NPY 仅支持单通道实数或整数计数。")
    return height, width, dtype


def read_array(data):
    src = BytesIO(data)
    try:
        h, w, dtype = _npy_header(src)
        if len(data) != src.tell() + h * w * dtype.itemsize:
            raise WorkbenchError("NPY 头部的尺寸与文件长度不符。")
        src.seek(0)
        return np.load(src, allow_pickle=False, max_header_size=MAX_NPY_HEADER)
    except (ValueError, EOFError, OSError, OverflowError, TypeError) as exc:
        raise WorkbenchError("无法读取探测器 NPY 数值矩阵。") from exc


def _pillow_array(payload):
    try:
        with Image.open(BytesIO(payload)) as view:
            fmt = view.format
            if fmt not in {"PNG", "TIFF"}:
                raise WorkbenchError("请导入单帧灰度 PNG、TIFF 或二维 NPY。")
            image_size((view.height, view.width))
            if getattr(view, "n_frames", 1) != 1 or getattr(view, "is_animated", False):
                raise WorkbenchError("请从多页或多帧文件中先导出一个计数平面。")
            if view.mode not in PNG_MODES:
                raise WorkbenchError("文件是彩色图像，请选用原始单通道计数图。")
            if fmt == "TIFF" and view.getexif().get(274, 1) != 1:
                raise WorkbenchError("TIFF 需要按探测器原始行列方向存储。")
            return np.asarray(view).copy(), fmt
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise WorkbenchError("无法解码探测器图像。") from exc


def normalize_array(data):
    raw = np.asarray(data)
    image_size(raw.shape)
    if not _numeric_dtype(raw.dtype):
        raise WorkbenchError("采集数据需要每个像素一个实数。")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(raw, dtype="<f4").copy(order="C")
    if np.any(np.isfinite(raw) & ~np.isfinite(result)):
        raise WorkbenchError("文件中的计数超出 float32 可保存范围。")
    result[~np.isfinite(result)] = np.nan
    if not np.any(np.isfinite(result)):
        raise WorkbenchError("图像没有有限的计数像素。")
    return result


def capture_from_array(data, title, *, source_name=None, source_format="NPY", source_dtype=None):
    arr = normalize_array(data)
    h, w = arr.shape
    if h > 1 and w > 1 and not arr.flags.c_contiguous:
        raise WorkbenchError("无法为这个矩阵建立探测器行列。")
    return Capture(
        arr,
        label(title, "图像名"),
        label(source_name or "detector.npy", "来源文件", 140),
        source_format,
        source_dtype or str(data.dtype),
        digest(array_bytes(arr)),
    )


def upload_capture(stream, name, title=None):
    data = bounded_bytes(stream, MAX_IMAGE_BODY)
    name = (name or "detector").replace("\\", "/").split("/")[-1]
    if data.startswith(b"\x93NUMPY"):
        raw = read_array(data)
        fmt = "NPY"
    else:
        raw, fmt = _pillow_array(data)
    return capture_from_array(
        raw, title or Path(name).stem[:78] or "探测器图像",
        source_name=name[:140], source_format=fmt,
    )


def finite_summary(data, valid=None):
    sample = data[np.isfinite(data)] if valid is None else data[valid & np.isfinite(data)]
    if sample.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "percentiles": None,
        }
    q = np.percentile(sample, [1, 5, 25, 50, 75, 95, 99]).tolist()
    return {
        "count": int(sample.size),
        "min": float(sample.min()),
        "max": float(sample.max()),
        "mean": float(np.mean(sample, dtype=np.float64)),
        "percentiles": q,
    }


def json_bytes(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def csv_bytes(header, rows):
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(["" if value is None else value for value in row])
    return stream.getvalue().encode("utf-8")


def numbers_for_json(values):
    return [float(v) if math.isfinite(v) else None for v in values]


def png_bytes(pixels):
    out = BytesIO()
    Image.fromarray(np.ascontiguousarray(pixels)).save(out, format="PNG", optimize=False)
    return out.getvalue()
