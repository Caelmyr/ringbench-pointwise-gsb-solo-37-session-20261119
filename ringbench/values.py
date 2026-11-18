"""Validation shared by API requests and restored workspaces."""
import math
import re

from .errors import WorkbenchError

MAX_SIDE = 4096
MAX_PIXELS = 4_194_304
MAX_FRAMES = 8
MAX_MASKS = 32
MAX_MARKS = 256
MAX_SPOTS = 128
MAX_BODY = 142 * 1024 * 1024
MAX_ARCHIVE_TOTAL = MAX_FRAMES * (MAX_PIXELS * 4 + 1024) + MAX_PIXELS + 1024 * 1024
MAX_IMAGE_BODY = MAX_PIXELS * 8 + 1024 * 1024
TOKEN_PATTERN = re.compile(r"[0-9a-f]{32}")
FILENAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]")


def mapping(value, field):
    if not isinstance(value, dict):
        raise WorkbenchError(f"{field}必须为对象。")
    return value


def selection(value, options, field):
    if value not in options:
        raise WorkbenchError(f"{field}的选项无效。")
    return value


def boolean(value, field):
    if not isinstance(value, bool):
        raise WorkbenchError(f"{field}必须是布尔值。")
    return value


def number(value, field, low=None, high=None, *, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise WorkbenchError(f"{field}必须为有限数字。")
    try:
        v = float(value)
    except (ValueError, TypeError, OverflowError):
        raise WorkbenchError(f"{field}必须为有限数字。") from None
    if not math.isfinite(v):
        raise WorkbenchError(f"{field}必须为有限数字。")
    if low is not None and v < low:
        raise WorkbenchError(f"{field}不能小于{low:g}。")
    if high is not None and v > high:
        raise WorkbenchError(f"{field}不能大于{high:g}。")
    return v


def integer(value, field, low=None, high=None):
    v = number(value, field, low, high)
    if not v.is_integer():
        raise WorkbenchError(f"{field}必须为整数。")
    return int(v)


def label(value, field="名称", max_length=80):
    if not isinstance(value, str):
        raise WorkbenchError(f"{field}必须为文字。")
    v = value.strip()
    if not v or len(v) > max_length or any(ord(c) < 32 for c in v):
        raise WorkbenchError(f"{field}需要1到{max_length}个可见字符。")
    return v


def items(value, field, *, minimum=0, maximum):
    if not isinstance(value, (list, tuple)) or not minimum <= len(value) <= maximum:
        raise WorkbenchError(f"{field}的数量应在{minimum}到{maximum}之间。")
    return value


def token(value, field):
    if not isinstance(value, str) or not TOKEN_PATTERN.fullmatch(value):
        raise WorkbenchError(f"{field}无效。")
    return value


def filename(value, fallback):
    safe = FILENAME_PATTERN.sub("-", value.strip())[:90].strip("-.")
    return safe or fallback


def image_size(shape):
    if len(shape) != 2:
        raise WorkbenchError("探测器文件必须是一个二维计数矩阵。")
    height, width = shape
    if not (
        16 <= width <= MAX_SIDE and 16 <= height <= MAX_SIDE
        and width * height <= MAX_PIXELS
    ):
        raise WorkbenchError("探测器边长应在16至4096之间，像素总数最多4194304。")
    return int(height), int(width)


def pair(value, field, bounds=None):
    v = items(value, field, minimum=2, maximum=2)
    if bounds is None:
        bounds = (-MAX_SIDE, 2 * MAX_SIDE, -MAX_SIDE, 2 * MAX_SIDE)
    x = number(v[0], f"{field} X", bounds[0], bounds[1])
    y = number(v[1], f"{field} Y", bounds[2], bounds[3])
    return x, y


def xy_bounds(width, height, margin=0):
    return (-margin, width - 1 + margin, -margin, height - 1 + margin)


def ordered(value, field, low=None, high=None):
    a, b = items(value, field, minimum=2, maximum=2)
    start = number(a, field, low, high)
    end = number(b, field, low, high)
    if end <= start:
        raise WorkbenchError(f"{field}的结束值必须大于起始值。")
    return start, end


def optional_ordered(value, field, low=None, high=None):
    if value is None:
        return None
    return ordered(value, field, low, high)


def unique(values, field):
    if len(set(values)) != len(values):
        raise WorkbenchError(f"{field}包含重复项。")
    return values
