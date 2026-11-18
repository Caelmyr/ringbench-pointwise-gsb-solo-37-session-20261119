"""Portable RingBench archives with checked array identities and replay settings."""
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import PurePosixPath
import re
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np

from .detector_io import capture_from_array, digest, json_bytes, mask_bytes, read_array
from .errors import WorkbenchError
from .masks import read_imported_mask
from .values import (
    MAX_ARCHIVE_TOTAL, MAX_FRAMES, MAX_PIXELS, integer, items, label, mapping, token,
)

ARCHIVE_FORMAT = "ringbench-workspace"
ARCHIVE_VERSION = 1
MANIFEST = "workspace.json"
MASK_PATH = "detector/excluded.npy"
MAX_MANIFEST = 256 * 1024
HASH = re.compile(r"[0-9a-f]{64}")
SOURCE_FORMATS = ("PNG", "TIFF", "NPY", "synthetic", "derived")
MAX_FRAME_BYTES = MAX_PIXELS * 4 + 1024


@dataclass(frozen=True)
class Archive:
    title: str
    captures: list
    original_ids: list
    settings: dict
    excluded: np.ndarray | None

    def install(self, store, title=None):
        return store.create(
            self.title if title is None else title,
            self.captures,
            state=self.settings,
            original_ids=self.original_ids,
            imported=self.excluded,
        )


def frame_path(index):
    return f"detector/frame-{index + 1:02d}.npy"


def pack(project, arrays, imported):
    stream = BytesIO()
    frames = []
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=4) as zipout:
        for index, (frame, data) in enumerate(arrays):
            path = frame_path(index)
            zipout.writestr(path, data)
            frames.append({
                **frame.document(), "file": path, "bytes": len(data),
                "normalized_sha256": digest(data),
            })
        mask_doc = None
        if imported is not None:
            array = mask_bytes(imported)
            zipout.writestr(MASK_PATH, array)
            mask_doc = {"file": MASK_PATH, "bytes": len(array), "sha256": digest(array)}
        manifest = {
            "format": ARCHIVE_FORMAT,
            "version": ARCHIVE_VERSION,
            "title": project.title,
            "coordinate_convention": "zero_based_pixel_centers_y_down",
            "width": project.width,
            "height": project.height,
            "created_at": project.created_at,
            "exported_revision": project.revision,
            "frames": frames,
            "settings": project.state,
            "detector_mask": mask_doc,
        }
        zipout.writestr(MANIFEST, json_bytes(manifest))
    return stream.getvalue()


def _unambiguous_pairs(entries):
    out = {}
    for k, v in entries:
        if k in out:
            raise WorkbenchError("归档清单中存在重复字段。")
        out[k] = v
    return out


def _invalid_json_constant(value):
    raise WorkbenchError(f"归档清单中出现非法数值 {value}。")


class ArchiveReader:
    def __init__(self, zipdata):
        self.zipdata = zipdata
        self.entries = {}
        self.read_members = set()
        for info in zipdata.infolist():
            self._register(info)
        if not 2 <= len(self.entries) <= MAX_FRAMES + 2:
            raise WorkbenchError("归档条目数量不支持。")
        if sum(item.file_size for item in self.entries.values()) > MAX_ARCHIVE_TOTAL:
            raise WorkbenchError("归档展开后数据大于工作台容量。")

    def _register(self, info):
        name = info.filename
        p = PurePosixPath(name)
        if (
            p.is_absolute() or ".." in p.parts or str(p) != name or "\\" in name
            or info.is_dir() or name in self.entries or info.flag_bits & 1
            or info.compress_type not in (ZIP_STORED, ZIP_DEFLATED)
        ):
            raise WorkbenchError("归档包含不支持的路径、重复文件或加密内容。")
        self.entries[name] = info

    def payload(self, path, max_size, *, size=None, sha=None):
        info = self.entries.get(path)
        if info is None or not 0 < info.file_size <= max_size:
            raise WorkbenchError(f"归档中缺失或超出读取限制的条目 {path}。")
        if size is not None and info.file_size != size:
            raise WorkbenchError(f"归档文件长度与清单不符 {path}。")
        with self.zipdata.open(info) as src:
            data = src.read(max_size + 1)
        if len(data) != info.file_size:
            raise WorkbenchError("归档文件的展开长度不符。")
        if sha is not None and digest(data) != sha:
            raise WorkbenchError(f"归档校验失败 {path}。")
        self.read_members.add(path)
        return data

    def manifest(self):
        raw = self.payload(MANIFEST, MAX_MANIFEST)
        try:
            obj = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unambiguous_pairs,
                parse_constant=_invalid_json_constant,
            )
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise WorkbenchError("无法读取 UTF-8 工程清单。") from exc
        v = mapping(obj, "归档清单")
        if v.get("format") != ARCHIVE_FORMAT or type(v.get("version")) is not int or v["version"] != ARCHIVE_VERSION:
            raise WorkbenchError("这个 ZIP 不是受支持的 RingBench 工程版本。")
        if v.get("coordinate_convention") != "zero_based_pixel_centers_y_down":
            raise WorkbenchError("该归档未声明已支持的探测器坐标约定。")
        return v

    def frame(self, index, entry, shape):
        v = mapping(entry, "已归档探测器帧")
        fid = token(v.get("id"), "原帧编号")
        path = frame_path(index)
        if v.get("file") != path:
            raise WorkbenchError("归档探测器帧编号与保存路径不符。")
        saved_sha = _sha(v.get("normalized_sha256"))
        expected_sha = _sha(v.get("digest"))
        length = integer(v.get("bytes"), "帧文件长度", 1, MAX_FRAME_BYTES)
        raw = self.payload(path, MAX_FRAME_BYTES, size=length, sha=saved_sha)
        data = read_array(raw)
        if data.shape != shape or data.dtype != np.dtype("<f4") or saved_sha != expected_sha:
            raise WorkbenchError("归档计数文件的坐标、数据类型或标识与清单不一致。")
        if v.get("source_format") not in SOURCE_FORMATS:
            raise WorkbenchError("归档中的采集来源类型不支持。")
        capture = capture_from_array(
            data,
            label(v.get("title"), "帧名称"),
            source_name=label(v.get("source_name"), "原文件名", 140),
            source_format=v["source_format"],
            source_dtype=label(v.get("source_dtype"), "采集数据类型", 40),
        )
        if capture.digest != saved_sha:
            raise WorkbenchError("归档计数平面的规范化哈希不匹配。")
        return fid, capture

    def mask(self, entry, shape):
        if entry is None:
            return None
        v = mapping(entry, "已归档二值遮罩")
        if v.get("file") != MASK_PATH:
            raise WorkbenchError("二值遮罩没有保存到约定目录。")
        maximum = MAX_PIXELS + 1024
        length = integer(v.get("bytes"), "遮罩文件大小", 1, maximum)
        raw = self.payload(MASK_PATH, maximum, size=length, sha=_sha(v.get("sha256")))
        return read_imported_mask(raw, shape)

    def complete(self):
        if self.read_members != self.entries.keys():
            raise WorkbenchError("工程归档包含清单之外的条目。")


def _sha(value):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise WorkbenchError("归档中缺少有效的 SHA-256 校验值。")
    return value


def unpack(payload):
    try:
        with ZipFile(BytesIO(payload)) as z:
            reader = ArchiveReader(z)
            v = reader.manifest()
            width = integer(v.get("width"), "已保存探测器宽", 16, 4096)
            height = integer(v.get("height"), "已保存探测器高", 16, 4096)
            if height * width > MAX_PIXELS:
                raise WorkbenchError("已保存探测器尺寸超出支持范围。")
            shape = height, width
            frames = items(v.get("frames"), "归档探测器帧", minimum=1, maximum=MAX_FRAMES)
            decoded = [reader.frame(i, frame, shape) for i, frame in enumerate(frames)]
            ids, captures = map(list, zip(*decoded))
            excluded = reader.mask(v.get("detector_mask"), shape)
            reader.complete()
            return Archive(
                label(v.get("title"), "归档工程名"), captures, ids,
                mapping(v.get("settings"), "归档工程设置"), excluded,
            )
    except (BadZipFile, OSError, OverflowError, EOFError, RuntimeError) as exc:
        raise WorkbenchError("无法解压并校验工程归档。") from exc
