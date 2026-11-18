"""Transactional project snapshots and aligned detector frames."""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json
import sqlite3

import numpy as np

from .detector_io import array_bytes, mask_bytes, read_array
from .errors import Changed, Missing, WorkbenchError
from .state import default_state, read_state
from .values import MAX_FRAMES, label, mapping, token

MAX_PROJECTS = 64
MAX_DATA_BYTES = 2_048 * 1024 * 1024

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    revision INTEGER NOT NULL,
    state_json TEXT NOT NULL,
    imported_mask BLOB,
    created_at TEXT NOT NULL,
    changed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS frames (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    metadata_json TEXT NOT NULL,
    data BLOB NOT NULL,
    UNIQUE(project_id, position)
);
CREATE INDEX IF NOT EXISTS frames_project ON frames(project_id, position);
PRAGMA user_version = 1;
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _id():
    return uuid4().hex


def _encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class Frame:
    id: str
    position: int
    metadata: dict

    @property
    def title(self):
        return self.metadata["title"]

    def document(self):
        return {"id": self.id, "position": self.position, **self.metadata}


@dataclass(frozen=True)
class Project:
    id: str
    title: str
    width: int
    height: int
    revision: int
    state: dict
    frames: list
    imported: bool
    created_at: str
    changed_at: str

    @property
    def shape(self):
        return self.height, self.width

    @property
    def frame_ids(self):
        return [frame.id for frame in self.frames]

    def select(self, frame_id):
        active = frame_id or self.state["active_frame"]
        for frame in self.frames:
            if frame.id == active:
                return frame
        raise Missing("这张图像不在当前工程中。")

    def document(self):
        return {
            "id": self.id,
            "title": self.title,
            "width": self.width,
            "height": self.height,
            "revision": self.revision,
            "state": self.state,
            "frames": [frame.document() for frame in self.frames],
            "imported_mask": self.imported,
            "created_at": self.created_at,
            "changed_at": self.changed_at,
        }


@dataclass(frozen=True)
class DetectorSnapshot:
    project: Project
    frame: Frame
    counts: np.ndarray
    imported: np.ndarray | None
    reference: Frame | None = None
    reference_counts: np.ndarray | None = None


class Store:
    def __init__(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / "ringbench.sqlite3"
        with self.connection() as con:
            version = con.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError("The workspace database version is unsupported")
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA)

    @contextmanager
    def connection(self, write=False):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=10000")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if db.in_transaction:
                db.commit()
        except BaseException:
            if db.in_transaction:
                db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _project(con, pid):
        token(pid, "工程编号")
        row = con.execute(
            """
            SELECT id, title, width, height, revision, state_json, created_at,
                   changed_at, imported_mask IS NOT NULL AS has_mask
            FROM projects WHERE id = ?
            """, (pid,),
        ).fetchone()
        if row is None:
            raise Missing("未找到工程。")
        frames = con.execute(
            "SELECT id, position, metadata_json FROM frames WHERE project_id=? ORDER BY position",
            (pid,),
        ).fetchall()
        return Project(
            row["id"], row["title"], row["width"], row["height"], row["revision"],
            json.loads(row["state_json"]),
            [Frame(f["id"], f["position"], json.loads(f["metadata_json"])) for f in frames],
            bool(row["has_mask"]), row["created_at"], row["changed_at"],
        )

    @staticmethod
    def _counts(con, fid):
        row = con.execute("SELECT data FROM frames WHERE id = ?", (fid,)).fetchone()
        if row is None:
            raise Missing("未找到探测器帧。")
        return read_array(row["data"])

    @staticmethod
    def _mask(con, project):
        if not project.imported:
            return None
        raw = con.execute("SELECT imported_mask FROM projects WHERE id = ?", (project.id,)).fetchone()
        return read_array(raw[0]).astype(bool, copy=False)

    @staticmethod
    def _require_revision(project, expected):
        if project.revision != expected:
            raise Changed(project.revision)

    @staticmethod
    def _usage(con):
        frames = con.execute("SELECT COALESCE(SUM(LENGTH(data)), 0) FROM frames").fetchone()[0]
        masks = con.execute("SELECT COALESCE(SUM(LENGTH(imported_mask)), 0) FROM projects").fetchone()[0]
        return int(frames + masks)

    @staticmethod
    def _save_project(con, project, *, state=None, title=None):
        doc = project.state if state is None else state
        new_title = project.title if title is None else title
        con.execute(
            """
            UPDATE projects
            SET title = ?, state_json = ?, revision = revision + 1, changed_at = ?
            WHERE id = ?
            """, (new_title, _encode(doc), _now(), project.id),
        )

    @staticmethod
    def _insert_frame(con, pid, fid, capture, position, payload):
        con.execute(
            "INSERT INTO frames VALUES (?, ?, ?, ?, ?)",
            (fid, pid, position, _encode(capture.metadata()), payload),
        )

    def projects(self):
        with self.connection() as con:
            rows = con.execute(
                """
                SELECT p.id, p.title, p.width, p.height, p.changed_at, p.revision,
                       COUNT(f.id) AS frame_count
                FROM projects p JOIN frames f ON f.project_id = p.id
                GROUP BY p.id ORDER BY p.changed_at DESC, p.id DESC
                """
            )
            return [dict(row) for row in rows]

    def project(self, project_id):
        with self.connection() as con:
            con.execute("BEGIN")
            return self._project(con, project_id)

    def create(self, title, captures, *, state=None, original_ids=None, imported=None):
        title = label(title, "工程名")
        if not 1 <= len(captures) <= MAX_FRAMES:
            raise WorkbenchError(f"工程至少一帧，最多{MAX_FRAMES}帧。")
        height, width = captures[0].shape
        if any(capture.shape != (height, width) for capture in captures):
            raise WorkbenchError("同一工程的探测器帧必须拥有相同行列。")
        fid_list = [_id() for _ in captures]
        if state is None:
            state = default_state(width, height)
            state["active_frame"] = fid_list[0]
        else:
            state = _remap_frames(state, original_ids, fid_list)
        settings = read_state(state, width, height, fid_list)
        mask = None if imported is None else mask_bytes(imported)
        if imported is not None and imported.shape != (height, width):
            raise WorkbenchError("归档遮罩尺寸与探测器不一致。")
        payloads = [array_bytes(capture.data) for capture in captures]
        incoming_bytes = sum(map(len, payloads)) + (len(mask) if mask else 0)
        pid = _id()
        now = _now()
        with self.connection(write=True) as con:
            count = con.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
            if count >= MAX_PROJECTS or incoming_bytes + self._usage(con) > MAX_DATA_BYTES:
                raise WorkbenchError("本地工程空间已满，请先归档并移除不用的工程。", status=409)
            con.execute(
                "INSERT INTO projects VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)",
                (pid, title, width, height, _encode(settings), mask, now, now),
            )
            for pos, (fid, capture, data) in enumerate(zip(fid_list, captures, payloads)):
                self._insert_frame(con, pid, fid, capture, pos, data)
            return self._project(con, pid)

    def update(self, pid, expected, changes=None, *, title=None):
        changes = mapping(changes or {}, "设置变更")
        with self.connection(write=True) as con:
            project = self._project(con, pid)
            self._require_revision(project, expected)
            unknown = set(changes) - project.state.keys()
            if unknown:
                raise WorkbenchError("包含未知工程设置。")
            patch = dict(project.state, **changes)
            new_state = read_state(patch, project.width, project.height, project.frame_ids)
            new_title = None if title is None else label(title, "工程名")
            self._save_project(con, project, state=new_state, title=new_title)
            return self._project(con, pid)

    def add_frame(self, pid, expected, capture, *, select_added=True):
        encoded, fid = array_bytes(capture.data), _id()
        with self.connection(write=True) as con:
            project = self._project(con, pid)
            self._require_revision(project, expected)
            if project.shape != capture.shape:
                raise WorkbenchError("请为同一工程导入同尺寸、同一探测器坐标下的帧。")
            if len(project.frames) >= MAX_FRAMES or len(encoded) + self._usage(con) > MAX_DATA_BYTES:
                raise WorkbenchError("当前工程或工作区已达到帧容量。", status=409)
            pos = max(frame.position for frame in project.frames) + 1
            self._insert_frame(con, pid, fid, capture, pos, encoded)
            state = dict(project.state)
            if select_added:
                state["active_frame"] = fid
            self._save_project(con, project, state=state)
            return self._project(con, pid)

    def drop_frame(self, pid, fid, expected):
        with self.connection(write=True) as con:
            project = self._project(con, pid)
            self._require_revision(project, expected)
            removing = project.select(fid)
            if len(project.frames) == 1:
                raise WorkbenchError("工程需要保留至少一个探测器帧。")
            con.execute("DELETE FROM frames WHERE id = ? AND project_id = ?", (removing.id, pid))
            surviving = [frame.id for frame in project.frames if frame.id != removing.id]
            state = dict(project.state)
            if state["active_frame"] == removing.id:
                state["active_frame"] = surviving[0]
            if state["difference"]["reference"] == removing.id:
                state["difference"] = dict(state["difference"], reference=None)
            self._save_project(con, project, state=state)
            return self._project(con, pid)

    def rename_frame(self, pid, fid, expected, title):
        title = label(title, "图像名")
        with self.connection(write=True) as con:
            project = self._project(con, pid)
            self._require_revision(project, expected)
            frame = project.select(fid)
            metadata = dict(frame.metadata, title=title)
            con.execute("UPDATE frames SET metadata_json = ? WHERE id = ?", (_encode(metadata), fid))
            self._save_project(con, project)
            return self._project(con, pid)

    def replace_mask(self, pid, expected, imported):
        with self.connection(write=True) as con:
            project = self._project(con, pid)
            self._require_revision(project, expected)
            if imported is None:
                payload = None
            else:
                if imported.shape != project.shape:
                    raise WorkbenchError("遮罩尺寸不匹配。")
                payload = mask_bytes(imported)
                old_size = con.execute(
                    "SELECT COALESCE(LENGTH(imported_mask), 0) FROM projects WHERE id=?", (pid,),
                ).fetchone()[0]
                if len(payload) - old_size + self._usage(con) > MAX_DATA_BYTES:
                    raise WorkbenchError("工作区已达到数据容量。", status=409)
            con.execute("UPDATE projects SET imported_mask = ? WHERE id = ?", (payload, pid))
            self._save_project(con, project)
            return self._project(con, pid)

    def delete(self, pid, expected):
        with self.connection(write=True) as con:
            self._require_revision(self._project(con, pid), expected)
            con.execute("DELETE FROM projects WHERE id = ?", (pid,))

    def snapshot(self, pid, fid=None, revision=None, *, reference=False):
        with self.connection() as con:
            con.execute("BEGIN")
            project = self._project(con, pid)
            if revision is not None:
                self._require_revision(project, revision)
            frame = project.select(fid)
            ref_id = project.state["difference"]["reference"]
            ref = project.select(ref_id) if reference and ref_id is not None else None
            return DetectorSnapshot(
                project,
                frame,
                self._counts(con, frame.id),
                self._mask(con, project),
                ref,
                self._counts(con, ref.id) if ref and ref.id != frame.id else None,
            )

    def related_snapshot(self, pid, fid, extra_ids, revision=None):
        with self.connection() as con:
            con.execute("BEGIN")
            project = self._project(con, pid)
            if revision is not None:
                self._require_revision(project, revision)
            focus = project.select(fid)
            extra = []
            for chosen in extra_ids:
                if chosen is not None:
                    frame = project.select(chosen)
                    extra.append((frame, self._counts(con, frame.id)))
            snapshot = DetectorSnapshot(
                project, focus, self._counts(con, focus.id), self._mask(con, project),
            )
            return snapshot, extra

    def archive_snapshot(self, pid, revision=None):
        with self.connection() as con:
            con.execute("BEGIN")
            project = self._project(con, pid)
            if revision is not None:
                self._require_revision(project, revision)
            data = [
                (frame, con.execute("SELECT data FROM frames WHERE id=?", (frame.id,)).fetchone()[0])
                for frame in project.frames
            ]
            return project, data, self._mask(con, project)


def _remap_frames(state, original_ids, new_ids):
    if original_ids is None or len(original_ids) != len(new_ids):
        raise WorkbenchError("归档状态缺少其原探测器帧编号。")
    frame_map = dict(zip(original_ids, new_ids))
    if len(frame_map) != len(new_ids):
        raise WorkbenchError("归档探测器帧编号重复。")
    v = dict(mapping(state, "原工程设置"))
    original = v.get("active_frame")
    if original not in frame_map:
        raise WorkbenchError("原工程当前帧与归档不符。")
    v["active_frame"] = frame_map[original]
    diff = dict(mapping(v.get("difference"), "原工程差图"))
    reference = diff.get("reference")
    if reference is not None and reference not in frame_map:
        raise WorkbenchError("原工程参考帧与归档不符。")
    diff["reference"] = frame_map.get(reference)
    v["difference"] = diff
    return v
