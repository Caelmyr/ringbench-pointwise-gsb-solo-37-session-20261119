"""Read-only integrity check before sharing a local experiment."""
from dataclasses import dataclass

import numpy as np

from .detector_io import digest, read_array
from .errors import WorkbenchError
from .state import read_state


@dataclass(frozen=True)
class Integrity:
    workspace: str
    revision: int
    frames: list
    mask: dict
    settings: str

    def document(self):
        return {
            "project": self.workspace,
            "revision": self.revision,
            "frame_checks": self.frames,
            "exclusion_plane": self.mask,
            "settings": self.settings,
            "valid": self.settings == "valid" and all(f["ok"] for f in self.frames) and self.mask["ok"],
        }


def check_integrity(project, arrays, imported):
    shape = project.shape
    entries = []
    for frame, encoded in arrays:
        record = {"title": frame.title, "id": frame.id, "size_bytes": len(encoded)}
        if digest(encoded) != frame.metadata["digest"]:
            entries.append({**record, "ok": False, "reason": "sha256_mismatch"})
            continue
        try:
            result = read_array(encoded)
        except WorkbenchError:
            entries.append({**record, "ok": False, "reason": "invalid_array"})
            continue
        finite = int(np.count_nonzero(np.isfinite(result)))
        ok = result.shape == shape and result.dtype == np.dtype("<f4") and finite > 0
        entries.append({
            **record, "ok": ok, "reason": None if ok else "detector_shape_or_values",
            "finite_pixels": finite, "missing_count_values": int(result.size - finite),
        })
    if imported is None:
        mask = {"present": False, "ok": not project.imported}
    else:
        mask = {
            "present": True,
            "ok": project.imported and imported.shape == shape,
            "excluded": int(np.count_nonzero(imported)),
        }
    try:
        read_state(project.state, project.width, project.height, project.frame_ids)
        settings = "valid"
    except WorkbenchError:
        settings = "invalid_saved_settings"
    return Integrity(project.id, project.revision, entries, mask, settings)
