import sys, tempfile
sys.path.insert(0, "/workspace")
import numpy as np
from ringbench.web import create_app
from ringbench.detector_io import capture_from_array

rng = np.random.default_rng(1)
H, W = 220, 200
img = 100 + rng.normal(0, 3, (H, W)); img[20:200, 70] *= 0.3
cap = capture_from_array(img, "曝光")
with tempfile.TemporaryDirectory() as d:
    app = create_app(d); store = app.extensions["ringbench_store"]
    p = store.create("缓存", [cap]); pid = p.id
    wb = app.extensions["ringbench_workbench"]; fid = p.frames[0].id
    r1 = wb.column_defects(pid, fid, p.revision)[1]
    assert r1["count"] == 1
    # repeated call at same revision is served (cache key stable)
    assert wb.column_defects(pid, fid, p.revision)[1]["count"] == 1
    # mask the whole column -> new revision must no longer suggest it
    p2 = store.update(pid, p.revision, {"masks": [{
        "kind": "rectangle", "title": "矩形坏区 1", "enabled": True,
        "a": [70, 0], "b": [70, H - 1]}]})
    r2 = wb.column_defects(pid, fid, p2.revision)[1]
    assert r2["count"] == 0, "masked column must not be suggested at the new revision"
    print("revision-keyed cache invalidates correctly after mask change")
    print("\nCACHE CHECK PASSED")
