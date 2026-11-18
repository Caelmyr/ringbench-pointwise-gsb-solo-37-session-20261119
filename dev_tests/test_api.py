import sys, tempfile, os
sys.path.insert(0, "/workspace")
import numpy as np
from ringbench.web import create_app
from ringbench.detector_io import capture_from_array

rng = np.random.default_rng(3)
H, W = 260, 220
yy, xx = np.mgrid[0:H, 0:W]
img = 110 + 0.04*xx + 0.02*yy + rng.normal(0, 3.5, (H, W))
img[110:118, :] += 230.0          # bright ring
# half-frame low column with ring crossing
img[15:110, 50] *= 0.35
img[118:210, 50] *= 0.35
# a normal-reading gap splits this one into two
img[20:90, 120] *= 0.35
img[150:230, 120] *= 0.35
capture = capture_from_array(img, "主曝光")

with tempfile.TemporaryDirectory() as d:
    app = create_app(d)
    client = app.test_client()
    # create project directly via store
    from ringbench.store import Store
    store = app.extensions["ringbench_store"]
    project = store.create("接口测试", [capture])
    pid = project.id
    rev = project.revision
    fid = project.frames[0].id
    resp = client.get(f"/api/projects/{pid}/column-defects?frame={fid}&revision={rev}")
    assert resp.status_code == 200, resp.data
    payload = resp.get_json()
    print("count:", payload["count"], "pixels:", payload["pixel_count"])
    for r in payload["regions"]:
        print(" ", r)
    xs = {r["x0"] for r in payload["regions"]}
    assert 50 in xs and 120 in xs
    # column 50 bridged into one segment across the ring
    c50 = [r for r in payload["regions"] if r["x0"] == 50]
    assert len(c50) == 1 and c50[0]["y0"] <= 17 and c50[0]["y1"] >= 208
    # column 120 split into two segments
    c120 = [r for r in payload["regions"] if r["x0"] == 120]
    assert len(c120) == 2
    print("thresholds:", payload["thresholds"])
    print("\nAPI END-TO-END CHECK PASSED")
