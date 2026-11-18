import sys, tempfile
sys.path.insert(0, "/workspace")
import numpy as np
from ringbench.web import create_app
from ringbench.detector_io import capture_from_array
from ringbench.masks import mask_layers, evaluate
from ringbench.geometry import Geometry

rng = np.random.default_rng(5)
H, W = 200, 180
img = 100 + rng.normal(0, 3, (H, W))
img[20:180, 70] *= 0.3
cap = capture_from_array(img, "曝光")
with tempfile.TemporaryDirectory() as d:
    app = create_app(d)
    store = app.extensions["ringbench_store"]
    p = store.create("遮罩应用", [cap])
    pid, rev, fid = p.id, p.revision, p.frames[0].id
    client = app.test_client()
    sug = client.get(f"/api/projects/{pid}/column-defects?frame={fid}&revision={rev}").get_json()
    # Mimic the frontend: append rectangle masks from suggestions.
    serial = len(p.state["masks"])
    added = [{
        "kind": "rectangle", "title": f"纵向坏区 {i+1}", "enabled": True,
        "a": [r["x0"], r["y0"]], "b": [r["x1"], r["y1"] - 1],
    } for i, r in enumerate(sug["regions"])]
    # also keep a pre-existing intensity mask to confirm it still validates
    existing = [{"kind": "intensity", "title": "饱和", "enabled": True, "low": None, "high": 4094}]
    resp = client.patch(f"/api/projects/{pid}", json={
        "revision": rev, "settings": {"masks": existing + added},
    })
    assert resp.status_code == 200, resp.data
    newp = store.project(pid)
    layers = mask_layers(newp.state["masks"], W, H)
    print("layers:", [(l.kind, l.title, l.enabled, l.args) for l in layers])
    g = Geometry.read(newp.state["geometry"], W, H)
    counts = store.snapshot(pid).counts
    result = evaluate(counts, g, layers, None)
    # pixels inside the suggested column must now be excluded
    r = sug["regions"][0]
    inside = result.codes[(r["y0"] + r["y1"]) // 2, r["x0"]]
    print("code at suggested column midpoint:", inside, "(nonzero = excluded)")
    assert inside > 0
    print("summary:", result.summary()["excluded"], "excluded; layers:", len(layers))
    print("\nMASK-APPLY CHECK PASSED")
