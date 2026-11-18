import sys, tempfile
sys.path.insert(0, "/workspace")
import numpy as np
from ringbench.web import create_app
from ringbench.demos import make_demo
from ringbench.detector_io import capture_from_array

with tempfile.TemporaryDirectory() as d:
    app = create_app(d); store = app.extensions["ringbench_store"]; client = app.test_client()
    title, captures, state, ids, imported = make_demo("powder")

    # Column A: low evidence above and below a SHORT ring crossing -> ONE region.
    a = captures[0].data.copy()
    a[40:130, 250] *= 0.35
    a[150:225, 250] *= 0.35     # ~20-row crossing gap, within MAX_BRIDGE
    capA = capture_from_array(a, "主曝光")
    pA = store.create("补齐", [capA, captures[1]], state=state, original_ids=ids, imported=imported)
    rA = client.get(f"/api/projects/{pA.id}/column-defects?frame={pA.frames[0].id}&revision={pA.revision}").get_json()
    segA = [r for r in rA["regions"] if r["x0"] == 250]
    print("A ring-masked column:", segA)
    assert len(segA) == 1 and segA[0]["y0"] <= 42 and segA[0]["y1"] >= 223

    # Column B: two low runs with a LONG normal reading gap -> stay separate.
    b = captures[0].data.copy()
    b[40:110, 260] *= 0.35
    b[200:300, 260] *= 0.35     # ~90-row normal gap, longer than MAX_BRIDGE
    capB = capture_from_array(b, "主曝光")
    pB = store.create("分段", [capB, captures[1]], state=state, original_ids=ids, imported=imported)
    rB = client.get(f"/api/projects/{pB.id}/column-defects?frame={pB.frames[0].id}&revision={pB.revision}").get_json()
    segB = [r for r in rB["regions"] if r["x0"] == 260]
    print("B normal-gap column:", segB)
    assert len(segB) == 2
    assert all(r["y1"] <= 115 or r["y0"] >= 195 for r in segB)

    print("\nEVIDENCE BRIDGE/SPLIT CHECKS PASSED")
