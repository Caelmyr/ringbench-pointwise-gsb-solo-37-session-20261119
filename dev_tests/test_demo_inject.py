import sys, tempfile
sys.path.insert(0, "/workspace")
import numpy as np
from ringbench.web import create_app
from ringbench.demos import make_demo
from ringbench.detector_io import capture_from_array

with tempfile.TemporaryDirectory() as d:
    app = create_app(d)
    store = app.extensions["ringbench_store"]
    client = app.test_client()
    title, captures, state, ids, imported = make_demo("powder")
    # Inject a half-frame low column into the main exposure, crossing rings.
    data = captures[0].data.copy()
    col = 300
    data[30:260, col] *= 0.35
    injected = capture_from_array(data, captures[0].title)
    p = store.create(title, [injected, captures[1]], state=state, original_ids=ids, imported=imported)
    fid = p.frames[0].id
    payload = client.get(f"/api/projects/{p.id}/column-defects?frame={fid}&revision={p.revision}").get_json()
    print("suggestions:", payload["count"])
    for r in payload["regions"]:
        print("   ", r)
    hit = [r for r in payload["regions"] if r["x0"] == col]
    assert hit, "injected half-frame column should be detected amid rings and masks"
    print("\nDEMO INJECTION CHECK PASSED")
