"""Ensure pre-existing detector/mask/analysis paths still work after the change."""
import sys, tempfile
sys.path.insert(0, "/workspace")
from ringbench.web import create_app
from ringbench.demos import make_demo

with tempfile.TemporaryDirectory() as d:
    app = create_app(d); store = app.extensions["ringbench_store"]; client = app.test_client()
    title, captures, state, ids, imported = make_demo("powder")
    p = store.create(title, captures, state=state, original_ids=ids, imported=imported)
    pid, fid, rev = p.id, p.frames[0].id, p.revision
    def get(path):
        r = client.get(f"/api/projects/{pid}/{path}")
        assert r.status_code == 200, (path, r.data)
        return r
    # core detector views
    get(f"detector?frame={fid}&revision={rev}")
    get(f"detector.png?frame={fid}&revision={rev}")
    get(f"detector.png?view=mask&frame={fid}&revision={rev}")
    get(f"detector.png?view=annotated&frame={fid}&revision={rev}")
    get(f"excluded.npy?frame={fid}&revision={rev}")
    get(f"probe?x=10&y=10&frame={fid}&revision={rev}")
    # existing analysis endpoints used by the workbench
    get(f"curves/profile?frame={fid}&revision={rev}")
    get(f"cake?frame={fid}&revision={rev}")
    get(f"spots?frame={fid}&revision={rev}")
    get(f"integrity?frame={fid}&revision={rev}")
    # the new endpoint coexists and returns the documented schema
    payload = get(f"column-defects?frame={fid}&revision={rev}").get_json()
    assert set(payload) >= {"regions", "count", "pixel_count", "thresholds"}
    assert isinstance(payload["regions"], list)
    print("all pre-existing endpoints OK; new endpoint schema OK")
    print("\nREGRESSION CHECK PASSED")
