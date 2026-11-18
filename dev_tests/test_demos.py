import sys, tempfile
sys.path.insert(0, "/workspace")
from ringbench.web import create_app
from ringbench.demos import make_demo

with tempfile.TemporaryDirectory() as d:
    app = create_app(d)
    store = app.extensions["ringbench_store"]
    client = app.test_client()
    for key in ("powder", "texture", "edge"):
        title, captures, state, ids, imported = make_demo(key)
        p = store.create(title, captures, state=state, original_ids=ids, imported=imported)
        fid = p.frames[0].id
        resp = client.get(f"/api/projects/{p.id}/column-defects?frame={fid}&revision={p.revision}")
        assert resp.status_code == 200, resp.data
        payload = resp.get_json()
        print(f"demo {key}: {payload['count']} suggestions, {payload['pixel_count']} px")
        for r in payload["regions"][:5]:
            print("   ", r)
