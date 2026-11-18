import os, sys, tempfile, threading, time, urllib.request, json
sys.path.insert(0, "/workspace")
import numpy as np
from ringbench.web import create_app
from ringbench.demos import make_demo
from werkzeug.serving import make_server

with tempfile.TemporaryDirectory() as d:
    app = create_app(d)
    store = app.extensions["ringbench_store"]
    title, captures, state, ids, imported = make_demo("powder")
    # inject a defect
    data = captures[0].data.copy(); data[40:220, 300] *= 0.35
    from ringbench.detector_io import capture_from_array
    cap = capture_from_array(data, captures[0].title)
    p = store.create(title, [cap, captures[1]], state=state, original_ids=ids, imported=imported)
    srv = make_server("127.0.0.1", 18951, app, threaded=True)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    try:
        base = f"http://127.0.0.1:18951"
        # static asset
        js = urllib.request.urlopen(base + "/static/detector.js").read().decode()
        assert "column-defects" in js and "columnDefects" in js
        print("static detector.js serves new UI")
        # health
        h = json.load(urllib.request.urlopen(base + "/api/health"))
        assert h["status"] == "ok"
        # endpoint over real HTTP
        url = f"{base}/api/projects/{p.id}/column-defects?frame={p.frames[0].id}&revision={p.revision}"
        payload = json.load(urllib.request.urlopen(url))
        print("live endpoint count:", payload["count"])
        assert payload["count"] >= 1
        assert all(set(["x0","x1","y0","y1","width","height","median_ratio"]) <= set(r) for r in payload["regions"])
        # 404 for unknown project still handled
        from urllib.error import HTTPError
        try:
            urllib.request.urlopen(base + "/api/projects/0"*32 + "/column-defects")
            raise AssertionError("expected 404")
        except HTTPError as e:
            assert e.code == 404
        print("error handling OK")
    finally:
        srv.shutdown()
print("\nLIVE SMOKE TEST PASSED")
