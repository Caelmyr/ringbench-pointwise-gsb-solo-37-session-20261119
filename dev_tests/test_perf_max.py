import sys, time
sys.path.insert(0, "/workspace")
import numpy as np
from ringbench.column_defects import suggest_column_defects

# Largest allowed: total pixels <= 4,194,304. Use 2048x2048 (4.19M) and a wide shape.
for (H, W) in [(2048, 2048), (1024, 4096), (4096, 1024)]:
    rng = np.random.default_rng(0)
    img = 100 + rng.normal(0, 4, (H, W)).astype(np.float32)
    img[int(H*0.1):int(H*0.9), W//2] *= 0.35
    valid = np.ones((H, W), bool)
    t0 = time.time()
    res = suggest_column_defects(img, valid)
    dt = time.time() - t0
    hit = any(r["x0"] == W//2 for r in res["regions"])
    print(f"{H}x{W} ({H*W/1e6:.2f}M px): {dt:.2f}s, regions={res['count']}, column hit={hit}")
    assert hit and dt < 45
print("\nMAX-SIZE PERFORMANCE OK")
