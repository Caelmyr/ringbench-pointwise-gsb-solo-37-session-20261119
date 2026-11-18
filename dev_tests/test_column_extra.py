import sys, time
sys.path.insert(0, "/workspace")
import numpy as np
from ringbench.column_defects import suggest_column_defects

rng = np.random.default_rng(11)

def base(H, W):
    yy, xx = np.mgrid[0:H, 0:W]
    img = 100 + 0.04*xx + 0.03*yy + rng.normal(0, 3.5, (H, W))
    img[120:128, :] += 220.0   # bright ring
    return img

# A. Half-frame-only low column (top half), healthy bottom half -> one segment.
H, W = 320, 200
img = base(H, W); valid = np.ones((H, W), bool)
img[10:150, 40] *= 0.3
res = suggest_column_defects(img, valid)
seg = [r for r in res["regions"] if r["x0"] == 40]
print("A half-frame:", seg)
assert len(seg) == 1 and seg[0]["y0"] <= 12 and seg[0]["y1"] >= 148

# B. Beam-stop EDGE: a soft shadow column next to a fully-masked disk.
img2 = base(H, W); valid2 = np.ones((H, W), bool)
yy, xx = np.mgrid[0:H, 0:W]
cy, cx, rad = 160, 100, 28
valid2[(yy-cy)**2 + (xx-cx)**2 <= rad**2] = False
img2[:, 130:136] *= 0.45   # 6px broad shadow arm (wide band -> reject)
res2 = suggest_column_defects(img2, valid2)
print("B beam-stop edge regions:", res2["regions"])
assert all(not (130 <= r["x0"] <= 135 or 130 <= r["x1"] <= 135) for r in res2["regions"])

# C. Bright ring crossing a low column must be bridged into a single region.
img3 = base(H, W)
img3[20:120, 55] *= 0.35
# rows 120-128 are the bright ring (normal/high in this column)
img3[128:240, 55] *= 0.35
res3 = suggest_column_defects(img3, valid)
seg3 = [r for r in res3["regions"] if r["x0"] == 55]
print("C ring-bridged:", seg3)
assert len(seg3) == 1 and seg3[0]["y0"] <= 22 and seg3[0]["y1"] >= 238

# D. Two distinct thin columns far apart are both reported independently.
img4 = base(H, W)
img4[30:200, 20] *= 0.3
img4[60:250, 180] *= 0.3
res4 = suggest_column_defects(img4, valid)
xs = {r["x0"] for r in res4["regions"]}
print("D independent columns:", xs)
assert {20, 180} <= xs

# E. A mild gradient (whole-frame slow variation) must not raise false positives.
img5 = base(H, W)
yy5, xx5 = np.mgrid[0:H, 0:W]
img5 += 0.25 * xx5
res5 = suggest_column_defects(img5, valid)
print("E gradient false positives:", res5["count"])
assert res5["count"] == 0

# F. Performance on a large detector.
bigH, bigW = 2048, 2048
bimg = 100 + rng.normal(0, 4, (bigH, bigW))
bimg[200:1800, 700] *= 0.35
bvalid = np.ones((bigH, bigW), bool)
t0 = time.time()
bres = suggest_column_defects(bimg, bvalid)
dt = time.time() - t0
print(f"F large 2048x2048 in {dt:.2f}s, regions={bres['count']}")
assert any(r["x0"] == 700 for r in bres["regions"])
assert dt < 30, "detection should run comfortably within the request budget"

print("\nALL EXTRA CHECKS PASSED")
