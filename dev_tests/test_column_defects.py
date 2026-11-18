"""Standalone checks for the narrow-column low-response suggestion."""
import sys
sys.path.insert(0, "/workspace")

import numpy as np
from ringbench.column_defects import suggest_column_defects

rng = np.random.default_rng(7)
H, W = 300, 240


def baseline_image():
    yy, xx = np.mgrid[0:H, 0:W]
    base = 120.0 + 0.05 * xx + 0.02 * yy
    # A bright diffraction ring band (adds signal to ALL columns on those rows).
    ring = np.zeros((H, W))
    ring[140:150, :] += 260.0
    # A diffuse dark ring (lowers ALL columns on those rows equally).
    base[200:208, :] *= 0.55
    img = base + ring + rng.normal(0, 4, (H, W))
    return img


def report(name, regions):
    print(f"\n== {name} ==")
    for r in regions:
        print(f"  x {r['x0']}-{r['x1']}  y {r['y0']}..{r['y1']} (excl)  "
              f"w={r['width']} h={r['height']} ratio={r['median_ratio']}")


# 1. Half-frame low column, interrupted by the bright ring (should be bridged).
img = baseline_image()
valid = np.ones((H, W), dtype=bool)
col = 60
img[20:140, col] *= 0.35          # low upper half
img[140:150, col] = img[140:150, col]  # ring rows stay bright (normal/high)
img[150:220, col] *= 0.35         # low continues below ring
res = suggest_column_defects(img, valid)
report("half-frame column with bright-ring gap", res["regions"])
assert any(r["x0"] == col and r["x1"] == col and r["y0"] <= 24 and r["y1"] >= 216
           for r in res["regions"]), "should bridge the bright ring into one segment"

# 2. Two low runs separated by a stretch of NORMAL readings -> separate segments.
img2 = baseline_image()
img2[30:110, col] *= 0.35
img2[110:180, col] = img2[110:180, col]   # normal reading stretch
img2[180:260, col] *= 0.35
res2 = suggest_column_defects(img2, valid)
report("two runs separated by normal readings", res2["regions"])
col_regions = [r for r in res2["regions"] if r["x0"] == col]
assert len(col_regions) >= 2, "normal-reading gap must split into separate segments"
assert all(r["y1"] <= 182 or r["y0"] >= 108 for r in col_regions)

# 3. Diffuse dark ring alone must not produce any suggestion.
img3 = baseline_image()
res3 = suggest_column_defects(img3, valid)
report("dark ring only (expect none)", res3["regions"])
assert res3["count"] == 0, "a ring-wide dark band is not a column defect"

# 4. Sparse isolated dark pixels must not produce a suggestion.
img4 = baseline_image()
for _ in range(40):
    py, px = rng.integers(2, H - 2), rng.integers(2, W - 2)
    img4[py, px] *= 0.1
res4 = suggest_column_defects(img4, valid)
report("sparse dark pixels (expect none)", res4["regions"])
assert res4["count"] == 0, "isolated dark pixels are point artifacts"

# 5. A wide low band (module/shadow) must be rejected as too broad.
img5 = baseline_image()
img5[40:260, 90:99] *= 0.4
res5 = suggest_column_defects(img5, valid)
report("wide low band (expect none)", res5["regions"])
assert res5["count"] == 0, "a broad low band and its edges are not thin columns"

# 6. Beam-stop-like fully masked circular region must not be reported.
img6 = baseline_image()
valid6 = valid.copy()
cy, cx, rad = 150, 120, 30
yy, xx = np.mgrid[0:H, 0:W]
valid6[(yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2] = False
res6 = suggest_column_defects(img6, valid6)
report("masked beam-stop disk (expect none)", res6["regions"])
assert res6["count"] == 0, "excluded beam-stop pixels are ignored"

# 7. Two-pixel-wide low column should merge into one width-2 region.
img7 = baseline_image()
img7[50:200, 70:72] *= 0.3
res7 = suggest_column_defects(img7, valid)
report("two-pixel column", res7["regions"])
assert any(r["x0"] == 70 and r["x1"] == 71 and r["width"] == 2 for r in res7["regions"])

# 8. A short low column (below minimum length) must be ignored.
img8 = baseline_image()
img8[100:108, 80] *= 0.2
res8 = suggest_column_defects(img8, valid)
report("short column (expect none)", res8["regions"])
assert all(not (r["x0"] == 80) for r in res8["regions"])

print("\nALL COLUMN-DEFECT CHECKS PASSED")
