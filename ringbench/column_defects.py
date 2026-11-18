"""Suggest narrow vertical low-response column segments from one exposure.

The detector artifact addressed here is a thin column (a few pixels wide) whose
readout stays low over a long row interval, sometimes disappearing for a short
stretch where a bright diffraction ring crosses the column.  Real dark rings,
the beam-stop edge and isolated dark pixels must not be reported: those are
wide or point-like features rather than sustained narrow-column depressions.

Two complementary baselines are used per row:

* a global row median, which reveals the full horizontal extent of every low
  region but also follows genuine dark rings (which lower an entire row), and
* a local peer-column median, which ignores row-wide features such as dark
  rings because every peer column is lowered together.

A pixel is a candidate only when it is low against *both* baselines.  Broad low
bands (module gaps, shadows) are measured on the global baseline and dropped,
along with the boundary fragments a local-only test would otherwise produce.
Inside a column, a short gap where the readout stays low or only briefly
returns to normal (a ring crossing masking the defect) is joined, while a longer
stretch of normal readings splits the evidence into separate segments.
"""
import numpy as np
from scipy import ndimage

from .errors import WorkbenchError

# Detection thresholds are deliberately fixed: the suggestion is an advisory
# overlay and never edits data on its own.
PEER_RADIUS = 6          # half width of the local peer-column window
MIN_PEERS = 8            # minimum usable peer columns for a local baseline
LOW_RATIO = 0.60         # below 60 % of a baseline counts as a low reading
DEFECT_RATIO = 0.75      # a retained column's median response must stay below this
RECOVER_RATIO = 0.85     # a gap row at/above this reads as normal (not masked)
BAND_RATIO = 0.80        # below 80 % of the row median maps a broad band
MIN_RUN = 12             # shortest retained low-response run (pixels)
MAX_BRIDGE = 28          # normal-reading gap shorter than this joins one column
MIN_SEGMENT = 18         # shortest retained segment after bridging
MAX_SUGGESTIONS = 96
MAX_COLUMN_WIDTH = 4     # wider low regions are modules/shadows, not columns


def _row_baseline(work):
    """Global per-row median over finite, currently valid detector pixels."""
    height = work.shape[0]
    baseline = np.empty(height, dtype=np.float64)
    for y in range(height):
        finite = work[y][np.isfinite(work[y])]
        baseline[y] = np.median(finite) if finite.size else np.nan
    # Rows entirely excluded (e.g. behind the beam stop) get a neutral fill so
    # they neither trigger low flags nor qualify as bright ring rows.
    missing = ~np.isfinite(baseline)
    if np.any(missing):
        fill = np.nanmedian(baseline)
        baseline[missing] = fill if np.isfinite(fill) else 1.0
    return baseline


def _local_ratios(work, usable_columns):
    """Per-pixel ratio against a local peer-column median."""
    height, width = work.shape
    ratios = np.full((height, width), np.nan, dtype=np.float64)
    for x in range(width):
        if not usable_columns[x]:
            continue
        left = max(0, x - PEER_RADIUS)
        right = min(width, x + PEER_RADIUS + 1)
        peers = [c for c in range(left, right) if c != x and usable_columns[c]]
        if len(peers) < MIN_PEERS:
            continue
        peer_median = np.median(work[:, peers], axis=1)
        values = work[:, x]
        finite = np.isfinite(values) & (peer_median > 0)
        ratios[finite, x] = values[finite] / peer_median[finite]
    return ratios


def _broad_band_mask(global_ratios):
    """Rows covered by a horizontally broad run of sub-normal columns.

    On every row, maximal runs of consecutive columns reading below
    ``BAND_RATIO`` of the row median that are wider than the thin-column limit
    are marked.  A genuine dark ring lowers the whole row together, so the row
    median falls with it and no column becomes sub-normal relative to that
    median; only localized low regions appear here.
    """
    below = np.isfinite(global_ratios) & (global_ratios < BAND_RATIO)
    band = np.zeros(global_ratios.shape, dtype=bool)
    height, width = below.shape
    # Vectorized run-length marking per row: pad with False column boundaries.
    padded = np.concatenate(
        [np.zeros((height, 1), dtype=bool), below, np.zeros((height, 1), dtype=bool)],
        axis=1,
    )
    for y in range(height):
        edges = np.flatnonzero(padded[y, 1:] != padded[y, :-1])
        for start, end in zip(edges[0::2], edges[1::2]):
            if end - start > MAX_COLUMN_WIDTH:
                band[y, start:end] = True
    return band


def _column_segments(low, sub_normal, ratio):
    """Low-response segments of one column, joining masked crossings only.

    Following a low run, the gap before the next low run decides whether they
    are one masked column or two separate defects:

    * a short gap dominated by sub-normal rows is a ring or strong feature
      masking the defect, so the runs are joined;
    * a gap dominated by rows that read at the normal peer level is genuine
      readout.  Such a gap joins the runs only when it is very short; a longer
      normal stretch separates two independent defects.

    Isolated dark pixels never form a tall segment because the gaps around them
    consist almost entirely of normal readout.
    """
    height = low.shape[0]
    segments = []
    y = 0
    while y < height:
        if not low[y]:
            y += 1
            continue
        start = y
        y += 1
        while y < height and low[y]:
            y += 1
        run_end = y  # low rows currently cover [start, run_end)
        while y < height:
            gap_start = y
            gap = 0
            normal_gap = 0
            while y < height and not low[y]:
                if not sub_normal[y]:
                    normal_gap += 1
                gap += 1
                y += 1
            if y >= height or not low[y]:
                y = gap_start
                break
            masked_crossing = normal_gap * 2 < gap  # sub-normal rows dominate
            if masked_crossing or normal_gap <= MAX_BRIDGE:
                while y < height and low[y]:
                    y += 1
                run_end = y
                continue
            y = gap_start  # a long normal gap: two genuinely separate defects
            break
        if run_end - start >= MIN_RUN:
            observed = ratio[start:run_end]
            observed = observed[np.isfinite(observed)]
            median_ratio = float(np.median(observed)) if observed.size else 0.0
            segments.append((start, run_end, median_ratio))
    return segments


def suggest_column_defects(data, valid):
    """Suggest thin vertical low-response regions from a single exposure.

    ``valid`` is the existing exclusion mask (True where the pixel is usable),
    so currently masked or missing pixels are ignored as evidence instead of
    being mistaken for detector defects.
    """
    if data.ndim != 2 or valid.shape != data.shape:
        raise WorkbenchError("纵向坏区建议需要同尺寸的计数矩阵和遮罩。")
    height, width = data.shape
    work = np.where(valid & np.isfinite(data), data.astype(np.float64), np.nan)
    row_baseline = _row_baseline(work)
    usable_columns = [
        np.count_nonzero(np.isfinite(work[:, x])) >= MIN_SEGMENT
        for x in range(width)
    ]
    local_ratios = _local_ratios(work, usable_columns)
    with np.errstate(invalid="ignore", divide="ignore"):
        global_ratios = work / row_baseline[:, None]
    broad_band = _broad_band_mask(global_ratios)

    finite = np.isfinite(local_ratios)
    low = finite & (local_ratios < LOW_RATIO) & (global_ratios < LOW_RATIO)
    # Sub-normal rows are still below the peer readout level; normal rows have
    # recovered to it.  Their balance inside a gap decides bridge vs. split.
    sub_normal = finite & (local_ratios < RECOVER_RATIO)

    low_flags = np.zeros((height, width), dtype=bool)
    for x in range(width):
        for y0, y1, _median in _column_segments(
            low[:, x], sub_normal[:, x], local_ratios[:, x],
        ):
            low_flags[y0:y1, x] = True

    # Four-connected components keep a bright-ring bridge connected vertically
    # while separating columns whose low rows only touch diagonally.
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labels, component_count = ndimage.label(low_flags, structure=structure)
    regions = []
    for code in range(1, component_count + 1):
        ys, xs = np.nonzero(labels == code)
        x0, x1 = int(xs.min()), int(xs.max())
        if x1 - x0 + 1 > MAX_COLUMN_WIDTH:
            continue  # intrinsically broad component: not a thin column
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        if y1 - y0 < MIN_SEGMENT:
            continue
        # Drop boundary fragments that belong to a broad low band.
        if np.count_nonzero(broad_band[y0:y1, x0:x1 + 1]) >= (y1 - y0) // 2:
            continue
        component_ratios = local_ratios[labels == code]
        component_ratios = component_ratios[np.isfinite(component_ratios)]
        median_ratio = float(np.median(component_ratios)) if component_ratios.size else 0.0
        # Noise around the low threshold can chain into tall fragments; a real
        # defect column stays clearly depressed over its whole extent.
        if median_ratio >= DEFECT_RATIO:
            continue
        regions.append({
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "width": x1 - x0 + 1,
            "height": y1 - y0,
            "median_ratio": round(median_ratio, 4),
        })
    regions.sort(key=lambda item: (item["y0"], item["x0"]))
    regions = regions[:MAX_SUGGESTIONS]
    covered = sum(item["width"] * item["height"] for item in regions)
    return {
        "regions": regions,
        "count": len(regions),
        "pixel_count": int(covered),
        "thresholds": {
            "peer_radius": PEER_RADIUS,
            "low_ratio": LOW_RATIO,
            "defect_ratio": DEFECT_RATIO,
            "recover_ratio": RECOVER_RATIO,
            "band_ratio": BAND_RATIO,
            "min_run": MIN_RUN,
            "max_bridge": MAX_BRIDGE,
            "min_segment": MIN_SEGMENT,
            "max_column_width": MAX_COLUMN_WIDTH,
        },
    }
