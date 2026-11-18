"""Build results-only review packages from a single detector snapshot."""
from dataclasses import dataclass
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile


from .detector_io import csv_bytes, digest, json_bytes, mask_bytes
from .errors import WorkbenchError
from .imaging import annotated_detector, difference_view, draw_curve_png, reduction_view
from .reduction import integrate_azimuth, integrate_cake, integrate_profile
from .section import cross_section
from .spots import catalog_csv, measure
from .state import AzimuthSettings, CakeSettings, DifferenceSettings, LineSettings, ProfileSettings
from .values import items, mapping, selection, unique
from .difference import compare

PRODUCTS = (
    "detector", "mask", "profile", "azimuth", "cake", "line", "spots", "difference",
)
PRODUCT_NAMES = {
    "detector": "带参考环与标点的探测器图",
    "mask": "探测器排除像素和平面统计",
    "profile": "径向积分谱",
    "azimuth": "环带角向曲线",
    "cake": "极坐标图",
    "line": "原探测器横截面",
    "spots": "已标斑点积分计数",
    "difference": "同位曝光差图",
}
MAX_RESULT_BYTES = 52 * 1024 * 1024


@dataclass(frozen=True)
class Artifact:
    path: str
    category: str
    media: str
    data: bytes

    def document(self):
        return {
            "file": self.path,
            "kind": self.category,
            "media": self.media,
            "bytes": len(self.data),
            "sha256": digest(self.data),
        }


@dataclass(frozen=True)
class Product:
    label: str
    summary: dict
    files: list

    def document(self):
        return {
            "label": self.label,
            "summary": self.summary,
            "artifacts": [f.document() for f in self.files],
        }


def product_ids(value):
    content = mapping(value, "分析交接内容")
    chosen = items(content.get("products"), "分析交接产品", minimum=1, maximum=len(PRODUCTS))
    return unique([selection(p, PRODUCTS, "分析交接类型") for p in chosen], "交接类型")


def _png(stem, category, payload):
    return Artifact(f"images/{stem}.png", category, "image/png", payload)


def _table(stem, category, payload):
    return Artifact(f"tables/{stem}.csv", category, "text/csv", payload)


def _array(stem, category, payload):
    return Artifact(f"arrays/{stem}.npy", category, "application/octet-stream", payload)


def _detector(det):
    output = annotated_detector(det.counts, det.mask, det.annotation(), det.geometry, det.rings())
    cover = det.detector_data()
    summary = {
        "width": det.project.width,
        "height": det.project.height,
        "count_statistics": cover["stats"],
        "display_range": cover["range"],
        "range_clipped": cover["range_clipped"],
        "reference_rings": cover["references"],
    }
    return summary, [_png("detector", "detector", output)]


def _mask(det):
    summary = det.mask.summary()
    rows = [(entry["title"], entry["enabled"], entry["excluded_first"]) for entry in summary["layers"]]
    rows.insert(0, ("imported detector flag", summary["imported"], summary["imported_count"]))
    rows.insert(0, ("missing source count", True, summary["missing"]))
    return summary, [
        _array("excluded", "mask", mask_bytes(~det.valid)),
        _table("excluded-sources", "mask", csv_bytes(["source", "enabled", "first_excluded_pixels"], rows)),
    ]


def _profile(det):
    spec = ProfileSettings.read(det.settings["profile"])
    product = integrate_profile(det.counts, det.valid, det.geometry, spec)
    details = product.document()
    peaks = details["peaks"]
    table = csv_bytes(
        [
            f"coordinate_{spec.unit}", "mean_count", "prominence_or_component_height",
            "half_prominence_width", "component_fwhm", "width_source", "valid_pixels",
            "components", "resolution_status",
        ],
        (
            (
                r["coordinate"], r["mean"], r["prominence"], r["width_at_half_prominence"],
                r["fwhm"], r["width_source"], r["valid_pixels"], r["components"],
                r["resolution"]["status"],
            )
            for r in peaks
        ),
    )
    split = sum(1 for r in peaks if r["resolution"]["status"] == "resolved")
    return {
        **details["summary"],
        "identified_peaks": len(peaks),
        "resolved_components": split,
        "unresolved_shoulders": sum(
            1 for r in peaks if r["resolution"]["status"] == "insufficient_evidence"
        ),
    }, [
        _png("profile", "profile", draw_curve_png(product)),
        _table("profile", "profile", product.csv()),
        _table("radial-peaks", "profile", table),
    ]


def _azimuth(det):
    spec = AzimuthSettings.read(det.settings["azimuth"], det.project.width, det.project.height)
    product = integrate_azimuth(det.counts, det.valid, det.geometry, spec)
    return product.moments.summary(), [
        _png("azimuth", "azimuth", draw_curve_png(product)),
        _table("azimuth", "azimuth", product.csv()),
    ]


def _cake(det):
    spec = CakeSettings.read(det.settings["cake"])
    product = integrate_cake(det.counts, det.valid, det.geometry, spec)
    summary = product.moments.summary()
    summary["radial_unit"] = product.radial.unit
    summary["radial_cells"] = product.radial.cells
    summary["angle_cells"] = product.angular.cells
    return summary, [
        _png("polar-counts", "cake", reduction_view(product, det.display)),
        _png("polar-support", "cake", reduction_view(product, det.display, support=True)),
        _table("polar-grid", "cake", product.csv()),
    ]


def _line(det):
    spec = LineSettings.read(det.settings["line"], det.project.width, det.project.height)
    product = cross_section(det.counts, det.valid, spec)
    return product.summary(), [_table("cross-section", "line", product.csv())]


def _spots(det):
    rows = measure(det.counts, det.valid, det.geometry, det.spots(), det.aperture())
    summary = {
        "marked": len(rows),
        "measured": sum(row["measured"] for row in rows),
        "subtracted_local_background": sum(
            row["counts"]["background"] is not None for row in rows if row["measured"]
        ),
    }
    return summary, [_table("spot-apertures", "spots", catalog_csv(rows))]


def _difference(det):
    spec = DifferenceSettings.read(det.settings["difference"])
    image = det.snapshot
    if image.reference_counts is None:
        raise WorkbenchError("把差图加入交接包前，需要选择不同的主曝光和参照曝光。")
    ref_mask = det.other_mask(image.reference_counts)
    product = compare(det.counts, det.valid, image.reference_counts, ref_mask.valid, spec)
    preview, window = difference_view(product, spec)
    summary = {
        **product.document(), "reference": image.reference.document(), "display_window": window,
    }
    return summary, [
        _png("exposure-difference", "difference", preview),
        _array("exposure-difference", "difference", product.npy()),
    ]


GENERATORS = {
    "detector": _detector,
    "mask": _mask,
    "profile": _profile,
    "azimuth": _azimuth,
    "cake": _cake,
    "line": _line,
    "spots": _spots,
    "difference": _difference,
}


def _manifest(det, products):
    return {
        "format": "ringbench-analysis-review",
        "version": 1,
        "project": det.project.title,
        "project_id": det.project.id,
        "workspace_revision": det.project.revision,
        "source": det.frame.document(),
        "coordinate_convention": det.description()["coordinate_convention"],
        "geometry": det.geometry.document(),
        "settings": det.settings,
        "products": {key: product.document() for key, product in products},
    }


def _notes(det, products):
    listed = [
        "RingBench 分析交接包",
        f"工程 {det.project.title} / 曝光 {det.frame.title}",
        f"工程版本 {det.project.revision}",
        "",
        "探测器坐标采用从 0 起算的像素中心，X 向右，Y 向下。",
        "角度由探测器正 X 轴沿图像方向顺时针增加，再减去保存的角度零位。",
        "曲线 CSV 的空白数字表示该位置没有可用像素。数组中 NaN 表示缺失。",
        "索引 analysis.json 包含几何、参数、结果摘要及每个交付文件的 SHA-256。",
        "要重新编辑工程，请在工作台单独下载工程 ZIP。",
        "",
    ]
    for key, product in products:
        listed.append(f"{PRODUCT_NAMES[key]}")
        listed.extend(f"  {item.path}" for item in product.files)
    return ("\n".join(listed) + "\n").encode("utf-8")


def produce_report(det, chosen):
    results = []
    artifact_bytes = 0
    for section in chosen:
        summary, files = GENERATORS[section](det)
        results.append((section, Product(PRODUCT_NAMES[section], summary, files)))
        artifact_bytes += sum(len(item.data) for item in files)
        if artifact_bytes > MAX_RESULT_BYTES:
            raise WorkbenchError("分析包过大，请取消部分产品后重新导出。")
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=4) as dst:
        dst.writestr("analysis.json", json_bytes(_manifest(det, results)))
        dst.writestr("使用说明.txt", _notes(det, results))
        for _, product in results:
            for item in product.files:
                dst.writestr(item.path, item.data)
    return stream.getvalue()
