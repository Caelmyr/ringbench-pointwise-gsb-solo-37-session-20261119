"""RingBench HTTP interface. All numeric results use the project's detector grid."""
from io import BytesIO
from pathlib import Path
from threading import BoundedSemaphore

from flask import Blueprint, Flask, Response, current_app, g, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException

from . import archive
from .report import PRODUCT_NAMES, product_ids
from .demos import choices, make_demo
from .detector_io import array_bytes, bounded_bytes, json_bytes, upload_capture
from .errors import WorkbenchError
from .integrity import check_integrity
from .masks import read_imported_mask
from .store import Store
from .values import MAX_BODY, MAX_IMAGE_BODY, filename, integer, mapping, number, optional_ordered, selection
from .workbench import Workbench

assets = Path(__file__).resolve().parent.parent / "static"
api = Blueprint("ringbench_api", __name__, url_prefix="/api")
DOWNLOAD = "application/octet-stream"


def store():
    return current_app.extensions["ringbench_store"]


def workbench():
    return current_app.extensions["ringbench_workbench"]


def json_content():
    if not request.is_json:
        raise WorkbenchError("请用 JSON 提交工作台设置。", status=415)
    return mapping(request.get_json(), "请求")


def evolution(content):
    return integer(content.get("revision"), "工程版本", 1, 2 ** 47)


def selection_args():
    rev = request.args.get("revision")
    fid = request.args.get("frame")
    return fid, None if rev is None else integer(rev, "工程版本", 1, 2 ** 47)


def src_file(field):
    source = request.files.get(field)
    if source is None:
        raise WorkbenchError("请选择要导入的文件。")
    return source


def form_upload():
    source = src_file("image")
    return upload_capture(source.stream, source.filename, request.form.get("frame_title"))


def file_response(data, name, content_type, *, attachment=True):
    response = send_file(
        BytesIO(data),
        download_name=name,
        mimetype=content_type,
        as_attachment=attachment,
        conditional=False,
        max_age=0,
    )
    response.headers["Cache-Control"] = "no-store"
    return response


def output_file(pid, stem, extension, data, content_type, *, attachment=True):
    title = store().project(pid).title
    root = filename(title, "ringbench")
    return file_response(data, f"{root}-{stem}.{extension}", content_type, attachment=attachment)


def png_response(payload):
    return file_response(payload, "preview.png", "image/png", attachment=False)


@api.get("/health")
def health():
    return jsonify(status="ok", service="ringbench", schema=1)


@api.get("/config")
def config():
    return jsonify(
        demos=choices(),
        formats=["PNG grayscale", "TIFF grayscale", "NPY numeric"],
        coordinate="zero_based_pixel_centers_y_down",
        cache=workbench().recent.status(),
        limits={"pixels": 4194304, "frames": 8, "projects": 64},
    )


@api.get("/projects")
def projects():
    return jsonify(projects=store().projects())


@api.post("/projects")
def create_project():
    frame = form_upload()
    title = request.form.get("title") or frame.title
    doc = store().create(title, [frame]).document()
    return jsonify(project=doc), 201


@api.post("/projects/demo")
def demo_project():
    body = json_content()
    key = selection(body.get("key"), ("powder", "texture", "edge"), "示范素材")
    title, captures, state, ids, mask = make_demo(key)
    doc = store().create(title, captures, state=state, original_ids=ids, imported=mask)
    return jsonify(project=doc.document()), 201


@api.post("/projects/restore")
def restore_project():
    upload = src_file("archive")
    payload = bounded_bytes(upload.stream, MAX_BODY)
    restored = archive.unpack(payload).install(store(), request.form.get("title"))
    return jsonify(project=restored.document()), 201


@api.get("/projects/<pid>")
def get_project(pid):
    return jsonify(project=store().project(pid).document())


@api.patch("/projects/<pid>")
def change_project(pid):
    body = json_content()
    patch = body.get("settings")
    if patch is not None:
        mapping(patch, "设置变更")
    doc = store().update(pid, evolution(body), patch, title=body.get("title"))
    return jsonify(project=doc.document())


@api.delete("/projects/<pid>")
def delete_project(pid):
    body = json_content()
    store().delete(pid, evolution(body))
    workbench().recent.discard_project(pid)
    return jsonify(deleted=pid)


@api.post("/projects/<pid>/frames")
def append_frame(pid):
    expected = evolution(request.form)
    frame = form_upload()
    project = store().add_frame(pid, expected, frame)
    return jsonify(project=project.document()), 201


@api.patch("/projects/<pid>/frames/<fid>")
def name_frame(pid, fid):
    body = json_content()
    project = store().rename_frame(pid, fid, evolution(body), body.get("title"))
    return jsonify(project=project.document())


@api.delete("/projects/<pid>/frames/<fid>")
def remove_frame(pid, fid):
    project = store().drop_frame(pid, fid, evolution(json_content()))
    return jsonify(project=project.document())


@api.put("/projects/<pid>/mask")
def import_mask(pid):
    source = src_file("mask")
    expected = evolution(request.form)
    snap = store().project(pid)
    raw = bounded_bytes(source.stream, MAX_IMAGE_BODY)
    arr = read_imported_mask(raw, snap.shape)
    saved = store().replace_mask(pid, expected, arr)
    return jsonify(project=saved.document())


@api.delete("/projects/<pid>/mask")
def delete_mask(pid):
    project = store().replace_mask(pid, evolution(json_content()), None)
    return jsonify(project=project.document())


@api.get("/projects/<pid>/detector")
def detector(pid):
    return jsonify(workbench().view_json(pid, *selection_args()))


@api.get("/projects/<pid>/detector.png")
def show_detector(pid):
    mode = selection(request.args.get("view", "counts"), ("counts", "mask", "annotated"), "探测器视图")
    payload = workbench().detector_png(
        pid, *selection_args(), mask_mode=mode == "mask", annotations=mode == "annotated",
    )
    if request.args.get("download") == "1":
        return output_file(pid, mode, "png", payload, "image/png")
    return png_response(payload)


@api.get("/projects/<pid>/detector.npy")
def raw_counts(pid):
    data = workbench().raw(pid, *selection_args())
    return output_file(pid, "counts", "npy", data, DOWNLOAD)


@api.get("/projects/<pid>/excluded.npy")
def excluded_pixels(pid):
    data = workbench().excluded(pid, *selection_args())
    return output_file(pid, "excluded", "npy", data, DOWNLOAD)


@api.get("/projects/<pid>/probe")
def probe(pid):
    point = (request.args.get("x"), request.args.get("y"))
    return jsonify(workbench().probe(pid, *selection_args(), point))


@api.get("/projects/<pid>/column-defects")
def column_defects(pid):
    desc, suggestion = workbench().column_defects(pid, *selection_args())
    return jsonify({**desc, **suggestion})


@api.post("/projects/<pid>/calibration/circle")
def find_center(pid):
    body = json_content()
    fit = workbench().calibration(pid, evolution(body), body.get("marks"))
    return jsonify(fit)


@api.post("/projects/<pid>/calibration/distance")
def estimate_distance(pid):
    body = json_content()
    model = workbench().calibrate_distance(pid, evolution(body), body.get("radius_px"), body.get("d_a"))
    return jsonify(model)


@api.post("/projects/<pid>/calibration/distance-fit")
def calibration_distances(pid):
    body = json_content()
    version = evolution(body)
    result = workbench().distance_model(pid, version, body.get("settings"))
    output = selection(body.get("output", "summary"), ("summary", "csv"), "多环标定输出")
    if output == "csv":
        return output_file(pid, "distance-calibration", "csv", result.csv(), "text/csv")
    return jsonify(model=result.document(), revision=version)


@api.get("/projects/<pid>/curves/<kind>")
def curve_values(pid, kind):
    kind = selection(kind, ("profile", "azimuth", "line"), "曲线类型")
    value = number(request.args.get("prominence"), "径向突出度", 0, allow_none=True)
    output = workbench().curve_data(pid, *selection_args(), kind, value)
    return jsonify(output)


@api.get("/projects/<pid>/curves/<kind>.csv")
def curve_table(pid, kind):
    kind = selection(kind, ("profile", "azimuth", "line"), "曲线类型")
    data = workbench().curve_csv(pid, *selection_args(), kind)
    return output_file(pid, kind, "csv", data, "text/csv")


@api.get("/projects/<pid>/curves/<kind>.png")
def curve_chart(pid, kind):
    kind = selection(kind, ("profile", "azimuth"), "可下载图表类型")
    bars = selection(request.args.get("series", "mean"), ("mean", "support"), "图表序列")
    data = workbench().curve_png(pid, *selection_args(), kind, bars)
    return output_file(pid, kind, "png", data, "image/png")


@api.get("/projects/<pid>/cake")
def cake_values(pid):
    cells = request.args.get("cells", "0") == "1"
    return jsonify(workbench().cake_data(pid, *selection_args(), cells=cells))


@api.get("/projects/<pid>/cake/cell")
def cake_cell(pid):
    x = integer(request.args.get("x"), "径向格", 0, 720)
    y = integer(request.args.get("y"), "方位格", 0, 360)
    return jsonify(workbench().cake_probe(pid, *selection_args(), x, y))


@api.get("/projects/<pid>/cake.png")
def cake_preview(pid):
    mode = selection(request.args.get("layer", "mean"), ("mean", "support"), "极坐标图图层")
    image = workbench().cake_png(pid, *selection_args(), mode == "support")
    if request.args.get("download") == "1":
        return output_file(pid, f"cake-{mode}", "png", image, "image/png")
    return png_response(image)


@api.get("/projects/<pid>/cake.csv")
def cake_table(pid):
    data = workbench().cake_csv(pid, *selection_args())
    return output_file(pid, "cake", "csv", data, "text/csv")


@api.get("/projects/<pid>/spots")
def spots(pid):
    return jsonify(workbench().spot_data(pid, *selection_args()))


@api.get("/projects/<pid>/spots.csv")
def spot_table(pid):
    output = workbench().spot_csv(pid, *selection_args())
    return output_file(pid, "spots", "csv", output, "text/csv")


@api.get("/projects/<pid>/spots/search")
def candidates(pid):
    threshold = number(request.args.get("threshold"), "候选计数下限", allow_none=True)
    distance = number(request.args.get("separation", 10), "候选点距离", 1, 80)
    maximum = integer(request.args.get("maximum", 16), "候选点上限", 1, 64)
    output = workbench().scan_spots(pid, *selection_args(), threshold, distance, maximum)
    return jsonify(output)


@api.get("/projects/<pid>/difference")
def diff_summary(pid):
    return jsonify(workbench().difference_data(pid, *selection_args()))


@api.get("/projects/<pid>/difference.png")
def diff_preview(pid):
    image = workbench().difference_png(pid, *selection_args())
    if request.args.get("download") == "1":
        return output_file(pid, "difference", "png", image, "image/png")
    return png_response(image)


@api.get("/projects/<pid>/difference.npy")
def diff_counts(pid):
    output = workbench().difference_npy(pid, *selection_args())
    return output_file(pid, "difference", "npy", output, DOWNLOAD)


@api.post("/projects/<pid>/inspection")
def inspection(pid):
    body = json_content()
    bins = integer(body.get("bins", 80), "直方图区间数", 12, 512)
    bounds = optional_ordered(body.get("counts"), "直方图计数区间")
    desc, result = workbench().inspect(pid, body.get("frame"), evolution(body), body.get("region"), bins, bounds)
    channel = selection(body.get("output", "summary"), ("summary", "histogram", "columns", "rows"), "检查结果")
    if channel == "summary":
        return jsonify({**desc, **result.document()})
    if channel == "histogram":
        data = result.hist.csv()
    elif channel == "columns":
        data = result.columns.csv()
    else:
        data = result.rows.csv()
    return output_file(pid, channel, "csv", data, "text/csv")


@api.post("/projects/<pid>/stability")
def repeat_frames(pid):
    body = json_content()
    minimum = integer(body.get("minimum", 2), "有效观测次数", 2, 8)
    output = selection(
        body.get("output", "summary"),
        ("summary", "mean.png", "std.png", "mean.npy", "std.npy", "outliers.csv"),
        "重复曝光输出",
    )
    product = workbench().repeat(pid, body.get("frame"), evolution(body), body.get("frames"), minimum)
    if output == "summary":
        return jsonify(product.document())
    if output.endswith(".png"):
        return png_response(workbench().repeat_image(product, output.split(".")[0]))
    if output == "outliers.csv":
        level = number(body.get("threshold", 10), "离散计数下限", 0)
        limit = integer(body.get("limit", 200), "离散像素上限", 1, 2048)
        return output_file(pid, "deviation-pixels", "csv", product.outlier_csv(level, limit), "text/csv")
    data = product.mean if output == "mean.npy" else product.deviation
    return output_file(pid, output.split(".")[0], "npy", array_bytes(data), DOWNLOAD)


@api.post("/projects/<pid>/correction")
def correction(pid):
    body = json_content()
    output = selection(
        body.get("output", "summary"), ("summary", "preview", "counts", "gain", "flags"), "校正输出",
    )
    desc, response = workbench().correction(pid, body.get("frame"), evolution(body), body.get("settings"))
    if output == "summary":
        return jsonify({**desc, **response.document()})
    if output == "preview":
        return png_response(workbench().corrected_preview(response))
    if output == "flags":
        return output_file(pid, "correction-flags", "csv", response.flags_csv(), "text/csv")
    arr = response.npy() if output == "counts" else response.gain_npy()
    return output_file(pid, f"corrected-{output}", "npy", arr, DOWNLOAD)


@api.get("/projects/<pid>/integrity")
def integrity(pid):
    _, version = selection_args()
    project, frames, mask = store().archive_snapshot(pid, version)
    return jsonify(check_integrity(project, frames, mask).document())


@api.get("/projects/<pid>/workspace.zip")
def zip_project(pid):
    _, rev = selection_args()
    project, arrays, mask = store().archive_snapshot(pid, rev)
    payload = archive.pack(project, arrays, mask)
    return file_response(payload, f"{filename(project.title, 'ringbench')}.zip", "application/zip")


@api.get("/projects/<pid>/settings.json")
def saved_settings(pid):
    project = store().project(pid)
    return file_response(json_bytes(project.document()), "workspace-settings.json", "application/json")


@api.post("/projects/<pid>/comparison")
def radial_comparison(pid):
    body = json_content()
    desc, curve = workbench().compare_profiles(pid, body.get("frame"), evolution(body), body.get("settings"))
    output = selection(body.get("output", "summary"), ("summary", "csv"), "径向比较输出")
    if output == "csv":
        return output_file(pid, "radial-comparison", "csv", curve.csv(), "text/csv")
    return jsonify({**desc, **curve.document()})


@api.post("/projects/<pid>/harmonics")
def harmonics(pid):
    body = json_content()
    desc, fit = workbench().harmonic(pid, body.get("frame"), evolution(body), body.get("settings"))
    output = selection(body.get("output", "summary"), ("summary", "csv"), "方位拟合输出")
    if output == "csv":
        return output_file(pid, "azimuth-fit", "csv", fit.csv(), "text/csv")
    return jsonify({**desc, **fit.document()})


@api.post("/projects/<pid>/trace")
def trace(pid):
    body = json_content()
    desc, result = workbench().trace(pid, body.get("frame"), evolution(body), body.get("settings"))
    output = selection(body.get("output", "summary"), ("summary", "csv"), "环线输出")
    if output == "csv":
        return output_file(pid, "ring-trace", "csv", result.csv(), "text/csv")
    return jsonify({**desc, **result.document()})


@api.post("/projects/<pid>/peak-area")
def peak_area(pid):
    body = json_content()
    desc, result = workbench().measure_area(pid, body.get("frame"), evolution(body), body.get("settings"))
    output = selection(body.get("output", "summary"), ("summary", "csv"), "峰区面积输出")
    if output == "csv":
        return output_file(pid, "peak-area", "csv", result.csv(), "text/csv")
    return jsonify({**desc, **result.document()})


@api.get("/report-products")
def available_report_items():
    return jsonify(options=[{"id": key, "label": label} for key, label in PRODUCT_NAMES.items()])


@api.post("/projects/<pid>/analysis.zip")
def export_review_package(pid):
    body = json_content()
    result = workbench().review_package(pid, body.get("frame"), evolution(body), product_ids(body))
    return output_file(pid, "analysis-review", "zip", result, "application/zip")


@api.post("/projects/<pid>/contours")
def contours(pid):
    body = json_content()
    desc, paths = workbench().contours(pid, body.get("frame"), evolution(body), body.get("settings"))
    view = selection(body.get("output", "summary"), ("summary", "csv"), "等强线输出")
    if view == "csv":
        return output_file(pid, "intensity-boundaries", "csv", paths.csv(), "text/csv")
    return jsonify({**desc, **paths.document()})


@api.post("/projects/<pid>/vectors")
def vectors(pid):
    body = json_content()
    source, result = workbench().vectors(pid, body.get("frame"), evolution(body), body.get("settings"))
    view = selection(body.get("output", "summary"), ("summary", "vectors", "pairs"), "斑点向量输出")
    if view != "summary":
        return output_file(pid, f"reciprocal-{view}", "csv", result.csv(view), "text/csv")
    return jsonify({**source, **result.document()})


@api.post("/projects/<pid>/coverage")
def coverage(pid):
    body = json_content()
    source, result = workbench().annular_coverage(
        pid, body.get("frame"), evolution(body), body.get("settings"),
    )
    view = selection(body.get("output", "summary"), ("summary", "csv"), "整环覆盖输出")
    if view == "csv":
        return output_file(pid, "annular-coverage", "csv", result.csv(), "text/csv")
    return jsonify({**source, **result.document()})


@api.post("/projects/<pid>/geometric-correction")
def geometric_correction(pid):
    body = json_content()
    selection_output = selection(
        body.get("output", "summary"),
        ("summary", "corrected.png", "response.png", "corrected.npy", "response.npy", "stats.csv"),
        "几何校正结果",
    )
    source, result, display = workbench().geometric(pid, body.get("frame"), evolution(body), body.get("settings"))
    if selection_output == "summary":
        return jsonify({**source, **result.document()})
    if selection_output.endswith(".png"):
        return png_response(workbench().geometric_preview(result, display, selection_output.split(".")[0]))
    if selection_output == "stats.csv":
        return output_file(pid, "geometry-flags", "csv", result.csv(), "text/csv")
    name = selection_output.split(".")[0]
    return output_file(pid, f"geometry-{name}", "npy", result.matrix(name), DOWNLOAD)


def create_app(data_dir):
    app = Flask(__name__, static_folder=str(assets))
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_BODY,
        MAX_FORM_MEMORY_SIZE=512 * 1024,
        MAX_FORM_PARTS=16,
    )
    db = Store(data_dir)
    app.extensions["ringbench_store"] = db
    app.extensions["ringbench_workbench"] = Workbench(db)
    app.extensions["ringbench_expensive"] = BoundedSemaphore(2)
    app.register_blueprint(api)

    @app.get("/")
    def index():
        return send_from_directory(assets, "index.html", max_age=0)

    @app.before_request
    def limit_parallel_work():
        g.ringbench_slot = False
        heavy = request.blueprint == api.name and request.endpoint not in {
            "ringbench_api.health", "ringbench_api.config", "ringbench_api.projects",
        }
        if not heavy:
            return
        slots = app.extensions["ringbench_expensive"]
        if not slots.acquire(timeout=5):
            raise WorkbenchError("工作台正在处理其他图像，请稍后重试。", status=503, code="busy")
        g.ringbench_slot = True

    @app.teardown_request
    def free_parallel_work(_error):
        if getattr(g, "ringbench_slot", False):
            g.ringbench_slot = False
            app.extensions["ringbench_expensive"].release()

    @app.after_request
    def harden_response(result):
        result.headers["X-Content-Type-Options"] = "nosniff"
        result.headers["Referrer-Policy"] = "same-origin"
        if request.blueprint == api.name:
            result.headers["Cache-Control"] = "no-store"
        return result

    @app.errorhandler(WorkbenchError)
    def user_error(exc):
        return jsonify(exc.payload()), exc.status

    @app.errorhandler(HTTPException)
    def request_error(exc):
        messages = {400: "请求内容无法读取。", 404: "没有找到该页面或输出。", 413: "上传文件超过大小限制。"}
        return jsonify(error={"code": f"http_{exc.code}", "message": messages.get(exc.code, exc.name)}), exc.code

    return app
