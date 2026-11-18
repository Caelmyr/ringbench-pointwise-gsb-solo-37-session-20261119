"""Compose persisted inputs with detector operations and transport-safe results."""
from dataclasses import asdict

import numpy as np

from .contours import ContourSettings, extract_contours
from .column_defects import suggest_column_defects
from .correction import ResponseSettings, correct
from .coverage import CoverageSpec, estimate_coverage
from .detector_io import array_bytes, finite_summary
from .difference import compare
from .distances import fit_distances
from .errors import Changed, WorkbenchError
from .geometric_response import GeometricResponseSpec, geometric_response
from .geometry import Geometry, fit_circle, reference_distance, reference_rings
from .imaging import (
    MAX_PREVIEW, Window, annotated_detector, choose_window, colors, detector_view,
    difference_view, draw_curve_png, mask_view, reduction_view,
)
from .inspection import Region, inspect_region, read_pixel, stability
from .masks import evaluate, mask_layers
from .orientation import HarmonicSettings, fit_orientation
from .peak_area import MeasureSettings, measure_area
from .recent import RecentProducts
from .reciprocal import VectorSettings, measure_vectors
from .report import produce_report
from .ring_trace import TraceSettings, trace_ring
from .series import SeriesSettings, compare_profile_series
from .reduction import integrate_azimuth, integrate_cake, integrate_profile, probe_cake
from .section import cross_section
from .spots import catalog_csv, find_candidates, measure
from .state import (
    Aperture, AzimuthSettings, CakeSettings, DifferenceSettings, Display,
    LineSettings, ProfileSettings, Spot,
)
from .values import items, token, unique


class Detector:
    """Decode only settings needed for the current detector snapshot."""

    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.project = snapshot.project
        self.settings = self.project.state
        self.frame = snapshot.frame
        self.counts = snapshot.counts
        self.geometry = Geometry.read(
            self.settings["geometry"], self.project.width, self.project.height,
        )
        self.layers = mask_layers(
            self.settings["masks"], self.project.width, self.project.height,
        )
        self.display = Display.read(self.settings["display"])
        self.mask = evaluate(self.counts, self.geometry, self.layers, self.snapshot.imported)
        self.valid = self.mask.valid

    def other_mask(self, data):
        return evaluate(data, self.geometry, self.layers, self.snapshot.imported)

    def support(self):
        return self.mask.summary()

    def rings(self):
        return reference_rings(self.geometry, self.settings["reference_d"], self.project.width, self.project.height)

    def description(self):
        return {
            "project": self.project.id,
            "revision": self.project.revision,
            "frame": self.frame.document(),
            "coordinate_convention": "zero_based_pixel_centers_y_down",
        }

    def detector_data(self):
        selected = self.counts[self.valid]
        window = choose_window(self.counts, self.valid, self.display)
        return {
            "mask": self.support(),
            "stats": finite_summary(self.counts, self.valid),
            "range": window.document(),
            "range_clipped": {
                "low": int(np.count_nonzero(selected < window.low)),
                "high": int(np.count_nonzero(selected > window.high)),
            },
            "references": self.rings(),
            "geometry": self.geometry.document(),
        }

    def spots(self):
        return [
            Spot.read(s, self.project.width, self.project.height)
            for s in self.settings["spots"]
        ]

    def aperture(self):
        return Aperture.read(self.settings["aperture"])

    def annotation(self):
        return {
            "display": self.display,
            "marks": self.settings["marks"],
            "spots": self.settings["spots"],
            "aperture": self.settings["aperture"],
        }


class Workbench:
    def __init__(self, store):
        self.store = store
        self.recent = RecentProducts(capacity=5, time_to_live=120)

    def detector(self, project_id, frame_id=None, revision=None, *, reference=False):
        image = self.store.snapshot(project_id, frame_id, revision, reference=reference)
        return Detector(image)

    def _version(self, project_id, frame_id, revision):
        project = self.store.project(project_id)
        if revision is not None and project.revision != revision:
            raise Changed(project.revision)
        frame = project.select(frame_id)
        return project, frame

    def _cached(self, pid, fid, rev, operation, make, extra=()):
        project, frame = self._version(pid, fid, rev)
        rev = project.revision
        key = (pid, rev, frame.id, operation, *extra)

        def build():
            attached = self.detector(pid, frame.id, rev, reference=(operation == "difference"))
            return make(attached)

        return self.recent.solve(key, build)

    def view_json(self, pid, fid, rev):
        det = self.detector(pid, fid, rev)
        return {**det.description(), **det.detector_data()}

    def detector_png(self, pid, fid, rev, *, mask_mode=False, annotations=False):
        det = self.detector(pid, fid, rev)
        if mask_mode:
            return mask_view(det.mask)
        if annotations:
            return annotated_detector(det.counts, det.mask, det.annotation(), det.geometry, det.rings())
        return detector_view(det.counts, det.mask, det.display)[0]

    def raw(self, pid, fid, rev):
        return array_bytes(self.store.snapshot(pid, fid, rev).counts)

    def excluded(self, pid, fid, rev):
        from .detector_io import mask_bytes
        return mask_bytes(~self.detector(pid, fid, rev).valid)

    def probe(self, pid, fid, rev, point):
        det = self.detector(pid, fid, rev)
        return {**det.description(), "pixel": read_pixel(det.counts, det.mask, det.geometry, point)}

    def column_defects(self, pid, fid, rev):
        def make(det):
            suggestion = suggest_column_defects(det.counts, det.valid)
            return det.description(), suggestion

        return self._cached(pid, fid, rev, "column-defects", make)

    def calibration(self, pid, rev, marks=None):
        project, _ = self._version(pid, None, rev)
        selected = project.state["marks"] if marks is None else marks
        model = fit_circle(selected, project.width, project.height)
        return {"fit": model, "revision": project.revision}

    def calibrate_distance(self, pid, rev, radius, d):
        project, _ = self._version(pid, None, rev)
        g = Geometry.read(project.state["geometry"], project.width, project.height)
        return {"model": reference_distance(g, radius, d), "revision": project.revision}

    def curve(self, pid, fid, rev, kind):
        def make(det):
            if kind == "profile":
                spec = ProfileSettings.read(det.settings["profile"])
                value = integrate_profile(det.counts, det.valid, det.geometry, spec)
            elif kind == "azimuth":
                spec = AzimuthSettings.read(det.settings["azimuth"], det.project.width, det.project.height)
                value = integrate_azimuth(det.counts, det.valid, det.geometry, spec)
            elif kind == "line":
                spec = LineSettings.read(det.settings["line"], det.project.width, det.project.height)
                value = cross_section(det.counts, det.valid, spec)
            else:
                raise WorkbenchError("未知的曲线类型。")
            return det.description(), value

        return self._cached(pid, fid, rev, kind, make)

    def curve_data(self, pid, fid, rev, kind, prominence=None):
        desc, product = self.curve(pid, fid, rev, kind)
        if kind == "profile" and prominence is not None:
            curve = product.document({"prominence": prominence})
        else:
            curve = product.document()
        return {**desc, **curve}

    def curve_png(self, pid, fid, rev, kind, bar):
        _, product = self.curve(pid, fid, rev, kind)
        return draw_curve_png(product, bar=bar)

    def curve_csv(self, pid, fid, rev, kind):
        _, product = self.curve(pid, fid, rev, kind)
        return product.csv()

    def cake(self, pid, fid, rev):
        def make(det):
            spec = CakeSettings.read(det.settings["cake"])
            output = integrate_cake(det.counts, det.valid, det.geometry, spec)
            return det.description(), output, det.display

        return self._cached(pid, fid, rev, "cake", make)

    def cake_data(self, pid, fid, rev, *, cells=False):
        desc, product, _ = self.cake(pid, fid, rev)
        obj = product.document() if cells else {
            "radial": product.radial.document(),
            "angular": product.angular.document(),
            "summary": product.moments.summary(),
        }
        return {**desc, **obj}

    def cake_probe(self, pid, fid, rev, x, y):
        desc, product, _ = self.cake(pid, fid, rev)
        return {**desc, "cell": probe_cake(product, x, y)}

    def cake_png(self, pid, fid, rev, support):
        _, product, display = self.cake(pid, fid, rev)
        return reduction_view(product, display, support=support)

    def cake_csv(self, pid, fid, rev):
        return self.cake(pid, fid, rev)[1].csv()

    def spot_data(self, pid, fid, rev):
        det = self.detector(pid, fid, rev)
        return {
            **det.description(),
            "aperture": asdict(det.aperture()),
            "spots": measure(det.counts, det.valid, det.geometry, det.spots(), det.aperture()),
        }

    def spot_csv(self, pid, fid, rev):
        return catalog_csv(self.spot_data(pid, fid, rev)["spots"])

    def scan_spots(self, pid, fid, rev, threshold, separation, maximum):
        det = self.detector(pid, fid, rev)
        found = find_candidates(
            det.counts, det.valid, det.geometry, det.aperture(), threshold, separation, maximum,
        )
        return {**det.description(), "candidates": found}

    def difference(self, pid, fid, rev):
        def make(det):
            image = det.snapshot
            if image.reference is None:
                raise WorkbenchError("设置差图时请选择一张参照曝光。")
            if image.reference_counts is None:
                raise WorkbenchError("主曝光和参照曝光需要是不同的帧。")
            ref_mask = det.other_mask(image.reference_counts)
            spec = DifferenceSettings.read(det.settings["difference"])
            diff = compare(det.counts, det.valid, image.reference_counts, ref_mask.valid, spec)
            return {**det.description(), "reference": image.reference.document()}, diff, spec

        return self._cached(pid, fid, rev, "difference", make)

    def difference_data(self, pid, fid, rev):
        desc, product, spec = self.difference(pid, fid, rev)
        _, scale = difference_view(product, spec)
        return {**desc, **product.document(), "display_limit": scale}

    def difference_png(self, pid, fid, rev):
        _, product, spec = self.difference(pid, fid, rev)
        return difference_view(product, spec)[0]

    def difference_npy(self, pid, fid, rev):
        return self.difference(pid, fid, rev)[1].npy()

    def inspect(self, pid, fid, rev, region, bins, interval):
        det = self.detector(pid, fid, rev)
        roi = Region.read(region, det.project.width, det.project.height)
        return det.description(), inspect_region(det.counts, det.valid, roi, bins, interval)

    def repeat(self, pid, fid, rev, frame_ids, minimum):
        frame_ids = unique(
            [token(fid, "重复曝光编号") for fid in items(frame_ids, "重复曝光", minimum=2, maximum=8)],
            "重复曝光",
        )
        project, frame = self._version(pid, fid, rev)
        if any(fid not in project.frame_ids for fid in frame_ids):
            raise WorkbenchError("重复曝光不在工程中。")
        ordered = tuple(sorted(frame_ids, key=project.frame_ids.index))
        key = (pid, project.revision, frame.id, "stability", minimum, *ordered)

        def build():
            snapshot, related = self.store.related_snapshot(pid, frame.id, ordered, project.revision)
            det = Detector(snapshot)
            return stability(
                ((meta, count, det.other_mask(count).valid) for meta, count in related),
                project.shape, minimum,
            )

        return self.recent.solve(key, build)

    def repeat_image(self, product, kind):
        image = product.deviation if kind == "std" else product.mean
        support = product.sufficient
        display = Display(stretch="sqrt")
        from .detector_io import png_bytes
        from .imaging import _thumbnail
        shown = colors(image, choose_window(image, support, display), "ice", display.stretch)
        return png_bytes(_thumbnail(shown, MAX_PREVIEW))

    def correction(self, pid, fid, rev, settings):
        spec = ResponseSettings.read(settings)
        project, frame = self._version(pid, fid, rev)
        if frame.id in (spec.dark, spec.flat):
            raise WorkbenchError("主曝光不能同时充当这次校正使用的参考图。")
        key = (pid, project.revision, frame.id, "correction", spec)

        def build():
            snapshot, related = self.store.related_snapshot(
                pid, frame.id, (spec.dark, spec.flat), project.revision,
            )
            det = Detector(snapshot)
            refs = {meta.id: (meta, data) for meta, data in related}
            dark = refs[spec.dark][1] if spec.dark is not None else None
            flat = refs[spec.flat][1] if spec.flat is not None else None
            common = np.ones(project.shape, dtype=bool)
            for _, data in related:
                common &= det.other_mask(data).valid
            result = correct(det.counts, dark, flat, det.valid, common, spec)
            desc = {**det.description(), "inputs": {
                "signal": frame.document(),
                "dark": None if spec.dark is None else refs[spec.dark][0].document(),
                "flat": None if spec.flat is None else refs[spec.flat][0].document(),
            }}
            return desc, result

        return self.recent.solve(key, build)

    def corrected_preview(self, result):
        from .detector_io import png_bytes
        from .imaging import _thumbnail
        display = Display(stretch="asinh")
        image = result.counts
        rgb = colors(image, choose_window(image, result.valid, display), display.palette, display.stretch)
        return png_bytes(_thumbnail(rgb, MAX_PREVIEW))

    def compare_profiles(self, pid, fid, rev, settings):
        spec = SeriesSettings.read(settings)
        project, frame = self._version(pid, fid, rev)
        if any(source not in project.frame_ids for source in spec.frames):
            raise WorkbenchError("比较的曝光不在工程中。")
        key = (pid, project.revision, frame.id, "series", spec)

        def build():
            snapshot, related = self.store.related_snapshot(pid, frame.id, spec.frames, project.revision)
            det = Detector(snapshot)
            support = [det.other_mask(data).valid for _, data in related]
            common = np.logical_and.reduce(support)
            profile = ProfileSettings.read(det.settings["profile"])
            product = compare_profile_series(related, common, support, det.geometry, profile, spec)
            return det.description(), product

        return self.recent.solve(key, build)

    def harmonic(self, pid, fid, rev, settings):
        spec = HarmonicSettings.read(settings)
        desc, curve = self.curve(pid, fid, rev, "azimuth")
        return desc, fit_orientation(curve, spec)

    def trace(self, pid, fid, rev, settings):
        spec = TraceSettings.read(settings)

        def make(det):
            return det.description(), trace_ring(det.counts, det.valid, det.geometry, spec)

        return self._cached(pid, fid, rev, "trace", make, (spec,))

    def measure_area(self, pid, fid, rev, settings):
        spec = MeasureSettings.read(settings)
        desc, curve = self.curve(pid, fid, rev, "profile")
        return desc, measure_area(curve, spec)

    def review_package(self, pid, fid, rev, chosen):
        det = self.detector(pid, fid, rev, reference="difference" in chosen)
        return produce_report(det, chosen)

    def contours(self, pid, fid, rev, settings):
        spec = ContourSettings.read(settings)

        def make(det):
            return det.description(), extract_contours(det.counts, det.valid, spec)

        return self._cached(pid, fid, rev, "contours", make, (spec,))

    def vectors(self, pid, fid, rev, settings):
        spec = VectorSettings.read(settings)
        det = self.detector(pid, fid, rev)
        return det.description(), measure_vectors(det.spots(), det.geometry, spec)

    def geometric(self, pid, fid, rev, settings):
        spec = GeometricResponseSpec.read(settings)

        def make(det):
            return det.description(), geometric_response(det.counts, det.valid, det.geometry, spec), det.display

        return self._cached(pid, fid, rev, "geometric", make, (spec,))

    def geometric_preview(self, product, display, kind):
        from .detector_io import png_bytes
        from .imaging import _thumbnail
        finite = product.valid
        if kind == "response":
            pixels = colors(product.factors, Window(0, 1), "support")
        else:
            image = product.values
            pixels = colors(image, choose_window(image, finite, display), display.palette, display.stretch)
        return png_bytes(_thumbnail(pixels, MAX_PREVIEW))

    def annular_coverage(self, pid, fid, rev, settings):
        det = self.detector(pid, fid, rev)
        spec = CoverageSpec.read(settings, det.project.shape)
        return det.description(), estimate_coverage(det.mask, det.geometry, spec)

    def distance_model(self, pid, rev, settings):
        project, _ = self._version(pid, None, rev)
        current = Geometry.read(project.state["geometry"], project.width, project.height)
        return fit_distances(current, settings)
