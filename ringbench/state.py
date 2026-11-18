"""Typed persisted workspace settings and coordinate-bound annotations."""
from dataclasses import asdict, dataclass

from .errors import WorkbenchError
from .geometry import Arc, Geometry, RADIAL_UNITS, ring_marks
from .masks import mask_layers
from .values import (
    boolean, integer, items, label, mapping, number, optional_ordered,
    pair, selection, token, xy_bounds,
)

COLOR_MAPS = ("ice", "sun", "gray")
STRETCHES = ("linear", "sqrt", "asinh")
VIEWS = (
    "detector", "profile", "azimuth", "cake", "line", "spots", "difference",
    "inspection", "stability", "correction", "comparison", "harmonics", "trace", "area",
    "contours", "vectors", "coverage", "geo_response",
)


@dataclass(frozen=True)
class Display:
    range_mode: str = "auto"
    lower: float = 0.0
    upper: float = 1000.0
    stretch: str = "asinh"
    palette: str = "ice"
    show_mask: bool = True

    @classmethod
    def read(cls, value):
        v = mapping(value, "显示参数")
        mode = selection(v.get("range_mode", "auto"), ("auto", "manual"), "计数窗口")
        low = number(v.get("lower", 0), "显示下限")
        high = number(v.get("upper", 1000), "显示上限")
        if mode == "manual" and high <= low:
            raise WorkbenchError("显示上限必须大于下限。")
        return cls(
            mode, low, high, selection(v.get("stretch", "asinh"), STRETCHES, "曲线"),
            selection(v.get("palette", "ice"), COLOR_MAPS, "色表"),
            boolean(v.get("show_mask", True), "显示遮罩"),
        )


@dataclass(frozen=True)
class ProfileSettings:
    unit: str = "px"
    bins: int = 160
    radial: tuple | None = None
    arc_start: float = 0
    arc_span: float = 360

    @property
    def arc(self):
        return Arc(self.arc_start, self.arc_span)

    @classmethod
    def read(cls, value):
        v = mapping(value, "径向积分参数")
        arc = Arc.read(v.get("arc_start", 0), v.get("arc_span", 360))
        return cls(
            selection(v.get("unit", "px"), RADIAL_UNITS, "径向单位"),
            integer(v.get("bins", 160), "径向分箱数", 12, 2048),
            optional_ordered(v.get("radial"), "径向范围", 0, 100_000),
            arc.start,
            arc.span,
        )


@dataclass(frozen=True)
class AzimuthSettings:
    inner_px: float
    outer_px: float
    bins: int = 72
    arc_start: float = 0
    arc_span: float = 360

    @property
    def arc(self):
        return Arc(self.arc_start, self.arc_span)

    @classmethod
    def default(cls, short_side):
        return cls(short_side * 0.20, short_side * 0.25)

    @classmethod
    def read(cls, value, width, height):
        v = mapping(value, "角向取样参数")
        cap = max(width, height) * 4
        inner = number(v.get("inner_px"), "取样内半径", 0, cap)
        outer = number(v.get("outer_px"), "取样外半径", 0, cap)
        if outer <= inner:
            raise WorkbenchError("取样外半径必须大于内半径。")
        arc = Arc.read(v.get("arc_start", 0), v.get("arc_span", 360))
        return cls(
            inner,
            outer,
            integer(v.get("bins", 72), "方位分箱数", 8, 720),
            arc.start,
            arc.span,
        )


@dataclass(frozen=True)
class CakeSettings:
    radial_bins: int = 180
    angular_bins: int = 120
    unit: str = "px"
    radial: tuple | None = None
    arc_start: float = 0
    arc_span: float = 360

    @property
    def arc(self):
        return Arc(self.arc_start, self.arc_span)

    @classmethod
    def read(cls, value):
        v = mapping(value, "极坐标图参数")
        arc = Arc.read(v.get("arc_start", 0), v.get("arc_span", 360))
        return cls(
            integer(v.get("radial_bins", 180), "径向分箱数", 24, 720),
            integer(v.get("angular_bins", 120), "方位分箱数", 18, 360),
            selection(v.get("unit", "px"), RADIAL_UNITS, "径向单位"),
            optional_ordered(v.get("radial"), "径向范围", 0, 100_000),
            arc.start,
            arc.span,
        )


@dataclass(frozen=True)
class LineSettings:
    a: tuple
    b: tuple
    half_width: float = 0.0
    step: float = 1.0

    @classmethod
    def default(cls, width, height):
        return cls((width * 0.08, height * 0.48), (width * 0.91, height * 0.48), 0.0, 1.0)

    @classmethod
    def read(cls, value, width, height):
        v = mapping(value, "横截面")
        bounds = xy_bounds(width, height, max(width, height))
        a = pair(v.get("a"), "起点", bounds)
        b = pair(v.get("b"), "终点", bounds)
        if (b[0]-a[0]) ** 2 + (b[1]-a[1]) ** 2 < 1:
            raise WorkbenchError("横截面端点至少相距一个像素。")
        return cls(
            a,
            b,
            number(v.get("half_width", 0), "半宽", 0, 30),
            number(v.get("step", 1), "剖面步距", 0.25, 20),
        )


@dataclass(frozen=True)
class Aperture:
    radius: float = 4.0
    background_inner: float = 7.0
    background_outer: float = 12.0

    @classmethod
    def read(cls, value):
        v = mapping(value, "斑点计数窗口")
        radius = number(v.get("radius", 4), "斑点半径", 1, 60)
        inner = number(v.get("background_inner", 7), "背景内半径", 1, 80)
        outer = number(v.get("background_outer", 12), "背景外半径", 2, 120)
        if not radius < inner < outer:
            raise WorkbenchError("斑点半径、背景内半径、背景外半径必须依次增大。")
        return cls(radius, inner, outer)


@dataclass(frozen=True)
class DifferenceSettings:
    reference: str | None = None
    normalization: str = "scale"
    scale: float = 1.0
    offset: float = 0.0
    window: float | None = None

    @classmethod
    def read(cls, value):
        v = mapping(value, "差图设置")
        ref = v.get("reference")
        return cls(
            token(ref, "参考帧") if ref is not None else None,
            selection(v.get("normalization", "scale"), ("scale", "fit"), "参照图归一化"),
            number(v.get("scale", 1), "参照图缩放", 0.001, 1000),
            number(v.get("offset", 0), "参照图偏移", -10 ** 12, 10 ** 12),
            number(v.get("window"), "差图显示窗口", 0.000001, 10 ** 15, allow_none=True),
        )


@dataclass(frozen=True)
class Spot:
    x: float
    y: float
    label: str

    @classmethod
    def read(cls, value, width, height):
        v = mapping(value, "斑点标记")
        x, y = pair([v.get("x"), v.get("y")], "斑点中心", xy_bounds(width, height, 0.5))
        return cls(x, y, label(v.get("label"), "斑点名", 48))


def reference_d_values(value):
    return [
        number(d, "参考 d 值 (Å)", 0.001, 5000)
        for d in items(value, "参考环间距", maximum=30)
    ]


def default_state(width, height):
    return {
        "geometry": Geometry.default(width, height).document(),
        "display": asdict(Display()),
        "masks": [],
        "marks": [],
        "reference_d": [],
        "profile": asdict(ProfileSettings()),
        "azimuth": asdict(AzimuthSettings.default(min(width, height))),
        "cake": asdict(CakeSettings()),
        "line": asdict(LineSettings.default(width, height)),
        "spots": [],
        "aperture": asdict(Aperture()),
        "difference": asdict(DifferenceSettings()),
        "active_frame": None,
        "active_view": "detector",
    }


def read_state(value, width, height, ids):
    v = mapping(value, "工程设置")
    g = Geometry.read(v.get("geometry"), width, height)
    display = Display.read(v.get("display"))
    masks = [layer.document() for layer in mask_layers(v.get("masks"), width, height)]
    marks = ring_marks(v.get("marks"), width, height)
    profile = ProfileSettings.read(v.get("profile"))
    azimuth = AzimuthSettings.read(v.get("azimuth"), width, height)
    cake = CakeSettings.read(v.get("cake"))
    line = LineSettings.read(v.get("line"), width, height)
    spots = [
        asdict(Spot.read(s, width, height))
        for s in items(v.get("spots"), "斑点标记", maximum=128)
    ]
    names = {spot["label"].casefold() for spot in spots}
    if len(spots) != len(names):
        raise WorkbenchError("斑点名称在同一工程中必须唯一。")
    difference = DifferenceSettings.read(v.get("difference"))
    if difference.reference is not None and difference.reference not in ids:
        raise WorkbenchError("差图设置引用了不在工程中的参考帧。")
    active_frame = v.get("active_frame")
    if active_frame not in ids:
        raise WorkbenchError("工程当前图像不在已保存的探测器帧列表中。")
    return {
        "geometry": g.document(),
        "display": asdict(display),
        "masks": masks,
        "marks": [list(point) for point in marks],
        "reference_d": reference_d_values(v.get("reference_d")),
        "profile": asdict(profile),
        "azimuth": asdict(azimuth),
        "cake": asdict(cake),
        "line": asdict(line),
        "spots": spots,
        "aperture": asdict(Aperture.read(v.get("aperture"))),
        "difference": asdict(difference),
        "active_frame": active_frame,
        "active_view": selection(v.get("active_view", "detector"), VIEWS, "工作视图"),
    }
