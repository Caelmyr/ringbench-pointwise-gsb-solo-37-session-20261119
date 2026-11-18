import {action, api, checkbox, el, escape, fields, fmt, link, numeric, on, select, stats, submit, table} from "./dom.js";
import {renderChart} from "./plot.js";

const units = [["px", "半径 px"], ["mm", "半径 mm"], ["2theta", "2θ °"], ["q", "q Å⁻¹"]];
const curveNames = {profile:"径向积分", azimuth:"角向曲线", line:"横截面"};

function arcControls(settings) {
  return fields([
    numeric("起始方位 °", "arc_start", settings.arc_start),
    numeric("方位宽度 °", "arc_span", settings.arc_span, {min:.1, max:360}),
  ], 2);
}

function radialControls(settings) {
  return fields([
    numeric("径向下界", "low", settings.radial?.[0], {min:0}),
    numeric("径向上界", "high", settings.radial?.[1], {min:0}),
  ], 2);
}

function readInterval(v) {
  const lo = v.number("low");
  const hi = v.number("high");
  return lo === null && hi === null ? null : [lo, hi];
}

function circleSettings(kind, spec) {
  const specInputs = kind === "profile" ? `${select("横轴单位", "unit", units, spec.unit)}
    ${numeric("径向分箱", "bins", spec.bins, {min:12, max:2048, step:1})}${radialControls(spec)}` :
    `${fields([numeric("内半径 px", "inner_px", spec.inner_px), numeric("外半径 px", "outer_px", spec.outer_px)], 2)}
    ${numeric("角向分箱", "bins", spec.bins, {min:8, max:720, step:1})}`;
  return `<form id="curve-form" class="side-group"><h2>${curveNames[kind]}参数</h2>
    ${specInputs}${arcControls(spec)}
    <button type="submit">更新曲线</button></form>`;
}

function lineSettings(spec) {
  return `<form id="curve-form" class="side-group"><h2>剖面线</h2>
    ${fields([numeric("起点 X","ax",spec.a[0]), numeric("起点 Y","ay",spec.a[1]),
      numeric("终点 X","bx",spec.b[0]), numeric("终点 Y","by",spec.b[1]),
      numeric("带宽半径 px","half_width",spec.half_width,{min:0,max:30}),
      numeric("采样步距 px","step",spec.step,{min:.25,max:20})], 2)}
    <button type="submit">更新横截面</button></form>`;
}

function readCurve(kind, v) {
  if (kind === "line") return {
    a:[v.number("ax"), v.number("ay")], b:[v.number("bx"), v.number("by")],
    half_width:v.number("half_width"), step:v.number("step"),
  };
  const quantities = {bins:v.number("bins"), arc_start:v.number("arc_start"), arc_span:v.number("arc_span")};
  if (kind === "profile") return {...quantities, unit:v.text("unit"), radial:readInterval(v)};
  return {...quantities, inner_px:v.number("inner_px"), outer_px:v.number("outer_px")};
}

function curveSummary(data) {
  if (data.kind === "line") return [
    ["路径长度", `${fmt(data.summary.length_px)} px`], ["有效取样",fmt(data.summary.observed_rows)],
    ["几何采样",fmt(data.summary.rows)], ["探测器外",fmt(data.summary.off_detector_rows)],
  ];
  return [
    ["计数像素",fmt(data.summary.included_pixels)], ["有效箱",fmt(data.summary.nonempty_cells)],
    ["整箱被遮",fmt(data.summary.masked_cells)], ["无像素箱",fmt(data.summary.off_detector_cells)],
  ];
}

function readCurveBin(ctx, data, index) {
  const value = data.values;
  const center = fmt(data.axis.centers[index], 4);
  const place = data.kind === "line" ? ` · (${fmt(value.x[index])}, ${fmt(value.y[index])})` : "";
  const support = data.kind === "line" ?
    `覆盖 ${fmt(100 * value.fraction[index], 1)}%` :
    `有效 ${fmt(value.count[index])}/${fmt(value.eligible[index])} 像素`;
  ctx.setReadout(`<strong>${center} ${escape(data.axis.unit)}${place} · ${fmt(value.mean[index], 3)} 计数</strong> · ${support}`);
}

function readPeak(ctx, peaks, index, unit) {
  const peak = peaks.find((v) => v.cell === index);
  if (!peak) return;
  ctx.setReadout(`<strong>径向峰 ${fmt(peak.coordinate, 4)} ${escape(unit)}</strong>
    · 突出度 ${fmt(peak.prominence, 3)} · 半突出度宽 ${fmt(peak.width_at_half_prominence, 4)}
    · ${fmt(peak.valid_pixels)} 像素`);
}

export async function curveTools(ctx) {
  const kind = ctx.state.active_view;
  const spec = ctx.state[kind];
  const isLine = kind === "line";
  ctx.outputs.innerHTML = link("下载曲线 CSV", ctx.url(`curves/${kind}.csv`));
  if (!isLine) ctx.outputs.innerHTML += link("下载图表 PNG", ctx.url(`curves/${kind}.png`), "quiet");
  ctx.controls.innerHTML = isLine ? lineSettings(spec) : circleSettings(kind, spec);
  ctx.controls.innerHTML += `<section class="side-group"><h2>曲线查看</h2>
    ${select("纵轴", "metric", [["mean","平均计数"],["count",isLine ? "有效覆盖比例" : "有效像素"],
      ["max","最大计数"]], ctx.session.curveMetric ?? "mean")}
    <p>${isLine ? "可在下方选择剖面线的两个端点。" : "图线中的空段表示原始探测器没有可用计数。"}</p>
    ${kind === "profile" ? `<form id="peaks-form" class="fields">
      ${numeric("最小径向峰突出度", "prominence", ctx.session.prominence, {min:0})}
      <button type="submit" class="secondary">检查径向峰</button></form>` : ""}
    </section>`;
  submit("curve-form", (v) => ctx.patch({[kind]:readCurve(kind, v)}));
  submit("peaks-form", (v) => {ctx.session.prominence = v.number("prominence"); ctx.render();});
  ctx.controls.querySelector("[name=metric]").addEventListener("change", (event) => {
    ctx.session.curveMetric = event.target.value;
    ctx.render();
  });
  const data = await ctx.get(`curves/${kind}`, kind === "profile" && ctx.session.prominence !== null &&
      ctx.session.prominence !== undefined ? {prominence:ctx.session.prominence} : {});
  if (!ctx.live()) return;
  const peaks = data.peaks ?? [];
  ctx.body.innerHTML = `<div class="plot-wrap">
    ${stats(curveSummary(data))}<div id="numeric-chart"></div>
    ${kind === "profile" ? `<h3>已识别的径向峰</h3>
    ${table(["位置", "平均计数", "突出度", "半突出度宽", "有效像素"],
    peaks.map((p) => [`${fmt(p.coordinate, 4)} ${data.axis.unit}`, fmt(p.mean), fmt(p.prominence),
       fmt(p.width_at_half_prominence, 4), fmt(p.valid_pixels)]))}` : ""}
    ${isLine ? '<h3>横截面在原探测器的位置</h3><div id="line-image"></div>' : ""}
    </div>`;
  renderChart(el("numeric-chart"), data, {
    title:curveNames[kind],
    metric:ctx.session.curveMetric ?? "mean",
    onRead:(i) => readCurveBin(ctx, data, i),
    onPeak:(i) => readPeak(ctx, peaks, i, data.axis.unit),
  });
  if (isLine) await ctx.stage({
    host:el("line-image"), guides:false, line:spec, anchor:ctx.session.lineAnchor,
    hint:"依次点击两个端点来改画横截面",
    onPoint:async (point) => {
      if (ctx.session.lineAnchor) {
        const a = ctx.session.lineAnchor;
        delete ctx.session.lineAnchor;
        await ctx.patch({line:{...spec,a,b:point}});
      } else {
        ctx.session.lineAnchor = point;
        ctx.render();
      }
    },
  });
}

export async function cakeTools(ctx) {
  const spec = ctx.state.cake;
  const layer = ctx.session.cakeLayer ?? "mean";
  ctx.outputs.innerHTML = `${link("下载极坐标 CSV",ctx.url("cake.csv"))}
    ${link("下载当前图 PNG",ctx.url("cake.png",{layer,download:1}),"quiet")}`;
  ctx.controls.innerHTML = `<form id="cake-form" class="side-group"><h2>极坐标网格</h2>
    ${select("径向单位", "unit", units, spec.unit)}
    ${fields([numeric("径向分箱","radial_bins",spec.radial_bins,{min:24,max:720,step:1}),
    numeric("方位分箱","angular_bins",spec.angular_bins,{min:18,max:360,step:1})], 2)}
    ${radialControls(spec)}${arcControls(spec)}<button type="submit">更新极坐标图</button></form>
    <section class="side-group">${select("显示图层","cake_layer",[["mean","平均计数"],["support","可用像素比例"]],layer)}</section>`;
  submit("cake-form", (v) => ctx.patch({cake:{
    radial_bins:v.number("radial_bins"), angular_bins:v.number("angular_bins"), unit:v.text("unit"),
    radial:readInterval(v), arc_start:v.number("arc_start"), arc_span:v.number("arc_span"),
  }}));
  ctx.controls.querySelector("[name=cake_layer]").addEventListener("change", (event) => {
    ctx.session.cakeLayer = event.target.value;
    ctx.render();
  });
  const grid = await ctx.get("cake");
  if (!ctx.live()) return;
  ctx.body.innerHTML = `<div class="plot-wrap">${stats([
    ["有效格",fmt(grid.summary.nonempty_cells)], ["计数像素",fmt(grid.summary.included_pixels)],
    ["探测器外格",fmt(grid.summary.off_detector_cells)],
  ])}<div id="cake-image"></div></div>`;
  const radial = grid.radial;
  const angular = grid.angular;
  await ctx.stage({
    host:el("cake-image"), source:ctx.url("cake.png",{layer}), overlays:false, cake:true,
    legend:[`径向 ${fmt(radial.edges[0])}–${fmt(radial.edges.at(-1))} ${radial.unit}`,
      layer === "mean" ? "平均计数" : "0 至 100% 有效", layer === "mean" ? ctx.state.display.palette : "support"],
    hint:`方位向上递增 ${fmt(angular.edges[0])}° 至 ${fmt(angular.edges.at(-1))}°，点击格子核对支撑`,
    onPoint:async (_point,[x,y]) => {
      const i = Math.floor(x*radial.centers.length);
      const j = Math.max(0, angular.centers.length - 1 - Math.floor(y*angular.centers.length));
      const {cell} = await ctx.get("cake/cell",{x:i,y:j});
      if (ctx.live()) ctx.setReadout(`<strong>${fmt(cell.radial[0], 4)}–${fmt(cell.radial[1],4)} ${escape(cell.unit)}</strong>
        · 方位 ${fmt(cell.angle[0])}°–${fmt(cell.angle[1])}° · ${fmt(cell.mean)} 计数
        · 有效 ${fmt(cell.valid_pixels)}/${fmt(cell.candidate_pixels)} 像素`);
    },
  });
}

export async function differenceTools(ctx) {
  const spec = ctx.state.difference;
  const refs = [["", "选择参照曝光"], ...ctx.current.frames.filter((frame) => frame.id !== ctx.chosen.id)
    .map((frame) => [frame.id, frame.title])];
  ctx.controls.innerHTML = `<form id="diff-form" class="side-group"><h2>曝光差图</h2>
    ${select("参照曝光","reference",refs,spec.reference ?? "")}
    ${select("归一化","normalization",[["scale","手动比例和偏移"],["fit","拟合相对曝光"]],spec.normalization)}
    ${fields([numeric("比例","scale",spec.scale), numeric("偏移","offset",spec.offset)], 2)}
    ${numeric("对称显示窗口", "window", spec.window)}
    ${checkbox("按环自动对齐参照图","align",spec.align)}
    ${numeric("平移搜索半径 px","align_search_px",spec.align_search_px ?? 6,{min:1,max:16})}
    <button type="submit">生成曝光差图</button></form>
    <div class="side-group"><p>差值为主曝光减去归一化的参照曝光。同一探测器位置上两图均有效时才计算差值。
    自动对齐只补偿参照图几像素的平移，不改变主曝光坐标；移出画幅或被遮的计数不参与比较。</p></div>`;
  submit("diff-form",(v) => ctx.patch({difference:{
    reference:v.text("reference") || null, normalization:v.text("normalization"),
    scale:v.number("scale"), offset:v.number("offset"), window:v.number("window"),
    align:v.has("align"), align_search_px:v.number("align_search_px"),
  }}));
  if (spec.reference === null || spec.reference === ctx.chosen.id) {
    ctx.body.innerHTML = '<p class="pending">选择另一张同尺寸曝光作为参照，即可生成差图。</p>';
    return;
  }
  const data = await ctx.get("difference");
  if (!ctx.live()) return;
  const alignment = data.alignment;
  const notice = alignment && !alignment.reliable ?
    `<p class="status-card" role="alert">${escape(alignment.message)}</p>` : "";
  ctx.outputs.innerHTML = `${link("下载差值 NPY",ctx.url("difference.npy"))}
    ${link("下载差图 PNG",ctx.url("difference.png",{download:1}),"quiet")}`;
  ctx.body.innerHTML = `<div class="plot-wrap">${notice}${stats([
    ["重合像素",fmt(data.overlap)], ["曝光比例",fmt(data.normalization.scale, 5)],
    ["计数偏移",fmt(data.normalization.offset, 4)], ["平均差值",fmt(data.residuals.mean, 4)],
  ])}${alignment?.applied ? `<p class="note">参照图已按环对齐：X ${fmt(alignment.shift_x, 2)} px，
    Y ${fmt(alignment.shift_y, 2)} px（相关性 ${fmt(alignment.correlation, 3)}）。</p>` :
    alignment?.reliable ? '<p class="note">按环对齐未发现需要补偿的平移，仍按同位比较。</p>' : ""}
    <div id="diff-image"></div></div>`;
  await ctx.stage({
    host:el("diff-image"), source:ctx.url("difference.png"), overlays:false,
    legend:[`${ctx.chosen.title} − 归一化的${data.reference.title}`,
      `−${fmt(data.display_limit)} 至 +${fmt(data.display_limit)}`, "signed"],
    hint:"蓝色为负差值，橙色为正差值，重合区外保持空白。",
  });
}

function spotLabel(known) {
  let i = 1;
  while (known.some((spot) => spot.label === `斑点 ${i}`)) i += 1;
  return `斑点 ${i}`;
}

export async function spotTools(ctx) {
  const spec = ctx.state.aperture;
  const spots = ctx.state.spots;
  ctx.outputs.innerHTML = link("下载斑点 CSV",ctx.url("spots.csv"));
  ctx.controls.innerHTML = `<form id="aperture-form" class="side-group"><h2>局部计数</h2>
    ${numeric("斑点半径 px","radius",spec.radius,{min:1,max:60})}
    ${fields([numeric("背景内半径","background_inner",spec.background_inner),
      numeric("背景外半径","background_outer",spec.background_outer)],2)}
    <button type="submit">更新计数窗口</button></form>
    <form id="candidate-form" class="side-group">
      <h2>搜索斑点</h2>${numeric("候选最低计数","threshold",ctx.session.spotThreshold)}
      ${numeric("相隔至少 px","separation",ctx.session.spotSeparation ?? 10,{min:1,max:80})}
      <button type="submit" class="secondary">寻找候选</button><div id="spot-candidates"></div>
    </form><section class="side-group"><h2>已标斑点</h2>
      ${spots.map((spot,index) => `<div class="spot-row"><span>${escape(spot.label)} · ${fmt(spot.x)}, ${fmt(spot.y)}</span>
        <button type="button" class="quiet" data-drop-spot="${index}">移除</button></div>`).join("")}</section>`;
  const add = (x,y) => ctx.patch({spots:[...spots,{x,y,label:spotLabel(spots)}]});
  ctx.controls.querySelectorAll("[data-drop-spot]").forEach((node) => node.addEventListener("click",() =>
    ctx.patch({spots:spots.filter((_, index) => index !== Number(node.dataset.dropSpot))})));
  submit("aperture-form",(v) => ctx.patch({aperture:{
    radius:v.number("radius"), background_inner:v.number("background_inner"), background_outer:v.number("background_outer"),
  }}));
  submit("candidate-form",async (v) => {
    const sep = v.number("separation");
    const threshold = v.number("threshold");
    ctx.session.spotThreshold = threshold;
    ctx.session.spotSeparation = sep;
    const search = {separation:sep};
    if (threshold !== null) search.threshold = threshold;
    const result = await ctx.own(() => ctx.get("spots/search",search));
    if (!result || !ctx.live()) return;
    const holder = el("spot-candidates");
    holder.innerHTML = result.candidates.slice(0,8).map((candidate,index) =>
      `<div class="spot-row"><span>${fmt(candidate.x)}, ${fmt(candidate.y)} · ${fmt(candidate.peak)}</span>
        <button type="button" class="quiet" data-candidate="${index}">加入</button></div>`).join("") ||
      '<p class="note">没有符合条件的候选。</p>';
    holder.querySelectorAll("[data-candidate]").forEach((node) => node.addEventListener("click",() => {
      const p = result.candidates[Number(node.dataset.candidate)];
      add(p.x,p.y);
    }));
  });
  const data = await ctx.get("spots");
  if (!ctx.live()) return;
  ctx.body.innerHTML = `<div class="plot-wrap"><div id="spot-image"></div>
    ${table(["标记","X","Y","q Å⁻¹","方位 °","净计数","可用像素"], data.spots.map((p) => [
      p.label,fmt(p.x),fmt(p.y),fmt(p.q_inverse_a,4),fmt(p.azimuth_deg),fmt(p.counts?.net),
      p.counts ? `${p.counts.valid_pixels}/${p.counts.full_pixels}` : "—",
    ]))}</div>`;
  await ctx.stage({
    host:el("spot-image"), spots:true, guides:false, hint:"点击斑点中心即添加局部计数",
    onPoint:([x,y]) => add(x,y),
  });
}
