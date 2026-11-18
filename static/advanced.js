import {action, checkbox, el, escape, fields, fmt, numeric, on, saveBlob, select, stats, submit, table} from "./dom.js";
import {renderChart} from "./plot.js";

const palette = ["#7adfba", "#ffbb72", "#7aa6ff", "#e9a6fd", "#ff738a", "#9bdef3"];

function exportData(ctx, route, output, settings, title, filename, extra = {}) {
  ctx.outputs.insertAdjacentHTML("beforeend", action(title, `send-${output.replace(".", "-")}`, extra.quiet ? "quiet" : ""));
  on(`send-${output.replace(".", "-")}`, "click", async () => {
    const payload = await ctx.own(() => ctx.analysis(route, {settings, output}));
    if (payload) saveBlob(payload, filename);
  });
}

function optionalPair(value, low, high) {
  const l = value.number(low);
  const h = value.number(high);
  return l === null && h === null ? null : [l,h];
}

function keyInput(title, type, pairs, selected) {
  return `<section class="side-group"><h3>${escape(title)}</h3>${pairs.map(([key,shown]) =>
    `<label class="checkbox"><input type="checkbox" name="${escape(type)}" value="${escape(key)}"
    ${selected.includes(key) ? "checked" : ""}>${escape(shown)}</label>`).join("")}</section>`;
}

export async function comparisonTools(ctx) {
  const route = "comparison";
  const registered = ctx.current.frames.map((f) => [f.id,f.title]);
  const settings = ctx.session.comparison ?? {frames:registered.slice(0,2).map(([id])=>id), support:"common",
    normalization:"raw", anchor:null};
  ctx.controls.innerHTML = `<form id="comparison-form" class="fields">
    ${keyInput("选择径向谱来源","series_frame",registered,settings.frames)}
    <section class="side-group"><h2>径向谱比较</h2>
    ${select("有效像素","support",[["common","各图共同像素"],["per_frame","各使用本图像素"]],settings.support)}
    ${select("计数标度","normalization",[["raw","保留原计数"],["area","按选区峰面积对齐"]],settings.normalization)}
    ${fields([numeric("对照区下界","low",settings.anchor?.[0]),numeric("对照区上界","high",settings.anchor?.[1])],2)}
    <p>比较使用径向积分页的坐标和分箱设置。归一以列表中首帧为参考。</p><button type="submit">比较径向谱</button>
    </section></form>`;
  submit("comparison-form",(v,form) => {
    ctx.session.comparison = {
      frames:new FormData(form).getAll("series_frame"), support:v.text("support"),
      normalization:v.text("normalization"), anchor:optionalPair(v,"low","high"),
    };
    ctx.render();
  });
  if (settings.frames.length < 2) {
    ctx.body.innerHTML = '<p class="pending">请选择两张以上的同尺寸曝光。</p>';
    return;
  }
  const data = await ctx.analysis(route,{settings});
  if (!ctx.live()) return;
  exportData(ctx,route,"csv",settings,"下载叠加谱 CSV","radial-series.csv");
  const series = data.frames.map((row,i) => ({values:row.values,color:palette[i]}));
  ctx.body.innerHTML = `<div class="plot-wrap"><div class="button-row">
    ${data.frames.map((row,i)=>`<span class="note" style="color:${palette[i]}">${escape(row.frame.title)}</span>`).join("")}</div>
    <div id="advanced-plot"></div>${table(["对照曝光","共同分箱","平均差值","RMS 差值","最大差值坐标"],data.metrics.map((p)=>[
      p.title,fmt(p.shared_bins),fmt(p.signed_mean_difference),fmt(p.rms_difference),fmt(p.max_difference?.position),
    ]))}</div>`;
  renderChart(el("advanced-plot"),{axis:data.axis,values:{mean:series[0].values}},{
    title:"对比曝光径向积分",extra:series.slice(1),color:palette[0],
    onRead:(i)=>ctx.setReadout(`<strong>${fmt(data.axis.centers[i],4)} ${escape(data.axis.unit)}</strong>
      · ${data.frames.map((row)=>`${escape(row.frame.title)} ${fmt(row.values[i])}`).join(" · ")}
      · 共用 ${fmt(data.common_pixels[i])} 像素`),
  });
}

export async function harmonicTools(ctx) {
  const route = "harmonics";
  const settings = ctx.session.harmonic ?? {order:2,minimum_pixels:2,minimum_fraction:.25,weights:"equal"};
  ctx.controls.innerHTML = `<form id="harmonic-form" class="side-group"><h2>取向谐波</h2>
    ${numeric("最高阶次","order",settings.order,{min:1,max:8,step:1})}
    ${numeric("每箱至少像素","minimum_pixels",settings.minimum_pixels,{min:1,step:1})}
    ${numeric("每箱覆盖下限","minimum_fraction",settings.minimum_fraction,{min:0,max:1})}
    ${select("方位权重","weights",[["equal","各箱等权"],["pixels","按像素数"]],settings.weights)}
    <p>使用角向曲线保存的环带。拟合线可越过缺口，只有标记为有效的箱实际参与计算。</p>
    <button type="submit">拟合取向</button></form>`;
  submit("harmonic-form",(v)=>{
    ctx.session.harmonic = {
      order:v.number("order"),minimum_pixels:v.number("minimum_pixels"),minimum_fraction:v.number("minimum_fraction"),
      weights:v.text("weights"),
    };
    ctx.render();
  });
  const data = await ctx.analysis(route,{settings});
  if (!ctx.live()) return;
  exportData(ctx,route,"csv",settings,"下载取向 CSV","orientation.csv");
  ctx.body.innerHTML = `<div class="plot-wrap">${stats([
    ["取向角",data.principal.direction_deg === null ? "—" : `${fmt(data.principal.direction_deg)}°`],
    ["最高阶振幅",fmt(data.principal.amplitude)],["参与角箱",fmt(data.summary.sample_bins)],
    ["加权 RMS",fmt(data.summary.weighted_rms)],["最大角间隙",`${fmt(data.summary.largest_circular_gap_deg)}°`],
  ])}<div id="advanced-plot"></div>
  ${table(["阶次","振幅","基准取向 °","周期 °","振幅标准误"],data.equation.components.map((h)=>[
    h.order,fmt(h.amplitude),fmt(h.direction_deg),fmt(h.period_deg),fmt(h.amplitude_standard_error),
  ]))}</div>`;
  renderChart(el("advanced-plot"),data,{
    title:"实测角向曲线与拟合",extra:[{values:data.values.fitted,color:palette[1]}],
    onRead:(i)=>ctx.setReadout(`<strong>${fmt(data.axis.centers[i])}°</strong> · 测得 ${fmt(data.values.mean[i])}
    · 拟合 ${fmt(data.values.fitted[i])} · ${data.values.fit_used[i] ? "用于拟合" : "未参与拟合"}`),
  });
}

function traceDefaults(ctx) {
  const a = ctx.state.azimuth;
  return {radius:(a.inner_px+a.outer_px)/2,tolerance:6,angular_bins:72,
    minimum_pixels:2,minimum_prominence:10,arc_start:a.arc_start,arc_span:a.arc_span};
}

export async function traceTools(ctx) {
  const route = "trace";
  const settings = ctx.session.trace ?? traceDefaults(ctx);
  ctx.controls.innerHTML = `<form id="trace-form" class="side-group"><h2>跟踪参考环</h2>
    ${fields([numeric("目标半径 px","radius",settings.radius,{min:2}),numeric("搜索半宽 px","tolerance",settings.tolerance,{min:1,max:100})],2)}
    ${numeric("方位扇区数","angular_bins",settings.angular_bins,{min:24,max:360,step:1})}
    ${fields([numeric("最少像素","minimum_pixels",settings.minimum_pixels,{min:1,step:1}),
    numeric("最低突出度","minimum_prominence",settings.minimum_prominence,{min:0})],2)}
    ${fields([numeric("方位起点 °","arc_start",settings.arc_start),numeric("覆盖角 °","arc_span",settings.arc_span,{min:.1,max:360})],2)}
    <button type="submit">跟踪环线</button></form>`;
  submit("trace-form",(v)=>{
    ctx.session.trace = Object.fromEntries(Object.keys(settings).map((key)=>[key,v.number(key)]));
    ctx.render();
  });
  const data = await ctx.analysis(route,{settings});
  if (!ctx.live()) return;
  exportData(ctx,route,"csv",settings,"下载环线 CSV","ring-trace.csv");
  const s=data.summary;
  ctx.body.innerHTML = `<div class="plot-wrap">${stats([
    ["目标半径",`${fmt(s.expected_radius_px)} px`],["有效扇区",`${s.observed_sectors}/${s.candidate_sectors}`],
    ["平均半径偏差",fmt(s.mean_deviation_px,4)],["偏差 RMS",fmt(s.deviation_rms_px,4)],
  ])}<h3>按方位的环线径向偏差</h3><div id="advanced-plot"></div><div id="advanced-image"></div></div>`;
  renderChart(el("advanced-plot"),data.series, {
    title:"参考环径向偏差",onRead:(i)=>{
      const p=data.points[i];
      ctx.setReadout(`<strong>${fmt(p.angle_deg)}° · 半径 ${fmt(p.radius_px,4)} px</strong>
        · 偏差 ${fmt(p.deviation_px,4)} px · ${fmt(p.peak_pixels)} 像素`);
    },
  });
  await ctx.stage({host:el("advanced-image"),guides:false, markers:data.points.filter((p)=>p.x_px!==null)
    .map((p)=>[p.x_px,p.y_px]),hint:"仅在找到有效峰的方位标点。"});
}

export async function areaTools(ctx) {
  const route = "peak-area";
  const detected = await ctx.get("curves/profile");
  if (!ctx.live()) return;
  const point = detected.peaks?.find((peak)=>peak.cell>12) ?? detected.peaks?.[0];
  const edgeSize = detected.axis.edges[1] - detected.axis.edges[0];
  const seed = point?.coordinate ?? detected.axis.centers[Math.floor(detected.axis.centers.length / 2)];
  const settings = ctx.session.peakArea ?? {
    peak:[seed-2*edgeSize,seed+2*edgeSize],left:[seed-6*edgeSize,seed-2.5*edgeSize],
    right:[seed+2.5*edgeSize,seed+6*edgeSize],baseline:"gradient",
  };
  const band = (key,label)=>fields([numeric(`${label}起点`,`${key}_low`,settings[key][0]),
    numeric(`${label}终点`,`${key}_high`,settings[key][1])],2);
  ctx.controls.innerHTML = `<form id="area-form" class="side-group"><h2>积分单个峰区</h2>
    ${band("left","峰前背景")}${band("peak","峰区")}${band("right","峰后背景")}
    ${select("基线","baseline",[["gradient","连接两侧平均值"],["constant","两侧等高基线"]],settings.baseline)}
    <button type="submit">计算峰区面积</button></form>`;
  submit("area-form",(v)=>{
    ctx.session.peakArea = {...Object.fromEntries(["left","peak","right"]
      .map((key)=>[key,[v.number(`${key}_low`),v.number(`${key}_high`)]])),baseline:v.text("baseline")};
    ctx.render();
  });
  const data = await ctx.analysis(route,{settings});
  if (!ctx.live()) return;
  exportData(ctx,route,"csv",settings,"下载峰区 CSV","peak-area.csv");
  ctx.body.innerHTML = `<div class="plot-wrap">${stats([
    ["净积分面积",fmt(data.net_area)],["估计背景面积",fmt(data.background_area)],
    ["正余量质心",fmt(data.net_positive_centroid,4)],["窗口有效宽度",`${fmt(data.fraction_observed*100)}%`],
  ])}<div id="advanced-plot"></div>${data.warnings.map((w)=>`<p class="note">${escape(w)}</p>`).join("")}</div>`;
  renderChart(el("advanced-plot"),data,{title:"峰区与参考基线",extra:[{values:data.values.baseline,color:palette[1]}],
    onRead:(i)=>ctx.setReadout(`<strong>${fmt(data.axis.centers[i],4)} ${escape(data.unit)}</strong>
    · 测得 ${fmt(data.values.mean[i])} · 背景 ${fmt(data.values.baseline[i])}`),
  });
}

export async function contourTools(ctx) {
  const route = "contours";
  const settings = ctx.session.contour ?? {mode:"percentile",level:null,percentile:96,minimum:20,maximum:32};
  ctx.controls.innerHTML = `<form id="contour-form" class="side-group"><h2>等强线</h2>
    ${select("计算阈值","mode",[["percentile","有效计数百分位"],["fixed","固定计数"]],settings.mode)}
    ${numeric("固定计数","level",settings.level)}
    ${numeric("计数百分位","percentile",settings.percentile,{min:.1,max:99.99})}
    ${fields([numeric("最短长度 px","minimum",settings.minimum,{min:0}),
      numeric("保留线条数","maximum",settings.maximum,{min:1,max:160,step:1})],2)}
    <button type="submit">提取等强线</button><p>线条到空白或遮罩边界会断开。可下载原始探测器坐标。</p>
    </form>`;
  submit("contour-form",(v)=>{
    ctx.session.contour = {mode:v.text("mode"),level:v.number("level"),percentile:v.number("percentile"),
      minimum:v.number("minimum"),maximum:v.number("maximum")};
    ctx.render();
  });
  const data = await ctx.analysis(route,{settings});
  if (!ctx.live()) return;
  exportData(ctx,route,"csv",settings,"下载等强线 CSV","intensity-boundaries.csv");
  const s = data.summary;
  ctx.body.innerHTML = `<div class="plot-wrap">${stats([
    ["等强计数",fmt(s.isovalue,4)],["追踪线条",fmt(s.traced_paths)],
    ["显示线条",fmt(s.displayed_paths)],["闭合线条",fmt(s.closed_paths)],
  ])}<div id="advanced-image"></div>
  ${table(["等强线","路径长度 px","封闭面积 px²","节点"],data.paths.map((p,i)=>[
    i+1,fmt(p.length_px),fmt(p.area_px2),fmt(p.vertex_count),
  ]))}</div>`;
  await ctx.stage({host:el("advanced-image"),guides:false,polylines:data.paths.map((item)=>item.xy),
    hint:"等强线叠加在原始探测器行列上。"});
}

export async function vectorTools(ctx) {
  const route = "vectors";
  const labels = ctx.state.spots.map((s)=>s.label);
  const available = labels.map((l)=>[l,l]);
  const settings = ctx.session.vectors ?? {labels:labels.slice(0,4),grouping:"pairs",reference:labels[0]};
  ctx.controls.innerHTML = `<form id="vectors-form" class="fields">
    ${keyInput("选中已标斑点","spot_id",available,settings.labels)}
    <section class="side-group"><h2>散射向量</h2>
    ${select("比较对象","grouping",[["pairs","每两个斑点"],["reference","依次与基准比较"]],settings.grouping)}
    ${select("基准斑点","reference",available,settings.reference)}
    <button type="submit">计算散射向量</button><p>qₓ 向右，qᵧ 向下，q𝓏 沿入射束轴投影。</p></section></form>`;
  submit("vectors-form",(v,f)=>{
    ctx.session.vectors = {labels:new FormData(f).getAll("spot_id"),grouping:v.text("grouping"),
      reference:v.text("reference")};
    ctx.render();
  });
  if (labels.length < 2 || settings.labels.length < 2) {
    ctx.body.innerHTML = '<p class="pending">在斑点测量中标记并选择至少两个斑点。</p>';
    return;
  }
  const data = await ctx.analysis(route,{settings});
  if (!ctx.live()) return;
  exportData(ctx,route,"pairs",settings,"下载向量差 CSV","reciprocal-pairs.csv");
  exportData(ctx,route,"vectors",settings,"下载散射向量 CSV","reciprocal-vectors.csv",{quiet:true});
  ctx.body.innerHTML = `<div class="plot-wrap">${stats([
    ["已测斑点",data.summary.spots],["斑点对",data.summary.comparisons],
    ["最小正间隔 Å⁻¹",fmt(data.summary.smallest_positive_separation,4)],
  ])}<h3>散射向量</h3>${table(["斑点","qₓ","qᵧ","q𝓏","|q|","d Å"],data.vectors.map((s)=>[
    s.label,fmt(s.qx,4),fmt(s.qy,4),fmt(s.qz,4),fmt(s.q_magnitude,4),fmt(s.d_a,4),
  ]))}
  <h3>斑点之间的间隔</h3>${table(["起点","终点","像素间隔","|Δq| Å⁻¹","向量夹角"],data.pairs.map((p)=>[
    p.first,p.second,fmt(p.detector_distance_px),fmt(p.delta_q,4),fmt(p.angle_between_q_deg),
  ]))}<div id="advanced-image"></div></div>`;
  const byLabel = new Map(data.vectors.map((spot)=>[spot.label,spot]));
  const linePoints = data.pairs.map((pair)=>[byLabel.get(pair.first),byLabel.get(pair.second)]
    .map((spot)=>[spot.detector_x_px,spot.detector_y_px]));
  await ctx.stage({host:el("advanced-image"),guides:false,spots:true,polylines:linePoints,
    hint:"连线表示当前分析的斑点对；线段位置与原图一致。"});
}

export async function coverageTools(ctx) {
  const route = "coverage";
  const az = ctx.state.azimuth;
  const settings = ctx.session.coverage ?? {inner:az.inner_px,outer:az.outer_px,bins:72,start:az.arc_start,span:az.arc_span};
  ctx.controls.innerHTML = `<form id="coverage-form" class="side-group"><h2>测量环带覆盖</h2>
    ${fields([numeric("内径 px","inner",settings.inner,{min:0}),numeric("外径 px","outer",settings.outer,{min:.01})],2)}
    ${fields([numeric("起始角 °","start",settings.start),numeric("方位宽度 °","span",settings.span,{min:.1,max:360})],2)}
    ${numeric("角度分箱","bins",settings.bins,{min:12,max:360,step:1})}
    <p>分母为完整选择的环带面积，包括探测器范围之外的区域。</p>
    <button type="submit">计算环带覆盖</button></form>`;
  submit("coverage-form",(v)=>{
    ctx.session.coverage = Object.fromEntries(Object.keys(settings).map((name)=>[name,v.number(name)]));
    ctx.render();
  });
  const data = await ctx.analysis(route,{settings});
  if (!ctx.live()) return;
  exportData(ctx,route,"csv",settings,"下载覆盖 CSV","annular-coverage.csv");
  const s=data.summary;
  ctx.body.innerHTML = `<div class="plot-wrap">${stats([
    ["探测器内",`${fmt(s.detector_fraction*100)}%`],["可用面积",`${fmt(s.supported_fraction*100)}%`],
    ["遮罩面积",`${fmt(s.masked_fraction*100)}%`],["最大空弧",`${fmt(s.longest_unsupported_arc_deg)}°`],
  ])}<div id="advanced-plot"></div><div id="advanced-image"></div></div>`;
  renderChart(el("advanced-plot"),data,{title:"探测器与有效环面覆盖",extra:[{values:data.values.on_detector,color:palette[1]}],
    onRead:(i)=>ctx.setReadout(`<strong>${fmt(data.axis.centers[i])}°</strong> · 探测器内 ${fmt(data.values.on_detector[i]*100)}%
       · 可用 ${fmt(data.values.mean[i]*100)}% · ${fmt(data.values.masked[i])} 个遮罩采样`),
  });
  await ctx.stage({host:el("advanced-image"),guides:false,
    radii:[data.inner_px,data.outer_px],hint:"轮廓标出在原探测器坐标下参与计算的径向边界。"});
}

export async function geometricTools(ctx) {
  const route = "geometric-correction";
  const settings = ctx.session.geomResponse ?? {solid_angle:true,polarization:"none",fraction:.5,direction:0,minimum:.08};
  const view = ctx.session.geoLayer ?? "corrected";
  ctx.controls.innerHTML = `<form id="geo-response-form" class="side-group"><h2>几何响应</h2>
    ${checkbox("按相对立体角校正","solid_angle",settings.solid_angle)}
    ${select("偏振模式","polarization",[["none","无"],["mixed","正交偏振混合"]],settings.polarization)}
    ${numeric("主方向偏振分数","fraction",settings.fraction,{min:0,max:1})}
    ${numeric("主方向 °","direction",settings.direction)}
    ${numeric("最小相对响应","minimum",settings.minimum,{min:.00001,max:.95})}
    <button type="submit">更新响应校正</button></form>
    <section class="side-group">${select("结果图层","geo_view",[["corrected","校正计数"],["response","相对响应"]],view)}</section>`;
  submit("geo-response-form",(v)=>{
    ctx.session.geomResponse = {solid_angle:v.has("solid_angle"),polarization:v.text("polarization"),
      fraction:v.number("fraction"),direction:v.number("direction"),minimum:v.number("minimum")};
    ctx.render();
  });
  ctx.controls.querySelector("[name=geo_view]").addEventListener("change",(event)=>{
    ctx.session.geoLayer=event.target.value; ctx.render();
  });
  const data = await ctx.analysis(route,{settings});
  if (!ctx.live()) return;
  exportData(ctx,route,`${view}.npy`,settings,"下载当前 NPY",`geometry-${view}.npy`);
  exportData(ctx,route,"stats.csv",settings,"下载遮罩统计","geometry-flags.csv",{quiet:true});
  ctx.body.innerHTML = `<div class="plot-wrap">${stats([
    ["输入均值",fmt(data.source_counts.mean)],["校正均值",fmt(data.output_counts.mean)],
    ["相对响应均值",fmt(data.relative_response.mean,4)],["保留像素",fmt(data.output_counts.count)],
  ])}<div id="advanced-image"></div>${table(["状态","像素"],data.rows.map((r)=>[r.reason,fmt(r.pixels)]))}</div>`;
  const image = await ctx.analysis(route,{settings,output:`${view}.png`});
  if (ctx.live()) await ctx.stage({host:el("advanced-image"),source:image,overlays:false,
    legend:[view === "corrected" ? "几何校正后计数" : "相对几何响应", "正入射平面探测器",
      view === "corrected" ? ctx.state.display.palette : "support"],hint:"相对响应过低的像素保持空白。"});
}
