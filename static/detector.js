import {action, api, checkbox, el, escape, fields, fmt, link, numeric, on, saveBlob, select, stats, submit, table} from "./dom.js";

const colors = {mint: "#7df6ca", orange: "#ffba69", cyan: "#87eef5"};

function shape(pen, color, width, build) {
  pen.beginPath();
  pen.strokeStyle = color;
  pen.lineWidth = width;
  build();
  pen.stroke();
}

function overlay(pen, canvas, config, ctx) {
  if (config.overlays === false) return;
  const {current, state, session} = ctx;
  const g = state.geometry;
  const X = (x) => (x + 0.5) * canvas.width / current.width;
  const Y = (y) => (y + 0.5) * canvas.height / current.height;
  const cx = X(g.cx);
  const cy = Y(g.cy);
  const rx = (r) => r * canvas.width / current.width;
  const ry = (r) => r * canvas.height / current.height;
  const oval = (x, y, radius, color, lineWidth = 1) => shape(pen, color, lineWidth,
    () => pen.ellipse(X(x), Y(y), rx(radius), ry(radius), 0, 0, 2*Math.PI));
  if (config.guides !== false) {
    shape(pen, colors.mint, 1.25, () => {
      pen.moveTo(cx-11, cy); pen.lineTo(cx+11, cy);
      pen.moveTo(cx, cy-11); pen.lineTo(cx, cy+11);
    });
    const rings = config.reference ?? [];
    for (const ring of rings) oval(g.cx, g.cy, ring.radius_px, "#7df6ca83");
    const used = session.fit?.used;
    for (const [index, [x, y]] of state.marks.entries()) {
      const dropped = used?.[index] === false;
      shape(pen, dropped ? "#ff9585" : colors.orange, dropped ? 2 : 1.5, () => {
        pen.arc(X(x), Y(y), 3.8, 0, 2*Math.PI);
        if (dropped) { pen.moveTo(X(x)-3.5, Y(y)-3.5); pen.lineTo(X(x)+3.5, Y(y)+3.5); }
      });
    }
    if (session.fit && session.fit.status === "ok") {
      oval(session.fit.cx, session.fit.cy, session.fit.radius_px, "#f6a9fd");
    }
  }
  if (config.line) {
    const {a, b, half_width} = config.line;
    const d = Math.hypot(b[0]-a[0], b[1]-a[1]);
    const dx = (a[1]-b[1]) / d * half_width;
    const dy = (b[0]-a[0]) / d * half_width;
    const line = (sign, color) => shape(pen, color, 1.4, () => {
      pen.moveTo(X(a[0]+sign*dx), Y(a[1]+sign*dy));
      pen.lineTo(X(b[0]+sign*dx), Y(b[1]+sign*dy));
    });
    line(0, colors.orange);
    if (half_width > 0) { line(-1, "#ffba6990"); line(1, "#ffba6990"); }
  }
  if (config.spots) {
    for (const [index, spot] of state.spots.entries()) {
      oval(spot.x, spot.y, Math.max(3, state.aperture.radius), colors.cyan, 1.6);
      pen.fillStyle = colors.cyan;
      pen.font = "12px system-ui";
      pen.fillText(String(index + 1), X(spot.x) + 8, Y(spot.y) - 8);
    }
  }
  if (config.roi) {
    const {start, stop} = config.roi;
    pen.strokeStyle = colors.orange;
    pen.lineWidth = 1.5;
    pen.strokeRect(X(start[0] - .5), Y(start[1] - .5),
      X(stop[0] - .5) - X(start[0] - .5), Y(stop[1] - .5) - Y(start[1] - .5));
  }
  for (const radius of config.radii ?? []) oval(g.cx, g.cy, radius, "#ffba69ae", 1.5);
  for (const [x,y] of config.markers ?? []) {
    shape(pen, colors.orange, 1.6, () => pen.arc(X(x), Y(y), 2.5, 0, 2*Math.PI));
  }
  for (const line of config.polylines ?? []) {
    shape(pen, colors.orange, 1.2, () => {
      line.forEach(([x,y], index) => index === 0 ? pen.moveTo(X(x),Y(y)) : pen.lineTo(X(x),Y(y)));
    });
  }
  if (config.anchor) {
    const [x, y] = config.anchor;
    shape(pen, colors.orange, 2, () => pen.arc(X(x), Y(y), 6, 0, 2*Math.PI));
  }
}

export async function drawStage(ctx, config) {
  if (!ctx.live()) return;
  const root = config.host ?? ctx.body;
  const legend = config.legend ?? ["原始探测器行列", "", "ice"];
  root.innerHTML = `<div class="detector-stage">
    <div class="stage-legend"><span>${escape(legend[0])}</span><span>${escape(legend[1])}
    <span class="legend-strip ${escape(legend[2])}" aria-hidden="true"></span></span></div>
    <canvas id="detector-canvas" class="detector-canvas ${config.cake ? "cake" : ""}"
      aria-label="${escape(config.label ?? "探测器图像")}"></canvas>
    <div class="stage-legend">${escape(config.hint ?? "点击图像读取像素")}</div>
    </div>`;
  const canvas = root.querySelector("#detector-canvas");
  const image = new Image();
  const source = config.source ?? ctx.url("detector.png");
  const src = source instanceof Blob ? URL.createObjectURL(source) : source;
  const loaded = new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error("图像预览无法读取，请重新生成。"));
    image.src = src;
  });
  try {
    await loaded;
  } finally {
    if (source instanceof Blob) URL.revokeObjectURL(src);
  }
  if (!ctx.live() || !canvas.isConnected) return;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  canvas.style.setProperty("--aspect", canvas.width / canvas.height);
  const pen = canvas.getContext("2d");
  pen.drawImage(image, 0, 0);
  overlay(pen, canvas, config, ctx);
  canvas.addEventListener("click", (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.min(0.999999, Math.max(0, (event.clientX-rect.left)/rect.width));
    const y = Math.min(0.999999, Math.max(0, (event.clientY-rect.top)/rect.height));
    const xy = [Math.floor(x*ctx.current.width), Math.floor(y*ctx.current.height)];
    ctx.own(() => config.onPoint?.(xy, [x, y]));
  });
}

const displayModes = [["auto","有效计数百分位"], ["manual","固定计数范围"]];
const maps = [["ice","青蓝"],["sun","铜红"],["gray","黑白"]];
const stretches = [["linear","线性"],["sqrt","平方根"],["asinh","反双曲正弦"]];

function displayPanel(display) {
  return `<form class="side-group" id="display-form">
  <h2>计数窗口</h2>${select("窗口", "range_mode", displayModes, display.range_mode)}
  ${fields([numeric("下限", "lower", display.lower), numeric("上限", "upper", display.upper)], 2)}
  ${fields([select("拉伸", "stretch", stretches, display.stretch), select("色表", "palette", maps, display.palette)], 2)}
  ${checkbox("显示被遮像素", "show_mask", display.show_mask)}<button type="submit" class="secondary">应用显示</button></form>`;
}

function geometryPanel(g, markCount, references, fit) {
  let model = "";
  if (fit) {
    const dropped = fit.rejected ?? [];
    const droppedText = dropped.length
      ? `<p class="note">未采用的标点 ${dropped.map((point) =>
          `#${point.index + 1} (${fmt(point.x, 1)}, ${fmt(point.y, 1)})`).join("、")}</p>`
      : "";
    model = fit.status === "ok"
      ? `<div class="status-card control-result" id="fit-result">
  <h3>拟合环心 ${fmt(fit.cx, 3)}, ${fmt(fit.cy, 3)}</h3>
  <p>半径 ${fmt(fit.radius_px, 3)} px · RMS ${fmt(fit.rms_px, 3)} px</p>
  <p>覆盖 ${fmt(fit.span_deg, 1)}° · 采用 ${fit.used_count}/${fit.count} 个标点</p>
  ${droppedText}
  ${action("应用拟合环心", "apply-fit")}</div>`
      : `<div class="status-card control-result" id="fit-result" role="alert">
  <h3>环心未能确定</h3>
  <p>${escape(fit.message)}</p>
  ${dropped.length ? `<ul class="note">${dropped.map((point) =>
    `<li>#${point.index + 1} (${fmt(point.x, 1)}, ${fmt(point.y, 1)})</li>`).join("")}</ul>` : ""}
  <p>现有环心保持 ${fmt(g.cx, 3)}, ${fmt(g.cy, 3)}。</p></div>`;
  }
  return `<section class="side-group">
  <h2>几何与标定</h2>
  <form id="geometry-form" class="fields">
  ${fields([numeric("环心 X", "cx", g.cx), numeric("环心 Y", "cy", g.cy)], 2)}
  ${fields([numeric("像素 mm", "pixel_mm", g.pixel_mm), numeric("距离 mm", "distance_mm", g.distance_mm)], 2)}
  ${fields([numeric("波长 Å", "wavelength_a", g.wavelength_a), numeric("起始角 °", "azimuth_zero", g.azimuth_zero)], 2)}
  <button type="submit" class="secondary">保存几何</button></form>
  <div class="button-row">${action(`拟合环心 · ${markCount} 点`, "estimate-center")}
    ${action("清空标点", "clear-marks", "quiet")}</div>
  ${model}
  <details><summary>参考环和距离</summary>
    <form id="refs-form" class="fields"><label>参考 d 值 Å（空格或逗号分隔）
    <input name="d_values" value="${escape(references.map((x) => x.toFixed(5)).join(" "))}"></label>
    <button type="submit" class="secondary">更新参考环</button></form>
    <form id="distance-form" class="fields">
    ${numeric("拟合参考环半径 px", "radius_px", fit?.status === "ok" ? fit.radius_px : undefined, {required:true})}
    ${numeric("已知 d 值 Å", "d_a", references[1] ?? references[0], {required:true})}
    <button type="submit" class="quiet">用参考环估计距离</button></form>
  </details>
  <details><summary>多环距离标定</summary>
    <form id="multi-distance" class="fields">
      <label>每行填写环半径 px、d 值 Å 和可选半径误差
        <textarea name="observations" placeholder="43 16.441 0.3" rows="4" required></textarea>
      </label>
      <button type="submit" class="secondary">拟合多环距离</button>
    </form><div id="distance-result" class="control-result"></div>
  </details></section>`;
}

function maskPanel(layers, imported) {
  return `<section class="side-group"><h2>探测器遮罩</h2>
  <div id="mask-list">${layers.map((layer, index) => `<div class="mask-row"><span>${escape(layer.title)}</span>
  <div class="button-row"><button type="button" class="quiet" data-toggle-mask="${index}">
  ${layer.enabled ? "停用" : "启用"}</button><button type="button" class="quiet" data-drop-mask="${index}"
  aria-label="移除${escape(layer.title)}">移除</button></div></div>`).join("") || '<p>未设置手工遮罩。</p>'}</div>
  <details><summary>按计数屏蔽</summary><form id="threshold-mask" class="fields">
  ${fields([numeric("低于或等于", "low"), numeric("高于或等于", "high")], 2)}
  <button type="submit" class="secondary">添加计数遮罩</button></form></details>
  <details><summary>导入二值遮罩</summary>
    <form id="mask-upload" class="fields">
      <label>同尺寸 NPY，1 表示排除<input type="file" accept=".npy" name="mask" required></label>
      <button type="submit" class="secondary">导入遮罩</button></form>
    ${imported ? action("移除导入层", "remove-mask-file", "quiet") : ""}
  </details>
  <div class="button-row">${link("遮罩 NPY", "#", "quiet").replace('href="#"', 'id="download-mask" href="#"')}</div>
  </section>`;
}

export async function detectorTools(ctx) {
  const {state, session, current} = ctx;
  const mode = session.detectorMode ?? "probe";
  const view = session.maskView ? "mask" : "counts";
  ctx.outputs.innerHTML = `${action(session.maskView ? "计数图" : "查看遮罩", "toggle-mask-view", "quiet")}
    ${link("下载标注图", ctx.url("detector.png", {view:"annotated", download:1}), "quiet")}`;
  on("toggle-mask-view", "click", () => {session.maskView = !session.maskView; ctx.render();});
  ctx.controls.innerHTML = `<section class="side-group"><h2>探测器读数</h2>
  ${select("点击图像", "click_mode", [["probe","读取像素"],["center","设为环心"],["mark","沿环放置标点"],["rectangle","两点圈定坏区"]], mode)}
  <div id="detector-summary" class="control-result"></div></section>
  ${displayPanel(state.display)}
  ${geometryPanel(state.geometry, state.marks.length, state.reference_d, session.fit)}
  ${maskPanel(state.masks, current.imported_mask)}`;
  ctx.controls.querySelector("[name=click_mode]").addEventListener("change", (event) => {
    session.detectorMode = event.target.value;
    delete session.maskAnchor;
    ctx.render();
  });
  el("download-mask").href = ctx.url("excluded.npy");
  submit("display-form", (v) => ctx.patch({display: {
    range_mode:v.text("range_mode"), lower:v.number("lower"), upper:v.number("upper"),
    stretch:v.text("stretch"), palette:v.text("palette"), show_mask:v.has("show_mask"),
  }}));
  submit("geometry-form", (v) => ctx.patch({geometry: Object.fromEntries(
    ["cx","cy","pixel_mm","distance_mm","wavelength_a","azimuth_zero"].map((key) => [key, v.number(key)]),
  )}));
  on("clear-marks", "click", () => {delete session.fit; ctx.patch({marks:[]});});
  on("estimate-center", "click", async () => {
    delete session.fit;
    const value = await ctx.own(() => ctx.analysis("calibration/circle", {}));
    if (value && ctx.live()) {session.fit = value.fit; ctx.render();}
  });
  on("apply-fit", "click", async () => {
    const fit = session.fit;
    if (fit?.status !== "ok") return;
    if (await ctx.patch({geometry: {...state.geometry, cx:fit.cx, cy:fit.cy}})) {
      delete session.fit;
      ctx.render();
      ctx.notify("拟合环心已应用");
    }
  });
  submit("refs-form", (v) => ctx.patch({reference_d: v.text("d_values").split(/[ ,，]+/)
    .filter(Boolean).map((x) => Number(x))}));
  submit("distance-form", async (v) => {
    const input = {radius_px:v.number("radius_px"), d_a:v.number("d_a")};
    const result = await ctx.own(() => ctx.analysis("calibration/distance", input));
    if (result && ctx.live()) {
      await ctx.patch({geometry:{...state.geometry, distance_mm:result.model.distance_mm}});
      ctx.notify(`探测器距离 ${fmt(result.model.distance_mm, 3)} mm`);
    }
  });

  submit("multi-distance", async (v) => {
    const observations = v.text("observations").split(/\r?\n/).filter((line)=>line.trim())
      .map((line) => {
        const [radius_px,d_a,sigma_px=1] = line.trim().split(/[ ,，]+/).map(Number);
        return {radius_px,d_a,sigma_px};
      });
    const settings = {observations};
    const value = await ctx.own(() => ctx.analysis("calibration/distance-fit",{settings}));
    if (!value || !ctx.live()) return;
    const m = value.model;
    el("distance-result").innerHTML = `<p>距离 ${fmt(m.distance_mm,4)} mm · RMS ${fmt(m.rms_residual_px,3)} px</p>
      ${table(["参考环","残差 px"], m.used_rings.map((r)=>[r.reference,fmt(r.residual_px,3)]))}
      <div class="button-row">${action("应用多环距离","apply-distance")}
      ${action("下载残差","download-distance","quiet")}</div>`;
    on("apply-distance","click",()=>ctx.patch({geometry:{...state.geometry,distance_mm:m.distance_mm}}));
    on("download-distance","click",async()=>{
      const data = await ctx.own(() => ctx.analysis("calibration/distance-fit",{settings,output:"csv"}));
      if (data) saveBlob(data,"ring-distance-calibration.csv");
    });
  });
  const changeMask = (index, drop) => {
    const next = state.masks.flatMap((layer, i) => i !== index ? [layer] : drop ? [] : [{...layer, enabled: !layer.enabled}]);
    ctx.patch({masks:next});
  };
  ctx.controls.querySelectorAll("[data-toggle-mask]").forEach((node) =>
    node.addEventListener("click", () => changeMask(Number(node.dataset.toggleMask), false)));
  ctx.controls.querySelectorAll("[data-drop-mask]").forEach((node) =>
    node.addEventListener("click", () => changeMask(Number(node.dataset.dropMask), true)));
  submit("threshold-mask", (v) => ctx.patch({masks:[...state.masks, {
    kind:"intensity", title:`计数阈值 ${state.masks.length + 1}`, enabled:true, low:v.number("low"), high:v.number("high"),
  }]}));
  submit("mask-upload", async (_v, form) => {
    const data = new FormData(form);
    data.set("revision", current.revision);
    const result = await ctx.own(() => api(ctx.url("mask"), {method:"PUT", body:data}), "已导入遮罩");
    if (result && ctx.live()) ctx.accept(result.project);
  });
  on("remove-mask-file", "click", () => ctx.transact("mask", "DELETE", {}, "导入遮罩已移除"));

  const data = await ctx.get("detector");
  if (!ctx.live()) return;
  el("detector-summary").innerHTML = stats([
    ["有效像素",fmt(data.mask.valid)], ["遮罩像素",fmt(data.mask.excluded)],
    ["平均计数",fmt(data.stats.mean)], ["超窗像素",fmt(data.range_clipped.low+data.range_clipped.high)],
  ]);
  const handle = async ([x,y]) => {
    if (mode === "center") {
      delete session.fit;
      await ctx.patch({geometry:{...state.geometry, cx:x, cy:y}});
      return;
    }
    if (mode === "mark") {
      delete session.fit;
      await ctx.patch({marks:[...state.marks, [x,y]]});
      return;
    }
    if (mode === "rectangle") {
      if (!session.maskAnchor) {
        session.maskAnchor = [x,y];
        ctx.render();
      } else {
        const a = session.maskAnchor;
        delete session.maskAnchor;
        await ctx.patch({masks:[...state.masks, {
          kind:"rectangle", title:`矩形坏区 ${state.masks.length+1}`, enabled:true, a, b:[x,y],
        }]});
      }
      return;
    }
    const answer = await ctx.get("probe", {x,y});
    if (ctx.live()) {
      const p = answer.pixel;
      ctx.setReadout(`<strong>X ${p.x} · Y ${p.y} · ${fmt(p.counts, 3)} 计数</strong>
        · ${escape(p.reason ?? "有效")} · 半径 ${fmt(p.radius_px)} px · 方位 ${fmt(p.azimuth_deg)}°
        · q ${fmt(p.q_inverse_a, 4)} Å⁻¹ · 邻域有效 ${p.nearby_valid} 点`);
    }
  };
  await ctx.stage({
    source:ctx.url("detector.png", {view}), reference:data.references, anchor:session.maskAnchor,
    legend:[view === "mask" ? "探测器遮罩" : "原始探测器计数",
      `${fmt(data.range.lower)} 至 ${fmt(data.range.upper)}`, state.display.palette],
    hint:mode === "mark" ? "沿同一条参考环点击；至少取三个分散的点" :
      mode === "rectangle" ? "依次点击坏区两个对角点" :
      mode === "center" ? "点击可用环心位置" : "点击像素查看计数、方位和遮罩原因",
    onPoint:handle,
  });
}
