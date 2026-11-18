import {action, el, fields, fmt, imageSrc, link, numeric, on, saveBlob, select, stats, submit, table} from "./dom.js";
import {renderChart, renderHistogram} from "./plot.js";

function inspectionParams(ctx) {
  return ctx.session.inspection ?? {bins:80, region:null};
}

export async function inspectionTools(ctx) {
  const parameters = inspectionParams(ctx);
  const region = parameters.region ?? {start:[0,0],stop:[ctx.current.width,ctx.current.height]};
  ctx.controls.innerHTML = `<form id="roi-form" class="side-group"><h2>检查矩形区域</h2>
    ${fields([numeric("左列","ax",region.start[0],{min:0,step:1}),
      numeric("顶行","ay",region.start[1],{min:0,step:1}),
      numeric("右列（不含）","bx",region.stop[0],{min:1,step:1}),
      numeric("底行（不含）","by",region.stop[1],{min:1,step:1})],2)}
    ${numeric("直方图区间数","bins",parameters.bins,{min:12,max:512,step:1})}
    <button type="submit">更新计数检查</button>${action("全幅检查","roi-reset","quiet")}</form>
    <section class="side-group"><h2>坏条带检查</h2>
    ${select("检查方向","row_or_column",[["columns","沿列统计"],["rows","沿行统计"]],ctx.session.stripes ?? "columns")}
    <p>剔除了导入遮罩、缺失像素和启用的遮罩层。图中空段没有可用计数。</p></section>`;
  submit("roi-form",(v) => {
    ctx.session.inspection = {bins:v.number("bins"), region:{
      start:[v.number("ax"),v.number("ay")], stop:[v.number("bx"),v.number("by")],
    }};
    ctx.render();
  });
  on("roi-reset","click", () => {delete ctx.session.inspection; ctx.render();});
  ctx.controls.querySelector("[name=row_or_column]").addEventListener("change",(ev) => {
    ctx.session.stripes = ev.target.value;
    ctx.render();
  });
  const result = await ctx.analysis("inspection", parameters);
  if (!ctx.live()) return;
  ctx.outputs.innerHTML = `${action("下载直方图 CSV","export-hist")}
    ${action("下载条带 CSV","export-stripe","quiet")}`;
  for (const [name,output] of [["export-hist","histogram"],["export-stripe",ctx.session.stripes ?? "columns"]]) {
    on(name,"click",async () => {
      const file = await ctx.own(() => ctx.analysis("inspection", {...parameters,output}));
      if (file) saveBlob(file,`${output}.csv`);
    });
  }
  const s = result.statistics;
  ctx.body.innerHTML = `<div class="plot-wrap">${stats([
    ["区域有效",`${fmt(s.count)} / ${fmt(s.candidate)}`],["最小计数",fmt(s.min)],["最大计数",fmt(s.max)],
    ["平均计数",fmt(s.mean)],["有效比例",`${fmt(s.valid_fraction*100,1)}%`],
  ])}<h3>有效计数直方图</h3><div id="histogram"></div>
  <h3>按行列比较计数</h3><div id="stripe-chart"></div>
  <h3>原探测器检查区域</h3><div id="inspect-image"></div></div>`;
  renderHistogram(el("histogram"),result.histogram);
  const curve = result[ctx.session.stripes ?? "columns"];
  renderChart(el("stripe-chart"),curve,{title:"条带检查曲线",onRead:(i) => {
    ctx.setReadout(`<strong>${fmt(curve.axis.centers[i])} px · ${fmt(curve.values.mean[i])} 计数</strong>
      · ${fmt(curve.values.count[i])} / ${fmt(curve.values.eligible[i])} 像素有效`);
  }});
  await ctx.stage({
    host:el("inspect-image"), guides:false, roi:result.region, anchor:ctx.session.roiAnchor,
    hint:"点击检查区域的两个对角点",
    onPoint:([x,y]) => {
      const a = ctx.session.roiAnchor;
      if (a) {
        delete ctx.session.roiAnchor;
        ctx.session.inspection = {...parameters, region:{
          start:[Math.min(a[0],x),Math.min(a[1],y)], stop:[Math.max(a[0],x)+1,Math.max(a[1],y)+1],
        }};
      } else ctx.session.roiAnchor = [x,y];
      ctx.render();
    },
  });
}

export async function repeatTools(ctx) {
  const options = ctx.session.repeat ?? {
    frames:ctx.current.frames.slice(0,2).map((item) => item.id), minimum:2,
  };
  ctx.controls.innerHTML = `<form id="repeat-form" class="side-group"><h2>重复曝光稳定性</h2>
    <div class="fields">${ctx.current.frames.map((f) => {
      const selected = options.frames.includes(f.id);
      return `<label class="checkbox"><input type="checkbox" name="selected_frame" value="${f.id}" ${selected ? "checked" : ""}>
        ${escapeText(f.title)}</label>`;
    }).join("")}</div>
    ${numeric("最少有效曝光次数","minimum",options.minimum,{min:2,max:8,step:1})}
    <button type="submit">计算重复曝光</button></form>
    <section class="side-group">${select("检查图层","repeat_layer",[["std","样本标准差"],["mean","重复计数均值"]],ctx.session.repeatLayer ?? "std")}
    ${numeric("离散点阈值","deviation_threshold",ctx.session.deviationThreshold ?? 15,{min:0})}
    ${action("下载离散点 CSV","deviation-outliers","quiet")}
    <p>请选择同一实验条件的重复曝光。同一像素的缺失观测从统计中排除。</p></section>`;
  submit("repeat-form",(v,form) => {
    ctx.session.repeat = {frames:new FormData(form).getAll("selected_frame"), minimum:v.number("minimum")};
    ctx.render();
  });
  ctx.controls.querySelector("[name=repeat_layer]").addEventListener("change",(event) => {
    ctx.session.repeatLayer = event.target.value;
    ctx.render();
  });
  if (options.frames.length < 2) {
    ctx.body.innerHTML = '<p class="pending">选择同条件下的至少两张曝光查看稳定性。</p>';
    return;
  }
  const summary = await ctx.analysis("stability",options);
  if (!ctx.live()) return;
  const layer = ctx.session.repeatLayer ?? "std";
  ctx.outputs.innerHTML = action(layer === "std" ? "下载离散 NPY" : "下载均值 NPY","repeat-export");
  on("repeat-export","click", async () => {
    const value = await ctx.own(() => ctx.analysis("stability",{...options,output:`${layer}.npy`}));
    if (value) saveBlob(value,`repeat-${layer}.npy`);
  });
  on("deviation-outliers","click", async () => {
    const threshold = Number(ctx.controls.querySelector("[name=deviation_threshold]").value);
    ctx.session.deviationThreshold = threshold;
    const value = await ctx.own(() => ctx.analysis("stability",{...options,output:"outliers.csv",threshold}));
    if (value) saveBlob(value,"deviation-pixels.csv");
  });
  ctx.body.innerHTML = `<div class="plot-wrap">${stats([
    ["重复曝光",summary.selected_frames.length],["支撑充分像素",fmt(summary.sufficient_pixels)],
    ["全图像素",fmt(summary.detector_pixels)],["平均离散",fmt(summary.sample_deviation.mean)],
  ])}<div id="repeat-image"></div>${table(["有效曝光数","探测器像素"],
      summary.observation_histogram.map((count,i) => [i,fmt(count)]))}</div>`;
  const preview = await ctx.analysis("stability",{...options,output:`${layer}.png`});
  if (ctx.live()) await ctx.stage({
    host:el("repeat-image"), source:preview, overlays:false,
    legend:[layer === "std" ? "样本标准差图" : "重复曝光计数均值", "有效支撑域", "ice"],
    hint:"观测不足的探测器像素保持空白。",
  });
}

function escapeText(value) {
  const item = document.createElement("span");
  item.textContent = value;
  return item.innerHTML;
}

export async function responseTools(ctx) {
  const active = ctx.chosen.id;
  const options = [["","不使用"],...ctx.current.frames.filter((f) => f.id !== active).map((f) => [f.id,f.title])];
  const spec = ctx.session.response ?? {dark:null,flat:null,minimum_signal:1,maximum_gain:6,keep_negative:true};
  ctx.controls.innerHTML = `<form id="response-form" class="side-group"><h2>暗场／平场校正</h2>
    ${select("暗场参考","dark",options,spec.dark ?? "")}
    ${select("均匀平场参考","flat",options,spec.flat ?? "")}
    ${numeric("平场净计数下限","minimum_signal",spec.minimum_signal,{min:.00001})}
    ${numeric("最大增益","maximum_gain",spec.maximum_gain,{min:1,max:1000})}
    <label class="checkbox"><input type="checkbox" name="keep_negative" ${spec.keep_negative ? "checked" : ""}>
    保留负校正值</label><button type="submit">应用参考帧</button></form>`;
  submit("response-form",(v) => {
    ctx.session.response = {
      dark:v.text("dark") || null, flat:v.text("flat") || null, minimum_signal:v.number("minimum_signal"),
      maximum_gain:v.number("maximum_gain"),keep_negative:v.has("keep_negative"),
    };
    ctx.render();
  });
  if (!spec.dark && !spec.flat) {
    ctx.body.innerHTML = '<p class="pending">为当前曝光选择独立采集的同尺寸暗场或平场后，校正结果会显示在这里。</p>';
    return;
  }
  const data = await ctx.analysis("correction",{settings:spec});
  if (!ctx.live()) return;
  ctx.outputs.innerHTML = `${action("下载校正 NPY","response-counts")}
    ${spec.flat ? action("下载增益 NPY","response-gain","quiet") : ""}
    ${action("下载排除报告","response-flags","quiet")}`;
  for (const output of ["counts","gain","flags"]) {
    on(`response-${output}`,"click",async () => {
      const result = await ctx.own(() => ctx.analysis("correction",{settings:spec,output}));
      if (result) saveBlob(result,`response-${output}.${output === "flags" ? "csv" : "npy"}`);
    });
  }
  ctx.body.innerHTML = `<div class="plot-wrap">${stats([
    ["可用像素",fmt(data.usable)],["候选像素",fmt(data.candidate)],["平场标度",fmt(data.normalization_reference)],
    ["平均校正计数",fmt(data.counts.mean)],["裁零像素",fmt(data.negative_values_clamped)],
  ])}<div id="response-image"></div>
  ${table(["排除原因","像素数量"],data.flags.map((f) => [f.reason,fmt(f.pixels)]))}</div>`;
  const preview = await ctx.analysis("correction",{settings:spec,output:"preview"});
  if (ctx.live()) await ctx.stage({
    host:el("response-image"), source:preview, overlays:false,
    legend:["相对响应校正后的计数", "有效支撑域", "ice"],
    hint:"校正图在原探测器坐标显示，原始曝光仍可单独读取。",
  });
}
