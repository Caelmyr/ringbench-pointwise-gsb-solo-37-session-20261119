import {$, action, api, checkbox, el, escape, fields, fmt, imageSrc, link, numeric, on,
  saveBlob, select, stats, submit, table} from "./dom.js";
import {renderChart, renderHistogram} from "./plot.js";
import {detectorTools, drawStage} from "./detector.js";
import {curveTools, cakeTools, differenceTools, spotTools} from "./views.js";
import {inspectionTools, repeatTools, responseTools} from "./tools.js";
import {areaTools, comparisonTools, contourTools, coverageTools, geometricTools, harmonicTools,
  traceTools, vectorTools} from "./advanced.js";

const panels = [
  ["detector", "探测器", detectorTools], ["profile", "径向积分", curveTools],
  ["azimuth", "角向曲线", curveTools], ["cake", "极坐标图", cakeTools],
  ["line", "横截面", curveTools], ["spots", "斑点测量", spotTools],
  ["difference", "曝光差图", differenceTools],
];
const additional = [
  ["inspection", "计数检查", inspectionTools], ["stability", "重复曝光", repeatTools],
  ["correction", "响应校正", responseTools], ["comparison", "径向谱比较", comparisonTools],
  ["harmonics", "取向谐波", harmonicTools], ["trace", "环线跟踪", traceTools],
  ["area", "峰区面积", areaTools], ["contours", "等强线", contourTools],
  ["vectors", "散射向量", vectorTools], ["coverage", "环带覆盖", coverageTools],
  ["geo_response", "几何响应", geometricTools],
];

let project = null;
let token = 0;
let toastTimer = null;
const session = {};
const welcome = el("welcome");
const workspace = el("workspace");
const side = el("input-sidebar");
const controls = el("controls");
const body = el("view-body");
const outputs = el("view-actions");
const readout = el("readout");
const manager = el("manager");
const KEY = "ringbench.open_project";

function notify(message, failure = false) {
  const notice = el("notice");
  notice.hidden = false;
  notice.classList.toggle("error", failure);
  notice.textContent = message;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {notice.hidden = true;}, 5300);
}

function fail(error) {
  notify(error.message, true);
}

async function own(task, success) {
  try {
    const result = await task();
    if (success) notify(success);
    return result;
  } catch (error) {
    fail(error);
    if (error.code === "revision_changed" && project) await openProject(project.id);
    return undefined;
  }
}

function displayProject(next, resetSession) {
  project = next;
  if (resetSession) {
    for (const name of Object.keys(session)) delete session[name];
  }
  if (project) localStorage.setItem(KEY, project.id);
  render();
}

async function patch(changes, title) {
  if (!project) return;
  const pid = project.id;
  const result = await own(() => api(`/api/projects/${pid}`, {
    method: "PATCH", json: {revision: project.revision, settings: changes, title},
  }));
  if (result && project?.id === pid) displayProject(result.project, false);
  return result;
}

function layoutSide() {
  side.innerHTML = `<h2 class="side-heading">探测器帧</h2>
  <div class="frame-list">${project.frames.map((item, index) => `<button type="button" class="frame"
    data-frame="${item.id}" aria-pressed="${item.id === project.state.active_frame}">
    <span>${escape(item.title)}</span><small>${index + 1} · ${escape(item.source_format)} · ${escape(item.source_dtype)}</small>
    </button>`).join("")}</div>
  <form id="add-frame" class="side-group">
    <label>添加同尺寸曝光<input type="file" name="image" accept=".png,.tif,.tiff,.npy" required></label>
    <button type="submit" class="secondary">添加曝光</button>
  </form>
  <div class="side-group">${action("当前曝光信息", "manage-frame", "quiet")}
  ${link("原始计数 NPY", projectURL("detector.npy"), "quiet")}
  ${link("当前设置 JSON", projectURL("settings.json"), "quiet")}</div>`;
  side.querySelectorAll("[data-frame]").forEach((item) => item.addEventListener("click", () => {
    if (item.dataset.frame !== project.state.active_frame) patch({active_frame: item.dataset.frame});
  }));
  submit("add-frame", async (_values, form) => {
    const pid = project.id;
    const data = new FormData(form);
    data.set("revision", project.revision);
    const result = await own(() => api(`/api/projects/${pid}/frames`, {method: "POST", body: data}), "曝光已添加");
    if (result && project?.id === pid) displayProject(result.project, false);
  });
  on("manage-frame", "click", () => showManager("frame"));
}

function projectURL(path, params = {}) {
  const query = new URLSearchParams({frame: project.state.active_frame, revision: project.revision, ...params});
  return `/api/projects/${project.id}/${path}?${query}`;
}

function showManager(kind) {
  const frame = project.frames.find((f) => f.id === project.state.active_frame);
  const current = kind === "frame" ? frame.title : project.title;
  const meta = kind === "frame" ? `<p class="note">${escape(frame.source_name)} · ${escape(frame.source_dtype)}</p>` : "";
  manager.innerHTML = `<form id="manager-form"><h2 id="manager-title">${kind === "frame" ? "当前曝光" : "工程设置"}</h2>
  ${meta}<label>${kind === "frame" ? "曝光名称" : "工程名称"}<input name="title" required maxlength="80" value="${escape(current)}"></label>
  <div class="button-row"><button type="submit">保存名称</button>${action("关闭", "close-manager")}
  ${action(kind === "frame" ? "移除曝光" : "删除工程", "remove-entry", "danger")}</div>
  ${kind === "project" ? `<div class="side-group">${action("检查本地工程","integrity-check","quiet")}
    <div id="integrity-result" class="note"></div></div>` : ""}</form>`;
  manager.showModal();
  on("close-manager", "click", () => manager.close());
  on("integrity-check", "click", async () => {
    const result = await own(() => api(projectURL("integrity")));
    if (result && manager.open && el("integrity-result")) {
      el("integrity-result").textContent = result.valid ?
        `已核对 ${result.frame_checks.length} 帧及保存设置 · 版本 ${result.revision}` : "发现无效帧或保存设置。";
    }
  });
  submit("manager-form", async (v) => {
    const pid = project.id;
    if (kind === "frame") {
      const result = await own(() => api(`/api/projects/${pid}/frames/${frame.id}`, {
        method: "PATCH", json: {title: v.text("title"), revision: project.revision},
      }), "已保存");
      if (result && project?.id === pid) displayProject(result.project, false);
      if (result) manager.close();
    } else if (await patch({}, v.text("title"))) manager.close();
  });
  on("remove-entry", "click", async () => {
    if (!window.confirm(kind === "frame" ? "从工程中移除这个曝光？" : "删除这个本地工程和其中所有曝光？")) return;
    const pid = project.id;
    const url = kind === "frame" ? `/api/projects/${pid}/frames/${frame.id}` : `/api/projects/${pid}`;
    const result = await own(() => api(url, {method: "DELETE", json: {revision: project.revision}}), "已移除");
    if (!result) return;
    manager.close();
    if (result.project) displayProject(result.project, false);
    else home();
  });
}

async function showReviewExport() {
  const pid = project.id;
  const {options} = await own(() => api("/api/report-products")) ?? {};
  if (!options || pid !== project?.id) return;
  manager.innerHTML = `<form id="review-form"><h2 id="manager-title">导出当前曝光的分析</h2>
    <p class="note">选中的图和表以同一个工程版本生成。包内含原始帧哈希、坐标和分析设置。</p>
    <div class="fields">${options.map((o) => `<label class="checkbox">
      <input name="product" type="checkbox" value="${escape(o.id)}" ${["detector","profile"].includes(o.id) ? "checked" : ""}>
      ${escape(o.label)}</label>`).join("")}</div>
    <div class="button-row"><button type="submit">下载分析 ZIP</button>${action("关闭","close-review")}</div></form>`;
  manager.showModal();
  on("close-review","click",()=>manager.close());
  submit("review-form", async (_v,form)=>{
    const products = new FormData(form).getAll("product");
    const result = await own(() => api(projectURL("analysis.zip"), {
      method:"POST",json:{products, revision:project.revision, frame:project.state.active_frame},
    }));
    if (result) {
      saveBlob(result, "ringbench-analysis.zip");
      manager.close();
    }
  });
}

function setReadout(content) {
  readout.hidden = !content;
  readout.innerHTML = content || "";
}

function render() {
  const visit = ++token;
  welcome.hidden = !!project;
  workspace.hidden = !project;
  const top = el("project-actions");
  if (!project) {
    top.innerHTML = "";
    return;
  }
  const current = project;
  const state = current.state;
  const chosen = current.frames.find((frame) => frame.id === state.active_frame);
  el("workspace-title").textContent = project.title;
  el("workspace-subtitle").textContent = `${chosen.title} · ${project.width} × ${project.height} px · 计数坐标从 0 开始`;
  top.innerHTML = `${action("切换工程", "home")}${action("工程设置", "edit-workspace", "quiet")}
    ${link("保存工程 ZIP", projectURL("workspace.zip"), "quiet")}
    ${action("导出分析", "review-package", "quiet")}`;
  on("home", "click", home);
  on("edit-workspace", "click", () => showManager("project"));
  on("review-package", "click", showReviewExport);
  layoutSide();
  outputs.innerHTML = "";
  const nav = el("views");
  nav.innerHTML = panels.map(([key, name]) => `<button type="button" class="tab" data-view="${key}"
    aria-selected="${key === state.active_view}">${name}</button>`).join("") +
    `<select id="more-views" aria-label="更多分析">
    <option value="" ${additional.some(([key]) => key === state.active_view) ? "" : "selected"}>更多分析</option>
    ${additional.map(([key,name]) => `<option value="${key}" ${key === state.active_view ? "selected" : ""}>${name}</option>`).join("")}
    </select>`;
  nav.querySelectorAll("[data-view]").forEach((node) => node.addEventListener("click", () => {
    if (node.dataset.view !== state.active_view) patch({active_view: node.dataset.view});
  }));
  on("more-views", "change", (event) => {
    if (event.target.value && event.target.value !== state.active_view) patch({active_view:event.target.value});
  });
  body.innerHTML = '<p class="pending">正在读取当前曝光…</p>';
  controls.innerHTML = "";
  setReadout("");
  const panel = [...panels, ...additional].find(([key]) => key === state.active_view)?.[2] ?? detectorTools;
  const ctx = {
    current, chosen, state, session, body, controls, outputs,
    url: projectURL, render, patch, own, notify, setReadout,
    live: () => visit === token,
    accept: (doc) => {if (project?.id === current.id) displayProject(doc, false);},
    get: (path, query) => api(projectURL(path, query)),
    transact: async (path, method, json, success) => {
      const result = await own(() => api(projectURL(path), {method, json: {...json, revision: project.revision}}), success);
      if (result?.project && project?.id === current.id) displayProject(result.project, false);
      return result;
    },
    analysis: (path, data) => api(projectURL(path), {method: "POST", json: {
      frame: state.active_frame, revision: current.revision, ...data,
    }}),
    stage: (config) => drawStage(ctx, config),
  };
  Promise.resolve(panel(ctx)).catch((error) => {
    if (ctx.live()) body.innerHTML = `<div class="pending"><p class="status-card" role="alert">${escape(error.message)}</p></div>`;
  });
}

async function openProject(id) {
  const result = await own(() => api(`/api/projects/${id}`));
  if (result) displayProject(result.project, result.project.id !== project?.id);
}

function home() {
  localStorage.removeItem(KEY);
  displayProject(null, true);
  refreshProjects().catch(fail);
  window.scrollTo(0, 0);
}

async function refreshProjects() {
  const data = await api("/api/projects");
  const host = el("saved-projects");
  host.innerHTML = data.projects.map((item) =>
    `<button type="button" class="frame" data-project="${item.id}">
      <span>${escape(item.title)}</span><small>${item.width} × ${item.height} · ${item.frame_count} 帧</small>
    </button>`).join("");
  host.querySelectorAll("[data-project]").forEach((item) =>
    item.addEventListener("click", () => openProject(item.dataset.project)));
}

async function initialize() {
  const cfg = await api("/api/config");
  el("demo-actions").innerHTML = cfg.demos.map((item) =>
    `<button type="button" class="${item.key === "powder" ? "" : "secondary"}" data-demo="${item.key}"
      title="${escape(item.description)}">载入${escape(item.title)}</button>`).join("");
  el("demo-actions").querySelectorAll("[data-demo]").forEach((node) => node.addEventListener("click", async () => {
    node.disabled = true;
    const result = await own(() => api("/api/projects/demo", {method: "POST", json: {key: node.dataset.demo}}));
    node.disabled = false;
    if (result) displayProject(result.project, true);
  }));
  for (const [id, route] of [["upload-project", "/api/projects"], ["restore-project", "/api/projects/restore"]]) {
    submit(id, async (_v, form) => {
      const result = await own(() => api(route, {method: "POST", body: new FormData(form)}));
      if (result) {
        form.reset();
        displayProject(result.project, true);
      }
    });
  }
  const last = localStorage.getItem(KEY);
  if (last) await openProject(last);
  await refreshProjects();
}

initialize().catch(fail);
