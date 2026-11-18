const DOM = document;

export const $ = (id, root = DOM) => root.querySelector(id);
export const el = (id) => DOM.getElementById(id);

export function escape(text) {
  const tags = {"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"};
  return String(text ?? "").replace(/[&<>"']/g, (char) => tags[char]);
}

export function fmt(value, precision = 2) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  if (Math.abs(value) >= 1e6 || (Math.abs(value) > 0 && Math.abs(value) < 1e-4)) {
    return Number(value).toExponential(2);
  }
  return Number(value).toLocaleString("zh-CN", {maximumFractionDigits: precision});
}

export function action(title, id, className = "secondary") {
  return `<button type="button" class="${className}" id="${id}">${escape(title)}</button>`;
}

export function link(title, url, className = "") {
  return `<a class="button ${className}" href="${escape(url)}">${escape(title)}</a>`;
}

export function numeric(title, name, value, options = {}) {
  const min = options.min === undefined ? "" : `min="${options.min}"`;
  const max = options.max === undefined ? "" : `max="${options.max}"`;
  return `<label>${escape(title)}<input type="number" name="${name}" value="${escape(value)}"
      step="${options.step ?? "any"}" ${min} ${max} ${options.required ? "required" : ""}></label>`;
}

export function select(title, name, options, current) {
  return `<label>${escape(title)}<select name="${name}">${options.map(([id, text]) =>
    `<option value="${escape(id)}" ${String(id) === String(current) ? "selected" : ""}>${escape(text)}</option>`
  ).join("")}</select></label>`;
}

export function checkbox(title, name, value) {
  return `<label class="checkbox"><input type="checkbox" name="${name}" ${value ? "checked" : ""}>${escape(title)}</label>`;
}

export function fields(entries, columns = 1) {
  return `<div class="fields ${columns === 2 ? "two" : columns === 3 ? "three" : ""}">${entries.join("")}</div>`;
}

export function stats(entries) {
  return `<div class="stats-grid">${entries.map(([name, value]) =>
    `<div class="stat"><strong>${escape(value)}</strong>${escape(name)}</div>`).join("")}</div>`;
}

export function table(headings, rows) {
  if (!rows.length) return '<p class="note">当前没有读数。</p>';
  return `<div class="table-scroll"><table><thead><tr>${headings.map((h) => `<th scope="col">${escape(h)}</th>`).join("")}
    </tr></thead><tbody>${rows.map((row) => `<tr>${row.map((item) => `<td>${escape(item)}</td>`).join("")}</tr>`).join("")}
    </tbody></table></div>`;
}

export function values(form) {
  const source = new FormData(form);
  return {
    text: (name) => String(source.get(name) ?? ""),
    number: (name) => source.get(name) === "" || !source.has(name) ? null : Number(source.get(name)),
    has: (name) => source.has(name),
  };
}

export function on(id, kind, handler) {
  el(id)?.addEventListener(kind, handler);
}

export function submit(id, handler) {
  on(id, "submit", (event) => {
    event.preventDefault();
    handler(values(event.currentTarget), event.currentTarget);
  });
}

export async function api(path, options = {}) {
  const {json, ...more} = options;
  if (json !== undefined) {
    more.headers = {...more.headers, "Content-Type": "application/json"};
    more.body = JSON.stringify(json);
  }
  const response = await fetch(path, more);
  const contentType = response.headers.get("Content-Type") ?? "";
  if (!response.ok) {
    const data = contentType.includes("json") ? await response.json() : null;
    const error = new Error(data?.error?.message ?? `请求失败 (${response.status})`);
    error.code = data?.error?.code;
    throw error;
  }
  return contentType.includes("json") ? await response.json() : response.blob();
}

export function saveBlob(blob, filename) {
  const uri = URL.createObjectURL(blob);
  const anchor = DOM.createElement("a");
  anchor.href = uri;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(uri), 1000);
}

export function imageSrc(image, source) {
  if (source instanceof Blob) {
    const url = URL.createObjectURL(source);
    image.onload = () => URL.revokeObjectURL(url);
    image.onerror = image.onload;
    image.src = url;
  } else {
    image.src = source;
  }
}
