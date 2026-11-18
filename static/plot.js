import {escape, fmt} from "./dom.js";

const size = {w: 860, h: 366, l: 70, r: 22, b: 62, t: 24};
const ranges = [[0, 1], [0, 1]];

function bounds(values, defaultValue) {
  const safe = values.filter(Number.isFinite);
  if (!safe.length) return defaultValue;
  const min = Math.min(...safe);
  const max = Math.max(...safe);
  return max === min ? [min, min + Math.max(1, Math.abs(min) * 1e-6)] : [min, max];
}

function sequence(xs, ys, xAt, yAt) {
  let d = "";
  let moved = false;
  ys.forEach((value, index) => {
    if (!Number.isFinite(value)) {
      moved = false;
      return;
    }
    d += `${moved ? "L" : "M"}${xAt(xs[index]).toFixed(2)} ${yAt(value).toFixed(2)}`;
    moved = true;
  });
  return d;
}

export function renderChart(host, curve, options) {
  const {axis, values, peaks = []} = curve;
  const xs = axis.centers;
  const ys = values[options.metric ?? "mean"];
  const {w, h, l, r, b, t} = size;
  const plotW = w - l - r;
  const plotH = h - t - b;
  const [minX, maxX] = bounds(axis.edges ?? xs, ranges[0]);
  const additions = (options.extra ?? []).filter((line) => line.values.length === xs.length);
  const span = [...ys, ...additions.flatMap((line) => line.values)];
  const [minY, maxY] = bounds(span, ranges[1]);
  const xAt = (x) => l + plotW * (x - minX) / (maxX - minX);
  const yAt = (y) => h - b - plotH * (y - minY) / (maxY - minY);
  const grid = Array.from({length: 6}, (_, i) => {
    const x = l + i * plotW / 5;
    const y = t + i * plotH / 5;
    return `<path d="M${l} ${y} H${w-r} M${x} ${t} V${h-b}" stroke="#2a374a" fill="none"/>
      <text x="${l-9}" y="${y+4}" text-anchor="end">${fmt(maxY-(maxY-minY)*i/5, 3)}</text>
      <text x="${x}" y="${h-b+20}" text-anchor="middle">${fmt(minX+(maxX-minX)*i/5, 3)}</text>`;
  }).join("");
  const pen = options.color ?? "#7adfba";
  const primary = sequence(xs, ys, xAt, yAt);
  const overlays = additions.map((line) => `<path d="${sequence(xs, line.values, xAt, yAt)}"
    fill="none" stroke="${line.color}" stroke-width="2" />`).join("");
  const marker = (options.metric ?? "mean") === "mean" ? peaks.map((peak) => {
    const x = xAt(peak.coordinate);
    const y = yAt(peak.mean);
    return `<circle data-peak="${peak.cell}" tabindex="0" role="button" r="4.5" cx="${x}" cy="${y}"
      fill="#ffbd74" aria-label="径向峰 ${escape(fmt(peak.coordinate, 3))}"></circle>`;
  }).join("") : "";
  host.innerHTML = `<svg viewBox="0 0 ${w} ${h}" class="chart" role="img" aria-label="${escape(options.title)}">
    <rect x="${l}" y="${t}" width="${plotW}" height="${plotH}" rx="2" fill="#101927" />
    ${grid}${overlays}<path d="${primary}" stroke="${pen}" stroke-width="2.4" fill="none"/>
    ${marker}
    <line data-pointer x1="${l}" x2="${l}" y1="${t}" y2="${h-b}" stroke="#eac489" opacity="0" />
    <text x="${w-r}" y="${h-13}" text-anchor="end">${escape(axis.label)}</text>
    </svg>`;
  const svg = host.firstElementChild;
  const pointer = svg.querySelector("[data-pointer]");
  const readAt = (clientX, clientY) => {
    const matrix = svg.getScreenCTM();
    if (!matrix) return;
    const displayed = new DOMPoint(clientX, clientY).matrixTransform(matrix.inverse());
    if (displayed.x < l || displayed.x > w-r || displayed.y < t || displayed.y > h-b) {
      pointer.setAttribute("opacity", "0");
      return;
    }
    const current = minX + (displayed.x - l) / plotW * (maxX - minX);
    let index = 0;
    xs.forEach((x, i) => { if (Math.abs(x-current) < Math.abs(xs[index]-current)) index = i; });
    const x = xAt(xs[index]);
    pointer.setAttribute("x1", x);
    pointer.setAttribute("x2", x);
    pointer.setAttribute("opacity", ".65");
    options.onRead?.(index);
  };
  svg.addEventListener("pointermove", (event) => readAt(event.clientX, event.clientY));
  svg.addEventListener("pointerleave", () => pointer.setAttribute("opacity", "0"));
  svg.querySelectorAll("[data-peak]").forEach((dot) => {
    dot.addEventListener("click", () => options.onPeak?.(Number(dot.dataset.peak)));
    dot.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        options.onPeak?.(Number(dot.dataset.peak));
      }
    });
  });
}

export function renderHistogram(host, histogram) {
  const c = histogram.count;
  const peak = Math.max(1, ...c);
  host.innerHTML = `<svg viewBox="0 0 860 180" class="chart" role="img" aria-label="有效像素计数直方图">
    <path d="M55 12 V148 H836" stroke="#455570" fill="none"/>
    ${c.map((count, i) => {
      const w = 780 / c.length;
      const h = 128 * count / peak;
      const x = 55 + w * i;
      return `<rect x="${x}" y="${148-h}" width="${Math.max(.5, w-1)}" height="${h}" fill="#6cbfba">
        <title>${fmt(histogram.edges[i], 2)}–${fmt(histogram.edges[i+1], 2)} · ${fmt(count)} 像素</title>
      </rect>`;
    }).join("")}
    <text x="55" y="168">${fmt(histogram.edges[0])}</text>
    <text x="836" y="168" text-anchor="end">${fmt(histogram.edges.at(-1))}</text>
    </svg>`;
}
