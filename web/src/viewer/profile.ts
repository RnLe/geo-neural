// SVG cross-section of reference and decoded heights along a line, in metres.

import { h, s } from "../data/dom";
import { formatSigned } from "../data/format";
import { bilinear, type GridFrame } from "./field";

export interface ProfilePoint {
  col: number;
  row: number;
}

export interface ProfileData {
  distance: Float64Array;
  reference: Float64Array;
  decoded: Float64Array;
  lengthM: number;
}

/** Samples both fields every `stepM` metres (bilinear between 20 m nodes). */
export function sampleProfile(
  f: GridFrame,
  reference: Float32Array,
  decoded: Float32Array,
  a: ProfilePoint,
  b: ProfilePoint,
  stepM = 10,
): ProfileData {
  const lengthM = Math.hypot(b.col - a.col, b.row - a.row) * f.spacingM;
  const n = Math.max(2, Math.ceil(lengthM / stepM) + 1);
  const distance = new Float64Array(n);
  const ref = new Float64Array(n);
  const dec = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const col = a.col + (b.col - a.col) * t;
    const row = a.row + (b.row - a.row) * t;
    distance[i] = lengthM * t;
    ref[i] = bilinear(reference, f.side, col, row);
    dec[i] = bilinear(decoded, f.side, col, row);
  }
  return { distance, reference: ref, decoded: dec, lengthM };
}

function niceStep(range: number, target: number): number {
  const raw = range / Math.max(1, target);
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

/** Renders the profile into `host` (replacing its content) at the host's width. */
export function renderProfile(host: HTMLElement, data: ProfileData, candidateLabel: string): void {
  host.replaceChildren();
  const width = Math.max(280, host.clientWidth || 600);
  const height = 220;
  const m = { left: 52, right: 14, top: 14, bottom: 38 };
  const pw = width - m.left - m.right;
  const ph = height - m.top - m.bottom;

  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < data.distance.length; i++) {
    lo = Math.min(lo, data.reference[i], data.decoded[i]);
    hi = Math.max(hi, data.reference[i], data.decoded[i]);
  }
  const pad = Math.max(1, (hi - lo) * 0.08);
  lo -= pad;
  hi += pad;
  const yStep = niceStep(hi - lo, 4);
  lo = Math.floor(lo / yStep) * yStep;
  hi = Math.ceil(hi / yStep) * yStep;
  const x = (d: number) => m.left + (d / data.lengthM) * pw;
  const y = (v: number) => m.top + (1 - (v - lo) / (hi - lo)) * ph;

  const path = (values: Float64Array) => {
    let out = "";
    for (let i = 0; i < values.length; i++) out += `${i ? "L" : "M"}${x(data.distance[i]).toFixed(1)},${y(values[i]).toFixed(1)}`;
    return out;
  };

  let worst = 0;
  let worstAt = 0;
  for (let i = 0; i < data.distance.length; i++) {
    const e = data.decoded[i] - data.reference[i];
    if (Math.abs(e) > Math.abs(worst)) {
      worst = e;
      worstAt = data.distance[i];
    }
  }

  const km = data.lengthM >= 2000;
  const xStep = niceStep(data.lengthM, Math.max(2, Math.floor(pw / 90)));
  const grid = s("g", { class: "gn-axis" });
  for (let v = lo; v <= hi + 1e-9; v += yStep) {
    grid.append(
      s("line", { x1: m.left, x2: m.left + pw, y1: y(v), y2: y(v), class: "gn-gridline" }),
      s("text", { x: m.left - 6, y: y(v) + 4, "text-anchor": "end" }, `${Math.round(v)}`),
    );
  }
  for (let v = 0; v <= data.lengthM + 1e-9; v += xStep) {
    grid.append(
      s("line", { x1: x(v), x2: x(v), y1: m.top + ph, y2: m.top + ph + 4, class: "gn-tick" }),
      s("text", { x: x(v), y: m.top + ph + 16, "text-anchor": "middle" }, km ? `${(v / 1000).toFixed(xStep < 1000 ? 1 : 0)}` : `${Math.round(v)}`),
    );
  }
  grid.append(
    s("text", { x: m.left + pw / 2, y: height - 4, "text-anchor": "middle" }, km ? "distance from A (km)" : "distance from A (m)"),
    s("text", { x: 12, y: m.top + ph / 2, transform: `rotate(-90 12 ${m.top + ph / 2})`, "text-anchor": "middle" }, "height (m)"),
  );

  // Plot aspect: how much the plot stretches heights relative to distance.
  const stretch = (ph / (hi - lo)) / (pw / data.lengthM);
  const titleId = `gn-profile-${Math.random().toString(36).slice(2)}`;
  const svg = s(
    "svg",
    { class: "gn-profile-svg", width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-labelledby": titleId },
    s("title", { id: titleId }, `Cross-section from A to B, ${(data.lengthM / 1000).toFixed(2)} km: reference and decoded heights`),
    grid,
    s("path", { d: path(data.reference), class: "gn-profile-ref" }),
    s("path", { d: path(data.decoded), class: "gn-profile-dec" }),
    s("text", { x: m.left + 4, y: m.top + 10, class: "gn-profile-end" }, "A"),
    s("text", { x: m.left + pw - 4, y: m.top + 10, class: "gn-profile-end", "text-anchor": "end" }, "B"),
  );
  host.append(
    svg,
    h(
      "ul",
      { class: "gn-legend-list gn-legend-inline" },
      h("li", {}, h("span", { class: "gn-swatch-line gn-swatch-line-ref", "aria-hidden": "true" }), "reference (solid)"),
      h("li", {}, h("span", { class: "gn-swatch-line gn-swatch-line-dec", "aria-hidden": "true" }), `decoded: ${candidateLabel} (dashed)`),
    ),
    h(
      "p",
      { class: "gn-note" },
      `Length ${(data.lengthM / 1000).toFixed(2)} km, sampled every 10 m between 20 m nodes. Heights in metres; the 3D exaggeration does not apply. ` +
        `The plot itself stretches heights about ${stretch.toFixed(0)}x relative to distance. ` +
        `Largest difference along the line: ${formatSigned(worst)} m at ${(worstAt / 1000).toFixed(2)} km.`,
    ),
  );
}
