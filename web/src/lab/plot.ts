// Small hand-built SVG line plots for the lab. Log axes take an explicit
// floor: values at or below it (zero included) sit on a labelled floor line
// as open markers instead of being plotted at an arbitrary height.

import { s } from "../data/dom";
import { formatSci } from "../data/format";

export type Marker = "circle" | "square" | "triangle" | "diamond" | "none";

export interface Series {
  label: string;
  points: [number, number][];
  /** CSS class carrying the stroke colour. */
  className: string;
  dash?: string;
  marker: Marker;
  /** Draw markers only, no connecting line. */
  markersOnly?: boolean;
}

export interface Axis {
  min: number;
  max: number;
  label: string;
  log?: boolean;
  /** Log axes: values at or below this are drawn on the floor line. */
  floor?: number;
  format?: (v: number) => string;
}

export interface PlotSpec {
  title: string;
  width: number;
  height: number;
  x: Axis;
  y: Axis;
  series: Series[];
}

function markerPath(m: Marker, r: number): string {
  switch (m) {
    case "square":
      return `M${-r},${-r}h${2 * r}v${2 * r}h${-2 * r}Z`;
    case "triangle":
      return `M0,${-1.2 * r}L${1.1 * r},${0.8 * r}L${-1.1 * r},${0.8 * r}Z`;
    case "diamond":
      return `M0,${-1.3 * r}L${1.1 * r},0L0,${1.3 * r}L${-1.1 * r},0Z`;
    case "circle":
      return `M${-r},0a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0`;
    default:
      return "";
  }
}

function linearTicks(min: number, max: number, target = 5): number[] {
  const raw = (max - min) / target;
  const p = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const m = raw / p;
  const step = (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  const out: number[] = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

function logTicks(min: number, max: number): number[] {
  const out: number[] = [];
  const lo = Math.ceil(Math.log10(min));
  const hi = Math.floor(Math.log10(max));
  const every = Math.max(1, Math.ceil((hi - lo + 1) / 6));
  for (let e = lo; e <= hi; e += every) out.push(Math.pow(10, e));
  return out;
}

/** Text for an HTML note under a plot with a log floor. */
export function floorNote(floor: number): string {
  return `Dashed line at ${formatSci(floor, 0)}: values at or below it, exact zero included, are drawn on that line as open markers; they are not measured at that height.`;
}

export function linePlot(spec: PlotSpec): SVGSVGElement {
  const { width, height, x, y } = spec;
  // Room for the direct labels: the longest one (about 6 px a character), within limits.
  const longest = Math.max(0, ...spec.series.map((series) => series.label.length));
  const labelSpace = Math.min(160, Math.max(70, width * 0.18, longest * 6 + 12));
  const mg = { left: y.log && y.floor !== undefined ? 70 : 56, right: labelSpace, top: 12, bottom: 38 };
  const pw = Math.max(40, width - mg.left - mg.right);
  const ph = Math.max(40, height - mg.top - mg.bottom);
  const floor = y.log ? (y.floor ?? y.min) : -Infinity;
  const yMin = y.log ? Math.min(y.min, floor) : y.min;
  const tx = (v: number) =>
    x.log
      ? mg.left + ((Math.log10(v) - Math.log10(x.min)) / (Math.log10(x.max) - Math.log10(x.min))) * pw
      : mg.left + ((v - x.min) / (x.max - x.min || 1)) * pw;
  const ty = (v: number) => {
    if (y.log) {
      const c = Math.max(v, floor);
      return mg.top + (1 - (Math.log10(c) - Math.log10(yMin)) / (Math.log10(y.max) - Math.log10(yMin))) * ph;
    }
    return mg.top + (1 - (v - y.min) / (y.max - y.min || 1)) * ph;
  };

  const fx = x.format ?? ((v: number) => String(v));
  const fy = y.format ?? ((v: number) => (y.log ? formatSci(v, 0) : String(v)));
  const axes = s("g", { class: "gn-axis" });
  for (const v of x.log ? logTicks(x.min, x.max) : linearTicks(x.min, x.max, Math.max(2, Math.floor(pw / 80)))) {
    axes.append(
      s("line", { x1: tx(v), x2: tx(v), y1: mg.top + ph, y2: mg.top + ph + 4, class: "gn-tick" }),
      s("text", { x: tx(v), y: mg.top + ph + 16, "text-anchor": "middle" }, fx(v)),
    );
  }
  const yTicks = y.log ? logTicks(Math.max(yMin, floor), y.max).filter((v) => v > floor * 1.0001) : linearTicks(y.min, y.max, 4);
  for (const v of yTicks) {
    axes.append(
      s("line", { x1: mg.left, x2: mg.left + pw, y1: ty(v), y2: ty(v), class: "gn-gridline" }),
      s("text", { x: mg.left - 6, y: ty(v) + 4, "text-anchor": "end" }, fy(v)),
    );
  }
  axes.append(
    s("line", { x1: mg.left, x2: mg.left, y1: mg.top, y2: mg.top + ph, class: "gn-axisline" }),
    s("line", { x1: mg.left, x2: mg.left + pw, y1: mg.top + ph, y2: mg.top + ph, class: "gn-axisline" }),
    s("text", { x: mg.left + pw / 2, y: height - 6, "text-anchor": "middle" }, x.label),
    s("text", { x: 12, y: mg.top + ph / 2, transform: `rotate(-90 12 ${mg.top + ph / 2})`, "text-anchor": "middle" }, y.label),
  );
  if (y.log && Number.isFinite(floor)) {
    axes.append(
      s("line", { x1: mg.left, x2: mg.left + pw, y1: ty(floor), y2: ty(floor), class: "gn-floorline" }),
      s("text", { x: mg.left - 6, y: ty(floor) + 4, "text-anchor": "end" }, `${fy(floor)} floor`),
    );
  }

  const body = s("g", {});
  const labels: { y: number; text: string; cls: string }[] = [];
  for (const series of spec.series) {
    const pts = series.points.filter(([px, py]) => Number.isFinite(px) && Number.isFinite(py) && (!x.log || px > 0));
    if (!pts.length) continue;
    if (!series.markersOnly && pts.length > 1) {
      const d = pts.map(([px, py], i) => `${i ? "L" : "M"}${tx(px).toFixed(1)},${ty(py).toFixed(1)}`).join("");
      body.append(s("path", { d, class: `gn-series ${series.className}`, "stroke-dasharray": series.dash ?? null }));
    }
    if (series.marker !== "none") {
      const every = Math.max(1, Math.ceil(pts.length / 24));
      pts.forEach(([px, py], i) => {
        if (i % every !== 0 && i !== pts.length - 1) return;
        const atFloor = y.log && py <= floor;
        body.append(
          s("path", {
            d: markerPath(series.marker, 3.5),
            transform: `translate(${tx(px).toFixed(1)},${ty(py).toFixed(1)})`,
            class: `gn-series-mark ${series.className}${atFloor ? " gn-at-floor" : ""}`,
          }),
        );
      });
    }
    const last = pts[pts.length - 1];
    labels.push({ y: ty(last[1]), text: series.label, cls: series.className });
  }
  // Direct labels at the right edge, spread so they do not overlap.
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 13);
  const overflow = labels.length ? labels[labels.length - 1].y - (mg.top + ph) : 0;
  if (overflow > 0) for (const l of labels) l.y -= overflow;
  for (const l of labels) body.append(s("text", { x: mg.left + pw + 6, y: l.y + 4, class: `gn-series-label ${l.cls}` }, l.text));

  return s(
    "svg",
    { class: "gn-plot-svg", width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": spec.title },
    s("title", {}, spec.title),
    axes,
    body,
  );
}
