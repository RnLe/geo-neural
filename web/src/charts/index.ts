// Bytes versus quality, hand-built SVG. Shapes encode family; markers with a
// viewer field are larger and select the candidate in the shared store.

import type { Bundle, ChartRow, Family } from "../data/bundle";
import { h, Listeners, s, uid } from "../data/dom";
import { formatBytes, formatMetres, formatRatio } from "../data/format";
import type { SelectionStore } from "../data/store";

export interface ChartOptions {
  bundle: Bundle;
  store: SelectionStore;
}

export interface ChartHandle {
  dispose(): void;
}

type Metric = "streamJaccard" | "maeM" | "maxM";

const METRICS: Record<Metric, { label: string; axis: string; log: boolean }> = {
  streamJaccard: { label: "Stream overlap (Jaccard)", axis: "stream overlap, Jaccard (higher is closer)", log: false },
  maeM: { label: "Mean absolute error", axis: "mean absolute error, m (log, lower is closer)", log: true },
  maxM: { label: "Maximum error", axis: "maximum absolute error, m (log, lower is closer)", log: true },
};

const FAMILIES: { key: Family; label: string; shape: string }[] = [
  { key: "conventional", label: "conventional", shape: "circle" },
  { key: "neural", label: "neural", shape: "triangle" },
  { key: "hybrid", label: "hybrid", shape: "square" },
  { key: "corrected", label: "corrected", shape: "diamond" },
  { key: "control", label: "control", shape: "cross" },
];

function shapePath(family: Family, r: number): string {
  switch (family) {
    case "neural":
      return `M0,${-1.2 * r}L${1.1 * r},${0.8 * r}L${-1.1 * r},${0.8 * r}Z`;
    case "hybrid":
      return `M${-0.85 * r},${-0.85 * r}h${1.7 * r}v${1.7 * r}h${-1.7 * r}Z`;
    case "corrected":
      return `M0,${-1.3 * r}L${1.1 * r},0L0,${1.3 * r}L${-1.1 * r},0Z`;
    case "control":
      return `M${-r},${-r}L${r},${r}M${-r},${r}L${r},${-r}`;
    default:
      return `M${-r},0a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0`;
  }
}

function shapeIcon(family: Family): SVGSVGElement {
  return s("svg", { width: 16, height: 16, viewBox: "-8 -8 16 16", "aria-hidden": "true", class: "gn-chart-icon" }, s("path", { d: shapePath(family, 5), class: `gn-mark gn-fam-${family}` }));
}

function niceLogTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) out.push(Math.pow(10, e));
  return out;
}

function tickLabel(v: number, metric: Metric | "bytes"): string {
  if (metric === "bytes") return v >= 1e6 ? `${v / 1e6} MB` : v >= 1e3 ? `${v / 1e3} kB` : `${v} B`;
  if (metric === "streamJaccard") return v.toFixed(1);
  return v >= 1 ? String(v) : String(Number(v.toPrecision(1)));
}

export function mountChart(el: HTMLElement, options: ChartOptions): ChartHandle {
  const { bundle, store } = options;
  const rows = bundle.manifest.chart;
  const viewerIds = new Set(bundle.manifest.candidates.map((c) => c.id));
  const byId = new Map(rows.map((r) => [r.id, r]));
  const listeners = new Listeners();
  let metric: Metric = "streamJaccard";
  let hovered: ChartRow | null = null;
  let picked: ChartRow | null = null;

  const root = h("section", { class: "gn-root gn-chart", "aria-label": "Bytes versus quality chart" });
  el.append(root);

  const metricId = uid("gn-metric");
  const metricSelect = h("select", { id: metricId, class: "gn-select" });
  for (const key of Object.keys(METRICS) as Metric[]) metricSelect.append(h("option", { value: key }, METRICS[key].label));

  const findId = uid("gn-find");
  const findSelect = h("select", { id: findId, class: "gn-select" }, h("option", { value: "" }, "choose a candidate"));
  for (const r of [...rows].sort((a, b) => a.bytes - b.bytes)) {
    findSelect.append(h("option", { value: r.id }, `${formatBytes(r.bytes)}: ${r.label}${viewerIds.has(r.id) ? " (viewer field)" : ""}`));
  }

  root.append(
    h(
      "div",
      { class: "gn-controls" },
      h("div", { class: "gn-field" }, h("label", { for: metricId }, "Vertical axis"), metricSelect),
      h("div", { class: "gn-field gn-field-wide" }, h("label", { for: findId }, "Find any candidate (sorted by bytes)"), findSelect),
    ),
  );
  const plot = h("div", { class: "gn-chart-plot" });
  const tooltip = h("div", { class: "gn-tooltip", role: "presentation", hidden: true });
  plot.append(tooltip);
  const legend = h("ul", { class: "gn-legend-list gn-legend-inline gn-chart-legend" });
  const omitted = h("p", { class: "gn-note" });
  const detail = h("div", { class: "gn-chart-detail", "aria-live": "polite" });
  const evidence = new Map<string, number>();
  for (const r of rows) evidence.set(r.evidence, (evidence.get(r.evidence) ?? 0) + 1);
  const evidenceText = [...evidence.entries()].map(([k, n]) => `${n} ${k}`).join(", ");
  const caption = h(
    "p",
    { class: "gn-caption" },
    `Every marker is one serialised candidate stored as independently compressed pages of the finest grid (${rows.length} in all, evidence: ${evidenceText}). x is the number of bytes stored to rebuild the field, on a log axis; ` +
      "y is measured against the 10 m reference. Stream overlap compares the stream cells a D8 routing algorithm derives from each field with those from the reference: " +
      "a routing diagnostic, not discharge. Larger outlined markers have a decoded field in the viewer; click or press Enter on one to show it there.",
  );
  const tableHost = h("details", { class: "gn-chart-table" }, h("summary", {}, `Table of all ${rows.length} candidates`));
  root.append(plot, legend, omitted, detail, caption, tableHost);

  for (const f of FAMILIES) {
    const n = rows.filter((r) => r.family === f.key).length;
    if (!n) continue;
    legend.append(h("li", {}, shapeIcon(f.key), `${f.label} (${f.shape}), ${n}`));
  }
  legend.append(
    h("li", {}, h("span", { class: "gn-chart-big", "aria-hidden": "true" }), "larger, outlined: field in the viewer"),
    h("li", {}, h("span", { class: "gn-chart-ring", "aria-hidden": "true" }), "ring: shown in the viewer now"),
  );

  // ---- layout state ---------------------------------------------------------
  interface Placed {
    row: ChartRow;
    x: number;
    y: number;
    node: SVGGElement;
  }
  let placed: Placed[] = [];
  let ring: SVGCircleElement | null = null;
  let pickRing: SVGCircleElement | null = null;
  let svg: SVGSVGElement | null = null;

  function describe(r: ChartRow): HTMLElement {
    const dominated = r.dominatedBy ? byId.get(r.dominatedBy) : null;
    return h(
      "dl",
      { class: "gn-facts" },
      h("dt", {}, "Candidate"),
      h("dd", {}, `${r.label} (${r.id})`),
      h("dt", {}, "Family"),
      h("dd", {}, `${r.family}, experiment ${r.experiment}`),
      h("dt", {}, "Serialised size"),
      h("dd", {}, `${formatBytes(r.bytes)} (${r.bytes.toLocaleString("en-US")} B)`),
      h("dt", {}, "Mean |error|"),
      h("dd", {}, formatMetres(r.maeM, 3)),
      h("dt", {}, "Max |error|"),
      h("dd", {}, formatMetres(r.maxM, 3) + (r.boundGuaranteed ? " (bound guaranteed)" : "")),
      h("dt", {}, "Stream Jaccard"),
      h("dd", {}, formatRatio(r.streamJaccard)),
      h("dt", {}, "Dominated by"),
      h("dd", {}, dominated ? `${dominated.label} (${formatBytes(dominated.bytes)})` : r.dominatedBy ?? "none in this table"),
      h("dt", {}, "Evidence"),
      h("dd", {}, r.evidence),
    );
  }

  function showDetail(r: ChartRow | null): void {
    if (!r) {
      detail.replaceChildren(h("p", { class: "gn-note" }, "Hover or focus a marker, or pick a candidate above, to see its numbers."));
      return;
    }
    const parts: (HTMLElement | null)[] = [describe(r)];
    if (viewerIds.has(r.id)) {
      if (store.get().candidateId === r.id) parts.push(h("p", { class: "gn-note" }, "Shown in the viewer now."));
      else {
        const b = h("button", { type: "button", class: "gn-button" }, "Show in viewer");
        b.addEventListener("click", () => store.set({ candidateId: r.id }));
        parts.push(b);
      }
    } else {
      parts.push(h("p", { class: "gn-note" }, "No decoded field for this candidate in the bundle; numbers only."));
    }
    detail.replaceChildren(...parts.filter((p): p is HTMLElement => p !== null));
  }

  function showTooltip(p: Placed | null): void {
    if (!p) {
      tooltip.hidden = true;
      return;
    }
    const r = p.row;
    tooltip.replaceChildren(
      h("strong", {}, r.label),
      h("br"),
      `${formatBytes(r.bytes)}, MAE ${formatMetres(r.maeM)}, max ${formatMetres(r.maxM)}, Jaccard ${formatRatio(r.streamJaccard)}`,
      h("br"),
      r.dominatedBy ? `dominated by ${r.dominatedBy}` : "not dominated in this table",
    );
    tooltip.hidden = false;
    const w = plot.clientWidth;
    const left = Math.min(Math.max(8, p.x + 14), Math.max(8, w - 260));
    tooltip.style.transform = `translate(${left}px, ${Math.max(0, p.y - 10)}px)`;
  }

  function updateSelection(): void {
    const id = store.get().candidateId;
    const sel = placed.find((p) => p.row.id === id);
    if (ring) {
      ring.setAttribute("visibility", sel ? "visible" : "hidden");
      if (sel) ring.setAttribute("transform", `translate(${sel.x},${sel.y})`);
    }
    for (const p of placed) p.node.classList.toggle("gn-selected", p.row.id === id);
    const pk = picked ? placed.find((p) => p.row.id === picked?.id) : undefined;
    if (pickRing) {
      pickRing.setAttribute("visibility", pk ? "visible" : "hidden");
      if (pk) pickRing.setAttribute("transform", `translate(${pk.x},${pk.y})`);
    }
  }

  function render(): void {
    const width = Math.max(300, plot.clientWidth || 640);
    const height = Math.round(Math.min(460, Math.max(300, width * 0.56)));
    const mg = { left: 58, right: 16, top: 14, bottom: 46 };
    const pw = width - mg.left - mg.right;
    const ph = height - mg.top - mg.bottom;
    const spec = METRICS[metric];
    const shown = rows.filter((r) => r[metric] !== null && Number.isFinite(r[metric] as number));
    const missing = rows.length - shown.length;
    omitted.textContent = missing
      ? `${missing} of ${rows.length} candidates have no ${spec.label.toLowerCase()} recorded and are not drawn in this view.`
      : `All ${rows.length} candidates are drawn.`;

    const bytes = rows.map((r) => r.bytes);
    const xLo = Math.min(...bytes) / 1.8;
    const xHi = Math.max(...bytes) * 1.8;
    const lx = (v: number) => mg.left + ((Math.log10(v) - Math.log10(xLo)) / (Math.log10(xHi) - Math.log10(xLo))) * pw;
    let yLo = 0;
    let yHi = 1;
    if (spec.log) {
      const vals = shown.map((r) => r[metric] as number);
      yLo = Math.min(...vals) / 1.6;
      yHi = Math.max(...vals) * 1.6;
    }
    const ly = (v: number) =>
      spec.log
        ? mg.top + (1 - (Math.log10(v) - Math.log10(yLo)) / (Math.log10(yHi) - Math.log10(yLo))) * ph
        : mg.top + (1 - (v - yLo) / (yHi - yLo)) * ph;

    const axes = s("g", { class: "gn-axis" });
    for (const v of niceLogTicks(xLo, xHi)) {
      axes.append(
        s("line", { x1: lx(v), x2: lx(v), y1: mg.top, y2: mg.top + ph, class: "gn-gridline" }),
        s("text", { x: lx(v), y: mg.top + ph + 16, "text-anchor": "middle" }, tickLabel(v, "bytes")),
      );
    }
    const yTicks = spec.log ? niceLogTicks(yLo, yHi) : [0, 0.2, 0.4, 0.6, 0.8, 1];
    for (const v of yTicks) {
      axes.append(
        s("line", { x1: mg.left, x2: mg.left + pw, y1: ly(v), y2: ly(v), class: "gn-gridline" }),
        s("text", { x: mg.left - 6, y: ly(v) + 4, "text-anchor": "end" }, tickLabel(v, metric)),
      );
    }
    axes.append(
      s("line", { x1: mg.left, x2: mg.left + pw, y1: mg.top + ph, y2: mg.top + ph, class: "gn-axisline" }),
      s("line", { x1: mg.left, x2: mg.left, y1: mg.top, y2: mg.top + ph, class: "gn-axisline" }),
      s("text", { x: mg.left + pw / 2, y: height - 8, "text-anchor": "middle" }, "serialised size, bytes (log)"),
      s("text", { x: 14, y: mg.top + ph / 2, transform: `rotate(-90 14 ${mg.top + ph / 2})`, "text-anchor": "middle" }, spec.axis),
    );

    const marks = s("g", { class: "gn-marks" });
    const front = s("g", { class: "gn-marks-front" });
    placed = [];
    // Small markers first, viewer markers on top.
    const ordered = [...shown].sort((a, b) => Number(viewerIds.has(a.id)) - Number(viewerIds.has(b.id)));
    for (const r of ordered) {
      const x = lx(r.bytes);
      const y = ly(r[metric] as number);
      const big = viewerIds.has(r.id);
      const cls = `gn-mark gn-fam-${r.family}${big ? " gn-mark-big" : ""}${r.evidence !== "reproduced" ? " gn-mark-other" : ""}`;
      const node = s("g", { transform: `translate(${x.toFixed(1)},${y.toFixed(1)})` }, s("path", { d: shapePath(r.family, big ? 7 : 4), class: cls }));
      if (big) {
        node.setAttribute("tabindex", "0");
        node.setAttribute("role", "button");
        node.setAttribute("class", "gn-mark-button");
        node.setAttribute(
          "aria-label",
          `${r.label}, ${r.family}, ${formatBytes(r.bytes)}, ${METRICS[metric].label} ${metric === "streamJaccard" ? formatRatio(r[metric]) : formatMetres(r[metric] as number)}. Press Enter to show in the viewer.`,
        );
        front.append(node);
      } else {
        marks.append(node);
      }
      placed.push({ row: r, x, y, node });
    }
    ring = s("circle", { r: 12, class: "gn-chart-selring", visibility: "hidden", "pointer-events": "none" });
    pickRing = s("circle", { r: 9, class: "gn-chart-pickring", visibility: "hidden", "pointer-events": "none" });

    const titleId = uid("gn-chart-title");
    svg = s(
      "svg",
      { class: "gn-chart-svg", width, height, viewBox: `0 0 ${width} ${height}`, role: "group", "aria-labelledby": titleId },
      s("title", { id: titleId }, `Serialised size versus ${spec.label.toLowerCase()} for ${shown.length} candidates`),
      axes,
      marks,
      pickRing,
      front,
      ring,
    );
    plot.querySelector("svg")?.remove();
    plot.prepend(svg);
    updateSelection();
  }

  function nearest(clientX: number, clientY: number): Placed | null {
    if (!svg) return null;
    const rect = svg.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    let best: Placed | null = null;
    let bestD = 16;
    for (const p of placed) {
      const d = Math.hypot(p.x - x, p.y - y) - (viewerIds.has(p.row.id) ? 4 : 0);
      if (d < bestD) {
        bestD = d;
        best = p;
      }
    }
    return best;
  }

  listeners.on(plot, "pointermove", (e: PointerEvent) => {
    const p = nearest(e.clientX, e.clientY);
    plot.classList.toggle("gn-chart-clickable", !!p && viewerIds.has(p.row.id));
    if (p?.row !== hovered) {
      hovered = p?.row ?? null;
      showTooltip(p);
      if (p) showDetail(p.row);
    }
  });
  listeners.on(plot, "pointerleave", () => {
    hovered = null;
    showTooltip(null);
    plot.classList.remove("gn-chart-clickable");
  });
  listeners.on(plot, "click", (e: MouseEvent) => {
    const p = nearest(e.clientX, e.clientY);
    if (!p) return;
    picked = p.row;
    findSelect.value = p.row.id;
    if (viewerIds.has(p.row.id)) store.set({ candidateId: p.row.id });
    showDetail(p.row);
    updateSelection();
  });
  listeners.on(plot, "focusin", (e: FocusEvent) => {
    const p = placed.find((q) => q.node === e.target);
    if (p) {
      showTooltip(p);
      showDetail(p.row);
    }
  });
  listeners.on(plot, "focusout", () => showTooltip(null));
  listeners.on(plot, "keydown", (e: KeyboardEvent) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const p = placed.find((q) => q.node === e.target);
    if (!p) return;
    e.preventDefault();
    store.set({ candidateId: p.row.id });
    showDetail(p.row);
  });
  listeners.on(metricSelect, "change", () => {
    metric = metricSelect.value as Metric;
    render();
  });
  listeners.on(findSelect, "change", () => {
    const r = byId.get(findSelect.value) ?? null;
    picked = r;
    showDetail(r);
    updateSelection();
  });
  listeners.on(tableHost, "toggle", () => {
    if (!tableHost.open || tableHost.querySelector("table")) return;
    const body = h("tbody");
    for (const r of [...rows].sort((a, b) => a.bytes - b.bytes)) {
      body.append(
        h(
          "tr",
          {},
          h("td", {}, r.label),
          h("td", {}, r.family),
          h("td", { class: "gn-num" }, r.bytes.toLocaleString("en-US")),
          h("td", { class: "gn-num" }, r.maeM === null ? "" : r.maeM.toFixed(3)),
          h("td", { class: "gn-num" }, r.maxM === null ? "" : r.maxM.toFixed(3)),
          h("td", { class: "gn-num" }, r.streamJaccard === null ? "" : r.streamJaccard.toFixed(3)),
          h("td", {}, r.dominatedBy ?? ""),
          h("td", {}, r.evidence),
        ),
      );
    }
    tableHost.append(
      h(
        "div",
        { class: "gn-table-wrap" },
        h(
          "table",
          { class: "gn-table" },
          h(
            "thead",
            {},
            h(
              "tr",
              {},
              ...["Candidate", "Family", "Bytes", "MAE (m)", "Max (m)", "Jaccard", "Dominated by", "Evidence"].map((t) => h("th", { scope: "col" }, t)),
            ),
          ),
          body,
        ),
      ),
    );
  });

  const unsubscribe = store.subscribe((next, prev) => {
    if (next.candidateId === prev.candidateId) return;
    updateSelection();
    const r = byId.get(next.candidateId);
    if (r && (hovered === null || hovered.id === r.id)) showDetail(r);
  });

  let lastWidth = 0;
  const resizeObserver = new ResizeObserver(() => {
    const w = plot.clientWidth;
    if (Math.abs(w - lastWidth) < 4) return;
    lastWidth = w;
    render();
  });
  resizeObserver.observe(plot);

  render();
  showDetail(byId.get(store.get().candidateId) ?? null);

  return {
    dispose() {
      unsubscribe();
      resizeObserver.disconnect();
      listeners.clear();
      root.remove();
    },
  };
}
