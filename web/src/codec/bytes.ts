// Where the bytes go: the learned coder's cost per field when one shared model serves a corpus of N fields,
// against SZ3 and cubic-ctx, whose cost per field does not depend on N. Hand-built SVG.

import { h, Listeners, s, uid } from "../data/dom";
import { formatBytes, formatInt } from "../data/format";
import { corpusSummary, perField, type CorpusSummary } from "./accounting";
import type { CodecBundle } from "./bundle";

export interface BytesOptions {
  bundle: CodecBundle;
}

export interface BytesHandle {
  dispose(): void;
}

const MARKS = [1, 10, 100];

function boundLabel(b: number): string {
  return b >= 0.01 ? `${b} m` : `${b * 1000} mm`;
}

function niceStep(range: number, target: number): number {
  const raw = range / Math.max(1, target);
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

function evenText(n: number | null, rival: string): string {
  if (n === null) return `never: the learned product alone is larger than ${rival}`;
  if (n === 1) return `from the first field (the model costs less than the saving on one field)`;
  return `${formatInt(n)} fields`;
}

export function mountBytes(el: HTMLElement, options: BytesOptions): BytesHandle {
  const m = options.bundle.manifest;
  const listeners = new Listeners();
  const bounds = m.bounds.filter((b) => corpusSummary(m, b) !== null);
  const root = h("section", { class: "gn-root gn-codec-bytes-view", "aria-label": "Bytes per field against corpus size" });
  el.append(root);
  if (!bounds.length) {
    root.append(h("p", { class: "gn-message" }, "The codec bundle has no bound with all three products, so there is nothing to compare."));
    return { dispose: () => root.remove() };
  }
  let bound = bounds[0];
  const boundId = uid("gn-codec-cbound");
  const boundSelect = h("select", { id: boundId, class: "gn-select" }, ...bounds.map((b) => h("option", { value: String(b) }, boundLabel(b))));
  const plot = h("div", { class: "gn-chart-plot" });
  const readout = h("p", { class: "gn-note gn-codec-readout" });
  const facts = h("div", { class: "gn-codec-even" });
  const tableHost = h("div", { class: "gn-table-wrap" });
  root.append(
    h("div", { class: "gn-controls" }, h("div", { class: "gn-field" }, h("label", { for: boundId }, "Error bound"), boundSelect)),
    plot,
    readout,
    h(
      "ul",
      { class: "gn-legend-list gn-legend-inline" },
      h("li", {}, h("span", { class: "gn-swatch-line gn-codec-swatch-learned", "aria-hidden": "true" }), "learned: product + model / N (dashed where extrapolated)"),
      h("li", {}, h("span", { class: "gn-swatch-line gn-codec-swatch-sz3", "aria-hidden": "true" }), "SZ3"),
      h("li", {}, h("span", { class: "gn-swatch-line gn-codec-swatch-cubic-ctx", "aria-hidden": "true" }), "cubic-ctx"),
    ),
    facts,
    h("details", { class: "gn-chart-table" }, h("summary", {}, "Bytes per field (table)"), tableHost),
  );

  function render(): void {
    const sum = corpusSummary(m, bound) as CorpusSummary;
    const measured = sum.regions.length;
    const evens = [sum.breakEvenSz3, sum.breakEvenCtx].filter((n): n is number => n !== null);
    const xMax = Math.max(100, ...evens.map((n) => Math.pow(10, Math.ceil(Math.log10(2 * n)))));
    const width = Math.max(300, plot.clientWidth || 640);
    const height = Math.round(Math.min(360, Math.max(240, width * 0.45)));
    const mg = { left: 70, right: 16, top: 14, bottom: 44 };
    const pw = width - mg.left - mg.right;
    const ph = height - mg.top - mg.bottom;
    const lx = (n: number) => mg.left + (Math.log10(n) / Math.log10(xMax)) * pw;
    const vals = [sum.learned, sum.learned + sum.model, sum.sz3, sum.ctx];
    let lo = Math.min(...vals);
    let hi = Math.max(...vals);
    const pad = Math.max((hi - lo) * 0.12, hi * 0.01);
    lo -= pad;
    hi += pad;
    const yStep = niceStep(hi - lo, 5);
    lo = Math.floor(lo / yStep) * yStep;
    hi = Math.ceil(hi / yStep) * yStep;
    const ly = (v: number) => mg.top + (1 - (v - lo) / (hi - lo)) * ph;

    const axes = s("g", { class: "gn-axis" });
    for (let e = 0; Math.pow(10, e) <= xMax; e++) {
      const v = Math.pow(10, e);
      axes.append(
        s("line", { x1: lx(v), x2: lx(v), y1: mg.top, y2: mg.top + ph, class: "gn-gridline" }),
        s("text", { x: lx(v), y: mg.top + ph + 16, "text-anchor": "middle" }, formatInt(v)),
      );
    }
    for (let v = lo; v <= hi + 1e-9; v += yStep) {
      axes.append(
        s("line", { x1: mg.left, x2: mg.left + pw, y1: ly(v), y2: ly(v), class: "gn-gridline" }),
        s("text", { x: mg.left - 6, y: ly(v) + 4, "text-anchor": "end" }, formatBytes(v)),
      );
    }
    axes.append(
      s("line", { x1: mg.left, x2: mg.left + pw, y1: mg.top + ph, y2: mg.top + ph, class: "gn-axisline" }),
      s("text", { x: mg.left + pw / 2, y: height - 8, "text-anchor": "middle" }, "fields sharing one model, N (log)"),
      s("text", { x: 14, y: mg.top + ph / 2, transform: `rotate(-90 14 ${mg.top + ph / 2})`, "text-anchor": "middle" }, "bytes per field (axis not from 0)"),
    );

    const curve = (from: number, to: number) => {
      const pts: string[] = [];
      const steps = 80;
      for (let i = 0; i <= steps; i++) {
        const n = from * Math.pow(to / from, i / steps);
        pts.push(`${i ? "L" : "M"}${lx(n).toFixed(1)},${ly(perField(sum, n)).toFixed(1)}`);
      }
      return pts.join("");
    };
    const marks = s("g", {});
    marks.append(
      s("line", { x1: lx(1), x2: lx(xMax), y1: ly(sum.sz3), y2: ly(sum.sz3), class: "gn-codec-line gn-codec-line-sz3" }),
      s("line", { x1: lx(1), x2: lx(xMax), y1: ly(sum.ctx), y2: ly(sum.ctx), class: "gn-codec-line gn-codec-line-cubic-ctx" }),
    );
    if (measured > 1) marks.append(s("path", { d: curve(1, measured), class: "gn-codec-line gn-codec-line-learned" }));
    marks.append(s("path", { d: curve(Math.max(1, measured), xMax), class: "gn-codec-line gn-codec-line-learned gn-codec-dashed" }));
    for (const n of MARKS.filter((v) => v <= xMax)) {
      marks.append(s("circle", { cx: lx(n), cy: ly(perField(sum, n)), r: 4.5, class: "gn-codec-dot gn-codec-dot-learned" }));
    }
    // Direct labels above the right end of each line, nudged apart when they would touch.
    const ends = [
      { v: perField(sum, xMax), text: "learned" },
      { v: sum.sz3, text: "SZ3" },
      { v: sum.ctx, text: "cubic-ctx" },
    ].sort((a, b) => ly(a.v) - ly(b.v));
    let last = -Infinity;
    for (const e of ends) {
      const y = Math.max(ly(e.v) - 6, last + 13);
      last = y;
      marks.append(s("text", { x: lx(xMax) - 4, y, "text-anchor": "end", class: "gn-codec-annot" }, e.text));
    }
    for (const [n, label] of [[sum.breakEvenSz3, "SZ3"], [sum.breakEvenCtx, "cubic-ctx"]] as [number | null, string][]) {
      if (n === null || n <= 1 || n > xMax) continue;
      marks.append(
        s("line", { x1: lx(n), x2: lx(n), y1: mg.top, y2: mg.top + ph, class: "gn-floorline" }),
        s("text", { x: lx(n) + 4, y: mg.top + 12, class: "gn-codec-annot" }, `break-even with ${label}: N = ${formatInt(n)}`),
      );
    }
    if (measured < xMax) {
      marks.append(s("text", { x: lx(Math.max(1, measured)) + 6, y: mg.top + ph - 8, class: "gn-codec-annot" }, `measured: N <= ${measured}; beyond, extrapolated`));
    }
    const cross = s("line", { y1: mg.top, y2: mg.top + ph, class: "gn-codec-cross", visibility: "hidden" });
    const titleId = uid("gn-codec-bytes-title");
    const svg = s(
      "svg",
      { class: "gn-chart-svg", width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-labelledby": titleId },
      s("title", { id: titleId }, `Bytes per field against corpus size at a ${boundLabel(bound)} bound`),
      axes,
      marks,
      cross,
    );
    const show = (n: number | null) => {
      if (n === null) {
        cross.setAttribute("visibility", "hidden");
        readout.textContent = "Move the pointer over the chart to read the bytes per field at any N.";
        return;
      }
      cross.setAttribute("x1", String(lx(n)));
      cross.setAttribute("x2", String(lx(n)));
      cross.setAttribute("visibility", "visible");
      readout.textContent =
        `N = ${formatInt(n)}${n > measured ? " (extrapolated)" : ""}: learned ${formatInt(perField(sum, n))} B per field, ` +
        `SZ3 ${formatInt(sum.sz3)} B, cubic-ctx ${formatInt(sum.ctx)} B.`;
    };
    svg.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const t = (ev.clientX - r.left - mg.left) / pw;
      show(t < -0.02 || t > 1.02 ? null : Math.max(1, Math.min(xMax, Math.round(Math.pow(xMax, Math.max(0, Math.min(1, t)))))));
    });
    svg.addEventListener("pointerleave", () => show(null));
    plot.replaceChildren(svg);
    show(null);

    facts.replaceChildren(
      h(
        "dl",
        { class: "gn-facts" },
        h("dt", {}, "Break-even with SZ3"),
        h("dd", {}, evenText(sum.breakEvenSz3, "SZ3")),
        h("dt", {}, "Break-even with cubic-ctx"),
        h("dd", {}, evenText(sum.breakEvenCtx, "cubic-ctx")),
        h("dt", {}, "Measured corpus"),
        h("dd", {}, `${measured} field${measured === 1 ? "" : "s"}: ${sum.regions.join(", ")}`),
        h("dt", {}, "Shared model"),
        h("dd", {}, `${formatInt(sum.model)} B, counted once per corpus`),
      ),
      h(
        "p",
        { class: "gn-note" },
        `Means over the measured fields at a ${boundLabel(bound)} bound. Beyond N = ${measured} the curve assumes further fields cost the mean of the measured ones; that part is an extrapolation, not a measurement. ` +
          "Each measured learned product used a model trained without its own region (same size for every region), so its bytes are those of a model that never saw the field. At N = 1 the standalone file, with the model embedded, is 9 bytes larger than product plus model.",
      ),
    );

    const body = h("tbody");
    for (const n of MARKS) {
      const l = perField(sum, n);
      body.append(
        h(
          "tr",
          {},
          h("th", { scope: "row" }, `${formatInt(n)}${n > measured ? " (extrapolated)" : ""}`),
          h("td", { class: "gn-num" }, formatInt(l)),
          h("td", { class: "gn-num" }, formatInt(sum.sz3)),
          h("td", { class: "gn-num" }, formatInt(sum.ctx)),
          h("td", { class: "gn-num" }, (l / sum.sz3).toFixed(3)),
          h("td", { class: "gn-num" }, (l / sum.ctx).toFixed(3)),
        ),
      );
    }
    tableHost.replaceChildren(
      h(
        "table",
        { class: "gn-table" },
        h(
          "thead",
          {},
          h("tr", {}, ...["N", "learned (B)", "SZ3 (B)", "cubic-ctx (B)", "learned / SZ3", "learned / cubic-ctx"].map((t) => h("th", { scope: "col" }, t))),
        ),
        body,
      ),
    );
  }

  listeners.on(boundSelect, "change", () => {
    bound = Number(boundSelect.value);
    render();
  });
  let lastWidth = 0;
  const resize = new ResizeObserver(() => {
    const w = plot.clientWidth;
    if (Math.abs(w - lastWidth) < 4) return;
    lastWidth = w;
    render();
  });
  resize.observe(plot);
  render();

  return {
    dispose() {
      resize.disconnect();
      listeners.clear();
      root.remove();
    },
  };
}
