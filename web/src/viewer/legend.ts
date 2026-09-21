// Legends with numbers for the four overlays.

import { h, uid } from "../data/dom";
import { formatInt, formatMetres, formatPercent, formatRatio } from "../data/format";
import { cssGradient, DIVERGING_STOPS, ELEVATION_STOPS, STREAM_COLORS } from "../data/palette";
import type { Candidate, LegendEntry } from "../data/bundle";
import { STREAM_BOTH, STREAM_LOST, STREAM_SPURIOUS } from "./terrain";

export interface FieldStats {
  count: number;
  belowRange: number;
  aboveRange: number;
  meanAbsErrorM: number;
  maxAbsErrorM: number;
  minHeightM: number;
  maxHeightM: number;
  both: number;
  lost: number;
  spurious: number;
}

export function computeStats(reference: Float32Array, refStreams: Uint8Array, decoded: Float32Array, candStreams: Uint8Array, rangeM: number): FieldStats {
  const n = reference.length;
  let below = 0;
  let above = 0;
  let sum = 0;
  let max = 0;
  let lo = Infinity;
  let hi = -Infinity;
  let both = 0;
  let lost = 0;
  let spurious = 0;
  for (let i = 0; i < n; i++) {
    const e = decoded[i] - reference[i];
    if (e < -rangeM) below++;
    else if (e > rangeM) above++;
    const a = Math.abs(e);
    sum += a;
    if (a > max) max = a;
    if (decoded[i] < lo) lo = decoded[i];
    if (decoded[i] > hi) hi = decoded[i];
    const r = refStreams[i];
    const c = candStreams[i];
    if (r && c) both++;
    else if (r) lost++;
    else if (c) spurious++;
  }
  return { count: n, belowRange: below, aboveRange: above, meanAbsErrorM: sum / n, maxAbsErrorM: max, minHeightM: lo, maxHeightM: hi, both, lost, spurious };
}

function ticks(lo: number, hi: number, step: number): number[] {
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(v);
  return out;
}

function bar(stops: readonly string[], marks: { at: number; label: string }[], clipped = false): HTMLElement {
  return h(
    "div",
    { class: "gn-ramp" },
    h(
      "div",
      { class: clipped ? "gn-ramp-bar gn-ramp-bar-clipped" : "gn-ramp-bar", style: `background:${cssGradient(stops)}`, "aria-hidden": "true" },
      clipped ? h("span", { class: "gn-ramp-clip gn-ramp-clip-low" }) : null,
      clipped ? h("span", { class: "gn-ramp-clip gn-ramp-clip-high" }) : null,
    ),
    h("div", { class: "gn-ramp-ticks" }, ...marks.map((m) => h("span", { class: "gn-ramp-tick", style: `left:${(100 * m.at).toFixed(2)}%` }, m.label))),
  );
}

export interface LegendContext {
  hMin: number;
  hMax: number;
  errRangeM: number;
  verticalCrs: string;
  candidate: Candidate;
  stats: FieldStats | null;
  geologyLegend: LegendEntry[];
  geologyCounts: number[];
  geologySource: string;
  geologyScale: string;
  geologyNote: string;
  streamThresholdCells: number;
  streamFilter: number;
  onStreamFilter: (bits: number) => void;
}

export function elevationLegend(c: LegendContext): HTMLElement {
  const marks = ticks(c.hMin, c.hMax, 20).map((v) => ({ at: (v - c.hMin) / (c.hMax - c.hMin), label: `${v}` }));
  const st = c.stats;
  return h(
    "div",
    { class: "gn-legend" },
    h("p", { class: "gn-legend-title" }, `Height (m, ${c.verticalCrs})`),
    bar(ELEVATION_STOPS, marks),
    h(
      "p",
      { class: "gn-note" },
      `Fixed scale ${c.hMin.toFixed(1)} to ${c.hMax.toFixed(1)} m: the reference range at 10 m. `,
      st ? `This decoded field spans ${st.minHeightM.toFixed(1)} to ${st.maxHeightM.toFixed(1)} m on the 20 m display grid.` : "",
    ),
  );
}

export function errorLegend(c: LegendContext): HTMLElement {
  const r = c.errRangeM;
  const marks = [-r, -r / 2, 0, r / 2, r].map((v) => ({ at: (v + r) / (2 * r), label: v === 0 ? "0" : v > 0 ? `+${v}` : `${v}` }));
  const st = c.stats;
  return h(
    "div",
    { class: "gn-legend" },
    h("p", { class: "gn-legend-title" }, "Signed error: decoded minus reference (m)"),
    bar(DIVERGING_STOPS, marks, true),
    h("p", { class: "gn-note" }, `Blue: decoded below the reference. Red: decoded above. One fixed scale of +/-${r} m for every candidate; the end blocks hold everything beyond it.`),
    st
      ? h(
          "dl",
          { class: "gn-facts" },
          h("dt", {}, `Below -${r} m`),
          h("dd", {}, `${formatInt(st.belowRange)} nodes (${formatPercent(st.belowRange / st.count, 2)})`),
          h("dt", {}, `Above +${r} m`),
          h("dd", {}, `${formatInt(st.aboveRange)} nodes (${formatPercent(st.aboveRange / st.count, 2)})`),
          h("dt", {}, "Mean |error|, 20 m grid"),
          h("dd", {}, `${formatMetres(st.meanAbsErrorM)} (recorded at 10 m: ${formatMetres(c.candidate.check.maeM)})`),
          h("dt", {}, "Max |error|, 20 m grid"),
          h("dd", {}, `${formatMetres(st.maxAbsErrorM)} (recorded at 10 m: ${formatMetres(c.candidate.check.maxM)})`),
        )
      : null,
  );
}

export function geologyLegend(c: LegendContext): HTMLElement {
  const total = c.geologyCounts.reduce((a, b) => a + b, 0) || 1;
  return h(
    "div",
    { class: "gn-legend" },
    h("p", { class: "gn-legend-title" }, "Mapped surface geology (share of display nodes)"),
    h(
      "ul",
      { class: "gn-legend-list" },
      ...c.geologyLegend.map((entry) =>
        h(
          "li",
          {},
          h("span", {
            class: entry.code === 0 ? "gn-swatch gn-swatch-unmapped" : "gn-swatch",
            style: entry.code === 0 ? "" : `background:${entry.color}`,
            "aria-hidden": "true",
          }),
          h("span", { class: "gn-legend-label" }, entry.label),
          h("span", { class: "gn-legend-num" }, formatPercent((c.geologyCounts[entry.code] ?? 0) / total)),
        ),
      ),
      h("li", {}, h("span", { class: "gn-swatch-line gn-swatch-line-fault", "aria-hidden": "true" }), h("span", { class: "gn-legend-label" }, "fault trace (dashed line)")),
    ),
    h("p", { class: "gn-note" }, `${c.geologySource}, ${c.geologyScale}. ${c.geologyNote} Colours are labels only; the class under any point is in the inspector.`),
  );
}

export function streamsLegend(c: LegendContext): HTMLElement {
  const st = c.stats;
  const rows: { bit: number; key: keyof typeof STREAM_COLORS; label: string; style: string; count: number | null }[] = [
    { bit: STREAM_BOTH, key: "both", label: "in both", style: "solid", count: st ? st.both : null },
    { bit: STREAM_LOST, key: "lost", label: "lost: reference only", style: "striped", count: st ? st.lost : null },
    { bit: STREAM_SPURIOUS, key: "spurious", label: "spurious: candidate only", style: "checkered", count: st ? st.spurious : null },
  ];
  const union = st ? st.both + st.lost + st.spurious : 0;
  const list = h("ul", { class: "gn-legend-list" });
  for (const row of rows) {
    const id = uid("gn-stream");
    const box = h("input", { type: "checkbox", id, checked: (c.streamFilter & row.bit) !== 0 });
    box.addEventListener("change", () => {
      const bits = box.checked ? c.streamFilter | row.bit : c.streamFilter & ~row.bit;
      c.streamFilter = bits;
      c.onStreamFilter(bits);
    });
    list.append(
      h(
        "li",
        {},
        box,
        h(
          "label",
          { for: id, class: "gn-legend-label" },
          h("span", { class: `gn-swatch gn-swatch-${row.key}`, style: `--gn-swatch:${STREAM_COLORS[row.key]}`, "aria-hidden": "true" }),
          `${row.label} (${row.style})`,
        ),
        h("span", { class: "gn-legend-num" }, row.count === null ? "" : `${formatInt(row.count)} nodes`),
      ),
    );
  }
  return h(
    "div",
    { class: "gn-legend" },
    h("p", { class: "gn-legend-title" }, "Derived stream cells, reference vs candidate"),
    list,
    h(
      "dl",
      { class: "gn-facts" },
      h("dt", {}, "Overlap, 20 m display nodes"),
      h("dd", {}, st && union ? formatRatio(st.both / union) : "not available"),
      h("dt", {}, "Recorded Jaccard at 10 m"),
      h("dd", {}, formatRatio(c.candidate.streamJaccard ?? c.candidate.check.streamJaccard)),
    ),
    h(
      "p",
      { class: "gn-note" },
      `Stream cells: D8 routing after depression filling, same settings on both fields; a stream cell drains at least ${c.streamThresholdCells} cells of 10 m ` +
        `(${((c.streamThresholdCells * 100) / 1e6).toFixed(2)} km2). Pooled to the 20 m display grid: a node is set if any 10 m cell around it is. ` +
        "A routing diagnostic, not discharge. Untick a class to hide it.",
    ),
  );
}
