// Preset 3: linear diffusion on the Essen lab terrain with a per-class
// diffusivity D0 x ratio[class]. The ratios are assumptions, not measurements.

import type { LabData, LegendEntry } from "../data/bundle";
import { h, uid } from "../data/dom";
import { formatPercent, formatSci, formatYears } from "../data/format";
import { cssGradient, DIVERGING_STOPS, ELEVATION_STOPS } from "../data/palette";
import type { RunFrame, RunSpec } from "./protocol";
import { drawClasses, drawDiverging, drawHeight, niceRange } from "./render";
import { controlStrip, diagnosticsList, logSlider, minMax, rampLegend, selectField, type Preset, type RunnerApi } from "./ui";

const FRAMES = 50;
const MAX_RATIO = 50;

export function materialPreset(api: RunnerApi, lab: Promise<LabData>, legend: LegendEntry[]): Preset {
  let data: LabData | null = null;
  let loadError: string | null = null;
  const ratios = new Map<number, number>(legend.map((e) => [e.code, 1]));

  const d0 = logSlider("Base diffusivity D0", 0.001, 0.1, 0.01, "m²/yr", 2);
  const years = selectField(
    "Run length N",
    [
      ["10000", "10,000 years"],
      ["100000", "100,000 years"],
      ["1000000", "1,000,000 years"],
    ],
    "100000",
  );
  d0.input.addEventListener("change", () => api.paramsChanged());
  years.select.addEventListener("change", () => api.paramsChanged());
  const strip = controlStrip(api);

  const tableBody = h("tbody");
  const inputs = new Map<number, HTMLInputElement>();
  const shareCells = new Map<number, HTMLElement>();
  const changeCells = new Map<number, HTMLElement>();
  for (const entry of legend) {
    const id = uid("gn-ratio");
    const input = h("input", { type: "number", id, min: 0, max: MAX_RATIO, step: 0.1, value: 1, class: "gn-number", inputmode: "decimal" });
    input.addEventListener("change", () => {
      let v = Number(input.value);
      if (!Number.isFinite(v) || v < 0) v = 0;
      if (v > MAX_RATIO) v = MAX_RATIO;
      input.value = String(v);
      ratios.set(entry.code, v);
      api.paramsChanged();
    });
    inputs.set(entry.code, input);
    const share = h("td", { class: "gn-num" });
    const change = h("td", { class: "gn-num" });
    shareCells.set(entry.code, share);
    changeCells.set(entry.code, change);
    tableBody.append(
      h(
        "tr",
        {},
        h(
          "th",
          { scope: "row" },
          h("span", { class: entry.code === 0 ? "gn-swatch gn-swatch-unmapped" : "gn-swatch", style: entry.code === 0 ? "" : `background:${entry.color}`, "aria-hidden": "true" }),
          h("label", { for: id }, entry.label),
        ),
        h("td", {}, input),
        share,
        change,
      ),
    );
  }
  const resetRatios = h("button", { type: "button", class: "gn-button" }, "All ratios to 1");
  resetRatios.addEventListener("click", () => {
    for (const [code, input] of inputs) {
      input.value = "1";
      ratios.set(code, 1);
    }
    api.paramsChanged();
  });

  const classCanvas = h("canvas", { class: "gn-lab-canvas", role: "img", "aria-label": "Mapped geology classes of the lab terrain" });
  const heightCanvas = h("canvas", { class: "gn-lab-canvas", role: "img", "aria-label": "Lab terrain height now" });
  const changeCanvas = h("canvas", { class: "gn-lab-canvas", role: "img", "aria-label": "Change in height since the start" });
  const heightLegend = h("div", {});
  const changeLegend = h("div", {});
  const diag = h("div", { class: "gn-lab-diag" });

  const el = h(
    "div",
    { class: "gn-lab-preset" },
    h(
      "p",
      { class: "gn-lab-intro" },
      "Linear diffusion on the Essen terrain sampled every 80 m (129 x 129 nodes), closed boundaries. Each node takes the diffusivity D0 x ratio of its mapped " +
        "surface class; faces use the harmonic mean of their two cells. A conditional sensitivity experiment: it shows how an assumed contrast would redistribute " +
        "height, not how the Ruhr valley erodes.",
    ),
    h("div", { class: "gn-controls" }, d0.el, years.el),
    h(
      "div",
      { class: "gn-lab-assume" },
      h("p", { class: "gn-legend-title" }, "Diffusivity ratio per class: assumed values, not measured properties"),
      h(
        "div",
        { class: "gn-table-wrap" },
        h(
          "table",
          { class: "gn-table" },
          h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Mapped class"), h("th", { scope: "col" }, "Ratio (assumed)"), h("th", { scope: "col" }, "Share of nodes"), h("th", { scope: "col" }, "Mean change (m)"))),
          tableBody,
        ),
      ),
      resetRatios,
    ),
    strip.el,
    h(
      "div",
      { class: "gn-lab-grid gn-lab-grid-3" },
      h("figure", { class: "gn-lab-card" }, h("figcaption", { class: "gn-legend-title" }, "Mapped classes (table colours)"), classCanvas),
      h("figure", { class: "gn-lab-card" }, h("figcaption", { class: "gn-legend-title" }, "Height now"), heightCanvas, heightLegend, diag),
      h("figure", { class: "gn-lab-card" }, h("figcaption", { class: "gn-legend-title" }, "Change in height since the start"), changeCanvas, changeLegend),
    ),
  );

  let frames: RunFrame[] = [];
  let current: Float32Array | null = null;
  let time = 0;
  let range: [number, number] = [0, 1];

  lab.then(
    (d) => {
      data = d;
      const t = d.terrain;
      range = minMax(t.height);
      const counts = new Map<number, number>();
      for (const c of t.classes) counts.set(c, (counts.get(c) ?? 0) + 1);
      for (const [code, cell] of shareCells) cell.textContent = formatPercent((counts.get(code) ?? 0) / t.classes.length);
      drawClasses(classCanvas, t.classes, t.side, new Map(legend.map((e) => [e.code, e.color])));
      // Nothing can have run before the terrain arrived.
      initial();
      strip.update(false, true, false, false);
    },
    (err: unknown) => {
      loadError = `Could not load the lab terrain: ${err instanceof Error ? err.message : String(err)}`;
      show();
    },
  );

  const N = () => Number(years.value());

  function initial(): void {
    frames = [];
    current = null;
    time = 0;
    show();
    if (data && !loadError) {
      const maxRatio = Math.max(...legend.map((e) => ratios.get(e.code) ?? 1));
      strip.status.textContent = `Ready: ${FRAMES} steps of ${formatYears(N() / FRAMES)}; largest D = ${formatSci(d0.value() * maxRatio, 2)} m²/yr.`;
    }
  }

  function show(): void {
    if (loadError) {
      strip.status.textContent = loadError;
      strip.update(false, false, false, false);
      return;
    }
    if (!data) {
      strip.status.textContent = "Loading the lab terrain...";
      strip.update(false, false, false, false);
      return;
    }
    const t = data.terrain;
    const now = current ?? t.height;
    drawHeight(heightCanvas, now, t.side, t.spacingM, range[0], range[1]);
    heightLegend.replaceChildren(rampLegend(cssGradient(ELEVATION_STOPS), `${range[0].toFixed(0)} m`, "", `${range[1].toFixed(0)} m`, "Fixed scale from the start surface."));
    const delta = new Float32Array(now.length);
    let rise = 0;
    let fall = 0;
    const sums = new Map<number, [number, number]>();
    for (let i = 0; i < now.length; i++) {
      const v = now[i] - t.height[i];
      delta[i] = v;
      rise = Math.max(rise, v);
      fall = Math.min(fall, v);
      const s = sums.get(t.classes[i]) ?? [0, 0];
      s[0] += v;
      s[1] += 1;
      sums.set(t.classes[i], s);
    }
    // Colour scale from the 99th percentile of |change|, so a few steep banks do not wash out the rest.
    const sorted = Float32Array.from(delta, Math.abs).sort();
    const p99 = sorted[Math.floor(0.99 * (sorted.length - 1))];
    const r = niceRange(Math.max(p99, 1e-3));
    let clipped = 0;
    for (const v of sorted) if (v > r) clipped++;
    drawDiverging(changeCanvas, delta, t.side, r);
    changeLegend.replaceChildren(
      rampLegend(
        cssGradient(DIVERGING_STOPS),
        `-${r} m`,
        "0",
        `+${r} m`,
        current
          ? `After ${formatYears(time)}: largest rise +${rise.toFixed(2)} m, largest lowering ${fall.toFixed(2)} m. Blue lowered, red raised. ` +
            `Scale from the 99th percentile of |change|; ${formatPercent(clipped / sorted.length)} of nodes lie beyond it and show the end colours.`
          : "No change yet.",
      ),
    );
    for (const [code, cell] of changeCells) {
      const s = sums.get(code);
      cell.textContent = current && s && s[1] ? (s[0] / s[1]).toFixed(3) : "";
    }
    const f = frames[0];
    diag.replaceChildren(diagnosticsList(f?.diagnostics ?? null, f?.stopped ?? null, t.side * t.side * t.spacingM * t.spacingM));
  }

  return {
    el,
    key: () => `material|${d0.value()}|${N()}|${legend.map((e) => ratios.get(e.code) ?? 1).join(",")}`,
    specs(): RunSpec[] | null {
      if (!data) return null;
      const t = data.terrain;
      const base = d0.value();
      const field = new Float64Array(t.side * t.side);
      for (let i = 0; i < field.length; i++) field[i] = base * (ratios.get(t.classes[i]) ?? 1);
      return [
        {
          key: "terrain",
          side: t.side,
          spacingM: t.spacingM,
          initial: Float64Array.from(t.height),
          model: "linear",
          boundary: "closed",
          params: {},
          diffusivity: field,
        },
      ];
    },
    dtYears: () => N() / FRAMES,
    untilYears: N,
    stepsPerMessage: () => 1,
    showInitial: initial,
    onFrame(next) {
      frames = next;
      const f = next[0];
      if (f?.surface) current = f.surface;
      time = f?.diagnostics?.timeYears ?? time;
      show();
      strip.status.textContent = `${formatYears(time)} of ${formatYears(N())}`;
    },
    finished: () => time >= N() - 1e-6 || frames.some((f) => f.stopped !== null),
    timeYears: () => time,
    setStatus: (text) => {
      strip.status.textContent = text;
    },
    setRunning(running) {
      strip.update(running, !!data && !loadError, time > 0, time >= N() - 1e-6);
    },
  };
}
