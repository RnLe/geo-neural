// Preset 2: the nonlinear teacher, the selected conductance closure and the
// three original learned closures from the same start, stepped in lockstep.
// Shows the volume balance and the distance to the teacher.

import type { ClosureMeta, LabData } from "../data/bundle";
import { h } from "../data/dom";
import { formatSci, formatYears } from "../data/format";
import { cssGradient, ELEVATION_STOPS } from "../data/palette";
import { floorNote, linePlot, type Marker } from "./plot";
import type { RunFrame, RunSpec } from "./protocol";
import { drawHeight } from "./render";
import { controlStrip, diagnosticsList, minMax, plotWidth, rampLegend, selectField, type Preset, type RunnerApi } from "./ui";

const STEPS = 64;
const FLOOR = 1e-16;
const FLAT = "flat";

interface Arm {
  key: string;
  model: "nonlinear" | "flux" | "kfield" | "penalty" | "conductance";
  title: string;
  text: string;
  className: string;
  marker: Marker;
  dash?: string;
}

const ARMS: Arm[] = [
  { key: "teacher", model: "nonlinear", title: "Teacher", text: "Critical-slope nonlinear diffusion, the target the arms were trained on.", className: "gn-s1", marker: "circle" },
  { key: "conductance", model: "conductance", title: "Conductance arm (selected)", text: "One bounded conductance per cell face from the face slope, never below the linear diffusivity. Flat ground stays flat; each face moves material from one cell to its neighbour.", className: "gn-s5", marker: "none" },
  { key: "flux", model: "flux", title: "Flux arm", text: "Learned flux per cell face; each face moves material from one cell to its neighbour.", className: "gn-s2", marker: "square", dash: "6 3" },
  { key: "kfield", model: "kfield", title: "K-field arm", text: "Learned diffusivity per cell times the cell's Laplacian. Trained without a conservation term.", className: "gn-s3", marker: "triangle", dash: "2 3" },
  { key: "penalty", model: "penalty", title: "Penalty arm", text: "Same K-field form, trained with a conservation penalty in the loss.", className: "gn-s4", marker: "diamond", dash: "8 3 2 3" },
];

/** The narrowest training range over the shown arms: every panel accepts a state inside it. */
function shownRange(meta: ClosureMeta): ClosureMeta["validated"] {
  const ranges = [meta.validated, ...ARMS.map((arm) => meta.arms[arm.model]?.validated).filter((v) => v !== undefined)];
  return {
    maxSlope: Math.min(...ranges.map((v) => v.maxSlope)),
    minHeightM: Math.max(...ranges.map((v) => v.minHeightM)),
    maxHeightM: Math.min(...ranges.map((v) => v.maxHeightM)),
  };
}

export function closurePreset(api: RunnerApi, lab: Promise<LabData>): Preset {
  let data: LabData | null = null;
  let loadError: string | null = null;

  const strip = controlStrip(api);
  const surfaceSel = selectField<string>("Start surface", [["0", "surface 1"]], "0");
  const boundarySel = selectField<"closed" | "fixed">(
    "Boundary",
    [
      ["closed", "closed: nothing crosses the edge"],
      ["fixed", "fixed: edge ring held at its start height"],
    ],
    "closed",
  );
  surfaceSel.select.addEventListener("change", () => api.paramsChanged());
  boundarySel.select.addEventListener("change", () => api.paramsChanged());

  const cards = new Map<string, { canvas: HTMLCanvasElement; diag: HTMLElement }>();
  const grid = h("div", { class: "gn-lab-grid gn-lab-grid-4" });
  let flat: Float32Array | null = null;
  for (const arm of ARMS) {
    const canvas = h("canvas", { class: "gn-lab-canvas", role: "img", "aria-label": `${arm.title} surface` });
    const diag = h("div", { class: "gn-lab-diag" });
    cards.set(arm.key, { canvas, diag });
    grid.append(
      h(
        "figure",
        { class: "gn-lab-card" },
        h("figcaption", { class: "gn-legend-title" }, h("span", { class: `gn-series-key ${arm.className}`, "aria-hidden": "true" }), arm.title),
        canvas,
        h("p", { class: "gn-note" }, arm.text),
        diag,
      ),
    );
  }
  const legendHost = h("div", {});
  const balanceHost = h("div", { class: "gn-lab-plot" });
  const errorHost = h("div", { class: "gn-lab-plot" });
  const rangeNote = h("p", { class: "gn-note" });

  const el = h(
    "div",
    { class: "gn-lab-preset" },
    h(
      "p",
      { class: "gn-lab-intro" },
      "Five models step the same start surface in lockstep: 64 steps of 200 years (12,800 years). The kernel splits a step into substeps below each model's " +
        "stability bound; the substep count is listed per panel. The conductance arm is the design the closure study selected: one conductance per cell face, " +
        "a = D + (aMax - D) sigmoid(NN(|g|)), applied as the flux a g. A flat surface is exactly still, a constant offset changes nothing, and each interior face " +
        "adds to one cell exactly what it takes from the next, so the integral can only change through the boundary. The flux arm also conserves, but it reads " +
        "absolute height and moves flat ground. The K-field arms multiply a per-cell diffusivity by the cell's Laplacian; neighbours see different K across the " +
        "same face, the terms no longer cancel, and a loss penalty shrinks the leak without closing it. Start from flat ground to see which models keep it flat.",
    ),
    h("div", { class: "gn-controls" }, surfaceSel.el, boundarySel.el),
    strip.el,
    rangeNote,
    legendHost,
    grid,
    h(
      "div",
      { class: "gn-lab-grid" },
      h("figure", { class: "gn-lab-card gn-lab-card-wide" }, h("figcaption", { class: "gn-legend-title" }, "Volume balance: |residual| / sum |h| a"), balanceHost),
      h("figure", { class: "gn-lab-card gn-lab-card-wide" }, h("figcaption", { class: "gn-legend-title" }, "Distance to the teacher (RMS height difference)"), errorHost),
    ),
    h(
      "p",
      { class: "gn-note" },
      "Residual = integral now - integral at start - boundary exchange. It is numerical volume balance on a synthetic surface, not sediment mass. " +
        "The start surfaces are fresh draws from the training distribution, not seen in training; flat ground is 0 m everywhere.",
    ),
  );

  let balance = new Map<string, [number, number][]>();
  let distance = new Map<string, [number, number][]>();
  let frames: RunFrame[] = [];
  let time = 0;
  let range: [number, number] = [0, 1];

  lab.then(
    (d) => {
      data = d;
      if (!d.closure || !d.closureText || !d.surfaces.length) {
        loadError = "This bundle has no learned closure.";
      } else {
        surfaceSel.select.replaceChildren(
          ...d.surfaces.map((_, i) => h("option", { value: String(i) }, `surface ${i + 1} of ${d.surfaces.length}`)),
          h("option", { value: FLAT }, "flat ground"),
        );
        const v = shownRange(d.closure);
        rangeNote.textContent =
          `The learned arms accept only states inside their training range (heights ${v.minHeightM.toFixed(0)} to ${v.maxHeightM.toFixed(0)} m, slope up to ${v.maxSlope.toFixed(2)}); ` +
          `the kernel refuses a step outside it and the panel says so. Grid ${d.surfaceSide} x ${d.surfaceSide} at ${d.surfaceSpacingM} m, teacher D = ${d.closure.teacher.diffusivity} m²/yr, critical slope ${d.closure.teacher.criticalSlope}.`;
      }
      // Nothing can have run before the inputs arrived.
      initial();
      if (!loadError) strip.update(false, true, false, false);
    },
    (err: unknown) => {
      loadError = `Could not load the lab inputs: ${err instanceof Error ? err.message : String(err)}`;
      show();
    },
  );

  function start(): Float32Array | null {
    if (!data || !data.surfaces.length) return null;
    if (surfaceSel.value() === FLAT) return (flat ??= new Float32Array(data.surfaceSide * data.surfaceSide));
    return data.surfaces[Number(surfaceSel.value()) || 0];
  }

  function drawCharts(): void {
    const T = STEPS * 200;
    const bal = ARMS.map((arm) => ({
      label: arm.title,
      points: (balance.get(arm.key) ?? []).filter(([t]) => t > 0),
      className: arm.className,
      marker: arm.marker,
      dash: arm.dash,
    }));
    let top = 1e-12;
    for (const s of bal) for (const [, v] of s.points) top = Math.max(top, v);
    balanceHost.replaceChildren(
      linePlot({
        title: "Relative volume residual over time, log axis with a floor",
        width: plotWidth(balanceHost),
        height: 240,
        x: { min: 0, max: T, label: "time (thousand years)", format: (v) => String(v / 1000) },
        y: {
          min: FLOOR,
          max: Math.pow(10, Math.ceil(Math.log10(top * 3))),
          log: true,
          floor: FLOOR,
          label: "|residual| relative (log)",
        },
        series: bal,
      }),
      h("p", { class: "gn-note" }, floorNote(FLOOR)),
    );
    const err = ARMS.filter((a) => a.key !== "teacher").map((arm) => ({
      label: arm.title,
      points: distance.get(arm.key) ?? [],
      className: arm.className,
      marker: arm.marker,
      dash: arm.dash,
    }));
    let errTop = 1;
    for (const s of err) for (const [, v] of s.points) errTop = Math.max(errTop, v);
    errorHost.replaceChildren(
      linePlot({
        title: "RMS height difference to the teacher over time",
        width: plotWidth(errorHost),
        height: 240,
        x: { min: 0, max: T, label: "time (thousand years)", format: (v) => String(v / 1000) },
        y: { min: 0, max: Math.ceil(errTop * 1.1), label: "RMS difference (m)" },
        series: err,
      }),
    );
  }

  function initial(): void {
    balance = new Map();
    distance = new Map();
    frames = [];
    time = 0;
    const s0 = start();
    if (s0) {
      const [lo, hi] = minMax(s0);
      // A flat start gets a few metres either side, so any movement shows.
      range = hi > lo ? [lo, hi] : [lo - 2, hi + 2];
    }
    show();
    if (data && s0 && !loadError) strip.status.textContent = `Ready: ${STEPS} steps of 200 years. The learned arms take roughly 50 to 150 ms per substep here.`;
  }

  function show(): void {
    const s0 = start();
    if (loadError) {
      strip.status.textContent = loadError;
      strip.update(false, false, false, false);
      return;
    }
    if (!data || !s0) {
      strip.status.textContent = "Loading lab inputs (closure weights and surfaces, about 450 kB)...";
      strip.update(false, false, false, false);
      return;
    }
    const side = data.surfaceSide;
    for (const arm of ARMS) {
      const card = cards.get(arm.key);
      if (!card) continue;
      const f = frames.find((x) => x.key === arm.key);
      drawHeight(card.canvas, f?.surface ?? s0, side, data.surfaceSpacingM, range[0], range[1]);
      card.diag.replaceChildren(diagnosticsList(f?.diagnostics ?? null, f?.stopped ?? null));
    }
    legendHost.replaceChildren(
      rampLegend(cssGradient(ELEVATION_STOPS), `${range[0].toFixed(0)} m`, "", `${range[1].toFixed(0)} m`, "Height, one fixed scale for all five panels, from the start surface."),
    );
    drawCharts();
  }

  return {
    el,
    key: () => `closure|${surfaceSel.value()}|${boundarySel.value()}`,
    specs(): RunSpec[] | null {
      const s0 = start();
      if (!data || !s0 || !data.closure || !data.closureText) return null;
      const initial = Float64Array.from(s0);
      const { teacher } = data.closure;
      const meta = data.closureText;
      const weights = data.weights;
      return ARMS.map((arm): RunSpec => {
        const spec: RunSpec = {
          key: arm.key,
          side: data!.surfaceSide,
          spacingM: data!.surfaceSpacingM,
          initial,
          model: arm.model,
          boundary: boundarySel.value(),
          params: {},
        };
        if (arm.model === "nonlinear") spec.params = { diffusivity: teacher.diffusivity, criticalSlope: teacher.criticalSlope };
        else {
          spec.weights = weights[arm.model];
          spec.weightsMeta = meta;
        }
        return spec;
      });
    },
    dtYears: () => 200,
    untilYears: () => STEPS * 200,
    stepsPerMessage: () => 1,
    showInitial: initial,
    onFrame(next, elapsedMs) {
      frames = next;
      const teacher = next.find((f) => f.key === "teacher");
      for (const f of next) {
        const d = f.diagnostics;
        if (!d) continue;
        const list = balance.get(f.key) ?? [];
        // Exact zero is drawn on the floor line, never at an invented value.
        if (!list.length || list[list.length - 1][0] !== d.timeYears) list.push([d.timeYears, Math.abs(d.residualRelative)]);
        balance.set(f.key, list);
        if (f.key !== "teacher" && f.surface && teacher?.surface && teacher.diagnostics) {
          let sum = 0;
          for (let i = 0; i < f.surface.length; i++) sum += (f.surface[i] - teacher.surface[i]) ** 2;
          const dl = distance.get(f.key) ?? [];
          if (!dl.length || dl[dl.length - 1][0] !== d.timeYears) dl.push([d.timeYears, Math.sqrt(sum / f.surface.length)]);
          distance.set(f.key, dl);
        }
      }
      time = Math.min(...next.filter((f) => f.diagnostics && !f.stopped).map((f) => f.diagnostics!.timeYears), STEPS * 200);
      if (!Number.isFinite(time)) time = STEPS * 200;
      show();
      const step = Math.round(time / 200);
      const flux = next.find((f) => f.key === "flux")?.diagnostics;
      strip.status.textContent =
        `Step ${step} of ${STEPS} (${formatYears(time)}), ${elapsedMs.toFixed(0)} ms for the last message.` +
        (flux ? ` Flux arm residual ${formatSci(flux.residualM3, 2)} m³.` : "");
    },
    finished: () => time >= STEPS * 200 - 1e-6 || (frames.length > 0 && frames.every((f) => f.stopped !== null)),
    timeYears: () => time,
    setStatus: (text) => {
      strip.status.textContent = text;
    },
    setRunning(running) {
      const ready = !!data && !loadError;
      strip.update(running, ready, time > 0, time >= STEPS * 200 - 1e-6);
    },
  };
}
