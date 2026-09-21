// Preset 1: linear diffusion of a synthetic ridge with small bumps, and a
// sine mode whose amplitude decay is compared with exp(-D k^2 t).

import { h } from "../data/dom";
import { formatInt, formatSci, formatYears } from "../data/format";
import { cssGradient, ELEVATION_STOPS } from "../data/palette";
import { floorNote, linePlot } from "./plot";
import type { RunFrame, RunSpec } from "./protocol";
import { drawHeight } from "./render";
import { modeAmplitude, ridgeSurface, sineMode } from "./surfaces";
import { controlStrip, diagnosticsList, logSlider, minMax, plotWidth, rampLegend, selectField, type Preset, type RunnerApi } from "./ui";

const SIDE = 96;
const SPACING = 50;
const FRAMES = 100;
const MODE = 16;
const MODE_AMPLITUDE = 5;

export function ridgePreset(api: RunnerApi): Preset {
  const initial = ridgeSurface(SIDE, SPACING);
  const [lo, hi] = minMax(initial);
  const sine = sineMode(SIDE, SPACING, MODE, MODE_AMPLITUDE);
  const wavelength = (2 * Math.PI) / sine.k;

  const diff = logSlider("Diffusivity D", 0.01, 1, 0.1, "m²/yr", 2);
  const duration = selectField(
    "Run length",
    [
      ["20000", "20,000 years"],
      ["100000", "100,000 years"],
      ["500000", "500,000 years"],
    ],
    "100000",
  );
  const strip = controlStrip(api);
  diff.input.addEventListener("change", () => api.paramsChanged());
  duration.select.addEventListener("change", () => api.paramsChanged());

  const canvas = h("canvas", { class: "gn-lab-canvas", width: SIDE, height: SIDE, role: "img", "aria-label": "Ridge surface, height colour with hillshade" });
  const diag = h("div", { class: "gn-lab-diag" });
  const profileHost = h("div", { class: "gn-lab-plot" });
  const modeHost = h("div", { class: "gn-lab-plot" });
  const modeText = h("p", { class: "gn-note" });

  const el = h(
    "div",
    { class: "gn-lab-preset" },
    h(
      "p",
      { class: "gn-lab-intro" },
      `Linear diffusion, dh/dt = D times the Laplacian of h, on a synthetic ${SIDE} x ${SIDE} grid at ${SPACING} m with closed boundaries (no flux across the edge). ` +
        "Short features decay faster than long ones: a wave of wavenumber k decays as exp(-D k² t), so the bumps vanish long before the ridge flattens.",
    ),
    h("div", { class: "gn-controls" }, diff.el, duration.el),
    strip.el,
    h(
      "div",
      { class: "gn-lab-grid" },
      h(
        "figure",
        { class: "gn-lab-card" },
        canvas,
        rampLegend(cssGradient(ELEVATION_STOPS), `${lo.toFixed(0)} m`, "", `${hi.toFixed(0)} m`, "Height, fixed scale from the start state. Sun from the north-west."),
        h("figcaption", { class: "gn-legend-title" }, "Ridge surface"),
        diag,
      ),
      h("figure", { class: "gn-lab-card gn-lab-card-wide" }, h("figcaption", { class: "gn-legend-title" }, "Middle row, west to east"), profileHost),
      h(
        "figure",
        { class: "gn-lab-card gn-lab-card-wide" },
        h("figcaption", { class: "gn-legend-title" }, `Sine mode: wavelength ${formatInt(wavelength)} m, same D and grid`),
        modeHost,
        modeText,
      ),
    ),
  );

  let times: number[] = [];
  let amps: number[] = [];
  let current: Float32Array | null = null;
  let time = 0;
  let lastFrames: RunFrame[] = [];

  const D = () => diff.value();
  const until = () => Number(duration.value());

  function drawProfile(): void {
    const row = SIDE >> 1;
    const start: [number, number][] = [];
    const now: [number, number][] = [];
    for (let c = 0; c < SIDE; c++) {
      const x = (c + 0.5) * SPACING;
      start.push([x, initial[row * SIDE + c]]);
      if (current) now.push([x, current[row * SIDE + c]]);
    }
    profileHost.replaceChildren(
      linePlot({
        title: "Height along the middle row at the start and now",
        width: plotWidth(profileHost),
        height: 200,
        x: { min: 0, max: SIDE * SPACING, label: "distance east (km)", format: (v) => String(v / 1000) },
        y: { min: Math.floor(lo / 10) * 10, max: Math.ceil(hi / 10) * 10, label: "height (m)" },
        series: [
          { label: "start", points: start, className: "gn-s2", dash: "5 4", marker: "none" },
          { label: "now", points: now, className: "gn-s1", marker: "none" },
        ],
      }),
    );
  }

  function drawMode(): void {
    const d = D();
    const k2 = sine.k * sine.k;
    const T = until();
    const analytic: [number, number][] = [];
    for (let i = 0; i <= 100; i++) {
      const t = (T * i) / 100;
      analytic.push([t, Math.exp(-d * k2 * t)]);
    }
    const measured = times.map((t, i) => [t, amps[i] / MODE_AMPLITUDE] as [number, number]);
    const floor = 1e-8;
    modeHost.replaceChildren(
      linePlot({
        title: "Sine mode amplitude over time: measured and exp(-D k^2 t)",
        width: plotWidth(modeHost),
        height: 220,
        x: { min: 0, max: T, label: "time (thousand years)", format: (v) => String(v / 1000) },
        y: { min: floor, max: 1.5, log: true, floor, label: "amplitude / start (log)" },
        series: [
          { label: "exp(-D k² t)", points: analytic, className: "gn-s2", dash: "6 4", marker: "none" },
          { label: "measured", points: measured, className: "gn-s1", marker: "circle", markersOnly: true },
        ],
      }),
    );
    modeHost.append(h("p", { class: "gn-note" }, floorNote(floor)));
    const n = times.length;
    if (n > 1 && amps[n - 1] > 0 && times[n - 1] > 0) {
      const rate = -Math.log(amps[n - 1] / MODE_AMPLITUDE) / times[n - 1];
      const grid = (4 / (SPACING * SPACING)) * Math.sin((sine.k * SPACING) / 2) ** 2;
      modeText.textContent =
        `At ${formatYears(times[n - 1])}: measured ${formatSci(amps[n - 1] / MODE_AMPLITUDE, 3)}, exp(-D k² t) ${formatSci(Math.exp(-d * k2 * times[n - 1]), 3)}. ` +
        `Measured decay rate / D k² = ${(rate / (d * k2)).toFixed(3)}. The grid's own rate is ${(grid / k2).toFixed(3)} D k² ` +
        "(the 5-point stencil sees sin² instead of k²); explicit time steps push the other way.";
    } else {
      modeText.textContent = `k = ${formatSci(sine.k, 3)} per m. Press Run to measure the decay.`;
    }
  }

  function draw(): void {
    drawHeight(canvas, current ?? initial, SIDE, SPACING, lo, hi);
    const ridge = lastFrames.find((f) => f.key === "ridge");
    diag.replaceChildren(diagnosticsList(ridge?.diagnostics ?? null, ridge?.stopped ?? null, SIDE * SIDE * SPACING * SPACING));
    drawProfile();
    drawMode();
  }

  return {
    el,
    key: () => `ridge|${D()}|${until()}`,
    specs(): RunSpec[] {
      const params = { diffusivity: D() };
      return [
        { key: "ridge", side: SIDE, spacingM: SPACING, initial, model: "linear", boundary: "closed", params },
        { key: "mode", side: SIDE, spacingM: SPACING, initial: sine.surface, model: "linear", boundary: "closed", params },
      ];
    },
    dtYears: () => until() / FRAMES,
    untilYears: until,
    stepsPerMessage: () => 1,
    showInitial() {
      times = [];
      amps = [];
      current = null;
      time = 0;
      lastFrames = [];
      draw();
      strip.status.textContent = `Ready: ${FRAMES} steps of ${formatYears(until() / FRAMES)}.`;
    },
    onFrame(frames) {
      lastFrames = frames;
      const ridge = frames.find((f) => f.key === "ridge");
      const mode = frames.find((f) => f.key === "mode");
      if (ridge?.surface) current = ridge.surface;
      time = ridge?.diagnostics?.timeYears ?? time;
      if (mode?.surface && mode.diagnostics) {
        times.push(mode.diagnostics.timeYears);
        amps.push(modeAmplitude(mode.surface, SIDE, sine.basis));
      }
      draw();
      strip.status.textContent = `${formatYears(time)} of ${formatYears(until())}`;
    },
    finished: () => time >= until() - 1e-6 || lastFrames.some((f) => f.stopped !== null),
    timeYears: () => time,
    setStatus: (text) => {
      strip.status.textContent = text;
    },
    setRunning(running) {
      strip.update(running, true, time > 0, time >= until() - 1e-6);
    },
  };
}
