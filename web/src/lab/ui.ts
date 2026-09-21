// Shared pieces of the lab presets: run controls, diagnostics, sliders.

import { h, uid } from "../data/dom";
import { formatSci, formatYears } from "../data/format";
import type { Diagnostics, RunFrame, RunSpec } from "./protocol";

export interface RunnerApi {
  run(): void;
  pause(): void;
  reset(): void;
  /** Parameters changed: drop live scenarios and show the start state. */
  paramsChanged(): void;
}

export interface Preset {
  readonly el: HTMLElement;
  /** Changes whenever a parameter that needs new scenarios changes. */
  key(): string;
  /** Null while inputs are loading or unavailable. */
  specs(): RunSpec[] | null;
  dtYears(): number;
  untilYears(): number;
  stepsPerMessage(): number;
  /** Clears history and draws the start state. */
  showInitial(): void;
  onFrame(frames: RunFrame[], elapsedMs: number): void;
  finished(): boolean;
  timeYears(): number;
  setStatus(text: string): void;
  setRunning(running: boolean): void;
}

export interface Strip {
  el: HTMLElement;
  status: HTMLElement;
  update(running: boolean, ready: boolean, started: boolean, finished: boolean): void;
}

export function controlStrip(api: RunnerApi): Strip {
  const run = h("button", { type: "button", class: "gn-button gn-button-primary" }, "Run");
  const pause = h("button", { type: "button", class: "gn-button", disabled: true }, "Pause");
  const reset = h("button", { type: "button", class: "gn-button" }, "Reset");
  const status = h("p", { class: "gn-lab-status", role: "status" });
  run.addEventListener("click", () => api.run());
  pause.addEventListener("click", () => api.pause());
  reset.addEventListener("click", () => api.reset());
  const el = h("div", { class: "gn-lab-strip" }, h("div", { class: "gn-inline" }, run, pause, reset), status);
  return {
    el,
    status,
    update(running, ready, started, finished) {
      run.disabled = running || !ready;
      pause.disabled = !running;
      run.textContent = finished ? "Run again" : started && !running ? "Continue" : "Run";
    },
  };
}

export function diagnosticsList(d: Diagnostics | null, stopped: string | null, area?: number): HTMLElement {
  if (!d) return h("p", { class: "gn-note" }, stopped ? `Not running: ${stopped}` : "No state yet.");
  const change = d.integralM3 - d.initialIntegralM3;
  const rows: [string, string][] = [
    ["Time", formatYears(d.timeYears)],
    ["Integral", `${formatSci(d.integralM3, 6)} m³`],
    ["Change in integral", `${formatSci(change, 3)} m³`],
    ["Boundary exchange", `${formatSci(d.boundaryExchangeM3, 3)} m³`],
    ["Residual", `${formatSci(d.residualM3, 3)} m³`],
    ["Residual, relative", formatSci(d.residualRelative, 2)],
    ["Substeps, last call", `${d.substeps} (stable step ${Number.isFinite(d.stableDtYears) ? formatYears(d.stableDtYears) : "none"})`],
    ["Max slope", d.maxSlope.toFixed(3)],
  ];
  if (area) rows.splice(3, 0, ["Mean height change", `${formatSci(change / area, 3)} m`]);
  return h(
    "div",
    {},
    h("dl", { class: "gn-facts gn-facts-compact" }, ...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])),
    stopped ? h("p", { class: "gn-lab-stopped" }, `Stopped: ${stopped}`) : null,
  );
}

/** A range input on a log scale. */
export function logSlider(label: string, min: number, max: number, value: number, unit: string, digits: number) {
  const id = uid("gn-slider");
  const steps = 200;
  const toValue = (pos: number) => Number((min * Math.pow(max / min, pos / steps)).toPrecision(digits));
  const toPos = (v: number) => Math.round((Math.log(v / min) / Math.log(max / min)) * steps);
  const input = h("input", { type: "range", id, min: 0, max: steps, step: 1, value: toPos(value), class: "gn-range" });
  const out = h("output", { for: id, class: "gn-output" });
  const show = () => {
    out.textContent = `${toValue(Number(input.value))} ${unit}`;
  };
  show();
  input.addEventListener("input", show);
  const el = h("div", { class: "gn-field" }, h("label", { for: id }, label), h("div", { class: "gn-inline" }, input, out));
  return { el, input, value: () => toValue(Number(input.value)) };
}

export function selectField<T extends string>(label: string, options: [T, string][], value: T) {
  const id = uid("gn-select");
  const select = h("select", { id, class: "gn-select" });
  for (const [v, text] of options) select.append(h("option", { value: v }, text));
  select.value = value;
  const el = h("div", { class: "gn-field" }, h("label", { for: id }, label), select);
  return { el, select, value: () => select.value as T };
}

export function minMax(data: ArrayLike<number>): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < data.length; i++) {
    if (data[i] < lo) lo = data[i];
    if (data[i] > hi) hi = data[i];
  }
  return [lo, hi];
}

export function plotWidth(host: HTMLElement, fallback = 420): number {
  return Math.max(260, Math.min(720, host.clientWidth || fallback));
}

/** Colour ramp legend for a canvas. */
export function rampLegend(gradient: string, left: string, mid: string, right: string, caption: string): HTMLElement {
  return h(
    "div",
    { class: "gn-ramp gn-ramp-small" },
    h("div", { class: "gn-ramp-bar", style: `background:${gradient}`, "aria-hidden": "true" }),
    h("div", { class: "gn-ramp-ends" }, h("span", {}, left), h("span", {}, mid), h("span", {}, right)),
    h("p", { class: "gn-note" }, caption),
  );
}
