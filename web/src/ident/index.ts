// Rate and age: what one or two surveys say about the common rate factor c.
// Precomputed expected log-likelihood profiles (geoneural export-ident); nothing is simulated here.

import { h } from "../data/dom";
import { linePlot, type Series } from "../lab/plot";

interface Design { delta: number[]; interval: [number, number] | null }
interface Variant { c: number[]; designs: Record<string, Design> }
interface Regime { years: number; reliefM: number; variants: Record<string, Variant> }
export interface IdentData { schema: string; lagsYears: number[]; regimes: Record<string, Regime> }

const DESIGNS: { key: string; label: string; short: string; className: string; dash?: string }[] = [
  { key: "terminal", label: "one survey", short: "one survey", className: "gn-s2", dash: "6 3" },
  { key: "fractional", label: "two surveys at fixed fractions of the age", short: "fractions of age", className: "gn-s3", dash: "2 3" },
  { key: "fixed-lag-20k", label: "two surveys 20,000 years apart", short: "20 kyr apart", className: "gn-s4" },
  { key: "fixed-lag-50k", label: "two surveys 50,000 years apart", short: "50 kyr apart", className: "gn-s5" },
];
const REGIMES: [string, string][] = [["transient", "young"], ["late", "older"], ["near-equilibrium", "near steady state"]];
const FLOOR = -40;

/** The range of c where the profile stays within `drop` log-units of its maximum, by linear interpolation on the
 *  grid; null ends mean the range reaches the grid edge. */
export function interval(c: number[], delta: number[], drop = 2): [number | null, number | null] {
  const top = Math.max(...delta);
  const inside = delta.map((d) => d >= top - drop);
  const first = inside.indexOf(true);
  const last = inside.lastIndexOf(true);
  const cross = (i: number, j: number) => {
    const t = (top - drop - delta[i]) / (delta[j] - delta[i]);
    return Math.exp(Math.log(c[i]) + t * (Math.log(c[j]) - Math.log(c[i])));
  };
  return [first > 0 ? cross(first - 1, first) : null, last < c.length - 1 ? cross(last + 1, last) : null];
}

export function mountIdent(el: HTMLElement, opts: { url: string }): { dispose(): void } {
  const root = h("div", { class: "gn-root gn-ident" });
  el.append(root);
  let alive = true;
  let regime = "transient";
  let variant = "rescaledDt";
  const plotHost = h("div", { class: "gn-ident-plot" });
  const readout = h("ul", { class: "gn-ident-readout" });
  const button = (label: string, pressed: boolean, on: () => void) => {
    const b = h("button", { type: "button", class: "gn-button", "aria-pressed": String(pressed) }, label);
    b.addEventListener("click", on);
    return b;
  };
  void fetch(opts.url).then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json() as Promise<IdentData>;
  }).then((data) => {
    if (!alive) return;
    const controls = h("div", { class: "gn-controls" });
    const draw = () => {
      const v = data.regimes[regime].variants[variant];
      const series: Series[] = DESIGNS.filter((d) => v.designs[d.key]).map((d) => ({
        label: d.short, className: d.className, dash: d.dash, marker: "none",
        points: v.c.map((c, i) => [c, Math.max(FLOOR, v.designs[d.key].delta[i])] as [number, number]),
      }));
      plotHost.replaceChildren(linePlot({
        title: "How well each common rate factor c fits the surveys", width: 640, height: 300,
        x: { min: 0.5, max: 2, label: "rates times c, time divided by c", log: true },
        y: { min: FLOOR, max: 2, label: "fit, log-likelihood below the best" },
        series,
      }));
      readout.replaceChildren(...DESIGNS.filter((d) => v.designs[d.key]).map((d) => {
        const [lo, hi] = interval(v.c, v.designs[d.key].delta);
        const text = lo === null && hi === null ? "every c from 0.5 to 2 fits: c is not identified"
          : `c between ${lo === null ? "below 0.5" : lo.toFixed(2)} and ${hi === null ? "above 2" : hi.toFixed(2)}`;
        return h("li", {}, `${d.label}: ${text}`);
      }));
      controls.replaceChildren(
        h("div", { class: "gn-inline", role: "group", "aria-label": "Landscape age" },
          ...REGIMES.map(([key, label]) => button(`${label} (${(data.regimes[key].years / 1e6).toFixed(1)} Myr)`,
            key === regime, () => { regime = key; draw(); }))),
        h("div", { class: "gn-inline", role: "group", "aria-label": "Time steps" },
          button("time steps scaled with c", variant === "rescaledDt", () => { variant = "rescaledDt"; draw(); }),
          button("fixed time-step cap", variant === "fixedDtCap", () => { variant = "fixedDtCap"; draw(); })),
      );
    };
    root.append(controls, plotHost, readout,
      h("p", { class: "gn-note" }, "Expected profiles from the simulation, without noise. With a fixed cap on the time step the curves look informative even for one survey: that information is an artifact of the numerics."));
    draw();
  }).catch((err: unknown) => {
    if (alive) root.append(h("p", { class: "gn-message" }, `The identifiability data could not be loaded: ${err instanceof Error ? err.message : String(err)}`));
  });
  return { dispose() { alive = false; root.remove(); } };
}
