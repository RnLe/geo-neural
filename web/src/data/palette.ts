// Colour ramps shared by the viewer and the lab. Maps keep the same colours in
// light and dark page themes; only the page chrome follows the theme.

/** Sequential, one warm hue, light (low) to dark (high). */
export const ELEVATION_STOPS = ["#f5f0e3", "#e6d8b6", "#d2b986", "#b6955e", "#8f6e40", "#634a2a"];

/** Diverging: blue (negative) to grey (zero) to red (positive). */
export const DIVERGING_STOPS = ["#184f95", "#3987e5", "#9ec5f4", "#ebe9e4", "#f2b1a7", "#df5f57", "#9e1f1e"];

/** Stream classes: both, only reference (lost), only candidate (spurious). */
export const STREAM_COLORS = { both: "#1c3f7a", lost: "#e2601f", spurious: "#c43f86" } as const;

export const NEUTRAL_MAP = "#e4e1d9";

export function hexToRgb(hex: string): [number, number, number, number] {
  const h = hex.replace("#", "");
  const n = h.length === 3 || h.length === 4 ? h.split("").map((c) => c + c).join("") : h;
  const v = (i: number) => parseInt(n.slice(i, i + 2), 16);
  return [v(0), v(2), v(4), n.length >= 8 ? v(6) : 255];
}

/** n RGBA entries interpolated linearly between evenly spaced stops. */
export function buildLut(stops: readonly string[], n = 256): Uint8Array {
  const rgb = stops.map(hexToRgb);
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * (rgb.length - 1);
    const k = Math.min(rgb.length - 2, Math.floor(t));
    const f = t - k;
    for (let c = 0; c < 3; c++) out[4 * i + c] = Math.round(rgb[k][c] * (1 - f) + rgb[k + 1][c] * f);
    out[4 * i + 3] = 255;
  }
  return out;
}

/** CSS linear-gradient for a legend bar. */
export function cssGradient(stops: readonly string[], direction = "to right"): string {
  return `linear-gradient(${direction}, ${stops.join(", ")})`;
}
