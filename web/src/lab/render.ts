// Canvas rendering for lab surfaces: height colour with a light hillshade,
// a diverging map for changes, and categorical classes.

import { buildLut, DIVERGING_STOPS, ELEVATION_STOPS, hexToRgb, NEUTRAL_MAP } from "../data/palette";

const ELEVATION = buildLut(ELEVATION_STOPS);
const DIVERGING = buildLut(DIVERGING_STOPS);

function context(canvas: HTMLCanvasElement, side: number): CanvasRenderingContext2D | null {
  if (canvas.width !== side) canvas.width = side;
  if (canvas.height !== side) canvas.height = side;
  return canvas.getContext("2d");
}

function lutIndex(t: number): number {
  return Math.max(0, Math.min(255, Math.round(t * 255))) * 4;
}

/** Height colour on a fixed [min, max] range, shaded from the north-west. */
export function drawHeight(canvas: HTMLCanvasElement, data: ArrayLike<number>, side: number, spacingM: number, min: number, max: number): void {
  const ctx = context(canvas, side);
  if (!ctx) return;
  const img = ctx.createImageData(side, side);
  const span = max - min || 1;
  const sun = Math.SQRT1_2;
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      const i = r * side + c;
      const e = data[r * side + Math.min(side - 1, c + 1)];
      const w = data[r * side + Math.max(0, c - 1)];
      const s = data[Math.min(side - 1, r + 1) * side + c];
      const n = data[Math.max(0, r - 1) * side + c];
      const gx = (e - w) / (2 * spacingM);
      const gy = (s - n) / (2 * spacingM);
      // Normal (-gx, 1, -gy) against a sun from the north-west at 45 degrees.
      const len = Math.hypot(gx, 1, gy);
      const dot = ((gx * 0.5 + gy * 0.5) + sun) / len;
      const shade = Math.max(0.55, Math.min(1.15, 1 + 1.0 * (dot - sun)));
      const k = lutIndex((data[i] - min) / span);
      img.data[4 * i] = Math.min(255, ELEVATION[k] * shade);
      img.data[4 * i + 1] = Math.min(255, ELEVATION[k + 1] * shade);
      img.data[4 * i + 2] = Math.min(255, ELEVATION[k + 2] * shade);
      img.data[4 * i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

/** Diverging colour for a signed field on [-range, range], clipped at the ends. */
export function drawDiverging(canvas: HTMLCanvasElement, data: ArrayLike<number>, side: number, range: number): void {
  const ctx = context(canvas, side);
  if (!ctx) return;
  const img = ctx.createImageData(side, side);
  const r = range || 1;
  for (let i = 0; i < side * side; i++) {
    const k = lutIndex(0.5 + 0.5 * Math.max(-1, Math.min(1, data[i] / r)));
    img.data[4 * i] = DIVERGING[k];
    img.data[4 * i + 1] = DIVERGING[k + 1];
    img.data[4 * i + 2] = DIVERGING[k + 2];
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

export function drawClasses(canvas: HTMLCanvasElement, classes: Uint8Array, side: number, colors: Map<number, string>): void {
  const ctx = context(canvas, side);
  if (!ctx) return;
  const img = ctx.createImageData(side, side);
  const neutral = hexToRgb(NEUTRAL_MAP);
  const cache = new Map<number, [number, number, number, number]>();
  for (let i = 0; i < side * side; i++) {
    let rgb = cache.get(classes[i]);
    if (!rgb) {
      const hex = colors.get(classes[i]);
      const v = hex ? hexToRgb(hex) : neutral;
      rgb = v[3] === 0 ? neutral : v;
      cache.set(classes[i], rgb);
    }
    img.data[4 * i] = rgb[0];
    img.data[4 * i + 1] = rgb[1];
    img.data[4 * i + 2] = rgb[2];
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** A round upper bound for a symmetric colour range. */
export function niceRange(v: number): number {
  if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}
