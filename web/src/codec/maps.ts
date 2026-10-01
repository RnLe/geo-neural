// Canvas rendering for the microscope: error codes on the diverging ramp, stream classes over a grey
// hillshade of the reference, and magnified crops.

import { buildLut, DIVERGING_STOPS, hexToRgb, STREAM_COLORS } from "../data/palette";

const DIVERGING = buildLut(DIVERGING_STOPS);
const BOTH = hexToRgb(STREAM_COLORS.both);
const LOST = hexToRgb(STREAM_COLORS.lost);
const SPURIOUS = hexToRgb(STREAM_COLORS.spurious);

function sized(canvas: HTMLCanvasElement, side: number): CanvasRenderingContext2D | null {
  if (canvas.width !== side) canvas.width = side;
  if (canvas.height !== side) canvas.height = side;
  return canvas.getContext("2d");
}

/** u8 error codes (128 is zero, 1 and 255 the scale) on the diverging ramp. */
export function drawErrorCodes(canvas: HTMLCanvasElement, codes: Uint8Array, side: number): void {
  const ctx = sized(canvas, side);
  if (!ctx) return;
  const img = ctx.createImageData(side, side);
  for (let i = 0; i < side * side; i++) {
    const k = Math.round((Math.max(1, codes[i]) - 1) * (255 / 254)) * 4;
    img.data[4 * i] = DIVERGING[k];
    img.data[4 * i + 1] = DIVERGING[k + 1];
    img.data[4 * i + 2] = DIVERGING[k + 2];
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** Grey hillshade (0..255) lit from the north-west. */
export function hillshade(height: Float32Array, side: number, spacingM: number): Uint8Array {
  const out = new Uint8Array(side * side);
  const sun = Math.SQRT1_2;
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      const e = height[r * side + Math.min(side - 1, c + 1)];
      const w = height[r * side + Math.max(0, c - 1)];
      const s = height[Math.min(side - 1, r + 1) * side + c];
      const n = height[Math.max(0, r - 1) * side + c];
      const gx = (e - w) / (2 * spacingM);
      const gy = (s - n) / (2 * spacingM);
      const len = Math.hypot(gx, 1, gy);
      const dot = (gx * 0.5 + gy * 0.5 + sun) / len;
      out[r * side + c] = Math.max(0, Math.min(255, Math.round(222 + 260 * (dot - sun))));
    }
  }
  return out;
}

/** Stream cells of reference and decoded field: both, reference only (lost), decoded only (spurious). */
export function drawStreams(canvas: HTMLCanvasElement, shade: Uint8Array, reference: Uint8Array, decoded: Uint8Array, side: number): void {
  const ctx = sized(canvas, side);
  if (!ctx) return;
  const img = ctx.createImageData(side, side);
  for (let i = 0; i < side * side; i++) {
    const r = reference[i];
    const d = decoded[i];
    const g = shade[i];
    const rgb = r && d ? BOTH : r ? LOST : d ? SPURIOUS : null;
    img.data[4 * i] = rgb ? rgb[0] : g;
    img.data[4 * i + 1] = rgb ? rgb[1] : g;
    img.data[4 * i + 2] = rgb ? rgb[2] : g;
    img.data[4 * i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** Copies a square window of `source` into `target`, magnified without smoothing. */
export function drawCrop(target: HTMLCanvasElement, source: HTMLCanvasElement, col0: number, row0: number, size: number, scale: number): void {
  const px = size * scale;
  const ctx = sized(target, px);
  if (!ctx) return;
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, px, px);
  ctx.drawImage(source, col0, row0, size, size, 0, 0, px, px);
}
