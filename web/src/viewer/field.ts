// Grid geometry helpers on the CPU side: frame conversion, bilinear sampling
// and an analytic ray to heightfield intersection (heights live in the shader,
// so the three.js raycaster cannot see them).

export interface GridFrame {
  side: number;
  spacingM: number;
  /** Bounds of the node centres: west, south, east, north. */
  west: number;
  south: number;
  north: number;
  /** Local origin: x = E - e0, y = h - h0, z = -(N - n0). */
  e0: number;
  n0: number;
  h0: number;
}

export function makeFrame(side: number, spacingM: number, bounds: [number, number, number, number], h0: number): GridFrame {
  const [west, south, east, north] = bounds;
  return { side, spacingM, west, south, north, e0: (west + east) / 2, n0: (south + north) / 2, h0 };
}

/** Continuous grid coordinates (column, row) of a local point. */
export function toGrid(f: GridFrame, x: number, z: number): { col: number; row: number } {
  const e = x + f.e0;
  const n = f.n0 - z;
  return { col: (e - f.west) / f.spacingM, row: (f.north - n) / f.spacingM };
}

export function toLocal(f: GridFrame, col: number, row: number): { x: number; z: number } {
  const e = f.west + col * f.spacingM;
  const n = f.north - row * f.spacingM;
  return { x: e - f.e0, z: -(n - f.n0) };
}

export function eastNorth(f: GridFrame, col: number, row: number): { e: number; n: number } {
  return { e: f.west + col * f.spacingM, n: f.north - row * f.spacingM };
}

export function inside(f: GridFrame, col: number, row: number): boolean {
  return col >= 0 && row >= 0 && col <= f.side - 1 && row <= f.side - 1;
}

/** Bilinear height at continuous grid coordinates, clamped to the grid. */
export function bilinear(field: Float32Array, side: number, col: number, row: number): number {
  const c = Math.min(Math.max(col, 0), side - 1);
  const r = Math.min(Math.max(row, 0), side - 1);
  const c0 = Math.min(Math.floor(c), side - 2);
  const r0 = Math.min(Math.floor(r), side - 2);
  const fc = c - c0;
  const fr = r - r0;
  const i = r0 * side + c0;
  const top = field[i] * (1 - fc) + field[i + 1] * fc;
  const bottom = field[i + side] * (1 - fc) + field[i + side + 1] * fc;
  return top * (1 - fr) + bottom * fr;
}

export interface Ray {
  origin: [number, number, number];
  direction: [number, number, number];
}

/**
 * First intersection of a ray with the displayed surface y = (h - h0) * exag.
 * Marches in half-cell steps inside the bounding box, then bisects.
 */
export function intersect(
  f: GridFrame,
  field: Float32Array,
  exag: number,
  yMin: number,
  yMax: number,
  ray: Ray,
): { x: number; y: number; z: number } | null {
  const half = ((f.side - 1) * f.spacingM) / 2;
  const lo = [-half, yMin, -half];
  const hi = [half, yMax, half];
  const [ox, oy, oz] = ray.origin;
  const d = ray.direction;
  let t0 = 0;
  let t1 = Infinity;
  const o = [ox, oy, oz];
  for (let a = 0; a < 3; a++) {
    if (Math.abs(d[a]) < 1e-12) {
      if (o[a] < lo[a] || o[a] > hi[a]) return null;
      continue;
    }
    let ta = (lo[a] - o[a]) / d[a];
    let tb = (hi[a] - o[a]) / d[a];
    if (ta > tb) [ta, tb] = [tb, ta];
    t0 = Math.max(t0, ta);
    t1 = Math.min(t1, tb);
    if (t0 > t1) return null;
  }
  const above = (t: number) => {
    const x = ox + d[0] * t;
    const y = oy + d[1] * t;
    const z = oz + d[2] * t;
    const g = toGrid(f, x, z);
    return y - (bilinear(field, f.side, g.col, g.row) - f.h0) * exag;
  };
  const len = Math.hypot(d[0], d[1], d[2]);
  const step = (f.spacingM * 0.5) / len;
  let prevT = t0;
  let prev = above(t0);
  if (prev < 0) return null;
  for (let t = t0 + step; t <= t1 + step; t += step) {
    const tt = Math.min(t, t1);
    const v = above(tt);
    if (v <= 0) {
      let a = prevT;
      let b = tt;
      for (let i = 0; i < 24; i++) {
        const m = 0.5 * (a + b);
        if (above(m) > 0) a = m;
        else b = m;
      }
      const tHit = 0.5 * (a + b);
      return { x: ox + d[0] * tHit, y: oy + d[1] * tHit, z: oz + d[2] * tHit };
    }
    prevT = tt;
    prev = v;
    if (tt >= t1) break;
  }
  return null;
}
