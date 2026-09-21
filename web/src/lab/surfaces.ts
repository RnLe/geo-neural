// Synthetic starting surfaces for the lab presets.

/** Small deterministic generator, so every visitor sees the same bumps. */
function lcg(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

/**
 * A ridge 40 m high running roughly north to south, plus small bumps and
 * pits a few metres high and about 100 m across.
 */
export function ridgeSurface(side: number, spacingM: number, seed = 7): Float64Array {
  const out = new Float64Array(side * side);
  const length = side * spacingM;
  const tilt = 0.12;
  const sigma = 520;
  const rand = lcg(seed);
  const bumps = Array.from({ length: 18 }, () => ({
    x: (0.08 + 0.84 * rand()) * length,
    y: (0.08 + 0.84 * rand()) * length,
    a: (rand() < 0.3 ? -1 : 1) * (2 + 4 * rand()),
    s: 50 + 70 * rand(),
  }));
  const cos = 1 / Math.hypot(1, tilt);
  for (let r = 0; r < side; r++) {
    const y = (r + 0.5) * spacingM;
    for (let c = 0; c < side; c++) {
      const x = (c + 0.5) * spacingM;
      const d = (x - (length / 2 + tilt * (y - length / 2))) * cos;
      let v = 40 * Math.exp(-(d * d) / (2 * sigma * sigma));
      for (const b of bumps) {
        const q = ((x - b.x) ** 2 + (y - b.y) ** 2) / (2 * b.s * b.s);
        if (q < 30) v += b.a * Math.exp(-q);
      }
      out[r * side + c] = v;
    }
  }
  return out;
}

/**
 * cos(k x) along the columns, with k = pi m / L. Under closed (zero-flux)
 * boundaries at cell centres this is an eigenmode of the discrete operator.
 */
export function sineMode(side: number, spacingM: number, mode: number, amplitudeM: number): { surface: Float64Array; k: number; basis: Float64Array } {
  const k = (Math.PI * mode) / (side * spacingM);
  const basis = new Float64Array(side);
  for (let c = 0; c < side; c++) basis[c] = Math.cos(k * (c + 0.5) * spacingM);
  const surface = new Float64Array(side * side);
  for (let r = 0; r < side; r++) for (let c = 0; c < side; c++) surface[r * side + c] = amplitudeM * basis[c];
  return { surface, k, basis };
}

/** Least-squares amplitude of the mode in a surface. */
export function modeAmplitude(surface: ArrayLike<number>, side: number, basis: Float64Array): number {
  let num = 0;
  let den = 0;
  for (let r = 0; r < side; r++) {
    for (let c = 0; c < side; c++) {
      num += surface[r * side + c] * basis[c];
      den += basis[c] * basis[c];
    }
  }
  return num / den;
}
