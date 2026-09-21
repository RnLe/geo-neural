// Number formatting shared by the modules. Decimal units (1 kB = 1000 B).

export function formatBytes(bytes: number): string {
  if (bytes < 1000) return `${bytes} B`;
  if (bytes < 1e6) return `${trim(bytes / 1e3)} kB`;
  return `${trim(bytes / 1e6)} MB`;
}

function trim(v: number): string {
  return v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
}

export function formatMetres(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "not recorded";
  return `${v.toFixed(digits)} m`;
}

export function formatSigned(v: number, digits = 2): string {
  const s = Math.abs(v).toFixed(digits);
  return Number(s) === 0 ? s : v > 0 ? `+${s}` : `-${s}`;
}

export function formatRatio(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "not recorded";
  return v.toFixed(digits);
}

/** Compact scientific notation, e.g. 1.2e-15, with a plain minus sign. */
export function formatSci(v: number, digits = 2): string {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 0.01 && a < 1000) return String(Number(v.toPrecision(digits + 1)));
  const [m, e] = v.toExponential(digits).split("e");
  return `${Number(m)}e${Number(e)}`;
}

/** Integer with comma thousands separators. */
export function formatInt(v: number): string {
  return Math.round(v).toLocaleString("en-US");
}

export function formatYears(v: number): string {
  return `${formatInt(v)} yr`;
}

export function formatPercent(fraction: number, digits = 1): string {
  return `${(100 * fraction).toFixed(digits)}%`;
}
