// Byte accounting shared by the microscope and the corpus chart: components of a product, the standalone form
// of a corpus product, and the cost per field when one model serves N fields.

import type { CodecManifest, CodecProduct, ComponentKind } from "./bundle";
import { modelOf, productsAt } from "./bundle";

/** Bar order; colours follow the kind (slot k is --gn-seg-k), not the position. */
export const SEGMENTS: { kind: ComponentKind; label: string; slot: number }[] = [
  { kind: "stream", label: "rANS stream", slot: 1 },
  { kind: "raw", label: "raw bits", slot: 2 },
  { kind: "coarse", label: "coarse lattice", slot: 3 },
  { kind: "params", label: "params", slot: 4 },
  { kind: "model", label: "model", slot: 5 },
  { kind: "context", label: "context", slot: 6 },
  { kind: "foreign", label: "SZ3 stream", slot: 8 },
  { kind: "container", label: "container", slot: 7 },
];

/** Directory entry the container spends on one more component. */
export const ENTRY_BYTES = 9;

/** Components of the product as one standalone file: a corpus product plus its model embedded. */
export function standaloneBreakdown(p: CodecProduct, modelBytes: number | null): Record<ComponentKind, number> {
  const out = { container: 0, coarse: 0, params: 0, stream: 0, raw: 0, model: 0, context: 0, foreign: 0 } as Record<ComponentKind, number>;
  for (const [k, v] of Object.entries(p.breakdown)) out[k as ComponentKind] += v ?? 0;
  if (p.model !== null && modelBytes !== null && !p.breakdown.model) {
    out.model += modelBytes;
    out.container += ENTRY_BYTES;
  }
  return out;
}

/** Smallest corpus size N >= 1 at which product + model / N <= rival, or null if never. */
export function breakEven(productBytes: number, modelBytes: number, rivalBytes: number): number | null {
  if (productBytes >= rivalBytes) return null;
  return Math.max(1, Math.ceil(modelBytes / (rivalBytes - productBytes)));
}

export interface CorpusSummary {
  boundM: number;
  /** Regions with all three products at this bound: the measured corpus. */
  regions: string[];
  sz3: number;
  ctx: number;
  /** Mean learned product without its model, and mean model file. */
  learned: number;
  model: number;
  breakEvenSz3: number | null;
  breakEvenCtx: number | null;
}

/** Means over the regions that have SZ3, cubic-ctx and learned products at the bound. */
export function corpusSummary(manifest: CodecManifest, boundM: number): CorpusSummary | null {
  const rows: { id: string; sz3: number; ctx: number; learned: number; model: number }[] = [];
  for (const region of manifest.regions) {
    const at = productsAt(region, boundM);
    const model = at.learned ? modelOf(manifest, at.learned) : null;
    if (!at.sz3 || !at["cubic-ctx"] || !at.learned || !model) continue;
    rows.push({ id: region.id, sz3: at.sz3.bytes, ctx: at["cubic-ctx"].bytes, learned: at.learned.bytes, model: model.bytes });
  }
  if (!rows.length) return null;
  const mean = (k: "sz3" | "ctx" | "learned" | "model") => rows.reduce((a, r) => a + r[k], 0) / rows.length;
  const s = { sz3: mean("sz3"), ctx: mean("ctx"), learned: mean("learned"), model: mean("model") };
  return {
    boundM,
    regions: rows.map((r) => r.id),
    ...s,
    breakEvenSz3: breakEven(s.learned, s.model, s.sz3),
    breakEvenCtx: breakEven(s.learned, s.model, s.ctx),
  };
}

/** Learned bytes per field when one model serves n fields. */
export function perField(s: CorpusSummary, n: number): number {
  return s.learned + s.model / n;
}
