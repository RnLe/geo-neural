// Loads the static data bundle written by geoneural.export.web.

import {
  codesToHeights,
  decodeBits,
  decodeF32,
  decodeU16Delta,
  decodeU8,
  gunzip,
  isGzip,
  sha256Hex,
  type Encoding,
} from "./decode";

export interface FileRef {
  file: string;
  encoding: Encoding;
  bytes: number;
  sha256: string;
}

export type Family = "conventional" | "neural" | "hybrid" | "corrected" | "control";

export interface Candidate {
  id: string;
  label: string;
  family: Family;
  evidence: string;
  bytes: number;
  maeM: number | null;
  maxM: number | null;
  streamJaccard: number | null;
  boundM: number | null;
  /** Recomputed by the exporter from the 10 m field it wrote. */
  check: { maeM: number; maxM: number; streamJaccard: number | null };
  height: FileRef;
  streams: FileRef;
}

export interface ChartRow {
  id: string;
  label: string;
  family: Family;
  experiment: string;
  bytes: number;
  maeM: number | null;
  maxM: number | null;
  streamJaccard: number | null;
  boundGuaranteed: boolean;
  evidence: string;
  dominatedBy: string | null;
}

export interface LegendEntry {
  code: number;
  label: string;
  color: string;
}

export interface Manifest {
  schema: string;
  createdUtc: string;
  region: {
    id: string;
    title: string;
    crs: string;
    verticalCrs: string;
    /** [west, south, east, north] of the node centres, metres. */
    bounds: [number, number, number, number];
    referenceSide: number;
    referenceSpacingM: number;
    referenceSha256: string;
  };
  grid: {
    side: number;
    spacingM: number;
    stride: number;
    rowOrder: string;
    heightOffsetM: number;
    heightQuantumM: number;
    note: string;
  };
  reference: { height: FileRef; streams: FileRef; minM: number; maxM: number };
  streamThresholdCells: number;
  candidates: Candidate[];
  geology: {
    classes: FileRef;
    legend: LegendEntry[];
    /** Polylines, metres east and north of the south-west node. */
    faults: [number, number][][];
    faultsFrame: string;
    source: string;
    scale: string;
    unknownFraction: number;
    note: string;
  };
  closure?: {
    meta: string;
    weights: Record<string, string>;
    encoding: string;
    surfaces: FileRef & { count: number; side: number; spacingM: number; seed: number; note: string };
  };
  labTerrain: {
    side: number;
    spacingM: number;
    height: FileRef;
    classes: FileRef;
    note: string;
  };
  chart: ChartRow[];
  attribution: string[];
}

export interface CandidateField {
  id: string;
  height: Float32Array;
  streams: Uint8Array;
}

export interface LabData {
  /** closure.json as text (the kernel parses it) and parsed. */
  closureText: string | null;
  closure: ClosureMeta | null;
  weights: Record<string, Float32Array>;
  surfaces: Float32Array[];
  surfaceSide: number;
  surfaceSpacingM: number;
  terrain: { side: number; spacingM: number; height: Float32Array; classes: Uint8Array };
}

export interface ClosureMeta {
  spacingM: number;
  dtYears: number;
  teacher: { diffusivity: number; criticalSlope: number };
  validated: { maxSlope: number; minHeightM: number; maxHeightM: number; note?: string };
  arms: Record<string, { apply: string; conservative: boolean; trainedWith?: string; floats: number; sha256?: string }>;
}

export interface Bundle {
  baseUrl: string;
  manifest: Manifest;
  /** Nodes per side of the display grid. */
  side: number;
  reference: { height: Float32Array; streams: Uint8Array };
  geology: Uint8Array;
  /** Decodes one viewer candidate (cached). */
  candidate(id: string): Promise<CandidateField>;
  /** Lab inputs, loaded on first use (cached). */
  lab(): Promise<LabData>;
}

export interface LoadOptions {
  signal?: AbortSignal;
  /** Check each file against its manifest SHA-256 where crypto.subtle exists. Default true. */
  verify?: boolean;
}

async function fetchBytes(url: string, signal?: AbortSignal): Promise<Uint8Array> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

/**
 * Fetches the manifest, the reference field, its streams and the geology
 * classes. Candidates and lab inputs load on demand.
 */
export async function loadBundle(baseUrl: string, options: LoadOptions = {}): Promise<Bundle> {
  const base = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
  const { signal } = options;
  const verify = options.verify ?? true;
  const res = await fetch(base + "manifest.json", { signal });
  if (!res.ok) throw new Error(`${base}manifest.json: HTTP ${res.status}`);
  const manifest = (await res.json()) as Manifest;
  if (manifest.schema !== "geoneural-web-bundle-v1") {
    throw new Error(`unexpected bundle schema ${String(manifest.schema)}`);
  }
  const side = manifest.grid.side;
  const count = side * side;

  async function raw(ref: FileRef): Promise<Uint8Array> {
    const bytes = await fetchBytes(base + ref.file, signal);
    if (verify && ref.sha256) {
      const hex = await sha256Hex(bytes);
      if (hex !== null && hex !== ref.sha256) throw new Error(`${ref.file}: checksum does not match the manifest`);
    }
    return gunzip(bytes);
  }
  const heights = async (ref: FileRef) => {
    if (ref.encoding !== "u16-delta") throw new Error(`${ref.file}: expected u16-delta, got ${ref.encoding}`);
    return codesToHeights(decodeU16Delta(await raw(ref), count), manifest.grid.heightOffsetM, manifest.grid.heightQuantumM);
  };
  const mask = async (ref: FileRef) => {
    if (ref.encoding !== "bits") throw new Error(`${ref.file}: expected bits, got ${ref.encoding}`);
    return decodeBits(await raw(ref), count);
  };

  const [refHeight, refStreams, geologyRaw] = await Promise.all([
    heights(manifest.reference.height),
    mask(manifest.reference.streams),
    raw(manifest.geology.classes),
  ]);
  const geology = decodeU8(geologyRaw, count);

  const fields = new Map<string, Promise<CandidateField>>();
  function candidate(id: string): Promise<CandidateField> {
    let pending = fields.get(id);
    if (!pending) {
      const row = manifest.candidates.find((c) => c.id === id);
      if (!row) return Promise.reject(new Error(`no viewer field for candidate ${id}`));
      pending = Promise.all([heights(row.height), mask(row.streams)]).then(([height, streams]) => ({ id, height, streams }));
      // A failed load may be retried.
      pending.catch(() => fields.delete(id));
      fields.set(id, pending);
    }
    return pending;
  }

  let labPending: Promise<LabData> | null = null;
  function lab(): Promise<LabData> {
    if (!labPending) {
      labPending = loadLab(base, manifest, raw, verify, signal);
      labPending.catch(() => {
        labPending = null;
      });
    }
    return labPending;
  }

  return { baseUrl: base, manifest, side, reference: { height: refHeight, streams: refStreams }, geology, candidate, lab };
}

async function loadLab(
  base: string,
  manifest: Manifest,
  raw: (ref: FileRef) => Promise<Uint8Array>,
  verify: boolean,
  signal?: AbortSignal,
): Promise<LabData> {
  const t = manifest.labTerrain;
  const tCount = t.side * t.side;
  const terrainPromise = Promise.all([raw(t.height), raw(t.classes)]).then(([h, c]) => ({
    side: t.side,
    spacingM: t.spacingM,
    height: decodeF32(h, tCount),
    classes: decodeU8(c, tCount),
  }));
  const closure = manifest.closure;
  if (!closure) {
    return {
      closureText: null,
      closure: null,
      weights: {},
      surfaces: [],
      surfaceSide: 0,
      surfaceSpacingM: 0,
      terrain: await terrainPromise,
    };
  }
  const textPromise = fetch(base + closure.meta, { signal }).then((r) => {
    if (!r.ok) throw new Error(`${closure.meta}: HTTP ${r.status}`);
    return r.text();
  });
  // Raw little-endian float32 per closure.json, checked against its sha256 there.
  const weightPromises = Object.entries(closure.weights).map(
    async ([arm, file]) => [arm, await fetchBytes(base + file, signal)] as const,
  );
  const s = closure.surfaces;
  const surfacesPromise = raw(s).then((bytes) => {
    const all = decodeF32(bytes, s.count * s.side * s.side);
    return Array.from({ length: s.count }, (_, i) => all.slice(i * s.side * s.side, (i + 1) * s.side * s.side));
  });
  const [closureText, weightBytes, surfaces, terrain] = await Promise.all([
    textPromise,
    Promise.all(weightPromises),
    surfacesPromise,
    terrainPromise,
  ]);
  const meta = JSON.parse(closureText) as ClosureMeta;
  const weights: [string, Float32Array][] = [];
  for (const [arm, stored] of weightBytes) {
    const spec = meta.arms[arm];
    if (verify && spec?.sha256) {
      const hex = await sha256Hex(stored);
      if (hex !== null && hex !== spec.sha256) throw new Error(`${arm}: weights do not match closure.json`);
    }
    // Inflate in case a later bundle gzips the weights too.
    const w = decodeF32(isGzip(stored) ? await gunzip(stored) : stored);
    if (spec && spec.floats !== w.length) throw new Error(`${arm}: expected ${spec.floats} weights, got ${w.length}`);
    weights.push([arm, w]);
  }
  return {
    closureText,
    closure: meta,
    weights: Object.fromEntries(weights),
    surfaces,
    surfaceSide: s.side,
    surfaceSpacingM: s.spacingM,
    terrain,
  };
}
