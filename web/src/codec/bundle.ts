// The codec bundle: what the compression microscope and the byte accounting read.
//
// Written by geoneural.codecs.fixtures.write_web_bundle, which defines the format; this comment documents it.
// Directory layout (paths in the manifest are relative to the bundle directory):
//
//   manifest.json
//   models/<model id>.gnm                     shared learned predictor, raw (its sha256 is the one products name)
//   <region>/reference.height.bin             reference at the display grid, u16-delta (gzip)
//   <region>/reference.streams.bin            reference stream cells at the display grid, bits (gzip)
//   <region>/<bound>/<coder>.gnc              the product exactly as stored, raw (learned at least)
//   <region>/<bound>/<coder>.error.bin        signed error at the display grid, u8 codes (gzip)
//   <region>/<bound>/<coder>.streams.bin      stream cells of the decoded field at the display grid, bits (gzip)
//
// manifest.json (schema "geoneural-codec-bundle-v1"):
//   latticeM, tableId           height lattice of the multilevel coders and the rANS table id they need
//   streamAreaM2                contributing area that makes a stream cell (D8 routing)
//   coders, bounds              coders in display order ("sz3", "cubic-ctx", "learned") and bounds in metres
//   models[]                    id, file, bytes, sha256, widths, heldOut (the region the model never saw)
//   regions[]                   id, title, side, spacingM, extent [west, south, east, north] of the node
//                               centres, crs, verticalCrs, crop (window of the source field or null),
//                               display {side, stride}, reference {height (+offsetM, quantumM), streams, minM,
//                               maxM, sha256 of the float32 field}, products[]
//   products[]                  id, boundM, E, coder, family, label, bytes (file as stored), standaloneBytes
//                               (with the model embedded), breakdown {container, coarse, params, stream, raw,
//                               model, context, foreign: bytes}, model (id of the shared model or null),
//                               metrics {maxM, rmseM, maeM, p99M, boundViolations, streamJaccard,
//                               streamTolerantF1} measured at full resolution against the reference,
//                               errorScaleM, latticeSha256 (sha256 of the decoded lattice as little-endian i64,
//                               multilevel only), decodeS (Python, unqualified), product, error, streams
//
// File references are {file, encoding, bytes, sha256}; sha256 is of the stored bytes. Encodings: "u16-delta"
// (h = offsetM + quantumM * code, first differences mod 2^16), "bits" (row-major, most significant bit first),
// "u8" (error codes: error = errorScaleM * (code - 128) / 127 at each display node, which is every stride-th
// node of the field), all gzip-compressed; "gnc" and "gnm" are raw. Display rasters sample the field at the
// display nodes; "streams" rasters mark a display node if any field node within stride / 2 is a stream cell.

import { decodeBits, decodeU16Delta, decodeU8, gunzip, sha256Hex } from "../data/decode";

export const CODEC_SCHEMA = "geoneural-codec-bundle-v1";

export type CodecEncoding = "u16-delta" | "bits" | "u8" | "gnc" | "gnm";

export interface CodecFileRef {
  file: string;
  encoding: CodecEncoding;
  bytes: number;
  sha256: string;
}

export type CoderId = "sz3" | "cubic-ctx" | "learned";
export type CodecFamily = "conventional" | "multilevel-fixed" | "learned";

export type ComponentKind = "container" | "coarse" | "params" | "stream" | "raw" | "model" | "context" | "foreign";

export interface CodecMetrics {
  maxM: number;
  rmseM: number;
  maeM: number;
  p99M: number;
  boundViolations: number;
  streamJaccard: number | null;
  streamTolerantF1: number | null;
}

export interface CodecProduct {
  id: string;
  boundM: number;
  E: number;
  coder: CoderId;
  family: CodecFamily;
  label: string;
  bytes: number;
  standaloneBytes: number;
  breakdown: Partial<Record<ComponentKind, number>>;
  model: string | null;
  metrics: CodecMetrics;
  errorScaleM: number;
  latticeSha256: string | null;
  decodeS: number;
  product: CodecFileRef | null;
  error: CodecFileRef;
  streams: CodecFileRef;
}

export interface CodecModel {
  id: string;
  file: CodecFileRef;
  bytes: number;
  sha256: string;
  widths: [number, number];
  heldOut: string | null;
}

export interface CodecRegion {
  id: string;
  title: string;
  side: number;
  spacingM: number;
  extent: [number, number, number, number];
  crs: string;
  verticalCrs: string;
  crop: { row: number; col: number } | null;
  display: { side: number; stride: number };
  reference: {
    height: CodecFileRef & { offsetM: number; quantumM: number };
    streams: CodecFileRef;
    minM: number;
    maxM: number;
    sha256: string;
  };
  products: CodecProduct[];
}

export interface CodecManifest {
  schema: string;
  createdUtc: string;
  latticeM: number;
  tableId: string;
  streamAreaM2: number;
  coders: CoderId[];
  bounds: number[];
  versions: Record<string, string>;
  models: CodecModel[];
  regions: CodecRegion[];
  timing: { qualified: boolean; note: string };
  attribution: string[];
}

export interface RegionRasters {
  height: Float32Array;
  streams: Uint8Array;
}

export interface ProductRasters {
  error: Uint8Array;
  streams: Uint8Array;
}

export interface CodecBundle {
  baseUrl: string;
  manifest: CodecManifest;
  /** Stored bytes of a file, checked against its sha256 where crypto.subtle exists, inflated if gzip. */
  file(ref: CodecFileRef): Promise<Uint8Array>;
  reference(region: CodecRegion): Promise<RegionRasters>;
  rasters(region: CodecRegion, product: CodecProduct): Promise<ProductRasters>;
}

/** Throws with a plain message if the manifest does not have the shape the views rely on. */
export function checkManifest(m: CodecManifest): void {
  if (m.schema !== CODEC_SCHEMA) throw new Error(`unexpected codec bundle schema ${String(m.schema)}`);
  if (!Array.isArray(m.regions) || !m.regions.length) throw new Error("the codec bundle lists no regions");
  const models = new Set(m.models.map((x) => x.id));
  for (const r of m.regions) {
    if (r.display.side !== Math.floor((r.side - 1) / r.display.stride) + 1) throw new Error(`${r.id}: display grid does not match the field`);
    for (const p of r.products) {
      if (p.model !== null && !models.has(p.model)) throw new Error(`${p.id}: unknown model ${p.model}`);
      const sum = Object.values(p.breakdown).reduce((a, b) => a + (b ?? 0), 0);
      if (sum !== p.bytes) throw new Error(`${p.id}: breakdown sums to ${sum}, not ${p.bytes} bytes`);
    }
  }
}

/** Products of one region at one bound, keyed by coder. */
export function productsAt(region: CodecRegion, boundM: number): Partial<Record<CoderId, CodecProduct>> {
  const out: Partial<Record<CoderId, CodecProduct>> = {};
  for (const p of region.products) if (p.boundM === boundM) out[p.coder] = p;
  return out;
}

/** Signed error in metres of one u8 code. */
export function errorFromCode(code: number, scaleM: number): number {
  return (scaleM * (code - 128)) / 127;
}

export function modelOf(manifest: CodecManifest, product: CodecProduct): CodecModel | null {
  return product.model === null ? null : manifest.models.find((m) => m.id === product.model) ?? null;
}

export async function loadCodecBundle(baseUrl: string, signal?: AbortSignal): Promise<CodecBundle> {
  const base = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";
  const res = await fetch(base + "manifest.json", { signal });
  if (!res.ok) throw new Error(`${base}manifest.json: HTTP ${res.status}`);
  const manifest = (await res.json()) as CodecManifest;
  checkManifest(manifest);
  const files = new Map<string, Promise<Uint8Array>>();

  function file(ref: CodecFileRef): Promise<Uint8Array> {
    let pending = files.get(ref.file);
    if (!pending) {
      pending = (async () => {
        const r = await fetch(base + ref.file, { signal });
        if (!r.ok) throw new Error(`${ref.file}: HTTP ${r.status}`);
        const bytes = new Uint8Array(await r.arrayBuffer());
        if (bytes.byteLength !== ref.bytes) throw new Error(`${ref.file}: expected ${ref.bytes} bytes, got ${bytes.byteLength}`);
        const hex = await sha256Hex(bytes);
        if (hex !== null && hex !== ref.sha256) throw new Error(`${ref.file}: checksum does not match the manifest`);
        return ref.encoding === "gnc" || ref.encoding === "gnm" ? bytes : gunzip(bytes);
      })();
      pending.catch(() => files.delete(ref.file));
      files.set(ref.file, pending);
    }
    return pending;
  }

  async function reference(region: CodecRegion): Promise<RegionRasters> {
    const n = region.display.side * region.display.side;
    const ref = region.reference;
    const [h, s] = await Promise.all([file(ref.height), file(ref.streams)]);
    const codes = decodeU16Delta(h, n);
    const height = new Float32Array(n);
    for (let i = 0; i < n; i++) height[i] = ref.height.offsetM + ref.height.quantumM * codes[i];
    return { height, streams: decodeBits(s, n) };
  }

  async function rasters(region: CodecRegion, product: CodecProduct): Promise<ProductRasters> {
    const n = region.display.side * region.display.side;
    const [e, s] = await Promise.all([file(product.error), file(product.streams)]);
    return { error: decodeU8(e, n), streams: decodeBits(s, n) };
  }

  return { baseUrl: base, manifest, file, reference, rasters };
}
