// Decoders for the bundle encodings. All bundle files are gzip-compressed;
// the learned closure weights are the one raw exception (see closure.json).

export type Encoding = "u16-delta" | "bits" | "u8" | "f32";

/** Copies into a fresh ArrayBuffer-backed array (what Blob and DataView want). */
function own(bytes: Uint8Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(bytes.byteLength);
  out.set(bytes);
  return out;
}

export function isGzip(bytes: Uint8Array): boolean {
  return bytes.byteLength >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

/** Inflates a gzip member with the platform DecompressionStream. */
export async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === "undefined") {
    throw new Error("this browser has no DecompressionStream, so the bundle cannot be read");
  }
  const stream = new Blob([own(bytes)]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * uint16 little-endian first differences (mod 2^16) of a row-major code
 * sequence. A running sum mod 65536 restores the codes.
 */
export function decodeU16Delta(bytes: Uint8Array, count: number): Uint16Array {
  if (bytes.byteLength !== count * 2) {
    throw new Error(`u16-delta: expected ${count * 2} bytes, got ${bytes.byteLength}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Uint16Array(count);
  let code = 0;
  for (let i = 0; i < count; i++) {
    code = (code + view.getUint16(2 * i, true)) & 0xffff;
    out[i] = code;
  }
  return out;
}

/** height = offset + quantum * code */
export function codesToHeights(codes: Uint16Array, offsetM: number, quantumM: number): Float32Array {
  const out = new Float32Array(codes.length);
  for (let i = 0; i < codes.length; i++) out[i] = offsetM + quantumM * codes[i];
  return out;
}

/** A packed boolean mask, most significant bit first (numpy packbits order), as 0/1 bytes. */
export function decodeBits(bytes: Uint8Array, count: number): Uint8Array {
  if (bytes.byteLength < Math.ceil(count / 8)) {
    throw new Error(`bits: expected at least ${Math.ceil(count / 8)} bytes, got ${bytes.byteLength}`);
  }
  const out = new Uint8Array(count);
  for (let i = 0; i < count; i++) out[i] = (bytes[i >> 3] >> (7 - (i & 7))) & 1;
  return out;
}

export function decodeU8(bytes: Uint8Array, count: number): Uint8Array {
  if (bytes.byteLength !== count) throw new Error(`u8: expected ${count} bytes, got ${bytes.byteLength}`);
  return own(bytes);
}

/** Little-endian float32. Pass count < 0 to take everything. */
export function decodeF32(bytes: Uint8Array, count = -1): Float32Array {
  if (bytes.byteLength % 4 !== 0) throw new Error("f32: byte length is not a multiple of 4");
  const n = bytes.byteLength / 4;
  if (count >= 0 && n !== count) throw new Error(`f32: expected ${count} values, got ${n}`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = view.getFloat32(4 * i, true);
  return out;
}

/** Hex SHA-256, or null where crypto.subtle is unavailable (insecure origins). */
export async function sha256Hex(bytes: Uint8Array): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  const digest = new Uint8Array(await subtle.digest("SHA-256", own(bytes)));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
}
