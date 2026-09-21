import { describe, expect, it } from "vitest";
import { codesToHeights, decodeBits, decodeF32, decodeU16Delta, gunzip, isGzip } from "../src/data/decode";

async function gzip(bytes: Uint8Array): Promise<Uint8Array> {
  const copy = new Uint8Array(bytes);
  const stream = new Blob([copy]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** What the exporter writes: first differences mod 2^16, uint16 little-endian. */
function encodeU16Delta(codes: number[]): Uint8Array {
  const out = new Uint8Array(codes.length * 2);
  const view = new DataView(out.buffer);
  let prev = 0;
  codes.forEach((code, i) => {
    view.setUint16(2 * i, (code - prev + 65536) & 0xffff, true);
    prev = code;
  });
  return out;
}

/** numpy.packbits order: most significant bit first, zero padded. */
function packBits(bits: number[]): Uint8Array {
  const out = new Uint8Array(Math.ceil(bits.length / 8));
  bits.forEach((b, i) => {
    if (b) out[i >> 3] |= 0x80 >> (i & 7);
  });
  return out;
}

describe("u16-delta", () => {
  it("round trips codes through gzip, including wrap-around", async () => {
    const codes = [0, 1, 65535, 0, 40000, 39999, 12345, 65535, 65534, 7];
    const raw = await gunzip(await gzip(encodeU16Delta(codes)));
    expect(Array.from(decodeU16Delta(raw, codes.length))).toEqual(codes);
  });

  it("maps codes to heights with offset and quantum", () => {
    const h = codesToHeights(new Uint16Array([0, 100, 13460]), 34, 0.01);
    expect(h[0]).toBeCloseTo(34, 6);
    expect(h[1]).toBeCloseTo(35, 6);
    expect(h[2]).toBeCloseTo(168.6, 4);
  });

  it("rejects a wrong length", () => {
    expect(() => decodeU16Delta(new Uint8Array(6), 4)).toThrow();
  });

  it("decodes a random grid exactly", async () => {
    let seed = 1;
    const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const codes = Array.from({ length: 33 * 33 }, () => Math.floor(rand() * 65536));
    const raw = await gunzip(await gzip(encodeU16Delta(codes)));
    expect(Array.from(decodeU16Delta(raw, codes.length))).toEqual(codes);
  });
});

describe("bits", () => {
  it("round trips a mask whose length is not a multiple of 8", async () => {
    const bits = [1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0];
    const packed = await gzip(packBits(bits));
    expect(isGzip(packed)).toBe(true);
    expect(Array.from(decodeBits(await gunzip(packed), bits.length))).toEqual(bits);
  });

  it("reads the most significant bit first", () => {
    expect(Array.from(decodeBits(new Uint8Array([0b10000001]), 8))).toEqual([1, 0, 0, 0, 0, 0, 0, 1]);
  });
});

describe("f32", () => {
  it("reads little-endian floats", () => {
    const src = new Float32Array([1.5, -2.25, 1e-3]);
    const bytes = new Uint8Array(src.length * 4);
    const view = new DataView(bytes.buffer);
    src.forEach((v, i) => view.setFloat32(4 * i, v, true));
    expect(Array.from(decodeF32(bytes, 3))).toEqual(Array.from(src));
  });
});
