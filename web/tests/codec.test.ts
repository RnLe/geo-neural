import { describe, expect, it } from "vitest";
import { breakEven, corpusSummary, perField, standaloneBreakdown } from "../src/codec/accounting";
import { checkManifest, errorFromCode, type CodecManifest, type CodecProduct, type CodecRegion, type CoderId } from "../src/codec/bundle";

const ref = { file: "x", encoding: "bits" as const, bytes: 1, sha256: "" };

function product(region: string, coder: CoderId, bytes: number, boundM = 0.1): CodecProduct {
  const breakdown = coder === "sz3" ? { container: 120, foreign: bytes - 120 } : { container: 147, coarse: 100, params: 0, stream: bytes - 347, raw: 100 };
  return {
    id: `${region}/${boundM}/${coder}`,
    boundM,
    E: 99,
    coder,
    family: coder === "sz3" ? "conventional" : coder === "learned" ? "learned" : "multilevel-fixed",
    label: coder,
    bytes,
    standaloneBytes: coder === "learned" ? bytes + 1009 : bytes,
    breakdown,
    model: coder === "learned" ? "m" : null,
    metrics: { maxM: 0.1, rmseM: 0.05, maeM: 0.04, p99M: 0.09, boundViolations: 0, streamJaccard: 0.5, streamTolerantF1: 0.8 },
    errorScaleM: boundM,
    latticeSha256: null,
    decodeS: 0,
    product: null,
    error: { ...ref, encoding: "u8" },
    streams: ref,
  };
}

function region(id: string, products: CodecProduct[]): CodecRegion {
  return {
    id,
    title: id,
    side: 1025,
    spacingM: 10,
    extent: [0, 0, 10240, 10240],
    crs: "EPSG:25832",
    verticalCrs: "EPSG:7837",
    crop: null,
    display: { side: 513, stride: 2 },
    reference: { height: { ...ref, encoding: "u16-delta", offsetM: 0, quantumM: 0.01 }, streams: ref, minM: 0, maxM: 1, sha256: "" },
    products,
  };
}

function manifest(regions: CodecRegion[]): CodecManifest {
  return {
    schema: "geoneural-codec-bundle-v1",
    createdUtc: "",
    latticeM: 0.001,
    tableId: "",
    streamAreaM2: 50000,
    coders: ["sz3", "cubic-ctx", "learned"],
    bounds: [0.1],
    versions: {},
    models: [{ id: "m", file: { ...ref, encoding: "gnm" }, bytes: 1000, sha256: "", widths: [32, 32], heldOut: null }],
    regions,
    timing: { qualified: false, note: "" },
    attribution: [],
  };
}

describe("break-even", () => {
  it("is the smallest N whose share of the model fits the saving", () => {
    expect(breakEven(100, 50, 200)).toBe(1);
    expect(breakEven(180, 100, 200)).toBe(5);
    expect(breakEven(190, 100, 200)).toBe(10);
    expect(breakEven(200, 100, 200)).toBeNull();
    expect(breakEven(250, 100, 200)).toBeNull();
  });

  it("averages the measured corpus and skips regions without all three products", () => {
    const m = manifest([
      region("a", [product("a", "sz3", 2000), product("a", "cubic-ctx", 1900), product("a", "learned", 1800)]),
      region("b", [product("b", "sz3", 4000), product("b", "cubic-ctx", 3700), product("b", "learned", 3600)]),
      region("c", [product("c", "sz3", 9000), product("c", "cubic-ctx", 9000)]),
    ]);
    const s = corpusSummary(m, 0.1);
    expect(s?.regions).toEqual(["a", "b"]);
    expect(s?.sz3).toBe(3000);
    expect(s?.learned).toBe(2700);
    expect(s?.breakEvenSz3).toBe(4);
    expect(s?.breakEvenCtx).toBe(10);
    expect(perField(s!, 10)).toBe(2800);
    expect(corpusSummary(m, 0.5)).toBeNull();
  });
});

describe("bundle", () => {
  it("embeds the model as one more component in the standalone form", () => {
    const p = product("a", "learned", 5000);
    const parts = standaloneBreakdown(p, 1000);
    expect(parts.model).toBe(1000);
    expect(parts.container).toBe(147 + 9);
    expect(Object.values(parts).reduce((a, b) => a + b, 0)).toBe(p.standaloneBytes);
    expect(standaloneBreakdown(p, null).model).toBe(0);
  });

  it("maps error codes back to metres", () => {
    expect(errorFromCode(128, 0.1)).toBe(0);
    expect(errorFromCode(255, 0.1)).toBeCloseTo(0.1, 12);
    expect(errorFromCode(1, 0.5)).toBeCloseTo(-0.5, 12);
  });

  it("refuses manifests the views cannot trust", () => {
    const good = manifest([region("a", [product("a", "sz3", 2000)])]);
    expect(() => checkManifest(good)).not.toThrow();
    expect(() => checkManifest({ ...good, schema: "other" })).toThrow(/schema/);
    const bad = manifest([region("a", [{ ...product("a", "sz3", 2000), bytes: 2001 }])]);
    expect(() => checkManifest(bad)).toThrow(/breakdown/);
    const orphan = manifest([region("a", [{ ...product("a", "learned", 2000), model: "missing" }])]);
    expect(() => checkManifest(orphan)).toThrow(/unknown model/);
  });
});
