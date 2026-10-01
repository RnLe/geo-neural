// Compression microscope: one region and one bound, the conventional product (SZ3), the fixed multilevel
// coder (cubic-ctx) and the learned coder side by side. Bytes by component, errors on a shared scale, stream
// cells, a cross-section, and the learned product decoded again live in a worker from its shipped file.

import { h, Listeners, s, uid } from "../data/dom";
import { formatBytes, formatInt, formatMetres, formatRatio, formatSigned } from "../data/format";
import { cssGradient, DIVERGING_STOPS, STREAM_COLORS } from "../data/palette";
import { SEGMENTS, standaloneBreakdown } from "./accounting";
import type { CodecBundle, CodecProduct, CodecRegion, CoderId, ComponentKind, ProductRasters, RegionRasters } from "./bundle";
import { errorFromCode, modelOf, productsAt } from "./bundle";
import { createDecoder, supported, type DecoderClient } from "./decoder";
import { drawCrop, drawErrorCodes, drawStreams, hillshade } from "./maps";
import type { DecodedMessage } from "./protocol";

export interface MicroscopeOptions {
  bundle: CodecBundle;
  /** URL of gnc_wasm_bg.wasm; a host site may serve its own copy. */
  wasmUrl: string | URL;
}

export interface MicroscopeHandle {
  dispose(): void;
}

const CODERS: CoderId[] = ["sz3", "cubic-ctx", "learned"];
const TITLES: Record<CoderId, { title: string; sub: string }> = {
  sz3: { title: "SZ3", sub: "conventional error-bounded codec" },
  "cubic-ctx": { title: "cubic-ctx", sub: "fixed multilevel predictor and tables" },
  learned: { title: "Learned", sub: "multilevel coder with a shared learned predictor" },
};
const CROP = 64;
const CROP_SCALE = 4;

type Overlay = "error" | "streams";
type Section = "ew" | "ns";

interface Card {
  coder: CoderId;
  root: HTMLElement;
  head: HTMLElement;
  bytes: HTMLElement;
  bar: HTMLElement;
  facts: HTMLElement;
  overview: HTMLCanvasElement;
  frame: HTMLElement;
  rect: HTMLElement;
  line: HTMLElement;
  crop: HTMLCanvasElement;
}

interface Live {
  productId: string;
  result: DecodedMessage;
  checksumMatch: boolean;
  breakdownMatch: boolean;
}

function boundsOf(region: CodecRegion, order: number[]): number[] {
  const have = new Set(region.products.map((p) => p.boundM));
  return order.filter((b) => have.has(b));
}

function niceStep(range: number, target: number): number {
  const raw = range / Math.max(1, target);
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

function boundLabel(b: number): string {
  return b >= 0.01 ? `${b} m` : `${b * 1000} mm`;
}

export function mountMicroscope(el: HTMLElement, options: MicroscopeOptions): MicroscopeHandle {
  const { bundle } = options;
  const m = bundle.manifest;
  const listeners = new Listeners();
  const wasmHref = new URL(String(options.wasmUrl), document.baseURI).href;
  let disposed = false;
  let region = m.regions[0];
  let bound = boundsOf(region, m.bounds)[0];
  let overlay: Overlay = "error";
  let section: Section = "ew";
  let crop = { col: Math.floor(region.display.side / 2), row: Math.floor(region.display.side / 2) };
  let refData: RegionRasters | null = null;
  let shade: Uint8Array | null = null;
  let rasters: Partial<Record<CoderId, ProductRasters>> = {};
  let live: Live | null = null;
  let loadSeq = 0;
  let started = false;
  let decoding = false;
  let decoder: DecoderClient | null = null;

  const root = h("section", { class: "gn-root gn-codec", "aria-label": "Compression microscope" });
  el.append(root);

  // ---- controls -------------------------------------------------------------
  const regionId = uid("gn-codec-region");
  const regionSelect = h("select", { id: regionId, class: "gn-select" });
  for (const r of m.regions) regionSelect.append(h("option", { value: r.id }, r.title));
  const boundId = uid("gn-codec-bound");
  const boundSelect = h("select", { id: boundId, class: "gn-select" });
  const overlayButtons = new Map<Overlay, HTMLButtonElement>();
  const overlayGroup = h("div", { class: "gn-inline", role: "group", "aria-label": "Map content" });
  for (const [key, label] of [["error", "Error"], ["streams", "Streams"]] as [Overlay, string][]) {
    const b = h("button", { type: "button", class: "gn-button", "aria-pressed": String(key === overlay) }, label);
    overlayButtons.set(key, b);
    overlayGroup.append(b);
    listeners.on(b, "click", () => {
      overlay = key;
      for (const [k, btn] of overlayButtons) btn.setAttribute("aria-pressed", String(k === overlay));
      drawMaps();
      renderLegend();
    });
  }
  const sectionId = uid("gn-codec-section");
  const sectionSelect = h(
    "select",
    { id: sectionId, class: "gn-select" },
    h("option", { value: "ew" }, "west to east through the crop"),
    h("option", { value: "ns" }, "north to south through the crop"),
  );
  root.append(
    h(
      "div",
      { class: "gn-controls" },
      h("div", { class: "gn-field" }, h("label", { for: regionId }, "Region"), regionSelect),
      h("div", { class: "gn-field" }, h("label", { for: boundId }, "Error bound"), boundSelect),
      h("div", { class: "gn-field" }, h("span", { class: "gn-label" }, "Maps show"), overlayGroup),
      h("div", { class: "gn-field" }, h("label", { for: sectionId }, "Cross-section"), sectionSelect),
    ),
    h(
      "p",
      { class: "gn-note gn-codec-labels" },
      h("strong", {}, "Measured reference: "),
      "the 10 m DGM1 grid. ",
      h("strong", {}, "Decoded reconstruction: "),
      "each product decoded from its own bytes and compared with the reference at every node. The learned product is also decoded live in this browser.",
    ),
  );

  // ---- cards ----------------------------------------------------------------
  const cardHost = h("div", { class: "gn-codec-cards" });
  const cards = new Map<CoderId, Card>();
  for (const coder of CODERS) {
    const head = h("div", { class: "gn-codec-head" });
    const bytes = h("div", { class: "gn-codec-bytes" });
    const bar = h("div", { class: "gn-codec-bar-wrap" });
    const facts = h("dl", { class: "gn-facts gn-facts-compact" });
    const overview = h("canvas", { class: "gn-codec-map", "aria-hidden": "true" });
    const rect = h("div", { class: "gn-codec-croprect", "aria-hidden": "true" });
    const line = h("div", { class: "gn-codec-sectionline", "aria-hidden": "true" });
    const frame = h(
      "div",
      { class: "gn-codec-frame", tabindex: "0", role: "button", "aria-label": `${TITLES[coder].title} map: click or use the arrow keys to move the crop` },
      overview,
      rect,
      line,
    );
    const cropCanvas = h("canvas", { class: "gn-codec-crop", role: "img", "aria-label": `${TITLES[coder].title}, magnified crop` });
    const card = h("article", { class: `gn-codec-card gn-codec-${coder}` }, head, bytes, bar, facts, frame, h("p", { class: "gn-codec-croplabel" }, "Crop, magnified"), cropCanvas);
    cards.set(coder, { coder, root: card, head, bytes, bar, facts, overview, frame, rect, line, crop: cropCanvas });
    cardHost.append(card);
    listeners.on(frame, "click", (e: MouseEvent) => {
      const r = overview.getBoundingClientRect();
      const side = region.display.side;
      moveCrop(Math.floor(((e.clientX - r.left) / r.width) * side), Math.floor(((e.clientY - r.top) / r.height) * side));
    });
    listeners.on(frame, "keydown", (e: KeyboardEvent) => {
      const step = e.shiftKey ? 32 : 8;
      const d: Record<string, [number, number]> = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
      const mv = d[e.key];
      if (!mv) return;
      e.preventDefault();
      moveCrop(crop.col + mv[0], crop.row + mv[1]);
    });
  }
  const legend = h("div", { class: "gn-codec-legend" });
  const live_ = h("div", { class: "gn-codec-live", "aria-live": "polite" });
  const sectionHost = h("div", { class: "gn-codec-section" });
  const sectionReadout = h("p", { class: "gn-note gn-codec-readout" });
  const tableHost = h("div", { class: "gn-table-wrap" });
  root.append(
    cardHost,
    legend,
    h("h3", { class: "gn-subhead" }, "Live decode of the learned product"),
    live_,
    h("h3", { class: "gn-subhead" }, "Cross-section"),
    sectionHost,
    sectionReadout,
    h("details", { class: "gn-chart-table" }, h("summary", {}, "Bytes by component (table)"), tableHost),
  );

  // ---- helpers ----------------------------------------------------------------
  const products = () => productsAt(region, bound);
  const cropSize = () => Math.min(CROP, region.display.side - 1);
  function cropOrigin(): { col0: number; row0: number } {
    const size = cropSize();
    const max = region.display.side - size;
    return {
      col0: Math.max(0, Math.min(max, crop.col - Math.floor(size / 2))),
      row0: Math.max(0, Math.min(max, crop.row - Math.floor(size / 2))),
    };
  }

  function moveCrop(col: number, row: number): void {
    const side = region.display.side;
    crop = { col: Math.max(0, Math.min(side - 1, col)), row: Math.max(0, Math.min(side - 1, row)) };
    placeCrop();
    drawCrops();
    renderSection();
  }

  function placeCrop(): void {
    const side = region.display.side;
    const size = cropSize();
    const { col0, row0 } = cropOrigin();
    const pct = (v: number) => `${((100 * v) / side).toFixed(3)}%`;
    for (const card of cards.values()) {
      Object.assign(card.rect.style, { left: pct(col0), top: pct(row0), width: pct(size), height: pct(size) });
      const mid = section === "ew" ? row0 + Math.floor(size / 2) : col0 + Math.floor(size / 2);
      if (section === "ew") Object.assign(card.line.style, { left: pct(col0), top: pct(mid + 0.5), width: pct(size), height: "0" });
      else Object.assign(card.line.style, { left: pct(mid + 0.5), top: pct(row0), width: "0", height: pct(size) });
      card.line.className = `gn-codec-sectionline gn-codec-sectionline-${section}`;
    }
  }

  // ---- cards: bytes and metrics -------------------------------------------------
  function renderCards(): void {
    const at = products();
    const scale = Math.max(...CODERS.map((c) => at[c]?.standaloneBytes ?? 0), 1);
    for (const card of cards.values()) {
      const p = at[card.coder];
      card.root.hidden = !p;
      if (!p) continue;
      const model = modelOf(m, p);
      card.head.replaceChildren(
        h("h3", { class: "gn-codec-title" }, TITLES[card.coder].title),
        h("p", { class: "gn-codec-sub" }, card.coder === "sz3" ? `${TITLES.sz3.sub}, ${p.label}` : TITLES[card.coder].sub),
      );
      const total = p.standaloneBytes;
      card.bytes.replaceChildren(
        h("span", { class: "gn-codec-big" }, formatBytes(total)),
        h("span", { class: "gn-codec-unit" }, ` ${formatInt(total)} B${model ? ", model embedded" : ""}`),
        ...(model ? [h("span", { class: "gn-codec-corpus" }, `As a corpus product: ${formatInt(p.bytes)} B per field plus the shared ${formatInt(model.bytes)} B model once per corpus.`)] : []),
      );
      card.bar.replaceChildren(stackedBar(p, model?.bytes ?? null, scale));
      const mt = p.metrics;
      card.facts.replaceChildren(
        h("dt", {}, "Max |error|"),
        h("dd", {}, `${formatMetres(mt.maxM, 4)} (bound ${boundLabel(p.boundM)})`),
        h("dt", {}, "RMSE"),
        h("dd", {}, formatMetres(mt.rmseM, 4)),
        h("dt", {}, "Bound violations"),
        h("dd", {}, formatInt(mt.boundViolations)),
        h("dt", {}, "Stream overlap"),
        h("dd", {}, `Jaccard ${formatRatio(mt.streamJaccard)}, tolerant F1 ${formatRatio(mt.streamTolerantF1)}`),
      );
    }
    renderTable(at);
  }

  function stackedBar(p: CodecProduct, modelBytes: number | null, scale: number): HTMLElement {
    const parts = standaloneBreakdown(p, modelBytes);
    const total = Object.values(parts).reduce((a, b) => a + b, 0);
    const track = h("div", { class: "gn-codec-bar", style: `width:${((100 * total) / scale).toFixed(2)}%` });
    const desc: string[] = [];
    for (const seg of SEGMENTS) {
      const v = parts[seg.kind];
      if (!v) continue;
      const share = v / total;
      const label = `${seg.label}: ${formatInt(v)} B (${(100 * share).toFixed(share < 0.01 ? 2 : 1)}%)`;
      desc.push(label);
      track.append(h("span", { class: `gn-codec-seg gn-seg-${seg.slot}${seg.kind === "model" ? " gn-codec-seg-model" : ""}`, style: `flex-grow:${v}`, title: label }));
    }
    return h("div", { class: "gn-codec-bar-track", role: "img", "aria-label": `Bytes by component: ${desc.join(", ")}` }, track);
  }

  function renderTable(at: Partial<Record<CoderId, CodecProduct>>): void {
    const cols: { title: string; parts: Record<ComponentKind, number> | null; total: number }[] = [];
    for (const c of CODERS) {
      const p = at[c];
      if (!p) continue;
      const model = modelOf(m, p);
      if (model) {
        cols.push({ title: `${TITLES[c].title} (standalone)`, parts: standaloneBreakdown(p, model.bytes), total: p.standaloneBytes });
        cols.push({ title: `${TITLES[c].title} (corpus file)`, parts: standaloneBreakdown(p, null), total: p.bytes });
      } else {
        cols.push({ title: TITLES[c].title, parts: standaloneBreakdown(p, null), total: p.bytes });
      }
    }
    const body = h("tbody");
    for (const seg of SEGMENTS) {
      body.append(
        h(
          "tr",
          {},
          h("th", { scope: "row" }, h("span", { class: `gn-swatch gn-seg-${seg.slot}`, "aria-hidden": "true" }), seg.label),
          ...cols.map((c) => h("td", { class: "gn-num" }, formatInt(c.parts?.[seg.kind] ?? 0))),
        ),
      );
    }
    body.append(h("tr", {}, h("th", { scope: "row" }, "total"), ...cols.map((c) => h("td", { class: "gn-num" }, h("strong", {}, formatInt(c.total))))));
    tableHost.replaceChildren(
      h("table", { class: "gn-table" }, h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Component (bytes)"), ...cols.map((c) => h("th", { scope: "col" }, c.title)))), body),
      h("p", { class: "gn-note" }, "Context is the geology class raster some learned products carry; none of these do. The standalone learned file embeds the model as one more component (9 directory bytes)."),
    );
  }

  // ---- legend -----------------------------------------------------------------
  function renderLegend(): void {
    const at = products();
    const present = new Set<ComponentKind>();
    for (const c of CODERS) {
      const p = at[c];
      if (!p) continue;
      const parts = standaloneBreakdown(p, modelOf(m, p)?.bytes ?? null);
      for (const seg of SEGMENTS) if (parts[seg.kind]) present.add(seg.kind);
    }
    const segs = h(
      "ul",
      { class: "gn-legend-list gn-legend-inline" },
      ...SEGMENTS.filter((sg) => present.has(sg.kind)).map((sg) => h("li", {}, h("span", { class: `gn-swatch gn-seg-${sg.slot}`, "aria-hidden": "true" }), sg.label)),
    );
    const b = bound;
    const map =
      overlay === "error"
        ? h(
            "div",
            { class: "gn-codec-ramp" },
            h("p", { class: "gn-legend-title" }, "Signed error, decoded minus reference (shared scale)"),
            h("div", { class: "gn-ramp" }, h("div", { class: "gn-ramp-bar", style: `background:${cssGradient(DIVERGING_STOPS)}`, "aria-hidden": "true" })),
            h("div", { class: "gn-ramp-ends" }, h("span", {}, `-${boundLabel(b)}`), h("span", {}, "0"), h("span", {}, `+${boundLabel(b)}`)),
            h("p", { class: "gn-note" }, `Errors at the display nodes, every ${region.display.stride * region.spacingM} m; the largest error of each product is in the numbers above.`),
          )
        : h(
            "div",
            {},
            h("p", { class: "gn-legend-title" }, `Stream cells (D8 routing, contributing area at least ${formatInt(m.streamAreaM2 / 1e4)} ha)`),
            h(
              "ul",
              { class: "gn-legend-list gn-legend-inline" },
              h("li", {}, h("span", { class: "gn-swatch", style: `background:${STREAM_COLORS.both}`, "aria-hidden": "true" }), "in both"),
              h("li", {}, h("span", { class: "gn-swatch", style: `background:${STREAM_COLORS.lost}`, "aria-hidden": "true" }), "reference only (lost)"),
              h("li", {}, h("span", { class: "gn-swatch", style: `background:${STREAM_COLORS.spurious}`, "aria-hidden": "true" }), "decoded only (spurious)"),
            ),
            h("p", { class: "gn-note" }, "Over a hillshade of the reference. A routing diagnostic, not discharge."),
          );
    legend.replaceChildren(h("div", {}, h("p", { class: "gn-legend-title" }, "Bytes by component"), segs), map);
  }

  // ---- maps ---------------------------------------------------------------------
  function drawMaps(): void {
    const side = region.display.side;
    for (const card of cards.values()) {
      const r = rasters[card.coder];
      if (!r) continue;
      if (overlay === "error") drawErrorCodes(card.overview, r.error, side);
      else if (refData && shade) drawStreams(card.overview, shade, refData.streams, r.streams, side);
    }
    drawCrops();
  }

  function drawCrops(): void {
    const size = cropSize();
    const { col0, row0 } = cropOrigin();
    for (const card of cards.values()) {
      if (rasters[card.coder]) drawCrop(card.crop, card.overview, col0, row0, size, CROP_SCALE);
    }
  }

  async function loadRasters(): Promise<void> {
    const seq = ++loadSeq;
    const at = products();
    const reg = region;
    try {
      const [ref, ...rs] = await Promise.all([
        bundle.reference(reg),
        ...CODERS.map((c) => (at[c] ? bundle.rasters(reg, at[c] as CodecProduct) : Promise.resolve(null))),
      ]);
      if (disposed || seq !== loadSeq) return;
      if (refData !== ref) shade = hillshade(ref.height, reg.display.side, reg.spacingM * reg.display.stride);
      refData = ref;
      rasters = {};
      CODERS.forEach((c, i) => {
        if (rs[i]) rasters[c] = rs[i] as ProductRasters;
      });
      drawMaps();
      renderSection();
    } catch (err) {
      if (disposed || seq !== loadSeq) return;
      sectionHost.replaceChildren(h("p", { class: "gn-message" }, `The maps could not be loaded: ${err instanceof Error ? err.message : String(err)}`));
    }
  }

  // ---- live decode ----------------------------------------------------------------
  function renderLive(state: string | null): void {
    const p = products().learned;
    if (!p) {
      live_.replaceChildren(h("p", { class: "gn-note" }, "No learned product at this bound."));
      return;
    }
    const model = modelOf(m, p);
    const parts: (HTMLElement | null)[] = [];
    if (state) parts.push(h("p", { class: "gn-codec-status" }, state));
    const l = live && live.productId === p.id ? live : null;
    if (l) {
      const r = l.result;
      parts.push(
        h(
          "p",
          { class: l.checksumMatch ? "gn-codec-ok" : "gn-codec-bad" },
          l.checksumMatch
            ? `Decoded here in ${r.decodeMs.toFixed(0)} ms: ${formatInt(r.rows)} x ${formatInt(r.cols)} nodes from the ${formatBytes(p.bytes)} product and the ${formatBytes(model?.bytes ?? 0)} shared model. The decoded field matches the precomputed reconstruction bit for bit.`
            : `Decoded here in ${r.decodeMs.toFixed(0)} ms, but the lattice checksum does NOT match the precomputed reconstruction.`,
        ),
      );
      parts.push(
        h(
          "dl",
          { class: "gn-facts gn-facts-compact" },
          h("dt", {}, "Product file"),
          h("dd", {}, `${p.product?.file ?? ""}, ${formatInt(p.bytes)} B`),
          h("dt", {}, "Shared model"),
          h("dd", {}, model ? `${model.id}, ${formatInt(model.bytes)} B, ${model.widths.join(" x ")} hidden units${model.heldOut ? `, trained without ${model.heldOut}` : ""}` : "none"),
          h("dt", {}, "Lattice checksum"),
          h("dd", { class: "gn-codec-mono" }, `${r.latticeSha256.slice(0, 16)}... here, ${p.latticeSha256?.slice(0, 16) ?? "none"}... precomputed`),
          h("dt", {}, "Components"),
          h("dd", {}, l.breakdownMatch ? "the sizes the decoder read from the file equal those listed above" : "the sizes read from the file differ from the manifest"),
          h("dt", {}, "Decode time"),
          h(
            "dd",
            {},
            `${r.decodeMs.toFixed(0)} ms for the decode call in a Web Worker (WebAssembly), on this device; not a benchmark. ` +
              "The first run on a page can include the browser's optimising compilation; Decode again shows the steady state. " +
              `The Python decoder took ${(p.decodeS * 1000).toFixed(0)} ms on a shared machine (unqualified).`,
          ),
        ),
      );
    }
    const again = h("button", { type: "button", class: "gn-button", disabled: decoding || !supported() }, l ? "Decode again" : "Decode now");
    again.addEventListener("click", () => void decodeLive());
    parts.push(h("div", { class: "gn-inline" }, again));
    live_.replaceChildren(...parts.filter((x): x is HTMLElement => x !== null));
  }

  async function decodeLive(): Promise<void> {
    const p = products().learned;
    if (!p || disposed) return;
    if (!supported()) {
      renderLive("Live decoding needs WebAssembly and Web Workers, which this browser does not provide. The precomputed numbers above do not depend on it.");
      return;
    }
    const model = modelOf(m, p);
    if (!p.product || !model) {
      renderLive("This bundle does not ship the learned product file or its model, so it cannot be decoded here.");
      return;
    }
    decoding = true;
    renderLive("Fetching the product and the model...");
    try {
      const [productBytes, modelBytes] = await Promise.all([bundle.file(p.product), bundle.file(model.file)]);
      if (disposed) return;
      renderLive("Decoding in a background worker...");
      decoder ??= createDecoder(wasmHref);
      const result = await decoder.decode(productBytes, modelBytes);
      if (disposed || products().learned?.id !== p.id) return;
      const info = JSON.parse(result.describe) as { breakdown: [string, number][] };
      const read = Object.fromEntries(info.breakdown);
      const keys = new Set([...Object.keys(read), ...Object.keys(p.breakdown).filter((k) => (p.breakdown[k as ComponentKind] ?? 0) > 0)]);
      const breakdownMatch = [...keys].every((k) => (read[k] ?? 0) === (p.breakdown[k as ComponentKind] ?? 0));
      live = { productId: p.id, result, checksumMatch: result.latticeSha256 === p.latticeSha256, breakdownMatch };
      decoding = false;
      renderLive(null);
      renderSection();
    } catch (err) {
      if (disposed) return;
      const msg = err instanceof Error ? err.message : String(err);
      if (msg === "superseded") return;
      decoding = false;
      renderLive(`The live decode failed: ${msg}`);
    }
  }

  // ---- cross-section ----------------------------------------------------------------
  function renderSection(): void {
    if (!refData) return;
    const side = region.display.side;
    const stride = region.display.stride;
    const size = cropSize();
    const { col0, row0 } = cropOrigin();
    const mid = section === "ew" ? row0 + Math.floor(size / 2) : col0 + Math.floor(size / 2);
    const n = size + 1;
    const idx = (k: number) => (section === "ew" ? mid * side + col0 + k : (row0 + k) * side + mid);
    const step = stride * region.spacingM;
    const dist = Array.from({ length: n }, (_, k) => k * step);
    const ref = dist.map((_, k) => refData!.height[idx(k)]);
    const p = products();
    const learnedLive = live && p.learned && live.productId === p.learned.id ? live.result : null;
    const dec = learnedLive
      ? dist.map((_, k) => {
          const r = section === "ew" ? mid : row0 + k;
          const c = section === "ew" ? col0 + k : mid;
          return learnedLive.heights[r * stride * learnedLive.cols + c * stride];
        })
      : null;
    const errs = CODERS.filter((c) => rasters[c] && p[c]).map((c) => ({
      coder: c,
      values: dist.map((_, k) => errorFromCode((rasters[c] as ProductRasters).error[idx(k)], (p[c] as CodecProduct).errorScaleM)),
    }));

    const width = Math.max(300, sectionHost.clientWidth || 640);
    const mg = { left: 60, right: 14, top: 12, bottom: 38 };
    const pw = width - mg.left - mg.right;
    const h1 = 150;
    const gap = 24;
    const rowH = 54;
    const rowGap = 18;
    const rowsTop = mg.top + h1 + gap;
    const bottom = rowsTop + errs.length * rowH + Math.max(0, errs.length - 1) * rowGap;
    const height = bottom + mg.bottom;
    const lengthM = dist[n - 1];
    const x = (d: number) => mg.left + (d / lengthM) * pw;
    let lo = Math.min(...ref, ...(dec ?? []));
    let hi = Math.max(...ref, ...(dec ?? []));
    const pad = Math.max(0.5, (hi - lo) * 0.08);
    lo -= pad;
    hi += pad;
    const yStep = niceStep(hi - lo, 4);
    lo = Math.floor(lo / yStep) * yStep;
    hi = Math.ceil(hi / yStep) * yStep;
    const y1 = (v: number) => mg.top + (1 - (v - lo) / (hi - lo)) * h1;
    const eb = bound * 1.1;
    const rowY = (i: number) => (v: number) => rowsTop + i * (rowH + rowGap) + (1 - (v + eb) / (2 * eb)) * rowH;
    const path = (vals: number[], y: (v: number) => number) => vals.map((v, k) => `${k ? "L" : "M"}${x(dist[k]).toFixed(1)},${y(v).toFixed(1)}`).join("");

    const axes = s("g", { class: "gn-axis" });
    for (let v = lo; v <= hi + 1e-9; v += yStep) {
      axes.append(
        s("line", { x1: mg.left, x2: mg.left + pw, y1: y1(v), y2: y1(v), class: "gn-gridline" }),
        s("text", { x: mg.left - 6, y: y1(v) + 4, "text-anchor": "end" }, `${Math.round(v)}`),
      );
    }
    const digits = bound < 0.1 ? 3 : 2;
    errs.forEach((e, i) => {
      const y = rowY(i);
      for (const v of [-bound, 0, bound]) {
        axes.append(s("line", { x1: mg.left, x2: mg.left + pw, y1: y(v), y2: y(v), class: v === 0 ? "gn-gridline" : "gn-floorline" }));
        if (v !== 0) axes.append(s("text", { x: mg.left - 6, y: y(v) + 4, "text-anchor": "end" }, formatSigned(v, digits)));
      }
      axes.append(s("text", { x: mg.left + pw, y: y(bound) - 4, "text-anchor": "end", class: "gn-codec-annot" }, `${TITLES[e.coder].title}: error (m)`));
    });
    const xStep = niceStep(lengthM, Math.max(2, Math.floor(pw / 90)));
    for (let v = 0; v <= lengthM + 1e-9; v += xStep) {
      axes.append(
        s("line", { x1: x(v), x2: x(v), y1: bottom, y2: bottom + 4, class: "gn-tick" }),
        s("text", { x: x(v), y: bottom + 16, "text-anchor": "middle" }, `${Math.round(v)}`),
      );
    }
    axes.append(
      s("text", { x: mg.left + pw / 2, y: height - 4, "text-anchor": "middle" }, `distance along the line (m), ${section === "ew" ? "west to east" : "north to south"}`),
      s("text", { x: 12, y: mg.top + h1 / 2, transform: `rotate(-90 12 ${mg.top + h1 / 2})`, "text-anchor": "middle" }, "height (m)"),
    );
    const marks = s("g", {});
    marks.append(s("path", { d: path(ref, y1), class: "gn-profile-ref" }));
    if (dec) marks.append(s("path", { d: path(dec, y1), class: "gn-codec-line gn-codec-line-learned gn-codec-dashed" }));
    errs.forEach((e, i) => {
      marks.append(s("path", { d: path(e.values.map((v) => Math.max(-eb, Math.min(eb, v))), rowY(i)), class: `gn-codec-line gn-codec-thin gn-codec-line-${e.coder}` }));
    });
    const cross = s("line", { x1: 0, x2: 0, y1: mg.top, y2: bottom, class: "gn-codec-cross", visibility: "hidden" });
    const titleId = uid("gn-codec-section-title");
    const svg = s(
      "svg",
      { class: "gn-profile-svg", width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-labelledby": titleId },
      s("title", { id: titleId }, `Cross-section of ${(lengthM / 1000).toFixed(2)} km: reference height, learned decoded height, and the signed error of each product`),
      axes,
      marks,
      cross,
    );
    const readout = (k: number | null) => {
      if (k === null) {
        sectionReadout.textContent = "Move the pointer over the plot to read values.";
        cross.setAttribute("visibility", "hidden");
        return;
      }
      cross.setAttribute("x1", String(x(dist[k])));
      cross.setAttribute("x2", String(x(dist[k])));
      cross.setAttribute("visibility", "visible");
      const errText = errs.map((e) => `${TITLES[e.coder].title} ${formatSigned(e.values[k], 3)} m`).join(", ");
      sectionReadout.textContent =
        `At ${Math.round(dist[k])} m: reference ${ref[k].toFixed(2)} m` + (dec ? `, learned decoded ${dec[k].toFixed(3)} m` : "") + `. Error: ${errText}.`;
    };
    svg.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const d = ((ev.clientX - r.left - mg.left) / pw) * lengthM;
      readout(d < -step || d > lengthM + step ? null : Math.max(0, Math.min(n - 1, Math.round(d / step))));
    });
    svg.addEventListener("pointerleave", () => readout(null));
    sectionHost.replaceChildren(
      svg,
      h(
        "ul",
        { class: "gn-legend-list gn-legend-inline" },
        h("li", {}, h("span", { class: "gn-swatch-line", "aria-hidden": "true" }), "measured reference (display grid, 1 cm steps)"),
        h("li", {}, h("span", { class: "gn-swatch-line gn-codec-swatch-learned gn-codec-dashed-swatch", "aria-hidden": "true" }), dec ? "decoded reconstruction, learned (decoded live here, dashed)" : "learned decoded height appears after the live decode"),
        ...errs.map((e) => h("li", {}, h("span", { class: `gn-swatch-line gn-codec-swatch-${e.coder}`, "aria-hidden": "true" }), `error, ${TITLES[e.coder].title}`)),
      ),
      h(
        "p",
        { class: "gn-note" },
        `Heights and errors every ${step} m along the line. At this scale the decoded height covers the reference; the differences are in the error rows, from the precomputed rasters, with dashed lines at the bound.`,
      ),
    );
    readout(null);
  }

  // ---- selection ----------------------------------------------------------------
  function fillBounds(): void {
    const bs = boundsOf(region, m.bounds);
    if (!bs.includes(bound)) bound = bs[0];
    boundSelect.replaceChildren(...bs.map((b) => h("option", { value: String(b) }, boundLabel(b))));
    boundSelect.value = String(bound);
  }

  function selectionChanged(): void {
    renderCards();
    renderLegend();
    placeCrop();
    renderLive(started ? null : "The learned product will be decoded when this section scrolls into view.");
    void loadRasters();
    if (started) void decodeLive();
  }

  listeners.on(regionSelect, "change", () => {
    region = m.regions.find((r) => r.id === regionSelect.value) ?? region;
    crop = { col: Math.floor(region.display.side / 2), row: Math.floor(region.display.side / 2) };
    refData = null;
    fillBounds();
    selectionChanged();
  });
  listeners.on(boundSelect, "change", () => {
    bound = Number(boundSelect.value);
    selectionChanged();
  });
  listeners.on(sectionSelect, "change", () => {
    section = sectionSelect.value as Section;
    placeCrop();
    renderSection();
  });

  let lastWidth = 0;
  const resize = new ResizeObserver(() => {
    const w = sectionHost.clientWidth;
    if (Math.abs(w - lastWidth) < 4) return;
    lastWidth = w;
    renderSection();
  });
  resize.observe(sectionHost);
  const start = () => {
    if (started || disposed) return;
    started = true;
    void decodeLive();
  };
  let seen: IntersectionObserver | null = null;
  if (typeof IntersectionObserver === "function") {
    seen = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        seen?.disconnect();
        start();
      }
    }, { rootMargin: "200px" });
    seen.observe(root);
  } else {
    start();
  }

  fillBounds();
  selectionChanged();

  return {
    dispose() {
      if (disposed) return;
      disposed = true;
      seen?.disconnect();
      resize.disconnect();
      listeners.clear();
      decoder?.dispose();
      decoder = null;
      root.remove();
    },
  };
}
