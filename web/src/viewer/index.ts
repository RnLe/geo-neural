// Terrain viewer: the 513 x 513 display grid with overlays, inspector and
// cross-section. Framework-free; mount into any element and call dispose().

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Bundle, Candidate, CandidateField } from "../data/bundle";
import { h, Listeners, uid } from "../data/dom";
import { formatBytes, formatMetres, formatRatio, formatSigned } from "../data/format";
import type { Overlay, SelectionStore } from "../data/store";
import { bilinear, eastNorth, inside, intersect, makeFrame, toGrid, toLocal } from "./field";
import {
  computeStats,
  elevationLegend,
  errorLegend,
  geologyLegend,
  streamsLegend,
  type FieldStats,
  type LegendContext,
} from "./legend";
import { renderProfile, sampleProfile, type ProfilePoint } from "./profile";
import { STREAM_BOTH, STREAM_LOST, STREAM_SPURIOUS, Terrain } from "./terrain";

export interface ViewerOptions {
  bundle: Bundle;
  store: SelectionStore;
  /** Half-width of the shared signed-error scale, metres. Default 5. */
  errorRangeM?: number;
}

export interface ViewerHandle {
  dispose(): void;
}

const OVERLAY_LABELS: Record<Overlay, string> = {
  elevation: "Elevation",
  error: "Signed error",
  geology: "Geology",
  streams: "Streams",
};

const EXAGGERATIONS = [1, 2, 3, 5, 10];
const OBLIQUE_POLAR = THREE.MathUtils.degToRad(58);
const LINE_LIFT_M = 3;

function webglAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!c.getContext("webgl2");
  } catch {
    return false;
  }
}

export function mountViewer(el: HTMLElement, options: ViewerOptions): ViewerHandle {
  const { bundle, store } = options;
  const m = bundle.manifest;
  const side = bundle.side;
  const errRange = options.errorRangeM ?? 5;
  const h0 = Math.floor(m.reference.minM);
  const frame = makeFrame(side, m.grid.spacingM, m.region.bounds, h0);
  const listeners = new Listeners();
  const root = h("section", { class: "gn-root gn-viewer", "aria-label": "Terrain viewer" });
  el.append(root);

  if (!webglAvailable()) {
    root.append(h("p", { class: "gn-message" }, "This browser cannot create a WebGL 2 context, so the terrain view is not available here. The chart and its table still show every candidate."));
    return { dispose: () => root.remove() };
  }

  // ---- controls -------------------------------------------------------------
  const candId = uid("gn-cand");
  const candSelect = h("select", { id: candId, class: "gn-select" });
  for (const c of m.candidates) candSelect.append(h("option", { value: c.id }, `${c.label} (${formatBytes(c.bytes)})`));
  const evidenceBadge = h("span", { class: "gn-badge", title: "Evidence status from the candidate table" });

  const overlayId = uid("gn-overlay");
  const overlaySelect = h("select", { id: overlayId, class: "gn-select" });
  for (const key of Object.keys(OVERLAY_LABELS) as Overlay[]) overlaySelect.append(h("option", { value: key }, OVERLAY_LABELS[key]));

  const surfaceId = uid("gn-surface");
  const surfaceSelect = h(
    "select",
    { id: surfaceId, class: "gn-select" },
    h("option", { value: "decoded" }, "decoded candidate"),
    h("option", { value: "reference" }, "reference"),
  );

  const exagId = uid("gn-exag");
  const exagSelect = h("select", { id: exagId, class: "gn-select" });
  for (const v of EXAGGERATIONS) exagSelect.append(h("option", { value: String(v) }, `${v}x`));

  const btn = (label: string, extra: Record<string, string> = {}) => h("button", { type: "button", class: "gn-button", ...extra }, label);
  const resetBtn = btn("Reset view");
  const topBtn = btn("Top-down");
  const obliqueBtn = btn("Oblique");
  const sectionBtn = btn("Cross-section", { "aria-pressed": "false" });
  const clearBtn = btn("Clear marks");

  const controlsBar = h(
    "div",
    { class: "gn-controls" },
    h("div", { class: "gn-field gn-field-wide" }, h("label", { for: candId }, "Candidate"), h("div", { class: "gn-inline" }, candSelect, evidenceBadge)),
    h("div", { class: "gn-field" }, h("label", { for: overlayId }, "Overlay"), overlaySelect),
    h("div", { class: "gn-field" }, h("label", { for: surfaceId }, "Surface shown"), surfaceSelect),
    h("div", { class: "gn-field" }, h("label", { for: exagId }, "Vertical exaggeration"), exagSelect),
    h("div", { class: "gn-field" }, h("span", { class: "gn-label" }, "View"), h("div", { class: "gn-inline", role: "group", "aria-label": "Camera presets" }, resetBtn, topBtn, obliqueBtn)),
    h("div", { class: "gn-field" }, h("span", { class: "gn-label" }, "Tools"), h("div", { class: "gn-inline" }, sectionBtn, clearBtn)),
  );
  const facts = h("dl", { class: "gn-facts gn-facts-row", "aria-live": "polite" });

  // ---- stage ----------------------------------------------------------------
  const canvas = h("canvas", { class: "gn-viewer-canvas" });
  const hintId = uid("gn-hint");
  const stage = h(
    "div",
    { class: "gn-viewer-stage", tabindex: "0", role: "application", "aria-roledescription": "3D terrain view", "aria-describedby": hintId },
    canvas,
  );
  const badge = h("div", { class: "gn-viewer-badge" }, "Recorded fields: decoded offline, rendered live");
  const exagLabel = h("div", { class: "gn-viewer-exag" }, "Vertical exaggeration 1x (display only)");
  const northArrow = h("div", { class: "gn-viewer-north", "aria-hidden": "true" });
  northArrow.innerHTML =
    '<svg viewBox="-16 -22 32 44" width="28" height="38"><path d="M0,-18 L8,6 L0,1 L-8,6 Z" class="gn-north-fill"/><path d="M0,-18 L-8,6 L0,1 Z" class="gn-north-half"/><text x="0" y="19" text-anchor="middle">N</text></svg>';
  const scaleBar = h("div", { class: "gn-viewer-scale", "aria-hidden": "true" }, h("div", { class: "gn-viewer-scale-bar" }), h("div", { class: "gn-viewer-scale-label" }));
  const status = h("div", { class: "gn-viewer-status", role: "status" });
  const pinMarker = h("div", { class: "gn-marker gn-marker-pin", "aria-hidden": "true" });
  const aMarker = h("div", { class: "gn-marker gn-marker-end", "aria-hidden": "true" }, "A");
  const bMarker = h("div", { class: "gn-marker gn-marker-end", "aria-hidden": "true" }, "B");
  stage.append(badge, exagLabel, northArrow, scaleBar, status, pinMarker, aMarker, bMarker);
  const hint = h(
    "p",
    { class: "gn-note", id: hintId },
    "Drag to rotate, right-drag or arrow keys to pan, wheel to zoom. Click the surface to inspect a point; Enter inspects the view centre. Local frame: x east, y up, z south, metres.",
  );

  const legendHost = h("div", { class: "gn-viewer-legend" });
  const inspector = h("div", { class: "gn-viewer-inspector", "aria-live": "polite" });
  const profileHost = h("div", { class: "gn-viewer-profile" });
  root.append(
    controlsBar,
    facts,
    stage,
    hint,
    h("div", { class: "gn-viewer-panels" }, legendHost, inspector),
    h("div", { class: "gn-viewer-profile-wrap" }, h("h3", { class: "gn-subhead" }, "Cross-section"), profileHost),
  );

  // ---- three.js -------------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setClearColor(0x000000, 0);
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  renderer.setPixelRatio(pixelRatio);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 20, 200000);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = false;
  controls.screenSpacePanning = false;
  controls.maxPolarAngle = THREE.MathUtils.degToRad(86);
  controls.minDistance = 300;
  controls.maxDistance = 60000;
  controls.listenToKeyEvents(stage);

  const terrain = new Terrain({
    side,
    spacingM: m.grid.spacingM,
    h0,
    hMin: m.reference.minM,
    hMax: m.reference.maxM,
    errRangeM: errRange,
    reference: bundle.reference.height,
    referenceStreams: bundle.reference.streams,
    geology: bundle.geology,
    legend: m.geology.legend,
  });
  terrain.setPixelRatio(pixelRatio);
  scene.add(terrain.mesh);

  const faultMaterial = new THREE.LineDashedMaterial({ color: 0x1b1b1b, dashSize: 90, gapSize: 50 });
  const faultGeometry = new THREE.BufferGeometry();
  const faults = new THREE.LineSegments(faultGeometry, faultMaterial);
  faults.frustumCulled = false;
  scene.add(faults);

  const profileMaterial = new THREE.LineBasicMaterial({ color: 0x111111 });
  const profileGeometry = new THREE.BufferGeometry();
  const profileLine = new THREE.Line(profileGeometry, profileMaterial);
  profileLine.frustumCulled = false;
  profileLine.visible = false;
  scene.add(profileLine);

  // ---- state ----------------------------------------------------------------
  let exag = 1;
  let showReference = false;
  let streamFilter = STREAM_BOTH | STREAM_LOST | STREAM_SPURIOUS;
  let current: CandidateField | null = null;
  let stats: FieldStats | null = null;
  let pin: ProfilePoint | null = null;
  let endA: ProfilePoint | null = null;
  let endB: ProfilePoint | null = null;
  let sectionMode: "off" | "a" | "b" = "off";
  let loadToken = 0;
  let disposed = false;
  let raf = 0;
  let visible = true;
  let dirty = true;

  const geologyCounts = new Array<number>(256).fill(0);
  for (let i = 0; i < bundle.geology.length; i++) geologyCounts[bundle.geology[i]]++;
  let globalMin = m.reference.minM;
  let globalMax = m.reference.maxM;

  const shown = (): Float32Array => (showReference || !current ? bundle.reference.height : current.height);
  const candidateRow = (): Candidate => m.candidates.find((c) => c.id === store.get().candidateId) ?? m.candidates[0];

  // ---- rendering ------------------------------------------------------------
  function requestRender(): void {
    dirty = true;
    if (!visible || raf || disposed) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      render();
    });
  }

  function render(): void {
    if (disposed) return;
    dirty = false;
    renderer.render(scene, camera);
    updateOverlays();
  }

  const tmp = new THREE.Vector3();
  function placeMarker(marker: HTMLElement, p: ProfilePoint | null): void {
    if (!p) {
      marker.hidden = true;
      return;
    }
    const { x, z } = toLocal(frame, p.col, p.row);
    const y = (bilinear(shown(), side, p.col, p.row) - h0) * exag;
    tmp.set(x, y, z).project(camera);
    if (tmp.z > 1 || tmp.z < -1) {
      marker.hidden = true;
      return;
    }
    marker.hidden = false;
    const w = stage.clientWidth;
    const hgt = stage.clientHeight;
    marker.style.transform = `translate(${((tmp.x + 1) / 2) * w}px, ${((1 - tmp.y) / 2) * hgt}px)`;
  }

  function updateOverlays(): void {
    // North arrow: screen direction of -z at the view centre.
    const az = controls.getAzimuthalAngle();
    const polar = controls.getPolarAngle();
    const angle = Math.atan2(Math.sin(az), Math.cos(az) * Math.cos(polar));
    northArrow.style.transform = `rotate(${angle.toFixed(4)}rad)`;
    // Scale bar at the orbit target.
    const d = camera.position.distanceTo(controls.target);
    const heightPx = stage.clientHeight || 1;
    const mpp = (2 * d * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / heightPx;
    const choices = [50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000];
    let len = choices[0];
    for (const c of choices) if (c / mpp <= 140) len = c;
    const bar = scaleBar.firstElementChild as HTMLElement;
    bar.style.width = `${Math.max(1, len / mpp).toFixed(1)}px`;
    (scaleBar.lastElementChild as HTMLElement).textContent = `${len >= 1000 ? `${len / 1000} km` : `${len} m`} at view centre`;
    placeMarker(pinMarker, pin);
    placeMarker(aMarker, endA);
    placeMarker(bMarker, endB);
  }

  // ---- lines draped on the shown surface -----------------------------------
  function drapedPolyline(points: { col: number; row: number }[], field: Float32Array, out: number[], segments: boolean): void {
    const maxStep = 1; // at most one grid cell between samples
    for (let i = 0; i < points.length - 1; i++) {
      const a = points[i];
      const b = points[i + 1];
      const n = Math.max(1, Math.ceil(Math.hypot(b.col - a.col, b.row - a.row) / maxStep));
      for (let k = 0; k < n; k++) {
        for (const t of segments ? [k / n, (k + 1) / n] : [k / n]) {
          const col = a.col + (b.col - a.col) * t;
          const row = a.row + (b.row - a.row) * t;
          const { x, z } = toLocal(frame, col, row);
          out.push(x, bilinear(field, side, col, row) - h0 + LINE_LIFT_M, z);
        }
      }
    }
    if (!segments && points.length) {
      const last = points[points.length - 1];
      const { x, z } = toLocal(frame, last.col, last.row);
      out.push(x, bilinear(field, side, last.col, last.row) - h0 + LINE_LIFT_M, z);
    }
  }

  function updateFaults(): void {
    const field = shown();
    const out: number[] = [];
    for (const line of m.geology.faults) {
      // Fault vertices are metres east and north of the south-west node.
      const pts = line.map(([fx, fy]) => ({ col: fx / frame.spacingM, row: (frame.north - (frame.south + fy)) / frame.spacingM }));
      drapedPolyline(pts, field, out, true);
    }
    faultGeometry.setAttribute("position", new THREE.Float32BufferAttribute(out, 3));
    faults.computeLineDistances();
    faults.scale.y = exag;
    faults.visible = store.get().overlay === "geology";
  }

  function updateProfileLine(): void {
    if (!endA || !endB) {
      profileLine.visible = false;
      return;
    }
    const out: number[] = [];
    drapedPolyline([endA, endB], shown(), out, false);
    profileGeometry.setAttribute("position", new THREE.Float32BufferAttribute(out, 3));
    profileLine.scale.y = exag;
    profileLine.visible = true;
  }

  // ---- panels ---------------------------------------------------------------
  function updateFacts(): void {
    const c = candidateRow();
    evidenceBadge.textContent = c.evidence;
    const mae = c.maeM ?? c.check.maeM;
    const maeNote = c.maeM === null ? " (exporter check; not in the candidate table)" : "";
    const jac = c.streamJaccard ?? c.check.streamJaccard;
    const jacNote = c.streamJaccard === null ? " (exporter check; not in the candidate table)" : "";
    facts.replaceChildren(
      h("dt", {}, "Family"),
      h("dd", {}, c.family),
      h("dt", {}, "Serialised size"),
      h("dd", {}, formatBytes(c.bytes)),
      h("dt", {}, "Mean |error|, 10 m"),
      h("dd", {}, formatMetres(mae) + maeNote),
      h("dt", {}, "Max |error|, 10 m"),
      h("dd", {}, formatMetres(c.maxM ?? c.check.maxM)),
      h("dt", {}, "Stream Jaccard, 10 m"),
      h("dd", {}, formatRatio(jac) + jacNote),
      h("dt", {}, "Error bound"),
      h("dd", {}, c.boundM === null ? "none" : `${c.boundM} m, guaranteed`),
    );
  }

  function legendContext(): LegendContext {
    return {
      hMin: m.reference.minM,
      hMax: m.reference.maxM,
      errRangeM: errRange,
      verticalCrs: m.region.verticalCrs,
      candidate: candidateRow(),
      stats,
      geologyLegend: m.geology.legend,
      geologyCounts,
      geologySource: m.geology.source,
      geologyScale: m.geology.scale,
      geologyNote: m.geology.note,
      streamThresholdCells: m.streamThresholdCells,
      streamFilter,
      onStreamFilter: (bits) => {
        streamFilter = bits;
        terrain.setStreamFilter(bits);
        requestRender();
      },
    };
  }

  function updateLegend(): void {
    const ctx = legendContext();
    const overlay = store.get().overlay;
    const node =
      overlay === "elevation" ? elevationLegend(ctx) : overlay === "error" ? errorLegend(ctx) : overlay === "geology" ? geologyLegend(ctx) : streamsLegend(ctx);
    legendHost.replaceChildren(node);
  }

  function updateInspector(): void {
    if (!pin || !current) {
      inspector.replaceChildren(
        h("p", { class: "gn-legend-title" }, "Point inspector"),
        h("p", { class: "gn-note" }, "Click the surface, or focus the view and press Enter, to read values at the nearest 20 m display node."),
      );
      return;
    }
    const col = Math.round(pin.col);
    const row = Math.round(pin.row);
    const i = row * side + col;
    const { e, n } = eastNorth(frame, col, row);
    const ref = bundle.reference.height[i];
    const dec = current.height[i];
    const cls = bundle.geology[i];
    const geo = m.geology.legend.find((g) => g.code === cls);
    const yesNo = (v: number) => (v ? "yes" : "no");
    inspector.replaceChildren(
      h("p", { class: "gn-legend-title" }, "Point inspector (nearest 20 m node)"),
      h(
        "dl",
        { class: "gn-facts" },
        h("dt", {}, "Easting"),
        h("dd", {}, `${e.toFixed(1)} m`),
        h("dt", {}, "Northing"),
        h("dd", {}, `${n.toFixed(1)} m (${m.region.crs})`),
        h("dt", {}, "Reference height"),
        h("dd", {}, `${ref.toFixed(2)} m`),
        h("dt", {}, "Decoded height"),
        h("dd", {}, `${dec.toFixed(2)} m`),
        h("dt", {}, "Signed error"),
        h("dd", {}, `${formatSigned(dec - ref)} m (decoded minus reference)`),
        h("dt", {}, "Geology class"),
        h("dd", {}, geo ? geo.label : `code ${cls}`),
        h("dt", {}, "Stream cell"),
        h("dd", {}, `reference ${yesNo(bundle.reference.streams[i])}, candidate ${yesNo(current.streams[i])}`),
        h("dt", {}, "Candidate"),
        h("dd", {}, current.id),
      ),
    );
  }

  function updateProfile(): void {
    if (!endA || !endB || !current) {
      profileHost.replaceChildren(
        h("p", { class: "gn-note" }, sectionMode === "off" ? "Press Cross-section, then click two points on the surface." : sectionMode === "a" ? "Click the first point (A)." : "Click the second point (B)."),
      );
      return;
    }
    const data = sampleProfile(frame, bundle.reference.height, current.height, endA, endB);
    renderProfile(profileHost, data, candidateRow().label);
  }

  function setStatus(text: string): void {
    status.textContent = text;
    status.hidden = text === "";
  }

  // ---- candidate switching --------------------------------------------------
  async function showCandidate(id: string): Promise<void> {
    const token = ++loadToken;
    candSelect.value = id;
    updateFacts();
    setStatus("Loading field...");
    try {
      const field = await bundle.candidate(id);
      if (disposed || token !== loadToken) return;
      current = field;
      terrain.setCandidate(field.height, field.streams);
      stats = computeStats(bundle.reference.height, bundle.reference.streams, field.height, field.streams, errRange);
      globalMin = Math.min(m.reference.minM, stats.minHeightM);
      globalMax = Math.max(m.reference.maxM, stats.maxHeightM);
      setStatus(sectionHint());
      updateLegend();
      updateInspector();
      updateFaults();
      updateProfileLine();
      updateProfile();
      requestRender();
    } catch (err) {
      if (token !== loadToken || disposed) return;
      setStatus(`Could not load this field: ${(err as Error).message}`);
    }
  }

  function sectionHint(): string {
    return sectionMode === "a" ? "Cross-section: click point A" : sectionMode === "b" ? "Cross-section: click point B" : "";
  }

  // ---- camera ---------------------------------------------------------------
  function setView(kind: "oblique" | "top"): void {
    const midH = (m.reference.minM + m.reference.maxM) / 2;
    const target = new THREE.Vector3(0, (midH - h0) * exag, 0);
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const fit = (terrain.half * 1.12) / Math.tan(fov / 2) / Math.min(1, camera.aspect);
    const polar = kind === "top" ? 0.0005 : OBLIQUE_POLAR;
    const dist = kind === "top" ? fit : fit * 0.8;
    const az = 0;
    camera.position.set(
      target.x + dist * Math.sin(polar) * Math.sin(az),
      target.y + dist * Math.cos(polar),
      target.z + dist * Math.sin(polar) * Math.cos(az),
    );
    controls.target.copy(target);
    controls.update();
    requestRender();
  }

  function setExaggeration(v: number): void {
    const ratio = v / exag;
    exag = v;
    terrain.setExaggeration(v);
    // Keep the view on the same terrain point as heights stretch.
    const dy = controls.target.y * (ratio - 1);
    controls.target.y += dy;
    camera.position.y += dy;
    controls.update();
    faults.scale.y = v;
    profileLine.scale.y = v;
    exagLabel.textContent = `Vertical exaggeration ${v}x (display only)`;
    requestRender();
  }

  // ---- picking --------------------------------------------------------------
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  function pickAt(clientX: number, clientY: number): ProfilePoint | null {
    const rect = canvas.getBoundingClientRect();
    ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const { origin, direction } = raycaster.ray;
    const hit = intersect(frame, shown(), exag, (globalMin - h0) * exag - 1, (globalMax - h0) * exag + 1, {
      origin: [origin.x, origin.y, origin.z],
      direction: [direction.x, direction.y, direction.z],
    });
    if (!hit) return null;
    const g = toGrid(frame, hit.x, hit.z);
    return inside(frame, g.col, g.row) ? g : null;
  }

  function handlePick(p: ProfilePoint | null): void {
    if (!p) return;
    if (sectionMode === "a") {
      endA = p;
      endB = null;
      sectionMode = "b";
      updateProfileLine();
    } else if (sectionMode === "b") {
      endB = p;
      sectionMode = "off";
      sectionBtn.setAttribute("aria-pressed", "false");
      updateProfileLine();
    } else {
      pin = { col: Math.round(p.col), row: Math.round(p.row) };
      updateInspector();
    }
    setStatus(sectionHint());
    updateProfile();
    requestRender();
  }

  let down: { x: number; y: number; t: number } | null = null;
  listeners.on(canvas, "pointerdown", (e: PointerEvent) => {
    down = e.button === 0 ? { x: e.clientX, y: e.clientY, t: performance.now() } : null;
  });
  listeners.on(canvas, "pointerup", (e: PointerEvent) => {
    if (!down || e.button !== 0) return;
    const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
    const quick = performance.now() - down.t < 700;
    down = null;
    if (moved < 5 && quick) handlePick(pickAt(e.clientX, e.clientY));
  });
  listeners.on(stage, "keydown", (e: KeyboardEvent) => {
    if (e.target !== stage) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      handlePick(pickAt(rect.left + rect.width / 2, rect.top + rect.height / 2));
    }
  });

  // ---- control wiring -------------------------------------------------------
  listeners.on(candSelect, "change", () => store.set({ candidateId: candSelect.value }));
  listeners.on(overlaySelect, "change", () => store.set({ overlay: overlaySelect.value as Overlay }));
  listeners.on(surfaceSelect, "change", () => {
    showReference = surfaceSelect.value === "reference";
    terrain.setShowReference(showReference);
    updateFaults();
    updateProfileLine();
    requestRender();
  });
  listeners.on(exagSelect, "change", () => setExaggeration(Number(exagSelect.value)));
  listeners.on(resetBtn, "click", () => {
    // Reset view also returns the exaggeration to 1x; Oblique keeps it.
    if (exag !== 1) {
      exagSelect.value = "1";
      setExaggeration(1);
    }
    setView("oblique");
  });
  listeners.on(obliqueBtn, "click", () => setView("oblique"));
  listeners.on(topBtn, "click", () => setView("top"));
  listeners.on(sectionBtn, "click", () => {
    const on = sectionMode === "off";
    sectionMode = on ? "a" : "off";
    sectionBtn.setAttribute("aria-pressed", String(on));
    setStatus(sectionHint());
    updateProfile();
  });
  listeners.on(clearBtn, "click", () => {
    pin = null;
    endA = null;
    endB = null;
    if (sectionMode !== "off") sectionMode = "a";
    updateProfileLine();
    updateInspector();
    updateProfile();
    setStatus(sectionHint());
    requestRender();
  });
  controls.addEventListener("change", requestRender);
  listeners.add(() => controls.removeEventListener("change", requestRender));

  listeners.on(canvas, "webglcontextlost", (e: Event) => {
    e.preventDefault();
    if (!disposed) setStatus("The graphics context was lost. Reload the page to restore the view.");
  });

  const unsubscribe = store.subscribe((next, prev) => {
    if (next.candidateId !== prev.candidateId) void showCandidate(next.candidateId);
    if (next.overlay !== prev.overlay) {
      overlaySelect.value = next.overlay;
      terrain.setOverlay(next.overlay);
      faults.visible = next.overlay === "geology";
      updateLegend();
      requestRender();
    }
  });

  // ---- size and visibility --------------------------------------------------
  let lastProfileWidth = 0;
  const resizeObserver = new ResizeObserver(() => {
    const w = Math.max(1, stage.clientWidth);
    const hgt = Math.max(1, stage.clientHeight);
    renderer.setSize(w, hgt, false);
    camera.aspect = w / hgt;
    camera.updateProjectionMatrix();
    const pw = profileHost.clientWidth;
    if (endA && endB && Math.abs(pw - lastProfileWidth) > 8) {
      lastProfileWidth = pw;
      updateProfile();
    }
    requestRender();
  });
  resizeObserver.observe(stage);
  resizeObserver.observe(profileHost);

  const intersectionObserver = new IntersectionObserver((entries) => {
    visible = entries.some((entry) => entry.isIntersecting);
    if (visible && dirty) requestRender();
  });
  intersectionObserver.observe(stage);

  // ---- start ----------------------------------------------------------------
  const initial = store.get();
  overlaySelect.value = initial.overlay;
  terrain.setOverlay(initial.overlay);
  const w0 = Math.max(1, stage.clientWidth);
  const h0px = Math.max(1, stage.clientHeight);
  renderer.setSize(w0, h0px, false);
  camera.aspect = w0 / h0px;
  camera.updateProjectionMatrix();
  setView("oblique");
  updateFaults();
  updateInspector();
  updateProfile();
  updateLegend();
  void showCandidate(initial.candidateId);

  return {
    dispose() {
      if (disposed) return;
      disposed = true;
      loadToken += 1;
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      unsubscribe();
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      listeners.clear();
      controls.dispose();
      terrain.dispose();
      faultGeometry.dispose();
      faultMaterial.dispose();
      profileGeometry.dispose();
      profileMaterial.dispose();
      scene.clear();
      renderer.dispose();
      renderer.forceContextLoss();
      root.remove();
    },
  };
}
