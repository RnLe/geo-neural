// Landscape lab: three presets driven by a Web Worker that owns the wasm
// kernel. Runs start only on an explicit Run.

import type { Bundle, LabData } from "../data/bundle";
import { h, Listeners } from "../data/dom";
import { formatYears } from "../data/format";
import { closurePreset } from "./preset-closure";
import { materialPreset } from "./preset-material";
import { ridgePreset } from "./preset-ridge";
import type { FromWorker, ToWorker } from "./protocol";
import type { Preset, RunnerApi } from "./ui";

export interface LabOptions {
  bundle: Bundle;
  /** URL of landscape_wasm_bg.wasm; a host site may serve its own copy. */
  wasmUrl: string | URL;
}

export interface LabHandle {
  dispose(): void;
}

type PresetId = "ridge" | "closure" | "material";

const TABS: { id: PresetId; title: string }[] = [
  { id: "ridge", title: "Smooth the ridge" },
  { id: "closure", title: "Loss term or construction" },
  { id: "material", title: "Assumed material contrast" },
];

function supported(): boolean {
  return typeof Worker !== "undefined" && typeof WebAssembly === "object" && typeof WebAssembly.instantiate === "function";
}

export function mountLab(el: HTMLElement, options: LabOptions): LabHandle {
  const root = h("section", { class: "gn-root gn-lab", "aria-label": "Landscape lab" });
  el.append(root);
  if (!supported()) {
    root.append(h("p", { class: "gn-message" }, "The lab needs WebAssembly and Web Workers, which this browser does not provide. The rest of the page works without them."));
    return { dispose: () => root.remove() };
  }

  const listeners = new Listeners();
  const wasmHref = new URL(String(options.wasmUrl), document.baseURI).href;
  let worker: Worker | null = null;
  let workerReady: Promise<void> | null = null;
  let readyResolve: (() => void) | null = null;
  let readyReject: ((err: Error) => void) | null = null;
  let seq = 0;
  let liveId = -1;
  let liveKey = "";
  let running = false;
  let outstanding = 0;
  let disposed = false;
  let labData: Promise<LabData> | null = null;
  const lab = () => (labData ??= options.bundle.lab());

  // ---- tabs -----------------------------------------------------------------
  const tablist = h("div", { class: "gn-tabs", role: "tablist", "aria-label": "Lab presets" });
  const panel = h("div", { class: "gn-tabpanel", role: "tabpanel", tabindex: "0" });
  const tabButtons = new Map<PresetId, HTMLButtonElement>();
  for (const t of TABS) {
    const b = h("button", { type: "button", role: "tab", class: "gn-tab", id: `gn-tab-${t.id}-${Math.random().toString(36).slice(2, 8)}`, "aria-selected": "false", tabindex: "-1" }, t.title);
    tabButtons.set(t.id, b);
    tablist.append(b);
    listeners.on(b, "click", () => select(t.id));
  }
  listeners.on(tablist, "keydown", (e: KeyboardEvent) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const ids = TABS.map((t) => t.id);
    const at = ids.indexOf(activeId);
    const next = ids[(at + (e.key === "ArrowRight" ? 1 : ids.length - 1)) % ids.length];
    select(next);
    tabButtons.get(next)?.focus();
  });
  root.append(
    h(
      "p",
      { class: "gn-lab-badge" },
      "Recomputed live in your browser by the project's Rust kernel compiled to WebAssembly, in a background worker. Runs start only when you press Run.",
    ),
    tablist,
    panel,
  );

  // ---- worker ---------------------------------------------------------------
  function send(msg: ToWorker): void {
    worker?.postMessage(msg);
  }

  function onMessage(event: MessageEvent<FromWorker>): void {
    const msg = event.data;
    switch (msg.type) {
      case "ready":
        readyResolve?.();
        return;
      case "disposed":
        (event.target as Worker).terminate();
        return;
      case "error":
        if (msg.id === null && readyReject) readyReject(new Error(msg.message));
        if (msg.id !== null) outstanding = Math.max(0, outstanding - 1);
        if (msg.id === null || msg.id === liveId) {
          running = false;
          preset?.setStatus(`Error: ${msg.message}`);
          preset?.setRunning(false);
        }
        return;
      case "stale":
        outstanding = Math.max(0, outstanding - 1);
        pump();
        return;
      case "frame":
        outstanding = Math.max(0, outstanding - 1);
        // Results from a replaced, reset or cancelled scenario are dropped.
        if (msg.id === liveId && preset) preset.onFrame(msg.frames, msg.elapsedMs);
        pump();
        return;
    }
  }

  function ensureWorker(): Promise<void> {
    if (workerReady) return workerReady;
    const pending = new Promise<void>((resolve, reject) => {
      readyResolve = resolve;
      readyReject = reject;
      try {
        worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
      } catch (err) {
        reject(err instanceof Error ? err : new Error(String(err)));
        return;
      }
      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", (e) => reject(new Error(e.message || "the worker failed to start")));
      send({ type: "init", wasmUrl: wasmHref });
    });
    workerReady = pending.then(
      () => {
        readyResolve = null;
        readyReject = null;
      },
      (err: unknown) => {
        worker?.terminate();
        worker = null;
        workerReady = null;
        readyResolve = null;
        readyReject = null;
        throw err;
      },
    );
    return workerReady;
  }

  // ---- runner ---------------------------------------------------------------
  function pump(): void {
    if (!preset || disposed) return;
    if (!running || outstanding > 0) {
      preset.setRunning(running);
      return;
    }
    if (preset.finished()) {
      running = false;
      preset.setRunning(false);
      preset.setStatus(`Done: ${formatYears(preset.timeYears())}.`);
      return;
    }
    send({ type: "advance", id: liveId, steps: preset.stepsPerMessage(), dtYears: preset.dtYears(), untilYears: preset.untilYears() });
    outstanding += 1;
  }

  function cancelLive(): void {
    if (liveId >= 0 && worker) send({ type: "cancel", id: ++seq });
    liveId = -1;
    liveKey = "";
  }

  const api: RunnerApi = {
    async run() {
      if (running || !preset) return;
      const p = preset;
      const specs = p.specs();
      if (!specs) return;
      running = true;
      p.setRunning(true);
      p.setStatus("Starting the kernel...");
      try {
        await ensureWorker();
      } catch (err) {
        running = false;
        p.setRunning(false);
        p.setStatus(`The wasm kernel could not start: ${err instanceof Error ? err.message : String(err)}`);
        return;
      }
      if (disposed || !running || p !== preset) return;
      const key = p.key();
      if (liveId >= 0 && key === liveKey) {
        if (p.finished()) {
          liveId = ++seq;
          p.showInitial();
          send({ type: "reset", id: liveId });
          outstanding += 1;
        } else {
          pump();
        }
        return;
      }
      cancelLive();
      liveId = ++seq;
      liveKey = key;
      p.showInitial();
      p.setRunning(true);
      send({ type: "create", id: liveId, runs: specs });
      outstanding += 1;
    },
    pause() {
      if (!running || !preset) return;
      running = false;
      preset.setRunning(false);
      preset.setStatus(`Paused at ${formatYears(preset.timeYears())}.`);
    },
    reset() {
      if (!preset) return;
      running = false;
      if (liveId >= 0 && worker && preset.key() === liveKey) {
        liveId = ++seq;
        send({ type: "reset", id: liveId });
        outstanding += 1;
      } else {
        cancelLive();
      }
      preset.showInitial();
      preset.setRunning(false);
    },
    paramsChanged() {
      if (!preset) return;
      running = false;
      cancelLive();
      preset.showInitial();
      preset.setRunning(false);
    },
  };

  // ---- presets --------------------------------------------------------------
  const presets = new Map<PresetId, Preset>();
  let preset: Preset | null = null;
  let activeId: PresetId = "ridge";

  function make(id: PresetId): Preset {
    if (id === "ridge") return ridgePreset(api);
    if (id === "closure") return closurePreset(api, lab());
    return materialPreset(api, lab(), options.bundle.manifest.geology.legend);
  }

  function select(id: PresetId): void {
    if (preset && id === activeId) return;
    running = false;
    cancelLive();
    activeId = id;
    let p = presets.get(id);
    if (!p) {
      p = make(id);
      presets.set(id, p);
    }
    preset = p;
    for (const [tid, b] of tabButtons) {
      const on = tid === id;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
    }
    panel.setAttribute("aria-labelledby", tabButtons.get(id)?.id ?? "");
    panel.replaceChildren(p.el);
    p.showInitial();
    p.setRunning(false);
  }

  select("ridge");

  return {
    dispose() {
      if (disposed) return;
      disposed = true;
      running = false;
      listeners.clear();
      if (worker) {
        const w = worker;
        // The worker frees its scenarios, then closes; terminate regardless.
        w.postMessage({ type: "dispose" } satisfies ToWorker);
        setTimeout(() => w.terminate(), 300);
      }
      worker = null;
      root.remove();
    },
  };
}
