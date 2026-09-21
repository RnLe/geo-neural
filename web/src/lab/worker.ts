// Owns the wasm Scenario objects. Each message does a bounded amount of work
// and returns, so pause, reset and cancel take effect between messages.

import init, { Scenario } from "../wasm/landscape_wasm.js";
import { MAX_STEPS_PER_MESSAGE, type Diagnostics, type FromWorker, type RunFrame, type RunSpec, type ToWorker } from "./protocol";

interface WorkerScope {
  postMessage(message: FromWorker, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<ToWorker>) => void) | null;
  close(): void;
}
const scope = self as unknown as WorkerScope;

interface Run {
  key: string;
  scenario: Scenario | null;
  initial: Float64Array;
  last: Diagnostics | null;
  stopped: string | null;
}

let ready: Promise<unknown> | null = null;
let latest = -1;
let runs: Run[] = [];

function freeAll(): void {
  for (const run of runs) run.scenario?.free();
  runs = [];
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function build(spec: RunSpec): Run {
  const run: Run = { key: spec.key, scenario: null, initial: spec.initial, last: null, stopped: null };
  try {
    run.scenario = new Scenario(
      spec.side,
      spec.spacingM,
      spec.initial,
      spec.model,
      spec.boundary,
      JSON.stringify(spec.params),
      spec.weights ?? null,
      spec.weightsMeta ?? null,
      spec.diffusivity ?? null,
    );
    run.last = run.scenario.diagnostics();
  } catch (err) {
    run.stopped = message(err);
  }
  return run;
}

function post(id: number, started: number): void {
  const frames: RunFrame[] = [];
  const transfer: Transferable[] = [];
  for (const run of runs) {
    const surface = run.scenario ? run.scenario.surface() : null;
    if (surface) transfer.push(surface.buffer);
    frames.push({ key: run.key, diagnostics: run.last, surface, stopped: run.stopped });
  }
  scope.postMessage({ type: "frame", id, frames, elapsedMs: performance.now() - started }, transfer);
}

/** Advances every live run to a common target time, a bounded number of steps each. */
function advance(steps: number, dtYears: number, untilYears: number): void {
  const live = runs.filter((r) => r.scenario && !r.stopped);
  if (!live.length || !(dtYears > 0)) return;
  const common = Math.min(...live.map((r) => r.last?.timeYears ?? 0));
  const target = Math.min(untilYears, common + steps * dtYears);
  for (const run of live) {
    const scenario = run.scenario as Scenario;
    let calls = 0;
    // A truncated step (maxSubsteps reached) advances less; allow a few extra calls.
    while ((run.last?.timeYears ?? 0) < target - 1e-6 && calls < steps + 4) {
      const now = run.last?.timeYears ?? 0;
      const d = scenario.step(1, Math.min(dtYears, target - now));
      run.last = d;
      calls += 1;
      if (d.rejected) {
        run.stopped = d.message || "the model refused this state";
        break;
      }
    }
  }
}

async function handle(msg: ToWorker): Promise<void> {
  switch (msg.type) {
    case "init": {
      ready ??= init({ module_or_path: msg.wasmUrl });
      try {
        await ready;
        scope.postMessage({ type: "ready" });
      } catch (err) {
        ready = null;
        scope.postMessage({ type: "error", id: null, message: `could not load the wasm kernel: ${message(err)}` });
      }
      return;
    }
    case "create": {
      if (msg.id < latest) return scope.postMessage({ type: "stale", id: msg.id });
      latest = msg.id;
      if (!ready) return scope.postMessage({ type: "error", id: msg.id, message: "the kernel was not initialised" });
      await ready;
      // A newer message may have arrived while waiting.
      if (msg.id !== latest) return scope.postMessage({ type: "stale", id: msg.id });
      const started = performance.now();
      freeAll();
      runs = msg.runs.map(build);
      post(msg.id, started);
      return;
    }
    case "advance": {
      if (msg.id !== latest || !runs.length) return scope.postMessage({ type: "stale", id: msg.id });
      const started = performance.now();
      try {
        advance(Math.max(1, Math.min(MAX_STEPS_PER_MESSAGE, Math.floor(msg.steps))), msg.dtYears, msg.untilYears);
      } catch (err) {
        return scope.postMessage({ type: "error", id: msg.id, message: message(err) });
      }
      post(msg.id, started);
      return;
    }
    case "reset": {
      if (msg.id < latest || !runs.length) return scope.postMessage({ type: "stale", id: msg.id });
      latest = msg.id;
      const started = performance.now();
      for (const run of runs) {
        if (!run.scenario) continue;
        try {
          run.scenario.reset(run.initial);
          run.last = run.scenario.diagnostics();
          run.stopped = null;
        } catch (err) {
          run.stopped = message(err);
        }
      }
      post(msg.id, started);
      return;
    }
    case "cancel": {
      if (msg.id < latest) return;
      latest = msg.id;
      freeAll();
      return;
    }
    case "dispose": {
      latest = Number.MAX_SAFE_INTEGER;
      freeAll();
      scope.postMessage({ type: "disposed" });
      scope.close();
      return;
    }
  }
}

// Handle messages strictly in arrival order, even across the awaits above.
let queue: Promise<void> = Promise.resolve();
scope.onmessage = (event) => {
  queue = queue.then(() => handle(event.data)).catch((err) => {
    scope.postMessage({ type: "error", id: null, message: message(err) });
  });
};
