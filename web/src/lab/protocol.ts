// Messages between the lab and its worker. Every create, reset and cancel
// carries a new, larger scenario id; the worker ignores anything older and
// the page drops replies whose id is not the live one.

import type { Diagnostics } from "../wasm/landscape_wasm";

export type { Diagnostics };

export type ModelName = "linear" | "nonlinear" | "flux" | "kfield" | "penalty" | "conductance";
export type BoundaryName = "closed" | "fixed" | "periodic";

export interface RunSpec {
  key: string;
  side: number;
  spacingM: number;
  initial: Float64Array;
  model: ModelName;
  boundary: BoundaryName;
  /** Keys: diffusivity, criticalSlope, uplift, safety, maxSubsteps. */
  params: Record<string, number>;
  weights?: Float32Array;
  weightsMeta?: string;
  diffusivity?: Float64Array;
}

export interface RunFrame {
  key: string;
  diagnostics: Diagnostics | null;
  surface: Float32Array | null;
  /** Why this run stopped (construction failed or a step was refused), else null. */
  stopped: string | null;
}

export type ToWorker =
  | { type: "init"; wasmUrl: string }
  | { type: "create"; id: number; runs: RunSpec[] }
  | { type: "advance"; id: number; steps: number; dtYears: number; untilYears: number }
  | { type: "reset"; id: number }
  | { type: "cancel"; id: number }
  | { type: "dispose" };

export type FromWorker =
  | { type: "ready" }
  | { type: "frame"; id: number; frames: RunFrame[]; elapsedMs: number }
  | { type: "stale"; id: number }
  | { type: "error"; id: number | null; message: string }
  | { type: "disposed" };

/** Upper bound on steps one advance message may request. */
export const MAX_STEPS_PER_MESSAGE = 16;
