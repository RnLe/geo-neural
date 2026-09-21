// A tiny selection store shared by the viewer and the chart.

export type Overlay = "elevation" | "error" | "geology" | "streams";

export const OVERLAYS: readonly Overlay[] = ["elevation", "error", "geology", "streams"];

export interface Selection {
  candidateId: string;
  overlay: Overlay;
}

export type Listener = (next: Selection, previous: Selection) => void;

export interface SelectionStore {
  get(): Selection;
  /** Merges the patch; listeners run only if something changed. */
  set(patch: Partial<Selection>): void;
  /** Returns the unsubscribe function. */
  subscribe(listener: Listener): () => void;
}

export function createSelectionStore(initial: Selection): SelectionStore {
  let state: Selection = { ...initial };
  const listeners = new Set<Listener>();
  return {
    get: () => state,
    set(patch) {
      const next: Selection = { ...state, ...patch };
      if (next.candidateId === state.candidateId && next.overlay === state.overlay) return;
      const previous = state;
      state = next;
      for (const listener of [...listeners]) listener(state, previous);
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
}
