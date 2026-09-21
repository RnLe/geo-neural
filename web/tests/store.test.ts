import { describe, expect, it } from "vitest";
import { createSelectionStore, type Selection } from "../src/data/store";

describe("selection store", () => {
  it("notifies on change with the previous state", () => {
    const store = createSelectionStore({ candidateId: "a", overlay: "elevation" });
    const seen: [Selection, Selection][] = [];
    store.subscribe((next, prev) => seen.push([next, prev]));
    store.set({ candidateId: "b" });
    expect(store.get()).toEqual({ candidateId: "b", overlay: "elevation" });
    expect(seen).toHaveLength(1);
    expect(seen[0][1].candidateId).toBe("a");
  });

  it("stays quiet when nothing changes", () => {
    const store = createSelectionStore({ candidateId: "a", overlay: "error" });
    let calls = 0;
    store.subscribe(() => calls++);
    store.set({ candidateId: "a" });
    store.set({});
    expect(calls).toBe(0);
  });

  it("stops notifying after unsubscribe", () => {
    const store = createSelectionStore({ candidateId: "a", overlay: "error" });
    let calls = 0;
    const off = store.subscribe(() => calls++);
    store.set({ overlay: "streams" });
    off();
    store.set({ overlay: "geology" });
    expect(calls).toBe(1);
    expect(store.get().overlay).toBe("geology");
  });

  it("copies the initial state", () => {
    const initial: Selection = { candidateId: "a", overlay: "elevation" };
    const store = createSelectionStore(initial);
    store.set({ candidateId: "z" });
    expect(initial.candidateId).toBe("a");
  });
});
