import { describe, expect, it } from "vitest";
import { interval } from "../src/ident";

describe("interval", () => {
  it("reads the 2-log-unit range of a profile, and reports an open end at the grid edge", () => {
    const c = [0.5, 0.75, 1, 1.5, 2];
    const quad = c.map((x) => -50 * Math.log(x) ** 2);
    const [lo, hi] = interval(c, quad);
    expect(lo!).toBeGreaterThan(0.75);
    expect(hi!).toBeLessThan(1.5);
    expect(interval(c, [0, 0, 0, 0, 0])).toEqual([null, null]);
  });
});
