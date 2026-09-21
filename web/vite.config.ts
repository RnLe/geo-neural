import { defineConfig } from "vitest/config";

export default defineConfig({
  // Relative asset paths, so the built page works from any sub-path.
  base: "./",
  worker: { format: "es" },
  build: {
    target: "es2022",
    // The viewer chunk carries three.js; everything else stays small.
    chunkSizeWarningLimit: 700,
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
});
