import { defineConfig } from "vitest/config";

// Separate from vite.config.ts so the app's own dev/build config never
// needs to know about test-only settings (and so vite.config.ts, which
// tsconfig.node.json type-checks, doesn't need vitest's config types).
export default defineConfig({
  test: {
    environment: "node", // pure-function tests only (geometry.ts) -- no DOM
    include: ["src/**/*.test.ts"],
  },
});
