import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  outputDir: "output/playwright/test-results",
  workers: 1,
  retries: 0,
  reporter: "line",
  use: {
    baseURL: "http://127.0.0.1:3100",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run e2e:pack && npm run build && npm run start -- --hostname 127.0.0.1 --port 3100",
    env: {
      SCRYGLASS_E2E_LOCAL_PACK: "1",
    },
    url: "http://127.0.0.1:3100/chat",
    reuseExistingServer: false,
    timeout: 180_000,
  },
});
