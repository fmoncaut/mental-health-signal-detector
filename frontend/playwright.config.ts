import { defineConfig, devices } from "@playwright/test";

/**
 * Config Playwright E2E — "Comment vas-tu ?"
 *
 * Serveur Vite démarré automatiquement avant les tests (webServer).
 * API backend mockée via page.route() dans chaque test — pas de dépendance réseau.
 */
export default defineConfig({
  testDir: "./e2e",
  testMatch: "**/*.e2e.ts",
  timeout: 30_000,
  retries: 0,
  workers: 1,               // séquentiel — évite les conflits sur le port 5173

  reporter: [["list"], ["html", { open: "never" }]],

  use: {
    baseURL: "http://localhost:5173",
    headless: true,
    screenshot: "only-on-failure",
    trace: "on-first-retry",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
  },
});
