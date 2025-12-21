import { defineConfig, devices } from '@playwright/test';

/**
 * Read environment variables from file.
 * https://github.com/motdotla/dotenv
 */
// require('dotenv').config();

const REDIAI_MODE = process.env.REDIAI_MODE || 'mock';
const REDIAI_BASE_URL = process.env.REDIAI_BASE_URL || 'http://localhost:8080';

// Port configuration (M4: add env var support for flexibility)
const FRONTEND_PORT = process.env.FRONTEND_PORT || '5173';
const BACKEND_PORT = process.env.BACKEND_PORT || '8000';

// Database URL for E2E (explicit in CI, uses default locally)
const DATABASE_URL = process.env.DATABASE_URL || 'postgresql+asyncpg://postgres:postgres@localhost:5432/postgres';

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
  testDir: './tests',
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only - M4: reduced to 1 to avoid masking failures */
  retries: process.env.CI ? 1 : 0,
  /* Opt out of parallel tests on CI. */
  workers: process.env.CI ? 1 : undefined,
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: 'html',

  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    /* M4: Standardize on 127.0.0.1 (IPv4) to avoid ::1 (IPv6) connection issues */
    baseURL: `http://127.0.0.1:${FRONTEND_PORT}`,
    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: 'on-first-retry',
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  /* Run your local dev server before starting the tests */
  /* M4: Both servers explicitly bind to 127.0.0.1 to avoid IPv6 issues */
  webServer: [
    {
      command: `cd ../backend && uvicorn tunix_rt_backend.app:app --host 127.0.0.1 --port ${BACKEND_PORT}`,
      url: `http://127.0.0.1:${BACKEND_PORT}/api/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
      env: {
        REDIAI_MODE,
        REDIAI_BASE_URL,
        DATABASE_URL,
      },
    },
    {
      command: `cd ../frontend && npm run dev -- --host 127.0.0.1 --port ${FRONTEND_PORT}`,
      url: `http://127.0.0.1:${FRONTEND_PORT}`,
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
    },
  ],
});

