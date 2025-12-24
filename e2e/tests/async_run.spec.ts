import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

test('async run flow: enqueue, poll, terminal state', async ({ page, request }) => {
  await page.goto('/');

  // Guardrail: Ensure backend is healthy before starting
  await expect(page.getByTestId('sys:api-status')).toHaveText(/API: healthy/i);

  // 1. Create a minimal trace via API to ensure we have data
  const traceRes = await request.post('/api/traces', {
    data: {
      trace_version: '1.0',
      prompt: 'E2E Test Prompt',
      final_answer: 'E2E Test Answer',
      steps: [
        { i: 0, type: 'reasoning', content: 'Step 1' }
      ],
      meta: { source: 'e2e-async-test' }
    }
  });
  expect(traceRes.ok()).toBeTruthy();
  const traceData = await traceRes.json();
  const traceId = traceData.id;

  // 2. Build a dynamic dataset containing this trace
  const datasetName = `e2e-test-${Date.now()}`;
  const buildRes = await request.post('/api/datasets/build', {
    data: {
      dataset_name: datasetName,
      dataset_version: 'v1',
      filters: { source: 'e2e-async-test' }, // Filter by the source we just used
      limit: 10,
      selection_strategy: 'latest'
    }
  });
  expect(buildRes.ok()).toBeTruthy();
  const buildData = await buildRes.json();
  const datasetKey = buildData.dataset_key;

  // Verify dataset is not empty (Guardrail B)
  expect(buildData.trace_count).toBeGreaterThan(0);

  // Enable Async Mode
  await page.getByLabel('Run Async (Non-blocking)').check();

  // Fill form with the dynamic dataset key
  await page.getByTestId('tunix:dataset-key').fill(datasetKey);

  // Click Dry-run
  // Intercept the creation response to get the run ID
  const runResponsePromise = page.waitForResponse(resp =>
    resp.url().includes('/api/tunix/run') &&
    resp.status() === 200 &&
    resp.request().method() === 'POST'
  );
  await page.getByTestId('tunix:run-dry-btn').click();
  const runResponse = await runResponsePromise;
  const runData = await runResponse.json();
  const runId = runData.run_id;

  // Verify status is displayed
  await expect(page.getByTestId('tunix:run-status')).toBeVisible();

  // NOTE:
  // This test asserts terminal-state behavior, not success.
  // Success determinism is enforced in later milestones.
  // We expect either 'completed' or 'failed', but NOT pending/running forever.
  await expect(page.getByTestId('tunix:run-status')).toHaveText(/completed|failed|cancelled/, { timeout: 30000 });

  // Get final status
  const statusText = await page.getByTestId('tunix:run-status').innerText();

  if (statusText.includes('failed')) {
    // If failed, log details for debugging (this doesn't fail the test unless we throw,
    // but throwing helps CI fail fast with reason)

    // Fetch full run details to get stderr
    // We use the API context from the page (or request fixture if we had it, but page.request is available)
    const detailsResponse = await page.request.get(`/api/tunix/runs/${runId}`);
    if (detailsResponse.ok()) {
        const details = await detailsResponse.json();
        console.log('Run Failed. Details:', JSON.stringify(details, null, 2));

        // Attach details to test report
        await test.info().attach('run-details', {
            body: JSON.stringify(details, null, 2),
            contentType: 'application/json'
        });

        if (details.stderr) {
            await test.info().attach('stderr', {
                body: details.stderr,
                contentType: 'text/plain'
            });
        }
    }

    // We can try to get stderr from the UI if expanded, or just log what we see.
    // The UI shows message.
    const message = await page.getByTestId('tunix:run-message').innerText();

    // Expand details if possible to see stderr (optional but good for screenshots)
    // For now, just throw informative error
    throw new Error(`Async run reached terminal state 'failed'. Message: ${message}. See attachments for stderr.`);
  } else if (statusText.includes('cancelled')) {
      throw new Error(`Async run was cancelled unexpectedly.`);
  }

  // If completed, verify success message
  await expect(page.getByTestId('tunix:run-message')).toContainText('Dry-run validation successful');

  // Check history
  await page.getByTestId('tunix:toggle-history-btn').click();
  await expect(page.getByTestId('tunix:history-list')).toContainText(datasetKey);
  await expect(page.getByTestId('tunix:history-list')).toContainText('completed');
});
