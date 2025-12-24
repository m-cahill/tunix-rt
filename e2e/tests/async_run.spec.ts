import { test, expect } from '@playwright/test';

test('async run flow: enqueue, poll, terminal state', async ({ page }) => {
  await page.goto('/');

  // Wait for health
  await expect(page.getByTestId('sys:api-status')).toHaveText(/API: healthy/i);

  // Enable Async Mode
  await page.getByLabel('Run Async (Non-blocking)').check();

  // Fill form
  await page.getByTestId('tunix:dataset-key').fill('test-async-v1');

  // Click Dry-run
  await page.getByTestId('tunix:run-dry-btn').click();

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

    // We can try to get stderr from the UI if expanded, or just log what we see.
    // The UI shows message.
    const message = await page.getByTestId('tunix:run-message').innerText();

    // Expand details if possible to see stderr (optional but good for screenshots)
    // For now, just throw informative error
    throw new Error(`Async run reached terminal state 'failed'. Message: ${message}`);
  } else if (statusText.includes('cancelled')) {
      throw new Error(`Async run was cancelled unexpectedly.`);
  }

  // If completed, verify success message
  await expect(page.getByTestId('tunix:run-message')).toHaveText('Dry-run validation successful');

  // Check history
  await page.getByTestId('tunix:toggle-history-btn').click();
  await expect(page.getByTestId('tunix:history-list')).toContainText('test-async-v1');
  await expect(page.getByTestId('tunix:history-list')).toContainText('completed');
});
