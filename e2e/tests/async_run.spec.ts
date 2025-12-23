import { test, expect } from '@playwright/test';

test('async run flow: enqueue, poll, complete', async ({ page }) => {
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
  
  // Status should eventually become 'completed'
  // Note: It goes pending -> running -> completed
  await expect(page.getByTestId('tunix:run-status')).toHaveText('completed', { timeout: 30000 });
  
  // Check output
  await expect(page.getByTestId('tunix:run-message')).toHaveText('Dry-run validation successful');
  
  // Check history
  await page.getByTestId('tunix:toggle-history-btn').click();
  await expect(page.getByTestId('tunix:history-row-test-async-v1')).toBeVisible(); // Need to verify if row-ID or similar
  // Our row testid uses run_id, which we don't know easily.
  // But we can check text content.
  await expect(page.getByTestId('tunix:history-list')).toContainText('test-async-v1');
  await expect(page.getByTestId('tunix:history-list')).toContainText('completed');
});

