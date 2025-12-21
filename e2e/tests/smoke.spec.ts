import { test, expect } from '@playwright/test';

test.describe('Smoke Tests', () => {
  test('homepage loads successfully', async ({ page }) => {
    await page.goto('/');
    
    // Check that the page title is present
    await expect(page.locator('h1')).toContainText('Tunix RT');
  });

  test('displays API healthy status', async ({ page }) => {
    await page.goto('/');
    
    // Wait for the API status to load
    const apiStatus = page.getByTestId('api-status');
    await expect(apiStatus).toBeVisible();
    
    // In mock mode (CI), should always be healthy
    // In real mode, depends on actual backend
    await expect(apiStatus).toContainText('API:');
  });

  test('displays RediAI status', async ({ page }) => {
    await page.goto('/');
    
    // Wait for the RediAI status to load
    const rediStatus = page.getByTestId('redi-status');
    await expect(rediStatus).toBeVisible();
    
    // Should show some status (healthy or down)
    await expect(rediStatus).toContainText('RediAI:');
  });

  test('shows correct status indicators', async ({ page }) => {
    await page.goto('/');
    
    // Wait for loading to complete
    await page.waitForTimeout(1000);
    
    // Should have status cards
    const statusCards = page.locator('.status-card');
    await expect(statusCards).toHaveCount(2);
  });
});

test.describe('Trace Upload and Retrieval', () => {
  test('can load example, upload, and fetch trace', async ({ page }) => {
    await page.goto('/');
    
    // Wait for page to load
    await expect(page.locator('h1')).toContainText('Tunix RT');
    
    // Find and click "Load Example" button
    const loadExampleBtn = page.locator('button', { hasText: 'Load Example' });
    await loadExampleBtn.click();
    
    // Verify textarea is populated
    const traceTextarea = page.locator('#trace-json');
    const textareaValue = await traceTextarea.inputValue();
    expect(textareaValue.length).toBeGreaterThan(0);
    expect(textareaValue).toContain('trace_version');
    
    // Click upload button
    const uploadBtn = page.locator('button', { hasText: 'Upload' });
    await uploadBtn.click();
    
    // Wait for success message with trace ID
    const successMessage = page.locator('.trace-success');
    await expect(successMessage).toBeVisible({ timeout: 5000 });
    await expect(successMessage).toContainText('Trace uploaded with ID:');
    
    // Click fetch button
    const fetchBtn = page.locator('button', { hasText: 'Fetch' });
    await fetchBtn.click();
    
    // Wait for fetched trace to appear
    const traceResult = page.locator('.trace-result');
    await expect(traceResult).toBeVisible({ timeout: 5000 });
    
    // Verify the fetched trace contains expected data
    const resultPre = traceResult.locator('pre');
    const resultText = await resultPre.textContent();
    expect(resultText).toContain('payload');
    expect(resultText).toContain('Convert 68°F to Celsius');
    expect(resultText).toContain('20°C');
  });
});

