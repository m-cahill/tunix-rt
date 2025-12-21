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

test.describe('Trace Comparison and Evaluation', () => {
  test('can create two traces and compare them', async ({ page }) => {
    await page.goto('/');
    
    // Wait for page to load
    await expect(page.locator('h1')).toContainText('Tunix RT');
    
    // Create first trace (simple)
    const traceTextarea = page.locator('#trace-json');
    const simpleTrace = JSON.stringify({
      trace_version: '1.0',
      prompt: 'What is 2 + 2?',
      final_answer: '4',
      steps: [
        { i: 0, type: 'compute', content: 'Add 2 and 2' }
      ],
      meta: { source: 'e2e-test-simple' }
    });
    
    await traceTextarea.fill(simpleTrace);
    
    const uploadBtn = page.locator('button', { hasText: 'Upload' });
    await uploadBtn.click();
    
    // Wait for success and extract first trace ID
    const successMessage = page.locator('.trace-success');
    await expect(successMessage).toBeVisible({ timeout: 5000 });
    const successText = await successMessage.textContent();
    const firstTraceId = successText?.match(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/)?.[0];
    expect(firstTraceId).toBeTruthy();
    
    // Create second trace (complex)
    const complexTrace = JSON.stringify({
      trace_version: '1.0',
      prompt: 'Explain the process of photosynthesis in plants',
      final_answer: 'Photosynthesis is the process by which plants convert light energy into chemical energy',
      steps: [
        { i: 0, type: 'define', content: 'Photosynthesis is a biochemical process used by plants to convert light into energy' },
        { i: 1, type: 'explain', content: 'Light energy is absorbed by chlorophyll in the chloroplasts' },
        { i: 2, type: 'detail', content: 'Carbon dioxide and water are converted into glucose and oxygen' },
        { i: 3, type: 'conclude', content: 'This process is essential for plant growth and produces oxygen for the atmosphere' }
      ],
      meta: { source: 'e2e-test-complex' }
    });
    
    await traceTextarea.fill(complexTrace);
    await uploadBtn.click();
    
    // Wait for second success message and extract second trace ID
    await expect(successMessage).toBeVisible({ timeout: 5000 });
    const secondSuccessText = await successMessage.textContent();
    const secondTraceId = secondSuccessText?.match(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/)?.[0];
    expect(secondTraceId).toBeTruthy();
    expect(secondTraceId).not.toBe(firstTraceId);
    
    // Now perform comparison
    const baseTraceInput = page.locator('#base-trace-id');
    const otherTraceInput = page.locator('#other-trace-id');
    
    await baseTraceInput.fill(firstTraceId!);
    await otherTraceInput.fill(secondTraceId!);
    
    const compareBtn = page.locator('button', { hasText: 'Fetch & Compare' });
    await compareBtn.click();
    
    // Wait for comparison result
    const comparisonResult = page.locator('.comparison-result');
    await expect(comparisonResult).toBeVisible({ timeout: 5000 });
    
    // Verify side-by-side columns exist
    const columns = page.locator('.comparison-column');
    await expect(columns).toHaveCount(2);
    
    // Verify both traces are displayed
    await expect(page.locator('text=Base Trace')).toBeVisible();
    await expect(page.locator('text=Other Trace')).toBeVisible();
    
    // Verify scores are displayed
    const traceScores = page.locator('.trace-score');
    await expect(traceScores).toHaveCount(2);
    
    // Verify the simple trace has lower score than complex trace
    // (simple has 1 step, complex has 4 steps)
    const scoreTexts = await traceScores.allTextContents();
    const baseScore = parseFloat(scoreTexts[0].match(/[\d.]+/)?.[0] || '0');
    const otherScore = parseFloat(scoreTexts[1].match(/[\d.]+/)?.[0] || '0');
    expect(otherScore).toBeGreaterThan(baseScore);
    
    // Verify trace content is displayed
    await expect(page.locator('text=What is 2 + 2?')).toBeVisible();
    await expect(page.locator('text=Explain the process of photosynthesis in plants')).toBeVisible();
    
    // Verify steps are listed
    await expect(page.locator('text=Add 2 and 2')).toBeVisible();
    await expect(page.locator('text=Light energy is absorbed by chlorophyll in the chloroplasts')).toBeVisible();
  });
});

