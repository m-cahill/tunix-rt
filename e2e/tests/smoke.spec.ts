import { test, expect } from '@playwright/test';

/**
 * E2E Selector Policy (M6.3):
 * - ALL selectors must use either `getByTestId` or `getByRole` with scoped containers
 * - Global text selectors (`page.locator('text=...')`) are FORBIDDEN
 * - Text matching (`hasText`) is allowed ONLY within scoped containers or with role/testid
 * - Prefer data-testid for UI elements that may change copy/labels
 * - Use role-based selectors only for semantic HTML elements (buttons, headings, etc.)
 */

test.describe('Smoke Tests', () => {
  test('homepage loads successfully', async ({ page }) => {
    await page.goto('/');
    
    // Check that the page title is present
    await expect(page.locator('h1')).toContainText('Tunix RT');
  });

  test('displays API healthy status', async ({ page }) => {
    await page.goto('/');
    
    // Wait for the API status to load using data-testid
    const apiStatus = page.getByTestId('sys:api-status');
    await expect(apiStatus).toBeVisible();
    
    // In mock mode (CI), should always be healthy
    // In real mode, depends on actual backend
    await expect(apiStatus).toContainText('API:');
  });

  test('displays RediAI status', async ({ page }) => {
    await page.goto('/');
    
    // Wait for the RediAI status to load using data-testid
    const rediStatus = page.getByTestId('sys:redi-status');
    await expect(rediStatus).toBeVisible();
    
    // Should show some status (healthy or down)
    await expect(rediStatus).toContainText('RediAI:');
  });

  test('shows correct status indicators', async ({ page }) => {
    await page.goto('/');
    
    // Wait for loading to complete
    await page.waitForTimeout(1000);
    
    // Should have status cards using data-testid
    const apiCard = page.getByTestId('sys:api-card');
    const rediCard = page.getByTestId('sys:redi-card');
    await expect(apiCard).toBeVisible();
    await expect(rediCard).toBeVisible();
  });
});

test.describe('Trace Upload and Retrieval', () => {
  test('can load example, upload, and fetch trace', async ({ page }) => {
    await page.goto('/');
    
    // Wait for page to load
    await expect(page.locator('h1')).toContainText('Tunix RT');
    
    // Click "Load Example" button using data-testid
    const loadExampleBtn = page.getByTestId('trace:load-example');
    await loadExampleBtn.click();
    
    // Verify textarea is populated using data-testid
    const traceTextarea = page.getByTestId('trace:json');
    const textareaValue = await traceTextarea.inputValue();
    expect(textareaValue.length).toBeGreaterThan(0);
    expect(textareaValue).toContain('trace_version');
    
    // Click upload button using data-testid
    const uploadBtn = page.getByTestId('trace:upload');
    await uploadBtn.click();
    
    // Wait for success message with trace ID using data-testid
    const successMessage = page.getByTestId('trace:success');
    await expect(successMessage).toBeVisible({ timeout: 5000 });
    await expect(successMessage).toContainText('Trace uploaded with ID:');
    
    // Click fetch button using data-testid
    const fetchBtn = page.getByTestId('trace:fetch');
    await fetchBtn.click();
    
    // Wait for fetched trace to appear using data-testid
    const traceResult = page.getByTestId('trace:result');
    await expect(traceResult).toBeVisible({ timeout: 5000 });
    
    // Verify the fetched trace contains expected data using data-testid
    const resultContent = page.getByTestId('trace:result-content');
    const resultText = await resultContent.textContent();
    expect(resultText).toContain('payload');
    expect(resultText).toContain('Convert 68°F to Celsius');
    expect(resultText).toContain('20°C');
  });
});

test.describe('UNGAR Integration Panel', () => {
  test('UNGAR section renders with status', async ({ page }) => {
    await page.goto('/');
    
    // Wait for UNGAR section to load
    const ungarSection = page.getByTestId('ungar:section');
    await expect(ungarSection).toBeVisible();
    
    // Verify status container is displayed
    const statusContainer = page.getByTestId('ungar:status-container');
    await expect(statusContainer).toBeVisible();
    
    // Verify status text is displayed (either Available or Not Installed)
    const statusText = page.getByTestId('ungar:status');
    await expect(statusText).toBeVisible();
    await expect(statusText).toContainText('Status:');
    
    // In default E2E environment (no UNGAR installed), should show "Not Installed"
    await expect(statusText).toContainText('Not Installed');
  });
});

test.describe('Trace Comparison and Evaluation', () => {
  test('can create two traces and compare them', async ({ page }) => {
    await page.goto('/');
    
    // Wait for page to load
    await expect(page.locator('h1')).toContainText('Tunix RT');
    
    // Create first trace (simple) using data-testid
    const traceTextarea = page.getByTestId('trace:json');
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
    
    const uploadBtn = page.getByTestId('trace:upload');
    await uploadBtn.click();
    
    // Wait for success and extract first trace ID using data-testid
    const successMessage = page.getByTestId('trace:success');
    await expect(successMessage).toBeVisible({ timeout: 5000 });
    const firstSuccessText = await successMessage.textContent();
    const firstTraceId = firstSuccessText?.match(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/)?.[0];
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
    
    // Wait for success message to update with NEW trace ID (not the first one)
    await expect(successMessage).toBeVisible({ timeout: 5000 });
    // Wait for the trace ID to change from the first one using data-testid
    await page.waitForFunction(
      (oldId) => {
        const elem = document.querySelector('[data-testid="trace:success"]');
        return elem && elem.textContent && !elem.textContent.includes(oldId);
      },
      firstTraceId,
      { timeout: 5000 }
    );
    
    const secondSuccessText = await successMessage.textContent();
    const secondTraceId = secondSuccessText?.match(/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/)?.[0];
    expect(secondTraceId).toBeTruthy();
    expect(secondTraceId).not.toBe(firstTraceId);
    
    // Now perform comparison using data-testid
    const baseTraceInput = page.getByTestId('compare:base-id');
    const otherTraceInput = page.getByTestId('compare:other-id');
    
    await baseTraceInput.fill(firstTraceId!);
    await otherTraceInput.fill(secondTraceId!);
    
    const compareBtn = page.getByTestId('compare:submit');
    await compareBtn.click();
    
    // Wait for comparison result using data-testid
    const comparisonResult = page.getByTestId('compare:result');
    await expect(comparisonResult).toBeVisible({ timeout: 5000 });
    
    // Verify side-by-side columns exist using data-testid
    const baseColumn = page.getByTestId('compare:base-column');
    const otherColumn = page.getByTestId('compare:other-column');
    await expect(baseColumn).toBeVisible();
    await expect(otherColumn).toBeVisible();
    
    // Verify both traces are displayed using role-based selectors
    await expect(page.getByRole('heading', { name: 'Base Trace' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Other Trace' })).toBeVisible();
    
    // Verify scores are displayed using data-testid
    const baseScore = page.getByTestId('compare:base-score');
    const otherScore = page.getByTestId('compare:other-score');
    await expect(baseScore).toBeVisible();
    await expect(otherScore).toBeVisible();
    
    // Verify the simple trace has lower score than complex trace
    // (simple has 1 step, complex has 4 steps)
    const baseScoreText = await baseScore.textContent();
    const otherScoreText = await otherScore.textContent();
    const baseScoreValue = parseFloat(baseScoreText?.match(/[\d.]+/)?.[0] || '0');
    const otherScoreValue = parseFloat(otherScoreText?.match(/[\d.]+/)?.[0] || '0');
    expect(otherScoreValue).toBeGreaterThan(baseScoreValue);
    
    // Verify trace content is displayed using data-testid (no text collision)
    const basePrompt = page.getByTestId('compare:base-prompt');
    const otherPrompt = page.getByTestId('compare:other-prompt');
    await expect(basePrompt).toContainText('What is 2 + 2?');
    await expect(otherPrompt).toContainText('Explain the process of photosynthesis in plants');
    
    // Verify steps are listed using data-testid (no text collision)
    const baseSteps = page.getByTestId('compare:base-steps');
    const otherSteps = page.getByTestId('compare:other-steps');
    await expect(baseSteps).toContainText('Add 2 and 2');
    await expect(otherSteps).toContainText('Light energy is absorbed by chlorophyll in the chloroplasts');
  });
});

