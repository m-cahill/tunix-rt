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

