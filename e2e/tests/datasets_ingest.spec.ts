import { test, expect } from '@playwright/test';

/**
 * E2E tests for the dataset ingest endpoint (POST /api/datasets/ingest).
 *
 * Tests the full ingest → DB persistence → dataset build flow using
 * a small fixture file with known trace data.
 */

test.describe('Dataset Ingest E2E', () => {
  // Unique source name to avoid conflicts with other test data
  const sourceName = `e2e-ingest-${Date.now()}`;

  test('ingest JSONL file and verify persistence via dataset build', async ({ request }) => {
    // 1. Ingest the test fixture
    const ingestResponse = await request.post('/api/datasets/ingest', {
      data: {
        path: 'tests/fixtures/e2e/e2e_ingest.jsonl',
        source_name: sourceName,
      },
    });

    expect(ingestResponse.ok()).toBeTruthy();
    const ingestData = await ingestResponse.json();

    // Verify ingested_count > 0 (minimum required assertion)
    expect(ingestData.ingested_count).toBeGreaterThan(0);
    expect(ingestData.ingested_count).toBe(3); // We have exactly 3 traces in fixture
    expect(ingestData.trace_ids).toHaveLength(3);

    // 2. Verify persistence by building a dataset that filters by source_name
    // The ingest endpoint adds provenance metadata including source_name
    const buildResponse = await request.post('/api/datasets/build', {
      data: {
        dataset_name: `e2e-verify-${Date.now()}`,
        dataset_version: 'v1',
        filters: { source: sourceName },
        limit: 10,
        selection_strategy: 'latest',
      },
    });

    expect(buildResponse.ok()).toBeTruthy();
    const buildData = await buildResponse.json();

    // Verify that the dataset build found the ingested traces
    expect(buildData.trace_count).toBeGreaterThan(0);
  });
});
