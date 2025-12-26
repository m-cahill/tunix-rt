import { describe, it, expect, vi, beforeEach } from 'vitest'
import * as client from './client'

// Mock fetch
const fetchMock = vi.fn()
global.fetch = fetchMock

describe('API Client - Model Registry', () => {
  beforeEach(() => {
    fetchMock.mockReset()
  })

  it('createModelArtifact calls correct endpoint', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ id: '1', name: 'test' })
    })

    await client.createModelArtifact({ name: 'test' })

    expect(fetchMock).toHaveBeenCalledWith('/api/models', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ name: 'test' })
    }))
  })

  it('listModelArtifacts calls correct endpoint', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => []
    })

    await client.listModelArtifacts()

    expect(fetchMock).toHaveBeenCalledWith('/api/models', undefined)
  })

  it('getModelArtifact calls correct endpoint', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ id: '1' })
    })

    await client.getModelArtifact('1')

    expect(fetchMock).toHaveBeenCalledWith('/api/models/1', undefined)
  })

  it('promoteRunToVersion calls correct endpoint', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ id: 'v1' })
    })

    await client.promoteRunToVersion('1', { source_run_id: 'run-1' })

    expect(fetchMock).toHaveBeenCalledWith('/api/models/1/versions/promote', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ source_run_id: 'run-1' })
    }))
  })

  it('getModelVersion calls correct endpoint', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ id: 'v1' })
    })

    await client.getModelVersion('v1')

    expect(fetchMock).toHaveBeenCalledWith('/api/models/versions/v1', undefined)
  })

  it('getModelDownloadUrl returns correct string', () => {
    const url = client.getModelDownloadUrl('v1')
    expect(url).toBe('/api/models/versions/v1/download')
  })
})

describe('API Client - Guardrails', () => {
  it('exports core trace functions', () => {
    expect(client.getApiHealth).toBeDefined()
    expect(client.createTrace).toBeDefined()
    expect(client.listTraces).toBeDefined()
    expect(client.getTrace).toBeDefined()
    expect(client.scoreTrace).toBeDefined()
    expect(client.compareTraces).toBeDefined()
  })

  it('exports dataset functions', () => {
    expect(client.buildDataset).toBeDefined()
    expect(client.getDatasetExportUrl).toBeDefined()
  })

  it('exports Tunix functions', () => {
    expect(client.getTunixStatus).toBeDefined()
    expect(client.executeTunixRun).toBeDefined()
    expect(client.listTunixRuns).toBeDefined()
    expect(client.getTunixRun).toBeDefined()
  })

  it('exports evaluation functions', () => {
    expect(client.evaluateRun).toBeDefined()
    expect(client.getLeaderboard).toBeDefined()
  })

  it('exports tuning functions', () => {
    expect(client.createTuningJob).toBeDefined()
    expect(client.startTuningJob).toBeDefined()
    expect(client.getTuningJob).toBeDefined()
    expect(client.listTuningJobs).toBeDefined()
  })

  it('exports model registry functions', () => {
    expect(client.createModelArtifact).toBeDefined()
    expect(client.listModelArtifacts).toBeDefined()
    expect(client.getModelArtifact).toBeDefined()
    expect(client.promoteRunToVersion).toBeDefined()
    expect(client.getModelVersion).toBeDefined()
    expect(client.getModelDownloadUrl).toBeDefined()
  })

  it('prevents accidental file deletion - all core exports must exist', () => {
    // This test ensures that client.ts is never accidentally overwritten/deleted
    // If this test fails, it means core API exports are missing
    const coreExports = [
      'getApiHealth',
      'createTrace',
      'listTraces',
      'getTrace',
      'scoreTrace',
      'compareTraces',
      'buildDataset',
      'getTunixStatus',
      'executeTunixRun',
      'listTunixRuns',
      'evaluateRun',
      'getLeaderboard',
      'createTuningJob',
      'getTuningJob',
      'createModelArtifact',
      'listModelArtifacts',
    ]

    const missingExports = coreExports.filter(name => !(name in client))

    expect(missingExports).toEqual([])
  })
})
