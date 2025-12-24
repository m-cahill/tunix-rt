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
