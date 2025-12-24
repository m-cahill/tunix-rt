import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { ModelRegistry } from './ModelRegistry'
import * as client from '../api/client'

// Mock the API client
vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal()
  return {
    ...actual,
    listModelArtifacts: vi.fn(),
    createModelArtifact: vi.fn(),
    getModelArtifact: vi.fn(),
    promoteRunToVersion: vi.fn(),
    getModelDownloadUrl: vi.fn().mockImplementation((id) => `/api/models/versions/${id}/download`),
  }
})

describe('ModelRegistry', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders loading state initially', async () => {
    vi.mocked(client.listModelArtifacts).mockImplementation(() => new Promise(() => {})) // Pending promise
    render(<ModelRegistry />)
    expect(screen.getByText(/Loading.../i)).toBeInTheDocument()
  })

  it('renders empty state when no artifacts', async () => {
    vi.mocked(client.listModelArtifacts).mockResolvedValue([])
    render(<ModelRegistry />)
    await waitFor(() => {
      expect(screen.queryByText(/Loading.../i)).not.toBeInTheDocument()
    })
    expect(screen.getByText('Models')).toBeInTheDocument()
  })

  it('renders list of artifacts', async () => {
    const mockArtifacts = [
      { id: '1', name: 'model-a', task_type: 'sft', description: 'desc', created_at: '', updated_at: '' },
      { id: '2', name: 'model-b', task_type: 'sft', description: 'desc', created_at: '', updated_at: '' },
    ]
    vi.mocked(client.listModelArtifacts).mockResolvedValue(mockArtifacts)

    render(<ModelRegistry />)

    await waitFor(() => {
      expect(screen.getByText('model-a')).toBeInTheDocument()
      expect(screen.getByText('model-b')).toBeInTheDocument()
    })
  })

  it('creates a new artifact', async () => {
    vi.mocked(client.listModelArtifacts).mockResolvedValue([])
    vi.mocked(client.createModelArtifact).mockResolvedValue({
      id: '3', name: 'new-model', task_type: 'sft', description: 'new desc', created_at: '', updated_at: ''
    })

    render(<ModelRegistry />)
    await waitFor(() => expect(screen.queryByText(/Loading.../i)).not.toBeInTheDocument())

    fireEvent.change(screen.getByPlaceholderText('Model Name'), { target: { value: 'new-model' } })
    fireEvent.change(screen.getByPlaceholderText('Description'), { target: { value: 'new desc' } })

    fireEvent.click(screen.getByText('Create Model'))

    await waitFor(() => {
      expect(client.createModelArtifact).toHaveBeenCalledWith({
        name: 'new-model',
        description: 'new desc',
        task_type: 'sft'
      })
    })

    // Should refresh list
    expect(client.listModelArtifacts).toHaveBeenCalledTimes(2)
  })

  it('selects an artifact and shows details', async () => {
    const artifact = { id: '1', name: 'model-a', task_type: 'sft', description: 'desc', created_at: '', updated_at: '' }
    const artifactDetail = {
        ...artifact,
        latest_version: {
            id: 'v1-id', artifact_id: '1', version: 'v1', status: 'ready', sha256: 'abc', size_bytes: 100, source_run_id: 'run-1', created_at: '',
            metrics_json: {}, config_json: {}, provenance_json: {}, storage_uri: ''
        }
    }

    vi.mocked(client.listModelArtifacts).mockResolvedValue([artifact])
    vi.mocked(client.getModelArtifact).mockResolvedValue(artifactDetail)

    render(<ModelRegistry />)

    await waitFor(() => expect(screen.getByText('model-a')).toBeInTheDocument())

    fireEvent.click(screen.getByText('model-a'))

    await waitFor(() => {
        expect(screen.getByText('Promote Run to Version')).toBeInTheDocument()
        expect(screen.getByText('v1')).toBeInTheDocument()
        expect(screen.getByText('Download Artifacts')).toBeInTheDocument()
    })
  })

  it('handles promotion', async () => {
    const artifact = { id: '1', name: 'model-a', task_type: 'sft', description: 'desc', created_at: '', updated_at: '' }
    vi.mocked(client.listModelArtifacts).mockResolvedValue([artifact])
    vi.mocked(client.getModelArtifact).mockResolvedValue(artifact)
    vi.mocked(client.promoteRunToVersion).mockResolvedValue({} as any)

    // Mock window.alert
    const alertMock = vi.spyOn(window, 'alert').mockImplementation(() => {})

    render(<ModelRegistry />)

    await waitFor(() => expect(screen.getByText('model-a')).toBeInTheDocument())
    fireEvent.click(screen.getByText('model-a'))

    await waitFor(() => expect(screen.getByPlaceholderText('Source Run ID (UUID)')).toBeInTheDocument())

    fireEvent.change(screen.getByPlaceholderText('Source Run ID (UUID)'), { target: { value: 'run-uuid' } })
    fireEvent.change(screen.getByPlaceholderText('Version Label (optional)'), { target: { value: 'v2' } })

    fireEvent.click(screen.getByText('Promote'))

    await waitFor(() => {
        expect(client.promoteRunToVersion).toHaveBeenCalledWith('1', {
            source_run_id: 'run-uuid',
            version_label: 'v2'
        })
        expect(alertMock).toHaveBeenCalledWith('Promotion successful!')
    })
  })
})
