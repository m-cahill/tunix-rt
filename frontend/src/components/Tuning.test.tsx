import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { Tuning } from './Tuning'
import * as client from '../api/client'

// Mock the client module
vi.mock('../api/client')

describe('Tuning', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('handles promote best run', async () => {
    // Setup mock data
    const job = {
        id: 'job-1',
        name: 'test-job',
        dataset_key: 'ds',
        base_model_id: 'm',
        metric_name: 'score',
        metric_mode: 'max',
        status: 'completed',
        best_run_id: 'run-123',
        created_at: '',
        search_space_json: {}
    }
    vi.mocked(client.listTuningJobs).mockResolvedValue([job as any])
    vi.mocked(client.promoteRunToVersion).mockResolvedValue({} as any)

    // Mock window interactions
    vi.spyOn(window, 'alert').mockImplementation(() => {})
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    vi.spyOn(window, 'prompt').mockReturnValue('artifact-id')

    render(<Tuning />)

    // Wait for load
    await waitFor(() => expect(screen.getByText('test-job')).toBeInTheDocument())

    // Click Promote
    fireEvent.click(screen.getByText('Promote Best'))

    await waitFor(() => {
        expect(client.promoteRunToVersion).toHaveBeenCalledWith('artifact-id', {
            source_run_id: 'run-123',
            version_label: 'tuning-job-1'
        })
        expect(window.alert).toHaveBeenCalledWith('Promotion successful! Check Model Registry.')
    })
  })

  it('handles promotion failure', async () => {
    const job = {
        id: 'job-fail',
        name: 'fail-job',
        dataset_key: 'ds',
        base_model_id: 'm',
        metric_name: 'score',
        metric_mode: 'max',
        status: 'completed',
        best_run_id: 'run-fail',
        created_at: '',
        search_space_json: {}
    }
    vi.mocked(client.listTuningJobs).mockResolvedValue([job as any])
    vi.mocked(client.promoteRunToVersion).mockRejectedValue(new Error('API Error'))

    vi.spyOn(window, 'alert').mockImplementation(() => {})
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    vi.spyOn(window, 'prompt').mockReturnValue('artifact-id')

    render(<Tuning />)

    await waitFor(() => expect(screen.getByText('fail-job')).toBeInTheDocument())

    fireEvent.click(screen.getByText('Promote Best'))

    await waitFor(() => {
        expect(window.alert).toHaveBeenCalledWith('Promotion failed: API Error')
    })
  })
})
