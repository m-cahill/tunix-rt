import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { RunComparison } from './RunComparison'
import * as client from '../api/client'

// Mock API client
vi.mock('../api/client', () => ({
  getTunixRun: vi.fn(),
  getTunixRunMetrics: vi.fn(),
}))

describe('RunComparison', () => {
  const mockRunA = {
    run_id: 'run-a-1234',
    status: 'completed',
    mode: 'local',
    dataset_key: 'dataset-a',
    model_id: 'model-a',
    duration_seconds: 120.5,
    started_at: '2023-01-01T12:00:00Z',
    message: 'Success'
  }

  const mockRunB = {
    run_id: 'run-b-5678',
    status: 'failed',
    mode: 'local',
    dataset_key: 'dataset-b',
    model_id: 'model-b',
    duration_seconds: 10.2,
    started_at: '2023-01-01T12:05:00Z',
    message: 'Failed'
  }

  const mockMetricsA = [
    { step: 1, loss: 0.5, timestamp: '...' },
    { step: 2, loss: 0.4, timestamp: '...' },
  ]

  const mockMetricsB = [
    { step: 1, loss: 0.8, timestamp: '...' },
  ]

  it('renders loading state initially', () => {
    render(<RunComparison runAId="1" runBId="2" onClose={() => {}} />)
    expect(screen.getAllByText('Loading...')[0]).toBeInTheDocument()
  })

  it('renders comparison data when loaded', async () => {
    vi.mocked(client.getTunixRun)
      .mockResolvedValueOnce(mockRunA as any)
      .mockResolvedValueOnce(mockRunB as any)

    vi.mocked(client.getTunixRunMetrics)
      .mockResolvedValueOnce(mockMetricsA)
      .mockResolvedValueOnce(mockMetricsB)

    render(<RunComparison runAId="1" runBId="2" onClose={() => {}} />)

    await waitFor(() => {
      expect(screen.getByText('Run Comparison')).toBeInTheDocument()
    })

    // Check Run A details
    expect(screen.getByText(/Run A:/)).toBeInTheDocument()
    expect(screen.getByText('model-a')).toBeInTheDocument()
    expect(screen.getByText('dataset-a')).toBeInTheDocument()
    // Duration formatting check (my component uses toFixed(2))
    expect(screen.getByText('120.50s')).toBeInTheDocument()

    // Check Run B details
    expect(screen.getByText(/Run B:/)).toBeInTheDocument()
    expect(screen.getByText('model-b')).toBeInTheDocument()
    expect(screen.getByText('dataset-b')).toBeInTheDocument()

    // Check chart presence (by title)
    expect(screen.getByText('Loss Curve Comparison')).toBeInTheDocument()
  })

  it('handles API errors', async () => {
    vi.mocked(client.getTunixRun).mockRejectedValue(new Error('API Error'))

    render(<RunComparison runAId="1" runBId="2" onClose={() => {}} />)

    await waitFor(() => {
      // API Error might appear twice (for A and B)
      const errors = screen.getAllByText('API Error')
      expect(errors.length).toBeGreaterThan(0)
    })
  })

  it('calls onClose when close button clicked', async () => {
    vi.mocked(client.getTunixRun).mockResolvedValue(mockRunA as any)
    vi.mocked(client.getTunixRunMetrics).mockResolvedValue([])

    const handleClose = vi.fn()
    render(<RunComparison runAId="1" runBId="2" onClose={handleClose} />)

    await waitFor(() => {
      expect(screen.getByText('Run Comparison')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText('Close'))
    expect(handleClose).toHaveBeenCalled()
  })
})
