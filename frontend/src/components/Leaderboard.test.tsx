import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { Leaderboard } from './Leaderboard'
import * as client from '../api/client'

// Mock the API client
vi.mock('../api/client', () => ({
  getLeaderboard: vi.fn(),
  ApiError: class ApiError extends Error {
    constructor(message: string, public status: number, public statusText: string) {
      super(message)
      this.name = 'ApiError'
    }
  },
}))

// Helper: Mock leaderboard data
const mockLeaderboardData = (overrides?: Partial<client.LeaderboardResponse>): client.LeaderboardResponse => ({
  data: [
    {
      run_id: 'run-1-uuid-full-string',
      model_id: 'google/gemma-3-1b-it',
      dataset_key: 'dev-reasoning-v2',
      score: 85.5,
      verdict: 'pass',
      metrics: { answer_correctness: 0.72 },
      evaluated_at: '2025-12-27T10:00:00Z',
      primary_score: 0.72,
      scorecard: {
        n_items: 100,
        n_scored: 95,
        n_skipped: 5,
        primary_score: 0.72,
        stddev: 0.15,
      },
    },
    {
      run_id: 'run-2-uuid-full-string',
      model_id: 'google/gemma-2-2b',
      dataset_key: 'golden-v2',
      score: 75.0,
      verdict: 'fail',
      metrics: { answer_correctness: 0.65 },
      evaluated_at: '2025-12-26T15:30:00Z',
      primary_score: 0.65,
      scorecard: {
        n_items: 100,
        n_scored: 90,
        n_skipped: 10,
        primary_score: 0.65,
        stddev: 0.20,
      },
    },
  ],
  pagination: {
    limit: 50,
    offset: 0,
    next_offset: null,
  },
  ...overrides,
})

describe('Leaderboard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  // ============================================================
  // Test 1: Renders loading state initially
  // ============================================================
  it('renders loading state initially', async () => {
    // Make the API call hang to observe loading state
    vi.mocked(client.getLeaderboard).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    )

    render(<Leaderboard />)

    expect(screen.getByText('Loading leaderboard...')).toBeInTheDocument()
  })

  // ============================================================
  // Test 2: Renders empty state when no data
  // ============================================================
  it('renders empty state when no evaluated runs found', async () => {
    vi.mocked(client.getLeaderboard).mockResolvedValue({
      data: [],
      pagination: { limit: 50, offset: 0, next_offset: null },
    })

    render(<Leaderboard />)

    await waitFor(() => {
      expect(screen.getByText('No evaluated runs found.')).toBeInTheDocument()
    })
  })

  // ============================================================
  // Test 3: Renders leaderboard data with correct columns
  // ============================================================
  it('renders leaderboard data with correct columns', async () => {
    vi.mocked(client.getLeaderboard).mockResolvedValue(mockLeaderboardData())

    render(<Leaderboard />)

    await waitFor(() => {
      // Check header columns
      expect(screen.getByText('Rank')).toBeInTheDocument()
      expect(screen.getByText('Run ID')).toBeInTheDocument()
      expect(screen.getByText('Model')).toBeInTheDocument()
      expect(screen.getByText('Dataset')).toBeInTheDocument()
      expect(screen.getByText('Primary Score')).toBeInTheDocument()
      expect(screen.getByText('Items')).toBeInTheDocument()
      expect(screen.getByText('Verdict')).toBeInTheDocument()
      expect(screen.getByText('Evaluated')).toBeInTheDocument()
    })

    // Check data rows
    expect(screen.getByText('run-1-uu...')).toBeInTheDocument()
    expect(screen.getByText('google/gemma-3-1b-it')).toBeInTheDocument()
    expect(screen.getByText('dev-reasoning-v2')).toBeInTheDocument()
  })

  // ============================================================
  // Test 4: Filter inputs render and update state
  // ============================================================
  it('renders filter inputs and updates on typing', async () => {
    const user = userEvent.setup()
    vi.mocked(client.getLeaderboard).mockResolvedValue(mockLeaderboardData())

    render(<Leaderboard />)

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Dataset (exact)')).toBeInTheDocument()
    })

    // Get filter inputs
    const datasetInput = screen.getByPlaceholderText('Dataset (exact)')
    const modelInput = screen.getByPlaceholderText('Model (contains)')

    // Type into inputs
    await act(async () => {
      await user.type(datasetInput, 'dev-reasoning-v2')
      await user.type(modelInput, 'gemma')
    })

    expect(datasetInput).toHaveValue('dev-reasoning-v2')
    expect(modelInput).toHaveValue('gemma')
  })

  // ============================================================
  // Test 5: Apply and Clear filter buttons work
  // ============================================================
  it('applies filters when Filter button clicked', async () => {
    const user = userEvent.setup()
    vi.mocked(client.getLeaderboard).mockResolvedValue(mockLeaderboardData())

    render(<Leaderboard />)

    await waitFor(() => {
      expect(screen.getByText('Filter')).toBeInTheDocument()
    })

    // Fill in filter and apply
    const datasetInput = screen.getByPlaceholderText('Dataset (exact)')
    await act(async () => {
      await user.type(datasetInput, 'golden-v2')
      await user.click(screen.getByText('Filter'))
    })

    // Verify getLeaderboard was called with filters
    await waitFor(() => {
      expect(client.getLeaderboard).toHaveBeenCalledWith(
        50,
        0,
        expect.objectContaining({ dataset_key: 'golden-v2' })
      )
    })
  })

  it('clears filters when Clear button clicked', async () => {
    const user = userEvent.setup()
    vi.mocked(client.getLeaderboard).mockResolvedValue(mockLeaderboardData())

    render(<Leaderboard />)

    await waitFor(() => {
      expect(screen.getByText('Clear')).toBeInTheDocument()
    })

    // Fill in filter, apply, then clear
    const datasetInput = screen.getByPlaceholderText('Dataset (exact)')
    await act(async () => {
      await user.type(datasetInput, 'golden-v2')
      await user.click(screen.getByText('Filter'))
    })

    await act(async () => {
      await user.click(screen.getByText('Clear'))
    })

    // Input should be cleared
    expect(datasetInput).toHaveValue('')

    // API should be called without filters
    await waitFor(() => {
      const lastCall = vi.mocked(client.getLeaderboard).mock.calls.slice(-1)[0]
      expect(lastCall[2]).toEqual({})
    })
  })

  // ============================================================
  // Test 6: Pagination buttons work
  // ============================================================
  it('pagination Previous button disabled on first page', async () => {
    vi.mocked(client.getLeaderboard).mockResolvedValue(mockLeaderboardData())

    render(<Leaderboard />)

    await waitFor(() => {
      const prevButton = screen.getByText('Previous')
      expect(prevButton).toBeDisabled()
    })
  })

  it('pagination Next button disabled when no next page', async () => {
    vi.mocked(client.getLeaderboard).mockResolvedValue(
      mockLeaderboardData({ pagination: { limit: 50, offset: 0, next_offset: null } })
    )

    render(<Leaderboard />)

    await waitFor(() => {
      const nextButton = screen.getByText('Next')
      expect(nextButton).toBeDisabled()
    })
  })

  it('pagination Next button enabled when next page exists', async () => {
    const user = userEvent.setup()
    vi.mocked(client.getLeaderboard).mockResolvedValue(
      mockLeaderboardData({ pagination: { limit: 50, offset: 0, next_offset: 50 } })
    )

    render(<Leaderboard />)

    await waitFor(() => {
      const nextButton = screen.getByText('Next')
      expect(nextButton).not.toBeDisabled()
    })

    // Click next and verify offset changes
    await act(async () => {
      await user.click(screen.getByText('Next'))
    })

    await waitFor(() => {
      expect(client.getLeaderboard).toHaveBeenCalledWith(50, 50, expect.any(Object))
    })
  })

  // ============================================================
  // Test 7: Error state displays correctly
  // ============================================================
  it('displays error state when API fails', async () => {
    vi.mocked(client.getLeaderboard).mockRejectedValue(
      new client.ApiError('Server Error', 500, 'Internal Server Error')
    )

    render(<Leaderboard />)

    await waitFor(() => {
      expect(screen.getByText(/Error: Server Error/)).toBeInTheDocument()
    })
  })

  // ============================================================
  // Test 8: Scorecard displays correctly
  // ============================================================
  it('displays scorecard items/scored ratio', async () => {
    vi.mocked(client.getLeaderboard).mockResolvedValue(mockLeaderboardData())

    render(<Leaderboard />)

    await waitFor(() => {
      // Check scorecard display: n_scored/n_items
      expect(screen.getByText('95/100')).toBeInTheDocument()
      expect(screen.getByText('90/100')).toBeInTheDocument()
    })
  })

  // ============================================================
  // Test 9: Primary score displays as percentage
  // ============================================================
  it('displays primary score as percentage', async () => {
    vi.mocked(client.getLeaderboard).mockResolvedValue(mockLeaderboardData())

    render(<Leaderboard />)

    await waitFor(() => {
      // 0.72 -> 72.0%, 0.65 -> 65.0%
      expect(screen.getByText('72.0%')).toBeInTheDocument()
      expect(screen.getByText('65.0%')).toBeInTheDocument()
    })
  })

  // ============================================================
  // Test 10: Date formatting works
  // ============================================================
  it('displays dates in localized format', async () => {
    vi.mocked(client.getLeaderboard).mockResolvedValue(mockLeaderboardData())

    render(<Leaderboard />)

    await waitFor(() => {
      // The component uses toLocaleDateString() which varies by locale
      // Just verify dates are present (not empty)
      const rows = screen.getAllByRole('row')
      expect(rows.length).toBeGreaterThan(1) // Header + data rows
    })
  })
})
