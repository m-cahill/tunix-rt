import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import App from './App'

// Mock fetch globally
global.fetch = vi.fn()

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders heading', () => {
    render(<App />)
    expect(screen.getByText('Tunix RT')).toBeInTheDocument()
  })

  it('displays API healthy status', async () => {
    // Mock both fetch calls
    (global.fetch as any)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'healthy' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'healthy' }),
      })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByTestId('api-status')).toHaveTextContent('API: healthy')
    })
  })

  it('displays RediAI healthy status', async () => {
    (global.fetch as any)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'healthy' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'healthy' }),
      })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByTestId('redi-status')).toHaveTextContent('RediAI: healthy')
    })
  })

  it('displays RediAI down status with error', async () => {
    (global.fetch as any)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'healthy' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'down', error: 'Connection refused' }),
      })

    render(<App />)

    await waitFor(() => {
      const rediStatus = screen.getByTestId('redi-status')
      expect(rediStatus).toHaveTextContent('RediAI: down')
      expect(screen.getByText(/Connection refused/)).toBeInTheDocument()
    })
  })

  it('handles fetch errors gracefully', async () => {
    (global.fetch as any)
      .mockRejectedValueOnce(new Error('Network error'))
      .mockRejectedValueOnce(new Error('Network error'))

    render(<App />)

    await waitFor(() => {
      expect(screen.getByTestId('api-status')).toHaveTextContent('API: down')
      expect(screen.getByTestId('redi-status')).toHaveTextContent('RediAI: down')
    })
  })

  it('populates textarea when Load Example is clicked', async () => {
    const user = userEvent.setup()
    
    // Mock health checks
    ;(global.fetch as any)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'healthy' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'healthy' }),
      })

    render(<App />)

    const loadExampleButton = screen.getByText('Load Example')
    await user.click(loadExampleButton)

    const textarea = screen.getByPlaceholderText(/Enter trace JSON/i) as HTMLTextAreaElement
    expect(textarea.value).toContain('Convert 68°F to Celsius')
    expect(textarea.value).toContain('20°C')
  })

  it('uploads trace successfully and displays trace ID', async () => {
    const user = userEvent.setup()
    const mockTraceId = '550e8400-e29b-41d4-a716-446655440000'
    
    // Mock health checks + upload
    ;(global.fetch as any)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'healthy' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'healthy' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 201,
        json: async () => ({
          id: mockTraceId,
          created_at: '2025-12-21T10:30:00Z',
          trace_version: '1.0',
        }),
      })

    render(<App />)

    // Load example trace
    const loadExampleButton = screen.getByText('Load Example')
    await user.click(loadExampleButton)

    // Upload
    const uploadButton = screen.getByText('Upload')
    await user.click(uploadButton)

    await waitFor(() => {
      expect(screen.getByText(/Success!/i)).toBeInTheDocument()
      expect(screen.getByText(new RegExp(mockTraceId))).toBeInTheDocument()
    })
  })

  it('fetches trace successfully and displays JSON', async () => {
    const user = userEvent.setup()
    const mockTraceId = '550e8400-e29b-41d4-a716-446655440000'
    const mockTraceDetail = {
      id: mockTraceId,
      created_at: '2025-12-21T10:30:00Z',
      trace_version: '1.0',
      payload: {
        trace_version: '1.0',
        prompt: 'Convert 68°F to Celsius',
        final_answer: '20°C',
        steps: [
          { i: 0, type: 'parse', content: 'Parse the temperature conversion task' },
        ],
        meta: { source: 'example' },
      },
    }

    // Mock health checks + upload + fetch
    ;(global.fetch as any)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'healthy' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'healthy' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 201,
        json: async () => ({
          id: mockTraceId,
          created_at: '2025-12-21T10:30:00Z',
          trace_version: '1.0',
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockTraceDetail,
      })

    render(<App />)

    // Load example, upload, then fetch
    const loadExampleButton = screen.getByText('Load Example')
    await user.click(loadExampleButton)

    const uploadButton = screen.getByText('Upload')
    await user.click(uploadButton)

    await waitFor(() => {
      expect(screen.getByText(/Success!/i)).toBeInTheDocument()
    })

    const fetchButton = screen.getByText('Fetch')
    await user.click(fetchButton)

    await waitFor(() => {
      expect(screen.getByText(/Fetched Trace/i)).toBeInTheDocument()
      // Check that the JSON payload is rendered in a pre element
      const preElement = screen.getByRole('heading', { name: /Fetched Trace/i }).nextElementSibling as HTMLPreElement
      expect(preElement.tagName).toBe('PRE')
      expect(preElement.textContent).toContain(mockTraceId)
      expect(preElement.textContent).toContain('Convert 68°F to Celsius')
    })
  })
})

