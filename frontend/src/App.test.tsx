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
    // Mock all three health fetch calls (API, RediAI, UNGAR)
    (global.fetch as any)
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
        json: async () => ({ available: false, version: null }),
      })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByTestId('sys:api-status')).toHaveTextContent('API: healthy')
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
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ available: false, version: null }),
      })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByTestId('sys:redi-status')).toHaveTextContent('RediAI: healthy')
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
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ available: false, version: null }),
      })

    render(<App />)

    await waitFor(() => {
      const rediStatus = screen.getByTestId('sys:redi-status')
      expect(rediStatus).toHaveTextContent('RediAI: down')
      expect(screen.getByText(/Connection refused/)).toBeInTheDocument()
    })
  })

  it('handles fetch errors gracefully', async () => {
    (global.fetch as any)
      .mockRejectedValueOnce(new Error('Network error'))
      .mockRejectedValueOnce(new Error('Network error'))
      .mockRejectedValueOnce(new Error('Network error'))

    render(<App />)

    await waitFor(() => {
      expect(screen.getByTestId('sys:api-status')).toHaveTextContent('API: down')
      expect(screen.getByTestId('sys:redi-status')).toHaveTextContent('RediAI: down')
    })
  })

  it('populates textarea when Load Example is clicked', async () => {
    const user = userEvent.setup()
    
    // Mock health checks (API, RediAI, UNGAR)
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
        json: async () => ({ available: false, version: null }),
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
    
    // Mock health checks (API, RediAI, UNGAR) + upload
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
        json: async () => ({ available: false, version: null }),
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

    // Mock health checks (API, RediAI, UNGAR) + upload + fetch
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
        json: async () => ({ available: false, version: null }),
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

  it('compares two traces successfully and displays side-by-side', async () => {
    const user = userEvent.setup()
    const baseTraceId = '550e8400-e29b-41d4-a716-446655440000'
    const otherTraceId = '660f9500-f39c-52e5-b827-557766551111'
    
    const mockCompareResponse = {
      base: {
        id: baseTraceId,
        created_at: '2025-12-21T10:30:00Z',
        score: 25.5,
        trace_version: '1.0',
        payload: {
          trace_version: '1.0',
          prompt: 'Simple task',
          final_answer: 'Simple answer',
          steps: [
            { i: 0, type: 'think', content: 'Short reasoning' },
          ],
        },
      },
      other: {
        id: otherTraceId,
        created_at: '2025-12-21T10:35:00Z',
        score: 75.8,
        trace_version: '1.0',
        payload: {
          trace_version: '1.0',
          prompt: 'Complex task requiring detailed reasoning',
          final_answer: 'Detailed answer with extensive explanation',
          steps: [
            { i: 0, type: 'analyze', content: 'Deep analysis step' },
            { i: 1, type: 'compute', content: 'Complex computation' },
            { i: 2, type: 'verify', content: 'Verification step' },
          ],
        },
      },
    }

    // Mock health checks (API, RediAI, UNGAR) + compare
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
        json: async () => ({ available: false, version: null }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockCompareResponse,
      })

    render(<App />)

    // Fill in trace IDs
    const baseInput = screen.getByPlaceholderText(/Enter base trace UUID/i) as HTMLInputElement
    const otherInput = screen.getByPlaceholderText(/Enter other trace UUID/i) as HTMLInputElement
    
    await user.type(baseInput, baseTraceId)
    await user.type(otherInput, otherTraceId)

    // Click compare
    const compareButton = screen.getByText('Fetch & Compare')
    await user.click(compareButton)

    await waitFor(() => {
      // Check that both traces are displayed
      expect(screen.getByText('Base Trace')).toBeInTheDocument()
      expect(screen.getByText('Other Trace')).toBeInTheDocument()
      
      // Check scores are displayed
      expect(screen.getByText(/25.50/)).toBeInTheDocument()
      expect(screen.getByText(/75.80/)).toBeInTheDocument()
      
      // Check prompts are displayed
      expect(screen.getByText('Simple task')).toBeInTheDocument()
      expect(screen.getByText('Complex task requiring detailed reasoning')).toBeInTheDocument()
      
      // Check steps are displayed
      expect(screen.getByText(/Short reasoning/)).toBeInTheDocument()
      expect(screen.getByText(/Deep analysis step/)).toBeInTheDocument()
    })
  })

  it('displays error when comparison fails', async () => {
    const user = userEvent.setup()
    const baseTraceId = '550e8400-e29b-41d4-a716-446655440000'
    const otherTraceId = 'invalid-uuid'

    // Mock health checks (API, RediAI, UNGAR) + failed compare
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
        json: async () => ({ available: false, version: null }),
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        json: async () => ({ detail: 'Base trace not found' }),
      })

    render(<App />)

    const baseInput = screen.getByPlaceholderText(/Enter base trace UUID/i) as HTMLInputElement
    const otherInput = screen.getByPlaceholderText(/Enter other trace UUID/i) as HTMLInputElement
    
    await user.type(baseInput, baseTraceId)
    await user.type(otherInput, otherTraceId)

    const compareButton = screen.getByText('Fetch & Compare')
    await user.click(compareButton)

    await waitFor(() => {
      expect(screen.getByText(/Compare failed/i)).toBeInTheDocument()
    })
  })

  it('disables compare button when trace IDs are missing', () => {
    // Mock health checks (API, RediAI, UNGAR)
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
        json: async () => ({ available: false, version: null }),
      })

    render(<App />)

    const compareButton = screen.getByText('Fetch & Compare') as HTMLButtonElement
    expect(compareButton.disabled).toBe(true)
  })

  it('displays UNGAR available status', async () => {
    // Mock health checks with UNGAR available
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
        json: async () => ({ available: true, version: '0.1.0' }),
      })

    render(<App />)

    await waitFor(() => {
      const statusElement = screen.getByTestId('ungar:status')
      expect(statusElement).toHaveTextContent('Status:')
      expect(statusElement).toHaveTextContent('Available')
      expect(screen.getByTestId('ungar:version')).toHaveTextContent('0.1.0')
    })
  })

  it('generates UNGAR traces successfully', async () => {
    const user = userEvent.setup()
    const mockGenerateResponse = {
      trace_ids: ['abc-123', 'def-456'],
      preview: [
        { trace_id: 'abc-123', game: 'HighCardDuel', result: 'win', my_card: 'A' },
        { trace_id: 'def-456', game: 'HighCardDuel', result: 'loss', my_card: '5' },
      ],
    }

    // Mock health checks (API, RediAI, UNGAR available) + generate
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
        json: async () => ({ available: true, version: '0.1.0' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 201,
        json: async () => mockGenerateResponse,
      })

    render(<App />)

    // Wait for UNGAR to be available and button to appear
    await waitFor(() => {
      expect(screen.getByTestId('ungar:status')).toHaveTextContent('Available')
    })

    // Click generate button
    const generateButton = screen.getByTestId('ungar:generate-btn')
    await user.click(generateButton)

    await waitFor(() => {
      expect(screen.getByText(/Generated 2 Traces/i)).toBeInTheDocument()
    })
  })

  it('displays error when UNGAR generation fails', async () => {
    const user = userEvent.setup()

    // Mock health checks (API, RediAI, UNGAR available) + failed generate
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
        json: async () => ({ available: true, version: '0.1.0' }),
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 501,
        statusText: 'Not Implemented',
        json: async () => ({ detail: 'UNGAR not installed' }),
      })

    render(<App />)

    await waitFor(() => {
      expect(screen.getByTestId('ungar:status')).toHaveTextContent('Available')
    })

    const generateButton = screen.getByTestId('ungar:generate-btn')
    await user.click(generateButton)

    await waitFor(() => {
      const errorElement = screen.getByTestId('ungar:error')
      expect(errorElement).toHaveTextContent('Error:')
      expect(errorElement).toHaveTextContent('Generation failed')
      expect(errorElement).toHaveTextContent('Not Implemented')
    })
  })

  it('displays error when trace upload fails', async () => {
    const user = userEvent.setup()

    // Mock health checks (API, RediAI, UNGAR) + failed upload
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
        json: async () => ({ available: false, version: null }),
      })
      .mockResolvedValueOnce({
        ok: false,
        status: 422,
        statusText: 'Unprocessable Entity',
        json: async () => ({ detail: 'Invalid trace format' }),
      })

    render(<App />)

    const loadExampleButton = screen.getByTestId('trace:load-example')
    await user.click(loadExampleButton)

    const uploadButton = screen.getByTestId('trace:upload')
    await user.click(uploadButton)

    await waitFor(() => {
      const errorElement = screen.getByTestId('trace:error')
      expect(errorElement).toHaveTextContent('Error:')
      expect(errorElement).toHaveTextContent('Upload failed')
      expect(errorElement).toHaveTextContent('Unprocessable Entity')
    })
  })

  it('displays error when trace fetch fails', async () => {
    const user = userEvent.setup()
    const mockTraceId = '550e8400-e29b-41d4-a716-446655440000'

    // Mock health checks (API, RediAI, UNGAR) + upload success + fetch failure
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
        json: async () => ({ available: false, version: null }),
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
        ok: false,
        status: 404,
        statusText: 'Not Found',
        json: async () => ({ detail: 'Trace not found' }),
      })

    render(<App />)

    const loadExampleButton = screen.getByTestId('trace:load-example')
    await user.click(loadExampleButton)

    const uploadButton = screen.getByTestId('trace:upload')
    await user.click(uploadButton)

    await waitFor(() => {
      expect(screen.getByTestId('trace:success')).toHaveTextContent('Success!')
    })

    const fetchButton = screen.getByTestId('trace:fetch')
    await user.click(fetchButton)

    await waitFor(() => {
      const errorElement = screen.getByTestId('trace:error')
      expect(errorElement).toHaveTextContent('Error:')
      expect(errorElement).toHaveTextContent('Fetch failed')
      expect(errorElement).toHaveTextContent('Not Found')
    })
  })
})

