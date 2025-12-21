import { render, screen, waitFor } from '@testing-library/react'
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
        json: async () => ({ status: 'healthy' }),
      })
      .mockResolvedValueOnce({
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
        json: async () => ({ status: 'healthy' }),
      })
      .mockResolvedValueOnce({
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
        json: async () => ({ status: 'healthy' }),
      })
      .mockResolvedValueOnce({
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
})

