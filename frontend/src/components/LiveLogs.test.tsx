import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { LiveLogs } from './LiveLogs'

// Track mock instances for assertions
let mockEventSourceInstances: any[] = []
let lastMockInstance: any = null

// Mock EventSource
const MockEventSource = vi.fn().mockImplementation((url: string) => {
  const instance = {
    url,
    readyState: 0,
    onopen: null as ((e: any) => void) | null,
    onerror: null as ((e: any) => void) | null,
    eventListeners: new Map<string, ((e: any) => void)[]>(),
    addEventListener: vi.fn((type: string, listener: (e: any) => void) => {
      const listeners = instance.eventListeners.get(type) || []
      listeners.push(listener)
      instance.eventListeners.set(type, listeners)
    }),
    removeEventListener: vi.fn(),
    close: vi.fn(),
    // Helper to trigger events in tests
    _emit: (type: string, data: any) => {
      const listeners = instance.eventListeners.get(type) || []
      listeners.forEach(l => l({ data: JSON.stringify(data) }))
    },
    _connect: () => {
      instance.readyState = 1
      if (instance.onopen) instance.onopen({} as any)
    },
  }

  mockEventSourceInstances.push(instance)
  lastMockInstance = instance

  // Auto-connect in next tick
  setTimeout(() => instance._connect(), 0)

  return instance
})

// Replace global EventSource
const originalEventSource = globalThis.EventSource
beforeEach(() => {
  vi.clearAllMocks()
  mockEventSourceInstances = []
  lastMockInstance = null
  globalThis.EventSource = MockEventSource as any
})

afterEach(() => {
  globalThis.EventSource = originalEventSource
})

describe('LiveLogs', () => {
  // ============================================================
  // Test 1: Renders waiting state initially
  // ============================================================
  it('renders waiting state initially', () => {
    render(<LiveLogs runId="test-run-id" initialStatus="running" />)

    expect(screen.getByText('Waiting for logs...')).toBeInTheDocument()
    expect(screen.getByText('Live Logs')).toBeInTheDocument()
  })

  // ============================================================
  // Test 2: Creates EventSource with correct URL
  // ============================================================
  it('creates EventSource with correct URL', async () => {
    render(<LiveLogs runId="my-run-123" initialStatus="running" />)

    await waitFor(() => {
      expect(MockEventSource).toHaveBeenCalled()
    })

    expect(MockEventSource).toHaveBeenCalledWith(
      expect.stringContaining('/api/tunix/runs/my-run-123/logs')
    )
  })

  // ============================================================
  // Test 3: Connection status indicator
  // ============================================================
  it('shows connection status indicator after connection', async () => {
    render(<LiveLogs runId="test-run-id" initialStatus="running" />)

    // After connection, should show Live
    await waitFor(() => {
      expect(screen.getByText('● Live')).toBeInTheDocument()
    })
  })

  // ============================================================
  // Test 4: Shows initial status badge
  // ============================================================
  it('displays initial status in badge', () => {
    render(<LiveLogs runId="test-run-id" initialStatus="pending" />)

    expect(screen.getByText('pending')).toBeInTheDocument()
  })

  it('displays running status in badge', () => {
    render(<LiveLogs runId="test-run-id" initialStatus="running" />)

    expect(screen.getByText('running')).toBeInTheDocument()
  })

  // ============================================================
  // Test 5: Registers event listeners
  // ============================================================
  it('registers event listeners for log and status events', async () => {
    render(<LiveLogs runId="test-run-id" initialStatus="running" />)

    await waitFor(() => {
      expect(lastMockInstance).not.toBeNull()
    })

    // Check addEventListener was called for expected event types
    expect(lastMockInstance.addEventListener).toHaveBeenCalledWith(
      'log',
      expect.any(Function)
    )
    expect(lastMockInstance.addEventListener).toHaveBeenCalledWith(
      'status',
      expect.any(Function)
    )
  })
})
