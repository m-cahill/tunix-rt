import { useEffect, useState, useRef } from 'react'

interface LogChunk {
  seq: number
  stream: string
  chunk: string
  created_at: string
}

interface LiveLogsProps {
  runId: string
  initialStatus: string
  onStatusChange?: (status: string) => void
}

export function LiveLogs({ runId, initialStatus, onStatusChange }: LiveLogsProps) {
  const [logs, setLogs] = useState<LogChunk[]>([])
  const [status, setStatus] = useState(initialStatus)
  const [connected, setConnected] = useState(false)
  const logContainerRef = useRef<HTMLPreElement>(null)
  const eventSourceRef = useRef<EventSource | null>(null)
  const lastSeqRef = useRef(0)

  useEffect(() => {
    // If status is already terminal, don't connect unless we want to replay?
    // Requirement: "show “Live Logs” tab... append log lines as they arrive"
    // Also "replay logs later".
    // If terminal, we can still fetch logs via SSE (it will stream all and close).
    // So we should connect regardless, but maybe stop reconnecting if it closes.

    // Actually, if it's completed, getting logs via SSE is fine, it will dump all and close.
    // But `getTunixRun` returns full stdout/stderr.
    // Maybe LiveLogs is only for active runs?
    // "In run detail panel: show “Live Logs” tab"
    // "replace “wait + refresh” feel"
    // I will enable it for all, but maybe optimize if already completed (just show static).
    // For now, let's use SSE for everything to be consistent, or just for pending/running.

    // If passed status is terminal, maybe we shouldn't use LiveLogs but the static view?
    // The parent `App.tsx` shows static view if `selectedRunDetail` is loaded.
    // I will use LiveLogs when `selectedRunDetail` is NOT fully loaded or status is running.
    // Actually, `App.tsx` loads `selectedRunDetail` which has `stdout`.
    // So if status is running, `stdout` might be partial.
    // I will render `LiveLogs` INSTEAD of static details if status is running.

    const connect = () => {
      // Use relative URL, Vite proxy handles it
      // Note: In dev, Vite proxy might not handle EventSource correctly if not configured,
      // but usually it works.
      // If backend is on 8000 and frontend 5173.
      const url = `/api/tunix/runs/${runId}/logs?since_seq=${lastSeqRef.current}`

      const es = new EventSource(url)
      eventSourceRef.current = es

      es.onopen = () => setConnected(true)

      es.addEventListener('log', (e) => {
        const data = JSON.parse(e.data) as LogChunk
        setLogs(prev => {
            if (data.seq <= lastSeqRef.current) return prev
            return [...prev, data]
        })
        lastSeqRef.current = Math.max(lastSeqRef.current, data.seq)
      })

      es.addEventListener('status', (e) => {
        const data = JSON.parse(e.data)
        if (data.status) {
            setStatus(data.status)
            onStatusChange?.(data.status)
            if (['completed', 'failed', 'timeout', 'cancelled'].includes(data.status)) {
                es.close()
                setConnected(false)
            }
        }
      })

      es.addEventListener('heartbeat', () => {
          // keepalive
      })

      es.onerror = () => {
        setConnected(false)
        es.close()
      }
    }

    connect()

    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
      }
    }
  }, [runId]) // Dependencies: runId. status change shouldn't trigger reconnect.

  // Auto-scroll effect
  useEffect(() => {
    if (logContainerRef.current) {
        logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs])

  return (
    <div className="live-logs" data-testid="live-logs:container">
      <div className="logs-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px' }}>
        <h4>Live Logs</h4>
        <div>
            <span className={`status-badge status-${status}`} style={{ marginRight: '10px' }}>{status}</span>
            <span className="connection-status" style={{ fontSize: '0.8em', color: connected ? 'green' : 'gray' }}>
                {connected ? '● Live' : '○ Disconnected'}
            </span>
        </div>
      </div>
      <pre
        ref={logContainerRef}
        className="logs-content"
        data-testid="live-logs:content"
        style={{
            maxHeight: '400px',
            overflow: 'auto',
            backgroundColor: '#1e1e1e',
            color: '#d4d4d4',
            padding: '10px',
            borderRadius: '4px',
            fontFamily: 'monospace',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all'
        }}
      >
        {logs.length === 0 && <div style={{color: '#666'}}>Waiting for logs...</div>}
        {logs.map((log) => (
          <span key={log.seq} style={{ color: log.stream === 'stderr' ? '#ff6b6b' : 'inherit' }}>
            {log.chunk}
          </span>
        ))}
      </pre>
    </div>
  )
}
