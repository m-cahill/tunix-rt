import { useEffect, useState } from 'react'
import { 
  getApiHealth, 
  getRediHealth, 
  createTrace,
  getTrace,
  type HealthResponse, 
  type RediHealthResponse,
  type TraceDetail,
  ApiError,
} from './api/client'
import { EXAMPLE_TRACE } from './exampleTrace'

function App() {
  const [apiHealth, setApiHealth] = useState<HealthResponse | null>(null)
  const [rediHealth, setRediHealth] = useState<RediHealthResponse | null>(null)
  const [loading, setLoading] = useState(true)
  
  // Trace state
  const [traceInput, setTraceInput] = useState('')
  const [uploadedTraceId, setUploadedTraceId] = useState<string | null>(null)
  const [fetchedTrace, setFetchedTrace] = useState<TraceDetail | null>(null)
  const [traceError, setTraceError] = useState<string | null>(null)
  const [traceLoading, setTraceLoading] = useState(false)

  useEffect(() => {
    const fetchHealth = async () => {
      setLoading(true)
      
      try {
        // Fetch API health
        const apiData = await getApiHealth()
        setApiHealth(apiData)
      } catch (error) {
        setApiHealth({ status: 'down' })
      }

      try {
        // Fetch RediAI health
        const rediData = await getRediHealth()
        setRediHealth(rediData)
      } catch (error) {
        setRediHealth({ status: 'down', error: 'Failed to fetch' })
      }

      setLoading(false)
    }

    // Fetch immediately
    fetchHealth()

    // Set up polling every 30 seconds
    const intervalId = setInterval(fetchHealth, 30000)

    // Cleanup on unmount
    return () => {
      clearInterval(intervalId)
    }
  }, [])

  const getStatusClass = (status: string | undefined) => {
    if (!status) return 'status-loading'
    return status === 'healthy' ? 'status-healthy' : 'status-down'
  }

  const handleLoadExample = () => {
    setTraceInput(JSON.stringify(EXAMPLE_TRACE, null, 2))
    setTraceError(null)
  }

  const handleUpload = async () => {
    setTraceError(null)
    setTraceLoading(true)
    
    try {
      const trace = JSON.parse(traceInput)
      const response = await createTrace(trace)
      setUploadedTraceId(response.id)
      setTraceError(null)
    } catch (error) {
      if (error instanceof ApiError) {
        setTraceError(`Upload failed: ${error.message} (${error.status})`)
      } else if (error instanceof SyntaxError) {
        setTraceError('Invalid JSON format')
      } else {
        setTraceError('Upload failed: Unknown error')
      }
    } finally {
      setTraceLoading(false)
    }
  }

  const handleFetch = async () => {
    if (!uploadedTraceId) {
      setTraceError('No trace ID to fetch')
      return
    }
    
    setTraceError(null)
    setTraceLoading(true)
    
    try {
      const trace = await getTrace(uploadedTraceId)
      setFetchedTrace(trace)
      setTraceError(null)
    } catch (error) {
      if (error instanceof ApiError) {
        setTraceError(`Fetch failed: ${error.message} (${error.status})`)
      } else {
        setTraceError('Fetch failed: Unknown error')
      }
      setFetchedTrace(null)
    } finally {
      setTraceLoading(false)
    }
  }

  return (
    <div>
      <h1>Tunix RT</h1>
      <p>Reasoning-Trace Framework with RediAI Integration</p>

      <div className={`status-card ${getStatusClass(apiHealth?.status)}`}>
        <h2>API Status</h2>
        {loading ? (
          <p>Loading...</p>
        ) : (
          <>
            <p data-testid="api-status">
              API: {apiHealth?.status || 'unknown'}
            </p>
          </>
        )}
      </div>

      <div className={`status-card ${getStatusClass(rediHealth?.status)}`}>
        <h2>RediAI Integration</h2>
        {loading ? (
          <p>Loading...</p>
        ) : (
          <>
            <p data-testid="redi-status">
              RediAI: {rediHealth?.status || 'unknown'}
            </p>
            {rediHealth?.error && <p>Error: {rediHealth.error}</p>}
          </>
        )}
      </div>

      <div className="trace-section">
        <h2>Reasoning Traces</h2>
        
        <div className="trace-input">
          <label htmlFor="trace-json">Trace JSON:</label>
          <textarea
            id="trace-json"
            value={traceInput}
            onChange={(e) => setTraceInput(e.target.value)}
            placeholder="Enter trace JSON here or click 'Load Example'"
            rows={10}
          />
          
          <div className="trace-actions">
            <button onClick={handleLoadExample} disabled={traceLoading}>
              Load Example
            </button>
            <button onClick={handleUpload} disabled={traceLoading || !traceInput}>
              Upload
            </button>
            <button onClick={handleFetch} disabled={traceLoading || !uploadedTraceId}>
              Fetch
            </button>
          </div>
        </div>

        {traceError && (
          <div className="trace-error">
            <strong>Error:</strong> {traceError}
          </div>
        )}

        {uploadedTraceId && !traceError && (
          <div className="trace-success">
            <strong>Success!</strong> Trace uploaded with ID: {uploadedTraceId}
          </div>
        )}

        {fetchedTrace && (
          <div className="trace-result">
            <h3>Fetched Trace</h3>
            <pre>{JSON.stringify(fetchedTrace, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

