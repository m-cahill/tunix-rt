import { useEffect, useState } from 'react'
import { 
  getApiHealth, 
  getRediHealth, 
  createTrace,
  getTrace,
  compareTraces,
  type HealthResponse, 
  type RediHealthResponse,
  type TraceDetail,
  type CompareResponse,
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
  
  // Comparison state
  const [baseTraceId, setBaseTraceId] = useState('')
  const [otherTraceId, setOtherTraceId] = useState('')
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null)
  const [compareError, setCompareError] = useState<string | null>(null)
  const [compareLoading, setCompareLoading] = useState(false)

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

  const handleCompare = async () => {
    if (!baseTraceId || !otherTraceId) {
      setCompareError('Both trace IDs are required')
      return
    }
    
    setCompareError(null)
    setCompareLoading(true)
    
    try {
      const result = await compareTraces(baseTraceId, otherTraceId)
      setCompareResult(result)
      setCompareError(null)
    } catch (error) {
      if (error instanceof ApiError) {
        setCompareError(`Compare failed: ${error.message} (${error.status})`)
      } else {
        setCompareError('Compare failed: Unknown error')
      }
      setCompareResult(null)
    } finally {
      setCompareLoading(false)
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

      <div className="evaluation-section">
        <h2>Evaluate Traces</h2>
        
        <div className="comparison-input">
          <div className="input-group">
            <label htmlFor="base-trace-id">Base Trace ID:</label>
            <input
              id="base-trace-id"
              type="text"
              value={baseTraceId}
              onChange={(e) => setBaseTraceId(e.target.value)}
              placeholder="Enter base trace UUID"
            />
          </div>
          
          <div className="input-group">
            <label htmlFor="other-trace-id">Other Trace ID:</label>
            <input
              id="other-trace-id"
              type="text"
              value={otherTraceId}
              onChange={(e) => setOtherTraceId(e.target.value)}
              placeholder="Enter other trace UUID"
            />
          </div>
          
          <button 
            onClick={handleCompare} 
            disabled={compareLoading || !baseTraceId || !otherTraceId}
          >
            Fetch & Compare
          </button>
        </div>

        {compareError && (
          <div className="trace-error">
            <strong>Error:</strong> {compareError}
          </div>
        )}

        {compareResult && (
          <div className="comparison-result">
            <div className="comparison-columns">
              <div className="comparison-column">
                <h3>Base Trace</h3>
                <div className="trace-score">
                  <strong>Score:</strong> {compareResult.base.score.toFixed(2)}
                </div>
                <div className="trace-metadata">
                  <p><strong>ID:</strong> {compareResult.base.id}</p>
                  <p><strong>Created:</strong> {new Date(compareResult.base.created_at).toLocaleString()}</p>
                  <p><strong>Version:</strong> {compareResult.base.trace_version}</p>
                </div>
                <div className="trace-content">
                  <h4>Prompt:</h4>
                  <p>{compareResult.base.payload.prompt}</p>
                  <h4>Final Answer:</h4>
                  <p>{compareResult.base.payload.final_answer}</p>
                  <h4>Steps:</h4>
                  <ul>
                    {compareResult.base.payload.steps.map((step) => (
                      <li key={step.i}>
                        <strong>{step.type}:</strong> {step.content}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="comparison-column">
                <h3>Other Trace</h3>
                <div className="trace-score">
                  <strong>Score:</strong> {compareResult.other.score.toFixed(2)}
                </div>
                <div className="trace-metadata">
                  <p><strong>ID:</strong> {compareResult.other.id}</p>
                  <p><strong>Created:</strong> {new Date(compareResult.other.created_at).toLocaleString()}</p>
                  <p><strong>Version:</strong> {compareResult.other.trace_version}</p>
                </div>
                <div className="trace-content">
                  <h4>Prompt:</h4>
                  <p>{compareResult.other.payload.prompt}</p>
                  <h4>Final Answer:</h4>
                  <p>{compareResult.other.payload.final_answer}</p>
                  <h4>Steps:</h4>
                  <ul>
                    {compareResult.other.payload.steps.map((step) => (
                      <li key={step.i}>
                        <strong>{step.type}:</strong> {step.content}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

