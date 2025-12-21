import { useEffect, useState } from 'react'
import { 
  getApiHealth, 
  getRediHealth, 
  createTrace,
  getTrace,
  compareTraces,
  getUngarStatus,
  generateUngarTraces,
  type HealthResponse, 
  type RediHealthResponse,
  type TraceDetail,
  type CompareResponse,
  type UngarStatusResponse,
  type UngarGenerateResponse,
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
  
  // UNGAR state
  const [ungarStatus, setUngarStatus] = useState<UngarStatusResponse | null>(null)
  const [ungarCount, setUngarCount] = useState('5')
  const [ungarSeed, setUngarSeed] = useState('42')
  const [ungarResult, setUngarResult] = useState<UngarGenerateResponse | null>(null)
  const [ungarError, setUngarError] = useState<string | null>(null)
  const [ungarLoading, setUngarLoading] = useState(false)

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

      try {
        // Fetch UNGAR status
        const ungarData = await getUngarStatus()
        setUngarStatus(ungarData)
      } catch (error) {
        setUngarStatus({ available: false, version: null })
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

  const handleUngarGenerate = async () => {
    const count = parseInt(ungarCount, 10)
    const seed = ungarSeed ? parseInt(ungarSeed, 10) : null
    
    if (isNaN(count) || count < 1 || count > 100) {
      setUngarError('Count must be between 1 and 100')
      return
    }
    
    setUngarError(null)
    setUngarLoading(true)
    
    try {
      const result = await generateUngarTraces({ count, seed, persist: true })
      setUngarResult(result)
      setUngarError(null)
    } catch (error) {
      if (error instanceof ApiError) {
        setUngarError(`Generation failed: ${error.message} (${error.status})`)
      } else {
        setUngarError('Generation failed: Unknown error')
      }
      setUngarResult(null)
    } finally {
      setUngarLoading(false)
    }
  }

  return (
    <div>
      <h1>Tunix RT</h1>
      <p>Reasoning-Trace Framework with RediAI Integration</p>

      <div className={`status-card ${getStatusClass(apiHealth?.status)}`} data-testid="sys:api-card">
        <h2>API Status</h2>
        {loading ? (
          <p>Loading...</p>
        ) : (
          <>
            <p data-testid="sys:api-status">
              API: {apiHealth?.status || 'unknown'}
            </p>
          </>
        )}
      </div>

      <div className={`status-card ${getStatusClass(rediHealth?.status)}`} data-testid="sys:redi-card">
        <h2>RediAI Integration</h2>
        {loading ? (
          <p>Loading...</p>
        ) : (
          <>
            <p data-testid="sys:redi-status">
              RediAI: {rediHealth?.status || 'unknown'}
            </p>
            {rediHealth?.error && <p data-testid="sys:redi-error">Error: {rediHealth.error}</p>}
          </>
        )}
      </div>

      <div className="trace-section" data-testid="trace:section">
        <h2>Reasoning Traces</h2>
        
        <div className="trace-input" data-testid="trace:input-container">
          <label htmlFor="trace-json">Trace JSON:</label>
          <textarea
            id="trace-json"
            data-testid="trace:json"
            value={traceInput}
            onChange={(e) => setTraceInput(e.target.value)}
            placeholder="Enter trace JSON here or click 'Load Example'"
            rows={10}
          />
          
          <div className="trace-actions" data-testid="trace:actions">
            <button data-testid="trace:load-example" onClick={handleLoadExample} disabled={traceLoading}>
              Load Example
            </button>
            <button data-testid="trace:upload" onClick={handleUpload} disabled={traceLoading || !traceInput}>
              Upload
            </button>
            <button data-testid="trace:fetch" onClick={handleFetch} disabled={traceLoading || !uploadedTraceId}>
              Fetch
            </button>
          </div>
        </div>

        {traceError && (
          <div className="trace-error" data-testid="trace:error">
            <strong>Error:</strong> {traceError}
          </div>
        )}

        {uploadedTraceId && !traceError && (
          <div className="trace-success" data-testid="trace:success">
            <strong>Success!</strong> Trace uploaded with ID: {uploadedTraceId}
          </div>
        )}

        {fetchedTrace && (
          <div className="trace-result" data-testid="trace:result">
            <h3>Fetched Trace</h3>
            <pre data-testid="trace:result-content">{JSON.stringify(fetchedTrace, null, 2)}</pre>
          </div>
        )}
      </div>

      <div className="evaluation-section" data-testid="compare:section">
        <h2>Evaluate Traces</h2>
        
        <div className="comparison-input" data-testid="compare:input-container">
          <div className="input-group">
            <label htmlFor="base-trace-id">Base Trace ID:</label>
            <input
              id="base-trace-id"
              data-testid="compare:base-id"
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
              data-testid="compare:other-id"
              type="text"
              value={otherTraceId}
              onChange={(e) => setOtherTraceId(e.target.value)}
              placeholder="Enter other trace UUID"
            />
          </div>
          
          <button 
            data-testid="compare:submit"
            onClick={handleCompare} 
            disabled={compareLoading || !baseTraceId || !otherTraceId}
          >
            Fetch & Compare
          </button>
        </div>

        {compareError && (
          <div className="trace-error" data-testid="compare:error">
            <strong>Error:</strong> {compareError}
          </div>
        )}

        {compareResult && (
          <div className="comparison-result" data-testid="compare:result">
            <div className="comparison-columns" data-testid="compare:columns">
              <div className="comparison-column" data-testid="compare:base-column">
                <h3>Base Trace</h3>
                <div className="trace-score" data-testid="compare:base-score">
                  <strong>Score:</strong> {compareResult.base.score.toFixed(2)}
                </div>
                <div className="trace-metadata">
                  <p><strong>ID:</strong> {compareResult.base.id}</p>
                  <p><strong>Created:</strong> {new Date(compareResult.base.created_at).toLocaleString()}</p>
                  <p><strong>Version:</strong> {compareResult.base.trace_version}</p>
                </div>
                <div className="trace-content" data-testid="compare:base-content">
                  <h4>Prompt:</h4>
                  <p data-testid="compare:base-prompt">{compareResult.base.payload.prompt}</p>
                  <h4>Final Answer:</h4>
                  <p data-testid="compare:base-answer">{compareResult.base.payload.final_answer}</p>
                  <h4>Steps:</h4>
                  <ul data-testid="compare:base-steps">
                    {compareResult.base.payload.steps.map((step) => (
                      <li key={step.i} data-testid={`compare:base-step-${step.i}`}>
                        <strong>{step.type}:</strong> {step.content}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="comparison-column" data-testid="compare:other-column">
                <h3>Other Trace</h3>
                <div className="trace-score" data-testid="compare:other-score">
                  <strong>Score:</strong> {compareResult.other.score.toFixed(2)}
                </div>
                <div className="trace-metadata">
                  <p><strong>ID:</strong> {compareResult.other.id}</p>
                  <p><strong>Created:</strong> {new Date(compareResult.other.created_at).toLocaleString()}</p>
                  <p><strong>Version:</strong> {compareResult.other.trace_version}</p>
                </div>
                <div className="trace-content" data-testid="compare:other-content">
                  <h4>Prompt:</h4>
                  <p data-testid="compare:other-prompt">{compareResult.other.payload.prompt}</p>
                  <h4>Final Answer:</h4>
                  <p data-testid="compare:other-answer">{compareResult.other.payload.final_answer}</p>
                  <h4>Steps:</h4>
                  <ul data-testid="compare:other-steps">
                    {compareResult.other.payload.steps.map((step) => (
                      <li key={step.i} data-testid={`compare:other-step-${step.i}`}>
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

      <div className="ungar-section" data-testid="ungar:section">
        <h2>UNGAR Generator (Optional)</h2>
        
        <div className="ungar-status" data-testid="ungar:status-container">
          <p data-testid="ungar:status">
            <strong>Status:</strong> {loading ? 'Loading...' : (ungarStatus?.available ? '✅ Available' : '❌ Not Installed')}
          </p>
          {ungarStatus?.version && (
            <p data-testid="ungar:version">
              <strong>Version:</strong> {ungarStatus.version}
            </p>
          )}
        </div>

        {ungarStatus?.available && (
          <div className="ungar-generator" data-testid="ungar:generator">
            <div className="input-group">
              <label htmlFor="ungar-count">Trace Count (1-100):</label>
              <input
                id="ungar-count"
                data-testid="ungar:generate-count"
                type="number"
                min="1"
                max="100"
                value={ungarCount}
                onChange={(e) => setUngarCount(e.target.value)}
                disabled={ungarLoading}
              />
            </div>
            
            <div className="input-group">
              <label htmlFor="ungar-seed">Random Seed (optional):</label>
              <input
                id="ungar-seed"
                data-testid="ungar:generate-seed"
                type="number"
                value={ungarSeed}
                onChange={(e) => setUngarSeed(e.target.value)}
                disabled={ungarLoading}
                placeholder="42"
              />
            </div>
            
            <button
              data-testid="ungar:generate-btn"
              onClick={handleUngarGenerate}
              disabled={ungarLoading}
            >
              Generate High Card Duel Traces
            </button>
          </div>
        )}

        {ungarError && (
          <div className="trace-error" data-testid="ungar:error">
            <strong>Error:</strong> {ungarError}
          </div>
        )}

        {ungarResult && (
          <div className="ungar-results" data-testid="ungar:results">
            <h3>Generated {ungarResult.trace_ids.length} Traces</h3>
            <p><strong>Trace IDs:</strong></p>
            <ul data-testid="ungar:trace-ids">
              {ungarResult.trace_ids.map((id) => (
                <li key={id} data-testid={`ungar:trace-id-${id}`}>{id}</li>
              ))}
            </ul>
            {ungarResult.preview.length > 0 && (
              <div data-testid="ungar:preview">
                <p><strong>Preview:</strong></p>
                <ul>
                  {ungarResult.preview.map((p, idx) => (
                    <li key={idx}>
                      Game: {p.game}, Result: {p.result}, Card: {p.my_card}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App

