import { useEffect, useState, useRef } from 'react'
import {
  getApiHealth,
  getRediHealth,
  createTrace,
  getTrace,
  compareTraces,
  getUngarStatus,
  generateUngarTraces,
  getTunixStatus,
  exportTunixSft,
  generateTunixManifest,
  executeTunixRun,
  listTunixRuns,
  getTunixRun,
  getTunixRunStatus,
  listArtifacts,
  cancelRun,
  getEvaluation,
  evaluateRun,
  getTunixRunMetrics,
  type HealthResponse,
  type RediHealthResponse,
  type TraceDetail,
  type CompareResponse,
  type UngarStatusResponse,
  type UngarGenerateResponse,
  type TunixStatusResponse,
  type TunixManifestResponse,
  type TunixRunResponse,
  type TunixRunListResponse,
  type ArtifactListResponse,
  type EvaluationResponse,
  type TunixRunMetric,
  ApiError,
} from './api/client'
import { EXAMPLE_TRACE } from './exampleTrace'
import { LiveLogs } from './components/LiveLogs'
import { Leaderboard } from './components/Leaderboard'
import { Tuning } from './components/Tuning'
import { ModelRegistry } from './components/ModelRegistry'

function App() {
  const [currentPage, setCurrentPage] = useState<'home' | 'leaderboard' | 'tuning' | 'registry'>('home')

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

  // Tunix state (M12)
  const [tunixStatus, setTunixStatus] = useState<TunixStatusResponse | null>(null)
  const [tunixDatasetKey, setTunixDatasetKey] = useState('')
  const [tunixModelId, setTunixModelId] = useState('google/gemma-2b-it')
  const [tunixOutputDir, setTunixOutputDir] = useState('./output/tunix_run')
  const [tunixManifestResult, setTunixManifestResult] = useState<TunixManifestResponse | null>(null)
  const [tunixRunResult, setTunixRunResult] = useState<TunixRunResponse | null>(null)
  const [tunixError, setTunixError] = useState<string | null>(null)
  const [tunixLoading, setTunixLoading] = useState(false)
  const [tunixRunLoading, setTunixRunLoading] = useState(false)
  const [tunixAsyncMode, setTunixAsyncMode] = useState(false)
  const pollingIntervalRef = useRef<number | null>(null)

  // M14: Run history state
  const [runHistoryExpanded, setRunHistoryExpanded] = useState(false)
  const [runHistory, setRunHistory] = useState<TunixRunListResponse | null>(null)
  const [runHistoryLoading, setRunHistoryLoading] = useState(false)
  const [runHistoryError, setRunHistoryError] = useState<string | null>(null)
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [selectedRunDetail, setSelectedRunDetail] = useState<TunixRunResponse | null>(null)

  // M16: Artifacts state
  const [artifacts, setArtifacts] = useState<ArtifactListResponse | null>(null)
  const [artifactsLoading, setArtifactsLoading] = useState(false)

  // M17: Evaluation state
  const [selectedRunEvaluation, setSelectedRunEvaluation] = useState<EvaluationResponse | null>(null)
  const [evaluationLoading, setEvaluationLoading] = useState(false)

  // M26: Metrics state
  const [runMetrics, setRunMetrics] = useState<TunixRunMetric[] | null>(null)

  useEffect(() => {
    const fetchHealth = async () => {
      // Don't set global loading true on poll, only initial
      if (!apiHealth) setLoading(true)

      try {
        const apiData = await getApiHealth()
        setApiHealth(apiData)
      } catch (error) {
        setApiHealth({ status: 'down' })
      }

      try {
        const rediData = await getRediHealth()
        setRediHealth(rediData)
      } catch (error) {
        setRediHealth({ status: 'down', error: 'Failed to fetch' })
      }

      try {
        const ungarData = await getUngarStatus()
        setUngarStatus(ungarData)
      } catch (error) {
        setUngarStatus({ available: false, version: null })
      }

      try {
        const tunixData = await getTunixStatus()
        setTunixStatus(tunixData)
      } catch (error) {
        setTunixStatus({ available: false, version: null, runtime_required: false, message: 'Failed to fetch status' })
      }

      setLoading(false)
    }

    fetchHealth()
    const intervalId = setInterval(fetchHealth, 30000)

    return () => {
      clearInterval(intervalId)
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
      }
    }
  }, [])

  const startPollingRun = (runId: string) => {
    if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current)

    pollingIntervalRef.current = window.setInterval(async () => {
      try {
        const status = await getTunixRunStatus(runId)

        setTunixRunResult((prev) => {
          if (!prev) return null
          return {
            ...prev,
            status: status.status,
            message: `Async execution: ${status.status}`,
          }
        })

        // Poll metrics if running
        if (status.status === 'running') {
            try {
                const metrics = await getTunixRunMetrics(runId)
                setRunMetrics(metrics)
            } catch (e) {
                console.warn('Metrics poll failed', e)
            }
        }

        if (['completed', 'failed', 'timeout'].includes(status.status)) {
          if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current)
          pollingIntervalRef.current = null

          const result = await getTunixRun(runId)
          setTunixRunResult(result)

          if (runHistoryExpanded) {
            handleRefreshRunHistory()
          }
        }
      } catch (error) {
        if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current)
      }
    }, 4000)
  }

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

  const handleTunixExport = async () => {
    if (!tunixDatasetKey) {
      setTunixError('Dataset key is required')
      return
    }

    setTunixError(null)
    setTunixLoading(true)

    try {
      const blob = await exportTunixSft({ dataset_key: tunixDatasetKey })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${tunixDatasetKey}.jsonl`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      setTunixError(null)
    } catch (error) {
      if (error instanceof ApiError) {
        setTunixError(`Export failed: ${error.message} (${error.status})`)
      } else {
        setTunixError('Export failed: Unknown error')
      }
    } finally {
      setTunixLoading(false)
    }
  }

  const handleTunixManifest = async () => {
    if (!tunixDatasetKey || !tunixModelId || !tunixOutputDir) {
      setTunixError('Dataset key, model ID, and output directory are required')
      return
    }

    setTunixError(null)
    setTunixLoading(true)
    setTunixManifestResult(null)

    try {
      const result = await generateTunixManifest({
        dataset_key: tunixDatasetKey,
        model_id: tunixModelId,
        output_dir: tunixOutputDir,
      })
      setTunixManifestResult(result)
      setTunixError(null)
    } catch (error) {
      if (error instanceof ApiError) {
        setTunixError(`Manifest generation failed: ${error.message} (${error.status})`)
      } else {
        setTunixError('Manifest generation failed: Unknown error')
      }
      setTunixManifestResult(null)
    } finally {
      setTunixLoading(false)
    }
  }

  const handleTunixRun = async (dryRun: boolean) => {
    if (!tunixDatasetKey) {
      setTunixError('Dataset key is required')
      return
    }

    setTunixError(null)
    setTunixLoading(true)
    setTunixRunLoading(true)
    setTunixRunResult(null)

    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
      pollingIntervalRef.current = null
    }

    try {
      const result = await executeTunixRun(
        {
          dataset_key: tunixDatasetKey,
          model_id: tunixModelId,
          output_dir: tunixOutputDir || undefined,
          dry_run: dryRun,
        },
        tunixAsyncMode ? { mode: 'async' } : undefined
      )
      setTunixRunResult(result)
      setTunixError(null)

      if (tunixAsyncMode && result.status === 'pending') {
        startPollingRun(result.run_id)
      } else {
        if (runHistoryExpanded) {
          await handleRefreshRunHistory()
        }
      }
    } catch (error) {
      if (error instanceof ApiError) {
        setTunixError(`Run failed: ${error.message} (${error.status})`)
      } else {
        setTunixError('Run failed: Unknown error')
      }
      setTunixRunResult(null)
    } finally {
      setTunixLoading(false)
      setTunixRunLoading(false)
    }
  }

  const handleRefreshRunHistory = async () => {
    setRunHistoryLoading(true)
    setRunHistoryError(null)

    try {
      const result = await listTunixRuns({ limit: 20, offset: 0 })
      setRunHistory(result)
      setRunHistoryError(null)
    } catch (error) {
      if (error instanceof ApiError) {
        setRunHistoryError(error.message)
      } else {
        setRunHistoryError('Unknown error occurred')
      }
    } finally {
      setRunHistoryLoading(false)
    }
  }

  const handleViewRunDetail = async (runId: string) => {
    if (selectedRunId === runId) {
      setSelectedRunId(null)
      setSelectedRunDetail(null)
      setArtifacts(null)
      setSelectedRunEvaluation(null)
      return
    }

    setSelectedRunId(runId)
    setSelectedRunDetail(null)
    setArtifacts(null)
    setSelectedRunEvaluation(null)
    setRunMetrics(null)
    setRunHistoryError(null)

    try {
      const result = await getTunixRun(runId)
      setSelectedRunDetail(result)
      setRunHistoryError(null)

      // Auto-fetch evaluation if completed
      if (result.status === 'completed') {
        try {
            const evalRes = await getEvaluation(runId)
            setSelectedRunEvaluation(evalRes)
        } catch (e) {
            setSelectedRunEvaluation(null)
        }
      }

      // M26: Fetch metrics
      try {
          const metrics = await getTunixRunMetrics(runId)
          setRunMetrics(metrics)
      } catch (e) {
          console.warn('Failed to load metrics', e)
          setRunMetrics(null)
      }
    } catch (error) {
      if (error instanceof ApiError) {
        setRunHistoryError(error.message)
      } else {
        setRunHistoryError('Unknown error occurred')
      }
    }
  }

  const handleEvaluateRun = async (runId: string) => {
    setEvaluationLoading(true)
    try {
        const res = await evaluateRun(runId)
        setSelectedRunEvaluation(res)
    } catch (e) {
        alert('Evaluation failed')
    } finally {
        setEvaluationLoading(false)
    }
  }

  const handleToggleRunHistory = () => {
    const newExpanded = !runHistoryExpanded
    setRunHistoryExpanded(newExpanded)
    if (newExpanded && !runHistory) {
      handleRefreshRunHistory()
    }
  }

  const handleCancelRun = async (runId: string) => {
    if (!confirm('Are you sure you want to cancel this run?')) return
    try {
        await cancelRun(runId)
        handleViewRunDetail(runId)
        handleRefreshRunHistory()
    } catch (e) {
        alert('Failed to cancel run')
    }
  }

  const handleFetchArtifacts = async (runId: string) => {
    setArtifactsLoading(true)
    try {
        const res = await listArtifacts(runId)
        setArtifacts(res)
    } catch (e) {
        console.error(e)
    } finally {
        setArtifactsLoading(false)
    }
  }

  const renderMetricsChart = () => {
    if (!runMetrics || runMetrics.length === 0) return null

    const width = 600
    const height = 200
    const padding = 25

    const steps = runMetrics.map(m => m.step)
    const losses = runMetrics.map(m => m.loss)

    const maxStep = Math.max(...steps)
    const minStep = Math.min(...steps)
    const maxLoss = Math.max(...losses) * 1.1 // Add 10% headroom

    const points = runMetrics.map(m => {
        const x = padding + ((m.step - minStep) / (maxStep - minStep || 1)) * (width - 2 * padding)
        // Invert Y because SVG y=0 is top
        const y = height - padding - (m.loss / (maxLoss || 1)) * (height - 2 * padding)
        return `${x},${y}`
    }).join(' ')

    return (
        <div style={{ marginTop: '10px', padding: '15px', border: '1px solid #e0e0e0', borderRadius: '4px', backgroundColor: '#fff' }}>
            <h5 style={{ marginTop: 0 }}>Training Loss</h5>
            <div style={{ overflowX: 'auto' }}>
                <svg width={width} height={height} style={{ background: '#fafafa', border: '1px solid #eee' }}>
                    {/* X Axis */}
                    <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#ccc" />
                    {/* Y Axis */}
                    <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#ccc" />

                    {/* Data Line */}
                    <polyline points={points} fill="none" stroke="#2196f3" strokeWidth="2" strokeLinejoin="round" />

                    {/* X Label */}
                    <text x={width / 2} y={height - 5} textAnchor="middle" fontSize="10" fill="#666">
                        Step ({minStep} - {maxStep})
                    </text>

                    {/* Y Label (Latest Loss) */}
                    <text x={width - padding} y={padding} textAnchor="end" fontSize="10" fill="#666">
                        Loss
                    </text>
                </svg>
            </div>
            <div style={{ fontSize: '0.9em', marginTop: '10px', display: 'flex', gap: '20px' }}>
                <span><strong>Latest Step:</strong> {runMetrics[runMetrics.length - 1].step}</span>
                <span><strong>Latest Loss:</strong> {runMetrics[runMetrics.length - 1].loss.toFixed(4)}</span>
                {runMetrics[0].device && <span><strong>Device:</strong> {runMetrics[0].device}</span>}
            </div>
        </div>
    )
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h1>Tunix RT</h1>
        <nav style={{ marginBottom: '20px' }}>
          <button
            onClick={() => setCurrentPage('home')}
            style={{
              fontWeight: currentPage === 'home' ? 'bold' : 'normal',
              marginRight: '10px',
              padding: '8px 16px',
              background: currentPage === 'home' ? '#eee' : 'transparent',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
          >
            Home
          </button>
          <button
            onClick={() => setCurrentPage('leaderboard')}
            style={{
              fontWeight: currentPage === 'leaderboard' ? 'bold' : 'normal',
              padding: '8px 16px',
              background: currentPage === 'leaderboard' ? '#eee' : 'transparent',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
          >
            Leaderboard
          </button>
          <button
            onClick={() => setCurrentPage('tuning')}
            style={{
              fontWeight: currentPage === 'tuning' ? 'bold' : 'normal',
              marginLeft: '10px',
              padding: '8px 16px',
              background: currentPage === 'tuning' ? '#eee' : 'transparent',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
          >
            Tuning (M19)
          </button>
          <button
            onClick={() => setCurrentPage('registry')}
            style={{
              fontWeight: currentPage === 'registry' ? 'bold' : 'normal',
              marginLeft: '10px',
              padding: '8px 16px',
              background: currentPage === 'registry' ? '#eee' : 'transparent',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
          >
            Registry (M20)
          </button>
        </nav>
      </div>
      <p>Reasoning-Trace Framework with RediAI Integration</p>

      {currentPage === 'leaderboard' ? (
        <Leaderboard />
      ) : currentPage === 'tuning' ? (
        <Tuning />
      ) : currentPage === 'registry' ? (
        <ModelRegistry />
      ) : (
        <>
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

      <div className="tunix-section" data-testid="tunix:section">
        <h2>Tunix Integration (M12)</h2>

        <div className="tunix-status" data-testid="tunix:status-container">
          <p data-testid="tunix:status">
            <strong>Status:</strong> {loading ? 'Loading...' : tunixStatus?.message || 'Unknown'}
          </p>
          <p data-testid="tunix:runtime-required">
            <strong>Runtime Required:</strong> {tunixStatus?.runtime_required ? 'Yes' : 'No (Artifact-based)'}
          </p>
        </div>

        <div className="tunix-form" data-testid="tunix:form">
          <div className="input-group">
            <label htmlFor="tunix-dataset-key">Dataset Key (e.g., ungar_hcd-v1):</label>
            <input
              id="tunix-dataset-key"
              data-testid="tunix:dataset-key"
              type="text"
              value={tunixDatasetKey}
              onChange={(e) => setTunixDatasetKey(e.target.value)}
              disabled={tunixLoading}
              placeholder="my_dataset-v1"
            />
          </div>

          <div className="input-group">
            <label htmlFor="tunix-model-id">Model ID:</label>
            <input
              id="tunix-model-id"
              data-testid="tunix:model-id"
              type="text"
              value={tunixModelId}
              onChange={(e) => setTunixModelId(e.target.value)}
              disabled={tunixLoading}
              placeholder="google/gemma-2b-it"
            />
          </div>

          <div className="input-group">
            <label htmlFor="tunix-output-dir">Output Directory:</label>
            <input
              id="tunix-output-dir"
              data-testid="tunix:output-dir"
              type="text"
              value={tunixOutputDir}
              onChange={(e) => setTunixOutputDir(e.target.value)}
              disabled={tunixLoading}
              placeholder="./output/tunix_run"
            />
          </div>

          <div className="button-group">
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
              <input
                type="checkbox"
                id="tunix-async-mode"
                checked={tunixAsyncMode}
                onChange={(e) => setTunixAsyncMode(e.target.checked)}
                disabled={tunixLoading || tunixRunLoading}
              />
              <label htmlFor="tunix-async-mode" style={{ marginLeft: '8px' }}>
                Run Async (Non-blocking)
              </label>
            </div>

            <button
              data-testid="tunix:export-btn"
              onClick={handleTunixExport}
              disabled={tunixLoading || !tunixDatasetKey}
            >
              Export JSONL
            </button>

            <button
              data-testid="tunix:manifest-btn"
              onClick={handleTunixManifest}
              disabled={tunixLoading || !tunixDatasetKey}
            >
              Generate Manifest
            </button>

            <button
              data-testid="tunix:run-dry-btn"
              onClick={() => handleTunixRun(true)}
              disabled={tunixLoading || tunixRunLoading || !tunixDatasetKey}
            >
              Run (Dry-run)
            </button>

            <button
              data-testid="tunix:run-local-btn"
              onClick={() => handleTunixRun(false)}
              disabled={tunixLoading || tunixRunLoading || !tunixDatasetKey}
              title={tunixStatus?.runtime_required ? 'Requires Tunix installation' : 'Run with Tunix (Local)'}
            >
              Run with Tunix (Local)
            </button>
          </div>
        </div>

        {tunixRunLoading && (
          <p data-testid="tunix:run-loading">Executing Tunix run...</p>
        )}

        {tunixError && (
          <div className="trace-error" data-testid="tunix:error">
            <strong>Error:</strong> {tunixError}
          </div>
        )}

        {tunixRunResult && (
          <div className="tunix-run-result" data-testid="tunix:run-result">
            <h3>Run {tunixRunResult.status === 'completed' ? 'Completed' : 'Failed'}</h3>
            <div className="run-info">
              <p data-testid="tunix:run-id"><strong>Run ID:</strong> {tunixRunResult.run_id}</p>
              <p data-testid="tunix:run-status"><strong>Status:</strong> {tunixRunResult.status}</p>
              <p data-testid="tunix:run-mode"><strong>Mode:</strong> {tunixRunResult.mode}</p>
              <p data-testid="tunix:run-duration"><strong>Duration:</strong> {tunixRunResult.duration_seconds?.toFixed(2)}s</p>
              {tunixRunResult.exit_code !== null && (
                <p data-testid="tunix:run-exit-code"><strong>Exit Code:</strong> {tunixRunResult.exit_code}</p>
              )}
              <p data-testid="tunix:run-message"><strong>Message:</strong> {tunixRunResult.message}</p>
            </div>
            {tunixRunResult.stdout && (
              <details>
                <summary>Standard Output</summary>
                <pre data-testid="tunix:run-stdout">{tunixRunResult.stdout}</pre>
              </details>
            )}
            {tunixRunResult.stderr && (
              <details>
                <summary>Standard Error</summary>
                <pre data-testid="tunix:run-stderr">{tunixRunResult.stderr}</pre>
              </details>
            )}
          </div>
        )}

        {tunixManifestResult && (
          <div className="tunix-manifest-result" data-testid="tunix:manifest-result">
            <h3>Manifest Generated</h3>
            <p data-testid="tunix:manifest-message">{tunixManifestResult.message}</p>
            <details>
              <summary>View YAML Manifest</summary>
              <pre data-testid="tunix:manifest-yaml">{tunixManifestResult.manifest_yaml}</pre>
            </details>
          </div>
        )}

        {/* M14: Run History Section */}
        <div className="run-history-section" data-testid="tunix:run-history-section">
          <h3>
            <button
              data-testid="tunix:toggle-history-btn"
              onClick={handleToggleRunHistory}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                fontSize: '1.2em',
                padding: '0.5em',
                textDecoration: 'underline',
              }}
            >
              {runHistoryExpanded ? '▼' : '▶'} Run History
            </button>
            {runHistoryExpanded && (
              <button
                data-testid="tunix:refresh-history-btn"
                onClick={handleRefreshRunHistory}
                disabled={runHistoryLoading}
                style={{ marginLeft: '1em' }}
              >
                {runHistoryLoading ? 'Refreshing...' : 'Refresh'}
              </button>
            )}
          </h3>

          {runHistoryExpanded && (
            <div className="run-history-content" data-testid="tunix:run-history-content">
              {runHistoryLoading && <p>Loading run history...</p>}

              {runHistoryError && (
                <div className="trace-error" data-testid="tunix:history-error">
                  <strong>Error:</strong> {runHistoryError}
                </div>
              )}

              {runHistory && runHistory.data.length === 0 && (
                <p data-testid="tunix:history-empty">No runs found. Execute a run to see history.</p>
              )}

              {runHistory && runHistory.data.length > 0 && (
                <div className="run-history-list" data-testid="tunix:history-list">
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left', padding: '0.5em', borderBottom: '2px solid #ccc' }}>Status</th>
                        <th style={{ textAlign: 'left', padding: '0.5em', borderBottom: '2px solid #ccc' }}>Mode</th>
                        <th style={{ textAlign: 'left', padding: '0.5em', borderBottom: '2px solid #ccc' }}>Dataset</th>
                        <th style={{ textAlign: 'left', padding: '0.5em', borderBottom: '2px solid #ccc' }}>Model</th>
                        <th style={{ textAlign: 'left', padding: '0.5em', borderBottom: '2px solid #ccc' }}>Duration</th>
                        <th style={{ textAlign: 'left', padding: '0.5em', borderBottom: '2px solid #ccc' }}>Score</th>
                        <th style={{ textAlign: 'left', padding: '0.5em', borderBottom: '2px solid #ccc' }}>Started</th>
                        <th style={{ textAlign: 'left', padding: '0.5em', borderBottom: '2px solid #ccc' }}>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {runHistory.data.map((run) => (
                        <>
                          <tr key={run.run_id} data-testid={`tunix:history-row-${run.run_id}`}>
                            <td style={{ padding: '0.5em', borderBottom: '1px solid #eee' }}>
                              <span className={`status-badge status-${run.status}`} data-testid={`tunix:history-status-${run.run_id}`}>
                                {run.status}
                              </span>
                            </td>
                            <td style={{ padding: '0.5em', borderBottom: '1px solid #eee' }} data-testid={`tunix:history-mode-${run.run_id}`}>
                              {run.mode}
                            </td>
                            <td style={{ padding: '0.5em', borderBottom: '1px solid #eee', fontSize: '0.9em' }} data-testid={`tunix:history-dataset-${run.run_id}`}>
                              {run.dataset_key}
                            </td>
                            <td style={{ padding: '0.5em', borderBottom: '1px solid #eee', fontSize: '0.9em' }} data-testid={`tunix:history-model-${run.run_id}`}>
                              {run.model_id}
                            </td>
                            <td style={{ padding: '0.5em', borderBottom: '1px solid #eee' }} data-testid={`tunix:history-duration-${run.run_id}`}>
                              {run.duration_seconds ? `${run.duration_seconds.toFixed(2)}s` : 'N/A'}
                            </td>
                            <td style={{ padding: '0.5em', borderBottom: '1px solid #eee' }} data-testid={`tunix:history-score-${run.run_id}`}>
                              {run.metrics?.answer_correctness ? Number(run.metrics.answer_correctness).toFixed(2) : '-'}
                            </td>
                            <td style={{ padding: '0.5em', borderBottom: '1px solid #eee', fontSize: '0.9em' }} data-testid={`tunix:history-started-${run.run_id}`}>
                              {new Date(run.started_at).toLocaleString()}
                            </td>
                            <td style={{ padding: '0.5em', borderBottom: '1px solid #eee' }}>
                              <button
                                data-testid={`tunix:view-detail-btn-${run.run_id}`}
                                onClick={() => handleViewRunDetail(run.run_id)}
                                style={{ fontSize: '0.9em' }}
                              >
                                {selectedRunId === run.run_id ? 'Hide' : 'View'}
                              </button>
                            </td>
                          </tr>
                          {selectedRunId === run.run_id && selectedRunDetail && (
                            <tr key={`${run.run_id}-detail`}>
                              <td colSpan={8} style={{ padding: '1em', backgroundColor: '#f9f9f9' }}>
                                <div className="run-detail" data-testid={`tunix:run-detail-${run.run_id}`}>
                                  <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                                    <h4>Run Details</h4>
                                    {['pending', 'running'].includes(selectedRunDetail.status) && (
                                        <button
                                            onClick={() => handleCancelRun(run.run_id)}
                                            style={{backgroundColor: '#ff6b6b', color: 'white', border: 'none', padding: '0.5em 1em', cursor: 'pointer', borderRadius: '4px'}}
                                        >
                                            Cancel Run
                                        </button>
                                    )}
                                  </div>

                                  <p><strong>Run ID:</strong> {selectedRunDetail.run_id}</p>
                                  <p><strong>Message:</strong> {selectedRunDetail.message}</p>
                                  {selectedRunDetail.exit_code !== null && (
                                    <p><strong>Exit Code:</strong> {selectedRunDetail.exit_code}</p>
                                  )}

                                  {/* M17: Evaluation Section */}
                                  {selectedRunDetail.status === 'completed' && (
                                    <div className="evaluation-summary" style={{margin: '10px 0', padding: '10px', backgroundColor: '#e8f5e9', border: '1px solid #c8e6c9', borderRadius: '4px'}}>
                                        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                                            <h5>Evaluation</h5>
                                            <button
                                                onClick={() => handleEvaluateRun(run.run_id)}
                                                disabled={evaluationLoading}
                                                style={{fontSize: '0.8em', padding: '2px 8px'}}
                                            >
                                                {evaluationLoading ? 'Evaluating...' : 'Re-evaluate'}
                                            </button>
                                        </div>

                                        {selectedRunEvaluation ? (
                                            <div>
                                                <p><strong>Score:</strong> {selectedRunEvaluation.score.toFixed(1)} / 100</p>
                                                <p><strong>Verdict:</strong> <span className={`status-badge ${selectedRunEvaluation.verdict === 'pass' ? 'status-completed' : 'status-failed'}`}>{selectedRunEvaluation.verdict.toUpperCase()}</span></p>
                                                <div style={{fontSize: '0.9em', marginTop: '5px'}}>
                                                    <strong>Metrics:</strong>
                                                    <ul style={{listStyleType: 'none', paddingLeft: 0, margin: '5px 0'}}>
                                                        {Object.entries(selectedRunEvaluation.metrics).map(([key, val]) => (
                                                            <li key={key}>{key}: {val}</li>
                                                        ))}
                                                    </ul>
                                                </div>
                                            </div>
                                        ) : (
                                            <p>No evaluation available.</p>
                                        )}
                                    </div>
                                  )}

                                  {/* M26: Metrics Chart */}
                                  {renderMetricsChart()}

                                  {['pending', 'running'].includes(selectedRunDetail.status) ? (
                                    <LiveLogs
                                        runId={selectedRunDetail.run_id}
                                        initialStatus={selectedRunDetail.status}
                                        onStatusChange={(newStatus) => {
                                            if (['completed', 'failed', 'timeout'].includes(newStatus)) {
                                                handleViewRunDetail(run.run_id)
                                                handleRefreshRunHistory()
                                            }
                                        }}
                                    />
                                  ) : (
                                    <>
                                      {selectedRunDetail.stdout && (
                                        <details>
                                          <summary>Standard Output</summary>
                                          <pre style={{ maxHeight: '300px', overflow: 'auto' }} data-testid={`tunix:detail-stdout-${run.run_id}`}>
                                            {selectedRunDetail.stdout}
                                          </pre>
                                        </details>
                                      )}
                                      {selectedRunDetail.stderr && (
                                        <details>
                                          <summary>Standard Error</summary>
                                          <pre style={{ maxHeight: '300px', overflow: 'auto' }} data-testid={`tunix:detail-stderr-${run.run_id}`}>
                                            {selectedRunDetail.stderr}
                                          </pre>
                                        </details>
                                      )}
                                    </>
                                  )}

                                  {/* Artifacts Section (M16) */}
                                  <div className="artifacts-section" style={{marginTop: '20px', borderTop: '1px solid #ddd', paddingTop: '10px'}}>
                                    <h4>Artifacts</h4>
                                    <button onClick={() => handleFetchArtifacts(run.run_id)}>Load Artifacts</button>

                                    {artifactsLoading && <span style={{marginLeft: '10px'}}>Loading...</span>}

                                    {artifacts && (
                                        <ul style={{marginTop: '10px'}}>
                                            {artifacts.artifacts.length === 0 && <li>No artifacts found.</li>}
                                            {artifacts.artifacts.map(art => (
                                                <li key={art.name}>
                                                    <a href={`/api/tunix/runs/${run.run_id}/artifacts/${art.name}/download`} target="_blank" rel="noreferrer">
                                                        {art.name}
                                                    </a> <span style={{fontSize: '0.8em', color: '#666'}}>({art.size} bytes)</span>
                                                </li>
                                            ))}
                                        </ul>
                                    )}
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      </>
      )}
    </div>
  )
}

export default App
