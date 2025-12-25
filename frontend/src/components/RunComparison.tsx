import { useEffect, useState } from 'react'
import {
  getTunixRun,
  getEvaluation,
  getTunixRunMetrics,
  type TunixRunResponse,
  type EvaluationResponse,
  type TunixRunMetric,
} from '../api/client'

interface RunComparisonProps {
  runAId: string
  runBId: string
  onClose: () => void
}

interface RunData {
  details: TunixRunResponse | null
  evaluation: EvaluationResponse | null
  metrics: TunixRunMetric[] | null
  loading: boolean
  error: string | null
}

const MetricsChart = ({ metricsA, metricsB, labelA, labelB }: { metricsA: TunixRunMetric[] | null, metricsB: TunixRunMetric[] | null, labelA: string, labelB: string }) => {
    if ((!metricsA || metricsA.length === 0) && (!metricsB || metricsB.length === 0)) return <div>No metrics data</div>

    const width = 600
    const height = 300
    const padding = 40

    // Combine steps to find range
    const stepsA = metricsA?.map(m => m.step) || []
    const stepsB = metricsB?.map(m => m.step) || []
    const allSteps = [...stepsA, ...stepsB]

    // Combine losses to find range
    const lossesA = metricsA?.map(m => m.loss) || []
    const lossesB = metricsB?.map(m => m.loss) || []
    const allLosses = [...lossesA, ...lossesB]

    const minStep = Math.min(...allSteps)
    const maxStep = Math.max(...allSteps)
    const maxLoss = Math.max(...allLosses) * 1.1 // 10% headroom

    const getPoints = (metrics: TunixRunMetric[]) => {
        return metrics.map(m => {
            const x = padding + ((m.step - minStep) / (maxStep - minStep || 1)) * (width - 2 * padding)
            const y = height - padding - (m.loss / (maxLoss || 1)) * (height - 2 * padding)
            return `${x},${y}`
        }).join(' ')
    }

    return (
        <div style={{ padding: '15px', border: '1px solid #e0e0e0', borderRadius: '4px', backgroundColor: '#fff' }}>
            <h5 style={{ marginTop: 0 }}>Loss Curve Comparison</h5>
            <div style={{ overflowX: 'auto' }}>
                <svg width={width} height={height} style={{ background: '#fafafa', border: '1px solid #eee' }}>
                    {/* Axes */}
                    <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#ccc" />
                    <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#ccc" />

                    {/* Run A */}
                    {metricsA && (
                        <polyline points={getPoints(metricsA)} fill="none" stroke="#2196f3" strokeWidth="2" strokeLinejoin="round" />
                    )}

                    {/* Run B */}
                    {metricsB && (
                        <polyline points={getPoints(metricsB)} fill="none" stroke="#f44336" strokeWidth="2" strokeLinejoin="round" />
                    )}

                    {/* Legend */}
                    <g transform={`translate(${width - 150}, ${padding})`}>
                        <rect width="140" height="60" fill="white" stroke="#ccc" />
                        <line x1="10" y1="20" x2="40" y2="20" stroke="#2196f3" strokeWidth="2" />
                        <text x="50" y="25" fontSize="12">{labelA} (Blue)</text>
                        <line x1="10" y1="45" x2="40" y2="45" stroke="#f44336" strokeWidth="2" />
                        <text x="50" y="50" fontSize="12">{labelB} (Red)</text>
                    </g>
                </svg>
            </div>
        </div>
    )
}

export const RunComparison = ({ runAId, runBId, onClose }: RunComparisonProps) => {
  const [dataA, setDataA] = useState<RunData>({ details: null, evaluation: null, metrics: null, loading: true, error: null })
  const [dataB, setDataB] = useState<RunData>({ details: null, evaluation: null, metrics: null, loading: true, error: null })

  const fetchRunData = async (runId: string, setData: React.Dispatch<React.SetStateAction<RunData>>) => {
    try {
        const details = await getTunixRun(runId)

        let evaluation = null
        if (details.status === 'completed') {
            try {
                evaluation = await getEvaluation(runId)
            } catch (e) {
                // Ignore missing eval
            }
        }

        let metrics = null
        try {
            metrics = await getTunixRunMetrics(runId)
        } catch (e) {
            // Ignore missing metrics
        }

        setData({ details, evaluation, metrics, loading: false, error: null })
    } catch (e: any) {
        setData(prev => ({ ...prev, loading: false, error: e.message || 'Failed to fetch run' }))
    }
  }

  useEffect(() => {
    fetchRunData(runAId, setDataA)
    fetchRunData(runBId, setDataB)
  }, [runAId, runBId])

  const renderMetricDiff = (valA: number | undefined, valB: number | undefined, higherIsBetter = true) => {
      if (valA === undefined || valB === undefined) return '-'
      const diff = valA - valB
      const color = higherIsBetter
        ? (diff > 0 ? 'green' : (diff < 0 ? 'red' : 'black'))
        : (diff < 0 ? 'green' : (diff > 0 ? 'red' : 'black'))

      return (
          <span>
              {valA.toFixed(4)} vs {valB.toFixed(4)}
              <span style={{ color, marginLeft: '5px', fontWeight: 'bold' }}>
                  ({diff > 0 ? '+' : ''}{diff.toFixed(4)})
              </span>
          </span>
      )
  }

  return (
    <div className="comparison-view" style={{ padding: '20px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2>Run Comparison</h2>
        <button onClick={onClose} style={{ padding: '5px 15px' }}>Close</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        {/* Header */}
        <div style={{ padding: '10px', background: '#e3f2fd', borderRadius: '4px' }}>
            <h3>Run A: {runAId.substring(0, 8)}...</h3>
            {dataA.loading && <p>Loading...</p>}
            {dataA.error && <p className="error">{dataA.error}</p>}
        </div>
        <div style={{ padding: '10px', background: '#ffebee', borderRadius: '4px' }}>
            <h3>Run B: {runBId.substring(0, 8)}...</h3>
            {dataB.loading && <p>Loading...</p>}
            {dataB.error && <p className="error">{dataB.error}</p>}
        </div>

        {/* Details Comparison */}
        <div>
            <h4>Configuration</h4>
            <p><strong>Model:</strong> {dataA.details?.model_id}</p>
            <p><strong>Dataset:</strong> {dataA.details?.dataset_key}</p>
            <p><strong>Status:</strong> {dataA.details?.status}</p>
            <p><strong>Duration:</strong> {dataA.details?.duration_seconds?.toFixed(2)}s</p>
        </div>
        <div>
            <h4>Configuration</h4>
            <p><strong>Model:</strong> {dataB.details?.model_id}</p>
            <p><strong>Dataset:</strong> {dataB.details?.dataset_key}</p>
            <p><strong>Status:</strong> {dataB.details?.status}</p>
            <p><strong>Duration:</strong> {dataB.details?.duration_seconds?.toFixed(2)}s</p>
        </div>

        {/* Metrics Comparison */}
        <div style={{ gridColumn: '1 / -1' }}>
            <h4>Evaluation Score (Answer Correctness)</h4>
            <div style={{ padding: '15px', border: '1px solid #ddd', borderRadius: '4px' }}>
                {renderMetricDiff(
                    dataA.evaluation?.score,
                    dataB.evaluation?.score
                )}
            </div>
        </div>

        {/* Chart */}
        <div style={{ gridColumn: '1 / -1' }}>
            <MetricsChart
                metricsA={dataA.metrics}
                metricsB={dataB.metrics}
                labelA={`Run A (${runAId.substring(0, 6)})`}
                labelB={`Run B (${runBId.substring(0, 6)})`}
            />
        </div>
      </div>
    </div>
  )
}
