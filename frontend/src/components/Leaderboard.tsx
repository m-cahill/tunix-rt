import { useEffect, useState } from 'react'
import { getLeaderboard, type LeaderboardResponse, type LeaderboardFilters, ApiError } from '../api/client'

export const Leaderboard = () => {
  const [data, setData] = useState<LeaderboardResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [limit] = useState(50)
  const [offset, setOffset] = useState(0)

  // M35: Filter state
  const [filters, setFilters] = useState<LeaderboardFilters>({})
  const [filterDataset, setFilterDataset] = useState('')
  const [filterModel, setFilterModel] = useState('')

  const fetchLeaderboard = async (currentLimit: number, currentOffset: number, currentFilters: LeaderboardFilters) => {
    setLoading(true)
    setError(null)
    try {
      const result = await getLeaderboard(currentLimit, currentOffset, currentFilters)
      setData(result)
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message)
      } else {
        setError('Failed to load leaderboard')
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchLeaderboard(limit, offset, filters)
  }, [limit, offset, filters])

  // M35: Apply filters
  const handleApplyFilters = () => {
    setOffset(0) // Reset to first page
    setFilters({
      dataset_key: filterDataset || undefined,
      model_id: filterModel || undefined,
    })
  }

  const handleClearFilters = () => {
    setFilterDataset('')
    setFilterModel('')
    setFilters({})
    setOffset(0)
  }

  const handleNextPage = () => {
    if (data?.pagination?.next_offset) {
      setOffset(data.pagination.next_offset)
    }
  }

  const handlePrevPage = () => {
    const newOffset = Math.max(0, offset - limit)
    setOffset(newOffset)
  }

  if (error) return <div className="error">Error: {error}</div>

  return (
    <div className="leaderboard">
      <h2>Leaderboard</h2>

      {/* M35: Inline Filters */}
      <div style={{ display: 'flex', gap: '10px', marginBottom: '15px', alignItems: 'center', flexWrap: 'wrap' }}>
        <input
          type="text"
          placeholder="Dataset (exact)"
          value={filterDataset}
          onChange={(e) => setFilterDataset(e.target.value)}
          style={{ padding: '6px 10px', border: '1px solid #ccc', borderRadius: '4px', width: '150px' }}
        />
        <input
          type="text"
          placeholder="Model (contains)"
          value={filterModel}
          onChange={(e) => setFilterModel(e.target.value)}
          style={{ padding: '6px 10px', border: '1px solid #ccc', borderRadius: '4px', width: '150px' }}
        />
        <button onClick={handleApplyFilters} style={{ padding: '6px 12px' }}>
          Filter
        </button>
        <button onClick={handleClearFilters} style={{ padding: '6px 12px' }}>
          Clear
        </button>
        {(filters.dataset_key || filters.model_id) && (
          <span style={{ fontSize: '0.85em', color: '#666' }}>
            Filtering: {filters.dataset_key && `dataset=${filters.dataset_key}`} {filters.model_id && `model~${filters.model_id}`}
          </span>
        )}
      </div>

      {loading ? (
        <div>Loading leaderboard...</div>
      ) : !data || data.data.length === 0 ? (
        <div>No evaluated runs found.</div>
      ) : (
        <>
          <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '10px' }}>
            <thead>
              <tr style={{ backgroundColor: '#f5f5f5' }}>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Rank</th>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Run ID</th>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Model</th>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Dataset</th>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Primary Score</th>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Items</th>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Verdict</th>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Evaluated</th>
              </tr>
            </thead>
            <tbody>
              {data.data.map((item, index) => (
                <tr key={item.run_id} style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '10px' }}>
                    {offset + index + 1}
                  </td>
                  <td style={{ padding: '10px', fontFamily: 'monospace' }}>{item.run_id.substring(0, 8)}...</td>
                  <td style={{ padding: '10px', fontSize: '0.9em' }}>{item.model_id}</td>
                  <td style={{ padding: '10px', fontSize: '0.9em' }}>{item.dataset_key}</td>
                  <td style={{ padding: '10px', fontWeight: 'bold' }}>
                    {item.primary_score != null ? (item.primary_score * 100).toFixed(1) + '%' : item.score.toFixed(1)}
                  </td>
                  <td style={{ padding: '10px', fontSize: '0.85em', color: '#666' }}>
                    {item.scorecard ? `${item.scorecard.n_scored}/${item.scorecard.n_items}` : '-'}
                  </td>
                  <td style={{ padding: '10px' }}>
                    <span className={`status-badge ${item.verdict === 'pass' ? 'status-completed' : 'status-failed'}`}>
                      {item.verdict.toUpperCase()}
                    </span>
                  </td>
                  <td style={{ padding: '10px', fontSize: '0.85em', color: '#666' }}>
                    {new Date(item.evaluated_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'center', gap: '10px' }}>
            <button
              onClick={handlePrevPage}
              disabled={offset === 0 || loading}
              style={{ padding: '5px 10px' }}
            >
              Previous
            </button>
            <span style={{ padding: '5px' }}>
              Page {Math.floor(offset / limit) + 1}
            </span>
            <button
              onClick={handleNextPage}
              disabled={!data?.pagination?.next_offset || loading}
              style={{ padding: '5px 10px' }}
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  )
}
