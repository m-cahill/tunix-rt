import { useEffect, useState } from 'react'
import { getLeaderboard, type LeaderboardResponse, ApiError } from '../api/client'

export const Leaderboard = () => {
  const [data, setData] = useState<LeaderboardResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [limit] = useState(50)
  const [offset, setOffset] = useState(0)

  const fetchLeaderboard = async (currentLimit: number, currentOffset: number) => {
    setLoading(true)
    setError(null)
    try {
      const result = await getLeaderboard(currentLimit, currentOffset)
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
    fetchLeaderboard(limit, offset)
  }, [limit, offset])

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
      <h2>🏆 Leaderboard</h2>

      {loading ? (
        <div>Loading leaderboard...</div>
      ) : !data || data.data.length === 0 ? (
        <div>No evaluated runs found.</div>
      ) : (
        <>
          <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: '20px' }}>
            <thead>
              <tr style={{ backgroundColor: '#f5f5f5' }}>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Rank</th>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Run ID</th>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Model</th>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Score</th>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Verdict</th>
                <th style={{ padding: '10px', textAlign: 'left', borderBottom: '2px solid #ddd' }}>Evaluated At</th>
              </tr>
            </thead>
            <tbody>
              {data.data.map((item, index) => (
                <tr key={item.run_id} style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '10px' }}>
                    {offset + index + 1}
                  </td>
                  <td style={{ padding: '10px', fontFamily: 'monospace' }}>{item.run_id.substring(0, 8)}...</td>
                  <td style={{ padding: '10px' }}>{item.model_id}</td>
                  <td style={{ padding: '10px', fontWeight: 'bold' }}>{item.score.toFixed(1)}</td>
                  <td style={{ padding: '10px' }}>
                    <span className={`status-badge ${item.verdict === 'pass' ? 'status-completed' : 'status-failed'}`}>
                      {item.verdict.toUpperCase()}
                    </span>
                  </td>
                  <td style={{ padding: '10px', fontSize: '0.9em', color: '#666' }}>
                    {new Date(item.evaluated_at).toLocaleString()}
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
