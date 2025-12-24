import { useEffect, useState } from 'react'
import { getLeaderboard, type LeaderboardResponse, ApiError } from '../api/client'

export const Leaderboard = () => {
  const [data, setData] = useState<LeaderboardResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchLeaderboard = async () => {
      try {
        const result = await getLeaderboard()
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

    fetchLeaderboard()
  }, [])

  if (loading) return <div>Loading leaderboard...</div>
  if (error) return <div className="error">Error: {error}</div>
  if (!data || data.data.length === 0) return <div>No evaluated runs found.</div>

  return (
    <div className="leaderboard">
      <h2>🏆 Leaderboard</h2>
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
                {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : index + 1}
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
    </div>
  )
}
