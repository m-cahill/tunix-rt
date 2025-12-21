import { useEffect, useState } from 'react'
import { getApiHealth, getRediHealth, type HealthResponse, type RediHealthResponse } from './api/client'

function App() {
  const [apiHealth, setApiHealth] = useState<HealthResponse | null>(null)
  const [rediHealth, setRediHealth] = useState<RediHealthResponse | null>(null)
  const [loading, setLoading] = useState(true)

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
    </div>
  )
}

export default App

