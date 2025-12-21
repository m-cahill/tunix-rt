import { useEffect, useState } from 'react'

interface HealthStatus {
  status: string
  error?: string
}

function App() {
  const [apiHealth, setApiHealth] = useState<HealthStatus | null>(null)
  const [rediHealth, setRediHealth] = useState<HealthStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchHealth = async () => {
      setLoading(true)
      
      try {
        // Fetch API health
        const apiResponse = await fetch('/api/health')
        const apiData = await apiResponse.json()
        setApiHealth(apiData)
      } catch (error) {
        setApiHealth({ status: 'down', error: 'Failed to fetch' })
      }

      try {
        // Fetch RediAI health
        const rediResponse = await fetch('/api/redi/health')
        const rediData = await rediResponse.json()
        setRediHealth(rediData)
      } catch (error) {
        setRediHealth({ status: 'down', error: 'Failed to fetch' })
      }

      setLoading(false)
    }

    fetchHealth()
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
            {apiHealth?.error && <p>Error: {apiHealth.error}</p>}
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

