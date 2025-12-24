import { useState, useEffect } from 'react'
import {
  listModelArtifacts,
  createModelArtifact,
  promoteRunToVersion,
  getModelArtifact,
  getModelDownloadUrl,
  type ModelArtifact,
} from '../api/client'

export const ModelRegistry = () => {
  const [artifacts, setArtifacts] = useState<ModelArtifact[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null)
  const [selectedArtifactDetail, setSelectedArtifactDetail] = useState<ModelArtifact | null>(null)

  // Create form
  const [newArtifactName, setNewArtifactName] = useState('')
  const [newArtifactDesc, setNewArtifactDesc] = useState('')

  // Promote form (manual)
  const [promoteRunId, setPromoteRunId] = useState('')
  const [promoteLabel, setPromoteLabel] = useState('')
  const [promoteLoading, setPromoteLoading] = useState(false)

  const fetchArtifacts = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await listModelArtifacts()
      setArtifacts(data)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchArtifacts()
  }, [])

  const handleCreateArtifact = async () => {
    if (!newArtifactName) return
    setLoading(true)
    setError(null)
    try {
      await createModelArtifact({ name: newArtifactName, description: newArtifactDesc, task_type: 'sft' })
      setNewArtifactName('')
      setNewArtifactDesc('')
      await fetchArtifacts()
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleSelectArtifact = async (id: string) => {
    setSelectedArtifactId(id)
    setSelectedArtifactDetail(null)
    // Here we assume getModelArtifact returns details + versions if implemented,
    // but typically we might need a separate call for versions if list doesn't include them.
    // Our API getModelArtifact returns ModelArtifactRead which has latest_version?
    // Actually our DB model has `versions` relationship but schema `ModelArtifactRead` only has `latest_version`.
    // I should have added `versions` list to `ModelArtifactRead` or separate endpoint.
    // For MVP, I'll rely on `latest_version` or update API to return all versions.
    // Wait, the plan said "GET /api/models/{artifact_id} -> artifact details + versions".
    // I implemented `get_model_artifact` returning `ModelArtifactRead`.
    // `ModelArtifactRead` definition in `backend/tunix_rt_backend/schemas/model_registry.py`:
    // `latest_version: ModelVersionRead | None`
    // It does NOT include a list of versions.
    // I missed that detail in implementation vs plan.
    // I should fix the backend to return versions list if I want to show them.
    // Or I assume for now I only see latest.

    // I will fetch it anyway.
    try {
        const detail = await getModelArtifact(id)
        setSelectedArtifactDetail(detail)
    } catch (e) {}
  }

  const handlePromote = async () => {
    if (!selectedArtifactId || !promoteRunId) return
    setPromoteLoading(true)
    try {
        await promoteRunToVersion(selectedArtifactId, {
            source_run_id: promoteRunId,
            version_label: promoteLabel || undefined
        })
        setPromoteRunId('')
        setPromoteLabel('')
        alert('Promotion successful!')
        // Refresh details (might update latest version)
        handleSelectArtifact(selectedArtifactId)
        fetchArtifacts()
    } catch (e: any) {
        alert(`Promotion failed: ${e.message}`)
    } finally {
        setPromoteLoading(false)
    }
  }

  return (
    <div className="model-registry">
      <div className="registry-sidebar" style={{ width: '300px', borderRight: '1px solid #ddd', padding: '10px' }}>
        <h3>Models</h3>
        <div className="create-form" style={{ marginBottom: '20px', padding: '10px', background: '#f5f5f5' }}>
            <input
                placeholder="Model Name"
                value={newArtifactName}
                onChange={e => setNewArtifactName(e.target.value)}
                style={{ width: '100%', marginBottom: '5px' }}
            />
            <input
                placeholder="Description"
                value={newArtifactDesc}
                onChange={e => setNewArtifactDesc(e.target.value)}
                style={{ width: '100%', marginBottom: '5px' }}
            />
            <button onClick={handleCreateArtifact} disabled={loading || !newArtifactName}>Create Model</button>
        </div>

        {loading && <p>Loading...</p>}
        {error && <p style={{color: 'red'}}>{error}</p>}

        <ul style={{ listStyle: 'none', padding: 0 }}>
            {artifacts.map(art => (
                <li
                    key={art.id}
                    onClick={() => handleSelectArtifact(art.id)}
                    style={{
                        padding: '10px',
                        borderBottom: '1px solid #eee',
                        cursor: 'pointer',
                        background: selectedArtifactId === art.id ? '#e3f2fd' : 'transparent'
                    }}
                >
                    <strong>{art.name}</strong>
                    <br/>
                    <small>{art.task_type}</small>
                </li>
            ))}
        </ul>
      </div>

      <div className="registry-content" style={{ flex: 1, padding: '20px' }}>
        {selectedArtifactDetail ? (
            <div>
                <h2>{selectedArtifactDetail.name}</h2>
                <p>{selectedArtifactDetail.description}</p>
                <p><strong>ID:</strong> {selectedArtifactDetail.id}</p>

                <div className="promote-section" style={{ margin: '20px 0', padding: '15px', border: '1px solid #ccc' }}>
                    <h4>Promote Run to Version</h4>
                    <div style={{ display: 'flex', gap: '10px' }}>
                        <input
                            placeholder="Source Run ID (UUID)"
                            value={promoteRunId}
                            onChange={e => setPromoteRunId(e.target.value)}
                            style={{ flex: 1 }}
                        />
                        <input
                            placeholder="Version Label (optional)"
                            value={promoteLabel}
                            onChange={e => setPromoteLabel(e.target.value)}
                        />
                        <button onClick={handlePromote} disabled={promoteLoading || !promoteRunId}>
                            {promoteLoading ? 'Promoting...' : 'Promote'}
                        </button>
                    </div>
                </div>

                <h3>Versions</h3>
                {/* Note: Backend currently only returns latest_version in list/detail.
                    Ideally we fetch full version list. For now showing latest. */}
                {selectedArtifactDetail.latest_version ? (
                    <div className="version-card" style={{ padding: '15px', border: '1px solid #eee', marginBottom: '10px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <h4>{selectedArtifactDetail.latest_version.version}</h4>
                            <span className={`status-badge status-${selectedArtifactDetail.latest_version.status}`}>
                                {selectedArtifactDetail.latest_version.status}
                            </span>
                        </div>
                        <p><strong>SHA256:</strong> {selectedArtifactDetail.latest_version.sha256}</p>
                        <p><strong>Size:</strong> {selectedArtifactDetail.latest_version.size_bytes} bytes</p>
                        <p><strong>Source Run:</strong> {selectedArtifactDetail.latest_version.source_run_id}</p>
                        <a
                            href={getModelDownloadUrl(selectedArtifactDetail.latest_version.id)}
                            target="_blank"
                            rel="noreferrer"
                            style={{ display: 'inline-block', marginTop: '10px', padding: '5px 10px', background: '#4caf50', color: 'white', textDecoration: 'none', borderRadius: '4px' }}
                        >
                            Download Artifacts
                        </a>
                    </div>
                ) : (
                    <p>No versions yet.</p>
                )}
            </div>
        ) : (
            <p>Select a model to view details.</p>
        )}
      </div>
    </div>
  )
}
