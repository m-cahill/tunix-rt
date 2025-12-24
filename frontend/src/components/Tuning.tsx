import { useState, useEffect } from 'react'
import {
  TuningJob,
  TuningJobCreate,
  listTuningJobs,
  createTuningJob,
  startTuningJob,
  getTuningJob,
} from '../api/client'

export function Tuning() {
  const [jobs, setJobs] = useState<TuningJob[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Create Form State
  const [showCreate, setShowCreate] = useState(false)
  const [createName, setCreateName] = useState('New Experiment')
  const [datasetKey, setDatasetKey] = useState('')
  const [modelId, setModelId] = useState('google/gemma-2b-it')
  const [numSamples, setNumSamples] = useState(2)

  // Detail View State
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [selectedJob, setSelectedJob] = useState<TuningJob | null>(null)

  useEffect(() => {
    fetchJobs()
  }, [])

  const fetchJobs = async () => {
    setLoading(true)
    try {
      const data = await listTuningJobs()
      setJobs(data)
      setError(null)
    } catch (e) {
      setError('Failed to list jobs')
    } finally {
      setLoading(false)
    }
  }

  const handleCreate = async () => {
    if (!datasetKey) {
      setError('Dataset key required')
      return
    }

    setLoading(true)
    try {
      const payload: TuningJobCreate = {
        name: createName,
        dataset_key: datasetKey,
        base_model_id: modelId,
        metric_name: 'score',
        metric_mode: 'max',
        num_samples: numSamples,
        max_concurrent_trials: 1,
        search_space: {
          learning_rate: { type: 'loguniform', min: 1e-5, max: 1e-3 },
          batch_size: { type: 'choice', values: [4, 8] }
        }
      }

      await createTuningJob(payload)
      setShowCreate(false)
      fetchJobs()
    } catch (e: any) {
      setError(e.message || 'Create failed')
    } finally {
      setLoading(false)
    }
  }

  const handleView = async (id: string) => {
    if (selectedJobId === id) {
      setSelectedJobId(null)
      setSelectedJob(null)
      return
    }

    setSelectedJobId(id)
    try {
      const job = await getTuningJob(id)
      setSelectedJob(job)
    } catch (e) {
      console.error(e)
    }
  }

  const handleStart = async (id: string) => {
    try {
      await startTuningJob(id)
      fetchJobs()
      if (selectedJobId === id) handleView(id)
    } catch (e: any) {
      alert(`Start failed: ${e.message}`)
    }
  }

  return (
    <div className="tuning-page">
      <h2>Hyperparameter Tuning</h2>

      {error && <div className="error-banner" style={{color: 'red', margin: '10px 0'}}>{error}</div>}

      <div style={{marginBottom: '20px'}}>
        <button onClick={() => setShowCreate(!showCreate)}>
          {showCreate ? 'Cancel' : '+ New Experiment'}
        </button>
        <button onClick={fetchJobs} style={{marginLeft: '10px'}} disabled={loading}>
          Refresh
        </button>
      </div>

      {showCreate && (
        <div className="create-form" style={{border: '1px solid #ccc', padding: '15px', marginBottom: '20px'}}>
          <h3>Create Experiment</h3>
          <div style={{marginBottom: '10px'}}>
            <label>Name: </label>
            <input value={createName} onChange={e => setCreateName(e.target.value)} />
          </div>
          <div style={{marginBottom: '10px'}}>
            <label>Dataset: </label>
            <input value={datasetKey} onChange={e => setDatasetKey(e.target.value)} placeholder="ungar_hcd-v1" />
          </div>
          <div style={{marginBottom: '10px'}}>
            <label>Model: </label>
            <input value={modelId} onChange={e => setModelId(e.target.value)} />
          </div>
          <div style={{marginBottom: '10px'}}>
            <label>Samples: </label>
            <input type="number" value={numSamples} onChange={e => setNumSamples(parseInt(e.target.value))} />
          </div>
          <p><em>Search space is hardcoded to LR (1e-5..1e-3) + Batch Size [4,8] for demo.</em></p>
          <button onClick={handleCreate} disabled={loading}>Create Job</button>
        </div>
      )}

      <table style={{width: '100%', borderCollapse: 'collapse'}}>
        <thead>
          <tr style={{textAlign: 'left', borderBottom: '2px solid #ccc'}}>
            <th>Status</th>
            <th>Name</th>
            <th>Dataset</th>
            <th>Best Params</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map(job => (
            <>
              <tr key={job.id} style={{borderBottom: '1px solid #eee'}}>
                <td style={{padding: '10px'}}>
                  <span className={`status-badge status-${job.status}`}>{job.status}</span>
                </td>
                <td>{job.name}</td>
                <td>{job.dataset_key}</td>
                <td>
                  {job.best_params_json ? JSON.stringify(job.best_params_json) : '-'}
                </td>
                <td>
                  {job.status === 'created' && (
                    <button onClick={() => handleStart(job.id)} style={{marginRight: '5px'}}>Start</button>
                  )}
                  <button onClick={() => handleView(job.id)}>
                    {selectedJobId === job.id ? 'Hide' : 'Details'}
                  </button>
                </td>
              </tr>
              {selectedJobId === job.id && selectedJob && (
                <tr key={`${job.id}-detail`}>
                  <td colSpan={5} style={{padding: '15px', backgroundColor: '#f9f9f9'}}>
                    <h4>Trials ({selectedJob.trials?.length || 0})</h4>
                    {selectedJob.trials && (
                      <table style={{width: '100%', fontSize: '0.9em'}}>
                        <thead>
                          <tr>
                            <th>ID</th>
                            <th>Status</th>
                            <th>Params</th>
                            <th>Metric ({selectedJob.metric_name})</th>
                          </tr>
                        </thead>
                        <tbody>
                          {selectedJob.trials.map(t => (
                            <tr key={t.id}>
                              <td>{t.id.slice(0, 8)}</td>
                              <td>{t.status}</td>
                              <td>{JSON.stringify(t.params_json)}</td>
                              <td>{t.metric_value?.toFixed(4) || '-'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </td>
                </tr>
              )}
            </>
          ))}
        </tbody>
      </table>
    </div>
  )
}
