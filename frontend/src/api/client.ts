/**
 * Typed API client for tunix-rt backend
 */

/**
 * Response from the tunix-rt health endpoint
 */
export interface HealthResponse {
  status: string
}

/**
 * Response from the RediAI health endpoint
 */
export interface RediHealthResponse {
  status: string
  error?: string
}

/**
 * A single step in a reasoning trace
 */
export interface TraceStep {
  i: number
  type: string
  content: string
}

/**
 * Complete reasoning trace payload
 */
export interface ReasoningTrace {
  trace_version: string
  prompt: string
  final_answer: string
  steps: TraceStep[]
  meta?: Record<string, any>
}

/**
 * Response from creating a trace
 */
export interface TraceCreateResponse {
  id: string
  created_at: string
  trace_version: string
}

/**
 * Full trace details including payload
 */
export interface TraceDetail {
  id: string
  created_at: string
  trace_version: string
  payload: ReasoningTrace
}

/**
 * Trace list item (without full payload)
 */
export interface TraceListItem {
  id: string
  created_at: string
  trace_version: string
}

/**
 * Pagination information
 */
export interface PaginationInfo {
  limit: number
  offset: number
  next_offset: number | null
}

/**
 * Response from listing traces
 */
export interface TraceListResponse {
  data: TraceListItem[]
  pagination: PaginationInfo
}

/**
 * Custom error class for API errors with HTTP status information
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public statusText: string
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

/**
 * Fetch wrapper with error handling and JSON parsing
 * @template T - The expected response type
 * @param url - The URL to fetch
 * @param options - Fetch options (method, body, headers, etc.)
 * @returns Promise resolving to the parsed JSON response
 * @throws {ApiError} on HTTP error responses
 */
async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options)

  if (!response.ok) {
    throw new ApiError(
      `HTTP error: ${response.statusText}`,
      response.status,
      response.statusText
    )
  }

  return response.json()
}

/**
 * Get tunix-rt application health status
 * @returns Promise resolving to health status
 * @throws {ApiError} on HTTP error
 */
export async function getApiHealth(): Promise<HealthResponse> {
  return fetchJSON<HealthResponse>('/api/health')
}

/**
 * Get RediAI integration health status
 * @returns Promise resolving to RediAI health status with optional error details
 * @throws {ApiError} on HTTP error
 */
export async function getRediHealth(): Promise<RediHealthResponse> {
  return fetchJSON<RediHealthResponse>('/api/redi/health')
}

/**
 * Create a new reasoning trace
 * @param trace - The trace payload to create
 * @returns Promise resolving to trace creation response with ID and metadata
 * @throws {ApiError} on HTTP error (including 413 for oversized payload, 422 for validation errors)
 */
export async function createTrace(trace: ReasoningTrace): Promise<TraceCreateResponse> {
  return fetchJSON<TraceCreateResponse>('/api/traces', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(trace),
  })
}

/**
 * Get a trace by ID
 * @param id - The trace UUID
 * @returns Promise resolving to full trace details
 * @throws {ApiError} on HTTP error (including 404 if trace not found)
 */
export async function getTrace(id: string): Promise<TraceDetail> {
  return fetchJSON<TraceDetail>(`/api/traces/${id}`)
}

/**
 * List traces with pagination
 * @param params - Pagination parameters (limit and offset)
 * @returns Promise resolving to paginated list of traces
 * @throws {ApiError} on HTTP error
 */
export async function listTraces(params?: { limit?: number; offset?: number }): Promise<TraceListResponse> {
  const searchParams = new URLSearchParams()
  if (params?.limit) searchParams.set('limit', params.limit.toString())
  if (params?.offset) searchParams.set('offset', params.offset.toString())

  const url = `/api/traces${searchParams.toString() ? `?${searchParams.toString()}` : ''}`
  return fetchJSON<TraceListResponse>(url)
}

/**
 * Score details from evaluation
 */
export interface ScoreDetails {
  step_count: number
  avg_step_length: number
  total_chars: number
  step_score: number
  length_score: number
  criteria: string
  scored_at: string
}

/**
 * Response from scoring a trace
 */
export interface ScoreResponse {
  trace_id: string
  score: number
  details: ScoreDetails
}

/**
 * Trace with associated score
 */
export interface TraceWithScore {
  id: string
  created_at: string
  score: number
  trace_version: string
  payload: ReasoningTrace
}

/**
 * Response from comparing two traces
 */
export interface CompareResponse {
  base: TraceWithScore
  other: TraceWithScore
}

/**
 * Score a trace using specified criteria
 * @param id - The trace UUID
 * @param criteria - Scoring criteria (default: 'baseline')
 * @returns Promise resolving to score response with value and details
 * @throws {ApiError} on HTTP error (including 404 if trace not found)
 */
export async function scoreTrace(id: string, criteria: string = 'baseline'): Promise<ScoreResponse> {
  return fetchJSON<ScoreResponse>(`/api/traces/${id}/score`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ criteria }),
  })
}

/**
 * Compare two traces side-by-side with scores
 * @param baseId - UUID of the base trace
 * @param otherId - UUID of the other trace
 * @returns Promise resolving to comparison with both traces and scores
 * @throws {ApiError} on HTTP error (including 404 if either trace not found)
 */
export async function compareTraces(baseId: string, otherId: string): Promise<CompareResponse> {
  const searchParams = new URLSearchParams()
  searchParams.set('base', baseId)
  searchParams.set('other', otherId)

  return fetchJSON<CompareResponse>(`/api/traces/compare?${searchParams.toString()}`)
}

/**
 * UNGAR availability status
 */
export interface UngarStatusResponse {
  available: boolean
  version: string | null
}

/**
 * Request parameters for UNGAR trace generation
 */
export interface UngarGenerateRequest {
  count: number
  seed?: number | null
  persist?: boolean
}

/**
 * Response from UNGAR trace generation
 */
export interface UngarGenerateResponse {
  trace_ids: string[]
  preview: Array<{
    trace_id: string
    game: string | null
    result: string | null
    my_card: string | null
  }>
}

/**
 * Get UNGAR availability status
 * @returns Promise resolving to UNGAR status (always succeeds, check 'available' field)
 */
export async function getUngarStatus(): Promise<UngarStatusResponse> {
  return fetchJSON<UngarStatusResponse>('/api/ungar/status')
}

/**
 * Generate High Card Duel traces from UNGAR
 * @param request - Generation parameters (count, seed, persist)
 * @returns Promise resolving to generated trace IDs and preview
 * @throws {ApiError} on HTTP error (including 501 if UNGAR not installed)
 */
export async function generateUngarTraces(request: UngarGenerateRequest): Promise<UngarGenerateResponse> {
  return fetchJSON<UngarGenerateResponse>('/api/ungar/high-card-duel/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })
}

/**
 * Tunix availability status (M12)
 */
export interface TunixStatusResponse {
  available: boolean
  version: string | null
  runtime_required: boolean
  message: string
}

/**
 * Request parameters for Tunix export
 */
export interface TunixExportRequest {
  dataset_key?: string | null
  trace_ids?: string[] | null
  limit?: number
}

/**
 * Request parameters for Tunix manifest generation
 */
export interface TunixManifestRequest {
  dataset_key: string
  model_id: string
  output_dir: string
  learning_rate?: number
  num_epochs?: number
  batch_size?: number
  max_seq_length?: number
}

/**
 * Response from Tunix manifest generation
 */
export interface TunixManifestResponse {
  manifest_yaml: string
  dataset_key: string
  model_id: string
  format: string
  message: string
}

/**
 * Request parameters for Tunix run execution (M13)
 */
export interface TunixRunRequest {
  dataset_key: string
  model_id: string
  output_dir?: string | null
  dry_run?: boolean
  learning_rate?: number
  num_epochs?: number
  batch_size?: number
  max_seq_length?: number
}

/**
 * Response from Tunix run execution (M13)
 */
export interface TunixRunResponse {
  run_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'timeout' | 'cancel_requested' | 'cancelled'
  mode: 'dry-run' | 'local'
  dataset_key: string
  model_id: string
  output_dir: string
  exit_code: number | null
  stdout: string
  stderr: string
  duration_seconds: number | null
  started_at: string
  completed_at: string | null
  message: string
}

/**
 * Response from checking Tunix run status (M15)
 */
export interface TunixRunStatusResponse {
  run_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'timeout' | 'cancel_requested' | 'cancelled'
  queued_at: string
  started_at: string
  completed_at: string | null
  exit_code: number | null
}

/**
 * Get Tunix availability status
 * @returns Promise resolving to Tunix status (always succeeds, check fields)
 */
export async function getTunixStatus(): Promise<TunixStatusResponse> {
  return fetchJSON<TunixStatusResponse>('/api/tunix/status')
}

/**
 * Export traces in Tunix SFT format (JSONL)
 * @param request - Export parameters (dataset_key OR trace_ids)
 * @returns Promise resolving to JSONL blob
 * @throws {ApiError} on HTTP error (including 400 if neither dataset_key nor trace_ids provided)
 */
export async function exportTunixSft(request: TunixExportRequest): Promise<Blob> {
  const response = await fetch('/api/tunix/sft/export', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new ApiError(
      `HTTP error: ${response.statusText}`,
      response.status,
      response.statusText
    )
  }

  return response.blob()
}

/**
 * Generate Tunix SFT training manifest
 * @param request - Manifest generation parameters
 * @returns Promise resolving to manifest response with YAML content
 * @throws {ApiError} on HTTP error (including 404 if dataset not found)
 */
export async function generateTunixManifest(request: TunixManifestRequest): Promise<TunixManifestResponse> {
  return fetchJSON<TunixManifestResponse>('/api/tunix/sft/manifest', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })
}

/**
 * Execute a Tunix training run (M13/M15)
 * @param request - Run parameters (dataset_key, model_id, hyperparameters, dry_run flag)
 * @param options - Execution options (mode='async' for non-blocking)
 * @returns Promise resolving to run execution results
 * @throws {ApiError} on HTTP error (including 501 if Tunix not available and dry_run=false)
 */
export async function executeTunixRun(request: TunixRunRequest, options?: { mode?: 'async' }): Promise<TunixRunResponse> {
  const url = options?.mode === 'async' ? '/api/tunix/run?mode=async' : '/api/tunix/run'
  return fetchJSON<TunixRunResponse>(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })
}

/**
 * Get status of a Tunix training run (M15)
 * @param runId - UUID of the run
 * @returns Promise resolving to run status
 */
export async function getTunixRunStatus(runId: string): Promise<TunixRunStatusResponse> {
  return fetchJSON<TunixRunStatusResponse>(`/api/tunix/runs/${runId}/status`)
}

/**
 * M14: Run registry types and functions
 */

/**
 * Tunix run list item (summary without full logs)
 */
export interface TunixRunListItem {
  run_id: string
  dataset_key: string
  model_id: string
  mode: 'dry-run' | 'local'
  status: 'pending' | 'running' | 'completed' | 'failed' | 'timeout' | 'cancel_requested' | 'cancelled'
  started_at: string
  duration_seconds: number | null
  metrics?: Record<string, number | string | null> | null
}

/**
 * Response from listing Tunix runs
 */
export interface TunixRunListResponse {
  data: TunixRunListItem[]
  pagination: {
    limit: number
    offset: number
    next_offset: number | null
  }
}

/**
 * Parameters for filtering Tunix runs
 */
export interface ListTunixRunsParams {
  limit?: number
  offset?: number
  status?: string
  dataset_key?: string
  mode?: string
}

/**
 * List Tunix training runs with pagination and filtering (M14)
 * @param params - Pagination and filter parameters
 * @returns Promise resolving to paginated list of run summaries
 * @throws {ApiError} on HTTP error
 */
export async function listTunixRuns(params?: ListTunixRunsParams): Promise<TunixRunListResponse> {
  const searchParams = new URLSearchParams()
  if (params?.limit) searchParams.set('limit', params.limit.toString())
  if (params?.offset) searchParams.set('offset', params.offset.toString())
  if (params?.status) searchParams.set('status', params.status)
  if (params?.dataset_key) searchParams.set('dataset_key', params.dataset_key)
  if (params?.mode) searchParams.set('mode', params.mode)

  const url = `/api/tunix/runs${searchParams.toString() ? `?${searchParams.toString()}` : ''}`
  return fetchJSON<TunixRunListResponse>(url)
}

/**
 * Get full details of a Tunix training run by ID (M14)
 * @param runId - UUID of the run
 * @returns Promise resolving to full run details (reuses TunixRunResponse schema)
 * @throws {ApiError} on HTTP error (including 404 if run not found)
 */
export async function getTunixRun(runId: string): Promise<TunixRunResponse> {
  return fetchJSON<TunixRunResponse>(`/api/tunix/runs/${runId}`)
}

/**
 * M16: Artifacts and Cancellation
 */

export interface Artifact {
  name: string
  size: number
  path: string
}

export interface ArtifactListResponse {
  artifacts: Artifact[]
}

/**
 * List artifacts for a Tunix run
 * @param runId - UUID of the run
 * @returns Promise resolving to list of artifacts
 */
export async function listArtifacts(runId: string): Promise<ArtifactListResponse> {
  return fetchJSON<ArtifactListResponse>(`/api/tunix/runs/${runId}/artifacts`)
}

/**
 * Cancel a pending or running Tunix run
 * @param runId - UUID of the run
 * @returns Promise resolving to status message
 */
export async function cancelRun(runId: string): Promise<{ message: string }> {
  return fetchJSON<{ message: string }>(`/api/tunix/runs/${runId}/cancel`, {
    method: 'POST'
  })
}

/**
 * M17: Evaluation types and functions
 */

export interface EvaluationMetric {
  name: string
  score: number
  max_score: number
  details?: Record<string, any>
}

export interface EvaluationJudgeInfo {
  name: string
  version: string
}

export interface EvaluationResponse {
  evaluation_id: string
  run_id: string
  score: number
  verdict: 'pass' | 'fail' | 'uncertain'
  metrics: Record<string, number>
  detailed_metrics: EvaluationMetric[]
  judge: EvaluationJudgeInfo
  evaluated_at: string
}

/**
 * M35: Scorecard summary for quick display
 */
export interface ScorecardSummary {
  n_items: number
  n_scored: number
  n_skipped: number
  primary_score: number | null
  stddev: number | null
}

/**
 * M35: Leaderboard filter options
 */
export interface LeaderboardFilters {
  dataset_key?: string | null
  model_id?: string | null
  config_path?: string | null
  date_from?: string | null
  date_to?: string | null
}

export interface LeaderboardItem {
  run_id: string
  model_id: string
  dataset_key: string
  config_path?: string | null
  score: number
  verdict: string
  metrics: Record<string, number>
  evaluated_at: string
  primary_score?: number | null
  scorecard?: ScorecardSummary | null
}

export interface LeaderboardResponse {
  data: LeaderboardItem[]
  pagination?: PaginationInfo
  filters?: LeaderboardFilters | null
}

/**
 * Trigger evaluation for a completed run
 * @param runId - UUID of the run
 * @param request - Optional params
 */
export async function evaluateRun(runId: string, request?: { judge_override?: string }): Promise<EvaluationResponse> {
  return fetchJSON<EvaluationResponse>(`/api/tunix/runs/${runId}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: request ? JSON.stringify(request) : undefined
  })
}

/**
 * Get evaluation details for a run
 * @param runId - UUID of the run
 */
export async function getEvaluation(runId: string): Promise<EvaluationResponse> {
  return fetchJSON<EvaluationResponse>(`/api/tunix/runs/${runId}/evaluation`)
}

/**
 * Get leaderboard data with optional filtering (M35)
 * @param limit - Max items per page (default 50)
 * @param offset - Pagination offset (default 0)
 * @param filters - Optional filter criteria (dataset, model, config, date range)
 */
export async function getLeaderboard(
  limit: number = 50,
  offset: number = 0,
  filters?: LeaderboardFilters
): Promise<LeaderboardResponse> {
  const params = new URLSearchParams()
  params.set('limit', limit.toString())
  params.set('offset', offset.toString())

  // M35: Add filter parameters if provided
  if (filters?.dataset_key) params.set('dataset_key', filters.dataset_key)
  if (filters?.model_id) params.set('model_id', filters.model_id)
  if (filters?.config_path) params.set('config_path', filters.config_path)
  if (filters?.date_from) params.set('date_from', filters.date_from)
  if (filters?.date_to) params.set('date_to', filters.date_to)

  return fetchJSON<LeaderboardResponse>(`/api/tunix/evaluations?${params.toString()}`)
}

/**
 * M19: Tuning Types and Functions
 */

export interface TuningJobCreate {
  name: string
  dataset_key: string
  base_model_id: string
  metric_name?: string
  metric_mode?: 'max' | 'min'
  num_samples?: number
  max_concurrent_trials?: number
  search_space: Record<string, any>
}

export interface TuningTrial {
  id: string
  tuning_job_id: string
  run_id: string | null
  params_json: Record<string, any>
  metric_value: number | null
  status: string
  error: string | null
  created_at: string
  completed_at: string | null
}

export interface TuningJob {
  id: string
  name: string
  status: string
  dataset_key: string
  base_model_id: string
  mode: string
  metric_name: string
  metric_mode: string
  num_samples: number
  max_concurrent_trials: number
  search_space_json: Record<string, any>
  best_run_id: string | null
  best_params_json: Record<string, any> | null
  created_at: string
  started_at: string | null
  completed_at: string | null
  trials?: TuningTrial[]
}

export interface TuningJobStartResponse {
  job_id: string
  status: string
  message: string
}

export async function createTuningJob(request: TuningJobCreate): Promise<TuningJob> {
  return fetchJSON<TuningJob>('/api/tuning/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  })
}

export async function startTuningJob(jobId: string): Promise<TuningJobStartResponse> {
  return fetchJSON<TuningJobStartResponse>(`/api/tuning/jobs/${jobId}/start`, {
    method: 'POST'
  })
}

export async function listTuningJobs(limit: number = 20, offset: number = 0): Promise<TuningJob[]> {
  const params = new URLSearchParams()
  params.set('limit', limit.toString())
  params.set('offset', offset.toString())
  return fetchJSON<TuningJob[]>(`/api/tuning/jobs?${params.toString()}`)
}

export async function getTuningJob(jobId: string): Promise<TuningJob> {
  return fetchJSON<TuningJob>(`/api/tuning/jobs/${jobId}`)
}

/**
 * M20: Model Registry
 */

export interface ModelVersion {
  id: string
  artifact_id: string
  version: string
  source_run_id: string | null
  status: string
  metrics_json: Record<string, any> | null
  config_json: Record<string, any> | null
  provenance_json: Record<string, any> | null
  storage_uri: string
  sha256: string
  size_bytes: number
  created_at: string
}

export interface ModelArtifact {
  id: string
  name: string
  description: string | null
  task_type: string | null
  created_at: string
  updated_at: string
  latest_version?: ModelVersion
}

export interface ModelArtifactCreate {
  name: string
  description?: string
  task_type?: string
}

export interface ModelPromotionRequest {
  source_run_id: string
  version_label?: string
  description?: string
}

export async function createModelArtifact(request: ModelArtifactCreate): Promise<ModelArtifact> {
  return fetchJSON<ModelArtifact>('/api/models', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
}

export async function listModelArtifacts(): Promise<ModelArtifact[]> {
  return fetchJSON<ModelArtifact[]>('/api/models')
}

export async function getModelArtifact(artifactId: string): Promise<ModelArtifact> {
  return fetchJSON<ModelArtifact>(`/api/models/${artifactId}`)
}

export async function promoteRunToVersion(
  artifactId: string,
  request: ModelPromotionRequest
): Promise<ModelVersion> {
  return fetchJSON<ModelVersion>(`/api/models/${artifactId}/versions/promote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
}

export async function getModelVersion(versionId: string): Promise<ModelVersion> {
  return fetchJSON<ModelVersion>(`/api/models/versions/${versionId}`)
}

export function getModelDownloadUrl(versionId: string): string {
  return `/api/models/versions/${versionId}/download`
}

/**
 * M26: Tunix Run Metrics
 */
export interface TunixRunMetric {
  step: number
  epoch: number
  loss: number
  timestamp: string
  device?: string
}

export async function getTunixRunMetrics(runId: string): Promise<TunixRunMetric[]> {
  return fetchJSON<TunixRunMetric[]>(`/api/tunix/runs/${runId}/metrics`)
}

/**
 * M29: Dataset Building and Export
 */

/**
 * Request to build a new dataset from traces
 */
export interface DatasetBuildRequest {
  dataset_name: string
  dataset_version: string
  filters?: Record<string, any>
  limit?: number
  selection_strategy?: 'latest' | 'random'
  seed?: number | null
  session_id?: string | null
  parent_dataset_id?: string | null
  training_run_id?: string | null
  provenance?: Record<string, any>
}

/**
 * Response from dataset build request
 */
export interface DatasetBuildResponse {
  dataset_key: string
  build_id: string
  trace_count: number
  manifest_path: string
}

/**
 * Request to ingest traces from a JSONL file
 */
export interface DatasetIngestRequest {
  path: string
  source_name: string
}

/**
 * Response from dataset ingest request
 */
export interface DatasetIngestResponse {
  ingested_count: number
  trace_ids: string[]
}

/**
 * Build a new dataset from traces
 * @param request - Dataset build parameters
 * @returns Promise resolving to build response with dataset key and trace count
 * @throws {ApiError} on HTTP error
 */
export async function buildDataset(request: DatasetBuildRequest): Promise<DatasetBuildResponse> {
  return fetchJSON<DatasetBuildResponse>('/api/datasets/build', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
}

/**
 * Get the URL for exporting a dataset
 * @param datasetName - Dataset name
 * @param datasetVersion - Dataset version
 * @returns URL string for dataset export endpoint
 */
export function getDatasetExportUrl(datasetName: string, datasetVersion: string): string {
  return `/api/datasets/${datasetName}/${datasetVersion}/export`
}

/**
 * Export a dataset as a blob
 * @param datasetName - Dataset name
 * @param datasetVersion - Dataset version
 * @returns Promise resolving to dataset blob
 * @throws {ApiError} on HTTP error
 */
export async function exportDataset(datasetName: string, datasetVersion: string): Promise<Blob> {
  const response = await fetch(getDatasetExportUrl(datasetName, datasetVersion))
  if (!response.ok) {
    throw new ApiError(
      `HTTP error: ${response.statusText}`,
      response.status,
      response.statusText
    )
  }
  return response.blob()
}

/**
 * Ingest traces from a server-side JSONL file
 * @param request - Ingest parameters (path and source name)
 * @returns Promise resolving to ingest response with count and trace IDs
 * @throws {ApiError} on HTTP error
 */
export async function ingestDataset(request: DatasetIngestRequest): Promise<DatasetIngestResponse> {
  return fetchJSON<DatasetIngestResponse>('/api/datasets/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
}
