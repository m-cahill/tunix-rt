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
  status: 'pending' | 'running' | 'completed' | 'failed' | 'timeout'
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
 * Execute a Tunix training run (M13)
 * @param request - Run parameters (dataset_key, model_id, hyperparameters, dry_run flag)
 * @returns Promise resolving to run execution results
 * @throws {ApiError} on HTTP error (including 501 if Tunix not available and dry_run=false)
 */
export async function executeTunixRun(request: TunixRunRequest): Promise<TunixRunResponse> {
  return fetchJSON<TunixRunResponse>('/api/tunix/run', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })
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
  status: 'pending' | 'running' | 'completed' | 'failed' | 'timeout'
  started_at: string
  duration_seconds: number | null
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
