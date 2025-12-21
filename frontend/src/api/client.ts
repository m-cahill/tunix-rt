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

