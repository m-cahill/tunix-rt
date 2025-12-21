/**
 * Typed API client for tunix-rt backend
 */

export interface HealthResponse {
  status: string
}

export interface RediHealthResponse {
  status: string
  error?: string
}

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
 */
async function fetchJSON<T>(url: string): Promise<T> {
  const response = await fetch(url)
  
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
 */
export async function getApiHealth(): Promise<HealthResponse> {
  return fetchJSON<HealthResponse>('/api/health')
}

/**
 * Get RediAI integration health status
 */
export async function getRediHealth(): Promise<RediHealthResponse> {
  return fetchJSON<RediHealthResponse>('/api/redi/health')
}

