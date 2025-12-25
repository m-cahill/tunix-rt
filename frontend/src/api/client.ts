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
