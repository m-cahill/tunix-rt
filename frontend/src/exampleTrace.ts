/**
 * Example reasoning trace for demonstration
 */

import type { ReasoningTrace } from './api/client'

export const EXAMPLE_TRACE: ReasoningTrace = {
  trace_version: '1.0',
  prompt: 'Convert 68°F to Celsius',
  final_answer: '20°C',
  steps: [
    {
      i: 0,
      type: 'parse',
      content: 'Parse the temperature conversion task: 68°F to °C',
    },
    {
      i: 1,
      type: 'formula',
      content: 'Use the formula: °C = (°F - 32) × 5/9',
    },
    {
      i: 2,
      type: 'compute',
      content: 'Calculate: (68 - 32) = 36',
    },
    {
      i: 3,
      type: 'compute',
      content: 'Calculate: 36 × 5/9 = 20',
    },
    {
      i: 4,
      type: 'result',
      content: 'Final answer: 20°C',
    },
  ],
  meta: {
    source: 'example',
    tags: ['temperature', 'conversion', 'fahrenheit', 'celsius'],
  },
}
