/**
 * Health check — EdgeOne Makers Node Function
 * ============================================
 *
 * File path: cloud-functions/health/index.ts → maps to **GET /health**
 *
 * Simple liveness probe. No AI involvement, no sandbox required —
 * lives in cloud-functions/ rather than agents/.
 */

import type { CloudFunctionContext } from '@edgeone/types';
import { createLogger } from '../_logger';

const logger = createLogger('health');

export async function onRequest(context: CloudFunctionContext) {
  const data = {
    status: 'ok',
    service: 'multimodal-file-assistant-agent',
    runId: context.run_id,
    timestamp: new Date().toISOString(),
  };

  logger.log('healthcheck:', data.status);

  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { 'Content-Type': 'application/json; charset=UTF-8' },
  });
}
