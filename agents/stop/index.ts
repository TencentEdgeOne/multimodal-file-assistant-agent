/**
 * Stop handler — EdgeOne Makers Agent
 * ====================================
 *
 * File path: agents/stop/index.ts → maps to **POST /stop**
 *
 * Aborts the active agent run for the given conversation_id.
 *
 * IMPORTANT: the stop request MUST NOT carry the same `makers-conversation-id`
 * header as the chat request, otherwise EdgeOne sticky-routes /stop to the
 * busy chat instance and abortActiveRun() never reaches the runner.
 * The target conversation_id is therefore passed ONLY via the request body.
 */

import type { AgentContext } from '@edgeone/types';
const logger = {
  log(...args: unknown[]) {
    console.log(`[stop][${new Date().toISOString()}]`, ...args);
  },
  error(...args: unknown[]) {
    console.error(`[stop][${new Date().toISOString()}]`, ...args);
  },
};

export async function onRequest(context: AgentContext) {
  const { request } = context;
  const body = (request?.body ?? {}) as Record<string, unknown>;
  // Accept both snake_case and camelCase for backwards compatibility
  const conversationId = (body.conversation_id ?? body.conversationId) as string | undefined;
  logger.log('conversation_id:', conversationId);

  if (!conversationId) {
    logger.error('Missing conversation_id');
    return new Response('Missing conversation_id', { status: 400 });
  }

  const ret = context.utils.abortActiveRun(conversationId);
  logger.log('abortActiveRun result:', ret);

  const data = {
    status: ret?.aborted ? 'aborting' : 'idle',
    conversation_id: conversationId,
    ...ret,
  };

  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { 'Content-Type': 'application/json; charset=UTF-8' },
  });
}
