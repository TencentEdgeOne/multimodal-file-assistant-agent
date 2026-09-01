/**
 * Document Processing Agent — EdgeOne Makers Functions
 * ====================================================
 *
 * File path: agents/chat/index.ts → maps to **POST /chat**
 *
 * Uses @anthropic-ai/claude-agent-sdk with dual MCP server pattern:
 *   1. EdgeOne sandbox MCP server  — code_interpreter, commands, files
 *   2. Custom tools MCP server     — suggest_actions, deliver_file
 */

import type { AgentContext } from '@edgeone/types';
import {
  query,
  createSdkMcpServer,
  getSessionInfo,
} from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";
import { resolveModelName, collectGatewayEnv } from "../_model";
import { createLogger, sseEvent, createSSEResponse } from "../_shared";
import { buildSystemPrompt } from "./_skills";
import {
  shellQuote,
  canInlineFallbackFile,
  buildDefaultActions,
} from "./_tools";

const logger = createLogger("chat");

// Prevent SDK observability streams from crashing the process on EPIPE.
//
// The Claude Agent SDK spawns internal observability Sockets for telemetry.
// When the HTTP response stream is closed by the client (or the runtime
// truncates the pipe), those Sockets emit an unhandled 'error' event with
// code === 'EPIPE'. Node's default behavior is to crash the process on any
// unhandled 'error' event, which would tear down the agent runtime. We
// register defensive listeners at three layers:
//
//   1. process.stdout / process.stderr — catches the common case where the
//      SDK writes directly to stdio (e.g. console.log) after the pipe closes.
//   2. process.on('uncaughtException') — last-resort net for any internal
//      SDK Socket whose 'error' event had no handler. EPIPE here is benign
//      (the client is gone, nothing to write to); anything else is logged
//      and re-thrown so the runtime's own error handling still runs.
process.stdout.on("error", (err: any) => {
  if (err?.code === "EPIPE") return;
});
process.stderr.on("error", (err: any) => {
  if (err?.code === "EPIPE") return;
});
process.on("uncaughtException", (err: any) => {
  if (err?.code === "EPIPE") {
    // Client disconnected mid-stream; ignore and keep serving other requests.
    return;
  }
  // Log and re-throw so Node's default behavior (crash + restart) still
  // applies to real exceptions.
  console.error("[chat] uncaughtException:", err);
  throw err;
});

/** Normalize a string to a valid UUID or return null */
function normalizeUuid(id: string): string | null {
  if (!id) return null;
  const uuidRe =
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  if (uuidRe.test(id)) return id.toLowerCase();
  // Pad/truncate to a pseudo-UUID so sessions can be keyed by conversationId
  const hex = id
    .replace(/[^0-9a-f]/gi, "")
    .padEnd(32, "0")
    .slice(0, 32);
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(
    12,
    16
  )}-${hex.slice(16, 20)}-${hex.slice(20, 32)}`;
}

/**
 * Tracks session IDs this worker has already started a CLI session for.
 * The CLI distinguishes `--session-id` (start new) from `--resume` (continue):
 * passing `--session-id` for an ID that already exists fails with
 * "Session ID <id> is already in use" → the subprocess exits 1. So once we
 * have created a session in this process, every follow-up must `resume` it.
 */
const _startedSessions = new Set<string>();

/** Resolve session binding: resume existing session or create new one */
async function resolveClaudeSessionBinding(
  sessionStore: any,
  conversationId: string,
  cwd: string
): Promise<{ resume?: string; sessionId?: string }> {
  const sessionId = normalizeUuid(conversationId);
  if (!sessionId) return {};

  // In-process follow-up: this worker already opened the session, so resume it
  // instead of trying to re-create it (which collides with "already in use").
  if (_startedSessions.has(sessionId)) {
    logger.log(`[session] resuming (in-process) session: ${sessionId}`);
    return { resume: sessionId };
  }

  try {
    const infoOptions: any = { dir: cwd };
    if (sessionStore?.load) infoOptions.sessionStore = sessionStore;
    const info = await getSessionInfo(sessionId, infoOptions);
    if (info) {
      logger.log(`[session] resuming existing session: ${sessionId}`);
      _startedSessions.add(sessionId);
      return { resume: sessionId };
    }
  } catch {
    // getSessionInfo may throw if session store is unavailable
  }

  logger.log(`[session] creating new session: ${sessionId}`);
  _startedSessions.add(sessionId);
  return { sessionId };
}

/** In-process file cache: persists uploaded files across follow-up requests within same process */
const _sessionFileCache = new Map<
  string,
  Array<{ name: string; base64: string }>
>();

export async function onRequest(context: AgentContext) {
  const ctxEnv: Record<string, string | undefined> = context.env ?? process.env;

  const body = context.request.body ?? {};
  let message = typeof body.message === "string" ? body.message.trim() : "";
  const uploadedFiles: Array<{ name: string; base64: string }> =
    body.files ?? [];

  // Detect user locale from the language hint appended by the frontend
  const locale: "zh" | "en" = message.includes("[语言要求：")
    ? "zh"
    : message.includes("[Language:")
    ? "en"
    : "en";

  if (!message) {
    return new Response(JSON.stringify({ error: "'message' is required" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  const signal: AbortSignal | undefined = context.request.signal;
  // EdgeOne Makers auto-injects `context.conversation_id` from the
  // `makers-conversation-id` HTTP header.
  const conversationId: string = context.conversation_id || "";
  const sandbox = context.sandbox ?? null;

  // ─── Session file cache: persist uploaded files across follow-up requests ────
  // The EdgeOne sandbox /tmp/ is ephemeral — re-upload cached files on every request.
  const cachedSessionFiles: Array<{ name: string; base64: string }> =
    conversationId ? _sessionFileCache.get(conversationId) ?? [] : [];

  if (conversationId && uploadedFiles.length > 0) {
    // Merge new files with cached ones (new files override existing with same name)
    const mergedMap = new Map(cachedSessionFiles.map((f) => [f.name, f]));
    uploadedFiles.forEach((f) => mergedMap.set(f.name, f));
    _sessionFileCache.set(conversationId, Array.from(mergedMap.values()));
  }

  // On follow-up requests (no new files), re-upload all cached files to the (possibly fresh) sandbox
  const filesToUpload: Array<{ name: string; base64: string }> =
    uploadedFiles.length > 0
      ? _sessionFileCache.get(conversationId) ?? uploadedFiles
      : cachedSessionFiles;
  const store = context.store ?? null;
  const cwd = process.cwd();

  logger.log(
    `[request] cid=${conversationId}, message="${message.slice(
      0,
      80
    )}...", files=${uploadedFiles.length}`
  );

  // ─── Sandbox readiness check ────────────────────────────────────────────────
  let sandboxWorking = false;

  if (sandbox) {
    try {
      await sandbox.commands.run("ls /tmp", { timeout: 10 });
      sandboxWorking = true;
      logger.log("[sandbox] ready");
    } catch {
      // Retry with delay (sandbox may be cold-starting)
      for (let attempt = 0; attempt < 2; attempt++) {
        await new Promise((r) => setTimeout(r, 2000));
        try {
          await sandbox.commands.run("ls /tmp", { timeout: 10 });
          sandboxWorking = true;
          logger.log(`[sandbox] ready (after ${attempt + 1} retries)`);
          break;
        } catch {
          logger.log(
            `[sandbox] not ready, retrying... (attempt ${attempt + 1})`
          );
        }
      }
    }
  }

  // ─── Write uploaded files to sandbox ────────────────────────────────────────
  if (sandboxWorking && filesToUpload.length > 0) {
    for (const file of filesToUpload) {
      try {
        const sandboxPath = `/tmp/${file.name}`;
        let uploadSuccess = false;

        const runCmd = async (cmd: string): Promise<string> => {
          const r = await sandbox.commands.run(cmd, { timeout: 30 });
          return r.stdout || "";
        };

        // Strategy 1: write base64 as text → decode with shell
        try {
          const b64TmpPath = "/tmp/__upload_b64.tmp";
          await sandbox.files.write(b64TmpPath, file.base64);
          await runCmd(
            `base64 -d ${b64TmpPath} > ${shellQuote(
              sandboxPath
            )} && rm -f ${b64TmpPath}`
          );

          const sizeStr = await runCmd(
            `stat -c %s ${shellQuote(sandboxPath)} 2>/dev/null || echo 0`
          );
          if ((parseInt(sizeStr.trim(), 10) || 0) > 0) {
            uploadSuccess = true;
            logger.log(
              `[upload] success via files.write+decode: ${sandboxPath}`
            );
          }
        } catch (e) {
          logger.log(
            `[upload] files.write method failed: ${(e as Error).message}`
          );
        }

        // Strategy 2: Python base64 decode (fallback)
        if (!uploadSuccess) {
          try {
            const runCode = (code: string) =>
              sandbox.runCode(code, { language: "python" });

            if (file.base64.length <= 150_000) {
              await runCode(
                `import base64\nwith open("${sandboxPath}", "wb") as f:\n    f.write(base64.b64decode("${file.base64}"))\nprint("ok")`
              );
            } else {
              const chunkSize = 150_000;
              const totalChunks = Math.ceil(file.base64.length / chunkSize);
              await runCode(`open("/tmp/__b64tmp", "w").close()`);
              for (let i = 0; i < totalChunks; i++) {
                const chunk = file.base64.slice(
                  i * chunkSize,
                  (i + 1) * chunkSize
                );
                await runCode(`open("/tmp/__b64tmp", "a").write("${chunk}")`);
              }
              await runCode(
                `import base64, os\nwith open("/tmp/__b64tmp") as f:\n    d = base64.b64decode(f.read())\nwith open("${sandboxPath}", "wb") as f:\n    f.write(d)\nos.remove("/tmp/__b64tmp")\nprint(len(d))`
              );
            }

            const sizeStr = await runCmd(
              `stat -c %s ${shellQuote(sandboxPath)} 2>/dev/null || echo 0`
            );
            if ((parseInt(sizeStr.trim(), 10) || 0) > 0) {
              uploadSuccess = true;
              logger.log(`[upload] success via runCode: ${sandboxPath}`);
            }
          } catch (e) {
            logger.log(
              `[upload] runCode method failed: ${(e as Error).message}`
            );
          }
        }

        if (!uploadSuccess) {
          logger.error(`[upload] ALL methods failed for ${file.name}`);
          sandboxWorking = false;
          break;
        }
      } catch (e) {
        logger.error(`[upload] failed for ${file.name}:`, (e as Error).message);
        sandboxWorking = false;
        break;
      }
    }

    // Tell the AI files are ready — no need to list /tmp
    if (sandboxWorking && filesToUpload.length > 0) {
      const fileList = filesToUpload.map((f) => `/tmp/${f.name}`).join(", ");
      const sysMsg =
        locale === "zh"
          ? `\n\n[系统提示：文件已就绪，路径为: ${fileList}。请直接分析和处理，不需要先 list 目录确认。]`
          : `\n\n[System: Files are ready at: ${fileList}. Start analysis directly — do not list the directory first.]`;
      message = message + sysMsg;
    }
  }

  // ─── Fallback: inline text file content when sandbox unavailable ─────────────
  if (!sandboxWorking && uploadedFiles.length > 0) {
    logger.log(
      "[fallback] sandbox unavailable, inlining text file content into message"
    );
    let inlineContent =
      "\n\n--- FILE CONTENTS (sandbox unavailable, analyze text files below) ---\n";
    const skippedFiles: string[] = [];

    for (const file of uploadedFiles) {
      try {
        const content = Buffer.from(file.base64, "base64");
        if (!canInlineFallbackFile(file.name, content)) {
          skippedFiles.push(file.name);
          continue;
        }
        const decoded = content.toString("utf8");
        inlineContent += `\n### File: ${file.name}\n\`\`\`\n${decoded}\n\`\`\`\n`;
      } catch {
        skippedFiles.push(file.name);
      }
    }

    if (skippedFiles.length > 0) {
      inlineContent += `\nSkipped binary or non-text files because sandbox is unavailable: ${skippedFiles.join(
        ", "
      )}\n`;
    }
    message = message + inlineContent;
  }

  // ─── Custom MCP tools (suggest_actions + deliver_file) ───────────────────────
  // SSE side-channel: tool handlers push events; the message loop drains & yields
  const sseQueue: string[] = [];
  let suggestActionsCalled = false;
  let deliverFileCalled = false;

  const customMcpServer = createSdkMcpServer({
    name: "custom-tools",
    alwaysLoad: true,
    tools: [
      {
        name: "suggest_actions",
        description:
          "Present a list of recommended actions to the user as clickable options. " +
          "Use this when you have analysed files and want to suggest processing options.",
        inputSchema: {
          actions: z
            .array(
              z.object({
                id: z.string().describe("Unique action ID (e.g. action_1)"),
                emoji: z.string().describe("Emoji icon for the action"),
                title: z.string().describe("Short title (under 20 chars)"),
                description: z
                  .string()
                  .describe("Brief description of what this action does"),
              })
            )
            .describe("List of suggested actions"),
        },
        handler: async ({ actions }: { actions: any[] }) => {
          suggestActionsCalled = true;
          sseQueue.push(sseEvent({ type: "suggest_actions", actions }));
          return {
            content: [
              {
                type: "text" as const,
                text: "Suggestions have been displayed to the user. Wait for them to choose an action.",
              },
            ],
          };
        },
      },
      {
        name: "deliver_file",
        description:
          "Deliver a processed file to the user for download. " +
          "Call this after generating an output file (e.g. merged PDF, converted image). " +
          "The file will be sent as a downloadable link.",
        inputSchema: {
          path: z
            .string()
            .describe(
              "Absolute path to the output file in sandbox (e.g. /tmp/report.pdf)"
            ),
          filename: z
            .string()
            .describe("Display filename for the user (e.g. report.pdf)"),
          description: z
            .string()
            .optional()
            .describe("Brief description of the file content"),
        },
        handler: async ({
          path,
          filename,
          description,
        }: {
          path: string;
          filename: string;
          description?: string;
        }) => {
          deliverFileCalled = true;
          let base64 = "";
          try {
            if (sandbox?.commands?.run) {
              const r = await sandbox.commands.run(
                `base64 -w 0 ${shellQuote(path)}`
              );
              base64 = (r.stdout || "").trim();
            }
          } catch (e) {
            logger.error(
              "[deliver_file] failed to read file:",
              (e as Error).message
            );
            return {
              content: [
                {
                  type: "text" as const,
                  text: `Error reading file "${path}": ${(e as Error).message}`,
                },
              ],
            };
          }

          if (!base64 && sandbox?.commands?.run) {
            // Fallback: scan /tmp for the most recently created file matching the extension
            try {
              const ext = filename.split(".").pop() || "pdf";
              const scanResult = await sandbox.commands.run(
                `ls -t /tmp/*.${ext} 2>/dev/null | head -1`
              );
              const altPath = scanResult.stdout?.trim();
              if (altPath && altPath !== path) {
                logger.log(`[deliver_file] fallback: trying ${altPath}`);
                const r2 = await sandbox.commands.run(
                  `base64 -w 0 ${shellQuote(altPath)}`
                );
                base64 = (r2.stdout || "").trim();
              }
            } catch {
              // Ignore fallback scan errors
            }
          }

          if (!base64) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: `File not found or empty: ${path}`,
                },
              ],
            };
          }

          sseQueue.push(
            sseEvent({
              type: "file_output",
              base64,
              filename,
              description: description ?? "",
            })
          );
          return {
            content: [
              {
                type: "text" as const,
                text: `File "${filename}" has been delivered to the user for download.`,
              },
            ],
          };
        },
      },
    ],
  });

  // ─── EdgeOne sandbox MCP server ──────────────────────────────────────────────
  let sandboxMcpServer: any = null;
  let edgeoneMcp: any = null;
  if (
    sandboxWorking &&
    typeof context.tools?.toClaudeMcpServer === "function"
  ) {
    try {
      edgeoneMcp = context.tools.toClaudeMcpServer();
      sandboxMcpServer = createSdkMcpServer({
        name: edgeoneMcp.name,
        tools: edgeoneMcp.tools,
        alwaysLoad: true,
      });
      logger.log("[mcp] EdgeOne sandbox MCP server created");
    } catch (e) {
      logger.error(
        "[mcp] failed to create sandbox MCP server:",
        (e as Error).message
      );
    }
  }

  // Build mcpServers as an object (required by claude-agent-sdk)
  const mcpServers: Record<string, any> = {
    "custom-tools": customMcpServer,
  };
  if (sandboxMcpServer && edgeoneMcp) {
    mcpServers[edgeoneMcp.name] = sandboxMcpServer;
  }

  // ─── Abort controller ────────────────────────────────────────────────────────
  const abortController = new AbortController();
  if (signal) {
    signal.addEventListener("abort", () => abortController.abort(), {
      once: true,
    });
  }

  // ─── Session store ────────────────────────────────────────────────────────────
  // The Claude Agent SDK exposes `claudeSessionStore()` (camelCase, no arguments).
  const claudeSessionStore = store?.claudeSessionStore?.() ?? null;

  // ─── Session binding (resume vs new) ─────────────────────────────────────────
  const sessionBinding = await resolveClaudeSessionBinding(
    claudeSessionStore,
    conversationId,
    cwd
  );

  // ─── System prompt (skills-based, dynamic) ───────────────────────────────────
  const systemPrompt = buildSystemPrompt(uploadedFiles, sandboxWorking, locale);

  // ─── Build query options ──────────────────────────────────────────────────────
  // The Claude Agent SDK spawns the `claude` CLI subprocess with stderr set to
  // "ignore" by default — that is why a non-zero exit surfaces as a bare
  // "Claude Code process exited with code 1" with no detail. Wire `stderr` to
  // our logger so the real CLI diagnostics (MCP handshake errors, unknown
  // flags, auth issues, etc.) actually reach the dev terminal.
  const queryOptions: Record<string, any> = {
    model: resolveModelName(ctxEnv),
    systemPrompt,
    cwd,
    tools: [],
    // Custom tools must be listed by their full `mcp__<server>__<tool>` name so
    // the model sees them under the correct `custom-tools` prefix. Without this
    // the model only sees the edgeone sandbox tools and guesses a wrong prefix
    // (e.g. mcp__edgeone__suggest_actions → "No such tool available").
    allowedTools: [
      "mcp__custom-tools__suggest_actions",
      "mcp__custom-tools__deliver_file",
      ...(edgeoneMcp?.allowedTools ?? []),
    ],
    // Do not set settingSources / skills options — this project has no .claude/
    // directory; all skills are baked into the system prompt via _skills.ts.
    permissionMode: "bypassPermissions",
    maxTurns: 10,
    // Emit partial (incremental) assistant messages so we can stream text token
    // by token. Without this the SDK only emits the FULL assistant message once
    // per turn → the frontend receives a single text_delta with the entire reply
    // (no real-time typing effect). With it enabled the SDK also yields
    // `stream_event` messages carrying Anthropic raw content_block_delta events.
    includePartialMessages: true,
    env: {
      ...ctxEnv,
      ...collectGatewayEnv(ctxEnv),
      // The claude CLI needs a writable config directory. In the serverless
      // runtime there is no writable HOME, so point it at /tmp explicitly.
      CLAUDE_CONFIG_DIR: "/tmp/claude-agent-sdk",
      CLAUDE_CODE_TMPDIR: "/tmp",
    },
    mcpServers,
    abortController,
    // Capture CLI stderr — required to debug CLI exit-1 errors. See SDK
    // `spawnLocalProcess`: stderr is "ignore" unless either DEBUG_CLAUDE_AGENT_SDK
    // is set in env OR this callback is provided.
    stderr: (data: any) => {
      const text = typeof data === "string" ? data : data?.toString?.() || "";
      if (text.trim()) logger.error("[claude-stderr]", text.trimEnd());
    },
    ...sessionBinding,
  };

  logger.log(
    `[query] model=${queryOptions.model}, session=${JSON.stringify(
      sessionBinding
    )}`
  );

  // ─── Helper: extract stdout + stderr from code_interpreter result ───────────
  function extractCodeInterpreterOutput(raw: string): {
    stdout: string;
    stderr: string;
  } {
    if (!raw) return { stdout: "", stderr: "" };
    try {
      const parsed = JSON.parse(raw);
      const stdout =
        (Array.isArray(parsed.logs?.stdout)
          ? parsed.logs.stdout.join("")
          : "") ||
        parsed.stdout ||
        "";
      const errorStr =
        typeof parsed.error === "string"
          ? parsed.error
          : parsed.error?.message
          ? `${parsed.error.name ?? "Error"}: ${parsed.error.message}`
          : "";
      const stderr =
        (Array.isArray(parsed.logs?.stderr)
          ? parsed.logs.stderr.join("")
          : "") ||
        parsed.stderr ||
        errorStr ||
        "";
      return { stdout, stderr };
    } catch {
      // Not JSON — return as-is if it looks like plain text output
      const text = raw.trim().startsWith("{") ? "" : raw;
      return { stdout: text, stderr: "" };
    }
  }

  // ─── SSE generator ────────────────────────────────────────────────────────────
  async function* generate(): AsyncGenerator<string> {
    let fullAssistantText = "";
    let lastMsgType = "";

    // Per-block sent-length tracker to deduplicate incremental text updates
    const sentTextLenByBlock = new Map<number, number>();

    // Map tool_use block ids → short tool names (for code_output parsing)
    const toolUseIdToName = new Map<string, string>();

    const q = query({
      prompt: message,
      options: queryOptions,
    });

    try {
      for await (const msg of q as any) {
        if (signal?.aborted) break;

        // Drain SSE queue after every message
        for (const evt of sseQueue.splice(0)) yield evt;

        // Reset text deduplication when a new assistant turn begins
        if (msg.type === "assistant" && lastMsgType === "user") {
          sentTextLenByBlock.clear();
        }
        lastMsgType = msg.type;

        // ── Partial streaming events: emit incremental text deltas in real time ─
        // With includePartialMessages enabled, the SDK yields `stream_event`
        // messages wrapping Anthropic raw streaming events. We forward text
        // deltas immediately so the UI types out the reply token by token.
        // sentTextLenByBlock tracks how much of each content block we've already
        // sent, so the final full `assistant` message (below) won't re-emit it.
        if (msg.type === "stream_event") {
          const ev = (msg as any).event;
          if (ev?.type === "content_block_start") {
            // New content block begins — reset this block's sent-length counter
            sentTextLenByBlock.set(ev.index ?? 0, 0);
          } else if (
            ev?.type === "content_block_delta" &&
            ev.delta?.type === "text_delta"
          ) {
            const idx = ev.index ?? 0;
            const text: string = ev.delta.text || "";
            if (text) {
              sentTextLenByBlock.set(
                idx,
                (sentTextLenByBlock.get(idx) ?? 0) + text.length
              );
              fullAssistantText += text;
              yield sseEvent({ type: "text_delta", delta: text });
            }
          }
          continue;
        }

        // ── Assistant message: emit text deltas + tool_called events ──────────
        if (msg.type === "assistant") {
          const blocks: any[] = msg.message?.content ?? [];
          for (let idx = 0; idx < blocks.length; idx++) {
            const block = blocks[idx];

            if (block.type === "text") {
              const fullText: string = block.text || "";
              const alreadySent = sentTextLenByBlock.get(idx) ?? 0;
              if (fullText.length > alreadySent) {
                const delta = fullText.slice(alreadySent);
                sentTextLenByBlock.set(idx, fullText.length);
                fullAssistantText += delta;
                yield sseEvent({ type: "text_delta", delta });
              }
            } else if (block.type === "tool_use") {
              const rawName: string = block.name || "";
              // MCP tool names are "mcp__servername__toolname" — extract short name
              const toolName = rawName.split("__").pop() || rawName;
              toolUseIdToName.set(block.id, toolName);
              const inputSummary = JSON.stringify(block.input ?? {}).slice(
                0,
                200
              );
              logger.log(`[tool_use] ${toolName} | input=${inputSummary}`);
              yield sseEvent({
                type: "tool_called",
                tool: toolName,
                input: block.input ?? {},
              });
            }
          }

          // ── User message (tool results): emit code_output for code_interpreter ─
        } else if (msg.type === "user") {
          // Drain queue again — custom MCP handlers may have just pushed events
          for (const evt of sseQueue.splice(0)) yield evt;

          const blocks: any[] = msg.message?.content ?? [];
          for (const block of blocks) {
            if (block.type === "tool_result") {
              const toolName = toolUseIdToName.get(block.tool_use_id) || "";

              // Normalize content: may be string, array of blocks, or undefined
              const contentBlocks: Array<{ type: string; text?: string }> =
                Array.isArray(block.content)
                  ? block.content
                  : typeof block.content === "string"
                  ? [{ type: "text", text: block.content }]
                  : [];

              const fullText = contentBlocks
                .filter((b) => b.type === "text")
                .map((b) => b.text || "")
                .join("");

              // Log all tool results for diagnostics (truncated)
              logger.log(
                `[tool_result] ${toolName} | len=${
                  fullText.length
                } | preview=${fullText.slice(0, 300).replace(/\n/g, "↵")}`
              );

              if (toolName === "code_interpreter") {
                const { stdout, stderr } =
                  extractCodeInterpreterOutput(fullText);
                if (stdout.trim()) {
                  yield sseEvent({ type: "code_output", stdout });
                }
                if (stderr.trim()) {
                  yield sseEvent({ type: "code_error", stderr });
                }
              }
            }
          }

          // ── Result: query complete ─────────────────────────────────────────────
        } else if (msg.type === "result") {
          // Drain final queue
          for (const evt of sseQueue.splice(0)) yield evt;
          // The Claude Agent SDK does NOT throw on model/CLI failures — instead it
          // emits a result message with `is_error: true` or `subtype !== 'success'`
          // (e.g. error_max_turns, error_during_execution). Surface it instead of
          // silently completing with no output.
          const subtype: string = msg.subtype || "";
          const isError: boolean =
            msg.is_error === true || (subtype !== "" && subtype !== "success");
          if (isError) {
            logger.error(
              "[result] error result:",
              JSON.stringify(msg).slice(0, 800)
            );
            if (!fullAssistantText && !signal?.aborted) {
              const detail =
                (typeof msg.result === "string" && msg.result) ||
                subtype ||
                "unknown error";
              yield sseEvent({
                type: "error_message",
                content: `Model run failed (${subtype || "error"}): ${detail}`,
              });
            }
          }
          break;
        }
      }
    } catch (err: any) {
      logger.error("[query] error:", err?.stack || err);
      if (!signal?.aborted) {
        // The SDK's bare "Claude Code process exited with code N" message is
        // intentionally unhelpful because stderr was suppressed inside the SDK.
        // Our `stderr` option (see queryOptions) forwards the real CLI output
        // to logger.error — so check the dev terminal / function logs for the
        // underlying cause (MCP handshake failure, unknown CLI flag, auth
        // issue, etc.). Surface a short pointer to the user too.
        const msg: string = err?.message || String(err);
        const isSubprocessExit = /Claude Code process exited/i.test(msg);
        const tail = isSubprocessExit
          ? "\n\nThe full CLI error has been written to the server logs (look for `[claude-stderr]`). On EdgeOne Makers the gateway credentials are injected automatically; for local dev, make sure your .env values are real (not the .env.example placeholders)."
          : "";
        yield sseEvent({ type: "error_message", content: msg + tail });
      }
    }

    // ─── Fallback suggest_actions ────────────────────────────────────────────
    // If AI never called suggest_actions and never delivered a file, auto-emit defaults
    if (
      !suggestActionsCalled &&
      !deliverFileCalled &&
      uploadedFiles.length > 0
    ) {
      logger.log(
        "[fallback] AI did not call suggest_actions, generating defaults"
      );
      yield sseEvent({
        type: "suggest_actions",
        actions: buildDefaultActions(uploadedFiles),
      });
    }

    yield "data: [DONE]\n\n";
  }

  return createSSEResponse(generate, signal);
}
