/** Shared helpers for the /api/chat/* read-only endpoints. */

import { isIP } from "node:net";
import { NextResponse } from "next/server";
import { readPackJson, readPackManifest } from "@/lib/serverPack";

export const CHAT_QUESTION_MAX_CHARS = 500;
export const CHAT_NAME_MAX_CHARS = 100;
export const CHAT_BODY_MAX_BYTES = 8 * 1024;
export const CHAT_HANDLER_TIMEOUT_MS = 5_000;
// The planner query routes chain several sequential network legs: a manifest
// read, the support query index, a second manifest read inside
// executePublishedQueryPlan, and the bounded Supabase RPC (which itself holds
// a 5s statement_timeout). Healthy totals are 1.5-2.1s, but the sum against a
// 5s wall intermittently loses: measured failures returned at 5.27s and 5.28s,
// which is the wall budget itself, and post-activation samples put
// query_champions at 4.28s and query_drafts at 4.30s. Ten seconds gives the
// sum honest headroom while remaining far under the platform function limit.
// This is a wall-clock budget for a synthetic 504, not a slow-query allowance:
// the RPC statement timeout still caps real database work at 5s.
export const CHAT_QUERY_ROUTE_TIMEOUT_MS = 10_000;

export const CHAT_CACHE_HEADERS = {
  "Cache-Control": "public, max-age=0, must-revalidate",
} as const;

const NO_STORE_HEADERS = { "Cache-Control": "private, no-store" } as const;

export type RateLimitPolicy = {
  requestsPerWindow: number;
  windowMs: number;
  burst: number;
};

type RateBucket = {
  tokens: number;
  updatedAt: number;
};

export type RateLimitResult = {
  allowed: boolean;
  retryAfterSeconds: number;
  remaining: number;
};

export const CHAT_RATE_LIMIT: RateLimitPolicy = {
  requestsPerWindow: 60,
  windowMs: 60_000,
  burst: 20,
};

export const PUBLISH_RATE_LIMIT: RateLimitPolicy = {
  requestsPerWindow: 6,
  windowMs: 60_000,
  burst: 6,
};

const rateBuckets = new Map<string, RateBucket>();
const MAX_RATE_BUCKETS = 5_000;

function characterLength(value: string): number {
  return Array.from(value).length;
}

function clientAddress(request: Request): string {
  const forwarded = request.headers.get("x-vercel-forwarded-for")
    ?? request.headers.get("x-forwarded-for")
    ?? request.headers.get("x-real-ip")
    ?? "";
  const candidate = forwarded.split(",", 1)[0]?.trim() ?? "";
  return isIP(candidate) ? candidate : "unknown";
}

export function takeRateLimit(
  key: string,
  policy: RateLimitPolicy,
  now = Date.now(),
  buckets = rateBuckets,
): RateLimitResult {
  if (policy.requestsPerWindow < 1 || policy.windowMs < 1 || policy.burst < 1) {
    throw new Error("Rate-limit policy is invalid");
  }
  const refillPerMs = policy.requestsPerWindow / policy.windowMs;
  const previous = buckets.get(key);
  const elapsed = previous ? Math.max(0, now - previous.updatedAt) : 0;
  const tokens = previous
    ? Math.min(policy.burst, previous.tokens + elapsed * refillPerMs)
    : policy.burst;
  const allowed = tokens >= 1;
  const remainingTokens = allowed ? tokens - 1 : tokens;
  buckets.set(key, { tokens: remainingTokens, updatedAt: now });

  if (buckets === rateBuckets && buckets.size > MAX_RATE_BUCKETS) {
    const oldest = buckets.keys().next().value as string | undefined;
    if (oldest) buckets.delete(oldest);
  }

  const missing = Math.max(0, 1 - remainingTokens);
  return {
    allowed,
    retryAfterSeconds: allowed ? 0 : Math.max(1, Math.ceil(missing / refillPerMs / 1_000)),
    remaining: Math.max(0, Math.floor(remainingTokens)),
  };
}

export function rateLimitResponse(
  request: Request,
  namespace: string,
  policy: RateLimitPolicy,
): NextResponse | null {
  const result = takeRateLimit(`${namespace}:${clientAddress(request)}`, policy);
  if (result.allowed) return null;
  return NextResponse.json(
    { ok: false, reason: "Too many requests. Try again soon." },
    {
      status: 429,
      headers: {
        ...NO_STORE_HEADERS,
        "Retry-After": String(result.retryAfterSeconds),
      },
    },
  );
}

export function chatJson(data: unknown): NextResponse {
  return NextResponse.json({ ok: true, data }, { headers: CHAT_CACHE_HEADERS });
}

export function chatError(reason: string, status = 404): NextResponse {
  return NextResponse.json(
    { ok: false, reason },
    { status, headers: NO_STORE_HEADERS },
  );
}

function validateChatUrl(request: Request): NextResponse | null {
  if (request.url.length > 4_096) {
    return chatError("The request is too large.", 422);
  }
  const params = new URL(request.url).searchParams;
  for (const [name, value] of params) {
    const maximum = name === "q" ? CHAT_QUESTION_MAX_CHARS : CHAT_NAME_MAX_CHARS;
    if (characterLength(value.trim()) > maximum) {
      return chatError("A query parameter is too long.", 422);
    }
  }
  return null;
}

export function secureChatRoute(
  handler: (request: Request, signal: AbortSignal) => Response | Promise<Response>,
  timeoutMs = CHAT_HANDLER_TIMEOUT_MS,
): (request: Request) => Promise<Response> {
  return async (request: Request) => {
    const invalid = validateChatUrl(request);
    if (invalid) return invalid;
    const limited = rateLimitResponse(request, "chat", CHAT_RATE_LIMIT);
    if (limited) return limited;

    const controller = new AbortController();
    const signal = request.signal.aborted
      ? request.signal
      : AbortSignal.any([request.signal, controller.signal]);
    let didTimeout = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;
    const timeoutResponse = () => chatError("The request exceeded its time budget.", 504);
    const timedOut = new Promise<Response>((resolve) => {
      timeout = setTimeout(
        () => {
          didTimeout = true;
          controller.abort(new DOMException("Chat request time budget exceeded", "TimeoutError"));
          resolve(timeoutResponse());
        },
        timeoutMs,
      );
    });
    try {
      // Keep NextRequest intact. Wrapping it in the platform Request
      // constructor breaks its private state in Node's production runtime.
      // Handlers receive the deadline signal as a separate argument.
      const response = await Promise.race([Promise.resolve(handler(request, signal)), timedOut]);
      if (didTimeout) return response.status === 504 ? response : timeoutResponse();
      try {
        const manifest = await readPackManifest(signal);
        response.headers.set("X-Scryglass-Release", manifest.pack_id);
      } catch {
        // Preserve the route response. Readiness probes reject a missing release header.
      }
      return didTimeout ? timeoutResponse() : response;
    } catch {
      return chatError("The request could not be completed.", 503);
    } finally {
      if (timeout) clearTimeout(timeout);
    }
  };
}

export type JsonBodyResult =
  | { ok: true; value: unknown }
  | { ok: false; response: NextResponse };

export async function readJsonBody(
  request: Request,
  maximumBytes = CHAT_BODY_MAX_BYTES,
): Promise<JsonBodyResult> {
  const contentType = request.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
  if (contentType !== "application/json" && !contentType?.endsWith("+json")) {
    return { ok: false, response: chatError("Content-Type must be application/json.", 415) };
  }
  const declared = Number(request.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > maximumBytes) {
    return { ok: false, response: chatError("The request body is too large.", 413) };
  }
  const chunks: Uint8Array[] = [];
  const reader = request.body?.getReader();
  let received = 0;
  if (reader) {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      received += chunk.value.byteLength;
      if (received > maximumBytes) {
        await reader.cancel("request body is too large");
        return { ok: false, response: chatError("The request body is too large.", 413) };
      }
      chunks.push(chunk.value);
    }
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    return { ok: true, value: JSON.parse(new TextDecoder().decode(bytes)) as unknown };
  } catch {
    return { ok: false, response: chatError("The request body is invalid JSON.", 400) };
  }
}

export async function readChatJson<T>(relativePath: string, signal?: AbortSignal): Promise<T> {
  const manifest = await readPackManifest(signal);
  return readPackJson<T>(manifest, relativePath, signal);
}

/** Status for a thrown query error.
 *
 * "Public query ... has a different release" is a transient release
 * transition, not a bad request: for a few seconds after activation this
 * function instance's cached manifest can lag the release the RPC already
 * serves. Mapping it to 422 made the post-publish probe treat the window as a
 * permanent failure and roll back an otherwise good release twice. 503 tells
 * the truth (retry shortly) and the probe's propagation window already
 * retries 503.
 */
export function chatQueryFailureStatus(error: unknown): 422 | 503 {
  if (!(error instanceof Error)) return 422;
  // Timeouts are transient by definition. The bounded-query RPC carries its
  // own 5s AbortSignal.timeout, and in the seconds after a release activates
  // the new release's query rows are cold, so that inner abort fires and the
  // catch-all used to publish it as 422 - a permanent status the post-publish
  // probe refuses to retry. Measured live: {"reason":"The operation was
  // aborted due to timeout"} at exactly the activation window, rolling back
  // an otherwise good release.
  if (error.name === "TimeoutError" || error.name === "AbortError") return 503;
  if (/aborted due to.*timeout/i.test(error.message)) return 503;
  return /has a different release/.test(error.message) ? 503 : 422;
}

export function searchParams(request: Request): URLSearchParams {
  return new URL(request.url).searchParams;
}

export function clean(value: string | null): string {
  return (value ?? "").trim();
}
