import assert from "node:assert/strict";
import test from "node:test";

import {
  CHAT_BODY_MAX_BYTES,
  readChatJson,
  readJsonBody,
  secureChatRoute,
  takeRateLimit,
} from "./chatApi";
import { getRatings } from "./publicData";
import type { PackManifest } from "./pack";

const QUERY_MANIFEST = {
  pack_id: "v2026.08.13.183000",
  schema_version: "2.0.0",
  created_utc: "2026-08-13T18:30:00Z",
  filters: { years: [2026], leagues: "all" },
  attribution: "Scryglass",
  excluded: [],
  base_url: null,
  data_backend: "supabase",
  query_api: { schema_version: "scryglass:query-api:v1", status: "available" },
  total_bytes: 0,
  total_files: 0,
  files: [],
} satisfies PackManifest;

test("token bucket enforces a burst and refills over the policy window", () => {
  const buckets = new Map();
  const policy = { requestsPerWindow: 2, windowMs: 1_000, burst: 2 };
  assert.equal(takeRateLimit("client", policy, 0, buckets).allowed, true);
  assert.equal(takeRateLimit("client", policy, 0, buckets).allowed, true);
  const denied = takeRateLimit("client", policy, 0, buckets);
  assert.equal(denied.allowed, false);
  assert.equal(denied.retryAfterSeconds, 1);
  assert.equal(takeRateLimit("client", policy, 500, buckets).allowed, true);
});

test("JSON body reader requires JSON and rejects declared or streamed excess", async () => {
  const wrongType = await readJsonBody(new Request("https://scryglass.xyz/api", {
    method: "POST",
    headers: { "content-type": "text/plain" },
    body: "{}",
  }));
  assert.equal(wrongType.ok, false);
  if (!wrongType.ok) assert.equal(wrongType.response.status, 415);

  const declared = await readJsonBody(new Request("https://scryglass.xyz/api", {
    method: "POST",
    headers: {
      "content-length": String(CHAT_BODY_MAX_BYTES + 1),
      "content-type": "application/json",
    },
    body: "{}",
  }));
  assert.equal(declared.ok, false);
  if (!declared.ok) assert.equal(declared.response.status, 413);

  const streamed = await readJsonBody(new Request("https://scryglass.xyz/api", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text: "x".repeat(CHAT_BODY_MAX_BYTES) }),
  }));
  assert.equal(streamed.ok, false);
  if (!streamed.ok) assert.equal(streamed.response.status, 413);
});

test("JSON body reader accepts structured JSON vendor types", async () => {
  const result = await readJsonBody(new Request("https://scryglass.xyz/api", {
    method: "POST",
    headers: { "content-type": "application/vnd.scryglass+json; charset=utf-8" },
    body: JSON.stringify({ plan: { entity: "player" } }),
  }));
  assert.deepEqual(result, { ok: true, value: { plan: { entity: "player" } } });
});

test("secure chat routes reject long parameters without reflecting them", async () => {
  let called = false;
  const route = secureChatRoute(() => {
    called = true;
    return Response.json({ ok: true });
  });
  const response = await route(new Request(
    `https://scryglass.xyz/api/chat/player?name=${encodeURIComponent("x".repeat(101))}`,
  ));
  assert.equal(response.status, 422);
  assert.equal(called, false);
  assert.doesNotMatch(await response.text(), /x{20}/);
});

test("secure chat timeout aborts its bounded RPC fetch", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  let fetchSawAbort = false;
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => (
    new Promise<Response>((_resolve, reject) => {
      const signal = init?.signal;
      assert.ok(signal);
      const onAbort = () => {
        fetchSawAbort = true;
        reject(signal.reason);
      };
      if (signal.aborted) onAbort();
      else signal.addEventListener("abort", onAbort, { once: true });
    })
  )) as typeof fetch;
  const route = secureChatRoute(async (_request, signal) => {
    await getRatings(QUERY_MANIFEST, { kind: "players", limit: 1 }, signal);
    return Response.json({ ok: true });
  }, 10);
  try {
    const response = await route(new Request("https://scryglass.xyz/api/chat/leaderboards", {
      headers: { "x-vercel-forwarded-for": "192.0.2.12" },
    }));
    assert.equal(response.status, 504);
    assert.equal(fetchSawAbort, true);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});

test("secure chat timeout aborts its active-release fetch", async () => {
  const previousFetch = globalThis.fetch;
  const previousUrl = process.env.SCRYGLASS_SUPABASE_URL;
  const previousKey = process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
  process.env.SCRYGLASS_SUPABASE_URL = "https://abcdef.supabase.co";
  process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = "sb_publishable_test";
  let fetchSawAbort = false;
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => (
    new Promise<Response>((_resolve, reject) => {
      const signal = init?.signal;
      assert.ok(signal);
      const onAbort = () => {
        fetchSawAbort = true;
        reject(signal.reason);
      };
      if (signal.aborted) onAbort();
      else signal.addEventListener("abort", onAbort, { once: true });
    })
  )) as typeof fetch;
  const route = secureChatRoute(async (_request, signal) => {
    await readChatJson<Record<string, unknown>>("features/team_records.json", signal);
    return Response.json({ ok: true });
  }, 10);
  try {
    const response = await route(new Request("https://scryglass.xyz/api/chat/team?name=T1", {
      headers: { "x-vercel-forwarded-for": "192.0.2.13" },
    }));
    assert.equal(response.status, 504);
    assert.equal(fetchSawAbort, true);
  } finally {
    globalThis.fetch = previousFetch;
    if (previousUrl === undefined) delete process.env.SCRYGLASS_SUPABASE_URL;
    else process.env.SCRYGLASS_SUPABASE_URL = previousUrl;
    if (previousKey === undefined) delete process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY;
    else process.env.SCRYGLASS_SUPABASE_PUBLISHABLE_KEY = previousKey;
  }
});
