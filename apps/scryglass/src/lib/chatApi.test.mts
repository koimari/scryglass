import assert from "node:assert/strict";
import test from "node:test";

import {
  CHAT_BODY_MAX_BYTES,
  readJsonBody,
  secureChatRoute,
  takeRateLimit,
} from "./chatApi";

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
