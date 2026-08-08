import assert from "node:assert/strict";
import test from "node:test";
import { GET as sandboxGet, POST as sandboxPost } from "../app/api/draft-sandbox/route";
import { POST as draftWrPost } from "../app/api/draft-wr/route";
import { POST as draftScorePost } from "../app/api/v2/draft/score/route";
import {
  canonicalPublicPredictiveDraftsEnabled,
  DRAFT_UNAVAILABLE_RESPONSE,
  DRAFT_INTERNAL_ERROR_RESPONSE,
  publicPredictiveDraftsEnabled,
} from "./publicDraftGate";

const originalOverride = process.env.SCRYGLASS_ENABLE_UNREVIEWED_DRAFTS;

function restoreOverride() {
  if (originalOverride == null) delete process.env.SCRYGLASS_ENABLE_UNREVIEWED_DRAFTS;
  else process.env.SCRYGLASS_ENABLE_UNREVIEWED_DRAFTS = originalOverride;
}

test.afterEach(restoreOverride);

test("public predictive draft output cannot be enabled by environment override", () => {
  delete process.env.SCRYGLASS_ENABLE_UNREVIEWED_DRAFTS;
  assert.equal(publicPredictiveDraftsEnabled(), false);

  process.env.SCRYGLASS_ENABLE_UNREVIEWED_DRAFTS = "0";
  assert.equal(publicPredictiveDraftsEnabled(), false);

  process.env.SCRYGLASS_ENABLE_UNREVIEWED_DRAFTS = "1";
  assert.equal(publicPredictiveDraftsEnabled(), false);
  assert.equal(canonicalPublicPredictiveDraftsEnabled(), false);
});

test("all public draft route methods fail closed by default", async () => {
  delete process.env.SCRYGLASS_ENABLE_UNREVIEWED_DRAFTS;

  const responses = await Promise.all([
    sandboxGet(),
    sandboxPost(new Request("http://localhost/api/draft-sandbox", { method: "POST", body: "{}" })),
    draftWrPost(new Request("http://localhost/api/draft-wr", { method: "POST", body: "{}" })),
    draftScorePost(new Request("http://localhost/api/v2/draft/score", { method: "POST", body: "{}" })),
  ]);

  for (const response of responses) {
    assert.equal(response.status, 503);
    assert.deepEqual(await response.json(), DRAFT_UNAVAILABLE_RESPONSE);
  }
});

test("a complete canonical request receives the schema-shaped unavailable result", async () => {
  const response = await draftScorePost(new Request("http://localhost/api/v2/draft/score", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      side_a: { top: "Aatrox", jungle: "Nidalee", mid: "Ahri", bot: "Jinx", support: "Thresh" },
      side_b: { top: "Gnar", jungle: "Sejuani", mid: "Orianna", bot: "Aphelios", support: "Rakan" },
      event_start: "2026-07-01T12:00:00Z",
      source_available_at: "2026-07-01T11:00:00Z",
      source_record_id: "source:public-gate-fixture",
      source_payload_sha256: "a".repeat(64),
      source_rights_status: "reviewed",
    }),
  }));
  const payload = await response.json();
  assert.equal(response.status, 503);
  assert.equal(payload.status, "unavailable");
  assert.equal(payload.schema_version, "2.0.0");
  assert.equal(payload.error.code, "model_not_promoted");
  assert.equal(payload.provenance.required_input_status, "missing");
  assert.deepEqual(payload.error.missing_fields, [
    "independent_l2_authority",
    "promotion_receipt",
    "reliability_artifact",
    "replay_parity_evidence",
  ]);
});

test("the canonical route does not expose internal exception text", async () => {
  const response = await draftScorePost(new Request("http://localhost/api/v2/draft/score", {
    method: "POST",
    body: "not-json",
  }));
  assert.equal(response.status, 503);
  assert.deepEqual(await response.json(), DRAFT_INTERNAL_ERROR_RESPONSE);
});
