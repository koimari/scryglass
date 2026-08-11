import assert from "node:assert/strict";
import test from "node:test";
import { validPublishSecret, validReleaseId } from "./dataPublish";

test("release IDs use the immutable public format", () => {
  assert.equal(validReleaseId("v2026.08.09.234328"), true);
  assert.equal(validReleaseId("packs/v2026.08.09.234328"), false);
  assert.equal(validReleaseId("v2026.8.9.1"), false);
  assert.equal(validReleaseId(null), false);
});

test("publish secret requires an exact bearer token", () => {
  assert.equal(validPublishSecret("Bearer known", "known"), true);
  assert.equal(validPublishSecret("Bearer wrong", "known"), false);
  assert.equal(validPublishSecret(null, "known"), false);
  assert.equal(validPublishSecret("Bearer known", undefined), false);
});
