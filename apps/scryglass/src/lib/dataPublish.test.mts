import assert from "node:assert/strict";
import test from "node:test";
import { uploadPolicy, validPublishSecret } from "./dataPublish";

test("immutable pack files receive long-lived cache policy", () => {
  const policy = uploadPolicy("packs/v2026.08.09.234328/features/profile_records.json");
  assert.equal(policy?.allowOverwrite, false);
  assert.equal(policy?.cacheControlMaxAge, 31_536_000);
});

test("stable pointers and tier display can be replaced", () => {
  assert.equal(uploadPolicy("packs/manifest.json")?.allowOverwrite, true);
  assert.equal(uploadPolicy("packs/latest.json")?.allowOverwrite, true);
  assert.equal(uploadPolicy("rankings/tierlists.json")?.allowOverwrite, true);
  assert.equal(uploadPolicy("rankings/tierlists-latest.json")?.allowOverwrite, true);
});

test("upload policy rejects paths outside the public data contract", () => {
  assert.equal(uploadPolicy("packs/v2026.08.09.234328/../../secret"), null);
  assert.equal(uploadPolicy("packs/bad/features/profile_records.json"), null);
  assert.equal(uploadPolicy("research/private.json"), null);
});

test("publish secret requires an exact bearer token", () => {
  assert.equal(validPublishSecret("Bearer known", "known"), true);
  assert.equal(validPublishSecret("Bearer wrong", "known"), false);
  assert.equal(validPublishSecret(null, "known"), false);
  assert.equal(validPublishSecret("Bearer known", undefined), false);
});
