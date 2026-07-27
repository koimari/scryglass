import assert from "node:assert/strict";
import test from "node:test";
import {
  exposePatchContracts,
  normalizeExactPatch,
  normalizePatchOrBuild,
  patchContractFromPublic,
  patchContractFromSource,
  patchContractsFromSource,
} from "./patch.ts";

test("maps public 25.x and 26.x patches to their source keys", () => {
  assert.deepEqual(patchContractFromPublic("25.14"), {
    public_patch: "25.14",
    source_patch_key: "15.14",
  });
  assert.deepEqual(patchContractFromPublic("26.14"), {
    public_patch: "26.14",
    source_patch_key: "16.14",
  });
  assert.deepEqual(patchContractFromSource("15.01"), {
    public_patch: "25.01",
    source_patch_key: "15.01",
  });
  assert.deepEqual(patchContractFromSource("16.14"), {
    public_patch: "26.14",
    source_patch_key: "16.14",
  });
});

test("rejects ambiguous one-digit minors and unsupported major families", () => {
  for (const value of ["16.1", "26.1", "Patch 26.1", "26.1.7"]) {
    assert.equal(normalizeExactPatch(value), null);
    assert.equal(normalizePatchOrBuild(value), null);
    assert.equal(patchContractFromPublic(value), null);
    assert.equal(patchContractFromSource(value), null);
  }
  assert.equal(patchContractFromPublic("24.14"), null);
  assert.equal(patchContractFromPublic("16.14"), null);
  assert.equal(patchContractFromSource("26.14"), null);
  assert.equal(patchContractFromSource(16.14), null);
});

test("preserves exact two-digit minors including leading zeroes", () => {
  assert.equal(normalizeExactPatch("26.01"), "26.01");
  assert.equal(normalizeExactPatch("26.10"), "26.10");
  assert.equal(normalizePatchOrBuild("16.01.123.456"), "16.01");
  assert.equal(normalizePatchOrBuild("Patch 16.10"), "16.10");
  assert.equal(normalizeExactPatch("26.01.123"), null);
});

test("deduplicates source keys and exposes explicitly named metadata contracts", () => {
  assert.deepEqual(
    patchContractsFromSource(["16.14", "16.14", "15.10", "14.20"]),
    [
      { public_patch: "26.14", source_patch_key: "16.14" },
      { public_patch: "25.10", source_patch_key: "15.10" },
    ],
  );

  const exposed = exposePatchContracts({
    latest_observed_patch: "16.14",
    observed_holdout_patches: ["16.13", "16.14"],
    analysis_patches: ["15.14", "16.14"],
    supported_patches: ["15.14"],
    runtime_status: "available",
  });
  assert.equal(exposed.runtime_status, "available");
  assert.deepEqual(exposed.latest_observed_patch_contract, {
    public_patch: "26.14",
    source_patch_key: "16.14",
  });
  assert.ok(!("latest_observed_patch" in exposed));
  assert.ok(!("analysis_patches" in exposed));
});
