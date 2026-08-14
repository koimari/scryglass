import assert from "node:assert/strict";
import test from "node:test";
import { publicPatchLabel, samePublicPatch } from "./patchIdentity";

test("public patch labels use Riot's 26.x namespace", () => {
  assert.equal(publicPatchLabel("16.15"), "26.15");
  assert.equal(publicPatchLabel("16.15.1"), "26.15");
  assert.equal(publicPatchLabel("16.16"), "26.16");
  assert.equal(publicPatchLabel("26.16"), "26.16");
});

test("source and public patch labels compare by their canonical public value", () => {
  assert.equal(samePublicPatch("16.15", "26.15"), true);
  assert.equal(samePublicPatch("16.16", "26.15"), false);
});

