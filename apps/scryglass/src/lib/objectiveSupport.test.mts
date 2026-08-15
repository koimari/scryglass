import assert from "node:assert/strict";
import test from "node:test";
import { objectiveFieldsForPatch, supportsAtakhans } from "./objectiveSupport";

test("hides Atakhan after its 26.01 removal", () => {
  assert.equal(supportsAtakhans("26.16"), false);
  assert.equal(supportsAtakhans("16.16"), false);
  assert.equal(objectiveFieldsForPatch("26.16").some((field) => field.key === "atakhans"), false);
});

test("keeps Atakhan for supported 25.x patches", () => {
  assert.equal(supportsAtakhans("25.24"), true);
  assert.equal(supportsAtakhans("15.24"), true);
  assert.equal(objectiveFieldsForPatch("25.24").some((field) => field.key === "atakhans"), true);
});

test("fails closed when patch support is unknown", () => {
  assert.equal(supportsAtakhans(null), false);
  assert.equal(objectiveFieldsForPatch(null).some((field) => field.key === "atakhans"), false);
});
