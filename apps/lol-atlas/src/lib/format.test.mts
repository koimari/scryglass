import assert from "node:assert/strict";
import test from "node:test";
import {
  champIconUrl,
  formatClock,
  normalizePatchVersion,
} from "./format.ts";

test("uses Data Dragon identifiers for apostrophized champion names", () => {
  assert.match(champIconUrl("Kai'Sa") ?? "", /\/Kaisa_0\.jpg$/);
  assert.match(champIconUrl("Cho'Gath") ?? "", /\/Chogath_0\.jpg$/);
  assert.match(champIconUrl("Kha'Zix") ?? "", /\/Khazix_0\.jpg$/);
  assert.match(champIconUrl("Vel'Koz") ?? "", /\/Velkoz_0\.jpg$/);
  assert.doesNotMatch(champIconUrl("Kai'Sa") ?? "", /\/cdn\/\d+\.\d+\.\d+\//);
});

test("clock rounding carries sixty seconds into the next minute", () => {
  assert.equal(formatClock(null, 29.999), "30:00");
  assert.equal(formatClock(1799.6), "30:00");
  assert.equal(formatClock(1799), "29:59");
});

test("patch versions normalize full builds without prefix ambiguity", () => {
  assert.equal(normalizePatchVersion("16.14.794.5912"), "16.14");
  assert.equal(normalizePatchVersion("Patch 16.01"), "16.01");
  assert.equal(normalizePatchVersion("16.10"), "16.10");
  assert.equal(normalizePatchVersion("16.1"), null);
  assert.equal(normalizePatchVersion("Patch 16.1"), null);
  assert.equal(normalizePatchVersion("16"), null);
  assert.equal(normalizePatchVersion("16.1 beta"), null);
});
