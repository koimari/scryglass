import assert from "node:assert/strict";
import test from "node:test";

import { teamMarkUrl } from "./teamMarks.ts";

test("returns reviewed team marks for canonical names and aliases", () => {
  assert.equal(teamMarkUrl("Gen.G"), "/team-marks/gen-g.png");
  assert.equal(teamMarkUrl(" LØS "), "/team-marks/los.png");
  assert.equal(teamMarkUrl("Ground Zero"), "/team-marks/ground-zero-gaming.png");
});

test("omits a mark when the team has no reviewed asset", () => {
  assert.equal(teamMarkUrl("Unknown Team"), null);
  assert.equal(teamMarkUrl(null), null);
});
