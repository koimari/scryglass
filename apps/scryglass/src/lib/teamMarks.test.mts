import assert from "node:assert/strict";
import test from "node:test";

import { teamInitials, teamMarkUrl } from "./teamMarks.ts";

test("returns original transparent PNG marks for published teams", () => {
  assert.match(teamMarkUrl("Gen.G") ?? "", /Gen\.Glogo_square\.png\/revision\/latest\?format=original$/);
  assert.match(teamMarkUrl(" LØS ") ?? "", /L%C3%98Slogo_square\.png\/revision\/latest\?format=original$/);
  assert.match(teamMarkUrl("Ground Zero Gaming") ?? "", /\.png\/revision\/latest\?format=original$/);
});

test("uses a lettermark when no reviewed team image exists", () => {
  assert.equal(teamMarkUrl("Unknown Team"), null);
  assert.equal(teamInitials("Unknown Team"), "UT");
  assert.equal(teamInitials("T1"), "T1");
});
