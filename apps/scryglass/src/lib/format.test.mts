import assert from "node:assert/strict";
import test from "node:test";
import { champIconUrl } from "./format.ts";

test("uses Data Dragon identifiers for apostrophized champion names", () => {
  assert.match(champIconUrl("Kai'Sa") ?? "", /\/Kaisa\.png$/);
  assert.match(champIconUrl("Cho'Gath") ?? "", /\/Chogath\.png$/);
  assert.match(champIconUrl("Kha'Zix") ?? "", /\/Khazix\.png$/);
  assert.match(champIconUrl("Vel'Koz") ?? "", /\/Velkoz\.png$/);
});
