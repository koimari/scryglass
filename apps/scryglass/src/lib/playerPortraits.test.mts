import assert from "node:assert/strict";
import test from "node:test";

import { playerPortraitSource, playerPortraitUrl } from "./playerPortraits.ts";

test("returns only reviewed player portraits", () => {
  assert.match(playerPortraitUrl("Inspired") ?? "", /^https:\/\/static\.wikia\.nocookie\.net\//);
  assert.equal(playerPortraitSource("Viper"), "https://lol.fandom.com/wiki/Viper_(Park_Do-hyeon)");
  assert.equal(playerPortraitUrl("random"), null);
});

test("portrait lookup is case-insensitive and trims the handle", () => {
  assert.equal(playerPortraitUrl(" CHOVY "), playerPortraitUrl("chovy"));
});
