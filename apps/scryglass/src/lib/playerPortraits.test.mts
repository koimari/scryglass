import assert from "node:assert/strict";
import test from "node:test";

import { playerPortrait } from "./playerPortraits.ts";

test("uses team context to resolve duplicate player handles", () => {
  const viper = playerPortrait("Viper", "Bilibili Gaming");
  assert.match(viper?.src ?? "", /BLG_Viper_2026_Split_1\.png/);
  assert.equal(viper?.source, "https://lol.fandom.com/wiki/Viper_(Park_Do-hyeon)");
  assert.equal(playerPortrait("Viper", "Unknown Team"), null);
});

test("portrait lookup is case-insensitive and trims identity fields", () => {
  assert.deepEqual(
    playerPortrait(" CHOVY ", " GEN.G "),
    playerPortrait("chovy", "gen.g"),
  );
});

test("unknown duplicate handles fail closed", () => {
  assert.match(playerPortrait("Random", "Disruptors")?.src ?? "", /KHK_Random_2025_Split_1\.png/);
  assert.match(playerPortrait("random", "Team Solid")?.src ?? "", /VAS_random_2026\.png/);
  assert.equal(playerPortrait("random", "Unknown Team"), null);
});
