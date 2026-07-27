import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

function source(relative: string): string {
  return readFileSync(new URL(relative, import.meta.url), "utf8");
}

test("weekly player ranks are fetched only after the global ordering gate passes", () => {
  const page = source("../app/elo/page.tsx");
  const gate = page.indexOf(
    "const playerOrderingVerified = playerOutcomeOrderingVerified(",
  );
  const guardedLoad = page.indexOf("if (playerOrderingVerified)", gate);
  const weeklyArtifact = page.indexOf(
    '"features/player_weekly_ranks.json"',
    guardedLoad,
  );

  assert.ok(gate >= 0, "global ordering gate is missing");
  assert.ok(guardedLoad > gate, "weekly-rank load is not guarded");
  assert.ok(weeklyArtifact > guardedLoad, "weekly-rank artifact is loaded before the gate");
  assert.equal(
    page.indexOf('"features/player_weekly_ranks.json"', weeklyArtifact + 1),
    -1,
    "weekly-rank artifact has an additional unguarded load",
  );
});

test("player ladder delta surfaces share the server-provided ordering gate", () => {
  const ladder = source("../components/EloLadders.tsx");

  assert.match(
    ladder,
    /const showPlayerRankDeltas\s*=\s*playerOrderingVerified && playerWeeklyRanks\.as_of != null;/,
  );
  assert.match(ladder, /\{showPlayerRankDeltas \? \(\s*<th[^>]*>Δ<\/th>/);
  assert.match(ladder, /if \(!showPlayerRankDeltas\) return undefined;/);
  assert.doesNotMatch(ladder, /playerOutcomeOrderingVerified/);
});

test("team participants stay in name order and retain honestly labeled signals", () => {
  const page = source("../app/elo/team/[team]/page.tsx");
  const detail = source("../components/TeamEloDetail.tsx");

  assert.match(
    page,
    /playerOutcomeOrderingVerified\(\s*playerRatingsMeta,\s*players,\s*\)/,
  );
  assert.match(
    detail,
    /\[\.\.\.roster\]\.sort\(\(a, b\) => a\.player\.localeCompare\(b\.player\)\)/,
  );
  assert.match(detail, /Current-tournament participants · name order/);
  assert.match(detail, /Raw team-outcome signal/);
  assert.match(detail, /Uncertainty-adjusted team-outcome signal/);
  assert.doesNotMatch(
    detail,
    /topRatedPlayers|otherPlayers|by uncertainty-adjusted rating|Ordering uses the Player Dual Elo/,
  );
});

test("player profiles expose no peer median, difference, or rank comparison", () => {
  const page = source("../app/elo/player/[player]/page.tsx");
  const detail = source("../components/PlayerEloDetail.tsx");

  assert.match(
    page,
    /playerOutcomeOrderingVerified\(\s*playerRatingsMeta,\s*players,\s*\)/,
  );
  assert.match(detail, /Raw team-outcome signal/);
  assert.match(detail, /Regional outcome component/);
  assert.match(detail, /International-transfer component/);
  assert.doesNotMatch(
    detail,
    /const peerMedian|Compare vs|uncertainty-adjusted difference|intlPeers|peers:/,
  );
});
