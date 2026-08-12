import assert from "node:assert/strict";
import test from "node:test";

import {
  currentMatchDefaults,
  filterMatchResults,
  matchCompetitionLevel,
  matchIncludesTeam,
  matchTeamHref,
} from "./matchFilters";
import type { MatchSummary } from "./pack";

const game: MatchSummary = {
  game_id: "game-1",
  date: "2026-08-11T12:00:00Z",
  league: "LCK",
  blue_team: "Hanwha Life Esports",
  red_team: "T1",
  blue_win: 1,
  champions: [],
  grades_available: 10,
};

test("match defaults use Tier 1 and the current UTC month", () => {
  assert.deepEqual(currentMatchDefaults(new Date("2026-08-11T22:00:00Z")), {
    level: "tier1",
    year: "2026",
    month: "2026-08",
  });
});

test("legacy match summaries infer Tier 1 from the league", () => {
  assert.equal(matchCompetitionLevel(game), "tier1");
  assert.equal(matchCompetitionLevel({ ...game, league: "EWC" }), "international");
});

test("team filters match either side and preserve all other scopes", () => {
  assert.equal(matchIncludesTeam(game, "T1"), true);
  assert.equal(matchIncludesTeam(game, "Gen.G"), false);
  assert.deepEqual(filterMatchResults([game], {
    level: "tier1",
    year: "2026",
    month: "2026-08",
    team: "T1",
    league: "LCK",
  }), [game]);
});

test("team links open the Results view with the requested team", () => {
  assert.equal(matchTeamHref("Hanwha Life Esports"), "/matches?section=results&team=Hanwha+Life+Esports");
});
