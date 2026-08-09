import assert from "node:assert/strict";
import test from "node:test";

import {
  compactPlayerRatings,
  recentProfileGames,
  scopedTeamWr,
  type PlayerRating,
  type ProfileGame,
  type ProfileRecords,
  type TeamRecord,
} from "./pack";

function profileGame(game_id: string, date: string): ProfileGame {
  return {
    game_id,
    date,
    league: "LCS",
    blue_team: "Blue",
    red_team: "Red",
    blue_win: 1,
    players: [],
  };
}

test("recent match index orders accepted profile games newest first", () => {
  const older = profileGame("older", "2026-07-18T12:00:00Z");
  const newest = profileGame("newest", "2026-08-08T21:50:46Z");
  const records: ProfileRecords = {
    schema_version: "scryglass:profile-records:v2",
    window_days: 120,
    champion_images: {},
    games: { older, newest },
    players: {},
    teams: {},
  };

  assert.deepEqual(recentProfileGames(records, 1), [newest]);
});

test("Tier 1 scope reads the tier1 team record", () => {
  const record: TeamRecord = {
    leagues: ["LCK"],
    primary: "LCK",
    current_league: "LCK",
    current_tier: "tier1",
    intl: true,
    wins: 12,
    games: 20,
    wr: 0.6,
    by_tier: {
      tier1: { wins: 12, games: 20, wr: 0.6 },
    },
  };

  assert.equal(scopedTeamWr(record, ["TIER1"]), 0.6);
});

test("compact player ratings preserve the public evidence contract", () => {
  const row: PlayerRating = {
    player: "Chovy",
    mu_total: 1706.6,
    mu_regional: 1696.8,
    mu_meta: 9.8,
    sigma: 28,
    n_maps: 263,
    last_team: "Gen.G",
    evidence_interval_width: 109.8,
    evidence_precision_ratio: 1,
    evidence_stability: 2.6,
    evidence_freshness_days: 1.5,
    evidence_support_coverage: 1,
    evidence_fallback: 0,
    evidence_active: 1,
    evidence_disconnected: 0,
    evidence_ood: 0,
    evidence_state: "settled",
  };

  assert.deepEqual(compactPlayerRatings([row]), [row]);
});

test("compact player ratings exclude disconnected players", () => {
  const disconnected: PlayerRating = {
    player: "Baus",
    mu_total: 1672.2,
    mu_regional: 1672.2,
    mu_meta: 0,
    sigma: 28,
    n_maps: 120,
    last_team: null,
    evidence_disconnected: 1,
    evidence_state: "disconnected",
  };

  assert.deepEqual(compactPlayerRatings([disconnected]), []);
});
