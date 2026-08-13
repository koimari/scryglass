import assert from "node:assert/strict";
import test from "node:test";
import { draftRankingsFromProfile } from "./draftRankings";
import type { ProfileRecords } from "./pack";

function game(
  id: string,
  blueTeam: string,
  redTeam: string,
  blueSignal: number,
  redSignal: number,
  status: "available" | "limited" = "available",
) {
  return {
    game_id: id,
    date: "2026-08-01",
    league: "LCK",
    competition_tier: "tier1",
    blue_team: blueTeam,
    red_team: redTeam,
    blue_win: 1 as const,
    players: [
      { player: "BlueMid", side: "Blue" as const, role: "mid", champion: "Ahri", kills: null, deaths: null, assists: null },
      { player: "RedMid", side: "Red" as const, role: "mid", champion: "Azir", kills: null, deaths: null, assists: null },
    ],
    draft_contribution: {
      schema_version: "scryglass:composition-signal:v1" as const,
      status,
      model_version: "test",
      fit_through: null,
      blue: { signal: blueSignal, prior_role_games: 10 },
      red: { signal: redSignal, prior_role_games: 10 },
      picks: [
        { side: "Blue" as const, role: "mid", champion: "Ahri", contribution: 0.2, prior_role_games: 10, evidence_status: "available" as const },
        { side: "Red" as const, role: "mid", champion: "Azir", contribution: -0.1, prior_role_games: 10, evidence_status: "available" as const },
      ],
      note: "test",
    },
  };
}

test("derives team and player rankings when the leaderboard asset is missing", () => {
  const games = Object.fromEntries(
    Array.from({ length: 5 }, (_, index) => [
      `game-${index}`,
      game(`game-${index}`, "Team A", "Team B", 0.5, 0.1),
    ]),
  );
  const records = {
    schema_version: "scryglass:profile-records:v2",
    window_days: 120,
    champion_images: {},
    players: {},
    teams: {},
    games,
  } satisfies ProfileRecords;

  const result = draftRankingsFromProfile(records);
  assert.equal(result.scope, "profile_window");
  assert.equal(result.evidenceGames, 5);
  assert.deepEqual(result.teams, [
    { team: "Team A", games: 5, draft_edge: 0.4 },
    { team: "Team B", games: 5, draft_edge: -0.4 },
  ]);
  assert.deepEqual(result.players, [
    { player: "BlueMid", games: 5, draft_score: 0.2, role: "mid", team: "Team A" },
    { player: "RedMid", games: 5, draft_score: -0.1, role: "mid", team: "Team B" },
  ]);
});

test("accepts the limited status and normalizes role abbreviations", () => {
  const games = Object.fromEntries(
    Array.from({ length: 5 }, (_, index) => [
      `game-${index}`,
      game(`game-${index}`, "Team A", "Team B", 0.2, 0.1),
    ]),
  );
  const record = games["game-0"];
  record.draft_contribution!.status = "limited";
  for (const value of Object.values(games)) {
    value.draft_contribution!.picks[0].role = "jng";
    value.players[0].role = "jungle";
  }
  const records = {
    schema_version: "scryglass:profile-records:v2",
    window_days: 0,
    champion_images: {},
    players: {},
    teams: {},
    games,
  } satisfies ProfileRecords;

  const result = draftRankingsFromProfile(records);
  assert.equal(result.scope, "whole_archive");
  assert.equal(result.evidenceGames, 5);
  assert.equal(result.players[0]?.role, "jungle");
});
