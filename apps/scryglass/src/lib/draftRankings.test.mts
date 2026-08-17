import assert from "node:assert/strict";
import test from "node:test";
import { draftRankingsFromProfile, filterDraftRankings, hasCompleteCompositionEvidence, hasCompleteDraftEvidence } from "./draftRankings";
import type { DraftPool, ProfileGame, ProfileRecords } from "./pack";

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
    draft_pool: {
      schema_version: "scryglass:draft-pool:v1" as const,
      status: "complete" as const,
      source: "published-tier-list",
      patch: "26.16",
      bans: {
        Blue: ["BanA", "BanB", "BanC", "BanD", "BanE"],
        Red: ["BanF", "BanG", "BanH", "BanI", "BanJ"],
      },
      picked: [
        { side: "Blue" as const, role: "mid", champion: "Ahri", order: 1 },
        { side: "Red" as const, role: "mid", champion: "Azir", order: 2 },
        { side: "Blue" as const, role: "top", champion: "Gnar", order: 3 },
        { side: "Red" as const, role: "top", champion: "Ornn", order: 4 },
        { side: "Blue" as const, role: "jungle", champion: "Vi", order: 5 },
        { side: "Red" as const, role: "jungle", champion: "Sejuani", order: 6 },
        { side: "Blue" as const, role: "bot", champion: "Jinx", order: 7 },
        { side: "Red" as const, role: "bot", champion: "Varus", order: 8 },
        { side: "Blue" as const, role: "support", champion: "Nautilus", order: 9 },
        { side: "Red" as const, role: "support", champion: "Rakan", order: 10 },
      ],
      unpicked: [],
    },
    draft_contribution: {
      schema_version: "scryglass:draft-descriptive-signal:v1" as const,
      status,
      model_version: "test",
      fit_through: null,
      blue: { signal: blueSignal, prior_role_games: 10 },
      red: { signal: redSignal, prior_role_games: 10 },
      picks: [
        { side: "Blue" as const, role: "mid", champion: "Ahri", contribution: 0.2, prior_role_games: 10, evidence_status: "available" as const, best_available: true },
        { side: "Red" as const, role: "mid", champion: "Azir", contribution: -0.1, prior_role_games: 10, evidence_status: "available" as const, best_available: true },
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
    { team: "Team A", games: 5, draft_edge: 0.4, positive_edge_rate: 1, league: "LCK", tier: "tier1" },
    { team: "Team B", games: 5, draft_edge: -0.4, positive_edge_rate: 0, league: "LCK", tier: "tier1" },
  ]);
  assert.deepEqual(result.players, [
    { player: "BlueMid", games: 5, pick_contribution: 0.2, best_available_rate: 1, role: "mid", team: "Team A", league: "LCK", tier: "tier1" },
    { player: "RedMid", games: 5, pick_contribution: -0.1, best_available_rate: 1, role: "mid", team: "Team B", league: "LCK", tier: "tier1" },
  ]);
});

test("composition evidence does not require a complete ban and pick-order pool", () => {
  const value = game("composition-only", "Team A", "Team B", 0.5, 0.1) as ProfileGame;
  delete value.draft_pool;

  assert.equal(hasCompleteCompositionEvidence(value), true);
  assert.equal(hasCompleteDraftEvidence(value), false);

  const games = Object.fromEntries(Array.from({ length: 5 }, (_, index) => {
    const copy = structuredClone(value);
    copy.game_id = `composition-only-${index}`;
    return [copy.game_id, copy];
  }));
  const records = {
    schema_version: "scryglass:profile-records:v2",
    window_days: 120,
    champion_images: {},
    players: {},
    teams: {},
    games,
  } satisfies ProfileRecords;
  const result = draftRankingsFromProfile(records);
  assert.equal(result.evidenceGames, 5);
  assert.equal(result.teams.length, 2);
  assert.equal(result.players.length, 0);
});

test("ranks players by the share of best-available picks", () => {
  const games = Object.fromEntries(
    Array.from({ length: 5 }, (_, index) => {
      const value = game(`best-${index}`, "Team A", "Team B", 0.5, 0.1);
      value.draft_contribution!.picks[0].best_available = index < 4;
      value.draft_contribution!.picks[1].best_available = true;
      value.draft_contribution!.picks[0].contribution = index < 4 ? 0.2 : -0.2;
      value.draft_contribution!.picks[1].contribution = 0.1;
      value.players.push(
        { player: "BlueTop", side: "Blue" as const, role: "top", champion: "Gnar", kills: null, deaths: null, assists: null },
        { player: "RedTop", side: "Red" as const, role: "top", champion: "Ornn", kills: null, deaths: null, assists: null },
      );
      value.draft_contribution!.picks.push(
        { side: "Blue" as const, role: "top", champion: "Gnar", contribution: 0.1, prior_role_games: 10, evidence_status: "available" as const, best_available: false },
        { side: "Red" as const, role: "top", champion: "Ornn", contribution: 0.2, prior_role_games: 10, evidence_status: "available" as const, best_available: true },
      );
      return [`best-${index}`, value];
    }),
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
  assert.equal(result.players[0]?.player, "RedTop");
  assert.equal(result.players.find((row) => row.player === "BlueMid")?.best_available_rate, 0.8);
  assert.equal(result.players.find((row) => row.player === "BlueTop")?.best_available_rate, 0);
});

test("normalizes role abbreviations in complete descriptive evidence", () => {
  const games = Object.fromEntries(
    Array.from({ length: 5 }, (_, index) => [
      `game-${index}`,
      game(`game-${index}`, "Team A", "Team B", 0.2, 0.1),
    ]),
  );
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

test("fails closed when bans, pick order, or contribution status is incomplete", () => {
  const incomplete = game("incomplete", "Team A", "Team B", 0.5, 0.1);
  (incomplete.draft_pool as DraftPool).status = "limited";
  incomplete.draft_pool!.bans.Blue = incomplete.draft_pool!.bans.Blue.slice(0, 4);
  incomplete.draft_contribution!.status = "limited";
  const records = {
    schema_version: "scryglass:profile-records:v2",
    window_days: 120,
    champion_images: {},
    players: {},
    teams: {},
    games: { incomplete },
  } satisfies ProfileRecords;
  const result = draftRankingsFromProfile(records);
  assert.equal(result.evidenceGames, 0);
  assert.deepEqual(result.teams, []);
  assert.deepEqual(result.players, []);
});

test("filters scoped rows and aggregates teams across selected leagues", () => {
  const records = {
    schema_version: "scryglass:profile-records:v2",
    window_days: 120,
    champion_images: {},
    players: {},
    teams: {},
    games: {
      ...Object.fromEntries(Array.from({ length: 5 }, (_, index) => [`lck-${index}`, game(`lck-${index}`, "Team A", "Team B", 0.5, 0.1)])),
      ...Object.fromEntries(Array.from({ length: 5 }, (_, index) => [`lec-${index}`, { ...game(`lec-${index}`, "Team A", "Team C", 0.1, 0.0), league: "LEC" }])),
    },
  } satisfies ProfileRecords;
  const result = draftRankingsFromProfile(records);
  const lck = filterDraftRankings(result, { leagues: ["LCK"], minGames: 5 });
  assert.equal(lck.teams.length, 2);
  assert.equal(lck.teams[0]?.team, "Team A");
  const all = filterDraftRankings(result, { leagues: [], minGames: 5 });
  assert.equal(all.teams[0]?.team, "Team A");
});

test("player totals stay aggregated across scoped profile evidence", () => {
  const games = Object.fromEntries(
    Array.from({ length: 10 }, (_, index) => {
      const value = game(`player-scope-${index}`, "Team A", "Team B", 0.5, 0.1);
      value.league = index < 5 ? "LCK" : "LEC";
      value.players[0].player = "ScopedPlayer";
      value.draft_contribution!.picks[0].best_available = index < 6;
      return [`player-scope-${index}`, value];
    }),
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
  const player = filterDraftRankings(result, { leagues: [], minGames: 5 }).players.find(
    (row) => row.player === "ScopedPlayer",
  );
  assert.equal(player?.games, 10);
  assert.equal(player?.best_available_rate, 0.6);
  assert.equal(filterDraftRankings(result, { leagues: ["LCK"], minGames: 5 }).players[0]?.games, 5);
});

test("marks a profile payload as whole archive when it contains older games", () => {
  const games = Object.fromEntries(Array.from({ length: 5 }, (_, index) => {
    const value = game(`archive-${index}`, "Team A", "Team B", 0.5, 0.1);
    value.date = index === 0 ? "2025-01-01" : "2026-08-01";
    return [`archive-${index}`, value];
  }));
  const records = {
    schema_version: "scryglass:profile-records:v2",
    window_days: 120,
    champion_images: {},
    players: {},
    teams: {},
    games,
  } satisfies ProfileRecords;
  assert.equal(draftRankingsFromProfile(records).scope, "whole_archive");
});
