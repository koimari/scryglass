import assert from "node:assert/strict";
import test from "node:test";

import {
  bestChampionRecords,
  compactPlayerRatings,
  findRecordByName,
  findPlayerByRouteName,
  hasPromotedDraftAuthority,
  isActiveRating,
  packUrl,
  recentProfileGames,
  scopedTeamWr,
  type PlayerRating,
  type PackManifest,
  type ProfileGame,
  type ProfileRecords,
  type TeamRecord,
} from "./pack";

test("player route lookup preserves exact source casing for duplicate handles", () => {
  const players: PlayerRating[] = [
    { player: "Random", mu_total: 1600, mu_regional: 1600, mu_meta: 0, sigma: 40, n_maps: 20, last_team: "Disruptors" },
    { player: "random", mu_total: 1500, mu_regional: 1500, mu_meta: 0, sigma: 40, n_maps: 20, last_team: "Team Solid" },
  ];
  assert.equal(findPlayerByRouteName(players, "random")?.last_team, "Team Solid");
  assert.equal(findPlayerByRouteName(players, "Random")?.last_team, "Disruptors");
});

test("record joins ignore source casing and harmless spacing differences", () => {
  const records = { "Gen.G": { current_tier: "tier1" } };
  assert.deepEqual(findRecordByName(records, "  GEN.G  "), { current_tier: "tier1" });
});

test("draft authority stays closed until an independent receipt verifier exists", () => {
  const manifest = {
    pack_id: "v2026.08.13.1830",
    schema_version: "test",
    created_utc: "2026-08-13T18:31:17Z",
    filters: { years: [2026], leagues: "all" },
    attribution: "test",
    excluded: [],
    base_url: null,
    total_bytes: 0,
    total_files: 0,
    files: [],
  } satisfies PackManifest;

  assert.equal(hasPromotedDraftAuthority(manifest), false);
  assert.equal(hasPromotedDraftAuthority({
    ...manifest,
    draft_authority: {
      schema_version: "scryglass:draft-authority:v1",
      status: "promoted",
      release_id: manifest.pack_id,
      model_version: "draft-v1",
      receipt_sha256: "a".repeat(64),
    },
  }), false);
  assert.equal(hasPromotedDraftAuthority({
    ...manifest,
    draft_authority: {
      schema_version: "scryglass:draft-authority:v1",
      status: "promoted",
      release_id: "different-release",
      model_version: "draft-v1",
      receipt_sha256: "a".repeat(64),
    },
  }), false);
});

test("Supabase pack URLs stay behind the active-release proxy", () => {
  const manifest = {
    pack_id: "v2026.08.13.183000",
    schema_version: "test",
    created_utc: "2026-08-13T18:31:17Z",
    filters: { years: [2026], leagues: "all" },
    attribution: "test",
    excluded: [],
    base_url: "https://legacy.example/packs/old",
    data_backend: "supabase" as const,
    total_bytes: 0,
    total_files: 0,
    files: [],
  } satisfies PackManifest;

  assert.equal(
    packUrl(manifest, "features/team_records.json"),
    "/api/assets/v2026.08.13.183000/features%2Fteam_records.json",
  );
  assert.throws(() => packUrl(manifest, "../private.json"), /invalid/);
});

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

test("public rankings include only rows confirmed as active", () => {
  assert.equal(isActiveRating({ evidence_active: 1 }), true);
  assert.equal(isActiveRating({ evidence_active: 0 }), false);
  assert.equal(isActiveRating({ evidence_active: null }), false);
  assert.equal(isActiveRating({}), false);
});

test("best champion records reward results with enough evidence", () => {
  const ranked = bestChampionRecords([
    { champion: "One map", games: 1, wins: 1, losses: 0, wr: 1, kills: null, deaths: null, assists: null },
    { champion: "Proven", games: 12, wins: 9, losses: 3, wr: 0.75, kills: null, deaths: null, assists: null },
    { champion: "Losing", games: 20, wins: 8, losses: 12, wr: 0.4, kills: null, deaths: null, assists: null },
  ]);

  assert.deepEqual(ranked.map((record) => record.champion), ["Proven", "One map", "Losing"]);
});
