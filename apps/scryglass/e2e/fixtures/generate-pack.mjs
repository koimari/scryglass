import { createHash } from "node:crypto";
import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const appRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const packRoot = path.join(appRoot, "output", "playwright", "e2e-pack");
const releaseId = "v2026.08.13.000001";
const releaseRoot = path.join(packRoot, releaseId);

function ratingEvidence(active = 1) {
  return {
    evidence_interval_width: 48,
    evidence_precision_ratio: 0.9,
    evidence_stability: 0.92,
    evidence_freshness_days: 1,
    evidence_support_coverage: 1,
    evidence_fallback: 0,
    evidence_active: active,
    evidence_disconnected: 0,
    evidence_ood: 0,
    evidence_state: "connected",
  };
}

function playerRating(player, muTotal, maps, team) {
  return {
    player,
    mu_total: muTotal,
    mu_regional: muTotal - 1_500,
    mu_meta: 0,
    sigma: 28,
    n_maps: maps,
    last_team: team,
    ...ratingEvidence(),
  };
}

function teamRecord(league, wins, games) {
  return {
    leagues: [league],
    primary: league,
    intl: false,
    interregional: false,
    current_league: league,
    current_tier: "tier1",
    current_team: null,
    current_date: "2026-08-12",
    wins,
    games,
    wr: wins / games,
    by_league: { [league]: { wins, games, wr: wins / games } },
    by_tier: { tier1: { wins, games, wr: wins / games } },
  };
}

function playerRecord(team, league, role, wins, games) {
  return {
    wins,
    games,
    wr: wins / games,
    roles: [role],
    primary_role: role,
    leagues: [league],
    primary: league,
    intl: false,
    interregional: false,
    current_league: league,
    current_tier: "tier1",
    current_team: team,
    current_date: "2026-08-12",
  };
}

function championRecord(champion, games, wins) {
  return {
    champion,
    champion_image_url: null,
    games,
    wins,
    losses: games - wins,
    wr: wins / games,
    kills: null,
    deaths: null,
    assists: null,
  };
}

function tierRow(champion, championId, rank, tierBucket, playedMaps, tierValue) {
  return {
    scope_id: "patch:16.15:mid",
    role: "mid",
    patch: "16.15",
    champion,
    champion_id: `riot:champion:${championId}`,
    champion_image_url: null,
    rank,
    rank_delta: null,
    movement: "new",
    tier_bucket: tierBucket,
    played_maps: playedMaps,
    tier_value_pp: tierValue,
    counterability_status: "unavailable",
    matchup_maps: 0,
    matchup_opponents: 0,
    expected_counter_breadth: null,
  };
}

function participant(player, side, role, champion, availableGrade = false) {
  return {
    player,
    side,
    role,
    champion,
    kills: availableGrade ? 5 : 2,
    deaths: availableGrade ? 1 : 3,
    assists: availableGrade ? 8 : 5,
    team_kills: side === "Blue" ? 15 : 9,
    cs: role === "support" ? 35 : 260,
    gold: role === "support" ? 8_500 : 12_500,
    grade: availableGrade
      ? {
          status: "available",
          grade: "A",
          score: 82,
          baseline_games: 100,
          self_baseline_games: 50,
          components: { self: 0.7, team: 0.4, opponent: 0.6, league_role: 0.5 },
        }
      : { status: "unavailable", reason: "The compact E2E fixture has no grade baseline." },
  };
}

const game = {
  game_id: "e2e-game-1",
  date: "2026-08-12T18:00:00Z",
  league: "LCK",
  competition_tier: "tier1",
  patch: "16.15",
  blue_team: "T1",
  red_team: "Gen.G",
  blue_win: 1,
  duration_seconds: 1_932,
  team_stats: {
    Blue: { kills: 15, gold: 62_000, dragons: 3, heralds: 1, barons: 1, towers: 9 },
    Red: { kills: 9, gold: 56_000, dragons: 1, heralds: 0, barons: 0, towers: 4 },
  },
  players: [
    participant("Doran", "Blue", "top", "Kennen"),
    participant("Oner", "Blue", "jungle", "Xin Zhao"),
    participant("Faker", "Blue", "mid", "Annie", true),
    participant("Gumayusi", "Blue", "bot", "Jinx"),
    participant("Keria", "Blue", "support", "Nautilus"),
    participant("Kiin", "Red", "top", "Renekton"),
    participant("Canyon", "Red", "jungle", "Vi"),
    participant("Chovy", "Red", "mid", "Galio", true),
    participant("Ruler", "Red", "bot", "Kai'Sa"),
    participant("Duro", "Red", "support", "Rakan"),
  ],
};

const assets = {
  "features/ratings_snapshot.json": [
    {
      team: "T1",
      mu_total: 1_800,
      mu_regional: 300,
      mu_meta: 0,
      sigma: 25,
      rating_p10: 1_775,
      n_series: 120,
      n_maps: 314,
      home_league: "LCK",
      exact_roster: null,
      ...ratingEvidence(),
    },
    {
      team: "Gen.G",
      mu_total: 1_760,
      mu_regional: 260,
      mu_meta: 0,
      sigma: 25,
      rating_p10: 1_735,
      n_series: 110,
      n_maps: 263,
      home_league: "LCK",
      exact_roster: null,
      ...ratingEvidence(),
    },
    {
      team: "KT Rolster",
      mu_total: 1_690,
      mu_regional: 190,
      mu_meta: 0,
      sigma: 25,
      rating_p10: 1_665,
      n_series: 80,
      n_maps: 180,
      home_league: "LCK",
      exact_roster: null,
      ...ratingEvidence(),
    },
    {
      team: "LYON",
      mu_total: 1_650,
      mu_regional: 150,
      mu_meta: 0,
      sigma: 25,
      rating_p10: 1_625,
      n_series: 90,
      n_maps: 222,
      home_league: "LCS",
      exact_roster: null,
      ...ratingEvidence(),
    },
  ],
  "features/player_ratings_snapshot.json": [
    playerRating("Chovy", 1_766, 263, "Gen.G"),
    playerRating("Faker", 1_713, 314, "T1"),
    playerRating("Inspired", 1_709, 222, "LYON"),
    playerRating("Bdd", 1_680, 180, "KT Rolster"),
  ],
  "features/team_records.json": {
    "T1": teamRecord("LCK", 204, 314),
    "Gen.G": teamRecord("LCK", 192, 263),
    "KT Rolster": teamRecord("LCK", 100, 180),
    "LYON": teamRecord("LCS", 144, 222),
  },
  "features/player_records.json": {
    "Chovy": playerRecord("Gen.G", "LCK", "mid", 192, 263),
    "Faker": playerRecord("T1", "LCK", "mid", 204, 314),
    "Inspired": playerRecord("LYON", "LCS", "jungle", 144, 222),
    "Bdd": playerRecord("KT Rolster", "LCK", "mid", 100, 180),
  },
  "features/player_champion_records.json": {
    "Chovy": [championRecord("Galio", 37, 31)],
    "Bdd": [championRecord("Galio", 8, 5)],
    "Faker": [
      championRecord("Azir", 10, 5),
      championRecord("Annie", 10, 7),
      championRecord("Orianna", 10, 9),
    ],
    "Inspired": [
      championRecord("Skarner", 7, 3),
      championRecord("Vi", 17, 9),
      championRecord("Xin Zhao", 27, 20),
    ],
  },
  "features/profile_records.json": {
    schema_version: "scryglass:profile-records:v3",
    grade_contract: "scryglass:player-map-grade:v2",
    window_days: 120,
    champion_images: {},
    games: { [game.game_id]: game },
    players: {
      "Chovy": [game.game_id],
      "Faker": [game.game_id],
      "Inspired": [],
      "Bdd": [],
    },
    teams: {
      "T1": [game.game_id],
      "Gen.G": [game.game_id],
      "LYON": [],
      "KT Rolster": [],
    },
  },
  "features/match_index.json": {
    schema_version: "scryglass:match-index:v1",
    years: [2026],
    games: [{
      game_id: game.game_id,
      date: game.date,
      league: game.league,
      competition_tier: game.competition_tier,
      blue_team: game.blue_team,
      red_team: game.red_team,
      blue_win: game.blue_win,
      champions: game.players.map((row) => row.champion),
      grades_available: 2,
    }],
  },
  "features/team_weekly_ranks.json": {
    as_of: "2026-08-12",
    previous_as_of: "2026-08-05",
    current_through: "2026-08-12T18:00:00Z",
    by_team: {
      "T1": { rank: 1, delta: 0 },
      "Gen.G": { rank: 2, delta: 1 },
      "LYON": { rank: 3, delta: -1 },
      "KT Rolster": { rank: 4, delta: 0 },
    },
  },
  "features/player_weekly_ranks.json": {
    as_of: "2026-08-12",
    previous_as_of: "2026-08-05",
    current_through: "2026-08-12T18:00:00Z",
    by_player: {
      "Chovy": { tier1: { rank: 1, delta: 0 } },
      "Faker": { tier1: { rank: 2, delta: 1 } },
      "Inspired": { tier1: { rank: 3, delta: -1 } },
      "Bdd": { tier1: { rank: 4, delta: 0 } },
    },
  },
  "rankings/tierlists.json": {
    status: "available",
    generated_at: "2026-08-13T00:00:01Z",
    as_of: "2026-08-12T18:00:00Z",
    source_freshness: "oe_daily_export",
    options: {
      roles: ["mid"],
      patches: ["16.15"],
      tier_buckets: ["A", "B", "D"],
    },
    scopes: [{
      scope_id: "patch:16.15:mid",
      scope_kind: "patch",
      role: "mid",
      patch: "16.15",
      as_of: "2026-08-12T18:00:00Z",
      status: "production",
      row_count: 3,
    }],
    rows: [
      tierRow("Azir", 268, 1, "A", 100, 4.2),
      tierRow("Galio", 3, 2, "B", 45, 1.1),
      tierRow("Zed", 238, 3, "D", 20, -3.8),
    ],
  },
};

function encoded(value) {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function rowCount(value) {
  if (Array.isArray(value)) return value.length;
  if (value && typeof value === "object") {
    if (Array.isArray(value.games)) return value.games.length;
    return Object.keys(value).length;
  }
  return null;
}

const serializedAssets = Object.entries(assets).map(([relative, value]) => {
  const raw = encoded(value);
  return {
    relative,
    raw,
    bytes: Buffer.byteLength(raw),
    sha256: createHash("sha256").update(raw).digest("hex"),
    rows: rowCount(value),
  };
});

const forbiddenDraftData = serializedAssets.filter(({ relative, raw }) => (
  relative.toLowerCase().includes("draft")
  || /"draft_(?:pool|contribution)"/.test(raw)
));
if (forbiddenDraftData.length) {
  throw new Error(`The E2E fixture contains Draft data: ${forbiddenDraftData.map((row) => row.relative).join(", ")}`);
}

const manifest = {
  pack_id: releaseId,
  schema_version: "scryglass:e2e-public-pack:v1",
  created_utc: "2026-08-13T00:00:01Z",
  filters: { years: [2026], leagues: "compact_deterministic_e2e" },
  attribution: "Synthetic Scryglass browser-test fixture.",
  excluded: ["real game rows", "research artifacts", "predictive output", "Draft data"],
  base_url: null,
  data_backend: "local",
  tier: { status: "available", as_of: "2026-08-12T18:00:00Z" },
  draft_authority: {
    schema_version: "scryglass:draft-authority:v1",
    status: "unavailable",
    release_id: releaseId,
    model_version: null,
    receipt_sha256: null,
    reason: "The deterministic E2E fixture contains no Draft data or independent promotion receipt.",
  },
  ratings: {
    source_mode: "synthetic_e2e",
    source_as_of: "2026-08-12T18:00:00Z",
    window_years: [2026],
    team_rating_rows: assets["features/ratings_snapshot.json"].length,
    player_rating_rows: assets["features/player_ratings_snapshot.json"].length,
    claim_ceiling: "descriptive_test_fixture",
  },
  total_bytes: serializedAssets.reduce((total, file) => total + file.bytes, 0),
  total_files: serializedAssets.length,
  files: serializedAssets.map((file) => ({
    path: file.relative,
    relative: file.relative,
    rows: file.rows,
    cols: null,
    bytes: file.bytes,
    sha256: file.sha256,
  })),
};

await rm(packRoot, { recursive: true, force: true });
for (const file of serializedAssets) {
  const target = path.join(releaseRoot, file.relative);
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, file.raw, { encoding: "utf8", mode: 0o600 });
}
await mkdir(packRoot, { recursive: true });
await writeFile(path.join(packRoot, "manifest.json"), encoded(manifest), { encoding: "utf8", mode: 0o600 });

process.stdout.write(`${packRoot}\n`);
