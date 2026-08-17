import assert from "node:assert/strict";
import test from "node:test";
import type { ProfileRecords } from "./pack";
import { queryTeamDraftScores } from "./draftQuery";

const records = {
  schema_version: "scryglass:profile-records:v2",
  window_days: 90,
  champion_images: {},
  players: {},
  teams: {},
  games: {
    one: game("one", "T1", "Gen.G", 1.2, 0.4),
    two: game("two", "T1", "HLE", 0.8, 0.3),
    three: game("three", "T1", "KT", 1, 0.2),
    four: game("four", "Gen.G", "HLE", 0.6, 0.5),
    five: game("five", "Gen.G", "KT", 0.7, 0.1),
    six: game("six", "HLE", "KT", 0.4, 0),
    unavailable: game("unavailable", "T1", "Gen.G", 99, 99, "unavailable"),
    tierTwo: { ...game("tierTwo", "Academy", "Challengers", 50, 40), competition_tier: "tier2" },
  },
} satisfies ProfileRecords;

const aliasRecords = {
  ...records,
  games: {
    ...records.games,
    kcOne: game("kc-one", "Karmine Corp", "G2 Esports", 1.2, 0.4),
    kcTwo: game("kc-two", "G2 Esports", "Karmine Corp", 0.9, 0.3),
    kcThree: game("kc-three", "Karmine Corp", "G2 Esports", 1.1, 0.2),
  },
} satisfies ProfileRecords;

const shareRankingRecords = {
  ...records,
  games: {
    rawHighOne: game("raw-high-one", "RawHigh", "Opponent One", 3, -2),
    rawHighTwo: game("raw-high-two", "RawHigh", "Opponent Two", -2, 3),
    shareHighOne: game("share-high-one", "ShareHigh", "Opponent Three", 0.4, 0),
    shareHighTwo: game("share-high-two", "ShareHigh", "Opponent Four", 0.4, 0),
  },
} satisfies ProfileRecords;

const leagueRecords = {
  ...records,
  games: {
    ...records.games,
    lecOne: { ...game("lec-one", "LEC High", "LEC Opponent One", 2, 0), league: "LEC" },
    lecTwo: { ...game("lec-two", "LEC High", "LEC Opponent Two", 2, 0), league: "LEC" },
    lecThree: { ...game("lec-three", "LEC High", "LEC Opponent Three", 2, 0), league: "LEC" },
    lcsOne: { ...game("lcs-one", "LCS High", "LCS Opponent One", 1.5, 0), league: "LCS" },
    lcsTwo: { ...game("lcs-two", "LCS High", "LCS Opponent Two", 1.5, 0), league: "LCS" },
    lcsThree: { ...game("lcs-three", "LCS High", "LCS Opponent Three", 1.5, 0), league: "LCS" },
    pcsOne: { ...game("pcs-one", "PCS High", "PCS Opponent One", 1.4, 0), league: "PCS", competition_tier: "tier2" },
    pcsTwo: { ...game("pcs-two", "PCS High", "PCS Opponent Two", 1.4, 0), league: "PCS", competition_tier: "tier2" },
    pcsThree: { ...game("pcs-three", "PCS High", "PCS Opponent Three", 1.4, 0), league: "PCS", competition_tier: "tier2" },
    msiOne: { ...game("msi-one", "MSI High", "MSI Opponent One", 1.3, 0), league: "MSI", competition_tier: "international" },
    msiTwo: { ...game("msi-two", "MSI High", "MSI Opponent Two", 1.3, 0), league: "MSI", competition_tier: "international" },
    msiThree: { ...game("msi-three", "MSI High", "MSI Opponent Three", 1.3, 0), league: "MSI", competition_tier: "international" },
  },
} satisfies ProfileRecords;

function game(
  gameId: string,
  blueTeam: string,
  redTeam: string,
  blueSignal: number,
  redSignal: number,
  status: "available" | "unavailable" = "available",
): ProfileRecords["games"][string] {
  return {
    game_id: gameId,
    date: "2026-08-01",
    league: "LCK",
    competition_tier: "tier1",
    blue_team: blueTeam,
    red_team: redTeam,
    blue_win: 1,
    players: [],
    draft_pool: {
      schema_version: "scryglass:draft-pool:v1",
      status: "complete",
      source: "published-tier-list",
      patch: "26.16",
      bans: {
        Blue: ["BanA", "BanB", "BanC", "BanD", "BanE"],
        Red: ["BanF", "BanG", "BanH", "BanI", "BanJ"],
      },
      picked: [
        "Ahri", "Azir", "Jinx", "Nautilus", "Gnar",
        "Orianna", "Sejuani", "Varus", "Rakan", "Renekton",
      ].map((champion, index) => ({
        side: index < 5 ? "Blue" as const : "Red" as const,
        role: ["mid", "jungle", "bot", "support", "top"][index % 5],
        champion,
        order: index + 1,
      })),
      unpicked: [],
    },
    draft_contribution: {
      schema_version: "scryglass:draft-descriptive-signal:v1",
      status,
      model_version: "test",
      fit_through: null,
      blue: { signal: blueSignal, prior_role_games: 10 },
      red: { signal: redSignal, prior_role_games: 10 },
      picks: [],
      note: "test",
      ...(status === "unavailable" ? { reason: "test" } : {}),
    },
  };
}

test("team draft scores support best, worst, and the ordered rows between", () => {
  const best = queryTeamDraftScores(records, "which team has the best draft score");
  const worst = queryTeamDraftScores(records, "which team has the worst draft score");
  assert.deepEqual(best.rows.map((row) => row.team), ["T1", "Gen.G", "HLE", "KT"]);
  assert.deepEqual(worst.rows.map((row) => row.team), ["KT", "HLE", "Gen.G", "T1"]);
  assert.match(best.answer.headline, /T1 has the highest average descriptive draft edge/);
  assert.match(worst.answer.headline, /KT has the lowest average descriptive draft edge/);
  assert.match(best.answer.basis, /Tier 1/);

  const fullOrder = queryTeamDraftScores(records, "show team draft scores from best to worst");
  assert.deepEqual(fullOrder.rows.map((row) => row.team), ["T1", "Gen.G", "HLE", "KT"]);
});

test("team draft rankings support limits, sample floors, and named teams", () => {
  const bottom = queryTeamDraftScores(records, "bottom 2 team draft scores with at least 3 drafts");
  assert.deepEqual(bottom.rows.map((row) => row.team), ["KT", "HLE"]);

  const named = queryTeamDraftScores(records, "what is T1's draft score?");
  assert.deepEqual(named.rows.map((row) => row.team), ["T1"]);
  assert.match(named.answer.headline, /\+0\.70 model units/);
  assert.match(named.answer.headline, /across 3 games/);

  const allTiers = queryTeamDraftScores(records, "best team draft score across all tiers with at least 1 draft");
  assert.equal(allTiers.rows[0].team, "Academy");
});

test("team draft scores compare two teams over their published history", () => {
  const comparison = queryTeamDraftScores(records, "which team has better draft score historically between T1 and Gen.G?");
  assert.equal(comparison.kind, "team_draft_comparison");
  assert.deepEqual(comparison.rows.map((row) => row.team), ["T1", "Gen.G"]);
  assert.equal(comparison.comparison?.winner, "T1");
  assert.match(comparison.answer.headline, /T1 has the higher average descriptive draft edge in the active 90-day profile window/);
  assert.match(comparison.answer.headline, /\+0\.70 model units/);
  assert.match(comparison.answer.headline, /-0\.03 model units/);
  assert.match(comparison.answer.headline, /gap is 0\.73 model-unit for T1/);
  assert.ok((comparison.comparison?.difference ?? 0) > 0.7);
  assert.match(comparison.answer.headline, /[+-]\d+\.\d{2}/);
  assert.match(comparison.answer.basis, /active 90-day profile window/);
  assert.match(comparison.answer.basis, /not all seasons/);

  const insufficient = queryTeamDraftScores(records, "compare T1 and Gen.G draft scores with at least 4 drafts");
  assert.equal(insufficient.kind, "team_draft_comparison");
  assert.equal(insufficient.comparison, undefined);
  assert.match(insufficient.answer.headline, /do not support a complete historical comparison/);
});

test("team draft comparisons resolve common team aliases", () => {
  const comparison = queryTeamDraftScores(aliasRecords, "who has the best draft between KC and G2?");
  assert.equal(comparison.kind, "team_draft_comparison");
  assert.deepEqual(comparison.rows.map((row) => row.team), ["Karmine Corp", "G2 Esports"]);
});

test("draft rankings use the descriptive edge metric", () => {
  const result = queryTeamDraftScores(shareRankingRecords, "which team has the best draft with at least 2 drafts");
  assert.equal(result.rows[0]?.team, "ShareHigh");
  assert.match(result.answer.headline, /ShareHigh has the highest average descriptive draft edge at \+0\.40 model units/);
});

test("team draft rankings stay inside the requested league", () => {
  const result = queryTeamDraftScores(leagueRecords, "which team has the best draft in LEC?");
  assert.deepEqual(result.rows.map((row) => row.team), ["LEC High"]);
  assert.match(result.answer.headline, /LEC High has the highest average descriptive draft edge in LEC at \+2\.00 model units/);
  assert.match(result.answer.basis, /Ranked 1 team .* in LEC, Tier 1/);

  const emea = queryTeamDraftScores(leagueRecords, "which team has the best draft in EMEA?");
  assert.deepEqual(emea.rows.map((row) => row.team), ["LEC High"]);

  const americas = queryTeamDraftScores(leagueRecords, "which team has the best draft in the Americas?");
  assert.deepEqual(americas.rows.map((row) => row.team), ["LCS High"]);

  const secondary = queryTeamDraftScores(leagueRecords, "which team has the best draft in PCS?");
  assert.equal(secondary.tier, "tier2");
  assert.deepEqual(secondary.rows.map((row) => row.team), ["PCS High"]);

  const international = queryTeamDraftScores(leagueRecords, "which team has the best draft at MSI?");
  assert.equal(international.tier, "international");
  assert.deepEqual(international.rows.map((row) => row.team), ["MSI High"]);
});
