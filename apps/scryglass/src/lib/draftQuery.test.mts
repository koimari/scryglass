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
    draft_contribution: {
      schema_version: "scryglass:composition-signal:v1",
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
  assert.match(best.answer.headline, /T1 has the highest average published draft win share/);
  assert.match(worst.answer.headline, /KT has the lowest average published draft win share/);
  assert.match(best.answer.basis, /Tier 1/);

  const fullOrder = queryTeamDraftScores(records, "show team draft scores from best to worst");
  assert.deepEqual(fullOrder.rows.map((row) => row.team), ["T1", "Gen.G", "HLE", "KT"]);
});

test("team draft rankings support limits, sample floors, and named teams", () => {
  const bottom = queryTeamDraftScores(records, "bottom 2 team draft scores with at least 3 drafts");
  assert.deepEqual(bottom.rows.map((row) => row.team), ["KT", "HLE"]);

  const named = queryTeamDraftScores(records, "what is T1's draft score?");
  assert.deepEqual(named.rows.map((row) => row.team), ["T1"]);
  assert.match(named.answer.headline, /67%/);
  assert.match(named.answer.headline, /across 3 games/);

  const allTiers = queryTeamDraftScores(records, "best team draft score across all tiers with at least 1 draft");
  assert.equal(allTiers.rows[0].team, "Academy");
});

test("team draft scores compare two teams over their published history", () => {
  const comparison = queryTeamDraftScores(records, "which team has better draft score historically between T1 and Gen.G?");
  assert.equal(comparison.kind, "team_draft_comparison");
  assert.deepEqual(comparison.rows.map((row) => row.team), ["T1", "Gen.G"]);
  assert.equal(comparison.comparison?.winner, "T1");
  assert.match(comparison.answer.headline, /T1 has the higher historical average draft win share/);
  assert.match(comparison.answer.headline, /67%/);
  assert.match(comparison.answer.headline, /49%/);
  assert.match(comparison.answer.headline, /18 percentage-point edge/);
  assert.ok((comparison.comparison?.win_share_gap ?? 0) > 0.16);
  assert.doesNotMatch(comparison.answer.headline, /[+-]\d+\.\d{2}/);
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
