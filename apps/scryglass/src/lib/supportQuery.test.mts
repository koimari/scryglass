import assert from "node:assert/strict";
import test from "node:test";
import {
  buildSupportQueryIndex,
  executeQueryPlan,
  parseQueryPlan,
  planPlayerQuestion,
  wilsonLowerBound,
} from "./supportQuery";

const index = buildSupportQueryIndex({
  ratings: [
    { player: "Faker", mu_total: 1700, n_maps: 300, last_team: "T1", home_league: "LCK", evidence_active: 1 },
    { player: "Chovy", mu_total: 1800, n_maps: 280, last_team: "Gen.G", home_league: "LCK", evidence_active: 1 },
    { player: "Knight", mu_total: 1750, n_maps: 260, last_team: "Bilibili Gaming", home_league: "LPL", evidence_active: 1 },
    { player: "Tier Two Mid", mu_total: 1900, n_maps: 150, last_team: "Academy", home_league: "LCKC", evidence_active: 1 },
    { player: "MISSING", mu_total: 1000, n_maps: 20, last_team: "LNG Esports", home_league: "LPL", evidence_active: 1 },
    { player: "Random", mu_total: 1500, n_maps: 20, last_team: "Disruptors", home_league: "AL", evidence_active: 1 },
    { player: "random", mu_total: 1400, n_maps: 20, last_team: "Team Solid", home_league: "CD", evidence_active: 1 },
  ],
  records: {
    Faker: { wins: 195, games: 300, wr: 0.65, current_team: "T1", current_league: "LCK", current_tier: "tier1", primary_role: "mid" },
    Chovy: { wins: 196, games: 280, wr: 0.7, current_team: "Gen.G", current_league: "LCK", current_tier: "tier1", primary_role: "mid" },
    Knight: { wins: 169, games: 260, wr: 0.65, current_team: "Bilibili Gaming", current_league: "LPL", current_tier: "tier1", primary_role: "mid" },
    "Tier Two Mid": { wins: 100, games: 150, wr: 0.6667, current_team: "Academy", current_league: "LCKC", current_tier: "tier2", primary_role: "mid" },
    MISSING: { wins: 10, games: 20, wr: 0.5, current_team: "LNG Esports", current_league: "LPL", current_tier: "tier1", primary_role: "sup" },
    Random: { wins: 10, games: 20, wr: 0.5, current_team: "Disruptors", current_league: "AL", current_tier: "tier1", primary_role: "mid" },
    random: { wins: 10, games: 20, wr: 0.5, current_team: "Team Solid", current_league: "CD", current_tier: "tier1", primary_role: "mid" },
  },
  champions: {
    Faker: [
      { champion: "Galio", games: 30, wins: 24, wr: 0.8 },
      { champion: "Azir", games: 40, wins: 20, wr: 0.5 },
      { champion: "Orianna", games: 20, wins: 13, wr: 0.65 },
    ],
    Chovy: [{ champion: "Galio", games: 5, wins: 5, wr: 1 }],
    Knight: [{ champion: "Galio", games: 20, wins: 12, wr: 0.6 }],
  },
  profiles: {
    games: {
      one: { players: [{ player: "Faker", grade: { status: "available", grade: "A" } }] },
      two: { players: [{ player: "Chovy", grade: { status: "available", grade: "B" } }] },
    },
  },
});

test("Wilson score is fail-closed and rewards stronger evidence", () => {
  assert.equal(wilsonLowerBound(2, 1), null);
  assert.ok((wilsonLowerBound(24, 30) ?? 0) > (wilsonLowerBound(5, 5) ?? 1));
});

test("query-plan parser rejects arbitrary fields and operators", () => {
  const arbitraryField = parseQueryPlan({
    version: 1,
    dataset: "players",
    operation: "rank",
    filters: [{ field: "salary", op: "gte", value: 1 }],
    orderBy: [{ field: "rating", direction: "desc" }],
    limit: 5,
  });
  assert.equal(arbitraryField.ok, false);

  const arbitraryKey = parseQueryPlan({
    version: 1,
    dataset: "players",
    operation: "rank",
    filters: [],
    orderBy: [{ field: "rating", direction: "desc" }],
    limit: 5,
    sql: "select * from players",
  });
  assert.equal(arbitraryKey.ok, false);
});

test("dynamic entity resolution compares any published player names", () => {
  const planned = planPlayerQuestion("which player is the best rated between faker and chovy", index);
  assert.ok(planned.ok);
  const result = executeQueryPlan(planned.plan, index);
  assert.deepEqual(result.rows.map((row) => row.name), ["Chovy", "Faker"]);
  assert.match(result.answer.headline, /Chovy ranks higher than Faker by 100 rating points/);
});

test("dynamic entity resolution accepts a possessive player name", () => {
  const planned = planPlayerQuestion("what is Faker's rating", index);
  assert.ok(planned.ok);
  const result = executeQueryPlan(planned.plan, index);
  assert.deepEqual(result.rows.map((row) => row.name), ["Faker"]);
  assert.match(result.answer.headline, /Faker has the highest matching published rating at 1700/);
});

test("a named player lookup does not inherit the open-ranking tier default", () => {
  const planned = planPlayerQuestion("what is Tier Two Mid's rating", index);
  assert.ok(planned.ok);
  const result = executeQueryPlan(planned.plan, index);
  assert.deepEqual(result.rows.map((row) => row.name), ["Tier Two Mid"]);
});

test("win-rate comparisons report percentage points", () => {
  const planned = planPlayerQuestion("who has the better win rate, Faker or Chovy", index);
  assert.ok(planned.ok);
  const result = executeQueryPlan(planned.plan, index);
  assert.match(result.answer.headline, /Chovy ranks higher than Faker by 5 percentage points on win rate/);
});

test("vague champion question uses the declared Tier 1 evidence rule", () => {
  const planned = planPlayerQuestion("who is the best Galio player", index);
  assert.ok(planned.ok);
  assert.equal(planned.plan.dataset, "player_champions");
  assert.deepEqual(planned.plan.filters, [
    { field: "champion", op: "eq", value: "Galio" },
    { field: "active", op: "eq", value: 1 },
    { field: "tier", op: "eq", value: "tier1" },
    { field: "champion_games", op: "gte", value: 5 },
  ]);
  const result = executeQueryPlan(planned.plan, index);
  assert.equal(result.rows[0].name, "Faker");
  assert.match(result.answer.basis, /95% Wilson lower bound/);
  assert.match(result.answer.caveat ?? "", /not a champion-specific rating/);
});

test("best-rated champion question orders overall rating within champion evidence", () => {
  const planned = planPlayerQuestion("who is the best rated Galio player", index);
  assert.ok(planned.ok);
  const result = executeQueryPlan(planned.plan, index);
  assert.equal(result.rows[0].name, "Chovy");
  assert.equal(result.plan.orderBy[0].field, "rating");
});

test("a named player's worst champion returns the ordered record set", () => {
  const planned = planPlayerQuestion("what is Faker's worst champion?", index);
  assert.ok(planned.ok);
  assert.equal(planned.plan.dataset, "player_champions");
  assert.deepEqual(planned.plan.orderBy[0], { field: "champion_win_rate", direction: "asc" });
  assert.equal(planned.plan.limit, 20);
  const result = executeQueryPlan(planned.plan, index);
  assert.deepEqual(result.rows.map((row) => row.champion), ["Azir", "Orianna", "Galio"]);
  assert.match(result.answer.headline, /Faker's lowest published champion win rate is 50% on Azir across 40 games/);
});

test("best and bottom N champion questions use the same bidirectional ranking", () => {
  const best = planPlayerQuestion("show me Faker's best 2 champions", index);
  const worst = planPlayerQuestion("show me Faker's bottom 2 champions", index);
  assert.ok(best.ok);
  assert.ok(worst.ok);
  assert.deepEqual(executeQueryPlan(best.plan, index).rows.map((row) => row.champion), ["Galio", "Orianna"]);
  assert.deepEqual(executeQueryPlan(worst.plan, index).rows.map((row) => row.champion), ["Azir", "Orianna"]);
});

test("a named player's champion list supports both full ordering phrases", () => {
  const descending = planPlayerQuestion("list Faker's champions from best to worst", index);
  const ascending = planPlayerQuestion("rank Faker's champions from worst to best", index);
  assert.ok(descending.ok);
  assert.ok(ascending.ok);
  assert.deepEqual(executeQueryPlan(descending.plan, index).rows.map((row) => row.champion), ["Galio", "Orianna", "Azir"]);
  assert.deepEqual(executeQueryPlan(ascending.plan, index).rows.map((row) => row.champion), ["Azir", "Orianna", "Galio"]);
});

test("median performance selects the champion closest to the player's median win rate", () => {
  const planned = planPlayerQuestion("what is Faker's most median performance champion?", index);
  assert.ok(planned.ok);
  assert.equal(planned.plan.orderBy[0].field, "champion_median_distance");
  assert.equal(planned.plan.orderBy[0].direction, "asc");
  const result = executeQueryPlan(planned.plan, index);
  assert.equal(result.rows[0].champion, "Orianna");
  assert.match(result.answer.headline, /Faker's median-performance champion is Orianna at 65%/);
});

test("median performance uses the midpoint of an even champion sample", () => {
  const planned = planPlayerQuestion("what is Faker's most median performance champion?", {
    ...index,
    playerChampions: index.playerChampions.filter((row) => ["Faker", "Galio", "Azir"].includes(row.name) && row.champion !== "Orianna"),
  });
  assert.ok(planned.ok);
  const result = executeQueryPlan(planned.plan, {
    ...index,
    playerChampions: index.playerChampions.filter((row) => row.name === "Faker" && row.champion !== "Orianna"),
  });
  assert.equal(result.rows[0].champion, "Azir");
  assert.match(result.answer.headline, /median champion win rate of 65%/);
});

test("median performance resolves a player introduced with for", () => {
  const planned = planPlayerQuestion("what is the median performance champion for Faker?", index);
  assert.ok(planned.ok);
  assert.deepEqual(planned.plan.filters[0], { field: "name", op: "eq", value: "Faker" });
  assert.equal(executeQueryPlan(planned.plan, index).rows[0].champion, "Orianna");
});

test("average performance also chooses the closest champion record", () => {
  const planned = planPlayerQuestion("what is Faker's average performance champion?", index);
  assert.ok(planned.ok);
  assert.equal(planned.plan.orderBy[0].field, "champion_mean_distance");
  assert.equal(planned.plan.orderBy[0].direction, "asc");
  const result = executeQueryPlan(planned.plan, index);
  assert.equal(result.rows[0].champion, "Orianna");
  assert.match(result.answer.headline, /Faker's mean-performance champion is Orianna/);
});

test("filtered rankings apply dynamic league, role, tier, and games constraints", () => {
  const planned = planPlayerQuestion("best Tier 1 LCK mid with at least 100 games", index);
  assert.ok(planned.ok);
  const result = executeQueryPlan(planned.plan, index);
  assert.deepEqual(result.rows.map((row) => row.name), ["Chovy", "Faker"]);
  assert.ok(result.rows.every((row) => row.tier === "tier1" && row.league === "LCK" && row.role === "mid"));
});

test("unresolved comparison fails closed", () => {
  const planned = planPlayerQuestion("who is better, Faker or Missing Person", index);
  assert.equal(planned.ok, false);
  if (!planned.ok) assert.match(planned.reason, /could not resolve two/);
});

test("an unresolved named lookup fails closed instead of returning a leaderboard", () => {
  const planned = planPlayerQuestion("what is Missing Person's rating", index);
  assert.equal(planned.ok, false);
  if (!planned.ok) assert.match(planned.reason, /could not resolve that player name/);
});

test("duplicate handles require team context", () => {
  const ambiguous = planPlayerQuestion("what is Random's rating", index);
  assert.equal(ambiguous.ok, false);
  if (!ambiguous.ok) assert.match(ambiguous.reason, /ambiguous/);

  const resolved = planPlayerQuestion("what is Random's rating on Disruptors", index);
  assert.ok(resolved.ok);
  const result = executeQueryPlan(resolved.plan, index);
  assert.deepEqual(result.rows.map((row) => row.name), ["Random"]);
  assert.deepEqual(result.rows.map((row) => row.team), ["Disruptors"]);
});
