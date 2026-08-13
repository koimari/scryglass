import assert from "node:assert/strict";
import test from "node:test";
import { queryChampions } from "./championQuery";
import { buildSupportQueryIndex } from "./supportQuery";

const index = buildSupportQueryIndex({
  ratings: [
    { player: "Faker", mu_total: 1700, n_maps: 300, last_team: "T1", home_league: "LCK", evidence_active: 1 },
    { player: "Chovy", mu_total: 1800, n_maps: 280, last_team: "Gen.G", home_league: "LCK", evidence_active: 1 },
    { player: "Inactive", mu_total: 1900, n_maps: 100, last_team: "Old", home_league: "LCK", evidence_active: 0 },
  ],
  records: {
    Faker: { wins: 195, games: 300, wr: 0.65, current_team: "T1", current_league: "LCK", current_tier: "tier1", primary_role: "mid" },
    Chovy: { wins: 196, games: 280, wr: 0.7, current_team: "Gen.G", current_league: "LCK", current_tier: "tier1", primary_role: "mid" },
    Inactive: { wins: 90, games: 100, wr: 0.9, current_team: "Old", current_league: "LCK", current_tier: "tier1", primary_role: "mid" },
  },
  champions: {
    Faker: [
      { champion: "Tahm Kench", games: 60, wins: 20, wr: 1 / 3 },
      { champion: "Azir", games: 100, wins: 60, wr: 0.6 },
    ],
    Chovy: [
      { champion: "Tahm Kench", games: 60, wins: 30, wr: 0.5 },
      { champion: "Azir", games: 100, wins: 55, wr: 0.55 },
    ],
    Inactive: [{ champion: "Tahm Kench", games: 1000, wins: 0, wr: 0 }],
  },
});

test("general worst champion aggregates player records and excludes inactive rows", () => {
  const result = queryChampions(index, "what is the worst champion in general?");
  assert.equal(result.tier, "tier1");
  assert.equal(result.minimumGames, 100);
  assert.equal(result.rows[0].champion, "Tahm Kench");
  assert.equal(result.rows[0].games, 120);
  assert.equal(result.rows[0].wins, 50);
  assert.match(result.answer.headline, /Tahm Kench has the lowest published Tier 1 win rate at 42%/);
  assert.match(result.answer.caveat, /does not isolate champion strength/);
});

test("general champion rankings support direction, role, and sample controls", () => {
  const best = queryChampions(index, "what is the best Tier 1 mid champion with at least 100 games?");
  assert.equal(best.rows[0].champion, "Azir");
  assert.equal(best.rows[0].games, 200);

  const allTiers = queryChampions(index, "bottom 2 champions across all tiers with at least 20 games");
  assert.equal(allTiers.tier, "all");
  assert.equal(allTiers.rows[0].champion, "Tahm Kench");
  assert.equal(allTiers.rows.length, 2);
});
