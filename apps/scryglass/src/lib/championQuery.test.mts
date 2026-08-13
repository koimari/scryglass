import assert from "node:assert/strict";
import test from "node:test";
import { queryChampions, type PublishedTierBoard } from "./championQuery";
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

const tierBoard: PublishedTierBoard = {
  options: { patches: ["16.15"] },
  rows: [
    { champion: "Azir", role: "mid", patch: "16.15", rank: 1, tier_bucket: "A", played_maps: 100 },
    { champion: "Tahm Kench", role: "mid", patch: "16.15", rank: 2, tier_bucket: "B", played_maps: 100 },
    { champion: "Zed", role: "mid", patch: "16.15", rank: 3, tier_bucket: "D", played_maps: 100 },
    { champion: "Riven", role: "top", patch: "16.15", rank: 1, tier_bucket: "A", played_maps: 100 },
  ],
};

test("general worst champion follows the published tier list", () => {
  const result = queryChampions(index, "what is the worst champion in general?", tierBoard);
  assert.equal(result.tier, "tier1");
  assert.equal(result.metric, "tier");
  assert.equal(result.minimumGames, 1);
  assert.equal(result.rows[0].champion, "Zed");
  assert.equal(result.rows[0].rank, 3);
  assert.match(result.answer.headline, /Zed \(mid\) ranks last on the published 16\.15 champion tier list/);
  assert.match(result.answer.caveat, /published patch tier-list order/);
});

test("general champion rankings support direction, role, and sample controls", () => {
  const best = queryChampions(index, "what is the best Tier 1 mid champion with at least 100 games?", tierBoard);
  assert.equal(best.rows[0].champion, "Azir");
  assert.equal(best.rows[0].games, 100);

  const allTiers = queryChampions(index, "bottom 2 champions across all tiers with at least 20 games", tierBoard);
  assert.equal(allTiers.tier, "all");
  assert.equal(allTiers.rows[0].champion, "Zed");
  assert.equal(allTiers.rows[1].champion, "Tahm Kench");
  assert.equal(allTiers.rows.length, 2);
});

test("explicit champion win-rate questions keep the descriptive aggregate metric", () => {
  const result = queryChampions(index, "which champion has the lowest win rate?", tierBoard);
  assert.equal(result.metric, "win_rate");
  assert.equal(result.rows[0].champion, "Tahm Kench");
  assert.match(result.answer.headline, /lowest published Tier 1 win rate at 42%/);
});
