import assert from "node:assert/strict";
import test from "node:test";
import {
  firstPickMetric,
  regionalOptions,
  regionalViewForRole,
  rowsForMode,
  type TierRow,
  type TierScope,
} from "./tierBoard.ts";

function row(overrides: Partial<TierRow> = {}): TierRow {
  return {
    scope_id: "patch:16.15",
    role: "top",
    patch: "16.15",
    champion: "Riven",
    champion_id: "riot:champion:92",
    champion_image_url: null,
    rank: 1,
    rank_delta: null,
    movement: "new",
    tier_bucket: "A",
    played_maps: 8,
    counterability_status: "unavailable",
    matchup_maps: 0,
    matchup_opponents: 0,
    expected_counter_breadth: null,
    ...overrides,
  };
}

test("first-pick values fall back to the published tier bucket", () => {
  assert.equal(firstPickMetric(row()), "A");
  assert.equal(firstPickMetric(row({ tier_value_pp: 4.25 })), "+4.3 pp");
});

test("matchup-only modes omit rows without their required evidence", () => {
  const rows = [
    row(),
    row({ champion: "Kennen", champion_id: "riot:champion:85", blind_score_pp: 2.2 }),
    row({ champion: "Jax", champion_id: "riot:champion:24", countered_opponent_count: 3 }),
  ];
  assert.deepEqual(rowsForMode(rows, "blind").map((item) => item.champion), ["Kennen"]);
  assert.deepEqual(rowsForMode(rows, "counter").map((item) => item.champion), ["Jax"]);
  assert.equal(rowsForMode(rows, "first_pick").length, 3);
});

test("regional choices combine every role for the selected patch", () => {
  const scopes: TierScope[] = [
    {
      scope_id: "patch:16.15",
      scope_kind: "patch",
      role: "top",
      patch: "16.15",
      as_of: "2026-08-08T00:00:00Z",
      status: "production",
      row_count: 1,
      regional_views: [{ id: "LCK", label: "LCK", maps: 12, basis: "observed", rows: [] }],
    },
    {
      scope_id: "patch:16.15",
      scope_kind: "patch",
      role: "support",
      patch: "16.15",
      as_of: "2026-08-08T00:00:00Z",
      status: "production",
      row_count: 1,
      regional_views: [{ id: "LEC", label: "LEC", maps: 9, basis: "observed", rows: [] }],
    },
  ];
  assert.deepEqual(regionalOptions(scopes, "16.15"), [
    { id: "LCK", label: "LCK" },
    { id: "LEC", label: "LEC" },
  ]);
  assert.equal(regionalViewForRole(scopes, "16.15", "support", "LEC")?.maps, 9);
});
