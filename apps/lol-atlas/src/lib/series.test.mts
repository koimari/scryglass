import assert from "node:assert/strict";
import test from "node:test";
import {
  formatSeriesLabel,
  groupMapsIntoSeries,
  isQuarantinedSeriesRow,
  type SeriesRow,
} from "./series.ts";

function gridMap(overrides: Partial<SeriesRow>): SeriesRow {
  return {
    date: "2026-07-26 12:00:00",
    league: "LPL",
    tournament: "LPL 2026",
    patch: "16.14.794.5912",
    game: 1,
    source_grid: true,
    blue_result: 1,
    y_blue_win: 1,
    ...overrides,
  };
}

test("a GRID gameVersion shared by unrelated matches is never a series key", () => {
  const grouped = groupMapsIntoSeries([
    gridMap({
      game_uid: "LDL_91841",
      blue_teamname: "Bilibili Gaming",
      red_teamname: "Anyone's Legend",
    }),
    gridMap({
      game_uid: "LDL_91825",
      blue_teamname: "ThunderTalk Gaming",
      red_teamname: "Edward Gaming",
    }),
  ]);

  assert.equal(grouped.length, 2);
  assert.deepEqual(
    grouped.map((series) => [series.teamA, series.teamB]).sort(),
    [
      ["Anyone's Legend", "Bilibili Gaming"],
      ["Edward Gaming", "ThunderTalk Gaming"],
    ],
  );
});

test("a canonical series id keeps side-swapped games together", () => {
  const grouped = groupMapsIntoSeries([
    gridMap({
      grid_series_id: "series-42",
      canonical_series_id: "series-42",
      game_uid: "LDL_91841",
      game: 1,
      grid_game_index: 1,
      scheduled_best_of: 3,
      canonical_series_status: "completed",
      blue_teamname: "Bilibili Gaming",
      red_teamname: "Anyone's Legend",
    }),
    gridMap({
      grid_series_id: "series-42",
      canonical_series_id: "series-42",
      game_uid: "LDL_91845",
      game: 2,
      grid_game_index: 2,
      scheduled_best_of: 3,
      canonical_series_status: "completed",
      blue_teamname: "Anyone's Legend",
      red_teamname: "Bilibili Gaming",
      blue_result: 0,
      y_blue_win: 0,
    }),
  ]);

  assert.equal(grouped.length, 1);
  assert.equal(grouped[0].games.length, 2);
  assert.equal(grouped[0].winsA, 0);
  assert.equal(grouped[0].winsB, 2);
  assert.equal(grouped[0].bestOf, 3);
});

function seriesWithWinners(
  winners: Array<"Alpha" | "Beta">,
  bestOf: 3 | 5 | null = null,
) {
  return groupMapsIntoSeries(
    winners.map((winner, index) =>
      gridMap({
        grid_series_id: "format-series",
        canonical_series_id: "format-series",
        game_uid: `game-${index + 1}`,
        game: index + 1,
        grid_game_index: index + 1,
        scheduled_best_of: bestOf,
        canonical_series_status: bestOf ? "completed" : "unverified",
        blue_teamname: "Alpha",
        red_teamname: "Beta",
        blue_result: winner === "Alpha" ? 1 : 0,
        y_blue_win: winner === "Alpha" ? 1 : 0,
      }),
    ),
  )[0];
}

test("series format follows verified scheduled format, never final score", () => {
  const cases: Array<[Array<"Alpha" | "Beta">, number, number, 3 | 5]> = [
    [["Alpha", "Alpha"], 2, 0, 3],
    [["Alpha", "Beta", "Alpha"], 2, 1, 3],
    [["Alpha", "Alpha", "Alpha"], 3, 0, 5],
    [["Alpha", "Beta", "Alpha", "Alpha"], 3, 1, 5],
    [["Alpha", "Beta", "Alpha", "Beta", "Alpha"], 3, 2, 5],
  ];
  for (const [winners, winsA, winsB, bestOf] of cases) {
    const series = seriesWithWinners(winners, bestOf);
    assert.equal(series.winsA, winsA);
    assert.equal(series.winsB, winsB);
    assert.equal(series.bestOf, bestOf);
  }
  assert.equal(seriesWithWinners(["Alpha", "Alpha"]).bestOf, null);
  assert.equal(
    seriesWithWinners(["Alpha", "Alpha", "Alpha"]).bestOf,
    null,
  );
});

test("a tied two-map group is marked incomplete instead of claimed as Bo3", () => {
  const series = seriesWithWinners(["Alpha", "Beta"]);
  assert.equal(series.winsA, 1);
  assert.equal(series.winsB, 1);
  assert.equal(series.bestOf, null);
});

test("a GRID series with a missing map index is marked incomplete", () => {
  const grouped = groupMapsIntoSeries([
    gridMap({
      grid_series_id: "gap-series",
      grid_game_index: 1,
      game_uid: "gap-1",
      blue_teamname: "Alpha",
      red_teamname: "Beta",
    }),
    gridMap({
      grid_series_id: "gap-series",
      grid_game_index: 3,
      game_uid: "gap-3",
      blue_teamname: "Alpha",
      red_teamname: "Beta",
    }),
    gridMap({
      grid_series_id: "gap-series",
      grid_game_index: 4,
      game_uid: "gap-4",
      blue_teamname: "Alpha",
      red_teamname: "Beta",
    }),
  ]);
  assert.equal(grouped[0].bestOf, null);
});

test("a lone explicit GRID game is not promoted to a completed Bo1", () => {
  const grouped = groupMapsIntoSeries([
    gridMap({
      grid_series_id: "partial-series",
      grid_game_index: 2,
      game_uid: "partial-2",
      blue_teamname: "Alpha",
      red_teamname: "Beta",
    }),
  ]);
  assert.equal(grouped[0].bestOf, null);
});

test("GRID completion provenance survives grouping", () => {
  const grouped = groupMapsIntoSeries([
    gridMap({
      grid_series_id: "provenance-series",
      grid_game_index: 1,
      grid_completion_source: "end_state_summary",
      game_uid: "provenance-1",
      blue_teamname: "Alpha",
      red_teamname: "Beta",
    }),
    gridMap({
      grid_series_id: "provenance-series",
      grid_game_index: 2,
      grid_completion_source: "end_state_summary",
      game_uid: "provenance-2",
      blue_teamname: "Alpha",
      red_teamname: "Beta",
    }),
  ]);
  assert.equal(grouped[0].completionSource, "end_state_summary");
});

test("legacy map IDs never create per-map series records", () => {
  const grouped = groupMapsIntoSeries([
    gridMap({
      game_uid: "legacy-map-1",
      grid_series_id: "legacy-source-series",
      blue_teamname: "Alpha",
      red_teamname: "Beta",
    }),
    gridMap({
      game_uid: "legacy-map-2",
      grid_series_id: "different-source-series",
      blue_teamname: "Beta",
      red_teamname: "Alpha",
    }),
  ]);
  assert.equal(grouped.length, 1);
  assert.equal(grouped[0].recordKind, "unverified_map_group");
  assert.equal(grouped[0].bestOf, null);
  assert.equal(formatSeriesLabel(grouped[0]), "Unverified map group");
});

test("unknown or contradictory map outcomes are never assigned to red", () => {
  const grouped = groupMapsIntoSeries([
    gridMap({
      canonical_series_id: "unknown-outcome",
      canonical_series_status: "completed",
      blue_teamname: "Alpha",
      red_teamname: "Beta",
      blue_result: null,
      y_blue_win: null,
    }),
  ]);
  assert.equal(grouped[0].knownOutcomeMaps, 0);
  assert.equal(grouped[0].winsA, null);
  assert.equal(grouped[0].winsB, null);
});

test("canonical status must be recognized and completed", () => {
  assert.equal(
    isQuarantinedSeriesRow({
      canonical_series_id: "bad-status",
      canonical_series_status: "done",
    }),
    true,
  );
  assert.equal(
    isQuarantinedSeriesRow({
      canonical_series_id: "complete",
      canonical_series_status: "completed",
    }),
    false,
  );
  assert.equal(
    isQuarantinedSeriesRow({
      game_uid: "legacy",
      canonical_series_status: "done",
    }),
    false,
  );
});
