import assert from "node:assert/strict";
import test from "node:test";
import { groupMapsIntoSeries, type SeriesRow } from "./series";

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

test("a real GRID series id keeps side-swapped games together", () => {
  const grouped = groupMapsIntoSeries([
    gridMap({
      grid_series_id: "series-42",
      game_uid: "LDL_91841",
      game: 1,
      blue_teamname: "Bilibili Gaming",
      red_teamname: "Anyone's Legend",
    }),
    gridMap({
      grid_series_id: "series-42",
      game_uid: "LDL_91845",
      game: 2,
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
});
