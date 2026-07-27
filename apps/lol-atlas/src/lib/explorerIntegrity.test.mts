import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  aggregateChampionRows,
  favoriteHitRateFromRow,
  normalizeMapQueryRow,
  patchFilterClause,
  mapFilterClauses,
  resolveMapWinnerSide,
  sumKnownNumbers,
} from "./duck.ts";
import {
  detailSourceLabel,
  formatObjectiveCount,
  formatPlayerKda,
} from "../components/MatchScoreboard.tsx";
import { validatedGateSummary } from "../components/DraftWrPanel.tsx";
import { parseMinGamesParam } from "../components/EloLadders.tsx";
import { ratingHistorySource } from "../components/MatchLoader.tsx";
import { latestNormalizedPatches } from "../components/TeamEloDetail.tsx";

test("map winner resolution fails closed on absent or contradictory evidence", () => {
  assert.equal(resolveMapWinnerSide({}), null);
  assert.equal(resolveMapWinnerSide({ blue_result: 1, y_blue_win: 0 }), null);
  assert.equal(resolveMapWinnerSide({ blue_result: 0 }), "red");
  assert.equal(resolveMapWinnerSide({ y_blue_win: 1 }), "blue");
});

test("player outcomes require evidence from both sides and must agree", () => {
  assert.equal(
    resolveMapWinnerSide({}, [{ side: "Blue", result: 1 }]),
    null,
  );
  assert.equal(
    resolveMapWinnerSide(
      {},
      [
        { side: "Blue", result: 1 },
        { side: "Red", result: 0 },
      ],
    ),
    "blue",
  );
  assert.equal(
    resolveMapWinnerSide(
      { blue_result: 0 },
      [
        { side: "Blue", result: 1 },
        { side: "Red", result: 0 },
      ],
    ),
    null,
  );
});

test("unknown values never become observed zero in a derived total", () => {
  assert.equal(sumKnownNumbers([4, 7]), 11);
  assert.equal(sumKnownNumbers([0, 0]), 0);
  assert.equal(sumKnownNumbers([4, null]), null);
  assert.equal(sumKnownNumbers(["bad", 7]), null);
});

test("scoreboard count and KDA cells preserve missing and malformed values", () => {
  assert.equal(formatObjectiveCount(null), "—");
  assert.equal(formatObjectiveCount("bad"), "—");
  assert.equal(formatObjectiveCount(-1), "—");
  assert.equal(formatObjectiveCount(0), "0");
  assert.equal(
    formatPlayerKda({ kills: 4, deaths: null, assists: 7 }),
    "4/—/7",
  );
  assert.equal(
    formatPlayerKda({ kills: 0, deaths: 0, assists: 0 }),
    "0/0/0",
  );
});

test("map-detail provenance is explicit, including an unknown state", () => {
  assert.equal(detailSourceLabel("oe_wide_feature_map"), "OE wide-map detail");
  assert.equal(detailSourceLabel("grid_event_detail"), "GRID event detail");
  assert.equal(detailSourceLabel(""), "Detail provenance unavailable");
});

test("favorite hit rate distinguishes valid empty, query, and integrity states", () => {
  assert.deepEqual(favoriteHitRateFromRow({ n: 0, hits: 0 }), {
    status: "sample_empty",
    n: 0,
    hits: 0,
    rate: null,
  });
  assert.deepEqual(favoriteHitRateFromRow({ n: 4, hits: 3 }), {
    status: "ok",
    n: 4,
    hits: 3,
    rate: 0.75,
  });
  assert.equal(
    favoriteHitRateFromRow({
      n: 4,
      hits: 3,
      duplicate_history_games: 1,
    }).status,
    "error",
  );
  assert.equal(favoriteHitRateFromRow({ n: 2, hits: 3 }).status, "error");
});

test("canonical series fields take precedence in browser rows", () => {
  const normalized = normalizeMapQueryRow({
    grid_game_index: 9,
    canonical_game_index: 2,
    grid_completion_source: "events_game_end",
    canonical_series_completion_source: "canonical_verified",
  });
  assert.equal(normalized.grid_game_index, 2);
  assert.equal(normalized.grid_completion_source, "canonical_verified");
});

test("champion aggregates preserve missing metrics and expose coverage", () => {
  const [aggregate] = aggregateChampionRows([
    {
      champion: "Ahri",
      kills: null,
      deaths: 2,
      assists: 7,
      totalgold: null,
      dpm: 600,
      minionkills: 100,
      monsterkills: 10,
      result: 1,
    },
    {
      champion: "Ahri",
      kills: 4,
      deaths: null,
      assists: 5,
      totalgold: 10_000,
      dpm: null,
      cspm: 2,
      gamelength: 600,
      result: 0,
    },
  ]);
  assert.equal(aggregate.kills, 4);
  assert.equal(aggregate.killsN, 1);
  assert.equal(aggregate.deaths, 2);
  assert.equal(aggregate.deathsN, 1);
  assert.equal(aggregate.gold, 10_000);
  assert.equal(aggregate.goldN, 1);
  assert.equal(aggregate.dpm, 600);
  assert.equal(aggregate.dpmN, 1);
  assert.equal(aggregate.cs, 65);
  assert.equal(aggregate.csN, 2);
});

test("patch filtering compares normalized major and minor exactly", () => {
  const clause = patchFilterClause("16.01");
  assert.match(clause, /= 16/);
  assert.match(clause, /= 1/);
  assert.doesNotMatch(clause, /ILIKE|%/);
  assert.equal(patchFilterClause("16.1"), "FALSE");
  assert.equal(patchFilterClause("not-a-patch"), "FALSE");
});

test("side filtering accepts every selected team on the requested side", () => {
  const blue = mapFilterClauses({
    teams: ["T1", "Gen.G"],
    side: "blue",
  }).join("\n");
  assert.match(blue, /blue_teamname ILIKE '%T1%'/);
  assert.match(blue, /blue_teamname ILIKE '%Gen\.G%'/);
  const red = mapFilterClauses({
    teams: ["T1", "Gen.G"],
    side: "red",
  }).join("\n");
  assert.match(red, /red_teamname ILIKE '%T1%'/);
  assert.match(red, /red_teamname ILIKE '%Gen\.G%'/);
});

test("draft gate copy is derived only from validated API evidence", () => {
  assert.deepEqual(
    validatedGateSummary({
      status: "withheld_failed_chronological_gate",
      evidence_status: "verified_immutable_pack_artifact",
      gate_id: "draft-probability-test",
      evidence: {
        decision: "withheld",
        final_test_maps: 2477,
        final_test_end: "2026-07-18T00:00:00Z",
      },
    }),
    {
      gateId: "draft-probability-test",
      finalTestMaps: 2477,
      finalTestEnd: "2026-07-18T00:00:00Z",
    },
  );
  assert.equal(
    validatedGateSummary({
      evidence_status: "unavailable_or_invalid",
      evidence: null,
    }),
    null,
  );
});

test("malformed minimum-game query values fail to a finite default", () => {
  assert.equal(parseMinGamesParam("not-a-number"), 20);
  assert.equal(parseMinGamesParam("2"), 5);
  assert.equal(parseMinGamesParam("25.9"), 25);
});

test("rating provenance and recent patch labels reflect available inputs", () => {
  assert.equal(
    ratingHistorySource(10, null),
    "pre-match team rating history",
  );
  assert.equal(
    ratingHistorySource(null, 5),
    "pre-match player rating history",
  );
  assert.deepEqual(
    latestNormalizedPatches([
      "16.9.1.2",
      "16.10.3.4",
      "16.11",
      "16.12.5",
      "bad",
    ]),
    ["16.10", "16.11", "16.12"],
  );
});

test("owned explorer surfaces keep Draft WR disabled and avoid accuracy wording", async () => {
  const headToHead = await readFile(
    new URL("../components/HeadToHead.tsx", import.meta.url),
    "utf8",
  );
  const browse = await readFile(
    new URL("../components/BrowseMaps.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(headToHead, /api\/draft-wr/);
  assert.doesNotMatch(`${headToHead}\n${browse}`, /\bElo accuracy\b/i);
  assert.match(browse, /favorite hit rate/i);
});

test("kill benchmark query uses a strict earlier-date cutoff", async () => {
  const duck = await readFile(new URL("./duck.ts", import.meta.url), "utf8");
  assert.match(
    duck,
    /TRY_CAST\(date AS TIMESTAMP\) < TRY_CAST\('\$\{escapedDate\}' AS TIMESTAMP\)/,
  );
});
