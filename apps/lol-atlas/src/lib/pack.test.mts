import assert from "node:assert/strict";
import test from "node:test";
import { recordMatchesLeagues, scopedTeamWr, type TeamRecord } from "./pack.ts";

const currentTournaments = { LPL: "LPL - Split 3 2026" };

function teamRecord(currentTournament: string | null): TeamRecord {
  return {
    leagues: ["LPL"],
    primary: "LPL",
    intl: false,
    current_league: "LPL",
    current_tier: "tier1",
    current_date: "2026-07-25",
    current_tournament: currentTournament,
    wins: 8,
    games: 10,
    wr: 0.8,
    by_league: { LPL: { wins: 8, games: 10, wr: 0.8 } },
    by_tier: { tier1: { wins: 8, games: 10, wr: 0.8 } },
    by_tournament: {
      "LPL|LPL - Split 3 2026": { wins: 2, games: 3, wr: 0.6667 },
    },
  };
}

test("regional membership requires the pack-declared current tournament", () => {
  assert.equal(
    recordMatchesLeagues(teamRecord("LPL - Split 3 2026"), ["LPL"], {
      dataAsOf: "2026-07-26",
      currentTournaments,
    }),
    true,
  );
  assert.equal(
    recordMatchesLeagues(teamRecord("LPL - Split 2 2026"), ["LPL"], {
      dataAsOf: "2026-07-26",
      currentTournaments,
    }),
    false,
  );
  assert.equal(
    recordMatchesLeagues(teamRecord(null), ["LPL"], {
      dataAsOf: "2026-07-26",
      currentTournaments,
    }),
    false,
  );
});

test("scoped win rate uses current tournament observations when available", () => {
  const record = teamRecord("LPL - Split 3 2026");
  assert.equal(scopedTeamWr(record, ["LPL"], { currentTournaments }), 2 / 3);
  assert.equal(scopedTeamWr(record, ["TIER1"], { currentTournaments }), 2 / 3);
});

test("leagues without a tournament label retain the dated-observation fallback", () => {
  const record = teamRecord(null);
  assert.equal(
    recordMatchesLeagues(record, ["LPL"], {
      dataAsOf: "2026-07-26",
      currentTournaments,
    }),
    false,
  );
  assert.equal(
    recordMatchesLeagues(record, ["LPL"], {
      dataAsOf: "2026-07-26",
      currentTournaments: {},
    }),
    true,
  );
});
