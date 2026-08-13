import assert from "node:assert/strict";
import test from "node:test";
import { filterChatMatchesByTeam, findPlayerRecord, type ChatMatch } from "./chatData";

test("player records match rating names without losing source casing", () => {
  const record = findPlayerRecord(
    { Inspired: { wins: 119, games: 182, wr: 0.6538 } },
    "inspired",
  );

  assert.deepEqual(record, { wins: 119, games: 182, wr: 0.6538 });
});

test("exact team match excludes academy teams with the same prefix", () => {
  const match = (game_id: string, blue_team: string, red_team: string): ChatMatch => ({
    game_id,
    blue_team,
    red_team,
    date: "2026-08-08T00:00:00Z",
    league: "LCK",
    blue_win: 1,
    champions: [],
  });
  const games = [
    match("main", "T1", "Hanwha Life Esports"),
    match("academy", "T1 Esports Academy", "DN SOOPers Challengers"),
  ];

  assert.deepEqual(filterChatMatchesByTeam(games, "T1").map((game) => game.game_id), ["main"]);
});
