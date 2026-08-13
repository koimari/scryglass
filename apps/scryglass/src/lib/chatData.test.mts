import assert from "node:assert/strict";
import test from "node:test";
import { findPlayerRecord } from "./chatData";

test("player records match rating names without losing source casing", () => {
  const record = findPlayerRecord(
    { Inspired: { wins: 119, games: 182, wr: 0.6538 } },
    "inspired",
  );

  assert.deepEqual(record, { wins: 119, games: 182, wr: 0.6538 });
});
