import assert from "node:assert/strict";
import test from "node:test";
import { playerPositionDeltas } from "./playerMovement";

test("reads movement from the player's current tier payload", () => {
  const movement = playerPositionDeltas({
    tier1: {
      position_deltas: {
        "1m": { as_of: "2026-08-08", rank: 9, delta: -1 },
        "3m": { as_of: "2026-08-08", rank: 9, delta: 0 },
        "12m": { as_of: "2026-08-08", rank: 12, delta: 2 },
      },
    },
  }, "tier1");
  assert.equal(movement?.["1m"]?.delta, -1);
  assert.equal(movement?.["3m"]?.delta, 0);
  assert.equal(movement?.["12m"]?.delta, 2);
});

test("keeps the legacy direct payload shape readable", () => {
  const movement = playerPositionDeltas({
    position_deltas: { "1m": { as_of: "2026-08-08", rank: null, delta: null } },
  }, "tier1");
  assert.equal(movement?.["1m"]?.delta, null);
});

test("fails closed for malformed movement payloads", () => {
  assert.equal(playerPositionDeltas(null, "tier1"), undefined);
  assert.equal(playerPositionDeltas({ tier1: null }, "tier1"), undefined);
});
