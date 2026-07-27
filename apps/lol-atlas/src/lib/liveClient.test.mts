import assert from "node:assert/strict";
import test from "node:test";
import {
  parseExternalProbability,
  probabilityLabel,
  validProbabilityPair,
} from "./liveClient.ts";

test("live probability surfaces reject impossible or non-complementary values", () => {
  assert.equal(probabilityLabel(1.01), "—");
  assert.equal(probabilityLabel(-0.01), "—");
  assert.equal(validProbabilityPair(0.57, 0.43), true);
  assert.equal(validProbabilityPair(0.57, 0.57), false);
  assert.deepEqual(parseExternalProbability("57/43"), {
    pBlue: 0.57,
    pRed: 0.43,
    raw: {},
  });
  assert.throws(
    () => parseExternalProbability('{"p_blue": 0.8, "p_red": 0.4}'),
    /complementary/,
  );
  assert.throws(
    () => parseExternalProbability('{"p_blue": 101}'),
    /outside/,
  );
});
