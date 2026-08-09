import assert from "node:assert/strict";
import test from "node:test";

import { compactPlayerRatings, type PlayerRating } from "./pack";

test("compact player ratings preserve the public evidence contract", () => {
  const row: PlayerRating = {
    player: "Chovy",
    mu_total: 1706.6,
    mu_regional: 1696.8,
    mu_meta: 9.8,
    sigma: 28,
    n_maps: 263,
    last_team: "Gen.G",
    evidence_interval_width: 109.8,
    evidence_precision_ratio: 1,
    evidence_stability: 2.6,
    evidence_freshness_days: 1.5,
    evidence_support_coverage: 1,
    evidence_fallback: 0,
    evidence_active: 1,
    evidence_disconnected: 0,
    evidence_ood: 0,
    evidence_state: "settled",
  };

  assert.deepEqual(compactPlayerRatings([row]), [row]);
});
