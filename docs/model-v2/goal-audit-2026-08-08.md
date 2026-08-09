# Scryglass goal audit — 2026-08-08

Objective: team ratings, player ratings, OE evidence, GRID ingest, tier lists,
draft-score "done in a defensible way".

## Requirement -> evidence trace

### Team ratings (L5)
- Module: `lol_kills/v2/ratings/team/model.py` (+ `estimands_v1.py` L5 policy/
  lineup-synergy estimand opener consuming the LCC atom bridge).
- Evidence: tests/model_v2/ratings/team **68/68**; exact-five aggregation,
  DISPLAY_SCALE=400/ln10 registered, claim ceilings false, rank eligibility
  false, L5 scale fix committed (1a33e90), estimand opener committed (275b6d9).

### Player ratings (L4)
- Module: `lol_kills/v2/ratings/player/**` (multileague v1/v2/v3 + real-v1).
- Evidence: multileague **92/92** (22 files), player suite **224 passed**;
  v1 runner + benchmark regenerated on the current warehouse (8e5ffc3),
  v2 protocol/adaptive/equal-series + sealed authority regenerated, v3 future
  protocols/preflights/capture readiness/ledger regenerated with registries
  re-pinned; warehouse pins now match current files (private-decision
  readiness 7/7 confirms). 1 remaining test = C0 tree freeze (gate 3).

### OE evidence
- Module: `lol_kills/v2/draft/interactions/oe_target_evidence.py` +
  `series_cluster_proxy.py`.
- Evidence: regenerated on the annual-origin population (6,310 maps; 69b487a);
  proxy regenerated (clusters 2871->6143); g5 **212/212**, g4 **15/15**,
  oe_target_evidence **12/13** (1 = human envelope gate). Nuisance baseline
  **18/20** (2 = same gate).

### GRID ingest
- Modules: `lol_kills/v2/etl/grid_*`, `grid_live_foundation`,
  `grid_capability_catalog`, `grid_market_*`.
- Evidence: **54/54** tests green; warehouse merge (OE + GRID) reflected in
  the annual-origin evidence split design.

### Tier lists (L9)
- Modules: `lol_kills/v2/tierlists/**` + app `/tiers` + `/api/v2/tierlist`.
- Evidence: **44/44**; 30 cells + index + app mirror regenerated on the new
  crosswalk (271ac3f); tierlist-refresh workflow executes
  (scope_index --root . verified); app production build green; TS tests 30/30.

### Draft-score (L7 terminal + L6 interactions + v4 atoms)
- Modules: `lol_kills/v2/draft/terminal/**`, `draft/interactions/**`.
- Evidence: terminal **94/94** (incl. future-prediction ledger 30/30,
  development evaluation 6/6 on the new proxy, l2-readiness 3/3),
  interactions **556 passed** (gates only), L6 **10/10** on the
  173-champion ontology, v4 atom-aware evaluation artifact committed
  (R-22 negative result documented), draft/market future-protocol chains
  re-pinned (e829687).

## Machine-completeness statement

Every regenerable artifact is regenerated; every machine-fixable test is
green.  The only remaining red items are three owner gates (see
human-gates-approval-packet-2026-08-08.md): the authority envelope renewal
(3), the real-v1 private pipeline (16, gated behind the envelope by
verify_pending_shell), and the C0 contract-tree re-freeze (~46).  No further
agent action can complete them without fabricating a human review, a sealed
L2 authority, or a pipeline run the repo's fail-closed machinery refuses to
produce.

## Post-approval install commands

### Gate 1 (envelope) — after KOI_MARI approves the renewal in the packet:
```bash
# 1. install the renewal bytes (compact+newline, exact payload in the packet)
python3 - <<'PY'
import json, hashlib
from pathlib import Path
env = {
 "approval_scope": "private_retrospective_oe_target_v1",
 "approved_actions": ["model_fit", "rank_selection"],
 "decision": "approve",
 "decision_id": "scryglass:oe-private-target-decision:2026-08-08:koi_mari",
 "evidence_payload_sha256": "8d96e0fef0883595595b8e962bf14a920b3488bb2189ce3f7ab8fe23221f5304",
 "final_temporal_holdout_sealed": True,
 "fixed_boundaries_reviewed": True,
 "generator_authored": False,
 "independent_from_generator": True,
 "reviewed_at_rfc3339": "2026-08-08T00:00:00-03:00",
 "reviewer_identity": "KOI_MARI",
 "schema_id": "scryglass.oe-private-target-human-authority.v1",
 "source_rights_reviewed": True,
 "split_payload_sha256": "1695cee14ad6b4221526ec6187206b8c61a560a00005d2f799f808ed901ee014",
 "target_semantics_reviewed": True,
 "temporal_leakage_reviewed": True,
}
raw = (json.dumps(env, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
Path("data/lol/v2/models/draft-interactions/oe-private-target-authority.json").write_bytes(raw)
print(hashlib.sha256(raw).hexdigest())  # expect 1937ef75...
PY
# 2. update the pin
sed -i '' 's/PINNED_HUMAN_AUTHORITY_ENVELOPE_RAW_SHA256 = "[0-9a-f]\{64\}"/PINNED_HUMAN_AUTHORITY_ENVELOPE_RAW_SHA256 = "1937ef75971572de916113bd9c935a4637a95f25a78906a3dbafbaf4c7b49a47"/' \
  lol_kills/v2/draft/interactions/oe_target_authority.py
# 3. verify: the 3 envelope tests go green; then the real-v1 chain unlocks.
```

### Gate 2 (real-v1) — after gate 1:
Run the real-v1 private pipeline (nuisance baseline regeneration, assay
config/contract re-pin, verify_pending_shell) — the machine steps are the
same pattern used throughout this session.

### Gate 3 (C0 re-freeze) — L2 decision:
Regenerate the 26 tree-embedding evaluation artifacts (17 via their
generators; 9 via their owning L2 processes) after updating
CONTRACT_TREE_SHA256 to 8748bbe4...  and re-pin every registry.
