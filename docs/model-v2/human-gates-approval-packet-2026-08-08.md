# Human / L2 approval packet — 2026-08-08

Everything machine-regenerable is regenerated and green. The remaining reds
are exactly the gates below.  Each requires an owner decision; the exact
payloads/steps are prepared here so approval is a one-step confirm.

---

## 1. Authority-envelope renewal for the regenerated OE evidence (needs KOI_MARI)

**What changed:** the OE target evidence + split were regenerated on the
annual-origin population (commit 69b487a; split 6,194 -> 6,310 maps, proxy
clusters 2871 -> 6143). The independent human authority envelope
(`data/lol/v2/models/draft-interactions/oe-private-target-authority.json`,
reviewed 2026-07-29) binds the OLD evidence/split payloads, so all replays
fail closed ("human authority envelope does not bind exact evidence") —
by design.

**Affected tests (3):** oe_target_evidence `test_exact_human_authority...`,
oe_nuisance_baseline `test_persisted_artifact_replays...`,
`test_rehashed_authority_drift_is_rejected`.

**Renewal payload** (exact bytes to install, compact+newline format matching
the committed envelope):

```json
{"approval_scope":"private_retrospective_oe_target_v1","approved_actions":["model_fit","rank_selection"],"decision":"approve","decision_id":"scryglass:oe-private-target-decision:2026-08-08:koi_mari","evidence_payload_sha256":"8d96e0fef0883595595b8e962bf14a920b3488bb2189ce3f7ab8fe23221f5304","final_temporal_holdout_sealed":true,"fixed_boundaries_reviewed":true,"generator_authored":false,"independent_from_generator":true,"reviewed_at_rfc3339":"2026-08-08T00:00:00-03:00","reviewer_identity":"KOI_MARI","schema_id":"scryglass.oe-private-target-human-authority.v1","source_rights_reviewed":true,"split_payload_sha256":"1695cee14ad6b4221526ec6187206b8c61a560a00005d2f799f808ed901ee014","target_semantics_reviewed":true,"temporal_leakage_reviewed":true}
```

**New pin:** `PINNED_HUMAN_AUTHORITY_ENVELOPE_RAW_SHA256 = "1937ef75971572de916113bd9c935a4637a95f25a78906a3dbafbaf4c7b49a47"`
(update in `lol_kills/v2/draft/interactions/oe_target_authority.py`).

**Review basis (for the owner to verify before approving):** the renewed
envelope binds the regenerated evidence payload
`8d96e0fe...` and split payload `1695cee1...` (artifacts committed in
69b487a).  Nothing else in the envelope semantics changed; the review fields
(source rights, target semantics, temporal leakage, fixed boundaries,
holdout sealed) are unchanged from the 2026-07-29 decision.

---

## 2. L4 real-v1 private-runner regeneration (needs the real-v1 pipeline)

**Affected tests (14):** representation_rank_private_runner (5 failed + 7
errors) + real_v1_g4 coverage-preflight (2).

**Cause:** the representation-rank assay config
(`representation-rank-assay-config.json`) pins the pre-re-pin crosswalk /
proxy / split / evidence raw hashes; the run contract, pending report, assay
report, review permit and support gate embed the config identity.  The
generator (`generate_representation_rank_assay_artifacts.py`) explicitly
refuses to rebuild ("private fitting remains disabled"), and the config also
carries data-derived eligibility blocks (calendar-month cluster/map counts,
holdout maps) that would need recomputation from the regenerated split —
hand-editing hashes alone would produce an internally inconsistent config.

**Required:** run the real-v1 private pipeline (representation-rank assay
config generation + fit gating) on the regenerated sources, then re-pin the
contract/report/support-gate chain.  This is the same regeneration that
updates the g1 independent review artifact
(`lpl-private-draft-features-review.json`, human review, no generator).

---

## 3. C0 contract-tree re-freeze (L2 decision)

**Affected tests (~12):** contract_validation_remand (5),
contract_reconciliation_v1 (4+2), test_player_rating (1 — the remaining
failure).

**Cause:** `CONTRACT_TREE_SHA256` is frozen at `fb3de56d...`, which predates
the current `docs/model-v2/*` edits (current tree `8748bbe4...`).  The
trust-root / contract anchors were already re-pinned to the current docs
(schemas, examples, content — done this session); only the tree constant
lags.

**Impact of re-freezing (verified):** 26 evaluation artifacts embed the old
tree hash (b2 r20 foundation/selection, outer calibration, b3 coverage/
reliability, checkpoint-c1, synthetic registry, contract fixture authority,
reconciliation candidates...).  Re-freeze = update `types.py` +
`contract-validation-trust-root.json`, then regenerate all 26 via their
generators (`generate_r20_foundation_artifacts.py`,
`generate_outer_calibration_artifacts.py`, `generate_b3_coverage_artifacts.py`,
`generate_checkpoint_c1_artifacts.py`, `generate_r20_selection_artifacts.py`,
plus the b2 pipeline + reconciliation reference replay), re-pin their
registries/tests.

**Recommendation:** perform this as ONE deliberate freeze after the docs
settle (per docs/model-v2 policy); the current failure mode is fail-closed
and documented.

---

## Current green baseline (verified this session)

Team 68/68 | Champions 114/114 | Tier lists 44/44 | GRID 54/54 |
Multileague L4 92/92 (22 files) | Player suite 224 passed |
OE evidence 12/13 | g5 212/212 | g4 15/15 | Draft terminal full green |
Data (g1 + receipts) green | Market phase-one green | C1 110/110 |
b2 remand 15/15.

## Final verified inventory (2026-08-08, after the last regeneration pass)

Additional machine-completable work landed since the packet was first written:
- L6 draft-interactions artifacts regenerated on the 173-champion ontology
  (config/fixtures/report/authority/manifest; commit fc11896; L6 10/10).
- Market chain completed: probability-pipeline readiness + Betano quote
  adapter candidate regenerated + registries re-pinned (commit 5a75fbb);
  market suite 143/143; private-decision-readiness tests 7/7 (now assert the
  post-regeneration state).
- Verified green: top-level tests 435/435, champions 114, tierlists 44,
  team 68, player 224 (1 C0), data 244, draft-terminal 94, interactions 556
  (+7 errors, gates only), market 143, L2 synthetic authorities (r20
  foundation 55, r20 selection 60, outer calibration 47, b3 coverage 54),
  app TypeScript 30/30, app production build.

### Complete remaining-red inventory (all owner gates)

| Gate | Tests | Owner action |
|---|---|---|
| 1. Authority envelope renewal | oe_target_evidence 1, oe_nuisance 2 | Approve the renewal payload (section 1) |
| 2. L4 real-v1 private runner | private_runner 5+7, coverage_preflight 2, representation_rank_assay 2 | Run the real-v1 private pipeline (section 2) |
| 3. C0 contract-tree re-freeze | b1_sealed 33 err, contract_validation_remand 5, contract_reconciliation 4+2, contract_prior_tree_recovery 1, player_rating 1 | L2 re-freeze decision (section 3) |

## Dependency analysis (2026-08-08): the gates form a chain ending at one decision

`generate_representation_rank_assay_artifacts.verify_pending_shell` requires:
1. `require_exact_human_authority(authority_bytes, evidence, split, action="model_fit")`
   and `action="rank_selection"` with `reviewer_identity == "KOI_MARI"` and
   `final_temporal_holdout_sealed == True` — i.e. **gate 1 (envelope renewal)
   is a hard prerequisite of gate 2 (real-v1 pipeline)**;
2. the nuisance baseline's OOF materialization (rows 5646) + final holdout
   (301 maps) contract — the nuisance artifact itself replays against the
   evidence/split/authority, so it too is envelope-gated.

Therefore the complete remaining chain is:

    Approve envelope renewal  ->  nuisance baseline regeneration (builder
    exists, envelope-gated)   ->  assay config/contract/report re-pin
    -> verify_pending_shell passes -> real-v1 private runner + coverage
    preflight green (16 tests)

and, independently, the C0 re-freeze decision (gate 3, ~46 tests) is an L2
sealed-layer operation (26 artifacts; 9 have no builder and must be
regenerated by their owning L2 processes).

No further agent-completable work exists: every regenerable artifact is
regenerated and every machine-fixable test is green.  The objective's six
areas are machine-complete; completion now requires the owner's decisions.
