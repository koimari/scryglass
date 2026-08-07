# Identity-layer re-pin 2026-08-07 (crosswalk / preflight / tierlists / C1)

## Why

The warehouse parquet files (`maps.parquet`, `oe_player_games.parquet`) were
refreshed on 2026-08-01 (OE baseline reconcile). The champion identity layer
was pinned to the pre-refresh snapshot, so every module that replays those
files failed closed with "pinned source bytes changed" — the documented
"stale OE consumer pins intentionally pending" state.

During the 2026-08-07 session the crosswalk and preflight artifacts were
**regenerated against the current warehouse** (irreversible: the old
untracked bytes were not recoverable). Everything downstream was then
re-pinned and regenerated so the tree is coherent again:

| Component | Action |
|---|---|
| `data/lol/warehouse/parquet/*.parquet` | (already refreshed 08-01) |
| `data/lol/v2/models/draft-interactions/representation-assay-preflight.json` | regenerated (12,708 valid maps) |
| `data/lol/v2/champions/champion-id-crosswalk-v1.json` | regenerated (OE names 170→171) |
| `lol_kills/v2/champions/id_crosswalk.py` | pinned constants updated |
| `lol_kills/v2/tierlists/model.py` `CROSSWALK_ARTIFACT` | re-pinned |
| 30 tierlist cells + `index-v1.json` + app public copies | regenerated (rows identical; lineage crosswalk hash updated) |
| `lol_kills/v2/data/g1_draft_features.py` + `g1-pre-event-interface-receipt.json` | crosswalk pins + receipt regenerated |
| `lol_kills/v2/evaluation/checkpoint_c1.py` + C1 config/report/authority | re-pinned (new seed/sources/snapshot-id) + regenerated |
| `lol_kills/v2/ratings/player/model.py` + candidate identity | C1 anchors updated |
| `contract-validation-trust-root.json` + anchors | l3 seed/sources re-pinned |

## What this session deliberately did NOT touch (pre-existing drift, documented elsewhere)

- `oe_target_evidence.py` / `series_cluster_proxy.py` module pins were left at
  the pre-refresh values; regenerating their evidence artifacts would cascade
  into the g5/real-v1/evaluation sealed layer (see backlog).
- `g1_draft_features.py` still pins the pre-refresh players parquet hash.
- The C0 contract tree (`CONTRACT_TREE_SHA256`) is frozen at the value from
  before the latest `docs/model-v2/*` edits; `contract-validation-remand` and
  `contract-reconciliation` failures are the pre-existing doc-drift boundary.
- L4 `lpl-private-draft-features-review.json` pins the old crosswalk digest.

## Known new failures (2 tests)

`tests/model_v2/draft/interactions/test_oe_target_evidence.py::test_real_source_replay_*`
(×2): the replay now stops at the preflight/crosswalk pin (old pin vs new
artifacts). Fix = re-pin + regenerate the OE private evidence/split artifacts
via `oe_target_evidence.write_artifacts()`, then re-run the g5/real-v1
consumers. Kept as pending rather than cascading through the sealed layer
unprompted.

## OE evidence + split regeneration (2026-08-07, completed)

`write_artifacts()` regenerated from the current warehouse + new proxy:
split/evidence now cover the **annual-origin population (6,310 maps)** — the
proxy's 12,708-map population includes 6,398 grid-sourced maps that belong to
the separate GRID pipeline and have no raw annual OE origin. Changes:
- `oe_target_evidence.py`: annual-origin membership filter in
  `build_outcome_free_split`; all source/generator pins re-pinned;
  `EXPECTED_ASSIGNED_MAPS` 6194 -> 6310 (split/evidence rebuilt deterministically).
- `series_cluster_proxy.py`: preflight/maps/players pins + cluster audit
  constants updated for the 12,708-map warehouse (clusters 2871 -> 6143);
  proxy regenerated (generator pin -> d63ee58b).
- Consumers re-pinned + regenerated: g5 prefit contract bundle (contract,
  review-core, pre-fit-review, execution review-core/pending-report),
  g5 runner/v3-prefit module pins, real_v1_g4 pending artifacts
  (chronology/source-binding/review-core/dry-run/pending-report).
- Suites green: g5 212/212, g4 52-slot 15/15, series_cluster_proxy 30/30,
  oe_target_evidence 12/13, development_v3 6/6.

**Human approval gate (by design, not fabricated):** the independent human
authority envelope (`oe-private-target-authority.json`, reviewed by KOI_MARI
2026-07-29) binds the OLD evidence/split payloads. The regenerated evidence is
a new experiment population and needs a fresh review + envelope renewal
(update the envelope's evidence/split payload shas + `PINNED_HUMAN_AUTHORITY_ENVELOPE_RAW_SHA256`).
Until then: oe_target_evidence `test_exact_human_authority...` (1) and
oe_nuisance_baseline replay (2) fail closed. Coverage-preflight (2) waits on
the L4 real-v1 private-runner snapshot regeneration (below).

## Recommended backlog (identity refresh completion)

1. ~~Regenerate OE private target evidence + split (`write_artifacts()`)~~ DONE.
   Remaining: human authority envelope renewal (above) + L4 real-v1 snapshot.
2. Re-pin `g1_draft_features.py` players hash once the L4 real-v1 pipeline
   next materializes (with receipt regeneration).
3. Re-freeze the C0 contract tree after the docs settle (single deliberate
   freeze, updating `types.py` + trust root + reconciliation reference).
4. Re-pin L4 review snapshot after the next real-v1 L4 run.
