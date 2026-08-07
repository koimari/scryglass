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

## Recommended backlog (identity refresh completion)

1. Regenerate OE private target evidence + split (`write_artifacts()`), re-pin
   `oe_target_evidence.py` / `series_cluster_proxy.py`, re-run interactions.
2. Re-pin `g1_draft_features.py` players hash once the L4 real-v1 pipeline
   next materializes (with receipt regeneration).
3. Re-freeze the C0 contract tree after the docs settle (single deliberate
   freeze, updating `types.py` + trust root + reconciliation reference).
4. Re-pin L4 review snapshot after the next real-v1 L4 run.
