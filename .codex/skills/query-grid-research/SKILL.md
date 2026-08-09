---
name: query-grid-research
description: Safely inspect and query the private Scryglass GRID integration for personal research. Use when Codex needs GRID schema or capability discovery, series/team/player/game identity crosswalks, bounded metadata queries, League of Legends live/final state field selection, checkpoint-safe feature sourcing, file-list discovery, provenance receipts, or routing a research question to GRID Central Data, Series State, Series Events, or already-local Riot files.
---

# Query GRID research

Keep every operation private and read-only. Never print, serialize, interpolate
into logs, or return `GRID_API_KEY`. Do not publish or redistribute raw GRID
data. Do not claim model, betting, probability, latency, or market-edge
authority from data access alone.

## Start from the catalog

Read `assets/grid-capability-catalog.v1.json` before constructing a query.
Verify `catalog_sha256`, `generated_at`, endpoint schema hashes, capability
status, and limitations. Treat `unknown`, `unverified`, `not_tested`, and
`restricted` as unavailable.

Read:

- `references/query-workflows.md` for routing and safe query templates.
- `references/field-guide.md` for League of Legends fields, temporal
  classification, identity, and checkpoint rules.

Regenerate the catalog only when freshness matters or schema drift is
suspected:

```bash
python3 /Users/river/.codex/skills/query-grid-research/scripts/build_catalog.py \
  --repo /Users/river/scryglass \
  --probe-series-id VERIFIED_ALREADY_LOCAL_SERIES_ID
```

The probe series must already be locally known and professional. The builder
performs introspection and small metadata/file-list probes only. It does not
download files or open the WebSocket.

## Query workflow

1. Define the question, population, exact as-of boundary, and whether the
   result is metadata, checkpoint state, outcome, or a derived feature.
2. Route through the table in `references/query-workflows.md`.
3. Resolve provider identity before querying state. Keep GRID IDs and Riot IDs
   in separate typed fields.
4. Query the smallest field set and bounded page needed. Serialize requests,
   honor `Retry-After`, and stop on 429.
5. Record endpoint, canonical query hash, variables with secrets excluded,
   response/source hash, retrieval time, provider `updatedAt`, series/game IDs,
   pagination cutoffs, and catalog/schema hashes.
6. For checkpoint evidence, select the last state/event whose provider game
   time is at or before the checkpoint. Apply the declared maximum-age rule.
   Never use receipt time as a substitute for game time.
7. Fail closed on ambiguous identity, missing clocks/sequences, gaps,
   revisions, stale state, incomplete outcomes, or schema drift.

## Hard boundaries

- Never guess IDs or scan sequential IDs.
- Never infer PUUIDs, teams, sides, players, patches, or leagues from names
  alone.
- Never use `won`, `finished`, `forfeited`, final duration/totals, end-state
  files, or post-checkpoint revisions as predictor inputs.
- Never treat schema presence as proof a field was populated at a historical
  checkpoint.
- Never download historical files unless the current task explicitly
  authorizes that separate phase.
- Never query mutation roots. Their presence in Central Data introspection is
  capability metadata, not write authorization.
- Never expose signed file URLs. Retain only sanitized file metadata.
- Never treat GRID access as evidence of bookmaker odds access or market edge.
