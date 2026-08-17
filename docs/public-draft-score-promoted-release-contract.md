# Promoted Draft Score release contract

## Purpose

This contract opens public Draft probability only after the frozen evaluation
and independent decision pass. It keeps the existing descriptive score valid
for releases that have descriptive authority.

## Required private inputs

The pack builder receives these files and expected SHA-256 values explicitly:

- `scryglass:public-draft-score-promotion-receipt:v1`
- the frozen candidate artifact;
- the frozen protocol;
- the final evaluation receipt;
- the independent decision;
- the paired intervention receipts used by the evaluation;
- one release result file for each published prediction.

Ambient checkout paths and environment-only receipt values are invalid inputs.
The pack builder hashes every file before parsing it.

## Promotion receipt checks

The pack builder must require:

- `status = promoted` and `authority = promoted`;
- exact model version and candidate receipt;
- exact protocol, evaluation, decision, and outcome hashes;
- the complete nonempty paired intervention receipt list;
- approved public fields in this exact order:
  `match_win_probability`, `controlled_draft_score`,
  `side_recommendation`;
- public probability and recommendation flags set to true;
- betting, odds, expected value, and stake authority set to false;
- a valid UTC issue time;
- a valid canonical receipt digest.

Any missing, extra, changed, or malformed binding closes the promoted lane.

## Manifest authority

The active manifest uses `scryglass:draft-authority:v1`:

```json
{
  "status": "promoted",
  "authority": "promoted",
  "release_id": "vYYYY.MM.DD.HHMMSS",
  "model_version": "public-draft-score-v1",
  "artifact_sha256": "<candidate file sha256>",
  "receipt_sha256": "<promotion receipt sha256>",
  "issued_utc": "<promotion issue time>",
  "estimand": "prematch_map_win_probability_with_controlled_draft_intervention",
  "probability_authority": true,
  "recommendation_authority": true,
  "betting_authority": false,
  "reason": null
}
```

The release ID is added during pack construction. The signed promotion receipt
does not claim a future release ID.

## Public result

Each result uses `scryglass:public-draft-score-result:v1` and contains:

- active release ID;
- model version;
- promotion receipt SHA-256;
- evidence window;
- Blue and Red match win probabilities that sum to one;
- controlled Draft model units;
- controlled Draft edge in percentage points;
- stronger Draft side;
- role-matched swap method and intervention receipt;
- full-probability side recommendation.

The result contains no odds, expected value, betting, or stake fields.

The promoted results live in a distinct allowlisted asset. Do not mix them into
the descriptive `features/draft_records.json` contract. Proposed path:

`features/promoted_draft_results.json`

This asset requires its own byte limit, row limit, MIME check, manifest digest,
Storage digest, active-release check, query projection, and cache invalidation.

## Supabase activation rules

The final activation function accepts these authority states:

- `unavailable`, with zero Draft assets;
- `descriptive`, with exactly one descriptive Draft asset;
- `promoted`, with exactly one descriptive asset and one promoted result asset.

For promoted authority, activation validates:

- exact fixed manifest fields;
- exact result schema and release binding;
- exact candidate and promotion receipt hashes;
- nonempty paired intervention receipts;
- probability complement and range;
- controlled Draft direction;
- recommendation direction;
- absence of forbidden betting fields;
- database metadata and immutable Storage bytes.

Restore repeats the same checks. Active and superseded objects remain
immutable. Anon and authenticated roles receive only fixed active-release RPC
projections. Base tables remain private.

## Web gate

`hasPromotedDraftAuthority()` can return true only after the active manifest RPC
has returned a sanitized promoted authority object. The promoted result parser
must bind the result release ID, model version, and receipt SHA to that manifest.

The web app does not parse private evaluation, decision, outcome, or reviewer
records. Those stay outside the public schema.

## Product behavior

When promoted authority is valid, the Draft tab and complete match drafts show:

- match win expectation;
- controlled Draft edge;
- stronger Draft side;
- side recommendation;
- model version, evidence window, release ID, and receipt;
- a statement that the recommendation is research output and contains no
  betting guidance.

An incomplete draft, malformed role assignment, duplicate champion, receipt
mismatch, inactive release, missing asset, or corrupt asset fails closed.

## Required tests

Python tests must cover receipt file tampering, canonical receipt tampering,
wrong candidate or protocol, missing paired evidence, extra public fields,
probability mismatch, recommendation mismatch, and forbidden betting fields.

SQL tests must cover zero-state replay, each authority state, direct-table
denial, active-only RPC output, Storage access, corruption, restore, and object
immutability.

Web tests must cover promoted rendering, descriptive rendering, unavailable
rendering, release rotation, receipt mismatch, malformed probabilities,
controlled Draft direction, and recommendation direction.

Production verification must prove the active release, manifest, result asset,
candidate, promotion receipt, and Storage bytes share the expected hashes.

## Rollout order

1. Apply additive promoted validators and RPC fields.
2. Deploy web code that understands all three authority states.
3. Produce a fresh promoted Storage-backed release.
4. Verify every hash and result through the candidate app.
5. Apply strict activation and direct-table cutover rules.
6. Activate the promoted release.
7. Invalidate manifest, Draft, match, profile, chat, and asset caches.
8. Probe production and the prior-release rollback path.

The descriptive active release remains available until the promoted release
passes every cutover probe.
