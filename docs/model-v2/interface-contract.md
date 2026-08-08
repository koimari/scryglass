# Interface contract

Contract version is `2.0.0`. Prose defines meaning; JSON Schemas define the
wire format. Generated Python and TypeScript types must come from the schemas
or prove structural parity in CI.

## 1. One estimator path

For each output and `model_version`, the model registry points to exactly one:

- Python estimator implementation;
- training snapshot and split plan;
- calibration transform;
- research artifact;
- production/runtime artifact; and
- evaluation report.

Python, artifact replay, API, and TypeScript runtime must return equivalent
pre-rounded values from identical input snapshots. Legacy estimators remain
named benchmark adapters only. No request-time cascade among draft engines is
permitted.

The registry refuses manifests whose evaluation artifact hash, served
calibration hash, or schema version differs from the promoted record.

## 2. Version fields

Every top-level response contains:

- `schema_version`: response schema, semantic version;
- `model_version`: immutable model release;
- `as_of`: input-information cutoff;
- `status`: canonical outputs use `ok` or `unavailable`; the typed noncanonical
  Archetype extrapolation branch alone uses `research_only`;
- `lineage`: manifest, training/source snapshot, cutoff, deterministic
  source-tree digest, optional candidate commit/release commit, and artifact
  hashes;
- `provenance`: immutable prediction-time record; and
- output-specific payload or a structured error, never both.

`status=ok` additionally requires provenance
`required_input_status=complete`, every required freshness check true, and no
missing/stale/conflict state. Every other required-input state uses
`status=unavailable`, retains safe lineage/provenance/error detail, and contains
no principal numeric payload.

Breaking field/meaning changes increment schema major. Model retraining changes
`model_version`, not schema version. Corrections produce a new artifact and
prediction ID.

## 3. Python boundary

The implementation package is isolated under `lol_kills/v2/`. Public protocols:

```python
class RatingEstimator(Protocol):
    def fit(self, snapshot: TrainingSnapshot, config: ModelConfig) -> ModelManifest: ...
    def snapshot(self, as_of: datetime, scope: RatingScope) -> RatingArtifact: ...

class TerminalDraftEstimator(Protocol):
    def score(self, state: TerminalDraftState, context: DraftContext | None,
              as_of: datetime) -> DraftScoreArtifact: ...

class PartialDraftEstimator(Protocol):
    def analyze(self, state: PartialDraftState, context: DraftContext | None,
                as_of: datetime, search: SearchBudget) -> PartialDraftArtifact: ...

class TierListEstimator(Protocol):
    def build(self, league_id: str, patch_id: str, role: Role,
              as_of: datetime) -> TierListArtifact: ...
```

All entry points:

- accept stable IDs and explicit `as_of`;
- load exactly one manifest by version or promoted channel;
- validate inputs before model execution;
- return schema-valid artifacts;
- expose deterministic seeds/config for approximate work;
- never download or refresh data implicitly; and
- never substitute neutral for requested stale contextual identity.

Training and serving code share pure feature/score functions. Training-only
objects cannot be imported by the browser bundle.

## 4. Artifact boundary

Canonical artifacts are immutable JSON plus columnar arrays where size
requires. JSON carries metadata and hashes for every referenced file.

Required schemas:

- [model-manifest.schema.json](contracts/model-manifest.schema.json)
- [prediction-provenance.schema.json](contracts/prediction-provenance.schema.json)
- [player-rating.schema.json](contracts/player-rating.schema.json)
- [team-rating.schema.json](contracts/team-rating.schema.json)
- [draft-score.schema.json](contracts/draft-score.schema.json)
- [partial-draft-state.schema.json](contracts/partial-draft-state.schema.json)
- [tier-list.schema.json](contracts/tier-list.schema.json)
- [publication-matrix.schema.json](contracts/publication-matrix.schema.json)

SHA-256 is computed on canonical bytes or an explicitly named canonicalization
method. Runtime compaction may quantize parameters only when parity and interval
coverage are re-evaluated on the compact artifact.

Schema-valid machine examples:

- [Player Rating](contracts/examples/player-rating.example.json)
- [Team Rating](contracts/examples/team-rating.example.json)
- [Terminal Draft Score](contracts/examples/draft-score.example.json)
- [Partial draft state](contracts/examples/partial-draft-state.example.json)
- [Tier list](contracts/examples/tier-list.example.json)

### Structural and semantic validation

Draft 2020-12 schemas validate shape, enums, required branches, and local
constants. They do not claim to prove cross-field arithmetic, uniqueness by a
nested ID, protocol legality, or interval ordering. Before any artifact may be
stored or returned as `status=ok`, the same versioned executable semantic
validator runs in training output, Python serving, artifact compilation,
TypeScript/runtime replay, and CI.

It must reject at least:

- `lower > upper` or a posterior mean outside its interval;
- a Team Rating roster with repeated player IDs or anything other than exactly
  one distinct player per canonical role;
- an illegal terminal action sequence, duplicate champion across sides,
  anything other than five assignments per side and one of each role, or a
  final assignment inconsistent with current role constraints;
- any mismatch between posterior mean and the single public rating field if a
  compatibility representation retains both;
- Draft Score other than `100 * probability`, scores that are not exact
  complements, calibrated-transform mismatch, or a raw logit that differs from
  its ledger sum beyond the declared tolerance;
- a terminal/partial transform lacking open-support, monotone-nondecreasing, and
  complement-symmetry proof fields or failing their replay;
- a partial terminal state that differs from canonical Terminal Draft Score;
- an Archetype extrapolation research response containing a 0–100 score,
  probability/advantage field, calibrated transform, or Reliability object, or
  lacking explicit support-gap, fallback, and uncertainty labels;
- an `ok` output without the required nonempty Reliability diagnostics
  (proper-score comparison, calibration, interval coverage, resolved-cluster
  support, and OOD state), or `reliability=high` when any required diagnostic is
  absent/failed, cluster support is zero, the transform is unapproved, or the
  output is OOD;
- a Reliability object whose recorded validation stratum and mapping hash do
  not equal the manifest's frozen total context/OOD mapping, including
  no-match=`unrated`;
- `settled=true` without strict greater-than-95% precision/stability, permitted
  interval width, current eligibility, fresh complete inputs, no material
  fallback/OOD, and validated coverage; and
- a promoted manifest without an opened sealed holdout, passing decision, all
  gate evidence, unchanged end-to-end sealed pipeline, required parity reports,
  and a required 1500/400 rating-display contract for rating artifacts.

Each output schema carries machine-readable negative mutation fixtures for its
structural failures and semantic-invariant IDs for executable checks. The
validation suite applies those mutations to the five canonical examples and
must observe the registered failure. A schema-valid but semantically invalid
artifact is still invalid and fails closed with `semantic_validation_failed`.

## 5. HTTP API

Recommended versioned routes:

| Method/path | Result |
|---|---|
| `GET /api/v2/ratings/players` | Player Rating snapshot/filter |
| `GET /api/v2/ratings/teams` | Exact-roster Team Rating snapshot/filter |
| `POST /api/v2/draft/score` | Terminal neutral/contextual Draft Score |
| `POST /api/v2/draft/partial` | Canonical Partial Draft Score, flex, signed strategic-response adjustment, recommendations |
| `POST /api/v2/draft/archetype-extrapolation` | Noncanonical ordinal research response for unsupported/new-champion actions; no score/probability/Reliability |
| `GET /api/v2/tier-lists` | One league-current-patch-role list |
| `GET /api/v2/models/{version}` | Public allowlisted model manifest |
| `GET /api/v2/predictions/{id}` | Immutable forecast/hindsight record, access permitting |

### Abbreviated terminal draft request fragment

This fragment shows request context only. Its one action is illustrative, not a
legal terminal request; a successful terminal request must supply the complete
protocol-legal action sequence and final role assignments.

```json
{
  "schema_version": "2.0.0",
  "as_of": "2026-07-27T18:00:00Z",
  "identity_mode": "contextual",
  "competition_scope_id": "scryglass:competition-scope:lpl",
  "patch_id": "26.14",
  "protocol_id": "scryglass:protocol:pro-standard-2026",
  "side_mapping": {
    "side_a_game_side": "blue",
    "side_b_game_side": "red",
    "side_a_draft_order": "first",
    "side_b_draft_order": "second",
    "mapping_source_id": "scryglass:source:protocol-example",
    "available_at": "2026-07-27T17:59:00Z"
  },
  "actions": [
    {"slot": 1, "kind": "pick", "canonical_side": "A", "champion_id": "riot:champion:115", "role_set": ["bot", "mid"]}
  ],
  "roster_a_id": "scryglass:roster:example-a",
  "roster_b_id": "scryglass:roster:example-b"
}
```

In contextual mode both exact roster IDs are required and must resolve fresh at
`as_of`. `identity_mode=neutral` is an explicit user choice and rejects roster
IDs to avoid ambiguous semantics.

### Successful Draft Score field fragment

The fragment below highlights semantics and deliberately omits the full
lineage/provenance objects. The linked machine example above is standalone and
schema-valid.

```json
{
  "schema_version": "2.0.0",
  "model_version": "draft-v2.0.0",
  "as_of": "2026-07-27T18:00:00Z",
  "status": "ok",
  "identity_mode": "contextual",
  "baseline_strength_equalized": true,
  "score_a": 56.2,
  "score_b": 43.8,
  "standardized_map_win_probability_a": 0.562,
  "interval_95": {"lower": 0.511, "upper": 0.611, "level": 0.95}
}
```

The fragment intentionally omits evidence, reliability, ledger, lineage, and
provenance rather than showing structurally invalid substitutes. The linked
full example is authoritative.

`score_a = 100 * standardized_map_win_probability_a` and
`score_b = 100 - score_a` before display rounding. Contextual output requires
`baseline_strength_equalized=true`. Team Rating values may appear in an
adjacent Match Forecast response, never inside the Draft Score transform.

### Unavailable field fragment

Production responses also include complete schema-valid lineage and provenance.

```json
{
  "schema_version": "2.0.0",
  "model_version": "draft-v2.0.0",
  "as_of": "2026-07-27T18:00:00Z",
  "status": "unavailable",
  "identity_mode": "contextual",
  "error": {
    "code": "stale_context",
    "message": "The selected roster context is older than this model permits.",
    "retryable": true,
    "missing_fields": [],
    "stale_fields": ["roster_a.policy_snapshot"]
  }
}
```

It must not contain a neutral score.

## 6. Error and fail-closed states

Stable error codes:

- `invalid_request`
- `unknown_entity`
- `ambiguous_roster`
- `illegal_draft_state`
- `patch_conflict`
- `missing_required_input`
- `stale_context`
- `model_not_promoted`
- `calibration_not_approved`
- `prefix_calibration_not_approved`
- `semantic_validation_failed`
- `artifact_hash_mismatch`
- `schema_mismatch`
- `prediction_time_violation`
- `source_access_blocked`
- `internal_error`

Errors disclose no credentials, private paths, raw licensed payloads, or stack
traces. Retryability and a public-safe message are explicit.

## 7. Prediction provenance

Every score/rating points to a provenance object conforming to
[prediction-provenance.schema.json](contracts/prediction-provenance.schema.json).
It records mode (`state_snapshot`, `current_analysis`, `forecast`,
`forecast_simulation`, or `hindsight`), input snapshot, estimator and
calibration IDs, event start if applicable, fallback levels, freshness checks,
OOD flags, and output hash. Ratings and tier lists use `state_snapshot`;
current/ad-hoc sandbox scoring uses `current_analysis`; `forecast_simulation`
is reserved for historical availability replay.

The partial schema is a discriminated union. The canonical branch has
`analysis_kind=partial_draft_score` with `status=ok|unavailable`. Every
successful canonical partial response additionally records
`partial_probability_calibration`: exact search policy and method, explicit
temperature or null, search artifact hash, transform hash, prefix-calibration
report hash, prefix stratum, open probability domain, monotonicity and
complement-symmetry proof, and the stratum-specific wording decision. The
response is unavailable unless that decision is true.
The partial model manifest carries the same policy/temperature/artifact/
transform identity, approved strata, and a terminal-consistency report.

The noncanonical branch has
`analysis_kind=archetype_extrapolation`, `status=research_only`, ordinal
recommendation groups, explicit support-gap/fallback and wide-uncertainty
labels, and no canonical score payload. It is returned only from the dedicated
research route. It forbids 0–100, probability/advantage fields, calibrated
transform/provenance, and Reliability, so no client can mistake model-prior
support for validated policy evidence. Its `identity_mode` may be `neutral` or
`contextual`; contextual research requires exact fresh roster/provenance and
forbids unsupported exact player-champion residuals, allowing only general
player state and broad archetype-fit priors.

A forecast record is immutable. Hindsight links to but cannot replace it.

## 8. TypeScript/runtime boundary

L10 generates TypeScript types and validators from the schemas into
`apps/scryglass/src/model-v2/generated/`. UI code consumes discriminated unions
on `analysis_kind` and `status`; accessing a score without both the canonical
branch and `status === "ok"` is a type and lint failure.

The browser does not recreate calibration, infer missing units, or synthesize a
fallback. It displays server-provided labels and values. If local scoring is
required for sandbox latency, the compact runtime is registry-addressed and
must pass the same golden replay suite as Python.

## 9. Numerical parity

Golden fixtures cover:

- swapped sides and transformed action order;
- every partial prefix/slot stratum and its exact policy, temperature, search
  artifact, and transform;
- partial-terminal equality with canonical Terminal Draft Score;
- permuted input arrays with fixed role mapping;
- new champion archetype fallback;
- sparse exact interactions;
- contextual equalization;
- flex role sets;
- missing/stale context;
- patch fallback;
- every error code; and
- ledger reconciliation.

Tolerance is derived from the artifact's declared numeric representation and
must be tighter than the smallest public display resolution. Exact complement
and schema invariants are checked before rounding.

## 10. Authentication and caching

Public/authenticated responses share schemas and estimands. Authorization
controls only rows, candidate breadth, and artifacts explicitly approved by the
publication matrix and post-C4 user decision—not mathematical meaning.
Authentication does not expose private weights or licensed rows by implication.

Cache keys include route, schema/model version, `as_of` snapshot, identity mode,
all stable entity/state IDs, and authorization class. A neutral response cannot
satisfy a contextual cache key. Stale-while-revalidate may be used only within
the source freshness contract; after it expires the result is unavailable.
