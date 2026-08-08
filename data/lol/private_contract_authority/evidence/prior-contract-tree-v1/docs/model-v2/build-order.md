# Build order

This order is binding for the initial cycle. It is structured so each builder
has one owned boundary and receives versioned upstream artifacts rather than
inventing semantics.

## Agent topology

- **S0 — one GPT-5.6 Sol xHigh:** definition and contract owner; this pack.
- **L1–L12 — one GPT-5.3 Codex Spark xHigh per structure:** implementation owners below.
- **S∞ — one GPT-5.6 Sol Ultra:** final adversarial reviewer and remand owner.

Do not silently substitute a model class. If the requested class is unavailable,
orchestration pauses or receives explicit user approval for a substitute. A
remand returns to the same owner; it does not create an unowned repair agent.

No owner commits, pushes, deploys, or edits another owner's files without the
explicit release instruction and allowlist. All builders preserve the dirty
worktree and existing user changes. Every candidate records its canonical
`source_tree_sha256` even when no commit exists; promotion requires a commit
containing that exact source tree.

## Proposed implementation namespace

V2 is isolated until promotion:

```text
lol_kills/v2/
data/lol/v2/
tests/model_v2/
apps/lol-atlas/src/model-v2/
apps/lol-atlas/src/app/api/v2/
apps/lol-atlas/src/app/(model-v2)/
apps/lol-atlas/src/components/model-v2/
apps/lol-atlas/data/model-v2/
```

Existing v1 files are read-only baselines until the migration checkpoint.
Builders do not patch legacy engines in place.

## Wave 0 — contract freeze

### S0: definition and creation

**Owns:** `docs/model-v2/**` only.

**Done when:**

- all prose and schemas agree;
- every public number has an interpretation and conditioning set;
- neutral/contextual plus state-snapshot/current-analysis/forecast/
  forecast-simulation/hindsight boundaries are explicit;
- evaluation and promotion are benchmark-driven;
- schemas validate under Draft 2020-12; and
- S0 records current-repo conflicts without editing production.

**Checkpoint C0:** S0 pack accepted by root orchestration. Downstream builders
pin its tree hash and may not reinterpret it.

## Wave 1 — foundations (parallel)

### L1: data, provenance, and publication

**Owns:**

- `lol_kills/v2/data/**`
- `lol_kills/v2/provenance/**`
- `data/lol/v2/snapshots/**`
- `tests/model_v2/data/**`

**Inputs:** S0 contracts, existing source adapters read-only.

**Deliverables:**

- stable identity/crosswalk registry;
- availability-time warehouse views;
- exact roster, patch, league, protocol, and series resolvers;
- separate season/calendar fields and canonical-side/game-side/draft-order
  mappings with source/availability;
- immutable source/training snapshot manifests;
- canonical source-tree allowlists/digests, with candidate commits optional and
  promoted commits mandatory;
- source-by-source publication matrix;
- public artifact allowlist generator; and
- data-quality/freshness reports.

**Definition of Done:** all L1 gates in
[acceptance-gates.md](acceptance-gates.md), including no four-hour series
grouping and no credentials in fixtures.

### L2: independent evaluation harness

**Owns:**

- `lol_kills/v2/evaluation/**`
- `tests/model_v2/evaluation/**`
- `data/lol/v2/evaluation/**`

**Inputs:** S0 evaluation contract; synthetic fixtures initially, L1 snapshot
interface when ready.

**Deliverables:**

- immutable split registry;
- rolling and sealed holdout runner;
- leakage sentinels;
- structural and semantic negative-fixture runner for all five outputs;
- evidence-diagnostic candidate comparison;
- frozen total Reliability context/OOD-to-stratum mapping and replay;
- series-preserving multiway/hierarchical comparison across participant/team
  and tournament/time dependence, with small-cluster correction;
- draft-order/game-side support, positivity, rank, conditioning, and
  confounding plus admissible-decomposition sensitivity diagnostics;
- prospective/on-policy and sequential off-policy partial-policy evaluation,
  including positivity and effective-sample-size/weight diagnostics;
- end-to-end sealed replay from immutable raw snapshot through serving;
- calibration/coverage suite;
- baseline adapters; and
- promotion/rollback report generator.

**Definition of Done:** L2 can reject a deliberately leaky model, a fake
holdout, a mismatched serving transform, map-independent standard errors,
schema-valid semantic contradictions, and an order/side coefficient inferred
from a rank-deficient or collinear design. It also rejects naive historical
calibration of a different served partial policy and any sealed report assembled
from independently passing but post-opening-swapped components.

### L3: champion ontology and archetype priors

**Owns:**

- `lol_kills/v2/champions/**`
- `data/lol/v2/champions/**`
- `tests/model_v2/champions/**`

**Inputs:** S0 champion contract; official patch/kit sources allowed by L1
matrix.

**Deliverables:**

- versioned interpretable multi-label ontology;
- champion/role/patch representation;
- human review trail and source links;
- new/zero-play prior generator; and
- masked/new-champion evaluation fixtures.

**Definition of Done:** Ziggs/Xerath/Vel'Koz-like archetype transfer is
structurally possible without an empirical residual; ontology absence is an
explicit fallback; tier-list eligibility remains false without actual play.

**Checkpoint C1:** L1 snapshot/IDs, L2 benchmark protocol, and L3 ontology schema
are frozen together. Hashes become inputs to Wave 2.

## Wave 2 — independent mathematical cores (parallel)

### L4: dynamic Player Rating

**Owns:**

- `lol_kills/v2/ratings/player/**`
- `data/lol/v2/models/player/**`
- `tests/model_v2/ratings/player/**`

**Inputs:** C1 L1 snapshot and L2 harness.

**Deliverables:**

- dynamic Bayesian player state model;
- role-normalized auxiliary-channel registry;
- endogenous-resource joint/no-resource/lagged-policy candidates with temporal
  ordering and player-policy double-count sensitivity;
- forecast-time update/replay;
- distinct league-scoped and eligible bridged-global posterior means/95%
  intervals on the required 1500/400 display contract;
- frozen role-specific reference-policy scaling with player-to-roster
  replacement-equality fixtures;
- evidence, reliability, stability inputs; and
- player rating artifact/schema conformance.

**Definition of Done:** transfers carry player history; future outcomes cannot
change earlier states; supports are not penalized for low farm; decay candidates
including no-reset, boundary shock, and reset are compared rather than named by
fiat; League Rating never enters individual skill; plug-in Elo expectation is
separate from posterior-predictive probability; and the reference-policy
replacement equality holds in every role. Weak resource channels are excluded
from Player Rating or routed to policy only.

### L6: champion and composition interactions

**Owns:**

- `lol_kills/v2/draft/interactions/**`
- `data/lol/v2/models/draft-interactions/**`
- `tests/model_v2/draft/interactions/**`

**Inputs:** C1 L1, L2, L3.

**Deliverables:**

- champion main, ally, all-25-enemy, whole-team, and cross-team terms;
- hierarchical patch/league/role pooling;
- archetype residual transfer;
- frozen functional-ANOVA centering/orthogonality projections;
- legal-support rank/conditioning, co-occurrence, posterior-dependence, and
  source-removal diagnostics with grouped-residual fallback;
- posterior draws/approximation; and
- exact term-level attribution primitives.

**Definition of Done:** antisymmetry and role-input invariance pass by
construction; sparse effects shrink; every champion sees four allies and five
enemies; main/pair/whole residual labels are statistically identified under
legal support or collapse into a supported joint residual; stronger residual
families survive registered ablation.

**Checkpoint C2:** L4 and L6 emit schema-valid, replayable model artifacts and
L2 development-fold reports. No sealed promotion result yet.

## Wave 3 — roster and terminal score

### L5: exact-roster Team Rating, League Rating, and policy

**Owns:**

- `lol_kills/v2/ratings/team/**`
- `data/lol/v2/models/team/**`
- `tests/model_v2/ratings/team/**`

**Inputs:** C2 L4; C1 L1/L2.

**Deliverables:**

- exact-five aggregation;
- team policy learned from resource and non-resource impact;
- shrunken lineup synergy;
- explicit League Rating;
- constrained player-versus-league decomposition with transfer/mobility,
  bridge-rank, posterior-dependence, and reference sensitivity diagnostics;
- regional/global eligibility; and
- Team Rating artifacts.

**Definition of Done:** a roster move changes team identity immediately; a new
roster derives from player histories; no sticky organization or mislabeled
league component remains; ambiguous roster fails closed; and no hypothetical
five is emitted or ranked as Team Rating. Global output widens or fails closed
when player skill and League Rating are not separately identified.

### L7: canonical terminal Draft Score

**Owns:**

- `lol_kills/v2/draft/terminal/**`
- `data/lol/v2/models/draft-terminal/**`
- `tests/model_v2/draft/terminal/**`

**Inputs:** C2 L4/L6 and the stable interfaces from L5.

L7 may build the neutral terminal core in parallel with L5. Contextual
integration and promotion cannot complete until L5 emits exact-roster policy
artifacts.

**Deliverables:**

- one terminal estimator;
- neutral and equalized contextual modes;
- exact ledger and uncertainty propagation;
- calibrated 0–100 transform;
- Python/artifact replay; and
- migration comparison against all current draft engines.

**Definition of Done:** baseline strength is exactly excluded from contextual
Draft Score; requested stale context does not return neutral; side swap
complements exactly; ledger reconciles; any empirical order term passes the
identifiability audit or is exactly zero/unavailable by convention; collinear
combined-effect sensitivity is immaterial or the affected endpoint is
unavailable; L2 approves probability wording for the exact served transform.

**Checkpoint C3:** L5 + L7 joint replay produces Team Rating and Draft Score from
one as-of snapshot. L2 opens only development outer folds. Interface shapes are
frozen for Wave 4.

## Wave 4 — products of the terminal model (parallel build, ordered integration)

### L8: partial draft graph, flex, and recommendations

**Owns:**

- `lol_kills/v2/draft/partial/**`
- `data/lol/v2/models/draft-partial/**`
- `tests/model_v2/draft/partial/**`

**Inputs:** C3 L7 terminal score; L3 ontology; L5 context.

**Deliverables:** canonical graph/state IDs, legal protocol engine, explicit
role sets, soft-minimax candidates, committed value, signed strategic-response
adjustment, flex value, search
coverage, exact policy/temperature/transform manifests, prefix/slot calibration
artifacts, terminal-consistency fixtures, canonical recommendations/best
responses, and a typed noncanonical Archetype extrapolation response.

**Definition of Done:** legal transpositions agree; flex sets can change;
sampled/pruned results disclose approximation; every published probability has
finite open support and prefix/slot approval for the exact served
policy/temperature/transform; failed canonical strata return unavailable; only
the separate research response may rank unsupported zero-play actions, with no
0–100, probability/advantage, or Reliability; canonical recommendations show
posterior change and downside only in approved strata; any policy unlike
observed behavior has prospective on-policy evidence or valid sequential OPE
rather than naive replay; the strategic-response adjustment is signed and
equals strategic minus committed value under the same baseline policy; and
terminal partial evaluation exactly equals canonical Terminal Draft Score for
identical inputs/model version.

### L9: role-specific league-current-patch tier lists

**Owns:**

- `lol_kills/v2/tierlists/**`
- `data/lol/v2/tierlists/**`
- `tests/model_v2/tierlists/**`

**Inputs:** C3 L7; L1 current patch/appearances; L5 strength standardization.

**Deliverables:** incremental Tier Value, counterability, the nested
future-outcome adapter that substitutes pre-event TV rows, one league-patch-role
artifact, after-match refresh, and played-only membership.

**Definition of Done:** no raw-win-rate target, no zero-play row, no mixed patch
or league; counterability is nonnegative response-specific lower-tail regret;
and its weight earns future proper-score/calibration gain with legal support or
is zero while counterability remains descriptive.

### L10: canonical model registry and serving

**Owns:**

- `lol_kills/v2/registry/**`
- `apps/lol-atlas/src/model-v2/**`
- `apps/lol-atlas/src/app/api/v2/**`
- `apps/lol-atlas/data/model-v2/**`
- `tests/model_v2/serving/**`

**Inputs:** C3 interface freeze; L8/L9 final artifacts for integration.

L10 may scaffold generated types, registry, and error handling in parallel.
Endpoint completion waits for L8 and L9.

**Deliverables:** manifest registry, artifact compiler, generated TypeScript,
versioned APIs, shared structural/semantic validator, negative-fixture runner,
golden parity runner, cache keying, and fail-closed errors.

**Definition of Done:** one estimator/version, exact transform hash, Python/API/
TypeScript structural and semantic parity, no legacy cascade, no
secrets/browser training data, and status-safe types that cannot expose
successful values with incomplete/stale provenance.

**Checkpoint C4:** one end-to-end private preview passes all schemas, semantic
fixtures, parity, staleness, and development evaluation. It also records
measured training time, runtime latency, storage, refresh compute/cadence, and
hosting cost.

## User gate U1 — post-C4 scope and publication decision

Before Wave 5, L1/L10/L12 present the C4 cost report, complete publication
matrix, reconstruction/licensing risks, and proposed refresh/access options.
The user explicitly decides:

- refresh budget and cadence;
- public core versus authenticated advanced breadth;
- whether custom what-if rosters are in scope; and
- whether any code, data, artifacts, or weights may be published.

No authorization product work or code/data/weight publication proceeds by
default. Authentication never implies private-weight or licensed-row exposure.
After U1, L2 may open the sealed suite for the approved scope. A passing sealed
decision locks an immutable candidate for surface integration; it is not yet a
promoted model.

## Wave 5 — surfaces and access (parallel after C4 + U1 + sealed pass)

### L11: public product surfaces

**Owns:**

- `apps/lol-atlas/src/app/(model-v2)/**` excluding article/auth routes
- `apps/lol-atlas/src/components/model-v2/**` excluding article/auth components
- `tests/model_v2/ui/**`

**Inputs:** version-pinned L10 candidate APIs with a passing sealed decision.
No UI may read a mutable development channel or call the candidate promoted.

**Deliverables:** Player Rating, Team Rating, match explorer, sandbox, and
role-specific tier-list interfaces.

**Definition of Done:** principal numbers are concise; exact roster/scope/patch/
as-of/status are visible; provenance modes cannot be confused; unavailable
remains unavailable; accessibility and responsive checks pass; recorded
fan/journalist/analyst and analyst/pro-player reviews demonstrate that the
visuals communicate the numbers without explanatory clutter; and the approved
editorial typography avoids generic dashboard/AI-default presentation.

### L12: articles, authentication, access, and self-healing

**Owns:**

- v2 article routes/components
- v2 authentication/authorization middleware and policy adapters
- article quantitative-and-mechanics-insert/version registry
- `tests/model_v2/access/**`

**Inputs:** C4 L1 publication matrix, the version-pinned sealed-pass L10
candidate registry, U1, and L11 surface primitives.

**Deliverables:** selected-author workflow, user-approved public/authenticated
artifact policy, immutable article claim revisions, safe typed quantitative and
source-backed patch-mechanics inserts, and access tests.

**Definition of Done:** credentials stay server-side; auth changes approved
breadth not estimands; source and U1 decisions are enforced; self-healing
freezes on numerical-claim or mechanics-semantic change for author review; old
article values/mechanics remain reproducible.

**Checkpoint C5:** release candidate assembled from an explicit allowlist in a
clean worktree. Full Gate A–F repository, artifact, browser, accessibility,
auth, and reproduction checks pass. Only now is the candidate eligible for the
final promotion review.

## Final review — S∞ Sol Ultra

S∞ receives:

- S0 contract hash;
- owner-by-owner diff/allowlist;
- all C0–C5 evidence;
- sealed L2 decision report;
- U1 user decision record;
- public/private matrix;
- golden parity results;
- browser captures/accessibility results; and
- unresolved research register.

S∞ returns exactly one outcome:

- `ACCEPT`: all binding gates pass and the immutable candidate is eligible for
  promotion;
- `REMAND(owner, gate, evidence)`: bounded corrective work; or
- `BLOCK(contract_or_evidence_gap)`: cannot be repaired without a new S0
  contract/version or user decision.

### Remand routing

| Finding | Route |
|---|---|
| Ambiguous/contradictory semantics | S0, then affected owners |
| Identity, time, roster, patch, source, publication | L1 |
| Leakage, metrics, calibration, coverage, benchmark | L2 |
| Ontology/archetype prior | L3 |
| Player dynamics/individual channels | L4 |
| Team policy/roster/League Rating | L5 |
| Composition interactions/antisymmetry | L6 |
| Terminal score/equalization/ledger | L7 |
| Partial search/flex/recommendations | L8 |
| Tier membership/value/counterability | L9 |
| Registry/schema/API/parity/cache | L10 |
| Public UI/copy/accessibility | L11 |
| Articles/auth/public-private enforcement | L12 |

The same owner repairs only its files, reruns its Definition of Done, and sends
new evidence to S∞. Cross-owner remands list dependency order. S∞ re-reviews the
remanded gates plus integration consequences; it does not rewrite owner code.

## Release authorization

`ACCEPT` means technically releasable, not permission to release. Commit, push,
PR, migration, and production deployment each require the user's explicit
authorization and an exact file allowlist. Live remains outside the initial
cycle; any future live phase requires a new contract and explicit approval.
