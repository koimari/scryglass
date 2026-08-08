# Draft Score research and execution decision

Status: design decision; no production or public predictive authority
As of: 2026-07-31

## Decision

Draft Score should proceed as a dedicated L7/C3 Scryglass goal, not as a run of
Karpathy's `autoresearch` framework. A bounded, Scryglass-specific candidate
harness may be used inside the goal, but only on preregistered development
folds. The Karpathy loop is designed for five-minute single-GPU language-model
training with `val_bpb` as its objective; that metric, data contract, and
iteration protocol do not evaluate a chronological esports probability model.

The active deliverable is one canonical neutral terminal estimator. It accepts
the complete pick/ban history for legal-input validation, while the fitted
neutral coefficient model describes the final five-role picks; it does not
require player names or a roster feed. Partial-draft recommendations,
strategic policy claims, and contextual roster output are separate products
and remain unavailable until they earn their own evidence.

## Literature findings

SciSpace semantic search was used to surface relevant work; the links below
point to primary papers or official publication records.

- [Which Heroes to Pick?](https://arxiv.org/abs/2012.10171) treats MOBA drafting
  as a sequential combinatorial problem and uses tree search. It is relevant to
  the later partial-draft graph, not a reason to expose a terminal probability
  without a serving and calibration contract.
- [NeuralAC](https://ojs.aaai.org/index.php/AAAI/article/view/16528) separates
  within-team cooperation from between-team competition. These are candidate
  interaction families for registered ablations, not a license to treat a
  learned interaction table as causal champion value.
- [DraftComPromise](https://publica.fraunhofer.de/entities/publication/02b027c3-2ff0-4f59-92df-4db076e4c52d)
  reports that draft tools can be close while uncertainty remains material.
  This supports reporting calibration, intervals, support, and unavailable
  states rather than presenting a tool ranking as scientific authority.
- [The SIDO Performance Model](https://arxiv.org/abs/2403.04873) supports
  hierarchical partial pooling and explicit assessment of discrimination,
  independence, and stability for LoL performance measures. It is useful for
  context and player-strength separation, but it does not validate a draft
  outcome estimator.
- [SCOPE](https://ojs.aaai.org/index.php/AIIDE/article/view/5233) motivates
  systematic parameter selection for esports ratings using calibration and
  log loss, rather than one-off tuning against accuracy.
- [Model Assessment and Selection under Temporal Distribution Shift](https://arxiv.org/abs/2402.08672)
  supports rolling-window assessment when the data-generating process changes.
  [Calibration of Time-Series Forecasting](https://arxiv.org/abs/2310.14838)
  reinforces that calibration can drift with context; an adapter cannot be
  promoted without an authorized source and a new locked evaluation.
- [Machine Learning Methods for Predicting League of Legends Game Outcome](https://doi.org/10.1109/TG.2022.3153086)
  confirms that pregame LoL prediction is feasible, but model feasibility is
  not evidence of an identified draft effect or a publishable probability.

## Scryglass design consequence

The canonical terminal path should be evaluated in this order:

1. `M0`: a legal, role-aware additive draft baseline with regularized
   pre-event team-strength adjustment; the served neutral score sets that
   nuisance baseline to zero.
2. Registered interaction candidates: allied cooperation, cross-team
   counter interaction, and any composition residual, each with support and
   identifiability gates.
3. A single frozen served transform, calibrated only after fitting and before
   the untouched outer interval. The public score is exactly `100 * p_blue`
   and red is its exact complement.
4. Python, serialized-artifact, and TypeScript replay of the same ledger,
   model version, as-of cutoff, and hashes.

Primary evaluation is chronological and dependence-clustered for the neutral
model. It must include future-patch, league, international, and sparse/new
champion stress tests where support permits; roster-change testing applies
only when identity terms are added. The target is a pre-map
descriptive association: the product must not call it a causal draft effect,
recommendation, betting probability, or policy value.

## Promotion boundary

The existing L7/C3 contract remains binding. Public neutral `status=ok`
requires legal input, pre-event draft availability, complete provenance,
approved calibration, replay parity, and the independent reliability/promotion
record. Contextual mode additionally requires the G1 source bundle: an
authorized pre-event roster payload, source/update and retrieval timestamps,
exact starters and roles, rights, and a verifiable content hash. Missing,
stale, ambiguous, or retrospective context remains `unavailable`; it does not
block neutral scoring.

Until those gates pass, the current public Draft Score gate should stay closed.

## Current implementation status

The neutral L7 development slice now has one canonical Python/TypeScript
replay path, an exact-byte refit coefficient artifact, a preregistered
chronological candidate set, legal terminal-input validation, side-swap
complement tests, ledger reconciliation, and schema-valid unavailable and
success adapters. The current Oracle's Elixir snapshot contributes 16,324
complete five-role maps across 37 canonical patches, including 409 MSI/EWC
rows. An outcome-free dependence proxy covers 6,194 maps; the remaining
10,130 maps are deliberately kept as single-game clusters because an
authoritative series ID is unavailable. Three outer diagnostics select the
registered role-additive baseline, but the result and the new neutral artifact
remain development-only: participant-level IDs, source-time replay, rights,
independent reliability review, and promotion authority are still absent.
The fitted development baseline adjusts for a deterministic pre-event team
Elo nuisance signal, updated only after each dependence cluster; that signal
is not serialized into the served neutral artifact. Role/champion terms with
fewer than ten fitting-slice map appearances are excluded, and sparse/new
champions remain a named diagnostic rather than being treated as proven neutral
effects.
The contextual boundary now also has a strict G1 payload contract requiring
pre-event availability, exact five-role starters for both sides, reviewed
rights, and a hash of the exact source bytes; a verified roster still cannot
publish until the contextual model itself is independently validated.
Oracle's Elixir remains the Draft Score baseline. GRID is a private source
candidate only within a defined competition/date cohort and only after every
included game has an exact hash-verified record containing the complete
pre-draft picks, roles, side, patch, and result, with zero identity, sequence,
or leakage failures. GRID must match or improve OE on pre-declared held-out
validation and calibration checks, and a second replay must reproduce the same
data and model hashes. A passing gate makes GRID primary only for that cohort;
OE remains the public reproducibility benchmark. A failing gate reports the
exact missing or invalid records and leaves OE active. The gate verifies the
exact source payload bytes, binds held-out result hashes, and checks that the
replay data/model hashes are hashes of the verified manifest rather than merely
equal caller-provided strings.
Serving authorization is additionally bound to the exact L2 contract,
candidate registry, evaluation-summary, and independently issued authority-
record bytes, so a manually assembled receipt cannot unlock the endpoint.
Public rendering also requires an independent protocol validator for the
source-specific pick/ban order. The canonical `/api/v2/draft/score` route now
checks that promotion bundle dynamically: it remains closed with the current
artifacts, but a future valid authority record can open the canonical route
without a code-only switch. Legacy exploratory routes remain closed. A
complete request that is still unauthorized receives the full schema-shaped
unavailable result with the missing authority fields; an empty or malformed
request receives only the public-safe service error.

Current status is reported plainly:

- Public MVP lane: live for the non-Draft-Score features.
- Draft Score validation: not accepted for public use.
- Direction: the estimator mechanics are in place; independent validation and
  the required source authority are still blocking publication.

The remaining neutral blockers are the independent L2 chronological
evaluation and reliability record, authoritative source-time and
series-grouped replay, an approved served calibration transform, and the
migration comparison against current engines. Python/artifact/TypeScript
replay parity is locally verified but still needs to be bound into the
independent L2 evidence record. The G1 source bundle is a contextual-only
requirement; no speculative roster reconstruction is part of the neutral lane.
