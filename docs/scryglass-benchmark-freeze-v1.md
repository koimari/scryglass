# Real-v1 benchmark freeze

Version: 1.4

Status: frozen candidate pending independent acceptance

Final labels read: no

This package freezes every source-independent comparison rule before a real
snapshot is bound. It is a Phase A pre-binding template, not a self-approved
benchmark. Version 1.4 permits no source-bound transition: executable baselines
and `RESOLVED` candidates, slots, secondary authorities, held-out patches, and
pairs are structurally unreachable. The current records remain typed
unavailable or typed unresolved.

The sole future entry is G1's unified
`SEALED_PRE_OUTCOME_INPUT_PLUS_SEPARATELY_OWNED_REPLAY_FROM_RAW_SCORING_DERIVATION`
boundary. It must bind the source-dependent package as one sealed replay, not
mutate records within v1.4, and it cannot silently change the scientific rules
below.

The machine authority is
`data/lol/v2/evaluation/real-v1/contract-manifest.json`.

## Scope and primary comparison

The forecasted event is whether canonical side A wins a professional map.
Maps in one resolved series stay together. Unresolved series are excluded from
the primary comparison and may appear only as descriptive values without an
inferential interval.

The domestic primary population is exactly Tier-1 LEC, LCK, LPL, LCS, and LCP.
MSI, EWC, and other named international events are separate release-critical
strata and never enter a vague domestic or global aggregate.

The primary score is chronological macro-regional log loss:

1. Average map losses equally within each resolved series.
2. Average resolved series equally within each league and chronological fold.
3. Average exactly the five registered leagues equally within each fold.
4. Average registered folds equally.

If one registered league is absent from a fold, the primary result is
unavailable. Every paired difference is candidate minus baseline, so lower is
better. Map losses must be finite and nonnegative; booleans and negative or
nonfinite values invalidate the run.

The four outputs are Player Rating, exact-active-roster Team Rating, terminal
Draft Score, and partial Draft Score. Partial depths are champion-pick depths
0 through 9; depth 10 is terminal. Each partial depth must qualify separately.

## Diagnostics and support

The contract defines Brier score, calibration intercept and slope, a ten-bin
reliability diagram that retains empty bins, sharpness, descriptive macro AUC,
predictive-interval behavior, new-roster transfer, adjacent-refresh rank
stability, sparse or new-champion performance, and the draft increment on a
registered near-equal-strength overlap set.

The frozen future rule for new-roster transfer is an exact secondary projection
of its registered primary rows. It selects every row marked as the first
tournament for a new exact roster and carries a separately derived first-series
sensitivity whose membership is recomputed from the per-roster tournament
series order. The future near-equal-strength draft increment is not authorized
by a stored overlap label. Its membership is independently derived from a
registered strength source measured strictly before series start, an
absolute-difference tolerance, an exclude-missing rule, and an inclusive
boundary rule. An empty derived membership blocks the secondary comparison.
Both secondary authority records are currently `TYPED_UNRESOLVED`.

Critical strata are league, patch, game side, roster change, named
international event, and draft depth. An inferential critical-stratum cell
requires at least 30 resolved series, 30 effective clusters, and both outcomes.
Anything below that floor is descriptive only and cannot satisfy a no-harm or
promotion gate.

Gneiting, Balabdaoui, and Raftery (2007,
DOI: 10.1111/J.1467-9868.2007.00587.X) motivate keeping proper-score,
calibration, and sharpness evidence distinct. That literature motivates the
diagnostic structure; it does not establish that any Scryglass forecast is
calibrated.

## Dependence-aware uncertainty

The analysis row is one paired mean loss difference per resolved series. Its
macro weight is \(1/(F\,5\,n_{fold,league})\), so the estimand is the weighted
sum across equal league-fold cells, never a raw map or series average.

The frozen cluster dimensions are:

- **P:** a 28-day participant component linking series that share a team or
  player;
- **T:** authoritative tournament, with ISO UTC week as the fixed fallback;
  and
- **H:** major.minor patch, used as the required shared-shock sensitivity.

If G1 later materializes source-bound evidence, each WCR input must be an exact,
canonical per-series record. It binds the series, fold, league, candidate,
baseline, output, and stratum IDs; the complete
registered ordered fold inventory; map IDs; candidate and baseline
probabilities; binary outcomes; score kind; the submitted macro weight; P, T,
and H assignments; game-side, roster-change, international-event, and
draft-depth selectors; exact-roster identity; series order within the
exact-roster tournament; the registered pre-outcome strength-source identity;
candidate and baseline pre-outcome strengths; resolution status; input and
both prediction digests; and the shared canonical row-order digest. Input identity is independently
derived from the series, ordered map IDs and outcomes, and every registered
context, roster, ordering, and strength field; each forecast identity is
independently derived from the series, ordered map IDs, and exact probability
vector. Supplied digest labels must match those derivations. The score is derived from the
probabilities and outcomes. The registered fold inventory and league IDs are
used to derive the macro weights internally; a caller-supplied weight that
does not exactly match that derivation blocks the run. Both 0 and 1 must occur
in every inferential critical cell. The registered threshold \(\theta_0\) is
derived from the frozen slot and margin rule, never supplied by a caller.

Execution identity `multiway-wcr-cgm-v1.4-promotion-remand` is a null-imposed wild cluster
restricted bootstrap. For each run, exactly one active dimension \(D\) is the
bootstrap-clustering dimension. At threshold \(\theta_0\), the restricted
residual is \(e_i^0=d_i-\theta_0\). One cluster multiplier is drawn for each
cluster of \(D\), and one pseudo-outcome vector is formed:

\[
d_i^*=\theta_0+v_{D(i)}e_i^0.
\]

The weighted intercept is refit on every draw. Its residuals are then used to
recompute the complete Cameron-Gelbach-Miller inclusion-exclusion covariance
over every nonempty subset of the active dimensions. Primary runs therefore
use P, T, and PT covariance terms even though only P or T supplies the
multiplier field. Patch runs use all seven P, T, H, PT, PH, TH, and PTH terms
while selecting P, T, or H as the single bootstrap dimension. Independent
subset multipliers and signed sums of separately bootstrapped subset scores
are prohibited.

Observed and bootstrap covariance use

\[
V=\sum_{\emptyset\ne S\subseteq A}
(-1)^{|S|+1}\frac{G_S}{G_S-1}\sum_g Q_{S,g}^2.
\]

Every intersection must have at least two clusters. A nonfinite,
nonpositive, or at-most-\(10^{-12}\) variance blocks inference; invalid draws
are neither discarded nor redrawn.

Each run uses exactly 9,999 draws and deterministic seed identity `2026072901`.
The multiplier stream is HMAC-SHA-256 over the active dimensions, selected
bootstrap dimension, replicate, canonical cluster ID, and multiplier law.
Candidate and comparison IDs do not enter the stream, so paired candidates
receive aligned draws. A selected dimension with 30 through 49 effective
clusters uses Webb six-point multipliers; 50 or more uses Rademacher
multipliers. Fewer than 30 blocks that run.

The one-sided p-value uses the finite correction
\((1+\#\{t^*\le t\})/(9999+1)\). The authoritative upper endpoint is found by
inverting this null-imposed lower-tail test with a verified bracket and
deterministic bisection. A Hyndman-Fan type-7 bootstrap-t endpoint is retained
only as a diagnostic and cannot satisfy a gate. Every authoritative endpoint is
an **unadjusted one-sided 95%** endpoint. Holm controls the family of
hypothesis rejections; it does not adjust, relabel, or replace an endpoint. A
promotion member would pass only when its Holm decision and its unadjusted
endpoint both pass the registered threshold.

The primary P,T covariance suite must agree when bootstrapping on P and on T.
The P,T,H patch suite must agree when bootstrapping separately on P, T, and H,
and it must agree with the primary suite. The largest P, T, and H cluster is
then removed one dimension at a time; rows, weights, support, multiplier law,
fits, covariance, families, and decisions are fully recomputed. Every
leave-largest suite must remain available and agree. There is no singleton,
map-independent, quiet-series, or two-way substitute.

Webb (2023, DOI: 10.1111/caje.12661) provides simulation evidence for
six-point weights with few clusters. Menzel (2021,
DOI: 10.3982/ECTA15383) explains why multiway-cluster inference has no
uniformly consistent solution in full generality. The support floors and
agreement checks are conservative Scryglass governance choices, not universal
constants supplied by those papers.

## Margins, multiplicity, and complexity

The margin procedure is frozen but its numbers remain deferred. A registered
development-only bundle must contain at least 30 deterministic refresh/refit
replicas of the same baseline specification on the same evaluation rows. The
margin is the type-7 95th percentile of the absolute paired change in
macro-regional chronological log loss. The bundle binds the development
snapshot, baseline identity, rows, complete replicate payload, procedure,
construction identity, and independent review.

`derive_registered_margin(candidate_id, baseline_id)` is a frozen signature
only. Version 1.4 rejects the source-bound margin path before a bundle can be
used. The zero fallback remains a declarative threshold rule in the frozen
matrix, not a currently executable comparison route, and it does not
manufacture a favorable margin.

Multiplicity uses one-sided Holm step-down control at familywise alpha 0.05.
`compute_registered_holm(candidate_id, family_id)` freezes exact membership;
callers cannot supply or omit comparisons. If G1 materializes a complete
family, raw p-values are ordered by value and immutable slot ID, adjusted
monotonically, and coupled to the registered inverted endpoint. In v1.4 the
typed-unresolved slots block before source-bound evidence can authorize a
result. Missing evidence, incomplete support, disagreement, or family mutation
also blocks the family.

The candidate registry contains exactly two candidates. Each has three complete
candidate-scoped families: `primary:{candidate}`, `harm:{candidate}`, and
`secondary:{candidate}`. There are exactly 2,182 slots: per candidate, 61
primary, 908 harm, and 122 secondary. Primary and secondary slots are strict
superiority against zero. Harm slots use the registered margin when available,
otherwise the zero fallback. Every candidate and every static slot is currently
`TYPED_UNRESOLVED`; an unresolved typed slot remains in its family and blocks
the relevant decision.

The patch critical entry is only the
`patch:each-held-out-major-minor` template. The top-level held-out-patch
inventory is `TYPED_UNRESOLVED`, so the template cannot be used for inference
and cannot be replaced inside v1.4. Exact held-out patch children may be
derived only inside the unified G1 bundle; omission, mutation, or a mixture of
template and children would block that future family.

All 61 baseline records are `TYPED_UNAVAILABLE`, have
`NO_FINAL_LABEL_ACCESS`, and leave source-dependent execution `UNBOUND`. The
`EXECUTABLE_PREBOUND` baseline branch and every `RESOLVED` candidate, slot,
secondary, held-out-patch, and pair branch are structurally unsatisfiable in
v1.4. Adapter entry points, fixtures, self-written digests,
`AUTHORITY_SIDE_REEXECUTION` receipts, and evidence files therefore cannot
authorize a Phase A transition, separately or in combination. Those field
shapes document future G1 requirements only.

The NI-plus-secondary route is unavailable until a separately registered
margin-threshold primary rule exists. Zero-threshold primary results must never
be recycled as evidence for a positive-margin noninferiority claim.

The pair registry contains one global `complexity:global` family with four
explicit Scryglass-v2-minus-simplest-parent comparisons, one per output. Pair
records are currently `TYPED_UNRESOLVED`; pair authority cannot be staged after
a candidate-only transition in v1.4. The future unified G1 record must bind
output and A/B identities, orientation, the registered ordered folds and
observed league IDs, internally derived macro weights, binary outcomes, P/T/H
assignments, critical selectors, canonical aligned rows, derived differences,
both candidate prediction-row digests, the common independently derived
input-row digest, the canonical analysis-row digest, exact-roster identity,
within-tournament series order, pre-outcome strength-source identity and
values, explicit P/T/H and PT/PH/TH/PTH CGM intersection assignments, and the
exact common plan. Candidate swapping, row reordering, recomputed differences,
opaque weights, or plan substitution fails closed.

Complexity sensitivity is not a pooled generic confidence interval. It consists
of 20 separate centered wild-cluster CGM max-t runs: the full primary P/T and
patch P/T/H suites plus every P-, T-, and H-largest-cluster removal suite. Each
full or reduced sample is centered at its own internally derived
macro-weighted estimate. Each run produces a simultaneous two-sided 95% band
across all four registered outputs. The bands and their decision conclusions must all
agree; any unavailable or discordant run yields `NO_WINNER_REMAND` rather than
a complexity winner.

A complexity winner would exist only after the required candidates pass, the
simultaneous interval and no-harm rules resolve, and one unique nondominated
candidate remains. Otherwise the result is `NO_WINNER_REMAND`; a favorable
point estimate is not a tie-break.

## Frozen API surface and closed opening

Version 1.4 freezes these authority-derived names:

```text
verify_authoritative_preflight() -> VerifiedAuthority
compute_registered_holm(candidate_id, family_id) -> HolmReport
compute_registered_pairwise_intervals(pair_family_id) -> PairwiseIntervalReport
consume_bound_opening_permit(permit_raw) -> OpeningReceipt
derive_registered_margin(candidate_id, baseline_id) -> float
```

None accepts caller-provided roots, digests, path lists, candidate lists,
attestations, stores, schemas, or authority objects. Every operation starts
from the fixed production package and repository roots, reopens the
manifest-pinned contract and registries, and verifies raw and semantic digests.
`verify_authoritative_preflight()` can verify that source-independent package.
The family, pair, margin, and permit calls cannot reach source-bound
computation: current typed-unresolved records block, and attempted source-bound
paths fail with `G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED`. The signatures freeze
future semantics; they do not make a v1.4 transition reachable.

Expected reads are a canonical, nonempty, manifest-bound set. Component-wise
descriptor traversal rejects absolute or ambiguous paths, dot components,
symlinked roots, parents, or leaves, nonregular files, and hardlink aliases.
Every expected payload is also checked against the final-label boundary.

The blank permit template is schema-valid and nonauthorizing:
`status=NOT_REQUESTED`, `decision_scope=G9_FINAL_OPENING_ONLY`, and
`authorizing=false`. Version 1.4 cannot consume an approved permit and cannot
write an opening receipt or ledger entry; it rejects before permit parsing or
ledger access. No opening candidate is stored or derived in Phase A.

The preflight is a **cooperative-process integrity boundary only**. It prevents
ordinary caller injection and common file-substitution mistakes, but it is not
cryptographically unforgeable against hostile same-process monkeypatching,
same-account file replacement, or intercepted I/O. Production authority still
requires a separately owned verifier, independently signed artifacts with
keys outside evidence-generator access, and an append-only separately owned
ledger service.

The sole human-authority reuse recorded for the future G1 handoff is
`oe_target_evidence.require_exact_human_authority->representation_rank_private_runner`
with reviewer `KOI_MARI` and approval scope
`private_retrospective_oe_target_v1`. The manifest-bound handoff reopens
`data/lol/v2/models/draft-interactions/oe-private-target-authority.json` at raw
SHA-256
`b1d0a6e37abb9a74dee8689dc19ab54d30fd15516bd4ee454906a075d8f20788`.
Its only approved actions are `model_fit` and `rank_selection`, and the
artifact must retain `final_temporal_holdout_sealed=true`. It excludes
`final_temporal_holdout`, `G9_final_opening`, `promotion`, `publication`, and
`public_claims`. This freeze creates no new human authority, and the existing
approval cannot resolve any v1.4 record.

## Current gate state

G0-103 remains blocked because v1.4 both lacks and prohibits source-bound
candidate adapters, candidate artifacts, comparison evidence, margins, and
pair evidence. The 61 baselines remain typed unavailable; both candidates,
all 2,182 slots, both secondary authorities, the held-out-patch inventory, and
all four pairs remain typed unresolved. The registries do not invent adapters,
fixtures, execution receipts, labels, fits, winners, or publication claims.

G9 and production authority also remain blocked. No final labels have been
opened, and no rating, Draft Score, reliability, probability, SOTA, promotion,
or winner claim is authorized by this freeze.

The only next entry is G1: one unified authority bundle must begin from sealed
pre-outcome inputs and independently replay raw inputs through scoring
derivation under a separately owned boundary. It must materialize baselines,
candidates, exact held-out children, secondary projections, all candidate
families, and the common four-pair sample together; editing v1.4 records or
supplying fixtures, receipts, evidence, or permits cannot substitute for that
replay. The G1 handoff still excludes the final temporal holdout, G9, promotion,
publication, and public claims. Until such a separately implemented boundary
exists, the actionable result is to keep G0-103, G9, and production closed.

## Identity and tool record

The manifest binds exact UTF-8 bytes and sorted compact JSON semantics for all
12 package JSON files. It separately raw-binds
`lol_kills/v2/evaluation/benchmark_contract.py`. The authority-contract digest
covers the six schemas, the benchmark, baseline, and blank-permit artifacts,
plus the executable code. The full manifest identity also covers the
authority, candidate, and pair registries. Whitespace-only changes therefore
break raw authority even when semantics do not change. This document describes
the source-independent v1.4 ceiling only: it is not independent acceptance,
does not open final labels, and does not authorize a model, reliability,
probability, promotion, or winner claim.

SciSpace was used for the cited forecast-evaluation and clustered-inference
literature. Wolfram independently evaluated the frozen type-7 interpolation
oracle: the 95th percentile of `0, 0.001, ..., 0.029` is `0.02755`. The
Academic Writing Toolkit was used for the final logic and prose review. These
tools informed concrete identities and checks; none supplied or authorized
Scryglass evidence.
