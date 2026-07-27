# Scryglass app-wide scientific audit protocol

Version: 1.0
Date: 2026-07-26
Status: mandatory before scientific launch

## Standard of proof

Scryglass cannot honestly promise that an empirical estimate is “absolutely
correct.” No finite observational dataset can prove that. It can promise
something more useful to an expert reviewer:

- the question and estimand are exact;
- source facts and temporal joins are authoritative and reproducible;
- every calculation has an independent oracle or derivation;
- model selection is predeclared and out of sample;
- uncertainty and non-identifiability are explicit;
- simpler and credible state-of-the-art alternatives were benchmarked fairly;
- all public claims trace to immutable evidence;
- known counterexamples and limitations are published;
- an external reviewer can reproduce and challenge the result without private
  context.

“Industry SOTA” means the selected model wins or is statistically
indistinguishable from the best eligible benchmark on predeclared
out-of-sample criteria while satisfying calibration, stability, interpretability,
latency, and failure-safety gates. It does not mean choosing the newest or most
complex paper.

The expanded evidence universe, transfer rules from financial econometrics,
and executable tournament design are defined in
[`CROSS_DOMAIN_SOTA_RESEARCH_PROTOCOL.md`](CROSS_DOMAIN_SOTA_RESEARCH_PROTOCOL.md).

## Current audit surface

The current source tree contains approximately:

| Surface | Count |
|---|---:|
| Inventory files | 141 |
| Python/TypeScript source, test, and audit-tool lines | 44,974 |
| Public page/API/layout entry files | 18 |
| Extracted declared symbols | 1,125 |
| Public page routes | 14 including redirects/dynamic pages |
| Public API routes | 3 |

The deterministic inventory hash for this reviewed baseline is
`d9f88cffda8a88330048f13d238b329bcf7eda991e1ca520a15020b52eef1bc2`.
It is produced by `tools/scientific_audit_inventory.py`; any source change
changes the hash and reopens affected review rows.

The review includes:

- every line reachable from a public route, API, pack build, model fit, article
  generator, or scheduled refresh;
- every static sentence, label, tooltip, formula, table header, default, and
  empty/error state shown publicly;
- every API field, hidden pack field, downloadable file, and model artifact;
- every backend calculation that can feed a release;
- every test that purports to validate one of those calculations;
- every dormant/legacy path until it is either reviewed or mechanically
  quarantined from release.

Visual styling is reviewed by the separate styling task. This protocol owns
semantic labels, chart meaning, accessibility of scientific content, and
functional interactions, but it does not overwrite that task’s visual work.

## The three mandatory ledgers

### 1. Source ledger

One row per reviewable code region:

| Field | Meaning |
|---|---|
| `source_id` | Stable audit identifier |
| `file`, `start_line`, `end_line`, `sha256` | Exact reviewed source |
| `symbols` | Functions/classes/constants covered |
| `reachability` | Public runtime, release generator, research-only, dormant |
| `estimand_ids` | Quantities the region may affect |
| `inputs`, `outputs`, `side_effects` | Technical contract |
| `temporal_inputs` | Any time-sensitive facts or joins |
| `review_status` | Unreviewed, challenged, accepted, quarantined |
| `reviewers` | Primary and independent reviewer |
| `tests` | Exact regression/property/golden checks |

Any source change invalidates acceptance for the changed hash and downstream
calculation/claim rows.

### 2. Calculation ledger

One row per derived quantity, including “small” frontend helpers:

| Field | Meaning |
|---|---|
| `calculation_id` | Stable identifier returned in debug/reproduction output |
| `name`, `formula` | Plain language and mathematical definition |
| `estimand_id` | Question being answered |
| `grain` | Map, series, player-map, roster interval, model release, etc. |
| `units` | Probability, Elo points, seconds, percentage points, count |
| `inputs` | Names, types, units, valid ranges, provenance |
| `denominator` | Explicit population for rates/averages |
| `time_policy` | Effective-at, trained-through, as-of, no-future rule |
| `missing_policy` | Fail-closed behavior |
| `scope_policy` | League, tournament, patch, side, role |
| `uncertainty` | Definition and computation |
| `independent_oracle` | Separate implementation, symbolic derivation, or hand case |
| `invariants` | Bounds, symmetry, conservation, stability, monotonicity |
| `validation_artifact` | Immutable evidence |

Formatting and sorting calculations belong in this ledger when they can change
interpretation: rounding, tied ranks, score-to-format conversion, date windows,
league filters, and query limits are scientific behavior.

### 3. Claim ledger

One row per public assertion:

| Field | Meaning |
|---|---|
| `claim_id` | Stable identifier |
| `route`, `component`, `copy` | Exact public location and words |
| `claim_type` | Source fact, descriptive statistic, estimate, forecast, causal claim, limitation |
| `calculation_ids` | Supporting calculations |
| `source_ids` | Supporting code/data |
| `release_ids` | Frozen evidence |
| `precision_policy` | Rounding and significant digits |
| `uncertainty_copy` | Required language |
| `scope_copy` | Required time/league/patch/roster language |
| `status` | Accepted, needs qualification, unsupported, quarantined |

No visible number or scientific sentence ships without a claim-ledger row.
Repeated text may share a claim only when its context and scope are identical.

## Review passes for every code region

### Pass A — Semantics

- What exact question does this code answer?
- Is its data grain the same as the public claim?
- Are counts named as maps, games, series, players, teams, or observations?
- Does “current” mean current tournament registration, latest observation, or
  latest model state?
- Is raw draft composition kept separate from team/player context?

### Pass B — Data and temporal correctness

- Source authority and retrieval time.
- Identity, alias, rename, roster, league, tournament, patch, and season joins.
- OE/GRID overlap and completion provenance.
- Duplicate, missing, remake, forfeit, and boundary behavior.
- No future data in a historical feature or prediction.
- Explicit inclusion/exclusion population and denominators.

### Pass C — Mathematics

- Formula derived from the stated likelihood/estimand.
- Units and ranges.
- Independent implementation in a separate code path or symbolic system.
- Property/metamorphic tests.
- Numerical stability and deterministic tie handling.
- Correct uncertainty object; no standard-error/posterior/prediction interval
  confusion.

### Pass D — Statistics and model selection

- Train/validation/test split follows time and competition structure.
- Hyperparameter/model selection is nested away from the final test.
- Baselines and credible competing model families are present.
- Ablations identify where performance comes from.
- Calibration, discrimination, proper scores, stability, and slice behavior.
- Effective sample size and dependence/cluster structure.
- Multiple-comparison policy for exploratory research.

### Pass E — Product conclusion

- Does the wording say more than the evidence?
- Is an association presented as causal?
- Is a local utility presented as probability or optimal policy?
- Could current ratings be mistaken for historical forecasts?
- Does the failure state hide uncertainty or missing evidence?
- Is the displayed precision justified?

### Pass F — Reproduction and challenge

- Clean-room rebuild from the immutable public bundle.
- Golden cases and counterexamples.
- All links/hashes/runtime dependencies.
- Reviewer can inspect raw rows behind each headline.
- One-command or notebook reproduction within declared tolerance.

## Independent verification requirements

Every high-impact calculation needs at least two of:

- symbolic derivation or exact enumeration;
- second implementation by a different agent/language;
- property-based/metamorphic tests;
- hand-computed golden examples;
- simulation with known ground truth;
- comparison with an established library;
- external reviewer reproduction.

Launch blockers require three, including an independent implementation.

Examples:

- series score logic: exact enumeration plus property tests plus audited real
  cases;
- Bo3/Bo5 conversion: symbolic derivation, enumeration, side-swap identity;
- player identifiability: design-matrix rank/null-space calculation and
  synthetic recovery;
- calibration: independent recomputation from the prediction ledger and a
  second implementation of score/bin calculations;
- draft antisymmetry: algebraic derivation and randomized blue/red swaps.

## SOTA benchmark protocol

### Team and organization strength

Eligible benchmark families:

1. static Elo and logistic Bradley–Terry;
2. decayed/sequential Elo;
3. Glicko/OpenSkill-style uncertainty-aware baselines;
4. whole-history or dynamic Bradley–Terry;
5. hierarchical state-space models with competition/region structure;
6. roster-aware organization/lineup state models.

Primary selection criteria:

- chronological series-level log loss and Brier score;
- calibration;
- performance across region/event/time slices;
- uncertainty coverage/stability;
- robustness to roster movement and sparse international bridges.

Complexity is accepted only when it improves the predeclared criteria outside
the selection sample. The simple model remains published as a benchmark.

### Player performance and player-derived lineup strength

Eligible benchmark families:

1. shared team/lineup outcome signal with no individual separation;
2. role-adjusted box-score models;
3. regularized adjusted plus-minus;
4. SIDO-style hierarchical contextual contribution models;
5. PandaSkill-style role-specific performance followed by Bayesian skill
   updates;
6. joint hierarchical player, lineup, champion, opponent, and patch models.

Primary criteria are not only match prediction. They include:

- discrimination when lineups change;
- stability through time;
- independence from teammate/team where claimed;
- role fairness and champion-context robustness;
- posterior dependence and shrinkage;
- incremental pre-match prediction after organization state.

A player model that merely predicts wins by rediscovering the team is rejected.

### Match and series win prediction

Eligible benchmark families:

1. calibrated dynamic strength difference;
2. strength plus roster continuity;
3. strength plus complete draft;
4. interpretable generalized additive/hierarchical model;
5. gradient-boosted or interaction/graph model with strict temporal features;
6. a calibrated stack selected on nested rolling-origin validation.

Map and series targets are separate. Series conversion accounts for map-specific
side/draft/roster information when available. A more accurate opaque model may
support a forecast, but it does not replace the interpretable research model
unless explanation fidelity and stability pass.

### Complete-draft strength

Eligible benchmark families:

1. league/patch/side-only base rate;
2. role-aware champion main effects;
3. within-team synergy;
4. all cross-team opposition interactions;
5. regularized low-rank or attention/set interaction models;
6. calibrated ensembles.

Evaluation includes:

- chronological and future-patch log loss/Brier;
- held-out-league transfer;
- calibration;
- blue/red antisymmetry;
- role invariance;
- sparse champion/interaction behavior;
- ablations and explanation reconciliation.

Raw draft cannot use team/player identity. Context is a separate jointly
validated forecast.

### Pro-aware dynamic draft recommendation

This is a sequential decision problem, not a tier list.

The state includes:

- pick/ban order and all locked/unavailable champions;
- declared and uncertain role assignments;
- patch, tournament rules, side, and series state;
- current roster and player champion pools in contextual mode;
- team style/coordination only if learned before the match;
- uncertainty and out-of-distribution status.

The action value integrates:

- likely own future picks;
- likely opponent replies;
- role-flex resolution;
- complete-composition value;
- player comfort and team context in the contextual layer;
- search horizon and approximation.

Candidate policies:

- greedy local counterfactual baseline;
- beam/minimax search;
- Monte Carlo tree search;
- learned opponent/own-pick policies with search;
- risk-sensitive or robust policy under model uncertainty.

Required evaluation:

- legal-action rate;
- state sensitivity;
- held-out final-draft value;
- regret against deeper search on tractable subgames;
- robustness to opponent-policy misspecification;
- ablations for pro context, flex uncertainty, future picks, and replies;
- latency under production constraints;
- qualitative review of champion/game-mechanic plausibility.

Logged pro drafts do not reveal the counterfactual outcome of picks not made.
Offline policy evaluation therefore cannot prove causal optimality without
strong assumptions. Public language must say “model-recommended under this
policy,” not “correct pick,” and must expose alternatives and uncertainty.

Recent evaluation is also a warning against inflated promises:
DraftComPromise found that neither its own real-time recommender nor the four
compared tools established recommendations winning above 50% once margins of
error were considered. BPCoach supports patch-, player-, style-, and
series-aware professional draft planning, but its case studies and user study
are not causal proof of optimal picks. Scryglass should borrow the richer state
definition and the humility of the evaluation, not a SOTA label.

## Patch adaptation

Patch is a modeled time-varying context:

- normalized Riot patch ID and effective competition dates;
- champion rework/system-change indicators where available;
- hierarchical patch effects that shrink toward adjacent/global estimates;
- explicit cold-start behavior for new/reworked champions;
- recency/process drift selected chronologically;
- future-patch holdout and empirical interval coverage;
- hard “unsupported” state when no defensible transfer exists.

The runtime may not silently use a historical patch coefficient while labelling
the output current.

## External expert review package

The package intended for a reviewer such as xPetu contains:

1. a one-page map of estimands and model separation;
2. immutable public data/model release with hashes;
3. source, calculation, and claim ledgers;
4. model cards with fit populations, exclusions, and clocks;
5. leakage audit and temporal split specification;
6. independent recomputation notebook/script;
7. benchmark and ablation tables;
8. calibration plots and raw prediction ledger;
9. mathematical invariants and counterexamples;
10. representative raw OE/GRID/registry rows;
11. Sandbox policy definition, search algorithm, and offline-evaluation limits;
12. known limitations and unresolved decisions;
13. exact commands and expected tolerances.

The review request asks the expert to challenge:

- whether the estimand matches the product label;
- whether the data can identify the claimed quantity;
- whether any temporal or selection leakage remains;
- whether benchmark selection was fair;
- whether uncertainty and calibration support the displayed precision;
- whether draft recommendations adapt for the right reasons;
- whether a simpler model would be more defensible.

Agreement is recorded per claim/calculation, not as a vague endorsement of the
whole application.

## Release gates

The scientific launch is blocked until:

- 100% of source-ledger rows reachable from release are accepted;
- 100% of public calculation and claim rows are accepted;
- dormant/unreviewed code is mechanically excluded from release;
- all launch-blocker calculations have independent implementations;
- SOTA benchmark protocols are frozen before final test evaluation;
- model cards and public bundles reproduce exactly;
- external review has no unresolved launch blocker;
- the production pack and deployed app match the reviewed hashes.

Coverage is measured from hashes and reachability, not from a checklist that can
silently miss new files.

## Primary methodological references

- Dynamic skill: <https://arxiv.org/abs/2308.02414>,
  <https://arxiv.org/abs/1903.07746>.
- Player identifiability/context: <https://arxiv.org/abs/1201.0317>,
  <https://arxiv.org/abs/2403.04873>,
  <https://arxiv.org/abs/2501.10049>.
- Sequential drafting: <https://arxiv.org/abs/2012.10171>,
  <https://arxiv.org/abs/2204.12750>,
  <https://arxiv.org/abs/2311.05912>,
  <https://doi.org/10.1109/GEM61861.2024.10585636>.
- Calibration/proper scoring: <https://arxiv.org/abs/2008.03033>,
  <https://arxiv.org/abs/2203.07835>.
