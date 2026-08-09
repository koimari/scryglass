# Evaluation contract

L2 is independent from model builders. Model owners may supply candidate
artifacts and documented expectations; they may not change split assignments,
evaluation code, baselines, or pass/fail interpretation after seeing sealed
holdout results.

## 1. Evaluation registry

Before any final holdout is opened, L2 freezes:

- source/training snapshot hashes;
- entity, series, patch, league, and roster crosswalk versions;
- all development, calibration, and holdout assignments;
- primary and secondary metrics by output;
- baseline artifacts and exact served transforms;
- subgroup and missingness analyses;
- bootstrap seeds and cluster unit;
- candidate comparison and non-inferiority rules;
- preregistered meaningful-improvement threshold and uncertainty rule for any
  SOTA wording;
- interval-coverage procedure;
- total output-context/OOD-to-Reliability-stratum mapping, including
  no-match=`unrated`;
- semantic-validator fixture set and expected failures;
- frozen reference populations plus centering, orthogonality, and
  player-versus-league identification constraints;
- the complete end-to-end candidate pipeline from source rows through feature
  reconstruction, state inference, calibration, serialization, and serving;
- numerical parity tolerance; and
- allowed reasons to invalidate and rerun an evaluation.

The registry is content-addressed in the model manifest. A changed split or
metric creates a new benchmark version and leaves the old report intact.

## 2. Split design

### Development folds

Use rolling-origin folds. For each fold:

1. fit only on observations available before the training cutoff;
2. select hyperparameters on a later but pre-test validation interval;
3. fit calibration on its own later calibration interval; and
4. evaluate on the next untouched interval.

No random map-level K-fold split is valid. Maps from the same series remain in
one partition. Player, roster, patch, and champion histories are reconstructed
as of each fold rather than joined from a final snapshot.

Every data-informed step is refit inside each outer fold: feature transforms,
dynamic states, hierarchical priors, reference populations, ontology
components learned from outcomes, evidence/reliability thresholds, behavior
policies, search temperature, counterability weight, calibration, and settled
resolution. Only predeclared source-backed constants may be reused. A
full-data upstream fit followed by downstream cross-validation is leakage.

### Sealed holdouts

The benchmark includes all of the following:

1. **Temporal holdout** — latest sealed competition block after all development
   folds.
2. **Future-patch holdout** — complete patches absent from fitting, with an
   explicit no-exact-patch fallback.
3. **League holdout** — leave-one-tier-1-league-out fits and tests; the held
   league cannot influence priors through derived target statistics.
4. **International holdout** — complete named international events held out by
   event, with MSI, EWC, and other events kept separate.
5. **Roster-change holdout when identity is modeled** — first observations for
   newly assembled exact rosters. It is not a requirement for the neutral
   champion-draft estimator, which intentionally contains no player or roster
   identity terms.
6. **Sparse/new-champion simulation** — masked champion residuals plus truly new
   champions when available, evaluating archetype transfer.

Every applicable sealed temporal, patch, league, international, roster-change,
and champion assignment is opened once for one promotion decision by running the
preregistered end-to-end serving candidate from its sealed raw snapshot.
Passing isolated component reports is insufficient, and no preprocessing,
state, calibration, transform, or serializer component may be swapped after
any opening. Research after an unsuccessful opening uses a newly sealed future
block or benchmark version with previously unseen labels; it does not tune on
any failed suite.

## 3. Prediction-time and leakage audit

For every evaluated row L2 asserts:

\[
\max(feature.available\_at)<event\_start.
\]

The audit independently reconstructs rosters, ratings, league state, patch,
champion ontology, action order, and calibration version. It tests adversarial
sentinels:

- shuffling future results must not change earlier predictions;
- deleting all post-cutoff data must not change pre-cutoff artifacts;
- hindsight metadata cannot enter forecast tables;
- final roster/team snapshots cannot be joined backward;
- calibration is fit without test labels;
- `status=ok` is impossible with non-complete required inputs, a false required
  freshness check, or missing/stale/conflict provenance;
- current `state_snapshot`/`current_analysis` artifacts cannot masquerade as
  forecasts, and `forecast_simulation` replays historical availability only;
- same-series later maps cannot influence an earlier map forecast unless the
  product explicitly forecasts between maps and the earlier result was then
  available;
- any empirical average draft-order term passes a preregistered
  protocol × order × game-side support/positivity and design-rank audit; and
- where draft order and game side are not separately identified, the report
  labels zero order contribution as a convention and bounds the score under
  admissible combined-effect decompositions; and
- model/runtime artifact hashes match the manifest.

Any failure invalidates the run rather than becoming a caveat.

## 4. Clustered comparison

Every resolved series is an indivisible inner block, but series are not assumed
independent when teams, players, tournaments, or adjacent time blocks recur.
Before outcomes are opened, L2 freezes the coarsest defensible independent block
or a paired multiway/hierarchical resampling design spanning at least series,
participant/team, and tournament/time dependence. Patch is added when it is a
shared shock. Small-cluster corrections and the minimum effective top-level
cluster count are preregistered; inadequate support widens the interval or
blocks inference rather than reverting to map- or series-independent errors.

Unresolved-series maps are excluded from primary inferential comparisons and
from the primary bootstrap. They may appear as descriptive point metrics with
no confidence interval. A sensitivity analysis may include them only under a
preregistered deterministic coarser cluster—such as a fixed
tournament/date/participant-pair key—with its construction and limitations
published. They are never singleton “independent” clusters.

Report the paired distribution of candidate-minus-baseline metrics, its central
95% interval, the number/effective number and size distribution of clusters at
every level, recurrence of participants, and leave-large-cluster sensitivity.
Promotion must be unchanged under the registered higher-level dependence
analysis. Do not present map- or naive series-level standard errors as
independent.

## 5. Metrics

Regional predictive accuracy is the primary product objective. Aggregate
rating/draft comparisons therefore report a macro-average across eligible
tier-1 regional leagues as the primary summary so the largest league cannot
dominate. Named international-event performance is a separate release-critical
stratum: it need not dominate regional selection, but a global-rating candidate
cannot promote with statistically supported material harm there.

### Probabilistic outcomes

Primary: mean negative log likelihood/log loss.
Secondary: Brier score, calibration intercept and slope, reliability diagrams,
sharpness, and discrimination/AUC as descriptive context.

ECE may be reported only with binning and uncertainty disclosed; it is never
the sole calibration gate.

### Player and Team Rating

Ratings have no direct “true rank.” Evaluate:

- pre-series/map log loss and Brier score in the joint outcome model;
- paired improvement from adding dynamic player/exact-roster states to a
  league/time baseline;
- first-roster and roster-transfer forecast performance;
- pairwise ordering accuracy and probabilistic concordance on future results;
- interval behavior and settled-status false certainty;
- rank turnover and posterior movement under data refresh;
- role/league subgroup calibration;
- the role-specific reference-replacement intervention: under the reference
  policy, a player's displayed logit difference equals the exact change in
  roster aggregation with lineup synergy fixed at its zero reference; and
- posterior-predictive expected result integrated over joint uncertainty,
  separately from the plug-in Elo curve at posterior means.

Regional Team Rating is reconstructed only from league-scoped player states;
global Team Rating is reconstructed only from global player states before
League Rating is added. L2 rejects cross-scope substitution. Team-policy and
lineup-synergy interpretation additionally requires within-roster variation,
design-rank/conditioning, posterior-dependence and source-removal diagnostics,
plus separate ablations. If the split is weak, evaluation approves only the
identified total roster state or a simpler pooled/equal-policy,
zero-synergy fallback.

Resource-channel evaluation freezes event ordering and compares a joint
resource-to-performance measurement model against no-resource,
lagged/pre-map-policy, and player/policy double-count or collider-sensitivity
variants. Same-map aggregate resource is never treated as an exogenous control
or evidence of a causal holding-resources-equal effect. A weak channel updates
policy only or is excluded; team outcome remains the player-strength anchor.

League-scoped and bridged-global Player Rating are evaluated separately.
Global-player evaluation requires structurally eligible active players,
leave-league/international bridge tests, transfer continuity, and an ablation
showing that League Rating is controlled as an environment term rather than
added to individual skill. L2 also reports the constrained design rank,
player-mobility support, player/league posterior dependence, and sensitivity to
reference league, bridge sources, and auxiliary-channel league intercepts.
International outcomes alone are not treated as identifying the split. Weak or
disconnected player-versus-league decompositions must widen or fail closed, not
inherit a precise global rank.

Individual auxiliary channels are evaluated out of sample, but KDA, gold, or
another channel is not treated as ground-truth player strength.

Settled-status evaluation includes adversarial inactive, stale, OOD, F5,
wide-interval, below/equal-95%-precision, and unstable cases. Exactly 0.95 is a
failure because the rule is strictly greater than 95%.

### League Rating

Evaluate leave-one-league-out and held-international-event forecasts, bridge
age/strength sensitivity, and posterior coverage in simulation. Report the
effect of removing League Rating from global forecasts. A globally eligible but
weakly connected league must widen rather than receive a precise default.

### Terminal Draft Score

The observed event is map outcome \(y\); there is no observed binary label that
a draft “has the advantage.” Because the map outcome includes roster strength,
naive Draft-Score-versus-outcome calibration is confounded. L2 evaluates the
draft logit inside the joint heldout outcome model with the independently
reconstructed strength and league logit as an offset:

\[
\operatorname{logit}\Pr(y=1)
=\eta_{strength}+\alpha+\beta\eta_D.
\]

Ideal draft calibration is \(\alpha=0,\beta=1\). L2 also evaluates:

- full joint forecast with and without the draft component;
- a preregistered near-equal-strength overlap subset;
- standardized predictions with the strength offset set to zero;
- neutral versus contextual improvement where valid identity exists;
- international-event fixtures use one named competition/meta scope shared by
  both sides and are invariant to swapping either team's domestic-league
  provenance;
- legal-support design-rank/conditioning, posterior-dependence, co-occurrence,
  and source/patch-removal diagnostics for the main/pair/whole composition
  decomposition;
- joint rank/dependence and source-removal diagnostics for player-champion
  \(h\) versus policy \(q\), including deterministic-overlap fixtures;
- side-swap, role-input, and contribution reconciliation invariants; and
- draft-order/game-side support, positivity, design-rank, condition-number, and
  confounding diagnostics; and
- calibration of the **exact serialized serving transform**.

An empirical average draft-order coefficient is eligible only if registered
protocol variation or another preregistered source identifies it separately
from game side. Perfect collinearity forces the coefficient to zero and the
manifest-linked identification status to `unavailable_collinear`; a
zero-centered prior is not identification. Because zero is then a
standardization convention, not an estimated absence of effect, L2 varies the
split of the identified combined side/order effect over its preregistered
admissible set. Material movement in score or calibration makes the affected
Draft Score endpoint unavailable.
Structural action-tree value from the legal action tree is evaluated separately and
does not establish a population-average order effect.

Probability wording for Draft Score is promoted only when the conditional
calibration evidence supports the served transform on untouched folds and the
equal-strength standardization has adequate overlap. Otherwise the canonical
Draft Score endpoint fails closed.

Weak component identification does not necessarily invalidate an identified,
calibrated total Draft Score. It does prohibit separate component claims: L2
requires the smallest supported grouped residual, widened uncertainty, exact
ledger reconciliation/coverage, and `unresolved_collinear` interpretation
status. Prior separation or Shapley allocation is not a pass.

### Partial Draft Score

Use time-safe historical action prefixes. Evaluate:

- eventual map-outcome calibration of projected standardized probabilities
  separately by pick slot/prefix stratum, with support/overlap diagnostics;
- the exact served response policy, policy version, temperature, search
  approximation, and prefix probability transform as one indivisible
  calibration target;
- whether evaluation is on-policy/prospective or sequential off-policy;
- normalization and strict positive support of the reference policy over every
  legal action included in soft search;
- committed-value rollout uses that same registered baseline policy, and the
  signed strategic-response adjustment exactly equals strategic minus
  committed value, including negative opponent-to-act fixtures;
- terminal-score prediction from every prefix;
- chosen-action likelihood under the fitted response policy as a diagnostic,
  not proof of optimality;
- value regret against the terminal result and against stronger search;
- search coverage/bounds, transposition correctness, and determinism;
- flex-set value under masked final roles;
- recommendation stability across posterior draws; and
- performance by pick slot, side, patch, and league.

The search model must beat or match simple empirical-completion and hard-greedy
baselines before its strategic-response-adjustment wording is promoted. Probability wording is
approved separately for each registered prefix stratum only when the exact
served policy/temperature/transform calibrates on untouched outcomes with
adequate support. Terminal Draft Score approval is not sufficient. A failed
prefix gate makes Partial Draft Score unavailable in that stratum; it may not
silently serve an index, and recommendations may not carry
calibrated-probability wording.

Historical action-prefix outcomes are on-policy evidence only for the observed
behavior policy. When the served policy differs, L2 requires either prospective
on-policy outcomes or a preregistered sequential off-policy evaluation. The
latter must justify consistency, sequential exchangeability, positivity, and
behavior-policy estimation; report effective sample size, weight
concentration/truncation, and doubly robust or equivalent sensitivity; and fail
closed by prefix stratum when those assumptions or diagnostics are inadequate.
Naive replay of observed continuations cannot calibrate a soft-minimax or other
counterfactual policy.

Zero-play/new-champion actions fail logged-policy positivity unless independent
support exists. L2 may evaluate a separate research-only Archetype
extrapolation view for ordinal usefulness and uncertainty, but it may not
approve a canonical partial payload, 0–100 score, probability/advantage
wording, or Reliability label for that unsupported stratum.

L2 also replays every legal terminal fixture through both paths and requires
the partial evaluator's terminal value and displayed score to equal canonical
Terminal Draft Score for identical inputs and model version.

### Tier lists

Every list is evaluated prospectively: the list after match \(m\) predicts only
later eligible appearances. The primary adapter scores observed future map
outcomes with roster/strength offsets and the exact serialized draft component
using proper scores; model-derived latent residuals are never pseudo-ground
truth. Evaluate calibration of that adapter, counterability ablation, rank
stability, and coverage of new/sparse cells. Pairwise ordering of a heldout
latent role-patch residual may appear only as a simulation/internal diagnostic.
Raw champion win rate is a baseline, not a target.

For each candidate \(\lambda_C\), the prospective adapter directly inserts the
pre-event tier row into the heldout outcome logit:

\[
\eta^{tier}_{g,\lambda_C}
=\eta^{-c}_{g}
+s_g\,\beta_{\lambda_C}
\frac{TV^{pre}_{c,r,L,P,t_g}}{100}.
\]

\(\eta^{-c}_g\) contains only time-safe roster/league and other-draft offsets
and removes the evaluated champion's ordinary contribution; \(s_g\) is
positive for side A and negative for side B. The adapter coefficient,
calibration, row computation, and \(\lambda_C\) are refit inside each outer
fold. L2 compares \(\lambda_C=0\) with registered candidates by future
observed-outcome proper score and calibration on the exact serialized adapter.
If legal common support/overlap or effective dependence-aware cluster support
is inadequate, \(\lambda_C=0\) and counterability remains descriptive.

Counterability tests operate on the response-specific
\(\Delta_c(z,a)\), reproduce the registered lower-tail plausible-response
regret, and assert nonnegativity before applying any validated weight. The
allied-context, response, and reference-champion distributions are common
across champions in the cell and every replacement is legal under bans,
uniqueness, roles, and protocol.

## 6. Baselines

At minimum, freeze:

### Ratings

- constant 50% / league-side frequency;
- classical Elo with no uncertainty;
- static Bradley–Terry;
- current production Dual Elo;
- current hierarchical Bradley–Terry;
- dynamic Glicko/TrueSkill-style paired-comparison baseline;
- a reproducible state-space online skill-rating baseline/reviewed candidate;
- a TrueSkill 2-style auxiliary-performance baseline where its inputs pass the
  availability and leakage audit;
- player-average roster without policy or synergy; and
- exact-roster model without League Rating.

### Draft

- league/patch/side frequency;
- role-aware champion additive model;
- same-role matchup model;
- all-pair ally/enemy model without whole-set residual;
- a factorization-machine sparse-interaction baseline;
- neutral model without archetype transfer;
- neutral model without patch or league deviations;
- contextual model without player-champion fit;
- contextual model without team policy; and
- current production serving estimator(s), evaluated exactly as served.

### Partial and tier list

- empirical legal continuation policy;
- greedy terminal-score search;
- hard minimax where tractable;
- published MOBA MCTS/neural-value/tree-search drafting baselines where their
  protocol and data can be reproduced without leakage;
- raw win rate;
- Elo/strength-controlled additive champion value; and
- incremental value without counterability.

Current conflicting estimators are preserved only as named baselines. They
cannot serve as runtime fallbacks.

## 7. Required ablations

Report paired metrics for removing:

- temporal dynamics/decay;
- individual auxiliary channels;
- team policy and lineup synergy separately;
- League Rating;
- ontology priors and champion residuals separately;
- ally pairs, enemy pairs, whole-team residual, and cross-team residual;
- the registered functional-ANOVA centering/orthogonality projections;
- composition legal-support rank/co-occurrence and contextual \(h\)-versus-\(q\)
  identification;
- patch, league, and role deviations;
- player-champion conditional response;
- exact contextual equalization;
- calibration;
- partial strategic-response adjustment and flex handling; and
- tier counterability.

An ablation result changes interpretation: if a complex component does not
improve registered out-of-sample performance, it is removed or retained only as
an explicitly non-principal research diagnostic.

## 7A. Evidence-diagnostic selection

L2 evaluates posterior displacement/information, precision/interval
contraction, and source/context coverage as separate method families under
R-20. Candidate diagnostics are tested in simulation and rolling replay for
stability, sensitivity to prior scale, interpretability, and whether they add
information beyond games played/popularity. Heldout Reliability is evaluated
separately and cannot be folded into an “evidence confidence” scalar. The
selected methods, units, priors, and any compact-label boundaries are frozen
before the sealed holdout. L2 hashes and replays the total Reliability stratum
mapping; every output context selects exactly one registered stratum or
`unrated`, so runtime cannot choose a favorable comparison after seeing the
output.

## 8. Calibration and 95% coverage

Calibration data is later than fitting data and earlier than test data inside
each outer fold. Candidate transforms include identity, Platt/temperature,
beta calibration, and monotone isotonic calibration where support is adequate.
For Draft Score each candidate must be constrained or symmetrized so
\(g(-z)=1-g(z)\); an unconstrained transform that breaks side-swap
complementarity is ineligible. Every terminal or partial transform must also be
monotone nondecreasing, so calibration cannot reverse the raw-logit ordering.
Every transform consumed by partial search or used for a partial probability
must map to \((0,1)\). A candidate capable of emitting zero or one is ineligible
unless its finite boundary treatment is frozen in the manifest and tested as
part of the exact served transform. The simplest transform not distinguishably
worse under paired outer-fold scoring is preferred.

Coverage is tested in two ways:

1. **Simulation-based calibration** for parameter/posterior computation,
   including rank statistics and known latent ratings/interactions.
2. **Heldout aggregate coverage** for empirical forecast cells and future
   series summaries, using the registered series-preserving, higher-level
   dependence-aware uncertainty design. Individual Bernoulli
   outcomes are not misused as observations of a latent probability interval.

Reports show nominal 95%, empirical coverage, its uncertainty, interval width,
and results by registered strata. “Exact 95%” is public wording only when the
applicable report supports it. Otherwise use “95% model range” and disclose
under/overcoverage.

## 9. Drift and decay research

Candidate relevance laws are evaluated by simulated historical as-of ratings:
fit at past cutoff \(t\), forecast successive windows, and measure loss as the
gap since evidence grows. Compare random walk, mean reversion, piecewise
patch/roster shocks, explicit season shock, calendar-boundary shock, full reset,
and no-reset carry-over. `season_id` and `calendar_year` remain separate.

Select on rolling-origin log loss and calibration, with role/league sensitivity.
Report the predictive horizon where baseline-relative skill becomes
indistinguishable, not a universal hand-picked half-life.

## 10. Promotion rule

A candidate is promotable only if:

1. the registered sealed holdout was actually opened exactly as preregistered;
2. its promotion decision is `pass`;
3. the same frozen end-to-end pipeline that consumed the sealed raw snapshot
   produced every evaluated state, calibrated output, and serialized serving
   result, with no post-opening component substitution;
4. every Gate A–F invariant, leakage, schema, semantic, lineage,
   authorization-boundary, parity, and fail-closed test passes;
5. the paired registered dependence-aware comparison shows superiority in the primary
   proper score, or its preregistered one-sided confidence bound demonstrates
   non-inferiority within a margin derived and frozen from the production
   baseline's own refit/refresh variation;
6. any non-inferiority promotion has a registered secondary benefit such as
   materially better calibration, coverage, sparse-cell performance, or
   interpretability, supported by the paired comparison;
7. every registered critical league, patch, side, roster-change, and
   international stratum independently demonstrates one-sided non-inferiority
   within its frozen harm margin under the preregistered multiplicity procedure;
8. calibration/coverage supports the public interpretation;
9. complexity ablations justify retained components; and
10. the identical serving artifact passes numerical and semantic replay parity.

No arbitrary raw metric cutoff is allowed. Margins derive from baseline
variation or a preregistered user-visible decision resolution and are stored
before the holdout opens. Failure to reject harm is not evidence of
non-inferiority: a small/low-power critical stratum that cannot place the
one-sided bound inside its margin blocks promotion. The registry freezes the
family of critical comparisons and its family-wise or false-discovery control
before any sealed label is opened.

### State-of-the-art wording

“State of the art” requires superiority on the complete registered Scryglass
suite against the strongest reproducible named baselines and any comparable
public benchmark available at decision time, exceeding the preregistered
meaningful-improvement threshold with its registered uncertainty criterion.
Only a promoted candidate whose sealed report passes may set SOTA wording true.
A candidate, failed/blocked model, or a model selected by
non-inferiority-plus-interpretability cannot claim SOTA. The claim names the
dataset, time window, metric, effect and uncertainty, baselines, and
limitations. Architecture alone never qualifies.

## 11. Rollback

Each release keeps the previous promoted manifest. Automatic rollback or
fail-closed serving is triggered by:

- artifact/hash or Python/runtime parity failure;
- stale/missing required inputs;
- invariant or schema failure;
- source correction that invalidates the training snapshot;
- prospective proper-score or calibration degradation beyond the frozen
  baseline-derived control limit;
- interval undercoverage supported by the registered monitoring test; or
- public/private boundary breach.

A rollback moves the registry pointer; it does not mutate versioned artifacts.
If the prior model cannot meet current freshness/data contracts, the output is
unavailable rather than served from stale state.
