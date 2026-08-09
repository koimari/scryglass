# Research register

Open questions remain open until the protocol below resolves them. A builder
may implement candidates, but may not silently choose the production answer.

## Decision protocol

For every item:

1. owner writes a preregistration with candidate set, estimand, data/splits,
   primary proper score, baselines, subgroup risks, and decision rule;
2. L1 freezes the required snapshot and L2 freezes the evaluation;
3. owner fits candidates on development data only;
4. L2 produces paired series-preserving, higher-level dependence-aware evidence
   and ablations;
5. S0 checks semantic compatibility;
6. the decision is stored in the model manifest with evidence hashes; and
7. S∞ may remand unsupported interpretation.

Resolution format:

```yaml
research_id: R-XX
decision: selected candidate or unresolved
scope: leagues/patches/roles/time
snapshot_id: ...
benchmark_version: ...
selected_config_hash: ...
paired_primary_metric: ...
calibration_and_coverage: ...
subgroup_results: ...
ablation_result: ...
limitations: ...
effective_model_version: ...
reviewers: [owner, L2, S0]
```

## Open questions

| ID | Question | Evidence needed | Owner | Depends on | Resolution |
|---|---|---|---|---|---|
| R-01 | How quickly do player and league states lose relevance, including across season/calendar boundaries? | Historical availability replay comparing continuous carry-over/no reset, random walk, mean reversion, explicit season shock, calendar-boundary shock, and reset candidates; rolling log loss/calibration by role/league; horizon where skill over baseline disappears | L4 + L2 | L1 snapshots with separate `season_id`/`calendar_year` | Select dynamics; publish fitted implication, never infer a January reset or permanent carry-over from the public filter |
| R-02 | How should exact-roster team policy weight roles, and can it be separated from lineup synergy? | Joint outcome and auxiliary-channel comparison of equal, league-pooled, and team-varying policy; resource and non-resource ablations; within-roster variation, design rank/conditioning, policy–synergy posterior dependence, source-removal sensitivity, support-specific calibration | L5 + L2 | L4, L1 | Select an identified policy/synergy split; otherwise use pooled/equal policy with zero synergy or expose only total Team Rating |
| R-03 | Which individual-performance and resource channels are defensible? | Availability/leakage and temporal-order audit; joint resource→performance measurement model versus no-resource and lagged/pre-map-policy variants; role-normalized predictive increment, player/policy double-count and collider sensitivity, missingness/drift, expert face review; no causal holding-resources-equal claim and no farm proxy alone | L4/L5 + L2 | L1 | Register identified channels; weak resource channels update policy only or are excluded from Player Rating |
| R-04 | What champion ontology best transfers to unseen champions? | Official-kit sourced labels, inter-rater review, leave-champion-out and true-new-champion performance, ontology-free/learned-embedding ablations | L3/L6 + L2 | L1 | Version ontology and prior; unresolved labels stay broad |
| R-05 | Which full-composition residual is justified and statistically separable on legal support? | Pair hypergraph vs low-rank vs Deep Sets/Set Transformer on sealed patch/league folds; functional-ANOVA gauge constraints; legal-support design rank/conditioning, co-occurrence, posterior dependence and source/patch-removal sensitivity; calibration, coverage, ablation, ledger feasibility | L6 + L2 | L3 | Simplest statistically competitive identified family; collapse weak component splits into a supported joint residual |
| R-06 | What partial-draft search policy models professional responses? | Prefix replay for the observed behavior policy; empirical policy, hard/soft minimax, risk-aware comparison; normalized positive reference policy; regret, chosen-action likelihood, search bounds and stability; prospective on-policy evaluation or sequential OPE with consistency, exchangeability, positivity, behavior-policy, effective-sample-size/weight and doubly robust sensitivity diagnostics; exact policy/temperature/transform calibration by slot; finite-logit and terminal-consistency tests | L8 + L2 | L7 | Freeze policy, temperature/risk, search budget, evaluation regime, open-support monotone transform, and approved prefix strata; naive historical replay cannot validate a different served policy, unsupported strata fail closed, and zero-play extrapolation remains a separate non-probability research view |
| R-07 | How is flex value normalized? | Counterfactual fixed-role vs declared-set prefix evaluation under the same baseline/served policies, legality and role-masking tests, posterior stability; keep it separate from the signed strategic-response adjustment | L8 + L2 | L3, L7 | Freeze reference weighting and public interpretation |
| R-08 | Which circuits and active players/rosters are structurally globally eligible at each date? | Versioned official qualification rules and roster eligibility; taxonomy review; bridge graph diagnostics kept separate | L1 + S0 | source review | Version `structurally_globally_eligible`; structural eligibility is not observed attendance or bridge strength |
| R-09 | How should League Rating evolve between international events? | Dynamic prior/bridge-age sensitivity, leave-event-out calibration and uncertainty, disconnected-league simulation | L5 + L2 | R-01, R-08 | Select evolution; wide interval if unresolved |
| R-10 | How is tier-list counterability defined and weighted? | Response-specific \(\Delta_c(z,a)\); common legal allied/response/reference distributions across champions; plausible response-set protocol; nonnegative lower-tail regret and quantile/tail candidates; nested prospective adapter that substitutes each pre-event \(TV=IV-\lambda_C C\) row for the evaluated champion contribution while other time-safe strength/draft terms are offsets; future observed-outcome proper score/calibration and overlap/support diagnostics | L9 + L2 | L7, L8 | Select legal reference support, tail, and weight; assert nonnegativity; \(\lambda_C=0\) and \(C\) descriptive if no supported gain |
| R-11 | Can exact 95% interval wording be defended? | Simulation-based calibration plus heldout aggregate/series coverage by output/stratum; posterior approximation comparison | L2 | L4–L9 | Approve exact wording per stratum or retain “model range” |
| R-12 | What rating distance counts as empirically resolved? | Series-clustered future pairwise discrimination versus production baseline across scopes; refresh variability | L2 | L4/L5 | Freeze `delta_res` per scope for settled/ties |
| R-13 | Which player-champion conditional structure is identifiable separately from team policy? | Main residual, ally/enemy low-rank response, team policy interaction; frozen ordered/cross-orthogonalization; joint rank, posterior-dependence and deterministic player-champion/policy overlap diagnostics; leave-player/champion/patch/source tests and shrinkage ablations | L4/L6/L7 + L2 | L3 | Select identifiable hierarchy; weak \(h/q\) split exposes only total contextual fit; no volume comfort |
| R-14 | Is an empirical average draft-order effect identifiable separately from game side, and does it add terminal predictive value? | Preregistered protocol/source variation; protocol × order × game-side support/positivity cells; design-matrix rank/condition number; confounding/posterior-correlation diagnostics; source-removal sensitivity; bounds over admissible decompositions of any combined side/order effect; side-transform invariance and heldout conditional calibration | L7/L8 + L2 | L1 | Include only if separately identified and validated; perfect collinearity means coefficient zero by convention and `unavailable_collinear`, with Draft Score unavailable if decomposition sensitivity is material; structural action-tree value remains separately labeled |
| R-15 | Which calibration transforms may serve terminal and partial Draft Score? | Nested identity/temperature/beta/isotonic comparison, equal-strength offset calibration, open-support/finite-boundary audit, monotone-nondecreasing and complement symmetry, prefix/slot calibration of exact search policy and temperature, overlap diagnostics, terminal consistency, runtime parity | L7/L8/L2/L10 | terminal model | Simplest supported exact transforms; terminal approval does not approve prefixes |
| R-16 | What current-patch rule is reliable per league? | Official announcement coverage vs latest completed match, conflict frequency, refresh latency | L1 | source matrix | Freeze source precedence and fail-closed conflicts |
| R-17 | What may be public, authenticated, or private? | Terms/license review per source/artifact, reconstruction risk, cost/monetization plan, credential audit | L1/L12 + user | source owners and C4 preview | Publication matrix is necessary but not sufficient; default private pending review and explicit user publication approval |
| R-18 | Which advanced/authenticated features and refresh budget justify cost? | Measured training/runtime/storage/refresh/hosting cost, publication matrix, and actual analyst need; no change to core estimands | L10/L12 + user | completed C4 private preview | Explicit post-C4 user decision on cadence, access scope, custom what-if rosters, and any code/data/weight publication |
| R-19 | When may article quantitative or patch-mechanics inserts self-heal? | Typed insert schemas, source-backed mechanics semantic signatures, claim tolerance, semantic-diff tests, versioned updates, author workflow review | L12 + S0 | L10 registry | Mechanics semantic change always freezes for author review; no silent prose rewrite |
| R-20 | Which evidence diagnostics are defensible for each output? | Compare posterior displacement/information candidates, interval-contraction/precision candidates, and source/context coverage rules on simulation, stability, interpretability, and sensitivity; keep heldout reliability separate | L2 + output owners | L1 lineage and posterior artifacts | Manifest selected diagnostics/units; no universal normalized evidence scalar and no game-count/popularity proxy |
| R-22 | Do LCC mechanistic atom features add incremental draft value or safe zero-play transfer? | Frozen clustered cohort; bridge-pinned atom family presence per role; incremental log loss/Brier vs `m0-role-additive` and baseline-only; ridge ablations; leave-champion-out and true-new-champion structural transfer check; feature support/positivity; no outcome-calibrated claim from atom presence alone | L7/L3 + L2 | L1, R-04, atom bridge v1 | Select if paired deltas nonpositive and transfer diagnostic passes; otherwise atoms remain descriptive priors only |
| R-21 | Which reference populations and constrained parameterization identify player, League, Team, neutral-composition, and contextual-fit components? | Frozen time-safe reference distributions; role-replacement equality; functional-ANOVA representation constraints plus legal-support rank/co-occurrence; international/transfer/mobility connected-design rank; player/league and \(h/q\) posterior dependence; reference/source-removal sensitivity; ordered/cross-orthogonalization and ledger invariance | L4/L5/L6/L7 + L2/S0 | L1, R-02, R-08, R-13 | Freeze the simplest statistically identified parameterization; constraints/priors alone are insufficient; affected global or component-level interpretations collapse, widen, or fail closed |

## Explicitly closed decisions

These are not research questions in v2:

- ratings use posterior means; ties prefer lower uncertainty;
- league and bridged-global Player Rating are distinct scopes; global individual
  skill excludes League Rating and requires structural eligibility;
- public interval target is 95%;
- Team Rating is exact active main roster, not organization history;
- tier-2/3 teams do not enter global ranks;
- contextual Draft Score holds baseline team strength equal;
- neutral is intentional identity omission, not missing-context fallback;
- Draft Score excludes in-game side advantage and post-draft events;
- international Draft Score uses one named event/meta competition scope shared
  by both sides, never either team's domestic league;
- full draft covers all allies and enemies with exact reconciliation;
- role-specific display scaling preserves one reference-roster replacement
  interpretation, and component labels require frozen centering/orthogonality
  constraints;
- tier lists are role×league×current-patch and played-only;
- forecast, historical forecast simulation, state snapshot, current analysis,
  and hindsight are distinct provenance modes;
- private credentials never enter public artifacts;
- the grubs 24% output is removed; and
- live is excluded from the initial cycle only; any future phase requires a new
  contract and explicit approval.
