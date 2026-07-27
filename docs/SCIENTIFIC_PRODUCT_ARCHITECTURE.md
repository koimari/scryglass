# Scryglass scientific product architecture

Version: 1.0
Date: 2026-07-26
Status: controlling specification for the scientific/data rebuild

App-wide review and evidence requirements are defined in
[`APP_WIDE_SCIENTIFIC_AUDIT_PROTOCOL.md`](APP_WIDE_SCIENTIFIC_AUDIT_PROTOCOL.md).
That protocol applies to every public line, backend calculation, API field,
artifact, and conclusion; this document defines what those reviewed surfaces
are allowed to mean.

## Product statement

Scryglass is a non-betting League of Legends research publication. Its job is
to make evidence-backed claims understandable and reproducible.

The articles are the publication. Ratings, Matches, H2H, profiles, and Sandbox
are research instruments that let a reader inspect the evidence, test a
counterfactual, or understand the scope of a claim. They are not independent
scoreboards sharing a convenient “Elo” label.

The product has four rules:

1. one public number answers one declared question;
2. time-sensitive facts carry an effective date and authoritative source;
3. probabilities are frozen before outcomes and validated out of sample; and
4. unverifiable or non-identifiable quantities fail closed.

## The estimand stack

The following quantities must remain separate:

| Estimand ID | Question | Observation grain | May use |
|---|---|---|---|
| `ORG_STATE` | How strong is this organization in this competition at time t? | Canonical series | Pre-series organization state, competition, side/format if validated |
| `LINEUP_STATE` | How strong is this exact five-player lineup at time t? | Canonical map/series with roster-at-date | Player latent states, lineup continuity, competition |
| `PLAYER_PERF` | How did this player perform relative to role/context? | Player-map or player-time segment | Role-specific predeclared outcomes, allies, opponents, game state |
| `DRAFT_RAW` | What is the blue-side map-win probability from the two complete compositions? | Complete 5v5 map draft | Champions, assigned roles, patch, league; no team/player identity |
| `MATCH_CONTEXT` | What was the pre-match win probability? | Frozen map/series prediction | Pre-event organization/lineup state, complete draft if known, format |
| `DRAFT_LOCAL` | How does one legal pick change the current partial-composition utility? | One legal partial draft state/action | Locked picks, roles/flex assignments, current runtime |
| `DRAFT_POLICY` | Which legal action has the highest expected value under future picks and replies? | Draft state plus policy horizon | Future-pick policy, opponent response, bans, player pools, series rules |

`DRAFT_LOCAL` is not a probability and is not `DRAFT_POLICY`.
`PLAYER_PERF` is not recoverable from team outcomes alone.
`ORG_STATE` is not automatically a current-roster projection.

## Canonical data contracts

### Organization identity

One row per identity interval:

- `organization_id`: immutable internal key;
- `display_name`;
- `alias`;
- `alias_type`: abbreviation, former name, source spelling, team brand;
- `valid_from`, `valid_to`;
- `predecessor_id` / `successor_id` only when competitive-slot continuity is
  documented;
- `source_url`, `retrieved_at`;
- `review_status`.

Aliases resolve within time and source context. A rename is not assumed to be
the same organization merely because a roster or slot continued. Historical
source spellings remain reproducible.

### Competition and tournament

One row per tournament edition:

- immutable `competition_id` and `tournament_id`;
- canonical league, region, tier, event kind, season, split/stage;
- scheduled start/end;
- match format rules;
- current status;
- official source and retrieval time.

Regional leagues, cross-region events, and international events are separate
scopes. LCS and CBLOL must never be collapsed to “Americas” for a domestic
ladder. MSI, EWC, First Stand, and Worlds remain separate events unless a
specific model card declares a pooled bridge.

### Tournament membership

One row per organization–tournament interval:

- `organization_id`, `tournament_id`;
- `status`: registered, active, eliminated, withdrawn, replaced;
- `valid_from`, `valid_to`;
- official source, retrieval time, confidence/review state.

Membership exists before a team plays. Match appearances are evidence of
participation, not the registry.

### Roster membership

One row per player–organization interval:

- immutable `player_id`;
- organization, role, starter/substitute/reserve status;
- effective timestamps;
- source and review state.

`current_team`, `last_team_seen`, and `rating_observed_with_team` are distinct
fields. A prediction joins the roster as it was known before the event.

### Map

One row per completed or explicitly incomplete map record:

- immutable canonical `map_id`;
- source-specific IDs and provenance flags;
- canonical series ID;
- raw source game index;
- canonical completed-map index;
- teams, side, result, date, patch, tournament;
- completion status and completion source;
- participant count, picks/bans, duration, kills;
- remake/cancellation relationship where applicable.

Source-specific raw values are retained alongside canonical values. OE/GRID
deduplication produces one map with source provenance, never two silent rows.

### Series

One row per scheduled series:

- immutable `series_id`;
- tournament, stage, scheduled format, status;
- team pair;
- ordered canonical map IDs;
- canonical score;
- completion/forfeit source;
- validation/quarantine reason.

Series identity may not depend on wall-clock buckets. A completed Bo3 ends at
2 wins and a completed Bo5 at 3 wins. Fixed-game groups and incomplete series
are not forced into knockout formats.

### Model release

One immutable release graph:

- `release_id`, data pack ID, registry versions;
- model ID and semantic version;
- created time and trained-through cutoff;
- feature/target schema hashes;
- fit population and exclusions;
- validation artifact and model-selection rule;
- calibration artifact;
- limitations and intended use;
- runtime hash.

The browser runtime and downloadable artifact are generated from the same model
release. An API refuses incompatible dependencies.

### Prediction ledger

One row per frozen pre-event prediction:

- prediction ID and creation time;
- map/series ID;
- model/data/registry IDs;
- all pre-event feature timestamps;
- raw score, calibrated probability, uncertainty definition;
- outcome joined only after completion.

This ledger is the only source for historical model-vs-actual pages.

## Model contracts

### Dynamic organization strength

Use a time-varying Bradley–Terry/state-space model at canonical series grain.
Organization strength evolves through time; observation likelihood and process
evolution are separate. International events can bridge regional pools only
through an explicit competition hierarchy and validated bridge policy.

Minimum validation:

- rolling-origin chronological evaluation;
- log loss, Brier score, calibration, and AUC;
- region/event slices with effective sample sizes;
- sensitivity to process variance and time decay;
- stable output under row reorder and time-zone conversion;
- no future roster, patch, or outcome leakage.

The public conservative rank may use a posterior quantile, but the label must
state what interval it represents. Local Hessian curvature is not automatically
a full posterior uncertainty.

### Lineup strength

Lineup strength is a roster-at-date projection, not an organization alias. It
may aggregate player states only after individual evidence is defined and
validated. Shared lineup effects should model coordination not explained by
individual effects.

New or changed rosters receive wider uncertainty. Player states travel across
organizations; organization coordination does not.

### Individual player performance

Team wins alone do not identify five individual teammate effects. The first
publishable version must choose one of two honest products:

1. **Shared lineup signal:** publish the team-derived signal, tied cohorts, and
   the statement that individual separation is not identified.
2. **Role-aware individual model:** use player-specific outcomes such as
   context-adjusted gold/damage/vision or another predeclared performance
   target, with hierarchical shrinkage by role, champion, patch, league,
   teammate, and opponent context.

The second path follows the identification strategy of role-aware work such as
SIDO and PandaSkill, but it must be validated on Scryglass data. Post-game
performance is appropriate for retrospective player evaluation; it must not
leak into a pre-match prediction.

Required diagnostics:

- discrimination between players with varied lineups;
- stability across adjacent windows;
- independence from teammate and organization where claimed;
- posterior correlations or bootstrap dependence;
- shrinkage/effective sample size;
- role and champion slices;
- ablation against team-outcome-only and simple box-score baselines.

Exact ties receive tied ranks. They are never separated with arbitrary
constants.

### Complete-draft composition

`DRAFT_RAW` estimates pre-match blue-side map-win probability conditional on:

- five champions per side;
- one declared role per champion;
- league and patch scope supported by the model.

It excludes team and player identity. The feature ledger may include
role-aware champion main effects, all within-team pair interactions, all 25
cross-team interactions, and a separately identified side intercept.

Required invariants:

- swapping complete blue/red compositions returns complementary probability;
- all displayed contributions reconcile to the model logit;
- role permutation changes only role-dependent terms;
- sparse interactions shrink toward neutral;
- no future patch or outcome leakage;
- prediction is bounded and calibrated on chronological, future-patch, and
  held-out-league slices.

Explanations include each champion’s direct role-aware effect, synergy with all
four allies, and interaction with all five enemies. No single matchup is
presented as the whole explanation.

### Contextual match prediction

Raw draft and pre-event strength remain separately visible. A contextual
probability must be a jointly trained or explicitly stacked model validated as
one model. Adding logits from separately calibrated components is not accepted
merely because each component is plausible.

Series probabilities derive from map probabilities only under declared
assumptions. For constant map probability `p`:

- Bo3 win probability is `3p² - 2p³`;
- Bo5 win probability is `10p³ - 15p⁴ + 6p⁵`.

Both satisfy side-swap complementarity, but constant `p` may be unrealistic
when side selection, draft, or roster changes by map. The runtime must declare
the assumption or simulate map-specific probabilities.

### Partial-draft analysis

The existing partial score may remain only as `DRAFT_LOCAL`: an uncalibrated
counterfactual utility with neutral unfilled seats.

`DRAFT_POLICY` requires:

- legal pick/ban sequence and unavailable champions;
- role assignment and flex uncertainty;
- expected future own picks;
- opponent response;
- player champion pools when contextual mode is selected;
- patch and competition rules;
- single-game or multi-game horizon;
- search approximation and uncertainty;
- offline policy evaluation against baselines.

Tree search is one defensible approach, as demonstrated in multi-round MOBA
draft research, but Scryglass must validate its own policy. Until then the UI
must not call local candidate ranking “best response” or imply it is comparable
to a production drafting assistant. Recent comparative work found that tested
draft tools did not establish above-50% winning recommendations once uncertainty
was considered; a recommendation margin is not proof of a correct pick.

### Probability and calibration

Accuracy is secondary. Every probability model reports:

- log loss and Brier score;
- calibration curve/reliability diagram with honest sample counts;
- AUC or another discrimination metric;
- base-rate and simple-model baselines;
- chronological and future-patch results;
- confidence intervals or bootstrap variability;
- the predeclared model-selection criterion.

Calibration error estimates are diagnostics, not model-selection scores unless
their statistical properties and binning are declared. Prediction intervals
must state whether they represent parameter, posterior predictive, or empirical
forecast uncertainty.

### Ranking and sorting

- Rank only within a declared comparable scope and model.
- Use competition-specific chips; never pool leagues silently.
- Default to a conservative posterior quantity only when its uncertainty is
  valid and comparable.
- Exact values receive tied ranks.
- Stable deterministic tie-breakers affect display order, not rank.
- “Games” and “Series” labels match the actual denominator.

## Page contracts

### `/`

- **User job:** understand what Scryglass studies and enter the latest
  evidence-backed work.
- **Primary grain:** article/research claim.
- **Required output:** latest article, its question, data/model date, and routes
  to supporting instruments.
- **Fail closed:** do not hard-code a scientific headline that is no longer
  backed by the current article artifact.
- **Non-goal:** a dashboard of unrelated model scores.

### `/articles`

- **User job:** browse the research library by question and scope.
- **Primary grain:** one versioned article.
- **Required output:** question, league/patch/time scope, publication/update
  date, pack/model release, and limitation summary.
- **Fail closed:** hide drafts without frozen evidence and reproduction bundle.

### `/articles/[slug]`

- **User job:** evaluate one claim from question through evidence.
- **Primary grain:** one explicit estimand and analysis version.
- **Required output:** plain-language result, uncertainty, assumptions,
  robustness, citations, frozen pack/model, and reproduction path.
- **Fail closed:** never substitute a different estimand because it yields a
  more dramatic headline.
- **Scenario rule:** a mathematically correct sensitivity result remains a
  scenario result. Its fitted uncertainty, scenario-input uncertainty, and
  causal limitations are co-primary with the point estimate.
- **Non-goal:** universal game advice or betting guidance.

### `/ratings` with `/elo` as a compatibility redirect

- **User job:** ask who appears strongest now, in which competition, with how
  much evidence and uncertainty.
- **Primary grain:** one model state at a declared date.
- **Required output:** separate tabs for organization state, lineup projection,
  and individual performance; scope, model date, evidence, uncertainty, and
  definitions.
- **Fail closed:** no individual ladder until individual separation is
  identified; no “current” filter without authoritative tournament membership.
- **Non-goal:** presenting every strength concept as Dual Elo.

### `/team/[id]`

- **User job:** understand one organization’s identity, current tournament
  membership, roster, strength, and history.
- **Primary grains:** organization interval, membership interval, roster
  interval, model state.
- **Required output:** verified current identity/league/tournament; roster as of
  a date; organization strength and lineup projection shown separately; series
  history and provenance.
- **Fail closed:** current roster does not fall back to last rating-team.

### `/player/[id]`

- **User job:** understand affiliation, role, evidence, and individual
  performance without confusing it with team success.
- **Primary grains:** player identity, roster interval, player-role performance
  posterior.
- **Required output:** current affiliation and role, retrospective
  player-performance estimate, shared lineup signal, uncertainty, effective
  sample, and model date.
- **Fail closed:** if only team outcomes exist, show the shared lineup signal
  and non-identifiability instead of an individual rank.

### `/matches` with `/browse` as a compatibility redirect

- **User job:** find complete and incomplete series, inspect the record, and
  evaluate frozen pre-match predictions.
- **Primary grain:** canonical series.
- **Required output:** series score first, map score second; scheduled format;
  status; remakes/forfeits; source provenance; frozen predictions where
  available.
- **Fail closed:** quarantined or boundary-truncated groups are explicitly
  incomplete and excluded from aggregate records.
- **Non-goal:** a headline hit-rate scoreboard.

### `/matches/head-to-head`

- **User job:** compare two organizations across series and roster eras.
- **Primary grain:** canonical series.
- **Required output:** series record, map record, event/patch filters, roster-era
  splits, complete chronology, and source status.
- **Fail closed:** never label a map-win total as a series record.

### `/matches/[series_id]`

- **User job:** inspect what happened and what was known before it happened.
- **Primary grain:** one canonical series plus ordered maps.
- **Required output:** observed map facts; raw/canonical indices; remake status;
  picks/bans; and frozen pre-match predictions with model IDs.
- **Fail closed:** current ratings cannot be substituted for historical
  predictions; post-match features cannot appear in the forecast block.
- **Non-goal:** betting lines or over/under framing.

### `/sandbox`

- **User job:** explore how a legal pick changes a draft and, once a policy
  exists, compare legal future actions.
- **Primary grain:** versioned draft state.
- **Required output:** selectable legal champions/roles, patch/model version,
  locked draft state, raw complete-composition probability only at 5v5, local
  partial utility before completion, and explicit raw/contextual toggle.
- **Fail closed:** unsupported role/patch states show “not estimated”; no local
  utility is labelled projected win rate or best response.
- **Non-goal:** deterministic global champion tier list disguised as a draft
  assistant.

### `/methodology`

- **User job:** learn exactly what each public number means and how it was
  validated.
- **Primary grain:** one model/data contract per section.
- **Required output:** estimand, observation grain, inputs/exclusions,
  equation, temporal behavior, uncertainty meaning, validation, and limits.
- **Fail closed:** content cannot claim an artifact or validation that is
  absent from the active release.

### `/reproduce`

- **User job:** reproduce a published number with the minimum public bundle.
- **Primary grain:** immutable release and article bundle.
- **Required output:** allowlisted files, hashes, schema, licenses/attribution,
  model cards, validation, exact query/command, and expected result tolerance.
- **Fail closed:** missing links or undeclared files fail release validation.
- **Non-goal:** exposing the internal workspace.

### `/live`

- **User job:** none until a verified live feed and safety contract exist.
- **Current behavior:** redirect/absent.
- **Fail closed:** do not revive from polling, placeholders, or mixed live and
  historical sources. If restored, live provenance and coverage are a separate
  contract.

## API response envelope

Every quantitative API response includes:

```json
{
  "estimand_id": "DRAFT_RAW",
  "value": 0.546,
  "unit": "probability",
  "data_pack_id": "immutable-id",
  "model_id": "immutable-id",
  "registry_versions": {
    "identity": "immutable-id",
    "competition": "immutable-id",
    "roster": "immutable-id"
  },
  "trained_through": "timestamp",
  "effective_at": "timestamp",
  "scope": {
    "league": "LPL",
    "patch": "26.14"
  },
  "uncertainty": {
    "kind": "approximate_conditional_parameter_interval",
    "level": 0.95,
    "lower": 0.459,
    "upper": 0.630
  },
  "limitations": ["Future-patch calibration is weaker."],
  "provenance": []
}
```

Partial or unverifiable results return a typed status and no fabricated value.

## Cross-cutting invariants

- Canonical team identities are stable under aliases and source spelling.
- Historical rows survive organization inactivity or rename.
- Current membership comes from the current tournament registry.
- Roster-at-date joins cannot see future movement.
- OE/GRID overlap produces one canonical map with both provenance flags.
- Raw source map index and canonical completed-map index remain separate.
- A completed series cannot tie or exceed its scheduled win threshold.
- Side swaps preserve complementary probabilities where the estimand is
  symmetric.
- Complete-draft contributions reconcile exactly.
- Duplicate maps cannot increase model weight.
- Series weighting is explicit and invariant to duration/timezone.
- Every denominator is named.
- Every sort is stable; exact scores receive tied ranks.
- Public files are allowlisted.

## Literature interpretation

Current skill-rating research supports time-varying latent states and explicit
inference approximations rather than a timeless leaderboard. Adjusted
plus-minus research demonstrates the collinearity problem created by frequent
teammates. SIDO and PandaSkill show defensible ways to introduce role-aware,
player-specific performance evidence. MOBA drafting research treats
recommendation as a sequential or personalized decision problem, not a
single-step score. Calibration research separates probability quality from
accuracy.

These studies define useful design constraints. They do not validate Scryglass
without independent recomputation, chronological holdouts, ablations, and
calibration on the actual production pipeline.

The cross-domain candidate universe and frozen selection rules are specified in
[`CROSS_DOMAIN_SOTA_RESEARCH_PROTOCOL.md`](CROSS_DOMAIN_SOTA_RESEARCH_PROTOCOL.md).

Primary references:

- <https://arxiv.org/abs/2308.02414>
- <https://arxiv.org/abs/1903.07746>
- <https://arxiv.org/abs/1201.0317>
- <https://arxiv.org/abs/2403.04873>
- <https://arxiv.org/abs/2501.10049>
- <https://arxiv.org/abs/2012.10171>
- <https://arxiv.org/abs/2204.12750>
- <https://arxiv.org/abs/2311.05912>
- <https://doi.org/10.1109/GEM61861.2024.10585636>
- <https://arxiv.org/abs/2008.03033>
- <https://arxiv.org/abs/2203.07835>
