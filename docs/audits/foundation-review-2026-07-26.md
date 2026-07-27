# Scryglass scientific foundation review — 2026-07-26

Status: **RELEASE CANDIDATE PASSES IMPLEMENTED DATA/MODEL GATES; DRAFT WIN
PROBABILITY REMAINS WITHHELD**

This began as the controlling review of production pack
`v2026.07.26.2343`. The remediated immutable candidate is
`v2026.07.27.0845` (schema `1.6.0`, observations through
`2026-07-26T21:25:07Z`). It contains 16,567 unique maps, 33,134 team-side
rows, and 165,670 player-role rows. Its fail-closed pack audit reports zero
launch blockers, majors, or minors; one gapped GRID series remains
quarantined and 274 old team-record histories remain available only outside
current scoped views.

The candidate binds data, ratings, model artifacts, and validation to one
immutable pack ID. The series dynamic Bradley–Terry release passed its frozen
1,771-series test against the selected rolling-Elo benchmark (log loss
0.58621 versus 0.62166; paired difference -0.03546, 95% moving-block interval
-0.04807 to -0.02191). This supports this release and benchmark only, not a
universal state-of-the-art claim. Bo5 calibration remains a stated limitation.

The composition probability pipeline did not beat the chronological overall
blue-side base-rate benchmark on the untouched final window (log loss 0.69427
versus 0.69021), so numeric Draft WR is withheld. The Sandbox release is an
explicit experimental policy utility refit on the full population, including
final-window labels; it is not described as a probability, optimal policy, or
the exact coefficient artifact scored on that final window. Player
team-outcome signals are not treated as identified individual skill and public
player ordering/rank movement is withheld.

Validation on the installed candidate: 268 backend tests plus 24 subtests,
117 frontend tests, TypeScript, lint, production build, immutable-pack audit,
and browser interaction checks all pass. Browser checks confirmed role-filtered
champion selection and recommendation changes after each draft action.

The remaining text records the defects found during the original review and
the rationale for the remediation; statements about the old production pack
are historical evidence, not the current candidate state.

The strongest parts of Scryglass are the editorial intent, the map-level source
provenance, and the complete-draft composition feature design. The basis of the
ratings, current-membership, series, partial-draft, and reproduction surfaces is
not yet scientifically or semantically coherent enough for publication.

The isolated worktree now contains production-code remediations and regression
tests described below. No styling, merge, pack publication, or deployment has
been performed.

## Executive decision

The app currently answers several different questions with labels that make
them look like one model:

1. organization strength inferred from series outcomes;
2. a sequential team Elo used for historical map predictions;
3. player ratings updated identically from team outcomes;
4. player-aggregated lineup strength;
5. raw five-versus-five draft composition;
6. composition plus current team/player context; and
7. a greedy partial-draft utility.

They have different grains, clocks, assumptions, uncertainty meanings, and
validation artifacts. Publishing them under “Dual Elo,” “player rating,” or
“model share” lets a reader infer comparability that the implementation does
not provide.

Three conditions are launch blockers:

- real series are split by four-hour wall-clock buckets, changing rating
  observations and outcomes;
- team-result-only player updates cannot identify individual teammates, yet the
  UI ranks them as individual skill; and
- current tournament membership is inferred from teams that have already
  appeared in the pack, excluding registered entrants before their first map.

## Worktree remediation status — 2026-07-27

Implemented and passing:

- canonical map-to-series IDs no longer use four-hour buckets;
- a series is rating-eligible only with a verified scheduled format,
  contiguous source indices, a decisive compatible score, and non-conflicting
  completion provenance;
- browser series cards consume the canonical format/status fields and never
  infer Bo3/Bo5 from a final score;
- current Tier 1 membership comes from a time-bounded Riot tournament registry,
  not appearances;
- all 56 registered current participants join the checked-in historical team
  records after stable sponsor/rebrand organization IDs were introduced;
- player snapshots now report exact team-outcome co-exposure groups, and exact
  rating ties use competition ranks instead of alphabetical pseudo-ranks;
- Sandbox actions now enforce pick order, one unique role per side, and
  selected/unavailable disjointness; stale API responses cannot supply role
  legality; a closed role resets after a pick;
- Sandbox scoring fails closed when the full-composition runtime cannot verify
  a role-complete state, rather than silently changing to the legacy scorer;
- public Sandbox copy now says “one-pick counterfactual” and explicitly states
  that future opponent replies are not simulated.

Still blocking:

- the checked-in 2025–2026 team-game pack has no authoritative scheduled-format
  field. The new ledger therefore quarantines all 7,967 candidate series
  instead of rebuilding ratings from guessed formats;
- the current registry supplies current membership but not a dated historical
  membership ledger;
- current player rosters remain last-observed match lineups, not an
  authoritative dated roster registry;
- the player, team, match, draft-probability, and sequential-policy model
  tournaments have not yet selected replacement champions.

Focused validation currently passes 72 Python tests, 25 application tests,
frontend lint, and a Next.js 16.2.11 production build.

The correct foundation is specified in
[`../SCIENTIFIC_PRODUCT_ARCHITECTURE.md`](../SCIENTIFIC_PRODUCT_ARCHITECTURE.md).
The implementation order and agent boundaries are in
[`../IMPLEMENTATION_WORKSTREAMS.md`](../IMPLEMENTATION_WORKSTREAMS.md).
The expanded mathematical, financial-econometric, calibration, and sequential
decision benchmark universe is in
[`../CROSS_DOMAIN_SOTA_RESEARCH_PROTOCOL.md`](../CROSS_DOMAIN_SOTA_RESEARCH_PROTOCOL.md).

## Severity system

| Severity | Meaning |
|---|---|
| Launch blocker | A public claim is materially false, non-identifiable, or based on corrupted observation grain. The affected surface must fail closed until fixed. |
| Major | The output may be numerically reproducible but its scope, provenance, probability meaning, or temporal state is misleading. |
| Minor | The result is basically interpretable but has a presentation, naming, or completeness defect that can cause confusion. |
| Informational | A limitation or future improvement that does not invalidate the present claim when clearly disclosed. |

## Exact production-pack scope

### Live production re-audit — 2026-07-27

The current `https://scryglass.vercel.app/api/pack-manifest` points to
`v2026.07.27.0406` (schema `1.3.0`, published
`2026-07-27T04:08:42.998250Z`). Its manifest omits `data_as_of`; direct row
inspection puts the latest observation at `2026-07-26T19:35:13.613Z`. A fresh
download, SHA-256 verification, and audit of all 43 files in that exact
immutable bundle found:

| Severity | Findings |
|---|---:|
| Launch blocker | 7 |
| Major | 6 |
| Minor | 0 |
| Informational | 1 |

| Exact live grain | Rows / games |
|---|---:|
| Public maps | 6,314 |
| OE-backed public maps | 6,252 |
| GRID-backed public maps | 62 |
| Team rows / unique games | 32,792 / 16,396 |
| Player rows / unique games | 163,960 / 16,396 |
| Team/player games absent from maps | 10,082 each |
| Map/source-year conflicts | 71 |
| Team/source-year conflicts | 496 |
| Player/source-year conflicts | 2,480 |
| Current-membership registry | missing |
| Quarantined public paths | 6 |
| Required public artifacts missing | 3 |

The live production pack therefore does **not** have one public analytical
population. Matches and rating history use 6,314 maps while team/player
records and rating inputs use 16,396. Its current-membership note explicitly
describes an appearance-derived signal and the manifest contains no reviewed
participant registry. It also exposes five tier-list CSVs and a
`blade_chest_role_matchups.json` working artifact that the governed allowlist
now quarantines, while omitting `draft_composition.json`,
`model_validation_2026-07-27.json`, and `meta/source_summary.json`.

The generator remediation now:

- coalesces `oe_year` before transport `year`, rewrites the canonical public
  `year`, and partitions on that same value;
- constructs the public map population from the complete two-team feed,
  preserving wide-feature rows where available and appending missing maps with
  explicit detail provenance;
- preserves OE versus GRID source flags at the appended-map grain;
- removes GRID-only games without `events_game_end` or verified
  `end_state_summary` provenance from team, player, and map populations
  together; and
- audits map/team/player identity equality and detail provenance as release
  gates.

Replaying the corrected population construction against the live files
produced 16,396 unique map identities. The old downloadable team/player feed
stripped completion provenance from all 62 GRID games, so the new completion
gate correctly rejects them rather than inventing provenance. The
representative corrected population is therefore 16,334 OE maps, 32,668 team
rows, and 163,340 player rows. A final production-pack count must be regenerated
from the source warehouse, where verified GRID completion provenance can be
retained.

| Item | Production value |
|---|---:|
| Pack ID | `v2026.07.27.0406` |
| Schema | `1.3.0` |
| Created | `2026-07-27T04:08:42.998250Z` |
| Data as of | absent from manifest; row maximum `2026-07-26T19:35:13.613Z` |
| Public map rows | 6,314 |
| OE map rows | 6,252 |
| GRID map rows | 62 |
| Full team/player rating population | 16,396 maps |
| Team rating pseudo-series accepted | 8,738 |
| Team rating maps accepted | 14,988 |
| Pseudo-series skipped as tied | 607 |
| Public manifest files | 43 |

The browsing/draft/map population is not the rating population. Team and player
ratings use 16,396 maps, while the public map explorer and prediction history
use 6,314. A model card must state both populations instead of presenting one
pack date as if it implied one analytical sample.

### Mutable-pointer drift during the audit

The production latest pointer changed from `v2026.07.26.2343` to
`v2026.07.27.0406` during this review. The exact counts above are now frozen to
the latter immutable bundle. This drift demonstrates why every
page/API/model response must use one request-scoped immutable bundle and why
publication time cannot be presented as observation time. The release audit
must be rerun against the final candidate immutable pack immediately before
approval.

## Finding inventory

### LB-01 — Real series are not the rating observation

- **Expected grain:** one canonical completed series, containing an ordered list
  of canonical completed maps plus raw source indices and remake/cancellation
  provenance.
- **Evidence:** OE team rows retain a source `game` number, but
  `build_maps_frame_from_team_games` drops it. `_series_key` then identifies
  non-GRID series by team pair and `date.floor("4h")`.
- **Root cause:** a clock bucket is being used as an identity key.
- **Affected scope:** all 16,396 rating-population maps without an explicit GRID
  series ID.
- **User impact:** series can be double-weighted, partially discarded, or
  assigned a pseudo-result opposite the actual series winner.
- **Fix:** construct and validate a canonical series ledger before any rating
  fit. Preserve source series ID where available, OE game number, raw map index,
  canonical completed-map index, scheduled format, completion status, and
  remake/cancellation relationships.
- **Regression check:** every accepted series has one stable ID; contiguous
  canonical indices; one team pair; a result compatible with its format; and
  the exact same map set under row reorder, timezone conversion, or exporter
  batching.

Independent reconstruction from source game-number resets and bounded
same-day/team-pair gaps found:

| Distortion | Count |
|---|---:|
| Likely real series | 7,951 |
| Real series with a winner | 7,866 |
| Tied/fixed-game groups | 85 |
| Completed real series represented by more than one accepted pseudo-series | 853 |
| Extra pseudo-observation weights | 855 |
| Completed real series only partly used | 532 |
| Maps dropped from completed real series | 1,226 |
| Pseudo-results opposite the full real-series winner | 196 |

The rating metadata separately reports 607 skipped clock buckets containing
1,408 maps: 510 two-map buckets and 97 four-map buckets.

Representative reproductions:

- `2025-01-23`, LPL, JD Gaming vs Oh My God: the real Bo5 is JDG 3–2.
  The current key creates an 08:00 bucket labelled OMG 2–1 and a 12:00 bucket
  labelled JDG 2–0. The series receives two observations and one has the wrong
  winner.
- `2025-02-09`, LPL, JD Gaming vs Weibo Gaming: the same split-and-conflicting-
  label pattern occurs.
- `2025-04-16`, LCK, BNK FEARX vs Gen.G: an early one-map bucket says BNK FEARX
  won, while a later two-map bucket says Gen.G won 2–0. The actual Bo3 winner is
  Gen.G.

Major-league distortion:

| League | Real series | Overweighted | Wrong pseudo-results | Partly used | Maps dropped |
|---|---:|---:|---:|---:|---:|
| LPL | 492 | 128 | 32 | 53 | 124 |
| LCK | 344 | 42 | 9 | 36 | 82 |
| LEC | 282 | 65 | 18 | 15 | 34 |
| LCP | 185 | 34 | 8 | 24 | 52 |
| CBLOL | 175 | 11 | 5 | 10 | 22 |
| LCS | 159 | 0 | 0 | 0 | 0 |
| EWC | 119 | 16 | 1 | 16 | 34 |
| Worlds | 44 | 7 | 1 | 4 | 12 |
| MSI | 40 | 13 | 2 | 6 | 20 |
| First Stand | 26 | 5 | 0 | 3 | 10 |

Code evidence:

- `lol_kills/export/pack_records.py`, `build_maps_frame_from_team_games`;
- `lol_kills/ratings/hierarchical_bt.py`, `_series_key`.

**Worktree verification:** the replacement canonical ledger sees 16,697 maps
and 7,967 candidate series in the checked-in `v2026.07.26` team-game pack.
Exactly 0 series are rating-eligible because all 7,967 lack a verified
scheduled format; 28 additionally have non-contiguous source game indices.
This is an intentional fail-closed result. A final score is not accepted as
proof of the scheduled format.

### LB-02 — The public player ladder does not identify individual skill

- **Expected grain:** one player-role performance observation with contextual
  features or a posterior player effect with explicit covariance and shared
  lineup uncertainty.
- **Evidence:** every player on a team receives the same residual, K factor
  logic, and uncertainty shrink on every map. Role weights are used only when
  aggregating the five player states into team strength.
- **Root cause:** team outcome is copied to five players without a
  player-specific observation.
- **Affected scope:** all 3,767 player states; the most visible symptom is exact
  equality among stable teammates.
- **User impact:** readers interpret the rank as individual performance even
  when the data only identify a lineup/team contrast.
- **Fix:** either publish a shared lineup signal and explicitly tied,
  non-identified teammate cohorts, or fit a role-aware hierarchical individual
  performance model using predeclared in-game outcomes. Do not add arbitrary
  role constants merely to break ties.
- **Regression check:** fixed-lineup simulations recover only the lineup
  contrast; the UI uses tied ranks for exact equality; individual separation
  requires observed player-specific evidence; posterior correlations and
  uncertainty are exposed.

Within the current Tier 1 player table:

| Exact-tie statistic | Count |
|---|---:|
| Players | 183 |
| Full-state exact-tie groups | 20 |
| Players in exact ties | 51 (27.9%) |
| Largest tie group | 5 |

Examples include Chovy/Ruler; Duro/Kiin; Bin/ON; Doran/Faker/Keria/Oner; and all
five listed Movistar KOI players. These are a mathematical consequence of the
update rule, not surprising evidence that the teammates have precisely equal
skill.

For a permanently fixed five-versus-five lineup, the team-outcome design matrix
has rank one and nullity nine. Only the lineup contrast is identified; nine
independent player contrasts remain unconstrained by the likelihood.

Code evidence: `lol_kills/ratings/player_elo.py`, map update loops.

**Checked-in pack reproduction:** among 3,846 players in 166,970 player-game
rows, 1,055 players fall into 376 exact shared team-outcome exposure groups;
the largest group contains five players. In the existing appearance-scoped
Tier 1 table, 97 players occupy 35 numerically exact adjusted-rating groups,
while 127 players occupy 48 groups at the displayed one-decimal precision.
Examples include all five G2 players at `1636.3552`, four T1 players at
`1672.9321`, and all five Movistar KOI players at `1575.7275`.

The exact design diagnostic shows Duro/Kiin share one signed outcome column;
Bin/ON share one; Doran/Faker/Keria share one; and
Nemesis/Rekkles/Velja share one. Ruler and Chovy do not have identical complete
exposure columns, so their equal displayed score is a separate consequence of
the update path and rounding rather than proof of equal individual skill.

### LB-03 — Current tournament membership is inferred from appearance

- **Expected grain:** one organization’s membership in one official tournament
  with `valid_from`, `valid_to`, source URL, source retrieval time, status, and
  confidence.
- **Evidence:** `build_current_tournament_membership` selects teams appearing in
  recent maps from the latest tournament family.
- **Root cause:** observation is treated as registration. A team cannot become
  “current” until the pack has already observed it playing.
- **Affected scope:** every current league-scoped ladder and filter.
- **User impact:** registered teams disappear before their first match; LCK
  disappears entirely when the map pack has no current-tournament row.
- **Fix:** ingest an authoritative tournament participant registry. Match
  appearances may confirm or challenge the registry but must not define it.
  Preserve historical participation separately.
- **Regression check:** all registered participants appear before match one;
  eliminated or withdrawn teams follow explicit tournament status rules; a
  missing source fails closed rather than falling back to “recently observed.”

Official current Split 3 pages list 56 participants across the six Tier 1
domestic leagues. The production pack marks 24:

| League | Official current participants | Pack-derived current | Missing |
|---|---:|---:|---:|
| LCK | 10 | 0 | 10 |
| LCS | 8 | 2 | 6 |
| LEC | 10 | 4 | 6 |
| CBLOL | 8 | 2 | 6 |
| LCP | 8 | 6 | 2 |
| LPL | 12 | 10 | 2 |
| **Total** | **56** | **24** | **32** |

The authoritative source is the active tournament, not a season-wide team list:
for example, the 2026 handbook lists 14 LPL organizations, while the current
LPL Split 3 page lists 12 participants.

Code evidence: `lol_kills/export/pack_records.py`,
`build_current_tournament_membership`.

**Worktree verification:** the reviewed Riot registry contains exactly 56
current participants: LCK 10, LPL 12, LEC 10, LCS 8, CBLOL 8, and LCP 8.
All 56 stable organization IDs join the checked-in team records; zero current
participants are missing. The registry expires on its declared review deadline
and domestic ladder filters fail closed when it is missing or stale.

### MA-01 — “Dual Elo” conflates incompatible models

- **Expected grain:** one named model, one fit population, one time state, one
  uncertainty definition, and one validation artifact.
- **Evidence:** the team ladder is hierarchical series-level
  Bradley–Terry; match history uses sequential Dual Elo; player profiles use a
  team-outcome player heuristic. The same pages and navigation call these
  “Dual Elo.”
- **Root cause:** route and product naming predate the current model stack.
- **Affected scope:** Ratings, team/player pages, match cards, methodology, and
  API consumers.
- **User impact:** numbers that are not on the same model or clock appear
  directly comparable.
- **Fix:** give each public estimand a stable model ID and plain-language name.
  Join outputs only through a versioned prediction model, never by label
  similarity.
- **Regression check:** every displayed number resolves to exactly one model
  card and immutable artifact ID.

### MA-02 — Roster state and rating history are mixed

- **Expected grain:** a dated player–organization membership interval,
  independent of a rating observation’s historical team.
- **Evidence:** 16 player `last_team` values differ from
  `player_records.current_team`. Team pages use rating `last_team` to construct a
  current roster. Los Ratones consequently shows two players where the current
  record contains six.
- **Root cause:** “last team in the rating feed” is used as a roster registry.
- **User impact:** stale teams and mixed rosters drive profile labels and
  contextual predictions.
- **Fix:** use a temporal roster registry for current pages; preserve
  `rating_observed_with_team` only as historical provenance.
- **Regression check:** roster-at-date joins cannot see future moves; current
  profile rosters reconcile to the official registry; substitutions and
  inactive/reserve status remain explicit.

### MA-03 — Partial-draft “best response” is a greedy local utility

- **Expected grain:** a legal draft state and an action value under a declared
  policy for future own picks, opponent responses, role assignment, bans, and
  series rules.
- **Evidence:** empty seats are scored as neutral; side intercept, calibration,
  and team strength are omitted; candidates are ranked by one-step change in a
  partial utility.
- **Root cause:** a useful local counterfactual was promoted into a draft
  recommendation.
- **Affected scope:** `/sandbox` before ten champions are locked.
- **User impact:** suggestions remain dominated by global champion effects and
  can look deterministic or unrelated to the developing draft.
- **Fix:** label the present number “local counterfactual utility” if retained.
  A real recommendation must evaluate future picks and opponent replies,
  marginalize flex-role assignments, enforce legal draft constraints, and
  declare whether the policy is single-game or series-aware.
- **Regression check:** a changed draft state changes candidate values for
  documented reasons; blue/red swap and draft-order symmetries hold where
  applicable; all recommended actions are legal; no uncalibrated utility is
  rendered as win probability.

The worktree fixes the role-filter dead end, stale legality race, illegal pick
sequence acceptance, and silent legacy fallback. The scientific complaint
remains valid because the current ranking is still a greedy one-step utility,
not a sequential policy comparable to a drafting product. Across 167 one-pick
states in the audited runtime, only 86 distinct top-15 support lists appeared;
63 materially different states shared the same ordering.

### MA-04 — Probability validation is incomplete and inconsistently used

- **Expected grain:** one frozen pre-event probability matched to one observed
  outcome, with chronological validation and no future features.
- **Evidence:** the current team and player holdouts expose accuracy/Brier/AUC,
  but the public surface emphasizes hit rate. The learned blend has Brier
  `0.21783`, slightly worse than the simple 60/40 blend at `0.21744`, yet is used
  at runtime.
- **Root cause:** model selection and public reporting are optimized around a
  familiar headline rather than a predeclared proper score and calibration
  policy.
- **Affected scope:** match explorer, model-vs-actual, contextual draft, and
  methodology.
- **User impact:** a model can appear better while being overconfident or while
  the selected blend is not the best held-out option.
- **Fix:** select models by chronological out-of-sample log loss/Brier with
  calibration and discrimination reported separately. Store a frozen
  prediction ledger before outcomes.
- **Regression check:** no post-event/current rating may populate a historical
  prediction; reliability diagrams use honest bins; all model selection
  decisions are reproducible from the validation artifact.

Descriptive holdout checks:

| Model / Elo band | n | Mean predicted | Actual | Gap | Wilson 95% interval |
|---|---:|---:|---:|---:|---:|
| Team, 40–50 | 59 | 61.09% | 52.54% | +8.55pp | 40.04–64.73% |
| Player, 40–50 | 85 | 64.46% | 48.24% | +16.23pp | 37.92–58.70% |

The team interval does not by itself prove miscalibration at 95%, but it is a
warning. The player-band mean prediction lies above the Wilson upper bound and
requires correction or a strong predeclared explanation.

### MA-05 — Model artifacts are on different clocks

- **Expected grain:** one immutable pack release containing data, all runtime
  models, validation, limitations, hashes, and creation/training cutoffs.
- **Evidence:** the browser composition runtime is a checked-in compressed
  asset; the hourly public pack contains older draft calibration artifacts but
  not the composition runtime. The runtime omits validation and limitations.
- **Root cause:** data refresh and model release are separate deployment paths.
- **Affected scope:** draft API, sandbox, methodology, and reproduction.
- **User impact:** “current pack” does not imply “current model,” and the visible
  model cannot be reproduced from the downloadable pack.
- **Fix:** release models atomically with a pack-level dependency graph. Every
  API response returns data pack ID, model ID, training cutoff, patch coverage,
  validation ID, and context registry version.
- **Regression check:** production refuses to load a model whose declared
  dependencies do not match the active pack.

### MA-06 — Composition uncertainty is narrower than its label suggests

- **Expected grain:** predictive uncertainty over parameters, calibration,
  sparse interactions, context inputs, and future match variation.
- **Evidence:** the runtime uses a diagonal Laplace approximation. It omits
  coefficient covariance, low-rank uncertainty, calibration uncertainty, and
  contextual-strength uncertainty.
- **Root cause:** a computational approximation is presented as a general
  interval.
- **User impact:** “95%” can be read as a calibrated predictive interval when it
  is only a conditional parameter approximation.
- **Fix:** call it an approximate conditional parameter interval, or replace it
  with posterior/predictive draws validated for interval coverage.
- **Regression check:** empirical coverage is reported on chronological,
  future-patch, and held-out-league slices.

### MA-07 — The public manifest exposes internal research artifacts

- **Expected grain:** an explicit public allowlist of the minimum files needed
  to reproduce published claims.
- **Evidence:** the UI presents a small file list, but the production manifest
  exposes all 43 files, including 16 grubs study notes, briefs, proof artifacts,
  and internal-looking text files.
- **Root cause:** manifest generation inventories the export directory instead
  of enforcing a public contract.
- **User impact:** downloadable surfaces leak irrelevant implementation
  artifacts and make the public reproduction contract ambiguous.
- **Fix:** generate the manifest from a hard allowlist; publish article-specific
  bundles separately.
- **Regression check:** CI rejects any public path not declared in the allowlist
  and checks all claimed links exist.

### MA-08 — Browser queries can cut a series at the result boundary

- **Expected grain:** whole canonical series returned by a series-level query.
- **Evidence:** the browser fetches an arbitrary map limit and groups into
  series afterward.
- **Root cause:** map pagination is applied before the public observation grain.
- **Affected scope:** Matches, H2H, and any series totals derived client-side.
- **User impact:** the first or last series can be incomplete because of query
  pagination rather than source data.
- **Fix:** query series IDs first, then fetch all maps for those IDs; or
  deliberately overfetch and discard boundary series.
- **Regression check:** page-size changes never change a retained series score.

### MA-09 — Public copy makes claims the artifacts do not support

- **Expected grain:** each sentence resolves to a current model card, data
  contract, or cited research artifact.
- **Evidence:** methodology says Bo3/Bo5 maps are collapsed to one series even
  though clock buckets split them; team pages say “player-aggregated strength”
  while displaying an organization Bradley–Terry rating; H2H copy says series
  while the headline record counts maps; the match page uses betting-style
  “over/under” wording despite Scryglass being a non-betting publication.
- **Root cause:** product copy was not versioned with model changes.
- **User impact:** even a correctly computed number can answer a different
  question from the label around it.
- **Fix:** derive labels from page/model contracts; delete unsupported claims
  until their artifacts exist.
- **Regression check:** a content-contract test maps every important label and
  number to an estimand ID.

### MA-10 — The lead article turns a scenario threshold into a general conclusion

- **Expected grain:** one explicitly parameterized opportunity-cost scenario,
  with uncertainty over both fitted coefficients and scenario inputs.
- **Evidence:** the 58.9% contest bar is algebraically reproducible from a
  side-neutral gold-at-10 logit with two waves of leave farm (`241.33g`), a
  `115.6g` objective swing, a symmetric `+/-600g` fight swing, secure-if-win,
  and opponent-secure-if-lose. At a 50% fight chance the same scenario produces
  a `-2.08pp` contest-minus-leave edge.
- **Root cause:** a sensitivity-model output is written as “Leave still wins at
  50/50,” while the scenario parameters are not estimated as one policy from
  logged decisions.
- **Affected scope:** home-page headline, article title/dek, article charts,
  methodology, and downloadable article artifacts.
- **User impact:** a correct formula value can be mistaken for a universal
  strategic conclusion or causal estimate.
- **Fix:** describe 58.9% as the output of the named reference scenario; make
  scenario uncertainty and sensitivity co-primary. A policy claim requires a
  decision model with observed state, action, outcome, confounding policy, and
  explicit causal assumptions.
- **Regression check:** article claims resolve to scenario parameters; altering
  each parameter updates the headline or invalidates it; fitted and scenario
  uncertainty remain separate.

Independent Wolfram recomputation gives `p*=0.5894271` and
`contest-minus-leave=-2.08137pp` at a 50% fight chance, agreeing with the
runtime rounding. The important problem is interpretation, not arithmetic.
Changing only assumed leave farm moves the threshold from 41.8% at `0g`, to
50.4% at one wave, 58.9% at two waves, 66.6% at `350g`, and 72.4% at `432g`.
The artifact itself states that its narrow delta-method interval omits
clustering and scenario-parameter uncertainty.

### MI-01 — Reproduction links and runtime health are incomplete

- **Expected grain:** every public download link and hydration path in the
  deployed release.
- **Evidence:** `major_teams.json` and the linked PDF are missing; production
  emitted React hydration error 418 during the browser pass.
- **User impact:** reproduction and trust are degraded even where the underlying
  data may be sound.
- **Fix:** link-check the immutable pack and run a zero-console-error browser
  acceptance suite.
- **Regression check:** all public links return the declared hash and content
  type; no console error appears on the canonical page suite.

## GRID completion and map-index findings

The current map pack contains three explicit GRID groups with non-contiguous raw
game indices:

- `2966866`: G2 vs Movistar KOI, raw indices 2 and 3. The 2–0 completed result is
  consistent with an omitted cancelled/remade raw game 1.
- `2975394`: Team WE vs JD Gaming, raw indices 1, 3, and 4. The 2–1 completed
  result is consistent with an omitted cancelled/remade raw game 2.
- `2975400`: ThunderTalk Gaming vs Edward Gaming, only raw index 2. This group is
  genuinely incomplete in the pack.

Raw source indices must never be overwritten with canonical completed-map
indices. A valid completed series can have raw gaps because of remake or
cancellation. Completion requires verified series summary/status provenance,
while display/order uses a separate canonical completed-map sequence.

The earlier “seven tied GRID series” audit was itself wrong: it summed
blue-side wins without first orienting results to a canonical team. This is a
regression class. Series aggregation must be invariant to which team appears on
blue side in each map.

## Corrected-population model tournaments — 2026-07-27

These are research evaluations, not production promotions. They use the
canonical-year, full-map reconstruction described above: 32,668 team-side
rows, 163,340 player rows, and 16,334 complete role-labelled maps from
2025-01-11 through 2026-07-18. No production artifact was overwritten.

### Draft composition

The predeclared tournament used four non-overlapping UTC-date groups:

- train: 8,987 maps, through 2025-09-16;
- validation: 2,425 maps, 2025-09-17 through 2026-02-18;
- model-family selection: 2,445 maps, 2026-02-19 through 2026-04-29; and
- untouched final test: 2,477 maps, 2026-04-30 through 2026-07-18.

Additive role/champion, ally-synergy, and opponent-interaction candidates were
fit without team, player, or roster identity. The overall blue-side base rate
won model selection. On the untouched final test:

| Model | Log loss | Brier | 10-bin ECE |
|---|---:|---:|---:|
| Overall blue-side base rate | 0.69058 | 0.24872 | 0.49% |
| League/side base rate | 0.69205 | 0.24944 | 0.93% |
| Best composition candidate (additive) | 0.69852 | 0.25181 | 5.99% |

The composition candidate also lost to the league/side baseline on the 2,309
final-test maps from patches unseen before the final split: log-loss lift
`-0.00461` and Brier lift `-0.00169`, where positive lift would favor
composition.

**Decision:** the current champion-composition family fails the probability
promotion gate. It must not be presented as calibrated “Draft WR,” “win
chance,” or a state-of-the-art forecast. A partial-draft search may expose a
clearly labelled experimental policy value, but it cannot inherit probability
language from this failed terminal model.

### Dynamic team strength

A diagonal-Gaussian online dynamic Bradley–Terry research candidate was
selected on the 2,445-map validation interval and then scored prequentially on
the same untouched 2,477-map final interval. It uses immutable organization
keys, predicts all maps sharing an exact timestamp before observing any result
from that timestamp, inflates uncertainty during inactivity, and contains no
series-score or post-map feature.

| Model | Final log loss | Final Brier |
|---|---:|---:|
| Dynamic Bradley–Terry candidate | 0.63223 | 0.22084 |
| Existing uncalibrated Dual Elo | 0.63806 | 0.22377 |
| Validation-period base rate | 0.69053 | 0.24869 |

A paired circular moving-block bootstrap over 1,024 inferred series clusters
(5,000 replicates, 12-series blocks, fixed seed) estimated candidate-minus-Dual
Elo:

- log loss: `-0.00583`, 95% interval `[-0.01256, +0.00076]`,
  **inconclusive** at the zero margin;
- Brier: `-0.00293`, 95% interval `[-0.00587, -0.00005]`,
  **superior**.

The inferred series IDs are suitable only as dependence clusters here: all
7,805 local series remain quarantined from rating-series eligibility because
the checked pack does not carry verified scheduled-format provenance.

**Decision:** this is promising but not yet a production promotion. Log-loss
superiority is not established, the calibration layer must be selected without
test labels, roster/rename stress tests remain mandatory, and the public
organization-rating estimand still needs a frozen model card and replayable
prediction ledger.

A subsequent validation-only calibration tournament selected a Platt map
(`intercept=-0.0472`, `slope=1.2294`; validation slope 95% local-Hessian
interval `[1.0943, 1.3644]`). It improved validation log loss from `0.60870` to
`0.60629`, but worsened the untouched final test from `0.63223` to `0.63440`;
test Brier likewise worsened from `0.22084` to `0.22130`. The frozen
calibration therefore does not transfer cleanly and is another reason not to
promote the candidate. The uncalibrated and calibrated ledgers remain separate.

A robustness challenger added a two-branch, outcome-supported shock update,
selected only on validation (`hazard=0.015`, added variance `0.50`, minimum
four prior observations). On the untouched final interval it scored `0.63179`
log loss, `0.22068` Brier, and `0.01857` 10-bin ECE, compared with `0.63223`,
`0.22084`, and `0.01924` for raw dynamic Bradley–Terry. In 5,000 paired
circular-block replicates, challenger-minus-raw-dynamic intervals crossed zero
for both log loss (`[-0.00232, +0.00147]`) and Brier
(`[-0.00101, +0.00068]`). Its intervals versus the old Dual Elo benchmark also
crossed zero. **Decision: not promoted.** The method is a two-branch
assumed-density approximation inspired by change-point models, not exact
Bayesian online change-point detection, and an outcome-supported shock does
not identify whether the cause was roster, patch, coaching, or organization.

### Player effects

The adjusted-plus-minus candidate correctly detects exact equal-exposure
teammate cohorts and reports prior-conditioned covariance-aware contrasts.
Its production-scale implementation now uses a sparse CSR design,
validation-only hyperparameter selection without per-candidate covariance, a
single bounded post-selection diagnostic pass, and sparse Hessian solves for
requested player contrasts. It never forms a full covariance matrix.

The frozen chronological evaluation used the 16,334 OE maps with complete
role-labelled lineups. The 62 GRID maps in the old downloadable player feed
remain excluded because that artifact stripped the verified completion
provenance needed to admit them. Six OE player-name cells contained leading or
trailing whitespace; the ingest root cause is now normalized and regression
tested. Stable `playerid` is used when present, a missing ID is filled from a
handle only when that handle maps to exactly one ID, and ambiguous missing-ID
handles fail closed.

The player-APM tournament used 11,412 maps through 2026-02-18 for fitting,
2,445 maps through 2026-04-29 for shrinkage selection, and the untouched 2,477
maps after that date for final scoring. The selected penalties were
`player_l2=0.1` and `nuisance_l2=10.0`. On the final test:

| Model | Final log loss | Final Brier |
|---|---:|---:|
| Sparse lineup adjusted-plus-minus | 0.64957 | 0.22848 |
| Dynamic organization Bradley–Terry | 0.63223 | 0.22084 |
| Historical blue-side base rate | 0.69058 | 0.24872 |

The lineup model beats the base rate but loses to the dynamic organization
model. More importantly, 638 of 2,477 test maps contain at least one player
unseen in the fit period (1,711 unknown-player assignments). In the refit
design, 1,064 players fall into 383 exact identical-exposure cohorts, with a
largest cohort of five. Those within-cohort player differences are determined
by the prior, not by map outcomes. The truncated sparse spectrum is reported
only as a lower bound and does not pretend to establish full-rank
identifiability.

**Decision:** no individual player ladder is promoted from team outcomes.
Lineup APM remains a diagnostic/context feature with explicit cohort ties.
Player-specific post-map performance models such as SIDO/PandaSkill-style
candidates are evaluated under a separate descriptive estimand and cannot be
silently substituted for pre-map skill.

### Player-specific early-resource performance

A separate SIDO/PandaSkill-inspired candidate uses no map result or team-win
target. Its narrow estimand is descriptive, role-relative 15-minute resource
performance: the equal-weight mean of training-only robust standardized gold,
experience, and creep-score differentials at 15 minutes. Separate role models
control for champion, team, opponent player, league, and patch context with
sparse ridge shrinkage.

The corrected 2025–2026 population contains 150,610 complete OE player rows
and 75,305 complete same-role matchups. Stable player IDs cover 72,801
matchups (96.67%); 2,905 complete player rows without an ID are excluded
rather than resolved from collision-prone names. The audit found zero target
antisymmetry violations, 47 normalized names mapping to multiple IDs, and one
ID with multiple display names.

The tournament selected both the player model and context-only penalty on the
chronological validation interval. The untouched final test begins
2026-04-01 and contains 35,062 player-role observations:

| Model | Final RMSE | Relative lift vs zero |
|---|---:|---:|
| Player + champion/team/opponent/context | 1.00916 | 6.96% |
| Champion/team/opponent/context only | 1.03070 | 4.98% |
| Zero baseline | 1.08469 | — |

The incremental player-identity RMSE lift over the context-only model is
2.09%. A paired calendar-day bootstrap over 102 days and 5,000 replicates gives
a 95% interval of 1.74% to 2.44%. The future-patch slice has 33,808 rows and a
6.88% lift over zero; the roster-move slice has 7,493 rows and a 6.26% lift.

**Decision:** the research gate passes for this narrow retrospective
early-resource estimand. It may support a separately named “15-minute resource
performance” surface after artifact integration and external review. It is
not general player skill, win contribution, complete-game performance, or a
pre-match probability, and it must not inherit the current player-rating
label.

## Public page consequences

Until the architecture gates pass:

- Ratings must not rank individual players from the present heuristic.
- Current league filters must not claim official membership from appearance
  data.
- Team ratings must not be regenerated from wall-clock pseudo-series.
- Partial Sandbox outputs must not be called win probability or best response.
- Historical match pages must not substitute current ratings for frozen
  pre-match predictions.
- Methodology must not claim series collapse, player aggregation, uncertainty,
  or reproducibility beyond the exact shipped artifact.

## Launch gates

All gates are mandatory:

1. **Identity and competition:** versioned organization, alias, tournament,
   membership, and roster registries with temporal intervals and authoritative
   sources.
2. **Series:** canonical series ledger rebuilt; all 16,396 rating maps reconciled
   or quarantined; no time-bucket identity remains.
3. **Ratings:** team state, lineup projection, and individual performance are
   separate models with separate labels and validations.
4. **Predictions:** immutable pre-event ledger; chronological holdouts;
   calibration, Brier, log loss, AUC, and coverage where relevant.
5. **Draft:** complete-composition and partial-draft estimands separated;
   sequential recommendation policy validated before “best response” language.
6. **Release:** atomic data/model pack with a hard public allowlist, model cards,
   dependency IDs, hashes, and no missing links.
7. **Surfaces:** every number/label mapped to an estimand; functional browser
   checks pass with zero hydration/console errors.

## Literature basis

The architecture uses these papers as methodological guidance, not as borrowed
proof that a Scryglass implementation is valid:

- Duffield, Power, and Rimella (2024), *A State-Space Perspective on Modelling
  and Inference for Online Skill Rating*:
  <https://arxiv.org/abs/2308.02414>.
- Maystre, Kristof, and Grossglauser (2019), *Pairwise Comparisons with Flexible
  Time-Dynamics*: <https://arxiv.org/abs/1903.07746>.
- Macdonald (2012), *Adjusted Plus-Minus for NHL Players using Ridge
  Regression*: <https://arxiv.org/abs/1201.0317>.
- Zhang and Naidu (2024), *The SIDO Performance Model for League of Legends*:
  <https://arxiv.org/abs/2403.04873>.
- De Bois et al. (2025), *PandaSkill — Player Performance and Skill Rating in
  Esports*: <https://arxiv.org/abs/2501.10049>.
- Chen et al. (2021), *Which Heroes to Pick? Learning to Draft in MOBA Games
  with Neural Networks and Tree Search*:
  <https://arxiv.org/abs/2012.10171>.
- Lee et al. (2022), *DraftRec: Personalized Draft Recommendation for Winning
  in Multi-Player Online Battle Arena Games*:
  <https://arxiv.org/abs/2204.12750>.
- Liu et al. (2024), *BPCoach: Exploring Hero Drafting in Professional MOBA
  Tournaments via Visual Analytics*: <https://arxiv.org/abs/2311.05912>.
- Horst, Meyer, and Dörner (2024), *DraftComPromise — On Draft Composition
  Recommendations in League of Legends*:
  <https://doi.org/10.1109/GEM61861.2024.10585636>.
- Dimitriadis, Gneiting, and Jordan (2021), *Evaluating probabilistic
  classifiers: Reliability diagrams and score decompositions revisited*:
  <https://arxiv.org/abs/2008.03033>.
- Gruber and Buettner (2022), *Better Uncertainty Calibration via Proper Scores
  for Classification and Beyond*: <https://arxiv.org/abs/2203.07835>.

Authoritative current-competition sources:

- 2026 League Handbook:
  <https://lolesports.com/en-GB/season/115547545029543948/handbook>.
- LPL Split 3:
  <https://lolesports.com/en-US/tournament/115616254668930796/overview>.
- LCS Split 3:
  <https://lolesports.com/en-GB/tournament/115564797158840434/stage/115564797161986163>.
- LEC Split 3:
  <https://lolesports.com/en-US/tournament/115548681802226458/overview>.
- CBLOL Split 3:
  <https://lolesports.com/en-US/tournament/115565671525288828/stage/115565671525813117>.
- LCK Split 3:
  <https://lolesports.com/en-US/tournament/115548147890329817/stage/115548147896621274>.
- LCP Split 3:
  <https://lolesports.com/en-US/tournament/115570728597462574/overview>.

## Remaining uncertainty

The official tournament pages resolve the current Tier 1 participant set for
this review. A production registry still needs an ingestion policy for:

- mid-tournament withdrawal or administrative replacement;
- temporary emergency substitutions versus roster membership;
- academy/affiliate organizations that appear on season-wide pages but not in
  the current Tier 1 tournament;
- organization renames that retain a competitive slot; and
- the exact effective timestamp of a roster move.

Those are data-policy decisions, not reasons to infer membership from the first
observed map.
