# Data contract

## 1. Canonical identity

Display names are never join keys. L1 owns an append-only identity registry.

| Entity | Stable ID | Required fields |
|---|---|---|
| Player | `player_id` | canonical ID, display name, known aliases, role history |
| Organization | `organization_id` | canonical ID, display name, aliases |
| Roster | `roster_id` | organization, five ordered role/player pairs, effective interval |
| League | `league_id` | canonical circuit, tier, international-connectivity rule |
| Competition scope | `competition_scope_id` | one regional league or one named international event/meta environment, effective interval |
| Tournament | `tournament_id` | league, `season_id`, split/stage, patch source |
| Series | `series_id` | source ID, tournament, participants, scheduled/actual interval |
| Map | `map_id` | source ID, series, map index, actual start, derived `calendar_year`, result |
| Champion | `champion_id` | immutable Riot numeric/key ID, versioned display name |
| Patch | `patch_id` | canonical major.minor competitive patch |
| Draft protocol | `protocol_id` | ordered legal pick/ban slots and side transform |
| Draft state | `draft_state_id` | protocol plus canonical ordered actions and role sets |
| Source snapshot | `source_snapshot_id` | source, retrieval, coverage, content hash |
| Training snapshot | `training_snapshot_id` | ordered source snapshots and filters |
| Model manifest | `manifest_id` | model, source-tree digest, optional/release commit, data, config, calibration, artifact hashes |
| Prediction | `prediction_id` | immutable inputs, estimator, manifest, mode, output hash |

IDs are opaque strings namespaced by issuer, for example `oe:game:...`,
`riot:champion:115`, or `scryglass:roster:<uuid>`. Name normalization creates
aliases, never new identity.

## 2. Time and as-of semantics

All timestamps are RFC 3339 UTC instants.

| Field | Meaning |
|---|---|
| `scheduled_start` | Source schedule time; mutable and not feature availability |
| `event_start` | Best verified actual start of map/series |
| `event_end` | Best verified completion time |
| `source_updated_at` | When source says the record changed |
| `observed_at` | When Scryglass first received this exact record |
| `ingested_at` | When the record entered the warehouse |
| `effective_from` / `effective_to` | Real-world validity interval for roster/taxonomy facts |
| `available_at` | Earliest defensible time the value could enter a forecast |
| `as_of` | Inclusive maximum `available_at` allowed in an artifact |
| `train_cutoff` | Maximum event/availability time permitted in model fitting |
| `created_at` | Artifact or prediction creation time |

Forecast features require `available_at < event_start`. When availability is
unknown, use `observed_at`, not an inferred earlier time. Same-map outcome and
performance rows become available only after `event_end`.

`as_of` is not “latest row date.” It is stored in every response and artifact,
and all source snapshots used must have `max_available_at <= as_of`.

### Artifact and prediction modes

- `mode=state_snapshot`: a current/as-of rating or tier-list artifact. It makes
  no claim that it was sealed before a future event.
- `mode=current_analysis`: an ad-hoc score or sandbox analysis using information
  available at its explicit `as_of`. It is not an event forecast.

- `mode=forecast`: created and sealed before `event_start`; immutable after
  creation. Corrections create a new correction record without replacing the
  forecast.
- `mode=hindsight`: created after the event, may use later metadata/model
  knowledge, and must point to the related forecast if one exists.

Historical pages show both only with explicit labels. A backtest prediction is
`forecast_simulation`, not a historical live forecast, and must replay source
availability rather than merely filter on event date.
`forecast_simulation` is reserved for that historical availability replay; it
is invalid for current ratings, current tier lists, or ad-hoc sandbox analysis.

### Season and calendar year

`season_id` is an authoritative competition label and is never derived from a
date. `calendar_year` is derived mechanically from the UTC `event_start` year
and is used as the default public query window. The two fields may differ. A
calendar-year boundary does not itself reset latent states; carry-over,
boundary-shock, and reset candidates are evaluated under R-01.

## 3. Competition taxonomy

The taxonomy is versioned independently of models.

- `tier1`: circuits whose main-roster teams are structurally able to qualify
  for a designated international event under the rules effective at `as_of`.
- `tier2` / `tier3`: developmental or lower circuits; never shown in global
  Team Rating ranks.
- `international`: designated cross-league events, stored separately by event
  (for example MSI and EWC are not merged into one public filter).
- source league and tournament labels are preserved alongside canonical IDs.

Structural global eligibility does not require the team itself to have played
internationally. It is stored as `structurally_globally_eligible`. Statistical
bridge strength is a separate model diagnostic: league nodes with weak or old
bridges have wider intervals or unavailable global ratings.

Regional scopes are explicit. No UI alias may silently merge circuits. The
calendar year is the default query window.

Every draft/tier artifact conditions on one `competition_scope_id`. For a
regional artifact it resolves to one league environment. For an international
draft it resolves to one named event/meta environment (for example MSI or EWC)
shared by both sides; either team's domestic `league_id` remains provenance and
cannot become the draft-conditioning scope.
Tier-list scope is restricted to one regional league; international/meta tier
lists are outside the initial contract.

## 4. Exact active roster

A valid active main roster has exactly one player for each canonical role:
`top`, `jungle`, `mid`, `bot`, `support`.

Roster evidence has precedence:

1. official league/team registration with an effective date;
2. official match roster or contract database;
3. verified tournament roster;
4. repeated recent starting lineup as a provisional inference.

The chosen precedence and source IDs are stored. Substitutes are separate and
do not enter Team Rating until designated as the active main player. A
single-map emergency substitution does not rewrite the active main roster.

If two players are plausibly active main in one role and no precedence rule
resolves it, Team Rating fails closed with `ambiguous_roster`. Selecting a
known team always resolves its exact active main roster. A hypothetical/custom
five is never Team Rating, never enters a rating ladder, and never inherits the
organization's identity. If a later post-C4 user decision authorizes custom
rosters, the sandbox labels the result **What-if roster context** and keeps it
outside rating artifacts.

Roster activity is registration-based, not “played the last event.” A roster
remains active when registered for the current tier-1 circuit even if it missed
one regional tournament. Participation in the latest local tournament is useful
corroboration, not a game-count rule; `settled` follows the formal
precision/stability/coverage gates. New rosters remain rateable from player
histories.

## 5. Series and map grouping

Use an authoritative `series_id` from source or schedule linkage. If multiple
sources disagree, retain all IDs and resolve through a versioned crosswalk.

Never group by a fixed time bucket. In particular, a four-hour floor can split
long series and merge unrelated series.

If no defensible series ID exists:

- preserve each map as provisional data, without inventing a cluster;
- set `series_resolution=unresolved`;
- exclude it from series-level training targets that require complete series;
- allow map-level training only if the model contract permits it; and
- never guess a best-of result;
- exclude it from primary inferential model comparisons and the primary
  series-preserving dependence-aware resampling; and
- report only descriptive point metrics unless a preregistered, deterministic
  coarser cluster construction is used as a labeled sensitivity analysis.

Resolved `series_id` is an indivisible inner block. The registered primary
design also accounts for recurring participant/team and tournament/time
dependence, adding patch when it is a shared shock. Unresolved maps are never
treated as independent singleton clusters for confidence intervals.

## 6. Patch rules

`patch_id` comes from authoritative competition metadata when available and is
checked against map records. A patch inferred only from date is provisional.

For league \(L\), **current league patch** at `as_of` is:

1. the authoritative announced competitive patch already effective for the
   league, if available and verified; otherwise
2. the patch of the most recent completed eligible league map.

A public tier list is published only after at least one completed eligible map
on that patch, because its membership requires actual league-patch-role play.
Conflicting patch sources fail the refresh rather than mixing cells.

Champion kit/ontology data is versioned by patch. A new patch may borrow
temporal priors, but the fallback appears in provenance and widens uncertainty.

## 7. Draft records

A canonical draft record contains:

- `protocol_id`;
- canonical analytical sides `A`/`B`;
- actual in-game sides `blue`/`red` for each canonical side;
- draft-order positions `first`/`second` for each canonical side;
- the protocol mapping among canonical side, game side, draft order, and legal
  action slots;
- mapping source ID, `observed_at`, `available_at`, and whether each mapping was
  observed or authoritatively reconstructed;
- ordered pick and ban actions with canonical actor side, stable action/pick ID,
  and slot;
- champion stable IDs;
- current role constraints plus append-only role-set revision/reassignment
  history;
- final role assignment for a terminal historical draft;
- one `competition_scope_id`, patch, event, roster, domestic-league provenance,
  and time IDs;
- source and correction lineage; and
- whether action order is observed or reconstructed from an authoritative
  protocol.

Each role revision stores previous/new sets, action-sequence position,
reason/source, and availability. Sets may legally widen, narrow, or reassign;
the original pick record is never rewritten. Final roles may be resolved in
hindsight from match records. The immutable forecast retains the role state and
history available at forecast time. Terminal states are valid canonical records
and must delegate to the terminal Draft Score path.

Duplicate champions, impossible role matchings, illegal action order, mixed
patches, or incomplete required actions are validation failures.

## 8. Player-performance and policy features

Every feature specification records:

- input source fields and units;
- role/champion/opponent/draft adjustment;
- transform and missingness policy;
- `available_at` rule;
- whether it updates player skill, team policy, or an explicitly identified
  joint measurement model;
- support-appropriate interpretation;
- known coverage changes; and
- leakage audit.

Post-map outcomes may update the next state but are not features for the same
forecast. Missing features are represented as missing with an explicit model
path; zero is never used unless zero is a valid observed value.

Resource allocation and non-resource impact are separate feature families.
Policy training must include at least one registered non-resource support
channel or use a broad equal-role prior; it may not infer low support importance
from farm. An endogenous resource aggregate cannot independently update both
player skill and policy; shared use requires the registered temporally ordered
joint model and double-count/collider sensitivity under R-03.

## 9. Staleness and missingness

Each source class has a freshness service-level objective selected from observed
publication cadence and recorded in the model manifest. A number is stale when
`as_of - source_updated_at` exceeds that source's registered limit or a newer
source revision is known but unprocessed.

Required inputs:

- model manifest and calibrated artifact;
- stable entities;
- event time and patch;
- legal draft protocol/state for Draft Score;
- exact roster and player-context snapshots when contextual mode was requested;
- current league patch for a tier list; and
- all lineage hashes.

Every successful output requires provenance
`required_input_status=complete`, all required freshness checks true, and no
missing/stale/conflict flag. Missing, stale, or conflicting required input
returns an unavailable error and no principal numeric payload. Optional
performance channels may use a registered missing-data model, but the fallback
and evidence impact are emitted.

## 10. Source publication matrix

L1 creates one row per source-artifact class using
[`publication-matrix.schema.json`](contracts/publication-matrix.schema.json).
The row includes:

- owner, source URL, access method, terms/license review date;
- raw/derived/code/weight/aggregate artifact class;
- credential requirement and secret locations;
- allowed storage, redistribution, retention, and attribution;
- public, authenticated, private, or prohibited decision;
- field allowlist/denylist and de-identification rule;
- derivative-reconstruction risk;
- reviewer and decision evidence; and
- next review date.

Default is `private_pending_review`. A public aggregate or authenticated
product does not imply the raw source, licensed rows, code, or weights are
public. Private credentials and recoverable secrets are always prohibited from
public artifacts, logs, schemas, examples, and source maps. After C4, measured
costs and the completed matrix go to an explicit user decision; no code, data,
or weights are published before that approval.

The initial matrix must cover Oracle's Elixir, Riot Data Dragon/API, any Riot
esports/live feed considered for a future separately approved phase,
Leaguepedia/Fandom, GRID or partner feeds,
manually authored ontology, Scryglass derived features, model code, model
weights, evaluation reports, and user/auth records.

## 11. Snapshot and artifact lineage

A training snapshot is content-addressed and immutable. Its manifest records:

- ordered source snapshot IDs and SHA-256 hashes;
- schema/taxonomy versions;
- inclusion/exclusion filters;
- row counts by year, league, patch, tier, and source;
- minimum/maximum event and availability times;
- duplicate, correction, missingness, and identity audits;
- train/validation/holdout assignment IDs; and
- deterministic `source_tree_sha256`, optional candidate code commit, and
  environment lock hash.

`source_tree_sha256` is defined by **Scryglass source-tree hash v1**:

1. start from an explicit allowlist of regular files that can influence the
   build, expressed as normalized repository-relative POSIX paths; reject
   absolute paths, `..`, duplicate paths, symlinks, and `.git`;
2. encode each path as UTF-8, keep file contents as exact raw bytes, and sort
   entries lexicographically by path bytes;
3. concatenate the ASCII prefix `scryglass-source-tree-v1` followed by a zero
   byte, then for every entry append
   `uint64_be(path_byte_length) || path_bytes ||
   uint64_be(content_byte_length) || content_bytes`; and
4. store the lowercase SHA-256 digest of that byte stream.

The allowlist is stored beside the manifest. Generated outputs and the manifest
itself are not source-tree entries. A candidate manifest may have no commit (or
`code_commit: null`) but must carry this deterministic digest. A promoted
manifest requires the 40-hex commit containing that exact source tree; its
top-level digest and lineage digest must match.

A model manifest points to one training snapshot, one split plan, one exact
estimator configuration, one calibration artifact, evaluation report hashes,
and every runtime artifact hash. A browser/runtime artifact is derived from the
same manifest and must prove numerical parity.

Public files are generated from an explicit allowlist. “Latest” pointers may
move only after promotion; versioned artifacts never change in place.

## 12. Data corrections

Corrections append a new source snapshot and cross-reference superseded records.
Forecasts remain immutable. Models may be rebuilt, but the new manifest and
hindsight output must show the corrected lineage. Silent mutation of a
published pack or prediction is prohibited.

## 13. Removed and prohibited fields

The grubs 24% result and derived labels are excluded from v2 source, training,
publication, and article manifests. Internal wagering/market features and
credentials are prohibited. A public pack does not inherit the broad v1
allowlist; L1 rebuilds it from the source publication matrix.
