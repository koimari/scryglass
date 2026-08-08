# Private decision-support readiness — 2026-08-02

Status: **ratings-, Draft-core-, GRID-source-capture-, outcome-free phase-one
collection-, prospective market-protocol-, and sealed phase-two evaluation-
implementation-ready; not rating-ready, Draft-ready, probability-ready, or
betting-ready**

This is an implementation checkpoint, not a promotion record. Public Scryglass
remains a non-betting research publication. No artifact below authorizes a
player rating, team rating, match probability, fair price, expected value,
recommendation, or wager.

## Current verification checkpoint

After the fresh-refit integration freeze, the complete private market test
directory passed `135/135`. The current v3 prediction-ledger,
terminal-Draft, and private-readiness selection passed `106/106` with two
deprecation warnings; the full future-ledger file passed `30/30`. The earlier
TypeScript terminal-Draft/public-boundary selection passed `30/30` and was not
rerun for this Python-only integration. Both registered readiness artifacts
and every old/new link in the latest supersession record replayed their exact
raw and canonical hashes, with every authority flag still false.

The deliberately broader all-generations ratings/Draft run was not green:
`366` tests passed, `6` failed, and `10` errored. The rating failures are
fail-closed evidence, not current v3 test regressions. Legacy development and
v2 artifacts pin earlier warehouse bytes, while the current warehouse hashes
are `04c0cce1d86a4358d9eeb5937f61d5288358953e66c693a1ce88b0b650295d08`
for `maps.parquet` and
`12f1cca978d683a0df8ceec0772999aeb03c723b4465f98674247f327dea71fa`
for `players.parquet`; the semantic output contract trust root is also
currently unanchored. Those artifacts must not be silently rebound to mutable
files merely to clear tests.

The latest ratings-only broad run reproduced that boundary: `293` passed,
`6` failed, and `10` errored. Fifteen failures/errors are the immutable legacy
source locks rejecting current warehouse bytes; the remaining failure is the
unanchored output-contract trust root. After the readiness changes, the focused
Draft, phase-one, and private-readiness selection passes `19/19` in `97.93s`.

The live private-readiness audit remains blocked and explicitly reports
`betting_ready: false`: `26` ratings blockers, `14` terminal-Draft blockers,
`15` match-winner-market blockers, and `6` live-total blockers (`61` total).
The fresh-refit/full-uncertainty integration blocker is now cleared by exact
slow/fast replay; no probability or betting authority was granted. The next
evidence-producing milestone is prospective joint ratings-plus-Draft capture
strictly after the registered `2026-08-03T00:00:00Z` boundary, followed by the
predeclared support floors and independent opening/evaluation reviews. Passing
implementation tests cannot substitute for that cohort.

## Clock-corrected ratings evidence

The operative future boundary is `2026-08-03T00:00:00` in the warehouse's
timezone-naive source-time semantics. The frozen candidate remains
`hierarchical-orgw100-orgv025-retain100`; its selection was adaptive and is not
independent validation.

| Evidence | UTC time | Raw SHA-256 | Canonical artifact SHA-256 | Meaning |
|---|---:|---|---|---|
| Corrected source preflight v3 | `2026-08-01T23:50:57Z` | `0e004e7561e1c4a9e44138e1ed6e9ac6a1987b0db267f92c0852618805eb876f` | `0b38cd8ddf57ceb7e334ad71f4e3846a387c3ea3df3231b09e8fbc7a9852afc8` | The frozen source can fit and replay numerically; it is adaptive rehearsal only. |
| Future protocol v3 | `2026-08-01T23:54:10.693949Z` | `c76a6b47079f757d62bfaba293b155da40c58cd11f618162ac67e188c080ded9` | `cfbee3194d65abcb3acf41552a8925a2c58e81e5b3914220bf4a07093f211731` | Empty, system-clocked future protocol; no future target was opened. |
| Capture readiness v3 | `2026-08-02T00:05:12.497121Z` | `a9b67bd8cc8dd441d1c5ca51e8ce2326640a1f29ffee3ff0510b7b2c68e7fb48` | `a5f0fe588aba163693ef54663cb39b10d2a77e0e05f4c7855e9b76346b1d3660` | The prediction and ledger paths sample UTC internally and expose no user timestamp argument. The ledger is empty. |

The timing-failure receipt remains part of the lineage. It rejects
`source-preflight-v2.json`, `future-protocol-lock-v2.json`, and
`capture-readiness-v1.json` because each file existed before its declared build
or lock time. `capture-readiness-v2.json` was subsequently superseded because
the prediction and ledger builders still accepted caller timestamps. None of
those receipts qualifies as future evidence.

The legacy development artifact still records hashes for warehouse files that
have since changed, and that mismatch remains visible. It is no longer treated
as a permanent promotion blocker because the operative v3 path does not replay
from those mutable locations: the corrected source snapshot is immutable and
registered, its fit/replay preflight passes, the candidate and future boundary
are preserved, and pre-event capture is locked to that path. If any of those
prospective bindings fails, the legacy source mismatch becomes blocking again.
Separately, the seven current public-output schema files do not match the
frozen contract trust root. That discrepancy is now an explicit readiness
blocker; the current bytes were not rehashed into authority without review.
A non-authorizing reconciliation candidate validates all nine current schemas,
unique schema IDs, and all five current examples while recording seven changed
runtime schemas and two auxiliary schemas. It does not activate those bytes.
Independent review remains blocked until the exact prior 25-file contract tree
and a complete candidate-anchor semantic replay are supplied, reviewed by two
distinct reviewers, and pinned outside the repository. Exact candidate time and
hashes live in the candidate itself rather than this contract-content tree, so
documenting a new freeze cannot invalidate the freeze it describes.

## Corrected terminal Draft Score lineage

The terminal model now has a stable, scientifically narrower development path.
The mutable warehouse is no longer the evaluation input: exact canonical bytes
freeze `6,194` complete maps in `2,871` outcome-free dependence clusters and
exclude `31,724` maps that lack a cluster assignment. The payload SHA-256 is
`e8105027c7b2406f735cdcca592e60071ed25fc6489f4966a663dd91bea4a7d8`;
the manifest raw SHA-256 is
`f395e0da4c138eaaaae6a749359e866a9f4fa71cfd24f334e104e2b458a74e48`.

The v1 evaluator's ridge expression effectively weakened regularization as the
sample grew. V3 instead minimizes mean logistic loss plus an explicit,
sample-size-invariant ridge penalty and leaves the pre-event strength nuisance
coefficient unpenalized. More importantly, it no longer judges an equal-team
draft index directly against outcomes from unequal real teams. Candidate
selection asks the identified question: does frozen pre-event context plus
draft improve on the same-input context-only model?

The adaptive development selection is
`m0-role-additive@ridge-0.05`. Across the three chronological validation folds,
its mean deltas versus context-only were `-0.005127` log loss and `-0.002152`
Brier (negative is better), with both metrics nonpositive in all three folds.
Nested outer tests were less stable: two folds improved, while one was slightly
harmful (`+0.000218` log loss and `+0.000039` Brier). This is evidence to keep
testing, not evidence to promote.

| Evidence | UTC time | Raw SHA-256 | Canonical artifact SHA-256 | Meaning |
|---|---:|---|---|---|
| V3 development evaluation summary | n/a | `4c5f71cbe9f5730cdebb1d4974b185e6c10d914235c6cbf4d025e6d866056f80` | n/a | Replays from the frozen clustered payload; one of three nested outer folds is harmful. |
| V3 neutral model bytes | model as-of `2026-07-18T16:33:48Z` | `d6ddfbd0238ff5b9d1b259e393b1724717bcb6dd9d7e407d92fc44395da6d6c3` | n/a | Equal-strength composition index only; not a directly outcome-calibrated win probability. |
| V3 candidate registry | `2026-08-02T00:35:55.995070Z` | `5fb9aebb0723b68c182b0bd556fcb9a20dab4233663b30b4c5421efa31c57895` | `0cdf6c2c4fd4a958c0cca851677239f06fb8711e644ca63396af353b0a9dc9c0` | Adaptive candidate frozen before the future boundary; no authority. |
| Draft future protocol v1 | `2026-08-02T00:37:19.269797Z` | `040efa545363c9c71facaaea88bbd2b3bb80bc56d204e79b5c7f6703e54bb299` | `0157d8bdba5aa57fc3241536b9f3dde919606513c22093c9969e22f6eba43925` | Empty incremental-evaluation protocol sharing the ratings boundary `2026-08-03T00:00:00Z`. |
| Draft capture readiness v1 | `2026-08-02T01:04:59.073962Z` | `c57be9bce656f6739184a69da5e49b9a4d7b75063e30caf46d136abf182ad183` | `3f92d35dbad0851c2be27a97db5e78cabc80a4fd7206eea9df3085fe4120cbf2` | System-clocked prediction, actual-map-start receipt, and receipt-bound ledger paths are locked. The ledger has zero entries and no authority. |
| GRID source readiness v1 | `2026-08-02T01:27:57.849474Z` | `1f06b057adac8c0e35c15dfed40cca01681bf34ab7be61dc780b0b968ae1440c` | `2aa23f24e281fc34926e96851ce5aff36046c31ebef857b41bc8910e51cc732c` | Five hash-matched completed archives each contain action slots 1–20 before map start, with 66.871–135.385 seconds observed from the last draft action to start. This is retrospective shape evidence only. |

The old app draft family's known adaptive harm remains bound and quarantined;
it was not relabeled as a pass. The new prospective target combines the exact
frozen ratings forecast with the exact frozen v3 draft term, then compares it
with that same ratings forecast without draft.

## Prospective match-winner market protocol

A separate two-stage Betano Brazil map-winner protocol was system-clocked and
locked before the shared `2026-08-03T00:00:00Z` future boundary. It cannot use
the ratings/Draft holdout as both final validation and a market test:

1. Phase one keeps the existing ratings and terminal-Draft outcomes sealed and
   evaluates them under their already registered stopping and opening rules.
   Both must independently pass. There is no fallback candidate if either
   fails.
2. Only after that pass may the predeclared bounded logistic recalibration and
   full-pipeline series-cluster bootstrap be fit and independently pinned.
3. Phase two begins strictly afterward on a disjoint future cohort. The model
   prediction must precede a contemporaneous open Betano quote, and both must
   precede actual map start. Outcomes remain sealed while metadata support is
   counted.

| Evidence | UTC time | Raw SHA-256 | Canonical artifact SHA-256 | Meaning |
|---|---:|---|---|---|
| Match-winner future protocol v1 | `2026-08-02T01:46:21.765186Z` | `b82ff719c02cc3aac81009e57cad0f6d0417519e7d02147de7c8bd3d78d5437a` | `c0ef32affcd73b682ed2ea973bbad24aa51b6efa667bfdb419b65d668494376f` | Empty, two-stage prospective market evaluation; no authority. |
| Phase-one collection readiness v1 | `2026-08-02T02:37:41.590661Z` | `7ca30d0b87ffea8b2e5e329e99a86942a8f6f1ff70c055fa39fce3c5acfa9b50` | `4c1abb28525130b17852071b9d64737bc78806c6a9a43419bd49670fd16f6629` | The exact plan, event-bundle, and joint-ledger implementation was system-clocked before the future boundary with zero plans, bundles, snapshots, or outcomes. |
| Quote-capture sub-contract | n/a | n/a | `06dc090de7c93d2625bc78c3ba7163eafda598e3d14deaea3f325c829cb55c68` | Exact response bytes, deterministic extraction, open two-way market, source transport timing, and pre-start ordering are required. Generic builder time explicitly does not count as transport time. |
| Settlement sub-contract | n/a | n/a | `6f1e885f7d49ee27555bdad7babb9579c8ee9b5057951e728957d44dbe253405` | Single-map cash-odds shadow settlement only; ambiguous remakes, forfeits, postponements, or rule conflicts are unavailable until exact bookmaker terms resolve them. |
| Public Betano terms snapshot v1 | `2026-08-02T02:13:59.564289Z` | `96f5c3228f7b8b1804cd764012677f912a2654da00f3684ec5d1a37bd3a45255` | `56407922787b0a954d90447a6ebb10c164005de74d8ad214c0c2520a53b7d9d5` | Exact system-clocked JSON bytes for the two public help articles. It is deliberately incomplete and has no independent alignment or settlement authority. |
| Betano quote-adapter candidate v1 | `2026-08-02T03:12:24.833831Z` | `91cad8e250ac34ce612255db81fb8fee2ec2c0ecb3afa9030a898925500a4001` | `17bd5cb9f8c2031ed409317b19855bbf200e8254ad9cc19e7cf8af1995705334` | Source-hashed public pre-event HTML transport and deterministic `window["initial_state"]` map-winner extraction. It uses a fresh unauthenticated Brave profile and pinned Playwright CLI, persists no request headers/cookies/credentials, contains no quote, and is not independently registered. |

The phase-two metadata floor is `500` quoted maps in `125` series, including
`75` maps from each of LCS/LEC/LCK/LPL, `50` international maps, three future
patches, `100` latest-patch maps, `50` roster-change maps, `50` sparse/new
player-or-champion maps, at least `80%` quote coverage, and `100` predeclared
shadow-policy signals. If the signal floor is still unmet at `1,000` quoted
maps, the test fails for insufficient support. These are frozen metadata floors,
not a post-hoc power claim.

The opened evaluation must compare the recalibrated ratings-plus-Draft model
with both the no-vig two-way market and the identically recalibrated
ratings-only model. Paired series-cluster bootstrap upper 95% bounds for log
loss and Brier deltas must be nonpositive, with at least one market delta
strictly below zero. Calibration, supported-stratum, quote coverage, timing,
extractor replay, and a fixed one-unit shadow-policy ROI gate must all pass.
Quoted shadow return is explicitly not proof that a price was executable or
that any stake was accepted.

Betano's public [LoL help](https://support.betano.bet.br/hc/pt-br/articles/34703909314589-Como-se-joga-o-E-sport-League-of-Legends-LoL)
identifies “Vencedor do Mapa” as the team that wins the map, while its general
[cancellation help](https://support.betano.bet.br/hc/pt-br/articles/6414148470301-O-que-acontece-se-o-jogo-em-que-apostei-for-adiado-ou-cancelado)
describes same-day resumption and refund handling. Those pages do not resolve
every esports remake, forfeit, or market-specific edge case. Therefore an exact
versioned bookmaker-terms snapshot and independent alignment review remain
mandatory before phase two; the current settlement contract does not invent
the missing cases.

## What is implemented

- Exact source, candidate, comparator, roster, patch, fixture, side, and player
  bindings are required for a prediction receipt.
- The candidate and both locked comparators are replayed from the immutable
  source for every captured event.
- Prediction and ledger timestamps are sampled inside their builders. The
  prediction CLI has no `--captured-at` option.
- Receipt validation rechecks ordering against the protocol, roster evidence,
  patch evidence, and event start. Ledger validation rechecks creation order.
- Event outcome fields are recursively rejected from input receipts,
  predictions, and the outcome-free ledger.
- Terminal-draft capture binds the exact ratings receipt, terminal assignments,
  source payload bytes, rights review, pick/ban validation, sides, patch, and
  frozen Draft artifact. A separate outcome-free receipt supplies authoritative
  actual map start; a prediction counts only when its system-clocked capture is
  strictly earlier.
- GRID WebSocket messages can now be persisted as exact-byte, SHA-256-bound,
  fsynced envelopes whose receive time is sampled inside the receiver. The API
  key is never included in the stored endpoint or envelope.
- One-shot GRID collection can stop immediately after action slot 20 for an
  exact game, then reconnect into a separate file and stop immediately on that
  game's `series-started-game` event. The draft adapter refuses a prediction if
  the target start was already received; the map-start adapter refuses a log
  that continues past the start transaction.
- The observed pre-start GRID events establish pick/ban order, champion, team,
  game, and blue/red side. They do **not** establish final player roles. The
  adapter therefore requires a separate reviewed ten-role assignment and fails
  closed on any missing, duplicate, ambiguous, or nonmatching assignment.
- Draft source and actual-map-start payloads must be strict outcome-free JSON.
  Standalone ledger validation reloads every declared receipt and verifies its
  exact canonical hash instead of trusting copied ledger metadata.
- Player and exact-roster diagnostics preserve unavailable lineup-synergy and
  team-policy components as unavailable/null, never zero.
- Post-validation probability inputs reload immutable fresh rating-source
  bytes under the exact event cutoff and 48-hour availability embargo. The
  slow 2,000-draw bootstrap, precomputed rating draws, fast terminal-Draft
  completion, combined point, and rating-only comparator share the same exact
  refit, ten-player roster, patch, and independent phase-one registry binding.
- Private ratings readiness no longer reads support or entry counts from the
  immutable empty protocol/capture-readiness receipt; it reloads and
  semantically validates the live receipt-bound ledger. The already frozen
  joint phase-one opening path binds the exact first support-met joint snapshot,
  requires separate ratings and terminal-Draft reviews, writes its marker
  before reading outcomes, and runs the predeclared one-time evaluator.
- Writes are no-clobber, source files are hash-bound, and deterministic replay
  is tested.
- Bookmaker quote receipts now embed the exact source body and a separate
  deterministic extraction payload. Event, market, settlement, both prices,
  capture-contract hash, settlement hash, extractor ID, and extractor source
  hash must replay. The CLI accepts no capture timestamp.
- The generic quote builder records honestly that its own system-clock sample
  is not network transport time. The Betano-specific candidate now surrounds
  the public document request with system UTC and monotonic clocks, embeds the
  exact response body once, and replays the source extraction into the generic
  quote receipt. An independent adapter registry and a fresh per-event receipt
  strictly after phase-two opening are still required.
- The adapter rejects redirects, non-200 or non-UTF-8 HTML, duplicate embedded
  state, duplicate JSON keys, event/league/team/map drift, any event/market/
  selection suspension, invalid prices, fewer or more
  than two selections, market close within five seconds, forbidden outcome
  fields, prediction-after-request timing, and prediction-to-response latency
  above 30 seconds. It never logs in, clicks a price, opens a bet slip, or
  submits credentials. It does not claim that a visible quote was executable
  or free of promotion/boost/limit constraints; that remains a separate review.
- Direct caller-supplied probabilities and intervals are diagnostics only. The
  decision layer can use a probability for fair odds or EV only when the exact
  event receipt appears in an independently pinned registry and its model,
  market protocol, calibration, uncertainty, source-prediction, and generation
  code hashes all match the separately approved market authority.
- Event-probability receipts replay bounded logistic recalibration, require at
  least `2,000` full-pipeline series-cluster bootstrap draws, bind the draw-set
  hash, and preserve the interval as epistemic rather than claiming binary
  outcome coverage. Neither the receipt nor its identity registry grants
  probability or betting authority.
- The two relevant public Betano help-center API responses are captured as
  exact base64 bytes with SHA-256, official article identity, update/edit
  timestamps, request/response clocks, and deterministic wording checks. The
  receipt enumerates six unresolved settlement/execution topics and cannot
  relabel itself as complete.
- Phase-one collection now has a pre-boundary, hash-pinned empty readiness
  receipt plus an immutable event plan created from the exact
  persisted ratings receipt, an event bundle that joins those exact bytes to
  the terminal-Draft and actual-map-start receipts, and one joint snapshot
  that rebuilds both registered child ledgers. All three reject outcome fields,
  sample their own UTC clocks, use no-clobber/fsynced writes, and keep every
  authority flag false.
- The event bundle refuses identity drift across fixture, series, game, league,
  patch, side, and organizations. It also decodes the ratings bytes embedded in
  the Draft receipt and requires byte-for-byte equality with the separately
  persisted ratings artifact; matching copied hashes alone are insufficient.

The production capture command is intentionally timestamp-free:

```bash
python3 -m lol_kills.v2.ratings.player.multileague_v3_prediction_ledger \
  --roster-receipt data/lol/private_pregame_rosters/receipts/EVENT.json \
  --patch-receipt data/lol/v2/evaluation/patch-receipts/EVENT.json \
  --series-id SERIES_ID \
  --game-number 1 \
  --out data/lol/v2/evaluation/multileague-v3/predictions/EVENT.json
```

This command creates evaluation evidence only. Its output must be externally
pinned and independently reviewed before it can count toward an opening.

The terminal Draft capture CLI is also timestamp-free. Capture occurs only
after all ten terminal assignments are verified:

```bash
python3 -m lol_kills.v2.draft.terminal.future_prediction_ledger capture \
  --ratings-receipt data/lol/v2/evaluation/multileague-v3/predictions/EVENT.json \
  --draft-metadata data/lol/v2/evaluation/draft-terminal-v1/input/EVENT.json \
  --draft-source-payload data/lol/v2/evaluation/draft-terminal-v1/source/EVENT.json \
  --out data/lol/v2/evaluation/draft-terminal-v1/predictions/EVENT.json
```

This first receipt remains pending. After the map begins, a separate
`map-start` capture binds outcome-free authoritative actual-start evidence; the
`ledger` command accepts only paired receipts whose prediction time is strictly
before that actual start.

For prospective private GRID capture, the source-specific adapter is the
preferred path. First collect a dedicated log that ends at terminal action 20:

```bash
python3 -m lol_kills.etl.grid_series_events \
  --series-id GRID_SERIES_ID \
  --seconds 0 \
  --full-state \
  --reconnect \
  --receipt-envelopes \
  --stop-after-draft-game-id GRID_GAME_ID \
  --out data/lol/v2/evaluation/draft-terminal-v1/grid/EVENT-draft.jsonl
```

The capture context must bind the exact event, GRID series/game and team IDs,
blue/red organizations, patch, reviewed provider fixture, and a separately
reviewed role-to-champion assignment for all ten picks. Once that context and
the already-created ratings receipt are available, create the pending Draft
prediction directly from the receipted log:

```bash
python3 -m lol_kills.v2.draft.terminal.grid_future_source_v1 capture \
  --receipt-log data/lol/v2/evaluation/draft-terminal-v1/grid/EVENT-draft.jsonl \
  --context data/lol/v2/evaluation/draft-terminal-v1/context/EVENT.json \
  --ratings-receipt data/lol/v2/evaluation/multileague-v3/predictions/EVENT.json \
  --out data/lol/v2/evaluation/draft-terminal-v1/predictions/EVENT.json
```

Then reconnect from the collector's printed last sequence into a new file and
stop on the exact target start:

```bash
python3 -m lol_kills.etl.grid_series_events \
  --series-id GRID_SERIES_ID \
  --from-sequence LAST_DRAFT_TRANSACTION_SEQUENCE \
  --seconds 0 \
  --full-state \
  --reconnect \
  --receipt-envelopes \
  --stop-after-start-game-id GRID_GAME_ID \
  --out data/lol/v2/evaluation/draft-terminal-v1/grid/EVENT-start.jsonl

python3 -m lol_kills.v2.draft.terminal.grid_future_source_v1 map-start \
  --receipt-log data/lol/v2/evaluation/draft-terminal-v1/grid/EVENT-start.jsonl \
  --context data/lol/v2/evaluation/draft-terminal-v1/context/EVENT.json \
  --out data/lol/v2/evaluation/draft-terminal-v1/map-start/EVENT.json
```

The adapter embeds only normalized, outcome-free fields and hashes; it does not
embed GRID's raw full-state start message. These commands create evaluation
evidence only and still grant no decision authority.

Once the ratings receipt exists, create its immutable phase-one plan at the
path the plan itself derives from the ratings filename:

```bash
python3 -m lol_kills.v2.market.phase_one_collection_v1 plan \
  --ratings-receipt data/lol/v2/evaluation/multileague-v3/predictions/EVENT.json \
  --out data/lol/v2/evaluation/match-winner-market-v1/phase-one/plans/EVENT.json
```

After the exact Draft and map-start receipts are persisted, assemble the event
bundle. The command reloads and semantically validates all four inputs:

```bash
python3 -m lol_kills.v2.market.phase_one_collection_v1 bundle \
  --plan data/lol/v2/evaluation/match-winner-market-v1/phase-one/plans/EVENT.json \
  --out data/lol/v2/evaluation/match-winner-market-v1/phase-one/bundles/EVENT.json
```

A metadata-only snapshot manifest contains only bundle locators, for example
`{"bundle_locators":["data/lol/v2/evaluation/match-winner-market-v1/phase-one/bundles/EVENT.json"]}`.
Build one immutable joint snapshot from it:

```bash
python3 -m lol_kills.v2.market.phase_one_collection_v1 snapshot \
  --bundle-manifest data/lol/v2/evaluation/match-winner-market-v1/phase-one/bundle-manifest.json \
  --snapshot-locator data/lol/v2/evaluation/match-winner-market-v1/phase-one/snapshots/SNAPSHOT.json \
  --out data/lol/v2/evaluation/match-winner-market-v1/phase-one/snapshots/SNAPSHOT.json
```

The snapshot contains the ratings and Draft ledger candidates built by their
registered implementations at one system-clock sample. It is still unreviewed,
unopened, and non-authorizing; an external digest pin and the registered
opening process remain mandatory.

## Phase-one evaluation and opening freeze

The outcome-opening and evaluation implementation was last re-frozen before the
future boundary at `2026-08-02T08:54:42.439430+00:00`. An earlier executable
outcome-free rehearsal found that the TypeScript replay used the ratings
series start instead of the ledger-bound actual map start. The corrected
replay now uses the map-specific start receipt. The earlier readiness bytes are
preserved under `phase-one/superseded/`, and the exact old/new hashes and reasons
are bound in the three `readiness-supersession-*.json` records. The latest
supersession also makes evaluator roster-entity extraction an explicit
fail-closed boundary: exactly two teams, five players per team, ten distinct
nonempty player IDs, and two distinct nonempty organization IDs are required
before the participant-aware sensitivity can run.

- readiness raw SHA-256:
  `78776f27ddb772921e36019cb96f8b82bb967fe3f75b739399f628d793d566d8`;
- readiness artifact SHA-256:
  `dfd19ad7578c8b862ff9ab7f90ba3b554e9c3ded67421f6eb748cd1ebf649108`;
- ratings bootstrap: `10,000` whole-series resamples, base seed `20260803`;
- Draft bootstrap: `10,000` whole-series resamples, base seed `20260804`;
- point estimates are map-weighted while uncertainty resamples complete series;
- Draft Python/TypeScript replay must cover every snapshot map with maximum
  absolute probability difference at most `1e-12`;
- the existing whole-series bootstrap is supplemented by a required
  entity-network HAC sensitivity: maps may covary when they share a series,
  exact player, or organization; it requires at least 20 series and 50 players,
  uses a predeclared `2.1` critical value, and both log-loss and Brier upper
  bounds must remain nonpositive;
- exactly two distinct independent opening reviews are required, one for the
  ratings holdout and one for the terminal-Draft holdout;
- the opening authority bytes require an out-of-band SHA-256 pin; and
- a durable no-clobber consumption marker is written before the first outcome
  read. A marker or result blocks a second opening, including after a crash.

The v3 ratings protocol required reliability to pass a “locked gate” but did
not define that gate numerically. Before outcomes existed, the evaluator fixed
that omission: the candidate calibration-intercept percentile interval must
include zero, its calibration-slope interval must include one, and the upper
95% ECE delta versus each frozen comparator must be at most `0.01`. This is a
pre-opening clarification, not a post-outcome threshold choice.

An evaluation result still has no authority by itself. Two further independent
result reviews must reproduce the registered seeds, resamples, strata, gates,
and exact opening/snapshot/outcome/result hashes. Their externally pinned
registry records either a pass or a terminal failure. A pass only permits the
separate recalibration, uncertainty, and phase-two preparation process; it does
not open phase two or authorize probabilities, odds, EV, recommendations, or
wagers.

## Post-pass recalibration and uncertainty freeze

The post-pass probability-pipeline implementation was separately frozen while
phase-one outcomes, evaluation results, recalibration artifacts, and event
uncertainty artifacts were all still absent. It was last re-frozen after the
fresh post-validation rating refit was wired through every rating draw and
point prediction at `2026-08-02T10:04:31.430181+00:00`:

- readiness raw SHA-256:
  `c383dd0ed7e03fa1cc077b410e60899e92132ff21d08c17bfb4e9215e7e9be55`;
- readiness artifact SHA-256:
  `c8358ac6d1e104745d162fbeb430a742a20a1efb474cfe20e41ee51dcf81fb16`;
- recalibration is one bounded L-BFGS-B fit for the combined model and an
  identical independent fit for the rating-only comparator, using unweighted
  phase-one map log loss, intercept bounds `[-2, 2]`, slope bounds
  `[0.25, 4]`, and no phase-two online update;
- event uncertainty uses exactly `2,000` series-cluster bootstrap draws with
  master seed `20260805` and percentile bounds `[0.025, 0.975]`;
- every draw resamples and refits the historical rating state, Draft training
  and calibration series, and phase-one recalibration;
- rating and Draft target predictions are recomputed in every draw, while the
  prospective phase-one predictions already captured before their maps are
  the fixed inputs resampled for the recalibration refit; and
- target-event outcomes and bookmaker prices are forbidden from the interval.

The frozen phase-one source remains correct for an independent model test, but
it is not an acceptable long-lived deployment state. A new post-pass refit
implementation therefore requires exact immutable fresh Parquet bytes, the
unchanged phase-one model family and hyperparameters, a strict target-event
cutoff, the model's 48-hour series-availability embargo, a maximum data age of
14 days, exact pre-event roster and patch receipts, and the independently
registered phase-one pass. It emits exact player ratings, roster-retained
player-plus-organization team strength, and a blue-minus-red interval retaining
cross-team posterior covariance. Lineup synergy and team policy remain null.
It emits no match probability, odds, EV, recommendation, stake, or authority.
The refit binds not merely the phase-one result but the exact externally pinned
independent registry locator, raw digest, registry ID, and registration time;
a refit timestamp before that registration is rejected.

The full-pipeline binding is now recorded as
`wired_replayed_and_independently_reviewable`. The slow bootstrap, precomputed
rating leg, fast bootstrap, combined point prediction, and rating-only
comparator all consume the exact fresh refit. The point calculation replays the
fresh rating probability and combines it only with the terminal Draft logit;
the historical phase-one-era rating point is no longer an input. Exact event,
roster, patch, source, phase-one registry, and chronology drift fail closed.
The superseded readiness bytes and this stronger contract are bound in
`readiness-supersession-fresh-refit-full-pipeline-v1.json`.

The freeze pins the Python, NumPy, and SciPy runtime identity plus every local
source module that affects those refits. A real-component smoke replay passed
against fresh immutable rating bytes and the actual Draft development snapshot,
and the slow and fast draw matched exactly. This establishes implementation
identity only: the 2,000-draw run
cannot occur until phase one independently passes, and its future artifacts
still require external reproduction and registration before phase two.

The independent registration formats are also implemented but deliberately
unissued. Recalibration/uncertainty registration requires two different
reviewers: one must exactly recompute both bounded fits; the other must replay
all 2,000 draws, seeds, sample digests, refits, probabilities, and percentile
bounds. Its verification target must be captured after recalibration and be
excluded from both phase one and phase two. The registry's exact bytes still
need an out-of-band digest.

Likewise, the incomplete public Betano help-page snapshot can no longer become
“reviewed” merely because a file and environment variable exist. The complete
terms authority requires exact Betano Brazil evidence bytes under the private
evidence prefix, hashes, source URLs, capture/effective times, complete coverage
of map-winner, non-start, postponement, cancellation, remake, resumption,
forfeit, void, refund, and cash-odds rules, and two distinct source/alignment
reviews. It grants settlement-rule identity only, never betting authority.

## Phase-two operational path partially implemented, not opened

The original time-critical phase-two path can precompute rating-history draws
before terminal draft and then refit Draft and phase-one recalibration for the
same draw IDs. One full draw using the registered frozen rating history and
Draft development snapshot matched the slow path exactly. That is no longer a
complete operational claim: the draw path must be rewired to the exact fresh
post-validation refit source before it can qualify for phase two. Independent
registration will still require equality across all 2,000 fresh-source draws
on a future verification event.

Event-probability v2 preserves percentile-bootstrap semantics: an ordered
95% percentile interval in `[0, 1]` is valid even when it does not contain the
plug-in point. Point containment is recorded and is not a gate. The Betano v2
transport wrapper preserves that probability and interval exactly; any wider
legacy compatibility interval exists only in memory to satisfy the frozen v1
transport shape and is neither persisted as probability evidence nor used for
price extraction.

The complete outcome-free collection path now has explicit contracts for:

- one-time phase-two opening, consumed before the first probability or quote;
- event-probability v2 identity registration;
- an immutable per-event quote-attempt plan written after probability creation
  but before the Betano request, reserving exact no-clobber quote and
  qualification paths even when the request fails;
- exact-byte Betano response capture after the probability;
- post-start joining to outcome-free actual-map-start authority;
- rejection unless the transport response preceded actual start by at least
  five seconds; and
- an independently reviewed, externally pinned qualified-quote registry.

The prospective probability identity now also preserves the frozen
recalibrated rating-only comparator plus patch, roster-change stratum, and
sparse/new-champion metadata. Those fields are required by the predeclared
phase-two market and supported-stratum gates; deriving or selecting them after
outcomes would not qualify. Quote coverage will use the pre-request event-plan
inventory as its denominator, not the successful-quote inventory.

Every plan is consumed exactly once into either a validated quote or a typed
failure receipt. Failure receipts retain no free-form exception text, headers,
cookies, or credentials and cannot be replaced by retrospective retries. After
authoritative map start, an outcome-free completion classifies the plan as a
qualified quote, a failed attempt, or a response inside the forbidden
five-second boundary. These completions are the sole stopping-rule inventory.

The metadata snapshot recomputes the exact locked 500 quoted maps, 125 series,
regional, patch, roster-change, sparse-map, 80% coverage, and 100 shadow-signal
floors. At 1,000 quoted maps, missing shadow support becomes a terminal
failure. A support-met snapshot still cannot open outcomes: a distinct
independent reviewer must replay every plan/completion/quote/failure/map-start
binding, attest that it is the first support-met snapshot, and externally pin
the exact registry bytes.

The sealed phase-two evaluator is also implemented, but not authorized or run.
Its outcome cohort must cover every map in that registered snapshot exactly,
bind authoritative actual map start, declare blue or red as the winner, and
hash exact official-result evidence observed strictly after map start. The
evaluator rejects a cohort created after the evaluation clock or any missing,
duplicate, substituted, or reordered map.

The evaluator applies all predeclared gates in one terminal result:

- combined versus no-vig market and combined versus identically recalibrated
  rating-only log loss and Brier deltas, with 10,000 whole-series bootstrap
  resamples and frozen seed `20260806`;
- 10-bin equal-frequency ECE, series-cluster ECE upper bound, calibration
  intercept/slope intervals, and league, patch, roster-change, sparse/new, and
  international strata;
- quote coverage over every completed plan, p95 prediction-to-response latency
  over every received quote, exact post-map-start count, extraction replay, and
  team/map binding gates; and
- the predeclared one-unit shadow rule, 1% positive-profit haircut, at least 100
  qualifying maps, series-cluster ROI interval, maximum drawdown, and longest
  losing run. This remains evaluation-only and never supplies a stake.

The five-second quote safety buffer is now represented separately from the
frozen `quote_after_map_start_count` gate. A quote three seconds before start is
too late to qualify but is not falsely counted as received after start. The p95
latency calculation includes all received quotes, including unqualified late
responses, rather than only successful qualifications.

Outcome opening requires two distinct reviewers with separate model-evaluation
and market-capture scopes plus an outcome custodian who is neither reviewer.
The externally pinned authority binds the independently registered first
support-met snapshot and a future evaluation-readiness registry. A durable
no-clobber marker is written before the first cohort byte is read; any marker or
result blocks reopening, even after a crash. Independent result registration
then reruns the entire evaluator from the bound snapshot and outcome bytes and
requires byte-for-byte equality before two further model-result and
market-result reviews can register pass or terminal failure.

The new evaluation-readiness source freezes these schemas, signatures, source
hashes, seed, resample count, empty outcome state, and non-authorizing result
semantics. Its artifact and independent registry are deliberately unissued:
the registered phase-two collection readiness on which it depends does not yet
exist. The collection- and evaluation-readiness registry validators themselves
are now part of their respective pre-freeze source-hash sets; no registry code
may be introduced after a readiness artifact is locked. Private readiness v15
also semantically validates these registries, the first support-met snapshot,
and the exact-replay phase-two result instead of treating file presence as a
pass. The latest combined terminal-Draft, market, and private-readiness
verification passes `232` tests with `2` deprecation warnings in `463.75s` via
`python3 -m pytest -q tests/model_v2/draft/terminal tests/model_v2/market tests/test_private_decision_readiness.py`.
Tests establish implementation behavior only, not future model or market
performance.

Private readiness semantically replays both the active collection-opening
marker and the qualified quote registry; file presence plus an environment digest is not
enough. The phase-two implementation-readiness source exists, but its artifact,
independent registry, opening authority, marker, probability receipts, quote
receipts, qualifications, quote registry, evaluation-readiness artifact,
sealed-outcome opening authority, evaluation result, and result registry are
deliberately unissued because
the required future phase-one pass and independent dependencies do not exist.
None of these implementation contracts establishes odds accuracy, probability
accuracy, expected value, a recommendation, a stake, or betting authority.

The older `market_decision.py` match-winner lane consumes v1 diagnostic
probability receipts and does not replay the new phase-two result registry. It
is now explicitly non-authorizing for match winner: even a syntactically valid,
externally hashed v1 authority returns `NO_AUTHORIZED_BET`, with probability,
fair odds, edge, and expected return withheld. A replacement semantic v2
consumer must bind the terminal phase-two pass and new live-production receipt
path before any private decision can be authorized.

That replacement lane is now implementation-complete but operationally
unissued. The semantic market
authority requires two new deployment reviewers and semantically replays the
terminal phase-two result, recalibration/uncertainty registry, complete Betano
terms, adapter registry, protocol, and exact production-source hashes. It may
authorize private probability/fair-odds/EV/BET-or-PASS calculations, but its
contract permanently excludes public output, transaction execution, account
control, and stake sizing. Private readiness no longer accepts file presence
plus an external hash as sufficient for this authority: it loads the receipt,
replays every binding, checks the validity window and both independent
deployment reviews, and confirms the transaction and stake exclusions. The
authority cannot currently be issued because the terminal future evaluation
and its independent dependencies do not yet exist.

The production probability source is present. It consumes the exact frozen
fast-uncertainty candidate only under an active semantic authority, replays the
rating-only comparator and all 2,000 draws, samples its own clock, rejects
post-start capture, contains no price or outcome, and grants no transaction or
stake authority. During persistence testing, both the phase-two and production
builders were corrected to return only their canonical receipt schema; derived
convenience fields remain validator outputs and are no longer accidentally
written into receipts that their own validators would reject.

The production Betano quote and semantic decision sources are also present.
The quote source binds the exact response bytes and transport clocks to the
production probability, semantic authority, event, scheduled start, and two
selection prices; it explicitly does not prove acceptance, executable limits,
transaction success, or stake authority. The decision source reloads and
validates that quote and authority, enforces the frozen 60-second probability
and 30-second quote limits, requires a pre-start quote after the probability,
normalizes the two-way implied market probabilities, and evaluates both sides.
It returns `BET` only when exactly one side's 95% lower probability bound still
has at least 2% expected return after the frozen 1% positive-profit haircut.
Otherwise it returns `PASS`; stale, malformed, mismatched, post-start, or
two-sided-qualifying inputs return `NO_AUTHORIZED_BET` with all numerical
decision outputs withheld. Both authorized outcomes expose both candidate
calculations for replay, while `selection` remains null for `PASS`. Stake is
always null and transaction authority is always false.

The evaluation-readiness source-hash inventory names the semantic
authority, production probability, production quote, and semantic decision
sources, all of which now exist. They cannot be changed after a readiness
artifact is issued and then silently inherit its pre-outcome status. The
readiness artifact and registry remain deliberately unissued until their
future-data prerequisites and independent reviews exist.

## What is still missing

The future ledgers and joint phase-one collection currently have zero entries.
The frozen ratings metadata stopping rule requires:

- at least 100 eligible future series overall;
- at least 20 series in each of LCS, LEC, LCK, and LPL; and
- at least 20 series where one or both exact rosters changed.

Outcomes must remain unopened while those counts are checked. The first
metadata-only snapshot meeting the rule must be externally pinned. Independent
protocol review and independent opening approval are both absent. Only then may
the one-time sealed evaluation run; proper-score, calibration-reliability,
regional, patch, international, roster-change, and replay gates still have to
pass. The live ratings ledger and the joint support-met snapshot, opening
authority, result, independent reviews, and external registry pin are absent.

The Draft Score core and GRID source adapter are now built, hash-pinned, and
tested, but the GRID source receipt is based only on five completed archives,
the actual future ledger is still absent, and it has zero eligible maps. The
new participant-dependence diagnostic binds the frozen Draft snapshot to the
exact player parquet. It finds ten unique role-assigned player IDs for `5,751`
of `6,194` maps (`92.85%`) across `970` players. However, connecting games that
share any player produces one transitive component containing all `5,751`
eligible maps. Treating those maps as independent participant clusters would
therefore be invalid. The pre-boundary phase-one freeze already supplies the
alternative path: whole-series bootstrap is supplemented by a graph-HAC paired
loss sensitivity linking maps that share an exact series, player, or
organization, with exact ten-player/two-organization identity required. Draft
readiness now recognizes that frozen method instead of permanently requiring an
impossible atomic component split. Participant-dependence support nevertheless
remains false until the untouched future result passes both log-loss and Brier
graph-HAC upper-bound gates and two independent reviewers register the exact
result. The
metadata-only stopping rule requires at least `1,000` eligible maps in `250`
series, `150` maps from each of LCS/LEC/LCK/LPL, `100` international maps,
three future patches, `200` maps on the latest future patch, and `50`
sparse/new-champion maps. Every receipt must bind terminal assignments,
pick/ban protocol validation, sides, patch, source bytes and rights, the ratings
receipt, and authoritative actual-map-start evidence; the system-clocked
prediction must precede map start. Retrospective backfill does not qualify.
No live Draft ledger, joint support-met snapshot, independent opening review,
evaluation result, or external result-registry pin exists.

After independent pinning, both the log-loss and Brier improvement upper 95%
bounds must be nonpositive under series-cluster bootstrap uncertainty, with at
least one strictly below zero. Both graph-HAC upper bounds must independently
remain nonpositive as the participant/organization dependence sensitivity.
Reliability, regional, future-patch,
international, and Python/TypeScript replay gates must also pass. Even then,
the neutral Draft output remains an equal-strength composition index; only the
combined contextual model can seek probability authority.

The match-winner protocol is locked but both stages are empty. The operational
phase-one join is implemented, but there are no real event plans, bundles, or
joint snapshots yet. Phase one has
not reached support, passed evaluation, or received independent opening
authority. The phase-one evaluator and its seeds are frozen, but its ledgers,
parity registry, outcome cohort, opening authority, result, and independent
result registry do not exist. Consequently there is no fitted phase-two
recalibration artifact, completed event-specific 2,000-draw uncertainty
artifact, independent calibration and uncertainty registration, complete
bookmaker-terms alignment review, independent
Betano adapter registration, phase-two opening, event-probability ledger,
qualifying quote ledger or registry, support-met stopping snapshot, sealed
outcome opening, phase-two evaluation result, independent result registry, or
market authority. The adapter
candidate is now built and code-pinned, and the public help pages are
byte-captured and pinned,
but remain informative rather than a complete settlement authority. A browser-visible price is not enough: exact
source bytes, deterministic extraction, independently pinned receipt identity,
transport timing, and settlement alignment must all replay.

Accordingly:

- **Player/Team Ratings:** numerically replayable development diagnostics, but
  no validated rating authority.
- **Terminal Draft Score:** corrected v3 candidate and prospective protocol are
  locked and outcome-free capture is ready, but future support, independent
  evaluation, reliability, and source-specific authority remain absent; no
  draft probability is authorized.
- **Total Kills:** remains a development candidate; current LCS patch support is
  too small for the frozen patch gate and there is no independent market
  authority.
- **Bet evaluation:** a two-stage prospective protocol and fail-closed receipt
  contracts are now locked, but both prospective cohorts and every downstream
  independent registry remain absent. Raw model numbers and browser prices can
  no longer reach fair odds or EV. Exact event probability, calibration,
  uncertainty, quote, settlement, freshness, and external authority receipts
  must all replay first.

Synthetic fixtures, retrospective predictions, internally consistent ratings,
bookmaker prices, and passing component tests do not satisfy these gates.
