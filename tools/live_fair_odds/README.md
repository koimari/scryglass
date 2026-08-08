# Private live market-audit worksheet

This local-only worksheet is intentionally separate from the public Scryglass `/live` page.

Run it from the repository root:

```bash
python3 -m tools.live_fair_odds.server --open
```

The default address is `http://127.0.0.1:8765`.

The worksheet uses:

- checked-in team aliases, draft context, and pace priors for diagnostics;
- a two-sided bookmaker quote for descriptive no-vig market context;
- the existing 10/15-minute live win artifact only after an exact-roster event
  rating is independently registered;
- the canonical terminal neutral Draft Score replay in explicit development mode;
- OE team/league pace priors;
- the versioned total-kills v2 artifact at exact validated checkpoints only, with
  calibration residuals clustered by series.

The win and total-kills outputs are research diagnostics, not authorized betting
probabilities. Winner probability is null by default. A registered Player/Team Rating may
feed the live-state development diagnostic, but it still cannot authorize a match-win
probability. Barons and inhibitors are recorded but receive no invented coefficient.

Pregame draft diagnostics require a verified event-start timestamp and a draft-source
availability timestamp strictly before it. Legacy context Elo and inferred lineups are not
used as rating fallbacks. Event ratings require an independently pinned registry that
replays the exact ordered pre-event roster, six model/evaluation artifacts, freshness,
an explicit status for every team component, posterior intervals, and every frozen
review gate. Identified components are included in the registered estimand; unavailable
synergy or policy must remain null and is never treated as zero. Rating
registration explicitly does not authorize match probability, fair odds, EV, or betting.

The successor Player/Team Rating lane is contamination-aware. Because the predecessor's
2026-Q1 validation results were inspected, that period is now labeled adaptive development,
not independent validation. The locked v2 protocol evaluates a fixed 12-candidate
player-plus-organization family and reserves the hash-bound post-2026-04-01 cohort as the
only final temporal holdout. The first calendar-window lock selected no candidate because one
window had only 13 series against a minimum of 20. That failed lock remains immutable. A
versioned metadata-only correction repartitions the same adaptive corpus into three
chronological blocks of 176 series, without changing candidates or thresholds; it selected an
adaptive candidate for possible independently approved final evaluation. The sealed cohort
remains unopened, so no production rating or probability is authorized.

The decision layer fails closed. It returns `NO_AUTHORIZED_BET` and keeps probability,
fair odds, edge, and expected return null unless all of the following are present:

- an independently reviewed market-authority receipt whose SHA-256 is pinned outside the
  receipt;
- exact binding to the current model, prospective market protocol, phase-two
  evaluation, calibration, uncertainty, bookmaker-terms, quote-capture, and
  settlement artifacts;
- passed out-of-sample market comparison, calibration, fixed shadow-policy,
  quote-coverage, settlement-review, latency, and dependence-aware uncertainty gates;
- an event-specific probability receipt in an independently pinned registry;
- replay of the receipt's bounded recalibration and dependence-aware interval,
  including exact source-prediction, generation-code, and bootstrap draw-set hashes;
- a fresh provenance-bearing quote containing both market sides and binding the exact
  event, selection, odds, and settlement rule, loaded through the independently
  pinned quote registry.

The numeric probability and interval passed by the worksheet are diagnostics only.
They are never used for fair odds or EV. A valid quote plus raw floats still returns
`NO_AUTHORIZED_BET`. The match-winner path has a separately locked two-stage
prospective protocol; the total-kills path has no corresponding registered market
protocol and therefore cannot borrow match-winner authority.

Phase-one evidence is collected through
`lol_kills.v2.market.phase_one_collection_v1`. Its event plan starts from an exact
persisted ratings receipt; its completed event bundle requires the corresponding
terminal-Draft and actual-map-start receipts; and its joint snapshot rebuilds both
registered child ledgers at one system-clock sample. The implementation was
hash-pinned in an empty readiness receipt before the shared future boundary. These
artifacts reject outcomes and keep all authority false. Metadata support, even when
complete, still requires an external digest pin and the registered one-time opening
review.

The operational entrypoint is
`lol_kills.v2.market.prospective_capture_v1`. It composes the frozen child
builders without adding a side, roster, patch, draft, or timestamp fallback:

```bash
python3 -m lol_kills.v2.market.prospective_capture_v1 --root . prepare \
  --input prospective-event.json \
  --roster-source-payload roster-source-package.json \
  --patch-receipt patch-receipt.json

python3 -m lol_kills.v2.market.prospective_capture_v1 --root . draft \
  --plan data/lol/v2/evaluation/match-winner-market-v1/phase-one/plans/<event>.json \
  --metadata terminal-draft-metadata.json \
  --source-payload terminal-draft-source.json

python3 -m lol_kills.v2.market.prospective_capture_v1 --root . map-start \
  --plan data/lol/v2/evaluation/match-winner-market-v1/phase-one/plans/<event>.json \
  --metadata actual-map-start-metadata.json \
  --source-payload actual-map-start-source.json
```

`prepare` requires explicit blue and red teams. Leaguepedia `Team1`/`Team2`
schedule order is not side authority. Each stage writes no-clobber artifacts;
the final stage publishes versioned joint, ratings, and Draft ledger candidates.
Any failure writes a validated capture-attempt receipt with
`eligible_evaluation_evidence=false`. A failure attempt is operational audit
evidence only and cannot enter either model ledger.

When blue/red is not publicly known before the scheduled series start, the
separately frozen side-neutral v2 protocol is the only allowed alternative. It
does not reinterpret `Team1`/`Team2` as sides. It seals both orientations at
one system-clock sample, then uses captured public JSON to select one child by
exact organization name:

```bash
python3 -m lol_kills.v2.ratings.player.pre_side_rating_envelope_v1 --root . \
  --input pre-side-input.json \
  --roster-source-payload roster-source-package.json \
  --patch-receipt patch-receipt.json \
  --out data/lol/v2/evaluation/multileague-v3/pre-side-rating-envelopes/<date>/<event>-g1.json

python3 -m lol_kills.v2.ratings.player.pre_side_rating_binding_v1 --root . \
  --envelope data/lol/v2/evaluation/multileague-v3/pre-side-rating-envelopes/<date>/<event>-g1.json \
  --input side-binding-input.json \
  --public-side-source public-side-source.json \
  --out data/lol/v2/evaluation/multileague-v3/pre-side-rating-bindings/<date>/<event>-g1.json

python3 -m lol_kills.v2.draft.terminal.side_neutral_prediction_v1 --root . \
  --side-binding data/lol/v2/evaluation/multileague-v3/pre-side-rating-bindings/<date>/<event>-g1.json \
  --draft-metadata terminal-draft-metadata.json \
  --draft-source terminal-draft-source.json \
  --out data/lol/v2/evaluation/draft-terminal-v1/side-neutral-predictions/<date>/<event>-g1.json

python3 -m lol_kills.v2.draft.terminal.future_prediction_ledger --root . map-start \
  --metadata actual-map-start-metadata.json \
  --source-payload actual-map-start-source.json \
  --out data/lol/v2/evaluation/draft-terminal-v1/map-start/<event>.json

python3 -m lol_kills.v2.market.side_neutral_capture_bundle_v1 --root . \
  --side-neutral-draft data/lol/v2/evaluation/draft-terminal-v1/side-neutral-predictions/<date>/<event>-g1.json \
  --map-start data/lol/v2/evaluation/draft-terminal-v1/map-start/<event>.json \
  --out data/lol/v2/evaluation/match-winner-market-v1/phase-one/side-neutral-bundles/<date>/<event>-g1.json
```

The required order is strictly `pre-side < side binding < terminal Draft <
actual map start`. The binding only selects existing rating bytes; it cannot
refit ratings. A complete bundle still contributes zero eligible maps until an
independent human review is written at
`data/lol/v2/authorities/multileague-v3/side-neutral-protocol-review-v2.json`
and its raw SHA-256 is supplied through
`SCRYGLASS_PRIVATE_SIDE_NEUTRAL_PROTOCOL_REVIEW_SHA256`. That review may
authorize only captures whose pre-side timestamp is after the review effective
time. It cannot authorize retrospective artifacts, outcome opening, ratings,
probabilities, odds, EV, recommendations, or betting.

The non-authorizing reviewer packet is frozen at
`data/lol/v2/review/multileague-v3/side-neutral-review-packet-v1.json`
(raw SHA-256
`8e2c082afbacf11a6bab881c5bf30a64610b99a682f84f8ca0e6a3edb70bf0e4`).
It binds the protocol, capture implementations, independent-review validator,
and post-review admission ledger. The reviewer must independently verify those
bytes and include both `reviewed_source_locks` and
`reviewed_admission_implementation` in the externally pinned review. The
packet is a request, not an approval, and all of its authority fields are false.

After review, the outcome-free admission ledger is built with:

```bash
python3 -m lol_kills.v2.market.side_neutral_ledger_v1 --root . \
  --bundle data/lol/v2/evaluation/match-winner-market-v1/phase-one/side-neutral-bundles/<date>/<event>-g1.json
```

Receipt files, if independently approved, live under
`data/lol/private_market_authority/`. Their digests must be registered through
`SCRYGLASS_PRIVATE_MATCH_WINNER_AUTHORITY_SHA256` or
`SCRYGLASS_PRIVATE_TOTAL_KILLS_AUTHORITY_SHA256`. No receipt is shipped by default, and a
self-consistent local file cannot authorize itself.

Exact roster receipts and rating receipts use separate pinned registries:

- `SCRYGLASS_PRIVATE_ROSTER_REGISTRY_SHA256` binds
  `data/lol/private_pregame_rosters/registry.json`;
- `SCRYGLASS_SEMANTIC_PRIVATE_RATING_AUTHORITY_SHA256` binds the short-lived
  semantic deployment authority at
  `data/lol/private_rating_authority/semantic-rating-authority-v1.json`. It must
  replay the exact independently registered joint future-evaluation result,
  approved player/team model bytes, serving sources, two new independent
  deployment reviews, and the 14-day data-freshness ceiling. It grants only
  private rating components—never probability, odds, EV, a recommendation, a
  stake, or a transaction;
- `SCRYGLASS_PRIVATE_RATING_REGISTRY_SHA256` binds
  `data/lol/private_rating_authority/registry.json`. This event registry is
  subordinate to the active semantic authority and must bind its exact approved
  artifact inventory plus the separately registered exact roster;
- `SCRYGLASS_PRIVATE_RATING_SEALED_OPENING_SHA256` may bind an independently
  authored one-time holdout-opening receipt; that receipt authorizes evaluation only,
  never production ratings or betting;
- `SCRYGLASS_PRIVATE_QUOTE_REGISTRY_SHA256` binds
  `data/lol/private_market_quotes/registry.json`.
- `SCRYGLASS_PRIVATE_MATCH_WINNER_PROBABILITY_REGISTRY_SHA256` binds
  `data/lol/v2/evaluation/match-winner-market-v1/event-probability-registry.json`.

None is shipped as approved authority by default. Manual prices in the browser form are
descriptive only and cannot create quote provenance or register themselves.

Terminal Draft promotion has the same two-layer boundary.
`SCRYGLASS_SEMANTIC_TERMINAL_DRAFT_AUTHORITY_SHA256` must externally pin
`data/lol/private_draft_authority/semantic-terminal-draft-authority-v1.json`.
That short-lived record replays the exact independently registered joint future
result, Draft subgroup/reliability gates, Python/TypeScript parity, approved
model bytes, and two new deployment reviews. Promotion receipt v1 is rejected;
v2 grants only the private equal-strength Draft component and explicitly leaves
public and event probability false. A combined match probability must separately
bind the exact semantically authorized rating receipt and pass the market
probability authority. The public TypeScript route does not opt into the private
component capability, so even a valid private receipt cannot open it.

`GET /api/readiness` runs and caches a full private blocker audit, including the terminal
Draft Score L2/GRID gate, Player/Team Rating artifact state, the locked two-stage
match-winner protocol, event-probability/quote/settlement registries, total-kills schema
and freshness, and the presence (never the value) of external digest pins. The audit is
non-authorizing and always leaves event approval to the exact score-request replay.

Run its focused tests with:

```bash
python3 -m pytest -q \
  tools/live_fair_odds/test_model.py \
  tests/test_market_decision.py \
  tests/test_bookmaker_quote_capture.py \
  tests/test_pregame_roster_capture.py \
  tests/test_private_rating_authority.py \
  tests/test_private_decision_readiness.py \
  tests/model_v2/ratings/test_semantic_rating_authority_v1.py \
  tests/test_live_totals_model.py \
  tests/model_v2/draft/terminal/test_future_prediction_ledger.py \
  tests/model_v2/draft/terminal/test_semantic_draft_authority_v1.py \
  tests/model_v2/market/test_prospective_capture_v1.py \
  tests/model_v2/market/test_match_winner_future_protocol_v1.py \
  tests/model_v2/market/test_event_probability_v1.py \
  tests/model_v2/ratings/player/test_multileague_development.py \
  tests/model_v2/ratings/player/test_multileague_runner.py \
  tests/model_v2/ratings/player/test_multileague_v2_protocol.py \
  tests/model_v2/ratings/player/test_multileague_v2_runner.py \
  tests/model_v2/ratings/player/test_multileague_v2_protocol_equal_series.py \
  tests/model_v2/ratings/player/test_multileague_v2_runner_equal_series.py \
  tests/model_v2/ratings/player/test_pre_side_rating_envelope_v1.py \
  tests/model_v2/ratings/player/test_multileague_v3_side_neutral_protocol_v1.py \
  tests/model_v2/ratings/player/test_multileague_v3_side_neutral_protocol_v2.py \
  tests/model_v2/ratings/player/test_multileague_v3_corrected_adaptive_diagnostic_v1.py \
  tests/model_v2/ratings/player/test_side_neutral_protocol_review_v1.py \
  tests/model_v2/ratings/player/test_side_neutral_review_packet_v1.py
```

The corrected-source adaptive rating diagnostic is frozen at
`data/lol/v2/models/player/multileague-v3/corrected-adaptive-diagnostic-v1.json`
(raw SHA-256
`b56b67f76d567d46f60cc7cddf6b15776a383c72bd6687e13a122cbee3375785`).
It replays all 12 previously declared hierarchical candidates on the immutable
pre-boundary snapshot. The registered incumbent beats the organization-only
comparator overall, but does not clear every LCS, domestic-league,
roster-change, international, or player-comparator uncertainty gate. The
adaptively ranked challenger also fails to beat the incumbent with nonpositive
upper 95% bounds overall, in LCS, and after roster changes. The diagnostic
therefore retains the frozen incumbent instead of tuning to exposed outcomes;
it explicitly does not validate that incumbent or authorize a rating or wager.

Current 2026-08-02 checkpoint: neither semantic authority file nor external pin
exists, and the v3 future rating/Draft ledgers still contain no eligible future
events. The first live LCS intake for LYON versus Shopify Rebellion has an exact
pre-event patch-26.15 revision and sourced ten-player roster, but the public
schedule has no blue/red field. Its immutable `prepare` attempt therefore
failed closed before creating a ratings prediction; schedule order was not used
as a side guess. The side-neutral v2 implementation and repository hash pin
were subsequently frozen before the 2026-08-03 holdout boundary, with zero
prediction, envelope, binding, Draft, or bundle artifacts present. No real
independent review digest or reviewed admission ledger exists yet, so the new
path remains collection-ineligible and does not rehabilitate the failed LCS
attempt. The current v3 warehouse bytes are unaliased and match their registered
snapshot hashes, but superseded v1/v2 rating artifacts correctly fail replay
against those newer bytes. The independent `docs/model-v2` contract content
anchor also remains out of agreement with the current documentation tree. These
are blockers, not reasons to substitute development output. Until new frozen
future receipts reach the locked stopping rule, the joint evaluation passes,
two result reviews and four deployment-scope reviews are independently pinned,
and event-specific roster/rating/Draft/quote receipts replay, all rating,
probability, odds, EV, and recommendation outputs remain unavailable.
