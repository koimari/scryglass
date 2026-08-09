# Side-neutral prospective capture revision

Status: implementation design frozen; not yet evaluation authority.

## Problem

The current rating receipt requires exact blue and red organizations before the
scheduled series start. Public Leaguepedia schedule rows expose `Team1` and
`Team2`, not map side. The 2026 LCS rules make Game 1 Choice and Remaining
Choice final the day before the match, but distribution to teams and League
Officials does not establish a public source. Treating schedule order as side
would therefore be an unsupported guess.

The evaluation must not discard every otherwise valid event simply because the
public side becomes observable only during champion select. It also must not
backdate a receipt, invent an expected map-start time, or refit after the map
has begun.

## Required event sequence

1. **Pre-side rating state, before the scheduled series start.** Capture the
   exact two organizations and ten starters in source order, the patch, the
   frozen rating model state, player/team posterior components, and both
   possible side-conditional predictions. The receipt labels the organizations
   `team1` and `team2`; it does not claim either is blue. Both conditional
   predictions are computed and sealed at the same system-clock sample.
2. **Side binding, during champion select.** Capture public source bytes that
   identify the actual blue and red organizations. The binding may only select
   one of the two already-sealed conditionals. It may not refit ratings, change
   either lineup, or add a new prediction.
3. **Terminal Draft Score.** Capture all ten champion-role assignments and the
   legal pick/ban action sequence. The terminal receipt binds the exact
   pre-side state and side-binding bytes before combining the selected rating
   conditional with Draft Score.
4. **Actual map start.** Capture an outcome-free authoritative start signal.
   The completed bundle must prove:

   `pre-side capture < side binding <= terminal draft capture < actual map start`

5. **Ledger admission.** Only a completed bundle with all four exact receipts
   enters the prospective ratings/Draft ledger. A pre-side state, an unused
   conditional, a pending side binding, and any failure receipt have zero
   eligible-map count.

## Timestamp semantics

- `scheduled_series_start_utc` is the public pre-event cutoff used to freeze
  rosters, patch evidence, and the training-data boundary.
- `actual_map_start_utc` is captured later and is the sole deadline used to
  prove that the realized side binding and terminal draft were prospective.
- No user-supplied “expected map start” is allowed.
- The source availability time and system capture time are distinct and both
  are retained.

## Fail-closed invariants

- `Team1`/`Team2`, page order, bookmaker selection order, and UI left/right
  position are never side authority.
- Both side conditionals must bind the same pre-side source bytes, model bytes,
  model state, patch, rosters, and clock sample.
- The two conditionals must pass an algebraic orientation check against the
  frozen model's neutral strength difference and side term. They are not
  assumed complementary: blue-side advantage applies in both orientations.
  Re-labeling the same physical orientation must preserve its selected-team
  probability within the frozen numerical tolerance.
- A side-binding source must identify both organizations, have reviewed rights
  for private evaluation, be hash-bound, and be available before actual start.
- More than one side binding for the same map is an ambiguity and invalidates
  the map.
- A completed bundle cannot contain outcomes, scores, kills, gold, map result,
  or winner fields.
- Immutable no-clobber artifacts and versioned snapshots are mandatory.
- No receipt in this sequence grants rating deployment, probability, odds, EV,
  recommendation, stake, transaction, public, or betting authority.

## Protocol transition

The existing v3 prospective cohort currently has zero eligible entries. A new
protocol may supersede its capture semantics only before any future outcome is
opened. The candidate models, source snapshot, support thresholds, evaluation
metrics, comparator set, uncertainty procedure, and future boundary remain
unchanged. The supersession receipt must explicitly state that no v3 outcome
or conditional prediction was used to design this revision.

Before collection, the implementation and protocol artifacts require fresh
hash registration and independent review. After the stopping rule is met, the
same independent outcome-opening, calibration, reliability, uncertainty,
semantic rating, semantic Draft, market-probability, quote, and promotion gates
still apply.
