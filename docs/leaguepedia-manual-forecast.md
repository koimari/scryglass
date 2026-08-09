# Manual Leaguepedia forecast workflow

This is the operating contract for the July 1 onward team/game study. It is
designed for human-reviewed Leaguepedia inputs and deterministic local scoring.

## The three phases

Each map gets one append-only ledger:

1. **Freeze pregame input.** Capture the roster evidence, team identity, side,
   draft, competition, and timestamps before the map starts. The frozen
   projection may not contain `winner`, `result`, `score`, kills, gold,
   duration, or any other finished-state field.
2. **Score the frozen input.** Run the checked-in scorer against only that
   projection. Record the pregame hash, scorer hash, draft runtime hash, and
   engine output hash.
3. **Reveal the outcome.** After the score is sealed, capture the match-history
   result and append it as a separate outcome object. The pregame bytes never
   change.

The implementation is in
`lol_kills/etl/manual_leaguepedia.py`:

```bash
python3 -m lol_kills.etl.manual_leaguepedia freeze \
  --input pregame-input.json \
  --output pregame-frozen.json

python3 -m lol_kills.etl.manual_leaguepedia score \
  --input pregame-frozen.json \
  --output scored.json

python3 -m lol_kills.etl.manual_leaguepedia reveal \
  --input scored.json \
  --outcome outcome.json \
  --output revealed.json

python3 -m lol_kills.etl.manual_leaguepedia verify \
  --input revealed.json \
  --require-score \
  --require-outcome
```

## Source capture

Capture the raw revision and rendered page together:

```bash
python3 -m lol_kills.etl.manual_leaguepedia capture-page \
  --title 'LØS/Match History' \
  --before '2026-07-15T14:00:00Z' \
  --output-dir snapshots/los-match-history
```

`--before` selects the latest MediaWiki revision at or before the cutoff.
`available_at` remains the actual retrieval time. A historical revision
reconstructed today is therefore useful retrospective evidence, but it does
not become strict pregame evidence merely because its revision timestamp is
old.

## Strict versus retrospective mode

- `strict`: every source snapshot and roster event must have been available
  before `event_start`; the model runtime `as_of` must also precede the map.
  This is the only mode that supports a clean historical forecast claim.
- `retrospective`: permits current/post-event source retrieval for replay and
  debugging, but the output must be labeled retrospective and cannot be used
  as a leakage-free backtest.

The current draft runtime is dated 2026-07-18, so it cannot be used as a
strict forecast runtime for July 15 maps. The runner rejects that combination
in strict mode instead of silently calling it pregame.

## Roster moves

Roster membership and expected starters are separate time-sliced assertions.
For example, Inspired's leave and Armao's temporary start are represented as
separate events with `effective_from`, `available_at`, role, status, and source
hash. The resolver blocks Inspired and selects Armao only for maps after the
effective change. Equal-precedence candidates produce `unavailable`; there is
no silent fallback.

Team aliases are identity-normalized, so `LØS`, `MIBR.LOS`, `Los Grandes`, and
`LOS` can be joined without treating the name change as a new organization.

## Batch run

The batch companion is `lol_kills/etl/manual_leaguepedia_batch.py`. It captures
all teams and all returned maps rather than sampling eight teams:

```bash
python3 -m lol_kills.etl.manual_leaguepedia_batch capture \
  --run-dir data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31

python3 -m lol_kills.etl.manual_leaguepedia_batch freeze \
  --run-dir data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31

python3 -m lol_kills.etl.manual_leaguepedia_batch score \
  --run-dir data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31 \
  --workers 8

python3 -m lol_kills.etl.manual_leaguepedia_batch reveal \
  --run-dir data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31
```

The run stores schedule-only catalog pages, result-free draft/player pages,
and outcome pages in separate raw directories. It deduplicates by game ID,
normalizes team aliases, preserves international and Tier-2 competition
labels, freezes every complete five-by-five draft, scores those frozen bytes,
and only then attaches the result. Since this run retrieves historical pages
after the maps, it is a result-blind retrospective study, not a strict
pre-event backtest.

If a historical player is absent from the fixed model context, the batch scorer
does not substitute the team's current roster. It emits the deterministic
champion-only Draft Score and marks player context unavailable; that map should
not be interpreted as having a player-comfort or roster-strength adjustment.
