# Temporal Draft Runtime

The meaningful upgrade is an expanding pre-event information set.

For a match at time `t`, the runtime is built from maps with `event_time < t`:

1. fit composition coefficients from prior drafts and results;
2. calculate team and player form before the match;
3. apply the prior maps' outcomes only after their scores are written;
4. score the new draft;
5. attach a contextual score only when starter evidence is available before `t`.

The implementation is [temporal_draft_runtime.py](/Users/river/scryglass/lol_kills/research/temporal_draft_runtime.py). It writes an outcome-free `temporal-scored-ledger.jsonl` before reading labels for evaluation.

## Roster and leave evidence

Roster changes are input records, not model guesses. Each record needs:

```json
{
  "team": "Team name",
  "role": "jungle",
  "player": "Player name",
  "status": "leave",
  "effective_from": "2026-07-01T00:00:00Z",
  "available_at": "2026-06-30T18:00:00Z",
  "source_snapshot_id": "announcement-or-roster-page-id",
  "source_sha256": "64-character-source-payload-hash"
}
```

The replacement uses the same shape with `status: "confirmed_substitute"` or `status: "temporary_starter"`. Its `effective_from` is the match-window boundary, while `available_at` must be when the public announcement or roster page became available. For a temporary change, set `effective_to` when the original starter returns.

For the Inspired/Armao case, add two records backed by the actual public announcement or roster snapshot: block Inspired with `leave`, then add Armao as the jungle `confirmed_substitute`. The system resolves the highest-precedence active candidate and returns `unavailable` on missing or conflicting evidence. It never infers a substitution merely because a later match page shows a different player.

## Current backtest

The pre-period capture contains 7,355 maps and 73,550 player rows. The hybrid runtime was fit before July 1 and then updated only with pre-match online ratings:

- `k=0.20`: **633/997 = 63.49%**;
- `k=0.80`: **628/997 = 62.99%**;
- pure composition component: **532/997 = 53.36%** at `k=0.20`;
- strict mode without a roster packet: contextual result **unavailable for 997/997 maps**.

The 63.49% match to the old full-run number is not being treated as proof of improvement. It is evidence that the same broad accuracy can be reached with a pre-July model and pre-match history, while the old runtime's 2026-07-18 snapshot is no longer required for the backtest.

This is still development evidence. The source pages were retrieved retrospectively, so the next validity gate is source availability before each event—not another accuracy claim.
