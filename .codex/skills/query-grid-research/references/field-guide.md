# League of Legends GRID field guide

The machine catalog is authoritative for field shape. This guide supplies
conservative temporal meaning; it does not assert that every schema field is
populated for every league, patch, series, or checkpoint.

## Source boundaries

- **Central Data**: titles, tournaments, series schedules, teams, players,
  external links, content catalogs, and provider ID resolvers. Treat as
  metadata. `updatedAt` can be later than a pregame/checkpoint cutoff.
- **Series State**: one known series' live-or-final snapshot. The same schema
  represents in-progress and completed state; check `started`, `finished`,
  game clock, and sequence.
- **Series Events**: ordered transactions with `sequenceNumber`, `occurredAt`,
  event(s), deltas, and optional full state. Local observations are not a
  complete taxonomy.
- **File Download**: series-scoped artifact listing. Listing is confirmed;
  payload download/completeness is a separate authorization and validation
  phase.

## LoL state types

`GameTeamStateLol` exposes candidate team fields including `money`,
`loadoutValue`, `netWorth`, `kills`, deaths/assists, `objectives`,
`structuresDestroyed`, `unitKills`, `players`, `damageDealt`,
`experiencePoints`, `moneyDifference`, `totalMoneyEarned`, and `visionScore`.

`GamePlayerStateLol` exposes candidate player fields including champion,
roles, money/loadout/net worth, kills/deaths/assists, inventory, objectives,
position, level-related `experiencePoints`, health/armor, damage, vision,
alive/respawn state, and total money earned.

`Objective` exposes `type`, `completedFirst`, and `completionCount`.
`StructureState` exposes type, side/team, health, destroyed state, position,
and respawn clock. `NonPlayerCharacterState` exposes type, side, alive,
position, and respawn clock. `ClockState.currentSeconds` is the checkpoint
clock candidate.

All of these remain **candidate** features until timestamp-complete empirical
coverage proves their value and leakage safety.

## Temporal classes

- `metadata_or_schedule`: Central Data fields; usable only with a suitable
  metadata as-of receipt.
- `live_and_final_snapshot_or_unknown`: Series State fields that may be
  populated live and remain in final state. Require a timestamped observation
  at the target checkpoint.
- `final_outcome_signal`: `finished`, `forfeited`, and team-state `won`.
  Labels/verification only; never predictors.
- `timestamped_event_stream`: may span the entire game. Filter by provider
  game time and sequence, not file row position or retrieval time.
- `final_game_only_artifact`: final GRID/Riot state files. Outcomes and
  verification only.
- `completed_game_replay_artifact`: full-game replay. Any checkpoint
  derivation requires a deterministic time-bounded parser.

## Identity rules

Keep separate columns for:

- GRID series ID and GRID game ID;
- GRID team/player IDs;
- external provider name plus external entity ID;
- Riot platform ID, Riot game ID/root game ID;
- PUUID only when explicitly present in a verified Riot record.

Require exact one-to-one mappings. Never derive a Riot ID or PUUID from a
nickname, display name, team tag, roster position, or ordering.

## Checkpoint rule

For checkpoint `T`, accept only a state/event with provider game time
`<= T`. Record the state age `T - state_time`, source sequence, stream
watermark, gaps, duplicates, late arrivals, and revisions. Apply the
task-declared maximum age; five seconds is provisional in the current
Scryglass foundation. Missing receive timestamps block prospective latency
claims but do not by themselves block retrospective evidence when provider
time, ordering, and no-post-checkpoint leakage are proven.

Final labels must come from a separately verified complete game-end path.
