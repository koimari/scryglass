# GRID query workflows

## Routing table

| Research question | Validated path | Required gates | Output ceiling |
|---|---|---|---|
| What series exist in a date/league window? | Central Data `allSeries` | `titleId=3`, `type=ESPORTS`, bounded time range/page, pro/private rejection | Metadata inventory |
| What are the exact series/team/player identities? | Central Data `series`, `team`, `player`, `externalLinks`, and `*IdByExternalId` | One-to-one crosswalk; preserve provider and external namespaces | Verified or quarantined identity |
| Is a known series started/finished? | Series State `seriesState(id)` with only status fields | Already-verified GRID series ID | Current/final status metadata |
| What was the state at a live checkpoint? | Captured Series Events/full state or timestamp-complete local Riot events | Provider game clock at/before checkpoint, sequence continuity, freshness bound, no revisions after cutoff | Historical checkpoint evidence only |
| What is final total kills/outcome? | Verified Riot `game_end` plus kill events; cross-check GRID final state | Exact game crosswalk, complete event file, unique game-end, both teams | Outcome label |
| What items, gold, XP, vision, damage, or objectives are available? | `GamePlayerStateLol`, `GameTeamStateLol`, `Objective`, `StructureState`, inventory types | Timestamped population evidence at the intended checkpoint; do not rely on schema alone | Candidate feature |
| Which files exist for a known series? | File Download list endpoint | Verified series ID; sanitize URLs; no download in discovery mode | File metadata only |
| What event families exist? | Already-local Series Events receipts | Report observed archive hashes and population; label non-exhaustive | Observed taxonomy |
| Does GRID expose bookmaker odds? | Introspected roots/types and separately licensed product metadata | Explicit odds endpoint/product confirmation | Currently unavailable |

## Private in-game market feasibility

Use `/Users/river/scryglass/lol_kills/grid_market_foundation.py` for the
bounded, non-model feasibility workflow covering first/total towers,
inhibitors, dragons, barons, and first blood. It:

- binds GRID series/game/team IDs to Riot platform/game IDs through the
  verified local context;
- derives checkpoints only from Riot LiveStats events and stats at or before
  10/15/20/25 minutes;
- verifies final labels independently against GRID cumulative final state;
- withholds incomplete game-end, conflicting labels, ambiguous identities,
  stale corroborating state, and unverified market definitions;
- never emits probability, price, fair odds, edge, expected value, or betting
  authority.

Treat `total_tower_destructions` and `unique_towers_destroyed` as different
estimands. Respawned Nexus turrets can create repeated destruction events, so
an external market's settlement rules are required before mapping either
derived value to “total towers.”

## Safe Central Data pattern

Use a small cursor page and explicit filters. Keep the cursor in the variables
receipt.

```graphql
query SeriesWindow($after: Cursor) {
  allSeries(
    first: 25
    after: $after
    filter: {
      titleId: 3
      type: ESPORTS
      startTimeScheduled: {gte: "START_Z", lte: "END_Z"}
    }
    orderBy: StartTimeScheduled
    orderDirection: ASC
  ) {
    pageInfo {hasNextPage endCursor}
    edges {
      node {
        id
        type
        startTimeScheduled
        updatedAt
        tournament {id name}
        teams {baseInfo {id name externalLinks {dataProvider {name} externalEntity {id}}}}
      }
    }
  }
}
```

Do not continue pagination after the declared window/row cap is satisfied.

## Safe identity pattern

Prefer explicit external-link resolvers:

- `seriesIdByExternalId(dataProviderName, externalSeriesId)`
- `gameIdByExternalId(dataProviderName, externalGameId)`
- `teamIdByExternalId(dataProviderName, externalTeamId, titleId)`
- `playerIdByExternalId(dataProviderName, externalPlayerId, titleId)`

An absent, conflicting, or many-to-one mapping is quarantine, never a fuzzy
name fallback. Names may be retained as descriptive labels only.

## Safe Series State pattern

Start with status and IDs. Add only fields required by the question.

```graphql
query KnownSeriesState($id: ID!) {
  seriesState(id: $id) {
    id
    valid
    started
    finished
    updatedAt
    games {
      id
      sequenceNumber
      started
      finished
      clock {id type ticking currentSeconds}
    }
  }
}
```

`updatedAt` is not game time. A current GraphQL snapshot cannot reconstruct a
past checkpoint. Use a timestamp-complete captured event stream for that.

## Provenance receipt minimum

Record:

- catalog version/hash and endpoint schema hash;
- endpoint identifier, canonical query hash, non-secret variables, and
  retrieval timestamp;
- provider series/game/team/player IDs and external-ID namespaces;
- page size, cursors, time bounds, and stop reason;
- response/content hash and local source receipt hash;
- provider `updatedAt`, game clock, sequence/watermark, and state age;
- identity, freshness, completeness, and leakage blockers.

Never store headers beyond allowlisted rate-limit fields.
