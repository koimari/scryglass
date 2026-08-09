# G1 roster-source decision

Status: `BLOCKED_DEPENDENCY` / non-authorizing research note  
As of: 2026-07-30

This note records the source branch that was actually checked. It is not a
roster receipt, does not authorize a model fit, and must not be used as a
fallback source for Team Rating or contextual Draft Score. G1 applies only to
player-aware contextual scoring; it does not block the neutral champion-draft
score.

## Acceptance target

The real-data spine needs a source assertion available no later than the
scheduled match start for each team and match. It must identify the provider
series, two teams, exactly five distinct main-roster players per team, one
canonical role (`top`, `jungle`, `mid`, `bot`, `support`) per player, source
update time, retrieval time, and an independently pinned source payload. A
missing, stale, substituted, duplicate, or ambiguous row is unavailable.

## Source results

| Source | What was observed | Decision |
| --- | --- | --- |
| GRID Central Data `series.players` | Historical LPL pages returned empty player lists. Upcoming pages sometimes returned incomplete team pools or substitutes; a bounded 50-series window produced zero strict five-versus-five, role-complete candidates. | Not authority yet. Keep the strict candidate validator, but do not infer starters. |
| GRID Series State | A completed series exposed participating players only in a mutable post-start state. | Observed participation only; cannot be used as pre-event input. |
| Oracle's Elixir / local warehouse | Match rows contain observed participants and team labels, but no pre-event roster receipt or historical availability timestamp. | Retrospective descriptive data only. |
| Local Leaguepedia player files | Player identity and current-team fields exist; dated membership intervals and exact match-time main-roster status do not. | Identity aid only. |
| Riot Global Contract Database | Official current affiliations and contract end dates are exposed publicly, but this branch did not establish historical match-time role-resolved starting rosters. | Current-contract evidence only. |
| Cito roster-history API | The public documentation advertises historical membership periods and roles, but no API key, rights review, or payload receipt exists in this workspace. | Procurement/review candidate, not bound. |

### Focused procurement follow-up (2026-07-30)

The provider's current public endpoint map still lists
`GET /api/v1/lol/teams/{slug}/roster/history`, but marks it as a Builder-plan
endpoint. The authentication guide requires an API key in the `x-api-key`
header. This confirms that a real intake needs approved provider access; the
public documentation itself supplies no payload, pre-event receipt, rights
grant, or verifiable source hash. The single procurement attempt therefore
failed to produce an authorizing source and G1 remains externally blocked.

### Cito access probe (2026-07-30)

The documented endpoint was probed read-only for a team-history payload:
`GET /api/v1/lol/teams/{slug}/roster/history`. The public demo credential was
rejected with `DEMO_KEY_RESTRICTED`; the response states that this endpoint
requires a full API key. No roster payload was received, retained, or used in
the model path. This confirms the procurement blocker; it does not establish
that the endpoint's historical rows satisfy the Scryglass contract.

The endpoint documentation and access terms remain external references only:
<https://lolesportsapi.com/lol-api-endpoints/> and
<https://citoapi.com/docs/>. A future intake must still independently verify
rights, historical coverage, pre-event availability, role semantics, and
immutable payload pinning before it can affect G1.

The Cito documentation illustrates player-team history rows with
`startedAt`, `endedAt`, and `role`. That is useful evidence that the provider's
data model may support interval checks, but it is not a received payload and
does not prove exact main-roster membership or availability before a match.

### Authorized GRID intake probe (2026-07-30)

Using the authorized workspace GRID credential, the Central Data `series(id)`
query for series `2974293` returned valid match metadata and an empty
`players` list. The same series' file listing exposed only timestamped event
streams, a completed-game replay, and final-state files; none carried a
pre-match roster assertion, source-available time, or rights receipt. No GRID
roster payload was retained or used in the model path.

The local warehouse audit found only one `roster.json` artifact, under the
single-game esports event `426848`; it contains observed participant rows and
no pre-event source timestamp. The GRID series JSON artifacts currently have
no `players` collection. These remain retrospective observations, not a
hidden G1 source.

Riot's documented `lobby-events/by-code/{tournamentCode}` endpoint exposes
pre-game joins and team-switch events, but its contract is tied to tournament
codes generated through the Tournament API. It is not a public historical
starting-lineup feed for Riot-sanctioned professional league matches, so it is
not a substitute for the missing G1 source.

## Decision

Do not combine these sources by player name, choose the most recent five, or
substitute observed lineups for pre-event rosters. Those shortcuts would make a
Team Rating look complete while violating the temporal and exact-roster
contract.

The smallest next action that can change G1 is one of:

1. provide an approved historical roster source with dated role/membership
   rows; or
2. explicitly authorize and provision a reviewed roster-history API, then
   create an independently pinned intake receipt.

Until one of those happens, downstream ratings and contextual Draft Score
remain private development artifacts or unavailable public results. Neutral
Draft Score follows its separate model-validation and independent-review gate.
