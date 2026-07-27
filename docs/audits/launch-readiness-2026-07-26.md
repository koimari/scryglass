# Scryglass launch-readiness audit — 2026-07-26

> **Superseded.** A deeper series-identity, player-identifiability, model-clock,
> and official-membership review invalidated this report's launch-ready
> conclusion. Use
> [`foundation-review-2026-07-26.md`](foundation-review-2026-07-26.md) as the
> controlling audit and
> [`../SCIENTIFIC_PRODUCT_ARCHITECTURE.md`](../SCIENTIFIC_PRODUCT_ARCHITECTURE.md)
> as the implementation contract. This file remains only as an audit trail of
> the earlier, narrower review.

Status: **GO received; implementation and validation complete in the isolated worktree. No merge or deploy performed.** This report covers the exact Blob pack currently pointed to by `apps/lol-atlas/public/packs/latest.json`, plus the membership-aware fixes in this worktree.

## Executive summary

The latest Blob pack is internally consistent at map, team-side, player-role, rating-history, and completion-provenance grains, but it is not launch-ready for public series/membership surfaces:

- **Launch blocker:** 3 GRID series have missing/non-contiguous game indices, and 8 multi-map GRID groups are tied. Existing format inference could label some as completed Bo3/Bo5. The frontend now fails closed to “Incomplete series”; the rating fit now excludes tied and gapped explicit GRID groups.
- **Major:** 4 GRID rows are classified as `INTL` even though their tournament titles are NACL and Circuito Desafiante. The ETL now maps those to `NACL` and `CD`, and unknown tournaments to `UNKNOWN` rather than silently promoting them to international scope.
- **Major:** 18 of 77 Tier 1 team records are more than 90 days older than the pack’s latest observation. Historical records remain intact; scoped ladders now apply a visible recent-observation guard and, when a tournament label exists, require observed participation in that league’s current tournament family.
- **Implemented:** current domestic tournament families are derived from the latest labeled tournament within the pack’s 90-day window. Stage suffixes such as LPL Group Ascend/Nirvana are merged into one family; leagues with no usable tournament label retain the dated-observation fallback.
- **Resolved in the latest refresh:** an earlier immutable pack (`v2026.07.26.1945`) omitted `grid_completion_source` and had `quality: null`; the current pack (`v2026.07.26.2209`) contains the full 104-column map contract, 61 event-confirmed GRID rows, 1 verified end-state-summary row, and a quality summary. The exporter guard remains implemented so this cannot silently regress.

## Exact current Blob-pack audit

Pack: `v2026.07.26.2209`, created `2026-07-26T22:11:38Z`, schema `1.3.0`.

| Grain | Result |
|---|---:|
| Maps | 6,314 (2025: 3,623; 2026: 2,691) |
| OE-backed maps | 6,252 |
| GRID-backed maps | 62 (61 event game-end, 1 verified end-state summary) |
| Unique `game_uid` / duplicate rows | 6,314 / 0 |
| Same-team sides | 0 |
| Result or `y_blue_win` inconsistencies | 0 / 0 |
| Required map-field null rows | 0 |
| Gamelength range | 1,103–3,558 seconds |
| Total-kills range | 7–85 |
| GRID series | 27 (13 two-map, 11 three-map, 3 singleton groups) |
| Gapped/non-contiguous GRID series | 3 |
| Tied multi-map GRID groups | 8 |
| Team rows | 32,792; 16,394 unique games; no duplicate `(gameid, side)` |
| Player rows | 163,960; 16,394 unique games; no duplicate `(gameid, side, position)` |
| Rating-history rows | 6,314; exact map-key overlap; no duplicate `game_uid` |

Representative series reproductions:

- `2975394` (LPL, Team WE vs JD Gaming) has indices `[1, 3, 4]`.
- `2975400` (LPL, ThunderTalk Gaming vs Edward Gaming) has only index `[2]`.
- `2966866` (LEC, Movistar KOI vs G2 Esports) has indices `[2, 3]` and is tied 1–1.
- The four mis-scoped `INTL` rows are two NACL games (`2975586`) and two Circuito Desafiante games (`2974291`, `2974292`).

The GRID refresh itself reports 34 discovered series, 29 files downloaded, 90 existing files, 62 parsed games, 66 skipped files, no failed files, no rate limiting, and 124 team rows / 620 player rows. This supports the freshness path; the latest pack also preserves completion-source provenance.

## Issue inventory

| Severity | Grain | Root cause | User impact | Fix / regression |
|---|---|---|---|---|
| Resolved / regression guard | Map/provenance | Export selected only columns present in the warehouse; older map frames could omit `grid_completion_source`. | The older `v2026.07.26.1945` pack could not distinguish event completion from verified summary fallback. | Export now materializes the complete map contract; audit fails closed on missing/null GRID provenance. Latest `v2026.07.26.2209` passes this check. Covered by `test_public_pack_audit.py` and existing live-grid provenance tests. |
| Launch blocker | Series | `inferBestOf` used winner score and map count without checking explicit GRID indices. | Missing games could be shown as completed Bo3/Bo5; singleton GRID records could be shown as Bo1. | Explicit GRID groups require contiguous indices and at least two maps; otherwise `bestOf=null`. Added four frontend regression cases and provenance labels. |
| Launch blocker / major | Rating series | Hierarchical BT used the first map to break tied series and accepted gapped explicit GRID groups. | Rating evidence depended on feed order or incomplete series. | Ties and gapped explicit GRID groups are excluded; skip counts are written to rating metadata. Added model regression coverage. |
| Major | Competition scope | GRID `_league_for` fell back to `INTL` and matched parent league substrings before developmental circuits. | NACL/CD rows leaked into international ladders and could distort scope-specific records. | Added specific developmental matches, word-boundary matching, and `UNKNOWN` fail-closed fallback. Added classification tests. |
| Major | Team membership | Records stored latest observed domestic affiliation but the UI called it current membership; old teams remained in Tier 1 chips. | Historical organizations could appear active in current scoped ladders. | Added 90-day recent-observation guard, `data_as_of`, plain-language caveat, and deterministic sort tie-breakers. Historical/all-scope views remain available. |
| Major / resolved in exporter | Current tournament membership | A league-only recency guard could include a team that had recent historical games but was absent from the current tournament. GRID’s historical `INTL` fallback could also be misread as a real scope. | Current regional/tier ladders could overstate active membership; developmental rows could leak into international scope. | Pack records now publish `current_tournament`; manifest publishes `current_tournaments`; regional/tier filters and scoped WR use the current tournament family when labeled. `INTL` canonicalizes to `UNKNOWN`, while future GRID ingestion maps NACL/CD explicitly. Added Python/TypeScript/audit regressions. |
| Resolved / regression guard | Pack manifest | An earlier Blob refresh had `quality: null`, despite the exporter producing a quality summary in this worktree. | The older pack did not expose pack-quality status from the public manifest. | Latest `v2026.07.26.2209` includes the quality summary; the audit reports any future omission. |

## Current-membership uncertainty requiring River’s decision

The official LoL Esports schedule remains the authoritative reference for active competition, not a warehouse-derived `current_date`: [official LEC schedule](https://lolesports.com/en-US/leagues/lec), [official LCS schedule](https://lolesports.com/en-US/leagues/lcs), [official LCP Split 3 standings](https://lolesports.com/en-US/tournament/115570728597462574/overview), and [official NACL Summer primer](https://lolesports.com/en-US/news/2026).

This worktree does not contain a maintained authoritative team-by-team membership/roster registry. The new tournament guard is therefore intentionally pack-derived and clearly labeled; it is not proof that a team is currently registered, nor proof that a team outside the window is inactive. A future authoritative registry can replace the signal without deleting historical rows.

## Validation evidence

- Python: `48` tests pass with `python3 -m unittest discover -s tests -p 'test_*.py'`; focused membership/audit tests pass.
- Frontend: `19/19` tests pass with `npm test`.
- Frontend lint: passes with `npm run lint`.
- Frontend production build: passes with `npm run build` on Next.js `16.2.11`.
- Representative v1.4 pack rebuild from the exact 6,314-map Blob slice: current tournament families were published for 6 leagues; the audit found 31 recent observed records outside those current families (intentionally excluded by scoped filters), plus the known 3 gapped and 8 tied GRID series.
- `git diff --check`: passes.
- No merge, commit, push, Blob publish, or deployment was performed.

## Remaining limitations

- This isolated worktree has no normalized OE/GRID warehouse snapshot, so a fresh full export could not be executed locally. The exact current Blob pack was downloaded read-only and audited directly; its map rows were replayed through the new membership derivation, which identified current labeled families for CBLOL, LCP, LCS, LEC, LJL, and LPL. LCK and other unlabeled leagues intentionally use the dated-observation fallback.
- The current Blob pack is still the prior immutable `v2026.07.26.2209` schema `1.3.0` artifact; it has not been republished with the new `1.4.0` membership fields. No external state was changed. A future refresh must rebuild the pack before the live `/elo` surface can enforce tournament membership.
- Official schedules establish current competition context, but a complete team-by-team current-membership reconciliation still needs an approved source/artifact. Historical participation must continue to be preserved.
