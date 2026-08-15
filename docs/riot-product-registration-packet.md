<!-- markdownlint-disable MD013 -->

# Riot Developer Portal registration packet

Status: **draft for owner submission**

This packet records the information for the Scryglass product registration. It
does not record approval. Keep `LEGAL-002` open until the owner has a Riot
Developer Portal receipt and a written scope decision.

## Source basis

The owner must check these pages again before submission:

- [Riot Games General Policies](https://developer.riotgames.com/policies/general)
- [Riot Games Developer Portal documentation](https://developer.riotgames.com/docs/portal)

The General Policies page says that products must be registered and audited
through the Developer Portal. It also says that product changes need a new
audit through the product page. The page was checked on 2026-08-15. Riot may
change its policies after this date.

## Submission values

| Portal field | Value to submit or confirm |
| --- | --- |
| Product name | Scryglass |
| Product type | Public website and independent research publication |
| Canonical URL | [scryglass.xyz](https://scryglass.xyz) |
| Game | League of Legends |
| Owner | Koi, exact owner or group name to be entered from the Developer Portal account |
| Availability | Public website. No user account is required. |
| Audience | Readers who want source-backed analysis of completed professional esports matches |
| Regions | Global readership. Historical competition scope is shown with each public result. |
| Monetization | Noncommercial. No betting, gambling, paid entry, subscription, advertising, or in-product purchase. |
| Riot API use in the public refresh | None in the current public refresh. The current pipeline uses the public sources listed below. |
| Riot API key | No Riot key is included in the public web application. The owner must confirm the account and key state in the Portal. |
| Direct client integration | None. Scryglass does not run inside the League client and does not change game state. |

### Product description

Scryglass is an independent, noncommercial research publication about
professional League of Legends esports. It lets readers inspect completed
professional matches, team and player research ratings, patch champion tier
lists, schedules, match details, and the methods and sources behind each
result. The site publishes derived research summaries with source dates and
release integrity checks. It has no game automation, client integration,
betting feature, or gambling feature.

Team and player ratings describe accepted professional match records. They are
research measures for esports analysis. They are not matchmaking, an in-game
rank, or a service for changing a player's official Riot account.

## Public features

The owner can describe these public surfaces:

- [Team and player ratings](https://scryglass.xyz/elo) from completed professional games.
- [Completed match browser](https://scryglass.xyz/matches) with lineups, player results, and published match facts.
- [Patch champion tier lists](https://scryglass.xyz/tiers) with the selected competition and patch scope.
- [Methodology](https://scryglass.xyz/methodology), [source notes](https://scryglass.xyz/sources), and [legal notices](https://scryglass.xyz/legal).
- [Ask Scryglass](https://scryglass.xyz/chat), which answers questions from the published data surface.

The public release keeps predictive Draft Score unavailable until an
independent promotion record passes the frozen evaluation and receipt process.
The public product has no Draft Score probability or other predictive draft
claim while that authority is unavailable.

## Data and ingest statement

| Source | Current use | Public boundary |
| --- | --- | --- |
| [Oracle's Elixir match data](https://lol.timsevenhuysen.com/matchdata/) | Completed professional game facts used to build accepted ratings, profiles, tier lists, and selected match summaries. | The [official FAQ answer](https://lol.timsevenhuysen.com/about/frequently-asked-questions/#comment-148) says that the data can only be used noncommercially. Scryglass is a noncommercial research publication. Raw source files stay private. This record closes the reuse gate for the current noncommercial product. Review it again before monetization or after a provider terms change. It is not a commercial-use license. |
| [League of Legends Wiki copyright information](https://wiki.leagueoflegends.com/en-us/League_of_Legends_Wiki:Copyrights) | Leaguepedia schedule and public identity metadata. | Public pages retain attribution and source links. The owner must verify the current license for each reused field. |
| [CommunityDragon](https://www.communitydragon.org/) | Champion image assets used on the public site. | The owner must confirm the permitted asset path with Riot or replace an asset when Riot requires a different source. |
| Riot supported services | No direct Riot API or League Client API call runs in the current public refresh. | Any future direct Riot ingest will use a supported service, a server-side key, the approved product scope, and the applicable rate limits. |
| GRID and other private research sources | Private research only. | No private source rows, credentials, signed URLs, or private source identifiers enter the public payload. |

The public refresh uses HTTPS, keeps secrets out of browser code, and serves
only accepted release assets. A failed source, identity, freshness, integrity,
or authority check keeps the previous accepted result active.

## Policy review points

The owner must ask Riot to review the product scope in the Portal. Riot's
General Policies name MMR and ELO calculators as prohibited alternatives to
official skill ranking systems. Scryglass uses team and player rating labels
for retrospective professional esports research. The owner must obtain Riot's
view on this feature scope before closing `LEGAL-002`.

The owner must also ask Riot to confirm the public use of CommunityDragon
champion images, or replace those images with an asset source that Riot
approves for the registered product.

Draft Score remains a separate authority gate. The public response uses
`authority: unavailable` until a frozen candidate, holdout evaluation,
independent review, and hash-bound receipt pass. Internal research artifacts
do not become public product claims through this registration.

## Public URLs for the application

| Purpose | URL |
| --- | --- |
| Canonical product | [scryglass.xyz](https://scryglass.xyz) |
| Ratings | [/elo](https://scryglass.xyz/elo) |
| Matches | [/matches](https://scryglass.xyz/matches) |
| Tier lists | [/tiers](https://scryglass.xyz/tiers) |
| Methodology | [/methodology](https://scryglass.xyz/methodology) |
| Data sources | [/sources](https://scryglass.xyz/sources) |
| Privacy | [/privacy](https://scryglass.xyz/privacy) |
| Legal and credits | [/legal](https://scryglass.xyz/legal) |
| Security reporting | [/security](https://scryglass.xyz/security) |
| RFC 9116 security contact | [security.txt](https://scryglass.xyz/.well-known/security.txt) |
| Repository | [github.com/koimari/scryglass](https://github.com/koimari/scryglass) |
| Private vulnerability report | [GitHub advisory form](https://github.com/koimari/scryglass/security/advisories/new) |

## Riot legal notice

This is the notice published on the current [Legal page](https://scryglass.xyz/legal):

> Scryglass isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.

The notice is visible on the public Legal route. Keep the text current if Riot
changes the required boilerplate.

## Owner fields

Fill these fields from the Developer Portal. Do not invent a value.

| Evidence field | Owner value |
| --- | --- |
| Developer Portal product ID | `OWNER: fill after registration` |
| Developer Portal account or group | `OWNER: fill exact account or group name` |
| Submission date and UTC time | `OWNER: fill after submission` |
| Portal status | `OWNER: fill current status` |
| Reviewer messages | `OWNER: attach or link the Portal message record` |
| Approved game and API scope | `OWNER: copy the exact approved scope` |
| Approved feature scope | `OWNER: copy the exact approved feature list` |
| Evidence screenshot or Portal export digest | `OWNER: record SHA-256 and private storage location` |
| Policy version checked before submission | `OWNER: record page date and check date` |
| Decision on team and player rating scope | `OWNER: record Riot guidance or approval message` |
| CommunityDragon asset decision | `OWNER: record Riot guidance or replacement asset decision` |

## Owner submission checklist

- [ ] Sign in to the [Riot Developer Portal](https://developer.riotgames.com/).
- [ ] Register Scryglass as a public product and verify ownership of the product.
- [ ] Paste the product description and the public URL list from this packet.
- [ ] State the noncommercial and non-betting status.
- [ ] State that the current public refresh uses public source data and has no direct Riot API call.
- [ ] Disclose team and player research ratings and request a policy decision under the official skill-ranking rule.
- [ ] Disclose CommunityDragon champion images and request a permitted asset decision.
- [ ] State that predictive Draft Score stays unavailable until independent promotion.
- [ ] Save the receipt, product ID, submission time, scope, and all Portal messages in the private owner record.

## Post-submission verification

- [ ] Confirm that the Portal record names Scryglass and the canonical URL `https://scryglass.xyz`.
- [ ] Confirm that the stored product ID matches the owner account and the evidence export.
- [ ] Check every reviewer message and record the required response or code change.
- [ ] Ship only features covered by the approved scope. Send each new feature or material change through the Portal audit path.
- [ ] Check the Legal, Privacy, Sources, Methodology, and Security pages in a clean browser session.
- [ ] Check that the Riot legal notice is visible and matches the approved text.
- [ ] Check that no Riot key, private source credential, or private research row reaches browser code or public assets.
- [ ] Check that the public Draft Score response remains unavailable while the promotion receipt is absent.
- [ ] Run the public-boundary, security-header, browser, and release-integrity checks after any approved change.
- [ ] Update `docs/public-release-readiness.md` with the receipt digest and approved scope. Close `LEGAL-002` only after owner review confirms the evidence.
