# <recommended_plugins>
Here is a list of plugins that are available but not installed.

- Atlassian R

| | |
|---|---|
| Session | `019f9c49-8430-7ff0-9446-522147c6b2b2` |
| Started | 2026-07-26T23:56:57.714Z |
| CWD | `/Users/river/scryglass` |
| Model provider | openai |
| CLI | 0.146.0-alpha.3.1 |
| Completed | True |
| Rollout | `/Users/river/.codex/sessions/2026/07/26/rollout-2026-07-26T20-56-57-019fa0dc-4368-7121-a035-6e1f1e62d947.jsonl` |

Tags: draft, ratings, tier-list, champion-atoms, grubs, live, market, frontend, data-warehouse, leaguepedia, evaluation, deploy

## User prompts

### 1

<recommended_plugins>
Here is a list of plugins that are available but not installed.

- Atlassian Rovo (atlassian-rovo@openai-curated-remote)
- Box (box@openai-curated-remote)
- Figma (figma@openai-curated-remote)
- Gmail (gmail@openai-curated-remote)
- Google Calendar (google-calendar@openai-curated-remote)
- Google Drive (google-drive@openai-curated-remote)
- Notion (notion@openai-curated-remote)
- Outlook Calendar (outlook-calendar@openai-curated-remote)
- Outlook Email (outlook-email@openai-curated-remote)
- SharePoint (sharepoint@openai-curated-remote)
- Slack (slack@openai-curated-remot

*(+4 further user messages; see rollout JSONL)*

## Final agent message

## Assessment B result

The deterministic scan is clean, with one tooling caveat and two rendered-only near-misses.

### Detector output

The requested wrapper failed identically for all targets:

`Error: bundled detector not found.`

Its expected detector bundle is absent. I therefore ran the cached Impeccable 3.3.1 engine directly and confirmed the same result with `npx --no-install`.

| Target | JSON | Count | Rules | Locations |
|---|---:|---:|---|---|
| [page.tsx](/Users/river/scryglass/apps/lol-atlas/src/app/elo/page.tsx:73) | `[]` | 0 | None | None |
| [EloLadders.tsx](/Users/river/scryglass/apps/lol-atlas/src/components/EloLadders.tsx:330) | `[]` | 0 | None | None |
| [SiteChrome.tsx](/Users/river/scryglass/apps/lol-atlas/src/components/SiteChrome.tsx:118) | `[]` | 0 | None | None |

All successful fallback scans exited `0`.

### Browser evidence

A fresh tab was opened at the requested URL and left available.

- Viewport: 1280×720, light presentation.
- Tier 1 is selected.
- 77 teams match; 20 rows are initially rendered.
- Default sort is adjusted rating.
- First three teams: T1, Gen.G, Hanwha Life Esports.
- 22 scope buttons are exposed simultaneously.
- Provenance panel

## Tools and files

| Tool | Calls |
|---|---|
| js | 9 |
| update_plan | 2 |
| wait | 1 |
