# so let's fix the maths on that please.

| | |
|---|---|
| Session | `019fa609-18f1-7390-9034-683f66c034bf` |
| Started | 2026-07-28T00:04:01.941Z |
| CWD | `/Users/river/scryglass` |
| Model provider | openai |
| CLI | 0.146.0-alpha.3.1 |
| Completed | True |
| Rollout | `/Users/river/.codex/sessions/2026/07/28/rollout-2026-07-28T13-08-12-019fa97b-d482-76e1-b22a-fdd64a660266.jsonl` |

Tags: draft, champion-atoms, market, frontend, leaguepedia, evaluation, deploy

## User prompts

### 1

so let's fix the maths on that please.

*(+7 further user messages; see rollout JSONL)*

## Final agent message

Read-only audit complete against the current built page. No files changed.

## Current copy load

Default Gen.G–T1 example, five captures:

| Surface | Current words | One-third target |
|---|---:|---:|
| Site header | 3 | 3 |
| Hero | 47 | 16–20 |
| Tabs | 3 | 3 |
| Analysis | 758 | 200–220 |
| Methodology | 437 | 140–155 |
| Sources | 159 | 35–45 |
| Footer | 39 | 26 |
| **Total** | **1,446** | **423–452** |

Analysis is roughly 705 words when closed dropdown options are excluded. Counts vary slightly with the selected teams and capture path.

## Replacement map

### Hero

At [DrakeStudy.tsx:1798](/Users/river/scryglass/apps/elemental-drakes/src/components/DrakeStudy.tsx:1798):

- Remove: “Professional League of Legends · GRID telemetry”
- Heading:

  > What is each dragon worth?

- Body:

  > Choose two lineups, then assign each dragon.

- Replace the four-part cohort ledger with:

  > 6,504 complete games · 6,382 modeled

Move Tier 1 and international counts to Methodology.

Metadata at [layout.tsx:6](/Users/river/scryglass/apps/elemental-drakes/app/layout.tsx:6):

- Description:

  > Compare how dragon sequences change estimated map-win probability for two lineups.

- Social t

## Tools and files

| Tool | Calls |
|---|---|
| _review__e3fd48ffcc3e | 3 |
| exec_command | 1 |
| send_message | 4 |
| write_stdin | 2 |
