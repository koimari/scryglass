# <recommended_plugins>
Here is a list of plugins that are available but not installed.

- Atlassian R

| | |
|---|---|
| Session | `019fa08e-5b6c-7a81-af29-6ff859f8022e` |
| Started | 2026-07-27T07:54:19.486Z |
| CWD | `/Users/river/.codex/worktrees/1681/scryglass` |
| Model provider | openai |
| CLI | 0.146.0-alpha.3.1 |
| Completed | True |
| Rollout | `/Users/river/.codex/sessions/2026/07/27/rollout-2026-07-27T04-54-19-019fa291-4dc2-7c83-aeb5-82b19e15877b.jsonl` |

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

*(+2 further user messages; see rollout JSONL)*

## Final agent message

Implemented the explicit public/source patch contract without touching `draftComposition.ts` or styling.

Key behavior:

- Public `25.xx` → source `15.xx`
- Public `26.xx` → source `16.xx`
- UI/API now use `public_patch`; `26.14` becomes internal `source_patch_key: 16.14`
- Ambiguous `16.1`, `26.1`, and build-like API values are rejected
- Exact minors such as `26.01` and `26.10` are preserved
- Public metadata exposes named patch-contract objects instead of ambiguous source arrays
- Shared Sandbox URLs now use `public_patch`

Files:

- New contract: [patch.ts](/Users/river/.codex/worktrees/1681/scryglass/apps/lol-atlas/src/lib/patch.ts)
- New focused tests: [patch.test.mts](/Users/river/.codex/worktrees/1681/scryglass/apps/lol-atlas/src/lib/patch.test.mts)
- Boundaries: [Sandbox page](/Users/river/.codex/worktrees/1681/scryglass/apps/lol-atlas/src/app/sandbox/page.tsx), [DraftSandbox.tsx](/Users/river/.codex/worktrees/1681/scryglass/apps/lol-atlas/src/components/DraftSandbox.tsx), [draft-sandbox route](/Users/river/.codex/worktrees/1681/scryglass/apps/lol-atlas/src/app/api/draft-sandbox/route.ts), [draft-wr route](/Users/river/.codex/worktrees/1681/scryglass/apps/lol-atlas/src/app

## Tools and files

| Tool | Calls |
|---|---|
| exec_command | 29 |
| run | 1 |
| update_plan | 9 |
| wait | 10 |
| write_stdin | 1 |
