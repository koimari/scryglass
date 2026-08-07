# Explain this project to me

| | |
|---|---|
| Session | `019db718-30fa-7991-be51-9c1538de760b` |
| Started | 2026-04-22T21:28:29.183Z |
| CWD | `/Users/river/.codex/worktrees/6448/LOL DRAFT TOOL - STV VOTING` |
| Model provider | openai |
| CLI | 0.122.0-alpha.13 |
| Completed | True |
| Rollout | `/Users/river/.codex/sessions/2026/04/22/rollout-2026-04-22T18-28-29-019db718-30fa-7991-be51-9c1538de760b.jsonl` |

Tags: draft, champion-atoms, live, frontend, data-warehouse, evaluation

## User prompts

### 1

Explain this project to me

*(+11 further user messages; see rollout JSONL)*

## Final agent message

**Handoff Prompt**
Paste this into the new session:

```text
Continue work in /Users/river/.codex/worktrees/6448/LOL DRAFT TOOL - STV VOTING.

Before doing any UI/UX work, verify that the `impeccable` skill is actually available in this session. It was installed at /Users/river/.codex/skills/impeccable, but the old session could not hot-load it. If `impeccable` is not in the available-skills list, stop and debug skill loading first. If it is available, follow it and any required related design-skill rules before touching the frontend.

Current implementation status:
- Added canonical application layer in src/stv_draft_tool/application/draft_service.py and moved web/CLI toward it.
- Reworked draft state for flex-role resolution, explicit role overrides, and undoable role assignment.
- Made schemas/scoring provenance-aware so placeholder damage data no longer counts as real signal.
- Added CDragon supplement parsing and merged-source scraping.
- Rewrote the FastAPI contract to canonical shapes and rewrote the browser client to match it.
- Regenerated normalized data and manifest from the merged DDragon+CDragon pipeline.

Key files changed:
- src/stv_draft_tool/application/draft_servi

## Tools and files

| Tool | Calls |
|---|---|
| exec_command | 193 |
| request_user_input | 1 |
| update_plan | 3 |
| write_stdin | 4 |
