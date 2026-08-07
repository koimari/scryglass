# open codex config.toml and change goals to = true

| | |
|---|---|
| Session | `019deeb3-ee38-70d1-ba8e-bdecbe28f408` |
| Started | 2026-05-03T16:37:42.594Z |
| CWD | `/Users/river/Code/finance` |
| Model provider | openai |
| CLI | 0.128.0 |
| Completed | True |
| Rollout | `/Users/river/.codex/sessions/2026/05/03/rollout-2026-05-03T13-37-42-019deeb3-ee38-70d1-ba8e-bdecbe28f408.jsonl` |

Tags: ratings, frontend

## User prompts

### 1

open codex config.toml and change goals to = true

*(+1 further user messages; see rollout JSONL)*

## Final agent message

It should work in a **new Codex CLI 0.128.0 terminal session**, but probably not in this current chat/thread.

I verified locally:

```text
codex-cli 0.128.0
goals  under development  true
```

So the config part is correct now. The likely reason it is “not working here” is:

1. `/goal` is a **client-side slash command**, not something I can activate from inside an already-running model turn.
2. This current session likely loaded before `goals = true` was added.
3. If you are using Codex Desktop/app chat rather than a fresh terminal `codex` TUI, the app shell may not expose the new slash command yet even though the CLI binary is 0.128.0.

Concrete next step:

```bash
cd /Users/river/Code/finance
codex
```

Then type `/goal` in the new Codex CLI session. Tiny robot note: if `/goal` still does not appear there, run:

```bash
codex features list | rg '^goals'
which codex
codex --version
```

Expected state is exactly what I saw: `/opt/homebrew/bin/codex`, `codex-cli 0.128.0`, and `goals ... true`.

## Tools and files

| Tool | Calls |
|---|---|
| exec_command | 19 |
