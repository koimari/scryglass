# does codex cli have access to use dynamic tool calls and/or computer use tool?

| | |
|---|---|
| Session | `019e0379-b1fa-7212-857c-f82393b1d332` |
| Started | 2026-05-07T17:26:07.622Z |
| CWD | `/Users/river/Documents/Codex/2026-05-07/does-codex-cli-have-access-to` |
| Model provider | openai |
| CLI | 0.128.0-alpha.1 |
| Completed | True |
| Rollout | `/Users/river/.codex/sessions/2026/05/07/rollout-2026-05-07T14-26-07-019e0379-b1fa-7212-857c-f82393b1d332.jsonl` |

Tags: frontend

## User prompts

### 1

does codex cli have access to use dynamic tool calls and/or computer use tool?

*(+1 further user messages; see rollout JSONL)*

## Final agent message

Yes. Diagnostic mode: **Codex CLI in the terminal can use Computer Use without an open Codex app session**, at least on this machine.

I verified it with a terminal-only `codex exec` run. The CLI successfully called:

`computer-use/list_apps`

and returned apps including `Brave Browser`, `ChatGPT`, and `Codex`.

Important nuance: this does **not** mean Computer Use is built into the raw terminal UI itself. The CLI launches/uses the configured `computer-use` MCP entry, which points at a bundled local helper app:

`Codex Computer Use.app/.../SkyComputerUseClient`

So the answer is:

- **Codex desktop app session open?** No, not required.
- **Codex CLI + installed/enabled Computer Use plugin/MCP?** Yes, required.
- **macOS Accessibility/privacy permissions?** Still may matter for real clicking/typing/control.
- **Dynamic tool discovery in terminal CLI?** Also yes here; `tool_search` is enabled, but it only discovers permitted/registered tools, not arbitrary magic tools.

Tiny robot correction: my first answer blurred “available in this current Codex app session” with “usable from standalone terminal CLI.” The terminal-only probe confirms the stronger version: **yes, CLI can use it.**


## Tools and files

| Tool | Calls |
|---|---|
| exec_command | 8 |
