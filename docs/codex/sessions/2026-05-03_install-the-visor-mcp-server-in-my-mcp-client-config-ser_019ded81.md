# Install the Visor MCP server in my MCP client config.

Server URL: https://dashboard.visorfinance.ap

| | |
|---|---|
| Session | `019ded81-6279-79c2-8b93-53986f060c27` |
| Started | 2026-05-03T11:02:52.803Z |
| CWD | `/Users/river/Documents/Codex/2026-05-03/install-the-visor-mcp-server-in` |
| Model provider | openai |
| CLI | 0.128.0-alpha.1 |
| Completed | True |
| Rollout | `/Users/river/.codex/sessions/2026/05/03/rollout-2026-05-03T08-02-52-019ded81-6279-79c2-8b93-53986f060c27.jsonl` |

Tags: ratings, live, frontend

## User prompts

### 1

Install the Visor MCP server in my MCP client config.

Server URL: https://dashboard.visorfinance.app/mcp
Transport: streamable HTTP with OAuth 2.0 (opens a browser to log in to Visor on first use).

Detect which MCP client I'm using and apply the matching install method:

- Claude Code (CLI): run `claude mcp add visor --transport http https://dashboard.visorfinance.app/mcp`
- Codex (CLI): run `codex mcp add visor -- npx mcp-remote https://dashboard.visorfinance.app/mcp`
- Cursor: edit `~/.cursor/mcp.json` and add to `mcpServers`:
  `"visor": { "url": "https://dashboard.visorfinance.app/mcp" }

*(+1 further user messages; see rollout JSONL)*

## Final agent message

Diagnostic result: Visor is installed and OAuth login succeeded.

I also upgraded the Codex config from the `mcp-remote` bridge to native streamable HTTP because this Codex build supports it directly and the bridge hit a Visor OAuth parsing error.

Current verified config:

```text
visor
transport: streamable_http
url: https://dashboard.visorfinance.app/mcp
auth: OAuth
status: enabled
```

I tried to call it from this running session, but the live tool registry has not reloaded yet, so Visor is not callable here until Codex restarts. Tiny robot note: restart Codex now, then start a new turn and say “call visor”; it should appear as an available MCP server/tool after reload.

## Tools and files

| Tool | Calls |
|---|---|
| exec_command | 20 |
| list_mcp_resources | 1 |
| write_stdin | 2 |
