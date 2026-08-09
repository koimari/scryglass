# <recommended_plugins>
Here is a list of plugins that are available but not installed.

- Airtable (a

| | |
|---|---|
| Session | `019fbb42-e1bc-7281-86fb-e0658e891c63` |
| Started | 2026-08-01T02:59:10.418Z |
| CWD | `/Users/river/scryglass` |
| Model provider | openai |
| CLI | 0.146.0-alpha.9.2 |
| Completed | True |
| Rollout | `/Users/river/.codex/sessions/2026/07/31/rollout-2026-07-31T23-59-10-019fbb42-e1bc-7281-86fb-e0658e891c63.jsonl` |

Tags: draft, ratings, tier-list, champion-atoms, grubs, live, market, frontend, data-warehouse, leaguepedia, evaluation, deploy

## Codex rollout summary

```text
thread_id: 019fbb42-e1bc-7281-86fb-e0658e891c63
updated_at: 2026-08-01T04:14:36+00:00
rollout_path: /Users/river/.codex/sessions/2026/07/31/rollout-2026-07-31T23-59-10-019fbb42-e1bc-7281-86fb-e0658e891c63.jsonl
cwd: /Users/river/scryglass
git_branch: codex/app-visual-revamp

# Improved League Wiki retrieval latency and precision

Rollout context: In `/Users/river/scryglass`, the user asked for precise League of Legends mechanics answers, then objected that a fact lookup took over a minute. The agent traced the delay to broad, sequential retrieval and verification rather than the local SQLite query itself.

## Task 1: Explain current minion-wave timing

Outcome: partial

Preference signals:
- The user asked for a clearer, more precise answer with “a table with the actual spawn times of the wave” to understand the exact map state around 8 minutes. Similar questions should provide explicit wave numbers, spawn times, arrival estimates, cannon status, and patch/mode assumptions rather than only a single approximate time.

Failures and how to do differently:
- The initial answer gave an approximate 20–22 second travel time but did not satisfy the user’s request for a complete timing table.
- Extensive web searching produced conflicting/dated information and never yielded a fully verified table in the shown rollout. For current LoL timing questions, prefer the local, revision-backed knowledge source when available and clearly distinguish spawn time from lane-arrival time.

Reusable knowledge:
- Patch 26.1 evidence showed minion spawn time changed to 30 seconds, base minion movement speed to 350, waves every 25 seconds after 14:00 and every 20 seconds after 30:00. These facts alone do not establish exact mid-lane arrival timestamps without lane-distance/travel-time evidence.

## Task 2: Calculate Malphite level-13 base AD

Outcome: success

Key steps:
- Retrieved current indexed champion data showing Malphite base AD 62 and AD growth 4.
- Retrieved the League Wiki primary growth formula: `base + growth × (level − 1) × (0.7025 + 0.0175 × (level − 1))`.
- Calculated level 13 as `62 + 4 × 12 × (0.7025 + 0.0175 × 12) = 105.8`, with HUD-style rounding to approximately 106.

Reusable knowledge:
- The local database is `/Users/river/scryglass/data/lol/knowledge/league-wiki.sqlite3`.
- The League Wiki query skill and database preserve page IDs, revision IDs, timestamps, source URLs, raw wikitext, and section boundaries. Champion structured data is available in `Module:ChampionData/data`; the growth formula is documented on `Champion statistic` / `Template:Growth scaling/doc`.

References:
- `python3 /Users/river/.codex/skills/league-wiki-query/scripts/query_league_wiki.py page --title "Module:ChampionData/data" --namespace 828 --no-text`
- `python3 /Users/river/.codex/skills/league-wiki-query/scripts/query_league_wiki.py page --title "Champion statistic" --namespace 0 --no-text`
- `python3 /Users/river/.codex/skills/league-wiki-query/scripts/query_league_wiki.py page --title "Template:Growth scaling/doc" --namespace 10 --no-text`

## Task 3: Reduce League Wiki MCP latency and payload size

Outcome: success

Preference signals:
- After the lookup took over a minute, the user said: “so can we fix that?” The agent should proactively optimize repeated local fact lookups instead of accepting multi-step retrieval latency.
- The user explicitly compared the result to “65-330 ms,” indicating that sub-second retrieval and avoiding unnecessary tool calls are important acceptance criteria.

Key steps:
- Added `response_mode=fast|standard` to `tools/league_wiki_mcp/server.py`; fast mode is now the default.
- Fast mode limits evidence, related pages, structured supporting evidence, and text passage size; broader retrieval remains available through standard mode or explicit fetch/search/SQL calls.
- Updated MCP initialization instructions and README to tell connected agents to use one compact query first, then fetch more only if needed.
- Added a regression test for compact fast-mode responses.

Validation:
- `python3 -m pytest -q tests/test_league_wiki_mcp.py` -> `6 passed`.
- `python3 -m py_compile tools/league_wiki_mcp/server.py` succeeded.
- Direct default-fast query benchmark: about `138 ms`, payload about `3860` bytes.
- MCP subprocess benchmark: initialization plus query about `233 ms` cumulative.

Failures and how to do differently:
- Before the change, the default query returned broad evidence and encouraged unnecessary follow-up retrieval, even though local queries were already sub-second.
- One attempted patch failed because the expected tool-schema context did not match; the agent inspected the actual file and reapplied the narrower patch successfully.
- The final `git diff` was empty because the relevant files were untracked in the displayed repository state; do not infer that no changes exist. Check file contents and `git status` carefully.

References:
- `/Users/river/scryglass/tools/league_wiki_mcp/server.py`
- `/Users/river/scryglass/tests/test_league_wiki_mcp.py`
- `/Users/river/scryglass/tools/league_wiki_mcp/README.md`
- MCP config: `/Users/river/.codex/config.toml`, section `[mcp_servers.scryglass_league_wiki]`, launching `/Users/river/scryglass/tools/league_wiki_mcp/server.py`.
- New normal path: `league_wiki_query` with default `response_mode="fast"`; use `response_mode="standard"`, `league_wiki_fetch`, or `league_wiki_sql` only when compact evidence is insufficient.
```

## User prompts

### 1

<recommended_plugins>
Here is a list of plugins that are available but not installed.

- Airtable (airtable@openai-curated-remote)
- Apollo.io (apollo@openai-curated-remote)
- Asana (asana@openai-curated-remote)
- Atlassian Rovo (atlassian-rovo@openai-curated-remote)
- Base44 (base44@openai-curated-remote)
- Box (box@openai-curated-remote)
- Build iOS Apps (build-ios-apps@openai-curated-remote)
- Build macOS Apps (build-macos-apps@openai-curated-remote)
- Build Web Apps (build-web-apps@openai-curated-remote)
- Canva (canva@openai-curated-remote)
- Cloudflare (cloudflare@openai-curated-remote)


*(+15 further user messages; see rollout JSONL)*

## Final agent message

Yes. I fixed the retrieval path.

Changes:

- `league_wiki_query` now defaults to a compact `fast` mode.
- It returns only the evidence needed for a first answer.
- Full page/section retrieval is reserved for an explicit second pass.
- The MCP instructions now tell the connected agent to use one query first.

Benchmark:

- Direct Wiki query: ~138 ms.
- MCP subprocess round trip: ~233 ms including startup and initialization.
- Tests: 6 passed.

Updated files: [server.py](/Users/river/scryglass/tools/league_wiki_mcp/server.py), [tests](/Users/river/scryglass/tests/test_league_wiki_mcp.py), and [README](/Users/river/scryglass/tools/league_wiki_mcp/README.md).

The current desktop session may need a restart/new task to reload the updated MCP tool definition.

## Tools and files

| Tool | Calls |
|---|---|
| wait | 2 |
