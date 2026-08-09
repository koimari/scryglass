# <recommended_plugins>
Here is a list of plugins that are available but not installed.

- Airtable (a

| | |
|---|---|
| Session | `019fbb95-4e20-7570-917d-1d047fcb80f0` |
| Started | 2026-08-01T04:29:12.123Z |
| CWD | `/Users/river/scryglass` |
| Model provider | openai |
| CLI | 0.146.0-alpha.9.2 |
| Completed | True |
| Rollout | `/Users/river/.codex/sessions/2026/08/01/rollout-2026-08-01T01-29-12-019fbb95-4e20-7570-917d-1d047fcb80f0.jsonl` |

Tags: draft, ratings, tier-list, champion-atoms, grubs, live, market, frontend, data-warehouse, leaguepedia, evaluation, deploy

## Codex rollout summary

```text
thread_id: 019fbb95-4e20-7570-917d-1d047fcb80f0
updated_at: 2026-08-01T05:20:21+00:00
rollout_path: /Users/river/.codex/sessions/2026/08/01/rollout-2026-08-01T01-29-12-019fbb95-4e20-7570-917d-1d047fcb80f0.jsonl
cwd: /Users/river/scryglass
git_branch: codex/app-visual-revamp

# Built and integrated a resident fast path for exact League mechanics lookups, but the final Viktor lookup still incurred avoidable orchestration latency.

Rollout context: Work occurred in `/Users/river/scryglass`. The user asked several precise LoL stat/mechanics questions and explicitly requires near-instant answers.

## Task 1: Initial Malphite stat lookup

Outcome: partial

Preference signals:
- After the lookup took over a minute, the user said: “I need near instant.” -> Simple exact stat lookups should be answered immediately, without broad web research or unnecessary provenance exploration.

Key steps:
- Queried the local League Wiki database and external search to establish Malphite’s base mana regeneration and level-scaling formula.
- Correct result: 13.32 MP5 at level 13.

Failures and how to do differently:
- The agent spent over a minute on a simple numeric lookup, repeatedly searching pages and attempting blocked web fetches. Use the resident/local mechanics fast path first; only use Wiki retrieval as fallback for unsupported or source-sensitive questions.

Reusable knowledge:
- League champion growth uses `base + growth × (level−1) × (0.7025 + 0.0175 × (level−1))`.
- Malphite’s relevant values are base mana regen 7.3 and growth 0.55, producing 13.3225 -> 13.32 at level 13.

## Task 2: Implement a fast local mechanics path

Outcome: success

Preference signals:
- The user’s complaint directly motivated a near-instant workflow. Future exact stat questions should default to the resident mechanics service rather than browsing or model-heavy reasoning.

Key steps:
- Added deterministic fastpack compiler: `lol_kills/knowledge/quick_mechanics_fastpack.py`.
- Added bounded query engine: `lol_kills/knowledge/quick_mechanics.py`.
- Added resident dependency-free MCP/JSON-RPC server: `tools/lol_mechanics_mcp/server.py`.
- Registered server in `/Users/river/.codex/config.toml` as `scryglass_lol_mechanics`; restart Codex is required for the already-running session to load it.
- Supported queries include champion MP5, MR, AD, full Voidgrub gold, and itemless auto-attacks-to-kill monsters. Typos such as `malphjite` resolve conservatively; unsupported mechanics fail closed.
- Exact 26.15 packet values validated: Malphite MP5 13.32, Malphite MR 50.45, Zaahen MP5 16.36, Renekton MP5 not applicable, full Voidgrub camp 90g, Tristana/Gromp example 58 attacks.

Reusable knowledge:
- Local packet: `data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json` (public patch 26.15, client patch 16.15).
- Wiki-grounded supplements include Gromp and Voidgrub data with revision/source metadata; supplements are unavailable for unvalidated patches rather than silently reused.
- Validation passed: `49 passed` across packet, engine, integration, and MCP tests.
- Benchmarks: cold first answer about 663 ms; resident direct dispatch about 0.08 ms; warm pipe mixed-query median 0.123 ms, p95 6.12 ms, max 6.33 ms.

Failures and how to do differently:
- Initial engine version mishandled duplicate base/cosmetic records and nested grub gold; it was fixed by base-first duplicate resolution and explicit nested reward handling.
- MCP registration was completed in config, but configuration changes do not affect the current Codex process until restart.

References:
- `/Users/river/scryglass/lol_kills/knowledge/quick_mechanics.py`
- `/Users/river/scryglass/lol_kills/knowledge/quick_mechanics_fastpack.py`
- `/Users/river/scryglass/tools/lol_mechanics_mcp/server.py`
- `/Users/river/.codex/config.toml:232`
- MCP tools: `league_mechanics_answer`, `league_mechanics_status`

## Task 3: Viktor base AD lookup and latency follow-up

Outcome: partial

Key steps:
- Raw packet showed Viktor base AD 53 and growth 3.0.
- Correct answer at level 17: 100.16 AD.

Failures and how to do differently:
- Despite the fast path being implemented, the final query launched the MCP cold and produced no visible output, then fell back to raw packet inspection; the user observed 35 seconds. The actual commands took about 1.4 seconds, with roughly 33 seconds attributed to Codex orchestration/inference. Restart Codex before relying on the newly configured MCP, and use the MCP directly rather than fallback probing.

References:
- Viktor packet lookup: base AD `53.0`, AD growth `3.0`.
- Final correct result: `100.16 AD` at level 17.
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


*(+13 further user messages; see rollout JSONL)*

## Final agent message

Because the new MCP was **configured but not loaded into this already-running Codex session**.

I then made it worse:

1. Launched Mochi cold; that call returned no visible output.
2. Fell back to querying the raw packet.
3. Codex added model/tool-routing latency around both calls.

The actual operations took only about **1.4 seconds combined**; roughly **33 seconds was Codex orchestration/inference latency**.

Restarting Codex will make Mochi resident and remove the cold/fallback path. Still, this chat interface cannot guarantee sub-second delivery—the only hard sub-second solution is a direct local command/UI that bypasses the model.

## Tools and files

| Tool | Calls |
|---|---|
| list_agents | 7 |
| send_message | 17 |
| spawn_agent | 3 |
| wait_agent | 8 |
