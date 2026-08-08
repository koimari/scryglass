# League oracle MCP server

`tools/lol_mechanics_mcp/server.py` is a local, dependency-free MCP
transport for one deterministic League answer path. At startup it loads one
compiled fastpack, the exact source-linked oracle, the semantic state engine,
and the read-only local Wiki index; repeated calls reuse all resident objects.
It never browses or makes network requests.

The normal process mode reads newline-delimited JSON-RPC messages from stdin:

```sh
python3 tools/lol_mechanics_mcp/server.py
```

For a direct local smoke test, use one-shot mode:

```sh
python3 tools/lol_mechanics_mcp/server.py --question \
  "How much MP5 does Malphite have at level 13?"
```

To make the process resident in Codex, add it as a local stdio MCP server
using the absolute command below, then restart Codex so it can discover the
two tools:

```toml
[mcp_servers.scryglass_lol_mechanics]
command = "/usr/bin/python3"
args = ["/Users/river/scryglass/tools/lol_mechanics_mcp/server.py"]
cwd = "/Users/river/scryglass"
startup_timeout_sec = 10
```

## Tools

`league_oracle_answer` is the only advertised answer tool and accepts
`{ "question": "...", "context": { ... } }`. Its fixed resident routing
order is:

1. exact patch-pinned mechanics;
2. semantic slot/state validation or execution;
3. compact local Wiki evidence for an unsupported rule;
4. explicit unavailable/unsupported result if evidence cannot authorize a
calculation.

The result includes `route`, `patch`, provenance, assumptions, and sources.
The connected model should display that structured result without a second
calculation or paraphrasing layer. Missing or unsupported mechanics remain
unavailable; Wiki evidence is never promoted into a numeric answer.

### Modifier-aware cast budgets

Cast-budget questions preserve explicit loadout state. The current exact packet
supports visible static item mana (for example, Sapphire Crystal), static
zero-mana items, and Manaflow Band's permanent maximum-mana component when its
state is stated as `0 stacks`, an exact count, `unstacked`, or `fully stacked`.
The answer includes item/rune components, total resource, floor division,
remainder, and source receipts. An omitted or partially described stack state,
passive item, or other rune returns unavailable rather than assuming full
stacks or silently dropping the modifier.

`league_oracle_status` reports resident exact, semantic, and Wiki components.
The old `league_mechanics_answer` and `league_mechanics_status` names remain
callable aliases for existing clients but are intentionally omitted from the
advertised tool list.

### Resident state warehouse

The same canonical tool also exposes a bounded, read-only warehouse operation:

```json
{
  "operation": "sql",
  "sql": "SELECT name, level, attack_damage FROM dim_champion JOIN fact_champion_stat USING (champion_key) WHERE level = 14 ORDER BY name LIMIT 20"
}
```

The normalized snapshot is materialized once at startup and contains
`dim_patch`, `dim_champion`, `fact_champion_stat`, `dim_ability`,
`fact_ability_value`, `dim_item`, `fact_item_stat`, `dim_rune`,
`dim_structure`, and `dim_rule_receipt`. `operation=state` accepts a structured
champion state (`champion`, `level`, `items`, `runes`, `rune_stacks`, and
optional live-state fields) and composes explicit flat components. Effect
text, non-empty buffs/debuffs, and transitions are returned as `unsupported`
with the blocker named; they are never silently omitted. `operation=schema`
and `operation=status` expose the resident table contract and snapshot receipt.

All SQL is a single `SELECT`/`WITH`, capped at 1,000 rows, and executes against
the patch-pinned in-memory connection. The warehouse benchmark reuses the
fixed 500-question corpus:

```sh
python3 benchmarks/run_league_warehouse_benchmark.py
```

The latest local run passed 500/500 champion-state rows. Startup was about
397 ms; SQL p95 was 0.035 ms and structured-state p95 was 0.062 ms. The
three-minute requirement is therefore met with a large resident margin, while
the snapshot hash and source packet remain attached to every result.

`league_mechanics_status` reports whether the engine and pack are resident,
the selected patch/index or configured fastpack path, answer count, and the
answer contract. It is an operational status check, not a source of game
facts.

## Pack selection

When `SCRYGLASS_LOL_MECHANICS_FASTPACK` is set, that exact path is loaded. A
missing or invalid explicit path is a startup error; the server never silently
falls back to an older patch. Without that variable, the newest local exact
`mechanics-index.json` under
`data/lol/knowledge/patch-packets/cdragon/` is selected by semantic patch
ordering (26.15 when present), compiled with `compile_fastpack(index_path)`,
and loaded with `load_fastpack(path)` when compilation returns a path. The
compiler may alternatively return an already-loaded pack.

The transport imports `QuickMechanicsEngine` from
`lol_kills.knowledge.quick_mechanics` and the compiler functions from
`lol_kills.knowledge.quick_mechanics_fastpack` lazily, with local fallback
module names so tests can inject fakes and the worker can evolve its module
split without changing the MCP boundary.

The intended hot path is parsing plus an in-memory lookup/calculation, with a
local process round trip comfortably below one second. The semantic and Wiki
objects are resident too, so an unsupported exact question does not spawn a
new Python process or choose an arbitrary external retrieval path.
