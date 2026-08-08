# League oracle modifier autoresearch — 2026-08-01

## Objective

Prevent a cast-budget answer from silently dropping item, rune, or stack state.
The fixed acceptance rule is: calculate only when every modifier is explicit
and patch-pinned; otherwise return an unavailable/unsupported result with the
missing state named. The response path must remain resident and deterministic.

## Fixed experiment

The corpus is [league_oracle_modifier_cases.jsonl](../benchmarks/league_oracle_modifier_cases.jsonl)
and currently contains 23 cases across:

- base/no-loadout casts;
- Sapphire Crystal plus zero, exact, unstacked, and fully stacked Manaflow;
- levels 1, 6, 10, and 18 and ranks 1, 3, and 5;
- zero-mana static items, multiple item quantities, and contradictory quantities;
- omitted or partially stated Manaflow state;
- named unsupported runes and passive/trigger items.

The runner keeps the questions and expected statuses fixed while measuring the
warm resident calculation path:

```sh
python3 benchmarks/run_league_oracle_autoresearch.py
```

## Iterations

| iteration | change | result |
| --- | --- | --- |
| baseline | `casts` matched the base budget before inspecting loadout | failed the four-part Malphite case: 6 returned, 13 expected |
| candidate A | composite static item + Manaflow arithmetic | 17/17 cases; named non-Manaflow runes still could be ignored |
| candidate B | patch-aware modifier packet, named-rune guard, explicit stack states, item quantities, zero-mana static items, and lookup fast-path | 23/23 (100%) |

Latest local run: 23/23, warm mean 2.90 ms, warm p95 4.03 ms, warm maximum
4.27 ms, packet-engine startup about 0.30 s. A fresh resident MCP subprocess
measured a first tool call around 12 ms after initialization and warm calls
around 3.4 ms; one-shot process startup was about 1.65 s.

## Packet contract now served

The answer carries `modifier_packet` version `modifier-packet-v1.0.0`, item
and rune components, total resource, floor division, remainder, patch hash,
and source receipts. Supported modifier states are:

- visible static item mana and visible static zero-mana items;
- explicit item quantities;
- Manaflow Band at `0 stacks`, an exact count, `unstacked`, or `fully stacked`.

The packet refuses to infer a full stack. It stays unavailable for omitted or
partially described Manaflow state, passive/trigger item effects, other named
runes, and temporary buffs/debuffs/shields/penetration.

## Resident reload note

The already-running Codex MCP process still reported server version `0.2.0`
and returned the old base-only result. The code is now server `0.3.0`; restart
Codex before trusting live `league_oracle_answer` calls so the resident process
loads the new packet. A new task is not required after the restart unless the
old tool definition remains cached.

## Remaining research boundary

This pass closes the item/rune composition bug for mana cast budgets; it does
not claim a general numeric packet for every rune, item passive, charge system,
mode, or live-state interaction. Those remain explicit unsupported cases until
their patch-specific rules and state transitions receive their own receipts and
benchmarks.

## Warehouse pass

The next autoresearch candidate materializes the same exact packet into a
resident, read-only SQLite star schema. This is the local equivalent of a
Snowflake-style serving warehouse: patch and source receipts are dimensions,
champion levels and item/ability values are facts, and every request carries a
snapshot hash. It is built once per MCP process and never rebuilt during a
question.

The fixed workload is the existing 500-question corpus, used as the benchmark
receipt while the warehouse exercises 500 deterministic champion/level rows.
The candidate is accepted only when all rows are exact and the resident p95 is
under the product ceiling of 180,000 ms:

```sh
python3 benchmarks/run_league_warehouse_benchmark.py
```

The latest run passed 500/500 rows (100% exact state accuracy). Startup was
397.293 ms; SQL p95 was 0.0345 ms; structured-state p95 was 0.0615 ms; and the
maximum measured warm operation was 0.3210 ms. The deterministic state digest
was `24b8bafe4a14105a41472011e9e04e18341d7596262730f4fe188baf3f40c850`.

The MCP keeps `league_oracle_answer` as the sole advertised tool to preserve
the deterministic path. `operation=sql`, `operation=schema`,
`operation=state`, and `operation=status` are optional operations on that tool;
the compatibility aliases `league_oracle_sql`, `league_warehouse_query`, and
`league_state_query` are callable for clients that prefer explicit names.

This closes the data/query layer, not the entire League emulator. Static flat
state composes immediately for every champion in the resident patch. Effect
text, runes without a revision-receipted executable rule, shields, buffs,
debuffs, and live transition state remain visible blockers until their own
typed rules and tests are added. That distinction is deliberate: a warehouse
can answer a broad query shape quickly without granting unsupported mechanics
false authority.
