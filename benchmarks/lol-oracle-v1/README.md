# LoL Oracle benchmark v1

This is the first fixed benchmark for a personal, source-linked League of
Legends calculation oracle. It contains exactly 500 questions: 100 each at
five difficulty levels.

The benchmark separates two claims:

1. `target.status`: whether the intended oracle should eventually answer the
   question exactly or block it with a machine-readable reason;
2. `baseline.expected_status`: what the current `quick-mechanics-v1` engine is
   expected to do today.

The current oracle is deliberately narrow. It has exact, patch-pinned champion
stat rows, stat deltas, resource windows, maximum resource/health lookups,
spell resource-cost budgets, conservative AP/AD direct-ability-damage graphs,
passive-free static item-stat additions, direct positive-resistance mitigation
for confirmed physical or magic damage, Gromp rows, Voidgrub rewards, and
itemless basic-attack math. Revision-receipted Wiki supplements now cover
Manaflow Band's capped permanent mana, Summoner's Rift turret health/attack
clocks and plate gold, explicit ordered numeric damage sequences, and narrow
Nasus/Thresh/Senna/Kindred/Touch-of-the-Void stack formulas. Raw damage graphs
that require penetration, missing level/stat inputs, target state, or
unsupported client primitives—and item passives, trigger timing, shields, and
underspecified live-state questions—remain blocked instead of being guessed.

Every row carries at least two source links. Client-backed numeric rows include
the exact 26.15 CommunityDragon packet and the human-readable League Wiki page
or formula; Wiki-rule rows include a revision ID, timestamp, and content hash
for the local source snapshot. A blocked row still links to the evidence needed
to resolve it or to explain why it cannot currently be resolved.

Run the generator and warm-engine benchmark from the repository root:

```text
python3 benchmarks/lol-oracle-v1/generate_benchmark.py
python3 benchmarks/lol-oracle-v1/run_benchmark.py
```

The 30-second product target is measured at the assistant-response boundary,
not just Python calculation time. The runner reports the latter separately so
that retrieval/tool startup, answer composition, and source-link formatting do
not get hidden behind a fast arithmetic kernel.

## Semantic state benchmark

The 100 `impossible` rows are now also a semantic benchmark. Run:

```text
python3 benchmarks/lol-oracle-v1/run_semantic_benchmark.py
```

The semantic layer does not turn missing facts into guessed numbers. It
classifies the 75 incomplete rows as `needs_input` and returns typed slots for
patch, mode, build, target, timeline, or win condition; contradictory or
nonexistent patch/mode rows are `invalid_scenario`. Closed completion fixtures
then route direct damage, deterministic fights, and counterfactual event
sequences through the exact packet/state kernel. The current semantic run is
100/100 contract-correct and 3/3 completion fixtures exact, with a measured
semantic startup of about 1.70 seconds and sub-21 ms worst warm triage row.

The resident MCP now exposes the same contract through one advertised
`league_oracle_answer` tool. Its route is recorded as `exact_packet`,
`semantic_contract`, `semantic_execution`, `wiki_evidence`, or an explicit
unsupported result. The old mechanics tool names remain compatibility aliases,
but new Codex sessions see only the canonical oracle answer/status tools.
