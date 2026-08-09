# LoL Oracle autoresearch journal

## Contract

- 500 frozen questions: 100 easy, 100 medium, 100 hard, 100 complex, 100 impossible.
- An answer is accepted only when its numeric result is exact under the pinned
  26.15 packet and it carries source links.
- Unsupported or impossible questions must block with a reason; a plausible
  number is a failure.
- The response-budget target is a hard ceiling of 30 seconds. Python kernel
  time and source-retrieval time are measured separately from assistant prose.

## Iterations

| Iteration | Change | Result | Decision |
| --- | --- | --- | --- |
| 0 | Existing quick engine, before benchmark hardening | 96.6% baseline-status agreement; 125/200 exact targets | Rejected: multi-level stat questions silently used the first level. |
| 1 | Reject multi-level single-stat questions | 100% baseline-status agreement; no silent level error; warm p95 about 0.21 ms | Accepted as a correctness guard. |
| 2 | Resident mechanics server | 500 warm calls: startup about 251 ms, p95 about 0.10 ms, total about 41 ms | Accepted as the fast transport path. |
| 3 | Resident Wiki search | 100 fast searches: p50 about 80 ms, p95 about 102 ms | Accepted as a fallback, but keep queries concise. |
| 4 | Add `LeagueOracleEngine` for stat deltas, resource windows, max
  resource/health, and spell cast budgets | 225/225 exact target rows; 500/500 rows carry links; baseline status accuracy 100%; warm p95 about 3 ms and max observed about 9 ms | Accepted. |
| 5 | Parse player shorthand such as `Q lvl 3 at level 6`; rerun the frozen set | 30 focused tests passed; 225/225 exact target rows; 500/500 rows carry links; warm p95 3.10 ms and max 8.92 ms; startup 244.68 ms | Accepted. |
| 6 | Add a fail-closed AP-only direct-damage graph kernel and promote only reproducible rows | 231/231 exact target rows; 500/500 rows carry links; baseline status accuracy 100%; warm p95 3.17 ms and max 9.44 ms; startup 282.70 ms | Accepted. |
| 7 | Parse manifest-verified passive-free static item stats and promote exact champion-plus-item totals | 256/256 exact target rows; 500/500 rows carry links; baseline status accuracy 100%; warm p95 13.27 ms and max 38.13 ms; startup 1.21 s | Accepted; still far below the 30-second ceiling. |
| 8 | Add positive magic-resistance mitigation for confirmed magic-tagged direct damage, with no penetration | 281/281 exact target rows; 500/500 rows carry links; baseline status accuracy 100%; warm p95 7.71 ms and max 26.31 ms; startup 529 ms | Accepted. |
| 9 | Add explicit level + total-AD direct physical graphs and armor mitigation; retain five blocked damage guards | 295/295 exact target rows; 500/500 rows carry links; baseline status accuracy 100%; warm p95 7.45 ms and max 21.70 ms; startup 566 ms | Accepted. |
| 10 | Add revision-receipted Wiki rule supplements for Manaflow Band, Summoner's Rift turrets/plates, and permanent stack formulas | 370/370 exact target rows; 500/500 rows carry links; baseline status accuracy 100%; warm p95 9.94 ms and max 36.14 ms; startup 867 ms | Accepted. The rune/structure/stack slices are explicit and fail closed on trigger or live-state ambiguity. |
| 11 | Run explicit numeric physical/magic/true damage sequences through the mechanics state kernel with trace hashes | 395/395 exact target rows; 500/500 rows carry links; baseline status accuracy 100%; warm p95 9.32 ms and max 32.87 ms; startup 1.09 s | Accepted. Sequence rows remain numeric and non-lethal; shield/item/rune/event-state variants stay blocked. |
| 12 | Add the semantic slot/state layer, explicit invalid-scenario validation, and closed direct/fight/counterfactual execution fixtures | 100/100 impossible-row contracts correct (75 `needs_input`, 25 `invalid_scenario`); 3/3 closed fixtures exact; triage p95 4.39 ms and max 20.73 ms; startup 1.70 s | Accepted. Missing state is now actionable instead of an opaque block; contradictory premises remain invalid rather than receiving fabricated numbers. |
| 13 | Put exact mechanics, semantic state, and revision-backed Wiki evidence behind one advertised resident MCP router | JSON-RPC verified with only `league_oracle_answer` + `league_oracle_status`; router startup 3.83 s; warm exact max 7.88 ms; warm semantic max 12.17 ms; exact and semantic route metadata present | Accepted. Legacy mechanics names remain compatibility aliases but are not advertised; Wiki evidence stays evidence-only and never becomes a guessed numeric result. |

## Current baseline

The final local run reports:

- 395 answerable target rows, all 395 exact;
- 105 rows intentionally blocked (five unsupported direct-damage graphs plus
  item passive/rune-trigger effects, shield/live-state variants, or
  impossible/underspecified prompts);
- 500/500 rows have at least two HTTP source links;
- resident engine startup about 0.81 seconds;
- latest warm 500-question calculation pass 2.18 s total, p95 7.56 ms,
  and a measured worst row of 26.71 ms—below 30 seconds by a wide margin.

This satisfies the latency ceiling for the local resident pipeline. It does
not yet make the oracle a full game emulator: the blocked 105-question slice
is the next capability frontier, and it must remain blocked until exact item
passive, rune-trigger, shield, and live event-state semantics are validated.
The accepted damage slice is intentionally limited to direct graphs whose
client calculation key and primitives are known, plus explicit AD/AP inputs,
positive armor/MR mitigation with a client-confirmed damage tag and no
penetration, and numeric ordered sequences with a trace receipt.

The semantic layer changes the shape of that frontier. Its 100 impossible
benchmark rows are no longer opaque: 75 receive a reproducible slot contract
and 25 are rejected as contradictory. Three completed state fixtures prove
that a filled contract can execute exact direct damage, deterministic fight,
and counterfactual event calculations. It still does not claim to emulate
unvalidated item passives, rune triggers, shields, or arbitrary champion
behavior; those remain explicit `unsupported` outcomes until their source
semantics are executable.
