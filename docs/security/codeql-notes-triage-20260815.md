<!-- markdownlint-disable MD013 -->

# CodeQL note alert triage, 2026-08-15

## Scope and snapshot

This inventory covers the open CodeQL alerts with note severity on `refs/heads/main`. The alert API returned 341 open CodeQL alerts: 341 notes, 0 warnings, and 0 errors. No security-severity alert is included in this document.

The latest Python CodeQL analysis is `1623973516` at `45e2ab0bb754599d0128f4d7a9b6845c59b533dd`. It reported `374` Python results. The JavaScript/TypeScript analysis at the same commit reported `0` results. The checkout and `origin/main` are `45e2ab0bb754599d0128f4d7a9b6845c59b533dd`.

The read-only API query was:

```text
gh api --paginate '/repos/koimari/scryglass/code-scanning/alerts?ref=refs/heads/main&state=open&tool_name=CodeQL&per_page=100'
```

The report uses the alert number, current analysis path, analysis line, finding message, and source line from the same head. It groups rows only when the CodeQL rule and source pattern share the same triage action. Every live alert remains listed.

### Live counts by rule

| Rule | Notes | Unique paths |
| --- | ---: | ---: |
| `py/unused-import` | 135 | 101 |
| `py/cyclic-import` | 74 | 44 |
| `py/unused-local-variable` | 51 | 38 |
| `py/empty-except` | 30 | 20 |
| `py/unused-global-variable` | 18 | 15 |
| `py/import-and-import-from` | 12 | 11 |
| `py/ineffectual-statement` | 12 | 4 |
| `py/unnecessary-lambda` | 5 | 5 |
| `py/imprecise-assert` | 2 | 2 |
| `py/catch-base-exception` | 1 | 1 |
| `py/unexpected-raise-in-special-method` | 1 | 1 |
| **Total** | **341** | **205** |

## Closed historical note alerts

The `state=all` query returned 46 historical note alerts with state `fixed`. These are excluded from the live inventory. They are listed here to show the stale records checked during the audit.

| Closed alert | Rule | Analysis commit | Path:line | Fixed at |
| ---: | --- | --- | --- | --- |
| 18 | `py/unused-local-variable` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/v2/ratings/team/estimands_v1.py:200` | 2026-08-15T19:07:01Z |
| 19 | `py/unused-local-variable` | `b1f25594314a304820d189896346ee85066c58a5` | `lol_kills/v2/data/g1_draft_features.py:303` | 2026-08-15T18:53:04Z |
| 23 | `py/unused-local-variable` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/research/grubs_intrinsic_value.py:1656` | 2026-08-15T19:07:01Z |
| 24 | `py/unused-local-variable` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/research/grubs_intrinsic_value.py:1657` | 2026-08-15T19:07:01Z |
| 25 | `py/unused-local-variable` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/research/grubs_intrinsic_value.py:1658` | 2026-08-15T19:07:01Z |
| 26 | `py/unused-local-variable` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/research/grubs_intrinsic_value.py:1659` | 2026-08-15T19:07:01Z |
| 27 | `py/unused-local-variable` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/research/grubs_intrinsic_value.py:2124` | 2026-08-15T19:07:01Z |
| 28 | `py/unused-local-variable` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/research/grubs_intrinsic_value.py:2125` | 2026-08-15T19:07:01Z |
| 29 | `py/unused-local-variable` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/research/grubs_intrinsic_value.py:2171` | 2026-08-15T19:07:01Z |
| 30 | `py/unused-local-variable` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/research/grubs_intrinsic_value.py:2174` | 2026-08-15T19:07:01Z |
| 41 | `py/unused-local-variable` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/v2/draft/interactions/model.py:1133` | 2026-08-15T19:07:01Z |
| 61 | `py/unused-local-variable` | `b1f25594314a304820d189896346ee85066c58a5` | `tests/model_v2/evaluation/test_evaluation_harness.py:381` | 2026-08-15T18:53:04Z |
| 62 | `py/unused-local-variable` | `b1f25594314a304820d189896346ee85066c58a5` | `tests/model_v2/data/test_g1_draft_features.py:45` | 2026-08-15T18:53:04Z |
| 63 | `py/unused-local-variable` | `97e58ec98b1809ffbbf3156f12d883c1eb092472` | `tests/model_v2/draft/interactions/test_g5_v2_math.py:392` | 2026-08-15T18:16:02Z |
| 64 | `py/unused-local-variable` | `97e58ec98b1809ffbbf3156f12d883c1eb092472` | `tests/model_v2/draft/interactions/test_g5_v2_math.py:393` | 2026-08-15T18:16:02Z |
| 116 | `py/unused-global-variable` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/ratings/player_elo.py:922` | 2026-08-15T19:07:01Z |
| 183 | `py/cyclic-import` | `97e58ec98b1809ffbbf3156f12d883c1eb092472` | `lol_kills/v2/market/match_winner_future_protocol_registry_v1.py:11` | 2026-08-15T18:16:02Z |
| 190 | `py/cyclic-import` | `97e58ec98b1809ffbbf3156f12d883c1eb092472` | `lol_kills/v2/market/phase_one_evaluation_readiness_registry_v1.py:10` | 2026-08-15T18:16:02Z |
| 225 | `py/cyclic-import` | `47f608fd2f3f09aa6bc88421ae25231564fbe7fa` | `lol_kills/v2/tierlists/pooled_candidate.py:49` | 2026-08-14T23:09:00Z |
| 244 | `py/import-and-import-from` | `97e58ec98b1809ffbbf3156f12d883c1eb092472` | `tests/model_v2/data/test_real_spine.py:358` | 2026-08-15T18:16:02Z |
| 245 | `py/import-and-import-from` | `97e58ec98b1809ffbbf3156f12d883c1eb092472` | `tests/model_v2/data/test_real_spine.py:649` | 2026-08-15T18:16:02Z |
| 246 | `py/import-and-import-from` | `97e58ec98b1809ffbbf3156f12d883c1eb092472` | `tests/model_v2/data/test_real_spine.py:679` | 2026-08-15T18:16:02Z |
| 258 | `py/unused-import` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/v2/tierlists/champion_elo.py:21` | 2026-08-15T19:07:01Z |
| 277 | `py/unused-import` | `a502b453fd2b1b9cf86a9f539f92a2f4855b27f5` | `lol_kills/ratings/dual_elo.py:13` | 2026-08-15T14:33:44Z |
| 278 | `py/unused-import` | `a502b453fd2b1b9cf86a9f539f92a2f4855b27f5` | `lol_kills/ratings/dual_elo.py:16` | 2026-08-15T14:33:44Z |
| 279 | `py/unused-import` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/v2/ratings/team/estimands_v1.py:28` | 2026-08-15T19:07:01Z |
| 294 | `py/unused-import` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/research/grubs_intrinsic_value.py:29` | 2026-08-15T19:07:01Z |
| 295 | `py/unused-import` | `b1f25594314a304820d189896346ee85066c58a5` | `lol_kills/research/grubs_ranked_contest_proof.py:27` | 2026-08-15T18:53:04Z |
| 301 | `py/unused-import` | `b1f25594314a304820d189896346ee85066c58a5` | `lol_kills/live_oe_prior.py:16` | 2026-08-15T18:53:04Z |
| 316 | `py/unused-import` | `b1f25594314a304820d189896346ee85066c58a5` | `lol_kills/v2/draft/interactions/oe_nuisance_baseline.py:15` | 2026-08-15T18:53:04Z |
| 322 | `py/unused-import` | `a502b453fd2b1b9cf86a9f539f92a2f4855b27f5` | `lol_kills/ratings/player_elo.py:38` | 2026-08-15T14:33:44Z |
| 324 | `py/unused-import` | `47f608fd2f3f09aa6bc88421ae25231564fbe7fa` | `lol_kills/v2/tierlists/pooled_candidate.py:49` | 2026-08-14T23:09:00Z |
| 331 | `py/unused-import` | `b1f25594314a304820d189896346ee85066c58a5` | `lol_kills/v2/data/real_spine.py:24` | 2026-08-15T18:53:04Z |
| 367 | `py/unused-import` | `b1f25594314a304820d189896346ee85066c58a5` | `tests/model_v2/evaluation/test_evaluation_harness.py:11` | 2026-08-15T18:53:04Z |
| 368 | `py/unused-import` | `b1f25594314a304820d189896346ee85066c58a5` | `tests/model_v2/evaluation/test_evaluation_harness.py:13` | 2026-08-15T18:53:04Z |
| 370 | `py/unused-import` | `b1f25594314a304820d189896346ee85066c58a5` | `tests/model_v2/data/test_g1_draft_features.py:3` | 2026-08-15T18:53:04Z |
| 371 | `py/unused-import` | `b1f25594314a304820d189896346ee85066c58a5` | `tests/model_v2/data/test_g1_draft_features_v3.py:3` | 2026-08-15T18:53:04Z |
| 378 | `py/unused-import` | `97e58ec98b1809ffbbf3156f12d883c1eb092472` | `tests/model_v2/ratings/player/test_multileague_v3_capture_readiness.py:9` | 2026-08-15T18:16:02Z |
| 379 | `py/unused-import` | `97e58ec98b1809ffbbf3156f12d883c1eb092472` | `tests/model_v2/ratings/player/test_multileague_v3_capture_readiness_v2.py:10` | 2026-08-15T18:16:02Z |
| 430 | `py/empty-except` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/research/draft_wr_study.py:440` | 2026-08-15T19:07:01Z |
| 434 | `py/empty-except` | `b1f25594314a304820d189896346ee85066c58a5` | `lol_kills/v2/data/g1_draft_features.py:347` | 2026-08-15T18:53:04Z |
| 438 | `py/empty-except` | `b1f25594314a304820d189896346ee85066c58a5` | `lol_kills/live_snapshots.py:204` | 2026-08-15T18:53:04Z |
| 439 | `py/empty-except` | `3426bedf2bda4c324116751e9e891012d2d9c9f6` | `lol_kills/export/pack_records.py:61` | 2026-08-15T17:06:17Z |
| 440 | `py/empty-except` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/ratings/player_elo.py:974` | 2026-08-15T19:07:01Z |
| 453 | `py/empty-except` | `b1f25594314a304820d189896346ee85066c58a5` | `tools/lol_mechanics_mcp/server.py:104` | 2026-08-15T18:53:04Z |
| 467 | `py/unused-import` | `2108f9809c9d55c6a08adf14a78ef85e89907326` | `lol_kills/ratings/player_elo.py:38` | 2026-08-15T19:07:01Z |

The current live set has 341 open note alerts. Each live instance has analysis commit `45e2ab0bb754599d0128f4d7a9b6845c59b533dd`; no fixed record is counted as live.

## Live inventory

Classification values are `confirmed cleanup`, `intentional/test-only`, `generated/false positive`, `cyclic architecture`, and `deferred source-contract review`. Release impact describes the risk of changing the code before the contract check. Closure actions are triage actions. This document does not call the dismissal API.

### `py/catch-base-exception`: BaseException catch with explicit interrupt re-raise (1)

- Classification: **intentional/test-only**
- Release impact: Moderate runtime impact. The catch converts unexpected pipeline failures into blocked evidence while preserving KeyboardInterrupt and SystemExit.
- Exact closure action: Keep the fail-closed boundary. Add an explanatory comment and review whether the catch can narrow to the remaining intended exception set. Run the G5 runner tests.

- **#421** `lol_kills/v2/draft/interactions/g5_exploratory/runner.py:462` — source evidence `462: except BaseException as error: / 463: if isinstance(error, (KeyboardInterrupt, SystemExit)):` — finding: Except block directly handles BaseException.

### `py/cyclic-import`: Import graph contains a cycle (74)

- Classification: **cyclic architecture**
- Release impact: High architectural impact. Import order can control module initialization and can prevent isolated tests.
- Exact closure action: Map each cycle. Move shared types, constants, or protocol contracts into a leaf module, or move the import into the smallest runtime function. Run import smoke tests and the affected package tests.

- **#159** `lol_kills/v2/market/betano_br_quote_adapter_v2.py:23` — source evidence `23: from . import event_probability_v2 as probability_v2 / 24: from . import phase_one_evaluation_v1 as evaluation` — finding: Import of module lol_kills.v2.market.event_probability_v2 begins an import cycle.
- **#160** `lol_kills/v2/market/betano_br_quote_adapter_v2.py:25` — source evidence `25: from . import phase_two_event_plan_v1 as event_plan / 26:` — finding: Import of module lol_kills.v2.market.phase_two_event_plan_v1 begins an import cycle.
- **#161** `lol_kills/v2/market/betano_br_quote_qualification_v1.py:20` — source evidence `20: from . import betano_br_quote_adapter_v2 as quote_v2 / 21: from . import phase_one_evaluation_v1 as evaluation` — finding: Import of module lol_kills.v2.market.betano_br_quote_adapter_v2 begins an import cycle.
- **#162** `lol_kills/v2/market/betano_br_quote_registry_v2.py:12` — source evidence `12: from . import betano_br_quote_qualification_v1 as qualification / 13: from . import event_probability_registry_v2 as probability_registry` — finding: Import of module lol_kills.v2.market.betano_br_quote_qualification_v1 begins an import cycle.
- **#163** `lol_kills/v2/market/betano_br_quote_registry_v2.py:13` — source evidence `13: from . import event_probability_registry_v2 as probability_registry / 14: from . import phase_one_evaluation_v1 as evaluation` — finding: Import of module lol_kills.v2.market.event_probability_registry_v2 begins an import cycle.
- **#164** `lol_kills/v2/draft/terminal/candidate_registry_v3.py:24` — source evidence `24: from .development_artifact_v3 import ( / 25: DEFAULT_ARTIFACT,` — finding: Import of module lol_kills.v2.draft.terminal.development_artifact_v3 begins an import cycle. Import of module terminal.development_artifact_v3 begins an import cycle.
- **#165** `lol_kills/v2/draft/terminal/candidate_registry_v3.py:34` — source evidence `34: from .model import TerminalDraft, TerminalModel, score_terminal_draft / 35:` — finding: Import of module lol_kills.v2.draft.terminal.model begins an import cycle. Import of module terminal.model begins an import cycle.
- **#166** `lol_kills/v2/draft/terminal/capture_readiness_registry_v1.py:11` — source evidence `11: from .capture_readiness_v1 import ( / 12: DEFAULT_OUTPUT,` — finding: Import of module terminal.capture_readiness_v1 begins an import cycle. Import of module lol_kills.v2.draft.terminal.capture_readiness_v1 begins an import cycle.
- **#167** `lol_kills/v2/draft/terminal/capture_readiness_v1.py:26` — source evidence `26: from .future_prediction_ledger import ( / 27: AUTHORITY_KEYS,` — finding: Import of module terminal.future_prediction_ledger begins an import cycle. Import of module lol_kills.v2.draft.terminal.future_prediction_ledger begins an import cycle.
- **#168** `lol_kills/v2/draft/terminal/capture_readiness_v1.py:35` — source evidence `35: from .future_protocol_registry_v1 import ( / 36: REGISTERED_PROTOCOL_ARTIFACT_SHA256,` — finding: Import of module lol_kills.v2.draft.terminal.future_protocol_registry_v1 begins an import cycle. Import of module terminal.future_protocol_registry_v1 begins an import cycle.
- **#169** `lol_kills/v2/tierlists/champion_elo.py:965` — source evidence `965: from .pooled_candidate import build_pooled_candidate / 966:` — finding: Import of module lol_kills.v2.tierlists.pooled_candidate begins an import cycle.
- **#170** `lol_kills/v2/draft/terminal/development_artifact_v3.py:42` — source evidence `42: from .model import TerminalDraft, TerminalModel, score_terminal_draft / 43:` — finding: Import of module lol_kills.v2.draft.terminal.model begins an import cycle. Import of module terminal.model begins an import cycle.
- **#171** `lol_kills/v2/market/event_probability_registry_v2.py:12` — source evidence `12: from . import event_probability_v2 as probability / 13: from . import phase_one_evaluation_v1 as evaluation` — finding: Import of module lol_kills.v2.market.event_probability_v2 begins an import cycle.
- **#172** `lol_kills/v2/market/event_probability_v2.py:16` — source evidence `16: from . import phase_two_opening_v1 as opening / 17: from .match_winner_future_protocol_registry_v1 import (` — finding: Import of module lol_kills.v2.market.phase_two_opening_v1 begins an import cycle.
- **#173** `lol_kills/v2/draft/terminal/future_prediction_ledger.py:33` — source evidence `33: from .future_protocol_registry_v1 import ( / 34: REGISTERED_PROTOCOL_ARTIFACT_SHA256,` — finding: Import of module lol_kills.v2.draft.terminal.future_protocol_registry_v1 begins an import cycle. Import of module terminal.future_protocol_registry_v1 begins an import cycle.
- **#174** `lol_kills/v2/draft/terminal/future_prediction_ledger.py:39` — source evidence `39: from .model import TerminalModel / 40:` — finding: Import of module lol_kills.v2.draft.terminal.model begins an import cycle. Import of module terminal.model begins an import cycle.
- **#175** `lol_kills/v2/draft/terminal/future_protocol_registry_v1.py:11` — source evidence `11: from .future_protocol_v1 import ( / 12: DEFAULT_OUTPUT,` — finding: Import of module terminal.future_protocol_v1 begins an import cycle. Import of module lol_kills.v2.draft.terminal.future_protocol_v1 begins an import cycle.
- **#176** `lol_kills/v2/draft/terminal/future_protocol_v1.py:37` — source evidence `37: from .candidate_registry_v3 import ( / 38: DEFAULT_OUTPUT as CANDIDATE_REGISTRY,` — finding: Import of module lol_kills.v2.draft.terminal.candidate_registry_v3 begins an import cycle. Import of module terminal.candidate_registry_v3 begins an import cycle.
- **#177** `lol_kills/research/grid_sequence_actions.py:22` — source evidence `22: from lol_kills.research import grid_sequence_review as core / 23:` — finding: Import of module lol_kills.research.grid_sequence_review begins an import cycle.
- **#178** `lol_kills/v2/draft/terminal/grid_source_readiness_registry_v1.py:11` — source evidence `11: from .grid_source_readiness_v1 import ( / 12: DEFAULT_OUTPUT,` — finding: Import of module terminal.grid_source_readiness_v1 begins an import cycle. Import of module lol_kills.v2.draft.terminal.grid_source_readiness_v1 begins an import cycle.
- **#179** `lol_kills/v2/draft/terminal/grid_source_readiness_v1.py:30` — source evidence `30: from .future_prediction_ledger import AUTHORITY_KEYS / 31:` — finding: Import of module terminal.future_prediction_ledger begins an import cycle. Import of module lol_kills.v2.draft.terminal.future_prediction_ledger begins an import cycle.
- **#180** `lol_kills/research/grid_sequence_review.py:2193` — source evidence `2193: from lol_kills.research.grid_sequence_actions import run_analysis_action_graph / 2194:` — finding: Import of module lol_kills.research.grid_sequence_actions begins an import cycle.
- **#181** `lol_kills/v2/draft/terminal/grid_source_readiness_v1.py:581` — source evidence `581: from .future_prediction_ledger import write_no_clobber / 582:` — finding: Import of module terminal.future_prediction_ledger begins an import cycle. Import of module lol_kills.v2.draft.terminal.future_prediction_ledger begins an import cycle.
- **#182** `lol_kills/knowledge/lol_oracle.py:1958` — source evidence `1958: from .semantic_engine import SemanticOracleEngine / 1959:` — finding: Import of module lol_kills.knowledge.semantic_engine begins an import cycle.
- **#184** `lol_kills/v2/market/match_winner_future_protocol_v1.py:32` — source evidence `32: from lol_kills.v2.draft.terminal.capture_readiness_registry_v1 import ( / 33: REGISTERED_CAPTURE_ARTIFACT_SHA256 as DRAFT_CAPTURE_ARTIFACT_SHA256,` — finding: Import of module lol_kills.v2.draft.terminal.capture_readiness_registry_v1 begins an import cycle. Import of module terminal.capture_readiness_registry_v1 begins an import cycle.
- **#185** `lol_kills/v2/market/match_winner_future_protocol_v1.py:38` — source evidence `38: from lol_kills.v2.draft.terminal.future_protocol_registry_v1 import ( / 39: REGISTERED_PROTOCOL_ARTIFACT_SHA256 as DRAFT_PROTOCOL_ARTIFACT_SHA256,` — finding: Import of module lol_kills.v2.draft.terminal.future_protocol_registry_v1 begins an import cycle. Import of module terminal.future_protocol_registry_v1 begins an import cycle.
- **#186** `lol_kills/v2/market/match_winner_future_protocol_v1.py:44` — source evidence `44: from lol_kills.v2.draft.terminal.grid_source_readiness_registry_v1 import ( / 45: REGISTERED_GRID_SOURCE_ARTIFACT_SHA256,` — finding: Import of module lol_kills.v2.draft.terminal.grid_source_readiness_registry_v1 begins an import cycle. Import of module terminal.grid_source_readiness_registry_v1 begins an import cycle.
- **#187** `lol_kills/v2/draft/terminal/model.py:20` — source evidence `20: from lol_kills.v2.draft.terminal.promotion import TerminalPromotionBindings, promotion_receipt_authorizes / 21:` — finding: Import of module lol_kills.v2.draft.terminal.promotion begins an import cycle. Import of module terminal.promotion begins an import cycle.
- **#188** `lol_kills/v2/draft/interactions/oe_target_authority.py:18` — source evidence `18: from .oe_target_evidence import OETargetEvidenceError / 19:` — finding: Import of module lol_kills.v2.draft.interactions.oe_target_evidence begins an import cycle. Import of module interactions.oe_target_evidence begins an import cycle.
- **#189** `lol_kills/v2/draft/interactions/oe_target_evidence.py:519` — source evidence `519: from .oe_target_authority import ( / 520: require_exact_human_authority as require_independent_authority,` — finding: Import of module lol_kills.v2.draft.interactions.oe_target_authority begins an import cycle. Import of module interactions.oe_target_authority begins an import cycle.
- **#191** `lol_kills/v2/market/phase_one_collection_v1.py:32` — source evidence `32: from lol_kills.v2.draft.terminal import future_prediction_ledger as draft_ledger / 33: from lol_kills.v2.draft.terminal.future_protocol_registry_v1 import (` — finding: Import of module terminal.future_prediction_ledger begins an import cycle. Import of module lol_kills.v2.draft.terminal.future_prediction_ledger begins an import cycle.
- **#192** `lol_kills/v2/market/phase_one_collection_v1.py:33` — source evidence `33: from lol_kills.v2.draft.terminal.future_protocol_registry_v1 import ( / 34: REGISTERED_PROTOCOL_ARTIFACT_SHA256 as DRAFT_PROTOCOL_ARTIFACT_SHA256,` — finding: Import of module lol_kills.v2.draft.terminal.future_protocol_registry_v1 begins an import cycle. Import of module terminal.future_protocol_registry_v1 begins an import cycle.
- **#193** `lol_kills/v2/market/phase_one_collection_v1.py:52` — source evidence `52: from .match_winner_future_protocol_registry_v1 import ( / 53: REGISTERED_PROTOCOL_ARTIFACT_SHA256 as MARKET_PROTOCOL_ARTIFACT_SHA256,` — finding: Import of module lol_kills.v2.market.match_winner_future_protocol_registry_v1 begins an import cycle.
- **#194** `lol_kills/v2/market/phase_one_evaluation_readiness_v1.py:20` — source evidence `20: from . import phase_one_opening_v1 as opening / 21: from .match_winner_future_protocol_registry_v1 import (` — finding: Import of module lol_kills.v2.market.phase_one_opening_v1 begins an import cycle.
- **#195** `lol_kills/v2/market/phase_one_evaluation_registry_v1.py:12` — source evidence `12: from . import phase_one_evaluation_v1 as evaluation / 13:` — finding: Import of module lol_kills.v2.market.phase_one_evaluation_v1 begins an import cycle.
- **#196** `lol_kills/v2/market/phase_one_evaluation_v1.py:26` — source evidence `26: from lol_kills.v2.draft.terminal import future_prediction_ledger as draft_ledger / 27: from lol_kills.v2.ratings.player import (` — finding: Import of module terminal.future_prediction_ledger begins an import cycle. Import of module lol_kills.v2.draft.terminal.future_prediction_ledger begins an import cycle.
- **#197** `lol_kills/v2/market/phase_one_evaluation_v1.py:31` — source evidence `31: from . import phase_one_collection_v1 as collection / 32:` — finding: Import of module lol_kills.v2.market.phase_one_collection_v1 begins an import cycle.
- **#198** `lol_kills/v2/market/phase_one_opening_v1.py:118` — source evidence `118: from .phase_one_evaluation_readiness_registry_v1 import ( / 119: REGISTERED_READINESS_ARTIFACT_SHA256,` — finding: Import of module lol_kills.v2.market.phase_one_evaluation_readiness_registry_v1 begins an import cycle.
- **#199** `lol_kills/v2/market/phase_two_attempt_completion_v1.py:13` — source evidence `13: from . import betano_br_quote_adapter_v2 as quote_v2 / 14: from . import betano_br_quote_qualification_v1 as qualification` — finding: Import of module lol_kills.v2.market.betano_br_quote_adapter_v2 begins an import cycle.
- **#200** `lol_kills/v2/market/phase_two_attempt_completion_v1.py:14` — source evidence `14: from . import betano_br_quote_qualification_v1 as qualification / 15: from . import phase_one_evaluation_v1 as evaluation` — finding: Import of module lol_kills.v2.market.betano_br_quote_qualification_v1 begins an import cycle.
- **#201** `lol_kills/v2/market/phase_two_attempt_completion_v1.py:16` — source evidence `16: from . import phase_two_event_plan_v1 as event_plan / 17: from . import phase_two_quote_attempt_v1 as attempt` — finding: Import of module lol_kills.v2.market.phase_two_event_plan_v1 begins an import cycle.
- **#202** `lol_kills/v2/market/phase_two_attempt_completion_v1.py:17` — source evidence `17: from . import phase_two_quote_attempt_v1 as attempt / 18:` — finding: Import of module lol_kills.v2.market.phase_two_quote_attempt_v1 begins an import cycle.
- **#203** `lol_kills/v2/market/phase_two_collection_readiness_registry_v1.py:12` — source evidence `12: from . import phase_two_collection_readiness_v1 as readiness / 13:` — finding: Import of module lol_kills.v2.market.phase_two_collection_readiness_v1 begins an import cycle.
- **#204** `lol_kills/v2/market/phase_two_collection_readiness_v1.py:14` — source evidence `14: from . import betano_br_quote_adapter_v2 as quote_v2 / 15: from . import betano_br_quote_qualification_v1 as quote_qualification` — finding: Import of module lol_kills.v2.market.betano_br_quote_adapter_v2 begins an import cycle.
- **#205** `lol_kills/v2/market/phase_two_collection_readiness_v1.py:15` — source evidence `15: from . import betano_br_quote_qualification_v1 as quote_qualification / 16: from . import betano_br_quote_registry_v2 as quote_registry` — finding: Import of module lol_kills.v2.market.betano_br_quote_qualification_v1 begins an import cycle.
- **#206** `lol_kills/v2/market/phase_two_collection_readiness_v1.py:16` — source evidence `16: from . import betano_br_quote_registry_v2 as quote_registry / 17: from . import event_probability_registry_v2 as probability_registry` — finding: Import of module lol_kills.v2.market.betano_br_quote_registry_v2 begins an import cycle.
- **#207** `lol_kills/v2/market/phase_two_collection_readiness_v1.py:17` — source evidence `17: from . import event_probability_registry_v2 as probability_registry / 18: from . import event_probability_v2 as probability_v2` — finding: Import of module lol_kills.v2.market.event_probability_registry_v2 begins an import cycle.
- **#208** `lol_kills/v2/market/phase_two_collection_readiness_v1.py:18` — source evidence `18: from . import event_probability_v2 as probability_v2 / 19: from . import event_rating_bootstrap_v1 as rating_bootstrap` — finding: Import of module lol_kills.v2.market.event_probability_v2 begins an import cycle.
- **#209** `lol_kills/v2/market/phase_two_collection_readiness_v1.py:22` — source evidence `22: from . import phase_two_opening_v1 as opening / 23: from . import phase_two_event_plan_v1 as event_plan` — finding: Import of module lol_kills.v2.market.phase_two_opening_v1 begins an import cycle.
- **#210** `lol_kills/v2/market/phase_two_collection_readiness_v1.py:23` — source evidence `23: from . import phase_two_event_plan_v1 as event_plan / 24: from . import phase_two_quote_attempt_v1 as quote_attempt` — finding: Import of module lol_kills.v2.market.phase_two_event_plan_v1 begins an import cycle.
- **#211** `lol_kills/v2/market/phase_two_collection_readiness_v1.py:24` — source evidence `24: from . import phase_two_quote_attempt_v1 as quote_attempt / 25: from . import phase_two_attempt_completion_v1 as attempt_completion` — finding: Import of module lol_kills.v2.market.phase_two_quote_attempt_v1 begins an import cycle.
- **#212** `lol_kills/v2/market/phase_two_collection_readiness_v1.py:25` — source evidence `25: from . import phase_two_attempt_completion_v1 as attempt_completion / 26: from . import phase_two_stopping_snapshot_v1 as stopping_snapshot` — finding: Import of module lol_kills.v2.market.phase_two_attempt_completion_v1 begins an import cycle.
- **#213** `lol_kills/v2/market/phase_two_collection_readiness_v1.py:26` — source evidence `26: from . import phase_two_stopping_snapshot_v1 as stopping_snapshot / 27: from . import phase_two_stopping_snapshot_registry_v1 as stopping_registry` — finding: Import of module lol_kills.v2.market.phase_two_stopping_snapshot_v1 begins an import cycle.
- **#214** `lol_kills/v2/market/phase_two_collection_readiness_v1.py:27` — source evidence `27: from . import phase_two_stopping_snapshot_registry_v1 as stopping_registry / 28: from .betano_br_quote_adapter_registry_v1 import (` — finding: Import of module lol_kills.v2.market.phase_two_stopping_snapshot_registry_v1 begins an import cycle.
- **#215** `lol_kills/v2/market/phase_two_evaluation_readiness_registry_v1.py:13` — source evidence `13: from . import phase_two_evaluation_readiness_v1 as readiness / 14:` — finding: Import of module lol_kills.v2.market.phase_two_evaluation_readiness_v1 begins an import cycle.
- **#216** `lol_kills/v2/market/phase_two_evaluation_readiness_v1.py:17` — source evidence `17: from . import phase_two_outcome_opening_v1 as opening / 18: from . import phase_two_stopping_snapshot_registry_v1 as snapshot_registry` — finding: Import of module lol_kills.v2.market.phase_two_outcome_opening_v1 begins an import cycle.
- **#217** `lol_kills/v2/market/phase_two_event_plan_v1.py:13` — source evidence `13: from . import event_probability_v2 as probability / 14: from . import fast_event_uncertainty_v1 as fast_uncertainty` — finding: Import of module lol_kills.v2.market.event_probability_v2 begins an import cycle.
- **#218** `lol_kills/v2/market/phase_two_opening_v1.py:243` — source evidence `243: from .phase_two_collection_readiness_registry_v1 import ( / 244: EXTERNAL_SHA256_ENV as READINESS_EXTERNAL_SHA256_ENV,` — finding: Import of module lol_kills.v2.market.phase_two_collection_readiness_registry_v1 begins an import cycle.
- **#219** `lol_kills/v2/market/phase_two_outcome_opening_v1.py:172` — source evidence `172: from .phase_two_evaluation_readiness_registry_v1 import ( / 173: EXTERNAL_SHA256_ENV as READINESS_EXTERNAL_SHA256_ENV,` — finding: Import of module lol_kills.v2.market.phase_two_evaluation_readiness_registry_v1 begins an import cycle.
- **#220** `lol_kills/v2/market/phase_two_quote_attempt_v1.py:13` — source evidence `13: from . import betano_br_quote_adapter_v2 as quote_v2 / 14: from . import phase_one_evaluation_v1 as evaluation` — finding: Import of module lol_kills.v2.market.betano_br_quote_adapter_v2 begins an import cycle.
- **#221** `lol_kills/v2/market/phase_two_quote_attempt_v1.py:15` — source evidence `15: from . import phase_two_event_plan_v1 as event_plan / 16:` — finding: Import of module lol_kills.v2.market.phase_two_event_plan_v1 begins an import cycle.
- **#222** `lol_kills/v2/market/phase_two_stopping_snapshot_registry_v1.py:13` — source evidence `13: from . import phase_two_stopping_snapshot_v1 as snapshot / 14:` — finding: Import of module lol_kills.v2.market.phase_two_stopping_snapshot_v1 begins an import cycle.
- **#223** `lol_kills/v2/market/phase_two_stopping_snapshot_v1.py:14` — source evidence `14: from . import betano_br_quote_adapter_v2 as quote_v2 / 15: from . import match_winner_future_protocol_v1 as protocol_source` — finding: Import of module lol_kills.v2.market.betano_br_quote_adapter_v2 begins an import cycle.
- **#224** `lol_kills/v2/market/phase_two_stopping_snapshot_v1.py:17` — source evidence `17: from . import phase_two_attempt_completion_v1 as completion / 18: from .match_winner_future_protocol_registry_v1 import (` — finding: Import of module lol_kills.v2.market.phase_two_attempt_completion_v1 begins an import cycle.
- **#226** `lol_kills/v2/draft/terminal/promotion.py:126` — source evidence `126: from .semantic_draft_authority_v1 import ( / 127: load_active_semantic_draft_authority_v1,` — finding: Import of module lol_kills.v2.draft.terminal.semantic_draft_authority_v1 begins an import cycle. Import of module terminal.semantic_draft_authority_v1 begins an import cycle.
- **#227** `lol_kills/v2/draft/terminal/promotion.py:236` — source evidence `236: from .semantic_draft_authority_v1 import ( / 237: load_active_semantic_draft_authority_v1,` — finding: Import of module lol_kills.v2.draft.terminal.semantic_draft_authority_v1 begins an import cycle. Import of module terminal.semantic_draft_authority_v1 begins an import cycle.
- **#228** `lol_kills/v2/draft/terminal/semantic_draft_authority_v1.py:19` — source evidence `19: from lol_kills.v2.market import phase_one_evaluation_registry_v1 as registry / 20: from lol_kills.v2.market import phase_one_evaluation_v1 as evaluation` — finding: Import of module lol_kills.v2.market.phase_one_evaluation_registry_v1 begins an import cycle.
- **#229** `lol_kills/v2/draft/terminal/semantic_draft_authority_v1.py:20` — source evidence `20: from lol_kills.v2.market import phase_one_evaluation_v1 as evaluation / 21:` — finding: Import of module lol_kills.v2.market.phase_one_evaluation_v1 begins an import cycle.
- **#230** `lol_kills/v2/draft/terminal/semantic_draft_authority_v1.py:22` — source evidence `22: from . import future_prediction_ledger as draft_ledger / 23: from .model import TerminalDraftError, TerminalModel` — finding: Import of module terminal.future_prediction_ledger begins an import cycle. Import of module lol_kills.v2.draft.terminal.future_prediction_ledger begins an import cycle.
- **#231** `lol_kills/v2/draft/terminal/semantic_draft_authority_v1.py:23` — source evidence `23: from .model import TerminalDraftError, TerminalModel / 24:` — finding: Import of module lol_kills.v2.draft.terminal.model begins an import cycle. Import of module terminal.model begins an import cycle.
- **#232** `lol_kills/knowledge/semantic_engine.py:36` — source evidence `36: from .lol_oracle import LeagueOracleEngine / 37: from .mechanics_engine import Combatant, Damage, Event, GameState, MechanicsEngine` — finding: Import of module lol_kills.knowledge.lol_oracle begins an import cycle.
- **#463** `lol_kills/v2/tierlists/pooled_candidate.py:50` — source evidence `50: from .champion_elo import ( / 51: ATOM_BRIDGE_LOCATORS,` — finding: Import of module lol_kills.v2.tierlists.champion_elo begins an import cycle.
- **#472** `lol_kills/v2/market/match_winner_future_protocol_registry_v1.py:11` — source evidence `11: from .match_winner_future_protocol_v1 import ( / 12: DEFAULT_OUTPUT,` — finding: Import of module lol_kills.v2.market.match_winner_future_protocol_v1 begins an import cycle.
- **#473** `lol_kills/v2/market/phase_one_evaluation_readiness_registry_v1.py:10` — source evidence `10: from .phase_one_evaluation_readiness_v1 import ( / 11: DEFAULT_OUTPUT,` — finding: Import of module lol_kills.v2.market.phase_one_evaluation_readiness_v1 begins an import cycle.

### `py/empty-except`: Broad empty exception in model calibration fallback (2)

- Classification: **confirmed cleanup**
- Release impact: Moderate runtime impact. A failed calibration transform or Elo prior load can silently change the output path.
- Exact closure action: Replace the broad catch with the expected exception types and record a blocker or diagnostic when the fallback runs. Run model training and prediction tests.

- **#454** `lol_kills/ml/train.py:496` — source evidence `496: except Exception: / 497: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#455** `lol_kills/ml/train.py:532` — source evidence `532: except Exception: / 533: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.

### `py/empty-except`: Empty exception body used for fallback or best-effort cleanup (28)

- Classification: **intentional/test-only**
- Release impact: Low to moderate runtime impact. Most rows are deliberate cleanup, parse fallback, or optional-artifact behavior. Broad catches can hide a degraded result.
- Exact closure action: Keep the behavior only after naming the reason in a nearby comment. Narrow the exception where possible. For fallback paths, record the blocker or fallback state. Run the focused module tests.

- **#422** `lol_kills/v2/evaluation/benchmark_contract.py:6300` — source evidence `6300: except FileNotFoundError: / 6301: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#423** `lol_kills/export/blob_retention.py:582` — source evidence `582: except LeaseError: / 583: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#424** `lol_kills/features/build.py:175` — source evidence `175: except (TypeError, ValueError): / 176: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#425** `lol_kills/research/composition_signal.py:478` — source evidence `478: except (TypeError, ValueError): / 479: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#426** `lol_kills/research/composition_signal.py:1425` — source evidence `1425: except (OSError, ValueError, KeyError, TypeError, CompositionSignalError): / 1426: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#427** `lol_kills/draft_dynamics.py:227` — source evidence `227: except Exception: / 228: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#428** `lol_kills/draft_tierlist.py:156` — source evidence `156: except Exception: / 157: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#429** `lol_kills/draft_tierlist.py:174` — source evidence `174: except Exception: / 175: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#431** `lol_kills/research/elemental_drakes.py:1997` — source evidence `1997: except (OSError, json.JSONDecodeError): / 1998: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#432** `lol_kills/research/elemental_drakes.py:2108` — source evidence `2108: except (OSError, json.JSONDecodeError): / 2109: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#433** `lol_kills/research/elemental_drakes.py:2128` — source evidence `2128: except (OSError, json.JSONDecodeError): / 2129: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#435** `lol_kills/v2/evaluation/generate_checkpoint_c1_artifacts.py:100` — source evidence `100: except FileNotFoundError: / 101: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#436** `lol_kills/etl/grid_patch_receipts.py:85` — source evidence `85: except (TypeError, ValueError): / 86: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#437** `lol_kills/knowledge/league_wiki_vault.py:151` — source evidence `151: except ValueError: / 152: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#441** `lol_kills/postgame_sync.py:150` — source evidence `150: except (OSError, ValueError, RuntimeError): / 151: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#442** `lol_kills/postgame_sync.py:159` — source evidence `159: except (OSError, ValueError, RuntimeError, TypeError, KeyError): / 160: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#443** `lol_kills/export/public_pack.py:676` — source evidence `676: except (TypeError, ValueError): / 677: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#444** `lol_kills/public_refresh.py:1349` — source evidence `1349: except Exception: / 1350: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#445** `lol_kills/public_refresh.py:1457` — source evidence `1457: except Exception: / 1458: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#446** `lol_kills/v2/provenance/publication.py:735` — source evidence `735: except ValueError: / 736: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#447** `lol_kills/v2/data/real_spine.py:246` — source evidence `246: except OSError: / 247: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#448** `lol_kills/etl/roster_receipts.py:243` — source evidence `243: except (OSError, json.JSONDecodeError): / 244: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#449** `lol_kills/v2/draft/interactions/g5_exploratory/runner.py:333` — source evidence `333: except FileNotFoundError: / 334: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#450** `lol_kills/v2/draft/interactions/g5_exploratory/runner.py:340` — source evidence `340: except FileNotFoundError: / 341: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#451** `lol_kills/v2/draft/interactions/g5_exploratory/runner.py:432` — source evidence `432: except Exception: / 433: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#452** `lol_kills/v2/draft/interactions/g5_exploratory/runner.py:554` — source evidence `554: except FileNotFoundError: / 555: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#456** `lol_kills/v2/draft/interactions/g5_exploratory/v2_runner.py:916` — source evidence `916: except FileNotFoundError: / 917: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.
- **#470** `lol_kills/export/pack_records.py:181` — source evidence `181: except (TypeError, ValueError): / 182: pass` — finding: 'except' clause does nothing but pass and there is no explanatory comment.

### `py/import-and-import-from`: Test imports one module namespace and named symbols from the same module (12)

- Classification: **intentional/test-only**
- Release impact: Test-only impact. The module alias supports namespace or monkeypatch checks while named imports support direct assertions.
- Exact closure action: Keep both forms when both are used. Otherwise consolidate the imports. If both are required, add a local rationale and retain the focused test coverage.

- **#233** `tests/test_blob_retention.py:9` — source evidence `9: import lol_kills.export.blob_retention as retention / 10: from lol_kills.export.blob_retention import (` — finding: Module 'lol_kills.export.blob_retention' is imported with both 'import' and 'import from'.
- **#234** `tests/model_v2/champions/test_champion_id_crosswalk.py:11` — source evidence `11: import lol_kills.v2.champions.id_crosswalk as id_crosswalk_module / 12: from lol_kills.v2.champions.id_crosswalk import (` — finding: Module 'lol_kills.v2.champions.id_crosswalk' is imported with both 'import' and 'import from'.
- **#235** `tests/model_v2/evaluation/test_checkpoint_c1.py:15` — source evidence `15: import lol_kills.v2.evaluation.checkpoint_c1 as c1 / 16: from lol_kills.v2.evaluation.checkpoint_c1 import (` — finding: Module 'lol_kills.v2.evaluation.checkpoint_c1' is imported with both 'import' and 'import from'.
- **#236** `tests/model_v2/draft/interactions/test_draft_interactions_l6.py:11` — source evidence `11: import lol_kills.v2.draft.interactions.artifacts as artifacts_module / 12: from lol_kills.v2.draft.interactions.artifacts import (` — finding: Module 'lol_kills.v2.draft.interactions.artifacts' is imported with both 'import' and 'import from'. Module 'interactions.artifacts' is imported with both 'import' and 'import from'.
- **#237** `tests/test_grid_market_cohort.py:9` — source evidence `9: import lol_kills.grid_market_cohort as cohort / 10: from lol_kills.grid_market_cohort import (` — finding: Module 'lol_kills.grid_market_cohort' is imported with both 'import' and 'import from'.
- **#238** `tests/test_live_grid.py:13` — source evidence `13: import lol_kills.etl.grid_ingest as grid_ingest / 14: from lol_kills.etl.grid_ingest import _download` — finding: Module 'lol_kills.etl.grid_ingest' is imported with both 'import' and 'import from'.
- **#239** `tests/model_v2/ratings/player/test_multileague_runner.py:9` — source evidence `9: import lol_kills.v2.ratings.player.multileague_runner as runner / 10: from lol_kills.v2.ratings.player.multileague_development import (` — finding: Module 'lol_kills.v2.ratings.player.multileague_runner' is imported with both 'import' and 'import from'. Module 'player.multileague_runner' is imported with both 'import' and 'import from'.
- **#240** `tests/test_postgame_sync.py:12` — source evidence `12: import lol_kills.postgame_sync as postgame_sync / 13: from lol_kills.export import pack_spec` — finding: Module 'lol_kills.postgame_sync' is imported with both 'import' and 'import from'.
- **#241** `tests/model_v2/ratings/player/test_private_development_runner.py:14` — source evidence `14: import lol_kills.v2.ratings.player.private_development_runner as runner / 15: from lol_kills.v2.ratings.player.model import DISPLAY_LOGIT_SCALE` — finding: Module 'lol_kills.v2.ratings.player.private_development_runner' is imported with both 'import' and 'import from'. Module 'player.private_development_runner' is imported with both 'import' and 'import from'.
- **#242** `tests/model_v2/data/test_publication_b2.py:11` — source evidence `11: import lol_kills.v2.provenance.allowlist as allowlist_module / 12: import lol_kills.v2.provenance.publication as publication_module` — finding: Module 'lol_kills.v2.provenance.allowlist' is imported with both 'import' and 'import from'.
- **#243** `tests/model_v2/data/test_publication_b2.py:12` — source evidence `12: import lol_kills.v2.provenance.publication as publication_module / 13: from lol_kills.v2.data.common import (` — finding: Module 'lol_kills.v2.provenance.publication' is imported with both 'import' and 'import from'.
- **#247** `tests/model_v2/draft/interactions/test_representation_rank_assay.py:11` — source evidence `11: import lol_kills.v2.draft.interactions.representation_rank_assay as assay / 12:` — finding: Module 'lol_kills.v2.draft.interactions.representation_rank_assay' is imported with both 'import' and 'import from'. Module 'interactions.representation_rank_assay' is imported with both 'import' and 'import from'.

### `py/imprecise-assert`: Equality expressed through assertTrue or assertFalse (2)

- Classification: **intentional/test-only**
- Release impact: Test-only impact. Failure output is less useful and does not change production behavior.
- Exact closure action: Use assertEqual or assertNotEqual with the existing operands. Run the two focused tests.

- **#419** `tests/model_v2/champions/test_champion_ontology.py:950` — source evidence `950: self.assertFalse(result["coverage"]["covered_champions"] == []) / 951: self.assertIn("input_hash", result)` — finding: assertFalse(a == b) cannot provide an informative message. Using assertNotEqual(a, b) instead will give more informative messages.
- **#420** `tests/test_competition_and_ratings.py:687` — source evidence `687: self.assertTrue(set(snapshot["model"]) == {"hierarchical_bt"}) / 688:` — finding: assertTrue(a == b) cannot provide an informative message. Using assertEqual(a, b) instead will give more informative messages.

### `py/ineffectual-statement`: Ellipsis body in a Protocol method (12)

- Classification: **deferred source-contract review**
- Release impact: Low direct runtime impact. The ellipsis is an interface declaration, so changing it can alter stub and runtime semantics.
- Exact closure action: Confirm the protocol contract. Keep the ellipsis with an explanatory suppression if the interface requires it, or replace it with the project-approved abstract-method form. Run type and import checks.

- **#403** `lol_kills/v2/market/betano_br_quote_adapter_v1.py:135` — source evidence `135: def __call__(self, request_url: str) -> PublicDocumentResponse: ... / 136:` — finding: This statement has no effect.
- **#404** `lol_kills/export/blob_retention.py:332` — source evidence `332: ) -> dict[str, object]: ... / 333:` — finding: This statement has no effect.
- **#405** `lol_kills/export/blob_retention.py:336` — source evidence `336: ) -> tuple[bytes, BlobIdentity] \| None: ... / 337:` — finding: This statement has no effect.
- **#406** `lol_kills/export/blob_retention.py:340` — source evidence `340: ) -> BlobIdentity \| None: ... / 341:` — finding: This statement has no effect.
- **#407** `lol_kills/export/blob_retention.py:350` — source evidence `350: ) -> BlobIdentity \| None: ... / 351:` — finding: This statement has no effect.
- **#408** `lol_kills/export/blob_retention.py:359` — source evidence `359: ) -> BlobIdentity \| None: ... / 360:` — finding: This statement has no effect.
- **#409** `tools/retire_public_blob_store.py:38` — source evidence `38: ) -> dict[str, object]: ... / 39:` — finding: This statement has no effect.
- **#410** `tools/retire_public_blob_store.py:47` — source evidence `47: ) -> BlobIdentity \| None: ... / 48:` — finding: This statement has no effect.
- **#411** `lol_kills/v2/evaluation/types.py:404` — source evidence `404: ... / 405:` — finding: This statement has no effect.
- **#412** `lol_kills/v2/evaluation/types.py:419` — source evidence `419: ... / 420:` — finding: This statement has no effect.
- **#413** `lol_kills/v2/evaluation/types.py:422` — source evidence `422: ... / 423:` — finding: This statement has no effect.
- **#414** `lol_kills/v2/evaluation/types.py:432` — source evidence `432: ...` — finding: This statement has no effect.

### `py/unexpected-raise-in-special-method`: Test sentinel raises AssertionError from `__getitem__` (1)

- Classification: **intentional/test-only**
- Release impact: Test-only impact. The sentinel proves that production code does not read a forbidden key.
- Exact closure action: Keep the assertion sentinel and add a local rationale, or use a test helper outside the mapping protocol. Preserve the negative-path test.

- **#138** `tests/model_v2/data/test_g1_draft_features.py:85` — source evidence `85: def __getitem__(self, key): / 86: if key == "target":` — finding: This method raises AssertionError - should raise a LookupError (KeyError or IndexError) instead.

### `py/unnecessary-lambda`: Lambda wraps a callable without changing its contract (5)

- Classification: **confirmed cleanup**
- Release impact: Low runtime impact. The wrapper adds noise and a small call layer.
- Exact closure action: Pass the callable directly when its signature matches. Retain a wrapper only when it adapts arguments, and document that adaptation. Run the focused tests.

- **#142** `lol_kills/v2/tierlists/artifact.py:640` — source evidence `640: object_pairs_hook=lambda pairs: _reject_duplicates(pairs), / 641: parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),` — finding: This 'lambda' is just a simple wrapper around a callable object. Use that object directly.
- **#143** `tools/live_fair_odds/model.py:197` — source evidence `197: for path in sorted(paths, key=lambda item: str(item)) / 198: }` — finding: This 'lambda' is just a simple wrapper around a callable object. Use that object directly.
- **#144** `lol_kills/v2/draft/interactions/oe_target_evidence.py:156` — source evidence `156: return canonical_sha256(sorted((dict(row) for row in rows), key=lambda x: canonical_bytes(x))) / 157:` — finding: This 'lambda' is just a simple wrapper around a callable object. Use that object directly.
- **#145** `tests/test_composition_signal.py:608` — source evidence `608: result = _production_style_recalibrate(games[:6], games[6:], None, lambda: LogisticRegression(), raw) / 609: assert np.allclose(result, np.clip(raw, 1e-5, 1 - 1e-5))` — finding: This 'lambda' is just a simple wrapper around a callable object. Use that object directly.
- **#146** `tests/model_v2/evaluation/test_r20_selection.py:289` — source evidence `289: lambda volume: np.ones_like(volume), / 290: ],` — finding: This 'lambda' is just a simple wrapper around a callable object. Use that object directly.

### `py/unused-global-variable`: Module constant or cache assignment has no local consumer (7)

- Classification: **confirmed cleanup**
- Release impact: Low to moderate impact. Dead constants obscure the source contract and can leave stale mechanics or paths in the module.
- Exact closure action: Remove the assignment after checking import-time side effects and external consumers. Preserve any right-hand-side validation or registration call. Run import and focused tests.

- **#102** `lol_kills/v2/evaluation/checkpoint_c1.py:126` — source evidence `126: _SPEC_BY_ROLE = {spec.role: spec for spec in _SPECS} / 127: _ARTIFACT_IDENTITIES = {` — finding: The global variable '_SPEC_BY_ROLE' is not used.
- **#106** `lol_kills/v2/draft/terminal/development_snapshot.py:35` — source evidence `35: _SHA256_RE = re.compile(r"^[0-9a-f]{64}$") / 36: _ROW_KEYS = {` — finding: The global variable '_SHA256_RE' is not used.
- **#109** `lol_kills/v2/tierlists/model.py:105` — source evidence `105: _FORBIDDEN_RAW_WIN_RATE_KEYS = frozenset(("win_rate", "raw_win_rate", "wr", "wins", "losses")) / 106:` — finding: The global variable '_FORBIDDEN_RAW_WIN_RATE_KEYS' is not used.
- **#110** `lol_kills/v2/ratings/player/model.py:118` — source evidence `118: _GH_NODES = ( / 119: -5.387480890011233,` — finding: The global variable '_GH_NODES' is not used.
- **#111** `lol_kills/v2/ratings/player/model.py:140` — source evidence `140: _GH_WEIGHTS = ( / 141: 2.229393645534151e-13,` — finding: The global variable '_GH_WEIGHTS' is not used.
- **#115** `lol_kills/v2/market/phase_two_evaluation_v1.py:39` — source evidence `39: CONFIDENCE_INTERVAL = (0.025, 0.975) / 40: ECE_BINS = 10` — finding: The global variable 'CONFIDENCE_INTERVAL' is not used.
- **#459** `lol_kills/research/draft_phase_curve.py:590` — source evidence `590: _BRIDGE_CACHE = result / 591: return result` — finding: The global variable '_BRIDGE_CACHE' is not used.

### `py/unused-global-variable`: Module-level path, schema, registry, or public constant lacks local use (9)

- Classification: **deferred source-contract review**
- Release impact: Moderate to high impact. These names can bind artifacts, receipts, registries, and cross-module contracts.
- Exact closure action: Audit every importer and generated receipt. Then export the intentional contract explicitly or remove the dead binding. Run provenance, registry, and artifact tests.

- **#105** `lol_kills/v2/evaluation/contract_reconciliation_review_v1.py:31` — source evidence `31: SOURCE_LOCATOR = "lol_kills/v2/evaluation/contract_reconciliation_review_v1.py" / 32: SCHEMA_VERSION = "scryglass:contract-validation-reconciliation-review:v1"` — finding: The global variable 'SOURCE_LOCATOR' is not used.
- **#107** `lol_kills/v2/market/event_probability_v1.py:53` — source evidence `53: DEFAULT_REGISTRY = Path( / 54: "data/lol/v2/evaluation/match-winner-market-v1/event-probability-registry.json"` — finding: The global variable 'DEFAULT_REGISTRY' is not used.
- **#108** `lol_kills/v2/market/full_pipeline_uncertainty_v1.py:49` — source evidence `49: OUTPUT_PREFIX = PurePosixPath( / 50: "data/lol/v2/evaluation/match-winner-market-v1/event-uncertainty"` — finding: The global variable 'OUTPUT_PREFIX' is not used.
- **#112** `lol_kills/v2/ratings/player/multileague_v3_temporal_failure_v2.py:18` — source evidence `18: SOURCE_LOCATOR = ( / 19: "lol_kills/v2/ratings/player/multileague_v3_temporal_failure_v2.py"` — finding: The global variable 'SOURCE_LOCATOR' is not used.
- **#113** `lol_kills/v2/evaluation/outer_calibration.py:2331` — source evidence `2331: _AUTHORITY_PRIVATE_CHECKER = _assert_runtime_integrity` — finding: The global variable '_AUTHORITY_PRIVATE_CHECKER' is not used.
- **#114** `lol_kills/v2/market/phase_one_evaluation_v1.py:51` — source evidence `51: OUTPUT_PREFIX = PurePosixPath( / 52: "data/lol/v2/evaluation/match-winner-market-v1/phase-one/evaluations"` — finding: The global variable 'OUTPUT_PREFIX' is not used.
- **#117** `lol_kills/v2/ratings/player/post_validation_refit_v1.py:49` — source evidence `49: SOURCE_LOCATOR = "lol_kills/v2/ratings/player/post_validation_refit_v1.py" / 50: SOURCE_SNAPSHOT_SCHEMA_VERSION = (` — finding: The global variable 'SOURCE_LOCATOR' is not used.
- **#118** `lol_kills/v2/ratings/player/post_validation_refit_v1.py:61` — source evidence `61: ROLES = ("top", "jungle", "mid", "bot", "support") / 62: SIDES = ("blue", "red")` — finding: The global variable 'ROLES' is not used.
- **#119** `lol_kills/v2/champions/atoms/schema.py:43` — source evidence `43: DIMENSIONS: tuple[str, ...] = tuple(DIMENSION_LABELS) / 44:` — finding: The global variable 'DIMENSIONS' is not used.

### `py/unused-global-variable`: Module-level cache or authority hook is consumed through global state (2)

- Classification: **generated/false positive**
- Release impact: Potentially high impact if treated as dead. These names participate in cache lookup or runtime integrity behavior.
- Exact closure action: Verify the global reads and the integrity hook. Keep the implementation, or expose the intended public name in `__all__` after the contract review. Do not dismiss the alert before recording the evidence.

- **#103** `lol_kills/research/composition_signal.py:149` — source evidence `149: _ATOM_CORPUS = payload / 150: return payload` — finding: The global variable '_ATOM_CORPUS' is not used.
- **#104** `lol_kills/research/composition_signal.py:277` — source evidence `277: _DEPTH4_KEYS = ( / 278: "d4_dmg_cd", "d4_burst_cd", "d4_cd_uptime", "d4_cd_x_uptime",` — finding: The global variable '_DEPTH4_KEYS' is not used.

### `py/unused-import`: Imported binding is not read (135)

- Classification: **confirmed cleanup**
- Release impact: Low direct runtime impact. Test rows affect test collection only. Production rows can add import cost or hide an incomplete refactor.
- Exact closure action: Remove the named binding. Preserve an import only when a verified registration side effect requires it. Run the focused test or module import smoke check.

- **#248** `lol_kills/v2/provenance/allowlist.py:9` — source evidence `9: from typing import Any, Mapping, Sequence / 10:` — finding: Import of 'Sequence' is not used.
- **#249** `lol_kills/v2/tierlists/artifact.py:23` — source evidence `23: from typing import Any, Iterable, Mapping, Sequence / 24:` — finding: Import of 'Iterable' is not used.
- **#250** `lol_kills/v2/tierlists/artifact.py:29` — source evidence `29: from .appearances import AppearanceScope, AppearanceTable, CellAppearances / 30: from .model import (` — finding: Import of 'AppearanceTable' is not used.
- **#251** `lol_kills/v2/tierlists/artifact.py:30` — source evidence `30: from .model import ( / 31: APPEARANCE_SOURCE,` — finding: Import of 'SOURCE_TREE_ALLOWLIST' is not used. Import of 'load_crosswalk_vocabulary' is not used.
- **#252** `lol_kills/v2/evaluation/b2_pipeline.py:5` — source evidence `5: from dataclasses import asdict / 6: from pathlib import Path` — finding: Import of 'asdict' is not used.
- **#253** `lol_kills/v2/market/betano_terms_authority_v1.py:15` — source evidence `15: from typing import Any, Mapping, Sequence / 16: from urllib.parse import urlparse` — finding: Import of 'Sequence' is not used.
- **#254** `lol_kills/v2/champions/atoms/bridge_v1.py:13` — source evidence `13: import math / 14: from datetime import datetime, timezone` — finding: Import of 'math' is not used.
- **#255** `lol_kills/research/champ_impact.py:26` — source evidence `26: from pathlib import Path / 27:` — finding: Import of 'Path' is not used.
- **#256** `lol_kills/research/champ_oe_lenses.py:33` — source evidence `33: from pathlib import Path / 34:` — finding: Import of 'Path' is not used.
- **#257** `lol_kills/research/champ_tierlist_blade_chest.py:41` — source evidence `41: from pathlib import Path / 42:` — finding: Import of 'Path' is not used.
- **#259** `tools/codex_import.py:25` — source evidence `25: import hashlib / 26: import json` — finding: Import of 'hashlib' is not used.
- **#260** `lol_kills/v2/draft/interactions/real_v1_g4/contract.py:12` — source evidence `12: from datetime import datetime / 13: import hashlib` — finding: Import of 'datetime' is not used.
- **#261** `lol_kills/v2/evaluation/contract_validation.py:7` — source evidence `7: from datetime import datetime, timezone / 8: import hashlib` — finding: Import of 'timezone' is not used.
- **#262** `lol_kills/v2/draft/interactions/real_v1_g4/coverage_preflight.py:14` — source evidence `14: from pathlib import Path / 15: from typing import Any, Mapping` — finding: Import of 'Path' is not used.
- **#263** `lol_kills/v2/draft/terminal/development_evaluation_v2.py:33` — source evidence `33: from .development_evaluation import ( / 34: BASELINE_CONFIG,` — finding: Import of '_brier' is not used. Import of '_log_loss' is not used. Import of '_parse_time' is not used.
- **#264** `lol_kills/v2/draft/terminal/development_evaluation_v2.py:51` — source evidence `51: from .development_snapshot import ( / 52: DEFAULT_MANIFEST,` — finding: Import of 'DEFAULT_MANIFEST' is not used.
- **#265** `lol_kills/v2/draft/terminal/development_evaluation_v4_atoms.py:27` — source evidence `27: import math / 28: from datetime import datetime, timezone` — finding: Import of 'math' is not used.
- **#266** `lol_kills/v2/draft/terminal/development_evaluation_v4_atoms.py:37` — source evidence `37: from lol_kills.v2.draft.terminal.development_evaluation import ( / 38: CALIBRATION_ORDER,` — finding: Import of '_cluster_metrics' is not used. Import of '_league_metrics' is not used.
- **#267** `lol_kills/v2/draft/terminal/development_evaluation_v4_atoms.py:49` — source evidence `49: from lol_kills.v2.draft.terminal.development_evaluation_v2 import ( / 50: RIDGE_STRENGTH_ORDER,` — finding: Import of 'baseline_adjusted_logits' is not used. Import of 'composition_logits' is not used.
- **#268** `lol_kills/research/draft_advantage_matrix.py:19` — source evidence `19: from pathlib import Path / 20:` — finding: Import of 'Path' is not used.
- **#269** `lol_kills/research/draft_advantage_matrix.py:24` — source evidence `24: from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score / 25: from sklearn.model_selection import TimeSeriesSplit` — finding: Import of 'log_loss' is not used.
- **#270** `lol_kills/research/draft_archetype_edges.py:16` — source evidence `16: from pathlib import Path / 17:` — finding: Import of 'Path' is not used.
- **#271** `lol_kills/draft_dynamics.py:26` — source evidence `26: from pathlib import Path / 27: from typing import Any` — finding: Import of 'Path' is not used.
- **#272** `lol_kills/draft_dynamics.py:29` — source evidence `29: from lol_kills.draft_archetypes import champ_tags, draft_archetype_features, side_archetype_counts / 30: from lol_kills.draft_phase_score import (` — finding: Import of 'champ_tags' is not used.
- **#273** `lol_kills/research/draft_phase_beatdown.py:19` — source evidence `19: from pathlib import Path / 20:` — finding: Import of 'Path' is not used.
- **#274** `lol_kills/draft_recommendation.py:35` — source evidence `35: from lol_kills.export.pack_records import build_player_records, build_team_records / 36: from lol_kills.draft_archetypes import ARCHETYPE_NAMES, champ_tags` — finding: Import of 'build_player_records' is not used. Import of 'build_team_records' is not used.
- **#275** `lol_kills/research/draft_score_autoresearch.py:23` — source evidence `23: from typing import Any, Callable, Iterable, Mapping / 24:` — finding: Import of 'Iterable' is not used. Import of 'Callable' is not used.
- **#276** `lol_kills/draft_tierlist.py:20` — source evidence `20: from pathlib import Path / 21:` — finding: Import of 'Path' is not used.
- **#280** `lol_kills/v2/market/fast_event_uncertainty_v1.py:24` — source evidence `24: from lol_kills.v2.draft.terminal import future_prediction_ledger as draft_ledger / 25: from lol_kills.v2.draft.terminal.development_evaluation import (` — finding: Import of 'draft_ledger' is not used.
- **#281** `lol_kills/v2/tierlists/filters.py:8` — source evidence `8: from typing import Any, Mapping, Sequence / 9:` — finding: Import of 'Sequence' is not used.
- **#282** `lol_kills/v2/evaluation/fixtures.py:9` — source evidence `9: from typing import Any, Mapping, Sequence / 10:` — finding: Import of 'Any' is not used.
- **#283** `lol_kills/v2/tierlists/forward_evaluation.py:14` — source evidence `14: from datetime import timezone / 15: import hashlib` — finding: Import of 'timezone' is not used.
- **#284** `lol_kills/research/furia_g2_grubs_brief_pdf.py:9` — source evidence `9: from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT / 10: from reportlab.lib.pagesizes import A4` — finding: Import of 'TA_LEFT' is not used.
- **#285** `lol_kills/research/furia_g2_grubs_brief_pdf.py:13` — source evidence `13: from reportlab.platypus import ( / 14: KeepTogether,` — finding: Import of 'KeepTogether' is not used.
- **#286** `reports/elemental-drakes-paper/generate_figures.py:7` — source evidence `7: import math / 8: import sys` — finding: Import of 'math' is not used.
- **#287** `lol_kills/grid_capability_catalog.py:22` — source evidence `22: from typing import Any, Iterable, Mapping, Sequence / 23:` — finding: Import of 'Iterable' is not used.
- **#288** `lol_kills/etl/grid_ingest.py:34` — source evidence `34: from typing import Any, Iterable, Iterator, Mapping, Sequence / 35:` — finding: Import of 'Iterable' is not used.
- **#289** `lol_kills/grid_market_cohort_intake.py:13` — source evidence `13: import time / 14: from collections import Counter` — finding: Import of 'time' is not used.
- **#290** `lol_kills/grid_market_evaluation.py:19` — source evidence `19: from typing import Any, Iterable, Mapping, Sequence / 20:` — finding: Import of 'Iterable' is not used.
- **#291** `lol_kills/grid_market_foundation.py:20` — source evidence `20: from typing import Any, Iterable, Iterator, Mapping, Sequence / 21:` — finding: Import of 'Iterable' is not used.
- **#292** `lol_kills/research/grubs_contest_study.py:39` — source evidence `39: from lol_kills.etl.riot_timelines import cache_path, load_cached, summarize_map_grubs / 40: from lol_kills.research.side_objective_edges import engineer` — finding: Import of 'cache_path' is not used.
- **#293** `lol_kills/research/grubs_decision_report.py:6` — source evidence `6: from pathlib import Path / 7:` — finding: Import of 'Path' is not used.
- **#296** `lol_kills/v2/champions/id_crosswalk.py:21` — source evidence `21: from typing import Any, Mapping / 22:` — finding: Import of 'Any' is not used.
- **#297** `lol_kills/v2/ratings/team/last_observed_real_v1.py:22` — source evidence `22: from typing import Any, Mapping, Sequence / 23:` — finding: Import of 'Sequence' is not used.
- **#298** `lol_kills/v2/champions/atoms/lcc_sources.py:19` — source evidence `19: from .schema import AtomBridgeError, require_object / 20:` — finding: Import of 'require_object' is not used.
- **#299** `lol_kills/knowledge/league_wiki_db.py:36` — source evidence `36: from collections.abc import Iterable, Iterator / 37: from datetime import datetime, timezone` — finding: Import of 'Iterable' is not used.
- **#300** `lol_kills/live_model.py:15` — source evidence `15: from typing import Any, Mapping, Sequence / 16:` — finding: Import of 'Sequence' is not used.
- **#302** `lol_kills/etl/manual_leaguepedia.py:23` — source evidence `23: import hashlib / 24: import json` — finding: Import of 'hashlib' is not used.
- **#303** `lol_kills/etl/manual_leaguepedia.py:35` — source evidence `35: from lol_kills.v2.data.common import ( / 36: ROLES,` — finding: Import of 'canonical_json_bytes' is not used.
- **#304** `lol_kills/research/mechanics_composite.py:14` — source evidence `14: from dataclasses import dataclass, field / 15: from datetime import datetime, timezone` — finding: Import of 'field' is not used.
- **#305** `lol_kills/research/mechanics_composite.py:15` — source evidence `15: from datetime import datetime, timezone / 16: from itertools import combinations` — finding: Import of 'timezone' is not used.
- **#306** `lol_kills/research/metric_correlation_matrix.py:16` — source evidence `16: from pathlib import Path / 17:` — finding: Import of 'Path' is not used.
- **#307** `lol_kills/v2/tierlists/model.py:25` — source evidence `25: from dataclasses import dataclass / 26: from datetime import datetime, timezone` — finding: Import of 'dataclass' is not used.
- **#308** `lol_kills/v2/tierlists/model.py:26` — source evidence `26: from datetime import datetime, timezone / 27: from pathlib import Path` — finding: Import of 'datetime' is not used. Import of 'timezone' is not used.
- **#309** `lol_kills/v2/tierlists/model.py:28` — source evidence `28: from typing import Any, Iterable, Mapping, Sequence / 29:` — finding: Import of 'Iterable' is not used.
- **#310** `lol_kills/v2/tierlists/model.py:30` — source evidence `30: from lol_kills.v2.data.common import ROLES, canonical_json_bytes, parse_rfc3339, sha256_bytes, to_rfc3339 / 31:` — finding: Import of 'ROLES' is not used. Import of 'parse_rfc3339' is not used. Import of 'sha256_bytes' is not used. Import of 'to_rfc3339' is not used.
- **#311** `lol_kills/v2/ratings/player/multileague_benchmark.py:27` — source evidence `27: import math / 28: import os` — finding: Import of 'math' is not used.
- **#312** `lol_kills/v2/ratings/player/multileague_benchmark.py:34` — source evidence `34: import numpy as np / 35:` — finding: Import of 'np' is not used.
- **#313** `lol_kills/v2/ratings/player/multileague_v2_runner.py:24` — source evidence `24: import numpy as np / 25:` — finding: Import of 'np' is not used.
- **#314** `lol_kills/v2/ratings/player/multileague_v2_runner_equal_series.py:8` — source evidence `8: import math / 9: from pathlib import Path` — finding: Import of 'math' is not used.
- **#315** `lol_kills/v2/ratings/player/multileague_v3_corrected_adaptive_diagnostic_v1.py:22` — source evidence `22: import math / 23: import os` — finding: Import of 'math' is not used.
- **#317** `lol_kills/v2/draft/terminal/participant_dependence_diagnostic_v1.py:14` — source evidence `14: import json / 15: import math` — finding: Import of 'json' is not used.
- **#318** `lol_kills/knowledge/patch_authority.py:20` — source evidence `20: from typing import Any, Iterable, Mapping / 21:` — finding: Import of 'Iterable' is not used.
- **#319** `lol_kills/v2/market/phase_one_opening_v1.py:13` — source evidence `13: from . import phase_one_collection_v1 as collection / 14: from . import phase_one_evaluation_v1 as evaluation` — finding: Import of 'collection' is not used.
- **#320** `lol_kills/v2/market/phase_two_collection_readiness_v1.py:21` — source evidence `21: from . import phase_one_evaluation_v1 as evaluation / 22: from . import phase_two_opening_v1 as opening` — finding: Import of 'evaluation' is not used.
- **#321** `lol_kills/v2/market/phase_two_stopping_snapshot_v1.py:8` — source evidence `8: import math / 9: import os` — finding: Import of 'math' is not used.
- **#323** `lol_kills/v2/tierlists/pooled_candidate.py:15` — source evidence `15: import json / 16: import math` — finding: Import of 'json' is not used.
- **#325** `lol_kills/v2/ratings/player/pre_side_rating_envelope_v1.py:24` — source evidence `24: import stat / 25: import tempfile` — finding: Import of 'stat' is not used.
- **#326** `lol_kills/knowledge/quick_mechanics_fastpack.py:25` — source evidence `25: import re / 26: import tempfile` — finding: Import of 're' is not used.
- **#327** `lol_kills/v2/evaluation/r20_foundation.py:12` — source evidence `12: import os / 13: from pathlib import Path` — finding: Import of 'os' is not used.
- **#328** `lol_kills/v2/evaluation/r20_foundation_algorithms.py:6` — source evidence `6: from dataclasses import dataclass / 7: import inspect` — finding: Import of 'dataclass' is not used.
- **#329** `lol_kills/v2/evaluation/r20_foundation_generator.py:8` — source evidence `8: from statistics import mean / 9: from typing import Any, Mapping, Sequence` — finding: Import of 'mean' is not used.
- **#330** `lol_kills/v2/evaluation/r20_foundation_generator.py:23` — source evidence `23: from .types import canonical_json, canonical_sha256 / 24:` — finding: Import of 'canonical_json' is not used.
- **#332** `lol_kills/v2/ratings/player/real_v1_adapter.py:16` — source evidence `16: import hashlib / 17: import json` — finding: Import of 'hashlib' is not used.
- **#333** `lol_kills/v2/ratings/team/real_v1_private_runner.py:21` — source evidence `21: from typing import Any, Mapping, Sequence / 22:` — finding: Import of 'Sequence' is not used.
- **#334** `lol_kills/etl/roster_receipts.py:22` — source evidence `22: from dataclasses import dataclass / 23: from datetime import datetime, timedelta, timezone` — finding: Import of 'dataclass' is not used.
- **#335** `lol_kills/etl/roster_receipts.py:26` — source evidence `26: from typing import Any, Iterable, Mapping / 27: from urllib.error import HTTPError, URLError` — finding: Import of 'Iterable' is not used.
- **#336** `benchmarks/lol-oracle-v1/run_benchmark.py:10` — source evidence `10: import statistics / 11: import sys` — finding: Import of 'statistics' is not used.
- **#337** `benchmarks/lol-oracle-v1/run_semantic_benchmark.py:16` — source evidence `16: import statistics / 17: import sys` — finding: Import of 'statistics' is not used.
- **#338** `lol_kills/v2/tierlists/schema.py:12` — source evidence `12: from lol_kills.v2.data.common import ROLES, canonicalize_role, parse_rfc3339 / 13:` — finding: Import of 'canonicalize_role' is not used.
- **#339** `lol_kills/v2/tierlists/scope_index.py:24` — source evidence `24: from .schema import COMPETITION_TIERS, INTERNATIONAL_SCOPES, REGIONS, TIERLIST_SCHEMA_ID / 25:` — finding: Import of 'COMPETITION_TIERS' is not used. Import of 'INTERNATIONAL_SCOPES' is not used.
- **#340** `lol_kills/v2/champions/atoms/seed_ontology_v1.py:21` — source evidence `21: from datetime import datetime, timezone / 22: from pathlib import Path` — finding: Import of 'datetime' is not used. Import of 'timezone' is not used.
- **#341** `lol_kills/research/side_objective_edges.py:22` — source evidence `22: from pathlib import Path / 23:` — finding: Import of 'Path' is not used.
- **#342** `lol_kills/research/side_objective_edges.py:26` — source evidence `26: from sklearn.linear_model import LogisticRegression / 27:` — finding: Import of 'LogisticRegression' is not used.
- **#343** `lol_kills/v2/data/source_tree.py:6` — source evidence `6: import os / 7: from pathlib import Path, PurePosixPath` — finding: Import of 'os' is not used.
- **#344** `lol_kills/research/temporal_draft_runtime.py:25` — source evidence `25: from datetime import datetime, timedelta, timezone / 26: from pathlib import Path` — finding: Import of 'datetime' is not used. Import of 'timezone' is not used. Import of 'timedelta' is not used.
- **#345** `lol_kills/research/temporal_draft_runtime.py:32` — source evidence `32: from sklearn.linear_model import SGDClassifier / 33:` — finding: Import of 'SGDClassifier' is not used.
- **#346** `tests/model_v2/tierlists/test_appearances.py:5` — source evidence `5: import json / 6: from pathlib import Path` — finding: Import of 'json' is not used.
- **#347** `tests/model_v2/tierlists/test_appearances.py:10` — source evidence `10: from lol_kills.v2.tierlists.appearances import ( / 11: AppearanceRow,` — finding: Import of 'AppearanceRow' is not used.
- **#348** `tests/model_v2/tierlists/test_artifact.py:6` — source evidence `6: import hashlib / 7: import json` — finding: Import of 'hashlib' is not used.
- **#349** `tests/model_v2/tierlists/test_artifact.py:7` — source evidence `7: import json / 8: from datetime import datetime, timezone` — finding: Import of 'json' is not used.
- **#350** `tests/model_v2/tierlists/test_artifact.py:8` — source evidence `8: from datetime import datetime, timezone / 9: from pathlib import Path` — finding: Import of 'datetime' is not used. Import of 'timezone' is not used.
- **#351** `tests/model_v2/tierlists/test_artifact.py:23` — source evidence `23: from lol_kills.v2.tierlists.appearances import AppearanceTable, league_scope, international_scope / 24: from lol_kills.v2.tierlists.model import (` — finding: Import of 'international_scope' is not used.
- **#352** `tests/model_v2/tierlists/test_artifact.py:24` — source evidence `24: from lol_kills.v2.tierlists.model import ( / 25: CLAIM_CEILING,` — finding: Import of 'CROSSWALK_ARTIFACT' is not used.
- **#353** `tests/model_v2/evaluation/test_b1_sealed_and_contracts.py:19` — source evidence `19: from lol_kills.v2.evaluation import ( / 20: AtomicSealedLedger,` — finding: Import of 'make_model_snapshot' is not used.
- **#354** `tests/model_v2/evaluation/test_b2_calibration.py:3` — source evidence `3: import math / 4:` — finding: Import of 'math' is not used.
- **#355** `tests/model_v2/evaluation/test_b2_coverage.py:3` — source evidence `3: from copy import deepcopy / 4:` — finding: Import of 'deepcopy' is not used.
- **#356** `tests/model_v2/evaluation/test_b2_pipeline_sealed.py:4` — source evidence `4: import hashlib / 5: import json` — finding: Import of 'hashlib' is not used.
- **#357** `tests/model_v2/evaluation/test_b2_pipeline_sealed.py:5` — source evidence `5: import json / 6: from pathlib import Path` — finding: Import of 'json' is not used.
- **#358** `tests/model_v2/evaluation/test_b2_pipeline_sealed.py:6` — source evidence `6: from pathlib import Path / 7:` — finding: Import of 'Path' is not used.
- **#359** `tests/model_v2/evaluation/test_b2_pipeline_sealed.py:24` — source evidence `24: from lol_kills.v2.evaluation.types import ArtifactRef / 25:` — finding: Import of 'ArtifactRef' is not used.
- **#360** `tests/model_v2/evaluation/test_b3_coverage.py:5` — source evidence `5: import importlib / 6: import json` — finding: Import of 'importlib' is not used.
- **#361** `tests/model_v2/market/test_betano_terms_authority_v1.py:3` — source evidence `3: from copy import deepcopy / 4: import hashlib` — finding: Import of 'deepcopy' is not used.
- **#362** `tests/model_v2/market/test_calibration_uncertainty_registry_v1.py:3` — source evidence `3: from copy import deepcopy / 4: import hashlib` — finding: Import of 'deepcopy' is not used.
- **#363** `tests/model_v2/data/test_core_contracts.py:5` — source evidence `5: import hashlib / 6: import pytest` — finding: Import of 'hashlib' is not used.
- **#364** `tests/model_v2/data/test_core_contracts.py:8` — source evidence `8: from lol_kills.v2.data.competitions import ( / 9: CompetitionTaxonomy,` — finding: Import of 'MapPatchRecord' is not used. Import of 'PatchConflictError' is not used.
- **#365** `tests/model_v2/draft/terminal/test_development_v3.py:3` — source evidence `3: from datetime import datetime, timezone / 4: import hashlib` — finding: Import of 'datetime' is not used.
- **#366** `tests/model_v2/ratings/team/test_estimands_v1.py:5` — source evidence `5: import json / 6: import math` — finding: Import of 'json' is not used.
- **#369** `tests/model_v2/market/test_event_probability_registry_v2.py:3` — source evidence `3: from copy import deepcopy / 4: import hashlib` — finding: Import of 'deepcopy' is not used.
- **#372** `tests/model_v2/draft/interactions/test_g5_execution_approval.py:4` — source evidence `4: import json / 5: import os` — finding: Import of 'json' is not used.
- **#373** `tests/test_grid_market_foundation.py:6` — source evidence `6: import pytest / 7:` — finding: Import of 'pytest' is not used.
- **#374** `tests/model_v2/draft/terminal/test_grid_promotion_gate.py:6` — source evidence `6: from datetime import datetime, timezone / 7:` — finding: Import of 'datetime' is not used. Import of 'timezone' is not used.
- **#375** `tests/test_leaderboards.py:3` — source evidence `3: import json / 4: from pathlib import Path` — finding: Import of 'json' is not used.
- **#376** `tests/test_leaderboards.py:4` — source evidence `4: from pathlib import Path / 5:` — finding: Import of 'Path' is not used.
- **#377** `tests/test_leaderboards.py:6` — source evidence `6: import pytest / 7:` — finding: Import of 'pytest' is not used.
- **#380** `tests/model_v2/market/test_phase_two_evaluation_v1.py:4` — source evidence `4: from datetime import datetime, timezone / 5: import hashlib` — finding: Import of 'datetime' is not used. Import of 'timezone' is not used.
- **#381** `tests/model_v2/ratings/player/test_player_rating.py:17` — source evidence `17: from lol_kills.v2.ratings.player.model import ( / 18: CLAIM_CEILING,` — finding: Import of 'ReplayResult' is not used.
- **#382** `tests/model_v2/ratings/player/test_private_development_runner.py:6` — source evidence `6: import importlib.util / 7: import json` — finding: Import of 'importlib' is not used.
- **#383** `tests/model_v2/data/test_publication_b2.py:31` — source evidence `31: from lol_kills.v2.provenance.publication import ( / 32: MODES,` — finding: Import of '_review_scope' is not used.
- **#384** `tests/model_v2/evaluation/test_r20_foundation.py:20` — source evidence `20: from lol_kills.v2.evaluation.r20_foundation import ( / 21: CANDIDATE_REGISTRY_LOCATOR,` — finding: Import of 'CANDIDATE_REGISTRY_LOCATOR' is not used.
- **#385** `tests/model_v2/evaluation/test_r20_foundation.py:33` — source evidence `33: from lol_kills.v2.evaluation.r20_foundation_generator import ( / 34: INITIAL_TRAIN_SERIES_PER_CELL,` — finding: Import of 'SOURCE_CONTEXT_PATTERNS' is not used.
- **#386** `tests/model_v2/evaluation/test_r20_selection.py:17` — source evidence `17: from lol_kills.v2.evaluation.r20_selection import ( / 18: AUTHORITY_LOCATOR,` — finding: Import of 'CONFIG_LOCATOR' is not used. Import of 'forecast_prefix_sha256' is not used.
- **#387** `tests/model_v2/data/test_real_spine.py:3` — source evidence `3: from copy import deepcopy / 4: from datetime import datetime, timedelta` — finding: Import of 'deepcopy' is not used.
- **#388** `tests/model_v2/draft/interactions/test_real_v1_g4_coverage_preflight.py:5` — source evidence `5: from pathlib import Path / 6:` — finding: Import of 'Path' is not used.
- **#389** `tests/model_v2/draft/interactions/test_representation_rank_private_coordinator.py:3` — source evidence `3: import json / 4: from pathlib import Path` — finding: Import of 'json' is not used.
- **#390** `tests/model_v2/ratings/player/test_side_neutral_protocol_review_v1.py:3` — source evidence `3: from copy import deepcopy / 4: from datetime import datetime, timezone` — finding: Import of 'deepcopy' is not used.
- **#391** `tests/model_v2/ratings/team/test_team_rating_publication.py:17` — source evidence `17: from lol_kills.v2.ratings.team import ( / 18: CLAIM_CEILING,` — finding: Import of 'CLAIM_CEILING' is not used.
- **#392** `lol_kills/total_kills_synthetic_prices.py:25` — source evidence `25: from lol_kills.grid_market_evaluation import ( / 26: CHECKPOINTS,` — finding: Import of 'EvaluationError' is not used.
- **#393** `lol_kills/ml/train.py:21` — source evidence `21: from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error / 22:` — finding: Import of 'mean_squared_error' is not used.
- **#394** `lol_kills/ml/train.py:25` — source evidence `25: from lol_kills.ml.eval import ( / 26: archive_models,` — finding: Import of 'crps_gaussian' is not used.
- **#395** `lol_kills/export/warehouse_snapshot.py:11` — source evidence `11: import io / 12: import json` — finding: Import of 'io' is not used.
- **#396** `lol_kills/export/warehouse_snapshot.py:15` — source evidence `15: from datetime import datetime, timezone / 16: from pathlib import Path` — finding: Import of 'datetime' is not used. Import of 'timezone' is not used.
- **#464** `lol_kills/v2/tierlists/pooled_candidate.py:50` — source evidence `50: from .champion_elo import ( / 51: ATOM_BRIDGE_LOCATORS,` — finding: Import of 'SOURCE_LOCATOR' is not used.
- **#474** `lol_kills/v2/tierlists/champion_elo.py:27` — source evidence `27: from scipy.special import expit, ndtr / 28:` — finding: Import of 'ndtr' is not used.
- **#475** `lol_kills/v2/tierlists/champion_elo.py:30` — source evidence `30: from lol_kills.v2.champions.atoms.consume import AtomBridge / 31:` — finding: Import of 'AtomBridge' is not used.

### `py/unused-local-variable`: Local result or temporary binding is not read (38)

- Classification: **confirmed cleanup**
- Release impact: Low direct runtime impact for pure expressions. A right-hand side can still perform validation or file work, so removal must preserve that call.
- Exact closure action: Remove the unused binding and keep the right-hand side as a standalone call when it has side effects. If the value should affect the result, add the missing consumer and a regression test.

- **#9** `lol_kills/v2/provenance/allowlist.py:395` — source evidence `395: raw_manifest = _resolve(lineage_manifest_locator).read_bytes() / 396: values = dict(source_id=output["source_id"], artifact_class=output["artifact_class"], locator=output["locator"], bytes_sha256=output["bytes_sha256"], object_sha256=output["object_sha256"], fields=tuple(sorted(output["fields"])), audience_mode=audience_mode, lineage_manifest_locator=lineage_manifest_locator, lineage_manifest_id=manifest["manifest_id"], lineage_manifest_sha256=sha256_canonical_object_hash(manifest))` — finding: Variable raw_manifest is not used.
- **#10** `lol_kills/v2/evaluation/b3_coverage.py:536` — source evidence `536: alpha_tail = (1.0 - float(config["nominal_interval"])) / 2.0 / 537: adapter_hash = _callable_fingerprint(_control_inference_adapter)` — finding: Variable alpha_tail is not used.
- **#12** `lol_kills/v2/evaluation/checks.py:558` — source evidence `558: roster_id = row.roster_id / 559: if row.roster_id and isinstance(roster_roles, Sequence) and len(roster_roles) > 0:` — finding: Variable roster_id is not used.
- **#13** `tools/codex_import.py:173` — source evidence `173: user_text = "\n\n".join(self.user_messages) / 174: assistant_text = "\n\n".join(self.assistant_parts)` — finding: Variable user_text is not used.
- **#14** `lol_kills/research/composition_signal.py:1988` — source evidence `1988: names = tuple(baseline.feature_names) / 1989: blocks = [` — finding: Variable names is not used.
- **#15** `lol_kills/v2/draft/terminal/development_artifact.py:62` — source evidence `62: train_logits = composition_logits(train_rows, fit) / 63: calibration_logits = composition_logits(calibration_rows, fit)` — finding: Variable train_logits is not used.
- **#16** `lol_kills/research/draft_advantage_matrix.py:498` — source evidence `498: old = live_win_prob( / 499: p_pre=p_pre,` — finding: Variable old is not used.
- **#17** `lol_kills/draft_recommendation.py:410` — source evidence `410: fold_validation_games = games[fold_train_end:fold_validation_end] / 411: fold_train_matrix = development_matrix[:fold_train_end]` — finding: Variable fold_validation_games is not used.
- **#21** `lol_kills/v2/provenance/g1_source_authority.py:191` — source evidence `191: metadata = os.lstat(current) / 192: except FileNotFoundError as error:` — finding: Variable metadata is not used.
- **#22** `benchmarks/lol-oracle-v1/generate_benchmark.py:455` — source evidence `455: items = sorted( / 456: [` — finding: Variable items is not used.
- **#31** `lol_kills/etl/join.py:153` — source evidence `153: bi = pd.to_numeric(m.get("blue_inhibitors"), errors="coerce") / 154: ri = pd.to_numeric(m.get("red_inhibitors"), errors="coerce")` — finding: Variable bi is not used.
- **#32** `lol_kills/etl/join.py:154` — source evidence `154: ri = pd.to_numeric(m.get("red_inhibitors"), errors="coerce") / 155: # OE may not have first-inhib; leave NaN unless we can infer asymmetry early — skip inference` — finding: Variable ri is not used.
- **#33** `lol_kills/export/leaderboards.py:316` — source evidence `316: wins = int(record.get("wins") or 0) / 317: wr = _number(record.get("wr"))` — finding: Variable wins is not used.
- **#34** `lol_kills/knowledge/lol_oracle.py:356` — source evidence `356: normalized = _norm(question) / 357: matches: list[tuple[int, Mapping[str, Any]]] = []` — finding: Variable normalized is not used.
- **#35** `lol_kills/market_decision.py:532` — source evidence `532: checked_quote = validate_quote_receipt( / 533: quote,` — finding: Variable checked_quote is not used.
- **#36** `lol_kills/market_decision.py:547` — source evidence `547: checked_quote = None / 548: blockers.append("market_quote_receipt_invalid")` — finding: Variable checked_quote is not used.
- **#37** `lol_kills/research/mechanics_composite.py:98` — source evidence `98: available = _timestamp(self.available_at, "available_at") / 99: if end is not None and end <= start:` — finding: Variable available is not used.
- **#38** `lol_kills/research/mechanics_engine_run.py:269` — source evidence `269: matrix = {} / 270: expected_hash = None` — finding: Variable matrix is not used.
- **#39** `lol_kills/v2/ratings/team/model.py:454` — source evidence `454: claim_ceiling = payload["claim_ceiling"] / 455: schema_conformance = payload["schema_conformance"]` — finding: Variable claim_ceiling is not used.
- **#40** `lol_kills/v2/ratings/player/model.py:1335` — source evidence `1335: ctx = context_for(match, player_id) / 1336: master_key = (role, player_id)` — finding: Variable ctx is not used.
- **#43** `lol_kills/v2/market/phase_one_collection_v1.py:317` — source evidence `317: ratings = validate_ratings_protocol(root=root) / 318: draft = validate_draft_protocol(root=root)` — finding: Variable ratings is not used.
- **#44** `lol_kills/v2/market/phase_one_collection_v1.py:318` — source evidence `318: draft = validate_draft_protocol(root=root) / 319: market = validate_market_protocol(root=root)` — finding: Variable draft is not used.
- **#45** `lol_kills/v2/tierlists/pooled_candidate.py:1012` — source evidence `1012: pair_designs = { / 1013: (focal, opponent): design[index]` — finding: Variable pair_designs is not used.
- **#52** `lol_kills/v2/evaluation/r20_selection.py:2680` — source evidence `2680: candidate_validation_call = validate_candidate_replays / 2681: predictive_validation_call = validate_predictive_rows` — finding: Variable candidate_validation_call is not used.
- **#53** `lol_kills/etl/riot_timelines.py:154` — source evidence `154: meta = timeline.get("metadata") or {} / 155: # Match-V5: participantId 1-5 blue, 6-10 red typically` — finding: Variable meta is not used.
- **#54** `lol_kills/etl/riot_timelines.py:306` — source evidence `306: ok = miss = skip = 0 / 307: todo = [g for g in ids if not cache_path(g).exists()]` — finding: Variable skip is not used.
- **#55** `lol_kills/research/side_objective_edges.py:142` — source evidence `142: meta_cols = ["blue_date", "blue_league", "blue_year", "blue_split", "blue_playoffs", "blue_patch", "blue_oe_year", "blue_gamelength"] / 143: m = blue.merge(red, on="gameid", how="inner")` — finding: Variable meta_cols is not used.
- **#57** `tests/model_v2/evaluation/test_benchmark_contract.py:40` — source evidence `40: per_league_fold = count // (5 * folds) / 41: rows: list[dict] = []` — finding: Variable per_league_fold is not used.
- **#58** `tests/model_v2/evaluation/test_b3_coverage.py:468` — source evidence `468: legitimate = root_authority / 469: with pytest.raises(TypeError, match="loader-issued"):` — finding: Variable legitimate is not used.
- **#59** `tests/model_v2/evaluation/test_b3_coverage.py:477` — source evidence `477: class ForgedSubclass(b3.LoadedB3CoverageAuthority): / 478: pass` — finding: Variable ForgedSubclass is not used.
- **#60** `tests/model_v2/evaluation/test_checkpoint_c1.py:259` — source evidence `259: class ForgedSubclass(c1.CheckpointC1Authority): / 260: pass` — finding: Variable ForgedSubclass is not used.
- **#65** `tests/model_v2/evaluation/test_outer_calibration.py:305` — source evidence `305: signed_theta = theta if z >= 0 else -theta / 306: p = served_probability(` — finding: Variable signed_theta is not used.
- **#67** `tests/model_v2/evaluation/test_r20_foundation.py:250` — source evidence `250: expected = replay_foundation_row_candidate( / 251: authority=authority,` — finding: Variable expected is not used.
- **#70** `lol_kills/knowledge/turret_dps_optimizer.py:576` — source evidence `576: features = score["features"] / 577: names = list(result["names"])` — finding: Variable features is not used.
- **#71** `lol_kills/total_kills_synthetic_prices.py:739` — source evidence `739: prices = [] / 740: for line in lines:` — finding: Variable prices is not used.
- **#72** `lol_kills/knowledge/vayne_rammus_optimizer.py:350` — source evidence `350: trinity_ready_at = -1.0 / 351: total_damage = 0.0` — finding: Variable trinity_ready_at is not used.
- **#73** `lol_kills/knowledge/vayne_rammus_optimizer.py:436` — source evidence `436: physical_raw = base_physical / 437: magic_raw = 0.0` — finding: Variable physical_raw is not used.
- **#74** `lol_kills/knowledge/vayne_rammus_optimizer.py:437` — source evidence `437: magic_raw = 0.0 / 438: # On-hit physical effects. BORK is 6% for ranged champions on the` — finding: Variable magic_raw is not used.

### `py/unused-local-variable`: Unused value may carry a model, provenance, or authority contract (10)

- Classification: **deferred source-contract review**
- Release impact: Potentially high impact. The value may represent a missing validation, identity field, or model calculation.
- Exact closure action: Trace the value to its intended contract. Add a regression test, then either consume it or remove the calculation while preserving required side effects.

- **#11** `lol_kills/v2/evaluation/b3_coverage.py:630` — source evidence `630: coverage_error = abs(empirical_coverage - expected_interval_coverage) / 631: rank_mean_z = abs(` — finding: Variable coverage_error is not used.
- **#20** `lol_kills/v2/provenance/g1_pre_event_receipt.py:172` — source evidence `172: rows = _load_feature_rows(feature_rows_path) / 173: transform = _safe_path(transform_path, label="feature transform")` — finding: Variable rows is not used.
- **#42** `lol_kills/v2/ratings/player/model.py:2393` — source evidence `2393: identity = { / 2394: "player_id": player_id,` — finding: Variable identity is not used.
- **#46** `lol_kills/private_decision_readiness.py:743` — source evidence `743: lcs_comparison = next( / 744: (` — finding: Variable lcs_comparison is not used.
- **#47** `lol_kills/private_decision_readiness.py:754` — source evidence `754: strong_lcs = next( / 755: (` — finding: Variable strong_lcs is not used.
- **#48** `lol_kills/private_decision_readiness.py:762` — source evidence `762: strong_roster_change = next( / 763: (` — finding: Variable strong_roster_change is not used.
- **#50** `lol_kills/private_decision_readiness.py:2480` — source evidence `2480: settlement_contract = protocol.get("settlement_contract") or {} / 2481: registries = protocol.get("registries") or {}` — finding: Variable settlement_contract is not used.
- **#51** `lol_kills/v2/data/protocols.py:194` — source evidence `194: normalized_roles = _normalize_role_set(action.role_set) / 195: pick_actions_by_side[action.side].append(action)` — finding: Variable normalized_roles is not used.
- **#56** `lol_kills/research/temporal_draft_runtime.py:1063` — source evidence `1063: game = games_by_id.get(fixture_id) or _target_game(run) / 1064: lineup_status, lineup_evidence = _lineup_status(` — finding: Variable game is not used.
- **#69** `lol_kills/ml/train.py:176` — source evidence `176: calibrated = False / 177:` — finding: Variable calibrated is not used.

### `py/unused-local-variable`: Local binding exists for a validation or negative-path test (3)

- Classification: **intentional/test-only**
- Release impact: Low release impact. The expression or class declaration exercises a validation boundary.
- Exact closure action: Keep the validation call or class declaration. Discard the result explicitly or add a short rationale so the intentional unused binding is clear.

- **#49** `lol_kills/v2/ratings/player/private_development_runner.py:350` — source evidence `350: folds = _folds_by_id(input_data) / 351: recomputed_player_observations = sum(len(observation.player_observations) for fold in input_data.folds for observation in fold.map_observations)` — finding: Variable folds is not used.
- **#66** `tests/model_v2/evaluation/test_r20_foundation.py:85` — source evidence `85: class ForgedAuthority(VerifiedFoundationAuthority): / 86: pass` — finding: Variable ForgedAuthority is not used.
- **#68** `tests/model_v2/ratings/team/test_team_rating_publication.py:134` — source evidence `134: league = LeagueRating.from_mapping( / 135: fixtures["global"]["league_rating"]` — finding: Variable league is not used.

## Coverage manifest

The following manifest is the live set used to generate the inventory. The validator compared it with the API response by alert number.

`9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 31, 32, 33, 34, 35, 36, 37, 38`
`39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59`
`60, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 102, 103, 104, 105, 106, 107, 108, 109, 110`
`111, 112, 113, 114, 115, 117, 118, 119, 138, 142, 143, 144, 145, 146, 159, 160, 161, 162, 163, 164`
`165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 184, 185`
`186, 187, 188, 189, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206`
`207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 226, 227`
`228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 247, 248, 249, 250`
`251, 252, 253, 254, 255, 256, 257, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271`
`272, 273, 274, 275, 276, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 296`
`297, 298, 299, 300, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 317, 318`
`319, 320, 321, 323, 325, 326, 327, 328, 329, 330, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341`
`342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361`
`362, 363, 364, 365, 366, 369, 372, 373, 374, 375, 376, 377, 380, 381, 382, 383, 384, 385, 386, 387`
`388, 389, 390, 391, 392, 393, 394, 395, 396, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413`
`414, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 431, 432, 433, 435, 436, 437, 441, 442`
`443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 454, 455, 456, 459, 463, 464, 470, 472, 473, 474`
`475`

Coverage result: 341 live IDs listed, 341 unique, 0 duplicates, 0 omissions. Rule counts above sum to 341.
