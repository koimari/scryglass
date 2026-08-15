# CodeQL quality triage: 2026-08-15

## Snapshot

Repository: `koimari/scryglass`

Branch: `refs/heads/main`

Commit: `30ed1939cf0243ed24343211bc82de755145757e`

CodeQL analyses:

- Python analysis `1623897911`, created at `2026-08-15T18:16:02Z`.
- JavaScript and TypeScript analysis `1623894643`, created at `2026-08-15T18:14:08Z`.

The inventory used the Code Scanning API with `ref=refs/heads/main`,
`state=open`, and `per_page=100`. The review kept alerts whose most recent
instance commit equals `30ed1939cf0243ed24343211bc82de755145757e`. Notes are
outside this slice.

The snapshot has 34 errors and 40 warnings. Three alerts belong to the
security-fix work in PR #266 and stay outside this triage:

| Alert | Location | Owner | Reason |
| --- | --- | --- | --- |
| #5 | `apps/scryglass/scripts/build-visual-identities.mjs:117` | PR #266 | `js/http-to-file-access`; protected path |
| #7 | `lol_kills/research/elemental_drakes.py:1865` | PR #266 | `py/clear-text-storage-sensitive-data`; protected path |
| #8 | `lol_kills/grid_live_foundation.py:1385` | PR #266 | `py/overly-permissive-file`; protected path |

The in-scope count is 71: 33 errors and 38 warnings. Alert #399 has a code
change in this branch. GitHub alert state remains open on the 30ed1939 snapshot.

PR #266 merged as `919d13911b55380bf758f13c21d7653aba74ff7d` during this
review. The post-merge CodeQL analyses are Python `1623907648` and
JavaScript and TypeScript `1623905150`. They report zero security alerts and
the same 71 quality alerts: 33 errors and 38 warnings. The alert numbers and
locations below remain current on that commit.

## Triage rules

`confirmed defect` means the alert describes an observable runtime or resource
correctness problem with a narrow safe fix.

`intentional negative test` means the test deliberately exercises a rejected
call or boundary.

`generated/lazy-export false positive` means the query cannot follow a
documented dynamic export or generated boundary.

`needs follow-up` means the code needs a later cleanup, semantic review, or
query-specific decision. Each entry has a path-specific action.

## Findings by rule

### `js/useless-assignment-to-local` warning

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #6 | `apps/elemental-drakes/src/components/DrakeStudy.tsx:1892` | `handleTabKey` reads `current` for the arrow, Home, and End branches. Every accepted key assigns `next` before line 1899. The other branch returns at line 1897. | needs follow-up | Change `let next = current` to a definite numeric declaration in a UI-only cleanup. Add the keyboard navigation test. |

### `py/call/wrong-named-argument` errors

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #155, #156 | `tests/test_blob_retention.py:261,263` | `test_runtime_override_arguments_and_approval_surface_do_not_exist` calls `RetentionExecutor.execute` with `force` and `approval` inside `pytest.raises(TypeError)`. The test then checks that the approval surface is absent. | intentional negative test | Keep the calls. A test-only CodeQL dismissal is justified after the test classification and this evidence are recorded. |
| #157 | `tests/model_v2/evaluation/test_r20_foundation.py:339` | The test first checks that `boundaries` is absent from `replay_foundation_row_candidate`, then calls the keyword inside `pytest.raises(TypeError)`. | intentional negative test | Keep the call. A test-only CodeQL dismissal is justified. |

### `py/comparison-of-identical-expressions` warnings

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #148 | `lol_kills/live_snapshots.py:44` | `_number` converts the input to `float`. `number == number` rejects NaN and the following infinity check rejects both infinities. | needs follow-up | Replace the idiom with `math.isfinite(number)` in a behavior-preserving cleanup. Run live snapshot parity tests. |
| #149 | `lol_kills/v2/draft/interactions/representation_rank_private_result.py:284` | `_optional_finite` accepts `numbers.Real`, converts to `float`, then applies the same finite-value test before the nonnegative test. | needs follow-up | Replace the idiom with `math.isfinite(number)`. Re-run the private result validation suite. |
| #150 | `lol_kills/v2/champions/atoms/schema.py:80` | `require_number` converts numeric input to `float` and uses the expression as a finite-value guard. | needs follow-up | Replace the idiom with `math.isfinite(number)`. Re-run atom schema parity tests. |

### `py/constant-conditional-expression` warnings

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #417 | `lol_kills/v2/data/g1_draft_features.py:304` | `_safe_write_many` contains a false conditional expression whose true arm calls `_safe_file` with label `impossible`. The false arm returns `None`. | needs follow-up | Remove the dead expression after the source-authority review for this module. Add a test that the safe writer still preflights every destination. |
| #418 | `lol_kills/live_oe_prior.py:67` | A list comprehension filters with `if True`. The result equals the source column list. | needs follow-up | Remove the constant filter in a source-preserving cleanup. Run the OE prior fixture tests. |

### `py/duplicate-key-dict-literal` warnings

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #151 | `lol_kills/draft_archetypes.py:75` and `:178` | The two `Yuumi` entries have the same tag set. Python keeps the later entry. | needs follow-up | Remove the legacy duplicate after checking any generated feature or source digests. Add an exact `champ_tags("Yuumi")` regression. |
| #152 | `lol_kills/draft_archetypes.py:91` and `:177` | The two `Kai'Sa` entries have the same tag set. | needs follow-up | Remove the legacy duplicate after digest review. Add an exact tag regression. |
| #153 | `lol_kills/draft_archetypes.py:94` and `:176` | The two `Ezreal` entries have the same tag set. | needs follow-up | Remove the legacy duplicate after digest review. Add an exact tag regression. |
| #154 | `lol_kills/draft_archetypes.py:114` and `:172` | The two `Nocturne` entries have the same tag set. | needs follow-up | Remove the legacy duplicate after digest review. Add an exact tag regression. |

### `py/file-not-closed` warning

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #398 | `lol_kills/v2/evaluation/benchmark_contract.py:630` | `_open_root_directory` returns a descriptor. `_read_regular_under_root` stores it in `descriptors` and closes every descriptor in the reversed `finally` block at lines 721-723. The leaf descriptor closes at lines 719-720. | needs follow-up | Treat this instance as a path-specific CodeQL false positive. A dismissal with the descriptor call path and cleanup lines is justified. Keep the manual descriptor contract. |
| #399 | `tools/codex_import.py:314` | The source file was opened and read in one expression. The old path had no close operation. This branch uses a `with open(...)` block. `tests/test_codex_import.py` tracks and asserts closure. | confirmed defect | Land the context-manager fix and regression. Close the open alert after the branch analysis reports the fixed location. |

### `py/implicit-string-concatenation-in-list` warnings

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #139 | `lol_kills/research/grubs_ranked_contest_proof.py:639-642` | The adjacent literals form one paragraph item in the `lines` list. The text continues the same sentence across source lines. | needs follow-up | Make the paragraph assembly explicit with `"".join(...)` or a named string. Preserve generated report bytes, then decide on a path-specific dismissal. |
| #140 | `lol_kills/research/grubs_intrinsic_value.py:2485-2487` | The adjacent literals form one abstract paragraph item. The next list item starts at line 2488. | needs follow-up | Make the paragraph assembly explicit after checking generated article output and source digests. |
| #141 | `lol_kills/v2/draft/interactions/oe_nuisance_baseline.py:910-911` | The adjacent literals form one limitation string. The surrounding list contains one string per limitation. | needs follow-up | Make the string assembly explicit and run the artifact hash tests before any alert dismissal. |

### `py/import-of-mutable-attribute` warning

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #397 | `lol_kills/v2/ratings/player/multileague_runner.py:42` | The runner imports `SEALED_FINAL_START` by value. Diagnostic and preflight context managers assign `multileague_runner.SEALED_FINAL_START` at runtime before calling `_validate_input` and `_replay`. A by-value import can keep the old timestamp. The module is included in source locks for the development receipts. | needs follow-up | Review the source-lock contract before changing the import to a module-qualified read. Add a boundary-override regression and refresh the registered source receipt in the same authorized change. |

### `py/missing-equals` warning

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #158 | `tests/model_v2/draft/interactions/test_representation_rank_2026_support_gate.py:339` | `_ReadAudit` extends `dict` to record keys read through `get`. The test compares `read_keys`, and it does not compare `_ReadAudit` instances. The extra state is audit instrumentation. | intentional negative test | Keep the test helper. A test-only CodeQL dismissal is justified with this evidence. |

### `py/multiple-definition` warnings

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #121, #122 | `lol_kills/v2/tierlists/champion_elo.py:959,962` | `interaction_mean` is assigned in both branches. The later computation uses `interaction_draw`; `interaction_mean` has no read. | needs follow-up | Remove the dead assignments only with the tier-list source digest review. Re-run tier fit parity tests. |
| #123 | `lol_kills/draft_phase_score.py:334` | The nearest bucket assignment is followed by a complete `if/elif/else` chain that assigns `t` for every path. | needs follow-up | Remove the initial assignment in a Draft Score parity patch. Keep the output fixtures unchanged. |
| #124 | `lol_kills/research/draft_wr_study.py:576` | `p_joint` receives a sigmoid proxy and is overwritten by `LogisticRegression.predict_proba` at lines 587-589 before any read. | needs follow-up | Remove the proxy assignment after the study artifact and source digest review. |
| #125 | `lol_kills/ml/eval.py:131` and `:136` | The first `ok` expression checks the mean baseline and the Elo bound. The second assignment keeps only the mean-baseline check. The comment says the Elo tie rule is allowed. | needs follow-up | Obtain a metric-owner decision. Preserve current gate semantics until a parity fixture covers the intended rule. |
| #126 | `lol_kills/research/grid_event_ledger.py:143,188` | `previous_frame` receives the baseline and each processed frame. Resource deltas use `previous_team` and `previous_player`; `previous_frame` has no read. | needs follow-up | Remove the dead state after the ledger artifact digest review. Re-run ledger reconciliation tests. |
| #127 | `lol_kills/research/grubs_ranked_contest_proof.py:511,521` | `b` is calculated from the gold difference at line 511 and overwritten by `bin_gold(g)` at line 521 before the first value is read. | needs follow-up | Remove the first calculation after report-byte and source-digest review. |
| #128, #129 | `lol_kills/research/grubs_intrinsic_value.py:2164-2165,2168-2169` | Legacy kill values are loaded from the artifact and overwritten by the current `kill_net_gold_table` values before downstream use. | needs follow-up | Remove the legacy assignments after the intrinsic-value artifact review. Preserve all generated numbers. |
| #130 | `lol_kills/v2/draft/interactions/model.py:929` | `_ordered_pair` returns an orientation value. The function uses the ordered role and champion values, while `_orientation` has no read. | needs follow-up | Replace the unused binding with `_` after the model source-digest review. |
| #131 | `tests/model_v2/evaluation/test_evaluation_harness.py:388,394-395` | The test computes the test-window end twice. The second computation is the value appended to `test_windows`. | intentional negative test | Remove the first test-only assignment or keep a path-specific test dismissal. The assertion coverage stays the same. |

### `py/pythagorean` warnings

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #465 | `lol_kills/ratings/dual_elo.py:189` | Team sigma combines two finite state sigmas with `sqrt(sb.sigma**2 + sr.sigma**2)`. State updates cap each sigma at 150, while the query still identifies a general overflow path. | needs follow-up | Replace with `math.hypot` after exact-output parity tests and the rating source-contract review. |
| #466 | `lol_kills/ratings/player_elo.py:485` | Match sigma combines aggregate player sigmas. | needs follow-up | Use `math.hypot` after rating parity tests. |
| #135 | `lol_kills/ratings/player_elo.py:394` | Bridge uncertainty combines a bounded row sigma and bridge sigma. | needs follow-up | Use `math.hypot` after rating parity tests. |
| #137 | `lol_kills/ratings/player_elo.py:961` | Live player sigma combines the two aggregate sigmas. | needs follow-up | Use `math.hypot` after rating parity tests. |
| #134 | `lol_kills/ratings/hierarchical_bt.py:447` | Unbridged league uncertainty combines `sigma` and the league bridge term. | needs follow-up | Use `math.hypot` after hierarchical-rating parity tests. |
| #133 | `lol_kills/research/grubs_intrinsic_pdf.py:1142` | The PDF builder combines two standard errors before interval construction. | needs follow-up | Use `math.hypot` after generated PDF and source-digest checks. |

### `py/redundant-assignment` errors

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #415 | `lol_kills/v2/evaluation/pipeline.py:1123` | `test_predictions` is assigned to itself after role and side invariance checks. The next operation only reads the existing mapping. | needs follow-up | Remove the self-assignment in a pipeline source-contract patch. Run the complete evaluation gate suite. |
| #416 | `tools/lol_mechanics_mcp/server.py:414` | `configured_path` is assigned to itself after its conditional derivation and before the `configured_path is not None` branch. | needs follow-up | Remove the self-assignment in a focused MCP runtime patch. Run startup and fast-path tests. |

### `py/redundant-comparison` warning

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #147 | `lol_kills/v2/ratings/team/estimands_v1.py:115` | The preceding condition at line 113 rejects every `ref <= 0` value. The second check at line 115 has no reachable true path. | needs follow-up | Remove the duplicate check with the unreachable statement at line 116 in a model-source parity patch. |

### `py/undefined-export` errors

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #75 | `lol_kills/export/__init__.py:3` | `__all__` names `export_public_pack`. Module-level `__getattr__` imports and returns the symbol at lines 6-10. | generated/lazy-export false positive | Keep the lazy export. A path-specific false-positive dismissal is justified with the `__getattr__` evidence. |
| #76-#101 | `lol_kills/v2/draft/terminal/__init__.py:51-76` | `_EXPORTS` maps each public name to its source module and attribute at lines 11-37. `__getattr__` imports the mapped module, reads the attribute, caches it in `globals`, and returns it at lines 41-48. The names in `__all__` match the map. | generated/lazy-export false positive | Keep the lazy package surface. A path-specific false-positive dismissal is justified for the exact `_EXPORTS` and `__getattr__` implementation. |

### `py/uninitialized-local-variable` error

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #120 | `lol_kills/v2/data/real_spine.py:1242` | `result` is assigned in each accepted command branch. The exception handler calls `parser.error`, which raises `SystemExit` in the CLI path. The existing test `test_cli_rejects_an_unrecognized_command_before_printing_a_result` verifies that exit. | needs follow-up | Add an explicit post-`parser.error` return or initialize the local after the source-authority review. Keep the CLI error behavior and add a direct branch test. |

### `py/unreachable-statement` warnings

| Alerts | Location | Evidence | Classification | Closure action |
| --- | --- | --- | --- | --- |
| #400 | `lol_kills/v2/tierlists/champion_elo.py:405` | `_fit_hierarchical_cell` raises the retired-path error at line 402. The old fitter body starts at line 405. | needs follow-up | Remove the retired body only after tier-list source digest and historical replay review. |
| #401 | `lol_kills/v2/tierlists/champion_elo.py:1113` | `build_candidate` returns `build_pooled_candidate(...)` at line 1105. The legacy candidate builder starts at line 1113. | needs follow-up | Remove the legacy body only after tier-list source digest and historical replay review. |
| #402 | `lol_kills/v2/ratings/team/estimands_v1.py:116` | The preceding validation rejects `ref <= 0` at line 113. The second check at line 115 stays false, so the raise at line 116 is unreachable. | needs follow-up | Remove the duplicate validation with alert #147 in a model-source parity patch. |

## Changes in this branch

- `tools/codex_import.py` reads each imported skill file inside a context
  manager.
- `tests/test_codex_import.py` uses a tracked reader and asserts that every
  source handle closes.

## Dismissal guidance

The evidence supports path-specific dismissals for alerts #75-#101 as lazy
exports, #155-#157 and #158 as intentional test behavior, and #398 as an
explicit descriptor cleanup false positive. The adjacent literal findings
have intentional paragraph semantics and need a cleanup decision before a
dismissal. The remaining findings need code cleanup, metric review, source
digest review, or a narrow runtime patch. This branch does not call the alert
dismissal API.

## Verification record

Checks completed before the branch push:

- `python3 -m pytest -q tests/test_codex_import.py`: 1 passed.
- `python3 -m pytest -q tests/test_codex_import.py tests/model_v2/evaluation/test_benchmark_contract.py`: 87 passed.
- `python3 -m py_compile tools/codex_import.py`.
- `python3 -m compileall -q tools/codex_import.py tests/test_codex_import.py`.
- `mypy tools/codex_import.py`: clean.
- `git diff --check`: clean.

The combined diagnostic run with `tests/model_v2/data/test_real_spine.py`
reported 115 passed and 3 failures. The failures require private snapshot
files absent from this clean worktree:
`data/lol/v2/snapshots/real-v1/lpl-private-development-rows.jsonl` and its
manifest. The changed module tests pass.

The pushed-branch CodeQL run is the final check for alert #399 closure.
