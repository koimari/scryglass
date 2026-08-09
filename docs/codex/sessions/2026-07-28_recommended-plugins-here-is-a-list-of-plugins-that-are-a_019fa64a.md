# <recommended_plugins>
Here is a list of plugins that are available but not installed.

- Atlassian R

| | |
|---|---|
| Session | `019fa64a-2ca8-7922-a839-3529798c6762` |
| Started | 2026-07-28T01:15:06.830Z |
| CWD | `/Users/river/scryglass` |
| Model provider | openai |
| CLI | 0.146.0-alpha.3.1 |
| Completed | True |
| Rollout | `/Users/river/.codex/sessions/2026/07/27/rollout-2026-07-27T22-15-06-019fa64a-2ca8-7922-a839-3529798c6762.jsonl` |

Tags: draft, ratings, tier-list, champion-atoms, grubs, live, market, frontend, data-warehouse, leaguepedia, evaluation, deploy

## Codex rollout summary

```text
thread_id: 019fa64a-2ca8-7922-a839-3529798c6762
updated_at: 2026-07-28T02:35:57+00:00
rollout_path: /Users/river/.codex/sessions/2026/07/27/rollout-2026-07-27T22-15-06-019fa64a-2ca8-7922-a839-3529798c6762.jsonl
cwd: /Users/river/scryglass
git_branch: codex/app-visual-revamp

# Scryglass L1 B1/B2 provenance and authority remand completed

Rollout context: Worked in `/Users/river/scryglass` under strict ownership of v2 provenance, publication data, and model-v2 data tests. Frozen docs and unrelated work were preserved; no staging, commit, push, or deployment occurred.

## Task 1: B1 snapshot integrity hardening

Outcome: success

Key steps:
- Enforced repository-resolved source manifest locator plus independently recomputed manifest ID/object hash pair.
- Added real hashed environment-lock, split-assignment, and row-count evidence artifacts; count maps are derived from exact calendar-year/league/tier/patch/source grain rows, preventing total-preserving lies. Golden tier is `tier1`.
- Required positive finite freshness SLOs, parsed UTC instant comparisons, exact appearance/source-row availability ordering, recursive forbidden-field rejection, and exact lineage pair/tree/freshness coverage.
- Renamed leaf evidence semantics via `leaf_source_row_evidence`; leaf hashes are not treated as source-manifest pairs.
- Added mutation probes and regenerated B1 artifacts.

Validation: focused B1/data tests passed; final combined data suite passed `165 passed`.

## Task 2: B2 transform and publication authority hardening

Outcome: success

Key steps:
- Added independently pinned recipe registry at `data/lol/v2/publication/allowed-recipe-registry-b2.json`; raw SHA-256 `5f948d6d245aa983660fa7abcc9e7d5ff171d836095890719276dd05252676a3`. Loader verifies pinned bytes, registry lineage to the B1 source snapshot, exact recipe locator/ID, ordered roles/selectors/cardinality, output schema, and code/config hashes. Self-rehashed or alternate-locator recipes are rejected.
- Added empty production C4 authority root at `data/lol/v2/publication/c4-authority-registry-b2.json`; raw SHA-256 `6afcf98e948905578c2e871fd304fa2dafd0d644a7fa541267778d21996d2ffa`. It contains zero authorities and zero approved packets, so production/public approval fails closed.
- Added separately pinned test-only C4 registry at `data/lol/v2/publication/c4-test-authority-registry-b2.json`; raw SHA-256 `c991bedd9efb96a75bbe5ad1f1d42ad3d6ef440f2087d58c48039d8b63e1d5dc`. The same validation path checks exact packet locator/raw hash/object hash, authorized approver, scope, row/matrix bindings, terms/review chain, decision time, and C4 ID.
- Added direct probes for self-rehashed recipes, leaf removal, alternate copies, registry tampering, caller-created C4 packets/approvers, alternate packet copies, and rehashed authority roots.
- Public and authenticated allowlists remain empty; private allowlist was regenerated.

Final validation: `python3 -m pytest tests/model_v2/data -q` -> `165 passed`; frozen Draft 2020-12 schema valid; `compileall`, recursive fixture safety scan, tracked/untracked diff checks all clean. Key IDs: recipe `scryglass:transform-recipe:77bf176b…008551`; manifest `scryglass:transform-manifest:8af674f…2279e0`. SciSpace, Wolfram, and Academic Writing Toolkit were not used because executable authority validation was the material evidence.
```

## User prompts

### 1

<recommended_plugins>
Here is a list of plugins that are available but not installed.

- Atlassian Rovo (atlassian-rovo@openai-curated-remote)
- Box (box@openai-curated-remote)
- Figma (figma@openai-curated-remote)
- Gmail (gmail@openai-curated-remote)
- Google Calendar (google-calendar@openai-curated-remote)
- Google Drive (google-drive@openai-curated-remote)
- Notion (notion@openai-curated-remote)
- Outlook Calendar (outlook-calendar@openai-curated-remote)
- Outlook Email (outlook-email@openai-curated-remote)
- SharePoint (sharepoint@openai-curated-remote)
- Slack (slack@openai-curated-remot

*(+8 further user messages; see rollout JSONL)*

## Final agent message

Final authority-layer remand is fixed.

Trust roots:

- Recipe registry: `data/lol/v2/publication/allowed-recipe-registry-b2.json`
  - Raw SHA-256: `5f948d6d245aa983660fa7abcc9e7d5ff171d836095890719276dd05252676a3`
- Production C4 registry: `data/lol/v2/publication/c4-authority-registry-b2.json`
  - Raw SHA-256: `6afcf98e948905578c2e871fd304fa2dafd0d644a7fa541267778d21996d2ffa`
  - Contains zero authorities and zero approved packets.
- Test-only C4 oracle: `data/lol/v2/publication/c4-test-authority-registry-b2.json`
  - Raw SHA-256: `c991bedd9efb96a75bbe5ad1f1d42ad3d6ef440f2087d58c48039d8b63e1d5dc`

These locators and hashes are pinned in [allowlist.py](/Users/river/scryglass/lol_kills/v2/provenance/allowlist.py:22) and [publication.py](/Users/river/scryglass/lol_kills/v2/provenance/publication.py:34). Loaders verify the pinned raw hash before accepting content-derived IDs. Submitted manifests and decision chains cannot select another registry or authorize newly rehashed objects.

Direct probes now reject:

- Self-rehashed one-input recipes and leaf removal
- Canonical recipes copied to alternate locators
- Rehashed recipe-registry tampering
- Output binding substitutions
- Caller-

## Tools and files

| Tool | Calls |
|---|---|
| exec_command | 49 |
| update_plan | 18 |
| write_stdin | 6 |
