# <recommended_plugins>
Here is a list of plugins that are available but not installed.

- Atlassian R

| | |
|---|---|
| Session | `019fa488-f5af-7b61-b3bd-0d9c33cf574c` |
| Started | 2026-07-28T20:19:13.735Z |
| CWD | `/Users/river/scryglass` |
| Model provider | openai |
| CLI | 0.146.0-alpha.3.1 |
| Completed | True |
| Rollout | `/Users/river/.codex/sessions/2026/07/28/rollout-2026-07-28T18-09-09-019faa8f-5a9e-7990-8634-d02a50eacfbc.jsonl` |

Tags: draft, ratings, tier-list, champion-atoms, grubs, live, market, frontend, data-warehouse, leaguepedia, evaluation, deploy, replay

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

## Final agent message

Conclusion: no current JSON Schema governs the internal L6 interaction artifacts. This is an S0 contract gap, not just missing test coverage.

Key evidence:

- Draft 2020-12 schemas use stable web `$id` values, explicit `required` fields, and `additionalProperties: false` for complete object shapes: [common.schema.json](/Users/river/scryglass/docs/model-v2/contracts/common.schema.json:1), [model-manifest.schema.json](/Users/river/scryglass/docs/model-v2/contracts/model-manifest.schema.json:1).
- Shared refs cover semver, model version, timestamps, SHA-256, stable IDs, and lineage: [common.schema.json](/Users/river/scryglass/docs/model-v2/contracts/common.schema.json:6), [common.schema.json](/Users/river/scryglass/docs/model-v2/contracts/common.schema.json:91). There is no reusable common `artifact_ref`.
- The production manifest defines an inline strict artifact entry `{artifact_id,purpose,media_type,sha256}`: [model-manifest.schema.json](/Users/river/scryglass/docs/model-v2/contracts/model-manifest.schema.json:88).
- The stronger internal convention is L4’s exact `{role,artifact_id,locator,media_type,raw_sha256,canonical_sha256}` reference, with both byte and parsed-object verific

## Tools and files

| Tool | Calls |
|---|---|
| send_message | 1 |
