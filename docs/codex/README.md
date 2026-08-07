# Codex import

Distilled records of Codex CLI sessions whose working directory was this repo
(or its Codex worktrees), imported by `tools/codex_import.py`.

- `INDEX.md` — sortable table of every imported session with topic tags.
- `manifest.json` — machine-readable index (metadata, user prompts, final
  agent messages, tools, file touches).
- `sessions/` — one markdown record per session, including Codex's own
  rollout summary when one exists.

Raw JSONL rollouts remain at `~/.codex/sessions/**` and are not copied into
this repo (privacy + size). Re-run the importer after Codex work to refresh:

```bash
python tools/codex_import.py --repo /Users/river/scryglass --related --skills
```

## Skills

Curated user skills from `~/.codex/skills` are copied into `.codex/skills/`
(see `skills-import.md`): `league-wiki-query`, `query-grid-research`,
`who-wins-this-game`, `frontend-skill`, `frontend-design`, `write-website-copy`,
`pdf`, `playwright`. Heavy or unrelated skills (browser binaries, `sora`,
`visor-mcp`, `hatch-pet`, `tldraw-offline`, `.system/*`) are not copied.

## Unmerged Codex git branches

Codex work that has not landed on `main` (check `git worktree list` and
`git log main..<branch>` before relying on it):

- `codex/app-visual-revamp` — 4 commit(s) ahead of main: 55cfe8a feat(calculator): ship Blender Rift environment background; 403af38 Revamp Scryglass interface and voice; d79fbd3 feat: professionalize draft recommendations; 577326b feat: add live draft analysis sandbox
- `codex/draft-sandbox` — 1 commit(s) ahead of main: 577326b feat: add live draft analysis sandbox
- `codex/draft-sandbox-professional` — 2 commit(s) ahead of main: d79fbd3 feat: professionalize draft recommendations; 577326b feat: add live draft analysis sandbox
- `codex/fix-matches-schema` — 1 commit(s) ahead of main: 7feae31 fix: tolerate older match pack provenance schema
- `codex/contextual-ratings-build` — 16 commit(s) ahead of main: 5c47a21 Merge pull request #40 from koimari/codex/interface-voice-revamp; c3b303b Revamp Scryglass interface and voice; fe8976d Merge pull request #39 from koimari/codex/hourly-pack-refresh; 3143a11 Run public pack refresh hourly
- `codex/interface-voice-revamp` — 0 commit(s) ahead of main: 
