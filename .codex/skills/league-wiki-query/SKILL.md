---
name: league-wiki-query
description: Query the locally ingested League of Legends Wiki source vault through its provenance-preserving SQLite index. Use when answering specific LoL Wiki mechanics, champion, item, map, minion, rune, or patch-history questions that require exact page evidence, revision dates, source links, searchable snippets, or read-only SQL over the ingested wiki.
---

# League Wiki Query

Use the bundled read-only CLI to ground answers in the local League Wiki
database. Do not answer a specific Wiki question from memory when this skill is
available: search first, retrieve the relevant page or section, and report the
revision evidence with the answer.

## Locate the database

Use `--database` when the database is not at the default path. Otherwise let the
CLI resolve `SCRYGLASS_LEAGUE_WIKI_DB`, the current project path, or the
canonical Scryglass path:

```text
/Users/river/scryglass/data/lol/knowledge/league-wiki.sqlite3
```

Check the index before a question if its freshness or completeness matters:

```bash
python3 /Users/river/.codex/skills/league-wiki-query/scripts/query_league_wiki.py status
```

The database is built from the validated source-preserving vault snapshot. Its
metadata includes the source catalog hash, latest-snapshot hash, retrieval
status, page counts, and build time.

## Query workflow

1. Translate the user’s question into a few concrete search terms. Include the
   named object and the mechanic (for example, `Minion movement speed` or
   `Aatrox Q sweet spot`).
2. Search page text and then section text. Prefer section results because they
   provide a narrower evidence range:

   ```bash
   python3 /Users/river/.codex/skills/league-wiki-query/scripts/query_league_wiki.py search-sections "minion movement speed" --limit 8
   ```

3. Retrieve the matching page without truncating the source when the answer
   depends on a table, template, formula, or historical note:

   ```bash
   python3 /Users/river/.codex/skills/league-wiki-query/scripts/query_league_wiki.py page --title "Minion" --namespace 0
   ```

   Use `sections --page-id ID --heading "Heading words"` to isolate the
   relevant section. Use `page --no-text` when only revision metadata is needed.
4. Answer only what the retrieved source supports. Preserve qualifiers,
   exceptions, mode restrictions, and patch or date context. Give the Wiki
   title, revision ID/date, and `source_url` near the claim so the user can
   inspect it.
5. If multiple pages or namespaces match, disambiguate with `--namespace` or
   explain the ambiguity. If the index has no supporting passage, say that the
   ingested Wiki does not establish the answer instead of filling the gap from
   memory.

## Deterministic derived-stat path

For a patch-pinned gold-equivalent of an elemental Dragon Slayer buff, use this
fixed sidecar after the Wiki lookup rather than rebuilding the arithmetic in
prose:

1. Query the relevant Dragon Slayer/buff-data template and retain its
   `revision_id`, `revision_timestamp`, and `source_url`.
2. Keep the Wiki mechanic and the patch-pinned client-data anchors separate:
   the Wiki establishes the effect, while the CDragon patch packet supplies
   item prices/stat amounts and the fastpack supplies level-based champion
   values.
3. Build a closed JSON array of champion snapshots from the GRID checkpoint or
   another explicitly named state source. Do not infer missing health, current
   movement speed, or item-derived stats when they are not present.
4. Run the shared calculator with explicit patch, stack, missing-health, and
   duration inputs:

   ```bash
   python3 /Users/river/scryglass/lol_kills/research/dragon_gold_equivalent.py \
     ocean states.json --stacks 1 --missing-health 0.5 --duration 35
   ```

   From the repository root, the equivalent module invocation is
   `python3 -m lol_kills.research.dragon_gold_equivalent ...`.
5. Preserve the calculator output and source hashes with the analysis. Treat
   `gold_equivalent: null` as intentionally unpriced; never replace it with
   zero. Keep direct dragon kill gold and any causal or win-value claim
   separate from the stat-equivalent.

The calculator is at
`lol_kills/research/dragon_gold_equivalent.py`; its item anchors are read from
the selected CDragon packet, and its Ocean conversion compares healing with
the champion's native HP5 against a Rejuvenation Bead anchor.

## Querying directly

Use the SQL mode for exact counts, inventories, and joins. It accepts one
read-only `SELECT`, `WITH`, or read-only `PRAGMA` statement and opens the
database in read-only mode:

```bash
python3 /Users/river/.codex/skills/league-wiki-query/scripts/query_league_wiki.py sql \
  "SELECT title, revision_id, revision_timestamp FROM pages WHERE namespace = 0 AND has_text = 1 ORDER BY title LIMIT 20"
```

Read [references/schema.md](references/schema.md) before writing a non-trivial
query. Never mutate the index from the skill. Rebuild it from the source vault
when the source snapshot changes.

## Evidence and limits

- Treat `wikitext` and section `body` as source evidence, not executable game
  logic. Preserve the raw wording; do not silently convert prose or templates
  into a new formula.
- Treat `revision_timestamp` and `revision_id` as part of every factual answer.
  A current-looking page can still contain an older revision.
- Distinguish three things: what the Wiki says, what a patch-pinned client-data
  source says, and any inference made from them. The Wiki index alone does not
  establish current live-server values, a causal effect, a probability, or a
  counterfactual.
- Treat missing text pages as metadata-only entries. Namespace 6/file pages are
  intentionally cataloged without copied media bodies.
- Report uncertainty plainly when the question depends on an unindexed image,
  a template expansion, an external citation, an omitted patch, or a page that
  predates the user’s requested version.

## Bundled resource

Run `scripts/query_league_wiki.py` with Python 3. It returns JSON so results can
be inspected, piped into another local analysis, or quoted with their page and
revision fields intact.
