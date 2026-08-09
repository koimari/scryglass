# League Wiki MCP server

This is a local, read-only MCP server over the complete indexed League Wiki at
`data/lol/knowledge/league-wiki.sqlite3`.

The hot path is `league_wiki_query`. It accepts any natural-language question
about the indexed Wiki and returns ranked section/page evidence with source URL,
revision ID, revision timestamp, and an answer contract for the connected model.
Its default `response_mode=fast` is a compact single pass intended for ordinary
questions; use `response_mode=standard` or `league_wiki_fetch` when a broader
source excerpt is needed. It performs no network calls. Champions, abilities,
items, runes, objectives, maps, minions, patch history, and other indexed topics
use the same interface.

Use `league_wiki_fetch` for exact page/section text, `league_wiki_search` for a
narrow FTS query, `league_wiki_sql` for bounded warehouse-style analysis, and
`league_wiki_status` for freshness and coverage checks. The connected agent
should not make a second retrieval call unless the fast result is insufficient
or the user asks for deeper evidence.

The server is a retrieval/provenance layer, not a second language model. The
connected agent remains responsible for explaining evidence, labelling derived
calculations, and saying when the Wiki does not establish a claim.

## Local smoke test

```sh
python3 tools/league_wiki_mcp/server.py
```

Then send newline-delimited JSON-RPC messages on stdin. The Codex configuration
entry uses the same command and keeps the SQLite connection open for repeated
queries.
