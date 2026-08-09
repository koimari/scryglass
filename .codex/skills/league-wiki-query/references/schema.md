# League Wiki query index

Database schema version: `scryglass:league-wiki-query-db:v1`

The index is an atomic, local SQLite build over the source-preserving League
Wiki vault. `meta` records the source hashes and build counts. The source vault,
not the index, remains authoritative.

## Tables

| Table | Purpose | Important columns |
| --- | --- | --- |
| `meta` | Build and source provenance | `key`, JSON-encoded `value` |
| `pages` | Complete catalog plus latest revision metadata | `page_id`, `namespace`, `title`, `source_url`, `cataloged`, `has_text`, `revision_id`, `revision_timestamp`, `content_sha256`, `document_sha256` |
| `page_text` | Raw latest wikitext for text-bearing pages | `page_id`, `wikitext`, `search_text` |
| `sections` | Heading-level evidence ranges | `section_id`, `page_id`, `ordinal`, `level`, `heading`, `start_line`, `end_line`, `body` |
| `page_fts` | FTS5 index over page title and text | `page_id`, `title`, `text` |
| `section_fts` | FTS5 index over section title, heading, and text | `section_id`, `title`, `heading`, `text` |

`cataloged = 1` means the page is present in the inventory. A small number of
legacy latest revisions may be retained with `cataloged = 0` so the index does
not discard source bytes. `has_text = 0` is expected for metadata-only file
pages.

## Useful queries

Find current text-bearing gameplay pages:

```sql
SELECT page_id, title, revision_id, revision_timestamp, source_url
FROM pages
WHERE namespace = 0 AND has_text = 1
ORDER BY title COLLATE NOCASE;
```

Find evidence sections for a page:

```sql
SELECT s.heading, s.start_line, s.end_line, s.body,
       p.title, p.revision_id, p.revision_timestamp, p.source_url
FROM sections AS s
JOIN pages AS p ON p.page_id = s.page_id
WHERE p.title = 'Minion' AND p.namespace = 0
ORDER BY s.ordinal;
```

Use FTS5 through the CLI for normal user-entered search. For direct SQL, quote
the search expression and use `MATCH`:

```sql
SELECT p.title, p.revision_id, p.revision_timestamp,
       snippet(page_fts, 2, '[', ']', '…', 28) AS snippet
FROM page_fts
JOIN pages AS p ON CAST(page_fts.page_id AS INTEGER) = p.page_id
WHERE page_fts MATCH '"movement" AND "speed"'
ORDER BY bm25(page_fts)
LIMIT 20;
```

## Provenance contract

Every text result should be accompanied by `title`, `namespace`, `source_url`,
`revision_id`, and `revision_timestamp`. Use the `sections` line range when a
claim is based on a narrow passage. Do not present a search snippet as a full
quote; retrieve the page or section body before quoting or calculating from it.
