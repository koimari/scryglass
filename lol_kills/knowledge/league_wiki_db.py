"""Build and query a provenance-preserving SQLite index of the League Wiki vault.

The vault remains the source of record.  This module only materializes a local,
read-only-friendly query index over the validated ``latest.jsonl`` documents and
the complete catalog.  Raw wikitext, section boundaries, revision identifiers,
and source URLs are retained so a result can be checked against the source page.

Examples::

    python3 -m lol_kills.knowledge.league_wiki_db build \
      --vault data/lol/knowledge/obsidian/league-wiki \
      --database data/lol/knowledge/league-wiki.sqlite3

    python3 -m lol_kills.knowledge.league_wiki_db search \
      --database data/lol/knowledge/league-wiki.sqlite3 \
      "minion movement speed"

    python3 -m lol_kills.knowledge.league_wiki_db page \
      --database data/lol/knowledge/league-wiki.sqlite3 --title Minion

The database is deliberately an index, not an executable mechanics authority.
Patch-specific numeric claims still need a patch-pinned client-data source and
the caller must report the wiki revision date.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .league_wiki_vault import validate as validate_vault


DB_SCHEMA_VERSION = "scryglass:league-wiki-query-db:v1"
DEFAULT_VAULT = Path("data/lol/knowledge/obsidian/league-wiki")
DEFAULT_DATABASE = Path("data/lol/knowledge/league-wiki.sqlite3")
WIKITEXT_MARKER = (
    "<!-- Source-preserving wikitext begins below. "
    "Do not treat prose as executable mechanics without reconciliation. -->"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield value


def _extract_wikitext(document: str, *, source: Path) -> str:
    marker_index = document.find(WIKITEXT_MARKER)
    if marker_index < 0:
        raise ValueError(f"missing source marker in {source}")
    body = document[marker_index + len(WIKITEXT_MARKER) :]
    # _page_document writes one separator newline before and after the source
    # body.  Preserve the body itself, including a single terminal newline.
    if body.startswith("\n"):
        body = body[1:]
    if body.endswith("\n"):
        body = body[:-1]
    return body


_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_EXTERNAL_LINK_RE = re.compile(r"\[https?://\S+\s+([^\]]+)\]")
_STYLE_RE = re.compile(r"'{2,5}")
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")


def _search_text(wikitext: str) -> str:
    """Make a stable, human-searchable view without discarding raw wikitext."""

    text = _COMMENT_RE.sub(" ", wikitext)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _LINK_RE.sub(lambda match: match.group(2) or match.group(1), text)
    text = _EXTERNAL_LINK_RE.sub(lambda match: match.group(1), text)
    text = text.replace("{{", " ").replace("}}", " ")
    text = text.replace("[", " ").replace("]", " ")
    text = _STYLE_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


_HEADING_RE = re.compile(r"^(={2,6})[ \t]*(.*?)[ \t]*\1[ \t]*$", re.MULTILINE)


def _sections(wikitext: str) -> list[dict[str, Any]]:
    """Return line-addressable sections for evidence snippets and navigation."""

    matches = list(_HEADING_RE.finditer(wikitext))
    if not matches:
        return []
    lines = wikitext.splitlines()
    sections: list[dict[str, Any]] = []
    for ordinal, match in enumerate(matches):
        start_offset = match.start()
        start_line = wikitext.count("\n", 0, start_offset) + 1
        end_offset = matches[ordinal + 1].start() if ordinal + 1 < len(matches) else len(wikitext)
        end_line = wikitext.count("\n", 0, end_offset) + 1
        heading = match.group(2).strip()
        level = len(match.group(1))
        # The first line is the heading; sections store only the following body
        # while preserving the source line range in metadata.
        body = wikitext[match.end() : end_offset].lstrip("\n").rstrip("\n")
        sections.append(
            {
                "ordinal": ordinal,
                "level": level,
                "heading": heading,
                "start_line": start_line,
                "end_line": max(start_line, end_line),
                "body": body,
                "search_text": _search_text(f"{heading}\n{body}"),
            }
        )
    # Keep the split-lines calculation intentional: it catches malformed
    # documents during development without making it part of the DB contract.
    del lines
    return sections


def _connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS page_fts "
            "USING fts5(page_id UNINDEXED, title, text)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS section_fts "
            "USING fts5(section_id UNINDEXED, title, heading, text)"
        )
    except sqlite3.OperationalError as exc:
        raise RuntimeError("this Python SQLite build must provide FTS5") from exc
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pages (
            page_id INTEGER PRIMARY KEY,
            namespace INTEGER NOT NULL,
            title TEXT NOT NULL,
            source_url TEXT NOT NULL,
            retrieved_at TEXT,
            cataloged INTEGER NOT NULL CHECK (cataloged IN (0, 1)),
            has_text INTEGER NOT NULL CHECK (has_text IN (0, 1)),
            document_path TEXT,
            revision_path TEXT,
            revision_id INTEGER,
            revision_timestamp TEXT,
            parent_revision_id INTEGER,
            api_sha1 TEXT,
            content_sha256 TEXT,
            document_sha256 TEXT,
            content_model TEXT,
            content_format TEXT,
            UNIQUE(namespace, title)
        );

        CREATE TABLE IF NOT EXISTS page_text (
            page_id INTEGER PRIMARY KEY REFERENCES pages(page_id) ON DELETE CASCADE,
            wikitext TEXT NOT NULL,
            search_text TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sections (
            section_id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id INTEGER NOT NULL REFERENCES pages(page_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            level INTEGER NOT NULL,
            heading TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            body TEXT NOT NULL,
            search_text TEXT NOT NULL,
            UNIQUE(page_id, ordinal)
        );

        CREATE INDEX IF NOT EXISTS pages_namespace_title_idx
            ON pages(namespace, title COLLATE NOCASE);
        CREATE INDEX IF NOT EXISTS pages_revision_idx
            ON pages(revision_timestamp);
        CREATE INDEX IF NOT EXISTS sections_page_ordinal_idx
            ON sections(page_id, ordinal);
        """
    )


def _set_meta(connection: sqlite3.Connection, values: dict[str, Any]) -> None:
    connection.executemany(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        [(key, json.dumps(value, ensure_ascii=False, sort_keys=True)) for key, value in values.items()],
    )


def _page_values(row: dict[str, Any], *, cataloged: bool, has_text: bool) -> tuple[Any, ...]:
    return (
        int(row.get("page_id", row.get("pageid"))),
        int(row["namespace"]),
        str(row["title"]),
        str(row["source_url"]),
        row.get("retrieved_at"),
        int(cataloged),
        int(has_text),
        row.get("document_path"),
        row.get("revision_path"),
        row.get("revision_id"),
        row.get("revision_timestamp"),
        row.get("parent_revision_id"),
        row.get("api_sha1"),
        row.get("content_sha256"),
        row.get("document_sha256"),
        row.get("content_model"),
        row.get("content_format"),
    )


_PAGE_COLUMNS = (
    "page_id, namespace, title, source_url, retrieved_at, cataloged, has_text, "
    "document_path, revision_path, revision_id, revision_timestamp, parent_revision_id, "
    "api_sha1, content_sha256, document_sha256, content_model, content_format"
)


def _upsert_page(connection: sqlite3.Connection, row: dict[str, Any], *, cataloged: bool, has_text: bool) -> None:
    connection.execute(
        f"INSERT INTO pages ({_PAGE_COLUMNS}) VALUES ({','.join('?' for _ in range(17))}) "
        "ON CONFLICT(page_id) DO UPDATE SET "
        "namespace=excluded.namespace, title=excluded.title, source_url=excluded.source_url, "
        "retrieved_at=excluded.retrieved_at, cataloged=MAX(pages.cataloged, excluded.cataloged), "
        "has_text=MAX(pages.has_text, excluded.has_text), document_path=COALESCE(excluded.document_path, pages.document_path), "
        "revision_path=COALESCE(excluded.revision_path, pages.revision_path), revision_id=COALESCE(excluded.revision_id, pages.revision_id), "
        "revision_timestamp=COALESCE(excluded.revision_timestamp, pages.revision_timestamp), parent_revision_id=COALESCE(excluded.parent_revision_id, pages.parent_revision_id), "
        "api_sha1=COALESCE(excluded.api_sha1, pages.api_sha1), content_sha256=COALESCE(excluded.content_sha256, pages.content_sha256), "
        "document_sha256=COALESCE(excluded.document_sha256, pages.document_sha256), content_model=COALESCE(excluded.content_model, pages.content_model), "
        "content_format=COALESCE(excluded.content_format, pages.content_format)",
        _page_values(row, cataloged=cataloged, has_text=has_text),
    )


def _read_document(vault: Path, row: dict[str, Any]) -> tuple[str, str]:
    relative = row.get("document_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"text page has no document_path: {row.get('namespace')}:{row.get('title')}")
    path = vault / relative
    payload = path.read_bytes()
    expected_document_hash = row.get("document_sha256")
    actual_document_hash = _sha256_bytes(payload)
    if expected_document_hash != actual_document_hash:
        raise ValueError(f"document hash mismatch for {row.get('namespace')}:{row.get('title')}")
    document = payload.decode("utf-8")
    return document, _extract_wikitext(document, source=path)


def _insert_text_and_sections(
    connection: sqlite3.Connection,
    *,
    row: dict[str, Any],
    wikitext: str,
) -> int:
    page_id = int(row["page_id"])
    search_text = _search_text(wikitext)
    connection.execute(
        "INSERT INTO page_text(page_id, wikitext, search_text) VALUES (?, ?, ?)",
        (page_id, wikitext, search_text),
    )
    connection.execute(
        "INSERT INTO page_fts(page_id, title, text) VALUES (?, ?, ?)",
        (str(page_id), str(row["title"]), f"{row['title']}\n{search_text}\n{wikitext}"),
    )
    section_count = 0
    for section in _sections(wikitext):
        cursor = connection.execute(
            "INSERT INTO sections(page_id, ordinal, level, heading, start_line, end_line, body, search_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                page_id,
                section["ordinal"],
                section["level"],
                section["heading"],
                section["start_line"],
                section["end_line"],
                section["body"],
                section["search_text"],
            ),
        )
        section_id = int(cursor.lastrowid)
        connection.execute(
            "INSERT INTO section_fts(section_id, title, heading, text) VALUES (?, ?, ?, ?)",
            (
                str(section_id),
                str(row["title"]),
                section["heading"],
                f"{row['title']}\n{section['heading']}\n{section['search_text']}\n{section['body']}",
            ),
        )
        section_count += 1
    return section_count


def build_database(
    vault: Path = DEFAULT_VAULT,
    database: Path = DEFAULT_DATABASE,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Build an atomic SQLite index from a complete, hash-validated vault."""

    vault = vault.expanduser()
    database = database.expanduser()
    if database.exists() and not force:
        raise FileExistsError(f"database exists; pass --force to replace it: {database}")
    validation = validate_vault(vault, require_complete=True)
    inventory_path = vault / "inventory-manifest.json"
    snapshot_path = vault / "snapshot-manifest.json"
    catalog_path = vault / "catalog.jsonl"
    latest_path = vault / "latest.jsonl"
    database.parent.mkdir(parents=True, exist_ok=True)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{database.name}.", suffix=".partial", dir=database.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    page_count = 0
    text_page_count = 0
    section_count = 0
    legacy_latest_count = 0
    try:
        connection = _connection(temporary)
        try:
            _create_schema(connection)
            connection.execute("BEGIN")
            catalog_keys: set[tuple[int, str]] = set()
            for row in _jsonl(catalog_path):
                key = (int(row["namespace"]), str(row["title"]))
                if key in catalog_keys:
                    raise ValueError(f"duplicate catalog key: {key[0]}:{key[1]}")
                catalog_keys.add(key)
                _upsert_page(connection, row, cataloged=True, has_text=False)
                page_count += 1

            seen_latest_ids: set[int] = set()
            for row in _jsonl(latest_path):
                namespace = int(row["namespace"])
                title = str(row["title"])
                page_id = int(row["page_id"])
                if page_id in seen_latest_ids:
                    raise ValueError(f"duplicate latest page ID: {page_id}")
                seen_latest_ids.add(page_id)
                if (namespace, title) not in catalog_keys:
                    legacy_latest_count += 1
                document, wikitext = _read_document(vault, row)
                _upsert_page(connection, row, cataloged=(namespace, title) in catalog_keys, has_text=True)
                sections_written = _insert_text_and_sections(connection, row=row, wikitext=wikitext)
                text_page_count += 1
                section_count += sections_written

            inventory_manifest = json.loads(inventory_path.read_text(encoding="utf-8"))
            snapshot_manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
            meta = {
                "schema_version": DB_SCHEMA_VERSION,
                "built_at": _utc_now(),
                "vault_path": str(vault.resolve()),
                "vault_schema_version": inventory_manifest.get("schema_version"),
                "source_api": inventory_manifest.get("source_api"),
                "source_url": "https://wiki.leagueoflegends.com/en-us/",
                "inventory_manifest_sha256": _file_sha256(inventory_path),
                "snapshot_manifest_sha256": _file_sha256(snapshot_path),
                "catalog_sha256": inventory_manifest.get("catalog_sha256"),
                "latest_jsonl_sha256": snapshot_manifest.get("latest_jsonl_sha256"),
                "snapshot_complete": snapshot_manifest.get("complete"),
                "catalog_page_count": page_count,
                "latest_page_count": text_page_count,
                "section_count": section_count,
                "legacy_latest_count": legacy_latest_count,
                "file_namespace_metadata_only": validation.get("file_namespace_metadata_only", True),
                "license_note": "CC BY-SA 3.0; additional terms may apply",
            }
            _set_meta(connection, meta)
            connection.execute("INSERT INTO page_fts(page_fts) VALUES ('optimize')")
            connection.execute("INSERT INTO section_fts(section_fts) VALUES ('optimize')")
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary, database)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "database": str(database),
        "schema_version": DB_SCHEMA_VERSION,
        "catalog_page_count": page_count,
        "text_page_count": text_page_count,
        "section_count": section_count,
        "legacy_latest_count": legacy_latest_count,
        "source_catalog_sha256": validation.get("catalog_sha256"),
        "source_completion_status": validation.get("completion_status"),
    }


def _fts_query(query: str) -> str:
    terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
    if not terms:
        raise ValueError("search query must contain at least one word")
    return " AND ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def connect_database(database: Path = DEFAULT_DATABASE) -> sqlite3.Connection:
    if not database.exists():
        raise FileNotFoundError(f"missing League Wiki database: {database}")
    connection = _connection(database)
    row = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None or json.loads(row["value"]) != DB_SCHEMA_VERSION:
        connection.close()
        raise ValueError(f"unsupported or incomplete database: {database}")
    return connection


def search_pages(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 10,
    namespace: int | None = None,
) -> list[dict[str, Any]]:
    if limit < 1 or limit > 100:
        raise ValueError("limit must be in [1, 100]")
    match = _fts_query(query)
    rows = connection.execute(
        "SELECT p.page_id, p.namespace, p.title, p.source_url, p.revision_id, "
        "p.revision_timestamp, p.has_text, p.cataloged, "
        "snippet(page_fts, 2, '[', ']', '…', 28) AS snippet, bm25(page_fts, 5.0, 1.0) AS rank "
        "FROM page_fts JOIN pages AS p ON CAST(page_fts.page_id AS INTEGER) = p.page_id "
        "WHERE page_fts MATCH ? AND (? IS NULL OR p.namespace = ?) "
        "ORDER BY rank, p.title COLLATE NOCASE LIMIT ?",
        (match, namespace, namespace, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def search_sections(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int = 20,
    namespace: int | None = None,
) -> list[dict[str, Any]]:
    if limit < 1 or limit > 100:
        raise ValueError("limit must be in [1, 100]")
    match = _fts_query(query)
    rows = connection.execute(
        "SELECT s.section_id, s.page_id, p.namespace, p.title, p.source_url, p.revision_id, "
        "p.revision_timestamp, s.ordinal, s.level, s.heading, s.start_line, s.end_line, "
        "snippet(section_fts, 3, '[', ']', '…', 36) AS snippet, bm25(section_fts, 5.0, 4.0, 1.0) AS rank "
        "FROM section_fts JOIN sections AS s ON CAST(section_fts.section_id AS INTEGER) = s.section_id "
        "JOIN pages AS p ON p.page_id = s.page_id "
        "WHERE section_fts MATCH ? AND (? IS NULL OR p.namespace = ?) "
        "ORDER BY rank, p.title COLLATE NOCASE, s.ordinal LIMIT ?",
        (match, namespace, namespace, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def get_page(
    connection: sqlite3.Connection,
    *,
    title: str | None = None,
    page_id: int | None = None,
    namespace: int | None = None,
    include_text: bool = True,
) -> dict[str, Any] | None:
    if title is None and page_id is None:
        raise ValueError("provide title or page_id")
    if page_id is not None:
        row = connection.execute(
            "SELECT p.*, t.wikitext, t.search_text FROM pages AS p "
            "LEFT JOIN page_text AS t ON t.page_id = p.page_id WHERE p.page_id = ?",
            (page_id,),
        ).fetchone()
    else:
        if namespace is None:
            rows = connection.execute(
                "SELECT p.*, t.wikitext, t.search_text FROM pages AS p "
                "LEFT JOIN page_text AS t ON t.page_id = p.page_id "
                "WHERE p.title = ? ORDER BY p.namespace LIMIT 2",
                (title,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT p.*, t.wikitext, t.search_text FROM pages AS p "
                "LEFT JOIN page_text AS t ON t.page_id = p.page_id "
                "WHERE p.title = ? AND p.namespace = ? LIMIT 2",
                (title, namespace),
            ).fetchall()
        if not rows:
            return None
        if len(rows) > 1 and namespace is None:
            raise ValueError(f"title is ambiguous across namespaces: {title}")
        row = rows[0]
    if row is None:
        return None
    result = dict(row)
    if not include_text:
        result.pop("wikitext", None)
        result.pop("search_text", None)
    return result


def get_sections(
    connection: sqlite3.Connection,
    page_id: int,
    *,
    heading: str | None = None,
) -> list[dict[str, Any]]:
    if heading is None:
        rows = connection.execute(
            "SELECT section_id, page_id, ordinal, level, heading, start_line, end_line, body, search_text "
            "FROM sections WHERE page_id = ? ORDER BY ordinal",
            (page_id,),
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT section_id, page_id, ordinal, level, heading, start_line, end_line, body, search_text "
            "FROM sections WHERE page_id = ? AND heading LIKE ? COLLATE NOCASE ORDER BY ordinal",
            (page_id, f"%{heading}%"),
        ).fetchall()
    return [dict(row) for row in rows]


def status(connection: sqlite3.Connection) -> dict[str, Any]:
    values = {
        row["key"]: json.loads(row["value"])
        for row in connection.execute("SELECT key, value FROM meta ORDER BY key")
    }
    values["counts"] = {
        "pages": connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0],
        "text_pages": connection.execute("SELECT COUNT(*) FROM pages WHERE has_text = 1").fetchone()[0],
        "cataloged_pages": connection.execute("SELECT COUNT(*) FROM pages WHERE cataloged = 1").fetchone()[0],
        "sections": connection.execute("SELECT COUNT(*) FROM sections").fetchone()[0],
    }
    return values


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="build an atomic index from a validated vault")
    build_parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    build_parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    build_parser.add_argument("--force", action="store_true")

    for name in ("search", "search-sections"):
        query_parser = subparsers.add_parser(name)
        query_parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
        query_parser.add_argument("query")
        query_parser.add_argument("--namespace", type=int)
        query_parser.add_argument("--limit", type=int, default=10 if name == "search" else 20)

    page_parser = subparsers.add_parser("page")
    page_parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    page_parser.add_argument("--title")
    page_parser.add_argument("--page-id", type=int)
    page_parser.add_argument("--namespace", type=int)
    page_parser.add_argument("--no-text", action="store_true")

    sections_parser = subparsers.add_parser("sections")
    sections_parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    sections_parser.add_argument("--page-id", type=int, required=True)
    sections_parser.add_argument("--heading")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)

    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            _print_json(build_database(args.vault, args.database, force=args.force))
            return 0
        connection = connect_database(args.database)
        try:
            if args.command == "search":
                _print_json(search_pages(connection, args.query, limit=args.limit, namespace=args.namespace))
            elif args.command == "search-sections":
                _print_json(search_sections(connection, args.query, limit=args.limit, namespace=args.namespace))
            elif args.command == "page":
                if args.title is None and args.page_id is None:
                    raise ValueError("page requires --title or --page-id")
                result = get_page(
                    connection,
                    title=args.title,
                    page_id=args.page_id,
                    namespace=args.namespace,
                    include_text=not args.no_text,
                )
                if result is None:
                    raise ValueError("page not found")
                _print_json(result)
            elif args.command == "sections":
                _print_json(get_sections(connection, args.page_id, heading=args.heading))
            elif args.command == "status":
                _print_json(status(connection))
            else:
                raise ValueError(f"unsupported command: {args.command}")
        finally:
            connection.close()
        return 0
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
