#!/usr/bin/env python3
"""Read-only CLI for the Scryglass League Wiki SQLite index."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "scryglass:league-wiki-query-db:v1"
DEFAULT_DATABASE = Path("/Users/river/scryglass/data/lol/knowledge/league-wiki.sqlite3")


def database_path(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    configured = os.environ.get("SCRYGLASS_LEAGUE_WIKI_DB")
    if configured:
        return Path(configured).expanduser()
    local = Path.cwd() / "data/lol/knowledge/league-wiki.sqlite3"
    return local if local.exists() else DEFAULT_DATABASE


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise FileNotFoundError(f"missing League Wiki database: {path}")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    row = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None or json.loads(row["value"]) != SCHEMA_VERSION:
        connection.close()
        raise ValueError(f"unsupported or incomplete League Wiki database: {path}")
    return connection


def fts_query(query: str) -> str:
    terms = re.findall(r"[\w]+", query, flags=re.UNICODE)
    if not terms:
        raise ValueError("search query must contain at least one word")
    return " AND ".join(f'"{term}"' for term in terms)


def rows_as_dict(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def search(connection: sqlite3.Connection, query: str, limit: int, namespace: int | None) -> list[dict[str, Any]]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be in [1, 100]")
    rows = connection.execute(
        "SELECT p.page_id, p.namespace, p.title, p.source_url, p.revision_id, "
        "p.revision_timestamp, p.has_text, p.cataloged, "
        "snippet(page_fts, 2, '[', ']', '…', 28) AS snippet "
        "FROM page_fts JOIN pages AS p ON CAST(page_fts.page_id AS INTEGER) = p.page_id "
        "WHERE page_fts MATCH ? AND (? IS NULL OR p.namespace = ?) "
        "ORDER BY bm25(page_fts, 5.0, 1.0), p.title COLLATE NOCASE LIMIT ?",
        (fts_query(query), namespace, namespace, limit),
    ).fetchall()
    return rows_as_dict(rows)


def search_sections(connection: sqlite3.Connection, query: str, limit: int, namespace: int | None) -> list[dict[str, Any]]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be in [1, 100]")
    rows = connection.execute(
        "SELECT s.section_id, s.page_id, p.namespace, p.title, p.source_url, p.revision_id, "
        "p.revision_timestamp, s.ordinal, s.level, s.heading, s.start_line, s.end_line, "
        "snippet(section_fts, 3, '[', ']', '…', 36) AS snippet "
        "FROM section_fts JOIN sections AS s ON CAST(section_fts.section_id AS INTEGER) = s.section_id "
        "JOIN pages AS p ON p.page_id = s.page_id "
        "WHERE section_fts MATCH ? AND (? IS NULL OR p.namespace = ?) "
        "ORDER BY bm25(section_fts, 5.0, 4.0, 1.0), p.title COLLATE NOCASE, s.ordinal LIMIT ?",
        (fts_query(query), namespace, namespace, limit),
    ).fetchall()
    return rows_as_dict(rows)


def page(
    connection: sqlite3.Connection,
    *,
    title: str | None,
    page_id: int | None,
    namespace: int | None,
    include_text: bool,
) -> dict[str, Any] | None:
    if title is None and page_id is None:
        raise ValueError("provide --title or --page-id")
    if page_id is not None:
        row = connection.execute(
            "SELECT p.*, t.wikitext, t.search_text FROM pages AS p "
            "LEFT JOIN page_text AS t ON t.page_id = p.page_id WHERE p.page_id = ?",
            (page_id,),
        ).fetchone()
    elif namespace is None:
        matches = connection.execute(
            "SELECT p.*, t.wikitext, t.search_text FROM pages AS p "
            "LEFT JOIN page_text AS t ON t.page_id = p.page_id "
            "WHERE p.title = ? ORDER BY p.namespace LIMIT 2",
            (title,),
        ).fetchall()
        if len(matches) > 1:
            raise ValueError(f"title is ambiguous across namespaces: {title}")
        row = matches[0] if matches else None
    else:
        row = connection.execute(
            "SELECT p.*, t.wikitext, t.search_text FROM pages AS p "
            "LEFT JOIN page_text AS t ON t.page_id = p.page_id "
            "WHERE p.title = ? AND p.namespace = ? LIMIT 1",
            (title, namespace),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    if not include_text:
        result.pop("wikitext", None)
        result.pop("search_text", None)
    return result


def sections(connection: sqlite3.Connection, page_id: int, heading: str | None) -> list[dict[str, Any]]:
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
    return rows_as_dict(rows)


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


def read_only_sql(connection: sqlite3.Connection, statement: str) -> list[dict[str, Any]]:
    normalized = statement.lstrip().lower()
    if not (normalized.startswith("select ") or normalized.startswith("with ") or normalized.startswith("pragma ")):
        raise ValueError("SQL mode is read-only: use SELECT, WITH, or read-only PRAGMA")
    if ";" in statement.rstrip().rstrip(";"):
        raise ValueError("SQL mode accepts one statement")
    return rows_as_dict(connection.execute(statement).fetchall())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", help="SQLite path; defaults to SCRYGLASS_LEAGUE_WIKI_DB or the project database")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="search page titles and text")
    search_parser.add_argument("query")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--namespace", type=int)

    section_parser = subparsers.add_parser("search-sections", help="search section-level text")
    section_parser.add_argument("query")
    section_parser.add_argument("--limit", type=int, default=20)
    section_parser.add_argument("--namespace", type=int)

    page_parser = subparsers.add_parser("page", help="retrieve a page and its revision evidence")
    page_parser.add_argument("--title")
    page_parser.add_argument("--page-id", type=int)
    page_parser.add_argument("--namespace", type=int)
    page_parser.add_argument("--no-text", action="store_true")

    sections_parser = subparsers.add_parser("sections", help="list sections for a page")
    sections_parser.add_argument("--page-id", type=int, required=True)
    sections_parser.add_argument("--heading")

    subparsers.add_parser("status", help="show source and database counts")

    sql_parser = subparsers.add_parser("sql", help="run one read-only SQL statement")
    sql_parser.add_argument("statement", nargs="?")

    args = parser.parse_args(argv)
    try:
        path = database_path(args.database)
        connection = connect(path)
        try:
            if args.command == "search":
                result = search(connection, args.query, args.limit, args.namespace)
            elif args.command == "search-sections":
                result = search_sections(connection, args.query, args.limit, args.namespace)
            elif args.command == "page":
                result = page(
                    connection,
                    title=args.title,
                    page_id=args.page_id,
                    namespace=args.namespace,
                    include_text=not args.no_text,
                )
                if result is None:
                    raise ValueError("page not found")
            elif args.command == "sections":
                result = sections(connection, args.page_id, args.heading)
            elif args.command == "status":
                result = status(connection)
            elif args.command == "sql":
                statement = args.statement or sys.stdin.read()
                result = read_only_sql(connection, statement)
            else:
                raise ValueError(f"unsupported command: {args.command}")
        finally:
            connection.close()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, sqlite3.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
