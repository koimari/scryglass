#!/usr/bin/env python3
"""Dependency-free MCP server for the local League Wiki knowledge base.

The primary tool, ``league_wiki_query``, accepts any natural-language question
and returns ranked, revision-backed Wiki evidence.  Exact retrieval, FTS search,
bounded read-only SQL, and index status are also exposed for follow-up work.

This is a retrieval/provenance layer for a connected AI agent, not a second
language model.  It performs no network calls and keeps one read-only SQLite
connection open, making normal queries millisecond-scale.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lol_kills.knowledge.league_wiki_db import (  # noqa: E402
    connect_database,
    get_page,
    get_sections,
    search_pages,
    search_sections,
    status,
)

DEFAULT_DATABASE = ROOT / "data/lol/knowledge/league-wiki.sqlite3"
SUPPORTED_PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "scryglass-league-wiki"
SERVER_VERSION = "0.1.0"

_STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "by",
    "can", "could", "does", "do", "for", "from", "get", "has", "have",
    "how", "i", "if", "in", "into", "is", "it", "its", "me", "of", "on",
    "or", "that", "the", "their", "this", "to", "what", "when", "where",
    "which", "who", "why", "with", "would", "work", "you", "your",
}
_TITLE_GENERIC_TERMS = {
    "ability", "abilities", "base", "champion", "champions", "exact", "grubs",
    "item", "items", "movement", "rules", "score", "speed", "spell", "stats",
    "summoner", "void",
}


class ToolError(ValueError):
    """An invalid MCP tool argument or unavailable query result."""


def _database_path() -> Path:
    configured = os.environ.get("SCRYGLASS_LEAGUE_WIKI_DB")
    return Path(configured).expanduser() if configured else DEFAULT_DATABASE


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _terms(query: str) -> list[str]:
    tokens = re.findall(r"[\w][\w'-]*", query.lower(), flags=re.UNICODE)
    result: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = token.strip("'-")
        token = re.sub(r"(?:'|’)s$", "", token)
        token = token.replace("'", "").replace("’", "")
        if len(token) < 2 or token in _STOPWORDS or token in seen:
            continue
        if not re.fullmatch(r"[\w]+", token, flags=re.UNICODE):
            continue
        seen.add(token)
        result.append(token)
    return result[:24]


def _fts_query(terms: Iterable[str], operator: str) -> str:
    quoted = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms]
    return f" {operator} ".join(quoted)


def _compact_text(value: str, limit: int = 3000) -> str:
    text = value.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _redirect_target(wikitext: str) -> str | None:
    match = re.search(r"#redirect\s*\[\[([^\]|#]+)", wikitext, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _title_matches(connection: sqlite3.Connection, query: str, limit: int) -> list[dict[str, Any]]:
    terms = _terms(query)
    if not terms:
        return []
    # Exact single-word titles (for example ``Minion`` or ``Flash``) should
    # outrank alphabetically earlier pages whose title merely contains a
    # generic term such as ``speed``.
    exact_terms = [term for term in terms[:8]]
    likes = [f"%{term}%" for term in exact_terms]
    conditions = " OR ".join("p.title LIKE ? COLLATE NOCASE" for _ in likes)
    exact_case = "CASE " + " ".join(
        f"WHEN lower(p.title) = lower(?) THEN {index}" for index, _ in enumerate(exact_terms)
    ) + f" ELSE {len(exact_terms) + 1} END"
    match_count = " + ".join(
        "CASE WHEN lower(p.title) LIKE lower(?) THEN 1 ELSE 0 END" for _ in exact_terms
    )
    rows = connection.execute(
        "SELECT p.page_id, p.namespace, p.title, p.source_url, p.revision_id, "
        "p.revision_timestamp, p.has_text, p.cataloged "
        f"FROM pages AS p WHERE {conditions} ORDER BY ({match_count}) DESC, {exact_case}, "
        "CASE WHEN p.namespace = 0 THEN 0 ELSE 1 END, CASE WHEN instr(p.title, '(') > 0 THEN 1 ELSE 0 END, "
        "length(p.title), p.has_text DESC, p.title COLLATE NOCASE LIMIT ?",
        (*likes, *likes, *exact_terms, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def _exact_title_matches(connection: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    terms = [term for term in _terms(query) if term not in _TITLE_GENERIC_TERMS]
    # Direct title lookup is intentionally conservative. A question containing
    # two named terms (for example “Rabadon's Deathcap” or “Void Grubs”) is
    # better handled by the multi-term title search than by returning separate
    # disambiguation pages for each word.
    if len(terms) != 1:
        return []
    placeholders = ",".join("?" for _ in terms)
    rows = connection.execute(
        "SELECT p.page_id, p.namespace, p.title, p.source_url, p.revision_id, "
        "p.revision_timestamp, p.has_text, p.cataloged FROM pages AS p "
        f"WHERE p.namespace = 0 AND lower(p.title) IN ({placeholders}) AND p.has_text = 1 "
        "ORDER BY length(p.title), p.title COLLATE NOCASE",
        tuple(terms),
    ).fetchall()
    return [dict(row) for row in rows]


def _section_search(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    namespace: int | None,
) -> tuple[list[dict[str, Any]], str]:
    terms = _terms(query)
    if not terms:
        return [], "none"
    for index, operator in enumerate(("AND", "OR")):
        rows = connection.execute(
            "SELECT s.section_id, s.page_id, p.namespace, p.title, p.source_url, "
            "p.revision_id, p.revision_timestamp, s.ordinal, s.level, s.heading, "
            "s.start_line, s.end_line, snippet(section_fts, 3, '[', ']', '…', 42) AS snippet, "
            "bm25(section_fts, 5.0, 4.0, 1.0) AS rank "
            "FROM section_fts JOIN sections AS s ON CAST(section_fts.section_id AS INTEGER) = s.section_id "
            "JOIN pages AS p ON p.page_id = s.page_id "
            "WHERE section_fts MATCH ? AND (? IS NULL OR p.namespace = ?) "
            "ORDER BY rank, p.title COLLATE NOCASE, s.ordinal LIMIT ?",
            (_fts_query(terms, operator), namespace, namespace, limit),
        ).fetchall()
        if rows:
            return [dict(row) for row in rows], "and" if index == 0 else "or_fallback"
    return [], "none"


def _page_search(
    connection: sqlite3.Connection,
    query: str,
    *,
    limit: int,
    namespace: int | None,
) -> tuple[list[dict[str, Any]], str]:
    terms = _terms(query)
    if not terms:
        return [], "none"
    for index, operator in enumerate(("AND", "OR")):
        rows = connection.execute(
            "SELECT p.page_id, p.namespace, p.title, p.source_url, p.revision_id, "
            "p.revision_timestamp, p.has_text, p.cataloged, "
            "snippet(page_fts, 2, '[', ']', '…', 42) AS snippet, "
            "bm25(page_fts, 5.0, 1.0) AS rank "
            "FROM page_fts JOIN pages AS p ON CAST(page_fts.page_id AS INTEGER) = p.page_id "
            "WHERE page_fts MATCH ? AND (? IS NULL OR p.namespace = ?) "
            "ORDER BY rank, p.title COLLATE NOCASE LIMIT ?",
            (_fts_query(terms, operator), namespace, namespace, limit),
        ).fetchall()
        if rows:
            return [dict(row) for row in rows], "and" if index == 0 else "or_fallback"
    return [], "none"


@dataclass
class SourceEvidence:
    page: dict[str, Any]
    section: dict[str, Any] | None = None
    snippet: str | None = None

    def as_dict(self, include_text: bool = True, text_limit: int = 3000) -> dict[str, Any]:
        result: dict[str, Any] = {
            "title": self.page.get("title"),
            "page_id": self.page.get("page_id"),
            "namespace": self.page.get("namespace"),
            "source_url": self.page.get("source_url"),
            "revision_id": self.page.get("revision_id"),
            "revision_timestamp": self.page.get("revision_timestamp"),
        }
        if self.page.get("redirected_from"):
            result["redirected_from"] = self.page["redirected_from"]
        if self.section is not None:
            result["section"] = {
                "section_id": self.section.get("section_id"),
                "heading": self.section.get("heading"),
                "start_line": self.section.get("start_line"),
                "end_line": self.section.get("end_line"),
            }
            if include_text:
                result["section_text"] = _compact_text(
                    str(self.section.get("body", "")), limit=text_limit
                )
        elif include_text and self.page.get("wikitext"):
            result["page_text"] = _compact_text(str(self.page["wikitext"]), limit=text_limit)
        if self.snippet:
            result["search_snippet"] = self.snippet
        return result


class LeagueWikiServer:
    """MCP dispatcher backed by one read-only SQLite connection."""

    def __init__(self, database: Path | None = None) -> None:
        self.database = (database or _database_path()).expanduser().resolve()
        self.connection = connect_database(self.database)
        self.connection.execute("PRAGMA query_only = ON")
        self.connection.execute("PRAGMA busy_timeout = 1000")
        self._pages: dict[int, dict[str, Any]] = {}
        self._sections: dict[int, dict[str, Any]] = {}
        self._index_status = status(self.connection)

    def close(self) -> None:
        self.connection.close()

    def _page(self, page_id: int) -> dict[str, Any] | None:
        if page_id not in self._pages:
            page = get_page(self.connection, page_id=page_id, include_text=False)
            if page is not None:
                self._pages[page_id] = page
        return self._pages.get(page_id)

    def _section(self, section_id: int) -> dict[str, Any] | None:
        if section_id not in self._sections:
            row = self.connection.execute(
                "SELECT section_id, page_id, ordinal, level, heading, start_line, end_line, body, search_text "
                "FROM sections WHERE section_id = ?",
                (section_id,),
            ).fetchone()
            if row is not None:
                self._sections[section_id] = dict(row)
        return self._sections.get(section_id)

    def _page_with_text(self, page_id: int) -> dict[str, Any] | None:
        visited: list[dict[str, Any]] = []
        current_id = page_id
        for _ in range(4):
            page = get_page(self.connection, page_id=current_id, include_text=True)
            if page is None:
                return None
            target = _redirect_target(str(page.get("wikitext", "")))
            if not target:
                if visited:
                    page["redirected_from"] = [item["title"] for item in visited]
                return page
            visited.append(page)
            target_row = self.connection.execute(
                "SELECT page_id FROM pages WHERE namespace = ? AND title = ? LIMIT 1",
                (int(page.get("namespace", 0)), target),
            ).fetchone()
            if target_row is None:
                page["redirect_target"] = target
                return page
            current_id = int(target_row["page_id"])
        raise ToolError(f"redirect chain is too deep for page {page_id}")

    def query(self, arguments: dict[str, Any]) -> dict[str, Any]:
        question = arguments.get("question", arguments.get("query"))
        if not isinstance(question, str) or not question.strip():
            raise ToolError("question must be a non-empty string")
        response_mode = str(arguments.get("response_mode", "fast")).lower()
        if response_mode not in {"standard", "fast"}:
            raise ToolError("response_mode must be 'standard' or 'fast'")
        if response_mode == "fast":
            limit = min(max(int(arguments.get("limit", 3)), 1), 8)
            text_limit = min(max(int(arguments.get("text_limit", 900)), 200), 1200)
            title_limit = min(6, max(3, limit + 1))
            supporting_limit = 1
        else:
            limit = min(max(int(arguments.get("limit", 6)), 1), 20)
            text_limit = 3000
            title_limit = 10
            supporting_limit = 3
        namespace = arguments.get("namespace")
        namespace = int(namespace) if namespace is not None else None
        include_text = bool(arguments.get("include_full_sections", True))
        include_titles = bool(arguments.get("include_title_matches", True))

        section_hits, section_mode = _section_search(
            self.connection, question, limit=limit, namespace=namespace
        )
        page_hits, page_mode = _page_search(
            self.connection, question, limit=max(3, min(limit, 10)), namespace=namespace
        )
        if include_titles:
            title_hits = _exact_title_matches(self.connection, question)
            seen_title_ids = {int(hit["page_id"]) for hit in title_hits}
            title_hits.extend(
                hit for hit in _title_matches(self.connection, question, title_limit)
                if int(hit["page_id"]) not in seen_title_ids
            )
        else:
            title_hits = []

        evidence: list[dict[str, Any]] = []
        seen_sections: set[int] = set()

        # If the question contains a recognizable Wiki title, put that page's
        # own evidence first. This prevents a generic word such as “stats” or
        # “rules” from outranking the named champion/item/objective.
        topic_terms = set(_terms(question))
        topic_pages_added = 0
        for hit in title_hits:
            title = str(hit.get("title", ""))
            if topic_pages_added >= min(2, limit) or int(hit.get("namespace", 0)) != 0:
                continue
            if "(" in title or title.startswith(("File:", "Template:", "Module:")):
                continue
            page_id = int(hit["page_id"])
            page = self._page_with_text(page_id)
            if page is None:
                continue
            canonical_page_id = int(page["page_id"])
            sections = get_sections(self.connection, canonical_page_id)
            ranked_sections = sorted(
                sections,
                key=lambda row: sum(term in str(row.get("search_text", "")).lower() for term in topic_terms),
                reverse=True,
            )
            if ranked_sections:
                section = ranked_sections[0]
                section_score = sum(term in str(section.get("search_text", "")).lower() for term in topic_terms)
                if section_score == 0:
                    continue
                seen_sections.add(int(section["section_id"]))
                evidence.append(
                    SourceEvidence(page, section, hit.get("snippet")).as_dict(
                        include_text=include_text, text_limit=text_limit
                    )
                )
                topic_pages_added += 1
            else:
                evidence.append(
                    SourceEvidence(page, snippet=hit.get("snippet")).as_dict(
                        include_text=include_text, text_limit=text_limit
                    )
                )
                topic_pages_added += 1

        for hit in section_hits:
            section_id = int(hit["section_id"])
            if section_id in seen_sections:
                continue
            section = self._section(section_id)
            page = self._page(int(hit["page_id"]))
            if section is None or page is None or section_id in seen_sections:
                continue
            seen_sections.add(section_id)
            evidence.append(
                SourceEvidence(page, section, hit.get("snippet")).as_dict(
                    include_text=include_text, text_limit=text_limit
                )
            )

        supporting_evidence: list[dict[str, Any]] = []
        seen_supporting: set[int] = set()
        # Item articles often render their numeric stats through a structured
        # Module:ItemData/data/<item> page. Include that page when the question
        # names an item, so the agent does not stop at an infobox template with
        # no literal numbers in the preserved wikitext.
        for hit in title_hits:
            if len(supporting_evidence) >= supporting_limit or int(hit.get("namespace", 0)) != 0:
                continue
            title = str(hit.get("title", ""))
            if "(" in title or title.startswith(("File:", "Template:", "Module:")):
                continue
            module_title = f"Module:ItemData/data/{title}"
            module_row = self.connection.execute(
                "SELECT page_id FROM pages WHERE namespace = 828 AND title = ? LIMIT 1",
                (module_title,),
            ).fetchone()
            if module_row is None:
                continue
            module_id = int(module_row["page_id"])
            if module_id in seen_supporting:
                continue
            module_page = self._page_with_text(module_id)
            if module_page is None:
                continue
            item_evidence = SourceEvidence(module_page).as_dict(
                include_text=include_text, text_limit=text_limit
            )
            item_evidence["relationship"] = "structured_item_data_for"
            supporting_evidence.append(item_evidence)
            seen_supporting.add(module_id)

        related: list[dict[str, Any]] = []
        seen_pages = {int(item["page_id"]) for item in evidence if item.get("page_id") is not None}
        for hit in [*title_hits, *page_hits]:
            page_id = int(hit["page_id"])
            if page_id in seen_pages:
                continue
            page = self._page(page_id)
            if page is None:
                continue
            seen_pages.add(page_id)
            related.append(SourceEvidence(page, snippet=hit.get("snippet")).as_dict(include_text=False))
            if len(related) >= limit:
                break

        return {
            "question": question,
            "terms_used": _terms(question),
            "search_strategy": {
                "section_match": section_mode,
                "page_match": page_mode,
                "title_matches_included": include_titles,
            },
            "response_mode": response_mode,
            "evidence": evidence,
            "supporting_evidence": supporting_evidence,
            "related_pages": related,
            "index": {
                "schema_version": self._index_status.get("schema_version"),
                "built_at": self._index_status.get("built_at"),
                "source_url": self._index_status.get("source_url"),
                "snapshot_complete": self._index_status.get("snapshot_complete"),
                "counts": self._index_status.get("counts"),
            },
            "answer_contract": {
                "use_evidence_before_inference": True,
                "cite_source_url_revision_id_and_timestamp": True,
                "label_derived_calculations_and_uncertainty": True,
                "if_evidence_is_missing": "say unavailable, then use league_wiki_fetch or league_wiki_sql",
            },
            "database": str(self.database),
        }

    def fetch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        title = arguments.get("title")
        page_id = arguments.get("page_id")
        if title is None and page_id is None:
            raise ToolError("provide title or page_id")
        requested_page = get_page(
            self.connection,
            title=str(title) if title is not None else None,
            page_id=int(page_id) if page_id is not None else None,
            namespace=int(arguments["namespace"]) if arguments.get("namespace") is not None else None,
            include_text=False,
        )
        if requested_page is None:
            raise ToolError("page not found")
        page = self._page_with_text(int(requested_page["page_id"]))
        if page is None:
            raise ToolError("page not found")
        if not bool(arguments.get("include_text", True)):
            page.pop("wikitext", None)
            page.pop("search_text", None)
        heading = arguments.get("heading")
        if heading is None:
            return page
        sections = get_sections(self.connection, int(page["page_id"]), heading=str(heading))
        return {
            "page": {key: value for key, value in page.items() if key not in {"wikitext", "search_text"}},
            "sections": sections,
        }

    def search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query must be a non-empty string")
        limit = min(max(int(arguments.get("limit", 10)), 1), 100)
        namespace = arguments.get("namespace")
        namespace = int(namespace) if namespace is not None else None
        mode = str(arguments.get("mode", "sections"))
        if mode == "pages":
            results = search_pages(self.connection, query, limit=limit, namespace=namespace)
        elif mode == "sections":
            results = search_sections(self.connection, query, limit=limit, namespace=namespace)
        else:
            raise ToolError("mode must be 'sections' or 'pages'")
        return {"query": query, "mode": mode, "results": results, "database": str(self.database)}

    def sql(self, arguments: dict[str, Any]) -> dict[str, Any]:
        statement = arguments.get("sql")
        if not isinstance(statement, str) or not statement.strip():
            raise ToolError("sql must be a single SELECT or WITH statement")
        normalized = statement.strip().lower()
        if not (normalized.startswith("select ") or normalized.startswith("with ")):
            raise ToolError("SQL mode is read-only: use SELECT or WITH")
        if ";" in statement.rstrip().rstrip(";"):
            raise ToolError("SQL mode accepts one statement")
        if re.search(r"\b(attach|detach|pragma|vacuum|load_extension|insert|update|delete|drop|alter|create)\b", normalized):
            raise ToolError("SQL statement contains a disallowed operation")
        max_rows = min(max(int(arguments.get("max_rows", 200)), 1), 1000)
        self.connection.set_progress_handler(lambda: 1, 100_000)
        try:
            cursor = self.connection.execute(statement)
            columns = [column[0] for column in cursor.description or ()]
            raw_rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.Error as exc:
            raise ToolError(str(exc)) from exc
        finally:
            self.connection.set_progress_handler(None, 0)
        return {
            "sql": statement,
            "columns": columns,
            "rows": [dict(zip(columns, row)) for row in raw_rows[:max_rows]],
            "truncated": len(raw_rows) > max_rows,
        }

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "league_wiki_query":
            return self.query(arguments)
        if name == "league_wiki_fetch":
            return self.fetch(arguments)
        if name == "league_wiki_search":
            return self.search(arguments)
        if name == "league_wiki_sql":
            return self.sql(arguments)
        if name == "league_wiki_status":
            return status(self.connection)
        raise ToolError(f"unknown tool: {name}")


def _tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "league_wiki_query",
            "title": "Ask the League Wiki",
            "description": (
                "Evidence-only local League Wiki retrieval. For numeric or interaction "
                "calculations, use the canonical resident league_oracle_answer tool first; "
                "use this tool for compact revision-backed source evidence, page text, and "
                "rules the oracle has explicitly left unavailable. It covers champions, "
                "abilities, items, runes, objectives, maps, minions, and patch history."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user's League Wiki question."},
                    "response_mode": {
                        "type": "string",
                        "enum": ["fast", "standard"],
                        "default": "fast",
                        "description": "Use fast for the normal one-pass answer path; standard returns broader evidence.",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 3},
                    "text_limit": {
                        "type": "integer",
                        "minimum": 200,
                        "maximum": 1200,
                        "default": 900,
                        "description": "Maximum characters of each evidence passage in fast mode.",
                    },
                    "namespace": {"type": "integer", "description": "Optional Wiki namespace filter; main articles are 0."},
                    "include_full_sections": {"type": "boolean", "default": True},
                    "include_title_matches": {"type": "boolean", "default": True},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
        {
            "name": "league_wiki_fetch",
            "title": "Fetch exact League Wiki evidence",
            "description": "Retrieve a named/page-ID page or matching section with preserved text and revision metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "page_id": {"type": "integer"},
                    "namespace": {"type": "integer"},
                    "heading": {"type": "string"},
                    "include_text": {"type": "boolean", "default": True},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "league_wiki_search",
            "title": "Search League Wiki index",
            "description": "Run a precise FTS page or section search when broad retrieval needs narrowing.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "mode": {"type": "string", "enum": ["sections", "pages"], "default": "sections"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                    "namespace": {"type": "integer"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "league_wiki_sql",
            "title": "Read-only League Wiki SQL",
            "description": "Run one bounded SELECT/WITH query against the local Wiki index, similar to a warehouse query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 200},
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
        {
            "name": "league_wiki_status",
            "title": "League Wiki index status",
            "description": "Return source snapshot, freshness, and row counts for the local index.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    ]


def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def dispatch(server: LeagueWikiServer, message: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatch one JSON-RPC message; return ``None`` for notifications."""

    method = message.get("method")
    request_id = message.get("id")
    notification = "id" not in message
    if method == "notifications/initialized":
        return None
    if method == "initialize":
        requested = (message.get("params") or {}).get("protocolVersion")
        version = requested if requested in {SUPPORTED_PROTOCOL_VERSION, "2025-03-26", "2025-06-18"} else SUPPORTED_PROTOCOL_VERSION
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": (
                    "Use league_oracle_answer first for League calculations. Use "
                    "league_wiki_query with response_mode=fast for evidence-only Wiki retrieval; "
                    "it is a single-pass compact retrieval path. Use league_wiki_fetch or "
                    "league_wiki_sql only for a second pass when the compact evidence is insufficient "
                    "or the user requests full source text. "
                    "Separate Wiki facts from calculations and inference, and preserve "
                    "source URL, revision ID, and revision timestamp."
                ),
            },
        }
    if method == "ping":
        return None if notification else {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return None if notification else {"jsonrpc": "2.0", "id": request_id, "result": {"tools": _tool_specs()}}
    if method == "tools/call":
        if notification:
            return None
        params = message.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return _error_response(request_id, -32602, "tools/call requires a tool name and object arguments")
        try:
            result = server.call_tool(name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": _json(result)}],
                    "structuredContent": result,
                },
            }
        except (ToolError, OSError, sqlite3.Error, ValueError) as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"isError": True, "content": [{"type": "text", "text": str(exc)}]},
            }
    if notification:
        return None
    return _error_response(request_id, -32601, f"method not found: {method}")


def main() -> int:
    try:
        server = LeagueWikiServer()
    except Exception as exc:  # pragma: no cover - integration-only startup diagnostics
        print(f"{SERVER_NAME}: startup failed: {exc}", file=sys.stderr, flush=True)
        return 1
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("JSON-RPC message must be an object")
                response = dispatch(server, message)
            except (json.JSONDecodeError, ValueError) as exc:
                response = _error_response(None, -32700, str(exc))
            if response is not None:
                print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
