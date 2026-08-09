from __future__ import annotations

from pathlib import Path

from tools.league_wiki_mcp.server import LeagueWikiServer, dispatch


DATABASE = Path("data/lol/knowledge/league-wiki.sqlite3")


def test_general_query_returns_revision_backed_evidence() -> None:
    server = LeagueWikiServer(DATABASE)
    try:
        result = server.query({"question": "How does minion movement speed work?"})
    finally:
        server.close()

    assert result["terms_used"] == ["minion", "movement", "speed"]
    assert result["evidence"]
    assert any(item["title"] == "Minion" for item in result["evidence"] + result["related_pages"])
    minion = next(item for item in result["evidence"] if item["title"] == "Minion")
    assert minion["source_url"].endswith("/Minion")
    assert minion["revision_id"] == 4041830
    assert minion["section_text"]


def test_query_is_not_topic_specific() -> None:
    server = LeagueWikiServer(DATABASE)
    try:
        result = server.query({"question": "What does the Flash summoner spell do?"})
    finally:
        server.close()

    assert result["evidence"] or result["related_pages"]
    titles = {item["title"] for item in result["evidence"] + result["related_pages"]}
    assert "Flash" in titles


def test_fast_query_is_compact_and_single_pass_ready() -> None:
    server = LeagueWikiServer(DATABASE)
    try:
        result = server.query(
            {
                "question": "What is Malphite's base attack damage at level 13?",
                "response_mode": "fast",
            }
        )
    finally:
        server.close()

    assert result["response_mode"] == "fast"
    assert result["evidence"]
    assert all(len(item.get("section_text", "")) <= 900 for item in result["evidence"])
    assert len(result["related_pages"]) <= 3


def test_query_resolves_redirects_and_item_data_modules() -> None:
    server = LeagueWikiServer(DATABASE)
    try:
        voidgrubs = server.query({"question": "What are the rules for Void Grubs?", "limit": 4})
        deathcap = server.query({"question": "What are Rabadon's Deathcap stats?", "limit": 4})
    finally:
        server.close()

    assert voidgrubs["evidence"][0]["title"] == "Voidgrub camp"
    assert voidgrubs["evidence"][0]["redirected_from"]
    assert deathcap["supporting_evidence"][0]["relationship"] == "structured_item_data_for"
    assert "ability power" in deathcap["supporting_evidence"][0]["page_text"].lower()


def test_mcp_initialize_and_query_call() -> None:
    server = LeagueWikiServer(DATABASE)
    try:
        initialized = dispatch(
            server,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
        )
        assert initialized is not None
        assert initialized["result"]["serverInfo"]["name"] == "scryglass-league-wiki"

        response = dispatch(
            server,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "league_wiki_query", "arguments": {"question": "What is Flash?"}},
            },
        )
        assert response is not None
        assert response["result"]["structuredContent"]["question"] == "What is Flash?"
    finally:
        server.close()


def test_sql_tool_is_read_only() -> None:
    server = LeagueWikiServer(DATABASE)
    try:
        result = server.sql({"sql": "SELECT COUNT(*) AS n FROM pages"})
        assert result["rows"][0]["n"] > 300_000
        try:
            server.sql({"sql": "DELETE FROM pages"})
        except ValueError as exc:
            assert "read-only" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("write SQL unexpectedly accepted")
    finally:
        server.close()
