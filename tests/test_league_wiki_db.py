from __future__ import annotations

import json
from pathlib import Path

from lol_kills.knowledge.league_wiki_db import (
    DB_SCHEMA_VERSION,
    build_database,
    connect_database,
    get_page,
    get_sections,
    search_pages,
    search_sections,
    status,
)
from lol_kills.knowledge.league_wiki_vault import (
    SCHEMA_VERSION,
    _page_document,
    _sha256_bytes,
)


def _fixture_vault(root: Path) -> Path:
    vault = root / "vault"
    page = {"pageid": 42, "ns": 0, "title": "Minion"}
    content = "== Movement speed ==\nMinions have 350 movement speed.\n== Waves ==\nWaves spawn every 30 seconds.\n"
    revision = {
        "revision_id": 99,
        "parent_revision_id": 98,
        "revision_timestamp": "2026-07-31T00:00:00Z",
        "api_sha1": "api",
        "content_sha256": _sha256_bytes(content.encode()),
        "content_model": "wikitext",
        "content_format": "text/x-wiki",
    }
    document = _page_document(page, revision, content)
    document_path = vault / "pages" / "ns-0" / "Minion.md"
    document_path.parent.mkdir(parents=True)
    document_path.write_bytes(document)
    catalog = {
        "pageid": 42,
        "namespace": 0,
        "title": "Minion",
        "source_url": "https://example.test/Minion",
        "retrieved_at": "2026-07-31T00:00:00Z",
    }
    latest = {
        "page_id": 42,
        "namespace": 0,
        "title": "Minion",
        "source_url": "https://example.test/Minion",
        "retrieved_at": "2026-07-31T00:00:00Z",
        **revision,
        "document_sha256": _sha256_bytes(document),
        "document_path": "pages/ns-0/Minion.md",
        "revision_path": "revisions/ns-0/Minion/rev-99.md",
    }
    vault.mkdir(exist_ok=True)
    catalog_bytes = (json.dumps(catalog, sort_keys=True) + "\n").encode()
    (vault / "catalog.jsonl").write_bytes(catalog_bytes)
    (vault / "latest.jsonl").write_text(json.dumps(latest, sort_keys=True) + "\n")
    (vault / "inventory-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "source_api": "https://example.test/api",
                "catalog_sha256": _sha256_bytes(catalog_bytes),
                "page_count": 1,
            }
        )
    )
    (vault / "snapshot-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "complete": True,
                "latest_jsonl_sha256": _sha256_bytes((vault / "latest.jsonl").read_bytes()),
            }
        )
    )
    return vault


def test_build_and_query_database_preserves_revision_evidence(tmp_path: Path) -> None:
    vault = _fixture_vault(tmp_path)
    database = tmp_path / "league-wiki.sqlite3"
    result = build_database(vault, database)
    assert result["schema_version"] == DB_SCHEMA_VERSION
    assert result["catalog_page_count"] == 1
    assert result["text_page_count"] == 1

    connection = connect_database(database)
    try:
        page = get_page(connection, title="Minion", namespace=0)
        assert page is not None
        assert page["revision_id"] == 99
        assert "350 movement speed" in page["wikitext"]
        assert search_pages(connection, "movement speed")[0]["page_id"] == 42
        section_hits = search_sections(connection, "350 movement speed")
        assert section_hits[0]["heading"] == "Movement speed"
        assert get_sections(connection, 42, heading="waves")[0]["heading"] == "Waves"
        assert status(connection)["counts"] == {
            "pages": 1,
            "text_pages": 1,
            "cataloged_pages": 1,
            "sections": 2,
        }
    finally:
        connection.close()
