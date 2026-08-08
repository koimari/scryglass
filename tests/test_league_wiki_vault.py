from __future__ import annotations

import json
from pathlib import Path

from lol_kills.knowledge.league_wiki_vault import (
    SCHEMA_VERSION,
    _sha256_bytes,
    _page_document,
    _parse_namespaces,
    _title_path,
    validate,
)


def test_default_and_full_namespace_selection() -> None:
    assert _parse_namespaces(None, False) == (0,)
    assert _parse_namespaces("14,0,14", False) == (0, 14)
    assert 0 in _parse_namespaces(None, True)


def test_title_path_is_safe_and_obsidian_readable(tmp_path: Path) -> None:
    path = _title_path(tmp_path, 0, "Aatrox/History")
    assert path == tmp_path / "pages" / "ns-0" / "Aatrox" / "History.md"
    dotted = _title_path(tmp_path, 0, "A.D.M.I.N. (Teamfight Tactics)")
    assert dotted.name == "A.D.M.I.N.%20(Teamfight%20Tactics).md"

    try:
        _title_path(tmp_path, 0, "../escape")
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal title must fail closed")


def test_page_document_contains_revision_lineage_and_exact_content_hash() -> None:
    content = "== Abilities ==\n{{Data Aatrox/Q|Ability}}\n"
    page = {"pageid": 1, "ns": 0, "title": "Aatrox"}
    revision = {
        "revision_id": 9,
        "parent_revision_id": 8,
        "revision_timestamp": "2026-07-31T00:00:00Z",
        "api_sha1": None,
        "content_sha256": "placeholder",
        "content_model": "wikitext",
        "content_format": "text/x-wiki",
    }
    document = _page_document(page, revision, content).decode("utf-8")
    assert f"schema_version: {json.dumps(SCHEMA_VERSION)}" in document
    assert "revision_id: 9" in document
    assert "Source-preserving wikitext begins below" in document
    assert content.rstrip() in document


def test_validate_requires_text_bodies_but_allows_file_metadata_only(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    catalog = [
        {"pageid": 1, "namespace": 0, "title": "Aatrox", "source_url": "https://example/Aatrox", "retrieved_at": "2026-07-31T00:00:00Z"},
        {"pageid": 2, "namespace": 6, "title": "Aatrox.png", "source_url": "https://example/Aatrox.png", "retrieved_at": "2026-07-31T00:00:00Z"},
    ]
    catalog_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in catalog).encode()
    (vault / "catalog.jsonl").write_bytes(catalog_bytes)
    (vault / "inventory-manifest.json").write_text(
        json.dumps({"catalog_sha256": _sha256_bytes(catalog_bytes)}), encoding="utf-8"
    )
    document = vault / "pages" / "ns-0" / "Aatrox.md"
    document.parent.mkdir(parents=True)
    document.write_text("source", encoding="utf-8")
    latest = {
        "namespace": 0,
        "title": "Aatrox",
        "document_path": "pages/ns-0/Aatrox.md",
        "document_sha256": _sha256_bytes(document.read_bytes()),
        "content_sha256": "a" * 64,
    }
    (vault / "latest.jsonl").write_text(json.dumps(latest) + "\n", encoding="utf-8")
    result = validate(vault)
    assert result["complete"] is True
    assert result["file_namespace_metadata_only"] is True
