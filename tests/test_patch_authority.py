from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from lol_kills.knowledge.patch_authority import (
    MechanicCell,
    PatchAuthorityError,
    PatchPacket,
    ReconciliationRecord,
    SourceBinding,
    build_cdragon_patch_packet,
)


SHA = "a" * 64


def _source() -> SourceBinding:
    return SourceBinding(
        source_id="fixture:client:26.13",
        source_kind="fixture",
        source_url="https://example.invalid/26.13.json",
        retrieved_at="2026-07-31T12:00:00Z",
        payload_sha256=SHA,
    )


def test_exact_cell_requires_source_and_round_trips(tmp_path: Path) -> None:
    packet = PatchPacket(
        patch="26.13",
        sources=(_source(),),
        cells=(
            MechanicCell(
                key="champion:Aatrox:Q:rank1",
                domain="champions",
                status="exact",
                source_ids=("fixture:client:26.13",),
                value={"base_damage": 70},
            ),
        ),
    )
    path = tmp_path / "packet.json"
    packet.write(path)
    loaded = PatchPacket.from_mapping(json.loads(path.read_text()))
    assert loaded.cell("champion:Aatrox:Q:rank1").executable is True
    assert loaded.payload_sha256 == packet.payload_sha256


def test_blocked_cell_cannot_smuggle_a_fallback_value() -> None:
    with pytest.raises(PatchAuthorityError, match="blocked cell"):
        PatchPacket(
            patch="26.13",
            sources=(),
            cells=(
                MechanicCell(
                    key="domain:champions",
                    domain="champions",
                    status="blocked",
                    value=0,
                    reason="missing source",
                ),
            ),
        )


def test_reconciled_cell_requires_review_record() -> None:
    source = _source()
    cell = MechanicCell(
        key="item:test:damage",
        domain="items",
        status="reconciled",
        source_ids=(source.source_id,),
        value=10,
    )
    with pytest.raises(PatchAuthorityError, match="reviewed record"):
        PatchPacket(patch="26.13", sources=(source,), cells=(cell,))
    packet = PatchPacket(
        patch="26.13",
        sources=(source,),
        cells=(cell,),
        reconciliations=(
            ReconciliationRecord(
                cell_key=cell.key,
                status="reviewed",
                source_ids=(source.source_id,),
                reason="two equivalent source representations",
                resolution="client payload wins after review",
            ),
        ),
    )
    assert packet.executable_cells == (cell,)


def test_semantic_only_and_blocked_cells_are_reported_separately() -> None:
    source = _source()
    semantic = MechanicCell(
        key="champion:test:ability",
        domain="champions.abilities",
        status="semantic_only",
        source_ids=(source.source_id,),
        value={"raw": True},
    )
    blocked = MechanicCell(
        key="domain:items",
        domain="items",
        status="blocked",
        reason="missing exact source",
    )
    packet = PatchPacket(patch="26.13", sources=(source,), cells=(semantic, blocked))
    assert packet.semantic_only_cells == (semantic,)
    assert packet.blocked_cells == (blocked,)


def test_cdragon_bridge_requires_exact_requested_patch_and_emits_only_supported_stats(tmp_path: Path) -> None:
    patch_dir = tmp_path / "26.13"
    raw_dir = patch_dir / "raw" / "game" / "data" / "characters" / "aatrox"
    raw_dir.mkdir(parents=True)
    bin_path = raw_dir / "aatrox.bin.json"
    bin_payload = b'{"raw":true}\n'
    bin_path.write_bytes(bin_payload)
    items_path = patch_dir / "raw" / "items.json"
    items_path.parent.mkdir(parents=True, exist_ok=True)
    items_payload = b"[]\n"
    items_path.write_bytes(items_payload)
    retrieved = "2026-07-31T12:00:00Z"
    index = {
        "schema_version": "scryglass:cdragon-patch-packet:v1",
        "patch": "26.13",
        "source_root": "https://raw.communitydragon.org/26.13/",
        "retrieved_at": retrieved,
        "champions": [
            {
                "id": 266,
                "name": "Aatrox",
                "bin_json_path": "raw/game/data/characters/aatrox/aatrox.bin.json",
                "mechanics": {
                    "raw_record_present": True,
                    "stats": {"base_health": 650.0, "health_per_level": 114.0},
                    "spells": [{"path": "Characters/Aatrox/Spells/AatroxQ"}],
                },
            }
        ],
    }
    index_path = patch_dir / "mechanics-index.json"
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")
    manifest_unsigned = {
        "schema_version": "scryglass:cdragon-patch-packet:v1",
        "patch": "26.13",
        "source": "CommunityDragon",
        "source_root": "https://raw.communitydragon.org/26.13/",
        "retrieved_at": retrieved,
        "exact_patch_source": True,
        "files": [
            {"path": "raw/game/data/characters/aatrox/aatrox.bin.json", "url": "https://example.invalid/aatrox.bin.json", "sha256": hashlib.sha256(bin_payload).hexdigest()},
            {"path": "raw/items.json", "url": "https://example.invalid/items.json", "sha256": hashlib.sha256(items_payload).hexdigest()},
        ],
    }
    manifest = dict(manifest_unsigned)
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest_unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest_path = patch_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    packet = build_cdragon_patch_packet(index_path, expected_patch="26.13", manifest_path=manifest_path)
    assert packet.cell("champion:266:stat:base_health").executable
    assert packet.cell("champion:266:spell:Characters/Aatrox/Spells/AatroxQ").status == "semantic_only"
    assert packet.cell("domain:items:raw").status == "semantic_only"
    with pytest.raises(PatchAuthorityError, match="patch mismatch"):
        build_cdragon_patch_packet(index_path, expected_patch="26.14", manifest_path=manifest_path)
