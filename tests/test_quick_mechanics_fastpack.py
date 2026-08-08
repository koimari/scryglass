from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lol_kills.knowledge.quick_mechanics_fastpack import (
    SCHEMA_VERSION,
    compile_fastpack,
    load_fastpack,
    write_fastpack,
)


INDEX = (
    Path(__file__).resolve().parents[1]
    / "data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json"
)


def _champion(pack: dict, alias: str) -> dict:
    return next(item for item in pack["champions"].values() if item["alias"] == alias)


def test_compile_current_patch_has_exact_fast_path_values() -> None:
    pack = compile_fastpack(INDEX)

    assert pack["schema_version"] == SCHEMA_VERSION
    assert pack["patch"] == "26.15"
    assert pack["client_patch"] == "16.15"
    assert pack["source_hashes"]["index_sha256"]
    assert len(pack["source_hashes"]["champion_bins"]) == 233
    assert pack["level_key_type"] == "string"

    malphite = _champion(pack, "Malphite")
    assert malphite["resource_type"] == "mana"
    assert malphite["levels"]["13"]["mp5"] == pytest.approx(13.32, abs=0.01)
    assert malphite["levels"]["13"]["magic_resist"] == pytest.approx(50.45, abs=0.01)

    zaahen = _champion(pack, "Zaahen")
    assert zaahen["resource_type"] == "mana"
    assert zaahen["levels"]["13"]["mp5"] == pytest.approx(16.36, abs=0.01)

    renekton = _champion(pack, "Renekton")
    assert renekton["resource_type"] != "mana"
    assert renekton["levels"]["13"]["mp5"] is None

    tristana = _champion(pack, "Tristana")
    assert tristana["levels"]["5"]["attack_damage"] == pytest.approx(70.506, abs=0.001)


def test_supplements_are_explicit_and_assumption_labelled() -> None:
    pack = compile_fastpack(INDEX)
    gromp = pack["monsters"]["gromp"]
    assert gromp["levels"]["5"]["max_health"] == pytest.approx(2870)
    assert gromp["levels"]["5"]["armor"] == pytest.approx(42)
    assert gromp["provenance"]["kind"] == "supplement"
    assert gromp["provenance"]["revision_id"] == 4016297
    assert gromp["provenance"]["content_sha256"]

    grubs = pack["objectives"]["void_grubs"]
    assert grubs["count"] == 3
    assert grubs["gold"]["local"] == 90
    assert grubs["gold"]["global"] == 0
    assert grubs["assumptions"]
    assert grubs["provenance"]["kind"] == "supplement"
    assert grubs["provenance"]["revision_id"] == 4015021
    assert grubs["provenance"]["content_sha256"]


def test_write_and_load_round_trip_is_network_free(tmp_path: Path) -> None:
    output = tmp_path / "quick-mechanics.fastpack.json"
    expected = write_fastpack(INDEX, output)
    loaded = load_fastpack(output)
    assert loaded == expected
    assert loaded["aliases"]["malphite"] == "54"
    assert loaded["aliases"]["voidgrubs"] == "objective:void_grubs"


def test_compile_fails_closed_on_tampered_source_hash(tmp_path: Path) -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    entry = dict(index["champions"][0])
    source_bin = INDEX.parent / entry["bin_json_path"]
    entry["bin_json_path"] = "annie.bin.json"
    shutil.copy2(source_bin, tmp_path / entry["bin_json_path"])
    entry["bin_sha256"] = "0" * 64
    index["champions"] = [entry]
    tampered = tmp_path / "mechanics-index.json"
    tampered.write_text(json.dumps(index), encoding="utf-8")
    # The referenced bin path is intentionally relative to the index.  The
    # missing directory is itself a fail-closed result, before any fallback.
    with pytest.raises((FileNotFoundError, ValueError)):
        compile_fastpack(tampered)


def test_supplements_do_not_carry_into_an_unvalidated_patch(tmp_path: Path) -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    entry = dict(index["champions"][0])
    source_bin = INDEX.parent / entry["bin_json_path"]
    entry["bin_json_path"] = "annie.bin.json"
    shutil.copy2(source_bin, tmp_path / entry["bin_json_path"])
    index["champions"] = [entry]
    index["patch"] = "26.14"
    historical = tmp_path / "mechanics-index.json"
    historical.write_text(json.dumps(index), encoding="utf-8")

    pack = compile_fastpack(historical)
    assert pack["supplements"]["status"] == "unavailable"
    assert pack["monsters"] == {}
    assert pack["objectives"] == {}
