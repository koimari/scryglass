from __future__ import annotations

from lol_kills.knowledge.cdragon_patch_packet import (
    CDragonClient,
    _extract_mechanics,
    _extract_stats,
    _client_patch_for,
    _requested_champions,
    capture_patch_matrix,
)


def test_public_2026_patch_maps_to_same_minor_client_namespace() -> None:
    assert _client_patch_for("26.01") == "16.1"
    assert _client_patch_for("26.13") == "16.13"
    assert _client_patch_for("16.13") == "16.13"


def test_client_keeps_public_label_separate_from_source_namespace() -> None:
    client = CDragonClient("26.13")
    assert client.patch == "26.13"
    assert client.source_patch == "16.13"


def test_extract_stats_reads_client_character_record() -> None:
    record = {
        "baseHPModifiable": {"baseValue": 650.0},
        "hpPerLevelModifiable": {"baseValue": 114.0},
        "baseDamageModifiable": {"baseValue": 60.0},
        "damagePerLevelModifiable": {"baseValue": 5.0},
        "baseArmorModifiable": {"baseValue": 38.0},
        "armorPerLevelModifiable": {"baseValue": 4.8},
        "baseMR": {"baseValue": 32.0},
        "baseMoveSpeedModifiable": {"baseValue": 345.0},
        "attackRangeModifiable": {"baseValue": 175.0},
        "attackSpeedModifiable": {"baseValue": 0.651},
        "attackSpeedPerLevelModifiable": {"baseValue": 2.5},
    }
    stats = _extract_stats(record)
    assert stats["base_health"] == 650.0
    assert stats["attack_damage_per_level"] == 5.0


def test_extract_mechanics_preserves_typed_data_values_and_formulas() -> None:
    payload = {
        "Characters/Test/CharacterRecords/Root": {
            "__type": "CharacterRecord",
            "mCharacterName": "Test",
            "characterToolData": {"championId": 999},
            "mAbilities": ["Characters/Test/Spells/TestQAbility"],
            "spellNames": ["TestQAbility/TestQ"],
        },
        "Characters/Test/Spells/TestQAbility": {
            "__type": "AbilityObject",
            "mChildSpells": ["Characters/Test/Spells/TestQAbility/TestQ"],
        },
        "Characters/Test/Spells/TestQAbility/TestQ": {
            "__type": "SpellObject",
            "ObjectName": "TestQ",
            "mSpell": {
                "DataValues": [{"name": "Base", "values": [10, 20, 30]}],
                "mSpellCalculations": {
                    "Damage": {"mFormulaParts": [{"mDataValue": "Base"}]}
                },
                "cooldownTime": [10, 9, 8],
            },
        },
    }
    mechanics = _extract_mechanics(payload)
    assert mechanics["stats"] == {}
    assert mechanics["spells"][0]["data_values"] == [
        {"name": "Base", "values": [10, 20, 30]}
    ]
    assert mechanics["spells"][0]["spell_calculations"]["Damage"]


def test_requested_champions_filters_by_id_name_or_alias() -> None:
    summary = [
        {"id": 1, "name": "Annie", "alias": "Annie"},
        {"id": 2, "name": "Olaf", "alias": "Olaf"},
        {"id": -1, "name": "None", "alias": "None"},
    ]
    assert [row["name"] for row in _requested_champions(summary, {"olaf"}, None)] == ["Olaf"]
    assert [row["name"] for row in _requested_champions(summary, {"1"}, None)] == ["Annie"]


def test_patch_matrix_preserves_exact_probe_status_without_fallback(tmp_path, monkeypatch) -> None:
    def probe(self, relative_path):
        return {
            "url": f"https://example.invalid/{self.patch}/{relative_path}",
            "status_code": 404 if self.patch == "26.13" else 200,
            "available": self.patch != "26.13",
            "sha256": "a" * 64,
            "bytes": 1,
        }

    monkeypatch.setattr(CDragonClient, "probe", probe)
    result = capture_patch_matrix(
        ["26.13", "26.12"],
        tmp_path,
        probe_only=True,
        delay=0,
    )
    rows = {row["patch"]: row for row in result["patches"]}
    assert rows["26.13"]["status"] == "blocked"
    assert rows["26.13"]["client_patch"] == "16.13"
    assert rows["26.13"]["exact_patch_source"] is False
    assert rows["26.12"]["status"] == "available_probe_only"
    assert (tmp_path / "26.13" / "probe.json").exists()
