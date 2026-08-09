from __future__ import annotations

import copy
import hashlib
import json
import pickle
from pathlib import Path

import pytest

import lol_kills.v2.champions.id_crosswalk as id_crosswalk_module
from lol_kills.v2.champions.id_crosswalk import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_METADATA_PATH,
    EXPECTED_ACTION_ORDER_COMPLETE,
    EXPECTED_ACTION_ORDER_MISSING,
    EXPECTED_MAPS,
    EXPECTED_METADATA_ENTRIES,
    EXPECTED_METADATA_RAW_SHA256,
    EXPECTED_OE_NAMES,
    EXPECTED_PREFLIGHT_PAYLOAD_SHA256,
    EXPECTED_ROLE_SLOTS,
    METADATA_VERSION,
    ChampionIdCrosswalkError,
    canonical_bytes,
    canonical_sha256,
    load_and_replay_artifact,
    load_metadata_bytes,
    normalize_name,
    require_exact_competition_patch_authority,
    resolve_champion_id,
    validate_artifact,
)


def _duplicate_first_numeric_id(data: dict) -> None:
    first, second = list(data)[:2]
    data[first]["key"] = data[second]["key"]


@pytest.fixture(scope="module")
def artifact() -> dict:
    return load_and_replay_artifact(DEFAULT_ARTIFACT_PATH)


def _rehash(payload: dict) -> dict:
    payload["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )
    return payload


def test_official_source_bytes_schema_and_base_ids_are_exact() -> None:
    raw = DEFAULT_METADATA_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_METADATA_RAW_SHA256
    metadata = load_metadata_bytes(raw)
    assert len(metadata) == EXPECTED_METADATA_ENTRIES
    assert len({record["numeric_id"] for record in metadata.values()}) == 173
    assert len({record["internal_id"] for record in metadata.values()}) == 173
    assert all(0 < record["numeric_id"] < 1000 for record in metadata.values())
    source = json.loads(raw)
    assert source["version"] == METADATA_VERSION
    assert all(key == record["id"] for key, record in source["data"].items())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.pop(next(iter(data))), "exactly 173"),
        (_duplicate_first_numeric_id, "duplicate Riot champion numeric ID"),
        (
            lambda data: data[next(iter(data))].update({"key": "0"}),
            "must be positive",
        ),
        (
            lambda data: data[next(iter(data))].update({"key": "1001"}),
            "mode-variant",
        ),
    ],
)
def test_metadata_duplicate_nonpositive_mode_and_count_fail_closed(
    mutation, message: str
) -> None:
    payload = json.loads(DEFAULT_METADATA_PATH.read_bytes())
    mutation(payload["data"])
    with pytest.raises(ChampionIdCrosswalkError, match=message):
        load_metadata_bytes(canonical_bytes(payload))


def test_metadata_normalized_display_collision_fails_closed() -> None:
    payload = json.loads(DEFAULT_METADATA_PATH.read_bytes())
    first, second = list(payload["data"])[:2]
    payload["data"][first]["name"] = payload["data"][second]["name"]
    with pytest.raises(ChampionIdCrosswalkError, match="duplicate normalized"):
        load_metadata_bytes(canonical_bytes(payload))


def test_exact_empirical_coverage_and_stable_ids(artifact: dict) -> None:
    validate_artifact(artifact)
    assert artifact["preflight"]["payload_sha256"] == EXPECTED_PREFLIGHT_PAYLOAD_SHA256
    assert artifact["metadata"]["version"] == METADATA_VERSION
    assert artifact["coverage"] == {
        "preflight_maps": EXPECTED_MAPS,
        "role_labeled_slots": EXPECTED_ROLE_SLOTS,
        "maps_with_ten_resolved_role_slots": EXPECTED_MAPS,
        "distinct_oe_names": EXPECTED_OE_NAMES,
        "distinct_oe_names_resolved": EXPECTED_OE_NAMES,
        "unresolved_oe_names": [],
    }
    assert len(artifact["entries"]) == EXPECTED_OE_NAMES
    assert len({entry["oe_name"] for entry in artifact["entries"]}) == EXPECTED_OE_NAMES
    assert len({entry["stable_champion_id"] for entry in artifact["entries"]}) == EXPECTED_OE_NAMES
    assert all(
        entry["stable_champion_id"] == f"riot:champion:{entry['riot_numeric_id']}"
        for entry in artifact["entries"]
    )


def test_source_internal_display_aliases_are_exact_not_fuzzy(artifact: dict) -> None:
    wukong = resolve_champion_id(artifact, "Wukong")
    assert wukong == resolve_champion_id(artifact, "MonkeyKing")
    nunu = resolve_champion_id(artifact, "Nunu & Willump")
    assert nunu == resolve_champion_id(artifact, "Nunu")
    assert normalize_name("  Kai’Sa\t") == "kai'sa"
    assert resolve_champion_id(artifact, "  Kai’Sa\t") == resolve_champion_id(
        artifact, "Kai'Sa"
    )


def test_plain_or_reconstructed_mapping_has_no_resolution_authority(
    artifact: dict,
) -> None:
    reconstructed = copy.deepcopy(artifact)
    validate_artifact(reconstructed)
    with pytest.raises(ChampionIdCrosswalkError, match="loader-issued"):
        resolve_champion_id(reconstructed, "Aatrox")

    unissued_shell = type(artifact)()
    with pytest.raises(ChampionIdCrosswalkError, match="loader-issued"):
        resolve_champion_id(unissued_shell, "Aatrox")


def test_no_caller_reachable_issuer_or_live_backing_accessor(
    artifact: dict,
) -> None:
    runtime_globals = resolve_champion_id.__globals__
    assert "_issue_verified_crosswalk" not in runtime_globals
    assert "_require_verified_crosswalk" not in runtime_globals

    exposed_entries = artifact["entries"]
    aatrox = next(
        entry for entry in exposed_entries if entry["oe_name"] == "Aatrox"
    )
    aatrox["riot_numeric_id"] = 12
    aatrox["stable_champion_id"] = "riot:champion:12"
    assert resolve_champion_id(artifact, "Aatrox") == "riot:champion:266"


def test_copy_deepcopy_pickle_and_unissued_shell_cannot_gain_authority(
    artifact: dict,
) -> None:
    for reconstructed in (copy.copy(artifact), copy.deepcopy(artifact)):
        assert isinstance(reconstructed, dict)
        with pytest.raises(ChampionIdCrosswalkError, match="loader-issued"):
            resolve_champion_id(reconstructed, "Aatrox")
    with pytest.raises(ChampionIdCrosswalkError, match="cannot be serialized"):
        pickle.dumps(artifact)
    with pytest.raises(ChampionIdCrosswalkError, match="loader-issued"):
        resolve_champion_id(type(artifact)(), "Aatrox")


def test_loader_issued_capability_fails_closed_after_generator_drift(
    artifact: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_identity = id_crosswalk_module._generator_identity()
    monkeypatch.setattr(
        id_crosswalk_module,
        "_generator_identity",
        lambda: {**real_identity, "version": "caller-forged-version"},
    )
    with pytest.raises(ChampionIdCrosswalkError, match="executable generator"):
        resolve_champion_id(artifact, "Aatrox")


def test_resolution_rechecks_every_source_byte_identity(
    artifact: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_raw_sha256 = id_crosswalk_module.raw_sha256
    observed: list[Path] = []

    def traced(path: Path) -> str:
        observed.append(path.resolve())
        return real_raw_sha256(path)

    monkeypatch.setattr(id_crosswalk_module, "raw_sha256", traced)
    assert resolve_champion_id(artifact, "Aatrox") == "riot:champion:266"
    expected = {
        id_crosswalk_module.DEFAULT_METADATA_PATH.resolve(),
        id_crosswalk_module.DEFAULT_PREFLIGHT_PATH.resolve(),
        id_crosswalk_module.DEFAULT_MAPS_PATH.resolve(),
        id_crosswalk_module.DEFAULT_PLAYERS_PATH.resolve(),
    }
    assert expected.issubset(set(observed))


def test_loader_issued_capability_fails_closed_after_source_byte_drift(
    artifact: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_raw_sha256 = id_crosswalk_module.raw_sha256
    metadata_path = id_crosswalk_module.DEFAULT_METADATA_PATH.resolve()

    def changed(path: Path) -> str:
        if path.resolve() == metadata_path:
            return "0" * 64
        return real_raw_sha256(path)

    monkeypatch.setattr(id_crosswalk_module, "raw_sha256", changed)
    with pytest.raises(ChampionIdCrosswalkError, match="Riot metadata source bytes"):
        resolve_champion_id(artifact, "Aatrox")


def test_loader_issued_verified_crosswalk_resolves(artifact: dict) -> None:
    assert resolve_champion_id(artifact, "Aatrox") == "riot:champion:266"


def test_rehashed_noncolliding_id_substitution_has_no_resolution_authority(
    artifact: dict,
) -> None:
    mutated = copy.deepcopy(artifact)
    aatrox = next(entry for entry in mutated["entries"] if entry["oe_name"] == "Aatrox")
    assert aatrox["riot_numeric_id"] == 266
    assert all(entry["riot_numeric_id"] != 11 for entry in mutated["entries"])
    aatrox["riot_numeric_id"] = 11
    aatrox["stable_champion_id"] = "riot:champion:11"
    _rehash(mutated)
    validate_artifact(mutated)
    with pytest.raises(ChampionIdCrosswalkError, match="loader-issued"):
        resolve_champion_id(mutated, "Aatrox")


def test_rehashed_invented_alias_has_no_resolution_authority(
    artifact: dict,
) -> None:
    mutated = copy.deepcopy(artifact)
    mutated["explicit_aliases"].append(
        {
            "input": "Dr Mundo",
            "normalized_input": "dr mundo",
            "riot_internal_id": "DrMundo",
            "basis": "Riot Data Dragon internal ID differs from display name",
        }
    )
    _rehash(mutated)
    validate_artifact(mutated)
    with pytest.raises(ChampionIdCrosswalkError, match="loader-issued"):
        resolve_champion_id(mutated, "Dr Mundo")


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        None,
        "Definitely Future Champion",
        "Kai",
        "Nunu and Willump",
        "Dr Mundo",
    ],
)
def test_unknown_future_blank_partial_and_punctuation_variants_fail_closed(
    artifact: dict, value: object
) -> None:
    with pytest.raises(ChampionIdCrosswalkError):
        resolve_champion_id(artifact, value)


def test_action_order_missingness_is_separate_from_role_identity(
    artifact: dict,
) -> None:
    availability = artifact["action_order_availability"]
    assert availability["complete_maps"] == EXPECTED_ACTION_ORDER_COMPLETE
    assert (
        availability["maps_missing_one_or_more_action_order_fields"]
        == EXPECTED_ACTION_ORDER_MISSING
    )
    assert availability["complete_maps"] + availability[
        "maps_missing_one_or_more_action_order_fields"
    ] == EXPECTED_MAPS
    assert availability["identity_coverage_affected"] is False
    assert artifact["coverage"]["maps_with_ten_resolved_role_slots"] == EXPECTED_MAPS


def test_metadata_version_is_not_a_competition_patch_and_float_is_no_authority(
    artifact: dict,
) -> None:
    patch = artifact["competition_patch"]
    assert patch["namespace"] == "Oracle's Elixir source patch token"
    assert patch["competition_patch_namespace"] == "Oracle's Elixir source patch token"
    assert patch["official_mapping_status"] == "none"
    assert patch["patch_mapping"] == "none"
    assert artifact["metadata"]["metadata_version"] == METADATA_VERSION
    assert patch["metadata_version_is_competition_patch"] is False
    assert patch["exact_patch_authority"] is False
    with pytest.raises(ChampionIdCrosswalkError, match="does not confer"):
        require_exact_competition_patch_authority(artifact, 16.14)
    with pytest.raises(ChampionIdCrosswalkError, match="does not confer"):
        require_exact_competition_patch_authority(artifact, "26.14")


def test_caller_rehash_cannot_make_mutation_replayable(
    artifact: dict, tmp_path: Path
) -> None:
    mutated = copy.deepcopy(artifact)
    mutated["metadata"]["documentation_url"] = "https://example.invalid/caller-rehash"
    _rehash(mutated)
    validate_artifact(mutated)
    path = tmp_path / "caller-rehashed.json"
    path.write_bytes(canonical_bytes(mutated))
    with pytest.raises(ChampionIdCrosswalkError, match="source-backed replay"):
        load_and_replay_artifact(path)


def test_crosswalk_stable_id_collision_is_rejected(artifact: dict) -> None:
    mutated = copy.deepcopy(artifact)
    mutated["entries"][0]["riot_numeric_id"] = mutated["entries"][1]["riot_numeric_id"]
    mutated["entries"][0]["stable_champion_id"] = mutated["entries"][1][
        "stable_champion_id"
    ]
    _rehash(mutated)
    with pytest.raises(ChampionIdCrosswalkError, match="entry collision"):
        validate_artifact(mutated)


def test_generator_code_identity_drift_is_rejected(
    artifact: dict, tmp_path: Path
) -> None:
    mutated = copy.deepcopy(artifact)
    mutated["generator"]["executable_dependency_boundary"][0]["raw_sha256"] = "0" * 64
    _rehash(mutated)
    path = tmp_path / "wrong-code.json"
    path.write_bytes(canonical_bytes(mutated))
    with pytest.raises(ChampionIdCrosswalkError, match="executable module"):
        load_and_replay_artifact(path)


def test_pinned_source_byte_drift_is_rejected(
    artifact: dict, tmp_path: Path
) -> None:
    changed_metadata = tmp_path / "changed-metadata.json"
    changed_metadata.write_bytes(DEFAULT_METADATA_PATH.read_bytes() + b"\n")
    mutated = copy.deepcopy(artifact)
    mutated["metadata"]["locator"] = changed_metadata.as_posix()
    _rehash(mutated)
    path = tmp_path / "wrong-source.json"
    path.write_bytes(canonical_bytes(mutated))
    with pytest.raises(ChampionIdCrosswalkError, match="pinned source bytes changed"):
        load_and_replay_artifact(path)


def test_publication_and_model_authority_ceiling(artifact: dict) -> None:
    assert artifact["publication_decision"] == "private_pending_review"
    assert artifact["development_only"] is True
    assert artifact["authority"] == {
        "authorizes_prediction": False,
        "authorizes_model_selection": False,
        "authorizes_publication": False,
        "content_addressing_confers_authority": False,
    }
    assert artifact["claim_scope"]["vocabulary_identity_only"] is True
    assert artifact["claim_scope"]["effect_estimation"] is False
