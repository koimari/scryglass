from __future__ import annotations

import hashlib
import json
import os
import shutil

import pytest

from lol_kills.v2.data import g1_draft_features as features


def _base() -> dict[str, object]:
    return {
        "source_game_id": "g1", "partition": "TRAIN", "source_local_event_start": "2025-01-01T00:00:00",
        "game_side": {"blue_team_id": "tb", "red_team_id": "tr"},
        "observed_lineups": [
            {"observed_game_side": "blue", "team_id": "tb", "player_ids_by_role": {role: f"b-{role}" for role in features.ROLES}},
            {"observed_game_side": "red", "team_id": "tr", "player_ids_by_role": {role: f"r-{role}" for role in features.ROLES}},
        ],
    }


def _source() -> list[dict[str, str]]:
    result = []
    for side, team, prefix in (("Blue", "tb", "b"), ("Red", "tr", "r")):
        for role, champion in zip(("top", "jng", "mid", "bot", "sup"), ("Aatrox", "Ahri", "Akali", "Alistar", "Amumu")):
            result.append({"gameid": "g1", "side": side, "position": role, "playerid": f"{prefix}-{features.ROLE_ALIASES[role]}", "teamid": team, "champion": champion if side == "Blue" else champion + "R"})
    return result


def _champions() -> dict[str, str]:
    return {row["champion"]: f"riot:{index}" for index, row in enumerate(_source())}



def test_real_materialization_and_verifier() -> None:
    manifest = features.build()
    checked = features.verify(expected_manifest_sha256=manifest["manifest_sha256"])
    assert checked["coverage"] == {"accepted_map_count": 1226, "feature_row_count": 1226, "pick_count": 12260, "identity_unavailable_map_count": 0}
    assert checked["source"]["selected_projection_row_count"] == 12260


def test_verifier_requires_external_pin_and_rejects_rehashed_relaxed_claim(tmp_path, monkeypatch) -> None:
    manifest = features.build()
    with pytest.raises(TypeError):
        features.verify()
    rows, altered = tmp_path / "rows", tmp_path / "manifest"
    shutil.copyfile(features.OUTPUT_ROWS, rows)
    payload = json.loads(features.OUTPUT_MANIFEST.read_text())
    payload["claim_ceiling"]["prediction"] = True
    payload.pop("manifest_sha256")
    payload["manifest_sha256"] = features.sha256(payload)
    altered.write_bytes(features.canonical_bytes(payload) + b"\n")
    monkeypatch.setattr(features, "_safe_file", lambda path, **_: path)
    with pytest.raises(features.G1DraftFeatureError, match="claim or final boundary"):
        features.verify(rows_path=rows, manifest_path=altered, expected_manifest_sha256=payload["manifest_sha256"])


def test_rehashed_crosswalk_substitution_fails_immutable_pin(tmp_path, monkeypatch) -> None:
    copy = tmp_path / "crosswalk.json"
    payload = json.loads(features.CROSSWALK.read_text())
    payload["entries"][0]["stable_champion_id"] = "riot:champion:999"
    payload["artifact_sha256"] = features.sha256({key: value for key, value in payload.items() if key != "artifact_sha256"})
    copy.write_bytes(features.canonical_bytes(payload))
    monkeypatch.setattr(features, "_safe_file", lambda path, **_: path)
    with pytest.raises(features.G1DraftFeatureError, match="immutable pin"):
        features._crosswalk_table(["Aatrox"], copy)


def test_authenticated_crosswalk_types_unknown_but_rejects_recognized_resolution_failure(monkeypatch) -> None:
    table, _ = features._crosswalk_table(["Aatrox", "Future Champion"])
    assert table["Aatrox"].startswith("riot:champion:") and "Future Champion" not in table
    original = features.id_crosswalk.resolve_champion_id
    def forged_resolver(capability, name):
        if name == "Aatrox":
            raise features.id_crosswalk.ChampionIdCrosswalkError("stale capability")
        return original(capability, name)
    monkeypatch.setattr(features.id_crosswalk, "resolve_champion_id", forged_resolver)
    with pytest.raises(features.G1DraftFeatureError, match="recognized-name resolution failed"):
        features._crosswalk_table(["Aatrox"])


def test_target_key_is_inaccessible_to_transform() -> None:
    class PoisonedBase(dict):
        def __getitem__(self, key):
            if key == "target":
                raise AssertionError("target must never be read")
            return super().__getitem__(key)

    rows, projection = features.materialize_from_projection(base_rows=[PoisonedBase(_base())], source_rows=_source(), champion_table=_champions())
    assert len(rows) == 1 and len(rows[0]["picks"]) == 10 and len(projection) == 10
    assert "target" not in rows[0]


@pytest.mark.parametrize("mutation,code", [
    (lambda rows: rows.__setitem__(0, {**rows[0], "champion": "Unknown"}), "TYPED_UNAVAILABLE_CHAMPION_IDENTITY"),
    (lambda rows: rows.__setitem__(0, {**rows[0], "position": "coach"}), "illegal source role"),
    (lambda rows: rows.__setitem__(1, {**rows[1], "playerid": rows[0]["playerid"]}), "source participant differs"),
    (lambda rows: rows.__setitem__(1, {**rows[1], "champion": rows[0]["champion"]}), "global champion duplication"),
])
def test_hostile_source_fields_fail_closed_or_type_unavailable(mutation, code) -> None:
    rows = _source(); mutation(rows)
    if code == "TYPED_UNAVAILABLE_CHAMPION_IDENTITY":
        output, _ = features.materialize_from_projection(base_rows=[_base()], source_rows=rows, champion_table=_champions())
        assert output[0]["availability"] == code
    else:
        with pytest.raises(features.G1DraftFeatureError, match=code):
            features.materialize_from_projection(base_rows=[_base()], source_rows=rows, champion_table=_champions())


def test_projection_is_champion_sensitive_and_order_deterministic() -> None:
    first = features._projection(_source())
    second = features._projection(reversed(_source()))
    assert first == second
    changed = [dict(row) for row in _source()]; changed[0]["champion"] = "Changed"
    assert features.sha256(first) != features.sha256(features._projection(changed))


def test_safe_writer_rejects_symlink_and_hardlink(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(features, "ROOT", tmp_path)
    safe = tmp_path / "safe"; safe.mkdir()
    target = safe / "target"; target.write_bytes(b"x")
    linked = safe / "linked"; linked.symlink_to(target)
    with pytest.raises(features.G1DraftFeatureError, match="unsafe output leaf"):
        features._safe_write_many(((linked, b"new"),))
    hard = safe / "hard"; os.link(target, hard)
    with pytest.raises(features.G1DraftFeatureError, match="unsafe output leaf"):
        features._safe_write_many(((hard, b"new"),))


def test_safe_writer_does_not_replace_when_staging_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(features, "ROOT", tmp_path)
    out = tmp_path / "out"; out.mkdir()
    first, second = out / "first.json", out / "second.json"
    original = features.tempfile.mkstemp
    calls = {"count": 0}
    def interrupted(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("simulated interruption")
        return original(*args, **kwargs)
    monkeypatch.setattr(features.tempfile, "mkstemp", interrupted)
    with pytest.raises(OSError):
        features._safe_write_many(((first, b"new"), (second, b"second")))
    assert not first.exists() and not second.exists()


@pytest.mark.parametrize("replacement", [1, 2])
def test_safe_writer_rolls_back_each_replace_failure(tmp_path, monkeypatch, replacement) -> None:
    monkeypatch.setattr(features, "ROOT", tmp_path)
    out = tmp_path / "out"; out.mkdir()
    first, second = out / "first.json", out / "second.json"
    original = features.os.replace; calls = {"count": 0}
    def interrupted(source, destination):
        calls["count"] += 1
        if calls["count"] == replacement:
            raise OSError("replace interruption")
        return original(source, destination)
    monkeypatch.setattr(features.os, "replace", interrupted)
    with pytest.raises(OSError):
        features._safe_write_many(((first, b"first"), (second, b"second")))
    assert not first.exists() and not second.exists()


def test_safe_writer_rejects_incomplete_or_mismatched_immutable_pair(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(features, "ROOT", tmp_path)
    out = tmp_path / "out"; out.mkdir()
    first, second = out / "first", out / "second"
    first.write_bytes(b"old")
    with pytest.raises(features.G1DraftFeatureError, match="incomplete"):
        features._safe_write_many(((first, b"first"), (second, b"second")))
    second.write_bytes(b"old")
    with pytest.raises(features.G1DraftFeatureError, match="different generation"):
        features._safe_write_many(((first, b"first"), (second, b"second")))
