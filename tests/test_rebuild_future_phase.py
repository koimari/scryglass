from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.rebuild_future_phase import (
    PhaseRebuildError,
    _build_rating_reference_partition_frame,
    build_phase_frame,
    select_accepted_rows,
    verify_source_bundle,
)
import lol_kills.research.future_value_rating as future_value_rating
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_source_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    freeze_root = tmp_path / "freeze"
    source_root = freeze_root / "source"
    source_root.mkdir(parents=True)
    ids = list(canonical_game_ids(["oe-api:1", "oe-api:2"]))
    census = {
        "schema_version": "scryglass:accepted-game-census:v1",
        "game_count": len(ids),
        "game_ids": ids,
        "source_identity_sha256": identity_sha256(ids),
    }
    census_path = freeze_root / "accepted-census.json"
    census_path.write_bytes(json.dumps(census, sort_keys=True).encode("utf-8"))
    labels = {
        "accepted_census": census_path,
        "annual_2025": source_root / "annual-2025.csv",
        "annual_2026": source_root / "annual-2026.csv",
        "bridge_oe_api_meta.json": source_root / "bridge-meta.json",
        "bridge_oe_api_player_games.parquet": source_root / "bridge-player.parquet",
        "bridge_oe_api_team_games.parquet": source_root / "bridge-team.parquet",
        "maps": source_root / "maps.parquet",
        "players": source_root / "players.parquet",
        "teams": source_root / "teams.parquet",
    }
    records: dict[str, dict[str, object]] = {}
    for label, path in labels.items():
        if label != "accepted_census":
            path.write_bytes(label.encode("utf-8"))
        records[label] = {
            "locator": str(path.relative_to(freeze_root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    receipt: dict[str, object] = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "source_as_of": "2026-01-01T00:00:00Z",
        "source_game_count": len(ids),
        "source_identity_sha256": identity_sha256(ids),
        "accepted_game_ids": ids,
        "model_eligible_game_count": len(ids),
        "model_eligible_game_ids": ids,
        "model_eligible_identity_sha256": identity_sha256(ids),
        "source_extra_game_ids": {"maps": [], "teams": []},
        "authority": {
            "research_only": True,
            "deployment": False,
            "merge": False,
            "promotion": False,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
        },
        "source_files": records,
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    receipt_path = freeze_root / "future-value-source-receipt.json"
    receipt_path.write_bytes(json.dumps(receipt, sort_keys=True).encode("utf-8"))
    return receipt_path, freeze_root, source_root


def test_verify_source_bundle_checks_durable_receipt_and_file_bytes(tmp_path: Path) -> None:
    receipt_path, freeze_root, source_root = _write_source_bundle(tmp_path)
    receipt, files = verify_source_bundle(
        receipt_path,
        freeze_root=freeze_root,
        source_root=source_root,
    )
    assert receipt["source_game_count"] == 2
    assert set(files) == {
        "accepted_census",
        "annual_2025",
        "annual_2026",
        "bridge_oe_api_meta.json",
        "bridge_oe_api_player_games.parquet",
        "bridge_oe_api_team_games.parquet",
        "maps",
        "players",
        "teams",
    }
    files["maps"].write_bytes(b"changed")
    with pytest.raises(PhaseRebuildError, match="source file bytes changed"):
        verify_source_bundle(receipt_path, freeze_root=freeze_root, source_root=source_root)


def test_select_accepted_rows_allows_only_declared_source_extras() -> None:
    frame = pd.DataFrame({"game_uid": ["1", "2", "extra"]})
    selected = select_accepted_rows(
        frame,
        accepted_ids=["1", "2"],
        declared_extra_ids=["extra"],
        label="maps",
    )
    assert selected["game_uid"].tolist() == ["1", "2"]
    with pytest.raises(PhaseRebuildError, match="undeclared game IDs"):
        select_accepted_rows(
            pd.concat([frame, pd.DataFrame({"game_uid": ["unknown"]})]),
            accepted_ids=["1", "2"],
            declared_extra_ids=["extra"],
            label="maps",
        )


def test_rating_reference_excludes_declared_raw_extra_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path, _freeze_root, _source_root = _write_source_bundle(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    extra_ids = ["excluded-duplicate", "excluded-invalid"]
    receipt["source_extra_game_ids"] = {"maps": extra_ids, "teams": []}
    receipt_payload = dict(receipt)
    receipt_payload.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt_payload)).hexdigest()
    accepted_ids = list(receipt["accepted_game_ids"])
    maps = pd.DataFrame(
        [
            {
                "game_uid": accepted_ids[0],
                "date": "2026-01-01T00:00:00Z",
                "blue_team_key": "blue",
                "red_team_key": "red",
            },
            {
                "game_uid": accepted_ids[1],
                "date": "2026-01-02T00:00:00Z",
                "blue_team_key": "blue",
                "red_team_key": "red",
            },
            {
                "game_uid": extra_ids[0],
                "date": "2026-01-03T00:00:00Z",
                "blue_team_key": "blue",
                "red_team_key": "red",
            },
            {
                "game_uid": extra_ids[0],
                "date": "2026-01-03T00:00:00Z",
                "blue_team_key": "blue",
                "red_team_key": "red",
            },
            {
                "game_uid": extra_ids[1],
                "date": "2026-01-04T00:00:00Z",
                "blue_team_key": "blue",
                "red_team_key": "red",
            },
        ]
    )
    crosswalk_path = tmp_path / "crosswalk.json"
    crosswalk_receipt_path = tmp_path / "crosswalk.receipt.json"
    crosswalk_path.write_bytes(b"crosswalk")
    crosswalk_receipt_path.write_bytes(b"crosswalk-receipt")
    crosswalk_artifact_sha256 = hashlib.sha256(crosswalk_path.read_bytes()).hexdigest()
    crosswalk_receipt_file_sha256 = hashlib.sha256(
        crosswalk_receipt_path.read_bytes()
    ).hexdigest()

    def bind(frame: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        result = frame.copy()
        result.attrs["verified_leaguepedia_series_crosswalk"] = {
            "mapped_game_ids": result["game_id"].astype(str).tolist(),
        }
        return result

    def model_frame(frame: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        result = frame.copy()
        result["series_id"] = "leaguepedia:fixture-series"
        result["_series_crosswalk_mapped"] = True
        result.attrs["series_cluster_source"] = (
            "mixed:leaguepedia_crosswalk+conservative_series_superset"
        )
        result.attrs["series_cluster_audit"] = {
            "source_receipt_sha256": str(
                kwargs["verified_source_receipt"]["receipt_sha256"]
            ),
            "key_fields": ["league", "tournament", "unordered_team_pair"],
            "crosswalk_assignment_sha256": "a" * 64,
            "crosswalk_sha256": "b" * 64,
            "crosswalk_artifact_sha256": crosswalk_artifact_sha256,
            "crosswalk_receipt_sha256": "d" * 64,
            "partial_series_blocker": False,
        }
        return result

    monkeypatch.setattr(
        future_value_rating,
        "bind_verified_leaguepedia_series_crosswalk",
        bind,
    )
    monkeypatch.setattr(future_value_rating, "_map_model_frame", model_frame)
    reference, eligible_assignment, stats = _build_rating_reference_partition_frame(
        maps,
        source_receipt=receipt,
        crosswalk_path=crosswalk_path,
        crosswalk_receipt_path=crosswalk_receipt_path,
        crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
    )
    assert reference.reference_game_count == len(accepted_ids)
    assert reference.reference_identity_sha256 == identity_sha256(accepted_ids)
    assert set(reference.frame["game_id"].astype(str)) == set(accepted_ids)
    assert not set(extra_ids).intersection(reference.frame["game_id"].astype(str))
    assert stats["reference_game_count"] == len(accepted_ids)
    assert stats["reference_promoted_game_count"] == len(accepted_ids)
    assert stats["reference_audit"]["partial_series_blocker"] is False
    assert eligible_assignment == reference.eligible_assignment_sha256


def test_build_phase_frame_keeps_incomplete_team_identity_missing(tmp_path: Path) -> None:
    ids = list(canonical_game_ids(["oe-api:1", "oe-api:2"]))
    maps = pd.DataFrame(
        [
            {"game_uid": ids[0], "date": "2026-01-01T00:00:00Z", "league": "LCK", "tournament": "Spring"},
            {"game_uid": ids[1], "date": "2026-01-02T00:00:00Z", "league": "LCK", "tournament": "Spring"},
        ]
    )
    rows: list[dict[str, object]] = []
    for game_index, game_id in enumerate(ids):
        for side, team_id, value in (
            ("Blue", "oe:team:blue", 1000.0),
            ("Red", "oe:team:red" if game_index == 0 else None, 900.0),
        ):
            row: dict[str, object] = {
                "game_uid": game_id,
                "date": maps.loc[game_index, "date"],
                "side": side,
                "teamid": team_id,
                "earnedgold": value + game_index,
                "dpm": value / 10 + game_index,
                "visionscore": value / 100 + game_index,
                "gamelength": 1800.0,
            }
            for phase in (10, 15, 20, 25):
                row[f"goldat{phase}"] = value + phase
                row[f"xpat{phase}"] = value + phase
            rows.append(row)
    teams = pd.DataFrame(rows)
    phase, stats = build_phase_frame(maps, teams, accepted_ids=ids)
    assert stats["complete_team_maps"] == 1
    assert stats["incomplete_team_identity_maps"] == 1
    assert stats["team_form_feature_rows"] == 1
    incomplete = phase.loc[phase["game_uid"].eq(ids[1])]
    assert incomplete["prior_form_earnedgold_diff"].isna().all()
