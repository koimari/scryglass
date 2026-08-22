from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.rebuild_future_value_tier_baseline import (
    BUNDLE_FILE,
    TierBaselineRebuildError,
    load_tier_baseline_bundle,
    rebuild_tier_baseline,
    sha256_path,
)
from lol_kills.research.future_value_rating import SOURCE_RECEIPT_AUTHORITY
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SOURCE_AS_OF = "2026-01-04T00:00:00Z"
IDS = ("g1", "g2")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")
    return sha256_path(path)


def _seal(value: dict[str, object], field: str) -> dict[str, object]:
    result = dict(value)
    result[field] = hashlib.sha256(
        _canonical({key: item for key, item in result.items() if key != field})
    ).hexdigest()
    return result


def _write_receipt(
    root: Path,
    *,
    accepted_ids: tuple[str, ...] = IDS,
    eligible_ids: tuple[str, ...] = ("g1",),
) -> tuple[Path, str, str]:
    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    frames = {
        "maps": pd.DataFrame({"game_uid": list(IDS)}),
        "players": pd.DataFrame({"game_uid": list(IDS)}),
        "teams": pd.DataFrame({"game_uid": list(IDS)}),
    }
    names = {
        "maps": "maps.parquet",
        "players": "oe_player_games.parquet",
        "teams": "oe_team_games.parquet",
    }
    for label, frame in frames.items():
        frame.to_parquet(source / names[label])
    _write_json(
        source / "accepted-census.json",
        {
            "schema_version": "scryglass:accepted-game-census:v1",
            "game_count": len(accepted_ids),
            "source_identity_sha256": identity_sha256(accepted_ids),
            "game_ids": list(accepted_ids),
        },
    )
    _write_json(source / "meta.json", {"source_as_of": SOURCE_AS_OF})
    records = {
        label: {
            "locator": name,
            "bytes": (source / name).stat().st_size,
            "sha256": sha256_path(source / name),
        }
        for label, name in names.items()
    }
    records["accepted_census"] = {
        "locator": "accepted-census.json",
        "bytes": (source / "accepted-census.json").stat().st_size,
        "sha256": sha256_path(source / "accepted-census.json"),
    }
    receipt = _seal(
        {
            "schema_version": "scryglass:future-value-rating-source:v1",
            "status": "accepted_source_bound_development_only",
            "source_as_of": SOURCE_AS_OF,
            "source_game_count": len(accepted_ids),
            "source_identity_sha256": identity_sha256(accepted_ids),
            "accepted_game_ids": list(accepted_ids),
            "model_eligible_game_count": len(eligible_ids),
            "model_eligible_identity_sha256": identity_sha256(eligible_ids),
            "model_eligible_game_ids": list(eligible_ids),
            "source_rows": {},
            "source_extra_game_ids": {},
            "identity_coverage": {},
            "checkpoint_coverage": {},
            "model_exclusions": {},
            "source_files": records,
            "model_contract": {},
            "authority": dict(SOURCE_RECEIPT_AUTHORITY),
            "receipt_sha256": "0" * 64,
        },
        "receipt_sha256",
    )
    receipt_path = root / "future-value-source-receipt.json"
    receipt_file_sha256 = _write_json(receipt_path, receipt)
    return receipt_path, receipt_file_sha256, str(receipt["receipt_sha256"])


def _candidate() -> dict[str, object]:
    return _seal(
        {
            "schema_version": "scryglass:champion-role-elo-candidate:v2",
            "artifact_kind": "tier_list_candidate",
            "status": "development_only",
            "development_only": True,
            "production_eligible": False,
            "publication_eligible": False,
            "as_of": SOURCE_AS_OF,
            "source": {
                "maps_replayed": len(IDS),
                "maps_used_in_joint_likelihood": len(IDS),
                "source_identity_sha256": identity_sha256(IDS),
                "source_latest_replayed": SOURCE_AS_OF,
            },
            "joint_model": {"map_ids": list(IDS)},
            "cells": [],
        },
        "artifact_sha256",
    )


@pytest.fixture
def fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "inputs"
    receipt_path, receipt_file_sha256, receipt_sha256 = _write_receipt(root)
    repo = tmp_path / "repo"
    repo.mkdir()
    captured: dict[str, object] = {}
    candidate = _candidate()

    def fake_builder(runtime_root: Path, **kwargs: object) -> dict[str, object]:
        captured["runtime_root"] = runtime_root
        captured["kwargs"] = kwargs
        return dict(candidate)

    return {
        "source_root": root / "source",
        "receipt": receipt_path,
        "receipt_file_sha256": receipt_file_sha256,
        "receipt_sha256": receipt_sha256,
        "repo": repo,
        "captured": captured,
        "candidate_builder": fake_builder,
        "accepted_identity": identity_sha256(IDS),
    }


def _build(fixture: dict[str, object], output: Path, **kwargs: object) -> dict[str, object]:
    return rebuild_tier_baseline(
        source_root=fixture["source_root"],
        source_receipt_path=fixture["receipt"],
        expected_source_receipt_file_sha256=fixture["receipt_file_sha256"],
        expected_source_receipt_sha256=fixture["receipt_sha256"],
        output_root=output,
        repository_root=fixture["repo"],
        expected_accepted_game_count=len(IDS),
        expected_accepted_identity_sha256=fixture["accepted_identity"],
        candidate_builder=fixture["candidate_builder"],
        **kwargs,
    )


def test_rebuild_binds_exact_ids_and_research_authority(
    fixture: dict[str, object], tmp_path: Path
) -> None:
    output = tmp_path / "output"
    bundle = _build(fixture, output)
    assert bundle["status"] == "research_only"
    assert bundle["authority"]["public_tierlist"] is False
    assert bundle["source"]["source_game_count"] == len(IDS)
    assert bundle["candidate"]["validation"]["map_count"] == len(IDS)
    assert bundle["source"]["source_identity_sha256"] == identity_sha256(IDS)
    kwargs = fixture["captured"]["kwargs"]
    assert tuple(kwargs["allowed_game_ids"]) == IDS
    loaded = load_tier_baseline_bundle(
        output / BUNDLE_FILE,
        expected_raw_sha256=sha256_path(output / BUNDLE_FILE),
    )
    assert loaded["bundle_sha256"] == bundle["bundle_sha256"]


def test_missing_accepted_id_fails_before_candidate_build(
    fixture: dict[str, object], tmp_path: Path
) -> None:
    players = fixture["source_root"] / "oe_player_games.parquet"
    pd.DataFrame({"game_uid": ["g1"]}).to_parquet(players)
    with pytest.raises(TierBaselineRebuildError, match="source players bytes changed"):
        _build(fixture, tmp_path / "output")


def test_missing_accepted_census_id_fails_closed(
    fixture: dict[str, object], tmp_path: Path
) -> None:
    receipt_path, receipt_file_sha256, receipt_sha256 = _write_receipt(
        fixture["source_root"].parent,
        accepted_ids=("g1",),
    )
    with pytest.raises(TierBaselineRebuildError, match="accepted census count changed"):
        rebuild_tier_baseline(
            source_root=fixture["source_root"],
            source_receipt_path=receipt_path,
            expected_source_receipt_file_sha256=receipt_file_sha256,
            expected_source_receipt_sha256=receipt_sha256,
            output_root=tmp_path / "output",
            repository_root=fixture["repo"],
            expected_accepted_game_count=len(IDS),
            expected_accepted_identity_sha256=fixture["accepted_identity"],
            candidate_builder=fixture["candidate_builder"],
        )


def test_extra_accepted_census_id_must_exist_in_each_source(
    fixture: dict[str, object], tmp_path: Path
) -> None:
    extra_ids = ("g1", "g2", "g3")
    receipt_path, receipt_file_sha256, receipt_sha256 = _write_receipt(
        fixture["source_root"].parent,
        accepted_ids=extra_ids,
    )
    with pytest.raises(TierBaselineRebuildError, match="missing from source"):
        rebuild_tier_baseline(
            source_root=fixture["source_root"],
            source_receipt_path=receipt_path,
            expected_source_receipt_file_sha256=receipt_file_sha256,
            expected_source_receipt_sha256=receipt_sha256,
            output_root=tmp_path / "output",
            repository_root=fixture["repo"],
            expected_accepted_game_count=len(extra_ids),
            expected_accepted_identity_sha256=identity_sha256(extra_ids),
            candidate_builder=fixture["candidate_builder"],
        )


def test_receipt_identity_drift_fails_closed(
    fixture: dict[str, object], tmp_path: Path
) -> None:
    with pytest.raises(TierBaselineRebuildError, match="accepted census identity changed"):
        rebuild_tier_baseline(
            source_root=fixture["source_root"],
            source_receipt_path=fixture["receipt"],
            expected_source_receipt_file_sha256=fixture["receipt_file_sha256"],
            expected_source_receipt_sha256=fixture["receipt_sha256"],
            output_root=tmp_path / "output",
            repository_root=fixture["repo"],
            expected_accepted_game_count=len(IDS),
            expected_accepted_identity_sha256="f" * 64,
            candidate_builder=fixture["candidate_builder"],
        )


def test_symlinked_source_fails_closed(
    fixture: dict[str, object], tmp_path: Path
) -> None:
    players = fixture["source_root"] / "oe_player_games.parquet"
    target = tmp_path / "players-target.parquet"
    target.write_bytes(players.read_bytes())
    players.unlink()
    players.symlink_to(target)
    with pytest.raises(TierBaselineRebuildError, match="symlink"):
        _build(fixture, tmp_path / "output")


def test_existing_output_and_tampered_bundle_fail_closed(
    fixture: dict[str, object], tmp_path: Path
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "existing").write_text("owned", encoding="utf-8")
    with pytest.raises(TierBaselineRebuildError, match="must be empty"):
        _build(fixture, output)

    clean_output = tmp_path / "clean-output"
    _build(fixture, clean_output)
    bundle_path = clean_output / BUNDLE_FILE
    bundle = json.loads(bundle_path.read_text())
    bundle["candidate"]["raw_sha256"] = "0" * 64
    bundle_path.write_bytes(_canonical(bundle) + b"\n")
    with pytest.raises(TierBaselineRebuildError, match="self hash changed"):
        load_tier_baseline_bundle(bundle_path)
