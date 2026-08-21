from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.future_value_tierlist_full_census_diff import (
    FullCensusTierDiffError,
    build_full_census_tier_diff,
    canonical_json_bytes,
    sha256_path,
    write_full_census_tier_diff,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SOURCE_AS_OF = "2026-08-20T14:51:29Z"
ACCEPTED = ("g1", "g2")
ELIGIBLE = ("g1",)
SOURCE_RECEIPT_SHA256 = ""


def _seal(payload: dict[str, object], field: str) -> dict[str, object]:
    value = dict(payload)
    value[field] = hashlib.sha256(
        canonical_json_bytes({key: item for key, item in value.items() if key != field})
    ).hexdigest()
    return value


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    return sha256_path(path)


def _source_receipt() -> dict[str, object]:
    receipt: dict[str, object] = {
        "accepted_game_ids": list(ACCEPTED),
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "merge": False,
            "deployment": False,
        },
        "checkpoint_coverage": {},
        "identity_coverage": {},
        "model_contract": {},
        "model_exclusions": {},
        "model_eligible_game_count": len(ELIGIBLE),
        "model_eligible_game_ids": list(ELIGIBLE),
        "model_eligible_identity_sha256": identity_sha256(ELIGIBLE),
        "receipt_sha256": "0" * 64,
        "schema_version": "scryglass:future-value-rating-source:v1",
        "source_as_of": SOURCE_AS_OF,
        "source_extra_game_ids": {},
        "source_files": {
            label: {"bytes": 1, "locator": f"{label}.bin", "sha256": "0" * 64}
            for label in ("maps", "players", "teams", "accepted_census")
        },
        "source_game_count": len(ACCEPTED),
        "source_identity_sha256": identity_sha256(ACCEPTED),
        "source_rows": {},
        "status": "accepted_source_bound_development_only",
    }
    return _seal(receipt, "receipt_sha256")


def _row(
    champion_id: str,
    champion: str,
    rank: int,
    tier: str,
) -> dict[str, object]:
    return {
        "champion_id": champion_id,
        "champion": champion,
        "rank": rank,
        "tier_bucket": tier,
        "tier_value_pp": float(rank),
        "strength_score": float(rank) / 10.0,
        "strength_sd_logit": 0.1,
        "rating": 1500.0 - rank,
        "played_maps": 2,
    }


def _candidate(
    *,
    game_count: int,
    source_identity: str,
    rows: list[dict[str, object]],
    source_receipt_sha256: str | None = None,
) -> dict[str, object]:
    candidate: dict[str, object] = {
        "artifact_kind": "tier_list_candidate",
        "as_of": SOURCE_AS_OF,
        "cells": [
            {
                "scope_id": "patch:16.16",
                "patches": ["16.16"],
                "role": "top",
                "rows": rows,
            }
        ],
        "development_only": True,
        "production_eligible": False,
        "publication_eligible": False,
        "schema_version": "scryglass:champion-role-elo-candidate:v2",
        "source": {
            "maps_replayed": game_count,
            "maps_used_in_joint_likelihood": game_count,
            "source_identity_sha256": source_identity,
            "source_latest_replayed": SOURCE_AS_OF,
        },
        "status": "development_only",
    }
    if source_receipt_sha256 is not None:
        candidate["pre_map_offset_override"] = {
            "applied": True,
            "game_count": game_count,
            "game_identity_sha256": source_identity,
            "offsets_sha256": "1" * 64,
            "provenance": {
                "schema_version": "scryglass:future-value-pre-map-offset-provenance:v1",
                "status": "research_only",
                "authority": False,
                "producer": "future_value_rating:future_player_form",
                "timing": "strict_prior_pre_map",
                "source_receipt_sha256": source_receipt_sha256,
                "source_identity_sha256": source_identity,
                "source_game_count": game_count,
                "offsets_sha256": "1" * 64,
                "receipt_sha256": "2" * 64,
            },
        }
    return _seal(candidate, "artifact_sha256")


def _fixture(tmp_path: Path) -> dict[str, object]:
    source_path = tmp_path / "source-receipt.json"
    source = _source_receipt()
    source_hash = _write_json(source_path, source)
    baseline_path = tmp_path / "baseline.json"
    baseline = _candidate(
        game_count=len(ACCEPTED),
        source_identity=identity_sha256(ACCEPTED),
        rows=[_row("a", "A", 1, "S"), _row("b", "B", 2, "A")],
    )
    baseline_hash = _write_json(baseline_path, baseline)
    v2_path = tmp_path / "v2.json"
    v2 = _candidate(
        game_count=len(ELIGIBLE),
        source_identity=identity_sha256(ELIGIBLE),
        rows=[_row("a", "A", 2, "A"), _row("c", "C", 1, "S")],
        source_receipt_sha256=str(source["receipt_sha256"]),
    )
    v2_hash = _write_json(v2_path, v2)
    return {
        "source_receipt_path": source_path,
        "expected_source_receipt_file_sha256": source_hash,
        "expected_source_receipt_sha256": source["receipt_sha256"],
        "baseline_candidate_path": baseline_path,
        "expected_baseline_candidate_sha256": baseline_hash,
        "v2_candidate_path": v2_path,
        "expected_v2_candidate_sha256": v2_hash,
    }


def test_builds_common_row_diff_with_closed_authority(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    report = build_full_census_tier_diff(**inputs)

    assert report["status"] == "research_only"
    assert report["authority"]["research_only"] is True
    assert not any(
        value
        for key, value in report["authority"].items()
        if key != "research_only"
    )
    comparison = report["comparison"]
    assert comparison["common_row_count"] == 1
    assert comparison["baseline_only_row_count"] == 1
    assert comparison["v2_only_row_count"] == 1
    assert comparison["changed_rank_count"] == 1
    assert comparison["changed_tier_count"] == 1
    assert report["source"]["accepted_game_count"] == 2
    assert report["source"]["model_eligible_game_count"] == 1
    assert len(report["rows"]) == 1


def test_source_receipt_file_hash_blocks_mutation(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    path = Path(inputs["source_receipt_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_game_count"] = 99
    path.write_bytes(canonical_json_bytes(payload) + b"\n")

    with pytest.raises(FullCensusTierDiffError, match="source receipt bytes changed"):
        build_full_census_tier_diff(**inputs)


def test_candidate_file_hash_blocks_value_mutation(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    path = Path(inputs["baseline_candidate_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cells"][0]["rows"][0]["rank"] = 99
    path.write_bytes(canonical_json_bytes(payload) + b"\n")

    with pytest.raises(FullCensusTierDiffError, match="baseline candidate bytes changed"):
        build_full_census_tier_diff(**inputs)


def test_candidate_identity_is_checked_after_external_hash_update(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    path = Path(inputs["v2_candidate_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source"]["source_identity_sha256"] = "f" * 64
    payload = _seal(payload, "artifact_sha256")
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    inputs["expected_v2_candidate_sha256"] = sha256_path(path)

    with pytest.raises(FullCensusTierDiffError, match="V2 candidate source identity changed"):
        build_full_census_tier_diff(**inputs)


def test_writes_report_and_receipt_with_input_bindings(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path)
    report = build_full_census_tier_diff(**inputs)
    report_path, receipt_path = write_full_census_tier_diff(
        report,
        output_root=tmp_path / "output",
    )
    written_report = json.loads(report_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"].endswith("-receipt:v1")
    assert receipt["report"]["sha256"] == sha256_path(report_path)
    assert receipt["report"]["bytes"] == report_path.stat().st_size
    assert receipt["comparison"] == written_report["comparison"]
    assert receipt["inputs"] == written_report["inputs"]
