from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.future_value_tierlist_full_census_diff import (
    FullCensusTierDiffError,
    _load_baseline_bundle_inputs,
    audit_final_v2_full_census_scoreability,
    build_full_census_fourway_diff,
    build_full_census_tier_diff,
    canonical_json_bytes,
    sha256_path,
    write_full_census_tier_diff,
)
from benchmarks.rebuild_future_value_tier_baseline import (
    BUNDLE_FILE,
    rebuild_tier_baseline,
)
from lol_kills.research.future_value_tierlist import VARIANTS, make_offset_provenance
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


def _final_v2_receipt(source_receipt_sha256: str) -> dict[str, object]:
    receipt: dict[str, object] = {
        "authority": {"research_only": True, "promotion": False},
        "blockers": [
            "authoritative_series_id_missing_proxy_cluster_used",
            "tournament_boundary_field_missing",
            "tournament_boundary_slice_missing",
        ],
        "feature_ledger_binding": {
            "artifact": {
                "bytes": 1,
                "path": "current-rating-ledger.parquet",
                "sha256": "3" * 64,
            },
            "game_identity_sha256": identity_sha256(ELIGIBLE),
            "rows": len(ELIGIBLE),
            "strict_prior_timing": "source_bound_current_rating_before_snapshot_as_of",
        },
        "fit_game_count": len(ELIGIBLE),
        "fit_game_identity_sha256": identity_sha256(ELIGIBLE),
        "schema_version": "scryglass:future-value-model-fit:v1",
        "source_binding": {
            "model_eligible_game_count": len(ELIGIBLE),
            "model_eligible_identity_sha256": identity_sha256(ELIGIBLE),
            "source_game_count": len(ACCEPTED),
            "source_identity_sha256": identity_sha256(ACCEPTED),
            "source_receipt_sha256": source_receipt_sha256,
        },
        "status": "research_only_blocked",
        "receipt_sha256": "0" * 64,
    }
    return _seal(receipt, "receipt_sha256")


def test_final_v2_full_census_scoreability_fails_closed_on_fit_subset(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source-receipt.json"
    source = _source_receipt()
    source_hash = _write_json(source_path, source)
    model_path = tmp_path / "final-v2-model-receipt.json"
    model = _final_v2_receipt(str(source["receipt_sha256"]))
    model_hash = _write_json(model_path, model)

    audit = audit_final_v2_full_census_scoreability(
        source_receipt_path=source_path,
        expected_source_receipt_file_sha256=source_hash,
        expected_source_receipt_sha256=str(source["receipt_sha256"]),
        model_receipt_path=model_path,
        expected_model_receipt_file_sha256=model_hash,
        expected_model_receipt_sha256=str(model["receipt_sha256"]),
    )

    assert audit["status"] == "research_only_blocked"
    assert audit["coverage"] == {
        "accepted_game_count": 2,
        "scored_game_count": 1,
        "missing_game_count": 1,
        "scored_identity_sha256": identity_sha256(ELIGIBLE),
        "matches_accepted_census": False,
        "matches_model_eligible_census": True,
    }
    assert audit["decision"]["can_score_accepted_census"] is False
    assert audit["decision"]["can_build_source_bound_tier_offset_ledger"] is False
    assert "final_v2_feature_ledger_does_not_cover_accepted_census" in audit[
        "blockers"
    ]
    assert "retrospective_full_census_model_fit_not_chronological_evaluation" in audit[
        "blockers"
    ]


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


def _fourway_fixture(tmp_path: Path) -> dict[str, object]:
    inputs = _fixture(tmp_path)
    source = _source_receipt()
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for variant in VARIANTS:
        candidate = _candidate(
            game_count=len(ELIGIBLE),
            source_identity=identity_sha256(ELIGIBLE),
            rows=[_row("a", "A", 1, "S"), _row("b", "B", 2, "A")],
        )
        provenance = make_offset_provenance(
            variant=variant,
            offsets={"g1": float(VARIANTS.index(variant))},
            source_receipt_sha256=str(source["receipt_sha256"]),
        )
        candidate["pre_map_offset_override"] = {
            "applied": True,
            "game_count": len(ELIGIBLE),
            "game_identity_sha256": identity_sha256(ELIGIBLE),
            "offsets_sha256": provenance["offsets_sha256"],
            "provenance": provenance,
        }
        candidate = _seal(candidate, "artifact_sha256")
        path = tmp_path / f"{variant}.json"
        hashes[variant] = _write_json(path, candidate)
        paths[variant] = path
    inputs["variant_candidate_paths"] = paths
    inputs["expected_variant_candidate_sha256"] = hashes
    return inputs


def _baseline_bundle_fixture(tmp_path: Path) -> tuple[Path, str]:
    input_root = tmp_path / "baseline-input"
    source_root = input_root / "source"
    source_root.mkdir(parents=True)
    source_names = {
        "maps": "maps.parquet",
        "players": "oe_player_games.parquet",
        "teams": "oe_team_games.parquet",
    }
    for label, name in source_names.items():
        pd.DataFrame({"game_uid": list(ACCEPTED)}).to_parquet(source_root / name)
    accepted_census = {
        "schema_version": "scryglass:accepted-game-census:v1",
        "game_count": len(ACCEPTED),
        "source_identity_sha256": identity_sha256(ACCEPTED),
        "game_ids": list(ACCEPTED),
    }
    _write_json(source_root / "accepted-census.json", accepted_census)
    _write_json(source_root / "meta.json", {"source_as_of": SOURCE_AS_OF})

    source_receipt = _source_receipt()
    source_receipt["source_files"] = {
        label: {
            "locator": name,
            "bytes": (source_root / name).stat().st_size,
            "sha256": sha256_path(source_root / name),
        }
        for label, name in source_names.items()
    }
    source_receipt["source_files"]["accepted_census"] = {
        "locator": "accepted-census.json",
        "bytes": (source_root / "accepted-census.json").stat().st_size,
        "sha256": sha256_path(source_root / "accepted-census.json"),
    }
    source_receipt = _seal(source_receipt, "receipt_sha256")
    source_receipt_path = input_root / "future-value-source-receipt.json"
    source_receipt_file_sha256 = _write_json(source_receipt_path, source_receipt)

    def candidate_builder(_runtime_root: Path, **_kwargs: object) -> dict[str, object]:
        candidate = _candidate(
            game_count=len(ACCEPTED),
            source_identity=identity_sha256(ACCEPTED),
            rows=[_row("a", "A", 1, "S"), _row("b", "B", 2, "A")],
        )
        candidate["joint_model"] = {"map_ids": list(ACCEPTED)}
        return _seal(candidate, "artifact_sha256")

    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "baseline-bundle"
    rebuild_tier_baseline(
        source_root=source_root,
        source_receipt_path=source_receipt_path,
        expected_source_receipt_file_sha256=source_receipt_file_sha256,
        expected_source_receipt_sha256=str(source_receipt["receipt_sha256"]),
        output_root=output,
        repository_root=repository,
        expected_accepted_game_count=len(ACCEPTED),
        expected_accepted_identity_sha256=identity_sha256(ACCEPTED),
        expected_model_eligible_game_count=len(ELIGIBLE),
        expected_model_eligible_identity_sha256=identity_sha256(ELIGIBLE),
        candidate_builder=candidate_builder,
    )
    bundle_path = output / BUNDLE_FILE
    return bundle_path, sha256_path(bundle_path)


def test_fourway_full_census_uses_identical_universe_and_keeps_baseline_unchanged(
    tmp_path: Path,
) -> None:
    report = build_full_census_fourway_diff(**_fourway_fixture(tmp_path))

    assert report["status"] == "research_only"
    assert report["timing"] == {
        "mode": "retrospective_full_model_eligible_census",
        "chronological_evaluation_suitable": False,
        "validation_offsets_used": False,
    }
    assert report["candidate_universe"]["identical"] is True
    assert report["candidate_universe"]["variants"] == list(VARIANTS)
    assert report["baseline_public_candidate"]["status"] == (
        "unchanged_non_comparable_full_census_reference"
    )
    assert report["comparisons"]["current_only"]["changed_rank_count"] == 0
    assert all(
        report["comparisons"][variant]["row_count"] == 2 for variant in VARIANTS
    )
    assert "chronological_subset_evidence_not_used_for_full_census_offsets" in report[
        "blockers"
    ]


def test_baseline_bundle_external_hash_rejects_changed_bytes(tmp_path: Path) -> None:
    bundle_path, bundle_sha256 = _baseline_bundle_fixture(tmp_path)
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["status"] = "changed"
    bundle_path.write_bytes(canonical_json_bytes(payload) + b"\n")

    with pytest.raises(
        FullCensusTierDiffError,
        match="Tier baseline bundle failed validation: trust-input bundle bytes changed",
    ):
        _load_baseline_bundle_inputs(
            baseline_bundle_path=bundle_path,
            expected_baseline_bundle_sha256=bundle_sha256,
        )


def test_baseline_bundle_rejects_resealed_source_mismatch(tmp_path: Path) -> None:
    bundle_path, _bundle_sha256 = _baseline_bundle_fixture(tmp_path)
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["source"]["source_game_count"] = 99
    payload = _seal(payload, "bundle_sha256")
    bundle_sha256 = _write_json(bundle_path, payload)

    with pytest.raises(
        FullCensusTierDiffError,
        match="Tier baseline bundle source binding changed: source_game_count",
    ):
        _load_baseline_bundle_inputs(
            baseline_bundle_path=bundle_path,
            expected_baseline_bundle_sha256=bundle_sha256,
        )


def test_fourway_receipt_binds_all_variant_comparisons(tmp_path: Path) -> None:
    report = build_full_census_fourway_diff(**_fourway_fixture(tmp_path))
    report_path, receipt_path = write_full_census_tier_diff(
        report,
        output_root=tmp_path / "fourway-output",
    )

    written_report = json.loads(report_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert "comparison" not in receipt
    assert receipt["comparisons"] == written_report["comparisons"]
    assert set(receipt["comparisons"]) == set(VARIANTS)
    assert receipt["report"]["sha256"] == sha256_path(report_path)


def test_fourway_full_census_rejects_missing_or_extra_candidate_rows(
    tmp_path: Path,
) -> None:
    inputs = _fourway_fixture(tmp_path)
    path = Path(inputs["variant_candidate_paths"]["both"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cells"][0]["rows"].pop()
    payload = _seal(payload, "artifact_sha256")
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    inputs["expected_variant_candidate_sha256"]["both"] = sha256_path(path)

    with pytest.raises(FullCensusTierDiffError, match="row universes differ"):
        build_full_census_fourway_diff(**inputs)
