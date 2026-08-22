from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.build_future_value_tier_trust import (
    TierTrustBuilderError,
    build_tier_trust_manifest,
    canonical_json_bytes,
    sha256_path,
)
from lol_kills.research.future_value_rating import validate_future_value_source_receipt_payload
from lol_kills.research.future_value_tierlist import load_trust_manifest
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SOURCE_AS_OF = "2026-01-04T00:00:00Z"
IDS = ["g1", "g2", "g3"]
VARIANTS = ("current_only", "future_player_form", "scaling_curve", "both")


def _seal(value: dict[str, object], field: str) -> dict[str, object]:
    output = dict(value)
    output[field] = hashlib.sha256(
        canonical_json_bytes({key: item for key, item in output.items() if key != field})
    ).hexdigest()
    return output


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    return sha256_path(path)


def _source(freeze: Path) -> tuple[Path, str, str]:
    source = freeze / "source"
    source.mkdir(parents=True)
    maps = source / "maps.parquet"
    pd.DataFrame(
        {
            "game_uid": IDS,
            "date": [
                "2026-01-02T00:00:00Z",
                "2026-01-03T00:00:00Z",
                "2026-01-04T00:00:00Z",
            ],
            "y_blue_win": [1, 0, 1],
        }
    ).to_parquet(maps)
    players = source / "oe_player_games.parquet"
    pd.DataFrame({"game_uid": IDS, "playerid": IDS}).to_parquet(players)
    teams = source / "oe_team_games.parquet"
    pd.DataFrame({"game_uid": IDS, "teamid": IDS}).to_parquet(teams)
    meta = source / "meta.json"
    _write_json(meta, {"source_as_of": SOURCE_AS_OF})
    accepted = freeze / "accepted-census.json"
    _write_json(accepted, {"accepted_game_ids": IDS})
    records = {}
    for label, path in (
        ("maps", maps),
        ("players", players),
        ("teams", teams),
        ("accepted_census", accepted),
    ):
        records[label] = {
            "locator": str(path.relative_to(freeze)),
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        }
    receipt = _seal(
        {
            "schema_version": "scryglass:future-value-rating-source:v1",
            "status": "accepted_source_bound_development_only",
            "source_as_of": SOURCE_AS_OF,
            "source_game_count": len(IDS),
            "source_identity_sha256": identity_sha256(IDS),
            "accepted_game_ids": IDS,
            "model_eligible_game_count": len(IDS),
            "model_eligible_identity_sha256": identity_sha256(IDS),
            "model_eligible_game_ids": IDS,
            "source_rows": {},
            "source_extra_game_ids": {},
            "identity_coverage": {},
            "checkpoint_coverage": {},
            "model_exclusions": {},
            "source_files": records,
            "model_contract": {},
            "authority": {
                "research_only": True,
                "public_player_rating": False,
                "public_team_rating": False,
                "public_probability": False,
                "promotion": False,
                "merge": False,
                "deployment": False,
            },
            "receipt_sha256": "0" * 64,
        },
        "receipt_sha256",
    )
    receipt_path = freeze / "future-value-source-receipt.json"
    receipt_file_hash = _write_json(receipt_path, receipt)
    return receipt_path, receipt_file_hash, str(receipt["receipt_sha256"])


def _candidate(freeze: Path, identity: str = identity_sha256(IDS)) -> tuple[Path, str]:
    candidate = {
        "artifact_kind": "tier_list_candidate",
        "as_of": SOURCE_AS_OF,
        "cells": [
            {
                "scope_id": "patch:26.16",
                "patches": ["26.16"],
                "role": "top",
                "rows": [
                    {
                        "champion": "Ahri",
                        "champion_id": "riot:champion:103",
                        "rank": 1,
                        "tier_bucket": "A",
                        "tier_value_pp": 1.0,
                        "strength_score": 0.5,
                        "strength_sd_logit": 0.1,
                        "rating": 1500.0,
                        "played_maps": 3,
                    }
                ],
            }
        ],
        "development_only": True,
        "production_eligible": False,
        "publication_eligible": False,
        "schema_version": "scryglass:champion-role-elo-candidate:v2",
        "source": {
            "maps_replayed": len(IDS),
            "maps_used_in_joint_likelihood": len(IDS),
            "source_identity_sha256": identity,
            "source_latest_replayed": SOURCE_AS_OF,
        },
        "status": "development_only",
        "claim_ceiling": {"research_only": True},
    }
    path = freeze / "baseline" / "tierlists" / "champion-elo-candidate-v1.json"
    digest = _write_json(path, _seal(candidate, "artifact_sha256"))
    return path, digest


def _model(
    root: Path,
    variant: str,
    source_receipt_sha: str,
    source_receipt_file_sha: str,
) -> tuple[Path, str]:
    rows = [
        {"fold": 1, "game_id": "g1", "target": 1.0, "candidate": 0.75},
        {"fold": 2, "game_id": "g2", "target": 0.0, "candidate": 0.25},
        {"fold": 3, "game_id": "g3", "target": 1.0, "candidate": 0.6},
    ]
    train_series = ["train:1", "train:2", "train:3"]
    validation_series = ["validation:1", "validation:2", "validation:3"]
    folds = []
    for fold, day, game in ((1, 2, "g1"), (2, 3, "g2"), (3, 4, "g3")):
        fit_day = day - 1
        cutoff = f"2026-01-0{day}T00:00:00Z"
        fit_end = f"2026-01-0{fit_day}T00:00:00Z"
        folds.append(
            {
                "fold": fold,
                "paired_game_ids": [game],
                "paired_game_id_count": 1,
                "validation_game_id_count": 1,
                "validation_game_identity_sha256": identity_sha256([game]),
                "train_end": fit_end,
                "validation_start": cutoff,
                "validation_end": cutoff,
                "validation_interval_start": cutoff,
                "validation_interval_end": cutoff,
                "feature_ledger_binding": {
                    "fit_date_max": fit_end,
                    "fit_window_end": fit_end,
                    "strict_prior_timing": "fit_rows_strictly_before_cutoff",
                    "same_timestamp_policy": "batch_exclude_same_timestamp",
                    "series_safety": {
                        "policy": "whole_series_disjoint",
                        "train_series_ids": [train_series[fold - 1]],
                        "validation_series_ids": [validation_series[fold - 1]],
                        "train_series_identity_sha256": identity_sha256([train_series[fold - 1]]),
                        "validation_series_identity_sha256": identity_sha256([validation_series[fold - 1]]),
                    },
                },
            }
        )
    ledger = {
        "schema_version": "scryglass:future-value-prediction-ledger:v1",
        "row_count": len(rows),
        "game_identity_sha256": identity_sha256([row["game_id"] for row in rows]),
        "rows": rows,
        "sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
    }
    result = {
        "status": "development_evaluated",
        "variant": variant,
        "authority": {"research_only": True, "deployment": False},
        "source": {
            "source_as_of": SOURCE_AS_OF,
            "source_game_count": len(IDS),
            "source_identity_sha256": identity_sha256(IDS),
            "source_receipt_sha256": source_receipt_sha,
            "source_receipt_file_sha256": source_receipt_file_sha,
            "model_eligible_game_count": len(IDS),
            "model_eligible_identity_sha256": identity_sha256(IDS),
            "accepted_game_ids": IDS,
            "model_eligible_game_ids": IDS,
        },
        "blockers": ["research_only"],
        "variant_receipt": {"receipt_sha256": "4" * 64},
        "prediction_ledger": ledger,
        "folds": folds,
    }
    document = {
        "schema_version": "scryglass:future-value-four-variant-evaluation:v1",
        "source": {
            "source_as_of": SOURCE_AS_OF,
            "source_game_count": len(IDS),
            "source_identity_sha256": identity_sha256(IDS),
            "source_receipt_sha256": source_receipt_sha,
        },
        "variants": {variant: result},
    }
    path = root / variant / "model.json"
    digest = _write_json(path, document)
    return path, digest


@pytest.fixture
def fixture(tmp_path: Path) -> dict[str, object]:
    freeze = tmp_path / "freeze"
    receipt_path, receipt_file_hash, receipt_sha = _source(freeze)
    candidate_path, candidate_hash = _candidate(freeze)
    repo = tmp_path / "repo"
    for locator in (
        "data/lol/v2/champions/champion-id-crosswalk-v1.json",
        "benchmarks/future_value_tierlist_fourway.py",
    ):
        _write_json(repo / locator, {"locator": locator})
    manifest = {
        "schema_version": "scryglass:tierlist-production-manifest:v1",
        "status": "approved",
        "candidate": {
            "locator": "ignored/current.json",
            "artifact_sha256": json.loads(candidate_path.read_text())["artifact_sha256"],
            "raw_sha256": candidate_hash,
        },
        "production_eligible": True,
        "decision": "promote",
    }
    manifest_path = repo / "production-manifest.json"
    _write_json(manifest_path, _seal(manifest, "artifact_sha256"))
    evaluation = {
        "schema_version": "scryglass:tierlist-forward-evaluation:v1",
        "status": "complete",
        "candidate": {
            "locator": "ignored/current.json",
            "artifact_sha256": json.loads(candidate_path.read_text())["artifact_sha256"],
            "as_of": SOURCE_AS_OF,
            "raw_sha256": candidate_hash,
        },
        "source": {
            "source_game_count": len(IDS),
            "source_identity_sha256": identity_sha256(IDS),
            "source_latest": SOURCE_AS_OF,
        },
    }
    evaluation_path = repo / "prospective-evaluation.json"
    _write_json(evaluation_path, _seal(evaluation, "artifact_sha256"))
    eval_root = tmp_path / "evaluations"
    models = {}
    for variant in VARIANTS:
        models[variant] = _model(eval_root, variant, receipt_sha, receipt_file_hash)[0]
    return {
        "freeze": freeze,
        "receipt": receipt_path,
        "receipt_file_hash": receipt_file_hash,
        "receipt_sha": receipt_sha,
        "candidate": candidate_path,
        "manifest": manifest_path,
        "evaluation": evaluation_path,
        "repo": repo,
        "eval_root": eval_root,
        "models": models,
    }


def _build(fixture: dict[str, object], output: Path) -> dict[str, object]:
    return build_tier_trust_manifest(
        source_root=fixture["freeze"],
        source_receipt_path=fixture["receipt"],
        expected_source_receipt_file_sha256=fixture["receipt_file_hash"],
        expected_source_receipt_sha256=fixture["receipt_sha"],
        baseline_candidate_path=fixture["candidate"],
        production_manifest_path=fixture["manifest"],
        prospective_evaluation_path=fixture["evaluation"],
        evaluation_root=fixture["eval_root"],
        evaluation_paths=fixture["models"],
        repository_root=fixture["repo"],
        tier_assets={
            "data/lol/v2/champions/champion-id-crosswalk-v1.json": fixture["repo"] / "data/lol/v2/champions/champion-id-crosswalk-v1.json",
        },
        implementation_files={
            "benchmarks/future_value_tierlist_fourway.py": fixture["repo"] / "benchmarks/future_value_tierlist_fourway.py",
        },
        output_path=output,
    )


def test_build_binds_census_assets_code_and_research_authority(fixture: dict[str, object], tmp_path: Path) -> None:
    output = tmp_path / "trust.json"
    trust = _build(fixture, output)
    assert trust["status"] == "research_only"
    assert trust["authority"]["public_tierlist"] is False
    assert trust["source"]["source_game_count"] == 3
    assert trust["source"]["model_eligible_game_count"] == 3
    assert set(trust["evaluations"]) == set(VARIANTS)
    assert "benchmarks/future_value_tierlist_fourway.py" in trust["tier_assets"]
    assert len(trust["trust_root_sha256"]) == 64
    loaded = load_trust_manifest(
        output,
        expected_raw_sha256=sha256_path(output),
    )
    assert loaded["trust_root_sha256"] == trust["trust_root_sha256"]


def test_source_drift_fails_closed(fixture: dict[str, object], tmp_path: Path) -> None:
    maps = fixture["freeze"] / "source/maps.parquet"
    maps.write_bytes(maps.read_bytes() + b"drift")
    with pytest.raises(TierTrustBuilderError, match="source file (?:bytes changed|maps bytes changed)"):
        _build(fixture, tmp_path / "trust.json")


def test_asset_symlink_and_escape_fail_closed(fixture: dict[str, object], tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    _write_json(outside, {"outside": True})
    with pytest.raises(TierTrustBuilderError, match="outside its declared root"):
        build_tier_trust_manifest(
            **{
                "source_root": fixture["freeze"],
                "source_receipt_path": fixture["receipt"],
                "expected_source_receipt_file_sha256": fixture["receipt_file_hash"],
                "expected_source_receipt_sha256": fixture["receipt_sha"],
                "baseline_candidate_path": fixture["candidate"],
                "production_manifest_path": fixture["manifest"],
                "prospective_evaluation_path": fixture["evaluation"],
                "evaluation_root": fixture["eval_root"],
                "evaluation_paths": fixture["models"],
                "repository_root": fixture["repo"],
                "tier_assets": {"escape.json": outside},
                "implementation_files": {
                    "benchmarks/future_value_tierlist_fourway.py": fixture["repo"] / "benchmarks/future_value_tierlist_fourway.py"
                },
                "output_path": tmp_path / "trust.json",
            }
        )


def test_resealed_candidate_census_fails_closed(fixture: dict[str, object], tmp_path: Path) -> None:
    path = fixture["candidate"]
    candidate = json.loads(path.read_text())
    candidate["source"]["source_identity_sha256"] = "f" * 64
    path.write_bytes(canonical_json_bytes(_seal(candidate, "artifact_sha256")) + b"\n")
    fixture["manifest"].write_bytes(fixture["manifest"].read_bytes())
    with pytest.raises(TierTrustBuilderError, match="failed the existing validator"):
        _build(fixture, tmp_path / "trust.json")
