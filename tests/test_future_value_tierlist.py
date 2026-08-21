from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.research.future_value_tierlist import (
    AUTHORITY,
    TRUST_SCHEMA_VERSION,
    VARIANTS,
    FutureValueTierListError,
    build_fourway_diff,
    canonical_json_bytes,
    load_prediction_offsets,
    load_trust_manifest,
    validate_common_prediction_universe,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SOURCE = {
    "source_as_of": "2026-08-20T14:51:29Z",
    "source_game_count": 2,
    "source_identity_sha256": "a" * 64,
    "source_receipt_sha256": "b" * 64,
    "source_receipt_file_sha256": "c" * 64,
    "player_source_sha256": "d" * 64,
    "maps_source_sha256": "e" * 64,
    "meta_source_sha256": "f" * 64,
}


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trust() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": TRUST_SCHEMA_VERSION,
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "source": dict(SOURCE),
        "evaluations": {
            variant: {"locator": f"{variant}/model.json", "raw_sha256": "1" * 64}
            for variant in VARIANTS
        },
        "tier_assets": {"data/asset.json": "2" * 64},
        "baseline_candidate": {"locator": "baseline.json", "raw_sha256": "3" * 64},
    }
    payload["trust_root_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def test_trust_manifest_requires_external_file_hash(tmp_path: Path) -> None:
    path = tmp_path / "trust.json"
    raw_hash = _write_json(path, _trust())
    loaded = load_trust_manifest(path, expected_raw_sha256=raw_hash)
    assert loaded["status"] == "research_only"
    changed = json.loads(path.read_text())
    changed["status"] = "available"
    _write_json(path, changed)
    with pytest.raises(FutureValueTierListError, match="file hash changed"):
        load_trust_manifest(path, expected_raw_sha256=raw_hash)


def _model(variant: str) -> dict[str, object]:
    rows = [
        {"fold": 1, "game_id": "g1", "target": 1.0, "candidate": 0.75},
        {"fold": 2, "game_id": "g2", "target": 0.0, "candidate": 0.25},
    ]
    ledger = {
        "schema_version": "scryglass:future-value-prediction-ledger:v1",
        "row_count": 2,
        "game_identity_sha256": identity_sha256(["g1", "g2"]),
        "rows": rows,
        "sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
    }
    result_source = {
        **{key: SOURCE[key] for key in ("source_as_of", "source_game_count", "source_identity_sha256", "source_receipt_sha256")},
        "source_receipt_file_sha256": SOURCE["source_receipt_file_sha256"],
    }
    return {
        "schema_version": "scryglass:future-value-four-variant-evaluation:v1",
        "source": {key: SOURCE[key] for key in ("source_as_of", "source_game_count", "source_identity_sha256", "source_receipt_sha256")},
        "variants": {
            variant: {
                "status": "development_evaluated",
                "variant": variant,
                "authority": {"research_only": True, "deployment": False},
                "source": result_source,
                "blockers": ["research_blocker"],
                "variant_receipt": {"receipt_sha256": "4" * 64},
                "prediction_ledger": ledger,
            }
        },
    }


def test_prediction_offsets_are_hash_bound_and_finite(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    raw_hash = _write_json(path, _model("future_player_form"))
    offsets, targets, binding = load_prediction_offsets(
        path,
        variant="future_player_form",
        expected_raw_sha256=raw_hash,
        source=SOURCE,
    )
    assert offsets["g1"] == pytest.approx(1.0986122886681098)
    assert targets == {"g1": 1.0, "g2": 0.0}
    assert binding["blockers"] == ["research_blocker"]
    changed = _model("future_player_form")
    changed["variants"]["future_player_form"]["prediction_ledger"]["rows"][0]["candidate"] = 1.0  # type: ignore[index]
    changed_hash = _write_json(path, changed)
    with pytest.raises(FutureValueTierListError, match="ledger values changed"):
        load_prediction_offsets(
            path,
            variant="future_player_form",
            expected_raw_sha256=changed_hash,
            source=SOURCE,
        )


def test_common_prediction_universe_checks_frozen_targets(tmp_path: Path) -> None:
    maps_path = tmp_path / "maps.parquet"
    pd.DataFrame({"game_uid": ["g1", "g2"], "y_blue_win": [1, 0]}).to_parquet(maps_path)
    maps_hash = hashlib.sha256(maps_path.read_bytes()).hexdigest()
    offsets = {variant: {"g1": 0.1, "g2": -0.1} for variant in VARIANTS}
    targets = {variant: {"g1": 1.0, "g2": 0.0} for variant in VARIANTS}
    game_ids, audit = validate_common_prediction_universe(
        offsets,
        targets,
        accepted_game_ids=["g1", "g2"],
        maps_path=maps_path,
        expected_maps_sha256=maps_hash,
    )
    assert game_ids == ["g1", "g2"]
    assert audit["game_count"] == 2
    broken = copy.deepcopy(targets)
    broken["both"]["g2"] = 1.0
    with pytest.raises(FutureValueTierListError, match="target changed"):
        validate_common_prediction_universe(
            offsets,
            broken,
            accepted_game_ids=["g1", "g2"],
            maps_path=maps_path,
            expected_maps_sha256=maps_hash,
        )


def _candidate(swapped: bool = False) -> dict[str, object]:
    first_rank, second_rank = ((2, 1) if swapped else (1, 2))
    return {
        "artifact_sha256": "5" * 64,
        "source": {"source_identity_sha256": "6" * 64},
        "pre_map_offset_override": {"applied": True},
        "cells": [
            {
                "scope_id": "patch:26.16",
                "role": "mid",
                "patches": ["26.16"],
                "rows": [
                    {
                        "champion": "Ahri",
                        "champion_id": "riot:champion:103",
                        "rank": first_rank,
                        "tier_bucket": "A" if not swapped else "B",
                        "tier_value_pp": 2.0 if not swapped else 1.0,
                        "strength_score": 0.52 if not swapped else 0.51,
                        "strength_sd_logit": 0.1,
                        "rating": 1520.0 if not swapped else 1510.0,
                        "played_maps": 20,
                        "counterability_status": "available",
                        "counterability": 60.0,
                        "matchup_maps": 10.0,
                        "matchup_opponents": 5,
                    },
                    {
                        "champion": "Orianna",
                        "champion_id": "riot:champion:61",
                        "rank": second_rank,
                        "tier_bucket": "B" if not swapped else "A",
                        "tier_value_pp": 1.0 if not swapped else 2.0,
                        "strength_score": 0.51 if not swapped else 0.52,
                        "strength_sd_logit": 0.1,
                        "rating": 1510.0 if not swapped else 1520.0,
                        "played_maps": 20,
                        "counterability_status": "available",
                        "counterability": 60.0,
                        "matchup_maps": 10.0,
                        "matchup_opponents": 5,
                    },
                ],
            }
        ],
    }


def test_fourway_diff_uses_exact_stable_identity_and_rank_direction() -> None:
    candidates = {variant: _candidate(variant == "future_player_form") for variant in VARIANTS}
    bindings = {variant: {"blockers": []} for variant in VARIANTS}
    report = build_fourway_diff(
        candidates,
        source={"source_as_of": SOURCE["source_as_of"]},
        universe={"game_count": 2, "game_identity_sha256": "7" * 64},
        model_bindings=bindings,
        trust_manifest_raw_sha256="8" * 64,
        baseline_candidate_raw_sha256="9" * 64,
    )
    comparison = report["comparisons"]["future_player_form"]
    assert comparison["changed_rank_count"] == 2
    assert comparison["changed_tier_count"] == 2
    assert comparison["maximum_absolute_rank_movement"] == 1
    ahri = next(
        row
        for row in report["rows"]["future_player_form"]
        if row["key"]["champion_id"] == "riot:champion:103"
    )
    assert ahri["delta"]["rank_delta"] == -1
    assert report["authority"]["public_tierlist"] is False
