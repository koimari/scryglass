from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

import benchmarks.future_value_tierlist_fourway as fourway_benchmark
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
from lol_kills.research.future_value_tierlist import sha256_path
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
    "model_eligible_game_count": 2,
    "model_eligible_identity_sha256": "1" * 64,
}
REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_v11_shadow_freeze_binds_the_completed_evaluation_files() -> None:
    path = REPO_ROOT / "data/lol/v2/evaluation/future-value-tierlist-shadow-freeze-v2.json"
    expected_hash = (
        "782d86edc3b80aeba59c68239c3d35d7d1382567b38202ff08618c77c9b4a3b2"
    )
    assert sha256_path(path) == expected_hash
    trust = load_trust_manifest(path, expected_raw_sha256=expected_hash)
    assert trust["source"]["source_game_count"] == 17764
    assert trust["source"]["model_eligible_game_count"] == 16553
    assert trust["source"]["source_identity_sha256"] == (
        "591820cb87bcb847da449af11349c9f75f4993a9295998cd46db17e1535c5cfb"
    )
    assert trust["source"]["model_eligible_identity_sha256"] == (
        "c2804529b489ea68a05aef4bdc594ba6babb97c32fa19e2b90a589fba693a044"
    )
    assert trust["evaluations"] == {
        "both": {
            "locator": "both/model.json",
            "raw_sha256": "1e4fc6b13f2d80cc73e982acffbdae8ff1f4a239ed0cbb95b55e0ec28a7afbcc",
        },
        "current_only": {
            "locator": "current_only/model.json",
            "raw_sha256": "1d6ac7dc2b4809d02e6aa55b651f09faa1aa38be56c0da2c70792e895e1109e4",
        },
        "future_player_form": {
            "locator": "future_player_form/model.json",
            "raw_sha256": "39ddc0adebd18b2a6b29e26c3e14190db45a9690dc13f996dbd708acd8509f0c",
        },
        "scaling_curve": {
            "locator": "scaling_curve/model.json",
            "raw_sha256": "fdc0d379dbdb3462bc1021f56725d4564ddfc811467d5b061114cc1d64069aa5",
        },
    }
    assert fourway_benchmark.PINNED_TRUST_MANIFEST_RAW_SHA256 == expected_hash


def test_implementation_binding_rejects_a_dirty_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if list(command)[0:4] == [
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
        ]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=" M lol_kills/v2/tierlists/pooled_candidate.py\n",
                stderr="",
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(fourway_benchmark.subprocess, "run", fake_run)
    with pytest.raises(FutureValueTierListError, match="working tree must be clean"):
        fourway_benchmark._implementation_binding(REPO_ROOT)


def test_implementation_binding_records_the_committed_source_and_runtime() -> None:
    binding = fourway_benchmark._implementation_binding(REPO_ROOT)
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", "HEAD^{commit}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert binding["git_commit"] == commit
    files = binding["files"]
    assert isinstance(files, dict)
    assert files["lol_kills/v2/tierlists/champion_elo.py"] == sha256_path(
        REPO_ROOT / "lol_kills/v2/tierlists/champion_elo.py"
    )
    assert files["lol_kills/v2/tierlists/atom_matchup_features.py"] == sha256_path(
        REPO_ROOT / "lol_kills/v2/tierlists/atom_matchup_features.py"
    )
    assert files["lol_kills/v2/tierlists/patch_mapping.py"] == sha256_path(
        REPO_ROOT / "lol_kills/v2/tierlists/patch_mapping.py"
    )
    assert set(files).issuperset(
        {
            "benchmarks/future_value_tierlist_fourway.py",
            "lol_kills/research/future_value_tierlist.py",
            "lol_kills/research/future_value_rating.py",
        }
    )
    runtime = binding["runtime"]
    assert runtime["python_version"]
    assert runtime["python_implementation"]
    assert set(runtime["packages"]) == {"numpy", "pandas", "scipy", "pyarrow"}


def _model(variant: str) -> dict[str, object]:
    rows = [
        {"fold": 1, "game_id": "g1", "target": 1.0, "candidate": 0.75},
        {"fold": 2, "game_id": "g2", "target": 0.0, "candidate": 0.25},
        {"fold": 3, "game_id": "g3", "target": 1.0, "candidate": 0.5},
    ]
    ledger = {
        "schema_version": "scryglass:future-value-prediction-ledger:v1",
        "row_count": 3,
        "game_identity_sha256": identity_sha256(["g1", "g2", "g3"]),
        "rows": rows,
        "sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
    }
    result_source = {
        **{key: SOURCE[key] for key in ("source_as_of", "source_game_count", "source_identity_sha256", "source_receipt_sha256")},
        "source_receipt_file_sha256": SOURCE["source_receipt_file_sha256"],
        "model_eligible_game_count": SOURCE["model_eligible_game_count"],
        "model_eligible_identity_sha256": SOURCE["model_eligible_identity_sha256"],
    }
    folds = []
    for fold, game_id in enumerate(("g1", "g2", "g3"), 1):
        train_series_ids = [f"train:{fold}"]
        validation_series_ids = [f"validation:{fold}"]
        folds.append(
            {
                "fold": fold,
                "paired_game_ids": [game_id],
                "paired_game_id_count": 1,
                "validation_game_id_count": 1,
                "validation_game_identity_sha256": identity_sha256([game_id]),
                "train_end": "2026-01-01T00:00:00Z",
                "validation_start": "2026-01-02T00:00:00Z",
                "validation_end": "2026-01-03T00:00:00Z",
                "validation_interval_start": "2026-01-02T00:00:00Z",
                "validation_interval_end": "2026-01-03T00:00:00Z",
                "feature_ledger_binding": {
                    "fit_date_max": "2026-01-01T00:00:00Z",
                    "fit_window_end": "2026-01-01T00:00:00Z",
                    "strict_prior_timing": "fit_rows_strictly_before_cutoff",
                    "same_timestamp_policy": "batch_exclude_same_timestamp",
                    "series_safety": {
                        "policy": "whole_series_disjoint",
                        "train_series_identity_sha256": identity_sha256(train_series_ids),
                        "validation_series_identity_sha256": identity_sha256(validation_series_ids),
                        "train_series_ids": train_series_ids,
                        "validation_series_ids": validation_series_ids,
                    },
                },
            }
        )
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
                "folds": folds,
            }
        },
    }


def test_prediction_offsets_are_hash_bound_and_finite(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    raw_hash = _write_json(path, _model("future_player_form"))
    maps_path = tmp_path / "maps.parquet"
    pd.DataFrame(
        {
            "game_uid": ["g1", "g2", "g3"],
            "date": [
                "2026-01-02T12:00:00Z",
                "2026-01-02T12:00:00Z",
                "2026-01-02T12:00:00Z",
            ],
            "y_blue_win": [1, 0, 1],
        }
    ).to_parquet(maps_path)
    maps_hash = hashlib.sha256(maps_path.read_bytes()).hexdigest()
    offsets, targets, binding = load_prediction_offsets(
        path,
        variant="future_player_form",
        expected_raw_sha256=raw_hash,
        source=SOURCE,
        maps_path=maps_path,
        expected_maps_sha256=maps_hash,
    )
    assert offsets["g1"] == pytest.approx(1.0986122886681098)
    assert targets == {"g1": 1.0, "g2": 0.0, "g3": 1.0}
    assert binding["blockers"] == ["research_blocker"]
    bad_maps = pd.DataFrame(
        {
            "game_uid": ["g1", "g2", "g3"],
            "date": [
                "2026-01-01T23:59:59Z",
                "2026-01-02T00:00:00Z",
                "2026-01-02T00:00:00Z",
            ],
            "y_blue_win": [1, 0, 1],
        }
    )
    bad_maps_path = tmp_path / "bad-maps.parquet"
    bad_maps.to_parquet(bad_maps_path)
    with pytest.raises(FutureValueTierListError, match="validation window"):
        load_prediction_offsets(
            path,
            variant="future_player_form",
            expected_raw_sha256=raw_hash,
            source=SOURCE,
            maps_path=bad_maps_path,
            expected_maps_sha256=hashlib.sha256(bad_maps_path.read_bytes()).hexdigest(),
        )
    changed = _model("future_player_form")
    changed["variants"]["future_player_form"]["prediction_ledger"]["rows"][0]["candidate"] = 1.0  # type: ignore[index]
    changed_hash = _write_json(path, changed)
    with pytest.raises(FutureValueTierListError, match="ledger values changed"):
        load_prediction_offsets(
            path,
            variant="future_player_form",
            expected_raw_sha256=changed_hash,
            source=SOURCE,
            maps_path=maps_path,
            expected_maps_sha256=maps_hash,
        )


def test_missing_series_ids_blocks_chronology_and_binds_blocker(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    model = _model("future_player_form")
    for fold in model["variants"]["future_player_form"]["folds"]:  # type: ignore[index]
        fold["feature_ledger_binding"]["series_safety"].pop("train_series_ids")  # type: ignore[index]
        fold["feature_ledger_binding"]["series_safety"].pop("validation_series_ids")  # type: ignore[index]
    raw_hash = _write_json(path, model)
    maps_path = tmp_path / "maps.parquet"
    pd.DataFrame(
        {
            "game_uid": ["g1", "g2", "g3"],
            "date": [
                "2026-01-02T12:00:00Z",
                "2026-01-02T12:00:00Z",
                "2026-01-02T12:00:00Z",
            ],
            "y_blue_win": [1, 0, 1],
        }
    ).to_parquet(maps_path)
    _, _, binding = load_prediction_offsets(
        path,
        variant="future_player_form",
        expected_raw_sha256=raw_hash,
        source=SOURCE,
        maps_path=maps_path,
        expected_maps_sha256=sha256_path(maps_path),
    )
    assert binding["chronology"]["status"] == "blocked"  # type: ignore[index]
    assert binding["chronology"]["blockers"] == [  # type: ignore[index]
        "series_disjointness_not_independently_verified"
    ]
    assert binding["blockers"] == [  # type: ignore[index]
        "research_blocker",
        "series_disjointness_not_independently_verified",
    ]


def test_series_identity_mutation_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    model = _model("future_player_form")
    safety = model["variants"]["future_player_form"]["folds"][0]["feature_ledger_binding"]["series_safety"]  # type: ignore[index]
    safety["train_series_ids"] = ["train:forged"]
    raw_hash = _write_json(path, model)
    maps_path = tmp_path / "maps.parquet"
    pd.DataFrame(
        {
            "game_uid": ["g1", "g2", "g3"],
            "date": [
                "2026-01-02T12:00:00Z",
                "2026-01-02T12:00:00Z",
                "2026-01-02T00:00:00Z",
            ],
            "y_blue_win": [1, 0, 1],
        }
    ).to_parquet(maps_path)
    with pytest.raises(FutureValueTierListError, match="train series identity changed"):
        load_prediction_offsets(
            path,
            variant="future_player_form",
            expected_raw_sha256=raw_hash,
            source=SOURCE,
            maps_path=maps_path,
            expected_maps_sha256=sha256_path(maps_path),
        )


def test_series_overlap_fails_closed_even_with_resealed_digests(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    model = _model("future_player_form")
    safety = model["variants"]["future_player_form"]["folds"][0]["feature_ledger_binding"]["series_safety"]  # type: ignore[index]
    safety["train_series_ids"] = ["series:shared"]
    safety["validation_series_ids"] = ["series:shared"]
    safety["train_series_identity_sha256"] = identity_sha256(["series:shared"])
    safety["validation_series_identity_sha256"] = identity_sha256(["series:shared"])
    raw_hash = _write_json(path, model)
    maps_path = tmp_path / "maps.parquet"
    pd.DataFrame(
        {
            "game_uid": ["g1", "g2", "g3"],
            "date": [
                "2026-01-02T12:00:00Z",
                "2026-01-02T12:00:00Z",
                "2026-01-02T12:00:00Z",
            ],
            "y_blue_win": [1, 0, 1],
        }
    ).to_parquet(maps_path)
    with pytest.raises(FutureValueTierListError, match="series IDs overlap"):
        load_prediction_offsets(
            path,
            variant="future_player_form",
            expected_raw_sha256=raw_hash,
            source=SOURCE,
            maps_path=maps_path,
            expected_maps_sha256=sha256_path(maps_path),
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


def _candidate(swapped: bool = False, variant: str = "current_only") -> dict[str, object]:
    first_rank, second_rank = ((2, 1) if swapped else (1, 2))
    candidate: dict[str, object] = {
        "artifact_sha256": "",
        "schema_version": "scryglass:champion-role-elo-candidate:v2",
        "status": "development_only",
        "development_only": True,
        "publication_eligible": False,
        "production_eligible": False,
        "source": {
            "maps_replayed": 2,
            "maps_used_in_joint_likelihood": 2,
            "source_identity_sha256": "7" * 64,
        },
        "pre_map_offset_override": {
            "applied": True,
            "game_count": 2,
            "game_identity_sha256": "7" * 64,
            "offsets_sha256": "5" * 64,
            "provenance": {},
        },
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
    provenance: dict[str, object] = {
        "schema_version": "scryglass:tierlist-pre-map-offset-override:v1",
        "status": "research_only",
        "authority": False,
        "producer": f"future_value_rating:{variant}",
        "timing": "strict_prior_pre_map",
        "source_receipt_sha256": "8" * 64,
        "source_identity_sha256": "7" * 64,
        "source_game_count": 2,
        "offsets_sha256": "5" * 64,
    }
    provenance["receipt_sha256"] = hashlib.sha256(canonical_json_bytes(provenance)).hexdigest()
    candidate["pre_map_offset_override"]["provenance"] = provenance  # type: ignore[index]
    candidate["artifact_sha256"] = hashlib.sha256(
        canonical_json_bytes({key: value for key, value in candidate.items() if key != "artifact_sha256"})
    ).hexdigest()
    return candidate


def test_fourway_diff_uses_exact_stable_identity_and_rank_direction() -> None:
    candidates = {
        variant: _candidate(variant == "future_player_form", variant)
        for variant in VARIANTS
    }
    bindings = {
        variant: {
            "blockers": [],
            "offsets_sha256": "5" * 64,
            "producer": f"future_value_rating:{variant}",
        }
        for variant in VARIANTS
    }
    report = build_fourway_diff(
        candidates,
        source={
            "source_as_of": SOURCE["source_as_of"],
            "source_identity_sha256": "7" * 64,
            "source_receipt_sha256": "8" * 64,
        },
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


def test_fourway_diff_preserves_blocked_chronology_evidence() -> None:
    candidates = {
        variant: _candidate(variant == "future_player_form", variant)
        for variant in VARIANTS
    }
    bindings = {
        variant: {
            "blockers": [],
            "offsets_sha256": "5" * 64,
            "producer": f"future_value_rating:{variant}",
            "chronology": {
                "status": "blocked",
                "blockers": ["series_disjointness_not_independently_verified"],
            },
        }
        for variant in VARIANTS
    }
    report = build_fourway_diff(
        candidates,
        source={
            "source_as_of": SOURCE["source_as_of"],
            "source_identity_sha256": "7" * 64,
            "source_receipt_sha256": "8" * 64,
        },
        universe={"game_count": 2, "game_identity_sha256": "7" * 64},
        model_bindings=bindings,
        trust_manifest_raw_sha256="8" * 64,
        baseline_candidate_raw_sha256="9" * 64,
    )
    assert "series_disjointness_not_independently_verified" in report["blockers"]
    assert report["model_bindings"]["current_only"]["chronology"]["status"] == "blocked"


def test_fourway_diff_rejects_resealed_offset_digest() -> None:
    candidates = {
        variant: _candidate(variant == "future_player_form", variant)
        for variant in VARIANTS
    }
    broken = copy.deepcopy(candidates["future_player_form"])
    override = broken["pre_map_offset_override"]  # type: ignore[index]
    override["offsets_sha256"] = "f" * 64
    override["provenance"]["offsets_sha256"] = "f" * 64
    override["provenance"]["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {
                key: value
                for key, value in override["provenance"].items()
                if key != "receipt_sha256"
            }
        )
    ).hexdigest()
    broken["artifact_sha256"] = hashlib.sha256(
        canonical_json_bytes(
            {key: value for key, value in broken.items() if key != "artifact_sha256"}
        )
    ).hexdigest()
    candidates["future_player_form"] = broken
    bindings = {
        variant: {
            "blockers": [],
            "offsets_sha256": "5" * 64,
            "producer": f"future_value_rating:{variant}",
        }
        for variant in VARIANTS
    }
    with pytest.raises(FutureValueTierListError, match="differ from the verified model"):
        build_fourway_diff(
            candidates,
            source={
                "source_as_of": SOURCE["source_as_of"],
                "source_identity_sha256": "7" * 64,
                "source_receipt_sha256": "8" * 64,
            },
            universe={"game_count": 2, "game_identity_sha256": "7" * 64},
            model_bindings=bindings,
            trust_manifest_raw_sha256="8" * 64,
            baseline_candidate_raw_sha256="9" * 64,
        )
