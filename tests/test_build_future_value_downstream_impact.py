from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.build_future_value_downstream_impact import (
    BOOTSTRAP_COMPARISONS,
    DownstreamImpactError,
    VARIANTS,
    _verify_snapshot_variant,
    build_downstream_impact_report,
    write_report,
)
from lol_kills.research.future_value_snapshots import snapshot_capability_matrix
from lol_kills.research.future_value_rating import (
    SCHEMA_VERSION,
    SOURCE_RECEIPT_AUTHORITY,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


AUTHORITY = {
    "research_only": True,
    "public_player_rating": False,
    "public_team_rating": False,
    "public_probability": False,
    "promotion": False,
    "deployment": False,
    "odds": False,
    "expected_value": False,
    "recommendation": False,
    "betting": False,
}
IDS = ["game-1", "game-2", "game-3"]
ELIGIBLE_IDS = ["game-1", "game-2"]


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _self_hash(value: dict, field: str) -> dict:
    result = dict(value)
    result[field] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")
    return path


def _source(root: Path) -> tuple[Path, dict, dict]:
    source = {
        "schema_version": SCHEMA_VERSION,
        "status": "accepted_source_bound_development_only",
        "source_as_of": "2026-08-20T00:00:00Z",
        "source_game_count": len(IDS),
        "source_identity_sha256": identity_sha256(IDS),
        "model_eligible_game_count": len(ELIGIBLE_IDS),
        "model_eligible_identity_sha256": identity_sha256(ELIGIBLE_IDS),
        "accepted_game_ids": IDS,
        "model_eligible_game_ids": ELIGIBLE_IDS,
        "source_files": {
            name: {"bytes": 1, "locator": f"{name}.bin", "sha256": str(index) * 64}
            for index, name in enumerate(("maps", "players", "teams", "accepted_census"), 1)
        },
        "source_rows": {"maps": 3, "players": 30, "teams": 6},
        "source_extra_game_ids": {},
        "identity_coverage": {"stable_game_ids": 3},
        "checkpoint_coverage": {"goldat10": 1.0},
        "model_exclusions": {},
        "model_contract": {"pregame": True},
        "authority": dict(SOURCE_RECEIPT_AUTHORITY),
    }
    source["receipt_sha256"] = hashlib.sha256(_canonical(source)).hexdigest()
    path = _write(root / "source-receipt.json", source)
    return path, source, {
        "source_as_of": source["source_as_of"],
        "source_game_count": source["source_game_count"],
        "source_identity_sha256": source["source_identity_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
        "model_eligible_game_count": source["model_eligible_game_count"],
        "model_eligible_identity_sha256": source["model_eligible_identity_sha256"],
    }


def _evaluation(root: Path, source: dict, source_fields: dict, variant: str) -> tuple[Path, Path]:
    rows = [
        {"game_id": "game-1", "candidate": 0.6, "actual": 1},
        {"game_id": "game-2", "candidate": 0.4, "actual": 0},
    ]
    ledger = {
        "schema_version": "scryglass:future-value-prediction-ledger:v2",
        "row_count": len(rows),
        "rows": rows,
        "game_identity_sha256": identity_sha256([row["game_id"] for row in rows]),
    }
    ledger["sha256"] = hashlib.sha256(_canonical(rows)).hexdigest()
    payload = {
        "authority": dict(AUTHORITY),
        "status": "development_evaluated",
        "variant": variant,
        "blockers": [],
        "evaluation": {
            "pooled_rows": len(rows),
            "valid_folds": 2,
            "requested_folds": 2,
            "pooled_candidate": {"auc": 0.6, "brier": 0.2, "log_loss": 0.5, "rows": len(rows)},
            "pooled_raw_candidate": {"auc": 0.59, "brier": 0.21, "log_loss": 0.51, "rows": len(rows)},
            "pooled_calibration": {"status": "available", "expected_calibration_error": 0.02, "max_absolute_error": 0.04, "rows": len(rows)},
        },
        "prediction_ledger": ledger,
    }
    model = {
        "schema_version": "scryglass:future-value-four-variant-evaluation:v1",
        "authority": dict(AUTHORITY),
        "source": {
            **source_fields,
            "accepted_game_ids": list(IDS),
            "model_eligible_game_count": source["model_eligible_game_count"],
            "model_eligible_identity_sha256": source["model_eligible_identity_sha256"],
            "source_receipt_file_sha256": hashlib.sha256(Path(root / "source-receipt.json").read_bytes()).hexdigest(),
        },
        "variants": {variant: payload},
    }
    model_path = _write(root / f"{variant}-model.json", model)
    receipt = _self_hash(
        {
            "schema_version": "scryglass:future-value-model-receipt:v1",
            "source_receipt_sha256": source["receipt_sha256"],
            "authority": dict(AUTHORITY),
            "artifact": {
                "path": str(model_path),
                "bytes": model_path.stat().st_size,
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            },
        },
        "receipt_sha256",
    )
    receipt_path = _write(root / f"{variant}-receipt.json", receipt)
    return model_path, receipt_path


def _snapshot_bundle(root: Path, source: dict, sf: dict) -> tuple[Path, Path, Path]:
    player = _write(root / "player-snapshot.json", {"rows": [{"player_id": "p1"}], "source_receipt_sha256": source["receipt_sha256"], "authority": dict(AUTHORITY)})
    team = _write(root / "team-snapshot.json", {"rows": [{"team_id": "t1"}], "source_receipt_sha256": source["receipt_sha256"], "authority": dict(AUTHORITY)})
    player_rank = _write(
        root / "player-rank-diffs.json",
        {
            "schema_version": "scryglass:future-value-snapshot:v1",
            "source_receipt_sha256": source["receipt_sha256"],
            "authority": dict(AUTHORITY),
            "status": "research_only_partial",
            "rows": [
                {"player_id": "p1", "current_rank": 2, "future_rank": 1, "rank_delta": 1}
            ],
        },
    )
    team_rank = _write(
        root / "team-rank-diffs.json",
        {
            "schema_version": "scryglass:future-value-snapshot:v1",
            "source_receipt_sha256": source["receipt_sha256"],
            "authority": dict(AUTHORITY),
            "status": "research_only_partial",
            "rows": [
                {"team_id": "t1", "current_rank": 1, "future_rank": 2, "rank_delta": -1}
            ],
        },
    )
    current_player = _write(root / "current-player.json", {"rows": [{"player_id": "p1"}]})
    current_team = _write(root / "current-team.json", {"rows": [{"team_id": "t1"}]})
    snapshot = {
        "schema_version": "scryglass:future-value-snapshot-receipt:v1",
        "status": "research_only_partial",
        "authority": dict(AUTHORITY),
        "source": {
            **sf,
            "model_eligible_game_count": source["model_eligible_game_count"],
            "model_eligible_identity_sha256": source["model_eligible_identity_sha256"],
        },
        "current_rating_inputs": {
            "schema_version": "scryglass:future-value-current-rating-input-binding:v1",
            **sf,
            "snapshots": {
                "player": {"path": str(current_player), "bytes": current_player.stat().st_size, "sha256": hashlib.sha256(current_player.read_bytes()).hexdigest()},
                "team": {"path": str(current_team), "bytes": current_team.stat().st_size, "sha256": hashlib.sha256(current_team.read_bytes()).hexdigest()},
            },
        },
        "player_row_count": 1,
        "team_row_count": 1,
        "player_rank_diff_count": 1,
        "team_rank_diff_count": 1,
        "rank_coverage": {
            "player": {"matched_rows": 1, "rank_direction": "descending", "rank_universe": "common_verified_finite_ids", "full_snapshot_rank_status": "incomparable"},
            "team": {"matched_rows": 1, "rank_direction": "descending", "rank_universe": "common_verified_finite_ids", "full_snapshot_rank_status": "incomparable"},
        },
        "blockers": ["research_only"],
        "fit": {"status": "blocked"},
        "model": {"variant": "future_player_form"},
        "tierlists": {"status": "unchanged"},
    }
    snapshot["receipt_sha256"] = hashlib.sha256(_canonical(snapshot)).hexdigest()
    snapshot_path = _write(root / "snapshot-receipt.json", snapshot)
    comparison = _self_hash(
        {
            "schema_version": "scryglass:future-value-snapshot-comparison:v1",
            "status": "research_only_partial",
            "authority": dict(AUTHORITY),
            "source": dict(sf),
            "full_snapshot_rank_status": "incomparable",
            "independent_join": {"status": "verified"},
            "snapshot_comparisons": {
                "player": {"matched_rows": 1, "common_universe_size": 1},
                "team": {"matched_rows": 1, "common_universe_size": 1},
            },
            "blockers": [],
        },
        "report_sha256",
    )
    comparison_path = _write(root / "snapshot-comparison.json", comparison)
    files = {}
    for name, path in {
        "player_snapshot": player,
        "team_snapshot": team,
        "player_rank_diffs": player_rank,
        "team_rank_diffs": team_rank,
        "receipt": snapshot_path,
    }.items():
        files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = _self_hash(
        {
            "schema_version": "scryglass:future-value-snapshot:v1",
            "status": "research_only_partial",
            "authority": dict(AUTHORITY),
            "source_receipt_sha256": source["receipt_sha256"],
            "files": files,
        },
        "manifest_sha256",
    )
    manifest_path = _write(root / "manifest.json", manifest)
    return snapshot_path, comparison_path, manifest_path


def _all_variant_snapshot_bundles(root: Path, source: dict, sf: dict) -> dict[str, tuple[Path, Path]]:
    """Create compact source-bound bundles for the all-variant verifier tests."""

    source_binding = {
        **sf,
        "model_eligible_game_count": source["model_eligible_game_count"],
        "model_eligible_identity_sha256": source["model_eligible_identity_sha256"],
    }
    matrix = snapshot_capability_matrix()
    outputs: dict[str, tuple[Path, Path]] = {}
    for variant in VARIANTS:
        variant_root = root / variant
        player_snapshot = _write(
            variant_root / "future-player-value-snapshot.json",
            {"rows": [] if variant == "scaling_curve" else [{"player_id": "p1"}]},
        )
        team_snapshot = _write(
            variant_root / "future-team-value-snapshot.json",
            {"rows": [] if variant == "scaling_curve" else [{"team_id": "t1"}]},
        )
        rows = [] if variant == "scaling_curve" else [{"player_id": "p1", "rank_delta": 0}]
        team_rows = [] if variant == "scaling_curve" else [{"team_id": "t1", "rank_delta": 0}]
        player_rank = _write(
            variant_root / "future-player-rank-diffs.json",
            {
                "schema_version": "scryglass:future-value-snapshot:v1",
                "source_receipt_sha256": source["receipt_sha256"],
                "authority": dict(AUTHORITY),
                "rows": rows,
            },
        )
        team_rank = _write(
            variant_root / "future-team-rank-diffs.json",
            {
                "schema_version": "scryglass:future-value-snapshot:v1",
                "source_receipt_sha256": source["receipt_sha256"],
                "authority": dict(AUTHORITY),
                "rows": team_rows,
            },
        )
        coverage = {
            "player": {
                "status": "not_applicable" if variant == "scaling_curve" else "complete",
                "row_policy": "no_rows" if variant == "scaling_curve" else "common_verified_finite_ids",
                "rank_universe": "common_verified_finite_ids",
            },
            "team": {
                "status": "not_applicable" if variant == "scaling_curve" else "complete",
                "row_policy": "no_rows" if variant == "scaling_curve" else "common_verified_finite_ids",
                "rank_universe": "common_verified_finite_ids",
            },
        }
        receipt = {
            "schema_version": "scryglass:future-value-snapshot-receipt:v1",
            "capability_schema_version": "scryglass:future-value-snapshot-capability:v1",
            "status": "research_only",
            "authority": dict(AUTHORITY),
            "variant": variant,
            "source": source_binding,
            "capability": matrix[variant],
            "player_row_count": 0 if variant == "scaling_curve" else 1,
            "team_row_count": 0 if variant == "scaling_curve" else 1,
            "player_rank_diff_count": len(rows),
            "team_rank_diff_count": len(team_rows),
            "rank_coverage": coverage,
            "blockers": [],
        }
        receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
        receipt_path = _write(variant_root / "future-value-snapshot-receipt.json", receipt)
        files = {
            "player_snapshot": player_snapshot,
            "team_snapshot": team_snapshot,
            "player_rank_diffs": player_rank,
            "team_rank_diffs": team_rank,
            "receipt": receipt_path,
        }
        manifest = {
            "schema_version": "scryglass:future-value-snapshot:v1",
            "capability_schema_version": "scryglass:future-value-snapshot-capability:v1",
            "status": "research_only",
            "authority": dict(AUTHORITY),
            "variant": variant,
            "capability": matrix[variant],
            "source_receipt_sha256": source["receipt_sha256"],
            "files": {
                name: {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in files.items()
            },
            "blockers": [],
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        manifest_path = _write(variant_root / "manifest.json", manifest)
        outputs[variant] = (receipt_path, manifest_path)
    return outputs


def _tier(root: Path, source: dict, sf: dict) -> tuple[Path, Path]:
    key = {"scope_id": "s", "patch": "p", "role": "mid", "champion_id": "c"}
    rows = [{"key": key, "baseline": {"rank": 1}, "v2": {"rank": 2}, "delta": {"rank_delta": -1, "tier_changed": True}}]
    report = {
        "schema_version": "scryglass:future-value-tierlist-full-census-diff:v1",
        "status": "research_only",
        "authority": {**AUTHORITY, "public_tierlist": False},
        "source": {
            "accepted_game_count": source["source_game_count"],
            "accepted_identity_sha256": source["source_identity_sha256"],
            "model_eligible_game_count": source["model_eligible_game_count"],
            "model_eligible_identity_sha256": source["model_eligible_identity_sha256"],
            "source_as_of": source["source_as_of"],
            "source_receipt_sha256": source["receipt_sha256"],
        },
        "comparison": {
            "reference": "current_only",
            "candidate": "future_player_form",
            "common_row_count": 1,
            "baseline_only_row_count": 0,
            "v2_only_row_count": 0,
            "changed_rank_count": 1,
            "changed_tier_count": 1,
            "mean_absolute_rank_movement": 1.0,
            "maximum_absolute_rank_movement": 1,
            "common_identity_sha256": hashlib.sha256(_canonical([key])).hexdigest(),
            "paired_rows_sha256": hashlib.sha256(_canonical(rows)).hexdigest(),
        },
        "inputs": {},
        "rows": rows,
        "blockers": ["retrospective_full_census_model_fit_not_chronological_evaluation"],
    }
    report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    report_path = _write(root / "tier-diff.json", report)
    receipt = _self_hash(
        {
            "schema_version": "scryglass:future-value-tierlist-full-census-diff-receipt:v1",
            "status": "research_only",
            "authority": report["authority"],
            "source": report["source"],
            "report": {"path": str(report_path), "bytes": report_path.stat().st_size, "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(), "report_sha256": report["report_sha256"]},
        },
        "receipt_sha256",
    )
    receipt_path = _write(root / "tier-receipt.json", receipt)
    return report_path, receipt_path


def _draft(root: Path, source: dict, sf: dict) -> Path:
    variants = {
        variant: {
            "status": "evaluated",
            "valid_fold_count": 2,
            "folds": [{"fold": 1, "status": "evaluated"}, {"fold": 2, "status": "evaluated"}],
            "feature_names": [variant],
            "producer_requirements": ["current_rating"],
        }
        for variant in VARIANTS
    }
    report = {
        "schema_version": "scryglass:future-value-draft-score-fourway:v1",
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "source": dict(sf),
        "coverage": {"accepted_game_count": source["source_game_count"], "descriptive_subset_game_count": source["source_game_count"], "folds": []},
        "static_atom": {},
        "variants": variants,
        "blockers": [],
        "claim_ceiling": "research",
    }
    report["report_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    return _write(root / "draft-score.json", report)


def _bootstrap(root: Path, source: dict, summaries: dict[str, dict]) -> Path:
    identity = summaries["current_only"]["coverage"]["ledger_identity_sha256"]
    rows = summaries["current_only"]["coverage"]["ledger_rows"]
    comparisons = {
        name: {"draws_accepted": 4, "draws_rejected": 0, "metrics": {"auc": {}, "brier": {}, "log_loss": {}}, "rows": rows}
        for name in BOOTSTRAP_COMPARISONS
    }
    value = {
        "schema_version": "scryglass:future-value-paired-uncertainty:v1",
        "status": "research_only",
        "authority": dict(AUTHORITY),
        "source": {**source, "source_receipt_file_sha256": "ignored"},
        "coverage": {"rows": rows, "game_identity_sha256": identity, "folds": {}, "series_count": 1},
        "comparisons": comparisons,
        "method": {"draws": 4},
        "bindings": {},
    }
    return _write(root / "bootstrap.json", value)


def _build_fixture(tmp_path: Path):
    source_path, source, sf = _source(tmp_path)
    evaluations = {}
    receipts = {}
    for variant in VARIANTS:
        evaluations[variant], receipts[variant] = _evaluation(tmp_path, source, sf, variant)
    snapshot_receipt, snapshot_comparison, snapshot_manifest = _snapshot_bundle(tmp_path, source, sf)
    tier, tier_receipt = _tier(tmp_path, source, sf)
    draft = _draft(tmp_path, source, sf)
    summaries = {variant: {"coverage": {"ledger_rows": 2, "ledger_identity_sha256": identity_sha256(ELIGIBLE_IDS)}} for variant in VARIANTS}
    bootstrap = _bootstrap(tmp_path, source, summaries)
    return locals()


def test_report_binds_dynamic_universes_and_preserves_research_authority(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    report = build_downstream_impact_report(
        source_receipt=fixture["source_path"],
        evaluations=fixture["evaluations"],
        evaluation_receipts=fixture["receipts"],
        snapshot_receipt=fixture["snapshot_receipt"],
        snapshot_comparison=fixture["snapshot_comparison"],
        snapshot_manifest=fixture["snapshot_manifest"],
        tier_diff=fixture["tier"],
        tier_receipt=fixture["tier_receipt"],
        draft_score_report=fixture["draft"],
        paired_uncertainty=fixture["bootstrap"],
    )
    assert report["status"] == "ready_research_only"
    assert report["source"]["source_game_count"] == 3
    assert report["evaluations"]["variants"]["both"]["metrics"]["auc"] == pytest.approx(0.6)
    assert report["snapshots"]["rank_movement"]["player"]["rows"] == 1
    assert report["tierlist"]["comparison"]["common_row_count"] == 1
    assert report["draft_score"]["coverage"]["descriptive_subset_game_count"] == 3
    assert set(report["variant_impacts"]) == set(VARIANTS)
    assert all("metrics" in report["variant_impacts"][variant]["prediction"] for variant in VARIANTS)
    assert all("tier" in report["variant_impacts"][variant] and "draft" in report["variant_impacts"][variant] for variant in VARIANTS)
    assert report["variant_impacts"]["scaling_curve"]["snapshot"]["intrinsic_rank_status"] == "not_applicable"
    assert not any(value is True for value in report["authority"].values() if isinstance(value, bool) and value is not report["authority"].get("research_only"))
    assert all(value is False for key, value in report["downstream_public_change_flags"].items() if key != "measured_changes_present")


def test_all_variant_snapshot_verifier_accepts_typed_scaling_na(tmp_path: Path) -> None:
    _source_path, source, sf = _source(tmp_path)
    bundles = _all_variant_snapshot_bundles(tmp_path / "snapshots", source, sf)

    summary, blockers = _verify_snapshot_variant(
        *bundles["scaling_curve"],
        variant="scaling_curve",
        source={
            **sf,
            "model_eligible_game_count": source["model_eligible_game_count"],
            "model_eligible_identity_sha256": source["model_eligible_identity_sha256"],
        },
    )

    assert blockers == []
    assert summary["rank_movement"]["player"]["status"] == "not_applicable"
    assert summary["rank_movement"]["team"]["rows"] == 0


def test_all_variant_snapshot_verifier_blocks_source_mismatch(tmp_path: Path) -> None:
    source_path, source, sf = _source(tmp_path)
    bundles = _all_variant_snapshot_bundles(tmp_path / "snapshots", source, sf)
    wrong_source = {
        **sf,
        "source_game_count": source["source_game_count"] + 1,
        "model_eligible_game_count": source["model_eligible_game_count"],
        "model_eligible_identity_sha256": source["model_eligible_identity_sha256"],
    }

    _, blockers = _verify_snapshot_variant(
        *bundles["current_only"],
        variant="current_only",
        source=wrong_source,
    )

    assert any("current_only_snapshot_source_game_count_mismatch" in blocker for blocker in blockers)


def test_model_bytes_mutation_is_rejected_by_external_receipt(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    path = fixture["evaluations"]["current_only"]
    model = json.loads(path.read_text())
    model["variants"]["current_only"]["evaluation"]["pooled_candidate"]["auc"] = 0.9
    path.write_bytes(_canonical(model) + b"\n")
    report = build_downstream_impact_report(
            source_receipt=fixture["source_path"],
            evaluations=fixture["evaluations"],
            evaluation_receipts=fixture["receipts"],
            snapshot_receipt=fixture["snapshot_receipt"],
            snapshot_comparison=fixture["snapshot_comparison"],
            snapshot_manifest=fixture["snapshot_manifest"],
            tier_diff=fixture["tier"],
            tier_receipt=fixture["tier_receipt"],
            draft_score_report=fixture["draft"],
            paired_uncertainty=fixture["bootstrap"],
        )
    assert report["status"] == "blocked"
    assert any("current_only_evaluation_receipt_artifact" in blocker for blocker in report["blockers"])


def test_symlinked_snapshot_file_blocks_closed_report(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    rank_path = fixture["snapshot_manifest"]
    manifest = json.loads(rank_path.read_text())
    target = Path(manifest["files"]["player_rank_diffs"]["path"])
    replacement = target.with_name("player-rank-diffs-real.json")
    replacement.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(replacement)
    report = build_downstream_impact_report(
            source_receipt=fixture["source_path"],
            evaluations=fixture["evaluations"],
            evaluation_receipts=fixture["receipts"],
            snapshot_receipt=fixture["snapshot_receipt"],
            snapshot_comparison=fixture["snapshot_comparison"],
            snapshot_manifest=fixture["snapshot_manifest"],
            tier_diff=fixture["tier"],
            tier_receipt=fixture["tier_receipt"],
            draft_score_report=fixture["draft"],
            paired_uncertainty=fixture["bootstrap"],
        )
    assert report["status"] == "blocked"
    assert any("snapshot player rank diff contains a symlink" in blocker for blocker in report["blockers"])


def test_missing_bootstrap_is_a_blocker_and_public_flags_stay_false(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    report = build_downstream_impact_report(
        source_receipt=fixture["source_path"],
        evaluations=fixture["evaluations"],
        evaluation_receipts=fixture["receipts"],
        snapshot_receipt=fixture["snapshot_receipt"],
        snapshot_comparison=fixture["snapshot_comparison"],
        snapshot_manifest=fixture["snapshot_manifest"],
        tier_diff=fixture["tier"],
        tier_receipt=fixture["tier_receipt"],
        draft_score_report=fixture["draft"],
        paired_uncertainty=None,
    )
    assert report["status"] == "blocked"
    assert "paired_bootstrap_receipt_missing" in report["blockers"]
    assert all(value is False for key, value in report["downstream_public_change_flags"].items() if key != "measured_changes_present")


def test_write_report_is_deterministic_json_surface(tmp_path: Path) -> None:
    output = write_report(tmp_path / "report.json", {"schema_version": "test", "status": "blocked"})
    assert output.read_text(encoding="utf-8").endswith("\n")
