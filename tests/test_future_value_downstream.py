from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.research.future_value_downstream import (
    EVALUATION_SCHEMA_VERSION,
    REQUIRED_EVALUATION_GATES,
    SCHEMA_VERSION,
    downstream_impact_contract,
    evaluate_downstream_impact,
    required_artifact_specs,
    write_downstream_report,
)


SOURCE = {
    "source_as_of": "2026-08-20T11:31:37Z",
    "source_game_count": 17756,
    "source_identity_sha256": "a" * 64,
}


def _evaluation() -> dict:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        **SOURCE,
        "gates": {name: {"status": "passed"} for name in REQUIRED_EVALUATION_GATES},
        "authority": {
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
        },
    }


def _manifest(*, source: dict | None = None, pack_id: str = "v2026.08.20.194336") -> dict:
    binding = source or SOURCE
    return {
        "pack_id": pack_id,
        **binding,
        "ratings": {
            **binding,
            "team_rating_rows": 2,
            "player_rating_rows": 2,
        },
        "total_files": 22,
        "total_bytes": 100,
    }


def _write_json(root: Path, relative: str, value: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_root(root: Path, *, changed: bool = False, source: dict | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    binding = source or SOURCE
    player_rows = [
        {"player": "A", "mu_total": 100.0 + (1.0 if changed else 0.0), "mu_effective": 101.0, "sigma": 10.0},
        {"player": "B", "mu_total": 90.0, "mu_effective": 91.0, "sigma": 12.0},
    ]
    team_rows = [
        {"team": "Alpha", "team_key": "alpha", "mu_total": 110.0 - (2.0 if changed else 0.0), "mu_effective": 111.0, "sigma": 9.0},
        {"team": "Beta", "team_key": "beta", "mu_total": 95.0, "mu_effective": 96.0, "sigma": 11.0},
    ]
    _write_json(root, "features/player_ratings_snapshot.json", player_rows)
    _write_json(root, "features/ratings_snapshot.json", team_rows)
    _write_json(
        root,
        "rankings/tierlists.json",
        {"rows": [{"champion": "Aatrox", "role": "top", "league": "LCK", "patch": "16.16", "rank": 1, "score": 0.5}]},
    )
    _write_json(
        root,
        "features/draft_records.json",
        {"games": [{"game_uid": "game-1", "draft_edge": 0.1, "base": 0.2}]},
    )
    _write_json(
        root,
        "features/profile_records.json",
        {"games": [{"game_uid": "game-1", "players": [{"player": "A", "mu_total": 100.0}]}]},
    )
    _write_json(
        root,
        "features/match_records_2026_q3.json",
        {"games": [{"game_uid": "game-1", "future_team_value": 0.2, "blue_result": 1}]},
    )
    _write_json(root, "manifest.json", _manifest(source=binding))


def test_complete_impact_report_binds_source_and_reports_deltas(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    candidate_root = tmp_path / "candidate"
    _write_root(old_root)
    _write_root(candidate_root, changed=True)
    _write_json(candidate_root, "future-value-evaluation-receipt.json", _evaluation())

    report = evaluate_downstream_impact(old_root, candidate_root, source_binding=SOURCE)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["status"] == "ready_research_only"
    assert report["source"] == SOURCE
    assert report["artifacts"]["player_ratings"]["comparison"]["matched_rows"] == 2
    assert report["artifacts"]["player_ratings"]["comparison"]["deltas"]["mu_total"]["mean"] == pytest.approx(0.5)
    assert report["artifacts"]["team_ratings"]["comparison"]["deltas"]["mu_total"]["mean"] == pytest.approx(-1.0)
    assert report["deltas"]["player_ratings"]["mu_total"]["changed_count"] == 1
    assert not any(report["authority"].values())


def test_missing_required_artifact_blocks_authority(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    candidate_root = tmp_path / "candidate"
    _write_root(old_root)
    _write_root(candidate_root)
    _write_json(candidate_root, "future-value-evaluation-receipt.json", _evaluation())
    (candidate_root / "features" / "draft_records.json").unlink()

    report = evaluate_downstream_impact(old_root, candidate_root, source_binding=SOURCE)

    assert report["status"] == "blocked"
    assert "candidate_draft_score_missing_or_invalid" in report["blockers"]
    assert not any(report["authority"].values())


def test_source_identity_mismatch_blocks_even_with_complete_outputs(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    candidate_root = tmp_path / "candidate"
    _write_root(old_root)
    changed_source = {**SOURCE, "source_identity_sha256": "b" * 64}
    _write_root(candidate_root, source=changed_source)
    evaluation = _evaluation()
    evaluation["source_identity_sha256"] = changed_source["source_identity_sha256"]
    _write_json(candidate_root, "future-value-evaluation-receipt.json", evaluation)

    report = evaluate_downstream_impact(old_root, candidate_root, source_binding=SOURCE)

    assert report["status"] == "blocked"
    assert "candidate_source_binding_mismatch" in report["blockers"]
    assert not any(report["authority"].values())


def test_missing_evaluation_receipt_blocks_authority(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    candidate_root = tmp_path / "candidate"
    _write_root(old_root)
    _write_root(candidate_root)

    report = evaluate_downstream_impact(old_root, candidate_root, source_binding=SOURCE)

    assert report["status"] == "blocked"
    assert "candidate_evaluation_receipt_missing" in report["blockers"]


def test_failed_evaluation_gate_blocks_authority(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    candidate_root = tmp_path / "candidate"
    _write_root(old_root)
    _write_root(candidate_root)
    evaluation = _evaluation()
    evaluation["gates"]["side_swap_invariance"] = {"status": "failed"}
    _write_json(candidate_root, "future-value-evaluation-receipt.json", evaluation)

    report = evaluate_downstream_impact(old_root, candidate_root, source_binding=SOURCE)

    assert report["status"] == "blocked"
    assert "evaluation_gate_side_swap_invariance_missing_or_failed" in report["blockers"]


def test_duplicate_identity_rows_block_comparison(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    candidate_root = tmp_path / "candidate"
    _write_root(old_root)
    _write_root(candidate_root)
    _write_json(candidate_root, "future-value-evaluation-receipt.json", _evaluation())
    _write_json(
        candidate_root,
        "features/player_ratings_snapshot.json",
        [
            {"player": "A", "mu_total": 100.0},
            {"player": "A", "mu_total": 101.0},
        ],
    )

    report = evaluate_downstream_impact(old_root, candidate_root, source_binding=SOURCE)

    assert report["status"] == "blocked"
    assert "candidate_player_ratings_duplicate_identity_rows" in report["blockers"]


def test_contract_lists_required_downstream_consumers() -> None:
    contract = downstream_impact_contract()
    names = {item["name"] for item in contract["required_artifacts"]}
    assert names == {
        "player_ratings",
        "team_ratings",
        "tierlists",
        "draft_score",
        "profiles",
        "matches",
        "public_manifest",
    }
    assert contract["required_evaluation_gates"] == list(REQUIRED_EVALUATION_GATES)
    assert not any(contract["authority"].values())
    assert len(required_artifact_specs()) == len(contract["required_artifacts"])


def test_frozen_contract_matches_executable_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads(
        (root / "data/lol/v2/evaluation/future-value-downstream-contract-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert frozen == downstream_impact_contract()


def test_report_writer_rejects_wrong_schema_and_writes_valid_report(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema"):
        write_downstream_report(tmp_path / "bad.json", {"schema_version": "wrong"})
    path = tmp_path / "report.json"
    write_downstream_report(path, {"schema_version": SCHEMA_VERSION, "status": "blocked"})
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "blocked"
