from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.research.future_value_downstream import (
    EVALUATION_SCHEMA_VERSION,
    FutureValueDownstreamError,
    REQUIRED_EVALUATION_GATES,
    SCHEMA_VERSION,
    SOURCE_RECEIPT_SCHEMA_VERSION,
    SOURCE_RECEIPT_STATUS,
    downstream_impact_contract,
    evaluate_downstream_impact,
    required_artifact_specs,
    write_downstream_report,
)


def _source_receipt(
    *,
    game_ids: list[str] | None = None,
    source_as_of: str = "2026-08-20T11:31:37Z",
) -> dict:
    ids = sorted(game_ids or ["game-1", "game-2"])
    identity = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    receipt = {
        "schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
        "status": SOURCE_RECEIPT_STATUS,
        "source_as_of": source_as_of,
        "source_game_count": len(ids),
        "source_identity_sha256": identity,
        "accepted_game_ids": ids,
        "source_files": {
            "fixture": {
                "bytes": 1,
                "locator": "fixture.bin",
                "sha256": "0" * 64,
            }
        },
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "deployment": False,
        },
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return receipt


SOURCE = _source_receipt()


def _evaluation(source: dict | None = None) -> dict:
    binding = source or SOURCE
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        **{field: binding[field] for field in ("source_as_of", "source_game_count", "source_identity_sha256")},
        "source_receipt": binding,
        "gates": {name: {"status": "passed"} for name in REQUIRED_EVALUATION_GATES},
        "authority": {
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
        },
    }


def _manifest(*, source: dict | None = None, pack_id: str = "v2026.08.20.194336") -> dict:
    binding = source or SOURCE
    source_fields = {
        field: binding[field]
        for field in ("source_as_of", "source_game_count", "source_identity_sha256")
    }
    return {
        "pack_id": pack_id,
        **source_fields,
        "source_receipt": binding,
        "ratings": {
            **source_fields,
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
    _write_json(root, "future-value-source-receipt.json", binding)


def test_complete_impact_report_binds_source_and_reports_deltas(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    candidate_root = tmp_path / "candidate"
    _write_root(old_root)
    _write_root(candidate_root, changed=True)
    _write_json(candidate_root, "future-value-evaluation-receipt.json", _evaluation())

    report = evaluate_downstream_impact(old_root, candidate_root, source_binding=SOURCE)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["status"] == "ready_research_only"
    assert report["source"]["source_as_of"] == SOURCE["source_as_of"]
    assert report["source"]["accepted_game_ids"] == SOURCE["accepted_game_ids"]
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
    changed_source = _source_receipt(game_ids=["game-1", "game-3"])
    _write_root(candidate_root, source=changed_source)
    _write_json(candidate_root, "future-value-evaluation-receipt.json", _evaluation(changed_source))

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


def test_source_binding_requires_a_complete_verified_receipt() -> None:
    with pytest.raises(FutureValueDownstreamError, match="schema|incomplete"):
        evaluate_downstream_impact(
            Path("old"),
            Path("candidate"),
            source_binding={
                "source_as_of": SOURCE["source_as_of"],
                "source_game_count": SOURCE["source_game_count"],
                "source_identity_sha256": SOURCE["source_identity_sha256"],
            },
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_game_count", 99, "census identity"),
        ("source_identity_sha256", "f" * 64, "census identity"),
        ("source_as_of", "2026-08-21T11:31:37Z", "hash"),
    ],
)
def test_source_receipt_rejects_forged_binding_fields(
    field: str, value: object, message: str
) -> None:
    forged = {**SOURCE, field: value}
    with pytest.raises(FutureValueDownstreamError, match=message):
        evaluate_downstream_impact(
            Path("old"),
            Path("candidate"),
            source_binding=forged,
        )


def test_source_receipt_rejects_forged_source_file_hash() -> None:
    forged = json.loads(json.dumps(SOURCE))
    forged["source_files"]["fixture"]["sha256"] = "f" * 64
    with pytest.raises(FutureValueDownstreamError, match="receipt hash"):
        evaluate_downstream_impact(
            Path("old"),
            Path("candidate"),
            source_binding=forged,
        )


def test_source_receipt_path_checks_explicit_source_file_bytes(tmp_path: Path) -> None:
    source_file = tmp_path / "fixture.bin"
    source_file.write_bytes(b"x")
    receipt = json.loads(json.dumps(SOURCE))
    receipt["source_files"]["fixture"]["path"] = source_file.name
    receipt["source_files"]["fixture"]["bytes"] = 1
    receipt["source_files"]["fixture"]["sha256"] = hashlib.sha256(b"x").hexdigest()
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    evaluate_downstream_impact(
        Path("old"),
        Path("candidate"),
        source_binding=receipt_path,
    )
    source_file.write_bytes(b"changed")
    with pytest.raises(FutureValueDownstreamError, match="source file changed"):
        evaluate_downstream_impact(
            Path("old"),
            Path("candidate"),
            source_binding=receipt_path,
        )


def test_invalid_durable_source_receipt_blocks_report(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    candidate_root = tmp_path / "candidate"
    _write_root(old_root)
    _write_root(candidate_root)
    _write_json(candidate_root, "future-value-evaluation-receipt.json", _evaluation())
    forged = json.loads(json.dumps(SOURCE))
    forged["accepted_game_ids"] = ["game-1", "game-3"]
    _write_json(candidate_root, "future-value-source-receipt.json", forged)

    report = evaluate_downstream_impact(old_root, candidate_root, source_binding=SOURCE)

    assert report["status"] == "blocked"
    assert "candidate_source_receipt_invalid" in report["blockers"]


def test_embedded_source_payload_must_carry_exact_census(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    candidate_root = tmp_path / "candidate"
    _write_root(old_root)
    _write_root(candidate_root)
    evaluation = _evaluation()
    evaluation["accepted_game_ids"] = ["game-1", "game-3"]
    _write_json(candidate_root, "future-value-evaluation-receipt.json", evaluation)

    report = evaluate_downstream_impact(old_root, candidate_root, source_binding=SOURCE)

    assert report["status"] == "blocked"
    assert "evaluation_source_census_invalid" in report["blockers"]
