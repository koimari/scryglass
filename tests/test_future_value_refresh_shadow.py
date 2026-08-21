from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.research.future_value_refresh_shadow import (
    FutureValueShadowError,
    FutureValueShadowPromotionError,
    run_future_value_refresh_shadow,
    reject_unauthorized_promotion,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


def _inputs(tmp_path: Path) -> dict[str, object]:
    ids = ["game-2", "game-1"]
    receipt = tmp_path / "accepted-source.json"
    receipt_payload = {
        "schema": "scryglass:oe-source-refresh:v1",
        "status": "refreshed",
        "year": "2026",
        "candidate": {"raw_sha256": "a" * 64},
        "authority": {
            "model_validation_authority": False,
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
    }
    receipt_payload["receipt_canonical_sha256"] = hashlib.sha256(
        json.dumps(
            receipt_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    artifacts: dict[str, str] = {}
    for name in ("model", "snapshot", "tier", "draft"):
        path = tmp_path / f"{name}.json"
        path.write_text(name, encoding="utf-8")
        artifacts[name] = str(path)
    return {
        "source_as_of": "2026-08-20T14:51:29Z",
        "source_game_ids": ids,
        "source_game_count": 2,
        "source_identity_sha256": identity_sha256(ids),
        "source_receipt_sha256": receipt_payload["receipt_canonical_sha256"],
        "accepted_source_receipt_path": receipt,
        "current_ratings": {
            "status": "published",
            "pack_id": "v2026.08.21.120000",
            "source_identity_sha256": identity_sha256(ids),
        },
        "artifacts": artifacts,
        "checked_at": NOW,
    }


def test_shadow_receipt_binds_source_and_artifacts_without_public_writes(tmp_path: Path) -> None:
    public_root = tmp_path / "public"
    public_root.mkdir()
    before = hashlib.sha256(b"public").hexdigest()
    public_file = public_root / "manifest.json"
    public_file.write_text("public", encoding="utf-8")

    result = run_future_value_refresh_shadow(runtime_root=tmp_path, **_inputs(tmp_path))

    assert result["status"] == "research_only_available"
    assert result["authority"]["research_only"] is True
    assert all(value is False for key, value in result["authority"].items() if key != "research_only")
    assert result["writes_public_artifacts"] is False
    assert result["stage_or_activation"] is False
    assert result["source"]["accepted_game_ids"] == ["game-1", "game-2"]
    assert result["coverage"]["accepted_game_count"] == 2
    assert result["coverage"]["artifact_count"] == 4
    receipt_path = Path(str(result["receipt_path"]))
    assert receipt_path.is_file()
    assert receipt_path.is_relative_to(tmp_path)
    assert hashlib.sha256(public_file.read_bytes()).hexdigest() == before
    saved = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert saved["receipt_sha256"] == result["receipt_sha256"]
    unsigned = dict(saved)
    claimed = unsigned.pop("receipt_sha256")
    assert claimed == hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_missing_artifacts_write_a_durable_blocked_receipt(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    values["artifacts"] = {}
    result = run_future_value_refresh_shadow(runtime_root=tmp_path, **values)

    assert result["status"] == "research_only_blocked"
    assert set(result["blockers"]) == {
        "draft_artifact_missing",
        "model_artifact_missing",
        "snapshot_artifact_missing",
        "tier_artifact_missing",
    }
    assert Path(str(result["receipt_path"])).is_file()


def test_source_identity_and_hash_drift_stays_blocked(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    values["source_identity_sha256"] = "b" * 64
    values["current_ratings"] = {"status": "published", "source_identity_sha256": "c" * 64}
    result = run_future_value_refresh_shadow(runtime_root=tmp_path, **values)

    assert result["status"] == "research_only_blocked"
    assert "accepted_source_identity_mismatch" in result["blockers"]
    assert "current_ratings_source_identity_mismatch" in result["blockers"]


def test_missing_current_rating_identity_stays_blocked(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    values["current_ratings"] = {"status": "published", "pack_id": "fixture"}

    result = run_future_value_refresh_shadow(runtime_root=tmp_path, **values)

    assert result["status"] == "research_only_blocked"
    assert "current_ratings_source_identity_missing" in result["blockers"]


def test_artifact_hash_mismatch_stays_blocked_and_does_not_copy(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    artifact = tmp_path / "model.json"
    values["artifacts"] = {
        **values["artifacts"],  # type: ignore[dict-item]
        "model": {"path": str(artifact), "sha256": "f" * 64},
    }
    result = run_future_value_refresh_shadow(runtime_root=tmp_path, **values)

    assert result["status"] == "research_only_blocked"
    assert "model_artifact_hash_mismatch" in result["blockers"]
    assert artifact.read_text(encoding="utf-8") == "model"


def test_hash_only_artifact_is_unverified_and_blocked(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    values["artifacts"] = {
        **values["artifacts"],  # type: ignore[dict-item]
        "model": "f" * 64,
    }

    result = run_future_value_refresh_shadow(runtime_root=tmp_path, **values)

    assert result["status"] == "research_only_blocked"
    assert result["artifacts"]["model"]["status"] == "unverified_hash"
    assert "model_artifact_unverified" in result["blockers"]


def test_tampered_accepted_source_receipt_is_blocked(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    receipt = Path(str(values["accepted_source_receipt_path"]))
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["year"] = "2025"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    result = run_future_value_refresh_shadow(runtime_root=tmp_path, **values)

    assert result["status"] == "research_only_blocked"
    assert "accepted_source_receipt_unavailable" in result["blockers"]


@pytest.mark.parametrize("run_id", ["../escape", "nested/run", "/tmp/escape", ".", ".."])
def test_run_id_is_a_safe_basename(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(FutureValueShadowError, match="run_id is unsafe"):
        run_future_value_refresh_shadow(
            runtime_root=tmp_path,
            run_id=run_id,
            **_inputs(tmp_path),
        )
    assert not (tmp_path.parent / "escape" / "future-value-shadow-receipt.json").exists()


def test_artifact_leaf_and_ancestor_symlinks_are_blocked(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    target = tmp_path / "model.json"
    leaf = tmp_path / "model-link.json"
    leaf.symlink_to(target)
    values["artifacts"] = {**values["artifacts"], "model": str(leaf)}  # type: ignore[dict-item]

    leaf_result = run_future_value_refresh_shadow(
        runtime_root=tmp_path, run_id="leaf-link", **values
    )

    assert "model_artifact_unavailable" in leaf_result["blockers"]

    real_dir = tmp_path / "real-artifacts"
    real_dir.mkdir()
    nested_target = real_dir / "model.json"
    nested_target.write_text("model", encoding="utf-8")
    linked_dir = tmp_path / "linked-artifacts"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    values["artifacts"] = {
        **values["artifacts"],  # type: ignore[dict-item]
        "model": str(linked_dir / "model.json"),
    }

    ancestor_result = run_future_value_refresh_shadow(
        runtime_root=tmp_path, run_id="ancestor-link", **values
    )

    assert "model_artifact_unavailable" in ancestor_result["blockers"]


def test_accepted_receipt_symlink_is_blocked(tmp_path: Path) -> None:
    values = _inputs(tmp_path)
    receipt = Path(str(values["accepted_source_receipt_path"]))
    linked = tmp_path / "accepted-source-link.json"
    linked.symlink_to(receipt)
    values["accepted_source_receipt_path"] = linked

    result = run_future_value_refresh_shadow(
        runtime_root=tmp_path, run_id="source-link", **values
    )

    assert "accepted_source_receipt_unavailable" in result["blockers"]


def test_runtime_output_ancestor_symlink_is_rejected(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (runtime_root / "data").symlink_to(outside, target_is_directory=True)

    with pytest.raises(FutureValueShadowError, match="contains a symlink"):
        run_future_value_refresh_shadow(
            runtime_root=runtime_root,
            run_id="safe-name",
            **_inputs(input_root),
        )
    assert not list(outside.rglob("future-value-shadow-receipt.json"))


def test_duplicate_source_ids_are_rejected_without_silent_deduplication(
    tmp_path: Path,
) -> None:
    values = _inputs(tmp_path)
    values["source_game_ids"] = ["game-1", "game-1", "game-2"]
    values["source_game_count"] = 3

    result = run_future_value_refresh_shadow(runtime_root=tmp_path, **values)

    assert result["status"] == "research_only_blocked"
    assert "accepted_source_census_duplicate_ids" in result["blockers"]


def test_explicit_promotion_is_rejected_before_any_shadow_work() -> None:
    with pytest.raises(FutureValueShadowPromotionError, match="independent authorization"):
        reject_unauthorized_promotion("future_player_form")
