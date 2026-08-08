"""Tests for the source-mode-aware tier-list refresh worker."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lol_kills.v2.tierlists import live_refresh


def _candidate(*, source_mode: str) -> dict[str, object]:
    return {
        "artifact_sha256": "c" * 64,
        "source_mode": source_mode,
        "as_of": "2026-08-08T12:00:00Z",
        "source_complete_through_expected_live_as_of": True,
        "source": {
            "maps_replayed": 10,
            "maps_in_live_window": 2,
        },
    }


def test_blob_publication_payloads_keep_cells_immutable_and_pointer_last() -> None:
    root = Path(__file__).resolve().parents[3]
    payloads = live_refresh._publication_payloads(root)
    pointer = json.loads(payloads["pointer_raw"])
    release_index = json.loads(payloads["release_index_raw"])

    assert payloads["release_index_path"].startswith(
        f"tierlists/releases/{payloads['release_id']}/"
    )
    assert pointer["base_url"] == f"./releases/{payloads['release_id']}/"
    assert release_index["base_url"] == "./"
    assert all(cell["locator"].startswith("cells/") for cell in pointer["cells"])
    assert payloads["pointer_raw"] != payloads["release_index_raw"]
    assert payloads["cell_count"] == 285


def test_blob_publication_writes_the_pointer_last() -> None:
    root = Path(__file__).resolve().parents[3]

    class FakeTransport:
        pointer_raw = b""

        def get_blob(self, _store_id: str, pathname: str, *, deadline_epoch: int):
            if pathname != live_refresh.BLOB_POINTER_PATH:
                return None
            return (
                self.pointer_raw,
                live_refresh.BlobIdentity(pathname, len(self.pointer_raw), "pointer-etag"),
            )

    transport = FakeTransport()

    class FakeExecutor:
        def __init__(self, _transport: FakeTransport):
            pass

        def execute(self, plan):
            assert plan.writes[-1].pathname == live_refresh.BLOB_POINTER_PATH
            assert plan.writes[-1].mode is live_refresh.WriteMode.NEW_IMMUTABLE
            assert all(
                write.pathname.startswith("tierlists/releases/")
                for write in plan.writes[:-1]
            )
            transport.pointer_raw = plan.writes[-1].content
            return SimpleNamespace(
                success=True,
                state=SimpleNamespace(value="normal"),
                current_retained_bytes=1,
                peak_retained_bytes=2,
                projected_final_bytes=2,
                actual_final_bytes=2,
                policy_sha256="p" * 64,
                operations=(),
            )

    with patch.object(
        live_refresh,
        "_publication_credentials",
        return_value=("https://store-test.public.blob.vercel-storage.com", "token", "store-test"),
    ), patch.object(live_refresh, "VercelBlobTransport", return_value=transport), patch.object(
        live_refresh, "_blob_inventory", return_value={}
    ), patch.object(live_refresh, "RetentionExecutor", FakeExecutor):
        result = live_refresh.publish_production_bundle(root)

    assert result["status"] == "published"
    assert result["pointer_mode"] == "NEW_IMMUTABLE"
    assert result["pointer_readback_verified"] is True
    assert result["cell_count"] == 285


def test_oe_only_skips_grid_and_can_be_ready_from_a_complete_oe_source(tmp_path: Path) -> None:
    oe_step = {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0}
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_api_meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(
        json.dumps({"source_latest": "2026-08-08T12:00:00Z", "player_detail_complete": True}),
        encoding="utf-8",
    )
    with patch.object(live_refresh, "_run_step", return_value=oe_step) as run_step, patch.object(
        live_refresh,
        "build_candidate",
        return_value=_candidate(source_mode="oe_only"),
    ), patch.object(live_refresh, "write_candidate", return_value="b" * 64):
        receipt = live_refresh.refresh_candidate(
            tmp_path,
            expected_live_as_of="2026-08-08T12:00:00Z",
            output_path=tmp_path / "candidate.json",
            receipt_path=tmp_path / "receipt.json",
            source_mode="oe_only",
        )

    assert run_step.call_count == 5
    assert "--skip-grid" in run_step.call_args_list[0].args[1]
    assert run_step.call_args_list[1].kwargs["source"] == "oe_api"
    assert run_step.call_args_list[2].kwargs["source"] == "champion_atomization"
    assert run_step.call_args_list[3].kwargs["source"] == "oe_live_source"
    assert run_step.call_args_list[4].kwargs["source"] == "ratings"
    assert receipt["source_mode"] == "oe_only"
    assert receipt["status"] == "ready_for_authority_review"
    assert receipt["source_steps"][5]["skipped"] is True
    assert receipt["source_steps"][5]["reason"] == "source_mode_oe_only"
    saved = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert saved["source_mode"] == "oe_only"


def test_oe_plus_grid_keeps_the_grid_step_available(tmp_path: Path) -> None:
    steps = [
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0},
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0},
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0},
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0},
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0},
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0},
    ]
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_api_meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(
        json.dumps({"source_latest": "2026-08-08T12:00:00Z", "player_detail_complete": True}),
        encoding="utf-8",
    )
    with patch.object(live_refresh, "_run_step", side_effect=steps) as run_step, patch.object(
        live_refresh,
        "build_candidate",
        return_value=_candidate(source_mode="oe_plus_grid"),
    ), patch.object(live_refresh, "write_candidate", return_value="b" * 64):
        receipt = live_refresh.refresh_candidate(
            tmp_path,
            expected_live_as_of="2026-08-08T12:00:00Z",
            output_path=tmp_path / "candidate.json",
            receipt_path=tmp_path / "receipt.json",
            source_mode="oe_plus_grid",
        )

    assert run_step.call_count == 6
    assert receipt["source_mode"] == "oe_plus_grid"
    assert run_step.call_args_list[1].kwargs["source"] == "oe_api"
    assert run_step.call_args_list[2].kwargs["source"] == "champion_atomization"
    assert run_step.call_args_list[3].kwargs["source"] == "oe_live_source"
    assert run_step.call_args_list[4].kwargs["source"] == "ratings"
    assert receipt["source_steps"][5]["completed"] is True


def test_skip_annual_oe_uses_the_committed_pack_baseline(tmp_path: Path) -> None:
    steps = [
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0}
        for _ in range(4)
    ]
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_api_meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(
        json.dumps({"source_latest": "2026-08-08T12:00:00Z", "player_detail_complete": True}),
        encoding="utf-8",
    )
    with patch.object(live_refresh, "_run_step", side_effect=steps) as run_step, patch.object(
        live_refresh,
        "build_candidate",
        return_value=_candidate(source_mode="oe_only"),
    ), patch.object(live_refresh, "write_candidate", return_value="b" * 64):
        receipt = live_refresh.refresh_candidate(
            tmp_path,
            expected_live_as_of="2026-08-08T12:00:00Z",
            output_path=tmp_path / "candidate.json",
            receipt_path=tmp_path / "receipt.json",
            source_mode="oe_only",
            skip_annual_oe=True,
        )

    assert run_step.call_count == 4
    assert receipt["source_steps"][0]["source"] == "oe_annual"
    assert receipt["source_steps"][0]["skipped"] is True
    assert receipt["source_steps"][0]["reason"] == "committed_public_pack_baseline"
    assert run_step.call_args_list[0].kwargs["source"] == "oe_api"


def test_promote_runs_evaluation_authority_and_bundle_after_source_refresh(tmp_path: Path) -> None:
    steps = [
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0}
        for _ in range(8)
    ]
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_api_meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(
        json.dumps({"source_latest": "2026-08-08T12:00:00Z", "player_detail_complete": True}),
        encoding="utf-8",
    )
    with patch.object(live_refresh, "_run_step", side_effect=steps) as run_step, patch.object(
        live_refresh,
        "build_candidate",
        return_value=_candidate(source_mode="oe_only"),
    ), patch.object(live_refresh, "write_candidate", return_value="b" * 64), patch.object(
        live_refresh,
        "publish_production_bundle",
        return_value={"status": "published", "cell_count": 285},
    ) as publish:
        receipt = live_refresh.refresh_candidate(
            tmp_path,
            expected_live_as_of="2026-08-08T12:00:00Z",
            receipt_path=tmp_path / "receipt.json",
            source_mode="oe_only",
            promote=True,
        )

    assert run_step.call_count == 8
    assert receipt["status"] == "production_promoted"
    assert receipt["promotion_status"] == "promoted"
    assert [call.kwargs["source"] for call in run_step.call_args_list[5:]] == [
        "forward_evaluation",
        "independent_authority",
        "production_bundle",
    ]
    publish.assert_called_once_with(tmp_path)
    assert receipt["authority"]["publication"] is True


def test_promote_stays_blocked_when_blob_publication_is_not_configured(tmp_path: Path) -> None:
    steps = [
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0}
        for _ in range(8)
    ]
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_api_meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(
        json.dumps({"source_latest": "2026-08-08T12:00:00Z", "player_detail_complete": True}),
        encoding="utf-8",
    )
    with patch.dict(live_refresh.os.environ, {}, clear=True), patch.object(
        live_refresh, "_run_step", side_effect=steps
    ), patch.object(
        live_refresh,
        "build_candidate",
        return_value=_candidate(source_mode="oe_only"),
    ), patch.object(live_refresh, "write_candidate", return_value="b" * 64):
        receipt = live_refresh.refresh_candidate(
            tmp_path,
            expected_live_as_of="2026-08-08T12:00:00Z",
            receipt_path=tmp_path / "receipt.json",
            source_mode="oe_only",
            promote=True,
        )

    assert receipt["status"] == "blocked_publication"
    assert receipt["promotion_status"] == "blocked_publication"
    assert receipt["authority"]["publication"] is False
    assert receipt["promotion_steps"][-1]["source"] == "blob_publication"
    assert "BLOB_READ_WRITE_TOKEN" in receipt["promotion_steps"][-1]["reason"]
