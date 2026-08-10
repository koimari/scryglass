"""Tests for the source-mode-aware tier-list refresh worker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from lol_kills.v2.tierlists import live_refresh
from lol_kills.etl.oe_live_source import _complete_player_game_ids, _merge
from lol_kills.v2.tierlists.forward_evaluation import _map_keys, _source_columns
from lol_kills.v2.tierlists.independent_authority import (
    _map_keys as _authority_map_keys,
    _source_columns as _authority_source_columns,
)


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


def test_rating_source_accepts_complete_identities_with_pending_statistics(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data/lol/warehouse/parquet/oe_live/meta.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "source_game_count": 10,
                "identity_complete_game_count": 10,
                "statistics_complete_game_count": 8,
                "player_statistics_complete": False,
            }
        ),
        encoding="utf-8",
    )

    assert live_refresh._oe_rating_source_complete(tmp_path) is True


def test_rating_source_rejects_an_incomplete_roster(tmp_path: Path) -> None:
    path = tmp_path / "data/lol/warehouse/parquet/oe_live/meta.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "source_game_count": 10,
                "identity_complete_game_count": 9,
                "statistics_complete_game_count": 9,
                "player_statistics_complete": False,
            }
        ),
        encoding="utf-8",
    )

    assert live_refresh._oe_rating_source_complete(tmp_path) is False


def test_live_merge_deduplicates_prefixed_and_canonical_game_ids() -> None:
    rows = [
        {"game_uid": "oe-api:g1", "gameid": "oe-api:g1", "date": "2026-08-08", "league": "LCK", "side": "Blue", "teamname": "A", "result": 1},
        {"game_uid": "oe-api:g1", "gameid": "oe-api:g1", "date": "2026-08-08", "league": "LCK", "side": "Red", "teamname": "B", "result": 0},
    ]
    supplement = [{**row, "game_uid": "g1", "gameid": "g1"} for row in rows]

    merged = _merge(pd.DataFrame(rows), pd.DataFrame(supplement), with_players=False)

    assert len(merged) == 2
    assert set(merged["game_uid"]) == {"g1"}
    assert set(merged["gameid"]) == {"g1"}


def test_live_merge_aligns_bridge_numbers_to_the_annual_schema() -> None:
    primary = pd.DataFrame(
        [
            {"gameid": "annual", "date": "2026-08-01", "side": "Blue", "game": 1.0},
            {"gameid": "annual", "date": "2026-08-01", "side": "Red", "game": 1.0},
        ]
    )
    supplement = pd.DataFrame(
        [
            {"gameid": "bridge", "date": "2026-08-08", "side": "Blue", "game": "2"},
            {"gameid": "bridge", "date": "2026-08-08", "side": "Red", "game": "2"},
        ]
    )

    merged = _merge(primary, supplement, with_players=False)

    assert merged["game"].dtype.kind == "f"
    assert merged["game"].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_live_source_excludes_games_with_missing_or_duplicate_player_names() -> None:
    rows = []
    for game_id, bad in (("good", False), ("missing", True), ("duplicate", True)):
        for side in ("Blue", "Red"):
            for role in ("top", "jng", "mid", "bot", "sup"):
                name = f"{side}-{role}"
                if game_id == "missing" and side == "Blue":
                    name = None
                if game_id == "duplicate" and side == "Blue" and role in {"bot", "sup"}:
                    name = "unknown player"
                rows.append(
                    {
                        "game_uid": f"oe-api:{game_id}",
                        "gameid": f"oe-api:{game_id}",
                        "side": side,
                        "position": role,
                        "playername": name,
                        "kills": 2,
                        "deaths": 2,
                        "assists": 8,
                        "teamkills": 15,
                        "gamelength": 1800,
                        "dpm": 500,
                        "damageshare": 0.2,
                        "totalgold": 10000,
                        "cspm": 7,
                        "wpm": 0.5,
                        "wcpm": 0.25,
                        "golddiffat10": 0,
                    }
                )

    assert _complete_player_game_ids(pd.DataFrame(rows)) == {"good"}


def test_forward_evaluation_accepts_annual_oe_rows_without_game_uid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "players.parquet"
    frame = pd.DataFrame(
        {
            "gameid": ["oe-api:game-1"],
            "date": ["2026-08-07"],
            "league": ["LCK"],
            "competition_tier": ["tier1"],
            "event_kind": [None],
            "patch": ["16.15"],
            "position": ["top"],
            "champion": ["Aatrox"],
            "side": ["Blue"],
            "teamname": ["T1"],
            "result": [1],
        }
    )
    frame.to_parquet(path, index=False)

    assert "game_uid" not in _source_columns(path)
    assert _map_keys(frame).tolist() == ["game-1"]
    assert "game_uid" not in _authority_source_columns(path)
    assert _authority_map_keys(frame).tolist() == ["game-1"]


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
    assert payloads["cell_count"] == 195
    assert json.loads(payloads["display_raw"])["schema_version"] == "rankings-tierlists-v2"


def test_blob_publication_writes_the_pointer_last() -> None:
    root = Path(__file__).resolve().parents[3]

    class FakeTransport:
        pointer_raw = b""
        movement_raw = b""
        display_raw = b""

        def get_blob(self, _store_id: str, pathname: str, *, deadline_epoch: int):
            if pathname == live_refresh.BLOB_MOVEMENT_PATH:
                if not self.movement_raw:
                    return None
                return (
                    self.movement_raw,
                    live_refresh.BlobIdentity(pathname, len(self.movement_raw), "movement-etag"),
                )
            if pathname == live_refresh.BLOB_DISPLAY_PATH:
                if not self.display_raw:
                    return None
                return (
                    self.display_raw,
                    live_refresh.BlobIdentity(pathname, len(self.display_raw), "display-etag"),
                )
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
                or write.pathname == live_refresh.BLOB_MOVEMENT_PATH
                or write.pathname == live_refresh.BLOB_DISPLAY_PATH
                for write in plan.writes[:-1]
            )
            movement_write = next(
                write for write in plan.writes if write.pathname == live_refresh.BLOB_MOVEMENT_PATH
            )
            transport.movement_raw = movement_write.content
            display_write = next(
                write for write in plan.writes if write.pathname == live_refresh.BLOB_DISPLAY_PATH
            )
            transport.display_raw = display_write.content
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
    assert result["display_readback_verified"] is True
    assert result["cell_count"] == 195
    assert set(result["previous_pointers"]) == {
        live_refresh.BLOB_POINTER_PATH,
        live_refresh.BLOB_MOVEMENT_PATH,
        live_refresh.BLOB_DISPLAY_PATH,
    }


def test_blob_publication_reuses_an_exact_partial_immutable_release() -> None:
    root = Path(__file__).resolve().parents[3]
    payloads = live_refresh._publication_payloads(root)
    reused_path, reused_raw = next(iter(sorted(payloads["cell_bytes"].items())))
    reused_identity = live_refresh.BlobIdentity(reused_path, len(reused_raw), "reused-etag")

    class FakeTransport:
        pointer_raw = b""
        movement_raw = b""
        display_raw = b""

        def get_blob(self, _store_id: str, pathname: str, *, deadline_epoch: int):
            if pathname == reused_path:
                return reused_raw, reused_identity
            if pathname == live_refresh.BLOB_MOVEMENT_PATH and self.movement_raw:
                return self.movement_raw, live_refresh.BlobIdentity(
                    pathname, len(self.movement_raw), "movement-etag"
                )
            if pathname == live_refresh.BLOB_DISPLAY_PATH and self.display_raw:
                return self.display_raw, live_refresh.BlobIdentity(
                    pathname, len(self.display_raw), "display-etag"
                )
            if pathname == live_refresh.BLOB_POINTER_PATH and self.pointer_raw:
                return self.pointer_raw, live_refresh.BlobIdentity(
                    pathname, len(self.pointer_raw), "pointer-etag"
                )
            return None

    transport = FakeTransport()

    class FakeExecutor:
        def __init__(self, _transport: FakeTransport):
            pass

        def execute(self, plan):
            assert reused_path not in {write.pathname for write in plan.writes}
            movement = next(
                write for write in plan.writes if write.pathname == live_refresh.BLOB_MOVEMENT_PATH
            )
            display = next(
                write for write in plan.writes if write.pathname == live_refresh.BLOB_DISPLAY_PATH
            )
            transport.movement_raw = movement.content
            transport.display_raw = display.content
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
        live_refresh, "_blob_inventory", return_value={reused_path: reused_identity}
    ), patch.object(live_refresh, "RetentionExecutor", FakeExecutor):
        result = live_refresh.publish_production_bundle(root)

    assert result["status"] == "published"
    assert result["reused_immutable_files"] == 1


def test_tier_step_timeout_is_recorded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def timed_out(*_args, **kwargs):
        assert kwargs["timeout"] == 2
        raise subprocess.TimeoutExpired(kwargs.get("args", "step"), 2)

    monkeypatch.setattr(live_refresh.subprocess, "run", timed_out)

    result = live_refresh._run_step(
        tmp_path,
        ["example.step"],
        source="example",
        step_timeout_seconds=2,
    )

    assert result["completed"] is False
    assert result["timed_out"] is True
    assert result["returncode"] == 124


def test_blob_publication_removes_new_stable_pointers_after_a_failed_write() -> None:
    root = Path(__file__).resolve().parents[3]

    class FakeTransport:
        def __init__(self) -> None:
            self.storage: dict[str, bytes] = {}

        def _identity(self, pathname: str) -> live_refresh.BlobIdentity:
            raw = self.storage[pathname]
            return live_refresh.BlobIdentity(pathname, len(raw), f"etag-{len(raw)}")

        def get_blob(self, _store_id: str, pathname: str, *, deadline_epoch: int):
            if pathname not in self.storage:
                return None
            return self.storage[pathname], self._identity(pathname)

        def delete_if_match(
            self,
            _store_id: str,
            pathname: str,
            *,
            etag: str,
            deadline_epoch: int,
        ):
            if pathname not in self.storage or self._identity(pathname).etag != etag:
                return None
            prior = self._identity(pathname)
            del self.storage[pathname]
            return prior

    transport = FakeTransport()

    class FailedExecutor:
        def __init__(self, _transport: FakeTransport):
            pass

        def execute(self, plan):
            for write in plan.writes:
                transport.storage[write.pathname] = write.content
            return SimpleNamespace(
                success=False,
                state=SimpleNamespace(value="failed"),
                operations=(SimpleNamespace(pathname=live_refresh.BLOB_POINTER_PATH, success=False),),
            )

    with patch.object(
        live_refresh,
        "_publication_credentials",
        return_value=("https://store-test.public.blob.vercel-storage.com", "token", "store-test"),
    ), patch.object(live_refresh, "VercelBlobTransport", return_value=transport), patch.object(
        live_refresh, "_blob_inventory", return_value={}
    ), patch.object(live_refresh, "RetentionExecutor", FailedExecutor):
        with pytest.raises(live_refresh.PublicationError, match="publication failed"):
            live_refresh.publish_production_bundle(root)

    assert live_refresh.BLOB_POINTER_PATH not in transport.storage
    assert live_refresh.BLOB_MOVEMENT_PATH not in transport.storage
    assert live_refresh.BLOB_DISPLAY_PATH not in transport.storage


def test_oe_only_skips_grid_and_can_be_ready_from_a_complete_oe_source(tmp_path: Path) -> None:
    oe_step = {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0}
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_live/meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(
        json.dumps({"source_latest": "2026-08-08T12:00:00Z", "player_statistics_complete": True}),
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

    assert run_step.call_count == 4
    assert "--skip-grid" in run_step.call_args_list[0].args[1]
    assert run_step.call_args_list[1].kwargs["source"] == "champion_atomization"
    assert run_step.call_args_list[2].kwargs["source"] == "oe_live_source"
    assert run_step.call_args_list[3].kwargs["source"] == "ratings"
    assert receipt["source_mode"] == "oe_only"
    assert receipt["status"] == "ready_for_authority_review"
    assert receipt["source_steps"][4]["skipped"] is True
    assert receipt["source_steps"][4]["reason"] == "public_refresh_oe_only"
    saved = json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8"))
    assert saved["source_mode"] == "oe_only"


def test_public_tier_refresh_rejects_grid_source_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be oe_only"):
        live_refresh.refresh_candidate(
            tmp_path,
            expected_live_as_of="2026-08-08T12:00:00Z",
            output_path=tmp_path / "candidate.json",
            receipt_path=tmp_path / "receipt.json",
            source_mode="oe_plus_grid",
        )


def test_skip_annual_oe_uses_the_cached_oe_source(tmp_path: Path) -> None:
    steps = [
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0}
        for _ in range(3)
    ]
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_live/meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(
        json.dumps({"source_latest": "2026-08-08T12:00:00Z", "player_statistics_complete": True}),
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

    assert run_step.call_count == 3
    assert receipt["source_steps"][0]["source"] == "oe_annual"
    assert receipt["source_steps"][0]["skipped"] is True
    assert receipt["source_steps"][0]["reason"] == "cached_oe_source"
    assert run_step.call_args_list[0].kwargs["source"] == "champion_atomization"


def test_promote_runs_evaluation_authority_and_bundle_after_source_refresh(tmp_path: Path) -> None:
    steps = [
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0}
        for _ in range(7)
    ]
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_live/meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(
        json.dumps({"source_latest": "2026-08-08T12:00:00Z", "player_statistics_complete": True}),
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

    assert run_step.call_count == 7
    assert receipt["status"] == "production_promoted"
    assert receipt["promotion_status"] == "promoted"
    assert [call.kwargs["source"] for call in run_step.call_args_list[4:]] == [
        "forward_evaluation",
        "independent_authority",
        "production_bundle",
    ]
    publish.assert_called_once_with(tmp_path)
    assert receipt["authority"]["publication"] is True


def test_promote_can_defer_publication_to_the_release_coordinator(tmp_path: Path) -> None:
    steps = [
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0}
        for _ in range(7)
    ]
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_live/meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(
        json.dumps({"source_latest": "2026-08-08T12:00:00Z", "player_statistics_complete": True}),
        encoding="utf-8",
    )
    with patch.object(live_refresh, "_run_step", side_effect=steps), patch.object(
        live_refresh,
        "build_candidate",
        return_value=_candidate(source_mode="oe_only"),
    ), patch.object(live_refresh, "write_candidate", return_value="b" * 64), patch.object(
        live_refresh,
        "publish_production_bundle",
    ) as publish:
        receipt = live_refresh.refresh_candidate(
            tmp_path,
            expected_live_as_of="2026-08-08T12:00:00Z",
            receipt_path=tmp_path / "receipt.json",
            source_mode="oe_only",
            promote=True,
            publish=False,
        )

    publish.assert_not_called()
    assert receipt["status"] == "production_built"
    assert receipt["promotion_status"] == "built"
    assert receipt["promotion_steps"][-1]["source"] == "deferred_publication"
    assert receipt["authority"]["publication"] is False
    assert receipt["authority"]["rank_eligibility"] is True


def test_promote_stays_blocked_when_blob_publication_is_not_configured(tmp_path: Path) -> None:
    steps = [
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0}
        for _ in range(7)
    ]
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_live/meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(
        json.dumps({"source_latest": "2026-08-08T12:00:00Z", "player_statistics_complete": True}),
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


def test_prepared_source_bundle_skips_source_ingestion(tmp_path: Path) -> None:
    steps = [
        {
            "source": source,
            "returncode": 0,
            "completed": True,
            "stdout_bytes": 0,
            "stderr_bytes": 0,
        }
        for source in ("oe_annual", "champion_atomization", "oe_live_source", "ratings")
    ]
    steps.append(
        {
            "source": "grid",
            "returncode": None,
            "completed": False,
            "skipped": True,
            "reason": "source_mode_oe_only",
        }
    )
    prepared = {
        "schema_version": "scryglass:tierlist-source-bundle:v1",
        "artifact_kind": "tier_list_source_bundle",
        "source_mode": "oe_only",
        "source_observed_through": "2026-08-08T12:00:00Z",
        "source_steps": steps,
    }
    with patch.object(live_refresh, "_run_step") as run_step, patch.object(
        live_refresh,
        "build_candidate",
        return_value=_candidate(source_mode="oe_only"),
    ), patch.object(live_refresh, "write_candidate", return_value="b" * 64):
        receipt = live_refresh.refresh_candidate(
            tmp_path,
            expected_live_as_of="2026-08-09T00:00:00Z",
            output_path=tmp_path / "candidate.json",
            receipt_path=tmp_path / "receipt.json",
            source_mode="oe_only",
            prepared_source=prepared,
        )

    run_step.assert_not_called()
    assert receipt["source_observed_through"] == "2026-08-08T12:00:00Z"
    assert receipt["source_steps"] == steps
