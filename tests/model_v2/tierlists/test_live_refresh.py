"""Tests for the source-mode-aware tier-list refresh worker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
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


def test_rating_step_fails_closed_on_a_different_accepted_census(tmp_path: Path) -> None:
    path = tmp_path / live_refresh.RATING_REFRESH_OUTPUT
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "source": {
                    "source_game_count": 2,
                    "source_identity_sha256": "b" * 64,
                }
            }
        ),
        encoding="utf-8",
    )

    result = live_refresh._bind_rating_step_to_census(
        tmp_path,
        {"completed": True, "returncode": 0},
        {"game_count": 1, "source_identity_sha256": "a" * 64},
    )

    assert result["completed"] is False
    assert result["returncode"] == 2
    assert result["reason"] == "rating refresh used a different accepted game census"


def test_regional_refresh_receipt_records_view_coverage() -> None:
    summary = live_refresh._regional_refresh_summary(
        {
            "cells": [
                {
                    "regional_views": [
                        {"id": "LCK", "maps": 12},
                        {"id": "LEC", "maps": 8},
                    ]
                },
                {"regional_views": []},
            ]
        }
    )

    assert summary == {
        "status": "available",
        "view_count": 2,
        "cell_count": 2,
        "cells_with_views": 1,
        "contexts": ["LCK", "LEC"],
        "basis": "patch_wide_model_with_regional_appearance_filter",
    }


def test_patch_refresh_receipt_keeps_26_16_pending_until_source_rows_arrive() -> None:
    summary = live_refresh._patch_refresh_summary(
        {"cells": [{"patches": ["16.15"]}, {"patches": ["26.15"]}]}
    )

    assert summary["current_public_patch"] == "26.16"
    assert summary["accepted_public_patches"] == ["26.15"]
    assert summary["latest_accepted_public_patch"] == "26.15"
    assert summary["status"] == "awaiting_source_rows"


def test_patch_refresh_receipt_marks_26_16_current_when_source_rows_arrive() -> None:
    summary = live_refresh._patch_refresh_summary(
        {"cells": [{"patches": ["16.15", "16.16"]}]}
    )

    assert summary["accepted_public_patches"] == ["26.15", "26.16"]
    assert summary["latest_accepted_public_patch"] == "26.16"
    assert summary["status"] == "current"


def test_tier_source_watermark_uses_the_latest_identity_complete_map(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data/lol/warehouse/parquet/oe_live/meta.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "source_latest": "2026-08-08T21:50:46Z",
                "statistics_complete_source_latest": "2026-07-28T23:45:16Z",
            }
        ),
        encoding="utf-8",
    )

    assert live_refresh._oe_source_latest(tmp_path) == "2026-08-08T21:50:46Z"


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


def test_live_source_excludes_games_with_invalid_identity_or_statistics() -> None:
    rows = []
    for game_id in (
        "good",
        "missing",
        "duplicate",
        "bad-share",
        "rounded-share",
        "partial",
    ):
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
                        "damageshare": (
                            0.3
                            if game_id == "bad-share" and side == "Blue" and role == "top"
                            else 0.2000016
                            if game_id == "rounded-share" and side == "Blue" and role == "top"
                            else 0.2
                        ),
                        "datacompleteness": (
                            "partial" if game_id == "partial" else "complete"
                        ),
                        "totalgold": 10000,
                        "cspm": 7,
                        "wpm": 0.5,
                        "wcpm": 0.25,
                        "golddiffat10": 0,
                    }
                )

    assert _complete_player_game_ids(pd.DataFrame(rows)) == {
        "good",
        "rounded-share",
    }


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


def test_legacy_blob_publication_and_restore_are_retired() -> None:
    message = "Vercel Blob tier-list publication is retired"
    with pytest.raises(live_refresh.PublicationError, match=message):
        live_refresh.publish_production_bundle(Path("."))
    with pytest.raises(live_refresh.PublicationError, match=message):
        live_refresh.restore_production_bundle({})


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


def test_step_timeout_default_is_2700_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCRYGLASS_TIER_STEP_TIMEOUT_SECONDS", raising=False)

    assert live_refresh._step_timeout_seconds() == 2700.0


def test_step_timeout_env_override_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRYGLASS_TIER_STEP_TIMEOUT_SECONDS", "1800")

    assert live_refresh._step_timeout_seconds() == 1800.0


def test_step_timeout_empty_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRYGLASS_TIER_STEP_TIMEOUT_SECONDS", "")

    assert live_refresh._step_timeout_seconds() == 2700.0


def test_step_timeout_whitespace_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRYGLASS_TIER_STEP_TIMEOUT_SECONDS", "   ")

    assert live_refresh._step_timeout_seconds() == 2700.0


def test_step_timeout_non_numeric_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRYGLASS_TIER_STEP_TIMEOUT_SECONDS", "not-a-number")

    with pytest.raises(ValueError):
        live_refresh._step_timeout_seconds()


def test_step_timeout_zero_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRYGLASS_TIER_STEP_TIMEOUT_SECONDS", "0")

    with pytest.raises(ValueError, match="must be positive"):
        live_refresh._step_timeout_seconds()


def test_step_timeout_negative_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRYGLASS_TIER_STEP_TIMEOUT_SECONDS", "-1")

    with pytest.raises(ValueError, match="must be positive"):
        live_refresh._step_timeout_seconds()


def test_step_timeout_nan_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRYGLASS_TIER_STEP_TIMEOUT_SECONDS", "nan")

    with pytest.raises(ValueError, match="must be positive"):
        live_refresh._step_timeout_seconds()


def test_step_timeout_inf_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCRYGLASS_TIER_STEP_TIMEOUT_SECONDS", "inf")

    with pytest.raises(ValueError, match="must be positive"):
        live_refresh._step_timeout_seconds()


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


def test_release_tier_refresh_does_not_rebuild_the_accepted_oe_source(tmp_path: Path) -> None:
    steps = [
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0}
        for _ in range(2)
    ]
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_live/meta.json"
    meta_path.parent.mkdir(parents=True)
    meta_path.write_text(
        json.dumps(
            {
                "source_latest": "2026-08-16T21:07:39+00:00",
                "source_game_count": 22,
                "identity_complete_game_count": 22,
            }
        ),
        encoding="utf-8",
    )
    with patch.object(live_refresh, "_run_step", side_effect=steps) as run_step, patch.object(
        live_refresh,
        "build_candidate",
        return_value=_candidate(source_mode="oe_only"),
    ), patch.object(live_refresh, "write_candidate", return_value="b" * 64):
        receipt = live_refresh.refresh_candidate(
            tmp_path,
            expected_live_as_of="2026-08-16T21:07:39Z",
            output_path=tmp_path / "candidate.json",
            receipt_path=tmp_path / "receipt.json",
            source_mode="oe_only",
            skip_annual_oe=True,
            skip_live_source=True,
        )

    assert run_step.call_count == 2
    assert all(call.kwargs["source"] != "oe_live_source" for call in run_step.call_args_list)
    assert receipt["source_steps"][2]["completed"] is True
    assert receipt["source_steps"][2]["reason"] == "accepted_oe_live_verified"


def test_ratings_step_receives_previous_refresh_as_of(tmp_path: Path) -> None:
    """Movement baseline: the ratings step gets --previous-as-of from the
    previous approved bundle so rank movement reflects the last cycle."""
    steps = [
        {"returncode": 0, "completed": True, "stdout_bytes": 0, "stderr_bytes": 0}
        for _ in range(4)
    ]
    previous_bundle = tmp_path / "previous-bundle.json"
    previous_bundle.write_text(
        json.dumps({"schema_version": "tier-list:v1", "as_of": "2026-08-07T12:00:00Z"}),
        encoding="utf-8",
    )
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
        live_refresh.refresh_candidate(
            tmp_path,
            expected_live_as_of="2026-08-08T12:00:00Z",
            previous_path=previous_bundle,
            output_path=tmp_path / "candidate.json",
            receipt_path=tmp_path / "receipt.json",
            source_mode="oe_only",
            skip_annual_oe=True,
        )
    ratings_calls = [
        call for call in run_step.call_args_list
        if any("rating_refresh" in str(part) for part in call.args[1])
    ]
    assert ratings_calls, "ratings step did not run"
    args = ratings_calls[0].args[1]
    assert "--previous-as-of" in args
    assert args[args.index("--previous-as-of") + 1] == "2026-08-07T12:00:00Z"


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
    assert receipt["regional_refresh"]["status"] == "unavailable"
    assert receipt["patch_refresh"]["status"] == "awaiting_source_rows"


def test_promote_stays_blocked_when_legacy_blob_publication_is_retired(tmp_path: Path) -> None:
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
    assert receipt["promotion_steps"][-1]["source"] == "publication_disabled"
    assert "Vercel Blob tier-list publication is retired" in receipt["promotion_steps"][-1]["reason"]


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
