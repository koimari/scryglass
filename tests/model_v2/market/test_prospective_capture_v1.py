from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from lol_kills.v2.market import prospective_capture_v1 as capture


NOW = datetime(2026, 8, 2, 7, 0, tzinfo=timezone.utc)


def _root(tmp_path: Path) -> Path:
    source = tmp_path / capture.SOURCE_LOCATOR
    source.parent.mkdir(parents=True)
    source.write_text("# frozen prospective capture fixture\n", encoding="utf-8")
    return tmp_path


def _players(prefix: str) -> list[dict[str, str]]:
    return [
        {"role": "top", "player_id": f"{prefix}-top", "display_name": "Top"},
        {
            "role": "jungle",
            "player_id": f"{prefix}-jungle",
            "display_name": "Jungle",
        },
        {"role": "mid", "player_id": f"{prefix}-mid", "display_name": "Mid"},
        {"role": "bot", "player_id": f"{prefix}-bot", "display_name": "Bot"},
        {
            "role": "support",
            "player_id": f"{prefix}-support",
            "display_name": "Support",
        },
    ]


def _prepare_input() -> dict[str, object]:
    return {
        "schema_version": capture.INPUT_SCHEMA_VERSION,
        "event": {
            "event_id": "lcs-2026-summer-week-2-match-3-game-1",
            "series_id": "lcs-2026-summer-week-2-match-3",
            "game_number": 1,
            "event_start_utc": "2026-08-02T20:00:00+00:00",
            "league": "LCS",
        },
        "roster_source": {
            "source": "public-test-source",
            "source_url": "https://example.test/rosters/fixture",
            "source_record_id": "fixture-revision-1",
            "source_updated_at_utc": "2026-08-02T06:00:00+00:00",
            "available_at_utc": "2026-08-02T06:30:00+00:00",
            "rights_status": "reviewed",
        },
        "teams": [
            {
                "side": "blue",
                "organization_id": "blue-org",
                "organization_name": "Blue Org",
                "roster_id": "blue-roster-v1",
                "players": _players("blue"),
            },
            {
                "side": "red",
                "organization_id": "red-org",
                "organization_name": "Red Org",
                "roster_id": "red-roster-v1",
                "players": _players("red"),
            },
        ],
    }


def _raw(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def _attempt(root: Path, locator: str) -> dict[str, object]:
    value = json.loads((root / locator).read_text(encoding="ascii"))
    return capture.validate_attempt_receipt(value, root=root)


def test_prepare_rejects_schedule_order_as_blue_red_authority(tmp_path: Path) -> None:
    root = _root(tmp_path)
    value = _prepare_input()
    teams = value["teams"]
    assert isinstance(teams, list)
    teams[0]["side"] = "team1"
    teams[1]["side"] = "team2"

    result = capture.prepare_capture(
        prepare_input_raw=_raw(value),
        roster_source_payload_raw=b"source bytes",
        patch_receipt_raw=b"{}\n",
        root=root,
        clock=lambda: NOW,
    )

    assert result["status"] == "FAILED_CLOSED"
    assert result["eligible_evaluation_evidence"] is False
    assert result["betting_authority"] is False
    attempt = _attempt(root, result["attempt_locator"])
    assert attempt["published_artifacts"] == []
    assert "teams must be ordered blue then red" in attempt["blockers"][0]
    assert not (root / "data/lol/v2/evaluation/multileague-v3/predictions").exists()


def test_prepare_rejects_non_exact_roster_without_partial_evidence(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    value = _prepare_input()
    teams = value["teams"]
    assert isinstance(teams, list)
    teams[0]["players"].append(
        {"role": "jungle", "player_id": "blue-sub", "display_name": "Sub"}
    )

    result = capture.prepare_capture(
        prepare_input_raw=_raw(value),
        roster_source_payload_raw=b"source bytes",
        patch_receipt_raw=b"{}\n",
        root=root,
        clock=lambda: NOW,
    )

    attempt = _attempt(root, result["attempt_locator"])
    assert attempt["status"] == "FAILED_CLOSED"
    assert attempt["eligible_evaluation_evidence"] is False
    assert attempt["published_artifacts"] == []
    assert "requires exactly five players" in attempt["blockers"][0]


def test_prepare_rejects_any_outcome_field_before_child_builders(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    value = _prepare_input()
    value["winner"] = "blue-org"

    result = capture.prepare_capture(
        prepare_input_raw=_raw(value),
        roster_source_payload_raw=b"source bytes",
        patch_receipt_raw=b"{}\n",
        root=root,
        clock=lambda: NOW,
    )

    attempt = _attempt(root, result["attempt_locator"])
    assert attempt["status"] == "FAILED_CLOSED"
    assert "prepare_input.winner" in attempt["blockers"][0]
    assert attempt["outcomes_present"] is False
    assert attempt["outcomes_accessed"] is False


def test_attempt_receipt_success_remains_non_authorizing(tmp_path: Path) -> None:
    root = _root(tmp_path)
    payload = capture.build_attempt_receipt(
        stage="draft",
        status="SUCCEEDED_EVALUATION_CANDIDATE_ONLY",
        event=_prepare_input()["event"],
        input_digests={"draft_input": "1" * 64},
        artifacts=[
            {
                "locator": "data/lol/v2/evaluation/draft-terminal-v1/predictions/x.json",
                "raw_sha256": "2" * 64,
                "artifact_sha256": "3" * 64,
            }
        ],
        blockers=[],
        root=root,
        clock=lambda: NOW,
    )

    checked = capture.validate_attempt_receipt(payload, root=root)
    assert checked["eligible_evaluation_evidence"] is True
    assert checked["authority"] == {name: False for name in capture.AUTHORITY_KEYS}
    assert checked["outcomes_present"] is False
    assert checked["outcomes_accessed"] is False

    forged = dict(payload)
    forged["authority"] = {**payload["authority"], "betting_authority": True}
    unsigned = {key: item for key, item in forged.items() if key != "artifact_sha256"}
    forged["artifact_sha256"] = capture._canonical_sha256(unsigned)
    with pytest.raises(capture.ProspectiveCaptureError, match="authority boundary"):
        capture.validate_attempt_receipt(forged, root=root)


def test_failure_requires_a_blocker_and_can_never_be_evidence(tmp_path: Path) -> None:
    root = _root(tmp_path)
    with pytest.raises(capture.ProspectiveCaptureError, match="blocker state"):
        capture.build_attempt_receipt(
            stage="prepare",
            status="FAILED_CLOSED",
            event=None,
            input_digests={"input": "4" * 64},
            artifacts=[],
            blockers=[],
            root=root,
            clock=lambda: NOW,
        )

