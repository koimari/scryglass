from __future__ import annotations

from datetime import datetime, timezone
import base64
import hashlib
import json
import os
from pathlib import Path

import pytest

from lol_kills.v2.market import side_neutral_prospective_capture_v1 as capture
from lol_kills.v2.ratings.player.side_neutral_protocol_review_v1 import (
    SideNeutralProtocolReviewError,
)


REVIEWED_AT = "2026-08-02T12:00:00+00:00"
CAPTURED_AT = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)


def _review() -> dict:
    return {
        "review_id": "external-side-neutral-review-1",
        "authorization": {
            "effective_at_utc": REVIEWED_AT,
            "prospective_collection_authorized": True,
        },
    }


def _activate_review(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        capture,
        "load_active_side_neutral_protocol_review",
        lambda **_kwargs: _review(),
    )


def _embedded(value: dict) -> dict:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return {
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_base64": base64.b64encode(raw).decode(),
        "value": value,
    }


def test_pre_side_refuses_to_run_without_externally_pinned_review(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        capture,
        "load_active_side_neutral_protocol_review",
        lambda **_kwargs: (_ for _ in ()).throw(
            SideNeutralProtocolReviewError("missing external review digest")
        ),
    )
    called = False

    def forbidden_builder(**_kwargs: object) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        capture.pre_side, "build_pre_side_rating_envelope", forbidden_builder
    )
    with pytest.raises(
        capture.SideNeutralProspectiveCaptureError,
        match="externally pinned independent",
    ):
        capture.capture_pre_side(
            input_raw=b"{}",
            roster_source_payload_raw=b"{}",
            patch_receipt_raw=b"{}",
            root=tmp_path,
            environment={},
            clock=lambda: CAPTURED_AT,
        )
    assert called is False
    assert list(tmp_path.rglob("*.json")) == []


def test_pre_side_writes_only_canonical_no_clobber_artifact_after_review(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _activate_review(monkeypatch)
    payload = {
        "captured_at_utc": CAPTURED_AT.isoformat(),
        "artifact_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        capture.pre_side,
        "build_pre_side_rating_envelope",
        lambda **_kwargs: payload,
    )
    locator = "data/lol/v2/evaluation/multileague-v3/pre-side-rating-envelopes/2026-08-03/event-g1.json"
    monkeypatch.setattr(capture.pre_side, "envelope_locator", lambda _value: locator)

    def writer(path: Path, value: dict) -> str:
        assert value is payload
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")
        return "b" * 64

    monkeypatch.setattr(capture.pre_side, "write_no_clobber", writer)
    result = capture.capture_pre_side(
        input_raw=b"{}",
        roster_source_payload_raw=b"{}",
        patch_receipt_raw=b"{}",
        root=tmp_path,
        environment={capture.INDEPENDENT_REVIEW_ENV: "c" * 64},
        clock=lambda: CAPTURED_AT,
    )
    assert result["stage"] == "pre-side"
    assert result["output"] == str(tmp_path / locator)
    assert result["independent_review_id"] == _review()["review_id"]
    assert result["outcomes_accessed"] is False
    assert result["probability_authority"] is False
    assert result["betting_authority"] is False


def test_later_stage_rejects_pre_side_capture_that_predates_review(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _activate_review(monkeypatch)
    checked_binding = {
        "pre_side_envelope": {
            "value": {"captured_at_utc": "2026-08-02T11:59:59+00:00"}
        }
    }
    monkeypatch.setattr(
        capture.side_binding,
        "validate_pre_side_rating_binding",
        lambda *_args, **_kwargs: checked_binding,
    )
    called = False

    def forbidden_builder(**_kwargs: object) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        capture.neutral_draft,
        "build_side_neutral_draft_prediction",
        forbidden_builder,
    )
    with pytest.raises(
        capture.SideNeutralProspectiveCaptureError,
        match="does not follow independent review",
    ):
        capture.capture_terminal_draft(
            side_binding_raw=b"{}",
            draft_metadata_raw=b"{}",
            draft_source_payload_raw=b"{}",
            root=tmp_path,
            environment={},
            clock=lambda: CAPTURED_AT,
        )
    assert called is False


def test_draft_stage_prospectively_materializes_frozen_phase_one_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _activate_review(monkeypatch)
    checked_binding = {
        "pre_side_envelope": {
            "value": {"captured_at_utc": "2026-08-03T11:00:00+00:00"}
        }
    }
    monkeypatch.setattr(
        capture.side_binding,
        "validate_pre_side_rating_binding",
        lambda *_args, **_kwargs: checked_binding,
    )
    rating = {"artifact_sha256": "a" * 64}
    child = {
        "artifact_sha256": "b" * 64,
        "event": {
            "event_id": "LCS/Event 1",
            "game_number": 1,
            "event_start_utc": "2026-08-03T12:30:00+00:00",
        },
        "input_receipts": {"ratings_prediction": _embedded(rating)},
    }
    wrapper = {
        "artifact_sha256": "c" * 64,
        "terminal_draft_prediction": _embedded(child),
        "selected_rating_binding": {
            "rating_receipt_artifact_sha256": rating["artifact_sha256"]
        },
    }
    monkeypatch.setattr(
        capture.neutral_draft,
        "build_side_neutral_draft_prediction",
        lambda **_kwargs: wrapper,
    )
    wrapper_locator = (
        "data/lol/v2/evaluation/draft-terminal-v1/side-neutral-predictions/"
        "2026-08-03/lcs-event-1-g1.json"
    )
    monkeypatch.setattr(
        capture.neutral_draft,
        "prediction_locator",
        lambda _payload: wrapper_locator,
    )
    monkeypatch.setattr(
        capture.ratings_ledger,
        "validate_pre_event_prediction_receipt",
        lambda *_args, **_kwargs: rating,
    )
    monkeypatch.setattr(
        capture.draft_ledger,
        "validate_draft_prediction_receipt",
        lambda *_args, **_kwargs: child,
    )
    tail = "2026-08-03/lcs-event-1-g1.json"
    plan_locator = (
        "data/lol/v2/evaluation/match-winner-market-v1/phase-one/plans/"
        + tail
    )

    def build_plan(**_kwargs: object) -> dict:
        rating_path = (
            tmp_path
            / "data/lol/v2/evaluation/multileague-v3/predictions"
            / tail
        )
        assert rating_path.read_bytes() == base64.b64decode(
            child["input_receipts"]["ratings_prediction"]["raw_base64"]
        )
        return {"locators": {"plan": plan_locator}, "artifact_sha256": "d" * 64}

    monkeypatch.setattr(capture.phase_one, "build_event_plan", build_plan)
    result = capture.capture_terminal_draft(
        side_binding_raw=b"{}",
        draft_metadata_raw=b"{}",
        draft_source_payload_raw=b"{}",
        root=tmp_path,
        environment={},
        clock=lambda: CAPTURED_AT,
    )

    bridge = result["phase_one_bridge"]
    assert result["stage"] == "draft"
    assert json.loads(Path(bridge["ratings_prediction"]).read_text()) == rating
    assert json.loads(Path(bridge["draft_prediction"]).read_text()) == child
    assert json.loads(Path(bridge["event_plan"]).read_text())["locators"][
        "plan"
    ] == plan_locator
    assert json.loads(Path(result["output"]).read_text()) == wrapper
    assert bridge["outcomes_accessed"] is False
    assert bridge["opening_authority"] is False


def test_map_start_stage_publishes_canonical_start_then_complete_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _activate_review(monkeypatch)
    checked_draft = {
        "event": {
            "event_id": "LCS/Event 1",
            "game_number": 1,
            "event_start_utc": "2026-08-03T12:05:00+00:00",
        },
        "side_binding": {
            "value": {
                "pre_side_envelope": {
                    "value": {"captured_at_utc": "2026-08-03T11:00:00+00:00"}
                }
            }
        }
    }
    monkeypatch.setattr(
        capture.neutral_draft,
        "validate_side_neutral_draft_prediction",
        lambda *_args, **_kwargs: checked_draft,
    )
    map_start = {
        "event": {
            "event_id": "LCS/Event 1",
            "game_number": 1,
            "actual_map_start_utc": "2026-08-03T12:05:00+00:00",
        },
        "artifact_sha256": "d" * 64,
    }
    monkeypatch.setattr(
        capture.draft_ledger, "build_map_start_receipt", lambda **_kwargs: map_start
    )

    bundle = {"artifact_sha256": "f" * 64}
    monkeypatch.setattr(
        capture.bundle_module,
        "build_side_neutral_capture_bundle",
        lambda **_kwargs: bundle,
    )
    bundle_locator = (
        "data/lol/v2/evaluation/multileague-v3/side-neutral-bundles/"
        "2026-08-03/lcs-event-1-g1.json"
    )
    monkeypatch.setattr(
        capture.bundle_module, "bundle_locator", lambda _payload: bundle_locator
    )
    tail = "2026-08-03/lcs-event-1-g1.json"
    plan_locator = (
        "data/lol/v2/evaluation/match-winner-market-v1/phase-one/plans/"
        + tail
    )
    map_start_locator = (
        "data/lol/v2/evaluation/draft-terminal-v1/map-start/" + tail
    )
    phase_bundle_locator = (
        "data/lol/v2/evaluation/match-winner-market-v1/phase-one/bundles/"
        + tail
    )
    plan = {
        "locators": {
            "plan": plan_locator,
            "map_start": map_start_locator,
            "event_bundle": phase_bundle_locator,
        }
    }
    plan_path = tmp_path / plan_locator
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("{}\n")
    monkeypatch.setattr(
        capture.phase_one, "validate_event_plan", lambda *_args, **_kwargs: plan
    )
    phase_bundle = {"artifact_sha256": "e" * 64}
    monkeypatch.setattr(
        capture.phase_one,
        "build_event_bundle",
        lambda **_kwargs: phase_bundle,
    )

    result = capture.capture_map_start_bundle(
        side_neutral_draft_raw=b"{}",
        map_start_metadata_raw=b"{}",
        map_start_source_payload_raw=b"{}",
        root=tmp_path,
        environment={},
        clock=lambda: CAPTURED_AT,
    )
    assert result["stage"] == "map-start"
    assert result["output"] == str(tmp_path / bundle_locator)
    assert result["map_start"]["output"].endswith(
        "draft-terminal-v1/map-start/2026-08-03/lcs-event-1-g1.json"
    )
    assert json.loads(Path(result["map_start"]["output"]).read_text()) == map_start
    assert json.loads(Path(result["output"]).read_text()) == bundle
    assert json.loads(
        Path(result["phase_one_bridge"]["event_bundle"]).read_text()
    ) == phase_bundle
    assert result["betting_authority"] is False


def test_two_artifact_publish_rolls_back_first_link_if_second_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = tmp_path / "first" / "one.json"
    second = tmp_path / "second" / "two.json"
    original_link = os.link
    calls = 0

    def fail_second_link(source: object, destination: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second-link failure")
        original_link(source, destination)

    monkeypatch.setattr(capture.os, "link", fail_second_link)
    with pytest.raises(OSError, match="second-link failure"):
        capture._atomic_no_clobber_batch(
            [(first, {"value": 1}), (second, {"value": 2})]
        )
    assert first.exists() is False
    assert second.exists() is False
    assert list(tmp_path.rglob("*.partial")) == []


def test_reviewed_ledger_publishes_exact_phase_one_snapshot_cohort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _activate_review(monkeypatch)
    side_bundle_locator = (
        "data/lol/v2/evaluation/match-winner-market-v1/phase-one/"
        "side-neutral-bundles/2026-08-03/lcs-event-1-g1.json"
    )
    side_bundle_path = tmp_path / side_bundle_locator
    side_bundle_path.parent.mkdir(parents=True)
    side_bundle_path.write_text("{}\n")
    selected_child = {
        "event": {
            "event_id": "LCS/Event 1",
            "game_number": 1,
            "event_start_utc": "2026-08-03T12:30:00+00:00",
        }
    }
    checked_bundle = {
        "input_receipts": {
            "side_neutral_draft": {
                "value": {"terminal_draft_prediction": {"value": selected_child}}
            }
        }
    }
    monkeypatch.setattr(
        capture.bundle_module,
        "validate_side_neutral_capture_bundle",
        lambda *_args, **_kwargs: checked_bundle,
    )
    ledger_payload = {
        "artifact_sha256": "a" * 64,
        "entries": [{"bundle_locator": side_bundle_locator}],
        "qualification": {"eligible_map_count": 1},
        "support": {"support_met": False},
    }
    monkeypatch.setattr(
        capture.ledger_module,
        "build_side_neutral_ledger",
        lambda **_kwargs: ledger_payload,
    )
    snapshot = {
        "artifact_sha256": "b" * 64,
        "ratings_ledger_candidate": {"artifact_sha256": "c" * 64},
        "draft_ledger_candidate": {"artifact_sha256": "d" * 64},
        "support": {"joint_metadata_support_met": False},
    }

    def build_snapshot(**kwargs: object) -> dict:
        assert kwargs["bundle_locators"] == [
            "data/lol/v2/evaluation/match-winner-market-v1/phase-one/"
            "bundles/2026-08-03/lcs-event-1-g1.json"
        ]
        return snapshot

    monkeypatch.setattr(
        capture.phase_one, "build_joint_ledger_snapshot", build_snapshot
    )
    result = capture.publish_ledger(
        bundle_locators=[side_bundle_locator],
        root=tmp_path,
        environment={},
        clock=lambda: CAPTURED_AT,
    )
    bridge = result["phase_one_bridge"]
    assert json.loads(Path(result["output"]).read_text()) == ledger_payload
    assert json.loads(Path(bridge["joint_snapshot"]).read_text()) == snapshot
    assert Path(bridge["ratings_ledger_candidate"]).is_file()
    assert Path(bridge["draft_ledger_candidate"]).is_file()
    assert bridge["joint_metadata_support_met"] is False
    assert bridge["outcomes_accessed"] is False


def test_cli_reports_missing_review_as_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("input.json", "roster.json", "patch.json"):
        (tmp_path / name).write_text("{}\n")
    monkeypatch.delenv(capture.INDEPENDENT_REVIEW_ENV, raising=False)
    code = capture.main(
        [
            "--root",
            str(tmp_path),
            "pre-side",
            "--input",
            str(tmp_path / "input.json"),
            "--roster-source-payload",
            str(tmp_path / "roster.json"),
            "--patch-receipt",
            str(tmp_path / "patch.json"),
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert code == 2
    assert output["status"] == "FAILED_CLOSED"
    assert output["stage"] == "pre-side"
    assert output["outcomes_accessed"] is False
    assert output["probability_authority"] is False
    assert output["betting_authority"] is False
