from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from lol_kills.v2.market import phase_two_attempt_completion_v1 as completion


def _plan() -> dict:
    return {
        "artifact_sha256": "1" * 64,
        "event": {
            "event_id": "event-1", "series_id": "series-1", "game_number": 1,
            "league": "LCS", "patch": "26.17",
            "roster_change_stratum": "UNCHANGED",
            "sparse_or_new_champion_map": False,
            "market_type": "match_winner", "selection": "winner:blue",
            "opposing_selection": "winner:red",
        },
        "reserved_outputs": {
            "quote_locator": "quotes/quote.json",
            "failure_locator": "failures/failure.json",
            "qualification_locator": "qualified/quote.json",
            "completion_locator": "completions/completion.json",
        },
    }


def _start() -> dict:
    return {
        "artifact_sha256": "2" * 64,
        "captured_at_utc": "2026-09-01T15:00:11+00:00",
        "event": {
            "event_id": "event-1", "series_id": "series-1", "game_number": 1,
            "league": "LCS", "patch": "26.17",
            "actual_map_start_utc": "2026-09-01T15:00:10+00:00",
        },
    }


def _install(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict]:
    plan = _plan()
    start = _start()
    monkeypatch.setattr(
        completion, "_plan",
        lambda **_kwargs: ("plans/plan.json", b"plan", plan),
    )
    monkeypatch.setattr(
        completion.qualification, "_map_start",
        lambda **_kwargs: ("starts/start.json", b"start", start),
    )
    monkeypatch.setattr(
        completion, "_source_locks",
        lambda _root: [{"locator": "completion.py", "bytes": 1, "raw_sha256": "3" * 64}],
    )
    return plan, start


def test_failed_attempt_completion_counts_in_denominator(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    failure_path = tmp_path / "failures/failure.json"
    failure_path.parent.mkdir(parents=True)
    failure_path.write_text("{}")
    failure = {
        "artifact_sha256": "4" * 64,
        "event_plan_binding": {"artifact_sha256": "1" * 64},
        "failure": {"code": "MARKET_NOT_OPEN_OR_MISSING"},
    }
    monkeypatch.setattr(
        completion, "_regular_json",
        lambda _root, _locator, _label: (b"failure", {}),
    )
    monkeypatch.setattr(
        completion.attempt, "validate_quote_attempt_failure_v1",
        lambda _object, **_kwargs: failure,
    )
    receipt = completion.build_phase_two_attempt_completion_v1(
        event_plan_locator="plans/plan.json",
        map_start_locator="starts/start.json",
        root=tmp_path,
        environment={},
        clock=lambda: datetime(2026, 9, 1, 15, 0, 12, tzinfo=timezone.utc),
    )
    assert receipt["status"] == "QUOTE_ATTEMPT_FAILED"
    assert receipt["coverage"]["counts_in_otherwise_eligible_denominator"] is True
    assert receipt["coverage"]["counts_as_qualified_quote"] is False
    assert receipt["authority"]["betting_authority"] is False


def test_late_quote_cannot_become_qualified(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    quote_path = tmp_path / "quotes/quote.json"
    quote_path.parent.mkdir(parents=True)
    quote_path.write_text("{}")
    quote = {
        "artifact_sha256": "5" * 64,
        "event_plan_binding": {"artifact_sha256": "1" * 64},
        "frozen_v1_transport_quote": {
            "transport": {
                "response_received_at_utc": "2026-09-01T15:00:06+00:00"
            }
        },
    }
    monkeypatch.setattr(
        completion, "_regular_json",
        lambda _root, _locator, _label: (b"quote", {}),
    )
    monkeypatch.setattr(
        completion.quote_v2, "validate_betano_map_winner_quote_v2",
        lambda _object, **_kwargs: quote,
    )
    receipt = completion.build_phase_two_attempt_completion_v1(
        event_plan_locator="plans/plan.json",
        map_start_locator="starts/start.json",
        root=tmp_path,
        environment={},
        clock=lambda: datetime(2026, 9, 1, 15, 0, 12, tzinfo=timezone.utc),
    )
    assert receipt["status"] == "QUOTE_RESPONSE_TOO_LATE"
    assert receipt["response_to_actual_start_seconds"] == 4.0
    assert receipt["qualification_binding"] is None
    assert receipt["coverage"]["quote_after_or_within_five_seconds_of_start"] is True

    forged = deepcopy(receipt)
    forged["status"] = "QUALIFIED_QUOTE"
    forged["artifact_sha256"] = completion._canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(completion.PhaseTwoAttemptCompletionError):
        completion.validate_phase_two_attempt_completion_v1(
            forged, root=tmp_path, environment={}
        )
