from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from lol_kills.v2.market import phase_two_quote_attempt_v1 as attempt


def _plan() -> dict:
    return {
        "artifact_sha256": "1" * 64,
        "planned_at_utc": "2026-09-01T15:00:00+00:00",
        "event": {
            "event_id": "event-1", "series_id": "series-1", "game_number": 1,
            "league": "LCS", "patch": "26.17",
            "roster_change_stratum": "UNCHANGED",
            "sparse_or_new_champion_map": False,
            "market_type": "match_winner", "selection": "winner:blue",
            "opposing_selection": "winner:red",
        },
        "probability_binding": {"locator": "probability.json"},
        "reserved_outputs": {
            "quote_locator": "quotes/quote.json",
            "qualification_locator": "qualified/quote.json",
            "failure_locator": "failures/failure.json",
            "completion_locator": "completions/completion.json",
        },
    }


def _install(monkeypatch: pytest.MonkeyPatch) -> dict:
    plan = _plan()
    monkeypatch.setattr(
        attempt, "_plan",
        lambda **_kwargs: ("plans/plan.json", b"plan", plan),
    )
    monkeypatch.setattr(
        attempt, "_source_locks",
        lambda _root: [{"locator": "attempt.py", "bytes": 1, "raw_sha256": "2" * 64}],
    )
    return plan


def test_failed_attempt_is_persisted_without_exception_text_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    monkeypatch.setattr(
        attempt.quote_v2,
        "capture_betano_map_winner_quote_v2",
        lambda **_kwargs: (_ for _ in ()).throw(
            attempt.quote_v2.BetanoQuoteAdapterV2Error(
                "exact map-winner market is missing or ambiguous"
            )
        ),
    )
    result = attempt.run_planned_quote_attempt_v1(
        event_plan_locator="plans/plan.json",
        request_url="https://example.test/event/1",
        betano_event_id="1",
        map_number=1,
        participant_bindings=[],
        fetcher=object(),
        root=tmp_path,
        environment={},
        clock=lambda: datetime(2026, 9, 1, 15, 0, 2, tzinfo=timezone.utc),
        monotonic_ns=lambda: 1,
    )
    assert result["status"] == "failed_persisted"
    assert result["failure_code"] == "MARKET_NOT_OPEN_OR_MISSING"
    payload = json.loads((tmp_path / "failures/failure.json").read_text())
    assert payload["failure"]["free_form_exception_text_persisted"] is False
    assert payload["failure"]["request_headers_cookies_or_credentials_persisted"] is False
    assert payload["failure"]["counts_in_quote_coverage_denominator"] is True
    assert payload["authority"]["betting_authority"] is False


def test_success_and_failure_are_mutually_exclusive_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch)
    monkeypatch.setattr(
        attempt.quote_v2,
        "capture_betano_map_winner_quote_v2",
        lambda **_kwargs: {"artifact_sha256": "3" * 64},
    )
    result = attempt.run_planned_quote_attempt_v1(
        event_plan_locator="plans/plan.json",
        request_url="https://example.test/event/1",
        betano_event_id="1",
        map_number=1,
        participant_bindings=[],
        fetcher=object(),
        root=tmp_path,
        environment={},
        clock=lambda: datetime(2026, 9, 1, 15, 0, 2, tzinfo=timezone.utc),
        monotonic_ns=lambda: 1,
    )
    assert result["status"] == "quote_persisted"
    assert (tmp_path / "quotes/quote.json").is_file()
    assert not (tmp_path / "failures/failure.json").exists()
    with pytest.raises(attempt.PhaseTwoQuoteAttemptError, match="already consumed"):
        attempt.run_planned_quote_attempt_v1(
            event_plan_locator="plans/plan.json",
            request_url="https://example.test/event/1",
            betano_event_id="1",
            map_number=1,
            participant_bindings=[],
            fetcher=object(),
            root=tmp_path,
            environment={},
        )
