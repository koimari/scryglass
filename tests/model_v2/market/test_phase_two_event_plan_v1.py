from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from lol_kills.v2.market import phase_two_event_plan_v1 as plan


def _install(monkeypatch: pytest.MonkeyPatch) -> None:
    receipt = {
        "captured_at_utc": "2026-09-01T15:00:00+00:00",
        "artifact_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
        "event": {
            "event_id": "event-1", "series_id": "series-1", "game_number": 1,
            "league": "LCS", "patch": "26.17",
            "roster_change_stratum": "UNCHANGED",
            "sparse_or_new_champion_map": False,
            "market_type": "match_winner", "selection": "winner:blue",
            "opposing_selection": "winner:red",
        },
    }
    monkeypatch.setattr(
        plan, "_probability",
        lambda **_kwargs: ("probability.json", b"probability", receipt),
    )
    monkeypatch.setattr(
        plan, "_event_start",
        lambda _root, _receipt, _environment: datetime(
            2026, 9, 1, 16, 0, tzinfo=timezone.utc
        ),
    )
    monkeypatch.setattr(
        plan, "_source_locks",
        lambda _root: [{"locator": "plan.py", "bytes": 1, "raw_sha256": "3" * 64}],
    )
    monkeypatch.setattr(
        plan.evaluation, "_locator", lambda value, _prefix, _field: str(value)
    )


def test_plan_freezes_denominator_before_quote_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    receipt = plan.build_phase_two_event_plan_v1(
        event_probability_locator="probability.json",
        quote_output_locator="quote.json",
        qualification_output_locator="qualified.json",
        failure_output_locator="failure.json",
        completion_output_locator="completion.json",
        environment={},
        clock=lambda: datetime(2026, 9, 1, 15, 0, 1, tzinfo=timezone.utc),
    )
    checked = plan.validate_phase_two_event_plan_v1(receipt, environment={})
    assert checked["denominator_contract"]["plan_persists_if_request_or_quote_fails"] is True
    assert checked["denominator_contract"]["success_only_plan_creation_permitted"] is False
    assert all(value is False for value in checked["authority"].values())


def test_plan_rejects_retrospective_or_authorizing_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    with pytest.raises(plan.PhaseTwoEventPlanError, match="before event"):
        plan.build_phase_two_event_plan_v1(
            event_probability_locator="probability.json",
            quote_output_locator="quote.json",
            qualification_output_locator="qualified.json",
            failure_output_locator="failure.json",
            completion_output_locator="completion.json",
            environment={},
            clock=lambda: datetime(2026, 9, 1, 16, 0, tzinfo=timezone.utc),
        )
    receipt = plan.build_phase_two_event_plan_v1(
        event_probability_locator="probability.json",
        quote_output_locator="quote.json",
        qualification_output_locator="qualified.json",
        failure_output_locator="failure.json",
        completion_output_locator="completion.json",
        environment={},
        clock=lambda: datetime(2026, 9, 1, 15, 0, 1, tzinfo=timezone.utc),
    )
    forged = deepcopy(receipt)
    forged["authority"]["betting_authority"] = True
    forged["artifact_sha256"] = plan._canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(plan.PhaseTwoEventPlanError, match="exceeds authority"):
        plan.validate_phase_two_event_plan_v1(forged, environment={})
