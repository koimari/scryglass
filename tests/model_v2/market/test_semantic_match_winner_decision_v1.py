from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from lol_kills.v2.market import semantic_match_winner_decision_v1 as decision


AS_OF = datetime(2026, 10, 5, 12, 0, 10, tzinfo=timezone.utc)
AUTHORITY = {
    "receipt": {
        "authority_id": "semantic-authority-1",
        "decision_policy": {
            "minimum_lower_bound_expected_return": 0.02,
            "maximum_probability_age_seconds": 60.0,
            "maximum_quote_age_seconds": 30.0,
            "positive_expected_return_haircut_fraction": 0.01,
        },
        "claim_ceiling": "private calculation only",
    },
    "receipt_raw_sha256": "a" * 64,
}
QUOTE = {
    "semantic_market_authority_binding": {
        "authority_raw_sha256": "a" * 64,
    },
    "response_received_at_utc": "2026-10-05T12:00:05+00:00",
    "prices": {"winner:blue": 2.0, "winner:red": 1.8},
    "probability": {
        "captured_at_utc": "2026-10-05T12:00:00+00:00",
        "semantic_market_authority_binding": {
            "authority_raw_sha256": "a" * 64,
        },
        "event": {
            "event_id": "event-1",
            "series_id": "series-1",
            "game_number": 1,
            "league": "LCS",
            "selection": "winner:blue",
            "opposing_selection": "winner:red",
            "scheduled_event_start_utc": "2026-10-05T12:30:00+00:00",
        },
        "probability": 0.60,
        "probability_interval": [0.55, 0.65],
    },
}


def _install(
    monkeypatch: pytest.MonkeyPatch,
    *, quote: dict | None = None,
    authority: dict | None = None,
) -> None:
    monkeypatch.setattr(
        decision,
        "_active_authority",
        lambda **_kwargs: deepcopy(authority or AUTHORITY),
    )
    monkeypatch.setattr(
        decision,
        "_quote",
        lambda **_kwargs: (
            "production-quotes-v1/quote.json",
            b"canonical quote bytes",
            deepcopy(quote or QUOTE),
        ),
    )


def test_semantic_decision_authorizes_one_lower_bound_positive_side_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    result = decision.evaluate_semantic_match_winner_v1(
        production_quote_locator="ignored.json",
        environment={},
        as_of=AS_OF,
    )
    assert result["status"] == "authorized"
    assert result["decision"] == "BET"
    assert result["selection"] == "winner:blue"
    assert result["evaluated_selection"] == "winner:blue"
    assert len(result["candidate_evaluations"]) == 2
    assert [item["qualifies"] for item in result["candidate_evaluations"]] == [
        True,
        False,
    ]
    assert result["authorized_probability"] == pytest.approx(0.60)
    assert result["probability_interval"] == [0.55, 0.65]
    assert result["fair_decimal_odds"] == pytest.approx(1.0 / 0.60)
    assert result["expected_return"] == pytest.approx(0.20)
    assert result["lower_bound_expected_return_after_haircut"] == pytest.approx(
        0.55 * 1.0 * 0.99 - 0.45
    )
    assert result["stake"] is None
    assert result["transaction_authorized"] is False


def test_semantic_decision_passes_when_neither_side_clears_frozen_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quote = deepcopy(QUOTE)
    quote["prices"] = {"winner:blue": 1.55, "winner:red": 1.55}
    _install(monkeypatch, quote=quote)
    result = decision.evaluate_semantic_match_winner_v1(
        production_quote_locator="ignored.json",
        environment={},
        as_of=AS_OF,
    )
    assert result["status"] == "authorized"
    assert result["decision"] == "PASS"
    assert result["selection"] is None
    assert result["evaluated_selection"] == "winner:blue"
    assert all(
        item["qualifies"] is False for item in result["candidate_evaluations"]
    )
    assert result["lower_bound_expected_return_after_haircut"] < 0.02
    assert result["stake"] is None
    assert result["transaction_authorized"] is False


def test_semantic_decision_withholds_all_numbers_when_quote_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quote = deepcopy(QUOTE)
    quote["response_received_at_utc"] = "2026-10-05T11:59:00+00:00"
    _install(monkeypatch, quote=quote)
    result = decision.evaluate_semantic_match_winner_v1(
        production_quote_locator="ignored.json",
        environment={},
        as_of=AS_OF,
    )
    assert result["status"] == "unavailable"
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert "production_quote_stale" in result["blockers"]
    for field in (
        "authorized_probability",
        "probability_interval",
        "fair_decimal_odds",
        "offered_decimal_odds",
        "no_vig_market_probability",
        "expected_return",
        "lower_bound_expected_return_after_haircut",
        "stake",
    ):
        assert result[field] is None
    assert result["selection"] is None
    assert result["evaluated_selection"] is None
    assert result["candidate_evaluations"] is None
    assert result["transaction_authorized"] is False


def test_semantic_decision_rejects_logically_inconsistent_two_sided_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quote = deepcopy(QUOTE)
    quote["probability"]["probability"] = 0.50
    quote["probability"]["probability_interval"] = [0.49, 0.51]
    quote["prices"] = {"winner:blue": 3.0, "winner:red": 3.0}
    _install(monkeypatch, quote=quote)
    result = decision.evaluate_semantic_match_winner_v1(
        production_quote_locator="ignored.json",
        environment={},
        as_of=AS_OF,
    )
    assert result["status"] == "unavailable"
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert result["blockers"] == ["both_sides_qualify_inconsistent"]
    assert result["authorized_probability"] is None
    assert result["stake"] is None
    assert result["transaction_authorized"] is False


def test_semantic_decision_fails_closed_when_authority_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        decision,
        "_active_authority",
        lambda **_kwargs: (_ for _ in ()).throw(
            decision.SemanticMatchWinnerDecisionError(
                "semantic market authority is unavailable"
            )
        ),
    )
    result = decision.evaluate_semantic_match_winner_v1(
        production_quote_locator="ignored.json",
        environment={},
        as_of=AS_OF,
    )
    assert result["status"] == "unavailable"
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert result["blockers"] == ["semantic market authority is unavailable"]
