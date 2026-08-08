from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from lol_kills.v2.market import betano_br_quote_qualification_v1 as qualification


def _install(monkeypatch: pytest.MonkeyPatch, *, start: str = "2026-09-01T15:00:10+00:00") -> None:
    probability = {
        "artifact_sha256": "1" * 64,
        "captured_at_utc": "2026-09-01T15:00:00+00:00",
        "event": {
            "event_id": "event-1", "series_id": "series-1", "game_number": 1,
            "league": "LCS", "market_type": "match_winner",
            "patch": "26.17", "roster_change_stratum": "UNCHANGED",
            "sparse_or_new_champion_map": False,
            "selection": "winner:blue", "opposing_selection": "winner:red",
        },
        "opening_binding": {"authority_id": "opening-1", "marker_raw_sha256": "2" * 64},
    }
    quote = {
        "artifact_sha256": "3" * 64,
        "event_probability_v2_binding": {"locator": "probability.json"},
        "event_plan_binding": {
            "locator": "plan.json", "raw_sha256": "7" * 64,
            "artifact_sha256": "8" * 64,
            "planned_at_utc": "2026-09-01T15:00:00.500000+00:00",
        },
        "frozen_v1_transport_quote": {
            "transport": {
                "request_started_at_utc": "2026-09-01T15:00:01+00:00",
                "response_received_at_utc": "2026-09-01T15:00:02+00:00",
            },
            "generic_quote_receipt_sha256": "4" * 64,
        },
    }
    map_start = {
        "artifact_sha256": "5" * 64,
        "captured_at_utc": "2026-09-01T15:00:11+00:00",
        "event": {
            "event_id": "event-1", "series_id": "series-1", "game_number": 1,
            "league": "LCS", "actual_map_start_utc": start,
        },
    }
    monkeypatch.setattr(
        qualification, "_quote",
        lambda **_kwargs: ("quote.json", b"quote", quote),
    )
    monkeypatch.setattr(
        qualification, "_map_start",
        lambda **_kwargs: ("start.json", b"start", map_start),
    )
    monkeypatch.setattr(
        qualification.quote_v2, "_probability",
        lambda **_kwargs: ("probability.json", b"probability", probability),
    )
    monkeypatch.setattr(
        qualification, "_source_locks",
        lambda _root: [{"locator": "qualification.py", "bytes": 1, "raw_sha256": "6" * 64}],
    )
    monkeypatch.setattr(
        qualification.evaluation,
        "_locator",
        lambda value, _prefix, _field: str(value),
    )
    monkeypatch.setattr(
        qualification.evaluation,
        "_read_regular",
        lambda _root, _locator, _label: b"plan",
    )
    monkeypatch.setattr(
        qualification.evaluation,
        "_strict_object",
        lambda _raw, _label: {},
    )
    monkeypatch.setattr(
        qualification.quote_v2.event_plan,
        "validate_phase_two_event_plan_v1",
        lambda _payload, **_kwargs: {
            "reserved_outputs": {
                "quote_locator": "quote.json",
                "qualification_locator": "qualified.json",
            }
        },
    )


def test_quote_is_qualified_only_after_authoritative_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    receipt = qualification.build_betano_quote_qualification_v1(
        quote_locator="quote.json",
        map_start_locator="start.json",
        qualification_output_locator="qualified.json",
        environment={},
        clock=lambda: datetime(2026, 9, 1, 15, 0, 12, tzinfo=timezone.utc),
    )
    checked = qualification.validate_betano_quote_qualification_v1(
        receipt, environment={}
    )
    assert checked["timing"]["quote_response_to_actual_map_start_seconds"] == 8.0
    assert checked["qualification"]["minimum_five_second_boundary_passed"] is True
    assert checked["qualification"]["event_outcome_accessed"] is False
    assert all(value is False for value in checked["authority"].values())


def test_quote_rejects_late_response_and_forged_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch, start="2026-09-01T15:00:06+00:00")
    with pytest.raises(
        qualification.BetanoQuoteQualificationError,
        match="five seconds",
    ):
        qualification.build_betano_quote_qualification_v1(
                quote_locator="quote.json",
                map_start_locator="start.json",
                qualification_output_locator="qualified.json",
            environment={},
            clock=lambda: datetime(2026, 9, 1, 15, 0, 12, tzinfo=timezone.utc),
        )

    _install(monkeypatch)
    receipt = qualification.build_betano_quote_qualification_v1(
        quote_locator="quote.json",
        map_start_locator="start.json",
        qualification_output_locator="qualified.json",
        environment={},
        clock=lambda: datetime(2026, 9, 1, 15, 0, 12, tzinfo=timezone.utc),
    )
    forged = deepcopy(receipt)
    forged["authority"]["betting_authority"] = True
    forged["artifact_sha256"] = qualification._canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(
        qualification.BetanoQuoteQualificationError,
        match="exceeds authority",
    ):
        qualification.validate_betano_quote_qualification_v1(
            forged, environment={}
        )
