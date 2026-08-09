from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from lol_kills import pregame_roster_capture as roster_capture
from lol_kills.v2.ratings.player import multileague_v3_prediction_ledger as ledger


EVENT_ID = "future-lcs-evaluation-map-1"
EVENT_START = "2026-08-08T20:00:00Z"
CAPTURED_AT = datetime(2026, 8, 2, 0, 0, tzinfo=timezone.utc)


def _teams() -> list[dict]:
    values = (
        (
            "blue",
            "oe:team:8eb884e168f28402ce685bedebb5250",
            "Team Liquid",
            (
                ("top", "oe:player:f73dde2d8c7879ae8d1fcdc03c4ffeb", "Morgan"),
                ("jungle", "oe:player:52b4f060a577ff87c99118d97fa75f4", "Josedeodo"),
                ("mid", "oe:player:8f8eaac8e99c203ee76e57c5c81c91f", "Quid"),
                ("bot", "oe:player:f6b8af05a5af22004a67a78f3046b92", "Yeon"),
                ("support", "oe:player:05fafeaf23dc3285558a4e968970b9f", "CoreJJ"),
            ),
        ),
        (
            "red",
            "oe:team:fc8e90107dabb9a35c490b0d86adea0",
            "Cloud9",
            (
                ("top", "oe:player:1efc4e5732e91520d03385900b903f9", "Thanatos"),
                ("jungle", "oe:player:ed455f823b6586e9e3621dfb5aeb42e", "Blaber"),
                ("mid", "oe:player:0a53c1102dafce61e89decc517b3175", "Loki"),
                ("bot", "oe:player:4e0619e57cb50c4ab1230c2dd72df1b", "Tactical"),
                ("support", "oe:player:81acc6bcde5c927c78e0990e9887f84", "Vulcan"),
            ),
        ),
    )
    return [
        {
            "side": side,
            "organization_id": organization_id,
            "organization_name": organization_name,
            "roster_id": f"{EVENT_ID}-{side}",
            "players": [
                {"role": role, "player_id": player_id, "display_name": name}
                for role, player_id, name in players
            ],
        }
        for side, organization_id, organization_name, players in values
    ]


def _roster_raw() -> bytes:
    receipt = roster_capture.build_pregame_roster_receipt(
        raw_source_payload=b'{"kind":"synthetic-pre-event-test-roster"}',
        source="test-fixture",
        source_url="https://example.test/future-lcs-evaluation-map-1",
        source_record_id="future-lcs-evaluation-map-1-roster",
        source_updated_at="2026-08-01T23:56:00Z",
        available_at="2026-08-01T23:56:30Z",
        captured_at="2026-08-01T23:57:00Z",
        event_id=EVENT_ID,
        event_start=EVENT_START,
        league="LCS",
        teams=_teams(),
        capture_protocol_sha256="a" * 64,
    )
    return (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()


def _patch_raw(*, fixture_id: str = EVENT_ID) -> bytes:
    evidence = {
        "revision_timestamp": "2026-08-01T23:40:00Z",
        "revision_id": 123456,
        "source_kind": "leaguepedia_data_page_revision",
        "content_sha256": "b" * 64,
    }
    receipt = {
        "schema_version": ledger.PATCH_RECEIPT_SCHEMA,
        "fixture_id": fixture_id,
        "event_start": EVENT_START,
        "as_of": "2026-08-01T23:58:00Z",
        "patch": "26.15",
        "client_patch": "16.15",
        "authority_status": "pre_event_revision",
        "pregame_authorized": True,
        "blockers": [],
        "evidence": evidence,
        "evidence_hash": ledger._canonical_sha256(
            {"fixture_id": fixture_id, "evidence": evidence}
        ),
    }
    return (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()


@pytest.fixture(scope="module")
def prediction_receipt() -> dict:
    return ledger.build_pre_event_prediction_receipt(
        roster_receipt_raw=_roster_raw(),
        patch_receipt_raw=_patch_raw(),
        series_id="future-lcs-evaluation-series-1",
        game_number=1,
        root=Path(".").resolve(),
        clock=lambda: CAPTURED_AT,
    )


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = ledger._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_capture_freezes_candidate_and_both_comparators_before_event(
    prediction_receipt: dict,
) -> None:
    assert prediction_receipt["result_state"] == ledger.RESULT_STATE
    assert set(prediction_receipt["evaluation_predictions"]) == set(ledger.MODEL_IDS)
    for prediction in prediction_receipt["evaluation_predictions"].values():
        assert prediction["p_blue"] + prediction["p_red"] == pytest.approx(1.0)
    assert prediction_receipt["event"]["roster_change_stratum"] == (
        "BOTH_ROSTERS_STABLE"
    )
    assert prediction_receipt["qualification"]["event_outcome_present"] is False
    assert prediction_receipt["qualification"]["event_outcome_accessed"] is False
    assert prediction_receipt["qualification"][
        "system_clock_sampled_inside_builder"
    ] is True
    assert prediction_receipt["clock_attestation"][
        "user_supplied_timestamp_allowed"
    ] is False
    assert all(value is False for value in prediction_receipt["authority"].values())


def test_rating_diagnostics_preserve_unavailable_components(
    prediction_receipt: dict,
) -> None:
    diagnostics = prediction_receipt["event_rating_diagnostics"]
    assert len(diagnostics["players"]) == 10
    assert len(diagnostics["teams"]) == 2
    for team in diagnostics["teams"]:
        assert team["components"]["lineup_synergy"] == {
            "status": "UNAVAILABLE",
            "posterior_mean_logit": None,
            "posterior_sd_logit": None,
        }
        assert team["components"]["team_policy"]["posterior_mean_logit"] is None
        assert team["unavailable_components_are_not_zero"] is True


def test_receipt_survives_sorted_json_round_trip(prediction_receipt: dict) -> None:
    round_trip = json.loads(json.dumps(prediction_receipt, sort_keys=True))
    checked = ledger.validate_pre_event_prediction_receipt(
        round_trip,
        root=Path(".").resolve(),
    )
    assert checked["artifact_sha256"] == prediction_receipt["artifact_sha256"]


def test_pre_event_prediction_replays_exactly(prediction_receipt: dict) -> None:
    checked = ledger.replay_pre_event_prediction_receipt(
        prediction_receipt,
        root=Path(".").resolve(),
    )
    assert checked["evaluation_predictions"] == prediction_receipt[
        "evaluation_predictions"
    ]


def test_receipt_rejects_forged_authority_or_event_outcome(
    prediction_receipt: dict,
) -> None:
    forged = deepcopy(prediction_receipt)
    forged["authority"]["probability_authority"] = True
    _resign(forged)
    with pytest.raises(ledger.PredictionLedgerError, match="exceeds authority"):
        ledger.validate_pre_event_prediction_receipt(
            forged,
            root=Path(".").resolve(),
        )

    leaked = deepcopy(prediction_receipt)
    leaked["event"]["winner"] = "blue"
    _resign(leaked)
    with pytest.raises(ledger.PredictionLedgerError, match="outcome field"):
        ledger.validate_pre_event_prediction_receipt(
            leaked,
            root=Path(".").resolve(),
        )


def test_outcome_free_registry_counts_metadata_without_opening(
    prediction_receipt: dict,
) -> None:
    value = ledger.build_prediction_ledger_registry(
        receipts=[
            (
                "data/lol/v2/evaluation/multileague-v3/predictions/future-1.json",
                prediction_receipt,
            )
        ],
        root=Path(".").resolve(),
        clock=lambda: datetime(2026, 8, 2, 0, 1, tzinfo=timezone.utc),
    )
    assert value["status"] == "COLLECTING_OUTCOME_FREE_PREDICTIONS"
    assert value["metadata_support"]["overall_series"] == 1
    assert value["outcomes_present"] is False
    assert value["independently_pinned"] is False
    assert value["opening_authority"] is False
    assert value["clock_attestation"]["user_supplied_timestamp_allowed"] is False


def test_capture_rejects_mismatched_fixture_before_model_fit() -> None:
    with pytest.raises(ledger.PredictionLedgerError, match="identities differ"):
        ledger.build_pre_event_prediction_receipt(
            roster_receipt_raw=_roster_raw(),
            patch_receipt_raw=_patch_raw(fixture_id="attacker-fixture"),
            series_id="future-lcs-evaluation-series-1",
            game_number=1,
            root=Path(".").resolve(),
            clock=lambda: CAPTURED_AT,
        )


def test_receipt_rejects_forged_clock_attestation(prediction_receipt: dict) -> None:
    forged = deepcopy(prediction_receipt)
    forged["clock_attestation"]["user_supplied_timestamp_allowed"] = True
    _resign(forged)
    with pytest.raises(ledger.PredictionLedgerError, match="clock attestation"):
        ledger.validate_pre_event_prediction_receipt(
            forged,
            root=Path(".").resolve(),
        )


def test_capture_rejects_naive_builder_clock() -> None:
    with pytest.raises(ledger.PredictionLedgerError, match="timezone-aware"):
        ledger.build_pre_event_prediction_receipt(
            roster_receipt_raw=_roster_raw(),
            patch_receipt_raw=_patch_raw(),
            series_id="future-lcs-evaluation-series-1",
            game_number=1,
            root=Path(".").resolve(),
            clock=lambda: datetime(2026, 8, 2, 0, 0),
        )
