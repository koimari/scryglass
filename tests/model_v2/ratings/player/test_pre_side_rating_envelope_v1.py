from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import (
    multileague_v3_prediction_ledger as rating_ledger,
)
from lol_kills.v2.ratings.player import pre_side_rating_envelope_v1 as envelope
from lol_kills.v2.ratings.player import pre_side_rating_binding_v1 as binding


EVENT_ID = "future-lcs-side-neutral-map-1"
SERIES_ID = "future-lcs-side-neutral-series-1"
EVENT_START = "2026-08-08T20:00:00Z"
CAPTURED_AT = datetime(2026, 8, 2, 0, 0, tzinfo=timezone.utc)
SOURCE_RAW = b'{"kind":"synthetic-pre-event-test-roster"}'
SIDE_CAPTURED_AT = datetime(2026, 8, 8, 20, 1, tzinfo=timezone.utc)
SIDE_SOURCE_RAW = (
    b'{"match":{"blue":{"name":"Cloud9"},'
    b'"red":{"name":"Team Liquid"}},"state":"champion_select"}'
)


def _teams() -> list[dict]:
    values = (
        (
            "team1",
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
            "team2",
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
            "slot": slot,
            "organization_id": organization_id,
            "organization_name": organization_name,
            "roster_id": f"{EVENT_ID}-{slot}",
            "players": [
                {"role": role, "player_id": player_id, "display_name": name}
                for role, player_id, name in players
            ],
        }
        for slot, organization_id, organization_name, players in values
    ]


def _input() -> dict:
    return {
        "schema_version": envelope.INPUT_SCHEMA_VERSION,
        "event": {
            "event_id": EVENT_ID,
            "series_id": SERIES_ID,
            "game_number": 1,
            "scheduled_series_start_utc": EVENT_START,
            "league": "LCS",
        },
        "roster_source": {
            "source": "test-fixture",
            "source_url": "https://example.test/future-lcs-side-neutral-map-1",
            "source_record_id": f"{EVENT_ID}-roster",
            "source_updated_at_utc": "2026-08-01T23:56:00Z",
            "available_at_utc": "2026-08-01T23:56:30Z",
            "rights_status": "reviewed",
        },
        "teams": _teams(),
    }


def _input_raw(value: dict | None = None) -> bytes:
    return (
        json.dumps(value or _input(), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()


def _patch_raw() -> bytes:
    evidence = {
        "revision_timestamp": "2026-08-01T23:40:00Z",
        "revision_id": 123456,
        "source_kind": "leaguepedia_data_page_revision",
        "content_sha256": "b" * 64,
    }
    receipt = {
        "schema_version": rating_ledger.PATCH_RECEIPT_SCHEMA,
        "fixture_id": EVENT_ID,
        "event_start": EVENT_START,
        "as_of": "2026-08-01T23:58:00Z",
        "patch": "26.15",
        "client_patch": "16.15",
        "authority_status": "pre_event_revision",
        "pregame_authorized": True,
        "blockers": [],
        "evidence": evidence,
        "evidence_hash": rating_ledger._canonical_sha256(
            {"fixture_id": EVENT_ID, "evidence": evidence}
        ),
    }
    return (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()


def _binding_input() -> dict:
    return {
        "schema_version": binding.INPUT_SCHEMA_VERSION,
        "event": {
            "event_id": EVENT_ID,
            "series_id": SERIES_ID,
            "game_number": 1,
        },
        "source": {
            "source": "test-public-side-feed",
            "source_url": "https://example.test/public-side-feed/map-1",
            "source_record_id": f"{EVENT_ID}-side-observation",
            "source_updated_at_utc": "2026-08-08T20:00:20Z",
            "available_at_utc": "2026-08-08T20:00:30Z",
            "rights_status": "reviewed",
        },
        "extraction": {
            "format": "strict_json_pointer_v1",
            "blue_organization_name_json_pointer": "/match/blue/name",
            "red_organization_name_json_pointer": "/match/red/name",
        },
    }


def _binding_input_raw(value: dict | None = None) -> bytes:
    return (
        json.dumps(
            value or _binding_input(), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode()


@pytest.fixture(scope="module")
def receipt() -> dict:
    return envelope.build_pre_side_rating_envelope(
        input_raw=_input_raw(),
        roster_source_payload_raw=SOURCE_RAW,
        patch_receipt_raw=_patch_raw(),
        root=Path(".").resolve(),
        clock=lambda: CAPTURED_AT,
    )


@pytest.fixture(scope="module")
def side_binding(receipt: dict) -> dict:
    return binding.build_pre_side_rating_binding(
        envelope_raw=(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode(),
        binding_input_raw=_binding_input_raw(),
        public_side_source_raw=SIDE_SOURCE_RAW,
        root=Path(".").resolve(),
        clock=lambda: SIDE_CAPTURED_AT,
    )


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = envelope._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_envelope_seals_both_orientations_without_authority(receipt: dict) -> None:
    assert receipt["result_state"] == envelope.RESULT_STATE
    assert set(receipt["side_conditionals"]) == {"team1_blue", "team2_blue"}
    assert receipt["qualification"]["actual_blue_red_side_known"] is False
    assert receipt["qualification"]["side_binding_present"] is False
    assert receipt["qualification"]["eligible_evaluation_map"] is False
    assert receipt["qualification"][
        "embedded_child_receipts_individually_ledger_eligible"
    ] is False
    assert all(value is False for value in receipt["authority"].values())
    for summary in receipt["conditional_summary"]["models"].values():
        assert summary["p_team1_if_red"] != pytest.approx(
            1.0 - summary["p_team1_if_blue"]
        )
        assert summary["selected_probability_uses_embedded_orientation"] is True
        assert summary["complementarity_assumed"] is False


def test_envelope_survives_json_round_trip_and_binds_exact_rosters(
    receipt: dict,
) -> None:
    checked = envelope.validate_pre_side_rating_envelope(
        json.loads(json.dumps(receipt, sort_keys=True)), root=Path(".").resolve()
    )
    assert checked["artifact_sha256"] == receipt["artifact_sha256"]

    forged = deepcopy(receipt)
    forged["source_order_teams"][0]["players"][0]["display_name"] = "Attacker"
    _resign(forged)
    with pytest.raises(envelope.PreSideRatingEnvelopeError, match="exact player roster"):
        envelope.validate_pre_side_rating_envelope(
            forged, root=Path(".").resolve()
        )


def test_envelope_rejects_resigned_source_or_authority_forgery(receipt: dict) -> None:
    forged_source = deepcopy(receipt)
    forged_source["roster_source"]["source_record_id"] = "attacker-source"
    _resign(forged_source)
    with pytest.raises(envelope.PreSideRatingEnvelopeError, match="roster source"):
        envelope.validate_pre_side_rating_envelope(
            forged_source, root=Path(".").resolve()
        )

    forged_authority = deepcopy(receipt)
    forged_authority["authority"]["probability_authority"] = True
    _resign(forged_authority)
    with pytest.raises(envelope.PreSideRatingEnvelopeError, match="authority"):
        envelope.validate_pre_side_rating_envelope(
            forged_authority, root=Path(".").resolve()
        )


def test_envelope_rejects_outcome_leakage_before_model_fit() -> None:
    leaked = _input()
    leaked["event"]["winner"] = "team1"
    with pytest.raises(envelope.PreSideRatingEnvelopeError, match="outcome field"):
        envelope.build_pre_side_rating_envelope(
            input_raw=_input_raw(leaked),
            roster_source_payload_raw=SOURCE_RAW,
            patch_receipt_raw=_patch_raw(),
            root=Path(".").resolve(),
            clock=lambda: CAPTURED_AT,
        )


def test_envelope_rejects_non_prospective_clock() -> None:
    with pytest.raises(envelope.PreSideRatingEnvelopeError, match="not pre-event"):
        envelope.build_pre_side_rating_envelope(
            input_raw=_input_raw(),
            roster_source_payload_raw=SOURCE_RAW,
            patch_receipt_raw=_patch_raw(),
            root=Path(".").resolve(),
            clock=lambda: datetime(2026, 8, 8, 20, 0, tzinfo=timezone.utc),
        )


def test_envelope_write_is_no_clobber(receipt: dict, tmp_path: Path) -> None:
    output = tmp_path / "envelope.json"
    envelope.write_no_clobber(output, receipt)
    with pytest.raises(envelope.PreSideRatingEnvelopeError, match="overwrite"):
        envelope.write_no_clobber(output, receipt)


def test_side_binding_selects_existing_orientation_without_refit(
    receipt: dict, side_binding: dict
) -> None:
    assert side_binding["selection"]["scenario"] == "team2_blue"
    assert side_binding["selection"]["blue_organization_name"] == "Cloud9"
    assert side_binding["selection"]["red_organization_name"] == "Team Liquid"
    selected = receipt["side_conditionals"]["team2_blue"]["rating_receipt"]
    assert side_binding["selection"][
        "selected_rating_receipt_raw_sha256"
    ] == selected["raw_sha256"]
    assert side_binding["selection"][
        "selected_rating_receipt_artifact_sha256"
    ] == selected["value"]["artifact_sha256"]
    assert side_binding["selection"]["rating_recomputed_after_side_observation"] is False
    assert side_binding["qualification"]["eligible_evaluation_map"] is False
    assert side_binding["qualification"][
        "binding_before_actual_map_start_verified"
    ] is False
    assert all(value is False for value in side_binding["authority"].values())


def test_side_binding_survives_round_trip(side_binding: dict) -> None:
    checked = binding.validate_pre_side_rating_binding(
        json.loads(json.dumps(side_binding, sort_keys=True)), root=Path(".").resolve()
    )
    assert checked["artifact_sha256"] == side_binding["artifact_sha256"]


def test_side_binding_rejects_unknown_or_schedule_order_names(receipt: dict) -> None:
    unknown_raw = SIDE_SOURCE_RAW.replace(b"Cloud9", b"Team1x")
    with pytest.raises(binding.PreSideRatingBindingError, match="exactly match"):
        binding.build_pre_side_rating_binding(
            envelope_raw=(json.dumps(receipt, sort_keys=True) + "\n").encode(),
            binding_input_raw=_binding_input_raw(),
            public_side_source_raw=unknown_raw,
            root=Path(".").resolve(),
            clock=lambda: SIDE_CAPTURED_AT,
        )


def test_side_binding_rejects_outcome_bearing_source(receipt: dict) -> None:
    leaked_source = json.dumps(
        {
            "match": {
                "blue": {"name": "Cloud9"},
                "red": {"name": "Team Liquid"},
            },
            "winner": "Cloud9",
        }
    ).encode()
    with pytest.raises(binding.PreSideRatingBindingError, match="outcome field"):
        binding.build_pre_side_rating_binding(
            envelope_raw=(json.dumps(receipt, sort_keys=True) + "\n").encode(),
            binding_input_raw=_binding_input_raw(),
            public_side_source_raw=leaked_source,
            root=Path(".").resolve(),
            clock=lambda: SIDE_CAPTURED_AT,
        )


def test_side_binding_rejects_source_from_the_future(receipt: dict) -> None:
    future = _binding_input()
    future["source"]["available_at_utc"] = "2026-08-08T20:01:01Z"
    with pytest.raises(binding.PreSideRatingBindingError, match="predates its public source"):
        binding.build_pre_side_rating_binding(
            envelope_raw=(json.dumps(receipt, sort_keys=True) + "\n").encode(),
            binding_input_raw=_binding_input_raw(future),
            public_side_source_raw=SIDE_SOURCE_RAW,
            root=Path(".").resolve(),
            clock=lambda: SIDE_CAPTURED_AT,
        )


def test_side_binding_rejects_resigned_selection_forgery(side_binding: dict) -> None:
    forged = deepcopy(side_binding)
    forged["selection"]["scenario"] = "team1_blue"
    forged["artifact_sha256"] = binding._canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(binding.PreSideRatingBindingError, match="selected rating"):
        binding.validate_pre_side_rating_binding(
            forged, root=Path(".").resolve()
        )


def test_side_binding_write_is_no_clobber(side_binding: dict, tmp_path: Path) -> None:
    output = tmp_path / "side-binding.json"
    binding.write_no_clobber(output, side_binding)
    with pytest.raises(binding.PreSideRatingBindingError, match="overwrite"):
        binding.write_no_clobber(output, side_binding)
