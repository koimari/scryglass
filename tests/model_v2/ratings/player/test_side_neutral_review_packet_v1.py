from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import side_neutral_review_packet_v1 as packet


TEST_CLOCK = datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc)


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = packet._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


@pytest.fixture(scope="module")
def review_packet() -> dict:
    return packet.build_side_neutral_review_packet(
        root=Path(".").resolve(), clock=lambda: TEST_CLOCK
    )


def test_packet_binds_protocol_capture_and_admission_without_authority(
    review_packet: dict,
) -> None:
    assert review_packet["result_state"] == packet.RESULT_STATE
    assert review_packet["pre_review_inventory"]["complete_bundles"] == 0
    assert len(review_packet["protocol"]["source_locks"]) >= 8
    assert {
        item["role"] for item in review_packet["admission_implementation"]
    } == {"external_review_validation", "post_review_bundle_admission"}
    assert len(review_packet["review_questions"]) == 10
    assert review_packet["requested_authorization"][
        "prospective_outcome_free_collection_only"
    ] is True
    assert review_packet["requested_authorization"]["betting"] is False
    assert all(value is False for value in review_packet["authority"].values())


def test_checked_in_packet_replays_exactly() -> None:
    path = Path(".").resolve() / packet.DEFAULT_OUTPUT
    raw = path.read_bytes()
    checked = packet.validate_side_neutral_review_packet(
        json.loads(raw), root=Path(".").resolve()
    )
    assert hashlib.sha256(raw).hexdigest() == (
        "fface1496156ed94823f9276f7fa11ec94ee1b243e4937c34cc28431741a674e"
    )
    assert checked["artifact_sha256"] == (
        "12ec9cf48c4612447cb65bf51b53beee74f6e4b3349495e3bd35ea3d29bca442"
    )


def test_packet_rejects_resigned_inventory_or_authority_forgery(
    review_packet: dict,
) -> None:
    contaminated = deepcopy(review_packet)
    contaminated["pre_review_inventory"]["complete_bundles"] = 1
    _resign(contaminated)
    with pytest.raises(packet.SideNeutralReviewPacketError, match="inventory"):
        packet.validate_side_neutral_review_packet(
            contaminated, root=Path(".").resolve()
        )

    forged = deepcopy(review_packet)
    forged["authority"]["betting_authority"] = True
    _resign(forged)
    with pytest.raises(packet.SideNeutralReviewPacketError, match="authority"):
        packet.validate_side_neutral_review_packet(
            forged, root=Path(".").resolve()
        )


def test_packet_write_is_no_clobber(review_packet: dict, tmp_path: Path) -> None:
    output = tmp_path / "packet.json"
    packet.write_no_clobber(output, review_packet)
    with pytest.raises(packet.SideNeutralReviewPacketError, match="overwrite"):
        packet.write_no_clobber(output, review_packet)
