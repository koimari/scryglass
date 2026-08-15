from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import pytest

from lol_kills.v2.data.common import sha256_canonical_object
from lol_kills.v2.market.match_winner_future_protocol_v1 import (
    MatchWinnerFutureProtocolError,
    build_match_winner_future_protocol_v1,
    validate_match_winner_future_protocol_v1,
    write_no_clobber,
)
from lol_kills.v2.market.match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_RAW_SHA256,
    REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
    REGISTERED_SETTLEMENT_CONTRACT_SHA256,
    validate_registered_match_winner_future_protocol_v1,
)


LOCK_TIME = datetime(2026, 8, 2, 1, 35, tzinfo=timezone.utc)


def _build(root: Path) -> dict:
    return build_match_winner_future_protocol_v1(
        root=root,
        clock=lambda: LOCK_TIME,
    )


def _rehash(payload: dict) -> None:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    payload["artifact_sha256"] = sha256_canonical_object(unsigned)


def test_protocol_is_empty_two_stage_and_non_authorizing(
    historical_capture_root: Path,
) -> None:
    payload = _build(historical_capture_root)

    assert payload["phase_one"]["status"] == "EMPTY_OUTCOMES_SEALED"
    assert payload["phase_two"]["status"] == "NOT_OPEN_NOT_STARTED"
    assert payload["phase_two"]["cohort_disjoint_from_phase_one"] is True
    assert payload["quote_capture_contract"][
        "generic_receipt_builder_time_counts_as_transport_time"
    ] is False
    assert payload["settlement_contract"][
        "bookmaker_terms_snapshot_status"
    ] == "NOT_YET_CAPTURED_OR_REVIEWED"
    assert all(value is False for value in payload["authority"].values())
    assert all(value is None for value in payload["decision_outputs"].values())
    assert all(value is None for value in payload["registries"].values())


def test_protocol_binds_exact_registered_prerequisites(
    historical_capture_root: Path,
) -> None:
    payload = _build(historical_capture_root)

    assert set(payload["prerequisites"]) == {
        "ratings_future_protocol",
        "ratings_capture_readiness",
        "draft_future_protocol",
        "draft_capture_readiness",
        "grid_terminal_draft_source_readiness",
    }
    for item in payload["prerequisites"].values():
        assert len(item["raw_sha256"]) == 64
        assert len(item["artifact_sha256"]) == 64


def test_rehashed_protocol_cannot_relax_quote_timing(
    historical_capture_root: Path,
) -> None:
    payload = deepcopy(_build(historical_capture_root))
    payload["quote_capture_contract"][
        "prediction_to_quote_response_seconds_maximum"
    ] = 300.0
    payload["quote_capture_contract_sha256"] = sha256_canonical_object(
        payload["quote_capture_contract"]
    )
    _rehash(payload)

    with pytest.raises(MatchWinnerFutureProtocolError, match="quote capture contract"):
        validate_match_winner_future_protocol_v1(
            payload, root=historical_capture_root
        )


def test_rehashed_protocol_cannot_grant_authority(
    historical_capture_root: Path,
) -> None:
    payload = deepcopy(_build(historical_capture_root))
    payload["authority"]["betting_authority"] = True
    _rehash(payload)

    with pytest.raises(MatchWinnerFutureProtocolError, match="granted authority"):
        validate_match_winner_future_protocol_v1(
            payload, root=historical_capture_root
        )


def test_rehashed_protocol_cannot_hide_source_drift(
    historical_capture_root: Path,
) -> None:
    payload = deepcopy(_build(historical_capture_root))
    payload["source_locks"][0]["raw_sha256"] = "0" * 64
    _rehash(payload)

    with pytest.raises(MatchWinnerFutureProtocolError, match="source drifted"):
        validate_match_winner_future_protocol_v1(
            payload, root=historical_capture_root
        )


def test_lock_must_precede_existing_future_boundary(
    historical_capture_root: Path,
) -> None:
    with pytest.raises(MatchWinnerFutureProtocolError, match="precede"):
        build_match_winner_future_protocol_v1(
            root=historical_capture_root,
            clock=lambda: datetime(2026, 8, 3, tzinfo=timezone.utc),
        )


def test_writer_refuses_to_replace_protocol(
    tmp_path: Path,
    historical_capture_root: Path,
) -> None:
    payload = _build(historical_capture_root)
    path = tmp_path / "protocol.json"

    first = write_no_clobber(path, payload)
    assert len(first) == 64
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_no_clobber(path, payload)


def test_registered_protocol_replays_exact_bytes_and_contracts(
    historical_capture_root: Path,
) -> None:
    payload = validate_registered_match_winner_future_protocol_v1(
        root=historical_capture_root
    )

    assert payload["artifact_sha256"] == REGISTERED_PROTOCOL_ARTIFACT_SHA256
    assert payload["quote_capture_contract_sha256"] == (
        REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256
    )
    assert payload["settlement_contract_sha256"] == (
        REGISTERED_SETTLEMENT_CONTRACT_SHA256
    )
    raw = (
        historical_capture_root
        / "data/lol/v2/evaluation/match-winner-market-v1/future-protocol-v1.json"
    ).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REGISTERED_PROTOCOL_RAW_SHA256
