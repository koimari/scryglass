from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from lol_kills.v2.market import phase_two_collection_readiness_v1 as readiness


def _install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        readiness,
        "_dependencies",
        lambda _root, _environment: {
            "phase_one_evaluation": {"raw_sha256": "1" * 64},
            "calibration_uncertainty_and_fast_parity": {"raw_sha256": "2" * 64},
            "bookmaker_terms": {"raw_sha256": "3" * 64},
            "source_specific_quote_adapter": {"registry_sha256": "4" * 64},
        },
    )
    monkeypatch.setattr(
        readiness,
        "_source_locks",
        lambda _root: [{"locator": "phase-two.py", "bytes": 1, "raw_sha256": "5" * 64}],
    )


def _receipt(monkeypatch: pytest.MonkeyPatch) -> dict:
    _install(monkeypatch)
    return readiness.build_phase_two_collection_readiness_v1(
        clock=lambda: datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc),
        environment={},
    )


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = readiness._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_readiness_freezes_fast_probability_quote_and_opening_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(monkeypatch)
    checked = readiness.validate_phase_two_collection_readiness_v1(
        receipt, environment={}
    )
    contract = checked["collection_contract"]
    assert contract["exact_2000_draw_slow_fast_parity_required"] is True
    assert contract["percentile_interval_need_not_contain_plugin_point"] is True
    assert contract["legacy_quote_bridge_is_transport_only"] is True
    assert checked["locked_empty_state"]["phase_two_started"] is False
    assert all(value is False for value in checked["authority"].values())
    assert all(value is None for value in checked["decision_outputs"].values())


def test_readiness_rejects_interval_bridge_or_authority_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = deepcopy(_receipt(monkeypatch))
    contract["collection_contract"][
        "percentile_interval_need_not_contain_plugin_point"
    ] = False
    _resign(contract)
    with pytest.raises(
        readiness.PhaseTwoCollectionReadinessError,
        match="contract changed",
    ):
        readiness.validate_phase_two_collection_readiness_v1(
            contract, environment={}
        )

    authority = deepcopy(_receipt(monkeypatch))
    authority["authority"]["betting_authority"] = True
    _resign(authority)
    with pytest.raises(
        readiness.PhaseTwoCollectionReadinessError,
        match="exceeds authority",
    ):
        readiness.validate_phase_two_collection_readiness_v1(
            authority, environment={}
        )
