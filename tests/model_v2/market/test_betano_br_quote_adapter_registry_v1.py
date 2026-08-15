from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path

import pytest

from lol_kills.v2.market import betano_br_quote_adapter_candidate_registry_v1 as pin
from lol_kills.v2.market import betano_br_quote_adapter_registry_v1 as registry
from lol_kills.v2.market.betano_br_quote_adapter_v1 import sha256_json


def independent_registry(root: Path) -> dict:
    candidate = pin.validate_registered_betano_quote_adapter_candidate_v1(root=root)
    issued = datetime.fromisoformat(candidate["locked_at_utc"]) + timedelta(minutes=1)
    return registry.build_betano_quote_adapter_registry_v1(
        independent_reviewer_id="external-reviewer-example",
        registry_id="betano-adapter-review-example",
        issued_at=issued.isoformat(),
        root=root,
    )


def test_candidate_code_pin_replays_without_independent_authority(
    historical_capture_root: Path,
) -> None:
    candidate = pin.validate_registered_betano_quote_adapter_candidate_v1(
        root=historical_capture_root
    )
    assert candidate["artifact_sha256"] == pin.REGISTERED_CANDIDATE_ARTIFACT_SHA256
    assert candidate["source_lock"]["raw_sha256"] == pin.REGISTERED_ADAPTER_SOURCE_SHA256
    assert candidate["registration"]["independently_registered"] is False
    assert all(value is False for value in candidate["authority"].values())


def test_independent_registry_contract_grants_adapter_identity_only(
    historical_capture_root: Path,
) -> None:
    value = independent_registry(historical_capture_root)
    checked = registry.validate_betano_quote_adapter_registry_v1(
        value,
        expected_registry_sha256=sha256_json(value),
        root=historical_capture_root,
    )
    assert checked["authority"]["source_adapter_identity_authority"] is True
    assert checked["authority"]["quote_identity_authority"] is False
    assert checked["authority"]["transaction_authority"] is False
    assert checked["authority"]["betting_authority"] is False


def test_registry_requires_out_of_band_digest_and_exact_candidate(
    historical_capture_root: Path,
) -> None:
    value = independent_registry(historical_capture_root)
    with pytest.raises(registry.BetanoQuoteAdapterRegistryError, match="out-of-band"):
        registry.validate_betano_quote_adapter_registry_v1(
            value,
            expected_registry_sha256=None,
            root=historical_capture_root,
        )
    changed = deepcopy(value)
    changed["review"]["future_market_close_rule_reviewed"] = False
    with pytest.raises(registry.BetanoQuoteAdapterRegistryError, match="review"):
        registry.validate_betano_quote_adapter_registry_v1(
            changed,
            expected_registry_sha256=sha256_json(changed),
            root=historical_capture_root,
        )


def test_loader_checks_the_exact_external_digest(
    tmp_path: Path,
    historical_capture_root: Path,
) -> None:
    value = independent_registry(historical_capture_root)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    checked = registry.load_registered_betano_quote_adapter_v1(
        expected_registry_sha256=sha256_json(value),
        root=historical_capture_root,
        registry_path=path,
    )
    assert checked["registry_sha256"] == sha256_json(value)
    with pytest.raises(registry.BetanoQuoteAdapterRegistryError, match="digest"):
        registry.load_registered_betano_quote_adapter_v1(
            expected_registry_sha256="0" * 64,
            root=historical_capture_root,
            registry_path=path,
        )
