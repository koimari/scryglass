from __future__ import annotations

import hashlib
import io
import json

import pytest

from lol_kills.etl.riot_patch_receipts import (
    RiotPatchReceiptError,
    fetch_game_patch,
    load_patch_receipts,
    parse_client_patch,
    receipt_from_response_bytes,
    validate_patch_receipt,
)


def _payload(patch: str = "16.15.800.4844") -> bytes:
    return json.dumps(
        {
            "gameMetadata": {"patchVersion": patch},
            "gameState": {"winningTeam": 100},
        },
        separators=(",", ":"),
    ).encode("utf-8")


def test_riot_build_version_maps_to_public_patch_without_date_guessing() -> None:
    assert parse_client_patch("16.15.800.4844") == "16.15"
    receipt = receipt_from_response_bytes(
        "115548147900619042",
        _payload(),
        retrieved_at_utc="2026-08-14T12:00:00Z",
    )
    assert receipt["observed"] == {
        "patch_version": "16.15.800.4844",
        "client_patch": "16.15",
        "public_patch": "26.15",
    }
    assert receipt["raw_response_sha256"] == hashlib.sha256(_payload()).hexdigest()
    assert receipt["authority"]["exact_game_patch"] is True
    assert validate_patch_receipt(receipt)["game_id"] == "115548147900619042"


def test_riot_16_16_receipt_maps_to_26_16() -> None:
    receipt = receipt_from_response_bytes(
        "game-16-16",
        _payload("16.16.801.5000"),
        retrieved_at_utc="2026-08-14T12:00:00Z",
    )
    assert receipt["observed"]["public_patch"] == "26.16"
    assert receipt["observed"]["client_patch"] == "16.16"


@pytest.mark.parametrize(
    "payload",
    [
        b'{"gameMetadata":{}}',
        b'{"gameMetadata":{"patchVersion":"16.15rc1"}}',
        b'{"gameMetadata":{"patchVersion":"14.24.1.1"}}',
    ],
)
def test_missing_or_unsupported_patch_fails_closed(payload: bytes) -> None:
    with pytest.raises(RiotPatchReceiptError):
        receipt_from_response_bytes("game-1", payload)


def test_receipt_rejects_wrong_source_and_tampering() -> None:
    payload = _payload()
    with pytest.raises(RiotPatchReceiptError, match="source URL"):
        receipt_from_response_bytes(
            "game-1",
            payload,
            source_url="https://example.com/livestats/v1/window/game-1",
        )
    receipt = receipt_from_response_bytes("game-1", payload)
    receipt["observed"]["public_patch"] = "26.16"
    with pytest.raises(RiotPatchReceiptError, match="canonical hash"):
        validate_patch_receipt(receipt)

    receipt = receipt_from_response_bytes("game-1", payload)
    receipt["authority"]["exact_game_patch"] = False
    unsigned = dict(receipt)
    unsigned.pop("receipt_canonical_sha256")
    receipt["receipt_canonical_sha256"] = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    with pytest.raises(RiotPatchReceiptError, match="authority"):
        validate_patch_receipt(receipt)


def test_receipt_catalog_validates_and_indexes_by_game_id(tmp_path) -> None:
    first = receipt_from_response_bytes("g1", _payload("16.15.800.4844"))
    second = receipt_from_response_bytes("g2", _payload("16.16.801.5000"))
    path = tmp_path / "riot-patch-receipts.json"
    path.write_text(
        json.dumps({"schema": "scryglass:riot-esports-patch-receipt:v1", "receipts": [first, second]}),
        encoding="utf-8",
    )
    catalog = load_patch_receipts(path)
    assert sorted(catalog) == ["g1", "g2"]
    assert catalog["g2"]["observed"]["public_patch"] == "26.16"

    duplicate = path.with_name("duplicate.json")
    duplicate.write_text(
        json.dumps({"receipts": [first, first]}),
        encoding="utf-8",
    )
    with pytest.raises(RiotPatchReceiptError, match="duplicate"):
        load_patch_receipts(duplicate)

class _Response:
    def __init__(self, payload: bytes) -> None:
        self.headers = {"Content-Length": str(len(payload))}
        self._stream = io.BytesIO(payload)

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


class _Opener:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.url = None

    def open(self, request: object, timeout: float) -> _Response:
        self.url = getattr(request, "full_url", None)
        assert timeout == 3.0
        return _Response(self.payload)


def test_fetch_uses_exact_official_feed_url_and_hashes_raw_bytes() -> None:
    opener = _Opener(_payload())
    receipt = fetch_game_patch(
        "game-1",
        opener=opener,
        timeout=3,
        retrieved_at_utc="2026-08-14T12:00:00Z",
    )
    assert opener.url == "https://feed.lolesports.com/livestats/v1/window/game-1"
    assert receipt["observed"]["public_patch"] == "26.15"
    assert receipt["raw_response_sha256"] == hashlib.sha256(_payload()).hexdigest()
