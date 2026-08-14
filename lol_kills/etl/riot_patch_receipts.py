"""Read exact per-game patch receipts from Riot's LoL Esports feed.

Oracle's Elixir can retain an older client token after a public patch release.
The official live-stats feed has the game-level ``gameMetadata.patchVersion``
field. This module keeps that field separate from the public Riot patch label,
binds it to the raw response hash, and fails closed when the field is absent.

The feed response contains match state and outcome data. A private patch receipt
keeps the game identity, source locator, patch identity, exact response bytes,
and transport hash. The bytes stay in the worker's private receipt catalog.
The receipt remains ingestion provenance rather than an outcome or
model-evaluation source.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lol_kills.net import NetworkTargetError, require_https_url
from lol_kills.v2.patch_identity import PatchIdentityError, canonical_patch


SCHEMA_VERSION = "scryglass:riot-esports-patch-receipt:v1"
SOURCE_HOST = "feed.lolesports.com"
SOURCE_ROOT = f"https://{SOURCE_HOST}/livestats/v1/window"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_GAME_ID_LENGTH = 128
_PATCH_VERSION_RE = re.compile(
    r"^v?(?P<major>\d{1,2})\.(?P<minor>\d{1,2})(?:\.\d+){0,4}$"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RiotPatchReceiptError(ValueError):
    """Raised when a Riot patch response cannot be bound safely."""


class _RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        raise RiotPatchReceiptError("unexpected redirect from Riot live stats")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decode_raw_response(value: object) -> bytes:
    """Decode bounded response evidence stored in a private receipt."""

    if not isinstance(value, str) or not value:
        raise RiotPatchReceiptError("Riot raw response evidence is missing")
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise RiotPatchReceiptError("Riot raw response evidence is not valid base64") from exc
    if not raw or len(raw) > MAX_RESPONSE_BYTES:
        raise RiotPatchReceiptError("Riot raw response evidence exceeds the bounded size")
    return raw


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise RiotPatchReceiptError(f"duplicate JSON key in Riot response: {key}")
        output[key] = value
    return output


def _decode_response_payload(raw_response: bytes) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            raw_response.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RiotPatchReceiptError("Riot response is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise RiotPatchReceiptError("Riot response root must be an object")
    return payload


def _validate_game_id(game_id: object) -> str:
    value = str(game_id or "").strip()
    if not value or len(value) > MAX_GAME_ID_LENGTH:
        raise RiotPatchReceiptError("Riot game ID is missing or too long")
    if any(ord(char) < 0x20 or char in "/\\" for char in value):
        raise RiotPatchReceiptError("Riot game ID contains an unsafe character")
    return value


def parse_client_patch(value: object) -> str:
    """Return ``16.x`` from a Riot client build such as ``16.15.800.4844``."""

    text = str(value or "").strip()
    match = _PATCH_VERSION_RE.fullmatch(text)
    if match is None:
        raise RiotPatchReceiptError("Riot patchVersion is malformed")
    try:
        return canonical_patch(f"{match.group('major')}.{match.group('minor')}").client_patch
    except PatchIdentityError as exc:
        raise RiotPatchReceiptError("Riot patchVersion is outside the supported namespace") from exc


def _patch_from_payload(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("gameMetadata")
    if not isinstance(metadata, Mapping):
        raise RiotPatchReceiptError("Riot response has no gameMetadata object")
    patch_version = metadata.get("patchVersion")
    if not isinstance(patch_version, str) or not patch_version.strip():
        raise RiotPatchReceiptError("Riot response has no gameMetadata.patchVersion")
    return patch_version.strip()


def _receipt_timestamp(value: object | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )
    text = str(value).strip()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise RiotPatchReceiptError("receipt timestamp is not RFC3339") from exc
    if parsed.tzinfo is None:
        raise RiotPatchReceiptError("receipt timestamp must include a timezone")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def build_patch_receipt(
    game_id: object,
    payload: Mapping[str, Any],
    *,
    source_url: str,
    raw_response_sha256: str,
    raw_response_b64: str | None = None,
    oe_game_id: object | None = None,
    retrieved_at_utc: object | None = None,
) -> dict[str, Any]:
    """Build one hash-bound receipt from a decoded Riot response.

    ``raw_response_sha256`` must be the SHA-256 of the exact response bytes.
    The caller must not replace it with a hash of a re-serialized JSON object.
    """

    if not isinstance(payload, Mapping):
        raise RiotPatchReceiptError("Riot response root must be an object")
    game = _validate_game_id(game_id)
    try:
        locator = require_https_url(source_url, hosts={SOURCE_HOST})
    except (NetworkTargetError, ValueError) as exc:
        raise RiotPatchReceiptError("Riot source URL is outside the approved host") from exc
    expected_url = f"{SOURCE_ROOT}/{urllib.parse.quote(game, safe='')}"
    if locator != expected_url:
        raise RiotPatchReceiptError("Riot source URL does not match the game ID")
    if not isinstance(raw_response_sha256, str) or not _SHA256_RE.fullmatch(
        raw_response_sha256.casefold()
    ):
        raise RiotPatchReceiptError("Riot response hash must be a SHA-256 digest")
    if raw_response_b64 is None:
        raise RiotPatchReceiptError("Riot raw response evidence is required")
    raw_response = _decode_raw_response(raw_response_b64)
    if _sha256_bytes(raw_response) != raw_response_sha256.casefold():
        raise RiotPatchReceiptError("Riot response evidence hash does not match")
    evidence_payload = _decode_response_payload(raw_response)
    if evidence_payload != payload:
        raise RiotPatchReceiptError("Riot response evidence does not match decoded payload")
    patch_version = _patch_from_payload(payload)
    client = parse_client_patch(patch_version)
    identity = canonical_patch(client)
    unsigned: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "game_id": game,
        "retrieved_at_utc": _receipt_timestamp(retrieved_at_utc),
        "source": {
            "provider": "Riot Games",
            "transport": "official_lolesports_livestats",
            "locator": locator,
        },
        "observed": {
            "patch_version": patch_version,
            "client_patch": identity.client_patch,
            "public_patch": identity.public_patch,
        },
        "raw_response_sha256": raw_response_sha256.casefold(),
        "raw_response_b64": raw_response_b64,
        "outcome_fields_excluded": ["winningTeam", "winner", "game_end"],
        "authority": {
            "exact_game_patch": True,
            "descriptive_source_freshness_evidence": True,
            "model_validation_authority": False,
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
    }
    if oe_game_id is not None:
        unsigned["oe_game_id"] = _validate_game_id(oe_game_id)
    receipt = dict(unsigned)
    receipt["receipt_canonical_sha256"] = _sha256_bytes(_canonical_bytes(unsigned))
    return receipt


def receipt_from_response_bytes(
    game_id: object,
    raw_response: bytes,
    *,
    source_url: str | None = None,
    oe_game_id: object | None = None,
    retrieved_at_utc: object | None = None,
) -> dict[str, Any]:
    """Decode one bounded response and create its exact patch receipt."""

    if not isinstance(raw_response, bytes):
        raise RiotPatchReceiptError("Riot response must be raw bytes")
    if not raw_response or len(raw_response) > MAX_RESPONSE_BYTES:
        raise RiotPatchReceiptError("Riot response exceeds the bounded receipt size")
    payload = _decode_response_payload(raw_response)
    game = _validate_game_id(game_id)
    locator = source_url or f"{SOURCE_ROOT}/{urllib.parse.quote(game, safe='')}"
    return build_patch_receipt(
        game,
        payload,
        source_url=locator,
        raw_response_sha256=_sha256_bytes(raw_response),
        raw_response_b64=base64.b64encode(raw_response).decode("ascii"),
        oe_game_id=oe_game_id,
        retrieved_at_utc=retrieved_at_utc,
    )


def fetch_game_patch(
    game_id: object,
    *,
    timeout: float = 30.0,
    opener: Any | None = None,
    retrieved_at_utc: object | None = None,
) -> dict[str, Any]:
    """Fetch one official game response and return a hash-bound patch receipt."""

    game = _validate_game_id(game_id)
    if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 60:
        raise RiotPatchReceiptError("Riot request timeout must be between 0 and 60 seconds")
    url = f"{SOURCE_ROOT}/{urllib.parse.quote(game, safe='')}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "scryglass/riot-patch-receipts/1",
        },
    )
    client = opener or urllib.request.build_opener(_RejectRedirect())
    try:
        with client.open(request, timeout=float(timeout)) as response:
            content_length = int(response.headers.get("Content-Length") or 0)
            if content_length > MAX_RESPONSE_BYTES:
                raise RiotPatchReceiptError("Riot response advertises an unsafe size")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(min(64 * 1024, MAX_RESPONSE_BYTES - total + 1))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise RiotPatchReceiptError("Riot response exceeds the bounded size")
    except urllib.error.HTTPError as exc:
        raise RiotPatchReceiptError(f"Riot live stats returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RiotPatchReceiptError("Riot live stats request failed") from exc
    except TimeoutError as exc:
        raise RiotPatchReceiptError("Riot live stats request timed out") from exc
    return receipt_from_response_bytes(
        game,
        b"".join(chunks),
        source_url=url,
        retrieved_at_utc=retrieved_at_utc,
    )


def validate_patch_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a receipt before it is allowed to override an OE row."""

    if not isinstance(receipt, Mapping) or receipt.get("schema") != SCHEMA_VERSION:
        raise RiotPatchReceiptError("unsupported Riot patch receipt schema")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_canonical_sha256", None)
    if not isinstance(claimed, str) or not _SHA256_RE.fullmatch(claimed.casefold()):
        raise RiotPatchReceiptError("Riot receipt canonical hash is missing")
    if _sha256_bytes(_canonical_bytes(unsigned)) != claimed.casefold():
        raise RiotPatchReceiptError("Riot receipt canonical hash mismatch")
    game = _validate_game_id(receipt.get("game_id"))
    source = receipt.get("source")
    observed = receipt.get("observed")
    authority = receipt.get("authority")
    if not isinstance(source, Mapping) or not isinstance(observed, Mapping):
        raise RiotPatchReceiptError("Riot receipt source or observed block is invalid")
    if (
        source.get("provider") != "Riot Games"
        or source.get("transport") != "official_lolesports_livestats"
        or not isinstance(authority, Mapping)
        or authority.get("exact_game_patch") is not True
    ):
        raise RiotPatchReceiptError("Riot receipt authority is not exact-game patch evidence")
    client = parse_client_patch(observed.get("patch_version"))
    identity = canonical_patch(client)
    if (
        observed.get("client_patch") != identity.client_patch
        or observed.get("public_patch") != identity.public_patch
    ):
        raise RiotPatchReceiptError("Riot receipt patch labels do not agree")
    raw_response = _decode_raw_response(receipt.get("raw_response_b64"))
    raw_hash = str(receipt.get("raw_response_sha256") or "").casefold()
    if not _SHA256_RE.fullmatch(raw_hash) or _sha256_bytes(raw_response) != raw_hash:
        raise RiotPatchReceiptError("Riot raw response evidence hash mismatch")
    raw_payload = _decode_response_payload(raw_response)
    if _patch_from_payload(raw_payload) != observed.get("patch_version"):
        raise RiotPatchReceiptError("Riot raw response patch does not match the receipt")
    oe_game_id = receipt.get("oe_game_id")
    if oe_game_id is not None:
        _validate_game_id(oe_game_id)
    built = build_patch_receipt(
        game,
        raw_payload,
        source_url=str(source.get("locator") or ""),
        raw_response_sha256=raw_hash,
        raw_response_b64=str(receipt.get("raw_response_b64") or ""),
        oe_game_id=oe_game_id,
        retrieved_at_utc=receipt.get("retrieved_at_utc"),
    )
    if built["receipt_canonical_sha256"] != claimed.casefold():
        raise RiotPatchReceiptError("Riot receipt fields failed canonical reconstruction")
    return dict(receipt)


def load_patch_receipts(path: Path) -> dict[str, dict[str, Any]]:
    """Load a validated OE game-ID to Riot receipt crosswalk."""

    try:
        raw = path.expanduser().resolve().read_text(encoding="utf-8")
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RiotPatchReceiptError("Riot patch receipt catalog is invalid JSON") from exc
    if isinstance(payload, Mapping):
        if payload.get("schema") not in {None, SCHEMA_VERSION}:
            raise RiotPatchReceiptError("unsupported Riot patch receipt catalog schema")
        entries = payload.get("receipts")
    else:
        entries = payload
    if not isinstance(entries, list):
        raise RiotPatchReceiptError("Riot patch receipt catalog must contain receipts")
    result: dict[str, dict[str, Any]] = {}
    seen_receipts: set[str] = set()
    for entry in entries:
        validated = validate_patch_receipt(entry)
        receipt_digest = str(validated["receipt_canonical_sha256"])
        if receipt_digest in seen_receipts:
            raise RiotPatchReceiptError(
                f"duplicate Riot patch receipt for game {validated['game_id']!r}"
            )
        seen_receipts.add(receipt_digest)
        keys = [str(validated["game_id"])]
        if validated.get("oe_game_id") is not None:
            keys.append(str(validated["oe_game_id"]))
        for game_id in keys:
            previous = result.get(game_id)
            if previous is not None:
                raise RiotPatchReceiptError(f"duplicate Riot patch receipt for game {game_id!r}")
            result[game_id] = validated
    return result


__all__ = [
    "MAX_RESPONSE_BYTES",
    "RiotPatchReceiptError",
    "SCHEMA_VERSION",
    "SOURCE_HOST",
    "SOURCE_ROOT",
    "build_patch_receipt",
    "fetch_game_patch",
    "load_patch_receipts",
    "parse_client_patch",
    "receipt_from_response_bytes",
    "validate_patch_receipt",
]
