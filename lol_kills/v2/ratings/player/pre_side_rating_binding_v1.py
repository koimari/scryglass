"""Bind public map sides to one already-sealed rating conditional.

The builder is deliberately a selector, not a forecaster.  It validates an
outcome-free pre-side envelope, reads blue and red organization names from
captured public JSON bytes using explicit JSON Pointers, and selects exactly
one embedded rating receipt.  It cannot refit the model, edit a lineup, admit
an evaluation map, or grant rating, probability, odds, EV, recommendation, or
betting authority.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

from . import pre_side_rating_envelope_v1 as envelope


ROOT = Path(__file__).resolve().parents[4]
INPUT_SCHEMA_VERSION = "scryglass:pre-side-rating-binding-input:v1"
BINDING_SCHEMA_VERSION = "scryglass:pre-side-rating-side-binding:v1"
RESULT_STATE = "SIDE_SELECTED_FROM_PRESEALED_RATING_AWAITING_DRAFT_AND_MAP_START"
SOURCE_LOCATOR = "lol_kills/v2/ratings/player/pre_side_rating_binding_v1.py"
BINDING_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/multileague-v3/pre-side-rating-bindings"
)
MAX_SOURCE_PAYLOAD_BYTES = 5_000_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_SLUG_RE = re.compile(r"[^a-z0-9._-]+")
OUTCOME_KEYS = envelope.OUTCOME_KEYS
AUTHORITY_KEYS = envelope.AUTHORITY_KEYS
CLAIM_CEILING = (
    "This receipt only proves that captured outcome-free public JSON selected "
    "one pre-sealed side conditional without refitting. It is not an eligible "
    "evaluation map until terminal Draft and authoritative actual-map-start "
    "receipts complete the timing chain. It grants no rating, probability, "
    "odds, EV, recommendation, or betting authority."
)


class PreSideRatingBindingError(ValueError):
    """A side-binding candidate failed closed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PreSideRatingBindingError("side-binding value is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PreSideRatingBindingError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreSideRatingBindingError(f"{field} must be a non-empty string")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    text = _nonempty(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PreSideRatingBindingError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PreSideRatingBindingError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PreSideRatingBindingError(
            "side-binding clock must return a timezone-aware datetime"
        )
    return value.astimezone(timezone.utc)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PreSideRatingBindingError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_json(raw: bytes, field: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PreSideRatingBindingError(
                    f"non-finite JSON number in {field}: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreSideRatingBindingError(f"{field} is not strict UTF-8 JSON") from exc


def _strict_object(raw: bytes, field: str) -> dict[str, Any]:
    value = _strict_json(raw, field)
    if not isinstance(value, dict):
        raise PreSideRatingBindingError(f"{field} must be a JSON object")
    return value


def _assert_no_outcomes(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
            if normalized in OUTCOME_KEYS:
                raise PreSideRatingBindingError(
                    f"event outcome field is forbidden: {path}.{key}"
                )
            _assert_no_outcomes(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_outcomes(item, f"{path}[{index}]")


def _source_record(root: Path) -> dict[str, Any]:
    path = root / SOURCE_LOCATOR
    if not path.is_file() or path.is_symlink():
        raise PreSideRatingBindingError("side-binding implementation is unavailable")
    return {
        "locator": SOURCE_LOCATOR,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _authority_false() -> dict[str, bool]:
    return {name: False for name in AUTHORITY_KEYS}


def _decode_pointer_token(token: str) -> str:
    index = 0
    result = ""
    while index < len(token):
        if token[index] != "~":
            result += token[index]
            index += 1
            continue
        if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
            raise PreSideRatingBindingError("JSON Pointer escape is invalid")
        result += "~" if token[index + 1] == "0" else "/"
        index += 2
    return result


def _json_pointer(document: Any, pointer: Any, field: str) -> Any:
    text = _nonempty(pointer, field)
    if not text.startswith("/"):
        raise PreSideRatingBindingError(f"{field} must be a non-root JSON Pointer")
    value = document
    for raw_token in text[1:].split("/"):
        token = _decode_pointer_token(raw_token)
        if isinstance(value, Mapping):
            if token not in value:
                raise PreSideRatingBindingError(f"{field} does not resolve")
            value = value[token]
        elif isinstance(value, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise PreSideRatingBindingError(f"{field} array index is invalid")
            index = int(token)
            if index >= len(value):
                raise PreSideRatingBindingError(f"{field} does not resolve")
            value = value[index]
        else:
            raise PreSideRatingBindingError(f"{field} traverses a scalar")
    return value


def _validate_input(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "event",
        "source",
        "extraction",
    }:
        raise PreSideRatingBindingError("side-binding input structure changed")
    _assert_no_outcomes(value, "side_binding_input")
    if value.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise PreSideRatingBindingError("side-binding input schema changed")
    event = value.get("event")
    if not isinstance(event, Mapping) or set(event) != {
        "event_id",
        "series_id",
        "game_number",
    }:
        raise PreSideRatingBindingError("side-binding event structure changed")
    for field in ("event_id", "series_id"):
        _nonempty(event.get(field), f"event.{field}")
    game_number = event.get("game_number")
    if isinstance(game_number, bool) or not isinstance(game_number, int) or game_number < 1:
        raise PreSideRatingBindingError("event.game_number must be positive")
    source = value.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "source",
        "source_url",
        "source_record_id",
        "source_updated_at_utc",
        "available_at_utc",
        "rights_status",
    }:
        raise PreSideRatingBindingError("side-binding source structure changed")
    for field in ("source", "source_url", "source_record_id"):
        _nonempty(source.get(field), f"source.{field}")
    parsed_url = urlparse(str(source["source_url"]))
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise PreSideRatingBindingError("source.source_url must be absolute HTTPS")
    _timestamp(source.get("source_updated_at_utc"), "source.source_updated_at_utc")
    _timestamp(source.get("available_at_utc"), "source.available_at_utc")
    if source.get("rights_status") != "reviewed":
        raise PreSideRatingBindingError("side-binding source rights are not reviewed")
    extraction = value.get("extraction")
    if not isinstance(extraction, Mapping) or set(extraction) != {
        "format",
        "blue_organization_name_json_pointer",
        "red_organization_name_json_pointer",
    }:
        raise PreSideRatingBindingError("side-binding extraction structure changed")
    if extraction.get("format") != "strict_json_pointer_v1":
        raise PreSideRatingBindingError("side-binding extraction format changed")
    blue_pointer = _nonempty(
        extraction.get("blue_organization_name_json_pointer"),
        "extraction.blue_organization_name_json_pointer",
    )
    red_pointer = _nonempty(
        extraction.get("red_organization_name_json_pointer"),
        "extraction.red_organization_name_json_pointer",
    )
    if blue_pointer == red_pointer:
        raise PreSideRatingBindingError("blue and red JSON Pointers must differ")
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "event": dict(event),
        "source": dict(source),
        "extraction": dict(extraction),
    }


def _embedded(raw: bytes, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_sha256": _sha256_bytes(raw),
        "raw_base64": base64.b64encode(raw).decode("ascii"),
        "value": dict(value),
    }


def _decode_embedded(value: Any, field: str) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "raw_sha256",
        "raw_base64",
        "value",
    }:
        raise PreSideRatingBindingError(f"{field} embedding changed")
    try:
        raw = base64.b64decode(_nonempty(value.get("raw_base64"), field), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise PreSideRatingBindingError(f"{field} base64 is invalid") from exc
    if _sha256_bytes(raw) != _sha(value.get("raw_sha256"), f"{field}.raw_sha256"):
        raise PreSideRatingBindingError(f"{field} raw hash changed")
    parsed = _strict_object(raw, field)
    if parsed != value.get("value"):
        raise PreSideRatingBindingError(f"{field} value changed")
    return raw, parsed


def _selection(checked_envelope: Mapping[str, Any], blue_name: str, red_name: str) -> dict[str, Any]:
    teams = checked_envelope["source_order_teams"]
    if blue_name == teams[0]["organization_name"] and red_name == teams[1]["organization_name"]:
        scenario, blue_index, red_index = "team1_blue", 0, 1
    elif blue_name == teams[1]["organization_name"] and red_name == teams[0]["organization_name"]:
        scenario, blue_index, red_index = "team2_blue", 1, 0
    else:
        raise PreSideRatingBindingError(
            "public side names do not exactly match the pre-side organizations"
        )
    child = checked_envelope["side_conditionals"][scenario]
    rating = child["rating_receipt"]
    return {
        "scenario": scenario,
        "blue_slot": teams[blue_index]["slot"],
        "red_slot": teams[red_index]["slot"],
        "blue_organization_id": teams[blue_index]["organization_id"],
        "blue_organization_name": teams[blue_index]["organization_name"],
        "red_organization_id": teams[red_index]["organization_id"],
        "red_organization_name": teams[red_index]["organization_name"],
        "selected_rating_receipt_raw_sha256": rating["raw_sha256"],
        "selected_rating_receipt_artifact_sha256": rating["value"][
            "artifact_sha256"
        ],
        "rating_recomputed_after_side_observation": False,
    }


def build_pre_side_rating_binding(
    *,
    envelope_raw: bytes,
    binding_input_raw: bytes,
    public_side_source_raw: bytes,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    captured = _clock_sample(clock)
    envelope_object = _strict_object(envelope_raw, "pre-side envelope")
    checked_envelope = envelope.validate_pre_side_rating_envelope(
        envelope_object, root=root
    )
    input_object = _strict_object(binding_input_raw, "side-binding input")
    checked_input = _validate_input(input_object)
    if checked_input["event"] != {
        field: checked_envelope["event"][field]
        for field in ("event_id", "series_id", "game_number")
    }:
        raise PreSideRatingBindingError("side binding and pre-side event differ")
    if captured <= _timestamp(checked_envelope["captured_at_utc"], "envelope capture"):
        raise PreSideRatingBindingError("side binding must follow pre-side capture")
    source = checked_input["source"]
    latest_source_time = max(
        _timestamp(source["source_updated_at_utc"], "source_updated_at_utc"),
        _timestamp(source["available_at_utc"], "available_at_utc"),
    )
    if latest_source_time > captured:
        raise PreSideRatingBindingError("side binding predates its public source")
    if not public_side_source_raw or len(public_side_source_raw) > MAX_SOURCE_PAYLOAD_BYTES:
        raise PreSideRatingBindingError("public side source is empty or too large")
    source_document = _strict_json(public_side_source_raw, "public side source")
    _assert_no_outcomes(source_document, "public_side_source")
    extraction = checked_input["extraction"]
    blue_name = _nonempty(
        _json_pointer(
            source_document,
            extraction["blue_organization_name_json_pointer"],
            "blue organization JSON Pointer",
        ),
        "extracted blue organization name",
    )
    red_name = _nonempty(
        _json_pointer(
            source_document,
            extraction["red_organization_name_json_pointer"],
            "red organization JSON Pointer",
        ),
        "extracted red organization name",
    )
    if blue_name == red_name:
        raise PreSideRatingBindingError("public side source repeats one organization")
    selection = _selection(checked_envelope, blue_name, red_name)
    payload: dict[str, Any] = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "captured_at_utc": captured.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": captured.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "event": dict(checked_input["event"]),
        "pre_side_envelope": _embedded(envelope_raw, checked_envelope),
        "binding_input": {
            "raw_sha256": _sha256_bytes(binding_input_raw),
            "value": checked_input,
        },
        "public_side_source": {
            **source,
            "payload_raw_sha256": _sha256_bytes(public_side_source_raw),
            "payload_raw_base64": base64.b64encode(public_side_source_raw).decode(
                "ascii"
            ),
            "extraction": {
                **extraction,
                "extracted_blue_organization_name": blue_name,
                "extracted_red_organization_name": red_name,
            },
        },
        "selection": selection,
        "qualification": {
            "pre_side_envelope_valid": True,
            "binding_strictly_after_pre_side_capture": True,
            "public_source_available_not_after_binding": True,
            "actual_blue_red_identified": True,
            "selected_existing_conditional_without_refit": True,
            "terminal_draft_present": False,
            "actual_map_start_present": False,
            "binding_before_actual_map_start_verified": False,
            "eligible_evaluation_map": False,
            "event_outcome_present": False,
            "event_outcome_accessed": False,
        },
        "implementation": _source_record(root),
        "authority": _authority_false(),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_pre_side_rating_binding(payload, root=root)


def validate_pre_side_rating_binding(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PreSideRatingBindingError("side binding must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "side_binding")
    expected_keys = {
        "schema_version",
        "result_state",
        "captured_at_utc",
        "clock_attestation",
        "event",
        "pre_side_envelope",
        "binding_input",
        "public_side_source",
        "selection",
        "qualification",
        "implementation",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }
    if set(value) != expected_keys:
        raise PreSideRatingBindingError("side-binding structure changed")
    if value.get("schema_version") != BINDING_SCHEMA_VERSION or value.get(
        "result_state"
    ) != RESULT_STATE:
        raise PreSideRatingBindingError("side-binding identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PreSideRatingBindingError("side-binding hash changed")
    captured = _timestamp(value.get("captured_at_utc"), "captured_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": captured.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise PreSideRatingBindingError("side-binding clock changed")
    _envelope_raw, envelope_value = _decode_embedded(
        value.get("pre_side_envelope"), "pre_side_envelope"
    )
    checked_envelope = envelope.validate_pre_side_rating_envelope(
        envelope_value, root=root
    )
    binding_input = value.get("binding_input")
    if not isinstance(binding_input, Mapping) or set(binding_input) != {
        "raw_sha256",
        "value",
    }:
        raise PreSideRatingBindingError("binding input record changed")
    _sha(binding_input.get("raw_sha256"), "binding_input.raw_sha256")
    checked_input = _validate_input(binding_input.get("value"))
    if value.get("event") != checked_input["event"] or checked_input["event"] != {
        field: checked_envelope["event"][field]
        for field in ("event_id", "series_id", "game_number")
    }:
        raise PreSideRatingBindingError("side-binding event changed")
    if captured <= _timestamp(checked_envelope["captured_at_utc"], "envelope capture"):
        raise PreSideRatingBindingError("side-binding timing changed")
    source = value.get("public_side_source")
    expected_source_keys = set(checked_input["source"]) | {
        "payload_raw_sha256",
        "payload_raw_base64",
        "extraction",
    }
    if not isinstance(source, Mapping) or set(source) != expected_source_keys:
        raise PreSideRatingBindingError("public side source record changed")
    if any(source.get(field) != item for field, item in checked_input["source"].items()):
        raise PreSideRatingBindingError("public side source metadata changed")
    try:
        source_raw = base64.b64decode(
            _nonempty(source.get("payload_raw_base64"), "source payload"),
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise PreSideRatingBindingError("public side source base64 is invalid") from exc
    if (
        not source_raw
        or len(source_raw) > MAX_SOURCE_PAYLOAD_BYTES
        or _sha256_bytes(source_raw)
        != _sha(source.get("payload_raw_sha256"), "source payload hash")
    ):
        raise PreSideRatingBindingError("public side source bytes changed")
    source_document = _strict_json(source_raw, "public side source")
    _assert_no_outcomes(source_document, "public_side_source")
    extraction = source.get("extraction")
    expected_extraction_keys = set(checked_input["extraction"]) | {
        "extracted_blue_organization_name",
        "extracted_red_organization_name",
    }
    if not isinstance(extraction, Mapping) or set(extraction) != expected_extraction_keys:
        raise PreSideRatingBindingError("public side extraction changed")
    if any(
        extraction.get(field) != item
        for field, item in checked_input["extraction"].items()
    ):
        raise PreSideRatingBindingError("public side extraction pointers changed")
    blue_name = _nonempty(
        _json_pointer(
            source_document,
            extraction["blue_organization_name_json_pointer"],
            "blue organization JSON Pointer",
        ),
        "extracted blue organization name",
    )
    red_name = _nonempty(
        _json_pointer(
            source_document,
            extraction["red_organization_name_json_pointer"],
            "red organization JSON Pointer",
        ),
        "extracted red organization name",
    )
    if extraction.get("extracted_blue_organization_name") != blue_name or extraction.get(
        "extracted_red_organization_name"
    ) != red_name:
        raise PreSideRatingBindingError("extracted side names changed")
    latest_source_time = max(
        _timestamp(source["source_updated_at_utc"], "source_updated_at_utc"),
        _timestamp(source["available_at_utc"], "available_at_utc"),
    )
    if latest_source_time > captured:
        raise PreSideRatingBindingError("side-binding source timing changed")
    expected_selection = _selection(checked_envelope, blue_name, red_name)
    if value.get("selection") != expected_selection:
        raise PreSideRatingBindingError("selected rating conditional changed")
    expected_qualification = {
        "pre_side_envelope_valid": True,
        "binding_strictly_after_pre_side_capture": True,
        "public_source_available_not_after_binding": True,
        "actual_blue_red_identified": True,
        "selected_existing_conditional_without_refit": True,
        "terminal_draft_present": False,
        "actual_map_start_present": False,
        "binding_before_actual_map_start_verified": False,
        "eligible_evaluation_map": False,
        "event_outcome_present": False,
        "event_outcome_accessed": False,
    }
    if value.get("qualification") != expected_qualification:
        raise PreSideRatingBindingError("side-binding qualification changed")
    if value.get("implementation") != _source_record(root):
        raise PreSideRatingBindingError("side-binding implementation changed")
    if value.get("authority") != _authority_false() or value.get(
        "claim_ceiling"
    ) != CLAIM_CEILING:
        raise PreSideRatingBindingError("side-binding authority boundary changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PreSideRatingBindingError(f"refusing to overwrite side binding: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise PreSideRatingBindingError(
                f"refusing to overwrite side binding: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return _sha256_bytes(raw)


def _slug(value: str) -> str:
    result = SAFE_SLUG_RE.sub("-", value.casefold()).strip("-.")
    if not result:
        raise PreSideRatingBindingError("event id cannot form a safe locator")
    return result[:160]


def binding_locator(payload: Mapping[str, Any]) -> str:
    embedded = payload.get("pre_side_envelope") or {}
    envelope_value = embedded.get("value") or {}
    event = envelope_value.get("event") or {}
    start = _timestamp(
        event.get("scheduled_series_start_utc"), "scheduled_series_start_utc"
    )
    event_id = _nonempty(event.get("event_id"), "event.event_id")
    game_number = event.get("game_number")
    return (
        BINDING_PREFIX
        / start.date().isoformat()
        / f"{_slug(event_id)}-g{game_number}.json"
    ).as_posix()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--envelope", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--public-side-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        payload = build_pre_side_rating_binding(
            envelope_raw=args.envelope.read_bytes(),
            binding_input_raw=args.input.read_bytes(),
            public_side_source_raw=args.public_side_source.read_bytes(),
            root=root,
        )
        expected = root / binding_locator(payload)
        if args.out.resolve(strict=False) != expected.resolve(strict=False):
            raise PreSideRatingBindingError(
                f"output must match bound locator: {expected.relative_to(root)}"
            )
        raw_sha256 = write_no_clobber(args.out, payload)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "result_state": payload["result_state"],
                "selected_scenario": payload["selection"]["scenario"],
                "eligible_evaluation_map": False,
                "betting_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BINDING_PREFIX",
    "BINDING_SCHEMA_VERSION",
    "INPUT_SCHEMA_VERSION",
    "PreSideRatingBindingError",
    "binding_locator",
    "build_pre_side_rating_binding",
    "validate_pre_side_rating_binding",
    "write_no_clobber",
]
