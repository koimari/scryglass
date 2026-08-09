"""Build terminal Draft Score from a publicly side-bound sealed rating.

This adapter extracts the one rating receipt selected by a validated public
side-binding receipt and passes those exact bytes to the frozen terminal Draft
builder.  It never recomputes ratings and remains ineligible until an
authoritative actual-map-start receipt completes the prospective timing chain.
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

from lol_kills.v2.ratings.player import pre_side_rating_binding_v1 as side_binding

from . import future_prediction_ledger as draft_ledger


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:draft-terminal-side-neutral-prediction:v1"
RESULT_STATE = "SIDE_BOUND_TERMINAL_DRAFT_CAPTURED_AWAITING_ACTUAL_MAP_START"
SOURCE_LOCATOR = "lol_kills/v2/draft/terminal/side_neutral_prediction_v1.py"
PREDICTION_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/draft-terminal-v1/side-neutral-predictions"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_SLUG_RE = re.compile(r"[^a-z0-9._-]+")
OUTCOME_KEYS = side_binding.OUTCOME_KEYS
AUTHORITY_KEYS = (
    "model_validation_authority",
    "player_rating_authority",
    "team_rating_authority",
    "draft_validation_authority",
    "incremental_draft_authority",
    "probability_authority",
    "odds_authority",
    "expected_value_authority",
    "recommendation_authority",
    "betting_authority",
)
CLAIM_CEILING = (
    "Outcome-free side-bound terminal Draft evaluation candidate only. The "
    "ratings bytes were selected from the pre-side envelope without refitting, "
    "but actual-map-start timing and independent ledger admission are still "
    "absent. No rating, Draft, probability, odds, EV, recommendation, or "
    "betting authority is granted."
)


class SideNeutralDraftPredictionError(ValueError):
    """The side-neutral terminal Draft wrapper failed closed."""


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
        raise SideNeutralDraftPredictionError(
            "side-neutral Draft value is not canonical"
        ) from exc


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
        raise SideNeutralDraftPredictionError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SideNeutralDraftPredictionError(f"{field} must be non-empty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    text = _nonempty(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SideNeutralDraftPredictionError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise SideNeutralDraftPredictionError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SideNeutralDraftPredictionError(
            "side-neutral Draft clock must return a timezone-aware datetime"
        )
    return value.astimezone(timezone.utc)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SideNeutralDraftPredictionError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_object(raw: bytes, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SideNeutralDraftPredictionError(
                    f"non-finite JSON number in {field}: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SideNeutralDraftPredictionError(
            f"{field} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise SideNeutralDraftPredictionError(f"{field} must be a JSON object")
    return value


def _assert_no_outcomes(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
            if normalized in OUTCOME_KEYS:
                raise SideNeutralDraftPredictionError(
                    f"event outcome field is forbidden: {path}.{key}"
                )
            _assert_no_outcomes(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_outcomes(item, f"{path}[{index}]")


def _source_record(root: Path) -> dict[str, Any]:
    path = root / SOURCE_LOCATOR
    if not path.is_file() or path.is_symlink():
        raise SideNeutralDraftPredictionError("adapter implementation is unavailable")
    return {
        "locator": SOURCE_LOCATOR,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
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
        raise SideNeutralDraftPredictionError(f"{field} embedding changed")
    try:
        raw = base64.b64decode(
            _nonempty(value.get("raw_base64"), f"{field}.raw_base64"),
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise SideNeutralDraftPredictionError(f"{field} base64 is invalid") from exc
    if _sha256_bytes(raw) != _sha(value.get("raw_sha256"), f"{field}.raw_sha256"):
        raise SideNeutralDraftPredictionError(f"{field} raw hash changed")
    parsed = _strict_object(raw, field)
    if parsed != value.get("value"):
        raise SideNeutralDraftPredictionError(f"{field} value changed")
    return raw, parsed


def _selected_rating_raw(binding: Mapping[str, Any]) -> bytes:
    scenario = binding["selection"]["scenario"]
    rating = binding["pre_side_envelope"]["value"]["side_conditionals"][scenario][
        "rating_receipt"
    ]
    try:
        raw = base64.b64decode(rating["raw_base64"], validate=True)
    except (KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise SideNeutralDraftPredictionError(
            "selected rating receipt bytes are unavailable"
        ) from exc
    if (
        _sha256_bytes(raw) != binding["selection"]["selected_rating_receipt_raw_sha256"]
        or _sha256_bytes(raw) != rating["raw_sha256"]
    ):
        raise SideNeutralDraftPredictionError("selected rating receipt hash changed")
    return raw


def _authority_false() -> dict[str, bool]:
    return {name: False for name in AUTHORITY_KEYS}


def build_side_neutral_draft_prediction(
    *,
    side_binding_raw: bytes,
    draft_metadata_raw: bytes,
    draft_source_payload_raw: bytes,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    captured = _clock_sample(clock)
    binding_object = _strict_object(side_binding_raw, "side binding")
    checked_binding = side_binding.validate_pre_side_rating_binding(
        binding_object, root=root
    )
    if captured <= _timestamp(checked_binding["captured_at_utc"], "side binding"):
        raise SideNeutralDraftPredictionError(
            "terminal Draft must be captured after side binding"
        )
    ratings_raw = _selected_rating_raw(checked_binding)
    child = draft_ledger.build_draft_prediction_receipt(
        ratings_receipt_raw=ratings_raw,
        draft_metadata_raw=draft_metadata_raw,
        draft_source_payload_raw=draft_source_payload_raw,
        root=root,
        clock=lambda: captured,
    )
    event = child["event"]
    selection = checked_binding["selection"]
    for field in (
        "blue_organization_id",
        "blue_organization_name",
        "red_organization_id",
        "red_organization_name",
    ):
        if event.get(field) != selection.get(field):
            raise SideNeutralDraftPredictionError(
                f"terminal Draft and public side binding differ: {field}"
            )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "captured_at_utc": captured.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": captured.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "event": dict(event),
        "side_binding": _embedded(side_binding_raw, checked_binding),
        "terminal_draft_prediction": _embedded(
            (
                json.dumps(child, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode(),
            child,
        ),
        "selected_rating_binding": {
            "scenario": selection["scenario"],
            "rating_receipt_raw_sha256": selection[
                "selected_rating_receipt_raw_sha256"
            ],
            "rating_receipt_artifact_sha256": selection[
                "selected_rating_receipt_artifact_sha256"
            ],
            "rating_recomputed_after_side_observation": False,
        },
        "qualification": {
            "pre_side_envelope_valid": True,
            "public_side_binding_valid": True,
            "terminal_draft_strictly_after_side_binding": True,
            "terminal_draft_prediction_valid": True,
            "selected_rating_bytes_bound_exactly": True,
            "rating_recomputed_after_side_observation": False,
            "actual_map_start_present": False,
            "terminal_draft_before_actual_map_start_verified": False,
            "eligible_evaluation_map": False,
            "event_outcome_present": False,
            "event_outcome_accessed": False,
        },
        "implementation": _source_record(root),
        "authority": _authority_false(),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_side_neutral_draft_prediction(payload, root=root)


def validate_side_neutral_draft_prediction(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SideNeutralDraftPredictionError("side-neutral Draft must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "side_neutral_draft")
    expected = {
        "schema_version",
        "result_state",
        "captured_at_utc",
        "clock_attestation",
        "event",
        "side_binding",
        "terminal_draft_prediction",
        "selected_rating_binding",
        "qualification",
        "implementation",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }
    if set(value) != expected:
        raise SideNeutralDraftPredictionError("side-neutral Draft structure changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get(
        "result_state"
    ) != RESULT_STATE:
        raise SideNeutralDraftPredictionError("side-neutral Draft identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise SideNeutralDraftPredictionError("side-neutral Draft hash changed")
    captured = _timestamp(value.get("captured_at_utc"), "captured_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": captured.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise SideNeutralDraftPredictionError("side-neutral Draft clock changed")
    _binding_raw, binding_object = _decode_embedded(
        value.get("side_binding"), "side_binding"
    )
    checked_binding = side_binding.validate_pre_side_rating_binding(
        binding_object, root=root
    )
    child_raw, child_object = _decode_embedded(
        value.get("terminal_draft_prediction"), "terminal_draft_prediction"
    )
    child = draft_ledger.validate_draft_prediction_receipt(child_object, root=root)
    if captured != _timestamp(child["captured_at_utc"], "child capture") or captured <= _timestamp(
        checked_binding["captured_at_utc"], "side binding"
    ):
        raise SideNeutralDraftPredictionError("side-neutral Draft timing changed")
    if value.get("event") != child["event"]:
        raise SideNeutralDraftPredictionError("side-neutral Draft event changed")
    selection = checked_binding["selection"]
    for field in (
        "blue_organization_id",
        "blue_organization_name",
        "red_organization_id",
        "red_organization_name",
    ):
        if child["event"].get(field) != selection.get(field):
            raise SideNeutralDraftPredictionError(
                f"terminal Draft side binding changed: {field}"
            )
    ratings_raw = _selected_rating_raw(checked_binding)
    child_ratings = child["input_receipts"]["ratings_prediction"]
    if (
        child_ratings["raw_sha256"] != _sha256_bytes(ratings_raw)
        or child_ratings["raw_base64"]
        != base64.b64encode(ratings_raw).decode("ascii")
    ):
        raise SideNeutralDraftPredictionError(
            "terminal Draft did not use selected rating bytes"
        )
    expected_binding = {
        "scenario": selection["scenario"],
        "rating_receipt_raw_sha256": selection[
            "selected_rating_receipt_raw_sha256"
        ],
        "rating_receipt_artifact_sha256": selection[
            "selected_rating_receipt_artifact_sha256"
        ],
        "rating_recomputed_after_side_observation": False,
    }
    if value.get("selected_rating_binding") != expected_binding:
        raise SideNeutralDraftPredictionError("selected rating binding changed")
    expected_qualification = {
        "pre_side_envelope_valid": True,
        "public_side_binding_valid": True,
        "terminal_draft_strictly_after_side_binding": True,
        "terminal_draft_prediction_valid": True,
        "selected_rating_bytes_bound_exactly": True,
        "rating_recomputed_after_side_observation": False,
        "actual_map_start_present": False,
        "terminal_draft_before_actual_map_start_verified": False,
        "eligible_evaluation_map": False,
        "event_outcome_present": False,
        "event_outcome_accessed": False,
    }
    if value.get("qualification") != expected_qualification:
        raise SideNeutralDraftPredictionError("side-neutral Draft qualification changed")
    if value.get("implementation") != _source_record(root):
        raise SideNeutralDraftPredictionError("adapter implementation changed")
    if value.get("authority") != _authority_false() or value.get(
        "claim_ceiling"
    ) != CLAIM_CEILING:
        raise SideNeutralDraftPredictionError("side-neutral Draft authority changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise SideNeutralDraftPredictionError(f"refusing to overwrite Draft: {path}")
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
            raise SideNeutralDraftPredictionError(
                f"refusing to overwrite Draft: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return _sha256_bytes(raw)


def _slug(value: str) -> str:
    result = SAFE_SLUG_RE.sub("-", value.casefold()).strip("-.")
    if not result:
        raise SideNeutralDraftPredictionError("event id cannot form a safe locator")
    return result[:160]


def prediction_locator(payload: Mapping[str, Any]) -> str:
    event = payload.get("event") or {}
    event_id = _nonempty(event.get("event_id"), "event.event_id")
    game_number = event.get("game_number")
    binding_value = (payload.get("side_binding") or {}).get("value") or {}
    envelope_value = (binding_value.get("pre_side_envelope") or {}).get("value") or {}
    start = _timestamp(
        (envelope_value.get("event") or {}).get("scheduled_series_start_utc"),
        "scheduled_series_start_utc",
    )
    return (
        PREDICTION_PREFIX
        / start.date().isoformat()
        / f"{_slug(event_id)}-g{game_number}.json"
    ).as_posix()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--side-binding", type=Path, required=True)
    parser.add_argument("--draft-metadata", type=Path, required=True)
    parser.add_argument("--draft-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        payload = build_side_neutral_draft_prediction(
            side_binding_raw=args.side_binding.read_bytes(),
            draft_metadata_raw=args.draft_metadata.read_bytes(),
            draft_source_payload_raw=args.draft_source.read_bytes(),
            root=root,
        )
        expected = root / prediction_locator(payload)
        if args.out.resolve(strict=False) != expected.resolve(strict=False):
            raise SideNeutralDraftPredictionError(
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
    "PREDICTION_PREFIX",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "SideNeutralDraftPredictionError",
    "build_side_neutral_draft_prediction",
    "prediction_locator",
    "validate_side_neutral_draft_prediction",
    "write_no_clobber",
]
