"""Bind ratings and terminal-Draft phase-one evidence without opening outcomes.

The three artifacts in this module are deliberately operational rather than
authorizing:

* an immutable event plan is created from an already-valid ratings receipt;
* an event bundle joins that exact ratings receipt to its terminal-Draft and
  actual-map-start receipts; and
* a joint ledger snapshot rebuilds the registered ratings and Draft ledgers
  from one set of event bundles.

Every timestamp is sampled inside its builder.  No artifact can contain an
outcome, approve an opening, or grant rating, probability, odds, EV,
recommendation, or betting authority.
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
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

from lol_kills.v2.draft.terminal import future_prediction_ledger as draft_ledger
from lol_kills.v2.draft.terminal.future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as DRAFT_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as DRAFT_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256 as DRAFT_PROTOCOL_RAW_SHA256,
    validate_registered_future_protocol_v1 as validate_draft_protocol,
)
from lol_kills.v2.ratings.player import (
    multileague_v3_prediction_ledger as ratings_ledger,
)
from lol_kills.v2.ratings.player.multileague_v3_future_protocol import (
    FUTURE_SEALED_START,
)
from lol_kills.v2.ratings.player.multileague_v3_registry_v3 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as RATINGS_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as RATINGS_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256 as RATINGS_PROTOCOL_RAW_SHA256,
    validate_registered_future_protocol_v3 as validate_ratings_protocol,
)

from .match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as MARKET_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as MARKET_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256 as MARKET_PROTOCOL_RAW_SHA256,
    validate_registered_match_winner_future_protocol_v1 as validate_market_protocol,
)


ROOT = Path(__file__).resolve().parents[3]
PLAN_SCHEMA_VERSION = "scryglass:match-winner-phase-one-event-plan:v1"
BUNDLE_SCHEMA_VERSION = "scryglass:match-winner-phase-one-event-bundle:v1"
SNAPSHOT_SCHEMA_VERSION = "scryglass:match-winner-phase-one-joint-ledger:v1"
PLAN_RESULT_STATE = "RATINGS_CAPTURED_AWAITING_TERMINAL_DRAFT"
BUNDLE_RESULT_STATE = "OUTCOME_FREE_PHASE_ONE_EVENT_BUNDLE_COMPLETE"
PLAN_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-one/plans"
)
BUNDLE_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-one/bundles"
)
SNAPSHOT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-one/snapshots"
)
SOURCE_LOCATOR = "lol_kills/v2/market/phase_one_collection_v1.py"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PATCH_RE = re.compile(r"^26\.(?:0[1-9]|1[0-9]|2[0-9])$")
OUTCOME_KEYS = frozenset(
    {
        "actualbluewin",
        "bluekills",
        "bluescore",
        "bluewin",
        "defeat",
        "gameoutcome",
        "gameresult",
        "iswinner",
        "kills",
        "losingteam",
        "lossteam",
        "mapscores",
        "outcome",
        "outcomes",
        "redkills",
        "redscore",
        "redwin",
        "result",
        "results",
        "score",
        "team1score",
        "team2score",
        "totalkills",
        "victory",
        "winner",
        "winnerteamid",
        "winningteam",
        "winteam",
        "won",
    }
)
AUTHORITY_KEYS = (
    "ratings_validation_authority",
    "player_rating_authority",
    "team_rating_authority",
    "draft_validation_authority",
    "incremental_draft_authority",
    "outcome_opening_authority",
    "calibration_authority",
    "probability_authority",
    "odds_authority",
    "expected_value_authority",
    "recommendation_authority",
    "betting_authority",
)
PLAN_CLAIM_CEILING = (
    "Outcome-free phase-one collection plan only; no model, opening, rating, "
    "probability, odds, recommendation, or betting authority."
)
BUNDLE_CLAIM_CEILING = (
    "Outcome-free phase-one receipt bundle eligible only for ledger-candidate "
    "construction; no independent validation or betting authority."
)
SNAPSHOT_CLAIM_CEILING = (
    "Outcome-free joint ratings and Draft ledger candidate only; metadata support "
    "does not authorize outcome opening, probability, odds, EV, or betting."
)


class PhaseOneCollectionError(ValueError):
    """A phase-one collection artifact failed closed."""


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
        raise PhaseOneCollectionError("phase-one value is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhaseOneCollectionError(f"{field} must be a non-empty string")
    return value.strip()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PhaseOneCollectionError(f"{field} must be a lowercase SHA-256")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    text = _nonempty(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseOneCollectionError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PhaseOneCollectionError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime], field: str) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise PhaseOneCollectionError(
            f"{field} clock must return a timezone-aware datetime"
        )
    return observed.astimezone(timezone.utc)


def _strict_object(raw: bytes, field: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise PhaseOneCollectionError(
                    f"{field} contains duplicate key {key!r}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except PhaseOneCollectionError:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise PhaseOneCollectionError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PhaseOneCollectionError(f"{field} must be a JSON object")
    return value


def _assert_no_outcomes(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
            if normalized in OUTCOME_KEYS:
                raise PhaseOneCollectionError(
                    f"event outcome field is forbidden: {path}.{key}"
                )
            _assert_no_outcomes(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_outcomes(item, f"{path}[{index}]")


def _locator(value: Any, prefix: PurePosixPath, field: str) -> str:
    path = PurePosixPath(_nonempty(value, field))
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or tuple(path.parts[: len(prefix.parts)]) != prefix.parts
        or path.suffix != ".json"
    ):
        raise PhaseOneCollectionError(f"{field} is outside its artifact root")
    return path.as_posix()


def _safe_repo_file(
    root: Path, locator: str, prefix: PurePosixPath, field: str
) -> Path:
    relative = PurePosixPath(_locator(locator, prefix, field))
    root_real = root.resolve(strict=True)
    current = root_real
    for part in relative.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise PhaseOneCollectionError(f"{field} is missing") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise PhaseOneCollectionError(f"{field} symlink is rejected")
    metadata = os.lstat(current)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise PhaseOneCollectionError(
            f"{field} must be an unaliased regular file"
        )
    try:
        current.resolve(strict=True).relative_to(root_real)
    except ValueError as exc:
        raise PhaseOneCollectionError(f"{field} escaped the repository") from exc
    return current


def _relative_receipt_tail(locator: str) -> PurePosixPath:
    path = PurePosixPath(
        _locator(locator, ratings_ledger.RECEIPT_PREFIX, "ratings_locator")
    )
    tail = PurePosixPath(*path.parts[len(ratings_ledger.RECEIPT_PREFIX.parts) :])
    if not tail.parts:
        raise PhaseOneCollectionError("ratings locator has no receipt filename")
    return tail


def _source_record(root: Path) -> dict[str, Any]:
    path = root / SOURCE_LOCATOR
    if not path.is_file():
        raise PhaseOneCollectionError("phase-one implementation is missing")
    return {
        "locator": SOURCE_LOCATOR,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _validate_source_record(value: Any, root: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "locator",
        "bytes",
        "raw_sha256",
    }:
        raise PhaseOneCollectionError("phase-one implementation binding changed")
    expected = _source_record(root)
    if dict(value) != expected:
        raise PhaseOneCollectionError("phase-one implementation source drifted")
    return expected


def _protocol_bindings(root: Path) -> dict[str, Any]:
    ratings = validate_ratings_protocol(root=root)
    draft = validate_draft_protocol(root=root)
    market = validate_market_protocol(root=root)
    phase_one = market.get("phase_one") or {}
    if not (
        phase_one.get("same_event_predictions_must_bind_exact_rating_and_draft_receipts")
        is True
        and phase_one.get("ratings_protocol_and_capture_must_pass_their_registered_rules")
        is True
        and phase_one.get("draft_protocol_and_capture_must_pass_their_registered_rules")
        is True
        and phase_one.get("status") == "EMPTY_OUTCOMES_SEALED"
    ):
        raise PhaseOneCollectionError("market phase-one contract is not sealed")
    return {
        "ratings": {
            "locator": RATINGS_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": RATINGS_PROTOCOL_RAW_SHA256,
            "artifact_sha256": RATINGS_PROTOCOL_ARTIFACT_SHA256,
        },
        "terminal_draft": {
            "locator": DRAFT_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": DRAFT_PROTOCOL_RAW_SHA256,
            "artifact_sha256": DRAFT_PROTOCOL_ARTIFACT_SHA256,
        },
        "match_winner_market": {
            "locator": MARKET_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": MARKET_PROTOCOL_RAW_SHA256,
            "artifact_sha256": MARKET_PROTOCOL_ARTIFACT_SHA256,
        },
    }


def _authority_false() -> dict[str, bool]:
    return {name: False for name in AUTHORITY_KEYS}


def _clock_attestation(observed: datetime, noun: str) -> dict[str, Any]:
    return {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": observed.isoformat(),
        "user_supplied_timestamp_allowed": False,
        f"{noun}_time_not_after_builder_observation": True,
    }


def _rating_event(value: Mapping[str, Any]) -> dict[str, Any]:
    event = value.get("event") or {}
    fields = (
        "event_id",
        "series_id",
        "game_number",
        "event_start_utc",
        "league",
        "patch",
        "blue_organization_id",
        "blue_organization_name",
        "red_organization_id",
        "red_organization_name",
    )
    if not isinstance(event, Mapping) or any(field not in event for field in fields):
        raise PhaseOneCollectionError("ratings event identity is incomplete")
    result = {field: event[field] for field in fields}
    for field in (
        "event_id",
        "series_id",
        "league",
        "patch",
        "blue_organization_id",
        "blue_organization_name",
        "red_organization_id",
        "red_organization_name",
    ):
        result[field] = _nonempty(result[field], f"event.{field}")
    game_number = result["game_number"]
    if isinstance(game_number, bool) or not isinstance(game_number, int) or game_number < 1:
        raise PhaseOneCollectionError("event.game_number must be a positive integer")
    if not PATCH_RE.fullmatch(result["patch"]):
        raise PhaseOneCollectionError("event.patch is invalid")
    result["event_start_utc"] = _timestamp(
        result["event_start_utc"], "event.event_start_utc"
    ).isoformat()
    return result


def _binding(locator: str, raw: bytes, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "locator": locator,
        "raw_sha256": _sha256_bytes(raw),
        "artifact_sha256": _sha(
            payload.get("artifact_sha256"), f"{locator}.artifact_sha256"
        ),
    }


def build_event_plan(
    *,
    ratings_prediction_locator: str,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Create an immutable phase-one plan from a persisted ratings receipt."""

    planned = _clock_sample(clock, "phase-one plan")
    rating_locator = _locator(
        ratings_prediction_locator,
        ratings_ledger.RECEIPT_PREFIX,
        "ratings_prediction_locator",
    )
    rating_path = _safe_repo_file(
        root,
        rating_locator,
        ratings_ledger.RECEIPT_PREFIX,
        "ratings prediction receipt",
    )
    ratings_raw = rating_path.read_bytes()
    ratings = ratings_ledger.validate_pre_event_prediction_receipt(
        _strict_object(ratings_raw, "ratings prediction receipt"), root=root
    )
    event = _rating_event(ratings)
    ratings_captured = _timestamp(ratings["captured_at_utc"], "ratings.captured_at")
    event_start = _timestamp(event["event_start_utc"], "event.event_start_utc")
    if planned < ratings_captured:
        raise PhaseOneCollectionError("phase-one plan predates its ratings receipt")
    if planned >= event_start:
        raise PhaseOneCollectionError("phase-one plan is not pre-event")
    if event_start.replace(tzinfo=None) < FUTURE_SEALED_START:
        raise PhaseOneCollectionError("phase-one plan predates the future boundary")
    tail = _relative_receipt_tail(rating_locator)
    locators = {
        "plan": (PLAN_PREFIX / tail).as_posix(),
        "ratings_prediction": rating_locator,
        "draft_prediction": (draft_ledger.PREDICTION_PREFIX / tail).as_posix(),
        "map_start": (draft_ledger.MAP_START_PREFIX / tail).as_posix(),
        "event_bundle": (BUNDLE_PREFIX / tail).as_posix(),
    }
    protocols = _protocol_bindings(root)
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "result_state": PLAN_RESULT_STATE,
        "planned_at_utc": planned.isoformat(),
        "clock_attestation": _clock_attestation(planned, "plan"),
        "protocols": protocols,
        "event": event,
        "locators": locators,
        "ratings_prediction": _binding(rating_locator, ratings_raw, ratings),
        "collection_requirements": {
            "ratings_prediction_must_remain_exact": True,
            "terminal_draft_must_bind_exact_ratings_bytes": True,
            "draft_prediction_must_precede_actual_map_start": True,
            "map_start_receipt_must_be_outcome_free": True,
            "joint_ledgers_must_replay_registered_builders": True,
            "outcomes_must_remain_unopened_during_metadata_counting": True,
            "independent_digest_pin_required_before_opening": True,
        },
        "outcomes_present": False,
        "outcomes_accessed": False,
        "independently_pinned": False,
        "opening_authority": False,
        "implementation": _source_record(root),
        "authority": _authority_false(),
        "claim_ceiling": PLAN_CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return _validate_event_plan(payload, root=root, protocols=protocols)


def _validate_event_plan(
    payload: Mapping[str, Any],
    *,
    root: Path,
    protocols: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseOneCollectionError("phase-one event plan must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "event_plan")
    if set(value) != {
        "schema_version",
        "result_state",
        "planned_at_utc",
        "clock_attestation",
        "protocols",
        "event",
        "locators",
        "ratings_prediction",
        "collection_requirements",
        "outcomes_present",
        "outcomes_accessed",
        "independently_pinned",
        "opening_authority",
        "implementation",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise PhaseOneCollectionError("phase-one event plan structure changed")
    if (
        value.get("schema_version") != PLAN_SCHEMA_VERSION
        or value.get("result_state") != PLAN_RESULT_STATE
    ):
        raise PhaseOneCollectionError("phase-one event plan identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseOneCollectionError("phase-one event plan hash changed")
    planned = _timestamp(value.get("planned_at_utc"), "planned_at_utc")
    if value.get("clock_attestation") != _clock_attestation(planned, "plan"):
        raise PhaseOneCollectionError("phase-one event plan clock changed")
    if value.get("protocols") != protocols:
        raise PhaseOneCollectionError("phase-one protocol binding changed")
    event = _rating_event({"event": value.get("event")})
    event_start = _timestamp(event["event_start_utc"], "event.event_start_utc")
    if planned >= event_start or event_start.replace(tzinfo=None) < FUTURE_SEALED_START:
        raise PhaseOneCollectionError("phase-one event plan timing changed")
    locators = value.get("locators")
    if not isinstance(locators, Mapping) or set(locators) != {
        "plan",
        "ratings_prediction",
        "draft_prediction",
        "map_start",
        "event_bundle",
    }:
        raise PhaseOneCollectionError("phase-one plan locator structure changed")
    rating_locator = _locator(
        locators["ratings_prediction"],
        ratings_ledger.RECEIPT_PREFIX,
        "locators.ratings_prediction",
    )
    tail = _relative_receipt_tail(rating_locator)
    expected_locators = {
        "plan": (PLAN_PREFIX / tail).as_posix(),
        "ratings_prediction": rating_locator,
        "draft_prediction": (draft_ledger.PREDICTION_PREFIX / tail).as_posix(),
        "map_start": (draft_ledger.MAP_START_PREFIX / tail).as_posix(),
        "event_bundle": (BUNDLE_PREFIX / tail).as_posix(),
    }
    if dict(locators) != expected_locators:
        raise PhaseOneCollectionError("phase-one plan locators changed")
    rating_path = _safe_repo_file(
        root,
        rating_locator,
        ratings_ledger.RECEIPT_PREFIX,
        "ratings prediction receipt",
    )
    ratings_raw = rating_path.read_bytes()
    ratings = ratings_ledger.validate_pre_event_prediction_receipt(
        _strict_object(ratings_raw, "ratings prediction receipt"), root=root
    )
    if value.get("ratings_prediction") != _binding(
        rating_locator, ratings_raw, ratings
    ):
        raise PhaseOneCollectionError("phase-one ratings receipt binding changed")
    if _rating_event(ratings) != event:
        raise PhaseOneCollectionError("phase-one plan and ratings event differ")
    if planned < _timestamp(ratings["captured_at_utc"], "ratings.captured_at"):
        raise PhaseOneCollectionError("phase-one plan predates its ratings receipt")
    if value.get("collection_requirements") != {
        "ratings_prediction_must_remain_exact": True,
        "terminal_draft_must_bind_exact_ratings_bytes": True,
        "draft_prediction_must_precede_actual_map_start": True,
        "map_start_receipt_must_be_outcome_free": True,
        "joint_ledgers_must_replay_registered_builders": True,
        "outcomes_must_remain_unopened_during_metadata_counting": True,
        "independent_digest_pin_required_before_opening": True,
    }:
        raise PhaseOneCollectionError("phase-one collection requirements changed")
    if any(
        value.get(field) is not False
        for field in (
            "outcomes_present",
            "outcomes_accessed",
            "independently_pinned",
            "opening_authority",
        )
    ):
        raise PhaseOneCollectionError("phase-one event plan exceeds authority")
    authority = value.get("authority") or {}
    if authority != _authority_false():
        raise PhaseOneCollectionError("phase-one event plan exceeds authority")
    _validate_source_record(value.get("implementation"), root)
    if value.get("claim_ceiling") != PLAN_CLAIM_CEILING:
        raise PhaseOneCollectionError("phase-one event plan claim changed")
    return value


def validate_event_plan(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    protocols = _protocol_bindings(root)
    return _validate_event_plan(payload, root=root, protocols=protocols)


def _load_plan(
    locator: str, root: Path, protocols: Mapping[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    path = _safe_repo_file(root, locator, PLAN_PREFIX, "phase-one event plan")
    raw = path.read_bytes()
    return raw, _validate_event_plan(
        _strict_object(raw, "phase-one event plan"),
        root=root,
        protocols=protocols,
    )


def _load_receipts(
    plan: Mapping[str, Any], root: Path
) -> tuple[
    bytes,
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, Any],
]:
    locators = plan["locators"]
    ratings_path = _safe_repo_file(
        root,
        locators["ratings_prediction"],
        ratings_ledger.RECEIPT_PREFIX,
        "ratings prediction receipt",
    )
    draft_path = _safe_repo_file(
        root,
        locators["draft_prediction"],
        draft_ledger.PREDICTION_PREFIX,
        "draft prediction receipt",
    )
    start_path = _safe_repo_file(
        root,
        locators["map_start"],
        draft_ledger.MAP_START_PREFIX,
        "map-start receipt",
    )
    ratings_raw = ratings_path.read_bytes()
    draft_raw = draft_path.read_bytes()
    start_raw = start_path.read_bytes()
    ratings = ratings_ledger.validate_pre_event_prediction_receipt(
        _strict_object(ratings_raw, "ratings prediction receipt"), root=root
    )
    draft = draft_ledger.validate_draft_prediction_receipt(
        _strict_object(draft_raw, "draft prediction receipt"), root=root
    )
    start = draft_ledger.validate_map_start_receipt(
        _strict_object(start_raw, "map-start receipt"), root=root
    )
    return ratings_raw, ratings, draft_raw, draft, start_raw, start


def _verify_event_join(
    *,
    plan: Mapping[str, Any],
    ratings_raw: bytes,
    ratings: Mapping[str, Any],
    draft: Mapping[str, Any],
    start: Mapping[str, Any],
) -> dict[str, Any]:
    expected = plan["event"]
    if _rating_event(ratings) != expected:
        raise PhaseOneCollectionError("plan and ratings event identity differ")
    draft_event = draft.get("event") or {}
    for field in (
        "event_id",
        "series_id",
        "game_number",
        "league",
        "patch",
        "blue_organization_id",
        "blue_organization_name",
        "red_organization_id",
        "red_organization_name",
    ):
        if draft_event.get(field) != expected[field]:
            raise PhaseOneCollectionError(
                f"plan and Draft event identity differ: {field}"
            )
    start_event = start.get("event") or {}
    for field in ("event_id", "series_id", "game_number", "league", "patch"):
        if start_event.get(field) != expected[field]:
            raise PhaseOneCollectionError(
                f"plan and map-start event identity differ: {field}"
            )
    embedded = (draft.get("input_receipts") or {}).get("ratings_prediction") or {}
    try:
        embedded_raw = base64.b64decode(
            _nonempty(embedded.get("raw_base64"), "embedded ratings raw_base64"),
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise PhaseOneCollectionError("embedded ratings bytes are invalid") from exc
    if (
        embedded_raw != ratings_raw
        or embedded.get("raw_sha256") != _sha256_bytes(ratings_raw)
    ):
        raise PhaseOneCollectionError(
            "terminal Draft does not bind the exact ratings receipt bytes"
        )
    ratings_time = _timestamp(ratings["captured_at_utc"], "ratings.captured_at")
    draft_time = _timestamp(draft["captured_at_utc"], "draft.captured_at")
    actual_start = _timestamp(
        start_event.get("actual_map_start_utc"), "map_start.actual_map_start"
    )
    start_capture = _timestamp(start["captured_at_utc"], "map_start.captured_at")
    if not ratings_time < draft_time < actual_start <= start_capture:
        raise PhaseOneCollectionError("phase-one receipt timing is not prospective")
    return {
        "ratings_captured_at_utc": ratings_time.isoformat(),
        "draft_captured_at_utc": draft_time.isoformat(),
        "actual_map_start_utc": actual_start.isoformat(),
        "map_start_captured_at_utc": start_capture.isoformat(),
        "ratings_before_draft": True,
        "draft_before_actual_map_start": True,
        "map_start_captured_at_or_after_actual_start": True,
    }


def build_event_bundle(
    *,
    plan_locator: str,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Join one plan and its three persisted outcome-free receipts."""

    created = _clock_sample(clock, "phase-one event bundle")
    protocols = _protocol_bindings(root)
    plan_locator = _locator(plan_locator, PLAN_PREFIX, "plan_locator")
    plan_raw, plan = _load_plan(plan_locator, root, protocols)
    if plan["locators"]["plan"] != plan_locator:
        raise PhaseOneCollectionError("phase-one plan was loaded from the wrong locator")
    ratings_raw, ratings, draft_raw, draft, start_raw, start = _load_receipts(
        plan, root
    )
    timing = _verify_event_join(
        plan=plan,
        ratings_raw=ratings_raw,
        ratings=ratings,
        draft=draft,
        start=start,
    )
    if created < _timestamp(
        timing["map_start_captured_at_utc"], "map_start.captured_at"
    ):
        raise PhaseOneCollectionError("phase-one event bundle predates an input receipt")
    payload: dict[str, Any] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "result_state": BUNDLE_RESULT_STATE,
        "created_at_utc": created.isoformat(),
        "clock_attestation": _clock_attestation(created, "creation"),
        "protocols": plan["protocols"],
        "event": {
            **plan["event"],
            "actual_map_start_utc": timing["actual_map_start_utc"],
        },
        "bundle_locator": plan["locators"]["event_bundle"],
        "receipt_bindings": {
            "plan": _binding(plan_locator, plan_raw, plan),
            "ratings_prediction": _binding(
                plan["locators"]["ratings_prediction"], ratings_raw, ratings
            ),
            "draft_prediction": _binding(
                plan["locators"]["draft_prediction"], draft_raw, draft
            ),
            "map_start": _binding(
                plan["locators"]["map_start"], start_raw, start
            ),
        },
        "timing": timing,
        "qualification": {
            "exact_event_identity_joined": True,
            "exact_ratings_bytes_embedded_by_terminal_draft": True,
            "actual_map_start_authority_present": True,
            "prediction_strictly_before_actual_map_start": True,
            "eligible_for_outcome_free_joint_ledger_candidate": True,
            "event_outcome_present": False,
            "event_outcome_accessed": False,
            "independently_pinned": False,
            "opening_authority": False,
        },
        "outcomes_present": False,
        "outcomes_accessed": False,
        "independently_pinned": False,
        "opening_authority": False,
        "implementation": _source_record(root),
        "authority": _authority_false(),
        "claim_ceiling": BUNDLE_CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return _validate_event_bundle(payload, root=root, protocols=protocols)


def _validate_event_bundle(
    payload: Mapping[str, Any],
    *,
    root: Path,
    protocols: Mapping[str, Any],
    receipt_context: dict[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseOneCollectionError("phase-one event bundle must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "event_bundle")
    if set(value) != {
        "schema_version",
        "result_state",
        "created_at_utc",
        "clock_attestation",
        "protocols",
        "event",
        "bundle_locator",
        "receipt_bindings",
        "timing",
        "qualification",
        "outcomes_present",
        "outcomes_accessed",
        "independently_pinned",
        "opening_authority",
        "implementation",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise PhaseOneCollectionError("phase-one event bundle structure changed")
    if (
        value.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or value.get("result_state") != BUNDLE_RESULT_STATE
    ):
        raise PhaseOneCollectionError("phase-one event bundle identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseOneCollectionError("phase-one event bundle hash changed")
    created = _timestamp(value.get("created_at_utc"), "created_at_utc")
    if value.get("clock_attestation") != _clock_attestation(created, "creation"):
        raise PhaseOneCollectionError("phase-one event bundle clock changed")
    bundle_locator = _locator(
        value.get("bundle_locator"), BUNDLE_PREFIX, "bundle_locator"
    )
    binding_values = value.get("receipt_bindings")
    if not isinstance(binding_values, Mapping) or set(binding_values) != {
        "plan",
        "ratings_prediction",
        "draft_prediction",
        "map_start",
    }:
        raise PhaseOneCollectionError("phase-one bundle bindings changed")
    plan_binding = binding_values["plan"]
    if not isinstance(plan_binding, Mapping):
        raise PhaseOneCollectionError("phase-one plan binding is malformed")
    plan_locator = _locator(
        plan_binding.get("locator"), PLAN_PREFIX, "receipt_bindings.plan.locator"
    )
    plan_raw, plan = _load_plan(plan_locator, root, protocols)
    if plan["locators"]["event_bundle"] != bundle_locator:
        raise PhaseOneCollectionError("phase-one bundle locator differs from plan")
    ratings_raw, ratings, draft_raw, draft, start_raw, start = _load_receipts(
        plan, root
    )
    expected_bindings = {
        "plan": _binding(plan_locator, plan_raw, plan),
        "ratings_prediction": _binding(
            plan["locators"]["ratings_prediction"], ratings_raw, ratings
        ),
        "draft_prediction": _binding(
            plan["locators"]["draft_prediction"], draft_raw, draft
        ),
        "map_start": _binding(plan["locators"]["map_start"], start_raw, start),
    }
    if dict(binding_values) != expected_bindings:
        raise PhaseOneCollectionError("phase-one receipt bytes or hashes drifted")
    timing = _verify_event_join(
        plan=plan,
        ratings_raw=ratings_raw,
        ratings=ratings,
        draft=draft,
        start=start,
    )
    if value.get("timing") != timing or created < _timestamp(
        timing["map_start_captured_at_utc"], "map_start.captured_at"
    ):
        raise PhaseOneCollectionError("phase-one event bundle timing changed")
    expected_event = {
        **plan["event"],
        "actual_map_start_utc": timing["actual_map_start_utc"],
    }
    if value.get("event") != expected_event:
        raise PhaseOneCollectionError("phase-one event bundle identity changed")
    if value.get("protocols") != plan["protocols"] or value.get(
        "protocols"
    ) != protocols:
        raise PhaseOneCollectionError("phase-one bundle protocol binding changed")
    if value.get("qualification") != {
        "exact_event_identity_joined": True,
        "exact_ratings_bytes_embedded_by_terminal_draft": True,
        "actual_map_start_authority_present": True,
        "prediction_strictly_before_actual_map_start": True,
        "eligible_for_outcome_free_joint_ledger_candidate": True,
        "event_outcome_present": False,
        "event_outcome_accessed": False,
        "independently_pinned": False,
        "opening_authority": False,
    }:
        raise PhaseOneCollectionError("phase-one event bundle qualification changed")
    if any(
        value.get(field) is not False
        for field in (
            "outcomes_present",
            "outcomes_accessed",
            "independently_pinned",
            "opening_authority",
        )
    ) or value.get("authority") != _authority_false():
        raise PhaseOneCollectionError("phase-one event bundle exceeds authority")
    _validate_source_record(value.get("implementation"), root)
    if value.get("claim_ceiling") != BUNDLE_CLAIM_CEILING:
        raise PhaseOneCollectionError("phase-one event bundle claim changed")
    if receipt_context is not None:
        receipt_context.update(
            {
                "plan": plan,
                "ratings": ratings,
                "draft": draft,
                "map_start": start,
            }
        )
    return value


def validate_event_bundle(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    protocols = _protocol_bindings(root)
    return _validate_event_bundle(payload, root=root, protocols=protocols)


def _snapshot_components(
    *,
    bundle_locators: Sequence[str],
    root: Path,
    created: datetime,
    protocols: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not bundle_locators:
        raise PhaseOneCollectionError("joint ledger requires at least one event bundle")
    records: list[dict[str, Any]] = []
    ratings_receipts: list[tuple[str, Mapping[str, Any]]] = []
    draft_receipts: list[
        tuple[str, Mapping[str, Any], str, Mapping[str, Any]]
    ] = []
    identities: set[tuple[str, int]] = set()
    seen_locators: set[str] = set()
    for raw_locator in bundle_locators:
        locator = _locator(raw_locator, BUNDLE_PREFIX, "bundle_locator")
        if locator in seen_locators:
            raise PhaseOneCollectionError("joint ledger repeats an event bundle")
        seen_locators.add(locator)
        path = _safe_repo_file(root, locator, BUNDLE_PREFIX, "phase-one event bundle")
        raw = path.read_bytes()
        receipt_context: dict[str, Mapping[str, Any]] = {}
        bundle = _validate_event_bundle(
            _strict_object(raw, "phase-one event bundle"),
            root=root,
            protocols=protocols,
            receipt_context=receipt_context,
        )
        if bundle["bundle_locator"] != locator:
            raise PhaseOneCollectionError("phase-one bundle loaded from wrong locator")
        event = bundle["event"]
        identity = (event["event_id"], event["game_number"])
        if identity in identities:
            raise PhaseOneCollectionError("joint ledger repeats a map identity")
        identities.add(identity)
        plan = receipt_context["plan"]
        ratings = receipt_context["ratings"]
        draft = receipt_context["draft"]
        start = receipt_context["map_start"]
        ratings_receipts.append(
            (plan["locators"]["ratings_prediction"], ratings)
        )
        draft_receipts.append(
            (
                plan["locators"]["draft_prediction"],
                draft,
                plan["locators"]["map_start"],
                start,
            )
        )
        records.append(
            {
                "event_id": event["event_id"],
                "series_id": event["series_id"],
                "game_number": event["game_number"],
                "league": event["league"],
                "patch": event["patch"],
                "actual_map_start_utc": event["actual_map_start_utc"],
                "bundle_locator": locator,
                "bundle_raw_sha256": _sha256_bytes(raw),
                "bundle_artifact_sha256": bundle["artifact_sha256"],
            }
        )
    records.sort(
        key=lambda item: (
            item["actual_map_start_utc"],
            item["event_id"],
            item["game_number"],
        )
    )
    ratings_candidate = ratings_ledger.build_prediction_ledger_registry(
        receipts=ratings_receipts,
        root=root,
        clock=lambda: created,
    )
    draft_candidate = draft_ledger.build_prediction_ledger(
        receipts=draft_receipts,
        root=root,
        clock=lambda: created,
    )
    return records, ratings_candidate, draft_candidate


def build_joint_ledger_snapshot(
    *,
    bundle_locators: Sequence[str],
    snapshot_locator: str,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Build one atomic candidate containing both registered child ledgers."""

    created = _clock_sample(clock, "phase-one joint ledger")
    protocols = _protocol_bindings(root)
    snapshot_locator = _locator(
        snapshot_locator, SNAPSHOT_PREFIX, "snapshot_locator"
    )
    records, ratings_candidate, draft_candidate = _snapshot_components(
        bundle_locators=bundle_locators,
        root=root,
        created=created,
        protocols=protocols,
    )
    support_met = (
        ratings_candidate["status"] == "SUPPORT_MET_OUTCOMES_UNOPENED"
        and draft_candidate["status"] == "SUPPORT_MET_OUTCOMES_UNOPENED"
    )
    payload: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "status": (
            "PHASE_ONE_METADATA_SUPPORT_MET_OUTCOMES_UNOPENED"
            if support_met
            else "COLLECTING_OUTCOME_FREE_PHASE_ONE_EVIDENCE"
        ),
        "created_at_utc": created.isoformat(),
        "clock_attestation": _clock_attestation(created, "creation"),
        "snapshot_locator": snapshot_locator,
        "protocols": protocols,
        "event_bundles": records,
        "ratings_ledger_candidate": ratings_candidate,
        "draft_ledger_candidate": draft_candidate,
        "support": {
            "event_bundles": len(records),
            "ratings_metadata_support_met": (
                ratings_candidate["status"] == "SUPPORT_MET_OUTCOMES_UNOPENED"
            ),
            "draft_metadata_support_met": (
                draft_candidate["status"] == "SUPPORT_MET_OUTCOMES_UNOPENED"
            ),
            "joint_metadata_support_met": support_met,
            "model_evaluation_passed": False,
        },
        "outcomes_present": False,
        "outcomes_accessed": False,
        "independently_pinned": False,
        "opening_authority": False,
        "implementation": _source_record(root),
        "authority": _authority_false(),
        "claim_ceiling": SNAPSHOT_CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return _validate_joint_ledger_snapshot(
        payload, root=root, protocols=protocols
    )


def _validate_joint_ledger_snapshot(
    payload: Mapping[str, Any],
    *,
    root: Path,
    protocols: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseOneCollectionError("phase-one joint ledger must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "joint_ledger")
    if set(value) != {
        "schema_version",
        "status",
        "created_at_utc",
        "clock_attestation",
        "snapshot_locator",
        "protocols",
        "event_bundles",
        "ratings_ledger_candidate",
        "draft_ledger_candidate",
        "support",
        "outcomes_present",
        "outcomes_accessed",
        "independently_pinned",
        "opening_authority",
        "implementation",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise PhaseOneCollectionError("phase-one joint ledger structure changed")
    if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION or value.get(
        "status"
    ) not in {
        "COLLECTING_OUTCOME_FREE_PHASE_ONE_EVIDENCE",
        "PHASE_ONE_METADATA_SUPPORT_MET_OUTCOMES_UNOPENED",
    }:
        raise PhaseOneCollectionError("phase-one joint ledger identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseOneCollectionError("phase-one joint ledger hash changed")
    created = _timestamp(value.get("created_at_utc"), "created_at_utc")
    if value.get("clock_attestation") != _clock_attestation(created, "creation"):
        raise PhaseOneCollectionError("phase-one joint ledger clock changed")
    _locator(value.get("snapshot_locator"), SNAPSHOT_PREFIX, "snapshot_locator")
    records = value.get("event_bundles")
    if not isinstance(records, list) or not records:
        raise PhaseOneCollectionError("phase-one joint ledger bundles are missing")
    expected_record_keys = {
        "event_id",
        "series_id",
        "game_number",
        "league",
        "patch",
        "actual_map_start_utc",
        "bundle_locator",
        "bundle_raw_sha256",
        "bundle_artifact_sha256",
    }
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected_record_keys:
            raise PhaseOneCollectionError("phase-one bundle record is malformed")
        _sha(record["bundle_raw_sha256"], "bundle_raw_sha256")
        _sha(record["bundle_artifact_sha256"], "bundle_artifact_sha256")
    bundle_locators = [record["bundle_locator"] for record in records]
    rebuilt_records, ratings_candidate, draft_candidate = _snapshot_components(
        bundle_locators=bundle_locators,
        root=root,
        created=created,
        protocols=protocols,
    )
    if records != rebuilt_records:
        raise PhaseOneCollectionError("phase-one joint bundle index changed")
    if value.get("ratings_ledger_candidate") != ratings_candidate:
        raise PhaseOneCollectionError("ratings ledger candidate replay changed")
    if value.get("draft_ledger_candidate") != draft_candidate:
        raise PhaseOneCollectionError("Draft ledger candidate replay changed")
    if value.get("protocols") != protocols:
        raise PhaseOneCollectionError("phase-one joint protocol binding changed")
    support_met = (
        ratings_candidate["status"] == "SUPPORT_MET_OUTCOMES_UNOPENED"
        and draft_candidate["status"] == "SUPPORT_MET_OUTCOMES_UNOPENED"
    )
    expected_support = {
        "event_bundles": len(records),
        "ratings_metadata_support_met": (
            ratings_candidate["status"] == "SUPPORT_MET_OUTCOMES_UNOPENED"
        ),
        "draft_metadata_support_met": (
            draft_candidate["status"] == "SUPPORT_MET_OUTCOMES_UNOPENED"
        ),
        "joint_metadata_support_met": support_met,
        "model_evaluation_passed": False,
    }
    expected_status = (
        "PHASE_ONE_METADATA_SUPPORT_MET_OUTCOMES_UNOPENED"
        if support_met
        else "COLLECTING_OUTCOME_FREE_PHASE_ONE_EVIDENCE"
    )
    if value.get("support") != expected_support or value.get("status") != expected_status:
        raise PhaseOneCollectionError("phase-one joint support status changed")
    if any(
        value.get(field) is not False
        for field in (
            "outcomes_present",
            "outcomes_accessed",
            "independently_pinned",
            "opening_authority",
        )
    ) or value.get("authority") != _authority_false():
        raise PhaseOneCollectionError("phase-one joint ledger exceeds authority")
    _validate_source_record(value.get("implementation"), root)
    if value.get("claim_ceiling") != SNAPSHOT_CLAIM_CEILING:
        raise PhaseOneCollectionError("phase-one joint ledger claim changed")
    return value


def validate_joint_ledger_snapshot(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    protocols = _protocol_bindings(root)
    return _validate_joint_ledger_snapshot(
        payload, root=root, protocols=protocols
    )


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhaseOneCollectionError(f"refusing to overwrite phase-one artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise PhaseOneCollectionError(
                f"refusing to overwrite phase-one artifact: {path}"
            ) from exc
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return _sha256_bytes(raw)


def _expected_output(root: Path, locator: str) -> Path:
    expected = root / PurePosixPath(locator)
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--ratings-receipt", required=True)
    plan_parser.add_argument("--out", type=Path, required=True)
    bundle_parser = subparsers.add_parser("bundle")
    bundle_parser.add_argument("--plan", required=True)
    bundle_parser.add_argument("--out", type=Path, required=True)
    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--bundle-manifest", type=Path, required=True)
    snapshot_parser.add_argument("--snapshot-locator", required=True)
    snapshot_parser.add_argument("--out", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "plan":
            payload = build_event_plan(
                ratings_prediction_locator=args.ratings_receipt,
                root=root,
            )
            expected_locator = payload["locators"]["plan"]
        elif args.command == "bundle":
            payload = build_event_bundle(plan_locator=args.plan, root=root)
            expected_locator = payload["bundle_locator"]
        elif args.command == "snapshot":
            manifest = _strict_object(
                args.bundle_manifest.read_bytes(), "bundle manifest"
            )
            if set(manifest) != {"bundle_locators"} or not isinstance(
                manifest["bundle_locators"], list
            ):
                raise PhaseOneCollectionError("bundle manifest is malformed")
            payload = build_joint_ledger_snapshot(
                bundle_locators=manifest["bundle_locators"],
                snapshot_locator=args.snapshot_locator,
                root=root,
            )
            expected_locator = payload["snapshot_locator"]
        else:
            raw = args.artifact.read_bytes()
            artifact = _strict_object(raw, "phase-one artifact")
            schema = artifact.get("schema_version")
            if schema == PLAN_SCHEMA_VERSION:
                payload = validate_event_plan(artifact, root=root)
                expected_locator = payload["locators"]["plan"]
            elif schema == BUNDLE_SCHEMA_VERSION:
                payload = validate_event_bundle(artifact, root=root)
                expected_locator = payload["bundle_locator"]
            elif schema == SNAPSHOT_SCHEMA_VERSION:
                payload = validate_joint_ledger_snapshot(artifact, root=root)
                expected_locator = payload["snapshot_locator"]
            else:
                raise PhaseOneCollectionError("unknown phase-one artifact schema")
            if args.artifact.resolve(strict=False) != _expected_output(
                root, expected_locator
            ).resolve(strict=False):
                raise PhaseOneCollectionError(
                    "artifact was not loaded from its bound locator"
                )
            print(
                json.dumps(
                    {
                        "schema_version": schema,
                        "artifact_sha256": payload["artifact_sha256"],
                        "valid": True,
                        "authority": False,
                    },
                    sort_keys=True,
                )
            )
            return 0
        expected_output = _expected_output(root, expected_locator)
        if args.out.resolve(strict=False) != expected_output.resolve(strict=False):
            raise PhaseOneCollectionError(
                f"output must match the artifact's bound locator: {expected_locator}"
            )
        raw_sha256 = write_no_clobber(args.out, payload)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "command": args.command,
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "outcomes_accessed": False,
                "opening_authority": False,
                "betting_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_PREFIX",
    "BUNDLE_SCHEMA_VERSION",
    "PLAN_PREFIX",
    "PLAN_SCHEMA_VERSION",
    "PhaseOneCollectionError",
    "SNAPSHOT_PREFIX",
    "SNAPSHOT_SCHEMA_VERSION",
    "build_event_bundle",
    "build_event_plan",
    "build_joint_ledger_snapshot",
    "validate_event_bundle",
    "validate_event_plan",
    "validate_joint_ledger_snapshot",
    "write_no_clobber",
]
