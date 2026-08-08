"""Capture and ledger outcome-free future terminal Draft Score predictions.

Prediction capture occurs after the terminal draft is available.  Because an
authoritative actual map-start timestamp may only arrive later, the prediction
receipt is initially pending.  A separate outcome-free map-start receipt then
proves the system-clocked prediction preceded the map, and only the combined
ledger entry can count toward prospective support.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

from lol_kills.v2.data.common import ROLES, sha256_canonical_object
from lol_kills.v2.ratings.player import (
    multileague_v3_prediction_ledger as ratings_ledger,
)
from lol_kills.v2.ratings.player.multileague_v3_future_protocol import (
    FUTURE_SEALED_START,
)

from .future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_future_protocol_v1,
)
from .model import TerminalModel


PREDICTION_SCHEMA_VERSION = "scryglass:draft-terminal-future-prediction:v1"
MAP_START_SCHEMA_VERSION = "scryglass:draft-terminal-map-start-receipt:v1"
LEDGER_SCHEMA_VERSION = "scryglass:draft-terminal-future-prediction-ledger:v1"
PREDICTION_RESULT_STATE = "OUTCOME_FREE_DRAFT_PREDICTION_PENDING_MAP_START"
MAP_START_RESULT_STATE = "OUTCOME_FREE_ACTUAL_MAP_START_AUTHORITY_CAPTURED"
RATING_MODEL_ID = ratings_ledger.MODEL_IDS[0]
PREDICTION_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/draft-terminal-v1/predictions"
)
MAP_START_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/draft-terminal-v1/map-start"
)
DEFAULT_LEDGER = Path(
    "data/lol/v2/evaluation/draft-terminal-v1/prediction-ledger.json"
)
PATCH_RE = re.compile(r"^26\.(?:0[1-9]|1[0-9]|2[0-9])$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OUTCOME_KEYS = frozenset(
    {
        "actualbluewin",
        "bluewin",
        "bluewins",
        "defeat",
        "gameoutcome",
        "gameresult",
        "iswinner",
        "losingteam",
        "losingteamid",
        "lossteam",
        "outcome",
        "outcomes",
        "redwins",
        "result",
        "results",
        "victory",
        "winner",
        "winningteam",
        "winningteamid",
        "winnerteamid",
        "winteam",
        "won",
    }
)
AUTHORITY_KEYS = (
    "model_validation_authority",
    "incremental_draft_authority",
    "neutral_probability_authority",
    "contextual_probability_authority",
    "odds_authority",
    "expected_value_authority",
    "recommendation_authority",
    "betting_authority",
)
SOURCE_LOCKS = (
    "lol_kills/v2/draft/terminal/future_prediction_ledger.py",
    "lol_kills/v2/draft/terminal/future_protocol_v1.py",
    "lol_kills/v2/draft/terminal/future_protocol_registry_v1.py",
    "lol_kills/v2/draft/terminal/model.py",
    "lol_kills/v2/ratings/player/multileague_v3_prediction_ledger.py",
    REGISTERED_PROTOCOL_LOCATOR.as_posix(),
)
PREDICTION_CLAIM_CEILING = (
    "Outcome-free evaluation prediction pending actual map-start authority. "
    "It grants no probability, odds, expected-value, recommendation, or betting authority."
)
MAP_START_CLAIM_CEILING = (
    "Outcome-free actual-map-start evidence only; no model or betting authority."
)
LEDGER_CLAIM_CEILING = (
    "Outcome-free Draft Score evaluation ledger only; no probability or betting authority."
)


class DraftPredictionLedgerError(ValueError):
    """A future Draft Score receipt or outcome-free ledger failed closed."""


def _canonical_sha256(value: object) -> str:
    return sha256_canonical_object(value)


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        raise DraftPredictionLedgerError(f"bound source is missing: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DraftPredictionLedgerError(f"{field} must be a non-empty string")
    return value.strip()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise DraftPredictionLedgerError(f"{field} must be a lowercase SHA-256")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    text = _nonempty(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DraftPredictionLedgerError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise DraftPredictionLedgerError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime], field: str) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise DraftPredictionLedgerError(
            f"{field} clock must return a timezone-aware datetime"
        )
    return observed.astimezone(timezone.utc)


def _strict_object(raw: bytes, field: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise DraftPredictionLedgerError(
                    f"{field} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except DraftPredictionLedgerError:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise DraftPredictionLedgerError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise DraftPredictionLedgerError(f"{field} must be an object")
    return value


def _assert_no_outcomes(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
            if normalized in OUTCOME_KEYS:
                raise DraftPredictionLedgerError(
                    f"event outcome field is forbidden: {path}.{key}"
                )
            _assert_no_outcomes(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_outcomes(item, f"{path}[{index}]")


def _embedded(raw: bytes, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_sha256": _sha256_bytes(raw),
        "raw_base64": base64.b64encode(raw).decode("ascii"),
        "value": dict(value),
    }


def _decode_embedded(value: Mapping[str, Any], field: str) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "raw_sha256",
        "raw_base64",
        "value",
    }:
        raise DraftPredictionLedgerError(f"{field} embedded bytes are malformed")
    try:
        raw = base64.b64decode(
            _nonempty(value.get("raw_base64"), f"{field}.raw_base64"),
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise DraftPredictionLedgerError(f"{field} base64 is invalid") from exc
    if _sha256_bytes(raw) != _sha(value.get("raw_sha256"), f"{field}.raw_sha256"):
        raise DraftPredictionLedgerError(f"{field} raw hash changed")
    decoded = _strict_object(raw, field)
    if decoded != value.get("value"):
        raise DraftPredictionLedgerError(f"{field} parsed value differs from its bytes")
    return raw, decoded


def _validate_source(
    source: Any,
    *,
    payload_raw: bytes,
    field: str,
) -> dict[str, Any]:
    expected = {
        "source_id",
        "source_url",
        "source_record_id",
        "available_at_utc",
        "rights_status",
        "payload_raw_sha256",
    }
    if not isinstance(source, Mapping) or set(source) != expected:
        raise DraftPredictionLedgerError(f"{field} source structure changed")
    for name in ("source_id", "source_url", "source_record_id"):
        _nonempty(source.get(name), f"{field}.{name}")
    _timestamp(source.get("available_at_utc"), f"{field}.available_at_utc")
    if source.get("rights_status") != "reviewed":
        raise DraftPredictionLedgerError(f"{field} source rights are not reviewed")
    if source.get("payload_raw_sha256") != _sha256_bytes(payload_raw):
        raise DraftPredictionLedgerError(f"{field} source payload hash changed")
    source_payload = _strict_object(payload_raw, f"{field}.source_payload")
    _assert_no_outcomes(source_payload, f"{field}.source_payload")
    return dict(source)


def _validate_side(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(ROLES):
        raise DraftPredictionLedgerError(f"{field} must contain exactly five roles")
    result = {role: _nonempty(value.get(role), f"{field}.{role}") for role in ROLES}
    if len(set(result.values())) != len(ROLES):
        raise DraftPredictionLedgerError(f"{field} contains duplicate champions")
    return result


def _canonicalize_actions(
    actions: Any,
    assignments: Any,
    *,
    side_map: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(actions, list) or not isinstance(assignments, list):
        raise DraftPredictionLedgerError("terminal actions and assignments must be lists")
    canonical_actions: list[dict[str, Any]] = []
    action_by_id: dict[str, dict[str, Any]] = {}
    champions: set[str] = set()
    for expected_slot, action in enumerate(actions, 1):
        if not isinstance(action, Mapping) or set(action) != {
            "slot",
            "action_id",
            "side",
            "kind",
            "champion_id",
            "role_set",
        }:
            raise DraftPredictionLedgerError("terminal action structure changed")
        if action.get("slot") != expected_slot:
            raise DraftPredictionLedgerError("terminal action slots are not contiguous")
        action_id = _nonempty(action.get("action_id"), "action.action_id")
        side = action.get("side")
        kind = action.get("kind")
        champion = _nonempty(action.get("champion_id"), "action.champion_id")
        role_set = action.get("role_set")
        if action_id in action_by_id or champion in champions:
            raise DraftPredictionLedgerError("terminal actions are duplicated")
        if side not in side_map or kind not in {"pick", "ban"}:
            raise DraftPredictionLedgerError("terminal action side or kind is invalid")
        if not isinstance(role_set, list) or len(set(role_set)) != len(role_set):
            raise DraftPredictionLedgerError("terminal action role_set is invalid")
        if kind == "pick":
            if not role_set or not set(role_set).issubset(set(ROLES)):
                raise DraftPredictionLedgerError("terminal pick role_set is invalid")
        elif role_set:
            raise DraftPredictionLedgerError("terminal ban role_set must be empty")
        canonical = {
            "slot": expected_slot,
            "action_id": action_id,
            "canonical_side": side_map[str(side)],
            "kind": kind,
            "champion_id": champion,
            "role_set": list(role_set),
        }
        canonical_actions.append(canonical)
        action_by_id[action_id] = canonical
        champions.add(champion)
    picks = [action for action in canonical_actions if action["kind"] == "pick"]
    if (
        len(picks) != 10
        or sum(action["canonical_side"] == "A" for action in picks) != 5
        or sum(action["canonical_side"] == "B" for action in picks) != 5
    ):
        raise DraftPredictionLedgerError("terminal draft does not have five picks per side")
    if len(assignments) != 10:
        raise DraftPredictionLedgerError("terminal draft must have ten assignments")
    canonical_assignments: list[dict[str, Any]] = []
    seen_actions: set[str] = set()
    seen_champions: set[str] = set()
    roles_by_side: dict[str, set[str]] = {"A": set(), "B": set()}
    for assignment in assignments:
        if not isinstance(assignment, Mapping) or set(assignment) != {
            "action_id",
            "side",
            "champion_id",
            "role",
        }:
            raise DraftPredictionLedgerError("terminal assignment structure changed")
        action_id = _nonempty(assignment.get("action_id"), "assignment.action_id")
        action = action_by_id.get(action_id)
        side = assignment.get("side")
        champion = _nonempty(
            assignment.get("champion_id"), "assignment.champion_id"
        )
        role = assignment.get("role")
        if (
            action is None
            or action["kind"] != "pick"
            or side not in side_map
            or side_map[str(side)] != action["canonical_side"]
            or champion != action["champion_id"]
            or role not in ROLES
            or role not in action["role_set"]
            or action_id in seen_actions
            or champion in seen_champions
            or role in roles_by_side[action["canonical_side"]]
        ):
            raise DraftPredictionLedgerError("terminal assignment is invalid")
        canonical = {
            "action_id": action_id,
            "canonical_side": action["canonical_side"],
            "champion_id": champion,
            "role": role,
        }
        canonical_assignments.append(canonical)
        seen_actions.add(action_id)
        seen_champions.add(champion)
        roles_by_side[action["canonical_side"]].add(str(role))
    if any(roles != set(ROLES) for roles in roles_by_side.values()):
        raise DraftPredictionLedgerError("terminal assignments do not fill every role")
    return canonical_actions, canonical_assignments


def _validate_draft_metadata(
    value: Mapping[str, Any], *, source_payload_raw: bytes
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "event_id",
        "series_id",
        "game_number",
        "league",
        "patch",
        "blue_organization_id",
        "blue_organization_name",
        "red_organization_id",
        "red_organization_name",
        "source",
        "protocol_validation",
        "blue",
        "red",
        "actions",
        "final_assignments",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DraftPredictionLedgerError("draft metadata structure changed")
    _assert_no_outcomes(value, "draft_metadata")
    if value.get("schema_version") != "scryglass:terminal-draft-capture-input:v1":
        raise DraftPredictionLedgerError("draft metadata schema changed")
    for field in (
        "event_id",
        "series_id",
        "league",
        "blue_organization_id",
        "blue_organization_name",
        "red_organization_id",
        "red_organization_name",
    ):
        _nonempty(value.get(field), f"draft_metadata.{field}")
    game_number = value.get("game_number")
    if isinstance(game_number, bool) or not isinstance(game_number, int) or game_number < 1:
        raise DraftPredictionLedgerError("draft metadata game_number is invalid")
    if not PATCH_RE.fullmatch(_nonempty(value.get("patch"), "draft_metadata.patch")):
        raise DraftPredictionLedgerError("draft metadata patch is invalid")
    if value["blue_organization_id"] == value["red_organization_id"]:
        raise DraftPredictionLedgerError("draft metadata teams are identical")
    blue = _validate_side(value.get("blue"), "draft_metadata.blue")
    red = _validate_side(value.get("red"), "draft_metadata.red")
    if len(set((*blue.values(), *red.values()))) != 10:
        raise DraftPredictionLedgerError("terminal draft must contain ten unique champions")
    source = _validate_source(
        value.get("source"), payload_raw=source_payload_raw, field="draft_metadata"
    )
    protocol = value.get("protocol_validation")
    expected_protocol = {
        "protocol_id",
        "validator_id",
        "validator_sha256",
        "validated_at_utc",
        "action_order_verified",
        "pick_ban_counts_verified",
        "blue_red_side_mapping_verified",
    }
    if not isinstance(protocol, Mapping) or set(protocol) != expected_protocol:
        raise DraftPredictionLedgerError("draft protocol validation structure changed")
    for field in ("protocol_id", "validator_id"):
        _nonempty(protocol.get(field), f"protocol_validation.{field}")
    _sha(protocol.get("validator_sha256"), "protocol_validation.validator_sha256")
    _timestamp(protocol.get("validated_at_utc"), "protocol_validation.validated_at_utc")
    if any(
        protocol.get(field) is not True
        for field in (
            "action_order_verified",
            "pick_ban_counts_verified",
            "blue_red_side_mapping_verified",
        )
    ):
        raise DraftPredictionLedgerError("draft protocol validation did not pass")
    canonical_blue_is_a = value["blue_organization_id"] < value["red_organization_id"]
    side_map = {
        "blue": "A" if canonical_blue_is_a else "B",
        "red": "B" if canonical_blue_is_a else "A",
    }
    canonical_actions, canonical_assignments = _canonicalize_actions(
        value.get("actions"), value.get("final_assignments"), side_map=side_map
    )
    return {
        **dict(value),
        "source": source,
        "blue": blue,
        "red": red,
        "canonical_side_map": side_map,
        "canonical_actions": canonical_actions,
        "canonical_assignments": canonical_assignments,
    }


def _pair(first: str, second: str) -> str:
    return "|".join(sorted((first, second)))


def _sigmoid(logit: float) -> float:
    if logit >= 40:
        return 1.0
    if logit <= -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-logit))


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise DraftPredictionLedgerError("rating probability is outside (0,1)")
    return math.log(probability / (1.0 - probability))


def _score_composition(
    metadata: Mapping[str, Any], model: TerminalModel
) -> dict[str, Any]:
    blue = dict(metadata["blue"])
    red = dict(metadata["red"])
    blue_is_a = metadata["canonical_side_map"]["blue"] == "A"
    side_a = blue if blue_is_a else red
    side_b = red if blue_is_a else blue
    raw = 0.0
    ledger: list[dict[str, Any]] = []
    for canonical_side, side, sign in (("A", side_a, 1.0), ("B", side_b, -1.0)):
        for role in ROLES:
            champion = side[role]
            value = sign * model.champion_role_logit.get(f"{role}|{champion}", 0.0)
            raw += value
            ledger.append(
                {
                    "component_type": "champion_role",
                    "canonical_side": canonical_side,
                    "role": role,
                    "champion_id": champion,
                    "signed_logit": value,
                }
            )
        champions = [side[role] for role in ROLES]
        for index, first in enumerate(champions):
            for second in champions[index + 1 :]:
                value = sign * model.ally_synergy_logit.get(_pair(first, second), 0.0)
                raw += value
                if value:
                    ledger.append(
                        {
                            "component_type": "ally_synergy",
                            "canonical_side": canonical_side,
                            "champion_ids": [first, second],
                            "signed_logit": value,
                        }
                    )
    for role in ROLES:
        first = side_a[role]
        second = side_b[role]
        first_key, second_key = sorted((first, second))
        value = model.counter_logit.get(
            f"{role}|{first_key}|{second_key}", 0.0
        )
        if first > second:
            value = -value
        raw += value
        if value:
            ledger.append(
                {
                    "component_type": "counter",
                    "role": role,
                    "champion_ids": [first, second],
                    "signed_logit": value,
                }
            )
    ledger_sum = sum(float(item["signed_logit"]) for item in ledger)
    if not math.isclose(raw, ledger_sum, abs_tol=1e-12):
        raise DraftPredictionLedgerError("draft component ledger does not reconcile")
    scaled_a = model.calibration_slope * raw
    scaled_blue = scaled_a if blue_is_a else -scaled_a
    sparse = [
        {"side": side_name, "role": role, "champion_id": picks[role]}
        for side_name, picks in (("blue", blue), ("red", red))
        for role in ROLES
        if f"{role}|{picks[role]}" not in model.champion_role_logit
    ]
    return {
        "canonical_side_a": "blue" if blue_is_a else "red",
        "canonical_side_b": "red" if blue_is_a else "blue",
        "raw_logit_a": raw,
        "scaled_logit_a": scaled_a,
        "scaled_logit_blue": scaled_blue,
        "equal_strength_index_a": _sigmoid(scaled_a),
        "equal_strength_index_b": 1.0 - _sigmoid(scaled_a),
        "ledger": ledger,
        "ledger_logit_sum": ledger_sum,
        "sparse_or_new_champion_assignments": sparse,
        "sparse_or_new_champion_map": bool(sparse),
        "neutral_output_directly_outcome_calibrated": False,
    }


def _model(root: Path, protocol: Mapping[str, Any]) -> TerminalModel:
    candidate = protocol["locked_candidate"]
    raw = (root / candidate["artifact_locator"]).read_bytes()
    return TerminalModel.from_artifact_bytes(
        raw, expected_artifact_sha256=candidate["artifact_raw_sha256"]
    )


def build_draft_prediction_receipt(
    *,
    ratings_receipt_raw: bytes,
    draft_metadata_raw: bytes,
    draft_source_payload_raw: bytes,
    root: Path = Path("."),
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    captured_at = _clock_sample(clock, "draft prediction")
    ratings_object = _strict_object(ratings_receipt_raw, "ratings receipt")
    ratings = ratings_ledger.validate_pre_event_prediction_receipt(
        ratings_object, root=root
    )
    metadata_object = _strict_object(draft_metadata_raw, "draft metadata")
    metadata = _validate_draft_metadata(
        metadata_object, source_payload_raw=draft_source_payload_raw
    )
    protocol = validate_registered_future_protocol_v1(root=root)
    protocol_locked = _timestamp(protocol["locked_at_utc"], "protocol.locked_at")
    source_available = _timestamp(
        metadata["source"]["available_at_utc"], "draft.source.available_at"
    )
    validated_at = _timestamp(
        metadata["protocol_validation"]["validated_at_utc"],
        "draft.protocol_validation.validated_at",
    )
    ratings_captured = _timestamp(
        ratings["captured_at_utc"], "ratings.captured_at"
    )
    if captured_at <= protocol_locked:
        raise DraftPredictionLedgerError("draft prediction predates its protocol")
    if captured_at <= max(source_available, validated_at, ratings_captured):
        raise DraftPredictionLedgerError(
            "draft prediction is not strictly after required evidence"
        )
    event = ratings["event"]
    bindings = {
        "event_id": "event_id",
        "series_id": "series_id",
        "game_number": "game_number",
        "league": "league",
        "patch": "patch",
        "blue_organization_id": "blue_organization_id",
        "blue_organization_name": "blue_organization_name",
        "red_organization_id": "red_organization_id",
        "red_organization_name": "red_organization_name",
    }
    for metadata_field, rating_field in bindings.items():
        if metadata.get(metadata_field) != event.get(rating_field):
            raise DraftPredictionLedgerError(
                f"draft and ratings identity differ: {metadata_field}"
            )
    event_start = _timestamp(event["event_start_utc"], "ratings.event_start")
    if event_start.replace(tzinfo=None) < FUTURE_SEALED_START:
        raise DraftPredictionLedgerError("draft event predates future boundary")
    if source_available < FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise DraftPredictionLedgerError("draft source predates future boundary")
    model = _model(root, protocol)
    if _timestamp(model.model_as_of, "model.model_as_of") >= captured_at:
        raise DraftPredictionLedgerError("draft model is not frozen before prediction")
    composition = _score_composition(metadata, model)
    rating_prediction = ratings["evaluation_predictions"][RATING_MODEL_ID]
    rating_p_blue = float(rating_prediction["p_blue"])
    rating_logit = _logit(rating_p_blue)
    combined_logit = rating_logit + float(composition["scaled_logit_blue"])
    combined_p_blue = _sigmoid(combined_logit)
    payload: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "result_state": PREDICTION_RESULT_STATE,
        "captured_at_utc": captured_at.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": captured_at.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "capture_time_not_after_builder_observation": True,
        },
        "protocol": {
            "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "future_holdout_start": FUTURE_SEALED_START.replace(
                tzinfo=timezone.utc
            ).isoformat(),
        },
        "event": {
            key: metadata[key]
            for key in (
                "event_id",
                "series_id",
                "game_number",
                "league",
                "patch",
                "blue_organization_id",
                "blue_organization_name",
                "red_organization_id",
                "red_organization_name",
            )
        },
        "input_receipts": {
            "ratings_prediction": _embedded(ratings_receipt_raw, ratings_object),
            "draft_metadata": _embedded(draft_metadata_raw, metadata_object),
            "draft_source_payload": {
                "raw_sha256": _sha256_bytes(draft_source_payload_raw),
                "raw_base64": base64.b64encode(draft_source_payload_raw).decode(
                    "ascii"
                ),
            },
        },
        "model": {
            "model_version": model.model_version,
            "model_as_of": model.model_as_of,
            "artifact_locator": protocol["locked_candidate"]["artifact_locator"],
            "artifact_raw_sha256": model.artifact_sha256,
            "candidate_id": protocol["locked_candidate"]["candidate_id"],
            "variant_id": protocol["locked_candidate"]["variant_id"],
        },
        "draft_index": composition,
        "evaluation_predictions": {
            "ratings_only": {
                "p_blue": rating_p_blue,
                "p_red": 1.0 - rating_p_blue,
                "logit_blue": rating_logit,
            },
            "ratings_plus_draft": {
                "p_blue": combined_p_blue,
                "p_red": 1.0 - combined_p_blue,
                "logit_blue": combined_logit,
            },
        },
        "qualification": {
            "event_on_or_after_future_boundary": True,
            "ratings_prediction_receipt_valid": True,
            "terminal_draft_source_hash_verified": True,
            "terminal_draft_source_rights_reviewed": True,
            "terminal_assignments_complete": True,
            "pick_ban_protocol_validated": True,
            "system_clock_sampled_inside_builder": True,
            "actual_map_start_authority_present": False,
            "prediction_strictly_before_actual_map_start": None,
            "eligible_future_evidence": False,
            "event_outcome_present": False,
            "event_outcome_accessed": False,
            "independently_pinned_ledger_entry": False,
        },
        "source_locks": [_source_record(root, locator) for locator in SOURCE_LOCKS],
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": PREDICTION_CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_draft_prediction_receipt(payload, root=root)


def _decode_source_payload(value: Any, field: str) -> bytes:
    if not isinstance(value, Mapping) or set(value) != {"raw_sha256", "raw_base64"}:
        raise DraftPredictionLedgerError(f"{field} payload structure changed")
    try:
        raw = base64.b64decode(
            _nonempty(value.get("raw_base64"), f"{field}.raw_base64"),
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise DraftPredictionLedgerError(f"{field} payload base64 is invalid") from exc
    if _sha256_bytes(raw) != _sha(value.get("raw_sha256"), f"{field}.raw_sha256"):
        raise DraftPredictionLedgerError(f"{field} payload hash changed")
    return raw


def validate_draft_prediction_receipt(
    payload: Mapping[str, Any], *, root: Path = Path(".")
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DraftPredictionLedgerError("draft prediction receipt must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "draft_prediction")
    if set(value) != {
        "schema_version",
        "result_state",
        "captured_at_utc",
        "clock_attestation",
        "protocol",
        "event",
        "input_receipts",
        "model",
        "draft_index",
        "evaluation_predictions",
        "qualification",
        "source_locks",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise DraftPredictionLedgerError(
            "draft prediction receipt structure changed"
        )
    if (
        value.get("schema_version") != PREDICTION_SCHEMA_VERSION
        or value.get("result_state") != PREDICTION_RESULT_STATE
    ):
        raise DraftPredictionLedgerError("draft prediction receipt identity changed")
    declared = value.get("artifact_sha256")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if declared != _canonical_sha256(unsigned):
        raise DraftPredictionLedgerError("draft prediction canonical hash changed")
    protocol = validate_registered_future_protocol_v1(root=root)
    protocol_record = value.get("protocol") or {}
    if protocol_record != {
        "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
        "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
        "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "future_holdout_start": FUTURE_SEALED_START.replace(
            tzinfo=timezone.utc
        ).isoformat(),
    }:
        raise DraftPredictionLedgerError("draft prediction protocol binding changed")
    captured = _timestamp(value.get("captured_at_utc"), "captured_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": captured.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "capture_time_not_after_builder_observation": True,
    }:
        raise DraftPredictionLedgerError("draft prediction clock attestation changed")
    inputs = value.get("input_receipts") or {}
    if set(inputs) != {
        "ratings_prediction",
        "draft_metadata",
        "draft_source_payload",
    }:
        raise DraftPredictionLedgerError("draft prediction input structure changed")
    ratings_raw, ratings_object = _decode_embedded(
        inputs.get("ratings_prediction") or {}, "ratings prediction"
    )
    ratings = ratings_ledger.validate_pre_event_prediction_receipt(
        ratings_object, root=root
    )
    metadata_raw, metadata_object = _decode_embedded(
        inputs.get("draft_metadata") or {}, "draft metadata"
    )
    source_payload_raw = _decode_source_payload(
        inputs.get("draft_source_payload"), "draft source"
    )
    metadata = _validate_draft_metadata(
        metadata_object, source_payload_raw=source_payload_raw
    )
    if _sha256_bytes(ratings_raw) != inputs["ratings_prediction"]["raw_sha256"]:
        raise DraftPredictionLedgerError("ratings prediction raw binding changed")
    if _sha256_bytes(metadata_raw) != inputs["draft_metadata"]["raw_sha256"]:
        raise DraftPredictionLedgerError("draft metadata raw binding changed")
    event = value.get("event") or {}
    expected_event = {
        key: metadata[key]
        for key in (
            "event_id",
            "series_id",
            "game_number",
            "league",
            "patch",
            "blue_organization_id",
            "blue_organization_name",
            "red_organization_id",
            "red_organization_name",
        )
    }
    if event != expected_event:
        raise DraftPredictionLedgerError("draft prediction event binding changed")
    ratings_event = ratings["event"]
    if any(event[key] != ratings_event[key] for key in expected_event):
        raise DraftPredictionLedgerError("draft prediction ratings identity changed")
    if captured <= _timestamp(protocol["locked_at_utc"], "protocol.locked_at"):
        raise DraftPredictionLedgerError("draft prediction predates protocol")
    if captured <= max(
        _timestamp(metadata["source"]["available_at_utc"], "draft.source.available_at"),
        _timestamp(
            metadata["protocol_validation"]["validated_at_utc"],
            "draft.protocol.validated_at",
        ),
        _timestamp(ratings["captured_at_utc"], "ratings.captured_at"),
    ):
        raise DraftPredictionLedgerError(
            "draft prediction is not strictly after evidence"
        )
    event_start = _timestamp(ratings["event"]["event_start_utc"], "ratings.event_start")
    source_available = _timestamp(
        metadata["source"]["available_at_utc"], "draft.source.available_at"
    )
    if (
        event_start.replace(tzinfo=None) < FUTURE_SEALED_START
        or source_available < FUTURE_SEALED_START.replace(tzinfo=timezone.utc)
    ):
        raise DraftPredictionLedgerError("draft prediction predates future boundary")
    model = _model(root, protocol)
    if _timestamp(model.model_as_of, "model.model_as_of") >= captured:
        raise DraftPredictionLedgerError("draft model is not frozen before prediction")
    model_record = value.get("model") or {}
    if model_record != {
        "model_version": model.model_version,
        "model_as_of": model.model_as_of,
        "artifact_locator": protocol["locked_candidate"]["artifact_locator"],
        "artifact_raw_sha256": model.artifact_sha256,
        "candidate_id": protocol["locked_candidate"]["candidate_id"],
        "variant_id": protocol["locked_candidate"]["variant_id"],
    }:
        raise DraftPredictionLedgerError("draft prediction model binding changed")
    expected_composition = _score_composition(metadata, model)
    if value.get("draft_index") != expected_composition:
        raise DraftPredictionLedgerError("draft prediction composition replay changed")
    rating_p_blue = float(
        ratings["evaluation_predictions"][RATING_MODEL_ID]["p_blue"]
    )
    rating_logit = _logit(rating_p_blue)
    combined_logit = rating_logit + expected_composition["scaled_logit_blue"]
    combined_p_blue = _sigmoid(combined_logit)
    if value.get("evaluation_predictions") != {
        "ratings_only": {
            "p_blue": rating_p_blue,
            "p_red": 1.0 - rating_p_blue,
            "logit_blue": rating_logit,
        },
        "ratings_plus_draft": {
            "p_blue": combined_p_blue,
            "p_red": 1.0 - combined_p_blue,
            "logit_blue": combined_logit,
        },
    }:
        raise DraftPredictionLedgerError("draft evaluation prediction replay changed")
    qualification = value.get("qualification") or {}
    if qualification != {
        "event_on_or_after_future_boundary": True,
        "ratings_prediction_receipt_valid": True,
        "terminal_draft_source_hash_verified": True,
        "terminal_draft_source_rights_reviewed": True,
        "terminal_assignments_complete": True,
        "pick_ban_protocol_validated": True,
        "system_clock_sampled_inside_builder": True,
        "actual_map_start_authority_present": False,
        "prediction_strictly_before_actual_map_start": None,
        "eligible_future_evidence": False,
        "event_outcome_present": False,
        "event_outcome_accessed": False,
        "independently_pinned_ledger_entry": False,
    }:
        raise DraftPredictionLedgerError("draft prediction qualification changed")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(authority.values()):
        raise DraftPredictionLedgerError("draft prediction exceeds authority")
    if value.get("claim_ceiling") != PREDICTION_CLAIM_CEILING:
        raise DraftPredictionLedgerError("draft prediction claim ceiling changed")
    records = value.get("source_locks")
    if (
        not isinstance(records, list)
        or [item.get("locator") for item in records if isinstance(item, Mapping)]
        != list(SOURCE_LOCKS)
    ):
        raise DraftPredictionLedgerError("draft prediction source inventory changed")
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "locator",
            "bytes",
            "raw_sha256",
        }:
            raise DraftPredictionLedgerError(
                "draft prediction source lock is malformed"
            )
        path = root / str(record["locator"])
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256_path(path) != record.get("raw_sha256")
        ):
            raise DraftPredictionLedgerError(
                f"draft prediction source drifted: {record.get('locator')}"
            )
    return value


def replay_draft_prediction_receipt(
    payload: Mapping[str, Any], *, root: Path = Path(".")
) -> dict[str, Any]:
    value = validate_draft_prediction_receipt(payload, root=root)
    inputs = value["input_receipts"]
    ratings_raw, _ = _decode_embedded(
        inputs["ratings_prediction"], "ratings prediction"
    )
    metadata_raw, _ = _decode_embedded(inputs["draft_metadata"], "draft metadata")
    source_raw = _decode_source_payload(
        inputs["draft_source_payload"], "draft source"
    )
    rebuilt = build_draft_prediction_receipt(
        ratings_receipt_raw=ratings_raw,
        draft_metadata_raw=metadata_raw,
        draft_source_payload_raw=source_raw,
        root=root,
        clock=lambda: _timestamp(value["captured_at_utc"], "captured_at_utc"),
    )
    if rebuilt != value:
        raise DraftPredictionLedgerError("draft prediction receipt replay changed")
    return value


def _validate_map_start_metadata(
    value: Mapping[str, Any], *, source_payload_raw: bytes
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "event_id",
        "series_id",
        "game_number",
        "league",
        "patch",
        "actual_map_start_utc",
        "source",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DraftPredictionLedgerError("map-start metadata structure changed")
    _assert_no_outcomes(value, "map_start_metadata")
    if value.get("schema_version") != "scryglass:actual-map-start-capture-input:v1":
        raise DraftPredictionLedgerError("map-start metadata schema changed")
    for field in ("event_id", "series_id", "league"):
        _nonempty(value.get(field), f"map_start.{field}")
    game_number = value.get("game_number")
    if isinstance(game_number, bool) or not isinstance(game_number, int) or game_number < 1:
        raise DraftPredictionLedgerError("map-start game_number is invalid")
    if not PATCH_RE.fullmatch(_nonempty(value.get("patch"), "map_start.patch")):
        raise DraftPredictionLedgerError("map-start patch is invalid")
    actual_start = _timestamp(
        value.get("actual_map_start_utc"), "map_start.actual_map_start_utc"
    )
    if actual_start.replace(tzinfo=None) < FUTURE_SEALED_START:
        raise DraftPredictionLedgerError("actual map start predates future boundary")
    source = _validate_source(
        value.get("source"), payload_raw=source_payload_raw, field="map_start"
    )
    source_available = _timestamp(
        source["available_at_utc"], "map_start.source.available_at"
    )
    if source_available < actual_start:
        raise DraftPredictionLedgerError(
            "actual map-start authority was available before the claimed start"
        )
    return {**dict(value), "source": source}


def build_map_start_receipt(
    *,
    map_start_metadata_raw: bytes,
    map_start_source_payload_raw: bytes,
    root: Path = Path("."),
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    captured = _clock_sample(clock, "map-start receipt")
    metadata_object = _strict_object(map_start_metadata_raw, "map-start metadata")
    metadata = _validate_map_start_metadata(
        metadata_object, source_payload_raw=map_start_source_payload_raw
    )
    protocol = validate_registered_future_protocol_v1(root=root)
    if captured <= _timestamp(protocol["locked_at_utc"], "protocol.locked_at"):
        raise DraftPredictionLedgerError("map-start receipt predates protocol")
    if captured < _timestamp(
        metadata["source"]["available_at_utc"], "map_start.source.available_at"
    ):
        raise DraftPredictionLedgerError("map-start receipt predates source availability")
    payload: dict[str, Any] = {
        "schema_version": MAP_START_SCHEMA_VERSION,
        "result_state": MAP_START_RESULT_STATE,
        "captured_at_utc": captured.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": captured.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "capture_time_not_after_builder_observation": True,
        },
        "protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "event": {
            key: metadata[key]
            for key in (
                "event_id",
                "series_id",
                "game_number",
                "league",
                "patch",
                "actual_map_start_utc",
            )
        },
        "input_receipts": {
            "map_start_metadata": _embedded(map_start_metadata_raw, metadata_object),
            "map_start_source_payload": {
                "raw_sha256": _sha256_bytes(map_start_source_payload_raw),
                "raw_base64": base64.b64encode(map_start_source_payload_raw).decode(
                    "ascii"
                ),
            },
        },
        "qualification": {
            "actual_map_start_authority_present": True,
            "source_payload_hash_verified": True,
            "source_rights_reviewed": True,
            "event_outcome_present": False,
            "event_outcome_accessed": False,
        },
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": MAP_START_CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_map_start_receipt(payload, root=root)


def validate_map_start_receipt(
    payload: Mapping[str, Any], *, root: Path = Path(".")
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DraftPredictionLedgerError("map-start receipt must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "map_start_receipt")
    if set(value) != {
        "schema_version",
        "result_state",
        "captured_at_utc",
        "clock_attestation",
        "protocol_artifact_sha256",
        "event",
        "input_receipts",
        "qualification",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise DraftPredictionLedgerError("map-start receipt structure changed")
    if (
        value.get("schema_version") != MAP_START_SCHEMA_VERSION
        or value.get("result_state") != MAP_START_RESULT_STATE
    ):
        raise DraftPredictionLedgerError("map-start receipt identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise DraftPredictionLedgerError("map-start receipt canonical hash changed")
    protocol = validate_registered_future_protocol_v1(root=root)
    if value.get("protocol_artifact_sha256") != protocol.get("artifact_sha256"):
        raise DraftPredictionLedgerError("map-start protocol binding changed")
    captured = _timestamp(value.get("captured_at_utc"), "captured_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": captured.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "capture_time_not_after_builder_observation": True,
    }:
        raise DraftPredictionLedgerError("map-start clock attestation changed")
    inputs = value.get("input_receipts") or {}
    if set(inputs) != {"map_start_metadata", "map_start_source_payload"}:
        raise DraftPredictionLedgerError("map-start input structure changed")
    metadata_raw, metadata_object = _decode_embedded(
        inputs.get("map_start_metadata") or {}, "map-start metadata"
    )
    source_raw = _decode_source_payload(
        inputs.get("map_start_source_payload"), "map-start source"
    )
    metadata = _validate_map_start_metadata(
        metadata_object, source_payload_raw=source_raw
    )
    expected_event = {
        key: metadata[key]
        for key in (
            "event_id",
            "series_id",
            "game_number",
            "league",
            "patch",
            "actual_map_start_utc",
        )
    }
    if value.get("event") != expected_event:
        raise DraftPredictionLedgerError("map-start event binding changed")
    if captured < _timestamp(
        metadata["source"]["available_at_utc"], "map_start.source.available_at"
    ):
        raise DraftPredictionLedgerError("map-start receipt predates source")
    if value.get("qualification") != {
        "actual_map_start_authority_present": True,
        "source_payload_hash_verified": True,
        "source_rights_reviewed": True,
        "event_outcome_present": False,
        "event_outcome_accessed": False,
    }:
        raise DraftPredictionLedgerError("map-start qualification changed")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(authority.values()):
        raise DraftPredictionLedgerError("map-start receipt exceeds authority")
    if value.get("claim_ceiling") != MAP_START_CLAIM_CEILING:
        raise DraftPredictionLedgerError("map-start claim ceiling changed")
    return value


def _receipt_locator(value: Any, prefix: PurePosixPath, field: str) -> str:
    path = PurePosixPath(_nonempty(value, field))
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or tuple(path.parts[: len(prefix.parts)]) != prefix.parts
        or path.suffix != ".json"
    ):
        raise DraftPredictionLedgerError(f"{field} is outside its receipt root")
    return path.as_posix()


def _patch_key(value: str) -> tuple[int, int]:
    major, minor = value.split(".", 1)
    return int(major), int(minor)


def _support(entries: Sequence[Mapping[str, Any]], rule: Mapping[str, Any]) -> dict[str, Any]:
    patches = sorted({str(entry["patch"]) for entry in entries}, key=_patch_key)
    latest_patch = patches[-1] if patches else None
    series = {str(entry["series_id"]) for entry in entries}
    maps_by_league = {
        league: sum(entry["league"] == league for entry in entries)
        for league in rule["domestic_leagues"]
    }
    international_maps = sum(entry["league"] in {"MSI", "EWC"} for entry in entries)
    support = {
        "eligible_maps": len(entries),
        "eligible_series": len(series),
        "maps_by_domestic_league": maps_by_league,
        "international_maps": international_maps,
        "distinct_future_patches": len(patches),
        "latest_future_patch": latest_patch,
        "latest_future_patch_maps": sum(
            entry["patch"] == latest_patch for entry in entries
        )
        if latest_patch
        else 0,
        "sparse_or_new_champion_maps": sum(
            bool(entry["sparse_or_new_champion_map"]) for entry in entries
        ),
    }
    support["support_met"] = (
        support["eligible_maps"] >= rule["eligible_maps_minimum"]
        and support["eligible_series"] >= rule["eligible_series_minimum"]
        and all(
            maps_by_league[league] >= rule["each_domestic_league_maps_minimum"]
            for league in rule["domestic_leagues"]
        )
        and support["international_maps"] >= rule["international_maps_minimum"]
        and support["distinct_future_patches"]
        >= rule["distinct_future_patches_minimum"]
        and support["latest_future_patch_maps"]
        >= rule["latest_future_patch_maps_minimum"]
        and support["sparse_or_new_champion_maps"]
        >= rule["sparse_or_new_champion_maps_minimum"]
    )
    return support


def _ledger_entry(
    *,
    prediction_locator: str,
    prediction_payload: Mapping[str, Any],
    map_start_locator: str,
    map_start_payload: Mapping[str, Any],
    created: datetime,
    root: Path,
) -> dict[str, Any]:
    prediction = validate_draft_prediction_receipt(prediction_payload, root=root)
    map_start = validate_map_start_receipt(map_start_payload, root=root)
    prediction_locator = _receipt_locator(
        prediction_locator, PREDICTION_PREFIX, "prediction_locator"
    )
    map_start_locator = _receipt_locator(
        map_start_locator, MAP_START_PREFIX, "map_start_locator"
    )
    event = prediction["event"]
    start_event = map_start["event"]
    for field in ("event_id", "series_id", "game_number", "league", "patch"):
        if event[field] != start_event[field]:
            raise DraftPredictionLedgerError(
                f"prediction and map-start identity differ: {field}"
            )
    prediction_time = _timestamp(
        prediction["captured_at_utc"], "prediction.captured_at"
    )
    actual_start = _timestamp(
        start_event["actual_map_start_utc"], "actual_map_start"
    )
    map_start_capture = _timestamp(
        map_start["captured_at_utc"], "map_start.captured_at"
    )
    metadata = prediction["input_receipts"]["draft_metadata"]["value"]
    draft_source_time = _timestamp(
        metadata["source"]["available_at_utc"], "draft.source.available_at"
    )
    protocol_validation_time = _timestamp(
        metadata["protocol_validation"]["validated_at_utc"],
        "draft.protocol.validated_at",
    )
    model_as_of = _timestamp(
        prediction["model"]["model_as_of"], "model.model_as_of"
    )
    if not (
        max(draft_source_time, protocol_validation_time, model_as_of)
        < prediction_time
        < actual_start
    ):
        raise DraftPredictionLedgerError(
            "draft prediction is not strictly after evidence and before map start"
        )
    if created < max(prediction_time, map_start_capture):
        raise DraftPredictionLedgerError("draft ledger predates an input receipt")
    return {
        "event_id": event["event_id"],
        "series_id": event["series_id"],
        "game_number": event["game_number"],
        "league": event["league"],
        "patch": event["patch"],
        "prediction_captured_at_utc": prediction["captured_at_utc"],
        "actual_map_start_utc": start_event["actual_map_start_utc"],
        "prediction_locator": prediction_locator,
        "prediction_artifact_sha256": prediction["artifact_sha256"],
        "map_start_locator": map_start_locator,
        "map_start_artifact_sha256": map_start["artifact_sha256"],
        "sparse_or_new_champion_map": prediction["draft_index"][
            "sparse_or_new_champion_map"
        ],
    }


def build_prediction_ledger(
    *,
    receipts: Sequence[
        tuple[str, Mapping[str, Any], str, Mapping[str, Any]]
    ],
    root: Path = Path("."),
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    created = _clock_sample(clock, "draft prediction ledger")
    protocol = validate_registered_future_protocol_v1(root=root)
    entries: list[dict[str, Any]] = []
    receipt_payloads: dict[
        tuple[str, str], tuple[Mapping[str, Any], Mapping[str, Any]]
    ] = {}
    for prediction_locator, prediction_payload, start_locator, start_payload in receipts:
        prediction_locator = _receipt_locator(
            prediction_locator, PREDICTION_PREFIX, "prediction_locator"
        )
        start_locator = _receipt_locator(
            start_locator, MAP_START_PREFIX, "map_start_locator"
        )
        key = (prediction_locator, start_locator)
        if key in receipt_payloads:
            raise DraftPredictionLedgerError("draft ledger repeats a receipt pair")
        receipt_payloads[key] = (prediction_payload, start_payload)
        entries.append(
            _ledger_entry(
                prediction_locator=prediction_locator,
                prediction_payload=prediction_payload,
                map_start_locator=start_locator,
                map_start_payload=start_payload,
                created=created,
                root=root,
            )
        )
    entries.sort(
        key=lambda item: (
            item["actual_map_start_utc"],
            item["event_id"],
            item["game_number"],
        )
    )
    identities = {
        (entry["event_id"], entry["game_number"]) for entry in entries
    }
    if len(identities) != len(entries):
        raise DraftPredictionLedgerError("draft ledger has duplicate map identities")
    rule = protocol["future_holdout"]["metadata_only_support_stopping_rule"]
    support = _support(entries, rule)
    payload: dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "status": (
            "SUPPORT_MET_OUTCOMES_UNOPENED"
            if support["support_met"]
            else "COLLECTING_OUTCOME_FREE_DRAFT_PREDICTIONS"
        ),
        "created_at_utc": created.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": created.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "creation_time_not_after_builder_observation": True,
        },
        "protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "entries": entries,
        "metadata_support": support,
        "support_stopping_rule": rule,
        "outcomes_present": False,
        "outcomes_accessed": False,
        "independently_pinned": False,
        "opening_authority": False,
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": LEDGER_CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return _validate_prediction_ledger(
        payload,
        root=root,
        receipt_payloads=receipt_payloads,
    )


def _validate_prediction_ledger(
    payload: Mapping[str, Any],
    *,
    root: Path = Path("."),
    receipt_payloads: Mapping[
        tuple[str, str], tuple[Mapping[str, Any], Mapping[str, Any]]
    ]
    | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DraftPredictionLedgerError("draft prediction ledger must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "draft_prediction_ledger")
    if set(value) != {
        "schema_version",
        "status",
        "created_at_utc",
        "clock_attestation",
        "protocol_artifact_sha256",
        "entries",
        "metadata_support",
        "support_stopping_rule",
        "outcomes_present",
        "outcomes_accessed",
        "independently_pinned",
        "opening_authority",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise DraftPredictionLedgerError("draft prediction ledger structure changed")
    if value.get("schema_version") != LEDGER_SCHEMA_VERSION or value.get(
        "status"
    ) not in {
        "COLLECTING_OUTCOME_FREE_DRAFT_PREDICTIONS",
        "SUPPORT_MET_OUTCOMES_UNOPENED",
    }:
        raise DraftPredictionLedgerError("draft prediction ledger identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise DraftPredictionLedgerError("draft prediction ledger hash changed")
    protocol = validate_registered_future_protocol_v1(root=root)
    if value.get("protocol_artifact_sha256") != protocol.get("artifact_sha256"):
        raise DraftPredictionLedgerError("draft prediction ledger protocol changed")
    created = _timestamp(value.get("created_at_utc"), "created_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": created.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "creation_time_not_after_builder_observation": True,
    }:
        raise DraftPredictionLedgerError("draft prediction ledger clock changed")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise DraftPredictionLedgerError("draft prediction ledger entries are malformed")
    expected_keys = {
        "event_id",
        "series_id",
        "game_number",
        "league",
        "patch",
        "prediction_captured_at_utc",
        "actual_map_start_utc",
        "prediction_locator",
        "prediction_artifact_sha256",
        "map_start_locator",
        "map_start_artifact_sha256",
        "sparse_or_new_champion_map",
    }
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != expected_keys:
            raise DraftPredictionLedgerError("draft ledger entry structure changed")
        prediction_time = _timestamp(
            entry["prediction_captured_at_utc"], "entry.prediction_captured_at"
        )
        actual_start = _timestamp(
            entry["actual_map_start_utc"], "entry.actual_map_start"
        )
        if not prediction_time < actual_start or created < prediction_time:
            raise DraftPredictionLedgerError("draft ledger entry timing changed")
        prediction_locator = _receipt_locator(
            entry["prediction_locator"], PREDICTION_PREFIX, "prediction_locator"
        )
        map_start_locator = _receipt_locator(
            entry["map_start_locator"], MAP_START_PREFIX, "map_start_locator"
        )
        _sha(entry["prediction_artifact_sha256"], "prediction_artifact_sha256")
        _sha(entry["map_start_artifact_sha256"], "map_start_artifact_sha256")
        if entry["league"] not in protocol["future_holdout"]["eligibility"]["leagues"]:
            raise DraftPredictionLedgerError("draft ledger league is ineligible")
        if not PATCH_RE.fullmatch(str(entry["patch"])):
            raise DraftPredictionLedgerError("draft ledger patch is invalid")
        if not isinstance(entry["sparse_or_new_champion_map"], bool):
            raise DraftPredictionLedgerError("draft ledger sparse flag is invalid")
        pair = (prediction_locator, map_start_locator)
        if receipt_payloads is None:
            prediction_payload = _strict_object(
                (root / prediction_locator).read_bytes(), "draft prediction"
            )
            map_start_payload = _strict_object(
                (root / map_start_locator).read_bytes(), "map-start receipt"
            )
        else:
            try:
                prediction_payload, map_start_payload = receipt_payloads[pair]
            except KeyError as exc:
                raise DraftPredictionLedgerError(
                    "draft ledger receipt pair is not available for validation"
                ) from exc
        expected_entry = _ledger_entry(
            prediction_locator=prediction_locator,
            prediction_payload=prediction_payload,
            map_start_locator=map_start_locator,
            map_start_payload=map_start_payload,
            created=created,
            root=root,
        )
        if dict(entry) != expected_entry:
            raise DraftPredictionLedgerError(
                "draft ledger entry differs from its bound receipts"
            )
    ordered = sorted(
        entries,
        key=lambda item: (
            item["actual_map_start_utc"],
            item["event_id"],
            item["game_number"],
        ),
    )
    if entries != ordered or len(
        {(item["event_id"], item["game_number"]) for item in entries}
    ) != len(entries):
        raise DraftPredictionLedgerError("draft ledger entries are unordered or duplicated")
    rule = protocol["future_holdout"]["metadata_only_support_stopping_rule"]
    if value.get("support_stopping_rule") != rule:
        raise DraftPredictionLedgerError("draft ledger stopping rule changed")
    expected_support = _support(entries, rule)
    if value.get("metadata_support") != expected_support:
        raise DraftPredictionLedgerError("draft ledger metadata support changed")
    expected_status = (
        "SUPPORT_MET_OUTCOMES_UNOPENED"
        if expected_support["support_met"]
        else "COLLECTING_OUTCOME_FREE_DRAFT_PREDICTIONS"
    )
    if value.get("status") != expected_status:
        raise DraftPredictionLedgerError("draft ledger status differs from support")
    if any(
        value.get(field) is not False
        for field in (
            "outcomes_present",
            "outcomes_accessed",
            "independently_pinned",
            "opening_authority",
        )
    ):
        raise DraftPredictionLedgerError("draft ledger exceeds opening boundary")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(authority.values()):
        raise DraftPredictionLedgerError("draft ledger exceeds authority")
    if value.get("claim_ceiling") != LEDGER_CLAIM_CEILING:
        raise DraftPredictionLedgerError("draft ledger claim ceiling changed")
    return value


def validate_prediction_ledger(
    payload: Mapping[str, Any], *, root: Path = Path(".")
) -> dict[str, Any]:
    """Validate a ledger against the receipt bytes at every declared locator."""

    return _validate_prediction_ledger(
        payload,
        root=root,
        receipt_payloads=None,
    )


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DraftPredictionLedgerError(f"refusing to overwrite receipt: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise DraftPredictionLedgerError(
                f"refusing to overwrite receipt: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256_bytes(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--ratings-receipt", type=Path, required=True)
    capture.add_argument("--draft-metadata", type=Path, required=True)
    capture.add_argument("--draft-source-payload", type=Path, required=True)
    capture.add_argument("--out", type=Path, required=True)
    map_start = subparsers.add_parser("map-start")
    map_start.add_argument("--metadata", type=Path, required=True)
    map_start.add_argument("--source-payload", type=Path, required=True)
    map_start.add_argument("--out", type=Path, required=True)
    ledger_parser = subparsers.add_parser("ledger")
    ledger_parser.add_argument("--pair-manifest", type=Path, required=True)
    ledger_parser.add_argument("--out", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            payload = build_draft_prediction_receipt(
                ratings_receipt_raw=args.ratings_receipt.read_bytes(),
                draft_metadata_raw=args.draft_metadata.read_bytes(),
                draft_source_payload_raw=args.draft_source_payload.read_bytes(),
                root=args.root,
            )
        elif args.command == "map-start":
            payload = build_map_start_receipt(
                map_start_metadata_raw=args.metadata.read_bytes(),
                map_start_source_payload_raw=args.source_payload.read_bytes(),
                root=args.root,
            )
        else:
            manifest = _strict_object(
                args.pair_manifest.read_bytes(), "draft ledger pair manifest"
            )
            pairs = manifest.get("pairs")
            if not isinstance(pairs, list):
                raise DraftPredictionLedgerError("pair manifest pairs are missing")
            loaded = []
            for pair in pairs:
                if not isinstance(pair, Mapping) or set(pair) != {
                    "prediction_locator",
                    "map_start_locator",
                }:
                    raise DraftPredictionLedgerError("pair manifest entry is malformed")
                prediction_locator = _receipt_locator(
                    pair["prediction_locator"],
                    PREDICTION_PREFIX,
                    "prediction_locator",
                )
                start_locator = _receipt_locator(
                    pair["map_start_locator"], MAP_START_PREFIX, "map_start_locator"
                )
                prediction = _strict_object(
                    (args.root / prediction_locator).read_bytes(), "draft prediction"
                )
                start = _strict_object(
                    (args.root / start_locator).read_bytes(), "map-start receipt"
                )
                loaded.append(
                    (prediction_locator, prediction, start_locator, start)
                )
            payload = build_prediction_ledger(receipts=loaded, root=args.root)
        raw_sha256 = write_no_clobber(args.out, payload)
    except (OSError, ValueError, DraftPredictionLedgerError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "command": args.command,
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_KEYS",
    "DEFAULT_LEDGER",
    "DraftPredictionLedgerError",
    "LEDGER_SCHEMA_VERSION",
    "MAP_START_SCHEMA_VERSION",
    "PREDICTION_SCHEMA_VERSION",
    "build_draft_prediction_receipt",
    "build_map_start_receipt",
    "build_prediction_ledger",
    "replay_draft_prediction_receipt",
    "validate_draft_prediction_receipt",
    "validate_map_start_receipt",
    "validate_prediction_ledger",
    "write_no_clobber",
]
