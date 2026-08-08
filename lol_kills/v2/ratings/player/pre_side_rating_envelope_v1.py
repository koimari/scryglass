"""Seal both rating side conditionals before public map sides are known.

This is a non-authorizing bridge artifact for prospective collection.  It
builds the already-registered rating receipt twice at one system-clock sample:
once with source-order team1 on blue and once with team2 on blue.  The child
receipts remain embedded and are explicitly ineligible for the v3 prediction
ledger.  A future reviewed side-binding protocol may select exactly one child;
this module cannot make that selection, create a phase-one plan, or grant any
rating, probability, odds, EV, recommendation, or betting authority.
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
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

from lol_kills import pregame_roster_capture as roster_capture

from . import multileague_v3_prediction_ledger as rating_ledger


ROOT = Path(__file__).resolve().parents[4]
INPUT_SCHEMA_VERSION = "scryglass:pre-side-rating-input:v1"
ENVELOPE_SCHEMA_VERSION = "scryglass:pre-side-rating-envelope:v1"
RESULT_STATE = "BOTH_SIDE_CONDITIONALS_SEALED_AWAITING_PUBLIC_SIDE_BINDING"
SOURCE_LOCATOR = "lol_kills/v2/ratings/player/pre_side_rating_envelope_v1.py"
ENVELOPE_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/multileague-v3/pre-side-rating-envelopes"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_SLUG_RE = re.compile(r"[^a-z0-9._-]+")
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
    "model_validation_authority",
    "player_rating_authority",
    "team_rating_authority",
    "side_binding_authority",
    "prediction_ledger_authority",
    "probability_authority",
    "odds_authority",
    "expected_value_authority",
    "recommendation_authority",
    "betting_authority",
)
CLAIM_CEILING = (
    "Both side-conditional rating evaluation receipts were computed and sealed "
    "before the scheduled series start. Neither embedded child is eligible "
    "ledger evidence until a separately reviewed public side-binding and actual "
    "map-start protocol selects exactly one. No rating, probability, odds, EV, "
    "recommendation, or betting authority is granted."
)


class PreSideRatingEnvelopeError(ValueError):
    """A pre-side rating envelope failed closed."""


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
        raise PreSideRatingEnvelopeError("pre-side value is not canonical") from exc


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
        raise PreSideRatingEnvelopeError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreSideRatingEnvelopeError(f"{field} must be a non-empty string")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    text = _nonempty(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PreSideRatingEnvelopeError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PreSideRatingEnvelopeError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PreSideRatingEnvelopeError(
            "pre-side capture clock must return a timezone-aware datetime"
        )
    return value.astimezone(timezone.utc)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PreSideRatingEnvelopeError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_object(raw: bytes, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PreSideRatingEnvelopeError(
                    f"non-finite JSON number in {field}: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreSideRatingEnvelopeError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PreSideRatingEnvelopeError(f"{field} must be a JSON object")
    return value


def _assert_no_outcomes(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
            if normalized in OUTCOME_KEYS:
                raise PreSideRatingEnvelopeError(
                    f"event outcome field is forbidden: {path}.{key}"
                )
            _assert_no_outcomes(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_outcomes(item, f"{path}[{index}]")


def _source_record(root: Path) -> dict[str, Any]:
    path = root / SOURCE_LOCATOR
    if not path.is_file() or path.is_symlink():
        raise PreSideRatingEnvelopeError("pre-side implementation is unavailable")
    return {
        "locator": SOURCE_LOCATOR,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _authority_false() -> dict[str, bool]:
    return {name: False for name in AUTHORITY_KEYS}


def _validate_team(value: Any, slot: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "slot",
        "organization_id",
        "organization_name",
        "roster_id",
        "players",
    }:
        raise PreSideRatingEnvelopeError(f"{slot} team structure changed")
    if value.get("slot") != slot:
        raise PreSideRatingEnvelopeError("source-order teams must be team1 then team2")
    players = value.get("players")
    if not isinstance(players, list) or len(players) != len(roster_capture.ROLES):
        raise PreSideRatingEnvelopeError(f"{slot} must contain exactly five players")
    normalized_players: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, role in enumerate(roster_capture.ROLES):
        player = players[index]
        if not isinstance(player, Mapping) or set(player) != {
            "role",
            "player_id",
            "display_name",
        }:
            raise PreSideRatingEnvelopeError(f"{slot}.{role} player changed")
        if player.get("role") != role:
            raise PreSideRatingEnvelopeError(
                f"{slot} players must be ordered {roster_capture.ROLES}"
            )
        player_id = _nonempty(player.get("player_id"), f"{slot}.{role}.player_id")
        if player_id in seen:
            raise PreSideRatingEnvelopeError(f"{slot} repeats a player")
        seen.add(player_id)
        normalized_players.append(
            {
                "role": role,
                "player_id": player_id,
                "display_name": _nonempty(
                    player.get("display_name"), f"{slot}.{role}.display_name"
                ),
            }
        )
    return {
        "slot": slot,
        "organization_id": _nonempty(
            value.get("organization_id"), f"{slot}.organization_id"
        ),
        "organization_name": _nonempty(
            value.get("organization_name"), f"{slot}.organization_name"
        ),
        "roster_id": _nonempty(value.get("roster_id"), f"{slot}.roster_id"),
        "players": normalized_players,
    }


def validate_input(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PreSideRatingEnvelopeError("pre-side input must be an object")
    _assert_no_outcomes(value, "pre_side_input")
    if set(value) != {"schema_version", "event", "roster_source", "teams"}:
        raise PreSideRatingEnvelopeError("pre-side input structure changed")
    if value.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise PreSideRatingEnvelopeError("pre-side input schema changed")
    event = value.get("event")
    if not isinstance(event, Mapping) or set(event) != {
        "event_id",
        "series_id",
        "game_number",
        "scheduled_series_start_utc",
        "league",
    }:
        raise PreSideRatingEnvelopeError("pre-side event structure changed")
    for field in ("event_id", "series_id", "league"):
        _nonempty(event.get(field), f"event.{field}")
    league = str(event["league"])
    if league != league.upper():
        raise PreSideRatingEnvelopeError("event.league must be uppercase")
    game_number = event.get("game_number")
    if isinstance(game_number, bool) or not isinstance(game_number, int) or game_number < 1:
        raise PreSideRatingEnvelopeError("event.game_number must be positive")
    _timestamp(
        event.get("scheduled_series_start_utc"),
        "event.scheduled_series_start_utc",
    )
    source = value.get("roster_source")
    if not isinstance(source, Mapping) or set(source) != {
        "source",
        "source_url",
        "source_record_id",
        "source_updated_at_utc",
        "available_at_utc",
        "rights_status",
    }:
        raise PreSideRatingEnvelopeError("pre-side roster source structure changed")
    for field in ("source", "source_url", "source_record_id"):
        _nonempty(source.get(field), f"roster_source.{field}")
    _timestamp(source.get("source_updated_at_utc"), "source_updated_at_utc")
    _timestamp(source.get("available_at_utc"), "available_at_utc")
    if source.get("rights_status") != "reviewed":
        raise PreSideRatingEnvelopeError("pre-side roster source rights are not reviewed")
    teams = value.get("teams")
    if not isinstance(teams, list) or len(teams) != 2:
        raise PreSideRatingEnvelopeError("pre-side input requires team1 and team2")
    team1 = _validate_team(teams[0], "team1")
    team2 = _validate_team(teams[1], "team2")
    if team1["organization_id"] == team2["organization_id"]:
        raise PreSideRatingEnvelopeError("pre-side organizations must be distinct")
    if len(
        {
            player["player_id"]
            for team in (team1, team2)
            for player in team["players"]
        }
    ) != 10:
        raise PreSideRatingEnvelopeError("pre-side players must be unique across teams")
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "event": dict(event),
        "roster_source": dict(source),
        "teams": [team1, team2],
    }


def _as_side_team(team: Mapping[str, Any], side: str) -> dict[str, Any]:
    return {
        "side": side,
        "organization_id": team["organization_id"],
        "organization_name": team["organization_name"],
        "roster_id": team["roster_id"],
        "players": team["players"],
    }


def _child_raw(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


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
        raise PreSideRatingEnvelopeError(f"{field} embedding changed")
    try:
        raw = base64.b64decode(
            _nonempty(value.get("raw_base64"), f"{field}.raw_base64"),
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise PreSideRatingEnvelopeError(f"{field} base64 is invalid") from exc
    if _sha256_bytes(raw) != _sha(value.get("raw_sha256"), f"{field}.raw_sha256"):
        raise PreSideRatingEnvelopeError(f"{field} raw hash changed")
    parsed = _strict_object(raw, field)
    if parsed != value.get("value"):
        raise PreSideRatingEnvelopeError(f"{field} embedded value changed")
    return raw, parsed


def _conditional_summary(
    team1_blue: Mapping[str, Any], team2_blue: Mapping[str, Any]
) -> dict[str, Any]:
    first = team1_blue.get("evaluation_predictions") or {}
    second = team2_blue.get("evaluation_predictions") or {}
    if set(first) != set(rating_ledger.MODEL_IDS) or set(second) != set(
        rating_ledger.MODEL_IDS
    ):
        raise PreSideRatingEnvelopeError("conditional model inventory changed")
    models: dict[str, Any] = {}
    for model_id in rating_ledger.MODEL_IDS:
        p_team1_if_blue = float(first[model_id]["p_blue"])
        p_team2_if_blue = float(second[model_id]["p_blue"])
        p_team1_if_red = float(second[model_id]["p_red"])
        p_team2_if_red = float(first[model_id]["p_red"])
        if not math.isclose(
            p_team1_if_red,
            1.0 - p_team2_if_blue,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            p_team2_if_red,
            1.0 - p_team1_if_blue,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise PreSideRatingEnvelopeError(
                "conditional selected-team probability changed under orientation"
            )
        first_latent = float(first[model_id]["latent_mean"])
        second_latent = float(second[model_id]["latent_mean"])
        if not math.isfinite(first_latent) or not math.isfinite(second_latent):
            raise PreSideRatingEnvelopeError("conditional latent mean is invalid")
        models[model_id] = {
            "p_team1_if_blue": p_team1_if_blue,
            "p_team1_if_red": p_team1_if_red,
            "p_team2_if_blue": p_team2_if_blue,
            "p_team2_if_red": p_team2_if_red,
            "team1_blue_latent_mean": first_latent,
            "team2_blue_latent_mean": second_latent,
            "derived_neutral_team1_latent": (first_latent - second_latent) / 2.0,
            "derived_blue_side_latent": (first_latent + second_latent) / 2.0,
            "selected_probability_uses_embedded_orientation": True,
            "complementarity_assumed": False,
        }
    return {
        "models": models,
        "both_conditionals_computed_at_same_clock_sample": True,
        "blue_side_advantage_applies_in_both_orientations": True,
    }


def build_pre_side_rating_envelope(
    *,
    input_raw: bytes,
    roster_source_payload_raw: bytes,
    patch_receipt_raw: bytes,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    captured = _clock_sample(clock)
    input_object = _strict_object(input_raw, "pre-side input")
    checked = validate_input(input_object)
    event = checked["event"]
    scheduled_start = _timestamp(
        event["scheduled_series_start_utc"], "scheduled_series_start_utc"
    )
    if captured >= scheduled_start:
        raise PreSideRatingEnvelopeError("pre-side envelope is not pre-event")
    source = checked["roster_source"]
    if captured < max(
        _timestamp(source["source_updated_at_utc"], "source_updated_at_utc"),
        _timestamp(source["available_at_utc"], "available_at_utc"),
    ):
        raise PreSideRatingEnvelopeError("pre-side envelope predates roster evidence")

    children: dict[str, dict[str, Any]] = {}
    for scenario, ordered in (
        ("team1_blue", (checked["teams"][0], checked["teams"][1])),
        ("team2_blue", (checked["teams"][1], checked["teams"][0])),
    ):
        roster = roster_capture.build_pregame_roster_receipt(
            raw_source_payload=roster_source_payload_raw,
            source=source["source"],
            source_url=source["source_url"],
            source_record_id=f"{source['source_record_id']}:{scenario}",
            source_updated_at=source["source_updated_at_utc"],
            available_at=source["available_at_utc"],
            captured_at=captured.isoformat(),
            event_id=event["event_id"],
            event_start=event["scheduled_series_start_utc"],
            league=event["league"],
            teams=[_as_side_team(ordered[0], "blue"), _as_side_team(ordered[1], "red")],
            capture_protocol_sha256=_source_record(root)["raw_sha256"],
            rights_status=source["rights_status"],
        )
        roster_raw = _child_raw(roster)
        rating = rating_ledger.build_pre_event_prediction_receipt(
            roster_receipt_raw=roster_raw,
            patch_receipt_raw=patch_receipt_raw,
            series_id=event["series_id"],
            game_number=event["game_number"],
            root=root,
            clock=lambda: captured,
        )
        children[scenario] = {
            "blue_slot": ordered[0]["slot"],
            "red_slot": ordered[1]["slot"],
            "rating_receipt": _embedded(_child_raw(rating), rating),
        }

    payload: dict[str, Any] = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "captured_at_utc": captured.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": captured.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "event": dict(event),
        "input_bindings": {
            "pre_side_input_raw_sha256": _sha256_bytes(input_raw),
            "roster_source_payload_raw_sha256": _sha256_bytes(
                roster_source_payload_raw
            ),
            "patch_receipt_raw_sha256": _sha256_bytes(patch_receipt_raw),
        },
        "roster_source": dict(source),
        "source_order_teams": checked["teams"],
        "side_conditionals": children,
        "conditional_summary": _conditional_summary(
            children["team1_blue"]["rating_receipt"]["value"],
            children["team2_blue"]["rating_receipt"]["value"],
        ),
        "qualification": {
            "scheduled_series_start_is_public_cutoff": True,
            "both_side_conditionals_sealed_pre_event": True,
            "actual_blue_red_side_known": False,
            "side_binding_present": False,
            "embedded_child_receipts_individually_ledger_eligible": False,
            "eligible_evaluation_map": False,
            "event_outcome_present": False,
            "event_outcome_accessed": False,
        },
        "implementation": _source_record(root),
        "authority": _authority_false(),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_pre_side_rating_envelope(payload, root=root)


def validate_pre_side_rating_envelope(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PreSideRatingEnvelopeError("pre-side envelope must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "pre_side_envelope")
    expected = {
        "schema_version",
        "result_state",
        "captured_at_utc",
        "clock_attestation",
        "event",
        "input_bindings",
        "roster_source",
        "source_order_teams",
        "side_conditionals",
        "conditional_summary",
        "qualification",
        "implementation",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }
    if set(value) != expected:
        raise PreSideRatingEnvelopeError("pre-side envelope structure changed")
    if (
        value.get("schema_version") != ENVELOPE_SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise PreSideRatingEnvelopeError("pre-side envelope identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PreSideRatingEnvelopeError("pre-side envelope hash changed")
    captured = _timestamp(value.get("captured_at_utc"), "captured_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": captured.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise PreSideRatingEnvelopeError("pre-side envelope clock changed")
    event = value.get("event")
    roster_source = value.get("roster_source")
    input_view = validate_input(
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "event": event,
            "roster_source": roster_source,
            "teams": value.get("source_order_teams"),
        }
    )
    if event != input_view["event"]:
        raise PreSideRatingEnvelopeError("pre-side event changed")
    bindings = value.get("input_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "pre_side_input_raw_sha256",
        "roster_source_payload_raw_sha256",
        "patch_receipt_raw_sha256",
    }:
        raise PreSideRatingEnvelopeError("pre-side input bindings changed")
    for field, digest in bindings.items():
        _sha(digest, field)
    children = value.get("side_conditionals")
    if not isinstance(children, Mapping) or set(children) != {
        "team1_blue",
        "team2_blue",
    }:
        raise PreSideRatingEnvelopeError("side conditional inventory changed")
    checked_children: dict[str, dict[str, Any]] = {}
    for scenario, blue_slot, red_slot in (
        ("team1_blue", "team1", "team2"),
        ("team2_blue", "team2", "team1"),
    ):
        child = children[scenario]
        if not isinstance(child, Mapping) or set(child) != {
            "blue_slot",
            "red_slot",
            "rating_receipt",
        }:
            raise PreSideRatingEnvelopeError(f"{scenario} structure changed")
        if child.get("blue_slot") != blue_slot or child.get("red_slot") != red_slot:
            raise PreSideRatingEnvelopeError(f"{scenario} orientation changed")
        child_raw, child_value = _decode_embedded(
            child.get("rating_receipt"), f"{scenario}.rating_receipt"
        )
        if child_raw != _child_raw(child_value):
            raise PreSideRatingEnvelopeError(
                f"{scenario} rating receipt bytes are not canonical"
            )
        checked_child = rating_ledger.validate_pre_event_prediction_receipt(
            child_value, root=root
        )
        if _timestamp(checked_child["captured_at_utc"], "child.captured_at") != captured:
            raise PreSideRatingEnvelopeError("conditional capture clocks differ")
        child_event = checked_child["event"]
        for field in ("event_id", "series_id", "game_number", "league"):
            if child_event.get(field) != event.get(field):
                raise PreSideRatingEnvelopeError(f"conditional event differs: {field}")
        if child_event.get("event_start_utc") != _timestamp(
            event.get("scheduled_series_start_utc"), "scheduled_series_start_utc"
        ).isoformat():
            raise PreSideRatingEnvelopeError("conditional scheduled start changed")
        roster_receipt = checked_child["input_receipts"]["roster"]["receipt"]
        patch_record = checked_child["input_receipts"]["patch"]
        expected_teams = [
            _as_side_team(
                input_view["teams"][0 if blue_slot == "team1" else 1], "blue"
            ),
            _as_side_team(
                input_view["teams"][1 if red_slot == "team2" else 0], "red"
            ),
        ]
        if roster_receipt.get("teams") != expected_teams:
            raise PreSideRatingEnvelopeError(
                f"{scenario} exact player roster binding changed"
            )
        expected_source_record_id = (
            f"{input_view['roster_source']['source_record_id']}:{scenario}"
        )
        expected_source_fields = {
            "source": input_view["roster_source"]["source"],
            "source_url": input_view["roster_source"]["source_url"],
            "source_record_id": expected_source_record_id,
            "source_updated_at": input_view["roster_source"][
                "source_updated_at_utc"
            ],
            "available_at": input_view["roster_source"]["available_at_utc"],
            "rights_status": "reviewed",
            "source_payload_sha256": bindings[
                "roster_source_payload_raw_sha256"
            ],
        }
        if any(
            roster_receipt.get(field) != expected
            for field, expected in expected_source_fields.items()
        ):
            raise PreSideRatingEnvelopeError(
                f"{scenario} roster source binding changed"
            )
        if patch_record.get("raw_sha256") != bindings["patch_receipt_raw_sha256"]:
            raise PreSideRatingEnvelopeError(
                f"{scenario} patch byte binding changed"
            )
        checked_children[scenario] = checked_child
    teams = value["source_order_teams"]
    first_event = checked_children["team1_blue"]["event"]
    second_event = checked_children["team2_blue"]["event"]
    if (
        first_event["blue_organization_id"] != teams[0]["organization_id"]
        or first_event["red_organization_id"] != teams[1]["organization_id"]
        or second_event["blue_organization_id"] != teams[1]["organization_id"]
        or second_event["red_organization_id"] != teams[0]["organization_id"]
    ):
        raise PreSideRatingEnvelopeError("conditional organization orientation changed")
    for field in (
        "protocol",
        "source_snapshot",
        "source_preflight",
        "source_locks",
    ):
        if checked_children["team1_blue"].get(field) != checked_children[
            "team2_blue"
        ].get(field):
            raise PreSideRatingEnvelopeError(
                f"conditional frozen model binding differs: {field}"
            )
    if checked_children["team1_blue"]["input_receipts"]["patch"] != checked_children[
        "team2_blue"
    ]["input_receipts"]["patch"]:
        raise PreSideRatingEnvelopeError("conditional patch receipts differ")
    expected_summary = _conditional_summary(
        checked_children["team1_blue"], checked_children["team2_blue"]
    )
    if value.get("conditional_summary") != expected_summary:
        raise PreSideRatingEnvelopeError("conditional summary changed")
    expected_qualification = {
        "scheduled_series_start_is_public_cutoff": True,
        "both_side_conditionals_sealed_pre_event": True,
        "actual_blue_red_side_known": False,
        "side_binding_present": False,
        "embedded_child_receipts_individually_ledger_eligible": False,
        "eligible_evaluation_map": False,
        "event_outcome_present": False,
        "event_outcome_accessed": False,
    }
    if value.get("qualification") != expected_qualification:
        raise PreSideRatingEnvelopeError("pre-side qualification changed")
    implementation = value.get("implementation")
    if not isinstance(implementation, Mapping) or implementation != _source_record(root):
        raise PreSideRatingEnvelopeError("pre-side implementation changed")
    if value.get("authority") != _authority_false() or value.get(
        "claim_ceiling"
    ) != CLAIM_CEILING:
        raise PreSideRatingEnvelopeError("pre-side authority boundary changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PreSideRatingEnvelopeError(f"refusing to overwrite envelope: {path}")
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
            raise PreSideRatingEnvelopeError(
                f"refusing to overwrite envelope: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return _sha256_bytes(raw)


def _slug(value: str) -> str:
    slug = SAFE_SLUG_RE.sub("-", value.casefold()).strip("-.")
    if not slug:
        raise PreSideRatingEnvelopeError("event id cannot form a safe locator")
    return slug[:160]


def envelope_locator(payload: Mapping[str, Any]) -> str:
    event = payload.get("event") or {}
    start = _timestamp(
        event.get("scheduled_series_start_utc"), "scheduled_series_start_utc"
    )
    event_id = _nonempty(event.get("event_id"), "event.event_id")
    game_number = event.get("game_number")
    return (
        ENVELOPE_PREFIX
        / start.date().isoformat()
        / f"{_slug(event_id)}-g{game_number}.json"
    ).as_posix()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--roster-source-payload", type=Path, required=True)
    parser.add_argument("--patch-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        payload = build_pre_side_rating_envelope(
            input_raw=args.input.read_bytes(),
            roster_source_payload_raw=args.roster_source_payload.read_bytes(),
            patch_receipt_raw=args.patch_receipt.read_bytes(),
            root=root,
        )
        expected = root / envelope_locator(payload)
        if args.out.resolve(strict=False) != expected.resolve(strict=False):
            raise PreSideRatingEnvelopeError(
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
                "actual_side_known": False,
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
    "ENVELOPE_PREFIX",
    "ENVELOPE_SCHEMA_VERSION",
    "INPUT_SCHEMA_VERSION",
    "PreSideRatingEnvelopeError",
    "build_pre_side_rating_envelope",
    "envelope_locator",
    "validate_input",
    "validate_pre_side_rating_envelope",
    "write_no_clobber",
]
