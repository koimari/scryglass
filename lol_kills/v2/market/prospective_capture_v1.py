"""Operate the outcome-free phase-one ratings and terminal-Draft capture.

This module joins the already-frozen child builders into one fail-closed
workflow.  It deliberately does not fetch, infer, or repair source facts.  In
particular, schedule order is not treated as blue/red side authority and an
ambiguous roster cannot be converted into a prediction.

The three commands correspond to facts becoming available over time:

* ``prepare`` captures an exact, sourced blue/red roster, the patch receipt,
  the ratings evaluation prediction, and the immutable event plan;
* ``draft`` captures the terminal draft after all ten assignments are known;
* ``map-start`` captures an outcome-free actual-start signal, completes the
  event bundle, and publishes versioned joint and child ledger candidates.

Every write is no-clobber.  A failed command writes a separate attempt receipt
with ``eligible_evaluation_evidence=false``; failure receipts are never inputs
to any ratings, Draft, calibration, probability, odds, EV, recommendation, or
betting gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

from lol_kills import pregame_roster_capture as roster_capture
from lol_kills.v2.draft.terminal import future_prediction_ledger as draft_ledger
from lol_kills.v2.ratings.player import (
    multileague_v3_prediction_ledger as ratings_ledger,
)

from . import phase_one_collection_v1 as phase_one


ROOT = Path(__file__).resolve().parents[3]
INPUT_SCHEMA_VERSION = "scryglass:prospective-match-winner-capture-input:v1"
ATTEMPT_SCHEMA_VERSION = "scryglass:prospective-match-winner-capture-attempt:v1"
ATTEMPT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-one/capture-attempts"
)
RATINGS_LEDGER_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/multileague-v3/ledgers"
)
DRAFT_LEDGER_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/draft-terminal-v1/ledgers"
)
SOURCE_LOCATOR = "lol_kills/v2/market/prospective_capture_v1.py"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_SLUG_RE = re.compile(r"[^a-z0-9._-]+")
STAGES = ("prepare", "draft", "map-start")
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
    "stake_authority",
    "transaction_authority",
    "betting_authority",
)
CLAIM_CEILING = (
    "Operational outcome-free capture audit only. Success creates evaluation "
    "candidates and failure creates no eligible evidence. Neither state grants "
    "rating, probability, odds, EV, recommendation, stake, transaction, or "
    "betting authority."
)


class ProspectiveCaptureError(ValueError):
    """The prospective capture workflow failed closed."""


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
        raise ProspectiveCaptureError("capture value is not canonical") from exc


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
        raise ProspectiveCaptureError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProspectiveCaptureError(f"{field} must be a non-empty string")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    text = _nonempty(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProspectiveCaptureError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise ProspectiveCaptureError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime], field: str) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ProspectiveCaptureError(
            f"{field} clock must return a timezone-aware datetime"
        )
    return value.astimezone(timezone.utc)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ProspectiveCaptureError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_object(raw: bytes, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ProspectiveCaptureError(
                    f"non-finite JSON number in {field}: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProspectiveCaptureError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ProspectiveCaptureError(f"{field} must be a JSON object")
    return value


def _assert_no_outcomes(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
            if normalized in OUTCOME_KEYS:
                raise ProspectiveCaptureError(
                    f"event outcome field is forbidden: {path}.{key}"
                )
            _assert_no_outcomes(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_outcomes(item, f"{path}[{index}]")


def _source_record(root: Path) -> dict[str, Any]:
    path = root / SOURCE_LOCATOR
    if not path.is_file() or path.is_symlink():
        raise ProspectiveCaptureError("prospective capture implementation is unavailable")
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
        raise ProspectiveCaptureError("capture implementation binding changed")
    if value.get("locator") != SOURCE_LOCATOR:
        raise ProspectiveCaptureError("capture implementation locator changed")
    path = root / SOURCE_LOCATOR
    if (
        not path.is_file()
        or path.is_symlink()
        or value.get("bytes") != path.stat().st_size
        or value.get("raw_sha256") != _sha256_path(path)
    ):
        raise ProspectiveCaptureError("capture implementation bytes changed")
    return dict(value)


def _authority_false() -> dict[str, bool]:
    return {name: False for name in AUTHORITY_KEYS}


def _slug(value: str) -> str:
    result = SAFE_SLUG_RE.sub("-", value.casefold()).strip("-.")
    if not result:
        raise ProspectiveCaptureError("event_id cannot form a safe artifact name")
    return result[:160]


def _event_tail(event: Mapping[str, Any]) -> PurePosixPath:
    event_start = _timestamp(event.get("event_start_utc"), "event.event_start_utc")
    event_id = _nonempty(event.get("event_id"), "event.event_id")
    game_number = event.get("game_number")
    if (
        isinstance(game_number, bool)
        or not isinstance(game_number, int)
        or game_number < 1
    ):
        raise ProspectiveCaptureError("event.game_number must be a positive integer")
    return PurePosixPath(event_start.date().isoformat()) / (
        f"{_slug(event_id)}-g{game_number}.json"
    )


def _safe_repo_output(root: Path, locator: PurePosixPath, field: str) -> Path:
    if locator.is_absolute() or any(part in {"", ".", ".."} for part in locator.parts):
        raise ProspectiveCaptureError(f"{field} locator is invalid")
    path = root / locator
    root_real = root.resolve()
    parent_real = path.parent.resolve(strict=False)
    try:
        parent_real.relative_to(root_real)
    except ValueError as exc:
        raise ProspectiveCaptureError(f"{field} path escapes the repository") from exc
    if path.exists() or path.is_symlink():
        raise ProspectiveCaptureError(f"{field} already exists: {locator.as_posix()}")
    ancestor = path.parent
    while ancestor != root and ancestor != ancestor.parent:
        try:
            mode = ancestor.lstat().st_mode
        except FileNotFoundError:
            ancestor = ancestor.parent
            continue
        if stat.S_ISLNK(mode):
            raise ProspectiveCaptureError(f"{field} parent symlink is rejected")
        ancestor = ancestor.parent
    return path


def _safe_repo_file(
    root: Path, locator: str, prefix: PurePosixPath, field: str
) -> Path:
    relative = PurePosixPath(_nonempty(locator, field))
    if (
        relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or tuple(relative.parts[: len(prefix.parts)]) != prefix.parts
        or relative.suffix != ".json"
    ):
        raise ProspectiveCaptureError(f"{field} locator is outside its root")
    path = root / relative
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ProspectiveCaptureError(f"{field} is missing") from exc
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise ProspectiveCaptureError(f"{field} must be a regular non-symlink file")
    try:
        path.resolve(strict=True).relative_to(root.resolve())
    except ValueError as exc:
        raise ProspectiveCaptureError(f"{field} path escapes the repository") from exc
    return path


def _write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ProspectiveCaptureError(f"refusing to overwrite capture artifact: {path}")
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
            raise ProspectiveCaptureError(
                f"refusing to overwrite capture artifact: {path}"
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


def _validate_prepare_input(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProspectiveCaptureError("prepare input must be an object")
    _assert_no_outcomes(value, "prepare_input")
    if set(value) != {
        "schema_version",
        "event",
        "roster_source",
        "teams",
    }:
        raise ProspectiveCaptureError("prepare input structure changed")
    if value.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ProspectiveCaptureError("prepare input schema changed")
    event = value.get("event")
    if not isinstance(event, Mapping) or set(event) != {
        "event_id",
        "series_id",
        "game_number",
        "event_start_utc",
        "league",
    }:
        raise ProspectiveCaptureError("prepare event structure changed")
    _nonempty(event.get("event_id"), "event.event_id")
    _nonempty(event.get("series_id"), "event.series_id")
    _event_tail(event)
    league = _nonempty(event.get("league"), "event.league")
    if league != league.upper():
        raise ProspectiveCaptureError("event.league must use its canonical uppercase id")
    source = value.get("roster_source")
    if not isinstance(source, Mapping) or set(source) != {
        "source",
        "source_url",
        "source_record_id",
        "source_updated_at_utc",
        "available_at_utc",
        "rights_status",
    }:
        raise ProspectiveCaptureError("roster source structure changed")
    for field in ("source", "source_url", "source_record_id"):
        _nonempty(source.get(field), f"roster_source.{field}")
    _timestamp(
        source.get("source_updated_at_utc"),
        "roster_source.source_updated_at_utc",
    )
    _timestamp(source.get("available_at_utc"), "roster_source.available_at_utc")
    if source.get("rights_status") != "reviewed":
        raise ProspectiveCaptureError("roster source rights are not reviewed")
    teams = value.get("teams")
    if not isinstance(teams, list):
        raise ProspectiveCaptureError("teams must be an exact blue/red list")
    # The roster builder supplies the canonical role, identity, uniqueness,
    # organization, and exact-side validation.  Calling it later is deliberate:
    # no schedule-order fallback exists here.
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "event": dict(event),
        "roster_source": dict(source),
        "teams": teams,
    }


def _artifact_binding(locator: str, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "locator": locator,
        "raw_sha256": _sha256_path(path),
        "artifact_sha256": _sha(
            payload.get("artifact_sha256"), f"{locator}.artifact_sha256"
        ),
    }


def build_attempt_receipt(
    *,
    stage: str,
    status: str,
    event: Mapping[str, Any] | None,
    input_digests: Mapping[str, str],
    artifacts: Sequence[Mapping[str, Any]],
    blockers: Sequence[str],
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    recorded = _clock_sample(clock, "capture attempt")
    if stage not in STAGES:
        raise ProspectiveCaptureError("capture attempt stage is invalid")
    if status not in {"SUCCEEDED_EVALUATION_CANDIDATE_ONLY", "FAILED_CLOSED"}:
        raise ProspectiveCaptureError("capture attempt status is invalid")
    normalized_digests = {
        _nonempty(key, "input digest name"): _sha(value, f"input_digests.{key}")
        for key, value in sorted(input_digests.items())
    }
    normalized_blockers = sorted(
        {_nonempty(blocker, "blocker") for blocker in blockers}
    )
    if (status == "FAILED_CLOSED") != bool(normalized_blockers):
        raise ProspectiveCaptureError("capture attempt blocker state is inconsistent")
    normalized_event: dict[str, Any] | None = None
    if event is not None:
        normalized_event = dict(event)
        _assert_no_outcomes(normalized_event, "attempt.event")
    normalized_artifacts = [dict(item) for item in artifacts]
    payload: dict[str, Any] = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "status": status,
        "stage": stage,
        "recorded_at_utc": recorded.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": recorded.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "event": normalized_event,
        "input_digests": normalized_digests,
        "published_artifacts": normalized_artifacts,
        "blockers": normalized_blockers,
        "eligible_evaluation_evidence": status
        == "SUCCEEDED_EVALUATION_CANDIDATE_ONLY",
        "outcomes_present": False,
        "outcomes_accessed": False,
        "implementation": _source_record(root),
        "authority": _authority_false(),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_attempt_receipt(payload, root=root)


def validate_attempt_receipt(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ProspectiveCaptureError("capture attempt must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "capture_attempt")
    expected = {
        "schema_version",
        "status",
        "stage",
        "recorded_at_utc",
        "clock_attestation",
        "event",
        "input_digests",
        "published_artifacts",
        "blockers",
        "eligible_evaluation_evidence",
        "outcomes_present",
        "outcomes_accessed",
        "implementation",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }
    if set(value) != expected or value.get("schema_version") != ATTEMPT_SCHEMA_VERSION:
        raise ProspectiveCaptureError("capture attempt structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise ProspectiveCaptureError("capture attempt hash changed")
    status = value.get("status")
    if status not in {"SUCCEEDED_EVALUATION_CANDIDATE_ONLY", "FAILED_CLOSED"}:
        raise ProspectiveCaptureError("capture attempt status changed")
    if value.get("stage") not in STAGES:
        raise ProspectiveCaptureError("capture attempt stage changed")
    recorded = _timestamp(value.get("recorded_at_utc"), "recorded_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": recorded.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise ProspectiveCaptureError("capture attempt clock changed")
    digests = value.get("input_digests")
    if not isinstance(digests, Mapping) or not digests:
        raise ProspectiveCaptureError("capture attempt input digests are missing")
    for key, digest in digests.items():
        _nonempty(key, "input digest name")
        _sha(digest, f"input_digests.{key}")
    artifacts = value.get("published_artifacts")
    if not isinstance(artifacts, list):
        raise ProspectiveCaptureError("capture attempt artifact list changed")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "locator",
            "raw_sha256",
            "artifact_sha256",
        }:
            raise ProspectiveCaptureError(
                f"capture attempt artifact {index} is malformed"
            )
        _nonempty(artifact.get("locator"), f"artifact.{index}.locator")
        _sha(artifact.get("raw_sha256"), f"artifact.{index}.raw_sha256")
        _sha(
            artifact.get("artifact_sha256"),
            f"artifact.{index}.artifact_sha256",
        )
    blockers = value.get("blockers")
    if (
        not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or any(not isinstance(item, str) or not item for item in blockers)
    ):
        raise ProspectiveCaptureError("capture attempt blockers changed")
    success = status == "SUCCEEDED_EVALUATION_CANDIDATE_ONLY"
    if (
        bool(blockers) == success
        or value.get("eligible_evaluation_evidence") is not success
        or value.get("outcomes_present") is not False
        or value.get("outcomes_accessed") is not False
        or value.get("authority") != _authority_false()
        or value.get("claim_ceiling") != CLAIM_CEILING
    ):
        raise ProspectiveCaptureError("capture attempt authority boundary changed")
    _validate_source_record(value.get("implementation"), root)
    return value


def _attempt_locator(
    *, stage: str, event: Mapping[str, Any] | None, recorded: datetime, digest: str
) -> PurePosixPath:
    if event is None:
        event_slug = "unknown-event"
    else:
        event_slug = _slug(str(event.get("event_id") or "unknown-event"))
    stamp = recorded.strftime("%Y%m%dT%H%M%S.%fZ")
    return ATTEMPT_PREFIX / recorded.date().isoformat() / (
        f"{event_slug}-{stage}-{stamp}-{digest[:12]}.json"
    )


def _write_attempt(
    *,
    stage: str,
    status: str,
    event: Mapping[str, Any] | None,
    input_digests: Mapping[str, str],
    artifacts: Sequence[Mapping[str, Any]],
    blockers: Sequence[str],
    root: Path,
    recorded: datetime,
) -> tuple[str, dict[str, Any]]:
    payload = build_attempt_receipt(
        stage=stage,
        status=status,
        event=event,
        input_digests=input_digests,
        artifacts=artifacts,
        blockers=blockers,
        root=root,
        clock=lambda: recorded,
    )
    locator = _attempt_locator(
        stage=stage,
        event=event,
        recorded=recorded,
        digest=payload["artifact_sha256"],
    )
    path = _safe_repo_output(root, locator, "capture attempt")
    _write_no_clobber(path, payload)
    return locator.as_posix(), payload


def prepare_capture(
    *,
    prepare_input_raw: bytes,
    roster_source_payload_raw: bytes,
    patch_receipt_raw: bytes,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Capture exact pre-event ratings evidence and its phase-one plan."""

    captured = _clock_sample(clock, "prepare capture")
    input_digests = {
        "prepare_input_raw_sha256": _sha256_bytes(prepare_input_raw),
        "roster_source_payload_raw_sha256": _sha256_bytes(
            roster_source_payload_raw
        ),
        "patch_receipt_raw_sha256": _sha256_bytes(patch_receipt_raw),
    }
    event: Mapping[str, Any] | None = None
    published: list[dict[str, Any]] = []
    try:
        prepare_object = _validate_prepare_input(
            _strict_object(prepare_input_raw, "prepare input")
        )
        event = prepare_object["event"]
        if captured >= _timestamp(event["event_start_utc"], "event.event_start_utc"):
            raise ProspectiveCaptureError("prepare capture is not pre-event")
        source = prepare_object["roster_source"]
        roster = roster_capture.build_pregame_roster_receipt(
            raw_source_payload=roster_source_payload_raw,
            source=source["source"],
            source_url=source["source_url"],
            source_record_id=source["source_record_id"],
            source_updated_at=source["source_updated_at_utc"],
            available_at=source["available_at_utc"],
            captured_at=captured.isoformat(),
            event_id=event["event_id"],
            event_start=event["event_start_utc"],
            league=event["league"],
            teams=prepare_object["teams"],
            capture_protocol_sha256=_source_record(root)["raw_sha256"],
            rights_status=source["rights_status"],
        )
        roster_raw = (
            json.dumps(roster, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        ).encode("ascii")
        rating = ratings_ledger.build_pre_event_prediction_receipt(
            roster_receipt_raw=roster_raw,
            patch_receipt_raw=patch_receipt_raw,
            series_id=event["series_id"],
            game_number=event["game_number"],
            root=root,
            clock=lambda: captured,
        )
        tail = _event_tail(event)
        roster_locator = roster_capture.RECEIPT_PREFIX / tail
        rating_locator = ratings_ledger.RECEIPT_PREFIX / tail
        plan_locator = phase_one.PLAN_PREFIX / tail
        roster_path = _safe_repo_output(root, roster_locator, "roster receipt")
        rating_path = _safe_repo_output(root, rating_locator, "ratings receipt")
        plan_path = _safe_repo_output(root, plan_locator, "event plan")

        _write_no_clobber(roster_path, roster)
        published.append(
            {
                "locator": roster_locator.as_posix(),
                "raw_sha256": _sha256_path(roster_path),
                "artifact_sha256": roster_capture.sha256_json(roster),
            }
        )
        ratings_ledger.write_no_clobber(rating_path, rating)
        published.append(
            _artifact_binding(rating_locator.as_posix(), rating_path, rating)
        )
        plan = phase_one.build_event_plan(
            ratings_prediction_locator=rating_locator.as_posix(),
            root=root,
            clock=lambda: captured,
        )
        if plan["locators"]["plan"] != plan_locator.as_posix():
            raise ProspectiveCaptureError("event plan locator changed")
        phase_one.write_no_clobber(plan_path, plan)
        published.append(_artifact_binding(plan_locator.as_posix(), plan_path, plan))
        attempt_locator, attempt = _write_attempt(
            stage="prepare",
            status="SUCCEEDED_EVALUATION_CANDIDATE_ONLY",
            event=event,
            input_digests=input_digests,
            artifacts=published,
            blockers=[],
            root=root,
            recorded=captured,
        )
        return {
            "status": attempt["status"],
            "attempt_locator": attempt_locator,
            "attempt_artifact_sha256": attempt["artifact_sha256"],
            "event": dict(event),
            "locators": plan["locators"],
            "eligible_evaluation_evidence": True,
            "betting_authority": False,
        }
    except Exception as exc:
        blocker = f"{type(exc).__name__}:{str(exc)}"
        attempt_locator, attempt = _write_attempt(
            stage="prepare",
            status="FAILED_CLOSED",
            event=event,
            input_digests=input_digests,
            artifacts=published,
            blockers=[blocker],
            root=root,
            recorded=captured,
        )
        return {
            "status": attempt["status"],
            "attempt_locator": attempt_locator,
            "attempt_artifact_sha256": attempt["artifact_sha256"],
            "event": dict(event) if event is not None else None,
            "blockers": attempt["blockers"],
            "eligible_evaluation_evidence": False,
            "betting_authority": False,
        }


def _load_plan(root: Path, locator: str) -> tuple[Path, dict[str, Any]]:
    path = _safe_repo_file(root, locator, phase_one.PLAN_PREFIX, "event plan")
    raw = path.read_bytes()
    plan = phase_one.validate_event_plan(
        _strict_object(raw, "event plan"), root=root
    )
    if plan["locators"]["plan"] != locator:
        raise ProspectiveCaptureError("event plan was loaded from the wrong locator")
    return path, plan


def capture_terminal_draft(
    *,
    plan_locator: str,
    draft_metadata_raw: bytes,
    draft_source_payload_raw: bytes,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Capture the terminal draft bound to an existing pre-event plan."""

    captured = _clock_sample(clock, "terminal draft capture")
    inputs = {
        "plan_locator_sha256": _sha256_bytes(plan_locator.encode("utf-8")),
        "draft_metadata_raw_sha256": _sha256_bytes(draft_metadata_raw),
        "draft_source_payload_raw_sha256": _sha256_bytes(draft_source_payload_raw),
    }
    event: Mapping[str, Any] | None = None
    published: list[dict[str, Any]] = []
    try:
        _plan_path, plan = _load_plan(root, plan_locator)
        event = plan["event"]
        ratings_path = _safe_repo_file(
            root,
            plan["locators"]["ratings_prediction"],
            ratings_ledger.RECEIPT_PREFIX,
            "ratings prediction",
        )
        draft_locator = plan["locators"]["draft_prediction"]
        draft_path = _safe_repo_output(
            root, PurePosixPath(draft_locator), "draft prediction"
        )
        draft = draft_ledger.build_draft_prediction_receipt(
            ratings_receipt_raw=ratings_path.read_bytes(),
            draft_metadata_raw=draft_metadata_raw,
            draft_source_payload_raw=draft_source_payload_raw,
            root=root,
            clock=lambda: captured,
        )
        draft_ledger.write_no_clobber(draft_path, draft)
        published.append(_artifact_binding(draft_locator, draft_path, draft))
        attempt_locator, attempt = _write_attempt(
            stage="draft",
            status="SUCCEEDED_EVALUATION_CANDIDATE_ONLY",
            event=event,
            input_digests=inputs,
            artifacts=published,
            blockers=[],
            root=root,
            recorded=captured,
        )
        return {
            "status": attempt["status"],
            "attempt_locator": attempt_locator,
            "attempt_artifact_sha256": attempt["artifact_sha256"],
            "draft_prediction_locator": draft_locator,
            "draft_prediction_artifact_sha256": draft["artifact_sha256"],
            "eligible_evaluation_evidence": True,
            "betting_authority": False,
        }
    except Exception as exc:
        attempt_locator, attempt = _write_attempt(
            stage="draft",
            status="FAILED_CLOSED",
            event=event,
            input_digests=inputs,
            artifacts=published,
            blockers=[f"{type(exc).__name__}:{str(exc)}"],
            root=root,
            recorded=captured,
        )
        return {
            "status": attempt["status"],
            "attempt_locator": attempt_locator,
            "attempt_artifact_sha256": attempt["artifact_sha256"],
            "blockers": attempt["blockers"],
            "eligible_evaluation_evidence": False,
            "betting_authority": False,
        }


def _bundle_locators(root: Path) -> list[str]:
    directory = root / phase_one.BUNDLE_PREFIX
    if not directory.exists():
        return []
    locators: list[str] = []
    for path in sorted(directory.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ProspectiveCaptureError("phase-one bundle inventory contains a symlink")
        relative = path.relative_to(root).as_posix()
        phase_one.validate_event_bundle(
            _strict_object(path.read_bytes(), "phase-one event bundle"), root=root
        )
        locators.append(relative)
    return locators


def _version_tail(recorded: datetime, event: Mapping[str, Any]) -> PurePosixPath:
    return PurePosixPath(recorded.date().isoformat()) / (
        f"{recorded.strftime('%Y%m%dT%H%M%S.%fZ')}-{_slug(str(event['event_id']))}.json"
    )


def capture_map_start_and_publish_snapshot(
    *,
    plan_locator: str,
    map_start_metadata_raw: bytes,
    map_start_source_payload_raw: bytes,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Capture actual start, complete the event, and publish versioned ledgers."""

    captured = _clock_sample(clock, "map-start capture")
    inputs = {
        "plan_locator_sha256": _sha256_bytes(plan_locator.encode("utf-8")),
        "map_start_metadata_raw_sha256": _sha256_bytes(map_start_metadata_raw),
        "map_start_source_payload_raw_sha256": _sha256_bytes(
            map_start_source_payload_raw
        ),
    }
    event: Mapping[str, Any] | None = None
    published: list[dict[str, Any]] = []
    try:
        _plan_path, plan = _load_plan(root, plan_locator)
        event = plan["event"]
        # Refuse to complete an event without the exact bound draft receipt.
        _safe_repo_file(
            root,
            plan["locators"]["draft_prediction"],
            draft_ledger.PREDICTION_PREFIX,
            "draft prediction",
        )
        start_locator = plan["locators"]["map_start"]
        bundle_locator = plan["locators"]["event_bundle"]
        start_path = _safe_repo_output(
            root, PurePosixPath(start_locator), "map-start receipt"
        )
        bundle_path = _safe_repo_output(
            root, PurePosixPath(bundle_locator), "event bundle"
        )
        start = draft_ledger.build_map_start_receipt(
            map_start_metadata_raw=map_start_metadata_raw,
            map_start_source_payload_raw=map_start_source_payload_raw,
            root=root,
            clock=lambda: captured,
        )
        draft_ledger.write_no_clobber(start_path, start)
        published.append(_artifact_binding(start_locator, start_path, start))
        bundle = phase_one.build_event_bundle(
            plan_locator=plan_locator,
            root=root,
            clock=lambda: captured,
        )
        phase_one.write_no_clobber(bundle_path, bundle)
        published.append(_artifact_binding(bundle_locator, bundle_path, bundle))

        bundle_locators = _bundle_locators(root)
        version_tail = _version_tail(captured, event)
        snapshot_locator = (phase_one.SNAPSHOT_PREFIX / version_tail).as_posix()
        snapshot_path = _safe_repo_output(
            root, PurePosixPath(snapshot_locator), "joint ledger snapshot"
        )
        ratings_ledger_locator = (RATINGS_LEDGER_PREFIX / version_tail).as_posix()
        ratings_ledger_path = _safe_repo_output(
            root, PurePosixPath(ratings_ledger_locator), "ratings ledger candidate"
        )
        draft_ledger_locator = (DRAFT_LEDGER_PREFIX / version_tail).as_posix()
        draft_ledger_path = _safe_repo_output(
            root, PurePosixPath(draft_ledger_locator), "draft ledger candidate"
        )
        snapshot = phase_one.build_joint_ledger_snapshot(
            bundle_locators=bundle_locators,
            snapshot_locator=snapshot_locator,
            root=root,
            clock=lambda: captured,
        )
        phase_one.write_no_clobber(snapshot_path, snapshot)
        published.append(
            _artifact_binding(snapshot_locator, snapshot_path, snapshot)
        )
        rating_candidate = snapshot["ratings_ledger_candidate"]
        ratings_ledger.write_no_clobber(ratings_ledger_path, rating_candidate)
        published.append(
            _artifact_binding(
                ratings_ledger_locator, ratings_ledger_path, rating_candidate
            )
        )
        draft_candidate = snapshot["draft_ledger_candidate"]
        draft_ledger.write_no_clobber(draft_ledger_path, draft_candidate)
        published.append(
            _artifact_binding(draft_ledger_locator, draft_ledger_path, draft_candidate)
        )
        attempt_locator, attempt = _write_attempt(
            stage="map-start",
            status="SUCCEEDED_EVALUATION_CANDIDATE_ONLY",
            event=event,
            input_digests=inputs,
            artifacts=published,
            blockers=[],
            root=root,
            recorded=captured,
        )
        return {
            "status": attempt["status"],
            "attempt_locator": attempt_locator,
            "attempt_artifact_sha256": attempt["artifact_sha256"],
            "event_bundle_locator": bundle_locator,
            "joint_snapshot_locator": snapshot_locator,
            "ratings_ledger_candidate_locator": ratings_ledger_locator,
            "draft_ledger_candidate_locator": draft_ledger_locator,
            "joint_metadata_support_met": snapshot["support"][
                "joint_metadata_support_met"
            ],
            "eligible_evaluation_evidence": True,
            "betting_authority": False,
        }
    except Exception as exc:
        attempt_locator, attempt = _write_attempt(
            stage="map-start",
            status="FAILED_CLOSED",
            event=event,
            input_digests=inputs,
            artifacts=published,
            blockers=[f"{type(exc).__name__}:{str(exc)}"],
            root=root,
            recorded=captured,
        )
        return {
            "status": attempt["status"],
            "attempt_locator": attempt_locator,
            "attempt_artifact_sha256": attempt["artifact_sha256"],
            "blockers": attempt["blockers"],
            "eligible_evaluation_evidence": False,
            "betting_authority": False,
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--roster-source-payload", type=Path, required=True)
    prepare.add_argument("--patch-receipt", type=Path, required=True)
    draft = subparsers.add_parser("draft")
    draft.add_argument("--plan", required=True)
    draft.add_argument("--metadata", type=Path, required=True)
    draft.add_argument("--source-payload", type=Path, required=True)
    start = subparsers.add_parser("map-start")
    start.add_argument("--plan", required=True)
    start.add_argument("--metadata", type=Path, required=True)
    start.add_argument("--source-payload", type=Path, required=True)
    validate = subparsers.add_parser("validate-attempt")
    validate.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "prepare":
            result = prepare_capture(
                prepare_input_raw=args.input.read_bytes(),
                roster_source_payload_raw=args.roster_source_payload.read_bytes(),
                patch_receipt_raw=args.patch_receipt.read_bytes(),
                root=root,
            )
        elif args.command == "draft":
            result = capture_terminal_draft(
                plan_locator=args.plan,
                draft_metadata_raw=args.metadata.read_bytes(),
                draft_source_payload_raw=args.source_payload.read_bytes(),
                root=root,
            )
        elif args.command == "map-start":
            result = capture_map_start_and_publish_snapshot(
                plan_locator=args.plan,
                map_start_metadata_raw=args.metadata.read_bytes(),
                map_start_source_payload_raw=args.source_payload.read_bytes(),
                root=root,
            )
        else:
            payload = _strict_object(
                args.artifact.read_bytes(), "capture attempt"
            )
            checked = validate_attempt_receipt(payload, root=root)
            result = {
                "valid": True,
                "status": checked["status"],
                "stage": checked["stage"],
                "artifact_sha256": checked["artifact_sha256"],
                "eligible_evaluation_evidence": checked[
                    "eligible_evaluation_evidence"
                ],
                "betting_authority": False,
            }
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc))
        return 2
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result.get("status") != "FAILED_CLOSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_SCHEMA_VERSION",
    "INPUT_SCHEMA_VERSION",
    "ProspectiveCaptureError",
    "build_attempt_receipt",
    "capture_map_start_and_publish_snapshot",
    "capture_terminal_draft",
    "prepare_capture",
    "validate_attempt_receipt",
]
