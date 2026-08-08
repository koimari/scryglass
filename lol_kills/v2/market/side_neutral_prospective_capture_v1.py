"""Operate reviewed side-neutral prospective ratings and Draft capture.

The underlying builders deliberately remain independently usable and
non-authorizing.  This operator adds the operational guard required for the
real prospective cohort: an externally hash-pinned independent review must be
active before the first pre-side envelope and before every later stage.

No command opens outcomes or grants rating, probability, odds, expected-value,
recommendation, stake, transaction, or betting authority.  Every artifact is
written to the canonical immutable locator selected by its frozen builder.
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

from lol_kills.v2.draft.terminal import future_prediction_ledger as draft_ledger
from lol_kills.v2.draft.terminal import side_neutral_prediction_v1 as neutral_draft
from lol_kills.v2.ratings.player import pre_side_rating_binding_v1 as side_binding
from lol_kills.v2.ratings.player import pre_side_rating_envelope_v1 as pre_side
from lol_kills.v2.ratings.player import (
    multileague_v3_prediction_ledger as ratings_ledger,
)
from lol_kills.v2.ratings.player.multileague_v3_side_neutral_protocol_v2 import (
    INDEPENDENT_REVIEW_ENV,
)
from lol_kills.v2.ratings.player.side_neutral_protocol_review_v1 import (
    SideNeutralProtocolReviewError,
    load_active_side_neutral_protocol_review,
)

from . import side_neutral_capture_bundle_v1 as bundle_module
from . import side_neutral_ledger_v1 as ledger_module
from . import phase_one_collection_v1 as phase_one


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/side_neutral_prospective_capture_v1.py"
SAFE_SLUG_RE = re.compile(r"[^a-z0-9._-]+")
STAGES = ("pre-side", "bind-side", "draft", "map-start", "ledger")
PHASE_ONE_BRIDGE_STAGES = ("draft", "map-start", "ledger")
RATINGS_LEDGER_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/multileague-v3/ledgers"
)
DRAFT_LEDGER_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/draft-terminal-v1/ledgers"
)


class SideNeutralProspectiveCaptureError(ValueError):
    """A reviewed side-neutral prospective operation failed closed."""


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SideNeutralProspectiveCaptureError(
            "operator clock must return a timezone-aware datetime"
        )
    return value.astimezone(timezone.utc)


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SideNeutralProspectiveCaptureError(f"{field} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SideNeutralProspectiveCaptureError(
            f"{field} must be RFC-3339"
        ) from exc
    if parsed.tzinfo is None:
        raise SideNeutralProspectiveCaptureError(
            f"{field} must include a timezone"
        )
    return parsed.astimezone(timezone.utc)


def _strict_object(path: Path, field: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise SideNeutralProspectiveCaptureError(
                    f"{field} contains duplicate key: {key}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SideNeutralProspectiveCaptureError(
                    f"{field} contains non-finite number: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SideNeutralProspectiveCaptureError(
            f"{field} must be strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise SideNeutralProspectiveCaptureError(f"{field} must be an object")
    return value


def _artifact_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise SideNeutralProspectiveCaptureError(
            "capture artifact is not canonical JSON"
        ) from exc


def _strict_raw_object(raw: bytes, field: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise SideNeutralProspectiveCaptureError(
                    f"{field} contains duplicate key: {key}"
                )
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SideNeutralProspectiveCaptureError(
                    f"{field} contains non-finite number: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SideNeutralProspectiveCaptureError(
            f"{field} must be strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise SideNeutralProspectiveCaptureError(f"{field} must be an object")
    return value


def _embedded_raw(value: Any, field: str) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "raw_sha256",
        "raw_base64",
        "value",
    }:
        raise SideNeutralProspectiveCaptureError(f"{field} embedding changed")
    try:
        raw = base64.b64decode(str(value.get("raw_base64")), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise SideNeutralProspectiveCaptureError(
            f"{field} base64 is invalid"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != value.get("raw_sha256"):
        raise SideNeutralProspectiveCaptureError(f"{field} raw hash changed")
    parsed = _strict_raw_object(raw, field)
    if parsed != value.get("value"):
        raise SideNeutralProspectiveCaptureError(f"{field} value changed")
    return raw, parsed


def _event_tail(event: Mapping[str, Any]) -> PurePosixPath:
    event_start = _timestamp(
        event.get("event_start_utc"), "event.event_start_utc"
    )
    event_id = str(event.get("event_id") or "")
    game_number = event.get("game_number")
    if not event_id or isinstance(game_number, bool) or not isinstance(
        game_number, int
    ) or game_number < 1:
        raise SideNeutralProspectiveCaptureError(
            "event identity cannot form a phase-one locator"
        )
    return PurePosixPath(event_start.date().isoformat()) / (
        f"{_slug(event_id)}-g{game_number}.json"
    )


def _atomic_no_clobber_batch(
    artifacts: Sequence[tuple[Path, Mapping[str, Any] | bytes]],
) -> dict[Path, str]:
    """Publish a small artifact batch or leave none of its final paths."""

    paths = [path for path, _payload in artifacts]
    if len(set(paths)) != len(paths):
        raise SideNeutralProspectiveCaptureError(
            "capture batch repeats an output path"
        )
    staged: list[tuple[Path, Path, bytes]] = []
    linked: list[Path] = []
    try:
        for path, payload in artifacts:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() or path.is_symlink():
                raise SideNeutralProspectiveCaptureError(
                    f"refusing to overwrite capture artifact: {path}"
                )
            raw = payload if isinstance(payload, bytes) else _artifact_bytes(payload)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".partial", dir=path.parent
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((temporary, path, raw))
        for temporary, path, _raw in staged:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise SideNeutralProspectiveCaptureError(
                    f"refusing to overwrite capture artifact: {path}"
                ) from exc
            linked.append(path)
        for parent in {path.parent for path in linked}:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return {
            path: hashlib.sha256(raw).hexdigest()
            for _temporary, path, raw in staged
        }
    except Exception:
        for path in reversed(linked):
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _path, _raw in staged:
            temporary.unlink(missing_ok=True)


def _active_review(
    *,
    root: Path,
    environment: Mapping[str, str],
    observed: datetime,
) -> dict[str, Any]:
    try:
        review = load_active_side_neutral_protocol_review(
            root=root,
            environment=environment,
            as_of=observed,
        )
    except (OSError, ValueError, SideNeutralProtocolReviewError) as exc:
        raise SideNeutralProspectiveCaptureError(
            "externally pinned independent side-neutral review is required "
            f"before prospective capture: {exc}"
        ) from exc
    effective = _timestamp(
        review["authorization"]["effective_at_utc"], "review effective time"
    )
    if observed <= effective:
        raise SideNeutralProspectiveCaptureError(
            "prospective capture must occur after review effective time"
        )
    return review


def _require_capture_after_review(
    payload: Mapping[str, Any], review: Mapping[str, Any]
) -> None:
    captured = _timestamp(payload.get("captured_at_utc"), "pre-side capture")
    effective = _timestamp(
        review["authorization"]["effective_at_utc"], "review effective time"
    )
    if captured <= effective:
        raise SideNeutralProspectiveCaptureError(
            "pre-side capture does not follow independent review effective time"
        )


def _result(
    *,
    stage: str,
    output: Path,
    raw_sha256: str,
    artifact_sha256: str,
    review: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "status": "SUCCEEDED_OUTCOME_FREE_EVALUATION_CANDIDATE_ONLY",
        "stage": stage,
        "output": str(output),
        "raw_sha256": raw_sha256,
        "artifact_sha256": artifact_sha256,
        "independent_review_id": review["review_id"],
        "independent_review_effective_at_utc": review["authorization"][
            "effective_at_utc"
        ],
        "outcomes_accessed": False,
        "probability_authority": False,
        "betting_authority": False,
    }


def capture_pre_side(
    *,
    input_raw: bytes,
    roster_source_payload_raw: bytes,
    patch_receipt_raw: bytes,
    root: Path = ROOT,
    environment: Mapping[str, str],
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    observed = _clock_sample(clock)
    review = _active_review(
        root=root, environment=environment, observed=observed
    )
    payload = pre_side.build_pre_side_rating_envelope(
        input_raw=input_raw,
        roster_source_payload_raw=roster_source_payload_raw,
        patch_receipt_raw=patch_receipt_raw,
        root=root,
        clock=lambda: observed,
    )
    _require_capture_after_review(payload, review)
    output = root / pre_side.envelope_locator(payload)
    raw_sha256 = pre_side.write_no_clobber(output, payload)
    return _result(
        stage="pre-side",
        output=output,
        raw_sha256=raw_sha256,
        artifact_sha256=payload["artifact_sha256"],
        review=review,
    )


def capture_side_binding(
    *,
    envelope_raw: bytes,
    binding_input_raw: bytes,
    public_side_source_raw: bytes,
    root: Path = ROOT,
    environment: Mapping[str, str],
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    observed = _clock_sample(clock)
    review = _active_review(
        root=root, environment=environment, observed=observed
    )
    envelope_object = json.loads(envelope_raw)
    checked_envelope = pre_side.validate_pre_side_rating_envelope(
        envelope_object, root=root
    )
    _require_capture_after_review(checked_envelope, review)
    payload = side_binding.build_pre_side_rating_binding(
        envelope_raw=envelope_raw,
        binding_input_raw=binding_input_raw,
        public_side_source_raw=public_side_source_raw,
        root=root,
        clock=lambda: observed,
    )
    output = root / side_binding.binding_locator(payload)
    raw_sha256 = side_binding.write_no_clobber(output, payload)
    return _result(
        stage="bind-side",
        output=output,
        raw_sha256=raw_sha256,
        artifact_sha256=payload["artifact_sha256"],
        review=review,
    )


def capture_terminal_draft(
    *,
    side_binding_raw: bytes,
    draft_metadata_raw: bytes,
    draft_source_payload_raw: bytes,
    root: Path = ROOT,
    environment: Mapping[str, str],
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    observed = _clock_sample(clock)
    review = _active_review(
        root=root, environment=environment, observed=observed
    )
    binding_object = json.loads(side_binding_raw)
    checked_binding = side_binding.validate_pre_side_rating_binding(
        binding_object, root=root
    )
    envelope_object = checked_binding["pre_side_envelope"]["value"]
    _require_capture_after_review(envelope_object, review)
    payload = neutral_draft.build_side_neutral_draft_prediction(
        side_binding_raw=side_binding_raw,
        draft_metadata_raw=draft_metadata_raw,
        draft_source_payload_raw=draft_source_payload_raw,
        root=root,
        clock=lambda: observed,
    )
    child_raw, child = _embedded_raw(
        payload.get("terminal_draft_prediction"), "terminal Draft child"
    )
    rating_raw, rating = _embedded_raw(
        (child.get("input_receipts") or {}).get("ratings_prediction"),
        "selected ratings child",
    )
    checked_rating = ratings_ledger.validate_pre_event_prediction_receipt(
        rating, root=root
    )
    checked_child = draft_ledger.validate_draft_prediction_receipt(
        child, root=root
    )
    if checked_rating["artifact_sha256"] != payload["selected_rating_binding"][
        "rating_receipt_artifact_sha256"
    ]:
        raise SideNeutralProspectiveCaptureError(
            "selected ratings child does not match side binding"
        )
    tail = _event_tail(checked_child["event"])
    rating_output = root / ratings_ledger.RECEIPT_PREFIX / tail
    child_output = root / draft_ledger.PREDICTION_PREFIX / tail
    plan_output = root / phase_one.PLAN_PREFIX / tail
    output = root / neutral_draft.prediction_locator(payload)
    for path in (rating_output, child_output, plan_output, output):
        if path.exists() or path.is_symlink():
            raise SideNeutralProspectiveCaptureError(
                f"refusing to overwrite Draft bridge artifact: {path}"
            )

    # The frozen plan builder intentionally reloads the exact ratings bytes from
    # their canonical path. Expose those bytes only while constructing and
    # validating the plan, then publish the complete four-file batch together.
    _atomic_no_clobber_batch([(rating_output, rating_raw)])
    try:
        plan = phase_one.build_event_plan(
            ratings_prediction_locator=(
                ratings_ledger.RECEIPT_PREFIX / tail
            ).as_posix(),
            root=root,
            clock=lambda: observed,
        )
    finally:
        rating_output.unlink(missing_ok=True)
    expected_plan_locator = (phase_one.PLAN_PREFIX / tail).as_posix()
    if plan["locators"]["plan"] != expected_plan_locator:
        raise SideNeutralProspectiveCaptureError(
            "phase-one plan locator differs from selected ratings receipt"
        )
    raw_hashes = _atomic_no_clobber_batch(
        [
            (rating_output, rating_raw),
            (child_output, child_raw),
            (plan_output, plan),
            (output, payload),
        ]
    )
    raw_sha256 = raw_hashes[output]
    return _result(
        stage="draft",
        output=output,
        raw_sha256=raw_sha256,
        artifact_sha256=payload["artifact_sha256"],
        review=review,
    ) | {
        "phase_one_bridge": {
            "ratings_prediction": str(rating_output),
            "ratings_prediction_raw_sha256": raw_hashes[rating_output],
            "draft_prediction": str(child_output),
            "draft_prediction_raw_sha256": raw_hashes[child_output],
            "event_plan": str(plan_output),
            "event_plan_raw_sha256": raw_hashes[plan_output],
            "outcomes_accessed": False,
            "opening_authority": False,
        }
    }


def _slug(value: str) -> str:
    slug = SAFE_SLUG_RE.sub("-", value.casefold()).strip("-.")
    if not slug:
        raise SideNeutralProspectiveCaptureError(
            "event id cannot form a safe map-start locator"
        )
    return slug[:160]


def _map_start_locator(payload: Mapping[str, Any]) -> str:
    event = payload.get("event") or {}
    event_id = str(event.get("event_id") or "")
    game_number = event.get("game_number")
    actual_start = _timestamp(
        event.get("actual_map_start_utc"), "actual map start"
    )
    if isinstance(game_number, bool) or not isinstance(game_number, int):
        raise SideNeutralProspectiveCaptureError("map-start game number is invalid")
    return (
        draft_ledger.MAP_START_PREFIX
        / actual_start.date().isoformat()
        / f"{_slug(event_id)}-g{game_number}.json"
    ).as_posix()


def capture_map_start_bundle(
    *,
    side_neutral_draft_raw: bytes,
    map_start_metadata_raw: bytes,
    map_start_source_payload_raw: bytes,
    root: Path = ROOT,
    environment: Mapping[str, str],
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    observed = _clock_sample(clock)
    review = _active_review(
        root=root, environment=environment, observed=observed
    )
    draft_object = json.loads(side_neutral_draft_raw)
    checked_draft = neutral_draft.validate_side_neutral_draft_prediction(
        draft_object, root=root
    )
    envelope_object = checked_draft["side_binding"]["value"][
        "pre_side_envelope"
    ]["value"]
    _require_capture_after_review(envelope_object, review)
    map_start = draft_ledger.build_map_start_receipt(
        map_start_metadata_raw=map_start_metadata_raw,
        map_start_source_payload_raw=map_start_source_payload_raw,
        root=root,
        clock=lambda: observed,
    )
    tail = _event_tail(checked_draft["event"])
    plan_locator = (phase_one.PLAN_PREFIX / tail).as_posix()
    plan_path = root / plan_locator
    if not plan_path.is_file() or plan_path.is_symlink():
        raise SideNeutralProspectiveCaptureError(
            "prospective phase-one event plan is missing"
        )
    plan = phase_one.validate_event_plan(
        _strict_raw_object(plan_path.read_bytes(), "phase-one event plan"),
        root=root,
    )
    if plan["locators"]["plan"] != plan_locator:
        raise SideNeutralProspectiveCaptureError(
            "phase-one event plan was loaded from the wrong locator"
        )
    map_start_output = root / plan["locators"]["map_start"]
    map_start_raw = _artifact_bytes(map_start)
    bundle = bundle_module.build_side_neutral_capture_bundle(
        side_neutral_draft_raw=side_neutral_draft_raw,
        map_start_receipt_raw=map_start_raw,
        root=root,
    )
    bundle_output = root / bundle_module.bundle_locator(bundle)
    phase_bundle_output = root / plan["locators"]["event_bundle"]
    for path in (map_start_output, bundle_output, phase_bundle_output):
        if path.exists() or path.is_symlink():
            raise SideNeutralProspectiveCaptureError(
                f"refusing to overwrite map-start bridge artifact: {path}"
            )

    # The frozen bundle builder reloads the map-start receipt from its canonical
    # path. Remove the temporary publication before atomically linking the
    # complete three-file batch.
    _atomic_no_clobber_batch([(map_start_output, map_start_raw)])
    try:
        phase_bundle = phase_one.build_event_bundle(
            plan_locator=plan_locator,
            root=root,
            clock=lambda: observed,
        )
    finally:
        map_start_output.unlink(missing_ok=True)
    raw_hashes = _atomic_no_clobber_batch(
        [
            (map_start_output, map_start_raw),
            (bundle_output, bundle),
            (phase_bundle_output, phase_bundle),
        ]
    )
    map_start_raw_sha256 = raw_hashes[map_start_output]
    bundle_raw_sha256 = raw_hashes[bundle_output]
    result = _result(
        stage="map-start",
        output=bundle_output,
        raw_sha256=bundle_raw_sha256,
        artifact_sha256=bundle["artifact_sha256"],
        review=review,
    )
    result["map_start"] = {
        "output": str(map_start_output),
        "raw_sha256": map_start_raw_sha256,
        "artifact_sha256": map_start["artifact_sha256"],
    }
    result["phase_one_bridge"] = {
        "event_plan": str(plan_path),
        "event_bundle": str(phase_bundle_output),
        "event_bundle_raw_sha256": raw_hashes[phase_bundle_output],
        "event_bundle_artifact_sha256": phase_bundle["artifact_sha256"],
        "outcomes_accessed": False,
        "opening_authority": False,
    }
    return result


def publish_ledger(
    *,
    bundle_locators: Sequence[str],
    root: Path = ROOT,
    environment: Mapping[str, str],
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    observed = _clock_sample(clock)
    review = _active_review(
        root=root, environment=environment, observed=observed
    )
    payload = ledger_module.build_side_neutral_ledger(
        bundle_locators=bundle_locators,
        environment=environment,
        root=root,
        clock=lambda: observed,
    )
    output = root / ledger_module.DEFAULT_LEDGER
    phase_bundle_locators: list[str] = []
    for entry in payload["entries"]:
        source_path = root / entry["bundle_locator"]
        source_bundle = bundle_module.validate_side_neutral_capture_bundle(
            _strict_raw_object(
                source_path.read_bytes(), "reviewed side-neutral bundle"
            ),
            root=root,
        )
        selected_child = source_bundle["input_receipts"]["side_neutral_draft"][
            "value"
        ]["terminal_draft_prediction"]["value"]
        tail = _event_tail(selected_child["event"])
        phase_bundle_locators.append(
            (phase_one.BUNDLE_PREFIX / tail).as_posix()
        )

    version_name = (
        f"{observed.strftime('%Y%m%dT%H%M%S.%fZ')}-"
        f"side-neutral-{payload['artifact_sha256'][:16]}.json"
    )
    version_tail = PurePosixPath(observed.date().isoformat()) / version_name
    snapshot_locator = (phase_one.SNAPSHOT_PREFIX / version_tail).as_posix()
    snapshot = phase_one.build_joint_ledger_snapshot(
        bundle_locators=phase_bundle_locators,
        snapshot_locator=snapshot_locator,
        root=root,
        clock=lambda: observed,
    )
    snapshot_output = root / snapshot_locator
    ratings_output = root / RATINGS_LEDGER_PREFIX / version_tail
    draft_output = root / DRAFT_LEDGER_PREFIX / version_tail
    raw_hashes = _atomic_no_clobber_batch(
        [
            (output, payload),
            (snapshot_output, snapshot),
            (ratings_output, snapshot["ratings_ledger_candidate"]),
            (draft_output, snapshot["draft_ledger_candidate"]),
        ]
    )
    raw_sha256 = raw_hashes[output]
    return _result(
        stage="ledger",
        output=output,
        raw_sha256=raw_sha256,
        artifact_sha256=payload["artifact_sha256"],
        review=review,
    ) | {
        "eligible_map_count": payload["qualification"]["eligible_map_count"],
        "support_met": payload["support"]["support_met"],
        "phase_one_bridge": {
            "joint_snapshot": str(snapshot_output),
            "joint_snapshot_raw_sha256": raw_hashes[snapshot_output],
            "joint_snapshot_artifact_sha256": snapshot["artifact_sha256"],
            "ratings_ledger_candidate": str(ratings_output),
            "draft_ledger_candidate": str(draft_output),
            "joint_metadata_support_met": snapshot["support"][
                "joint_metadata_support_met"
            ],
            "outcomes_accessed": False,
            "opening_authority": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    subparsers = parser.add_subparsers(dest="command", required=True)
    first = subparsers.add_parser("pre-side")
    first.add_argument("--input", type=Path, required=True)
    first.add_argument("--roster-source-payload", type=Path, required=True)
    first.add_argument("--patch-receipt", type=Path, required=True)
    bind = subparsers.add_parser("bind-side")
    bind.add_argument("--envelope", type=Path, required=True)
    bind.add_argument("--input", type=Path, required=True)
    bind.add_argument("--public-side-source", type=Path, required=True)
    draft = subparsers.add_parser("draft")
    draft.add_argument("--side-binding", type=Path, required=True)
    draft.add_argument("--metadata", type=Path, required=True)
    draft.add_argument("--draft-source", type=Path, required=True)
    start = subparsers.add_parser("map-start")
    start.add_argument("--side-neutral-draft", type=Path, required=True)
    start.add_argument("--metadata", type=Path, required=True)
    start.add_argument("--source-payload", type=Path, required=True)
    ledger = subparsers.add_parser("ledger")
    ledger.add_argument("--bundle", action="append", required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    environment = dict(os.environ)
    try:
        if args.command == "pre-side":
            result = capture_pre_side(
                input_raw=args.input.read_bytes(),
                roster_source_payload_raw=args.roster_source_payload.read_bytes(),
                patch_receipt_raw=args.patch_receipt.read_bytes(),
                root=root,
                environment=environment,
            )
        elif args.command == "bind-side":
            result = capture_side_binding(
                envelope_raw=args.envelope.read_bytes(),
                binding_input_raw=args.input.read_bytes(),
                public_side_source_raw=args.public_side_source.read_bytes(),
                root=root,
                environment=environment,
            )
        elif args.command == "draft":
            result = capture_terminal_draft(
                side_binding_raw=args.side_binding.read_bytes(),
                draft_metadata_raw=args.metadata.read_bytes(),
                draft_source_payload_raw=args.draft_source.read_bytes(),
                root=root,
                environment=environment,
            )
        elif args.command == "map-start":
            result = capture_map_start_bundle(
                side_neutral_draft_raw=args.side_neutral_draft.read_bytes(),
                map_start_metadata_raw=args.metadata.read_bytes(),
                map_start_source_payload_raw=args.source_payload.read_bytes(),
                root=root,
                environment=environment,
            )
        else:
            result = publish_ledger(
                bundle_locators=args.bundle,
                root=root,
                environment=environment,
            )
    except (OSError, ValueError, RuntimeError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED_CLOSED",
                    "stage": args.command,
                    "error": f"{type(exc).__name__}:{exc}",
                    "outcomes_accessed": False,
                    "probability_authority": False,
                    "betting_authority": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INDEPENDENT_REVIEW_ENV",
    "PHASE_ONE_BRIDGE_STAGES",
    "SOURCE_LOCATOR",
    "STAGES",
    "SideNeutralProspectiveCaptureError",
    "capture_map_start_bundle",
    "capture_pre_side",
    "capture_side_binding",
    "capture_terminal_draft",
    "publish_ledger",
]
