"""Lock a genuinely future, prediction-ledger ratings evaluation protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd

from . import multileague_v2_protocol_equal_series as v2_protocol
from . import multileague_v2_runner_equal_series as v2_runner
from .multileague_source_snapshot import (
    CURRENT_MANIFEST_LOCATOR,
    CURRENT_MANIFEST_RAW_SHA256,
    validate_current_source_snapshot,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:multileague-rating-v3-future-protocol-lock:v1"
RESULT_STATE = "FUTURE_HOLDOUT_PROTOCOL_LOCKED_EMPTY"
SOURCE_LOCATOR = (
    "lol_kills/v2/ratings/player/multileague_v3_future_protocol.py"
)
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v3/future-protocol-lock-v1.json"
)
FUTURE_SEALED_START = datetime.fromisoformat("2026-08-03T00:00:00")
SELECTED_CANDIDATE_ID = "hierarchical-orgw100-orgv025-retain100"
DOMESTIC_LEAGUES = ("LCS", "LEC", "LCK", "LPL")
SOURCE_LOCKS = (
    SOURCE_LOCATOR,
    "lol_kills/v2/ratings/player/multileague_source_snapshot.py",
    "lol_kills/v2/ratings/player/multileague_v2_runner.py",
    "lol_kills/v2/ratings/player/multileague_runner.py",
    "lol_kills/v2/ratings/player/multileague_benchmark.py",
    "data/lol/v2/models/player/multileague-v2/protocol-lock-v2.json",
    "data/lol/v2/models/player/multileague-v2/adaptive-development-artifact-v2.json",
    CURRENT_MANIFEST_LOCATOR.as_posix(),
)


class FutureProtocolError(RuntimeError):
    """The future ratings protocol is malformed, contaminated, or unbound."""


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
        raise FutureProtocolError("future protocol value is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FutureProtocolError(f"cannot read bound artifact: {path}") from exc
    if not isinstance(value, dict):
        raise FutureProtocolError(f"bound artifact is not an object: {path}")
    return raw, value


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        raise FutureProtocolError(f"bound source is unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256(path),
    }


def _integrity_checked_v2_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _protocol_raw, protocol = _read_object(root / v2_protocol.DEFAULT_OUTPUT)
    try:
        v2_protocol.validate_equal_series_protocol_lock(protocol, root=root)
    except v2_protocol.EqualSeriesProtocolError as exc:
        if not str(exc).startswith("bound source drifted:"):
            raise FutureProtocolError(f"v2 protocol integrity failed: {exc}") from exc
    _selection_raw, selection = _read_object(root / v2_runner.DEFAULT_OUTPUT)
    try:
        v2_runner.validate_equal_series_adaptive_artifact(selection, root=root)
    except v2_runner.EqualSeriesRunnerError as exc:
        if not str(exc).startswith("bound source drifted:"):
            raise FutureProtocolError(f"v2 selection integrity failed: {exc}") from exc
    selected_id = (selection.get("selection") or {}).get("selected_candidate_id")
    if selected_id != SELECTED_CANDIDATE_ID:
        raise FutureProtocolError("adaptive selected candidate identity changed")
    return protocol, selection


def _selected_candidate(protocol: Mapping[str, Any]) -> dict[str, Any]:
    candidates = (protocol.get("candidate_family") or {}).get("candidates") or []
    selected = [
        dict(item)
        for item in candidates
        if isinstance(item, Mapping)
        and item.get("candidate_id") == SELECTED_CANDIDATE_ID
    ]
    if len(selected) != 1:
        raise FutureProtocolError("selected candidate definition is unavailable")
    return selected[0]


def _parse_locked_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise FutureProtocolError("locked_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise FutureProtocolError("locked_at must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    boundary_utc = FUTURE_SEALED_START.replace(tzinfo=timezone.utc)
    if parsed >= boundary_utc:
        raise FutureProtocolError("protocol must be locked before the future boundary")
    return parsed


def build_future_protocol_lock(
    *,
    locked_at: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    lock_time = _parse_locked_at(locked_at)
    source_snapshot = validate_current_source_snapshot(root=root)
    information = source_snapshot.get("information_boundary") or {}
    if (
        information.get("all_outcomes_present_in_snapshot_are_adaptive_development")
        is not True
        or information.get("future_sealed_targets_present") is not False
    ):
        raise FutureProtocolError("source snapshot information boundary changed")
    maps_record = (source_snapshot.get("files") or {}).get("maps") or {}
    maps_path = root / str(maps_record.get("locator") or "")
    try:
        dates = pd.to_datetime(
            pd.read_parquet(maps_path, columns=["date"])["date"],
            errors="raise",
        )
    except Exception as exc:  # noqa: BLE001
        raise FutureProtocolError("snapshot date metadata could not be read") from exc
    if dates.empty or dates.max().to_pydatetime().replace(tzinfo=None) >= FUTURE_SEALED_START:
        raise FutureProtocolError("source snapshot overlaps the future holdout")

    v2_lock, v2_selection = _integrity_checked_v2_inputs(root)
    candidate = _selected_candidate(v2_lock)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": lock_time.isoformat(),
        "adaptation_disclosure": {
            "all_results_available_through_source_snapshot_are_adaptive": True,
            "v2_candidate_selection_was_not_independent_validation": True,
            "v2_source_bytes_are_no_longer_exactly_replayable": True,
            "v2_sealed_results_are_not_reused_as_v3_holdout_evidence": True,
        },
        "source_snapshot": {
            "manifest_locator": CURRENT_MANIFEST_LOCATOR.as_posix(),
            "manifest_raw_sha256": CURRENT_MANIFEST_RAW_SHA256,
            "manifest_canonical_sha256": source_snapshot[
                "manifest_canonical_sha256"
            ],
            "package_id": source_snapshot["package_id"],
            "maps": source_snapshot["files"]["maps"],
            "players": source_snapshot["files"]["players"],
            "latest_observed_source_time": dates.max().isoformat(),
        },
        "locked_candidate": {
            "candidate_id": SELECTED_CANDIDATE_ID,
            "definition": candidate,
            "selection_artifact_locator": v2_runner.DEFAULT_OUTPUT.as_posix(),
            "selection_artifact_sha256": v2_selection["artifact_sha256"],
            "selection_status": "adaptive_development_choice_frozen_before_future_holdout",
        },
        "future_holdout": {
            "status": "EMPTY_NOT_YET_ACQUIRED",
            "start_inclusive_source_time": FUTURE_SEALED_START.isoformat(),
            "source_time_semantics": "timezone-naive warehouse timestamp",
            "series_atomic": True,
            "one_time_opening": True,
            "eligibility": {
                "professional_maps_only": True,
                "leagues": ["LCS", "LEC", "LCK", "LPL", "MSI", "EWC"],
                "exact_ten_player_identity_required": True,
                "pre_event_roster_receipt_required": True,
                "pre_event_prediction_ledger_required": True,
                "prediction_timestamp_strictly_before_event_start": True,
                "prediction_must_bind_candidate_source_roster_patch_and_fixture": True,
                "retrospective_prediction_generation_qualifies": False,
            },
            "support_stopping_rule": {
                "overall_series_minimum": 100,
                "each_domestic_league_series_minimum": 20,
                "domestic_leagues": list(DOMESTIC_LEAGUES),
                "one_or_both_rosters_changed_series_minimum": 20,
                "stop_at_first_independently_pinned_snapshot_meeting_metadata_only_thresholds": True,
                "outcomes_must_remain_unopened_while_checking_support": True,
            },
        },
        "evaluation": {
            "comparators": [
                "predecessor-player-random-walk",
                "predecessor-organization-random-walk",
            ],
            "metrics": ["log_loss", "brier"],
            "uncertainty": {
                "method": "series_cluster_bootstrap",
                "confidence": 0.95,
                "replicates": 10000,
                "seed": 20260803,
            },
            "required_strata": [
                "overall",
                "LCS",
                "each_domestic_league",
                "one_or_both_rosters_changed",
                "patch",
                "international",
            ],
            "pass_rule": (
                "candidate minus each comparator upper 95 percent series-cluster "
                "bootstrap bound must be nonpositive overall, for LCS, for every "
                "supported domestic league, and for the roster-change stratum; "
                "reliability must also pass its locked gate"
            ),
            "reliability_required": True,
            "failure_policy": "remain_unavailable_and_do_not_reopen_or_replace_the_holdout",
        },
        "opening_authority": {
            "independent_protocol_review_present": False,
            "independent_opening_approval_present": False,
            "approval_must_externally_pin_protocol_and_prediction_ledger_sha256": True,
            "self_authorizing": False,
        },
        "source_locks": [_source_record(root, locator) for locator in SOURCE_LOCKS],
        "decision_outputs": {
            "sealed_evaluation": None,
            "player_rating_authority": None,
            "team_rating_authority": None,
            "probability": None,
            "odds": None,
            "expected_value": None,
            "bet_recommendation": None,
        },
        "claim_ceiling": (
            "This lock creates an empty future evaluation protocol only. It does not "
            "validate or authorize player ratings, team ratings, probabilities, odds, "
            "expected value, recommendations, or wagers."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_future_protocol_lock(payload, root=root)


def validate_future_protocol_lock(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise FutureProtocolError("future protocol must be an object")
    value = dict(payload)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise FutureProtocolError("future protocol identity changed")
    declared = value.get("artifact_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    if not isinstance(declared, str) or declared != _canonical_sha256(unsigned):
        raise FutureProtocolError("future protocol canonical hash mismatch")
    _parse_locked_at(str(value.get("locked_at_utc")))
    future = value.get("future_holdout") or {}
    if (
        future.get("status") != "EMPTY_NOT_YET_ACQUIRED"
        or future.get("start_inclusive_source_time")
        != FUTURE_SEALED_START.isoformat()
        or future.get("one_time_opening") is not True
    ):
        raise FutureProtocolError("future holdout boundary changed")
    eligibility = future.get("eligibility") or {}
    if (
        eligibility.get("pre_event_prediction_ledger_required") is not True
        or eligibility.get("retrospective_prediction_generation_qualifies") is not False
    ):
        raise FutureProtocolError("prediction-ledger boundary changed")
    opening = value.get("opening_authority") or {}
    if (
        opening.get("independent_protocol_review_present") is not False
        or opening.get("independent_opening_approval_present") is not False
        or opening.get("self_authorizing") is not False
    ):
        raise FutureProtocolError("opening authority was fabricated")
    if any(item is not None for item in (value.get("decision_outputs") or {}).values()):
        raise FutureProtocolError("empty future protocol contains decision outputs")
    records = value.get("source_locks")
    if not isinstance(records, list) or len(records) != len(SOURCE_LOCKS):
        raise FutureProtocolError("future protocol source inventory changed")
    expected_locators = list(SOURCE_LOCKS)
    if [record.get("locator") for record in records if isinstance(record, Mapping)] != expected_locators:
        raise FutureProtocolError("future protocol source order changed")
    for record in records:
        locator = str(record["locator"])
        path = root / locator
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256(path) != record.get("raw_sha256")
        ):
            raise FutureProtocolError(f"future protocol source drifted: {locator}")
    source_snapshot = validate_current_source_snapshot(root=root)
    source_record = value.get("source_snapshot") or {}
    if (
        source_record.get("manifest_raw_sha256") != CURRENT_MANIFEST_RAW_SHA256
        or source_record.get("manifest_canonical_sha256")
        != source_snapshot.get("manifest_canonical_sha256")
        or source_record.get("package_id") != source_snapshot.get("package_id")
    ):
        raise FutureProtocolError("future protocol source snapshot binding changed")
    v2_lock, v2_selection = _integrity_checked_v2_inputs(root)
    candidate = value.get("locked_candidate") or {}
    if (
        candidate.get("candidate_id") != SELECTED_CANDIDATE_ID
        or candidate.get("definition") != _selected_candidate(v2_lock)
        or candidate.get("selection_artifact_sha256")
        != v2_selection.get("artifact_sha256")
    ):
        raise FutureProtocolError("future protocol candidate binding changed")
    return value


def write_protocol_lock_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    digest = hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace protocol lock: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-at", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_future_protocol_lock(locked_at=args.locked_at)
    raw_sha256 = write_protocol_lock_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "future_holdout_start": FUTURE_SEALED_START.isoformat(),
                "status": payload["result_state"],
                "rating_authority": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "FUTURE_SEALED_START",
    "FutureProtocolError",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "SELECTED_CANDIDATE_ID",
    "build_future_protocol_lock",
    "validate_future_protocol_lock",
    "write_protocol_lock_no_clobber",
]
