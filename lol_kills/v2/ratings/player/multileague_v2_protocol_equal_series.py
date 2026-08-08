"""Versioned adaptive protocol using equal-support chronological windows.

Protocol v1 failed before candidate selection because its calendar-Q4 window
contained only 13 series against the locked minimum of 20.  This successor
changes only the metadata-driven window assignment: the same adaptive corpus is
split into three chronological, series-atomic blocks with equal series counts.
The candidate family, regret threshold, and sealed-final cohort are unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import multileague_development as adapter
from . import multileague_v2_protocol as parent_protocol
from . import multileague_v2_runner as parent_runner


SCHEMA_VERSION = "scryglass:multileague-rating-v2-protocol-lock:v2"
RESULT_STATE = "EQUAL_SERIES_PROTOCOL_LOCKED_SEALED_FINAL_UNOPENED"
SOURCE_LOCATOR = (
    "lol_kills/v2/ratings/player/multileague_v2_protocol_equal_series.py"
)
PARENT_PROTOCOL = parent_protocol.DEFAULT_OUTPUT
PARENT_ADAPTIVE_ARTIFACT = parent_runner.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v2/protocol-lock-v2.json"
)
ADAPTIVE_START = datetime.fromisoformat("2025-07-01T00:00:00")
SEALED_START = datetime.fromisoformat("2026-04-01T00:00:00")
WINDOW_COUNT = 3


class EqualSeriesProtocolError(ValueError):
    """The equal-support protocol lock is malformed, stale, or unbound."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EqualSeriesProtocolError("protocol value is not canonical") from error


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("artifact_sha256", None)
    return _sha256(_canonical_bytes(body))


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EqualSeriesProtocolError(f"{label} must be a lowercase SHA-256")
    return value


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EqualSeriesProtocolError(f"cannot read bound artifact: {path}") from error
    if not isinstance(value, dict):
        raise EqualSeriesProtocolError(f"bound artifact is not an object: {path}")
    return raw, value


def _source_record(root: Path, locator: str, kind: str) -> dict[str, Any]:
    try:
        raw = (root / locator).read_bytes()
    except OSError as error:
        raise EqualSeriesProtocolError(f"bound source is unavailable: {locator}") from error
    return {
        "kind": kind,
        "locator": locator,
        "bytes": len(raw),
        "raw_sha256": _sha256(raw),
    }


def equal_series_windows(
    input_data: adapter.PrivateMultiLeagueRatingInput,
) -> tuple[tuple[adapter.DevelopmentSeries, ...], ...]:
    eligible = sorted(
        (
            series
            for series in input_data.development_series
            if ADAPTIVE_START
            <= adapter.source_local_datetime(series.source_local_start)
            < SEALED_START
        ),
        key=lambda series: (
            adapter.source_local_datetime(series.source_local_start),
            series.series_id,
        ),
    )
    if len(eligible) < WINDOW_COUNT:
        raise EqualSeriesProtocolError("adaptive corpus cannot fill equal-series windows")
    base, remainder = divmod(len(eligible), WINDOW_COUNT)
    sizes = [base + (1 if index < remainder else 0) for index in range(WINDOW_COUNT)]
    windows = []
    cursor = 0
    for size in sizes:
        windows.append(tuple(eligible[cursor : cursor + size]))
        cursor += size
    if cursor != len(eligible) or any(not window for window in windows):
        raise EqualSeriesProtocolError("equal-series partition does not reconcile")
    identities = [series.series_id for window in windows for series in window]
    if len(identities) != len(set(identities)):
        raise EqualSeriesProtocolError("series repeats across adaptive windows")
    return tuple(windows)


def window_manifests(
    windows: Sequence[Sequence[adapter.DevelopmentSeries]],
) -> list[dict[str, Any]]:
    manifests = []
    for index, window in enumerate(windows, start=1):
        identities = [series.series_id for series in window]
        starts = [
            adapter.source_local_datetime(series.source_local_start) for series in window
        ]
        manifests.append(
            {
                "window_id": f"adaptive-equal-series-{index}",
                "assignment": "chronological_series_atomic_equal_count",
                "series": len(window),
                "maps": sum(len(series.maps) for series in window),
                "first_source_local_start": min(starts).isoformat(),
                "last_source_local_start": max(starts).isoformat(),
                "ordered_series_ids_sha256": _sha256(_canonical_bytes(identities)),
            }
        )
    return manifests


def build_equal_series_protocol_lock(
    root: Path | str = Path("."),
    *,
    locked_at: str,
) -> dict[str, Any]:
    repo_root = Path(root)
    parent_raw, parent = _read_object(repo_root / PARENT_PROTOCOL)
    failed_raw, failed = _read_object(repo_root / PARENT_ADAPTIVE_ARTIFACT)
    try:
        parent = parent_protocol.validate_protocol_lock(parent, root=repo_root)
        failed = parent_runner.validate_adaptive_development_artifact(
            failed,
            root=repo_root,
        )
    except (
        parent_protocol.MultiLeagueV2ProtocolError,
        parent_runner.MultiLeagueV2RunnerError,
    ) as error:
        raise EqualSeriesProtocolError("parent adaptive evidence is invalid") from error
    if failed.get("result_state") != parent_runner.RESULT_NO_ELIGIBLE:
        raise EqualSeriesProtocolError("parent replay did not fail the support gate")
    parent_counts = [
        int(window["metrics"]["overall"]["series"])
        for window in failed["candidate_results"][0]["windows"]
    ]
    if parent_counts != [263, 7, 223]:
        raise EqualSeriesProtocolError("parent calendar-window support changed")

    binding = parent["input_binding"]
    input_data = adapter.load_multileague_development_input(
        expected_maps_sha256=str(binding["maps_sha256"]),
        expected_players_sha256=str(binding["players_sha256"]),
    )
    windows = equal_series_windows(input_data)
    manifests = window_manifests(windows)
    if [item["series"] for item in manifests] != [165, 164, 164]:
        raise EqualSeriesProtocolError("equal-series metadata support changed")

    source_locks = [
        _source_record(repo_root, PARENT_PROTOCOL.as_posix(), "parent_protocol_lock"),
        _source_record(
            repo_root,
            PARENT_ADAPTIVE_ARTIFACT.as_posix(),
            "failed_parent_adaptive_replay",
        ),
        _source_record(repo_root, adapter.DEFAULT_MAPS_LOCATOR, "warehouse_maps"),
        _source_record(repo_root, adapter.DEFAULT_PLAYERS_LOCATOR, "warehouse_players"),
        _source_record(
            repo_root,
            "lol_kills/v2/ratings/player/multileague_development.py",
            "input_adapter_source",
        ),
        _source_record(repo_root, parent_protocol.SOURCE_LOCATOR, "parent_protocol_source"),
        _source_record(repo_root, parent_runner.SOURCE_LOCATOR, "parent_runner_source"),
        _source_record(repo_root, SOURCE_LOCATOR, "equal_series_protocol_source"),
    ]
    if source_locks[0]["raw_sha256"] != _sha256(parent_raw):
        raise EqualSeriesProtocolError("parent protocol byte binding changed")
    if source_locks[1]["raw_sha256"] != _sha256(failed_raw):
        raise EqualSeriesProtocolError("parent replay byte binding changed")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "scryglass:multileague-rating-v2:equal-series-lock-2026-08-01",
        "locked_at": locked_at,
        "result_state": RESULT_STATE,
        "parent_evidence": {
            "protocol_locator": PARENT_PROTOCOL.as_posix(),
            "protocol_raw_sha256": _sha256(parent_raw),
            "protocol_artifact_sha256": parent["artifact_sha256"],
            "adaptive_artifact_locator": PARENT_ADAPTIVE_ARTIFACT.as_posix(),
            "adaptive_artifact_raw_sha256": _sha256(failed_raw),
            "adaptive_artifact_sha256": failed["artifact_sha256"],
            "adaptive_result_state": failed["result_state"],
        },
        "adaptation_disclosure": {
            "status": "METADATA_SUPPORT_CORRECTION_AFTER_FAILED_LOCK",
            "changed": ["adaptive_window_assignment"],
            "unchanged": [
                "adaptive_corpus",
                "candidate_family",
                "candidate_hyperparameters",
                "selection_rank",
                "maximum_worst_window_regret",
                "minimum_series_per_window",
                "sealed_final_cohort",
                "sealed_final_gate",
            ],
            "reason": (
                "calendar-Q4 contained 13 series, below the locked 20-series minimum; "
                "equal-series windows are derived only from chronology and series identity"
            ),
            "presealed_outcomes_remain_adaptive_not_independent_validation": True,
        },
        "input_binding": dict(binding),
        "information_boundary": dict(parent["information_boundary"]),
        "adaptive_development": {
            "corpus_start_inclusive": ADAPTIVE_START.isoformat(),
            "corpus_end_exclusive": SEALED_START.isoformat(),
            "assignment": "three_chronological_series_atomic_equal_count_windows",
            "assignment_uses_outcomes": False,
            "windows": manifests,
            "selection_rule": dict(
                parent["adaptive_development"]["selection_rule"]
            ),
            "series_frozen_predictions": True,
            "outcome_updates_strictly_after_series_end_plus_hours": 48,
        },
        "baselines": list(parent["baselines"]),
        "candidate_family": dict(parent["candidate_family"]),
        "sealed_final_gate": dict(parent["sealed_final_gate"]),
        "post_holdout_authorities_still_required": list(
            parent["post_holdout_authorities_still_required"]
        ),
        "source_locks": source_locks,
        "claim_ceiling": dict(parent["claim_ceiling"]),
        "decision_outputs": {
            "match_probability": None,
            "fair_odds": None,
            "expected_value": None,
            "bet_recommendation": None,
        },
    }
    payload["artifact_sha256"] = _artifact_sha256(payload)
    return validate_equal_series_protocol_lock(payload, root=repo_root)


def validate_equal_series_protocol_lock(
    payload: Mapping[str, Any],
    *,
    root: Path | str = Path("."),
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise EqualSeriesProtocolError("equal-series protocol must be an object")
    value = dict(payload)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise EqualSeriesProtocolError("equal-series protocol schema is unsupported")
    if value.get("result_state") != RESULT_STATE:
        raise EqualSeriesProtocolError("equal-series protocol is not locked")
    declared = _require_sha256(value.get("artifact_sha256"), "artifact_sha256")
    if declared != _artifact_sha256(value):
        raise EqualSeriesProtocolError("equal-series protocol digest mismatch")
    disclosure = value.get("adaptation_disclosure") or {}
    if (
        disclosure.get("status") != "METADATA_SUPPORT_CORRECTION_AFTER_FAILED_LOCK"
        or disclosure.get(
            "presealed_outcomes_remain_adaptive_not_independent_validation"
        )
        is not True
    ):
        raise EqualSeriesProtocolError("adaptation disclosure changed")
    boundary = value.get("information_boundary") or {}
    final_gate = value.get("sealed_final_gate") or {}
    if (
        boundary.get("sealed_final_targets_accessed") is not False
        or final_gate.get("opened") is not False
    ):
        raise EqualSeriesProtocolError("sealed-final isolation changed")
    windows = (value.get("adaptive_development") or {}).get("windows")
    if (
        not isinstance(windows, list)
        or len(windows) != WINDOW_COUNT
        or [item.get("series") for item in windows] != [165, 164, 164]
    ):
        raise EqualSeriesProtocolError("equal-series window manifest changed")
    if (value.get("candidate_family") or {}).get(
        "candidates"
    ) != parent_protocol._candidate_payloads():
        raise EqualSeriesProtocolError("candidate family changed")
    outputs = value.get("decision_outputs") or {}
    if any(item is not None for item in outputs.values()):
        raise EqualSeriesProtocolError("protocol contains decision outputs")

    records = value.get("source_locks")
    if not isinstance(records, list) or len(records) != 8:
        raise EqualSeriesProtocolError("source-lock inventory changed")
    repo_root = Path(root)
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise EqualSeriesProtocolError("source-lock record is malformed")
        locator = record.get("locator")
        if not isinstance(locator, str) or not locator or locator in seen:
            raise EqualSeriesProtocolError("source-lock locator is invalid")
        seen.add(locator)
        expected = _require_sha256(record.get("raw_sha256"), f"{locator} raw_sha256")
        try:
            raw = (repo_root / locator).read_bytes()
        except OSError as error:
            raise EqualSeriesProtocolError(f"bound source is unavailable: {locator}") from error
        if len(raw) != record.get("bytes") or _sha256(raw) != expected:
            raise EqualSeriesProtocolError(f"bound source drifted: {locator}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-at", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_equal_series_protocol_lock(locked_at=args.locked_at)
    raw_sha256 = parent_protocol.write_protocol_lock_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "window_series": [
                    item["series"]
                    for item in payload["adaptive_development"]["windows"]
                ],
                "sealed_final_opened": payload["sealed_final_gate"]["opened"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "EqualSeriesProtocolError",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "build_equal_series_protocol_lock",
    "equal_series_windows",
    "validate_equal_series_protocol_lock",
    "window_manifests",
]
