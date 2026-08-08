"""Frozen one-time phase-one model evaluation, without any market authority.

This module contains no outcome-opening permission.  It validates an exact
outcome cohort only after a separate caller has established independent opening
authority, evaluates the already-captured probabilities, and emits a
non-authorizing result.  Model selection, probability replacement, fitting, and
post-opening threshold changes are deliberately absent.
"""

from __future__ import annotations

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

import numpy as np

from lol_kills.v2.draft.terminal import future_prediction_ledger as draft_ledger
from lol_kills.v2.ratings.player import (
    multileague_v3_prediction_ledger as ratings_ledger,
)

from . import phase_one_collection_v1 as collection


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_one_evaluation_v1.py"
TYPESCRIPT_PARITY_LOCATOR = "apps/lol-atlas/scripts/phaseOneDraftParity.mts"
TYPESCRIPT_SCORER_LOCATOR = "apps/lol-atlas/src/lib/draftTerminalScore.ts"
PARITY_SCHEMA_VERSION = "scryglass:phase-one-draft-replay-parity-registry:v1"
PARITY_REPLAY_SCHEMA_VERSION = "scryglass:phase-one-draft-typescript-replay:v1"
OUTCOME_SCHEMA_VERSION = "scryglass:phase-one-sealed-outcome-cohort:v1"
RESULT_SCHEMA_VERSION = "scryglass:phase-one-model-evaluation:v1"
PARITY_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-one/parity"
)
OUTCOME_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-one/outcomes"
)
OUTCOME_EVIDENCE_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-one/outcome-evidence"
)
OUTPUT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-one/evaluations"
)
RATINGS_CANDIDATE = ratings_ledger.MODEL_IDS[0]
RATINGS_COMPARATORS = ratings_ledger.MODEL_IDS[1:]
RATINGS_BOOTSTRAP_REPLICATES = 10_000
RATINGS_BOOTSTRAP_SEED = 20_260_803
DRAFT_BOOTSTRAP_REPLICATES = 10_000
DRAFT_BOOTSTRAP_SEED = 20_260_804
CONFIDENCE_INTERVAL = (0.025, 0.975)
ECE_BINS = 10
ECE_DELTA_UPPER_MAXIMUM = 0.01
PARITY_TOLERANCE = 1e-12
ENTITY_NETWORK_HAC_CRITICAL_VALUE = 2.1
ENTITY_NETWORK_HAC_MINIMUM_SERIES = 20
ENTITY_NETWORK_HAC_MINIMUM_PARTICIPANTS = 50
DOMESTIC_LEAGUES = ("LCS", "LEC", "LCK", "LPL")
INTERNATIONAL_LEAGUES = ("MSI", "EWC")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
AUTHORITY_KEYS = (
    "phase_one_opening_authority",
    "ratings_validation_authority",
    "player_rating_authority",
    "team_rating_authority",
    "draft_validation_authority",
    "probability_authority",
    "odds_authority",
    "expected_value_authority",
    "recommendation_authority",
    "betting_authority",
)
RESULT_CLAIM_CEILING = (
    "One-time phase-one model evaluation result only. Any reported pass is a "
    "necessary research gate, not rating, probability, odds, expected-value, "
    "recommendation, transaction, or betting authority."
)


class PhaseOneEvaluationError(RuntimeError):
    """Phase-one evidence, replay parity, outcomes, or evaluation failed closed."""


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
        raise PhaseOneEvaluationError("value is not canonical JSON") from exc


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


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PhaseOneEvaluationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"invalid number: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PhaseOneEvaluationError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PhaseOneEvaluationError(f"{label} must contain an object")
    return value


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise PhaseOneEvaluationError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhaseOneEvaluationError(f"{field} must be nonempty")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseOneEvaluationError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PhaseOneEvaluationError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise PhaseOneEvaluationError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PhaseOneEvaluationError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise PhaseOneEvaluationError(f"{field} must be finite")
    return result


def _locator(value: Any, prefix: PurePosixPath, field: str) -> str:
    path = PurePosixPath(_nonempty(value, field))
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or tuple(path.parts[: len(prefix.parts)]) != prefix.parts
        or path.suffix != ".json"
    ):
        raise PhaseOneEvaluationError(f"{field} is outside its registered root")
    return path.as_posix()


def _read_regular(root: Path, locator: str, label: str) -> bytes:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise PhaseOneEvaluationError(f"{label} is not an unaliased regular file")
    return path.read_bytes()


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise PhaseOneEvaluationError(f"source is unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _snapshot(
    root: Path, locator: str
) -> tuple[bytes, dict[str, Any]]:
    locator = _locator(locator, collection.SNAPSHOT_PREFIX, "snapshot_locator")
    raw = _read_regular(root, locator, "joint snapshot")
    try:
        checked = collection.validate_joint_ledger_snapshot(
            _strict_object(raw, "joint snapshot"), root=root
        )
    except collection.PhaseOneCollectionError as exc:
        raise PhaseOneEvaluationError("joint snapshot is invalid") from exc
    if (
        checked["status"]
        != "PHASE_ONE_METADATA_SUPPORT_MET_OUTCOMES_UNOPENED"
        or checked["support"]["joint_metadata_support_met"] is not True
        or checked["outcomes_present"] is not False
        or checked["outcomes_accessed"] is not False
    ):
        raise PhaseOneEvaluationError("joint snapshot is not eligible for opening")
    return raw, checked


def _identity_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_sha256(
        [
            {
                "event_id": entry["event_id"],
                "series_id": entry["series_id"],
                "game_number": entry["game_number"],
                "prediction_artifact_sha256": entry["prediction_artifact_sha256"],
            }
            for entry in entries
        ]
    )


def build_draft_replay_parity_registry(
    *,
    snapshot_locator: str,
    typescript_replay_raw: bytes,
    parity_locator: str,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Bind an outcome-free TypeScript replay to every Draft receipt."""

    created = clock()
    if not isinstance(created, datetime) or created.tzinfo is None:
        raise PhaseOneEvaluationError("parity clock must be timezone-aware")
    created = created.astimezone(timezone.utc)
    snapshot_raw, snapshot = _snapshot(root, snapshot_locator)
    parity_locator = _locator(parity_locator, PARITY_PREFIX, "parity_locator")
    replay = _strict_object(typescript_replay_raw, "TypeScript parity replay")
    entries = snapshot["draft_ledger_candidate"]["entries"]
    comparisons = replay.get("comparisons")
    if (
        set(replay) != {"schema_version", "snapshot_artifact_sha256", "comparisons"}
        or replay.get("schema_version") != PARITY_REPLAY_SCHEMA_VERSION
        or replay.get("snapshot_artifact_sha256") != snapshot["artifact_sha256"]
        or not isinstance(comparisons, list)
    ):
        raise PhaseOneEvaluationError("TypeScript parity replay structure changed")
    expected_by_identity = {
        (entry["event_id"], entry["game_number"]): entry for entry in entries
    }
    expected_keys = {
        "event_id",
        "series_id",
        "game_number",
        "prediction_artifact_sha256",
        "python_draft_index_probability_a",
        "typescript_draft_index_probability_a",
        "draft_index_absolute_delta",
        "python_combined_probability_blue",
        "typescript_combined_probability_blue",
        "combined_absolute_delta",
    }
    seen: set[tuple[str, int]] = set()
    max_draft = 0.0
    max_combined = 0.0
    for item in comparisons:
        if not isinstance(item, Mapping) or set(item) != expected_keys:
            raise PhaseOneEvaluationError("parity comparison structure changed")
        identity = (str(item["event_id"]), int(item["game_number"]))
        expected = expected_by_identity.get(identity)
        if (
            expected is None
            or identity in seen
            or item["series_id"] != expected["series_id"]
            or item["prediction_artifact_sha256"]
            != expected["prediction_artifact_sha256"]
        ):
            raise PhaseOneEvaluationError("parity comparison identity changed")
        seen.add(identity)
        py_draft = _number(
            item["python_draft_index_probability_a"], "python draft probability"
        )
        ts_draft = _number(
            item["typescript_draft_index_probability_a"], "TypeScript draft probability"
        )
        py_combined = _number(
            item["python_combined_probability_blue"], "python combined probability"
        )
        ts_combined = _number(
            item["typescript_combined_probability_blue"],
            "TypeScript combined probability",
        )
        for probability in (py_draft, ts_draft, py_combined, ts_combined):
            if not 0.0 < probability < 1.0:
                raise PhaseOneEvaluationError("parity probability is outside (0,1)")
        draft_delta = _number(item["draft_index_absolute_delta"], "draft delta")
        combined_delta = _number(item["combined_absolute_delta"], "combined delta")
        if (
            not math.isclose(draft_delta, abs(py_draft - ts_draft), abs_tol=1e-15)
            or not math.isclose(
                combined_delta, abs(py_combined - ts_combined), abs_tol=1e-15
            )
        ):
            raise PhaseOneEvaluationError("parity delta does not reconcile")
        max_draft = max(max_draft, draft_delta)
        max_combined = max(max_combined, combined_delta)
    if seen != set(expected_by_identity):
        raise PhaseOneEvaluationError("parity replay does not cover the exact snapshot")
    passed = max(max_draft, max_combined) <= PARITY_TOLERANCE
    payload: dict[str, Any] = {
        "schema_version": PARITY_SCHEMA_VERSION,
        "result_state": (
            "EXACT_OUTCOME_FREE_PYTHON_TYPESCRIPT_REPLAY_PARITY"
            if passed
            else "PYTHON_TYPESCRIPT_REPLAY_PARITY_FAILED"
        ),
        "created_at_utc": created.isoformat(),
        "parity_locator": parity_locator,
        "snapshot": {
            "locator": snapshot_locator,
            "raw_sha256": _sha256_bytes(snapshot_raw),
            "artifact_sha256": snapshot["artifact_sha256"],
        },
        "typescript_replay": {
            "raw_sha256": _sha256_bytes(typescript_replay_raw),
            "raw_base64": base64.b64encode(typescript_replay_raw).decode("ascii"),
            "schema_version": PARITY_REPLAY_SCHEMA_VERSION,
        },
        "coverage": {
            "expected_events": len(entries),
            "replayed_events": len(comparisons),
            "event_identity_sha256": _identity_digest(entries),
            "exact_snapshot_coverage": True,
        },
        "numerical_parity": {
            "absolute_tolerance": PARITY_TOLERANCE,
            "maximum_draft_index_absolute_delta": max_draft,
            "maximum_combined_absolute_delta": max_combined,
            "passed": passed,
        },
        "source_locks": [
            _source_record(root, SOURCE_LOCATOR),
            _source_record(root, TYPESCRIPT_PARITY_LOCATOR),
            _source_record(root, TYPESCRIPT_SCORER_LOCATOR),
            _source_record(root, draft_ledger.SOURCE_LOCKS[0]),
        ],
        "outcomes_present": False,
        "outcomes_accessed": False,
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": (
            "Outcome-free Python/TypeScript replay parity only; no model, "
            "probability, recommendation, or betting authority."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_draft_replay_parity_registry(payload, root=root)


def validate_draft_replay_parity_registry(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseOneEvaluationError("parity registry must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "result_state",
        "created_at_utc",
        "parity_locator",
        "snapshot",
        "typescript_replay",
        "coverage",
        "numerical_parity",
        "source_locks",
        "outcomes_present",
        "outcomes_accessed",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise PhaseOneEvaluationError("parity registry structure changed")
    if value.get("schema_version") != PARITY_SCHEMA_VERSION:
        raise PhaseOneEvaluationError("parity registry schema changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseOneEvaluationError("parity registry hash changed")
    _timestamp(value.get("created_at_utc"), "parity created_at")
    _locator(value.get("parity_locator"), PARITY_PREFIX, "parity_locator")
    snapshot_record = value.get("snapshot")
    if not isinstance(snapshot_record, Mapping) or set(snapshot_record) != {
        "locator",
        "raw_sha256",
        "artifact_sha256",
    }:
        raise PhaseOneEvaluationError("parity snapshot binding changed")
    snapshot_raw, snapshot = _snapshot(root, str(snapshot_record["locator"]))
    if snapshot_record != {
        "locator": snapshot_record["locator"],
        "raw_sha256": _sha256_bytes(snapshot_raw),
        "artifact_sha256": snapshot["artifact_sha256"],
    }:
        raise PhaseOneEvaluationError("parity snapshot no longer matches")
    replay = value.get("typescript_replay")
    if not isinstance(replay, Mapping) or set(replay) != {
        "raw_sha256",
        "raw_base64",
        "schema_version",
    }:
        raise PhaseOneEvaluationError("parity replay binding changed")
    try:
        replay_raw = base64.b64decode(
            _nonempty(replay["raw_base64"], "typescript_replay.raw_base64"),
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise PhaseOneEvaluationError("parity replay base64 is invalid") from exc
    if _sha256_bytes(replay_raw) != _sha(
        replay["raw_sha256"], "typescript_replay.raw_sha256"
    ):
        raise PhaseOneEvaluationError("embedded parity replay hash changed")
    if replay["schema_version"] != PARITY_REPLAY_SCHEMA_VERSION:
        raise PhaseOneEvaluationError("parity replay schema changed")
    entries = snapshot["draft_ledger_candidate"]["entries"]
    replay_payload = _strict_object(replay_raw, "embedded TypeScript parity replay")
    comparisons = replay_payload.get("comparisons")
    if (
        set(replay_payload)
        != {"schema_version", "snapshot_artifact_sha256", "comparisons"}
        or replay_payload.get("schema_version") != PARITY_REPLAY_SCHEMA_VERSION
        or replay_payload.get("snapshot_artifact_sha256")
        != snapshot["artifact_sha256"]
        or not isinstance(comparisons, list)
    ):
        raise PhaseOneEvaluationError("embedded parity replay structure changed")
    expected_by_identity = {
        (entry["event_id"], entry["game_number"]): entry for entry in entries
    }
    comparison_keys = {
        "event_id",
        "series_id",
        "game_number",
        "prediction_artifact_sha256",
        "python_draft_index_probability_a",
        "typescript_draft_index_probability_a",
        "draft_index_absolute_delta",
        "python_combined_probability_blue",
        "typescript_combined_probability_blue",
        "combined_absolute_delta",
    }
    seen: set[tuple[str, int]] = set()
    max_draft = 0.0
    max_combined = 0.0
    for item in comparisons:
        if not isinstance(item, Mapping) or set(item) != comparison_keys:
            raise PhaseOneEvaluationError("embedded parity comparison changed")
        game = item.get("game_number")
        if isinstance(game, bool) or not isinstance(game, int):
            raise PhaseOneEvaluationError("embedded parity game number changed")
        identity = (str(item["event_id"]), game)
        expected = expected_by_identity.get(identity)
        if (
            expected is None
            or identity in seen
            or item["series_id"] != expected["series_id"]
            or item["prediction_artifact_sha256"]
            != expected["prediction_artifact_sha256"]
        ):
            raise PhaseOneEvaluationError("embedded parity identity changed")
        seen.add(identity)
        py_draft = _number(item["python_draft_index_probability_a"], "python draft")
        ts_draft = _number(item["typescript_draft_index_probability_a"], "ts draft")
        py_combined = _number(item["python_combined_probability_blue"], "python combined")
        ts_combined = _number(item["typescript_combined_probability_blue"], "ts combined")
        draft_delta = _number(item["draft_index_absolute_delta"], "draft delta")
        combined_delta = _number(item["combined_absolute_delta"], "combined delta")
        if (
            not math.isclose(draft_delta, abs(py_draft - ts_draft), abs_tol=1e-15)
            or not math.isclose(
                combined_delta, abs(py_combined - ts_combined), abs_tol=1e-15
            )
        ):
            raise PhaseOneEvaluationError("embedded parity delta does not reconcile")
        max_draft = max(max_draft, draft_delta)
        max_combined = max(max_combined, combined_delta)
    if seen != set(expected_by_identity):
        raise PhaseOneEvaluationError("embedded parity coverage changed")
    expected_coverage = {
        "expected_events": len(entries),
        "replayed_events": len(entries),
        "event_identity_sha256": _identity_digest(entries),
        "exact_snapshot_coverage": True,
    }
    if value.get("coverage") != expected_coverage:
        raise PhaseOneEvaluationError("parity coverage changed")
    numerical = value.get("numerical_parity")
    if not isinstance(numerical, Mapping) or set(numerical) != {
        "absolute_tolerance",
        "maximum_draft_index_absolute_delta",
        "maximum_combined_absolute_delta",
        "passed",
    }:
        raise PhaseOneEvaluationError("parity numerical result changed")
    if (
        numerical["absolute_tolerance"] != PARITY_TOLERANCE
        or numerical["maximum_draft_index_absolute_delta"] != max_draft
        or numerical["maximum_combined_absolute_delta"] != max_combined
        or max_draft > PARITY_TOLERANCE
        or max_combined > PARITY_TOLERANCE
        or numerical["passed"] is not True
        or value.get("result_state")
        != "EXACT_OUTCOME_FREE_PYTHON_TYPESCRIPT_REPLAY_PARITY"
    ):
        raise PhaseOneEvaluationError("Python/TypeScript parity did not pass")
    expected_sources = [
        _source_record(root, SOURCE_LOCATOR),
        _source_record(root, TYPESCRIPT_PARITY_LOCATOR),
        _source_record(root, TYPESCRIPT_SCORER_LOCATOR),
        _source_record(root, draft_ledger.SOURCE_LOCKS[0]),
    ]
    if value.get("source_locks") != expected_sources:
        raise PhaseOneEvaluationError("parity source lock changed")
    if value.get("outcomes_present") is not False or value.get("outcomes_accessed") is not False:
        raise PhaseOneEvaluationError("parity registry accessed outcomes")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(authority.values()):
        raise PhaseOneEvaluationError("parity registry exceeds authority")
    if value.get("claim_ceiling") != (
        "Outcome-free Python/TypeScript replay parity only; no model, "
        "probability, recommendation, or betting authority."
    ):
        raise PhaseOneEvaluationError("parity claim ceiling changed")
    return value


def validate_outcome_cohort(
    payload: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    """Validate the exact post-event labels against a previously frozen snapshot."""

    if not isinstance(payload, Mapping):
        raise PhaseOneEvaluationError("outcome cohort must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "created_at_utc",
        "snapshot_artifact_sha256",
        "rows",
        "artifact_sha256",
    }:
        raise PhaseOneEvaluationError("outcome cohort structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseOneEvaluationError("outcome cohort hash changed")
    if (
        value.get("schema_version") != OUTCOME_SCHEMA_VERSION
        or value.get("snapshot_artifact_sha256") != snapshot["artifact_sha256"]
    ):
        raise PhaseOneEvaluationError("outcome cohort identity changed")
    created = _timestamp(value.get("created_at_utc"), "outcome cohort created_at")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise PhaseOneEvaluationError("outcome rows must be a list")
    records = snapshot["event_bundles"]
    expected = {
        (record["event_id"], record["game_number"]): record for record in records
    }
    row_keys = {
        "event_id",
        "series_id",
        "game_number",
        "actual_map_start_utc",
        "winning_side",
        "source_system",
        "source_record_id",
        "source_revision_id",
        "source_observed_at_utc",
        "evidence_locator",
        "evidence_raw_sha256",
    }
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != row_keys:
            raise PhaseOneEvaluationError("outcome row structure changed")
        game = row.get("game_number")
        if isinstance(game, bool) or not isinstance(game, int) or game < 1:
            raise PhaseOneEvaluationError("outcome game_number is invalid")
        identity = (_nonempty(row.get("event_id"), "outcome event_id"), game)
        record = expected.get(identity)
        if (
            record is None
            or identity in seen
            or row.get("series_id") != record["series_id"]
            or row.get("actual_map_start_utc") != record["actual_map_start_utc"]
            or row.get("winning_side") not in {"blue", "red"}
        ):
            raise PhaseOneEvaluationError("outcome identity or side mapping changed")
        seen.add(identity)
        for field in ("source_system", "source_record_id", "source_revision_id"):
            _nonempty(row.get(field), f"outcome.{field}")
        start = _timestamp(row["actual_map_start_utc"], "actual_map_start")
        observed = _timestamp(row["source_observed_at_utc"], "source_observed_at")
        if observed <= start or created < observed:
            raise PhaseOneEvaluationError("outcome evidence timing is invalid")
        evidence_locator = _locator(
            row.get("evidence_locator"), OUTCOME_EVIDENCE_PREFIX, "evidence_locator"
        )
        evidence_raw = _read_regular(root, evidence_locator, "outcome evidence")
        if _sha256_bytes(evidence_raw) != _sha(
            row.get("evidence_raw_sha256"), "evidence_raw_sha256"
        ):
            raise PhaseOneEvaluationError("outcome evidence hash changed")
    if seen != set(expected):
        raise PhaseOneEvaluationError("outcome cohort is not the exact snapshot cohort")
    ordered = sorted(
        rows,
        key=lambda row: (
            row["actual_map_start_utc"],
            row["event_id"],
            row["game_number"],
        ),
    )
    if rows != ordered:
        raise PhaseOneEvaluationError("outcome rows are not deterministically ordered")
    return value


def _loss(probability: float, outcome: int, metric: str) -> float:
    probability = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    if metric == "log_loss":
        return -(outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability))
    if metric in {"brier", "brier_score"}:
        return (probability - outcome) ** 2
    raise PhaseOneEvaluationError(f"unsupported metric: {metric}")


def _derived_seed(base: int, label: str) -> int:
    return (base + int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)) % (2**32)


def _delta_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_key: str,
    comparator_key: str,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        return {"maps": 0, "series": 0, "point_delta": None, "lower_95": None, "upper_95": None}
    series_ids = sorted({str(row["series_id"]) for row in rows})
    index = {series_id: offset for offset, series_id in enumerate(series_ids)}
    sums = np.zeros(len(series_ids), dtype=float)
    counts = np.zeros(len(series_ids), dtype=float)
    all_deltas: list[float] = []
    for row in rows:
        outcome = int(row["blue_win"])
        delta = _loss(float(row[candidate_key]), outcome, metric) - _loss(
            float(row[comparator_key]), outcome, metric
        )
        position = index[str(row["series_id"])]
        sums[position] += delta
        counts[position] += 1.0
        all_deltas.append(delta)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(series_ids), size=(replicates, len(series_ids)))
    sampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "maps": len(rows),
        "series": len(series_ids),
        "point_delta": float(np.mean(all_deltas)),
        "lower_95": float(np.quantile(sampled, CONFIDENCE_INTERVAL[0])),
        "upper_95": float(np.quantile(sampled, CONFIDENCE_INTERVAL[1])),
    }


def _entity_network_hac_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_key: str,
    comparator_key: str,
    metric: str,
) -> dict[str, Any]:
    """Participant/team/series network-HAC sensitivity for a mean loss delta.

    Two maps are treated as potentially dependent when they share a series,
    exact participant, or organization.  The scalar sandwich meat includes
    every ordered dependent pair exactly once after de-duplicating overlapping
    memberships.  This is a stricter sensitivity alongside, not a replacement
    for, the registered whole-series bootstrap.
    """

    if not rows:
        return {
            "maps": 0,
            "series": 0,
            "participants": 0,
            "organizations": 0,
            "dependent_ordered_pairs": 0,
            "maximum_dependency_neighborhood": 0,
            "point_delta": None,
            "standard_error": None,
            "lower_95": None,
            "upper_95": None,
            "complete": False,
            "failure_reason": "empty_stratum",
        }
    memberships: dict[str, list[int]] = {}
    deltas: list[float] = []
    participants: set[str] = set()
    organizations: set[str] = set()
    series_ids: set[str] = set()
    for index, row in enumerate(rows):
        participant_ids = row.get("participant_ids")
        organization_ids = row.get("organization_ids")
        if (
            not isinstance(participant_ids, (list, tuple))
            or len(participant_ids) != 10
            or len(set(participant_ids)) != 10
            or any(not isinstance(item, str) or not item for item in participant_ids)
        ):
            raise PhaseOneEvaluationError(
                "entity-network sensitivity requires ten exact participant IDs"
            )
        if (
            not isinstance(organization_ids, (list, tuple))
            or len(organization_ids) != 2
            or len(set(organization_ids)) != 2
            or any(not isinstance(item, str) or not item for item in organization_ids)
        ):
            raise PhaseOneEvaluationError(
                "entity-network sensitivity requires two exact organization IDs"
            )
        series_id = str(row["series_id"])
        series_ids.add(series_id)
        memberships.setdefault(f"series:{series_id}", []).append(index)
        for participant_id in participant_ids:
            participants.add(participant_id)
            memberships.setdefault(f"participant:{participant_id}", []).append(index)
        for organization_id in organization_ids:
            organizations.add(organization_id)
            memberships.setdefault(f"organization:{organization_id}", []).append(index)
        outcome = int(row["blue_win"])
        deltas.append(
            _loss(float(row[candidate_key]), outcome, metric)
            - _loss(float(row[comparator_key]), outcome, metric)
        )
    values = np.asarray(deltas, dtype=float)
    point = float(values.mean())
    residuals = values - point
    neighborhoods = [set((index,)) for index in range(len(rows))]
    for indices in memberships.values():
        group = set(indices)
        for index in group:
            neighborhoods[index].update(group)
    covariance_sum = float(
        sum(
            residuals[index]
            * sum(residuals[other] for other in neighborhood)
            for index, neighborhood in enumerate(neighborhoods)
        )
    )
    variance = (
        len(rows) / (len(rows) - 1) * covariance_sum / (len(rows) ** 2)
        if len(rows) > 1
        else float("nan")
    )
    support_met = (
        len(series_ids) >= ENTITY_NETWORK_HAC_MINIMUM_SERIES
        and len(participants) >= ENTITY_NETWORK_HAC_MINIMUM_PARTICIPANTS
    )
    variance_valid = math.isfinite(variance) and variance >= -1e-18
    complete = support_met and variance_valid
    standard_error = math.sqrt(max(variance, 0.0)) if complete else None
    failure_reason = None
    if not support_met:
        failure_reason = "minimum_series_or_participant_support_not_met"
    elif not variance_valid:
        failure_reason = "network_hac_variance_negative_or_nonfinite"
    return {
        "maps": len(rows),
        "series": len(series_ids),
        "participants": len(participants),
        "organizations": len(organizations),
        "dependent_ordered_pairs": sum(len(item) for item in neighborhoods),
        "maximum_dependency_neighborhood": max(map(len, neighborhoods)),
        "point_delta": point,
        "standard_error": standard_error,
        "lower_95": (
            None
            if standard_error is None
            else point - ENTITY_NETWORK_HAC_CRITICAL_VALUE * standard_error
        ),
        "upper_95": (
            None
            if standard_error is None
            else point + ENTITY_NETWORK_HAC_CRITICAL_VALUE * standard_error
        ),
        "complete": complete,
        "failure_reason": failure_reason,
    }


def _entity_network_metric_report(
    *,
    candidate_key: str,
    comparator_keys: Sequence[str],
    metrics: Sequence[str],
    strata: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    return {
        "method": (
            "shared_series_participant_or_organization_network_hac_"
            "sandwich_for_mean_paired_loss_delta"
        ),
        "dependency_rule": (
            "maps_may_covary_when_they_share_series_id_any_exact_player_id_"
            "or_any_exact_organization_id"
        ),
        "critical_value": ENTITY_NETWORK_HAC_CRITICAL_VALUE,
        "minimum_series": ENTITY_NETWORK_HAC_MINIMUM_SERIES,
        "minimum_participants": ENTITY_NETWORK_HAC_MINIMUM_PARTICIPANTS,
        "status": "required_sensitivity_not_replacement_for_series_bootstrap",
        "metrics_by_stratum": {
            stratum: {
                comparator: {
                    metric: _entity_network_hac_interval(
                        subset,
                        candidate_key=candidate_key,
                        comparator_key=comparator,
                        metric=metric,
                    )
                    for metric in metrics
                }
                for comparator in comparator_keys
            }
            for stratum, subset in strata.items()
        },
    }


def _ece(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    if probabilities.size == 0:
        raise PhaseOneEvaluationError("ECE inputs are empty")
    order = np.argsort(probabilities, kind="stable")
    bins = np.array_split(order, min(ECE_BINS, probabilities.size))
    return float(
        sum(
            len(indices)
            / probabilities.size
            * abs(float(probabilities[indices].mean()) - float(outcomes[indices].mean()))
            for indices in bins
            if len(indices)
        )
    )


def _calibration_fit(probabilities: np.ndarray, outcomes: np.ndarray) -> tuple[float, float] | None:
    if probabilities.size < 2 or np.unique(outcomes).size < 2:
        return None
    logits = np.log(np.clip(probabilities, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probabilities, 1e-6, 1.0))
    beta = np.array([0.0, 1.0], dtype=float)
    design = np.column_stack((np.ones(probabilities.size), logits))
    for _ in range(100):
        eta = np.clip(design @ beta, -40.0, 40.0)
        fitted = 1.0 / (1.0 + np.exp(-eta))
        weights = np.maximum(fitted * (1.0 - fitted), 1e-9)
        gradient = design.T @ (outcomes - fitted)
        hessian = design.T @ (weights[:, None] * design)
        try:
            step = np.linalg.solve(hessian + np.eye(2) * 1e-10, gradient)
        except np.linalg.LinAlgError:
            return None
        candidate = beta + step
        candidate[0] = np.clip(candidate[0], -10.0, 10.0)
        candidate[1] = np.clip(candidate[1], 0.0, 10.0)
        if float(np.max(np.abs(candidate - beta))) <= 1e-10:
            return float(candidate[0]), float(candidate[1])
        beta = candidate
    return None


def _reliability(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_key: str,
    comparator_keys: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    probabilities = np.asarray([float(row[candidate_key]) for row in rows], dtype=float)
    comparators = {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in comparator_keys
    }
    outcomes = np.asarray([int(row["blue_win"]) for row in rows], dtype=float)
    series_order = sorted({str(row["series_id"]) for row in rows})
    indices_by_series = {
        series_id: np.asarray(
            [index for index, row in enumerate(rows) if row["series_id"] == series_id],
            dtype=int,
        )
        for series_id in series_order
    }
    point_fit = _calibration_fit(probabilities, outcomes)
    point_ece = _ece(probabilities, outcomes)
    point_comparator_ece = {key: _ece(values, outcomes) for key, values in comparators.items()}
    rng = np.random.default_rng(seed)
    intercepts: list[float] = []
    slopes: list[float] = []
    ece_deltas: dict[str, list[float]] = {key: [] for key in comparator_keys}
    failures = 0
    for _ in range(replicates):
        selected = rng.integers(0, len(series_order), size=len(series_order))
        indices = np.concatenate([indices_by_series[series_order[item]] for item in selected])
        sampled_probabilities = probabilities[indices]
        sampled_outcomes = outcomes[indices]
        fitted = _calibration_fit(sampled_probabilities, sampled_outcomes)
        if fitted is None:
            failures += 1
            continue
        intercepts.append(fitted[0])
        slopes.append(fitted[1])
        candidate_ece = _ece(sampled_probabilities, sampled_outcomes)
        for key, values in comparators.items():
            ece_deltas[key].append(candidate_ece - _ece(values[indices], sampled_outcomes))
    complete = point_fit is not None and failures == 0 and len(intercepts) == replicates
    return {
        "maps": len(rows),
        "series": len(series_order),
        "equal_frequency_bins": ECE_BINS,
        "bootstrap_replicates": replicates,
        "bootstrap_failures": failures,
        "point": {
            "calibration_intercept": None if point_fit is None else point_fit[0],
            "calibration_slope": None if point_fit is None else point_fit[1],
            "ece": point_ece,
            "comparator_ece": point_comparator_ece,
        },
        "intervals": {
            "calibration_intercept": None
            if not complete
            else [
                float(np.quantile(intercepts, CONFIDENCE_INTERVAL[0])),
                float(np.quantile(intercepts, CONFIDENCE_INTERVAL[1])),
            ],
            "calibration_slope": None
            if not complete
            else [
                float(np.quantile(slopes, CONFIDENCE_INTERVAL[0])),
                float(np.quantile(slopes, CONFIDENCE_INTERVAL[1])),
            ],
            "ece_delta_upper_95": {
                key: None
                if not complete
                else float(np.quantile(values, CONFIDENCE_INTERVAL[1]))
                for key, values in ece_deltas.items()
            },
        },
        "complete": complete,
    }


def _metric_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_key: str,
    comparator_keys: Sequence[str],
    metrics: Sequence[str],
    strata: Mapping[str, Sequence[Mapping[str, Any]]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    return {
        stratum: {
            comparator: {
                metric: _delta_interval(
                    subset,
                    candidate_key=candidate_key,
                    comparator_key=comparator,
                    metric=metric,
                    replicates=replicates,
                    seed=_derived_seed(seed, f"{stratum}|{comparator}|{metric}"),
                )
                for metric in metrics
            }
            for comparator in comparator_keys
        }
        for stratum, subset in strata.items()
    }


def _ratings_report(rows: Sequence[Mapping[str, Any]], *, replicates: int) -> dict[str, Any]:
    strata: dict[str, Sequence[Mapping[str, Any]]] = {"overall": rows}
    strata.update({f"league:{league}": [row for row in rows if row["league"] == league] for league in DOMESTIC_LEAGUES})
    strata["roster_change"] = [row for row in rows if row["roster_change"]]
    for patch in sorted({str(row["patch"]) for row in rows}):
        strata[f"patch:{patch}"] = [row for row in rows if row["patch"] == patch]
    strata["international"] = [row for row in rows if row["league"] in INTERNATIONAL_LEAGUES]
    metrics = _metric_report(
        rows,
        candidate_key="rating_candidate",
        comparator_keys=RATINGS_COMPARATORS,
        metrics=("log_loss", "brier"),
        strata=strata,
        replicates=replicates,
        seed=RATINGS_BOOTSTRAP_SEED,
    )
    reliability = _reliability(
        rows,
        candidate_key="rating_candidate",
        comparator_keys=RATINGS_COMPARATORS,
        replicates=replicates,
        seed=_derived_seed(RATINGS_BOOTSTRAP_SEED, "ratings-reliability"),
    )
    gated_strata = ["overall", *(f"league:{league}" for league in DOMESTIC_LEAGUES), "roster_change"]
    entity_network = _entity_network_metric_report(
        candidate_key="rating_candidate",
        comparator_keys=RATINGS_COMPARATORS,
        metrics=("log_loss", "brier"),
        strata={name: strata[name] for name in gated_strata},
    )
    primary_pass = all(
        metrics[stratum][comparator][metric]["upper_95"] is not None
        and metrics[stratum][comparator][metric]["upper_95"] <= 0.0
        for stratum in gated_strata
        for comparator in RATINGS_COMPARATORS
        for metric in ("log_loss", "brier")
    )
    entity_network_pass = all(
        item["complete"] is True
        and item["upper_95"] is not None
        and item["upper_95"] <= 0.0
        for stratum in gated_strata
        for comparator in RATINGS_COMPARATORS
        for item in (
            entity_network["metrics_by_stratum"][stratum][comparator][
                "log_loss"
            ],
            entity_network["metrics_by_stratum"][stratum][comparator]["brier"],
        )
    )
    intervals = reliability["intervals"]
    intercept = intervals["calibration_intercept"]
    slope = intervals["calibration_slope"]
    reliability_pass = (
        reliability["complete"]
        and intercept is not None
        and intercept[0] <= 0.0 <= intercept[1]
        and slope is not None
        and slope[0] <= 1.0 <= slope[1]
        and all(
            intervals["ece_delta_upper_95"][comparator] is not None
            and intervals["ece_delta_upper_95"][comparator] <= ECE_DELTA_UPPER_MAXIMUM
            for comparator in RATINGS_COMPARATORS
        )
    )
    return {
        "candidate": RATINGS_CANDIDATE,
        "comparators": list(RATINGS_COMPARATORS),
        "bootstrap": {
            "method": "paired_series_cluster_bootstrap",
            "replicates": replicates,
            "base_seed": RATINGS_BOOTSTRAP_SEED,
            "confidence": 0.95,
            "point_weighting": "map_weighted",
            "cluster_resampling": "whole_series_with_replacement",
        },
        "metrics_by_stratum": metrics,
        "entity_network_dependence_sensitivity": entity_network,
        "reliability": reliability,
        "locked_reliability_gate": {
            "calibration_intercept_interval_includes_zero": True,
            "calibration_slope_interval_includes_one": True,
            "candidate_minus_each_comparator_ece_upper_95_maximum": ECE_DELTA_UPPER_MAXIMUM,
            "note": "This exact gate resolves the pre-opening ambiguity in the v3 ratings protocol.",
        },
        "primary_gate_passed": primary_pass,
        "entity_network_dependence_gate_passed": entity_network_pass,
        "reliability_gate_passed": reliability_pass,
        "passed": primary_pass and entity_network_pass and reliability_pass,
    }


def _draft_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    parity: Mapping[str, Any],
    replicates: int,
) -> dict[str, Any]:
    strata: dict[str, Sequence[Mapping[str, Any]]] = {"overall": rows}
    strata.update({f"league:{league}": [row for row in rows if row["league"] == league] for league in DOMESTIC_LEAGUES})
    for patch in sorted({str(row["patch"]) for row in rows}):
        strata[f"patch:{patch}"] = [row for row in rows if row["patch"] == patch]
    strata["international"] = [row for row in rows if row["league"] in INTERNATIONAL_LEAGUES]
    metrics = _metric_report(
        rows,
        candidate_key="ratings_plus_draft",
        comparator_keys=("ratings_only",),
        metrics=("log_loss", "brier_score"),
        strata=strata,
        replicates=replicates,
        seed=DRAFT_BOOTSTRAP_SEED,
    )
    reliability = _reliability(
        rows,
        candidate_key="ratings_plus_draft",
        comparator_keys=("ratings_only",),
        replicates=replicates,
        seed=_derived_seed(DRAFT_BOOTSTRAP_SEED, "draft-reliability"),
    )
    overall = metrics["overall"]["ratings_only"]
    entity_network = _entity_network_metric_report(
        candidate_key="ratings_plus_draft",
        comparator_keys=("ratings_only",),
        metrics=("log_loss", "brier_score"),
        strata={"overall": rows},
    )
    primary_pass = (
        all(overall[metric]["point_delta"] is not None and overall[metric]["point_delta"] <= 0.0 for metric in ("log_loss", "brier_score"))
        and all(overall[metric]["upper_95"] is not None and overall[metric]["upper_95"] <= 0.0 for metric in ("log_loss", "brier_score"))
        and any(overall[metric]["upper_95"] is not None and overall[metric]["upper_95"] < 0.0 for metric in ("log_loss", "brier_score"))
    )
    diagnostic_strata = [
        *(f"league:{league}" for league in DOMESTIC_LEAGUES),
        *(name for name in metrics if name.startswith("patch:")),
        "international",
    ]
    subgroup_pass = all(
        metrics[stratum]["ratings_only"][metric]["point_delta"] is not None
        and metrics[stratum]["ratings_only"][metric]["point_delta"] <= 0.0
        for stratum in diagnostic_strata
        for metric in ("log_loss", "brier_score")
    )
    entity_network_pass = all(
        item["complete"] is True
        and item["upper_95"] is not None
        and item["upper_95"] <= 0.0
        for item in (
            entity_network["metrics_by_stratum"]["overall"]["ratings_only"][
                "log_loss"
            ],
            entity_network["metrics_by_stratum"]["overall"]["ratings_only"][
                "brier_score"
            ],
        )
    )
    intervals = reliability["intervals"]
    intercept = intervals["calibration_intercept"]
    slope = intervals["calibration_slope"]
    ece_upper = intervals["ece_delta_upper_95"]["ratings_only"]
    reliability_pass = (
        reliability["complete"]
        and intercept is not None
        and intercept[0] <= 0.0 <= intercept[1]
        and slope is not None
        and slope[0] <= 1.0 <= slope[1]
        and ece_upper is not None
        and ece_upper <= ECE_DELTA_UPPER_MAXIMUM
        and parity["numerical_parity"]["passed"] is True
    )
    return {
        "candidate": "ratings_plus_draft",
        "comparator": "ratings_only",
        "bootstrap": {
            "method": "paired_series_cluster_bootstrap",
            "replicates": replicates,
            "base_seed": DRAFT_BOOTSTRAP_SEED,
            "confidence": 0.95,
            "point_weighting": "map_weighted",
            "cluster_resampling": "whole_series_with_replacement",
        },
        "metrics_by_stratum": metrics,
        "entity_network_dependence_sensitivity": entity_network,
        "reliability": reliability,
        "typescript_parity_artifact_sha256": parity["artifact_sha256"],
        "primary_gate_passed": primary_pass,
        "subgroup_nonharm_gate_passed": subgroup_pass,
        "entity_network_dependence_gate_passed": entity_network_pass,
        "reliability_gate_passed": reliability_pass,
        "passed": (
            primary_pass
            and subgroup_pass
            and entity_network_pass
            and reliability_pass
        ),
    }


def _evaluation_entities(
    ratings: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    input_receipts = ratings.get("input_receipts")
    if not isinstance(input_receipts, Mapping):
        raise PhaseOneEvaluationError(
            "phase-one ratings receipt does not contain input receipts"
        )
    roster_binding = input_receipts.get("roster")
    if not isinstance(roster_binding, Mapping):
        raise PhaseOneEvaluationError(
            "phase-one ratings receipt does not contain a roster binding"
        )
    roster = roster_binding.get("receipt")
    if not isinstance(roster, Mapping):
        raise PhaseOneEvaluationError(
            "phase-one ratings receipt does not contain an exact roster receipt"
        )
    teams = roster.get("teams")
    if not isinstance(teams, list) or len(teams) != 2:
        raise PhaseOneEvaluationError(
            "phase-one evaluation roster does not contain two exact teams"
        )

    participant_ids: list[str] = []
    organization_ids: list[str] = []
    for team in teams:
        if not isinstance(team, Mapping):
            raise PhaseOneEvaluationError(
                "phase-one evaluation roster contains an invalid team"
            )
        organization_id = team.get("organization_id")
        if not isinstance(organization_id, str) or not organization_id:
            raise PhaseOneEvaluationError(
                "phase-one evaluation roster contains an invalid organization ID"
            )
        players = team.get("players")
        if not isinstance(players, list) or len(players) != 5:
            raise PhaseOneEvaluationError(
                "phase-one evaluation roster team does not contain five exact players"
            )
        organization_ids.append(organization_id)
        for player in players:
            if not isinstance(player, Mapping):
                raise PhaseOneEvaluationError(
                    "phase-one evaluation roster contains an invalid player"
                )
            player_id = player.get("player_id")
            if not isinstance(player_id, str) or not player_id:
                raise PhaseOneEvaluationError(
                    "phase-one evaluation roster contains an invalid player ID"
                )
            participant_ids.append(player_id)

    if len(participant_ids) != 10 or len(set(participant_ids)) != 10:
        raise PhaseOneEvaluationError(
            "phase-one evaluation roster does not contain ten exact players"
        )
    if len(organization_ids) != 2 or len(set(organization_ids)) != 2:
        raise PhaseOneEvaluationError(
            "phase-one evaluation roster does not contain two exact organizations"
        )
    return tuple(participant_ids), tuple(organization_ids)


def _evaluation_rows(
    *,
    snapshot: Mapping[str, Any],
    outcomes: Mapping[str, Any],
    root: Path,
) -> list[dict[str, Any]]:
    outcome_by_identity = {
        (row["event_id"], row["game_number"]): row for row in outcomes["rows"]
    }
    rows: list[dict[str, Any]] = []
    for entry in snapshot["draft_ledger_candidate"]["entries"]:
        prediction_raw = _read_regular(root, entry["prediction_locator"], "draft prediction")
        prediction = draft_ledger.validate_draft_prediction_receipt(
            _strict_object(prediction_raw, "draft prediction"), root=root
        )
        ratings = prediction["input_receipts"]["ratings_prediction"]["value"]
        participant_ids, organization_ids = _evaluation_entities(ratings)
        outcome = outcome_by_identity[(entry["event_id"], entry["game_number"])]
        rating_predictions = ratings["evaluation_predictions"]
        draft_predictions = prediction["evaluation_predictions"]
        rows.append(
            {
                "event_id": entry["event_id"],
                "series_id": entry["series_id"],
                "game_number": entry["game_number"],
                "league": entry["league"],
                "patch": entry["patch"],
                "roster_change": ratings["event"]["roster_change_stratum"]
                == "ONE_OR_BOTH_ROSTERS_CHANGED",
                "sparse_or_new_champion": entry["sparse_or_new_champion_map"],
                "participant_ids": participant_ids,
                "organization_ids": organization_ids,
                "blue_win": int(outcome["winning_side"] == "blue"),
                "rating_candidate": float(rating_predictions[RATINGS_CANDIDATE]["p_blue"]),
                RATINGS_COMPARATORS[0]: float(rating_predictions[RATINGS_COMPARATORS[0]]["p_blue"]),
                RATINGS_COMPARATORS[1]: float(rating_predictions[RATINGS_COMPARATORS[1]]["p_blue"]),
                "ratings_only": float(draft_predictions["ratings_only"]["p_blue"]),
                "ratings_plus_draft": float(draft_predictions["ratings_plus_draft"]["p_blue"]),
            }
        )
    return rows


def evaluate_phase_one(
    *,
    snapshot_locator: str,
    parity_locator: str,
    outcome_cohort_raw: bytes,
    outcome_cohort_locator: str,
    opening_authority_binding: Mapping[str, Any],
    run_id: str,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Evaluate once after a separate opening gate has validated authority."""

    evaluated = clock()
    if not isinstance(evaluated, datetime) or evaluated.tzinfo is None:
        raise PhaseOneEvaluationError("evaluation clock must be timezone-aware")
    evaluated = evaluated.astimezone(timezone.utc)
    snapshot_raw, snapshot = _snapshot(root, snapshot_locator)
    parity_locator = _locator(parity_locator, PARITY_PREFIX, "parity_locator")
    parity_raw = _read_regular(root, parity_locator, "parity registry")
    parity = validate_draft_replay_parity_registry(
        _strict_object(parity_raw, "parity registry"), root=root
    )
    if parity["snapshot"]["artifact_sha256"] != snapshot["artifact_sha256"]:
        raise PhaseOneEvaluationError("parity and snapshot differ")
    outcome_cohort_locator = _locator(
        outcome_cohort_locator, OUTCOME_PREFIX, "outcome_cohort_locator"
    )
    outcomes = validate_outcome_cohort(
        _strict_object(outcome_cohort_raw, "outcome cohort"),
        snapshot=snapshot,
        root=root,
    )
    rows = _evaluation_rows(snapshot=snapshot, outcomes=outcomes, root=root)
    ratings = _ratings_report(rows, replicates=RATINGS_BOOTSTRAP_REPLICATES)
    draft = _draft_report(
        rows, parity=parity, replicates=DRAFT_BOOTSTRAP_REPLICATES
    )
    phase_one_passed = ratings["passed"] and draft["passed"]
    payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_state": (
            "PHASE_ONE_MODELS_PASSED_PENDING_INDEPENDENT_REGISTRATION"
            if phase_one_passed
            else "PHASE_ONE_MODEL_GATE_FAILED_TERMINALLY"
        ),
        "run_id": _nonempty(run_id, "run_id"),
        "evaluated_at_utc": evaluated.isoformat(),
        "opening_authority_binding": dict(opening_authority_binding),
        "inputs": {
            "snapshot_locator": snapshot_locator,
            "snapshot_raw_sha256": _sha256_bytes(snapshot_raw),
            "snapshot_artifact_sha256": snapshot["artifact_sha256"],
            "parity_locator": parity_locator,
            "parity_raw_sha256": _sha256_bytes(parity_raw),
            "parity_artifact_sha256": parity["artifact_sha256"],
            "outcome_cohort_locator": outcome_cohort_locator,
            "outcome_cohort_raw_sha256": _sha256_bytes(outcome_cohort_raw),
            "outcome_cohort_artifact_sha256": outcomes["artifact_sha256"],
            "maps": len(rows),
            "series": len({row["series_id"] for row in rows}),
        },
        "ratings_evaluation": ratings,
        "draft_evaluation": draft,
        "phase_one_models_passed": phase_one_passed,
        "phase_two_opening_authorized": False,
        "recalibration_authorized": False,
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": RESULT_CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_phase_one_evaluation_result(payload)


def _validate_entity_network_report(
    report: Any,
    *,
    strata: Sequence[str],
    comparators: Sequence[str],
    metrics: Sequence[str],
) -> bool:
    if not isinstance(report, Mapping) or set(report) != {
        "method",
        "dependency_rule",
        "critical_value",
        "minimum_series",
        "minimum_participants",
        "status",
        "metrics_by_stratum",
    }:
        raise PhaseOneEvaluationError(
            "entity-network dependence report structure changed"
        )
    if report != {
        **dict(report),
        "method": (
            "shared_series_participant_or_organization_network_hac_"
            "sandwich_for_mean_paired_loss_delta"
        ),
        "dependency_rule": (
            "maps_may_covary_when_they_share_series_id_any_exact_player_id_"
            "or_any_exact_organization_id"
        ),
        "critical_value": ENTITY_NETWORK_HAC_CRITICAL_VALUE,
        "minimum_series": ENTITY_NETWORK_HAC_MINIMUM_SERIES,
        "minimum_participants": ENTITY_NETWORK_HAC_MINIMUM_PARTICIPANTS,
        "status": "required_sensitivity_not_replacement_for_series_bootstrap",
    }:
        raise PhaseOneEvaluationError(
            "entity-network dependence contract changed"
        )
    by_stratum = report.get("metrics_by_stratum")
    if not isinstance(by_stratum, Mapping) or set(by_stratum) != set(strata):
        raise PhaseOneEvaluationError(
            "entity-network dependence strata changed"
        )
    passed = True
    expected_item_keys = {
        "maps",
        "series",
        "participants",
        "organizations",
        "dependent_ordered_pairs",
        "maximum_dependency_neighborhood",
        "point_delta",
        "standard_error",
        "lower_95",
        "upper_95",
        "complete",
        "failure_reason",
    }
    for stratum in strata:
        by_comparator = by_stratum[stratum]
        if not isinstance(by_comparator, Mapping) or set(by_comparator) != set(
            comparators
        ):
            raise PhaseOneEvaluationError(
                "entity-network dependence comparators changed"
            )
        for comparator in comparators:
            by_metric = by_comparator[comparator]
            if not isinstance(by_metric, Mapping) or set(by_metric) != set(metrics):
                raise PhaseOneEvaluationError(
                    "entity-network dependence metrics changed"
                )
            for metric in metrics:
                item = by_metric[metric]
                if not isinstance(item, Mapping) or set(item) != expected_item_keys:
                    raise PhaseOneEvaluationError(
                        "entity-network dependence interval structure changed"
                    )
                for key in (
                    "maps",
                    "series",
                    "participants",
                    "organizations",
                    "dependent_ordered_pairs",
                    "maximum_dependency_neighborhood",
                ):
                    if (
                        isinstance(item[key], bool)
                        or not isinstance(item[key], int)
                        or item[key] < 0
                    ):
                        raise PhaseOneEvaluationError(
                            "entity-network dependence count changed"
                        )
                complete = item.get("complete") is True
                if complete:
                    point = _number(item["point_delta"], "network-HAC point")
                    standard_error = _number(
                        item["standard_error"], "network-HAC standard error"
                    )
                    lower = _number(item["lower_95"], "network-HAC lower")
                    upper = _number(item["upper_95"], "network-HAC upper")
                    if (
                        standard_error < 0.0
                        or item.get("failure_reason") is not None
                        or not math.isclose(
                            lower,
                            point
                            - ENTITY_NETWORK_HAC_CRITICAL_VALUE * standard_error,
                            abs_tol=1e-12,
                        )
                        or not math.isclose(
                            upper,
                            point
                            + ENTITY_NETWORK_HAC_CRITICAL_VALUE * standard_error,
                            abs_tol=1e-12,
                        )
                    ):
                        raise PhaseOneEvaluationError(
                            "entity-network dependence interval does not reconcile"
                        )
                else:
                    if any(
                        item.get(key) is not None
                        for key in ("standard_error", "lower_95", "upper_95")
                    ) or not isinstance(item.get("failure_reason"), str):
                        raise PhaseOneEvaluationError(
                            "incomplete entity-network dependence interval changed"
                        )
                    upper = None
                passed = passed and complete and upper is not None and upper <= 0.0
    return passed


def validate_phase_one_evaluation_result(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable result without granting or reopening authority."""

    if not isinstance(payload, Mapping):
        raise PhaseOneEvaluationError("phase-one result must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "result_state",
        "run_id",
        "evaluated_at_utc",
        "opening_authority_binding",
        "inputs",
        "ratings_evaluation",
        "draft_evaluation",
        "phase_one_models_passed",
        "phase_two_opening_authorized",
        "recalibration_authorized",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise PhaseOneEvaluationError("phase-one result structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseOneEvaluationError("phase-one result hash changed")
    _nonempty(value.get("run_id"), "run_id")
    _timestamp(value.get("evaluated_at_utc"), "evaluated_at_utc")
    binding = value.get("opening_authority_binding")
    if not isinstance(binding, Mapping) or set(binding) != {
        "authority_id",
        "authority_raw_sha256",
        "opening_marker_locator",
    }:
        raise PhaseOneEvaluationError("opening authority binding changed")
    _nonempty(binding.get("authority_id"), "authority_id")
    _sha(binding.get("authority_raw_sha256"), "authority_raw_sha256")
    marker = PurePosixPath(_nonempty(binding.get("opening_marker_locator"), "opening_marker_locator"))
    marker_prefix = PurePosixPath(
        "data/lol/v2/evaluation/match-winner-market-v1/phase-one/opening-markers"
    )
    if (
        marker.is_absolute()
        or tuple(marker.parts[: len(marker_prefix.parts)]) != marker_prefix.parts
        or marker.suffix != ".json"
    ):
        raise PhaseOneEvaluationError("opening marker locator changed")
    inputs = value.get("inputs")
    expected_input_keys = {
        "snapshot_locator",
        "snapshot_raw_sha256",
        "snapshot_artifact_sha256",
        "parity_locator",
        "parity_raw_sha256",
        "parity_artifact_sha256",
        "outcome_cohort_locator",
        "outcome_cohort_raw_sha256",
        "outcome_cohort_artifact_sha256",
        "maps",
        "series",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_input_keys:
        raise PhaseOneEvaluationError("phase-one result inputs changed")
    _locator(inputs["snapshot_locator"], collection.SNAPSHOT_PREFIX, "snapshot_locator")
    _locator(inputs["parity_locator"], PARITY_PREFIX, "parity_locator")
    _locator(inputs["outcome_cohort_locator"], OUTCOME_PREFIX, "outcome_cohort_locator")
    for key, item in inputs.items():
        if key.endswith("sha256"):
            _sha(item, f"inputs.{key}")
    if any(
        isinstance(inputs[key], bool)
        or not isinstance(inputs[key], int)
        or inputs[key] <= 0
        for key in ("maps", "series")
    ):
        raise PhaseOneEvaluationError("phase-one sample size changed")
    ratings = value.get("ratings_evaluation")
    draft = value.get("draft_evaluation")
    if not isinstance(ratings, Mapping) or not isinstance(draft, Mapping):
        raise PhaseOneEvaluationError("model evaluation reports are missing")
    if ratings.get("bootstrap") != {
        "method": "paired_series_cluster_bootstrap",
        "replicates": RATINGS_BOOTSTRAP_REPLICATES,
        "base_seed": RATINGS_BOOTSTRAP_SEED,
        "confidence": 0.95,
        "point_weighting": "map_weighted",
        "cluster_resampling": "whole_series_with_replacement",
    }:
        raise PhaseOneEvaluationError("ratings bootstrap contract changed")
    if draft.get("bootstrap") != {
        "method": "paired_series_cluster_bootstrap",
        "replicates": DRAFT_BOOTSTRAP_REPLICATES,
        "base_seed": DRAFT_BOOTSTRAP_SEED,
        "confidence": 0.95,
        "point_weighting": "map_weighted",
        "cluster_resampling": "whole_series_with_replacement",
    }:
        raise PhaseOneEvaluationError("Draft bootstrap contract changed")
    ratings_entity_passed = _validate_entity_network_report(
        ratings.get("entity_network_dependence_sensitivity"),
        strata=(
            "overall",
            *(f"league:{league}" for league in DOMESTIC_LEAGUES),
            "roster_change",
        ),
        comparators=RATINGS_COMPARATORS,
        metrics=("log_loss", "brier"),
    )
    draft_entity_passed = _validate_entity_network_report(
        draft.get("entity_network_dependence_sensitivity"),
        strata=("overall",),
        comparators=("ratings_only",),
        metrics=("log_loss", "brier_score"),
    )
    if (
        ratings.get("entity_network_dependence_gate_passed")
        is not ratings_entity_passed
        or draft.get("entity_network_dependence_gate_passed")
        is not draft_entity_passed
    ):
        raise PhaseOneEvaluationError(
            "entity-network dependence gate does not reconcile"
        )
    if ratings.get("passed") is not (
        ratings.get("primary_gate_passed") is True
        and ratings_entity_passed
        and ratings.get("reliability_gate_passed") is True
    ):
        raise PhaseOneEvaluationError("ratings evaluation gates do not reconcile")
    if draft.get("passed") is not (
        draft.get("primary_gate_passed") is True
        and draft.get("subgroup_nonharm_gate_passed") is True
        and draft_entity_passed
        and draft.get("reliability_gate_passed") is True
    ):
        raise PhaseOneEvaluationError("Draft evaluation gates do not reconcile")
    ratings_passed = ratings.get("passed") is True
    draft_passed = draft.get("passed") is True
    combined_passed = ratings_passed and draft_passed
    if value.get("phase_one_models_passed") is not combined_passed:
        raise PhaseOneEvaluationError("joint phase-one result does not reconcile")
    expected_state = (
        "PHASE_ONE_MODELS_PASSED_PENDING_INDEPENDENT_REGISTRATION"
        if combined_passed
        else "PHASE_ONE_MODEL_GATE_FAILED_TERMINALLY"
    )
    if value.get("schema_version") != RESULT_SCHEMA_VERSION or value.get("result_state") != expected_state:
        raise PhaseOneEvaluationError("phase-one result identity changed")
    if value.get("phase_two_opening_authorized") is not False or value.get("recalibration_authorized") is not False:
        raise PhaseOneEvaluationError("phase-one result exceeds its opening boundary")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(authority.values()):
        raise PhaseOneEvaluationError("phase-one result exceeds authority")
    if value.get("claim_ceiling") != RESULT_CLAIM_CEILING:
        raise PhaseOneEvaluationError("phase-one result claim ceiling changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhaseOneEvaluationError(f"refusing to replace evaluation artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PhaseOneEvaluationError(
                f"refusing to replace evaluation artifact: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256_bytes(raw)


__all__ = [
    "AUTHORITY_KEYS",
    "DRAFT_BOOTSTRAP_REPLICATES",
    "DRAFT_BOOTSTRAP_SEED",
    "OUTCOME_SCHEMA_VERSION",
    "PARITY_SCHEMA_VERSION",
    "PhaseOneEvaluationError",
    "RATINGS_BOOTSTRAP_REPLICATES",
    "RATINGS_BOOTSTRAP_SEED",
    "RESULT_SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "TYPESCRIPT_PARITY_LOCATOR",
    "build_draft_replay_parity_registry",
    "evaluate_phase_one",
    "validate_draft_replay_parity_registry",
    "validate_phase_one_evaluation_result",
    "validate_outcome_cohort",
    "write_no_clobber",
]
