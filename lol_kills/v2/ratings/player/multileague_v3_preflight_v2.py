"""Rehearse the frozen v3 ratings candidate on corrected pre-boundary data.

This is a source and numerical preflight, not an evaluation. Every outcome in
the snapshot predates the future holdout and remains adaptive development
evidence. The artifact intentionally contains no published player/team rating
and grants no probability or betting authority.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from . import multileague_development as adapter
from . import multileague_runner as rating
from . import multileague_v2_runner as hierarchical
from .multileague_v3_future_protocol import FUTURE_SEALED_START
from .multileague_v3_preflight_v1_registry import (
    REGISTERED_PREFLIGHT_ARTIFACT_SHA256,
    REGISTERED_PREFLIGHT_LOCATOR,
    REGISTERED_PREFLIGHT_RAW_SHA256,
    validate_registered_source_preflight_v1,
)
from .multileague_v3_registry import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_future_protocol,
)
from .multileague_v3_source_registry_v2 import (
    MANIFEST_CANONICAL_SHA256,
    MANIFEST_LOCATOR,
    MANIFEST_RAW_SHA256,
    PACKAGE_ID,
    validate_registered_source_snapshot_v2,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:multileague-rating-v3-source-preflight:v2"
RESULT_STATE = "CORRECTED_SOURCE_PREFLIGHT_PASSED_NON_AUTHORIZING"
SOURCE_LOCATOR = "lol_kills/v2/ratings/player/multileague_v3_preflight_v2.py"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v3/source-preflight-v2.json"
)
AUTHORITY_KEYS = (
    "model_validation_authority",
    "player_rating_authority",
    "team_rating_authority",
    "probability_authority",
    "recommendation_authority",
    "betting_authority",
)
SOURCE_LOCKS = (
    SOURCE_LOCATOR,
    "lol_kills/v2/ratings/player/multileague_development.py",
    "lol_kills/v2/ratings/player/multileague_runner.py",
    "lol_kills/v2/ratings/player/multileague_v2_runner.py",
    "lol_kills/v2/ratings/player/multileague_source_snapshot.py",
    "lol_kills/v2/ratings/player/multileague_v3_source_registry_v2.py",
    "lol_kills/v2/ratings/player/multileague_v3_future_protocol.py",
    "lol_kills/v2/ratings/player/multileague_v3_registry.py",
    "lol_kills/v2/ratings/player/multileague_v3_preflight_v1.py",
    "lol_kills/v2/ratings/player/multileague_v3_preflight_v1_registry.py",
    REGISTERED_PROTOCOL_LOCATOR.as_posix(),
    REGISTERED_PREFLIGHT_LOCATOR.as_posix(),
    MANIFEST_LOCATOR.as_posix(),
)


class CorrectedSourcePreflightError(RuntimeError):
    """The corrected source or frozen-candidate rehearsal failed closed."""


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
        raise CorrectedSourcePreflightError(
            "preflight value is not canonical"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        raise CorrectedSourcePreflightError(
            f"bound preflight source is unavailable: {locator}"
        )
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256(path),
    }


def _parse_built_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise CorrectedSourcePreflightError(
            "built_at must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise CorrectedSourcePreflightError("built_at must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    if parsed >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise CorrectedSourcePreflightError(
            "corrected source preflight must precede the future boundary"
        )
    return parsed


@contextmanager
def _future_boundary() -> Iterator[None]:
    boundary = pd.Timestamp(FUTURE_SEALED_START)
    old_adapter = adapter.SEALED_FINAL_START
    old_rating = rating.SEALED_FINAL_START
    adapter.SEALED_FINAL_START = boundary
    rating.SEALED_FINAL_START = boundary
    try:
        yield
    finally:
        adapter.SEALED_FINAL_START = old_adapter
        rating.SEALED_FINAL_START = old_rating


def _state_sha256(replay: hierarchical.ReplayResult) -> str:
    state = replay.state
    mean = np.asarray(state.mean, dtype="<f8", order="C")
    covariance = np.asarray(state.covariance, dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(
        _canonical_bytes(
            {
                "keys": list(state.keys),
                "mean_shape": list(mean.shape),
                "covariance_shape": list(covariance.shape),
            }
        )
    )
    digest.update(mean.tobytes(order="C"))
    digest.update(covariance.tobytes(order="C"))
    return digest.hexdigest()


def _fold_counts(input_data: adapter.PrivateMultiLeagueRatingInput) -> list[dict[str, Any]]:
    rows = []
    for fold in ("TRAIN", "DEVELOPMENT", "VALIDATION"):
        selected = [
            series for series in input_data.development_series if series.fold_id == fold
        ]
        rows.append(
            {
                "fold": fold,
                "series": len(selected),
                "maps": sum(len(series.maps) for series in selected),
            }
        )
    return rows


def build_corrected_source_preflight(
    *,
    built_at: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    build_time = _parse_built_at(built_at)
    source = validate_registered_source_snapshot_v2(root=root)
    protocol_v1 = validate_registered_future_protocol(root=root)
    failure_v1 = validate_registered_source_preflight_v1(root=root)
    files = source["files"]
    maps_path = root / files["maps"]["locator"]
    projected = pd.read_parquet(maps_path, columns=["date", "playoffs"])
    dates = pd.to_datetime(projected["date"], errors="raise")
    playoffs_dtype = str(projected["playoffs"].dtype)
    if dates.empty or dates.max().to_pydatetime().replace(tzinfo=None) >= FUTURE_SEALED_START:
        raise CorrectedSourcePreflightError(
            "corrected source overlaps the future holdout"
        )
    if playoffs_dtype not in {"bool", "boolean"}:
        raise CorrectedSourcePreflightError(
            "corrected source does not expose boolean playoffs semantics"
        )

    with _future_boundary():
        input_data = adapter.load_multileague_development_input(
            expected_maps_sha256=files["maps"]["raw_sha256"],
            expected_players_sha256=files["players"]["raw_sha256"],
            root=root,
            maps_locator=files["maps"]["locator"],
            players_locator=files["players"]["locator"],
        )
        rating._validate_input(
            input_data,
            expected_maps_sha256=files["maps"]["raw_sha256"],
            expected_players_sha256=files["players"]["raw_sha256"],
        )
        if input_data.sealed_series_metadata:
            raise CorrectedSourcePreflightError(
                "preflight source unexpectedly contains future sealed metadata"
            )
        candidate = hierarchical.CandidateSpec.from_payload(
            protocol_v1["locked_candidate"]["definition"]
        )
        replay = hierarchical.replay_candidate(input_data, candidate)

    psd = replay.state.assert_psd()
    coverage = dict(input_data.coverage)
    expected_coverage = {
        "selected_maps": 3524,
        "development_maps": 3521,
        "sealed_metadata_maps": 0,
        "quarantined_maps": 3,
        "development_series": 1419,
        "sealed_metadata_series": 0,
        "quarantined_clusters": 2,
    }
    if any(coverage.get(key) != value for key, value in expected_coverage.items()):
        raise CorrectedSourcePreflightError(
            "corrected source coverage differs from the pre-boundary rehearsal"
        )
    folds = _fold_counts(input_data)
    if folds != [
        {"fold": "TRAIN", "series": 528, "maps": 1234},
        {"fold": "DEVELOPMENT", "series": 270, "maps": 758},
        {"fold": "VALIDATION", "series": 621, "maps": 1529},
    ]:
        raise CorrectedSourcePreflightError("corrected source folds changed")
    if replay.applied_series != 1419 or replay.applied_maps != 3521:
        raise CorrectedSourcePreflightError("corrected source replay did not reconcile")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "built_at_utc": build_time.isoformat(),
        "lineage": {
            "supersedes_failed_protocol_version": 1,
            "protocol_v1_raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "protocol_v1_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "source_failure_v1_raw_sha256": REGISTERED_PREFLIGHT_RAW_SHA256,
            "source_failure_v1_artifact_sha256": REGISTERED_PREFLIGHT_ARTIFACT_SHA256,
            "source_failure_v1_result_state": failure_v1["result_state"],
            "future_outcomes_used_for_remediation": False,
        },
        "source_snapshot": {
            "package_id": PACKAGE_ID,
            "manifest_locator": MANIFEST_LOCATOR.as_posix(),
            "manifest_raw_sha256": MANIFEST_RAW_SHA256,
            "manifest_canonical_sha256": MANIFEST_CANONICAL_SHA256,
            "maps": files["maps"],
            "players": files["players"],
            "latest_observed_source_time": dates.max().isoformat(),
            "playoffs_dtype": playoffs_dtype,
        },
        "future_boundary": {
            "start_inclusive_source_time": FUTURE_SEALED_START.isoformat(),
            "future_holdout_maps_present": 0,
            "future_holdout_targets_accessed": False,
        },
        "locked_candidate": {
            "candidate_id": candidate.candidate_id,
            "definition": protocol_v1["locked_candidate"]["definition"],
            "selection_remains_adaptive": True,
        },
        "adapter_preflight": {
            "coverage": coverage,
            "folds": folds,
            "development_selected_rows_sha256": (
                input_data.development_selected_rows_sha256
            ),
            "player_selected_metadata_sha256": (
                input_data.player_selected_metadata_sha256
            ),
            "cluster_partition_sha256": input_data.cluster_partition_sha256,
        },
        "numerical_preflight": {
            "applied_series": replay.applied_series,
            "applied_maps": replay.applied_maps,
            "prediction_rows": len(replay.predictions),
            "prediction_rows_sha256": _canonical_sha256(replay.predictions),
            "latent_dimension": len(replay.state.keys),
            "posterior_state_sha256": _state_sha256(replay),
            "posterior_psd": psd,
            "players_with_available_state": len(replay.player_metadata),
            "teams_with_available_state": len(replay.team_lineups),
            "teams_with_home_league": len(replay.team_home_leagues),
            "bridge_diagnostics": replay.bridge_diagnostics,
            "roster_transition_diagnostics": replay.roster_transition_diagnostics,
        },
        "adaptation_disclosure": {
            "all_snapshot_outcomes_are_adaptive_development": True,
            "preflight_is_not_independent_validation": True,
            "preflight_is_not_a_holdout_opening": True,
            "candidate_was_selected_before_this_rehearsal": True,
        },
        "source_locks": [_source_record(root, locator) for locator in SOURCE_LOCKS],
        "decision_outputs": {
            "player_ratings": None,
            "team_ratings": None,
            "match_probability": None,
            "fair_odds": None,
            "expected_value": None,
            "bet_recommendation": None,
        },
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": (
            "This artifact proves only that the frozen candidate can be fit on the "
            "corrected, pre-boundary source. It is adaptive numerical rehearsal, not "
            "independent validation or rating, probability, recommendation, or betting authority."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_corrected_source_preflight(payload, root=root)


def validate_corrected_source_preflight(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CorrectedSourcePreflightError("corrected source preflight must be an object")
    value = dict(payload)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise CorrectedSourcePreflightError("corrected source preflight identity changed")
    declared = value.get("artifact_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    if not isinstance(declared, str) or declared != _canonical_sha256(unsigned):
        raise CorrectedSourcePreflightError("corrected source canonical hash mismatch")
    _parse_built_at(str(value.get("built_at_utc")))

    source = validate_registered_source_snapshot_v2(root=root)
    protocol_v1 = validate_registered_future_protocol(root=root)
    failure_v1 = validate_registered_source_preflight_v1(root=root)
    source_record = value.get("source_snapshot") or {}
    if (
        source_record.get("package_id") != source.get("package_id")
        or source_record.get("manifest_raw_sha256") != MANIFEST_RAW_SHA256
        or source_record.get("manifest_canonical_sha256") != MANIFEST_CANONICAL_SHA256
        or source_record.get("playoffs_dtype") not in {"bool", "boolean"}
    ):
        raise CorrectedSourcePreflightError("corrected source binding changed")
    lineage = value.get("lineage") or {}
    if (
        lineage.get("protocol_v1_artifact_sha256")
        != protocol_v1.get("artifact_sha256")
        or lineage.get("source_failure_v1_artifact_sha256")
        != failure_v1.get("artifact_sha256")
        or lineage.get("future_outcomes_used_for_remediation") is not False
    ):
        raise CorrectedSourcePreflightError("preflight lineage changed")
    future = value.get("future_boundary") or {}
    if future != {
        "start_inclusive_source_time": FUTURE_SEALED_START.isoformat(),
        "future_holdout_maps_present": 0,
        "future_holdout_targets_accessed": False,
    }:
        raise CorrectedSourcePreflightError("future outcome isolation changed")
    candidate = value.get("locked_candidate") or {}
    if (
        candidate.get("candidate_id")
        != protocol_v1["locked_candidate"]["candidate_id"]
        or candidate.get("definition")
        != protocol_v1["locked_candidate"]["definition"]
        or candidate.get("selection_remains_adaptive") is not True
    ):
        raise CorrectedSourcePreflightError("locked candidate binding changed")
    adaptation = value.get("adaptation_disclosure") or {}
    if set(adaptation) != {
        "all_snapshot_outcomes_are_adaptive_development",
        "preflight_is_not_independent_validation",
        "preflight_is_not_a_holdout_opening",
        "candidate_was_selected_before_this_rehearsal",
    } or any(item is not True for item in adaptation.values()):
        raise CorrectedSourcePreflightError("adaptation disclosure changed")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(
        authority.get(name) is not False for name in AUTHORITY_KEYS
    ):
        raise CorrectedSourcePreflightError("corrected preflight exceeds authority")
    if any(item is not None for item in (value.get("decision_outputs") or {}).values()):
        raise CorrectedSourcePreflightError("corrected preflight contains decision outputs")
    records = value.get("source_locks")
    if not isinstance(records, list) or len(records) != len(SOURCE_LOCKS):
        raise CorrectedSourcePreflightError("preflight source inventory changed")
    if [record.get("locator") for record in records if isinstance(record, Mapping)] != list(SOURCE_LOCKS):
        raise CorrectedSourcePreflightError("preflight source order changed")
    for record in records:
        locator = str(record["locator"])
        path = root / locator
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256(path) != record.get("raw_sha256")
        ):
            raise CorrectedSourcePreflightError(
                f"corrected preflight source drifted: {locator}"
            )
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
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
            raise FileExistsError(
                f"refusing to replace corrected source preflight: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--built-at", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_corrected_source_preflight(built_at=args.built_at)
    raw_sha256 = write_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "result_state": payload["result_state"],
                "latent_dimension": payload["numerical_preflight"]["latent_dimension"],
                "rating_authority": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_KEYS",
    "CorrectedSourcePreflightError",
    "DEFAULT_OUTPUT",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "build_corrected_source_preflight",
    "validate_corrected_source_preflight",
    "write_no_clobber",
]
