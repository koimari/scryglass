"""Contamination-aware protocol lock for second-generation private ratings.

The first multi-league experiment exposed its 2026-Q1 validation results.  It
would therefore be scientifically invalid to tune a successor in response and
continue calling that same period an independent validation set.  This module
records the disclosure, reclassifies every pre-2026-04-01 outcome as adaptive
development evidence, and binds the still outcome-free post-2026-04-01 cohort
as the only final temporal holdout.

This file locks a candidate family and decision rules; it does not fit a model,
open sealed outcomes, authorize probabilities, or authorize betting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from . import multileague_benchmark as benchmark
from . import multileague_development as adapter
from . import multileague_runner as rating


SCHEMA_VERSION = "scryglass:multileague-rating-v2-protocol-lock:v1"
RESULT_STATE = "PROTOCOL_LOCKED_SEALED_FINAL_UNOPENED"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v2/protocol-lock-v1.json"
)
RATING_ARTIFACT = Path(
    "data/lol/v2/models/player/multileague-v1/private-development-artifact-v1.json"
)
BENCHMARK_ARTIFACT = Path(
    "data/lol/v2/models/player/multileague-v1/strong-baseline-benchmark-v1.json"
)
SOURCE_LOCATOR = "lol_kills/v2/ratings/player/multileague_v2_protocol.py"

DISCOVERY_WINDOWS = (
    {
        "window_id": "adaptive-2025q3",
        "start_inclusive": "2025-07-01T00:00:00",
        "end_exclusive": "2025-10-01T00:00:00",
        "historical_fold_label": "DEVELOPMENT",
    },
    {
        "window_id": "adaptive-2025q4",
        "start_inclusive": "2025-10-01T00:00:00",
        "end_exclusive": "2026-01-01T00:00:00",
        "historical_fold_label": "DEVELOPMENT",
    },
    {
        "window_id": "adaptive-2026q1-observed",
        "start_inclusive": "2026-01-01T00:00:00",
        "end_exclusive": "2026-04-01T00:00:00",
        "historical_fold_label": "VALIDATION",
    },
)

ORGANIZATION_WEIGHTS = (0.25, 0.5, 1.0)
ORGANIZATION_PRIOR_VARIANCES = (0.25, 1.0)
ORGANIZATION_ROSTER_RETENTION_FLOORS = (0.5, 1.0)


class MultiLeagueV2ProtocolError(ValueError):
    """The protocol lock is malformed, stale, or no longer source-bound."""


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
        raise MultiLeagueV2ProtocolError(
            "protocol contains a non-canonical or non-finite value"
        ) from error


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MultiLeagueV2ProtocolError(f"{label} must be a lowercase SHA-256")
    return value


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MultiLeagueV2ProtocolError(f"cannot read bound artifact: {path}") from error
    if not isinstance(value, dict):
        raise MultiLeagueV2ProtocolError(f"bound artifact is not an object: {path}")
    return raw, value


def _source_record(root: Path, locator: str, kind: str) -> dict[str, Any]:
    path = root / locator
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise MultiLeagueV2ProtocolError(f"bound source is unavailable: {locator}") from error
    return {
        "kind": kind,
        "locator": locator,
        "bytes": len(raw),
        "raw_sha256": _sha256(raw),
    }


def _candidate_payloads() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for organization_weight in ORGANIZATION_WEIGHTS:
        for organization_prior_variance in ORGANIZATION_PRIOR_VARIANCES:
            for retention_floor in ORGANIZATION_ROSTER_RETENTION_FLOORS:
                candidate_id = (
                    f"hierarchical-orgw{int(organization_weight * 100):03d}"
                    f"-orgv{int(organization_prior_variance * 100):03d}"
                    f"-retain{int(retention_floor * 100):03d}"
                )
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "latent_components": [
                            "player",
                            "organization_residual",
                            "bridge_identified_home_league",
                            "blue_side",
                        ],
                        "player_weight_per_role": 0.2,
                        "player_prior_variance": 1.0,
                        "player_process_variance_per_day": 0.0005,
                        "organization_weight": organization_weight,
                        "organization_prior_variance": organization_prior_variance,
                        "organization_process_variance_per_day": 0.0005,
                        "organization_roster_retention": {
                            "kind": "retained_player_fraction_with_floor",
                            "floor": retention_floor,
                            "formula": "floor_plus_one_minus_floor_times_retained_exact_players_over_five",
                        },
                        "lineup_synergy_component": {
                            "status": "UNAVAILABLE",
                            "value": None,
                        },
                        "team_policy_component": {
                            "status": "UNAVAILABLE",
                            "value": None,
                        },
                    }
                )
    return candidates


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    return _sha256(_canonical_bytes(body))


def build_protocol_lock(
    root: Path | str = Path("."),
    *,
    locked_at: str,
) -> dict[str, Any]:
    repo_root = Path(root)
    rating_raw, rating_artifact = _read_object(repo_root / RATING_ARTIFACT)
    benchmark_raw, benchmark_artifact = _read_object(repo_root / BENCHMARK_ARTIFACT)
    try:
        rating.validate_multileague_development_artifact(rating_artifact)
        benchmark.validate_strong_baseline_benchmark(benchmark_artifact)
    except (rating.MultiLeagueRunnerError, benchmark.MultiLeagueBenchmarkError) as error:
        raise MultiLeagueV2ProtocolError("a predecessor evidence artifact is invalid") from error

    rating_selection = rating_artifact.get("selection") or {}
    benchmark_selection = benchmark_artifact.get("selection") or {}
    if (
        rating_selection.get("sealed_final_opened") is not False
        or benchmark_selection.get("sealed_final_opened") is not False
    ):
        raise MultiLeagueV2ProtocolError("sealed-final isolation was already lost")
    rating_input = rating_artifact.get("input") or {}
    coverage = rating_input.get("coverage") or {}
    if coverage.get("sealed_metadata_series") != 398:
        raise MultiLeagueV2ProtocolError("sealed metadata cohort count changed")

    records = [
        _source_record(repo_root, adapter.DEFAULT_MAPS_LOCATOR, "warehouse_maps"),
        _source_record(repo_root, adapter.DEFAULT_PLAYERS_LOCATOR, "warehouse_players"),
        _source_record(repo_root, RATING_ARTIFACT.as_posix(), "predecessor_rating_artifact"),
        _source_record(repo_root, BENCHMARK_ARTIFACT.as_posix(), "predecessor_benchmark_artifact"),
        _source_record(
            repo_root,
            "lol_kills/v2/ratings/player/multileague_development.py",
            "input_adapter_source",
        ),
        _source_record(
            repo_root,
            "lol_kills/v2/ratings/player/multileague_runner.py",
            "predecessor_runner_source",
        ),
        _source_record(
            repo_root,
            "lol_kills/v2/ratings/player/multileague_benchmark.py",
            "benchmark_source",
        ),
        _source_record(repo_root, SOURCE_LOCATOR, "protocol_source"),
    ]
    by_locator = {item["locator"]: item for item in records}
    if by_locator[adapter.DEFAULT_MAPS_LOCATOR]["raw_sha256"] != rating_input.get(
        "maps_sha256"
    ):
        raise MultiLeagueV2ProtocolError("warehouse maps no longer match predecessor pin")
    if by_locator[adapter.DEFAULT_PLAYERS_LOCATOR]["raw_sha256"] != rating_input.get(
        "players_sha256"
    ):
        raise MultiLeagueV2ProtocolError("warehouse players no longer match predecessor pin")
    if by_locator[RATING_ARTIFACT.as_posix()]["raw_sha256"] != _sha256(rating_raw):
        raise MultiLeagueV2ProtocolError("rating artifact byte pin is inconsistent")
    if by_locator[BENCHMARK_ARTIFACT.as_posix()]["raw_sha256"] != _sha256(benchmark_raw):
        raise MultiLeagueV2ProtocolError("benchmark artifact byte pin is inconsistent")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "scryglass:multileague-rating-v2:adaptive-lock-2026-08-01",
        "locked_at": locked_at,
        "result_state": RESULT_STATE,
        "source_locks": records,
        "input_binding": {
            "maps_sha256": rating_input.get("maps_sha256"),
            "players_sha256": rating_input.get("players_sha256"),
            "cluster_partition_sha256": rating_input.get("cluster_partition_sha256"),
            "development_selected_rows_sha256": rating_input.get(
                "development_selected_rows_sha256"
            ),
            "sealed_selected_metadata_sha256": rating_input.get(
                "sealed_selected_metadata_sha256"
            ),
            "predecessor_rating_raw_sha256": _sha256(rating_raw),
            "predecessor_rating_artifact_sha256": rating_artifact.get(
                "artifact_sha256"
            ),
            "predecessor_benchmark_raw_sha256": _sha256(benchmark_raw),
            "predecessor_benchmark_artifact_sha256": benchmark_artifact.get(
                "artifact_sha256"
            ),
        },
        "information_boundary": {
            "outcomes_available_for_adaptive_development_before": "2026-04-01T00:00:00",
            "sealed_final_start_inclusive": "2026-04-01T00:00:00",
            "sealed_final_end_inclusive_metadata": (
                rating_artifact.get("freshness") or {}
            ).get("latest_outcome_free_sealed_metadata_source_local_end"),
            "sealed_final_series": coverage.get("sealed_metadata_series"),
            "sealed_final_maps": coverage.get("sealed_metadata_maps"),
            "sealed_final_targets_accessed": False,
            "sealed_final_metadata_only": True,
            "source_time_semantics": "timezone-naive warehouse timestamp",
            "availability_embargo_hours": rating.AVAILABILITY_EMBARGO_HOURS,
        },
        "validation_disclosure": {
            "status": "RECLASSIFIED_AS_ADAPTIVE_DEVELOPMENT",
            "disclosed_on": "2026-08-01",
            "reason": (
                "the predecessor 2026-Q1 validation metrics and roster-change failures "
                "were inspected before this successor protocol was locked"
            ),
            "prohibited_claim": (
                "no successor result on any pre-2026-04-01 outcome may be described as "
                "independent validation or final holdout evidence"
            ),
        },
        "adaptive_development": {
            "warm_start": {
                "start_inclusive": "2025-01-01T00:00:00",
                "end_exclusive": "2025-07-01T00:00:00",
            },
            "windows": list(DISCOVERY_WINDOWS),
            "series_frozen_predictions": True,
            "outcome_updates_strictly_after_series_end_plus_hours": (
                rating.AVAILABILITY_EMBARGO_HOURS
            ),
            "selection_unit": "series_macro",
            "dependence_unit": "source_or_derived_series_cluster",
            "selection_rule": {
                "stage_1_finite_psd_replay_required": True,
                "stage_2_each_window_metrics_required": ["log_loss", "brier"],
                "stage_3_rank": [
                    "minimum_worst_window_log_loss_regret_against_better_baseline",
                    "minimum_pooled_series_macro_log_loss",
                    "minimum_pooled_series_macro_brier",
                    "candidate_id",
                ],
                "better_baseline_per_window": (
                    "minimum_loss_of_player_random_walk_and_organization_random_walk"
                ),
                "maximum_allowed_worst_window_log_loss_regret": 0.01,
                "minimum_series_per_scored_window": 20,
                "selection_is_exploratory_not_authority": True,
            },
        },
        "baselines": [
            {
                "baseline_id": "constant-half",
                "probability": 0.5,
            },
            {
                "baseline_id": "predecessor-player-random-walk",
                "candidate_id": "random_walk_no_reset",
                "artifact_sha256": rating_artifact.get("artifact_sha256"),
            },
            {
                "baseline_id": "predecessor-organization-random-walk",
                "candidate_id": benchmark_selection.get("organization_candidate_id"),
                "artifact_sha256": benchmark_artifact.get("artifact_sha256"),
            },
        ],
        "candidate_family": {
            "kind": "full_covariance_dynamic_player_plus_organization_residual",
            "candidates": _candidate_payloads(),
            "league_bridge_policy": (
                "home_league_effects_enter only when identities were available before "
                "an MSI or EWC bridge series"
            ),
            "identifiability": (
                "player and organization decomposition is prior-regularized; only their "
                "joint predictive contrast is treated as identified by map outcomes"
            ),
            "missing_component_policy": "unavailable_is_null_and_never_zero",
        },
        "sealed_final_gate": {
            "opened": False,
            "opening_requires_independent_approval_receipt": True,
            "opening_requires_locked_winner_artifact_sha256": True,
            "one_time_evaluation": True,
            "bootstrap_samples": 10000,
            "bootstrap_seed": 20260821,
            "cluster_unit": "source_or_derived_series_cluster",
            "minimum_series": {
                "overall": 100,
                "each_domestic_league": 20,
                "one_or_both_rosters_changed": 20,
            },
            "required_comparators": [
                "predecessor-player-random-walk",
                "predecessor-organization-random-walk",
            ],
            "required_metrics": ["log_loss", "brier"],
            "pass_rule": (
                "candidate_minus_each_comparator upper 95 percent series-cluster "
                "bootstrap bound must be nonpositive overall, for LCS, and for the "
                "one-or-both-rosters-changed stratum"
            ),
            "reliability_required": True,
            "failure_policy": "remain_unavailable_and_do_not_open_another_holdout",
            "passing_is_necessary_not_sufficient_for_probability_or_betting": True,
        },
        "post_holdout_authorities_still_required": [
            "independent_model_review",
            "calibration_transform_authority",
            "exact_pre_event_roster_registry",
            "event_bound_rating_registry",
            "freshness_and_patch_authority",
            "market_definition_authority",
            "immutable_bookmaker_quote_registry",
            "uncertainty_aware_decision_policy",
        ],
        "claim_ceiling": {
            "development_diagnostic": True,
            "production_rating": False,
            "match_probability": False,
            "fair_odds": False,
            "expected_value": False,
            "bet_recommendation": False,
            "public_betting_surface": False,
        },
        "decision_outputs": {
            "match_probability": None,
            "fair_odds": None,
            "expected_value": None,
            "bet_recommendation": None,
        },
    }
    payload["artifact_sha256"] = _artifact_sha256(payload)
    return validate_protocol_lock(payload, root=repo_root)


def validate_protocol_lock(
    payload: Mapping[str, Any],
    *,
    root: Path | str = Path("."),
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise MultiLeagueV2ProtocolError("protocol lock must be an object")
    value = dict(payload)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise MultiLeagueV2ProtocolError("protocol schema version is unsupported")
    if value.get("result_state") != RESULT_STATE:
        raise MultiLeagueV2ProtocolError("protocol result state is not locked and sealed")
    declared = _require_sha256(value.get("artifact_sha256"), "artifact_sha256")
    if declared != _artifact_sha256(value):
        raise MultiLeagueV2ProtocolError("protocol artifact digest mismatch")

    boundary = value.get("information_boundary") or {}
    disclosure = value.get("validation_disclosure") or {}
    final_gate = value.get("sealed_final_gate") or {}
    if (
        boundary.get("sealed_final_targets_accessed") is not False
        or boundary.get("sealed_final_metadata_only") is not True
        or final_gate.get("opened") is not False
        or disclosure.get("status") != "RECLASSIFIED_AS_ADAPTIVE_DEVELOPMENT"
    ):
        raise MultiLeagueV2ProtocolError("sealed or contamination semantics changed")
    if (value.get("candidate_family") or {}).get("candidates") != _candidate_payloads():
        raise MultiLeagueV2ProtocolError("locked hierarchical candidate family changed")
    windows = (value.get("adaptive_development") or {}).get("windows")
    if windows != list(DISCOVERY_WINDOWS):
        raise MultiLeagueV2ProtocolError("adaptive discovery windows changed")
    decision_outputs = value.get("decision_outputs") or {}
    if set(decision_outputs) != {
        "match_probability",
        "fair_odds",
        "expected_value",
        "bet_recommendation",
    } or any(item is not None for item in decision_outputs.values()):
        raise MultiLeagueV2ProtocolError("protocol lock cannot contain decision outputs")
    claim_ceiling = value.get("claim_ceiling") or {}
    if any(
        claim_ceiling.get(name) is not False
        for name in (
            "production_rating",
            "match_probability",
            "fair_odds",
            "expected_value",
            "bet_recommendation",
            "public_betting_surface",
        )
    ):
        raise MultiLeagueV2ProtocolError("protocol claim ceiling became authorizing")

    records = value.get("source_locks")
    if not isinstance(records, list) or len(records) != 8:
        raise MultiLeagueV2ProtocolError("protocol source-lock inventory changed")
    repo_root = Path(root)
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise MultiLeagueV2ProtocolError("source-lock record is malformed")
        locator = record.get("locator")
        if not isinstance(locator, str) or not locator or locator in seen:
            raise MultiLeagueV2ProtocolError("source-lock locator is invalid or repeated")
        seen.add(locator)
        expected = _require_sha256(record.get("raw_sha256"), f"{locator} raw_sha256")
        try:
            raw = (repo_root / locator).read_bytes()
        except OSError as error:
            raise MultiLeagueV2ProtocolError(f"bound source is unavailable: {locator}") from error
        if len(raw) != record.get("bytes") or _sha256(raw) != expected:
            raise MultiLeagueV2ProtocolError(f"bound source drifted: {locator}")
    return value


def write_protocol_lock_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(dict(payload), indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace existing protocol lock: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _sha256(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-at", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_protocol_lock(locked_at=args.locked_at)
    raw_sha256 = write_protocol_lock_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "candidates": len(payload["candidate_family"]["candidates"]),
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
    "DISCOVERY_WINDOWS",
    "MultiLeagueV2ProtocolError",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "build_protocol_lock",
    "validate_protocol_lock",
    "write_protocol_lock_no_clobber",
]
