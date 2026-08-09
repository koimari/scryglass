"""Corrected-source adaptive diagnostic for the frozen v3 rating candidate.

The v2 candidate family was selected before the corrected immutable source
snapshot existed.  This module therefore replays the complete, already
declared family on the corrected pre-2026-08-03 source and asks a deliberately
narrow question: is there strong enough *adaptive* evidence to justify
superseding the already frozen v3 candidate before prospective collection?

It cannot answer whether any rating is valid.  Every outcome used here was
available before this diagnostic, the reliability gate is not opened, and all
decision outputs remain null.  The only admissible decisions are to retain the
incumbent or to record that a separate superseding protocol is required.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from lol_kills.v2.evaluation.bootstrap import series_cluster_bootstrap

from . import multileague_benchmark as benchmark
from . import multileague_development as adapter
from . import multileague_runner as rating
from . import multileague_v2_protocol as candidate_family
from . import multileague_v2_runner as hierarchical
from .multileague_v3_future_protocol import FUTURE_SEALED_START
from .multileague_v3_preflight_v3_registry import (
    REGISTERED_PREFLIGHT_ARTIFACT_SHA256,
    REGISTERED_PREFLIGHT_RAW_SHA256,
    validate_registered_source_preflight_v3,
)
from .multileague_v3_registry_v3 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_future_protocol_v3,
)
from .multileague_v3_source_registry_v2 import (
    MANIFEST_CANONICAL_SHA256,
    MANIFEST_RAW_SHA256,
    PACKAGE_ID,
    validate_registered_source_snapshot_v2,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:multileague-rating-v3-corrected-adaptive-diagnostic:v1"
RESULT_STATE = "INCUMBENT_RETAINED_NO_ADAPTIVE_SUPERSESSION_EVIDENCE"
SOURCE_LOCATOR = (
    "lol_kills/v2/ratings/player/"
    "multileague_v3_corrected_adaptive_diagnostic_v1.py"
)
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v3/"
    "corrected-adaptive-diagnostic-v1.json"
)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260802
MINIMUM_SERIES = 20
EVALUATION_FOLD = "VALIDATION"
DOMESTIC_LEAGUES = ("LCS", "LEC", "LCK", "LPL")
ROSTER_CHANGE_STRATUM = "ONE_OR_BOTH_ROSTERS_CHANGED"
SOURCE_LOCKS = (
    SOURCE_LOCATOR,
    "lol_kills/v2/ratings/player/multileague_development.py",
    "lol_kills/v2/ratings/player/multileague_runner.py",
    "lol_kills/v2/ratings/player/multileague_benchmark.py",
    "lol_kills/v2/ratings/player/multileague_v2_protocol.py",
    "lol_kills/v2/ratings/player/multileague_v2_runner.py",
    "lol_kills/v2/ratings/player/multileague_v3_source_registry_v2.py",
    "lol_kills/v2/ratings/player/multileague_v3_preflight_v3_registry.py",
    "lol_kills/v2/ratings/player/multileague_v3_registry_v3.py",
)
AUTHORITY_KEYS = (
    "model_validation_authority",
    "player_rating_authority",
    "team_rating_authority",
    "probability_authority",
    "recommendation_authority",
    "betting_authority",
)


class CorrectedAdaptiveDiagnosticError(RuntimeError):
    """The diagnostic is malformed, stale, or crossed its boundary."""


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
        raise CorrectedAdaptiveDiagnosticError(
            "diagnostic value is not canonical"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        raise CorrectedAdaptiveDiagnosticError(
            f"bound diagnostic source unavailable: {locator}"
        )
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise CorrectedAdaptiveDiagnosticError(
            f"{label} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise CorrectedAdaptiveDiagnosticError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


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


def _evaluation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    league: str | None = None,
    roster_change_stratum: str | None = None,
    international: bool = False,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("fold_id") == EVALUATION_FOLD
        and (league is None or row.get("league") == league)
        and (
            not international
            or row.get("league") in adapter.INTERNATIONAL_LEAGUES
        )
        and (
            roster_change_stratum is None
            or row.get("roster_change_stratum") == roster_change_stratum
        )
    ]


def _strata() -> tuple[tuple[str, dict[str, Any]], ...]:
    return (
        ("overall", {}),
        *((league, {"league": league}) for league in DOMESTIC_LEAGUES),
        (
            "one_or_both_rosters_changed",
            {"roster_change_stratum": ROSTER_CHANGE_STRATUM},
        ),
        ("international", {"international": True}),
    )


def _point_metrics(
    rows: Sequence[Mapping[str, Any]], **selector: Any
) -> dict[str, Any]:
    selected = _evaluation_rows(rows, **selector)
    if not selected:
        raise CorrectedAdaptiveDiagnosticError("diagnostic stratum has no rows")
    metrics = rating._metric_payload(selected)
    return {
        "maps": metrics["maps"],
        "series": metrics["series"],
        "map_weighted": metrics["map_weighted"],
        "series_macro": metrics["series_macro"],
        "calibration": metrics["calibration"],
    }


def _paired_difference(
    candidate_rows: Sequence[Mapping[str, Any]],
    comparator_rows: Sequence[Mapping[str, Any]],
    **selector: Any,
) -> dict[str, Any]:
    candidate = {
        str(row["game_id"]): row
        for row in _evaluation_rows(candidate_rows, **selector)
    }
    comparator = {
        str(row["game_id"]): row
        for row in _evaluation_rows(comparator_rows, **selector)
    }
    if set(candidate) != set(comparator) or not candidate:
        raise CorrectedAdaptiveDiagnosticError(
            "candidate/comparator diagnostic populations differ"
        )
    game_ids = sorted(candidate)
    clusters = [str(candidate[game_id]["series_id"]) for game_id in game_ids]
    series = len(set(clusters))
    results: dict[str, Any] = {}
    for offset, metric in enumerate(("log_loss", "brier")):
        deltas = [
            rating._loss(
                float(candidate[game_id]["probability"]),
                int(candidate[game_id]["outcome"]),
                metric,
            )
            - rating._loss(
                float(comparator[game_id]["probability"]),
                int(comparator[game_id]["outcome"]),
                metric,
            )
            for game_id in game_ids
        ]
        boot = series_cluster_bootstrap(
            deltas,
            clusters,
            [True] * len(game_ids),
            row_ids=game_ids,
            n_boot=BOOTSTRAP_SAMPLES,
            random_seed=BOOTSTRAP_SEED + offset,
            cluster_unit="source-or-derived-series-dependence-cluster",
        )
        sizes: dict[str, int] = {}
        for size in boot.cluster_size_distribution.values():
            key = str(int(size))
            sizes[key] = sizes.get(key, 0) + 1
        results[metric] = {
            "candidate_minus_comparator": boot.point,
            "lower_95": boot.lower_95,
            "upper_95": boot.upper_95,
            "cluster_count": boot.cluster_count,
            "cluster_size_distribution": dict(
                sorted(sizes.items(), key=lambda item: int(item[0]))
            ),
        }
    enough = series >= MINIMUM_SERIES
    passed = enough and all(item["upper_95"] <= 0.0 for item in results.values())
    return {
        "status": (
            "PASS_NONPOSITIVE_UPPER_95"
            if passed
            else "FAIL_INSUFFICIENT_SERIES"
            if not enough
            else "FAIL_UPPER_95_ABOVE_ZERO"
        ),
        "maps": len(game_ids),
        "series": series,
        "minimum_required_series": MINIMUM_SERIES,
        "log_loss": results["log_loss"],
        "brier": results["brier"],
    }


def _rank_candidates(
    candidate_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    comparator_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    baselines = {
        comparator_id: {
            stratum: _point_metrics(rows, **selector)
            for stratum, selector in _strata()
        }
        for comparator_id, rows in comparator_rows.items()
    }
    reports: list[dict[str, Any]] = []
    for candidate_id, rows in candidate_rows.items():
        strata = {
            stratum: _point_metrics(rows, **selector)
            for stratum, selector in _strata()
        }
        regrets: list[dict[str, Any]] = []
        for stratum, _selector in _strata():
            for metric in ("log_loss", "brier"):
                candidate_loss = float(strata[stratum]["map_weighted"][metric])
                better_baseline = min(
                    float(payload[stratum]["map_weighted"][metric])
                    for payload in baselines.values()
                )
                regrets.append(
                    {
                        "stratum": stratum,
                        "metric": metric,
                        "candidate_loss": candidate_loss,
                        "better_comparator_loss": better_baseline,
                        "regret": candidate_loss - better_baseline,
                    }
                )
        worst = max(
            regrets,
            key=lambda item: (float(item["regret"]), item["stratum"], item["metric"]),
        )
        reports.append(
            {
                "candidate_id": candidate_id,
                "strata": strata,
                "worst_point_regret_against_better_comparator": worst,
                "rank_key": {
                    "maximum_point_regret": max(0.0, float(worst["regret"])),
                    "overall_map_weighted_log_loss": strata["overall"][
                        "map_weighted"
                    ]["log_loss"],
                    "overall_map_weighted_brier": strata["overall"][
                        "map_weighted"
                    ]["brier"],
                    "candidate_id": candidate_id,
                },
            }
        )
    return sorted(
        reports,
        key=lambda item: (
            float(item["rank_key"]["maximum_point_regret"]),
            float(item["rank_key"]["overall_map_weighted_log_loss"]),
            float(item["rank_key"]["overall_map_weighted_brier"]),
            str(item["candidate_id"]),
        ),
    )


def build_corrected_adaptive_diagnostic(
    *, built_at: str, root: Path = ROOT
) -> dict[str, Any]:
    build_time = _timestamp(built_at, "built_at")
    boundary = FUTURE_SEALED_START.replace(tzinfo=timezone.utc)
    if build_time >= boundary:
        raise CorrectedAdaptiveDiagnosticError(
            "diagnostic must be built before the future boundary"
        )
    source = validate_registered_source_snapshot_v2(root=root)
    protocol = validate_registered_future_protocol_v3(root=root)
    preflight = validate_registered_source_preflight_v3(root=root)
    files = source["files"]
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
            raise CorrectedAdaptiveDiagnosticError(
                "corrected snapshot unexpectedly crosses the future boundary"
            )
        player = rating._replay(
            input_data,
            next(
                item
                for item in rating.CANDIDATES
                if item.candidate_id == "random_walk_no_reset"
            ),
        )
        organization = benchmark._organization_replay(
            input_data,
            next(
                item
                for item in benchmark.ORGANIZATION_CANDIDATES
                if item.candidate_id == "organization_random_walk_no_reset"
            ),
        )
        player_rows = benchmark._attach_roster_strata(
            player.predictions, organization.predictions
        )
        comparator_rows = {
            "predecessor-player-random-walk": player_rows,
            "predecessor-organization-random-walk": organization.predictions,
        }
        replays: dict[str, hierarchical.ReplayResult] = {}
        candidate_rows: dict[str, list[dict[str, Any]]] = {}
        for definition in candidate_family._candidate_payloads():
            spec = hierarchical.CandidateSpec.from_payload(definition)
            replay = hierarchical.replay_candidate(input_data, spec)
            replay.state.assert_psd()
            replays[spec.candidate_id] = replay
            candidate_rows[spec.candidate_id] = benchmark._attach_roster_strata(
                replay.predictions, organization.predictions
            )

    ranked = _rank_candidates(candidate_rows, comparator_rows)
    incumbent_id = str(protocol["locked_candidate"]["candidate_id"])
    if incumbent_id not in candidate_rows:
        raise CorrectedAdaptiveDiagnosticError(
            "registered incumbent is outside the declared candidate family"
        )
    adaptive_challenger_id = str(ranked[0]["candidate_id"])
    incumbent_rows = candidate_rows[incumbent_id]
    challenger_rows = candidate_rows[adaptive_challenger_id]
    incumbent_vs_comparators = {
        comparator_id: {
            stratum: _paired_difference(
                incumbent_rows, rows, **selector
            )
            for stratum, selector in _strata()
        }
        for comparator_id, rows in comparator_rows.items()
    }
    challenger_vs_incumbent = {
        stratum: _paired_difference(
            challenger_rows, incumbent_rows, **selector
        )
        for stratum, selector in _strata()
    }
    required_superiority = (
        "overall",
        "LCS",
        "one_or_both_rosters_changed",
    )
    challenger_superior = all(
        challenger_vs_incumbent[stratum]["status"]
        == "PASS_NONPOSITIVE_UPPER_95"
        for stratum in required_superiority
    )
    if challenger_superior:
        raise CorrectedAdaptiveDiagnosticError(
            "adaptive challenger met the supersession threshold; a new protocol is required"
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "built_at_utc": build_time.isoformat(),
        "information_boundary": {
            "future_holdout_start_inclusive": boundary.isoformat(),
            "source_snapshot_latest_observation": source[
                "created_from_refresh_at_utc"
            ],
            "future_holdout_maps_present": 0,
            "future_holdout_targets_accessed": False,
            "all_evaluated_outcomes_are_adaptive": True,
            "evaluation_fold_label": EVALUATION_FOLD,
            "evaluation_fold_is_not_independent_validation": True,
        },
        "bindings": {
            "source_package_id": PACKAGE_ID,
            "source_manifest_raw_sha256": MANIFEST_RAW_SHA256,
            "source_manifest_canonical_sha256": MANIFEST_CANONICAL_SHA256,
            "protocol_raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "preflight_raw_sha256": REGISTERED_PREFLIGHT_RAW_SHA256,
            "preflight_artifact_sha256": REGISTERED_PREFLIGHT_ARTIFACT_SHA256,
            "preflight_posterior_state_sha256": preflight[
                "numerical_preflight"
            ]["posterior_state_sha256"],
        },
        "population": {
            "coverage": dict(input_data.coverage),
            "cluster_partition_sha256": input_data.cluster_partition_sha256,
            "development_selected_rows_sha256": (
                input_data.development_selected_rows_sha256
            ),
            "player_selected_metadata_sha256": (
                input_data.player_selected_metadata_sha256
            ),
        },
        "candidate_family": {
            "status": "previously_declared_family_replayed_adaptively",
            "definitions": candidate_family._candidate_payloads(),
            "rank_rule": [
                "minimum_nonnegative_worst_point_regret_against_better_comparator_across_locked_strata_and_metrics",
                "minimum_overall_map_weighted_log_loss",
                "minimum_overall_map_weighted_brier",
                "candidate_id",
            ],
            "ranked_results": ranked,
        },
        "incumbent": {
            "candidate_id": incumbent_id,
            "definition": protocol["locked_candidate"]["definition"],
            "posterior_state_sha256": hierarchical.rating._state_digest(
                replays[incumbent_id].state
            ),
            "versus_comparators": incumbent_vs_comparators,
        },
        "adaptive_challenger": {
            "candidate_id": adaptive_challenger_id,
            "definition": next(
                item
                for item in candidate_family._candidate_payloads()
                if item["candidate_id"] == adaptive_challenger_id
            ),
            "posterior_state_sha256": hierarchical.rating._state_digest(
                replays[adaptive_challenger_id].state
            ),
            "versus_incumbent": challenger_vs_incumbent,
        },
        "retention_decision": {
            "status": "RETAIN_REGISTERED_INCUMBENT",
            "challenger_superiority_required_strata": list(
                required_superiority
            ),
            "challenger_superiority_gate_passed": False,
            "reason": (
                "The adaptively ranked challenger did not have nonpositive upper "
                "95 percent series-cluster bootstrap bounds for both proper scores "
                "overall, in LCS, and after roster change. Preserve the prospective "
                "protocol instead of chasing corrected-source point estimates."
            ),
            "does_not_validate_incumbent": True,
        },
        "uncertainty": {
            "method": "series_cluster_bootstrap",
            "confidence": 0.95,
            "replicates": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "dependence_unit": "source-or-derived-series-dependence-cluster",
        },
        "reliability": {
            "status": "UNAVAILABLE_AS_AUTHORITY",
            "descriptive_bins_present_in_ranked_results": True,
            "locked_future_reliability_gate_opened": False,
        },
        "source_locks": [
            _source_record(root, locator) for locator in SOURCE_LOCKS
        ],
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
            "Adaptive corrected-source diagnostic only. It neither validates the "
            "incumbent nor authorizes ratings, probabilities, odds, expected value, "
            "recommendations, or betting."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_corrected_adaptive_diagnostic(payload, root=root)


def validate_corrected_adaptive_diagnostic(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CorrectedAdaptiveDiagnosticError("diagnostic must be an object")
    value = dict(payload)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise CorrectedAdaptiveDiagnosticError("diagnostic identity changed")
    declared = value.get("artifact_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    if not isinstance(declared, str) or declared != _canonical_sha256(unsigned):
        raise CorrectedAdaptiveDiagnosticError("diagnostic canonical hash mismatch")
    built_at = _timestamp(str(value.get("built_at_utc")), "built_at_utc")
    if built_at >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise CorrectedAdaptiveDiagnosticError("diagnostic crossed future boundary")
    source = validate_registered_source_snapshot_v2(root=root)
    protocol = validate_registered_future_protocol_v3(root=root)
    preflight = validate_registered_source_preflight_v3(root=root)
    bindings = value.get("bindings") or {}
    if bindings != {
        "source_package_id": PACKAGE_ID,
        "source_manifest_raw_sha256": MANIFEST_RAW_SHA256,
        "source_manifest_canonical_sha256": MANIFEST_CANONICAL_SHA256,
        "protocol_raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
        "protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "preflight_raw_sha256": REGISTERED_PREFLIGHT_RAW_SHA256,
        "preflight_artifact_sha256": REGISTERED_PREFLIGHT_ARTIFACT_SHA256,
        "preflight_posterior_state_sha256": preflight["numerical_preflight"][
            "posterior_state_sha256"
        ],
    }:
        raise CorrectedAdaptiveDiagnosticError("diagnostic bindings changed")
    boundary = value.get("information_boundary") or {}
    if (
        boundary.get("future_holdout_maps_present") != 0
        or boundary.get("future_holdout_targets_accessed") is not False
        or boundary.get("all_evaluated_outcomes_are_adaptive") is not True
        or boundary.get("evaluation_fold_is_not_independent_validation") is not True
    ):
        raise CorrectedAdaptiveDiagnosticError("diagnostic boundary changed")
    family = value.get("candidate_family") or {}
    ranked = family.get("ranked_results")
    expected_ids = {
        item["candidate_id"] for item in candidate_family._candidate_payloads()
    }
    if (
        family.get("definitions") != candidate_family._candidate_payloads()
        or not isinstance(ranked, list)
        or len(ranked) != len(expected_ids)
        or {item.get("candidate_id") for item in ranked} != expected_ids
    ):
        raise CorrectedAdaptiveDiagnosticError("candidate inventory changed")
    rank_keys = [
        (
            float(item["rank_key"]["maximum_point_regret"]),
            float(item["rank_key"]["overall_map_weighted_log_loss"]),
            float(item["rank_key"]["overall_map_weighted_brier"]),
            str(item["candidate_id"]),
        )
        for item in ranked
    ]
    if rank_keys != sorted(rank_keys):
        raise CorrectedAdaptiveDiagnosticError("candidate rank order changed")
    incumbent = value.get("incumbent") or {}
    challenger = value.get("adaptive_challenger") or {}
    if (
        incumbent.get("candidate_id")
        != protocol["locked_candidate"]["candidate_id"]
        or challenger.get("candidate_id") != ranked[0]["candidate_id"]
    ):
        raise CorrectedAdaptiveDiagnosticError("retention identities changed")
    decision = value.get("retention_decision") or {}
    if (
        decision.get("status") != "RETAIN_REGISTERED_INCUMBENT"
        or decision.get("challenger_superiority_gate_passed") is not False
        or decision.get("does_not_validate_incumbent") is not True
    ):
        raise CorrectedAdaptiveDiagnosticError("retention boundary changed")
    for section in (
        incumbent.get("versus_comparators") or {},
        {"challenger": challenger.get("versus_incumbent") or {}},
    ):
        if not isinstance(section, Mapping):
            raise CorrectedAdaptiveDiagnosticError("paired diagnostic missing")
    if any(value is not None for value in (payload.get("decision_outputs") or {}).values()):
        raise CorrectedAdaptiveDiagnosticError("diagnostic emitted a decision output")
    authority = payload.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(
        authority.get(name) is not False for name in AUTHORITY_KEYS
    ):
        raise CorrectedAdaptiveDiagnosticError("diagnostic exceeded authority")
    records = payload.get("source_locks")
    if not isinstance(records, list) or len(records) != len(SOURCE_LOCKS):
        raise CorrectedAdaptiveDiagnosticError("diagnostic source inventory changed")
    for record, locator in zip(records, SOURCE_LOCKS):
        path = root / locator
        if (
            not isinstance(record, Mapping)
            or record.get("locator") != locator
            or not path.is_file()
            or record.get("bytes") != path.stat().st_size
            or record.get("raw_sha256") != _sha256_path(path)
        ):
            raise CorrectedAdaptiveDiagnosticError(
                f"diagnostic source drifted: {locator}"
            )
    # Touch registered objects so a validator cannot accept stale lineage just
    # because the embedded scalar bindings still look plausible.
    if source.get("package_id") != PACKAGE_ID:
        raise CorrectedAdaptiveDiagnosticError("registered source identity changed")
    return value


def replay_corrected_adaptive_diagnostic(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    value = validate_corrected_adaptive_diagnostic(payload, root=root)
    rebuilt = build_corrected_adaptive_diagnostic(
        built_at=str(value["built_at_utc"]), root=root
    )
    if _canonical_bytes(rebuilt) != _canonical_bytes(value):
        raise CorrectedAdaptiveDiagnosticError("diagnostic replay mismatch")
    return value


def write_corrected_adaptive_diagnostic_no_clobber(
    payload: Mapping[str, Any], path: Path
) -> str:
    value = validate_corrected_adaptive_diagnostic(payload)
    raw = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        metadata = None
    if metadata is not None:
        raise CorrectedAdaptiveDiagnosticError(
            "diagnostic output exists; refusing to clobber"
        )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise CorrectedAdaptiveDiagnosticError(
                "diagnostic output is not a regular file"
            )
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(raw).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--built-at", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_corrected_adaptive_diagnostic(built_at=args.built_at)
    raw_sha256 = write_corrected_adaptive_diagnostic_no_clobber(
        payload, args.out
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "result_state": payload["result_state"],
                "incumbent_candidate_id": payload["incumbent"]["candidate_id"],
                "adaptive_challenger_id": payload["adaptive_challenger"][
                    "candidate_id"
                ],
                "challenger_superiority_gate_passed": payload[
                    "retention_decision"
                ]["challenger_superiority_gate_passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "BOOTSTRAP_SEED",
    "CorrectedAdaptiveDiagnosticError",
    "DEFAULT_OUTPUT",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "build_corrected_adaptive_diagnostic",
    "replay_corrected_adaptive_diagnostic",
    "validate_corrected_adaptive_diagnostic",
    "write_corrected_adaptive_diagnostic_no_clobber",
]
