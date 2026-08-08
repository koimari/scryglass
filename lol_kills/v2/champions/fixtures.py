"""Evaluation fixtures and contract-only transfer evaluation utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .catalog import (
    ChampionOntology,
    ChampionOntologyError,
    canonical_sha256,
    load_champion_ontology,
)
from .paths import (
    DEFAULT_FIXTURE_PATH,
    DEFAULT_ONTOLOGY_PATH,
    DEFAULT_REVIEW_PATH,
    DEFAULT_SOURCE_PATH,
)

EVALUATION_VERSION = "leave_one_out_v2"
C0_CONTRACT_HASH = "8748bbe48b273593b09304ac80923f11384de808b835f6e83e97c6fef48661dd"
ALLOWED_ALGORITHMS = {"inverse_distance_weighted_mean"}
ALLOWED_BASELINE_STRATEGIES = {"hierarchical_mean_or_zero"}
DEFAULT_DISTANCE_EPSILON = 1e-6


def load_evaluation_fixtures(path: Path | None = None) -> dict[str, Any]:
    fixture_path = path or DEFAULT_FIXTURE_PATH
    if not fixture_path.exists():
        raise ChampionOntologyError(f"missing fixture file: {fixture_path}")

    try:
        with fixture_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as err:
        raise ChampionOntologyError(f"invalid fixture json in {fixture_path}: {err}") from err

    if not isinstance(payload, dict):
        raise ChampionOntologyError("fixture payload must be an object")
    for field in ("masked", "transfer", "leave_one_out"):
        if field not in payload:
            raise ChampionOntologyError(f"fixture payload missing field: {field}")
    if not isinstance(payload["masked"], list):
        raise ChampionOntologyError("fixture field 'masked' must be a list")
    if not isinstance(payload["transfer"], list):
        raise ChampionOntologyError("fixture field 'transfer' must be a list")
    if not isinstance(payload["leave_one_out"], list):
        raise ChampionOntologyError("fixture field 'leave_one_out' must be a list")
    return payload


def masked_champion_fixtures(path: Path | None = None) -> list[dict[str, Any]]:
    payload = load_evaluation_fixtures(path)
    return list(payload["masked"])


def leave_one_out_fixtures(path: Path | None = None) -> list[dict[str, Any]]:
    payload = load_evaluation_fixtures(path)
    return list(payload["leave_one_out"])


def transfer_fixtures(path: Path | None = None) -> list[dict[str, Any]]:
    payload = load_evaluation_fixtures(path)
    return list(payload["transfer"])


def _write_temp_payload(payload: dict[str, Any]) -> Path:
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        return Path(handle.name)


def build_transfer_distances(
    ontology: ChampionOntology,
    fixture: dict[str, Any],
) -> list[float]:
    champions = list(fixture["champion_ids"])
    if not isinstance(champions, list) or len(champions) < 2:
        raise ChampionOntologyError("transfer fixture requires at least 2 champion_ids")
    role = fixture["role"]
    patch_id = fixture["patch_id"]
    league_id = fixture.get("league_id")

    vectors = []
    for champion_id in champions:
        vectors.append(
            ontology.build_archetype_prior(
                champion_id=champion_id,
                role=role,
                patch_id=patch_id,
                league_id=league_id,
            )["vector"]
        )

    distances = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            distances.append(round(ontology.profile_distance(vectors[i], vectors[j]), 6))
    return distances


def _validate_leave_one_out_fixture(fixture: dict[str, Any]) -> None:
    required = (
        "fixture_id",
        "schema_version",
        "c0_contract_hash",
        "as_of",
        "role",
        "patch_id",
        "league_id",
        "target_champion_ids",
        "cells",
    )
    for field in required:
        if field not in fixture:
            raise ChampionOntologyError(f"leave-one-out fixture missing field: {field}")

    if not isinstance(fixture["c0_contract_hash"], str) or not fixture["c0_contract_hash"]:
        raise ChampionOntologyError(
            "leave-one-out fixture c0_contract_hash must be non-empty string"
        )
    if fixture["c0_contract_hash"] != C0_CONTRACT_HASH:
        raise ChampionOntologyError(
            "leave-one-out fixture c0_contract_hash mismatch"
        )

    role = fixture["role"]
    patch_id = fixture["patch_id"]
    if not isinstance(role, str) or not role:
        raise ChampionOntologyError("leave-one-out fixture role must be non-empty string")
    if not isinstance(patch_id, str) or not patch_id:
        raise ChampionOntologyError("leave-one-out fixture patch_id must be non-empty string")

    target_ids = fixture["target_champion_ids"]
    if not isinstance(target_ids, list) or not target_ids:
        raise ChampionOntologyError("leave-one-out fixture requires at least one target_champion_id")
    if any(not isinstance(item, str) or not item for item in target_ids):
        raise ChampionOntologyError("leave-one-out target ids must be non-empty strings")

    cells = fixture["cells"]
    if not isinstance(cells, list) or not cells:
        raise ChampionOntologyError("leave-one-out fixture cells must be a non-empty list")
    for row in cells:
        if not isinstance(row, dict):
            raise ChampionOntologyError("leave-one-out empirical row must be object")
        for field in (
            "champion_id",
            "patch_id",
            "role",
            "league_id",
            "residual",
            "verified_appearance_count",
        ):
            if field not in row:
                raise ChampionOntologyError(f"leave-one-out row missing field: {field}")
    if any(row["champion_id"] == "" for row in cells):
        raise ChampionOntologyError("leave-one-out rows require champion id")


def _candidate_ids_for_evaluation(
    candidate_rows: list[dict[str, Any]],
    holdout_id: str,
) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for row in candidate_rows:
        if row["champion_id"] == holdout_id:
            continue
        if row["champion_id"] not in seen:
            ids.append(row["champion_id"])
            seen.add(row["champion_id"])
    return ids


def _squared_error(a: float, b: float) -> float:
    diff = float(a) - float(b)
    return round(diff * diff, 12)


def _compute_transfer_prediction(
    holdout_vector: list[float],
    candidate_prior_rows: list[dict[str, Any]],
    *,
    distance_epsilon: float,
) -> float:
    if not candidate_prior_rows:
        raise ChampionOntologyError("leave-one-out candidate rows required")
    if not (0.0 < distance_epsilon < 1.0):
        raise ChampionOntologyError("distance_epsilon must be between 0 and 1")

    weighted_sum = 0.0
    weight_sum = 0.0
    for row in candidate_prior_rows:
        distance = row["distance"]
        residual = row["residual"]
        weight = 1.0 / (distance + distance_epsilon)
        weighted_sum += weight * residual
        weight_sum += weight
    if weight_sum <= 0:
        raise ChampionOntologyError("leave-one-out transfer weight sum must be positive")
    return weighted_sum / weight_sum


def run_leave_one_out_prediction_evaluation(
    fixture: dict[str, Any],
    *,
    ontology_path: Path = DEFAULT_ONTOLOGY_PATH,
    source_path: Path = DEFAULT_SOURCE_PATH,
    review_path: Path = DEFAULT_REVIEW_PATH,
    as_of: str | None = None,
    algorithm: str = "inverse_distance_weighted_mean",
    baseline_strategy: str = "hierarchical_mean_or_zero",
    allow_training_target: bool = False,
    distance_epsilon: float = DEFAULT_DISTANCE_EPSILON,
) -> dict[str, Any]:
    if not isinstance(fixture, dict):
        raise ChampionOntologyError("leave-one-out fixture must be object")
    _validate_leave_one_out_fixture(fixture)

    if algorithm not in ALLOWED_ALGORITHMS:
        raise ChampionOntologyError(f"unsupported algorithm: {algorithm}")
    if baseline_strategy not in ALLOWED_BASELINE_STRATEGIES:
        raise ChampionOntologyError(f"unsupported baseline strategy: {baseline_strategy}")

    fixture_id = fixture["fixture_id"]
    schema_version = fixture["schema_version"]
    c0_contract_hash = fixture["c0_contract_hash"]
    fixture_as_of = fixture["as_of"]
    role = fixture["role"]
    patch_id = fixture["patch_id"]
    league_id = fixture["league_id"]
    target_champion_ids = list(fixture["target_champion_ids"])
    empirical_cells = list(fixture["cells"])
    for row in empirical_cells:
        if not isinstance(row, dict):
            raise ChampionOntologyError("leave-one-out cells must be objects")

    as_of = as_of or fixture_as_of
    if not isinstance(as_of, str) or not as_of:
        raise ChampionOntologyError("as_of must be provided")

    input_payload = {
        "schema_version": schema_version,
        "c0_contract_hash": c0_contract_hash,
        "as_of": fixture_as_of,
        "cells": empirical_cells,
    }

    evaluator_config = {
        "version": EVALUATION_VERSION,
        "algorithm": algorithm,
        "baseline_strategy": baseline_strategy,
        "allow_training_target": bool(allow_training_target),
        "distance_epsilon": float(distance_epsilon),
        "synthetic_contract_only": bool(fixture.get("synthetic_contract_only", True)),
    }

    config_hash = canonical_sha256(evaluator_config)
    fixture_hash = canonical_sha256(fixture)
    input_hash = canonical_sha256(input_payload)

    full_empirical_path = _write_temp_payload(input_payload)
    try:
        full_ontology = load_champion_ontology(
            ontology_path=ontology_path,
            source_path=source_path,
            review_path=review_path,
            empirical_path=full_empirical_path,
            as_of=as_of,
        )
    finally:
        full_empirical_path.unlink()

    dependency_lineage = {
        "c0_contract_hash": c0_contract_hash,
        "ontology_snapshot_sha256": full_ontology.ontology_snapshot_hash,
        "source_metadata_sha256": full_ontology.source_metadata_sha256,
        "manual_review_snapshot_sha256": full_ontology.review_snapshot_hash,
        "empirical_snapshot_sha256": full_ontology.empirical_snapshot_hash,
        "as_of": as_of,
        "review_as_of": full_ontology.as_of,
        "ontology_as_of": full_ontology.ontology_as_of,
        "source_as_of": full_ontology.source_as_of,
    }

    per_holdout: list[dict[str, Any]] = []
    missing_targets: list[str] = []

    for holdout_id in target_champion_ids:
        candidate_rows = [
            row for row in empirical_cells if row["champion_id"] != holdout_id
        ]
        if not candidate_rows and empirical_cells:
            raise ChampionOntologyError(f"no training rows for holdout {holdout_id}")

        includes_holdout = any(
            row["champion_id"] == holdout_id
            for row in candidate_rows
        )
        if includes_holdout:
            raise ChampionOntologyError(
                f"held-out champion leakage: {holdout_id} still present in candidate rows"
            )

        if allow_training_target and any(
            row["champion_id"] == holdout_id for row in empirical_cells
        ):
            per_holdout.append(
                {
                    "holdout_champion_id": holdout_id,
                    "status": "invalid_no_score",
                    "invalid_reason": "allow_training_target_explicit_leakage_path_not_supported",
                    "training_cell_count": len(candidate_rows),
                    "training_includes_holdout": True,
                    "candidate_count": 0,
                    "candidate_ids": [],
                    "uses_empirical_fixture": True,
                }
            )
            continue

        matched_rows = [
            row
            for row in empirical_cells
            if row["champion_id"] == holdout_id
            and row["patch_id"] == patch_id
            and row["role"] == role
            and row["league_id"] == league_id
        ]
        if not matched_rows:
            missing_targets.append(holdout_id)
            continue

        training_payload = {
            "schema_version": schema_version,
            "as_of": fixture_as_of,
            "cells": candidate_rows,
        }
        training_payload_hash = canonical_sha256(training_payload)
        training_path = _write_temp_payload(training_payload)
        try:
            training_ontology = load_champion_ontology(
                ontology_path=ontology_path,
                source_path=source_path,
                review_path=review_path,
                empirical_path=training_path,
                as_of=as_of,
            )
        finally:
            training_path.unlink()

        holdout_prior = full_ontology.build_archetype_prior(
            champion_id=holdout_id,
            role=role,
            patch_id=patch_id,
            league_id=league_id,
        )

        holdout_vector = holdout_prior["vector"]
        true_residual = float(holdout_prior["residual"]["mean"])

        candidate_ids = _candidate_ids_for_evaluation(
            candidate_rows=candidate_rows,
            holdout_id=holdout_id,
        )

        candidate_rows_for_scoring: list[dict[str, Any]] = []
        for candidate_id in candidate_ids:
            candidate_prior = training_ontology.build_archetype_prior(
                champion_id=candidate_id,
                role=role,
                patch_id=patch_id,
                league_id=league_id,
            )
            distance = training_ontology.profile_distance(
                holdout_vector,
                candidate_prior["vector"],
            )
            candidate_rows_for_scoring.append(
                {
                    "candidate_id": candidate_id,
                    "distance": distance,
                    "residual": float(candidate_prior["residual"]["mean"]),
                    "residual_status": candidate_prior["residual"]["status"],
                }
            )

        if not candidate_rows_for_scoring:
            raise ChampionOntologyError(f"holdout {holdout_id} has no candidate prior rows")

        transfer_prediction = _compute_transfer_prediction(
            holdout_vector=holdout_vector,
            candidate_prior_rows=[
                {
                    "distance": row["distance"],
                    "residual": row["residual"],
                }
                for row in candidate_rows_for_scoring
            ],
            distance_epsilon=distance_epsilon,
        )

        baseline_values = [
            row["residual"]
            for row in candidate_rows_for_scoring
            if row["residual_status"] in {"observed", "masked", "prior_only"}
        ]
        baseline_prediction = sum(baseline_values) / len(baseline_values) if baseline_values else 0.0

        transfer_error = _squared_error(transfer_prediction, true_residual)
        baseline_error = _squared_error(baseline_prediction, true_residual)

        per_holdout.append(
            {
                "holdout_champion_id": holdout_id,
                "training_cell_count": len(candidate_rows),
                "training_includes_holdout": includes_holdout,
                "candidate_count": len(candidate_rows_for_scoring),
                "candidate_ids": candidate_ids,
                "transfer_prediction": round(float(transfer_prediction), 8),
                "baseline_prediction": round(float(baseline_prediction), 8),
                "true_residual": round(float(true_residual), 8),
                "transfer_squared_error": transfer_error,
                "baseline_squared_error": baseline_error,
                "true_residual_status": holdout_prior["residual"]["status"],
                "true_residual_source": holdout_prior["residual_evidence"]["residual_source"],
                "residual_evidence_status": holdout_prior["residual_evidence"]["status"],
                "training_payload_hash": training_payload_hash,
                "uses_empirical_fixture": True,
            }
        )

    all_valid = [entry for entry in per_holdout if entry.get("status") != "invalid_no_score"]
    if all_valid:
        transfer_mse = sum(item["transfer_squared_error"] for item in all_valid) / max(
            1,
            len(all_valid),
        )
        baseline_mse = sum(item["baseline_squared_error"] for item in all_valid) / max(
            1,
            len(all_valid),
        )
        transfer_better_by = round(float(baseline_mse - transfer_mse), 12)
    else:
        transfer_mse = None
        baseline_mse = None
        transfer_better_by = None

    coverage = {
        "requested_holdouts": len(target_champion_ids),
        "covered_holdouts": len(all_valid),
        "requested_champions": list(target_champion_ids),
        "covered_champions": [entry["holdout_champion_id"] for entry in all_valid],
        "missing_champions": sorted(set(missing_targets)),
    }

    result = {
        "status": (
            "synthetic_contract_only"
            if all(entry.get("status") != "invalid_no_score" for entry in per_holdout)
            else "invalid_no_score"
        ),
        "synthetic_contract_only": bool(fixture.get("synthetic_contract_only", True)),
        "result_version": EVALUATION_VERSION,
        "fixture_id": fixture_id,
        "schema_version": schema_version,
        "patch_id": patch_id,
        "league_id": league_id,
        "role": role,
        "requested_as_of": as_of,
        "fixture_as_of": fixture_as_of,
        "config": evaluator_config,
        "dependency_lineage": dependency_lineage,
        "input_hash": input_hash,
        "fixture_hash": fixture_hash,
        "config_hash": config_hash,
        "coverage": coverage,
        "metrics": {
            "transfer_mse": None if transfer_mse is None else round(float(transfer_mse), 12),
            "baseline_mse": None if baseline_mse is None else round(float(baseline_mse), 12),
            "transfer_better_by": transfer_better_by,
        },
        "per_holdout": per_holdout,
        "source_dependency": {
            "fixture_cell_count": len(empirical_cells),
            "candidate_cell_count": len(empirical_cells),
        },
        "baseline_strategy": baseline_strategy,
    }
    result["result_hash"] = canonical_sha256(result)
    return result
