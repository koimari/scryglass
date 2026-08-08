"""Artifact builders for L6 draft interaction development artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from pathlib import PurePosixPath
import sys
from typing import Any, Mapping

import numpy as np

from lol_kills.v2.evaluation.types import canonical_sha256

from .fixtures import (
    load_synthetic_fixture,
    load_synthetic_rows,
    reveal_synthetic_seed,
)
from .model import DraftInteractionModel, run_candidate_selection

SCHEMA_VERSION = "2.0.0"
MODEL_VERSION = "l6-draft-composition-interactions-development-v1"
ARTIFACT_CONFIG_ID = "draft-interactions-config-l6-v1"
ARTIFACT_FIXTURE_ID = "draft-interactions-synthetic-fixtures-l6-v1"
ARTIFACT_REPORT_ID = "draft-interactions-development-report-l6-v1"
ARTIFACT_AUTHORITY_ID = "draft-interactions-candidate-identity-l6-v2"
ARTIFACT_MANIFEST_ID = "draft-interactions-manifest-l6-v1"
_REPO_ROOT = Path(__file__).resolve().parents[4]
_IMPLEMENTATION_LOCATORS = (
    "lol_kills/v2/draft/interactions/__init__.py",
    "lol_kills/v2/draft/interactions/artifacts.py",
    "lol_kills/v2/draft/interactions/fixtures.py",
    "lol_kills/v2/draft/interactions/model.py",
    "lol_kills/v2/draft/interactions/types.py",
    "tests/model_v2/draft/interactions/test_draft_interactions_l6.py",
)
_FOUNDATION_LOCATORS = (
    "lol_kills/v2/evaluation/checkpoint_c1.py",
    "lol_kills/v2/evaluation/types.py",
    "lol_kills/v2/evaluation/checks.py",
    "lol_kills/v2/champions/__init__.py",
    "lol_kills/v2/champions/catalog.py",
    "lol_kills/v2/champions/paths.py",
    "lol_kills/v2/champions/schema.py",
    "lol_kills/v2/champions/fixtures.py",
    "data/lol/v2/evaluation/b2/checkpoint-c1-authority.json",
    "data/lol/v2/evaluation/b2/checkpoint-c1-config.json",
    "data/lol/v2/evaluation/b2/checkpoint-c1-report.json",
    "data/lol/v2/champions/champion-ontology-seed.json",
    "data/lol/v2/champions/champion-ontology-sources.json",
    "data/lol/v2/champions/champion-review-log.jsonl",
    "data/lol/v2/champions/evaluation-fixtures.json",
    "requirements.txt",
)
def canonical_artifact_bytes(payload: dict[str, Any]) -> bytes:
    """Encode an artifact deterministically for byte-exact replay."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _owned_regular_path(locator: str, *, must_exist: bool = True) -> Path:
    pure = PurePosixPath(locator)
    if (
        not locator
        or pure.is_absolute()
        or str(pure) != locator
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "//" in locator
    ):
        raise ValueError(f"artifact locator is not canonical relative POSIX: {locator}")
    lexical_path = _REPO_ROOT / locator
    current = _REPO_ROOT
    for part in pure.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(
                f"artifact locator parent chain contains symlink: {locator}"
            )
    if must_exist and not lexical_path.exists():
        raise ValueError(f"artifact locator is missing: {locator}")
    if lexical_path.is_symlink():
        raise ValueError(f"artifact locator must not be a symlink: {locator}")
    resolved = lexical_path.resolve(strict=must_exist)
    try:
        resolved.relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"artifact locator escapes repository: {locator}") from exc
    if must_exist:
        stat = resolved.stat()
        if not resolved.is_file() or stat.st_nlink != 1:
            raise ValueError(
                f"artifact locator must be a single-link regular file: {locator}"
            )
    return resolved


def _validate_artifact_payload(
    payload: Mapping[str, Any],
    *,
    role: str,
    artifact_id: str,
) -> None:
    if payload.get("artifact_id") != artifact_id:
        raise ValueError(f"{role} artifact_id mismatch")
    if payload.get("schema_version") != SCHEMA_VERSION and role != "fixtures":
        raise ValueError(f"{role} schema_version mismatch")
    if role == "config":
        ceiling = payload.get("claim_ceiling", {})
        forbidden_authorities = (
            "publication_authorized",
            "promotion_authorized",
            "pass_b2",
            "c2",
            "reliability_authorized",
            "probability_authorized",
        )
        if (
            payload.get("production_eligible") is not False
            or payload.get("status") != "candidate"
            or payload.get("principal_estimand")
            != "neutral_five_versus_five_composition_value"
            or ceiling.get("synthetic_only") is not True
            or ceiling.get("no_probability_claim") is not True
            or any(ceiling.get(key) is True for key in forbidden_authorities)
        ):
            raise ValueError("config exceeds synthetic development authority")
    elif role == "fixtures":
        if payload.get("source_scope") != "synthetic":
            raise ValueError("fixtures must remain synthetic")
    elif role == "report":
        if (
            payload.get("production_eligible") is not False
            or payload.get("status") != "synthetic_mechanics_only"
            or not payload.get("blockers")
        ):
            raise ValueError("report exceeds synthetic development authority")


def _raw_sha256(locator: str) -> str:
    path = _owned_regular_path(locator)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_entry(
    *,
    role: str,
    artifact_id: str,
    locator: str,
) -> dict[str, Any]:
    path = _owned_regular_path(locator)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_artifact_payload(
        payload,
        role=role,
        artifact_id=artifact_id,
    )
    return {
        "role": role,
        "artifact_id": artifact_id,
        "locator": locator,
        "raw_sha256": _raw_sha256(locator),
        "canonical_sha256": canonical_sha256(payload),
    }


def write_owned_artifacts() -> dict[str, str]:
    """Regenerate the acyclic L6 artifact graph in dependency order."""
    base_payloads = (
        (
            "data/lol/v2/models/draft-interactions/draft-interactions-config.json",
            build_interactions_config(),
        ),
        (
            "data/lol/v2/models/draft-interactions/draft-interactions-fixtures.json",
            build_fixture_payload(),
        ),
        (
            "data/lol/v2/models/draft-interactions/draft-interactions-development-report.json",
            build_development_report(),
        ),
    )
    written: dict[str, str] = {}
    for locator, payload in base_payloads:
        path = _owned_regular_path(locator)
        encoded = canonical_artifact_bytes(payload)
        path.write_bytes(encoded)
        written[locator] = hashlib.sha256(encoded).hexdigest()

    authority_locator = (
        "data/lol/v2/models/draft-interactions/draft-interactions-authority.json"
    )
    authority_path = _owned_regular_path(authority_locator)
    authority_bytes = canonical_artifact_bytes(build_authority())
    authority_path.write_bytes(authority_bytes)
    written[authority_locator] = hashlib.sha256(authority_bytes).hexdigest()

    manifest_locator = (
        "data/lol/v2/models/draft-interactions/draft-interactions-manifest.json"
    )
    manifest_path = _owned_regular_path(manifest_locator)
    manifest_bytes = canonical_artifact_bytes(render_draft_interactions_manifest())
    manifest_path.write_bytes(manifest_bytes)
    written[manifest_locator] = hashlib.sha256(manifest_bytes).hexdigest()
    return written


def _row_payload_list(rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    payload_rows: list[dict[str, Any]] = []
    if rows is None:
        for row in load_synthetic_rows():
            payload_rows.append(row.to_payload())
        return payload_rows
    for row in rows:
        payload_rows.append(dict(row))
    return payload_rows


def build_fixture_payload() -> dict[str, Any]:
    """Return deterministic synthetic fixture payload."""
    return load_synthetic_fixture()


def build_interactions_config(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload_rows = _row_payload_list(rows)
    model = DraftInteractionModel(draw_count=64, draw_seed=20260728)
    reference_distribution = model.legal_reference_distribution()
    try:
        report = run_candidate_selection(payload_rows, draw_count=64, draw_seed=20260728)
        selected_family = report.selected_family
        selection_status = report.selection_status
    except Exception:
        selected_family = None
        selection_status = "unavailable"

    return {
        "artifact_id": ARTIFACT_CONFIG_ID,
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "principal_estimand": "neutral_five_versus_five_composition_value",
        "status": "candidate",
        "production_eligible": False,
        "development_only": True,
        "public_serving_status": "unavailable",
        "public_probability_authorized": False,
        "public_interval_authorized": False,
        "selection_seed": reveal_synthetic_seed(),
        "families": [
            {
                "family_id": family.family_id,
                "include_ally_sparse": family.include_ally_sparse,
                "include_ally_exact": family.include_ally_exact,
                "include_enemy_sparse": family.include_enemy_sparse,
                "include_enemy_exact": family.include_enemy_exact,
                "include_whole_team": family.include_whole_team,
                "include_cross_team": family.include_cross_team,
                "include_factorization": family.include_factorization,
                "use_archetype_transfer": family.use_archetype_transfer,
            }
            for family in model.FAMILY_REGISTRY
        ],
        "selection_status": selection_status,
        "selected_family": selected_family,
        "posterior_approximation": {
            "family": "laplace_woodbury_low_rank_plus_diagonal",
            "dependence_preserved": True,
            "stored_diagonal_semantics": "pre_correction_prior_diagonal_D",
            "stored_factor_semantics": "woodbury_B_where_covariance_D_minus_B_transpose_B",
            "draw_identity": "fit_draw_count_and_covariance_seed",
            "oracle": "deterministic_one_parameter_hessian_inverse",
        },
        "legal_reference_distribution": reference_distribution,
        "rows_checksum": canonical_sha256(payload_rows),
        "rows": payload_rows,
        "claim_ceiling": {
            "synthetic_only": True,
            "no_side_advantage_term": True,
            "no_patch_mapping": True,
            "no_player_context": True,
            "no_probability_claim": True,
        },
        "metadata": {
            "source": "synthetic_dev",
            "selection": {
                "selection_min_rows": model._SELECTION_MIN_ROWS,
                "seed": 20260728,
                "draw_count": 64,
                "selection_sha256": report.selection_sha256 if "report" in locals() else None,
            },
        },
    }


def build_development_report() -> dict[str, Any]:
    rows = load_synthetic_rows()
    report = run_candidate_selection(rows, draw_count=64, draw_seed=20260728)
    blockers = [
        "Synthetic mechanics only; authoritative performance evidence unavailable.",
        "Selection requires identified decomposition and stable source-removal diagnostics.",
        "No PASS-B2, Reliability, or production promotion claim is authorized.",
    ]
    return {
        "artifact_id": ARTIFACT_REPORT_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "synthetic_mechanics_only",
        "selection_rule": "minimum development-fold log loss, then Brier, then ECE",
        "candidate_ids": [candidate.family_id for candidate in report.candidates],
        "selected_family": report.selected_family,
        "selection_status": report.selection_status,
        "selection_sha256": report.selection_sha256,
        "candidate_count": report.candidate_count,
        "candidates": [candidate.as_payload for candidate in report.candidates],
        "blockers": blockers,
        "production_eligible": False,
        "development_only": True,
        "public_serving_status": "unavailable",
        "public_probability_authorized": False,
        "public_interval_authorized": False,
    }


def build_authority() -> dict[str, Any]:
    artifacts = [
        _artifact_entry(
            role="config",
            artifact_id=ARTIFACT_CONFIG_ID,
            locator="data/lol/v2/models/draft-interactions/draft-interactions-config.json",
        ),
        _artifact_entry(
            role="fixtures",
            artifact_id=ARTIFACT_FIXTURE_ID,
            locator="data/lol/v2/models/draft-interactions/draft-interactions-fixtures.json",
        ),
        _artifact_entry(
            role="report",
            artifact_id=ARTIFACT_REPORT_ID,
            locator="data/lol/v2/models/draft-interactions/draft-interactions-development-report.json",
        ),
    ]
    all_locators = (
        [item["locator"] for item in artifacts]
        + list(_IMPLEMENTATION_LOCATORS)
        + list(_FOUNDATION_LOCATORS)
    )
    if len(all_locators) != len(set(all_locators)):
        raise ValueError("authority locators must be unique")
    return {
        "artifact_id": ARTIFACT_AUTHORITY_ID,
        "schema_version": SCHEMA_VERSION,
        "identity_kind": "non_authorizing_candidate_identity",
        "independent_l6_authority_present": False,
        "external_authority_digest": None,
        "authorization": "unavailable_requires_external_c2_l2_registrar",
        "production_eligible": False,
        "artifacts": artifacts,
        "implementation_inputs": [
            {"locator": locator, "raw_sha256": _raw_sha256(locator)}
            for locator in _IMPLEMENTATION_LOCATORS
        ],
        "foundation_inputs": [
            {"locator": locator, "raw_sha256": _raw_sha256(locator)}
            for locator in _FOUNDATION_LOCATORS
        ],
        "legal_reference_transform": DraftInteractionModel(
            draw_count=64,
            draw_seed=20260728,
        ).legal_reference_distribution(),
        "environment_identity": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "byteorder": sys.byteorder,
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "blas_config_sha256": canonical_sha256(
                getattr(np.__config__, "CONFIG", {})
            ),
        },
        "claim_ceiling": {
            "synthetic_only": True,
            "development_only": True,
            "predictive_performance_authorized": False,
            "real_data_fit_authorized": False,
            "calibrated_public_probability_authorized": False,
            "empirical_95_coverage_authorized": False,
            "reliability_authorized": False,
            "promotion_authorized": False,
            "publication_authorized": False,
            "sota_authority": False,
            "pass_b2": False,
            "c2": False,
        },
    }


def load_authorized_l6_model() -> None:
    raise ValueError(
        "independent L6 authority is absent; external C2/L2 registration required"
    )


def render_draft_interactions_manifest() -> dict[str, Any]:
    authority = build_authority()
    artifact_hashes = {
        item["role"]: item["raw_sha256"] for item in authority["artifacts"]
    }
    artifact_hashes["candidate_identity"] = _raw_sha256(
        "data/lol/v2/models/draft-interactions/draft-interactions-authority.json"
    )
    return {
        "artifact_id": ARTIFACT_MANIFEST_ID,
        "schema_version": SCHEMA_VERSION,
        "status": "candidate",
        "model_version": MODEL_VERSION,
        "artifact_hashes": artifact_hashes,
        "hash_semantics": "sha256_of_exact_file_bytes",
        "principal_estimand": "neutral_five_versus_five_composition_value",
        "sealed_holdout_opened": False,
        "promotion_decision": "pending",
        "production_eligible": False,
        "sota_claim": {"status": "not_claimed"},
        "authority_ceiling": "candidate identity only; external C2/L2 registrar absent",
        "independent_l6_authority_present": False,
    }
