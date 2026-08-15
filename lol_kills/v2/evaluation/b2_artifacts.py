"""Strict content-addressed authority for synthetic-only PASS-B2 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from .checks import ValidationFailure
from .types import ArtifactRef, EvaluationRegistry, canonical_json, canonical_sha256


B2_ARTIFACT_IDS = (
    "scryglass:b2:reliability-registry:v1",
    "scryglass:b2:evidence-candidate-registry:v1",
    "scryglass:b2:calibration-candidate-registry:v1",
    "scryglass:b2:coverage-procedure:v1",
)
ZERO_SHA256 = "0" * 64
FROZEN_SYNTHETIC_REGISTRY_LOCATOR = Path(
    "data/lol/v2/evaluation/synthetic-registry-frozen.json"
)
REGISTRAR_TRUST_ROOT_LOCATOR = Path(
    "data/lol/v2/evaluation/registry-registrar-trust-root.json"
)
TEST_REGISTRAR_ID = "scryglass:test-only:l2-evaluation-registrar-v1"
REGISTRAR_TRUST_ROOT_RAW_SHA256 = (
    "7ab9ed349af98007d3385b18a769bf6b798a364032c204c354acdbdbb6486590"
)
REGISTRAR_TRUST_ROOT_OBJECT_SHA256 = (
    "c3328f4bfaeccdf7b9b9e30d9d9576bb689625fa8797bb695d2e19f9a32b5dd2"
)
# Updated only when the accepted, registrar-authorized synthetic registry bytes
# are intentionally regenerated.
FROZEN_SYNTHETIC_REGISTRY_RAW_SHA256 = (
    "824ebc1db08c7a7d35001515cfbe19a0d3d6543f1886dd4fefe43121d8391455"
)
FROZEN_SYNTHETIC_REGISTRY_OBJECT_SHA256 = (
    "3f215b05b2e5717b116216df2e170e0e35fe53ca0a4edb825c35d72f61a887c2"
)

_EXPECTED_FAMILIES = {
    "standardized_posterior_mean_displacement": (
        "posterior_information",
        "prior_standard_deviations",
        ("posterior_draws", "prior_draws"),
    ),
    "interval_contraction": (
        "precision",
        "fraction_of_reference_width",
        ("posterior_draws", "registered_reference_draws"),
    ),
    "deterministic_source_context_coverage": (
        "source_context_coverage",
        "typed_flags",
        (
            "source_lineage",
            "context_registry",
            "fallback_registry",
            "bridge_registry",
        ),
    ),
}
_EXPECTED_CALIBRATION_FAMILIES = (
    "identity",
    "symmetric_temperature",
    "symmetrized_platt",
    "symmetrized_beta",
    "symmetrized_bounded_isotonic",
)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationFailure(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValidationFailure(f"noncanonical JSON numeric constant: {value}")


def strict_json_bytes(
    raw: bytes, *, what: str, require_canonical_bytes: bool = True
) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValidationFailure(f"{what} is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise ValidationFailure(f"{what} must be a JSON object")
    if (
        require_canonical_bytes
        and raw != (canonical_json(payload) + "\n").encode("utf-8")
    ):
        raise ValidationFailure(f"{what} bytes are not canonical")
    _assert_finite(payload, what)
    return payload


def _assert_finite(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationFailure(f"{path} contains a nonfinite number")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def _safe_artifact_path(repo_root: Path, locator: str) -> Path:
    relative = Path(locator)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValidationFailure("B2 artifact locator must be repository-relative")
    root = repo_root.resolve()
    candidate = root.joinpath(relative)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValidationFailure(
                "B2 artifact locator contains a symlink component"
            )
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValidationFailure("B2 artifact locator escapes repository") from exc
    if not resolved.is_file():
        raise ValidationFailure("B2 artifact locator is missing or not a file")
    return resolved


def _valid_nonplaceholder_hash(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value == ZERO_SHA256:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return len(set(value.lower())) > 1


def _require_exact_fields(
    payload: MappingLike, expected: set[str], what: str
) -> None:
    if set(payload) != expected:
        raise ValidationFailure(f"{what} fields are missing or extra")


MappingLike = dict[str, Any]


def _validate_reliability(payload: MappingLike) -> None:
    _require_exact_fields(
        payload,
        {
            "artifact_id",
            "axes",
            "context_extraction",
            "context_universe",
            "diagnostics_to_label_policy",
            "ood_flag_sets",
            "production_eligible",
            "rules",
            "synthetic_only",
            "validation_strata",
        },
        "Reliability registry",
    )
    output_types = {
        "player_rating",
        "team_rating",
        "draft_score",
        "partial_draft_state",
        "tier_list",
    }
    if set(payload["context_extraction"]) != output_types:
        raise ValidationFailure("Reliability context extraction is not five-output exact")
    universe = list(payload["context_universe"])
    rules = list(payload["rules"])
    strata = list(payload["validation_strata"])
    if not universe or not rules or not strata:
        raise ValidationFailure("Reliability registry sets must be nonempty")
    context_hashes = [canonical_sha256(item) for item in universe]
    if len(context_hashes) != len(set(context_hashes)):
        raise ValidationFailure("Reliability contexts are duplicated")
    rule_ids = [item.get("rule_id") for item in rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValidationFailure("Reliability rules are duplicated")
    if len(strata) != len(set(strata)):
        raise ValidationFailure("Reliability strata are duplicated")
    referenced = {item.get("stratum_id") for item in rules}
    if referenced != set(strata):
        raise ValidationFailure("Reliability strata are unauthorized or unreferenced")
    from .reliability import audit_mapping_registry

    audit_mapping_registry(payload)


def _validate_evidence(payload: MappingLike) -> None:
    _require_exact_fields(
        payload,
        {
            "artifact_id",
            "benchmark_covariates",
            "boundaries_frozen_before_sealed_evidence",
            "candidates",
            "production_eligible",
            "selection",
            "synthetic_only",
        },
        "evidence candidate registry",
    )
    candidates = list(payload["candidates"])
    if len(candidates) != len(_EXPECTED_FAMILIES):
        raise ValidationFailure("evidence candidate set is not exact and nonempty")
    by_method = {item.get("method_id"): item for item in candidates}
    if set(by_method) != set(_EXPECTED_FAMILIES):
        raise ValidationFailure("evidence method set is unauthorized")
    for method, (family, units, roles) in _EXPECTED_FAMILIES.items():
        recipe = by_method[method]
        if recipe.get("family") != family or recipe.get("units") != units:
            raise ValidationFailure("evidence method family or units mismatch")
        if not _valid_nonplaceholder_hash(recipe.get("code_sha256")):
            raise ValidationFailure("evidence code hash is invalid or placeholder")
        if not _valid_nonplaceholder_hash(recipe.get("config_sha256")):
            raise ValidationFailure("evidence config hash is invalid or placeholder")
        dependencies = list(recipe.get("dependencies", ()))
        if tuple(item.get("role") for item in dependencies) != roles:
            raise ValidationFailure("evidence dependency role set/order mismatch")
        for dependency in dependencies:
            if not _valid_nonplaceholder_hash(dependency.get("raw_sha256")):
                raise ValidationFailure(
                    "evidence dependency hash is invalid or placeholder"
                )
    expected_selection = {
        "criteria",
        "rolling_replay",
        "simulation",
    }
    _require_exact_fields(payload["selection"], expected_selection, "evidence selection")
    if (
        payload["boundaries_frozen_before_sealed_evidence"] is not True
        or not payload["selection"]["criteria"]
        or payload["selection"]["rolling_replay"] != "chronological_series_blocked"
    ):
        raise ValidationFailure("evidence selection procedure is incomplete")


def _validate_calibration(payload: MappingLike) -> None:
    _require_exact_fields(
        payload,
        {
            "artifact_id",
            "boundary_epsilon",
            "candidates",
            "draft_score",
            "fit",
            "nested_selection",
            "production_eligible",
            "synthetic_only",
        },
        "calibration candidate registry",
    )
    candidates = list(payload["candidates"])
    families = tuple(item.get("family") for item in candidates)
    ranks = tuple(item.get("simplicity_rank") for item in candidates)
    if families != _EXPECTED_CALIBRATION_FAMILIES or ranks != tuple(range(5)):
        raise ValidationFailure("calibration candidate set/order is not exact")
    if not 0 < float(payload["boundary_epsilon"]) < 0.5:
        raise ValidationFailure("calibration boundary epsilon is invalid")
    if payload["fit"] != {
        "likelihood": "binomial",
        "link": "logit",
        "nonconvergence_status": "unavailable",
    }:
        raise ValidationFailure("calibration fit procedure is unauthorized")
    nested = payload["nested_selection"]
    if (
        set(nested)
        != {
            "outer_test_labels_available",
            "paired_uncertainty",
            "primary_score",
            "refit",
            "split",
        }
        or nested["outer_test_labels_available"] is not False
        or nested["split"] != "chronological_series_blocked"
        or nested["primary_score"] != "log_loss"
    ):
        raise ValidationFailure("calibration nested procedure is incomplete")


def _validate_coverage(payload: MappingLike) -> None:
    _require_exact_fields(
        payload,
        {
            "aggregate_forecast_coverage",
            "artifact_id",
            "production_eligible",
            "simulation_parameter_coverage",
            "synthetic_only",
            "wording",
        },
        "coverage procedure",
    )
    simulation = payload["simulation_parameter_coverage"]
    aggregate = payload["aggregate_forecast_coverage"]
    wording = payload["wording"]
    if (
        set(simulation)
        != {"aggregate_axes", "interval", "nominal", "required"}
        or simulation["nominal"] != 0.95
        or not simulation["aggregate_axes"]
        or not simulation["required"]
        or set(aggregate)
        != {
            "baseline_width_noninferiority_margin",
            "cell_partition",
            "dependence_design",
            "forbidden",
            "interval",
            "statistic",
        }
        or not aggregate["forbidden"]
        or aggregate["cell_partition"] != "exact_registered_partition"
        or set(wording) != {"b3_missing", "exact_95_probability_requires"}
        or wording["b3_missing"] != "95% model range"
        or not wording["exact_95_probability_requires"]
    ):
        raise ValidationFailure("coverage procedure is truncated or unauthorized")


_ARTIFACT_VALIDATORS = {
    B2_ARTIFACT_IDS[0]: _validate_reliability,
    B2_ARTIFACT_IDS[1]: _validate_evidence,
    B2_ARTIFACT_IDS[2]: _validate_calibration,
    B2_ARTIFACT_IDS[3]: _validate_coverage,
}


def verify_artifact_ref(ref: ArtifactRef, repo_root: Path) -> dict[str, Any]:
    if ref.artifact_id not in B2_ARTIFACT_IDS:
        raise ValidationFailure(f"unknown B2 artifact id: {ref.artifact_id}")
    for field, digest in (
        ("raw_sha256", ref.raw_sha256),
        ("canonical_payload_sha256", ref.canonical_payload_sha256),
    ):
        if len(digest) != 64 or digest == ZERO_SHA256:
            raise ValidationFailure(f"{ref.artifact_id} has invalid {field}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValidationFailure(f"{ref.artifact_id} has invalid {field}") from exc
    path = _safe_artifact_path(repo_root, ref.locator)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref.raw_sha256:
        raise ValidationFailure(f"{ref.artifact_id} raw bytes mismatch")
    payload = strict_json_bytes(raw, what=ref.artifact_id)
    if canonical_sha256(payload) != ref.canonical_payload_sha256:
        raise ValidationFailure(f"{ref.artifact_id} canonical payload mismatch")
    if payload.get("artifact_id") != ref.artifact_id:
        raise ValidationFailure(f"{ref.artifact_id} internal identity mismatch")
    if payload.get("synthetic_only") is not True or payload.get("production_eligible") is not False:
        raise ValidationFailure(f"{ref.artifact_id} violates synthetic-only boundary")
    _ARTIFACT_VALIDATORS[ref.artifact_id](payload)
    if ref.artifact_id == B2_ARTIFACT_IDS[1]:
        from .evidence import verify_evidence_registry

        verify_evidence_registry(payload, repo_root)
    return payload


def verify_frozen_b2_registry_authority(
    registry: EvaluationRegistry,
    repo_root: Path | str = Path("."),
) -> dict[str, str]:
    """Resolve B2 authority from code-pinned B1 registrar and registry bytes."""

    root = Path(repo_root)
    trust_path = _safe_artifact_path(root, str(REGISTRAR_TRUST_ROOT_LOCATOR))
    trust_raw = trust_path.read_bytes()
    trust = strict_json_bytes(trust_raw, what="B1 registry registrar trust root")
    if (
        hashlib.sha256(trust_raw).hexdigest()
        != REGISTRAR_TRUST_ROOT_RAW_SHA256
        or canonical_sha256(trust) != REGISTRAR_TRUST_ROOT_OBJECT_SHA256
    ):
        raise ValidationFailure("B1 registry registrar trust root is stale")
    registrars = trust.get("test_only", {}).get("registrars", [])
    matches = [
        item for item in registrars if item.get("registrar_id") == TEST_REGISTRAR_ID
    ]
    if (
        len(matches) != 1
        or trust.get("production", {}).get("registrars") != []
        or trust.get("production", {}).get("synthetic_ancestry_allowed") is not False
    ):
        raise ValidationFailure("B1 synthetic registrar authority is not exact")

    registry_path = _safe_artifact_path(
        root, str(FROZEN_SYNTHETIC_REGISTRY_LOCATOR)
    )
    registry_raw = registry_path.read_bytes()
    registry_payload = strict_json_bytes(
        registry_raw,
        what="authorized frozen synthetic registry",
        require_canonical_bytes=False,
    )
    if (
        hashlib.sha256(registry_raw).hexdigest()
        != FROZEN_SYNTHETIC_REGISTRY_RAW_SHA256
        or canonical_sha256(registry_payload)
        != FROZEN_SYNTHETIC_REGISTRY_OBJECT_SHA256
    ):
        raise ValidationFailure("frozen synthetic registry authority is stale")
    from .splitter import load_evaluation_registry

    authorized = load_evaluation_registry(registry_path)
    exact_frozen = (
        registry.to_payload() == authorized.to_payload()
        and registry.sha256() == authorized.sha256()
        and registry.is_synthetic_registry
    )
    if not exact_frozen:
        raise ValidationFailure(
            "B2 registry is not the exact B1-registrar-authorized synthetic registry"
        )
    return {
        "registrar_id": TEST_REGISTRAR_ID,
        "registrar_kind": "test_only",
        "registrar_raw_sha256": REGISTRAR_TRUST_ROOT_RAW_SHA256,
        "registrar_object_sha256": REGISTRAR_TRUST_ROOT_OBJECT_SHA256,
        "registry_locator": str(FROZEN_SYNTHETIC_REGISTRY_LOCATOR),
        "registry_raw_sha256": FROZEN_SYNTHETIC_REGISTRY_RAW_SHA256,
        "registry_object_sha256": FROZEN_SYNTHETIC_REGISTRY_OBJECT_SHA256,
        "registry_sha256": registry.sha256(),
        "authority_mode": "frozen_synthetic",
    }


def verify_b2_artifact_refs(
    registry: EvaluationRegistry,
    repo_root: Path | str = Path("."),
) -> dict[str, dict[str, Any]]:
    refs = tuple(registry.b2_artifact_refs)
    if len(refs) != 4:
        raise ValidationFailure("registry must contain exactly four B2 artifact refs")
    ids = [ref.artifact_id for ref in refs]
    if len(ids) != len(set(ids)) or set(ids) != set(B2_ARTIFACT_IDS):
        raise ValidationFailure("B2 artifact refs are missing, extra, or duplicated")
    locators = [ref.locator for ref in refs]
    if len(locators) != len(set(locators)):
        raise ValidationFailure("B2 artifact locators are duplicated")
    root = Path(repo_root)
    paths = [_safe_artifact_path(root, ref.locator) for ref in refs]
    resolved = [str(path.resolve()) for path in paths]
    inodes = [(os.stat(path).st_dev, os.stat(path).st_ino) for path in paths]
    if len(resolved) != len(set(resolved)) or len(inodes) != len(set(inodes)):
        raise ValidationFailure("B2 artifact refs contain a path or hard-link alias")
    artifacts = {
        ref.artifact_id: verify_artifact_ref(ref, root)
        for ref in refs
    }
    if not registry.is_synthetic_registry:
        raise ValidationFailure(
            "synthetic B2 artifact authority cannot be attached to a non-synthetic registry"
        )
    return artifacts


def refs_from_payloads(
    locators: Iterable[str],
    repo_root: Path | str = Path("."),
) -> tuple[ArtifactRef, ...]:
    """Build refs for fixture generation; verification still trusts registry bytes."""
    root = Path(repo_root)
    refs: list[ArtifactRef] = []
    for locator in locators:
        path = _safe_artifact_path(root, locator)
        raw = path.read_bytes()
        payload = strict_json_bytes(raw, what=locator)
        refs.append(
            ArtifactRef(
                artifact_id=str(payload["artifact_id"]),
                locator=locator,
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                canonical_payload_sha256=canonical_sha256(payload),
            )
        )
    return tuple(refs)
