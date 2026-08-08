"""Artifact-bound Reliability diagnostics and exact replay."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .checks import ValidationFailure
from .types import ArtifactRef, CONTRACT_TREE_SHA256, canonical_json, canonical_sha256


HIGH_PREREQUISITES = (
    "mapping_exact_one",
    "proper_score_skill",
    "typed_calibration_interface",
    "typed_aggregate_coverage_interface",
    "b3_authority",
    "approved_exact_transform",
    "positive_effective_clusters",
    "no_ood",
    "no_fallback",
    "probability_wording_eligible",
)
SYNTHETIC_DIAGNOSTIC_LOCATOR = Path(
    "data/lol/v2/evaluation/b2/reliability-positive-controls.json"
)
B3_AUTHORITY_LOCATOR = Path(
    "data/lol/v2/evaluation/b2/b3-reliability-authority.json"
)
# Filled after the owned artifacts are generated; these values are code authority,
# not values read from the submitted artifacts.
SYNTHETIC_DIAGNOSTIC_RAW_SHA256 = "085e3f6146df7dcba1a3b715bf85e429cdd386e88200d8d4b39d9eb389faeda9"
SYNTHETIC_DIAGNOSTIC_OBJECT_SHA256 = "dbb7313250be3b3551ca2f2b5b2860d6dad7535e58071b194d5fd2805ef3ed7f"
B3_AUTHORITY_RAW_SHA256 = "6a0a0e0cd078e0970d1b969e0d01f81b89371b81babe5bf1a438c366280bc569"
B3_AUTHORITY_OBJECT_SHA256 = "9309e417654270fda5b83cf00f5facd59ea7e7382913f05e115e2f43180e76a2"
_LOADER_AUTHORITY = object()
_ISSUED_MAPPING_IDS: set[int] = set()
_ISSUED_DIAGNOSTIC_IDS: set[int] = set()


def _valid_hash(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return len(set(value)) > 1


def _strict_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValidationFailure(f"{name} must be a boolean")
    return value


def _finite(value: object, name: str, *, lower: float | None = None, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationFailure(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationFailure(f"{name} must be finite")
    if lower is not None and number < lower:
        raise ValidationFailure(f"{name} is below its lower bound")
    if upper is not None and number > upper:
        raise ValidationFailure(f"{name} exceeds its upper bound")
    return number


def _stable_id(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not 3 <= len(value) <= 256
    ):
        raise ValidationFailure(
            f"{name} must be an unpadded stable ID of length 3..256"
        )
    return value


def _canonical_object(raw: bytes, what: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"{what} is not JSON") from exc
    if not isinstance(payload, dict) or raw != (canonical_json(payload) + "\n").encode():
        raise ValidationFailure(f"{what} is not a canonical JSON object")
    return payload


def _safe_path(root: Path, locator: str) -> Path:
    relative = Path(locator)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValidationFailure("Reliability artifact locator is unsafe")
    current = root.resolve()
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValidationFailure("Reliability artifact locator contains a symlink")
    resolved = current.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValidationFailure("Reliability artifact locator escapes repository") from exc
    if not resolved.is_file():
        raise ValidationFailure("Reliability artifact locator is missing")
    return resolved


@dataclass(frozen=True)
class VerifiedReliabilityMapping:
    ref: ArtifactRef
    repository_root: str
    authority_id: str
    payload: Mapping[str, Any]
    _authority_token: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class VerifiedReliabilityDiagnostic:
    artifact_locator: str
    artifact_raw_sha256: str
    artifact_object_sha256: str
    repository_root: str
    authority_id: str
    record: Mapping[str, Any]
    _authority_token: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class ReliabilityResolution:
    output_sha256: str
    context: Mapping[str, Any]
    context_sha256: str
    matching_rule_ids: tuple[str, ...]
    match_count: int
    stratum_id: str | None
    status: str
    reasons: tuple[str, ...]
    label: str
    mapping_raw_sha256: str
    mapping_object_sha256: str
    diagnostic_sha256: str | None
    resolution_sha256: str

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in self.__dict__.items()
            if key != "resolution_sha256"
        }


def load_verified_mapping(
    ref: ArtifactRef,
    repo_root: Path | str = Path("."),
) -> VerifiedReliabilityMapping:
    from .b2_artifacts import B2_ARTIFACT_IDS, verify_artifact_ref

    if ref.artifact_id != B2_ARTIFACT_IDS[0]:
        raise ValidationFailure("Reliability resolver received the wrong artifact ID")
    root = Path(repo_root).resolve()
    payload = verify_artifact_ref(ref, root)
    raw = _safe_path(root, ref.locator).read_bytes()
    parsed = _canonical_object(raw, "Reliability mapping")
    if (
        hashlib.sha256(raw).hexdigest() != ref.raw_sha256
        or canonical_sha256(parsed) != ref.canonical_payload_sha256
        or parsed != payload
        or parsed.get("artifact_id") != ref.artifact_id
    ):
        raise ValidationFailure("Reliability mapping ref, bytes, and object are detached")
    verified = VerifiedReliabilityMapping(
        ref=ref,
        repository_root=str(root),
        authority_id="b2-artifact-ref-authority-v1",
        payload=payload,
        _authority_token=_LOADER_AUTHORITY,
    )
    _ISSUED_MAPPING_IDS.add(id(verified))
    return verified


def _reverify_mapping(mapping: VerifiedReliabilityMapping) -> Mapping[str, Any]:
    if (
        not isinstance(mapping, VerifiedReliabilityMapping)
        or mapping._authority_token is not _LOADER_AUTHORITY
        or id(mapping) not in _ISSUED_MAPPING_IDS
        or mapping.authority_id != "b2-artifact-ref-authority-v1"
    ):
        raise ValidationFailure("Reliability mapping wrapper was not loader-issued")
    fresh = load_verified_mapping(mapping.ref, mapping.repository_root)
    if fresh.payload != mapping.payload or fresh.ref != mapping.ref:
        raise ValidationFailure("Reliability mapping in-memory payload differs from parsed raw bytes")
    return fresh.payload


def _verify_b3_authority(repo_root: Path | str = Path(".")) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    raw = _safe_path(root, str(B3_AUTHORITY_LOCATOR)).read_bytes()
    payload = _canonical_object(raw, "B3 Reliability authority")
    if (
        hashlib.sha256(raw).hexdigest() != B3_AUTHORITY_RAW_SHA256
        or canonical_sha256(payload) != B3_AUTHORITY_OBJECT_SHA256
        or payload.get("contract_tree_sha256") != CONTRACT_TREE_SHA256
        or payload.get("production_authorities") != []
        or payload.get("status") != "fail_closed_until_b3"
    ):
        raise ValidationFailure("B3 Reliability authority is stale or nonempty")
    return payload


def _validate_diagnostic_record(record: Mapping[str, Any]) -> None:
    required = {
        "record_id", "output_type", "stratum_id", "candidate_id",
        "candidate_version", "candidate_artifact_sha256", "baseline_id",
        "baseline_version", "baseline_artifact_sha256", "candidate_log_loss",
        "baseline_log_loss", "log_loss_skill", "candidate_brier",
        "baseline_brier", "brier_skill", "support", "effective_resolved_clusters",
        "calibration_status", "calibration_intercept", "calibration_slope",
        "aggregate_coverage_status", "aggregate_coverage", "nominal_coverage",
        "dependence_design_id", "transform_id", "transform_sha256",
        "transform_kind", "transform_approved", "ood_state", "fallback_state",
        "probability_wording_eligible", "source_snapshot_sha256",
        "registry_sha256", "synthetic_positive_control", "production_eligible",
        "record_sha256",
    }
    if set(record) != required:
        raise ValidationFailure("Reliability diagnostic record fields are missing or extra")
    for name in (
        "record_id", "output_type", "stratum_id", "candidate_id",
        "candidate_version", "baseline_id", "baseline_version",
        "calibration_status", "aggregate_coverage_status",
        "dependence_design_id", "transform_id", "ood_state", "fallback_state",
        "transform_kind",
    ):
        _stable_id(record[name], name)
    for name in (
        "candidate_artifact_sha256", "baseline_artifact_sha256",
        "transform_sha256", "source_snapshot_sha256", "registry_sha256",
    ):
        if not _valid_hash(record[name]):
            raise ValidationFailure(f"{name} is not a strict content hash")
    candidate_ll = _finite(record["candidate_log_loss"], "candidate_log_loss", lower=0)
    baseline_ll = _finite(record["baseline_log_loss"], "baseline_log_loss", lower=0)
    ll_skill = _finite(record["log_loss_skill"], "log_loss_skill")
    candidate_brier = _finite(record["candidate_brier"], "candidate_brier", lower=0, upper=1)
    baseline_brier = _finite(record["baseline_brier"], "baseline_brier", lower=0, upper=1)
    brier_skill = _finite(record["brier_skill"], "brier_skill")
    if abs(ll_skill - (baseline_ll - candidate_ll)) > 1e-12:
        raise ValidationFailure("log-loss skill does not recompute")
    if abs(brier_skill - (baseline_brier - candidate_brier)) > 1e-12:
        raise ValidationFailure("Brier skill does not recompute")
    if ll_skill <= 0 or brier_skill <= 0:
        raise ValidationFailure("candidate must improve both registered proper scores")
    if isinstance(record["support"], bool) or not isinstance(record["support"], int):
        raise ValidationFailure("support must be an integer count")
    _finite(record["support"], "support", lower=1)
    _finite(record["effective_resolved_clusters"], "effective_resolved_clusters", lower=1)
    _finite(record["calibration_intercept"], "calibration_intercept")
    slope = _finite(record["calibration_slope"], "calibration_slope")
    if slope <= 0 or slope > 100:
        raise ValidationFailure("calibration slope is outside the frozen interface bounds")
    aggregate_coverage = _finite(
        record["aggregate_coverage"], "aggregate_coverage", lower=0, upper=1
    )
    nominal_coverage = _finite(
        record["nominal_coverage"], "nominal_coverage", lower=0, upper=1
    )
    if aggregate_coverage != 0.95 or nominal_coverage != 0.95:
        raise ValidationFailure("synthetic coverage control must equal its frozen 95% value")
    for name in (
        "transform_approved", "probability_wording_eligible",
        "synthetic_positive_control", "production_eligible",
    ):
        _strict_bool(record[name], name)
    if record["calibration_status"] != "synthetic_positive_control":
        raise ValidationFailure("calibration adequacy is not authorized in checkpoint 2A")
    if record["aggregate_coverage_status"] != "synthetic_positive_control":
        raise ValidationFailure("coverage adequacy is not authorized in checkpoint 2A")
    if (
        record["synthetic_positive_control"] is not True
        or record["production_eligible"] is not False
        or record["transform_approved"] is not True
        or record["probability_wording_eligible"] is not True
        or record["ood_state"] != "known:"
        or record["fallback_state"] != "none"
    ):
        raise ValidationFailure("Reliability positive control violates synthetic boundary")
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    if record["record_sha256"] != canonical_sha256(unsigned):
        raise ValidationFailure("Reliability diagnostic record hash is invalid")


def load_verified_diagnostics(
    repo_root: Path | str = Path("."),
) -> tuple[VerifiedReliabilityDiagnostic, ...]:
    root = Path(repo_root).resolve()
    raw = _safe_path(root, str(SYNTHETIC_DIAGNOSTIC_LOCATOR)).read_bytes()
    payload = _canonical_object(raw, "Reliability positive controls")
    if (
        hashlib.sha256(raw).hexdigest() != SYNTHETIC_DIAGNOSTIC_RAW_SHA256
        or canonical_sha256(payload) != SYNTHETIC_DIAGNOSTIC_OBJECT_SHA256
        or payload.get("artifact_id") != "scryglass:b2:reliability-positive-controls:v1"
        or payload.get("contract_tree_sha256") != CONTRACT_TREE_SHA256
        or payload.get("synthetic_only") is not True
        or payload.get("production_eligible") is not False
    ):
        raise ValidationFailure("Reliability diagnostic authority is stale")
    from .b2_artifacts import (
        FROZEN_SYNTHETIC_REGISTRY_LOCATOR,
        verify_frozen_b2_registry_authority,
    )
    from .splitter import load_evaluation_registry

    registry = load_evaluation_registry(root / FROZEN_SYNTHETIC_REGISTRY_LOCATOR)
    verify_frozen_b2_registry_authority(registry, root)
    if payload.get("registry_sha256") != registry.sha256():
        raise ValidationFailure("Reliability diagnostics registry identity is stale")
    mapping_raw = _safe_path(
        root, "data/lol/v2/evaluation/b2/reliability-registry.json"
    ).read_bytes()
    mapping_payload = _canonical_object(mapping_raw, "Reliability mapping authority")
    if (
        payload.get("mapping_raw_sha256") != hashlib.sha256(mapping_raw).hexdigest()
        or payload.get("mapping_object_sha256") != canonical_sha256(mapping_payload)
    ):
        raise ValidationFailure("Reliability diagnostics are detached from mapping bytes")
    authorities = payload.get("transform_authorities")
    if not isinstance(authorities, list) or len(authorities) != 2:
        raise ValidationFailure("Reliability transform authority set is not exact")
    expected_transform_ids = {"identity-v1", "symmetrized-platt-v1"}
    loaded_authorities: dict[str, dict[str, Any]] = {}
    for authority in authorities:
        if not isinstance(authority, dict) or set(authority) != {
            "transform_id", "kind", "locator", "raw_sha256", "object_sha256",
        }:
            raise ValidationFailure("Reliability transform authority fields are invalid")
        transform_id = _stable_id(authority["transform_id"], "transform_id")
        if transform_id in loaded_authorities:
            raise ValidationFailure("Reliability transform authority is duplicated")
        transform_raw = _safe_path(root, authority["locator"]).read_bytes()
        transform_payload = _canonical_object(
            transform_raw, f"Reliability transform {transform_id}"
        )
        registry_record = registry.served_transform_identities.get(transform_id)
        if (
            hashlib.sha256(transform_raw).hexdigest() != authority["raw_sha256"]
            or canonical_sha256(transform_payload) != authority["object_sha256"]
            or transform_payload.get("artifact_id") != transform_id
            or transform_payload.get("kind") != authority["kind"]
            or transform_payload.get("synthetic_only") is not True
            or transform_payload.get("production_eligible") is not False
            or registry_record
            != {"kind": authority["kind"], "sha256": authority["raw_sha256"]}
        ):
            raise ValidationFailure(
                "Reliability transform bytes, semantics, and registry record are detached"
            )
        loaded_authorities[transform_id] = authority
    if set(loaded_authorities) != expected_transform_ids:
        raise ValidationFailure("Reliability transform identity set is not exact")
    _verify_b3_authority(root)
    records = list(payload.get("records", ()))
    if len(records) != 5:
        raise ValidationFailure("Reliability positive-control record set is not exact")
    keys: set[tuple[str, str]] = set()
    result = []
    for record in records:
        _validate_diagnostic_record(record)
        transform_authority = loaded_authorities.get(record["transform_id"])
        if (
            record["registry_sha256"] != payload.get("registry_sha256")
            or record["source_snapshot_sha256"]
            != payload.get("source_snapshot_sha256")
            or transform_authority is None
            or record["transform_kind"] != transform_authority["kind"]
            or record["transform_sha256"] != transform_authority["raw_sha256"]
        ):
            raise ValidationFailure(
                "Reliability diagnostic lineage or transform authority is detached"
            )
        key = (record["output_type"], record["stratum_id"])
        if key in keys:
            raise ValidationFailure("Reliability diagnostic records are duplicated")
        keys.add(key)
        verified = VerifiedReliabilityDiagnostic(
                artifact_locator=str(SYNTHETIC_DIAGNOSTIC_LOCATOR),
                artifact_raw_sha256=SYNTHETIC_DIAGNOSTIC_RAW_SHA256,
                artifact_object_sha256=SYNTHETIC_DIAGNOSTIC_OBJECT_SHA256,
                repository_root=str(root),
                authority_id="synthetic-reliability-positive-control-v1",
                record=record,
                _authority_token=_LOADER_AUTHORITY,
            )
        _ISSUED_DIAGNOSTIC_IDS.add(id(verified))
        result.append(verified)
    return tuple(result)


def _reverify_diagnostic(
    diagnostic: VerifiedReliabilityDiagnostic,
) -> Mapping[str, Any]:
    if (
        not isinstance(diagnostic, VerifiedReliabilityDiagnostic)
        or diagnostic._authority_token is not _LOADER_AUTHORITY
        or id(diagnostic) not in _ISSUED_DIAGNOSTIC_IDS
        or diagnostic.authority_id != "synthetic-reliability-positive-control-v1"
    ):
        raise ValidationFailure("Reliability diagnostic was not loader-issued")
    fresh = load_verified_diagnostics(diagnostic.repository_root)
    matches = [item for item in fresh if item.record["record_id"] == diagnostic.record.get("record_id")]
    if len(matches) != 1 or matches[0].record != diagnostic.record:
        raise ValidationFailure("Reliability diagnostic differs from frozen authority bytes")
    return matches[0].record


def _pointer(payload: Mapping[str, Any], pointer: str) -> Any:
    current: Any = payload
    for token in pointer.strip("/").split("/"):
        if not isinstance(current, Mapping) or token not in current:
            return None
        current = current[token]
    return current


def extract_context(output: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    output_type = str(output.get("output_type", ""))
    specs = registry.get("context_extraction", {})
    if output_type not in specs:
        raise ValidationFailure("output type lacks registered context extraction")
    context = {"output_type": output_type}
    for axis, pointer in specs[output_type].items():
        context[axis] = _pointer(output, str(pointer))
    flags = context.get("ood_flags")
    if flags is None:
        context["ood_flags"] = None
        context["ood_state"] = "missing"
    else:
        if not isinstance(flags, list) or len(flags) != len(set(map(str, flags))):
            raise ValidationFailure("OOD flags are malformed or duplicated")
        context["ood_flags"] = sorted(map(str, flags))
        known_sets = {tuple(item) for item in registry["ood_flag_sets"]}
        context["ood_state"] = (
            "known:" + ",".join(context["ood_flags"])
            if tuple(context["ood_flags"]) in known_sets
            else "unknown"
        )
    return context


def _matches(selector: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    return all(context.get(axis) == expected for axis, expected in selector.items())


def audit_mapping_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    universe = list(registry.get("context_universe", []))
    rules = list(registry.get("rules", []))
    axes = tuple(registry.get("axes", ()))
    if not universe or not rules or not axes or len(axes) != len(set(axes)):
        raise ValidationFailure("Reliability mapping axes or sets are incomplete")
    hashes = [canonical_sha256(item) for item in universe]
    if len(hashes) != len(set(hashes)):
        raise ValidationFailure("Reliability contexts are duplicated")
    rule_ids = [rule.get("rule_id") for rule in rules]
    selectors = [canonical_sha256(rule.get("selector", {})) for rule in rules]
    if len(rule_ids) != len(set(rule_ids)) or len(selectors) != len(set(selectors)):
        raise ValidationFailure("Reliability rule IDs or selectors are duplicated")
    strata = list(registry.get("validation_strata", ()))
    if not strata or len(strata) != len(set(strata)) or {r.get("stratum_id") for r in rules} != set(strata):
        raise ValidationFailure("Reliability strata are missing, duplicated, or unreferenced")
    evidence = []
    for context in universe:
        if set(context) != set(axes):
            raise ValidationFailure("registered Reliability context axes mismatch")
        matching = [rule for rule in rules if _matches(rule.get("selector", {}), context)]
        if len(matching) != 1:
            raise ValidationFailure(f"Reliability context has {'gap' if not matching else 'overlap'}")
        if set(matching[0]) != {"rule_id", "selector", "stratum_id"}:
            raise ValidationFailure("Reliability rule may select only a stratum")
        evidence.append({"context_sha256": canonical_sha256(context), "rule_id": matching[0]["rule_id"]})
    return {"registered_context_count": len(universe), "evidence": evidence}


def resolve_reliability(
    output: Mapping[str, Any],
    mapping: VerifiedReliabilityMapping,
    diagnostic: VerifiedReliabilityDiagnostic | None,
) -> ReliabilityResolution:
    if not isinstance(output.get("status"), str) or output.get("status") != "ok":
        raise ValidationFailure(
            "Reliability rating requires exact string output status 'ok'"
        )
    registry = _reverify_mapping(mapping)
    audit_mapping_registry(registry)
    context = extract_context(output, registry)
    missing_provenance = any(context.get(axis) is None for axis in registry["axes"])
    universe = {canonical_sha256(item) for item in registry["context_universe"]}
    if missing_provenance:
        matches = [rule for rule in registry["rules"] if rule.get("selector", {}).get("output_type") == context["output_type"]]
    elif canonical_sha256(context) in universe and context["ood_state"] != "unknown":
        matches = [rule for rule in registry["rules"] if _matches(rule["selector"], context)]
    else:
        matches = []
    stratum = matches[0]["stratum_id"] if len(matches) == 1 else None
    if stratum is None:
        status, reasons, label, diagnostic_hash = "unrated", ("no_registered_context_match",), "unrated", None
    elif missing_provenance:
        status, reasons, label, diagnostic_hash = "unavailable", ("context_provenance_missing",), "limited", None
    elif diagnostic is None:
        status, reasons, label, diagnostic_hash = "unavailable", ("diagnostic_missing",), "limited", None
    else:
        record = _reverify_diagnostic(diagnostic)
        if (
            record["output_type"] != output.get("output_type")
            or record["stratum_id"] != stratum
            or record["transform_id"] != context.get("transform_id")
        ):
            raise ValidationFailure("Reliability diagnostic output, stratum, or transform is mismatched")
        policy = registry["diagnostics_to_label_policy"]
        if (
            policy.get("synthetic_high_control") is not True
            or policy.get("production_b3_authority") is not False
            or "high" not in policy.get("labels", [])
        ):
            raise ValidationFailure("Reliability label policy is unauthorized")
        status, reasons, label = "ok", (), "high"
        diagnostic_hash = record["record_sha256"]
    base = {
        "output_sha256": canonical_sha256(output),
        "context": context,
        "context_sha256": canonical_sha256(context),
        "matching_rule_ids": [rule["rule_id"] for rule in matches],
        "match_count": len(matches),
        "stratum_id": stratum,
        "status": status,
        "reasons": list(reasons),
        "label": label,
        "mapping_raw_sha256": mapping.ref.raw_sha256,
        "mapping_object_sha256": mapping.ref.canonical_payload_sha256,
        "diagnostic_sha256": diagnostic_hash,
    }
    return ReliabilityResolution(
        output_sha256=base["output_sha256"],
        context=context,
        context_sha256=base["context_sha256"],
        matching_rule_ids=tuple(base["matching_rule_ids"]),
        match_count=len(matches),
        stratum_id=stratum,
        status=status,
        reasons=tuple(reasons),
        label=label,
        mapping_raw_sha256=mapping.ref.raw_sha256,
        mapping_object_sha256=mapping.ref.canonical_payload_sha256,
        diagnostic_sha256=diagnostic_hash,
        resolution_sha256=canonical_sha256(base),
    )


def verify_reliability_replay(
    resolution: ReliabilityResolution,
    output: Mapping[str, Any],
    mapping: VerifiedReliabilityMapping,
    diagnostic: VerifiedReliabilityDiagnostic | None,
) -> None:
    if not _valid_hash(resolution.resolution_sha256):
        raise ValidationFailure("Reliability resolution hash is malformed")
    if resolution.diagnostic_sha256 is not None and not _valid_hash(resolution.diagnostic_sha256):
        raise ValidationFailure("Reliability diagnostic hash is malformed")
    if resolve_reliability(output, mapping, diagnostic) != resolution:
        raise ValidationFailure("Reliability resolution does not replay exactly")
