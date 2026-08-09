"""Frozen-schema publication matrix and content-addressed policy evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from ..data import parse_rfc3339, to_rfc3339
from ..data.common import (
    canonical_json_bytes,
    sha256_canonical_object_hash,
    sha256_raw_bytes_hash,
)
from ..data.source_tree import (
    normalize_source_tree_path,
    resolve_repository_file,
)
from .snapshots import CONTRACT_TREE_SHA256, _validate_forbidden_filters

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_LOCATOR = "data/lol/v2/publication/publication-evidence-b2.json"
TRAINING_LOCATOR = "data/lol/v2/snapshots/b1/training-snapshot-passb1.json"
SOURCE_LOCATOR = "data/lol/v2/snapshots/b1/source-snapshot-passb1.json"
ARTIFACT_CLASSES = (
    "raw", "derived_rows", "aggregate", "code", "features", "weights",
    "evaluation", "documentation", "user_auth",
)
MODES = ("public", "authenticated", "private")
C4_AUTHORITY_ROOTS = {
    "production": (
        "data/lol/v2/publication/c4-authority-registry-b2.json",
        "d8cae6b7cd30456ab88dd5d706bb76a143ec507c68f192fbb5ffa6f09c6fc8a9",
    ),
    "test_only": (
        "data/lol/v2/publication/c4-test-authority-registry-b2.json",
        "368f05165fcae68606b4d15bc8157bd213671ce572e72ed807b380fd3398b728",
    ),
}


class PublicationMatrixError(ValueError):
    pass


class PublicationMatrixDecision:
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    PRIVATE = "private"
    PRIVATE_PENDING_REVIEW = "private_pending_review"
    PROHIBITED = "prohibited"


@dataclass(frozen=True)
class SourceRegistryEntry:
    source_id: str
    source_name: str
    source_url: str
    owner: str
    access_method: str
    credential_required: bool
    secret_storage_class: str
    artifact_classes: tuple[str, ...]
    source_row_bindings: tuple[str, ...] = ()


def registered_sources() -> tuple[SourceRegistryEntry, ...]:
    rows = (
        SourceRegistryEntry("scryglass:source:oracle-elixir", "Oracle's Elixir", "https://lol.timsevenhuysen.com/matchdata/", "Oracle's Elixir", "download", False, "none", ("raw", "aggregate")),
        SourceRegistryEntry("scryglass:source:riot-ddragon", "Riot Data Dragon", "https://ddragon.leagueoflegends.com/", "Riot Games", "public_api", False, "none", ("raw", "aggregate")),
        SourceRegistryEntry("scryglass:source:riot-docs", "Riot developer documentation", "https://developer.riotgames.com/docs/lol", "Riot Games", "public_api", False, "none", ("documentation",)),
        SourceRegistryEntry("scryglass:source:riot-api", "Riot authenticated API", "https://api.riotgames.com/", "Riot Games", "authenticated_api", True, "server_secret", ("raw", "aggregate")),
        SourceRegistryEntry("scryglass:source:riot-esports-live", "Riot esports live event feed", "https://feed.lolesports.com/livestats/v1/", "Riot Games", "authenticated_api", True, "server_secret", ("raw",)),
        SourceRegistryEntry("scryglass:source:leaguepedia", "Leaguepedia", "https://lol.fandom.com/wiki/League_of_Legends_Esports_Wiki", "Fandom", "manual", False, "none", ("raw", "aggregate")),
        SourceRegistryEntry("scryglass:source:grid", "GRID", "https://grid.gg/", "GRID", "partner_feed", True, "partner_managed", ("raw", "derived_rows", "aggregate")),
        SourceRegistryEntry("scryglass:source:manual-ontology", "Manually authored ontology and review", "repo://scryglass/data/lol/v2/publication/publication-evidence-b2.json", "Scryglass", "manual", False, "none", ("raw", "documentation"), ("row-json-b1",)),
        SourceRegistryEntry("scryglass:source:derived-features", "Scryglass derived features", "repo://scryglass/data/lol/v2/publication/transforms/two-source-derived-b2.json", "Scryglass", "derived", False, "none", ("derived_rows", "aggregate", "features")),
        SourceRegistryEntry("scryglass:source:model-code", "Scryglass model code", "repo://scryglass/lol_kills/v2/provenance/publication.py", "Scryglass", "derived", False, "none", ("code",), ("row-text-b1",)),
        SourceRegistryEntry("scryglass:source:model-weights", "Scryglass model weights", "repo://scryglass/data/lol/v2/publication/publication-evidence-b2.json", "Scryglass", "derived", False, "none", ("weights",)),
        SourceRegistryEntry("scryglass:source:evaluation-reports", "Scryglass evaluation reports", "repo://scryglass/data/lol/v2/evaluation/synthetic-registry-frozen.json", "Scryglass", "derived", False, "none", ("evaluation",)),
        SourceRegistryEntry("scryglass:source:user-auth", "Scryglass user and authentication records", "repo://scryglass/data/lol/v2/publication/publication-evidence-b2.json", "Scryglass", "derived", True, "prohibited", ("user_auth",)),
    )
    for row in rows:
        _validate_source_url(row.source_url)
    return rows


def _default_schema(source_id: str, artifact_class: str) -> dict[str, Any]:
    if source_id == "scryglass:source:derived-features" and artifact_class == "derived_rows":
        return {"type": "object", "required": ["record_count", "sources"], "additionalProperties": False, "properties": {"record_count": {"type": "integer"}, "sources": {"type": "array", "items": {"type": "string"}}}}
    if source_id == "scryglass:source:manual-ontology" and artifact_class == "raw":
        return {"type": "object", "required": ["alpha", "beta", "nested"], "additionalProperties": False, "properties": {"alpha": {"type": "array", "items": {"type": "integer"}}, "beta": {"type": "string"}, "nested": {"type": "object", "required": ["seed"], "additionalProperties": False, "properties": {"seed": {"type": "integer"}}}}}
    return {"type": "object", "required": ["record_id"], "additionalProperties": False, "properties": {"record_id": {"type": "string"}}}


def make_publication_evidence() -> dict[str, Any]:
    sources = []
    schemas = []
    for source in registered_sources():
        sources.append({
            "source_id": source.source_id, "source_url": source.source_url,
            "artifact_classes": list(source.artifact_classes),
            "source_row_bindings": list(source.source_row_bindings),
        })
        for artifact_class in source.artifact_classes:
            for mode in MODES:
                schemas.append({
                    "schema_id": f"scryglass:payload-schema:{source.source_id.split(':')[-1]}:{artifact_class}:{mode}",
                    "source_id": source.source_id, "artifact_class": artifact_class,
                    "mode": mode, "schema": _default_schema(source.source_id, artifact_class),
                })
    payload = {
        "schema_version": "2.0.0", "sources": sources,
        "payload_schemas": schemas,
        "source_snapshot_locator": SOURCE_LOCATOR,
        "applicability_status": "complete",
    }
    payload["evidence_id"] = "scryglass:publication-evidence:" + sha256_canonical_object_hash(payload)
    return payload


def load_publication_evidence(path: Path | None = None) -> dict[str, Any]:
    target = path or (REPO_ROOT / EVIDENCE_LOCATOR)
    payload = json.loads(target.read_bytes())
    expected = make_publication_evidence()
    if payload != expected:
        raise PublicationMatrixError("publication evidence drift")
    return payload


@dataclass(frozen=True)
class PublicationMatrixRow:
    row_id: str
    source_id: str
    source_name: str
    source_url: str
    owner: str
    access_method: str
    artifact_class: str
    credential_required: bool
    terms_review_status: str
    decision: str
    field_allowlist: tuple[str, ...]
    field_denylist: tuple[str, ...]
    derivative_reconstruction_risk: str
    reviewer: str
    decision_evidence: tuple[str, ...]
    next_review_at: str
    secret_storage_class: str | None = None
    license_or_terms: str | None = None
    terms_reviewed_at: str | None = None
    storage_rule: str | None = None
    redistribution_rule: str | None = None
    retention_rule: str | None = None
    attribution_rule: str | None = None
    deidentification_rule: str | None = None

    def __post_init__(self) -> None:
        _validate_row(self)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "row_id": self.row_id, "source_id": self.source_id,
            "source_name": self.source_name, "source_url": self.source_url,
            "owner": self.owner, "access_method": self.access_method,
            "artifact_class": self.artifact_class,
            "credential_required": self.credential_required,
            "terms_review_status": self.terms_review_status, "decision": self.decision,
            "field_allowlist": list(self.field_allowlist),
            "field_denylist": list(self.field_denylist),
            "derivative_reconstruction_risk": self.derivative_reconstruction_risk,
            "reviewer": self.reviewer, "decision_evidence": list(self.decision_evidence),
            "next_review_at": self.next_review_at,
        }
        for name in ("secret_storage_class", "license_or_terms", "terms_reviewed_at", "storage_rule", "redistribution_rule", "retention_rule", "attribution_rule", "deidentification_rule"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


@dataclass(frozen=True)
class PublicationMatrix:
    schema_version: str
    model_version: str
    as_of: str
    matrix_id: str
    reviewed_at: str
    rows: tuple[PublicationMatrixRow, ...]
    lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        validate_publication_matrix(self)

    def _payload_without_id(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "model_version": self.model_version,
                "as_of": self.as_of, "reviewed_at": self.reviewed_at,
                "rows": [row.to_payload() for row in sorted(self.rows, key=lambda r: (r.source_id, r.artifact_class))],
                "lineage": dict(self.lineage)}

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_id(), "matrix_id": self.matrix_id}

    def object_sha256(self) -> str:
        return sha256_canonical_object_hash(self.to_payload())

    def find_row(self, source_id: str, artifact_class: str) -> PublicationMatrixRow | None:
        return next((r for r in self.rows if r.source_id == source_id and r.artifact_class == artifact_class), None)

    def payload_schema(self, source_id: str, artifact_class: str, mode: str) -> Mapping[str, Any]:
        evidence = load_publication_evidence()
        return next(item["schema"] for item in evidence["payload_schemas"] if (item["source_id"], item["artifact_class"], item["mode"]) == (source_id, artifact_class, mode))


def _fields(source_id: str, artifact_class: str) -> tuple[str, ...]:
    schema = _default_schema(source_id, artifact_class)
    return tuple(sorted(schema["properties"]))


def make_default_publication_matrix(model_version: str = "Scryglass-v2.0.0-b2", as_of: str | datetime = "2026-07-27T14:00:00Z", *, matrix_id: str | None = None) -> PublicationMatrix:
    as_of_dt = _utc(as_of)
    evidence = load_publication_evidence()
    training = json.loads((REPO_ROOT / TRAINING_LOCATOR).read_bytes())
    source = json.loads((REPO_ROOT / SOURCE_LOCATOR).read_bytes())
    rows = []
    for entry in registered_sources():
        for artifact_class in entry.artifact_classes:
            decision = PublicationMatrixDecision.PROHIBITED if entry.source_id == "scryglass:source:user-auth" else (PublicationMatrixDecision.PRIVATE if entry.source_id == "scryglass:source:grid" else PublicationMatrixDecision.PRIVATE_PENDING_REVIEW)
            terms = "prohibited" if decision == PublicationMatrixDecision.PROHIBITED else ("restricted" if decision == PublicationMatrixDecision.PRIVATE else "pending")
            row_data = dict(source_id=entry.source_id, source_name=entry.source_name, source_url=entry.source_url, owner=entry.owner, access_method=entry.access_method, artifact_class=artifact_class, credential_required=entry.credential_required, terms_review_status=terms, decision=decision, field_allowlist=_fields(entry.source_id, artifact_class), field_denylist=("account_id", "api_key", "authorization", "contact", "email", "password", "secret", "token"), derivative_reconstruction_risk="high" if entry.source_id in {"scryglass:source:grid", "scryglass:source:user-auth", "scryglass:source:model-weights"} else "unknown", reviewer="l1-policy-unapproved", decision_evidence=("no_captured_terms_bytes", "no_user_publication_decision"), next_review_at=to_rfc3339(as_of_dt + timedelta(days=90)), secret_storage_class=entry.secret_storage_class, storage_rule="private_review_storage", redistribution_rule="deny_publication", retention_rule="retain_only_with_lineage", attribution_rule="source_attribution_required", deidentification_rule="publication_not_approved")
            row_id = "scryglass:publication-row:" + sha256_canonical_object_hash(_jsonable(row_data))
            rows.append(PublicationMatrixRow(row_id=row_id, **row_data))
    lineage = {
        "manifest_id": evidence["evidence_id"],
        "training_snapshot_id": training["snapshot_id"],
        "source_snapshot_ids": [source["snapshot_id"]],
        "artifact_sha256": sha256_canonical_object_hash(evidence),
        "source_tree_sha256": training["source_tree_sha256"],
        "train_cutoff": training["train_cutoff"],
        "environment_lock_sha256": training["environment_lock_sha256"],
        "code_commit": None,
    }
    matrix = PublicationMatrix("2.0.0", model_version, to_rfc3339(as_of_dt), "", to_rfc3339(as_of_dt), tuple(rows), lineage)
    expected = _matrix_id(matrix)
    if matrix_id and matrix_id != expected:
        raise PublicationMatrixError("matrix_id must be content-derived")
    object.__setattr__(matrix, "matrix_id", expected)
    return matrix


def validate_publication_matrix(matrix: PublicationMatrix) -> None:
    parse_rfc3339(matrix.as_of); parse_rfc3339(matrix.reviewed_at)
    if matrix.as_of != matrix.reviewed_at:
        raise PublicationMatrixError("reviewed_at must equal deterministic as_of")
    evidence = load_publication_evidence()
    if dict(matrix.lineage).get("manifest_id") != evidence["evidence_id"] or dict(matrix.lineage).get("artifact_sha256") != sha256_canonical_object_hash(evidence):
        raise PublicationMatrixError("matrix lineage evidence drift")
    expected = {(s.source_id, c) for s in registered_sources() for c in s.artifact_classes}
    actual = {(r.source_id, r.artifact_class) for r in matrix.rows}
    if actual != expected or len(actual) != len(matrix.rows):
        raise PublicationMatrixError("matrix applicability coverage incomplete")
    for row in matrix.rows:
        _validate_row(row, matrix=matrix)
    if matrix.matrix_id and matrix.matrix_id != _matrix_id(matrix):
        raise PublicationMatrixError("matrix_id must be content-derived")
    if not matrix.matrix_id:
        object.__setattr__(matrix, "matrix_id", _matrix_id(matrix))


def _validate_row(
    row: PublicationMatrixRow,
    *,
    matrix: PublicationMatrix | None = None,
) -> None:
    _validate_source_url(row.source_url)
    if row.decision in {PublicationMatrixDecision.PUBLIC, PublicationMatrixDecision.AUTHENTICATED}:
        _validate_legal_evidence(row, matrix=matrix)
    for field in row.field_allowlist:
        _validate_forbidden_filters((field,), "publication field")
    expected = "scryglass:publication-row:" + sha256_canonical_object_hash({k: v for k, v in row.to_payload().items() if k != "row_id"})
    if row.row_id != expected:
        raise PublicationMatrixError("row_id must be content-derived")


def _decision_mode(row: PublicationMatrixRow) -> str:
    return (
        "public"
        if row.decision == PublicationMatrixDecision.PUBLIC
        else "authenticated"
    )


def _review_scope(row: PublicationMatrixRow) -> str:
    return f"publication:{row.source_id}:{row.artifact_class}:{_decision_mode(row)}"


def _row_binding_id(row: PublicationMatrixRow) -> str:
    return "scryglass:publication-row-binding:" + sha256_canonical_object_hash(
        {
            "source_id": row.source_id,
            "artifact_class": row.artifact_class,
            "audience_mode": _decision_mode(row),
            "source_url": row.source_url,
        }
    )


def _matrix_binding_id(matrix: PublicationMatrix) -> str:
    return "scryglass:publication-matrix-binding:" + sha256_canonical_object_hash(
        {
            "schema_version": matrix.schema_version,
            "model_version": matrix.model_version,
            "as_of": matrix.as_of,
            "lineage": dict(matrix.lineage),
        }
    )


def _evidence_value(row: PublicationMatrixRow, prefix: str) -> str:
    matches = [
        value.removeprefix(prefix)
        for value in row.decision_evidence
        if value.startswith(prefix)
    ]
    if len(matches) != 1 or not matches[0]:
        raise PublicationMatrixError(
            "publication approval requires captured terms bytes and one "
            f"{prefix[:-1]}"
        )
    return matches[0]


def _load_typed_evidence(
    row: PublicationMatrixRow,
    kind: str,
) -> tuple[dict[str, Any], str]:
    locator = _evidence_value(row, f"{kind}_locator:")
    expected_hash = _evidence_value(row, f"{kind}_sha256:")
    try:
        normalized = normalize_source_tree_path(locator)
        if normalized != locator:
            raise ValueError("non-normalized")
        path = resolve_repository_file(REPO_ROOT, locator)
    except ValueError as err:
        raise PublicationMatrixError(
            f"{kind} requires a resolvable repository locator"
        ) from err
    raw = path.read_bytes()
    if sha256_raw_bytes_hash(raw) != expected_hash:
        raise PublicationMatrixError(f"{kind} raw-byte hash drift")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise PublicationMatrixError(f"{kind} must be typed JSON evidence") from err
    if not isinstance(payload, dict):
        raise PublicationMatrixError(f"{kind} must be a typed object")
    object_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"evidence_id", "object_sha256"}
    }
    object_hash = sha256_canonical_object_hash(object_payload)
    if payload.get("object_sha256") != object_hash:
        raise PublicationMatrixError(f"{kind} canonical object hash drift")
    expected_id = f"scryglass:{kind.replace('_', '-')}:{object_hash}"
    if payload.get("evidence_id") != expected_id:
        raise PublicationMatrixError(f"{kind} content-addressed id drift")
    return payload, locator


def _validate_evidence_scope(
    payload: Mapping[str, Any],
    row: PublicationMatrixRow,
    kind: str,
) -> None:
    common = {
        "schema_version", "evidence_type", "evidence_id", "source_id",
        "artifact_class", "audience_mode", "review_scope", "as_of",
        "object_sha256",
    }
    exact_fields = {
        "terms_capture": common | {
            "captured_at", "terms_url", "terms_content_locator",
            "terms_content_sha256",
        },
        "terms_review": common | {
            "reviewed_at", "reviewer", "decision_status", "terms_capture_id",
            "terms_locator", "terms_sha256",
        },
        "user_publication_decision": common | {
            "reviewed_at", "reviewer", "decision_status", "terms_capture_id",
            "terms_review_id", "c4_decision_reference", "row_binding_id",
            "matrix_binding_id", "c4_packet_locator", "c4_packet_id",
            "c4_packet_sha256",
        },
    }
    if set(payload) != exact_fields[kind]:
        raise PublicationMatrixError(f"{kind} typed schema mismatch")
    expected = {
        "source_id": row.source_id,
        "artifact_class": row.artifact_class,
        "audience_mode": _decision_mode(row),
        "review_scope": _review_scope(row),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise PublicationMatrixError(f"{kind} source/class/mode/scope mismatch")
    if payload.get("schema_version") != "2.0.0" or payload.get(
        "evidence_type"
    ) != kind:
        raise PublicationMatrixError(f"{kind} typed schema mismatch")


def _validate_terms_bytes(
    capture: Mapping[str, Any],
    row: PublicationMatrixRow,
) -> tuple[str, str]:
    terms_url = capture.get("terms_url")
    if not isinstance(terms_url, str):
        raise PublicationMatrixError("terms capture lacks source terms URL")
    _validate_source_url(terms_url)
    locator = capture.get("terms_content_locator")
    expected_hash = capture.get("terms_content_sha256")
    if not isinstance(locator, str) or not isinstance(expected_hash, str):
        raise PublicationMatrixError("terms capture lacks exact terms locator/hash")
    if not locator.startswith("data/lol/v2/publication/legal/terms/") or not locator.endswith(
        (".html", ".txt", ".md")
    ):
        raise PublicationMatrixError("unrelated file cannot serve as captured terms")
    try:
        terms_path = resolve_repository_file(REPO_ROOT, locator)
    except ValueError as err:
        raise PublicationMatrixError("captured terms locator does not resolve") from err
    raw = terms_path.read_bytes()
    if sha256_raw_bytes_hash(raw) != expected_hash:
        raise PublicationMatrixError("captured terms raw-byte hash drift")
    try:
        text = raw.decode("utf-8").casefold()
    except UnicodeDecodeError as err:
        raise PublicationMatrixError("captured terms must be reviewable text") from err
    hostname = urlparse(row.source_url).hostname or row.source_name
    if len(raw) < 80 or not any(
        marker in text
        for marker in ("terms", "license", "redistribution", "permission")
    ) or not any(
        marker.casefold() in text
        for marker in (row.source_name, hostname)
    ):
        raise PublicationMatrixError("unrelated file cannot serve as captured terms")
    return locator, expected_hash


def _validate_legal_evidence(
    row: PublicationMatrixRow,
    *,
    matrix: PublicationMatrix | None = None,
    authority_environment: str = "production",
) -> None:
    if row.terms_review_status != "approved" or not row.terms_reviewed_at:
        raise PublicationMatrixError("publication approval requires completed review")
    capture, _ = _load_typed_evidence(row, "terms_capture")
    review, _ = _load_typed_evidence(row, "terms_review")
    decision, _ = _load_typed_evidence(row, "user_publication_decision")
    for kind, payload in (
        ("terms_capture", capture),
        ("terms_review", review),
        ("user_publication_decision", decision),
    ):
        _validate_evidence_scope(payload, row, kind)
    terms_locator, terms_hash = _validate_terms_bytes(capture, row)
    if row.license_or_terms != f"repo://scryglass/{terms_locator}":
        raise PublicationMatrixError("row terms locator does not match typed capture")
    capture_at = parse_rfc3339(capture.get("captured_at", ""))
    capture_as_of = parse_rfc3339(capture.get("as_of", ""))
    review_at = parse_rfc3339(review.get("reviewed_at", ""))
    review_as_of = parse_rfc3339(review.get("as_of", ""))
    decision_at = parse_rfc3339(decision.get("reviewed_at", ""))
    decision_as_of = parse_rfc3339(decision.get("as_of", ""))
    row_reviewed_at = parse_rfc3339(row.terms_reviewed_at)
    if not (
        capture_at <= review_at <= review_as_of
        and capture_at <= decision_at <= decision_as_of
        and review_at == row_reviewed_at
        and capture_as_of == review_as_of == decision_as_of
    ):
        raise PublicationMatrixError("legal evidence time/as_of mismatch")
    if review.get("reviewer") != row.reviewer or decision.get("reviewer") != row.reviewer:
        raise PublicationMatrixError("legal evidence reviewer mismatch")
    if row.reviewer.casefold() in {"", "pending", "unreviewed", "l1-policy-unapproved"}:
        raise PublicationMatrixError("publication approval requires an identified reviewer")
    if review.get("decision_status") != "approved" or decision.get(
        "decision_status"
    ) != "approved":
        raise PublicationMatrixError("legal/user decision is not approved")
    if (
        review.get("terms_capture_id") != capture["evidence_id"]
        or review.get("terms_locator") != terms_locator
        or review.get("terms_sha256") != terms_hash
        or decision.get("terms_capture_id") != capture["evidence_id"]
        or decision.get("terms_review_id") != review["evidence_id"]
    ):
        raise PublicationMatrixError("legal evidence chain mismatch")
    row_binding = _row_binding_id(row)
    if decision.get("row_binding_id") != row_binding:
        raise PublicationMatrixError("user decision row binding mismatch")
    matrix_binding = decision.get("matrix_binding_id")
    if not isinstance(matrix_binding, str) or not matrix_binding.startswith(
        "scryglass:publication-matrix-binding:"
    ):
        raise PublicationMatrixError("user decision matrix binding missing")
    if matrix is not None:
        if capture_as_of != parse_rfc3339(matrix.as_of):
            raise PublicationMatrixError("legal evidence matrix as_of mismatch")
        if matrix_binding != _matrix_binding_id(matrix):
            raise PublicationMatrixError("user decision matrix binding mismatch")
    _validate_c4_packet(
        row,
        decision,
        capture,
        review,
        matrix=matrix,
        authority_environment=authority_environment,
    )


def _load_c4_authority_registry(environment: str) -> dict[str, Any]:
    root = C4_AUTHORITY_ROOTS.get(environment)
    if root is None:
        raise PublicationMatrixError("unknown C4 authority environment")
    locator, pinned_hash = root
    try:
        raw = resolve_repository_file(REPO_ROOT, locator).read_bytes()
    except ValueError as err:
        raise PublicationMatrixError("pinned C4 authority root does not resolve") from err
    if sha256_raw_bytes_hash(raw) != pinned_hash:
        raise PublicationMatrixError("pinned C4 authority root hash drift")
    payload = json.loads(raw)
    required = {
        "schema_version", "registry_id", "environment",
        "contract_tree_sha256", "matrix_locator", "matrix_id",
        "matrix_object_sha256", "authorities", "approved_packets",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != "2.0.0"
        or payload.get("environment") != environment
        or payload.get("contract_tree_sha256") != CONTRACT_TREE_SHA256
        or payload.get("matrix_locator")
        != "data/lol/v2/publication/publication-matrix-b2.json"
    ):
        raise PublicationMatrixError("invalid C4 authority root")
    expected_id = "scryglass:c4-authority-registry:" + (
        sha256_canonical_object_hash(
            {key: value for key, value in payload.items() if key != "registry_id"}
        )
    )
    if payload.get("registry_id") != expected_id:
        raise PublicationMatrixError("C4 authority root id drift")
    if environment == "production" and (
        payload["authorities"] or payload["approved_packets"]
    ):
        raise PublicationMatrixError("production C4 authority root must remain empty")
    return payload


def _validate_c4_packet(
    row: PublicationMatrixRow,
    decision: Mapping[str, Any],
    capture: Mapping[str, Any],
    review: Mapping[str, Any],
    *,
    matrix: PublicationMatrix | None,
    authority_environment: str,
) -> None:
    registry = _load_c4_authority_registry(authority_environment)
    if matrix is None:
        raise PublicationMatrixError(
            "C4 approval requires exact publication matrix binding"
        )
    if (
        registry["matrix_id"] != matrix.matrix_id
        or registry["matrix_object_sha256"] != matrix.object_sha256()
    ):
        raise PublicationMatrixError("C4 authority root matrix lineage mismatch")
    packet_locator = decision.get("c4_packet_locator")
    packet_id = decision.get("c4_packet_id")
    packet_hash = decision.get("c4_packet_sha256")
    registered = next(
        (
            item
            for item in registry["approved_packets"]
            if item.get("packet_locator") == packet_locator
            and item.get("packet_id") == packet_id
            and item.get("packet_bytes_sha256") == packet_hash
        ),
        None,
    )
    if registered is None:
        raise PublicationMatrixError(
            "C4 packet is not authorized by the pinned authority root"
        )
    try:
        raw = resolve_repository_file(REPO_ROOT, packet_locator).read_bytes()
    except (TypeError, ValueError) as err:
        raise PublicationMatrixError("registered C4 packet does not resolve") from err
    if sha256_raw_bytes_hash(raw) != packet_hash:
        raise PublicationMatrixError("registered C4 packet raw-byte hash drift")
    packet = json.loads(raw)
    required = {
        "schema_version", "packet_type", "packet_id", "object_sha256",
        "authority_id", "approver_id", "source_id", "artifact_class",
        "audience_mode", "review_scope", "row_binding_id",
        "matrix_binding_id", "terms_capture_id", "terms_review_id",
        "decision_time", "decision_status", "c4_decision_id",
    }
    if (
        not isinstance(packet, dict)
        or set(packet) != required
        or packet.get("schema_version") != "2.0.0"
        or packet.get("packet_type") != "c4_publication_decision"
    ):
        raise PublicationMatrixError("invalid typed C4 decision packet")
    object_payload = {
        key: value
        for key, value in packet.items()
        if key not in {"packet_id", "object_sha256"}
    }
    object_hash = sha256_canonical_object_hash(object_payload)
    if (
        packet["object_sha256"] != object_hash
        or packet["packet_id"] != f"scryglass:c4-decision-packet:{object_hash}"
    ):
        raise PublicationMatrixError("C4 decision packet object/id drift")
    expected_registered = {
        "packet_locator": packet_locator,
        "packet_id": packet["packet_id"],
        "packet_bytes_sha256": packet_hash,
        "packet_object_sha256": packet["object_sha256"],
        "c4_decision_id": packet["c4_decision_id"],
        "authority_id": packet["authority_id"],
        "approver_id": packet["approver_id"],
    }
    if registered != expected_registered:
        raise PublicationMatrixError("C4 registry packet binding mismatch")
    authority = next(
        (
            item
            for item in registry["authorities"]
            if item.get("authority_id") == packet["authority_id"]
        ),
        None,
    )
    if authority is None or packet["approver_id"] not in authority.get(
        "approver_ids", []
    ):
        raise PublicationMatrixError("C4 approver is not authorized")
    expected_binding = {
        "source_id": row.source_id,
        "artifact_class": row.artifact_class,
        "audience_mode": _decision_mode(row),
        "review_scope": _review_scope(row),
        "row_binding_id": _row_binding_id(row),
        "matrix_binding_id": _matrix_binding_id(matrix),
        "terms_capture_id": capture["evidence_id"],
        "terms_review_id": review["evidence_id"],
        "decision_time": decision["reviewed_at"],
        "decision_status": "approved",
    }
    if any(packet.get(key) != value for key, value in expected_binding.items()):
        raise PublicationMatrixError("C4 packet scope/lineage/time mismatch")
    c4_core = {
        key: packet[key]
        for key in (
            "authority_id", "approver_id", "source_id", "artifact_class",
            "audience_mode", "review_scope", "row_binding_id",
            "matrix_binding_id", "terms_capture_id", "terms_review_id",
            "decision_time", "decision_status",
        )
    }
    expected_c4 = "scryglass:c4-publication-decision:" + (
        sha256_canonical_object_hash(c4_core)
    )
    if (
        packet["c4_decision_id"] != expected_c4
        or decision.get("c4_decision_reference") != expected_c4
        or decision.get("reviewer") != packet["approver_id"]
    ):
        raise PublicationMatrixError("C4 decision/approver reference mismatch")


def publication_matrix_from_payload(payload: Mapping[str, Any]) -> PublicationMatrix:
    rows = []
    for raw in payload["rows"]:
        item = dict(raw)
        for key in ("field_allowlist", "field_denylist", "decision_evidence"):
            item[key] = tuple(item[key])
        rows.append(PublicationMatrixRow(**item))
    return PublicationMatrix(payload["schema_version"], payload["model_version"], payload["as_of"], payload["matrix_id"], payload["reviewed_at"], tuple(rows), payload["lineage"])


def load_publication_matrix(path: Path) -> PublicationMatrix:
    payload = json.loads(path.read_bytes())
    matrix = publication_matrix_from_payload(payload)
    if matrix.to_payload() != payload:
        raise PublicationMatrixError("matrix canonical drift")
    return matrix


def write_publication_matrix(matrix: PublicationMatrix, path: Path) -> str:
    data = canonical_json_bytes(matrix.to_payload()) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    return sha256_canonical_object_hash(matrix.to_payload())


def _validate_source_url(value: str) -> None:
    if value.startswith("repo://scryglass/"):
        relative = value.removeprefix("repo://scryglass/")
        try:
            resolve_repository_file(REPO_ROOT, relative)
        except ValueError as err:
            raise PublicationMatrixError("repository source_url must resolve") from err
        return
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query:
        raise PublicationMatrixError("invalid source_url")
    if parsed.hostname.endswith((".invalid", ".example")):
        raise PublicationMatrixError("placeholder source_url")
    try:
        addr = ip_address(parsed.hostname)
        if addr.is_private or addr.is_loopback or addr.is_reserved:
            raise PublicationMatrixError("private source_url")
    except ValueError:
        pass


def _matrix_id(matrix: PublicationMatrix) -> str:
    return "scryglass:publication-matrix:" + sha256_canonical_object_hash(matrix._payload_without_id())


def _utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None: raise PublicationMatrixError("timezone required")
        return value.astimezone(timezone.utc)
    return parse_rfc3339(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple): return [_jsonable(v) for v in value]
    if isinstance(value, Mapping): return {k: _jsonable(v) for k, v in value.items()}
    return value


__all__ = ["ARTIFACT_CLASSES", "MODES", "PublicationMatrix", "PublicationMatrixDecision", "PublicationMatrixError", "PublicationMatrixRow", "SourceRegistryEntry", "load_publication_evidence", "load_publication_matrix", "make_default_publication_matrix", "make_publication_evidence", "publication_matrix_from_payload", "registered_sources", "validate_publication_matrix", "write_publication_matrix"]
