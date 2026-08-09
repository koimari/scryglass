from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator, FormatChecker, RefResolver

import lol_kills.v2.provenance.allowlist as allowlist_module
import lol_kills.v2.provenance.publication as publication_module
from lol_kills.v2.data.common import (
    canonical_json_bytes,
    sha256_canonical_object_hash,
    sha256_raw_bytes_hash,
)
from lol_kills.v2.provenance.allowlist import (
    ArtifactAllowlistError,
    _scan_private,
    _load_allowed_recipe_registry,
    _validate_dag,
    _validate_transform_recipe,
    _validate_typed,
    enforce_candidate_publication,
    load_artifact_allowlist,
    load_transform_manifest,
    load_transform_recipe,
    make_candidate_artifact,
)
from lol_kills.v2.provenance.publication import (
    MODES,
    PublicationMatrixDecision,
    PublicationMatrixError,
    PublicationMatrixRow,
    _load_c4_authority_registry,
    _matrix_binding_id,
    _review_scope,
    _row_binding_id,
    _validate_legal_evidence,
    load_publication_evidence,
    load_publication_matrix,
    make_default_publication_matrix,
    make_publication_evidence,
    registered_sources,
)

ROOT = Path(__file__).resolve().parents[3]
PUB = ROOT / "data/lol/v2/publication"
MATRIX = PUB / "publication-matrix-b2.json"
TRANSFORM = PUB / "transforms/two-source-derived-b2.json"
RECIPE = PUB / "recipes/two-source-derived-b2.recipe.json"


def _matrix():
    return load_publication_matrix(MATRIX)


def test_b2_golden_matrix_validates_frozen_draft_202012_schema() -> None:
    contracts = ROOT / "docs/model-v2/contracts"
    schema = json.loads((contracts / "publication-matrix.schema.json").read_bytes())
    common = json.loads((contracts / "common.schema.json").read_bytes())
    instance = json.loads(MATRIX.read_bytes())
    resolver = RefResolver.from_schema(
        schema, store={common["$id"]: common, "common.schema.json": common}
    )
    errors = list(
        Draft202012Validator(
            schema,
            resolver=resolver,
            format_checker=FormatChecker(),
        ).iter_errors(instance)
    )
    assert errors == []
    assert set(instance) == {
        "schema_version", "model_version", "as_of", "matrix_id",
        "reviewed_at", "rows", "lineage",
    }
    required_row = set(schema["properties"]["rows"]["items"]["required"])
    assert all(required_row <= set(row) for row in instance["rows"])


def test_b2_registry_includes_every_contract_source_class() -> None:
    ids = {row.source_id for row in registered_sources()}
    assert {
        "scryglass:source:oracle-elixir",
        "scryglass:source:riot-ddragon",
        "scryglass:source:riot-docs",
        "scryglass:source:riot-api",
        "scryglass:source:riot-esports-live",
        "scryglass:source:leaguepedia",
        "scryglass:source:grid",
        "scryglass:source:manual-ontology",
        "scryglass:source:derived-features",
        "scryglass:source:model-code",
        "scryglass:source:model-weights",
        "scryglass:source:evaluation-reports",
        "scryglass:source:user-auth",
    } <= ids


def test_b2_evidence_has_complete_applicability_and_typed_mode_schemas() -> None:
    evidence = load_publication_evidence()
    assert evidence == make_publication_evidence()
    expected = {
        (source.source_id, artifact_class, mode)
        for source in registered_sources()
        for artifact_class in source.artifact_classes
        for mode in MODES
    }
    actual = {
        (row["source_id"], row["artifact_class"], row["mode"])
        for row in evidence["payload_schemas"]
    }
    assert actual == expected
    assert evidence["applicability_status"] == "complete"
    matrix = _matrix()
    assert matrix.lineage["manifest_id"] == evidence["evidence_id"]
    assert matrix.lineage["artifact_sha256"] == sha256_canonical_object_hash(evidence)


def test_b2_matrix_has_exact_applicable_rows_and_content_ids() -> None:
    matrix = _matrix()
    expected = {
        (source.source_id, artifact_class)
        for source in registered_sources()
        for artifact_class in source.artifact_classes
    }
    assert {(row.source_id, row.artifact_class) for row in matrix.rows} == expected
    assert len(matrix.rows) == len(expected)
    rebuilt = make_default_publication_matrix()
    assert rebuilt.matrix_id == matrix.matrix_id
    assert rebuilt.object_sha256() == matrix.object_sha256()


def test_b2_no_source_has_fabricated_public_or_authenticated_approval() -> None:
    matrix = _matrix()
    assert all(
        row.decision
        in {
            PublicationMatrixDecision.PRIVATE_PENDING_REVIEW,
            PublicationMatrixDecision.PRIVATE,
            PublicationMatrixDecision.PROHIBITED,
        }
        for row in matrix.rows
    )
    ddragon = matrix.find_row("scryglass:source:riot-ddragon", "raw")
    assert ddragon.decision == PublicationMatrixDecision.PRIVATE_PENDING_REVIEW
    assert ddragon.terms_review_status == "pending"
    assert ddragon.license_or_terms is None
    assert "no_user_publication_decision" in ddragon.decision_evidence


def test_b2_terms_label_hash_without_captured_bytes_cannot_approve() -> None:
    row = _matrix().find_row("scryglass:source:riot-ddragon", "raw")
    payload = row.to_payload()
    payload.update(
        row_id="scryglass:publication-row:tampered",
        decision="public",
        terms_review_status="approved",
        license_or_terms="repo://scryglass/data/lol/v2/publication/missing-terms.json",
        terms_reviewed_at="2026-07-27T14:00:00Z",
        decision_evidence=[
            "terms_sha256:" + "a" * 64,
            "user_decision_sha256:" + "b" * 64,
        ],
    )
    for name in ("field_allowlist", "field_denylist", "decision_evidence"):
        payload[name] = tuple(payload[name])
    with pytest.raises(PublicationMatrixError, match="captured terms bytes"):
        PublicationMatrixRow(**payload)


def _write_typed_evidence(root: Path, locator: str, payload: dict) -> tuple[str, str]:
    kind = payload["evidence_type"]
    payload["object_sha256"] = sha256_canonical_object_hash(payload)
    payload["evidence_id"] = (
        f"scryglass:{kind.replace('_', '-')}:{payload['object_sha256']}"
    )
    raw = canonical_json_bytes(payload) + b"\n"
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return payload["evidence_id"], sha256_raw_bytes_hash(raw)


def _route_legal_evidence_to_tmp(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = publication_module.resolve_repository_file

    def resolve(repo_root: Path, locator: str) -> Path:
        selected_root = (
            root
            if locator.startswith("data/lol/v2/publication/legal/")
            else ROOT
        )
        return original(selected_root, locator)

    monkeypatch.setattr(publication_module, "resolve_repository_file", resolve)


def _approved_test_row(
    root: Path,
    *,
    capture_overrides: dict | None = None,
    review_overrides: dict | None = None,
    decision_overrides: dict | None = None,
    unrelated_terms: bool = False,
) -> tuple[PublicationMatrixRow, SimpleNamespace, dict[str, Path]]:
    as_of = "2026-07-27T14:00:00Z"
    reviewed_at = "2026-07-27T13:00:00Z"
    terms_locator = "data/lol/v2/publication/legal/terms/riot-ddragon-terms.txt"
    terms_path = root / terms_locator
    terms_path.parent.mkdir(parents=True, exist_ok=True)
    terms_text = (
        "This is an unrelated implementation file with ordinary source code."
        if unrelated_terms
        else (
            "Riot Data Dragon terms and license capture. Redistribution and "
            "publication permission must follow the official Riot Games terms. "
        )
    )
    terms_path.write_text(terms_text, encoding="utf-8")
    terms_hash = sha256_raw_bytes_hash(terms_path.read_bytes())
    source_id = "scryglass:source:riot-ddragon"
    artifact_class = "raw"
    mode = "public"
    scope = f"publication:{source_id}:{artifact_class}:{mode}"
    capture = {
        "schema_version": "2.0.0",
        "evidence_type": "terms_capture",
        "source_id": source_id,
        "artifact_class": artifact_class,
        "audience_mode": mode,
        "review_scope": scope,
        "captured_at": "2026-07-27T12:00:00Z",
        "as_of": as_of,
        "terms_url": "https://www.riotgames.com/en/terms-of-service",
        "terms_content_locator": terms_locator,
        "terms_content_sha256": terms_hash,
    }
    capture.update(capture_overrides or {})
    capture_locator = "data/lol/v2/publication/legal/evidence/terms-capture.json"
    capture_id, capture_raw_hash = _write_typed_evidence(
        root, capture_locator, capture
    )
    review = {
        "schema_version": "2.0.0",
        "evidence_type": "terms_review",
        "source_id": source_id,
        "artifact_class": artifact_class,
        "audience_mode": mode,
        "review_scope": scope,
        "reviewed_at": reviewed_at,
        "as_of": as_of,
        "reviewer": "test-reviewer",
        "decision_status": "approved",
        "terms_capture_id": capture_id,
        "terms_locator": terms_locator,
        "terms_sha256": terms_hash,
    }
    review.update(review_overrides or {})
    review_locator = "data/lol/v2/publication/legal/evidence/terms-review.json"
    review_id, review_raw_hash = _write_typed_evidence(root, review_locator, review)
    row_scope = SimpleNamespace(
        source_id=source_id,
        artifact_class=artifact_class,
        decision=PublicationMatrixDecision.PUBLIC,
        source_url="https://ddragon.leagueoflegends.com/",
    )
    matrix = _matrix()
    matrix_binding = _matrix_binding_id(matrix)
    caller_packet_locator = (
        "data/lol/v2/publication/legal/evidence/caller-created-c4-packet.json"
    )
    caller_packet = {
        "schema_version": "2.0.0",
        "packet_type": "c4_publication_decision",
        "caller_created": True,
    }
    caller_packet_raw = canonical_json_bytes(caller_packet) + b"\n"
    caller_packet_path = root / caller_packet_locator
    caller_packet_path.parent.mkdir(parents=True, exist_ok=True)
    caller_packet_path.write_bytes(caller_packet_raw)
    caller_packet_hash = sha256_raw_bytes_hash(caller_packet_raw)
    decision = {
        "schema_version": "2.0.0",
        "evidence_type": "user_publication_decision",
        "source_id": source_id,
        "artifact_class": artifact_class,
        "audience_mode": mode,
        "review_scope": scope,
        "reviewed_at": "2026-07-27T13:30:00Z",
        "as_of": as_of,
        "reviewer": "test-reviewer",
        "decision_status": "approved",
        "terms_capture_id": capture_id,
        "terms_review_id": review_id,
        "row_binding_id": _row_binding_id(row_scope),
        "matrix_binding_id": matrix_binding,
        "c4_packet_locator": caller_packet_locator,
        "c4_packet_id": (
            "scryglass:c4-decision-packet:"
            + sha256_canonical_object_hash(caller_packet)
        ),
        "c4_packet_sha256": caller_packet_hash,
    }
    decision.update(decision_overrides or {})
    decision["c4_decision_reference"] = (
        decision_overrides or {}
    ).get(
        "c4_decision_reference",
        "scryglass:c4-publication-decision:"
        + sha256_canonical_object_hash(
            {
                "review_scope": decision["review_scope"],
                "row_binding_id": decision["row_binding_id"],
                "matrix_binding_id": decision["matrix_binding_id"],
                "terms_review_id": decision["terms_review_id"],
            }
        ),
    )
    decision_locator = (
        "data/lol/v2/publication/legal/evidence/user-publication-decision.json"
    )
    _, decision_raw_hash = _write_typed_evidence(root, decision_locator, decision)
    base = _matrix().find_row(source_id, artifact_class).to_payload()
    base.update(
        decision=PublicationMatrixDecision.PUBLIC,
        terms_review_status="approved",
        reviewer="test-reviewer",
        terms_reviewed_at=reviewed_at,
        license_or_terms=f"repo://scryglass/{terms_locator}",
        decision_evidence=[
            f"terms_capture_locator:{capture_locator}",
            f"terms_capture_sha256:{capture_raw_hash}",
            f"terms_review_locator:{review_locator}",
            f"terms_review_sha256:{review_raw_hash}",
            f"user_publication_decision_locator:{decision_locator}",
            f"user_publication_decision_sha256:{decision_raw_hash}",
        ],
    )
    base["row_id"] = "scryglass:publication-row:" + sha256_canonical_object_hash(
        {key: value for key, value in base.items() if key != "row_id"}
    )
    for name in ("field_allowlist", "field_denylist", "decision_evidence"):
        base[name] = tuple(base[name])
    row = SimpleNamespace(**base)
    return row, matrix, {
        "terms": terms_path,
        "capture": root / capture_locator,
        "review": root / review_locator,
        "decision": root / decision_locator,
        "packet": caller_packet_path,
    }


def test_b2_self_consistent_caller_legal_chain_cannot_mint_c4_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _route_legal_evidence_to_tmp(tmp_path, monkeypatch)
    row, matrix, _ = _approved_test_row(tmp_path)
    with pytest.raises(PublicationMatrixError, match="pinned authority root"):
        _validate_legal_evidence(row, matrix=matrix)


def _pinned_test_authority_row() -> SimpleNamespace:
    base = _matrix().find_row(
        "scryglass:source:riot-ddragon", "raw"
    ).to_payload()
    base.update(
        decision=PublicationMatrixDecision.PUBLIC,
        terms_review_status="approved",
        reviewer="scryglass:approver:test-oracle",
        terms_reviewed_at="2026-07-27T13:00:00Z",
        license_or_terms=(
            "repo://scryglass/data/lol/v2/publication/legal/terms/"
            "test-only-riot-ddragon-terms.txt"
        ),
        decision_evidence=[
            (
                "terms_capture_locator:data/lol/v2/publication/legal/"
                "test-authority/terms-capture.json"
            ),
            "terms_capture_sha256:f53f5ed18a6f016f0369f50c343c8b476f19de69b1fffb6f72132d22f5136e72",
            (
                "terms_review_locator:data/lol/v2/publication/legal/"
                "test-authority/terms-review.json"
            ),
            "terms_review_sha256:b580c8f4373332696fe93dabc51385715f96811ef31e8721229be09d325e82bf",
            (
                "user_publication_decision_locator:data/lol/v2/publication/"
                "legal/test-authority/user-publication-decision.json"
            ),
            "user_publication_decision_sha256:2be03693c65b3d719719da9ecd2abf843804597a1fd625f297040973d5cdccba",
        ],
    )
    for name in ("field_allowlist", "field_denylist", "decision_evidence"):
        base[name] = tuple(base[name])
    return SimpleNamespace(**base)


def test_b2_separately_pinned_test_authority_oracle_is_exact() -> None:
    row = _pinned_test_authority_row()
    matrix = _matrix()
    _validate_legal_evidence(
        row,
        matrix=matrix,
        authority_environment="test_only",
    )
    with pytest.raises(PublicationMatrixError, match="pinned authority root"):
        _validate_legal_evidence(row, matrix=matrix)


def test_b2_c4_packet_copy_at_alternate_locator_is_not_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet_source = (
        PUB / "legal/test-authority/c4-decision-packet.json"
    )
    alternate_packet_locator = (
        "data/lol/v2/publication/legal/audit-temp/copied-packet.json"
    )
    alternate_packet = tmp_path / alternate_packet_locator
    alternate_packet.parent.mkdir(parents=True, exist_ok=True)
    alternate_packet.write_bytes(packet_source.read_bytes())
    decision = json.loads(
        (PUB / "legal/test-authority/user-publication-decision.json").read_bytes()
    )
    decision["c4_packet_locator"] = alternate_packet_locator
    object_payload = {
        key: value
        for key, value in decision.items()
        if key not in {"evidence_id", "object_sha256"}
    }
    decision_hash = sha256_canonical_object_hash(object_payload)
    decision["object_sha256"] = decision_hash
    decision["evidence_id"] = (
        f"scryglass:user-publication-decision:{decision_hash}"
    )
    decision_locator = (
        "data/lol/v2/publication/legal/audit-temp/copied-packet-decision.json"
    )
    decision_path = tmp_path / decision_locator
    decision_raw = canonical_json_bytes(decision) + b"\n"
    decision_path.write_bytes(decision_raw)
    original = publication_module.resolve_repository_file

    def resolve(root: Path, locator: str) -> Path:
        if locator == alternate_packet_locator:
            return alternate_packet
        if locator == decision_locator:
            return decision_path
        return original(ROOT, locator)

    monkeypatch.setattr(publication_module, "resolve_repository_file", resolve)
    row = _pinned_test_authority_row()
    evidence = list(row.decision_evidence)
    evidence[-2] = f"user_publication_decision_locator:{decision_locator}"
    evidence[-1] = (
        "user_publication_decision_sha256:"
        + sha256_raw_bytes_hash(decision_raw)
    )
    row.decision_evidence = tuple(evidence)
    with pytest.raises(PublicationMatrixError, match="pinned authority root"):
        _validate_legal_evidence(
            row,
            matrix=_matrix(),
            authority_environment="test_only",
        )


def test_b2_c4_registry_tamper_and_self_rehash_cannot_mint_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = json.loads(
        (PUB / "c4-authority-registry-b2.json").read_bytes()
    )
    registry["authorities"] = [
        {
            "authority_id": "caller",
            "approver_ids": ["caller-reviewer"],
        }
    ]
    registry["registry_id"] = "scryglass:c4-authority-registry:" + (
        sha256_canonical_object_hash(
            {key: value for key, value in registry.items() if key != "registry_id"}
        )
    )
    tampered = tmp_path / "caller-c4-registry.json"
    tampered.write_bytes(canonical_json_bytes(registry) + b"\n")
    original = publication_module.resolve_repository_file
    root_locator = publication_module.C4_AUTHORITY_ROOTS["production"][0]

    monkeypatch.setattr(
        publication_module,
        "resolve_repository_file",
        lambda root, locator: (
            tampered if locator == root_locator else original(root, locator)
        ),
    )
    with pytest.raises(PublicationMatrixError, match="pinned.*hash drift"):
        _load_c4_authority_registry("production")


@pytest.mark.parametrize(
    "target,overrides,match",
    (
        ("capture", {"source_id": "scryglass:source:riot-docs"}, "source/class"),
        ("capture", {"artifact_class": "aggregate"}, "source/class"),
        ("review", {"audience_mode": "authenticated"}, "mode/scope"),
        ("review", {"review_scope": "publication:wrong"}, "mode/scope"),
        ("review", {"reviewed_at": "2026-07-27T15:00:00Z"}, "time/as_of"),
        ("review", {"terms_locator": "data/lol/v2/publication/legal/terms/other.txt"}, "chain"),
        ("decision", {"row_binding_id": "scryglass:publication-row-binding:wrong"}, "row binding"),
        ("decision", {"matrix_binding_id": "scryglass:publication-matrix-binding:wrong"}, "matrix binding"),
        ("decision", {"c4_decision_reference": "scryglass:c4-publication-decision:wrong"}, "C4"),
    ),
)
def test_b2_typed_legal_scope_time_chain_and_binding_mismatches_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    overrides: dict,
    match: str,
) -> None:
    _route_legal_evidence_to_tmp(tmp_path, monkeypatch)
    kwargs = {f"{target}_overrides": overrides}
    with pytest.raises(PublicationMatrixError, match=match):
        row, matrix, _ = _approved_test_row(tmp_path, **kwargs)
        _validate_legal_evidence(row, matrix=matrix)


def test_b2_unrelated_file_cannot_serve_as_terms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _route_legal_evidence_to_tmp(tmp_path, monkeypatch)
    with pytest.raises(PublicationMatrixError, match="unrelated file"):
        row, matrix, _ = _approved_test_row(tmp_path, unrelated_terms=True)
        _validate_legal_evidence(row, matrix=matrix)


@pytest.mark.parametrize("target", ("decision", "capture", "review"))
def test_b2_missing_typed_legal_file_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    _route_legal_evidence_to_tmp(tmp_path, monkeypatch)
    row, _, paths = _approved_test_row(tmp_path)
    paths[target].unlink()
    with pytest.raises(PublicationMatrixError, match="locator"):
        _validate_legal_evidence(row)


@pytest.mark.parametrize("target", ("terms", "capture", "review", "decision"))
def test_b2_typed_legal_raw_hash_drift_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    _route_legal_evidence_to_tmp(tmp_path, monkeypatch)
    row, _, paths = _approved_test_row(tmp_path)
    paths[target].write_bytes(paths[target].read_bytes() + b"mutated")
    with pytest.raises(PublicationMatrixError, match="hash drift"):
        _validate_legal_evidence(row)


def test_b2_transform_manifest_exactly_resolves_two_source_rows_and_output() -> None:
    manifest = load_transform_manifest(TRANSFORM.relative_to(ROOT).as_posix())
    recipe = load_transform_recipe(RECIPE.relative_to(ROOT).as_posix())
    assert manifest["recipe_id"] == recipe["recipe_id"]
    assert manifest["recipe_bytes_sha256"] == sha256_raw_bytes_hash(
        RECIPE.read_bytes()
    )
    assert manifest["output_node_id"] == "output"
    output = next(row for row in manifest["nodes"] if row["node_id"] == "output")
    assert output["direct_input_ids"] == ["input-json", "input-text"]
    candidate = make_candidate_artifact(
        lineage_manifest_locator=TRANSFORM.relative_to(ROOT).as_posix(),
        audience_mode="private",
    )
    row = enforce_candidate_publication(candidate, _matrix(), publication_mode="private")
    assert row.direct_input_ids == ("input-json", "input-text")


def _mutated_manifest(mutator) -> dict:
    payload = json.loads(TRANSFORM.read_bytes())
    mutator(payload)
    base = {key: value for key, value in payload.items() if key != "manifest_id"}
    payload["manifest_id"] = (
        "scryglass:transform-manifest:" + sha256_canonical_object_hash(base)
    )
    return payload


def _node(payload: dict, node_id: str) -> dict:
    return next(node for node in payload["nodes"] if node["node_id"] == node_id)


def test_b2_recipe_independently_rejects_removed_required_leaf() -> None:
    def mutate(payload: dict) -> None:
        payload["nodes"] = [
            node for node in payload["nodes"] if node["node_id"] != "input-text"
        ]
        _node(payload, "output")["direct_input_ids"] = ["input-json"]

    with pytest.raises(ArtifactAllowlistError, match="required recipe roles"):
        _validate_dag(_mutated_manifest(mutate))


def test_b2_recipe_rejects_reachable_extra_leaf() -> None:
    def mutate(payload: dict) -> None:
        extra = deepcopy(_node(payload, "input-json"))
        extra.update(node_id="extra-input", input_role="extra-role")
        payload["nodes"].append(extra)
        _node(payload, "output")["direct_input_ids"].append("extra-input")

    with pytest.raises(ArtifactAllowlistError, match="required recipe roles"):
        _validate_dag(_mutated_manifest(mutate))


def test_b2_recipe_rejects_duplicate_row_bound_under_another_role() -> None:
    def mutate(payload: dict) -> None:
        duplicate = deepcopy(_node(payload, "input-json"))
        duplicate.update(node_id="input-text", input_role="model_code_rows")
        payload["nodes"][1] = duplicate

    with pytest.raises(ArtifactAllowlistError, match="selector substitution"):
        _validate_dag(_mutated_manifest(mutate))


def test_b2_recipe_rejects_valid_source_rows_substituted_between_roles() -> None:
    def mutate(payload: dict) -> None:
        first = deepcopy(_node(payload, "input-json"))
        second = deepcopy(_node(payload, "input-text"))
        first.update(node_id="input-text", input_role="model_code_rows")
        second.update(node_id="input-json", input_role="ontology_records")
        payload["nodes"][0] = first
        payload["nodes"][1] = second

    with pytest.raises(ArtifactAllowlistError, match="selector substitution"):
        _validate_dag(_mutated_manifest(mutate))


def test_b2_manifest_recipe_identity_and_bytes_are_not_submitter_writable() -> None:
    for key in ("recipe_id", "recipe_bytes_sha256"):
        payload = _mutated_manifest(lambda value, key=key: value.update({key: "0" * 64}))
        with pytest.raises(ArtifactAllowlistError, match="recipe"):
            _validate_dag(payload)


@pytest.mark.parametrize(
    "field,match",
    (
        ("transform_code_sha256", "code hash drift"),
        ("transform_config_sha256", "config hash drift"),
    ),
)
def test_b2_recipe_code_and_config_hash_drift_fail(field: str, match: str) -> None:
    recipe = json.loads(RECIPE.read_bytes())
    recipe[field] = "1" * 64
    recipe["recipe_id"] = "scryglass:transform-recipe:" + (
        sha256_canonical_object_hash(
            {key: value for key, value in recipe.items() if key != "recipe_id"}
        )
    )
    with pytest.raises(ArtifactAllowlistError, match=match):
        _validate_transform_recipe(recipe)


def _route_audit_tmp(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = allowlist_module._resolve

    def resolve(locator: str) -> Path:
        if locator.startswith("data/lol/v2/publication/audit-temp/"):
            path = root / locator
            if not path.is_file():
                raise ArtifactAllowlistError("audit temp locator missing")
            return path
        return original(locator)

    monkeypatch.setattr(allowlist_module, "_resolve", resolve)


def _write_rehashed_manifest(root: Path, locator: str, payload: dict) -> None:
    payload["manifest_id"] = "scryglass:transform-manifest:" + (
        sha256_canonical_object_hash(
            {key: value for key, value in payload.items() if key != "manifest_id"}
        )
    )
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def test_b2_self_rehashed_temp_recipe_and_full_leaf_removal_are_unauthorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _route_audit_tmp(tmp_path, monkeypatch)
    recipe = json.loads(RECIPE.read_bytes())
    recipe["input_roles"] = recipe["input_roles"][:1]
    recipe["recipe_id"] = "scryglass:transform-recipe:" + (
        sha256_canonical_object_hash(
            {key: value for key, value in recipe.items() if key != "recipe_id"}
        )
    )
    recipe_locator = (
        "data/lol/v2/publication/audit-temp/self-rehashed-recipe.json"
    )
    recipe_raw = canonical_json_bytes(recipe) + b"\n"
    recipe_path = tmp_path / recipe_locator
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.write_bytes(recipe_raw)
    manifest = json.loads(TRANSFORM.read_bytes())
    manifest["recipe_locator"] = recipe_locator
    manifest["recipe_id"] = recipe["recipe_id"]
    manifest["recipe_bytes_sha256"] = sha256_raw_bytes_hash(recipe_raw)
    manifest["nodes"] = [
        node for node in manifest["nodes"] if node["node_id"] != "input-text"
    ]
    _node(manifest, "output")["direct_input_ids"] = ["input-json"]
    manifest_locator = (
        "data/lol/v2/publication/audit-temp/self-rehashed-manifest.json"
    )
    _write_rehashed_manifest(tmp_path, manifest_locator, manifest)
    with pytest.raises(ArtifactAllowlistError, match="not authorized"):
        load_transform_manifest(manifest_locator)
    with pytest.raises(ArtifactAllowlistError, match="not authorized"):
        make_candidate_artifact(
            lineage_manifest_locator=manifest_locator,
            audience_mode="private",
        )


def test_b2_canonical_recipe_at_an_alternate_locator_is_unauthorized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _route_audit_tmp(tmp_path, monkeypatch)
    alternate = "data/lol/v2/publication/audit-temp/recipe-copy.json"
    path = tmp_path / alternate
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(RECIPE.read_bytes())
    manifest = json.loads(TRANSFORM.read_bytes())
    manifest["recipe_locator"] = alternate
    manifest["recipe_bytes_sha256"] = sha256_raw_bytes_hash(path.read_bytes())
    locator = "data/lol/v2/publication/audit-temp/alternate-manifest.json"
    _write_rehashed_manifest(tmp_path, locator, manifest)
    with pytest.raises(ArtifactAllowlistError, match="not authorized"):
        load_transform_manifest(locator)


def test_b2_recipe_registry_tamper_and_self_rehash_cannot_move_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = PUB / "allowed-recipe-registry-b2.json"
    payload = json.loads(registry_path.read_bytes())
    payload["recipes"][0]["input_roles"] = payload["recipes"][0][
        "input_roles"
    ][:1]
    payload["registry_id"] = "scryglass:allowed-recipe-registry:" + (
        sha256_canonical_object_hash(
            {key: value for key, value in payload.items() if key != "registry_id"}
        )
    )
    tampered = tmp_path / "tampered-registry.json"
    tampered.write_bytes(canonical_json_bytes(payload) + b"\n")
    original = allowlist_module._resolve
    monkeypatch.setattr(
        allowlist_module,
        "_resolve",
        lambda locator: (
            tampered
            if locator == allowlist_module.RECIPE_REGISTRY_LOCATOR
            else original(locator)
        ),
    )
    with pytest.raises(ArtifactAllowlistError, match="pinned.*hash drift"):
        _load_allowed_recipe_registry()


def test_b2_self_asserted_data_dragon_source_identity_fails() -> None:
    payload = _mutated_manifest(
        lambda p: p["nodes"][0].update(
            source_id="scryglass:source:riot-ddragon"
        )
    )
    with pytest.raises(ArtifactAllowlistError, match="self-asserted"):
        _validate_dag(payload)


def test_b2_transform_output_cannot_self_assert_data_dragon_identity() -> None:
    payload = _mutated_manifest(
        lambda p: next(
            node for node in p["nodes"] if node["node_id"] == "output"
        ).update(
            source_id="scryglass:source:riot-ddragon",
            artifact_class="raw",
        )
    )
    with pytest.raises(ArtifactAllowlistError, match="self-assert external"):
        _validate_dag(payload)


@pytest.mark.parametrize(
    "mutator,match",
    (
        (
            lambda p: next(
                n for n in p["nodes"] if n["node_id"] == "output"
            ).update(direct_input_ids=["input-json"]),
            "disconnected",
        ),
        (
            lambda p: p["nodes"].append(
                {
                    **deepcopy(p["nodes"][0]),
                    "node_id": "extra-disconnected",
                }
            ),
            "disconnected",
        ),
        (
            lambda p: next(
                n for n in p["nodes"] if n["node_id"] == "output"
            ).update(direct_input_ids=["output"]),
            "cyclic",
        ),
        (
            lambda p: next(
                n for n in p["nodes"] if n["node_id"] == "input-json"
            ).update(bytes_sha256="1" * 63 + "2"),
            "lineage mismatch|mutated",
        ),
    ),
)
def test_b2_omitted_extra_cyclic_and_substituted_dag_inputs_fail(
    mutator, match: str
) -> None:
    with pytest.raises(ArtifactAllowlistError, match=match):
        _validate_dag(_mutated_manifest(mutator))


def test_b2_missing_or_mutated_candidate_lineage_fails() -> None:
    candidate = make_candidate_artifact(
        lineage_manifest_locator=TRANSFORM.relative_to(ROOT).as_posix(),
        audience_mode="private",
    )
    with pytest.raises(ArtifactAllowlistError, match="lineage"):
        enforce_candidate_publication(
            candidate.__class__(
                **{
                    **candidate.__dict__,
                    "lineage_manifest_sha256": "1" * 63 + "2",
                }
            ),
            _matrix(),
            publication_mode="private",
        )
    with pytest.raises(ArtifactAllowlistError, match="does not exist"):
        make_candidate_artifact(
            lineage_manifest_locator=(
                "data/lol/v2/publication/transforms/missing-transform.json"
            ),
            audience_mode="private",
        )


@pytest.mark.parametrize(
    "payload",
    (
        ["opaque"],
        "opaque",
        {},
    ),
)
def test_b2_top_level_list_string_and_empty_object_fail(payload) -> None:
    schema = _matrix().payload_schema(
        "scryglass:source:derived-features", "derived_rows", "private"
    )
    with pytest.raises(ArtifactAllowlistError):
        if payload == {}:
            _validate_typed(payload, schema)
        else:
            _validate_typed(payload, schema)


@pytest.mark.parametrize(
    "payload",
    (
        {"asset_path": {"account": "hidden"}},
        {"asset_path": {"email": "person@site.test"}},
        {"asset_path": {"contact": "hidden"}},
        {"asset_path": "sk_live_redacted"},
        {"asset_path": "person@site.test"},
    ),
)
def test_b2_nested_private_contact_and_secret_shaped_values_fail(payload) -> None:
    with pytest.raises(ArtifactAllowlistError):
        _scan_private(payload)


def test_b2_recursive_schema_rejects_nested_extra_fields_and_wrong_values() -> None:
    schema = _matrix().payload_schema(
        "scryglass:source:manual-ontology", "raw", "private"
    )
    valid = {"alpha": [1, 2, 3], "beta": "passb1", "nested": {"seed": 7}}
    _validate_typed(valid, schema)
    with pytest.raises(ArtifactAllowlistError, match="extra"):
        _validate_typed(
            {**valid, "nested": {"seed": 7, "email": "hidden"}}, schema
        )
    with pytest.raises(ArtifactAllowlistError, match="integer"):
        _validate_typed({**valid, "nested": {"seed": "7"}}, schema)


def test_b2_empty_public_allowlist_denies_even_empty_object() -> None:
    candidate = make_candidate_artifact(
        lineage_manifest_locator=TRANSFORM.relative_to(ROOT).as_posix(),
        audience_mode="public",
    )
    real = _matrix()
    empty = SimpleNamespace(
        field_allowlist=(),
        decision=PublicationMatrixDecision.PUBLIC,
    )
    fake = SimpleNamespace(
        find_row=lambda source_id, artifact_class: empty,
        payload_schema=real.payload_schema,
    )
    with pytest.raises(ArtifactAllowlistError, match="empty public"):
        enforce_candidate_publication(candidate, fake, publication_mode="public")


def test_b2_private_and_empty_public_authenticated_allowlists_round_trip() -> None:
    private = load_artifact_allowlist(PUB / "artifact-allowlist-private-b2.json")
    public = load_artifact_allowlist(PUB / "artifact-allowlist-public-b2.json")
    authenticated = load_artifact_allowlist(
        PUB / "artifact-allowlist-authenticated-b2.json"
    )
    assert len(private.rows) == 1
    assert public.rows == ()
    assert authenticated.rows == ()
    assert all(
        value.allowlist_id.startswith("scryglass:artifact-allowlist:")
        for value in (private, public, authenticated)
    )
