"""Artifact allowlists backed by exact, verified per-artifact DAG manifests."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..data import parse_rfc3339
from ..data.common import canonical_json_bytes, sha256_canonical_object_hash, sha256_raw_bytes_hash
from ..data.source_tree import normalize_source_tree_path, resolve_repository_file
from .publication import PublicationMatrix, PublicationMatrixDecision, load_publication_evidence, load_publication_matrix, registered_sources
from .snapshots import CONTRACT_TREE_SHA256, _reject_forbidden_recursive

REPO_ROOT = Path(__file__).resolve().parents[3]
MATRIX_LOCATOR = "data/lol/v2/publication/publication-matrix-b2.json"
SOURCE_LOCATOR = "data/lol/v2/snapshots/b1/source-snapshot-passb1.json"
SECRET = {"account", "accountid", "apikey", "authorization", "contact", "credential", "email", "password", "secret", "token"}
RECIPE_PREFIX = "scryglass:transform-recipe:"
RECIPE_REGISTRY_LOCATOR = (
    "data/lol/v2/publication/allowed-recipe-registry-b2.json"
)
RECIPE_REGISTRY_BYTES_SHA256 = (
    "c09f8a558c680b6177e617ee34b6ea5a656a6ba4e88522b4192514a922b4a142"
)


class ArtifactAllowlistError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateArtifact:
    artifact_id: str
    source_id: str
    artifact_class: str
    locator: str
    bytes_sha256: str
    object_sha256: str
    fields: tuple[str, ...]
    audience_mode: str
    lineage_manifest_locator: str
    lineage_manifest_id: str
    lineage_manifest_sha256: str

    def _payload_without_id(self) -> dict[str, Any]:
        return {k: v for k, v in self.to_payload().items() if k != "artifact_id"}

    def to_payload(self) -> dict[str, Any]:
        return {"artifact_id": self.artifact_id, "source_id": self.source_id, "artifact_class": self.artifact_class, "locator": self.locator, "bytes_sha256": self.bytes_sha256, "object_sha256": self.object_sha256, "fields": list(self.fields), "audience_mode": self.audience_mode, "lineage_manifest_locator": self.lineage_manifest_locator, "lineage_manifest_id": self.lineage_manifest_id, "lineage_manifest_sha256": self.lineage_manifest_sha256}


@dataclass(frozen=True)
class ArtifactAllowlistRow:
    artifact_id: str
    source_id: str
    artifact_class: str
    locator: str
    bytes_sha256: str
    object_sha256: str
    fields: tuple[str, ...]
    audience_mode: str
    lineage_manifest_locator: str
    lineage_manifest_id: str
    lineage_manifest_sha256: str
    effective_decision: str
    direct_input_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        payload = CandidateArtifact(self.artifact_id, self.source_id, self.artifact_class, self.locator, self.bytes_sha256, self.object_sha256, self.fields, self.audience_mode, self.lineage_manifest_locator, self.lineage_manifest_id, self.lineage_manifest_sha256).to_payload()
        payload.update(effective_decision=self.effective_decision, direct_input_ids=list(self.direct_input_ids))
        return payload


@dataclass(frozen=True)
class ArtifactAllowlist:
    schema_version: str
    model_version: str
    audience_mode: str
    as_of: str
    matrix_locator: str
    matrix_id: str
    matrix_object_sha256: str
    contract_tree_sha256: str
    rows: tuple[ArtifactAllowlistRow, ...]
    allowlist_id: str = ""

    def __post_init__(self) -> None:
        _validate_allowlist(self)

    def _payload_without_id(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "model_version": self.model_version, "audience_mode": self.audience_mode, "as_of": self.as_of, "matrix_locator": self.matrix_locator, "matrix_id": self.matrix_id, "matrix_object_sha256": self.matrix_object_sha256, "contract_tree_sha256": self.contract_tree_sha256, "rows": [r.to_payload() for r in sorted(self.rows, key=lambda r: r.artifact_id)]}

    def to_payload(self) -> dict[str, Any]:
        return {**self._payload_without_id(), "allowlist_id": self.allowlist_id}

    def object_sha256(self) -> str:
        return sha256_canonical_object_hash(self.to_payload())


def load_transform_manifest(locator: str) -> dict[str, Any]:
    payload = json.loads(_resolve(locator).read_bytes())
    base = {k: v for k, v in payload.items() if k != "manifest_id"}
    expected_id = "scryglass:transform-manifest:" + sha256_canonical_object_hash(base)
    if payload.get("manifest_id") != expected_id:
        raise ArtifactAllowlistError("transform manifest id drift")
    _validate_dag(payload)
    return payload


def load_transform_recipe(locator: str) -> dict[str, Any]:
    registry = _load_allowed_recipe_registry()
    registered = next(
        (
            item
            for item in registry["recipes"]
            if item["recipe_locator"] == locator
        ),
        None,
    )
    if registered is None:
        raise ArtifactAllowlistError(
            "transform recipe locator is not authorized by the pinned registry"
        )
    path = _resolve(locator)
    raw = path.read_bytes()
    if sha256_raw_bytes_hash(raw) != registered["recipe_bytes_sha256"]:
        raise ArtifactAllowlistError("registered transform recipe bytes drift")
    payload = json.loads(raw)
    _validate_transform_recipe(payload)
    expected = {
        "recipe_locator": locator,
        "recipe_id": payload["recipe_id"],
        "recipe_bytes_sha256": sha256_raw_bytes_hash(raw),
        "transform_code_locator": payload["transform_code_locator"],
        "transform_code_sha256": payload["transform_code_sha256"],
        "transform_config_locator": payload["transform_config_locator"],
        "transform_config_sha256": payload["transform_config_sha256"],
        "input_roles": payload["input_roles"],
        "output": payload["output"],
    }
    if registered != expected:
        raise ArtifactAllowlistError("transform recipe registry binding mismatch")
    return payload


def _load_allowed_recipe_registry() -> dict[str, Any]:
    raw = _resolve(RECIPE_REGISTRY_LOCATOR).read_bytes()
    if sha256_raw_bytes_hash(raw) != RECIPE_REGISTRY_BYTES_SHA256:
        raise ArtifactAllowlistError("pinned allowed-recipe registry hash drift")
    payload = json.loads(raw)
    required = {
        "schema_version", "registry_id", "contract_tree_sha256",
        "source_snapshot_locator", "source_snapshot_id",
        "source_snapshot_object_sha256", "recipes",
    }
    if set(payload) != required or payload.get("schema_version") != "2.0.0":
        raise ArtifactAllowlistError("invalid allowed-recipe registry")
    base = {key: value for key, value in payload.items() if key != "registry_id"}
    expected_id = (
        "scryglass:allowed-recipe-registry:"
        + sha256_canonical_object_hash(base)
    )
    if payload.get("registry_id") != expected_id:
        raise ArtifactAllowlistError("allowed-recipe registry id drift")
    if payload.get("contract_tree_sha256") != CONTRACT_TREE_SHA256:
        raise ArtifactAllowlistError("allowed-recipe contract lineage drift")
    if payload.get("source_snapshot_locator") != SOURCE_LOCATOR:
        raise ArtifactAllowlistError("allowed-recipe source locator drift")
    source = json.loads(_resolve(SOURCE_LOCATOR).read_bytes())
    if (
        payload.get("source_snapshot_id") != source.get("snapshot_id")
        or payload.get("source_snapshot_object_sha256")
        != sha256_canonical_object_hash(source)
    ):
        raise ArtifactAllowlistError("allowed-recipe source lineage drift")
    recipes = payload.get("recipes")
    if not isinstance(recipes, list) or not recipes:
        raise ArtifactAllowlistError("allowed-recipe registry is empty")
    keys = [
        (item.get("recipe_locator"), item.get("recipe_id"))
        for item in recipes
        if isinstance(item, Mapping)
    ]
    if len(keys) != len(recipes) or len(set(keys)) != len(recipes):
        raise ArtifactAllowlistError("allowed recipes must be unique")
    return payload


def _validate_transform_recipe(recipe: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "recipe_id", "transform_code_locator",
        "transform_code_sha256", "transform_config_locator",
        "transform_config_sha256", "input_roles", "output",
    }
    if set(recipe) != required or recipe.get("schema_version") != "2.0.0":
        raise ArtifactAllowlistError("invalid transform recipe shape")
    base = {key: value for key, value in recipe.items() if key != "recipe_id"}
    if recipe.get("recipe_id") != RECIPE_PREFIX + sha256_canonical_object_hash(base):
        raise ArtifactAllowlistError("transform recipe id drift")
    for kind in ("code", "config"):
        locator = recipe.get(f"transform_{kind}_locator")
        expected = recipe.get(f"transform_{kind}_sha256")
        if not isinstance(locator, str) or not isinstance(expected, str):
            raise ArtifactAllowlistError(f"transform {kind} evidence missing")
        if sha256_raw_bytes_hash(_resolve(locator).read_bytes()) != expected:
            raise ArtifactAllowlistError(f"transform {kind} hash drift")
    roles = recipe.get("input_roles")
    if not isinstance(roles, list) or not roles:
        raise ArtifactAllowlistError("transform recipe needs input roles")
    role_names = [role.get("role") for role in roles if isinstance(role, Mapping)]
    if len(role_names) != len(roles) or len(set(role_names)) != len(roles):
        raise ArtifactAllowlistError("transform recipe roles must be unique")
    for role in roles:
        if set(role) != {
            "role", "kind", "source_id", "artifact_class", "selector",
            "minimum", "maximum",
        }:
            raise ArtifactAllowlistError("invalid transform input role")
        if role["kind"] not in {"source_row", "artifact"}:
            raise ArtifactAllowlistError("invalid transform input kind")
        if role["minimum"] != 1 or role["maximum"] != 1:
            raise ArtifactAllowlistError("B2 recipes require exactly one input per role")
        if not isinstance(role["selector"], Mapping) or not role["selector"]:
            raise ArtifactAllowlistError("transform role needs an exact selector")
    output = recipe.get("output")
    if not isinstance(output, Mapping) or set(output) != {
        "source_id", "artifact_class", "fields", "schemas_by_mode",
    }:
        raise ArtifactAllowlistError("invalid transform recipe output")
    if not isinstance(output["fields"], list) or not output["fields"]:
        raise ArtifactAllowlistError("transform recipe output fields required")
    schemas = output["schemas_by_mode"]
    if not isinstance(schemas, Mapping) or not schemas:
        raise ArtifactAllowlistError("transform recipe output mode/schema required")
    evidence = load_publication_evidence()
    known = {
        (item["source_id"], item["artifact_class"], item["mode"]): item["schema_id"]
        for item in evidence["payload_schemas"]
    }
    for mode, schema_id in schemas.items():
        if mode not in {"public", "authenticated", "private"} or known.get(
            (output["source_id"], output["artifact_class"], mode)
        ) != schema_id:
            raise ArtifactAllowlistError("transform recipe output schema mismatch")


def _recipe_for_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    locator = manifest.get("recipe_locator")
    expected_id = manifest.get("recipe_id")
    expected_hash = manifest.get("recipe_bytes_sha256")
    if not all(isinstance(value, str) for value in (locator, expected_id, expected_hash)):
        raise ArtifactAllowlistError("transform manifest recipe evidence missing")
    recipe = load_transform_recipe(locator)
    registered = next(
        item
        for item in _load_allowed_recipe_registry()["recipes"]
        if item["recipe_locator"] == locator
    )
    if (
        recipe["recipe_id"] != expected_id
        or registered["recipe_id"] != expected_id
        or registered["recipe_bytes_sha256"] != expected_hash
    ):
        raise ArtifactAllowlistError("transform recipe identity drift")
    return recipe


def _node_matches_role(node: Mapping[str, Any], role: Mapping[str, Any]) -> bool:
    if (
        node.get("kind") != role["kind"]
        or node.get("source_id") != role["source_id"]
        or node.get("artifact_class") != role["artifact_class"]
    ):
        return False
    return all(node.get(key) == value for key, value in role["selector"].items())


def _validate_dag(manifest: Mapping[str, Any]) -> None:
    recipe = _recipe_for_manifest(manifest)
    nodes = manifest.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ArtifactAllowlistError("transform DAG needs nodes")
    by_id = {n.get("node_id"): n for n in nodes if isinstance(n, Mapping)}
    if len(by_id) != len(nodes) or None in by_id:
        raise ArtifactAllowlistError("duplicate/invalid DAG node")
    output_id = manifest.get("output_node_id")
    if output_id not in by_id:
        raise ArtifactAllowlistError("missing DAG output")
    output = by_id[output_id]
    recipe_output = recipe["output"]
    if (
        output.get("kind") != "artifact"
        or output.get("source_id") != recipe_output["source_id"]
        or output.get("artifact_class") != recipe_output["artifact_class"]
        or output.get("fields") != recipe_output["fields"]
    ):
        raise ArtifactAllowlistError(
            "transform output cannot self-assert external source or mismatch recipe"
        )
    leaves = [
        node for node in nodes
        if isinstance(node, Mapping) and not node.get("direct_input_ids")
    ]
    bindings: dict[str, Mapping[str, Any]] = {}
    for leaf in leaves:
        role_name = leaf.get("input_role")
        if not isinstance(role_name, str) or role_name in bindings:
            raise ArtifactAllowlistError(
                "disconnected additional or duplicate transform leaf binding"
            )
        bindings[role_name] = leaf
    required_roles = [role["role"] for role in recipe["input_roles"]]
    if set(bindings) != set(required_roles) or len(bindings) != len(required_roles):
        raise ArtifactAllowlistError("transform leaf set does not match required recipe roles")
    bound_identities: set[tuple[Any, ...]] = set()
    for role in recipe["input_roles"]:
        node = bindings[role["role"]]
        if not _node_matches_role(node, role):
            raise ArtifactAllowlistError(
                "self-asserted source identity or transform role selector substitution"
            )
        identity = (
            node.get("kind"), node.get("source_id"), node.get("artifact_class"),
            node.get("source_snapshot_row_id"), node.get("artifact_id"),
            node.get("bytes_sha256"),
        )
        if identity in bound_identities:
            raise ArtifactAllowlistError("duplicate transform input binding")
        bound_identities.add(identity)
    expected_direct_inputs = [bindings[role]["node_id"] for role in required_roles]
    if output_id in output.get("direct_input_ids", []):
        raise ArtifactAllowlistError("cyclic transform DAG")
    if output.get("direct_input_ids") != expected_direct_inputs:
        raise ArtifactAllowlistError(
            "disconnected transform output edges do not bind ordered recipe roles"
        )
    source_snapshot = json.loads((REPO_ROOT / SOURCE_LOCATOR).read_bytes())
    source_rows = {r["source_snapshot_row_id"]: r for r in source_snapshot["rows"]}
    source_bindings = {s["source_id"]: set(s["source_row_bindings"]) for s in load_publication_evidence()["sources"]}
    registry = {source.source_id: source for source in registered_sources()}
    visiting: set[str] = set(); visited: set[str] = set()
    def visit(node_id: str) -> None:
        if node_id in visiting: raise ArtifactAllowlistError("cyclic transform DAG")
        if node_id in visited: return
        visiting.add(node_id)
        node = by_id[node_id]
        source = registry.get(node.get("source_id"))
        if source is None or node.get("artifact_class") not in source.artifact_classes:
            raise ArtifactAllowlistError("DAG node source/artifact policy mismatch")
        inputs = node.get("direct_input_ids")
        if not isinstance(inputs, list) or len(inputs) != len(set(inputs)):
            raise ArtifactAllowlistError("invalid direct inputs")
        for input_id in inputs:
            if input_id not in by_id: raise ArtifactAllowlistError("unknown DAG input")
            visit(input_id)
        if node.get("kind") == "source_row":
            if inputs: raise ArtifactAllowlistError("source row cannot have inputs")
            row_id = node.get("source_snapshot_row_id")
            row = source_rows.get(row_id)
            if not row or row_id not in source_bindings.get(node.get("source_id"), set()):
                raise ArtifactAllowlistError("self-asserted or unbound source identity")
            if node.get("source_snapshot_id") != source_snapshot["snapshot_id"] or node.get("locator") != row["source_content_path"] or node.get("bytes_sha256") != row["source_content_sha256"]:
                raise ArtifactAllowlistError("source-row lineage mismatch")
            raw = _resolve(node["locator"]).read_bytes()
            if sha256_raw_bytes_hash(raw) != node["bytes_sha256"]:
                raise ArtifactAllowlistError("mutated source-row bytes")
        elif node.get("kind") == "artifact":
            if not inputs and "input_role" not in node:
                raise ArtifactAllowlistError("artifact node needs inputs or a recipe role")
            if source.access_method != "derived":
                raise ArtifactAllowlistError("transform output cannot self-assert external source")
            raw = _resolve(node.get("locator", "")).read_bytes()
            if sha256_raw_bytes_hash(raw) != node.get("bytes_sha256"):
                raise ArtifactAllowlistError("artifact output bytes mismatch")
            obj = json.loads(raw)
            if sha256_canonical_object_hash(obj) != node.get("object_sha256"):
                raise ArtifactAllowlistError("artifact output object mismatch")
        else:
            raise ArtifactAllowlistError("unknown DAG node kind")
        visiting.remove(node_id); visited.add(node_id)
    visit(output_id)
    if visited != set(by_id):
        raise ArtifactAllowlistError("disconnected or extra DAG input")


def make_candidate_artifact(*, lineage_manifest_locator: str, audience_mode: str) -> CandidateArtifact:
    manifest = load_transform_manifest(lineage_manifest_locator)
    recipe = _recipe_for_manifest(manifest)
    if audience_mode not in recipe["output"]["schemas_by_mode"]:
        raise ArtifactAllowlistError("candidate audience mode is not permitted by recipe")
    output = next(n for n in manifest["nodes"] if n["node_id"] == manifest["output_node_id"])
    raw_manifest = _resolve(lineage_manifest_locator).read_bytes()
    values = dict(source_id=output["source_id"], artifact_class=output["artifact_class"], locator=output["locator"], bytes_sha256=output["bytes_sha256"], object_sha256=output["object_sha256"], fields=tuple(sorted(output["fields"])), audience_mode=audience_mode, lineage_manifest_locator=lineage_manifest_locator, lineage_manifest_id=manifest["manifest_id"], lineage_manifest_sha256=sha256_canonical_object_hash(manifest))
    artifact_id = "scryglass:candidate-artifact:" + sha256_canonical_object_hash(_jsonable(values))
    return CandidateArtifact(artifact_id=artifact_id, **values)


def enforce_candidate_publication(candidate: CandidateArtifact, matrix: PublicationMatrix, *, publication_mode: str = "private") -> ArtifactAllowlistRow:
    if candidate.audience_mode != publication_mode:
        raise ArtifactAllowlistError("audience mode mismatch")
    manifest = load_transform_manifest(candidate.lineage_manifest_locator)
    if candidate.lineage_manifest_id != manifest["manifest_id"] or candidate.lineage_manifest_sha256 != sha256_canonical_object_hash(manifest):
        raise ArtifactAllowlistError("missing or mutated lineage manifest")
    output = next(n for n in manifest["nodes"] if n["node_id"] == manifest["output_node_id"])
    expected = make_candidate_artifact(lineage_manifest_locator=candidate.lineage_manifest_locator, audience_mode=publication_mode)
    if candidate != expected:
        raise ArtifactAllowlistError("candidate identity is not derived from verified lineage")
    policies = []
    for node in manifest["nodes"]:
        policy = matrix.find_row(node["source_id"], node["artifact_class"])
        if policy is None: raise ArtifactAllowlistError("DAG node has no matching source/artifact policy")
        policies.append(policy)
    effective = max((p.decision for p in policies), key=_rank)
    if publication_mode == "public":
        output_policy = matrix.find_row(output["source_id"], output["artifact_class"])
        if not output_policy.field_allowlist:
            raise ArtifactAllowlistError("empty public field allowlist denies entire artifact")
        if effective != PublicationMatrixDecision.PUBLIC:
            raise ArtifactAllowlistError("public mode blocked by exact DAG policy")
    elif publication_mode == "authenticated" and effective not in {PublicationMatrixDecision.PUBLIC, PublicationMatrixDecision.AUTHENTICATED}:
        raise ArtifactAllowlistError("authenticated mode blocked by exact DAG policy")
    elif effective == PublicationMatrixDecision.PROHIBITED:
        raise ArtifactAllowlistError("prohibited DAG input")
    raw = _resolve(candidate.locator).read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not payload:
        raise ArtifactAllowlistError("typed payload root must be a nonempty object")
    _scan_private(payload)
    schema = matrix.payload_schema(candidate.source_id, candidate.artifact_class, publication_mode)
    _validate_typed(payload, schema)
    if set(candidate.fields) != set(payload):
        raise ArtifactAllowlistError("candidate fields do not match typed payload")
    return ArtifactAllowlistRow(**candidate.__dict__, effective_decision=effective, direct_input_ids=tuple(output["direct_input_ids"]))


def _validate_typed(value: Any, schema: Mapping[str, Any]) -> None:
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict): raise ArtifactAllowlistError("expected object")
        properties = schema.get("properties", {})
        if set(value) != set(schema.get("required", [])) or (schema.get("additionalProperties") is False and set(value) - set(properties)):
            raise ArtifactAllowlistError("missing or nested extra typed field")
        for key, child in value.items(): _validate_typed(child, properties[key])
    elif kind == "array":
        if not isinstance(value, list): raise ArtifactAllowlistError("expected array")
        for child in value: _validate_typed(child, schema["items"])
    elif kind == "string":
        if not isinstance(value, str): raise ArtifactAllowlistError("expected string")
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int): raise ArtifactAllowlistError("expected integer")
    else:
        raise ArtifactAllowlistError("opaque payload type has no registered schema")


def _scan_private(value: Any) -> None:
    try: _reject_forbidden_recursive(value, "artifact payload")
    except ValueError as err: raise ArtifactAllowlistError(str(err)) from err
    if isinstance(value, dict):
        for key, child in value.items():
            compact = re.sub("[^a-z0-9]", "", key.casefold())
            if any(token in compact for token in SECRET): raise ArtifactAllowlistError("private/contact identifier field")
            _scan_private(child)
    elif isinstance(value, list):
        for child in value: _scan_private(child)
    elif isinstance(value, str):
        lower = value.casefold()
        if re.search(r"sk_live_[a-z0-9]+|[^\s@]+@[^\s@]+\.[^\s@]+|bearer\s+|(?:token|secret|password)\s*[=:]", lower):
            raise ArtifactAllowlistError("secret-shaped or contact value")


def generate_artifact_allowlist(matrix: PublicationMatrix, candidate_artifacts: list[CandidateArtifact], *, publication_mode: str = "private", allowlist_id: str | None = None, as_of: str | None = None, matrix_locator: str = MATRIX_LOCATOR) -> ArtifactAllowlist:
    if as_of and parse_rfc3339(as_of) != parse_rfc3339(matrix.as_of): raise ArtifactAllowlistError("as_of mismatch")
    rows = tuple(sorted((enforce_candidate_publication(c, matrix, publication_mode=publication_mode) for c in candidate_artifacts), key=lambda r: r.artifact_id))
    allowlist = ArtifactAllowlist("2.0.0", matrix.model_version, publication_mode, matrix.as_of, matrix_locator, matrix.matrix_id, matrix.object_sha256(), CONTRACT_TREE_SHA256, rows, "")
    expected = _allowlist_id(allowlist)
    if allowlist_id and allowlist_id != expected: raise ArtifactAllowlistError("allowlist id drift")
    object.__setattr__(allowlist, "allowlist_id", expected)
    return allowlist


def _validate_allowlist(value: ArtifactAllowlist) -> None:
    matrix = load_publication_matrix(_resolve(value.matrix_locator))
    if value.matrix_id != matrix.matrix_id or value.matrix_object_sha256 != matrix.object_sha256() or value.contract_tree_sha256 != CONTRACT_TREE_SHA256:
        raise ArtifactAllowlistError("allowlist lineage mismatch")
    if tuple(sorted(value.rows, key=lambda r: r.artifact_id)) != value.rows or len({r.artifact_id for r in value.rows}) != len(value.rows):
        raise ArtifactAllowlistError("allowlist rows must be sorted unique")
    for row in value.rows:
        candidate = CandidateArtifact(**{k: getattr(row, k) for k in CandidateArtifact.__dataclass_fields__})
        if enforce_candidate_publication(candidate, matrix, publication_mode=value.audience_mode) != row: raise ArtifactAllowlistError("allowlist row drift")
    if value.allowlist_id and value.allowlist_id != _allowlist_id(value): raise ArtifactAllowlistError("allowlist id drift")
    if not value.allowlist_id: object.__setattr__(value, "allowlist_id", _allowlist_id(value))


def artifact_allowlist_from_payload(payload: Mapping[str, Any]) -> ArtifactAllowlist:
    rows = []
    for raw in payload["rows"]:
        item = dict(raw)
        for key in ("fields", "direct_input_ids"): item[key] = tuple(item[key])
        rows.append(ArtifactAllowlistRow(**item))
    return ArtifactAllowlist(payload["schema_version"], payload["model_version"], payload["audience_mode"], payload["as_of"], payload["matrix_locator"], payload["matrix_id"], payload["matrix_object_sha256"], payload["contract_tree_sha256"], tuple(rows), payload["allowlist_id"])


def load_artifact_allowlist(path: Path) -> ArtifactAllowlist:
    payload = json.loads(path.read_bytes()); value = artifact_allowlist_from_payload(payload)
    if value.to_payload() != payload: raise ArtifactAllowlistError("allowlist canonical drift")
    return value


def write_artifact_allowlist(value: ArtifactAllowlist, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(canonical_json_bytes(value.to_payload()) + b"\n"); return value.object_sha256()


def _resolve(locator: str) -> Path:
    try:
        normalized = normalize_source_tree_path(locator)
        if normalized != locator: raise ValueError("non-normalized")
        return resolve_repository_file(REPO_ROOT, locator)
    except ValueError as err: raise ArtifactAllowlistError(f"invalid repository locator: {err}") from err


def _rank(decision: str) -> int:
    return {PublicationMatrixDecision.PUBLIC: 0, PublicationMatrixDecision.AUTHENTICATED: 1, PublicationMatrixDecision.PRIVATE_PENDING_REVIEW: 2, PublicationMatrixDecision.PRIVATE: 3, PublicationMatrixDecision.PROHIBITED: 4}[decision]


def _allowlist_id(value: ArtifactAllowlist) -> str:
    return "scryglass:artifact-allowlist:" + sha256_canonical_object_hash(value._payload_without_id())


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple): return [_jsonable(v) for v in value]
    if isinstance(value, Mapping): return {k: _jsonable(v) for k, v in value.items()}
    return value


__all__ = ["ArtifactAllowlist", "ArtifactAllowlistError", "ArtifactAllowlistRow", "CandidateArtifact", "artifact_allowlist_from_payload", "enforce_candidate_publication", "generate_artifact_allowlist", "load_artifact_allowlist", "load_transform_manifest", "load_transform_recipe", "make_candidate_artifact", "write_artifact_allowlist"]
