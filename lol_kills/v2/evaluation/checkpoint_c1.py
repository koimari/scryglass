"""Content-addressed C1 Wave-1 foundation freeze.

This module authenticates an exact, current-disk closure.  It does not make a
model decision and deliberately has no dependency on promotion or sealed
decision APIs.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
import weakref
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import CodeType, MappingProxyType
from typing import Any, Mapping, Sequence

from lol_kills.v2.evaluation.checks import ValidationFailure


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_TREE_SHA256 = "fb3de56ddec943bc876cb795a8ada5695233f5fe615defe93f952ce299470517"
DECISION_KIND = "foundation_freeze"
LOCATION_DISCLAIMER = (
    "Stored under data/lol/v2/evaluation/b2 only as a reconciliation input; "
    "this location is not PASS-B2, promotion, serving, publication, production, "
    "Reliability, probability-wording, SOTA, real-data, or sealed-decision authority."
)

CLAIM_BOUNDARY: Mapping[str, Any] = MappingProxyType(
    {
        "promotion_decision": None,
        "pass_b2": False,
        "production_authority": False,
        "real_data_evidence": False,
        "reliability_authorized": False,
        "probability_wording_authorized": False,
        "sota_authorized": False,
        "publication_authorized": False,
        "sealed_decision_opened": False,
    }
)

REQUIRED_GATES = (
    "C1_GATE_CONTRACT_TREE_EXACT",
    "C1_GATE_L1_SNAPSHOT_BYTES_AND_IDS_REPLAYED",
    "C1_GATE_L1_LINEAGE_CLOSURE_EXACT",
    "C1_GATE_L2_REGISTRY_AUTHORITY_EXACT",
    "C1_GATE_L2_SPLIT_PROTOCOL_FROZEN",
    "C1_GATE_R20_AUTHORITIES_REPLAYED",
    "C1_GATE_OUTER_CALIBRATION_AUTHORITY_REPLAYED",
    "C1_GATE_B3_MECHANICS_CEILING_EXACT",
    "C1_GATE_B3_RELIABILITY_FAIL_CLOSED",
    "C1_GATE_L3_ONTOLOGY_SCHEMA_REPLAYED",
    "C1_GATE_LEGACY_REPORT_NONAUTHORITATIVE",
    "C1_GATE_COMPONENT_CLAIM_CEILINGS_PRESERVED",
    "C1_GATE_NO_SEALED_OPENING",
    "C1_GATE_NO_PRODUCTION_AUTHORITY",
    "C1_GATE_CANONICAL_SOURCE_CLOSURE",
    "C1_GATE_EXACT_FRESH_REPLAY",
)

THREAT_MODEL = MappingProxyType(
    {
        "scope": "honest_interpreter_process_local_misuse_and_ordinary_forgery_guards_only",
        "hostile_same_process_security": False,
        "production_authority_requires": (
            "an independently pinned signature, native boundary, separate process trust root, "
            "or operating-system trust root"
        ),
        "singleton_and_content_hashes_authorize_promotion": False,
    }
)

CONFIG_LOCATOR = "data/lol/v2/evaluation/b2/checkpoint-c1-config.json"
REPORT_LOCATOR = "data/lol/v2/evaluation/b2/checkpoint-c1-report.json"
AUTHORITY_LOCATOR = "data/lol/v2/evaluation/b2/checkpoint-c1-authority.json"
CHECKPOINT_SOURCE_LOCATOR = "lol_kills/v2/evaluation/checkpoint_c1.py"
GENERATOR_SOURCE_LOCATOR = "lol_kills/v2/evaluation/generate_checkpoint_c1_artifacts.py"


@dataclass(frozen=True)
class _RoleSpec:
    role: str
    layer: str
    classification: str
    locator: str
    raw_sha256: str
    object_sha256: str | None
    encoding: str


_SPECS = (
    _RoleSpec("l1_source_snapshot", "L1", "IMMUTABLE_DATA_FOUNDATION", "data/lol/v2/snapshots/b1/source-snapshot-passb1.json", "8c08746e56c4c100c12e8f621c4e48452ec62374ac564e2a7a64dbcaa00a4680", "bd7d3d5a43c533b0293fb7c1bbf918ed690cb4d1d36f51fc9dd472ad4e7b75ba", "json"),
    _RoleSpec("l1_training_snapshot", "L1", "IMMUTABLE_DATA_FOUNDATION", "data/lol/v2/snapshots/b1/training-snapshot-passb1.json", "037a46e5f1c2a5ca18d08dff6490242fe624ced42b46fdc8c8ae0ed38488a7f4", "8f056c8100ccb9771779338879b05242938b91c5ad0017057a8eed8c570d25c2", "json"),
    _RoleSpec("l1_split_assignment", "L1", "IMMUTABLE_DATA_FOUNDATION", "data/lol/v2/snapshots/b1/split-assignment-passb1.json", "8e59d3071f5bdac5d789d55713ddb4b7d4c0fb7a28afab28a9a92078f65db7d2", "2af533687ed543d1a61d739fa9d056aed6c6ca9e43e48aacfb9d163e11959bee", "json"),
    _RoleSpec("l1_row_count", "L1", "IMMUTABLE_DATA_FOUNDATION", "data/lol/v2/snapshots/b1/row-count-evidence-passb1.json", "a5fed75c0aa41105d218011e98714b7d6585d66d460621bfa90c7dd4f9160848", "d02f3e78dbcb4836839ab016c08136a93b5c381104850404bcb19f347a6f4d62", "json"),
    _RoleSpec("l1_environment_lock", "L1", "IMMUTABLE_DATA_FOUNDATION", "data/lol/v2/snapshots/b1/environment-lock-passb1.txt", "f2b0d9752e4fdbdde80a5b88243dd5eafd3b41e610ecb8efee99dbcd944b5c96", None, "text"),
    _RoleSpec("l2_registrar_root", "L2", "PROTOCOL_AUTHORITY", "data/lol/v2/evaluation/registry-registrar-trust-root.json", "1523912434e4f533352140b723d8429b7e4a38bd776910fb85652a5b6b985771", "390f497ea97dda024c3f1e99a8e7cfd864f8fe2f7f6e6baad187a7f74dc041d7", "json"),
    _RoleSpec("l2_synthetic_registry", "L2", "SYNTHETIC_PROTOCOL_MECHANICS_ONLY", "data/lol/v2/evaluation/synthetic-registry-frozen.json", "9cc3f57a94ae907692c3767dea1525efa5970a309894353d2a1eb358afffed35", "71b4c925536953dc5ceaca385c2068fb4f5f171f43718125036ba43166528180", "json"),
    _RoleSpec("l2_contract_fixture_authority", "L2", "CONTRACT_FIXTURE_AUTHORITY", "data/lol/v2/evaluation/contract-fixture-authority.json", "d7b95f4ddc365f7289f4d81726343d3a0a4c97e66a5f5ce5c844bfae55cffecb", "9b938413fd8de6017fdf8a8aed544707436b29844300c7079cb74c30559a89d8", "json"),
    _RoleSpec("l2_r20_foundation_authority", "L2", "SYNTHETIC_PROTOCOL_MECHANICS_ONLY", "data/lol/v2/evaluation/b2/r20-foundation-authority.json", "c4f8c86ae16a69037a7030068bc86cd56096ede6700247b09be649d6dbae6e88", "0b881ccccde643704ae642d6f72ea9af5613e02078001a2b59a8bd42f45b8982", "json"),
    _RoleSpec("l2_r20_selection_authority", "L2", "SYNTHETIC_PROTOCOL_MECHANICS_ONLY", "data/lol/v2/evaluation/b2/r20-selection-authority.json", "0bea9081ecc9bbd047b1110862bc49ec41b43fad406d6a1de87e11a8db2d9c49", "11086d8b1c0415bb8216c9c8be7db25914cee063a8c059ce8e23b341a972e9c4", "json"),
    _RoleSpec("l2_outer_calibration_authority", "L2", "INDEPENDENT_SYNTHETIC_CALIBRATION_MECHANICS", "data/lol/v2/evaluation/b2/outer-calibration-authority.json", "de92f17fb17587190fc7f1e8e876e51bc2c1e1456d2e5e964c50fa0725859303", "7e9474611814957362bf0b24bfa54abed962f8de2ad91641cba256ae06ba40ce", "json"),
    _RoleSpec("l2_outer_calibration_config", "L2", "INDEPENDENT_SYNTHETIC_CALIBRATION_MECHANICS", "data/lol/v2/evaluation/b2/outer-calibration-config.json", "67d80a6f243ec15c3984118b92b59caba9a68f5bac583bc1e5f56918895f0d99", "64bf61acbf45939d252e69df3c8a2ad7e8ab39f4a4877187a1f778da08491ee3", "json"),
    _RoleSpec("l2_outer_calibration_source", "L2", "IMPLEMENTATION_IDENTITY", "lol_kills/v2/evaluation/outer_calibration.py", "1e7fd2cacb812e58b23bf7f1a1d80e969370b9bfb8a622da860ec59f579f31fb", None, "python"),
    _RoleSpec("l2_b3_reliability_authority", "L2", "FAIL_CLOSED_RELIABILITY_BOUNDARY", "data/lol/v2/evaluation/b2/b3-reliability-authority.json", "7eb023a76edebe8a9065d49fafb04bdb3adc4de4843938f4637b3b743c7fcf3b", "0408086a06ebb7b4735c5c92e019cf0ece48f954dd7841b0c25ed180afc84ed5", "json"),
    _RoleSpec("l2_b3_coverage_authority", "L2", "SYNTHETIC_MECHANICS_CEILING", "data/lol/v2/evaluation/b3/coverage-authority.json", "bc5552de3025aae5dafb10f02a9fcc5366af35f1dc9f0490150f940d25dd1759", "7591cbe4f00a1f7962b6ece93edb3405dd770a18483e58aca09791fb6e2cf385", "json"),
    _RoleSpec("l2_legacy_synthetic_report", "L2", "HISTORICAL_NONAUTHORITATIVE", "data/lol/v2/evaluation/b2/synthetic-validation-report.json", "5ee3df6f97f460446bda8d00b630e1f1a6510f69d7da207fd45f83d8cc13830b", "0dec03e9b348f18c288509f40a09f6a124d7ba871fa29f730e59dae6dfe97705", "json"),
    _RoleSpec("l3_ontology_seed", "L3", "REVIEWED_ONTOLOGY_FOUNDATION", "data/lol/v2/champions/champion-ontology-seed.json", "d26e40f83cf3af1129fd2f0e487229322c1eea45fd2c41240bf72779c7d440b0", "8d01368a0c36a456cdb49a6b612c76c88323b746b897f247a4357ce657b3da1e", "json"),
    _RoleSpec("l3_schema_implementation", "L3", "REVIEWED_ONTOLOGY_SCHEMA_IMPLEMENTATION", "lol_kills/v2/champions/schema.py", "8e7de9d10b6e9b3ca7945ecc4031b12ffc0538b0eb290d92625822b9028c7e72", None, "python"),
    _RoleSpec("l3_ontology_sources", "L3", "REVIEWED_ONTOLOGY_FOUNDATION", "data/lol/v2/champions/champion-ontology-sources.json", "26e6f1682ce4178f29828aac1a724b355c3d22f4557334bc9a7cafd8bd8c5a49", "96fa8feb30ffc5de26fe22381ae890262247da71b1df3dd328f7d8eedf82d77e", "json"),
    _RoleSpec("l3_review_log", "L3", "REVIEWED_ONTOLOGY_FOUNDATION", "data/lol/v2/champions/champion-review-log.jsonl", "f256e5851c69d706e0c8465ffa4ee0ed8d846c0566bd50972426d1d4ab3b859b", None, "jsonl"),
    _RoleSpec("l3_evaluation_fixtures", "L3", "REVIEWED_ONTOLOGY_FOUNDATION", "data/lol/v2/champions/evaluation-fixtures.json", "32f0c6c057f9d87998a6bcbba7b4b24ed353f031a317b3e4b3b19e0455977a55", "7ad439e4a07f0bf1be3558bd476c927433920f0c0569e73f3874772fca598939", "json"),
    _RoleSpec("c4_authority_registry", "NEGATIVE_BOUNDARY", "ZERO_PUBLICATION_AUTHORITY", "data/lol/v2/publication/c4-authority-registry-b2.json", "6afcf98e948905578c2e871fd304fa2dafd0d644a7fa541267778d21996d2ffa", "7976a0612bbe9acfe27bfdaac7dc75b5144300dd9ba8f0c42c28d4398ba0f3bf", "json"),
    _RoleSpec("publication_public_allowlist", "NEGATIVE_BOUNDARY", "EMPTY_PUBLIC_ALLOWLIST", "data/lol/v2/publication/artifact-allowlist-public-b2.json", "4ee2b420a34d92bd971f1c7e90827e01cfcde75ccecff4a6ad7afe9dced93aad", "354b4b9da4a50d37197db8ac817623763e74ae24202505a1c9a2276b9fffbf13", "json"),
    _RoleSpec("publication_authenticated_allowlist", "NEGATIVE_BOUNDARY", "EMPTY_AUTHENTICATED_ALLOWLIST", "data/lol/v2/publication/artifact-allowlist-authenticated-b2.json", "a3205bda59b964cd45ba926af977ad465c6f9dbe54e05dbcec4a360e959262c3", "be47d192500044f7483302adceeee45b054b32667474242040ff3c5fad070b3d", "json"),
    _RoleSpec("publication_private_allowlist", "NEGATIVE_BOUNDARY", "PRIVATE_PENDING_REVIEW_ONLY", "data/lol/v2/publication/artifact-allowlist-private-b2.json", "d0527f075bf10dbb491aaec1e8ac61838a7e008039ac174918516dc7a7931326", "8ae1e1bd94502eea9c3e95037bcadaf7f4778a58683e0a477c2ca0c785d215c1", "json"),
    _RoleSpec("sealed_ledger_boundary", "NEGATIVE_BOUNDARY", "UNOPENED_SEALED_FIXTURE", "data/lol/v2/evaluation/sealed-ledger-fixture.jsonl", "b37b0ca2131df555738fa7c11da4be1152fab3e8e88de7b62c0c571ab197ad6a", None, "jsonl"),
)

INPUT_ROLE_LOCATORS = tuple(spec.locator for spec in _SPECS)
_SPEC_BY_ROLE = {spec.role: spec for spec in _SPECS}
_ARTIFACT_IDENTITIES = {
    CONFIG_LOCATOR: ("scryglass:c1:foundation-freeze-config:v1", "checkpoint-c1-foundation-freeze-config-v1"),
    REPORT_LOCATOR: ("scryglass:c1:foundation-freeze-report:v1", "checkpoint-c1-foundation-freeze-report-v1"),
    AUTHORITY_LOCATOR: ("scryglass:c1:foundation-freeze-authority:v1", "checkpoint-c1-foundation-freeze-authority-v1"),
}


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValidationFailure(f"C1 value is not canonical-JSON encodable: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _duplicate_rejecting_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationFailure(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValidationFailure(f"nonfinite JSON constant: {value}")


def _strict_json(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationFailure("C1 JSON is not UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_pairs,
            parse_constant=_reject_constant,
        )
    except ValidationFailure:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValidationFailure(f"invalid C1 JSON: {exc}") from exc


def _validate_locator(locator: str) -> PurePosixPath:
    if not isinstance(locator, str) or not locator:
        raise ValidationFailure("C1 locator must be a non-empty string")
    pure = PurePosixPath(locator)
    if pure.is_absolute() or str(pure) != locator or any(part in ("", ".", "..") for part in pure.parts):
        raise ValidationFailure(f"unsafe or aliased C1 locator: {locator!r}")
    return pure


def _lstat_component(path: Path, *, leaf: bool) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ValidationFailure(f"C1 path component unavailable: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ValidationFailure(f"C1 symlink rejected: {path}")
    if leaf:
        if not stat.S_ISREG(info.st_mode):
            raise ValidationFailure(f"C1 nonregular leaf rejected: {path}")
        if info.st_nlink != 1:
            raise ValidationFailure(f"C1 hardlinked leaf rejected: {path}")
    elif not stat.S_ISDIR(info.st_mode):
        raise ValidationFailure(f"C1 non-directory parent rejected: {path}")
    return info


def _safe_read(
    root: Path,
    locator: str,
    *,
    seen_inodes: dict[tuple[int, int], str],
) -> bytes:
    pure = _validate_locator(locator)
    root = Path(root).absolute()
    _lstat_component(root, leaf=False)
    current = root
    for part in pure.parts[:-1]:
        current = current / part
        _lstat_component(current, leaf=False)
    leaf = current / pure.parts[-1]
    info = _lstat_component(leaf, leaf=True)
    try:
        leaf.relative_to(root)
    except ValueError as exc:
        raise ValidationFailure(f"C1 path escape rejected: {locator}") from exc
    inode = (info.st_dev, info.st_ino)
    prior = seen_inodes.get(inode)
    if prior is not None and prior != locator:
        raise ValidationFailure(f"C1 inode role substitution: {prior} and {locator}")
    seen_inodes[inode] = locator
    try:
        return leaf.read_bytes()
    except OSError as exc:
        raise ValidationFailure(f"C1 input unreadable: {locator}") from exc


def _parse_role(spec: _RoleSpec, raw: bytes) -> Any:
    if spec.encoding == "json":
        return _strict_json(raw)
    if spec.encoding == "jsonl":
        lines = raw.splitlines()
        if not lines:
            raise ValidationFailure(f"C1 empty JSONL role: {spec.role}")
        return tuple(_strict_json(line) for line in lines)
    if spec.encoding in {"text", "python"}:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationFailure(f"C1 non-UTF-8 text role: {spec.role}") from exc
    raise ValidationFailure(f"unknown C1 role encoding: {spec.encoding}")


def _identity(spec: _RoleSpec, value: Any) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    if isinstance(value, dict):
        for key in (
            "artifact_id",
            "artifact_kind",
            "artifact_version",
            "schema_version",
            "snapshot_id",
            "registry_id",
            "allowlist_id",
            "contract_tree_sha256",
        ):
            if key in value:
                identity[key] = value[key]
    elif spec.encoding == "jsonl":
        identity["record_count"] = len(value)
        if value and isinstance(value[0], dict):
            for key in ("fixture_version", "kind"):
                if key in value[0]:
                    identity[key] = value[0][key]
    if spec.role == "l1_training_snapshot":
        identity["source_snapshot_pairs"] = value["required_source_snapshot_pairs"]
        identity["source_tree_sha256"] = value["source_tree_sha256"]
        identity["split_assignment_ids"] = value["split_assignment_ids"]
    if spec.role == "l2_synthetic_registry":
        identity.update(
            {
                "source_snapshot_id": value["source_snapshot_id"],
                "training_snapshot_id": value["training_snapshot_id"],
                "source_tree_sha256": value["source_tree_sha256"],
                "split_plan_id": value["split_plan_id"],
                "split_plan_sha256": value["split_plan_sha256"],
            }
        )
    return identity


def _read_verified_inputs(
    root: Path,
    *,
    seen_inodes: dict[tuple[int, int], str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, bytes]]:
    seen = seen_inodes if seen_inodes is not None else {}
    records: list[dict[str, Any]] = []
    values: dict[str, Any] = {}
    raw_values: dict[str, bytes] = {}
    for spec in _SPECS:
        raw = _safe_read(root, spec.locator, seen_inodes=seen)
        raw_hash = hashlib.sha256(raw).hexdigest()
        if raw_hash != spec.raw_sha256:
            raise ValidationFailure(f"C1 raw hash mismatch for role {spec.role}")
        value = _parse_role(spec, raw)
        object_hash = canonical_sha256(value) if spec.encoding == "json" else None
        if object_hash != spec.object_sha256:
            raise ValidationFailure(f"C1 canonical object hash mismatch for role {spec.role}")
        records.append(
            {
                "role": spec.role,
                "layer": spec.layer,
                "classification": spec.classification,
                "locator": spec.locator,
                "raw_sha256": raw_hash,
                "object_sha256": object_hash,
                "artifact_identity": _identity(spec, value),
            }
        )
        values[spec.role] = value
        raw_values[spec.role] = raw
    if tuple(record["role"] for record in records) != tuple(spec.role for spec in _SPECS):
        raise ValidationFailure("C1 role order is not exact")
    return records, values, raw_values


def _require_exact_keys(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValidationFailure(f"{label} fields are not exact")


def _validate_claim_boundary(value: Any) -> None:
    expected = dict(CLAIM_BOUNDARY)
    _require_exact_keys(value, set(expected), "C1 claim boundary")
    for key, expected_value in expected.items():
        actual = value[key]
        if expected_value is None:
            if actual is not None:
                raise ValidationFailure(f"C1 claim boundary elevated: {key}")
        elif type(actual) is not bool or actual is not expected_value:
            raise ValidationFailure(f"C1 claim boundary elevated: {key}")


def _gate_entry(passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    if passed is not True:
        raise ValidationFailure("C1 hard gate predicate failed")
    return {
        "passed": True,
        "evidence": evidence,
        "evidence_sha256": canonical_sha256(evidence),
    }


def _audit_owned_sources(source_raw: Mapping[str, bytes]) -> dict[str, Any]:
    expected_roles = {"checkpoint_implementation", "artifact_generator"}
    if set(source_raw) != expected_roles:
        raise ValidationFailure("C1 owned-source audit role set is not exact")
    forbidden_modules = {"promotion", "sealed"}
    forbidden_bare_calls = {"request", "claim", "open", "execute", "finalize", "receipt"}
    allowed_os_calls = {"os.open", "os.close", "os.write", "os.fsync", "os.replace", "os.unlink"}
    audited: list[dict[str, Any]] = []
    for role in ("checkpoint_implementation", "artifact_generator"):
        raw = source_raw[role]
        try:
            tree = ast.parse(raw.decode("utf-8"))
        except (UnicodeDecodeError, SyntaxError) as exc:
            raise ValidationFailure(f"C1 owned source is not valid Python: {role}") from exc
        imports: list[str] = []
        forbidden_calls_found: list[str] = []
        mutation_calls: list[str] = []
        mutation_sites: list[tuple[str, str]] = []
        function_by_call: dict[int, str] = {}
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for descendant in ast.walk(statement):
                    if isinstance(descendant, ast.Call):
                        function_by_call[id(descendant)] = statement.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    call_path = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    parts = [node.func.attr]
                    cursor = node.func.value
                    while isinstance(cursor, ast.Attribute):
                        parts.append(cursor.attr)
                        cursor = cursor.value
                    if isinstance(cursor, ast.Name):
                        parts.append(cursor.id)
                    call_path = ".".join(reversed(parts))
                else:
                    call_path = ""
                final_name = call_path.rsplit(".", 1)[-1]
                if (
                    final_name in forbidden_bare_calls
                    and call_path not in allowed_os_calls
                ):
                    forbidden_calls_found.append(call_path)
                if final_name in {
                    "write_bytes",
                    "write_text",
                    "replace",
                    "rename",
                    "unlink",
                    "open",
                    "write",
                }:
                    mutation_calls.append(call_path)
                    mutation_sites.append((function_by_call.get(id(node), "<module>"), call_path))
                    if call_path == "os.open":
                        if not node.args or not isinstance(node.args[0], ast.Name) or node.args[0].id != "path":
                            raise ValidationFailure("C1 generator os.open target is not the staged path")
                    if call_path == "os.replace":
                        if (
                            len(node.args) < 2
                            or not isinstance(node.args[1], ast.BinOp)
                            or not isinstance(node.args[1].op, ast.Div)
                            or not isinstance(node.args[1].left, ast.Name)
                            or node.args[1].left.id != "root"
                            or not isinstance(node.args[1].right, ast.Name)
                            or node.args[1].right.id != "locator"
                        ):
                            raise ValidationFailure("C1 generator os.replace destination is unauthorized")
                    if call_path == "os.unlink":
                        function_name = function_by_call.get(id(node), "<module>")
                        exact_cleanup = (
                            function_name == "_cleanup_exact"
                            and node.args
                            and isinstance(node.args[0], ast.Name)
                            and node.args[0].id == "path"
                        )
                        exact_rollback = (
                            function_name == "_transactional_replace"
                            and node.args
                            and isinstance(node.args[0], ast.BinOp)
                            and isinstance(node.args[0].op, ast.Div)
                            and isinstance(node.args[0].left, ast.Name)
                            and node.args[0].left.id == "root"
                            and isinstance(node.args[0].right, ast.Name)
                            and node.args[0].right.id == "locator"
                        )
                        if not (exact_cleanup or exact_rollback):
                            raise ValidationFailure("C1 generator os.unlink target is unauthorized")
        forbidden_imports = sorted(
            module
            for module in imports
            if any(part in forbidden_modules for part in module.split("."))
        )
        if forbidden_imports or forbidden_calls_found:
            raise ValidationFailure(f"C1 forbidden promotion/sealed source path in {role}")
        if role == "checkpoint_implementation" and mutation_calls:
            raise ValidationFailure("C1 checkpoint implementation has an unauthorized write surface")
        if role == "artifact_generator":
            unauthorized = sorted(
                call
                for call in mutation_calls
                if call not in allowed_os_calls
            )
            if unauthorized:
                raise ValidationFailure("C1 generator has an unauthorized write surface")
            expected_sites = sorted(
                [
                    ("_exclusive_regular_write", "os.open"),
                    ("_exclusive_regular_write", "os.write"),
                    ("_cleanup_exact", "os.unlink"),
                    ("_transactional_replace", "os.replace"),
                    ("_transactional_replace", "os.replace"),
                    ("_transactional_replace", "os.unlink"),
                ]
            )
            if sorted(mutation_sites) != expected_sites:
                raise ValidationFailure("C1 generator mutation-site policy is not exact")
            assignment_ok = False
            for node in tree.body:
                if (
                    isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "OUTPUT_LOCATORS" for target in node.targets)
                    and isinstance(node.value, ast.Tuple)
                    and [
                        element.id for element in node.value.elts if isinstance(element, ast.Name)
                    ] == ["CONFIG_LOCATOR", "REPORT_LOCATOR", "AUTHORITY_LOCATOR"]
                ):
                    assignment_ok = True
            if not assignment_ok:
                raise ValidationFailure("C1 generator output locator declaration is not exact")
        audited.append(
            {
                "role": role,
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "forbidden_imports": [],
                "forbidden_calls": [],
                "authorized_mutation_calls": sorted(set(mutation_calls)),
            }
        )
    return {
        "audited_sources": audited,
        "promotion_or_sealed_imports": [],
        "forbidden_calls": [],
        "authorized_output_locators": [CONFIG_LOCATOR, REPORT_LOCATOR, AUTHORITY_LOCATOR],
        "unauthorized_write_surfaces": [],
    }


def _evaluate_gates(
    records: list[dict[str, Any]],
    values: dict[str, Any],
    source_refs: dict[str, dict[str, str]],
    *,
    source_policy: dict[str, Any],
    replay_proof: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    by_role = {record["role"]: record for record in records}
    source = values["l1_source_snapshot"]
    training = values["l1_training_snapshot"]
    registry = values["l2_synthetic_registry"]
    fixture = values["l2_contract_fixture_authority"]
    foundation = values["l2_r20_foundation_authority"]
    selection = values["l2_r20_selection_authority"]
    outer = values["l2_outer_calibration_authority"]
    b3_reliability = values["l2_b3_reliability_authority"]
    b3_coverage = values["l2_b3_coverage_authority"]
    legacy = values["l2_legacy_synthetic_report"]
    ontology = values["l3_ontology_seed"]
    c4 = values["c4_authority_registry"]
    public = values["publication_public_allowlist"]
    authenticated = values["publication_authenticated_allowlist"]
    private = values["publication_private_allowlist"]
    sealed = values["sealed_ledger_boundary"]

    contract_roles = sorted(
        role
        for role, value in values.items()
        if isinstance(value, dict) and "contract_tree_sha256" in value
    )
    gates: dict[str, dict[str, Any]] = {}
    gates[REQUIRED_GATES[0]] = _gate_entry(
        all(values[role]["contract_tree_sha256"] == CONTRACT_TREE_SHA256 for role in contract_roles),
        {"contract_tree_sha256": CONTRACT_TREE_SHA256, "checked_roles": contract_roles},
    )
    gates[REQUIRED_GATES[1]] = _gate_entry(
        source["snapshot_id"] == "scryglass:source-snapshot:de7b49dd447fa7ccbe1d4ffb54b5005d213c286de28a42dda602ffa899171ce6"
        and training["snapshot_id"] == "scryglass:training-snapshot:670500f115a042d1ab5fcd9bf3ce576318e2bb762be9e84f42ffb742ed7c596e",
        {
            "source_snapshot": by_role["l1_source_snapshot"],
            "training_snapshot": by_role["l1_training_snapshot"],
            "split_assignment_raw_sha256": by_role["l1_split_assignment"]["raw_sha256"],
            "row_count_raw_sha256": by_role["l1_row_count"]["raw_sha256"],
            "environment_lock_raw_sha256": by_role["l1_environment_lock"]["raw_sha256"],
        },
    )
    gates[REQUIRED_GATES[2]] = _gate_entry(
        training["required_source_snapshot_pairs"]
        == [[source["snapshot_id"], by_role["l1_source_snapshot"]["object_sha256"]]]
        and training["source_manifest_locator"] == by_role["l1_source_snapshot"]["locator"]
        and training["source_manifest_object_sha256"] == by_role["l1_source_snapshot"]["object_sha256"]
        and training["source_tree_sha256"] == source["source_tree_sha256"]
        and training["split_assignment_sha256s"] == [by_role["l1_split_assignment"]["raw_sha256"]]
        and training["row_count_evidence_sha256"] == by_role["l1_row_count"]["raw_sha256"]
        and training["environment_lock_sha256"] == by_role["l1_environment_lock"]["raw_sha256"],
        {
            "source_snapshot_id": source["snapshot_id"],
            "source_tree_sha256": source["source_tree_sha256"],
            "training_snapshot_id": training["snapshot_id"],
            "required_source_snapshot_pairs": training["required_source_snapshot_pairs"],
        },
    )
    gates[REQUIRED_GATES[3]] = _gate_entry(
        registry["is_synthetic_registry"] is True
        and registry["is_synthetic_placeholder"] is False
        and registry["source_snapshot_id"].startswith("source://synthetic/")
        and registry["training_snapshot_id"].startswith("source://synthetic/")
        and registry["source_snapshot_id"] != source["snapshot_id"]
        and registry["training_snapshot_id"] != training["snapshot_id"]
        and fixture["production_model_authority"]["status"] == "fail_closed",
        {
            "registrar_root": by_role["l2_registrar_root"],
            "synthetic_registry": by_role["l2_synthetic_registry"],
            "contract_fixture_authority": by_role["l2_contract_fixture_authority"],
            "b1_l2_real_lineage_integration": False,
        },
    )
    gates[REQUIRED_GATES[4]] = _gate_entry(
        registry["split_plan_id"] == "split-2026.07.27-s1"
        and registry["split_plan_sha256"] == "3ba537cbcab0776efd2c8e0c582fb8f374c0d158cde48bba098014047f0649bb"
        and len(registry["split_plan"]["folds"]) == 2
        and len(registry["split_plan"]["sealed_holdouts"]) == 11,
        {
            "split_plan_id": registry["split_plan_id"],
            "split_plan_sha256": registry["split_plan_sha256"],
            "fold_count": len(registry["split_plan"]["folds"]),
            "sealed_holdout_count": len(registry["split_plan"]["sealed_holdouts"]),
            "authority_scope": "synthetic_protocol_mechanics_only",
        },
    )
    gates[REQUIRED_GATES[5]] = _gate_entry(
        foundation["synthetic_only"] is True
        and foundation["production_eligible"] is False
        and selection["synthetic_only"] is True
        and selection["production_eligible"] is False,
        {
            "foundation": by_role["l2_r20_foundation_authority"],
            "selection": by_role["l2_r20_selection_authority"],
        },
    )
    outer_ceiling = outer["claim_ceiling"]
    gates[REQUIRED_GATES[6]] = _gate_entry(
        all(
            outer_ceiling[key] is False
            for key in (
                "c1",
                "coverage",
                "pass_b2",
                "probability_wording",
                "promotion",
                "real_predictive_performance",
                "reliability",
                "served_approval",
                "sota",
            )
        )
        and outer_ceiling["scope"] == "synthetic_calibration_mechanics_only",
        {
            "authority": by_role["l2_outer_calibration_authority"],
            "config": by_role["l2_outer_calibration_config"],
            "source": by_role["l2_outer_calibration_source"],
        },
    )
    gates[REQUIRED_GATES[7]] = _gate_entry(
        b3_coverage["synthetic_only"] is True
        and b3_coverage["production_eligible"] is False
        and b3_coverage["claim_ceiling"] == ["synthetic_sbc_coverage_dependence_mechanics_only"],
        {
            "coverage_authority": by_role["l2_b3_coverage_authority"],
            "claim_ceiling": b3_coverage["claim_ceiling"],
        },
    )
    gates[REQUIRED_GATES[8]] = _gate_entry(
        b3_reliability["status"] == "fail_closed_until_b3"
        and b3_reliability["production_authorities"] == []
        and b3_reliability["synthetic_authority_allowed"] is False,
        {
            "reliability_authority": by_role["l2_b3_reliability_authority"],
            "status": b3_reliability["status"],
            "production_authorities": [],
            "synthetic_authority_allowed": False,
        },
    )
    gates[REQUIRED_GATES[9]] = _gate_entry(
        ontology["schema_version"] == "v2-champion-ontology-1"
        and ontology["snapshot_id"] == "scryglass:v2:champion-ontology:2026-08-07",
        {
            "schema": by_role["l3_ontology_seed"],
            "schema_implementation": by_role["l3_schema_implementation"],
            "sources": by_role["l3_ontology_sources"],
            "review": by_role["l3_review_log"],
            "fixtures": by_role["l3_evaluation_fixtures"],
            "private_pending_review_promoted": False,
            "invalid_no_score_training_target_promoted": False,
        },
    )
    gates[REQUIRED_GATES[10]] = _gate_entry(
        legacy["synthetic_only"] is True
        and legacy["production_eligible"] is False
        and legacy["report_sha256"] == "e72abafd61f9b3e693241eac659f46c393f9ae21d5ae011dfbf2d69cb96e7c45"
        and registry["b2_validation_report_sha256"] == "e15ec3144594974d6a13e59587797aa3d275a81769f89fc2b20125e286ff7fe1",
        {
            "legacy_report": by_role["l2_legacy_synthetic_report"],
            "internal_report_sha256": legacy["report_sha256"],
            "stale_frozen_pointer": registry["b2_validation_report_sha256"],
            "classification": "HISTORICAL_NONAUTHORITATIVE",
            "stale_pointer_repaired": False,
            "inner_synthetic_reliability_inherited": False,
            "inner_probability_wording_inherited": False,
        },
    )
    gates[REQUIRED_GATES[11]] = _gate_entry(
        foundation["production_eligible"] is False
        and selection["production_eligible"] is False
        and b3_coverage["production_eligible"] is False
        and legacy["production_eligible"] is False,
        {
            "l1_authority": "immutable_data_foundation_only",
            "l2_authority": "protocol_and_synthetic_mechanics_only",
            "l3_authority": "reviewed_ontology_foundation_only",
            "merged_or_escalated_authority": False,
        },
    )
    gates[REQUIRED_GATES[12]] = _gate_entry(
        len(sealed) == 1
        and isinstance(sealed[0], dict)
        and set(sealed[0]) == {"fixture_version", "kind"}
        and type(sealed[0]["fixture_version"]) is int
        and sealed[0]["fixture_version"] == 1
        and sealed[0]["kind"] == "empty-sealed-ledger"
        and source_policy["promotion_or_sealed_imports"] == []
        and source_policy["forbidden_calls"] == []
        and source_policy["unauthorized_write_surfaces"] == []
        and source_policy["authorized_output_locators"]
        == [CONFIG_LOCATOR, REPORT_LOCATOR, AUTHORITY_LOCATOR]
        and dict(CLAIM_BOUNDARY)["sealed_decision_opened"] is False,
        {
            "sealed_boundary": by_role["sealed_ledger_boundary"],
            "sealed_fixture_record_count": len(sealed),
            "sealed_fixture_semantics": sealed[0],
            "source_policy_audit": source_policy,
            "sealed_decision_opened": False,
        },
    )
    private_rows = private["rows"]
    gates[REQUIRED_GATES[13]] = _gate_entry(
        c4["authorities"] == []
        and c4["approved_packets"] == []
        and public["rows"] == []
        and authenticated["rows"] == []
        and len(private_rows) == 1
        and private_rows[0]["effective_decision"] == "private_pending_review",
        {
            "c4_registry": by_role["c4_authority_registry"],
            "public_allowlist": by_role["publication_public_allowlist"],
            "authenticated_allowlist": by_role["publication_authenticated_allowlist"],
            "private_allowlist": by_role["publication_private_allowlist"],
            "private_status": "private_pending_review",
            "production_authority": False,
            "publication_authority": False,
        },
    )
    gates[REQUIRED_GATES[14]] = _gate_entry(
        tuple(by_role) == tuple(spec.role for spec in _SPECS)
        and len({record["locator"] for record in records}) == len(records),
        {
            "role_count": len(records),
            "role_order": [record["role"] for record in records],
            "source_closure_sha256": canonical_sha256(records),
            "source_code": source_refs,
        },
    )
    gates[REQUIRED_GATES[15]] = _gate_entry(
        set(replay_proof)
        == {
            "proof_kind",
            "one_pass_materializations",
            "probe_materializations",
            "probe_state_sha256",
            "probe_byte_identical",
            "final_materializations",
            "final_state_sha256",
            "final_state_byte_identical",
            "serialized_final_bundles",
            "serialized_final_byte_identical",
        }
        and replay_proof["proof_kind"] == "measured_internal_one_pass_replay"
        and type(replay_proof["one_pass_materializations"]) is int
        and replay_proof["one_pass_materializations"] == 4
        and replay_proof["probe_materializations"] == 2
        and replay_proof["probe_byte_identical"] is True
        and replay_proof["final_materializations"] == 2
        and replay_proof["final_state_byte_identical"] is True
        and replay_proof["serialized_final_bundles"] == 2
        and replay_proof["serialized_final_byte_identical"] is True
        and replay_proof["probe_state_sha256"] == replay_proof["final_state_sha256"],
        {
            **replay_proof,
            "config_source_closure_sha256": canonical_sha256(records),
            "implementation_sha256": source_refs["checkpoint_implementation"]["raw_sha256"],
            "generator_sha256": source_refs["artifact_generator"]["raw_sha256"],
        },
    )
    if tuple(gates) != REQUIRED_GATES:
        raise ValidationFailure("C1 hard gate set or order is not exact")
    return gates


def _source_refs(
    root: Path,
    *,
    seen_inodes: dict[tuple[int, int], str],
) -> tuple[dict[str, dict[str, str]], dict[str, bytes]]:
    result: dict[str, dict[str, str]] = {}
    raw_result: dict[str, bytes] = {}
    for role, locator in (
        ("checkpoint_implementation", CHECKPOINT_SOURCE_LOCATOR),
        ("artifact_generator", GENERATOR_SOURCE_LOCATOR),
    ):
        raw = _safe_read(root, locator, seen_inodes=seen_inodes)
        result[role] = {"locator": locator, "raw_sha256": hashlib.sha256(raw).hexdigest()}
        raw_result[role] = raw
    return result, raw_result


@dataclass(frozen=True)
class _Materialization:
    records: list[dict[str, Any]]
    values: dict[str, Any]
    source_refs: dict[str, dict[str, str]]
    source_policy: dict[str, Any]
    state_bytes: bytes


def _materialize_once(
    root: Path,
    *,
    observer: Any = None,
) -> _Materialization:
    root = Path(root).absolute()
    seen: dict[tuple[int, int], str] = {}
    records, values, _ = _read_verified_inputs(root, seen_inodes=seen)
    source_refs, source_raw = _source_refs(root, seen_inodes=seen)
    source_policy = _audit_owned_sources(source_raw)
    state_bytes = canonical_json_bytes(
        {
            "records": records,
            "source_refs": source_refs,
            "source_policy": source_policy,
            "semantic_values_sha256": canonical_sha256(values),
        }
    )
    if observer is not None:
        observer(hashlib.sha256(state_bytes).hexdigest())
    return _Materialization(records, values, source_refs, source_policy, state_bytes)


def _serialize_bundle(
    materialization: _Materialization,
    replay_proof: dict[str, Any],
) -> dict[str, bytes]:
    records = materialization.records
    values = materialization.values
    source_refs = materialization.source_refs
    config = {
        "artifact_id": _ARTIFACT_IDENTITIES[CONFIG_LOCATOR][0],
        "schema_version": _ARTIFACT_IDENTITIES[CONFIG_LOCATOR][1],
        "decision_kind": DECISION_KIND,
        "location_disclaimer": LOCATION_DISCLAIMER,
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "required_gates": list(REQUIRED_GATES),
        "input_roles": records,
        "source_code": source_refs,
        "threat_model": dict(THREAT_MODEL),
    }
    config_bytes = canonical_json_bytes(config)
    gates = _evaluate_gates(
        records,
        values,
        source_refs,
        source_policy=materialization.source_policy,
        replay_proof=replay_proof,
    )
    report = {
        "artifact_id": _ARTIFACT_IDENTITIES[REPORT_LOCATOR][0],
        "schema_version": _ARTIFACT_IDENTITIES[REPORT_LOCATOR][1],
        "decision_kind": DECISION_KIND,
        "status": "ACCEPT",
        "acceptance_scope": "exact_enumerated_wave_1_input_bytes_and_identities_frozen_for_downstream_wave_2_work_only",
        "location_disclaimer": LOCATION_DISCLAIMER,
        "config_ref": {
            "locator": CONFIG_LOCATOR,
            "raw_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "object_sha256": canonical_sha256(config),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "gates": gates,
        "input_closure_sha256": canonical_sha256(records),
    }
    report_bytes = canonical_json_bytes(report)
    authority = {
        "artifact_id": _ARTIFACT_IDENTITIES[AUTHORITY_LOCATOR][0],
        "schema_version": _ARTIFACT_IDENTITIES[AUTHORITY_LOCATOR][1],
        "decision_kind": DECISION_KIND,
        "authority_scope": "wave_1_foundation_freeze_only",
        "location_disclaimer": LOCATION_DISCLAIMER,
        "config_ref": {
            "locator": CONFIG_LOCATOR,
            "raw_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "object_sha256": canonical_sha256(config),
        },
        "report_ref": {
            "locator": REPORT_LOCATOR,
            "raw_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "object_sha256": canonical_sha256(report),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "gate_set": {gate: True for gate in REQUIRED_GATES},
        "input_closure_sha256": canonical_sha256(records),
        "source_code": source_refs,
        "threat_model": dict(THREAT_MODEL),
    }
    authority_bytes = canonical_json_bytes(authority)
    return {
        CONFIG_LOCATOR: config_bytes,
        REPORT_LOCATOR: report_bytes,
        AUTHORITY_LOCATOR: authority_bytes,
    }


def build_checkpoint_c1_bundle(
    root: Path = ROOT,
    *,
    _observer: Any = None,
) -> dict[str, bytes]:
    _assert_runtime_integrity()
    root = Path(root).absolute()
    probe_a = _materialize_once(root, observer=_observer)
    probe_b = _materialize_once(root, observer=_observer)
    probe_equal = probe_a.state_bytes == probe_b.state_bytes
    if not probe_equal:
        raise ValidationFailure("C1 independent probe materializations diverged")
    final_a = _materialize_once(root, observer=_observer)
    final_b = _materialize_once(root, observer=_observer)
    final_equal = final_a.state_bytes == final_b.state_bytes
    if not final_equal or final_a.state_bytes != probe_a.state_bytes:
        raise ValidationFailure("C1 independent final materializations diverged")
    proof = {
        "proof_kind": "measured_internal_one_pass_replay",
        "one_pass_materializations": 4,
        "probe_materializations": 2,
        "probe_state_sha256": hashlib.sha256(probe_a.state_bytes).hexdigest(),
        "probe_byte_identical": probe_equal,
        "final_materializations": 2,
        "final_state_sha256": hashlib.sha256(final_a.state_bytes).hexdigest(),
        "final_state_byte_identical": final_equal,
        "serialized_final_bundles": 2,
        "serialized_final_byte_identical": True,
    }
    serialized_a = _serialize_bundle(final_a, proof)
    serialized_b = _serialize_bundle(final_b, proof)
    serialized_equal = serialized_a == serialized_b
    if not serialized_equal:
        raise ValidationFailure("C1 independently serialized final bundles diverged")
    if proof["serialized_final_byte_identical"] is not serialized_equal:
        raise ValidationFailure("C1 serialized replay evidence disagrees with measurement")
    return serialized_a


_CONFIG_KEYS = {
    "artifact_id", "schema_version", "decision_kind", "location_disclaimer",
    "contract_tree_sha256", "claim_boundary", "required_gates", "input_roles",
    "source_code", "threat_model",
}
_REPORT_KEYS = {
    "artifact_id", "schema_version", "decision_kind", "status", "acceptance_scope",
    "location_disclaimer", "config_ref", "claim_boundary", "gates",
    "input_closure_sha256",
}
_AUTHORITY_KEYS = {
    "artifact_id", "schema_version", "decision_kind", "authority_scope",
    "location_disclaimer", "config_ref", "report_ref", "claim_boundary", "gate_set",
    "input_closure_sha256", "source_code", "threat_model",
}


def _validate_artifact(
    raw: bytes,
    locator: str,
    expected_keys: set[str],
) -> dict[str, Any]:
    value = _strict_json(raw)
    if not isinstance(value, dict):
        raise ValidationFailure(f"C1 artifact must be an object: {locator}")
    if raw != canonical_json_bytes(value):
        raise ValidationFailure(f"C1 artifact bytes are not canonical: {locator}")
    _require_exact_keys(value, expected_keys, locator)
    expected_id, expected_schema = _ARTIFACT_IDENTITIES[locator]
    if value["artifact_id"] != expected_id or value["schema_version"] != expected_schema:
        raise ValidationFailure(f"C1 artifact identity mismatch: {locator}")
    if value["decision_kind"] != DECISION_KIND:
        raise ValidationFailure(f"C1 decision kind mismatch: {locator}")
    if value["location_disclaimer"] != LOCATION_DISCLAIMER:
        raise ValidationFailure(f"C1 location disclaimer mismatch: {locator}")
    _validate_claim_boundary(value["claim_boundary"])
    return value


def _validate_ref(ref: Any, locator: str, raw: bytes, value: dict[str, Any]) -> None:
    _require_exact_keys(ref, {"locator", "raw_sha256", "object_sha256"}, f"C1 ref {locator}")
    if ref["locator"] != locator:
        raise ValidationFailure(f"C1 ref locator mismatch: {locator}")
    if ref["raw_sha256"] != hashlib.sha256(raw).hexdigest():
        raise ValidationFailure(f"C1 ref raw hash mismatch: {locator}")
    if ref["object_sha256"] != canonical_sha256(value):
        raise ValidationFailure(f"C1 ref object hash mismatch: {locator}")


def _validate_gate_bundle(gates: Any) -> None:
    if not isinstance(gates, dict) or set(gates) != set(REQUIRED_GATES):
        raise ValidationFailure("C1 report gate set is not exact")
    for gate, entry in gates.items():
        _require_exact_keys(entry, {"passed", "evidence", "evidence_sha256"}, f"C1 gate {gate}")
        if type(entry["passed"]) is not bool or entry["passed"] is not True:
            raise ValidationFailure(f"C1 gate is not exactly true: {gate}")
        if not isinstance(entry["evidence"], dict):
            raise ValidationFailure(f"C1 gate evidence is not bounded object: {gate}")
        if entry["evidence_sha256"] != canonical_sha256(entry["evidence"]):
            raise ValidationFailure(f"C1 gate evidence hash mismatch: {gate}")


def _authenticate_bundle(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _assert_runtime_integrity()
    root = Path(root).absolute()
    seen: dict[tuple[int, int], str] = {}
    records, _, _ = _read_verified_inputs(root, seen_inodes=seen)
    source_refs, source_raw = _source_refs(root, seen_inodes=seen)
    _audit_owned_sources(source_raw)
    config_raw = _safe_read(root, CONFIG_LOCATOR, seen_inodes=seen)
    report_raw = _safe_read(root, REPORT_LOCATOR, seen_inodes=seen)
    authority_raw = _safe_read(root, AUTHORITY_LOCATOR, seen_inodes=seen)
    config = _validate_artifact(config_raw, CONFIG_LOCATOR, _CONFIG_KEYS)
    report = _validate_artifact(report_raw, REPORT_LOCATOR, _REPORT_KEYS)
    authority = _validate_artifact(authority_raw, AUTHORITY_LOCATOR, _AUTHORITY_KEYS)

    if config["contract_tree_sha256"] != CONTRACT_TREE_SHA256:
        raise ValidationFailure("C1 config contract tree mismatch")
    if config["required_gates"] != list(REQUIRED_GATES):
        raise ValidationFailure("C1 config gate declaration mismatch")
    if config["input_roles"] != records:
        raise ValidationFailure("C1 config source closure mismatch")
    if config["source_code"] != source_refs or authority["source_code"] != source_refs:
        raise ValidationFailure("C1 source implementation identity mismatch")
    if config["threat_model"] != dict(THREAT_MODEL) or authority["threat_model"] != dict(THREAT_MODEL):
        raise ValidationFailure("C1 threat model mismatch")
    closure_hash = canonical_sha256(records)
    if report["input_closure_sha256"] != closure_hash or authority["input_closure_sha256"] != closure_hash:
        raise ValidationFailure("C1 input closure hash mismatch")

    _validate_ref(report["config_ref"], CONFIG_LOCATOR, config_raw, config)
    _validate_ref(authority["config_ref"], CONFIG_LOCATOR, config_raw, config)
    _validate_ref(authority["report_ref"], REPORT_LOCATOR, report_raw, report)
    if report["status"] != "ACCEPT" or report["acceptance_scope"] != "exact_enumerated_wave_1_input_bytes_and_identities_frozen_for_downstream_wave_2_work_only":
        raise ValidationFailure("C1 report freeze status semantics mismatch")
    _validate_gate_bundle(report["gates"])
    if not isinstance(authority["gate_set"], dict) or set(authority["gate_set"]) != set(REQUIRED_GATES):
        raise ValidationFailure("C1 authority gate set mismatch")
    if any(type(value) is not bool or value is not True for value in authority["gate_set"].values()):
        raise ValidationFailure("C1 authority contains a non-true gate")
    if authority["authority_scope"] != "wave_1_foundation_freeze_only":
        raise ValidationFailure("C1 authority scope mismatch")

    expected = build_checkpoint_c1_bundle(root)
    actual = {
        CONFIG_LOCATOR: config_raw,
        REPORT_LOCATOR: report_raw,
        AUTHORITY_LOCATOR: authority_raw,
    }
    if actual != expected:
        raise ValidationFailure("C1 exact fresh replay mismatch")
    return config, report, authority


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _code_payload(code: CodeType) -> dict[str, Any]:
    constants: list[Any] = []
    for value in code.co_consts:
        if isinstance(value, CodeType):
            constants.append({"code": _code_payload(value)})
        else:
            constants.append(
                {
                    "type": type(value).__qualname__,
                    "repr": repr(value),
                }
            )
    return {
        "bytecode": code.co_code.hex(),
        "constants": constants,
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "flags": code.co_flags,
    }


def _callable_fingerprint(function: Any) -> str:
    payload = {
        "code_sha256": canonical_sha256(_code_payload(function.__code__)),
        "defaults": repr(function.__defaults__),
        "kwdefaults": repr(function.__kwdefaults__),
    }
    return canonical_sha256(payload)


def _critical_constants_payload() -> dict[str, Any]:
    return {
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "decision_kind": DECISION_KIND,
        "location_disclaimer": LOCATION_DISCLAIMER,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "required_gates": list(REQUIRED_GATES),
        "threat_model": dict(THREAT_MODEL),
        "locators": {
            "config": CONFIG_LOCATOR,
            "report": REPORT_LOCATOR,
            "authority": AUTHORITY_LOCATOR,
            "checkpoint_source": CHECKPOINT_SOURCE_LOCATOR,
            "generator_source": GENERATOR_SOURCE_LOCATOR,
        },
        "artifact_identities": _ARTIFACT_IDENTITIES,
        "specs": [
            {
                "role": spec.role,
                "layer": spec.layer,
                "classification": spec.classification,
                "locator": spec.locator,
                "raw_sha256": spec.raw_sha256,
                "object_sha256": spec.object_sha256,
                "encoding": spec.encoding,
            }
            for spec in _SPECS
        ],
    }


def _make_authority_api(
    guard_cell: list[Any],
    authenticate_bundle: Any,
    freeze_value: Any,
) -> tuple[type, Any]:
    issued_roots: weakref.WeakKeyDictionary[Any, Path] = weakref.WeakKeyDictionary()

    class _CheckpointC1Authority:
        """Loader-issued, read-only projection over a freshly authenticated bundle."""

        __slots__ = ("__weakref__",)

        def __new__(cls, *args: Any, **kwargs: Any) -> "_CheckpointC1Authority":
            raise TypeError("CheckpointC1Authority instances are loader-issued only")

        def __init_subclass__(cls, **kwargs: Any) -> None:
            raise TypeError("CheckpointC1Authority cannot be subclassed")

        def __setattr__(self, name: str, value: Any) -> None:
            raise AttributeError("CheckpointC1Authority is read-only")

        def authenticate(self) -> Mapping[str, Any]:
            guard_cell[0]()
            if type(self) is not _CheckpointC1Authority:
                raise ValidationFailure("substituted C1 authority type")
            root = issued_roots.get(self)
            if root is None:
                raise ValidationFailure("unissued or forged C1 authority")
            _, _, authority = authenticate_bundle(root)
            return freeze_value(authority)

        @property
        def payload(self) -> Mapping[str, Any]:
            return self.authenticate()

    _CheckpointC1Authority.__name__ = "CheckpointC1Authority"
    _CheckpointC1Authority.__qualname__ = "CheckpointC1Authority"

    def _load(root: Path = ROOT) -> _CheckpointC1Authority:
        guard_cell[0]()
        checked_root = Path(root).absolute()
        authenticate_bundle(checked_root)
        authority = object.__new__(_CheckpointC1Authority)
        issued_roots[authority] = checked_root
        return authority

    return _CheckpointC1Authority, _load


_AUTHORITY_GUARD_CELL: list[Any] = [
    lambda: (_ for _ in ()).throw(ValidationFailure("C1 runtime guard is not initialized"))
]
CheckpointC1Authority, load_checkpoint_c1 = _make_authority_api(
    _AUTHORITY_GUARD_CELL,
    _authenticate_bundle,
    _freeze,
)


def _make_runtime_guard() -> Any:
    global_names = (
        "canonical_json_bytes",
        "canonical_sha256",
        "_duplicate_rejecting_pairs",
        "_reject_constant",
        "_strict_json",
        "_validate_locator",
        "_lstat_component",
        "_safe_read",
        "_parse_role",
        "_identity",
        "_read_verified_inputs",
        "_require_exact_keys",
        "_validate_claim_boundary",
        "_gate_entry",
        "_audit_owned_sources",
        "_evaluate_gates",
        "_source_refs",
        "_materialize_once",
        "_serialize_bundle",
        "build_checkpoint_c1_bundle",
        "_validate_artifact",
        "_validate_ref",
        "_validate_gate_bundle",
        "_authenticate_bundle",
        "_freeze",
        "_code_payload",
        "_callable_fingerprint",
        "_critical_constants_payload",
        "load_checkpoint_c1",
    )
    captured = {name: globals()[name] for name in global_names}
    captured_fingerprints = {
        name: _callable_fingerprint(function)
        for name, function in captured.items()
    }
    class_targets = {
        "CheckpointC1Authority.__new__": CheckpointC1Authority.__new__,
        "CheckpointC1Authority.__init_subclass__": CheckpointC1Authority.__init_subclass__.__func__,
        "CheckpointC1Authority.__setattr__": CheckpointC1Authority.__setattr__,
        "CheckpointC1Authority.authenticate": CheckpointC1Authority.authenticate,
        "CheckpointC1Authority.payload": CheckpointC1Authority.payload.fget,
    }
    class_fingerprints = {
        name: _callable_fingerprint(function)
        for name, function in class_targets.items()
    }
    constants_sha256 = canonical_sha256(_critical_constants_payload())
    fingerprint = _callable_fingerprint
    constants_payload = _critical_constants_payload

    def _guard() -> None:
        for name, expected in captured.items():
            current = globals().get(name)
            if current is not expected:
                raise ValidationFailure(f"C1 runtime helper substitution rejected: {name}")
            if fingerprint(current) != captured_fingerprints[name]:
                raise ValidationFailure(f"C1 runtime helper mutation rejected: {name}")
        current_class_targets = {
            "CheckpointC1Authority.__new__": CheckpointC1Authority.__new__,
            "CheckpointC1Authority.__init_subclass__": CheckpointC1Authority.__init_subclass__.__func__,
            "CheckpointC1Authority.__setattr__": CheckpointC1Authority.__setattr__,
            "CheckpointC1Authority.authenticate": CheckpointC1Authority.authenticate,
            "CheckpointC1Authority.payload": CheckpointC1Authority.payload.fget,
        }
        for name, expected in class_targets.items():
            current = current_class_targets[name]
            if current is not expected or fingerprint(current) != class_fingerprints[name]:
                raise ValidationFailure(f"C1 authority method mutation rejected: {name}")
        if canonical_sha256(constants_payload()) != constants_sha256:
            raise ValidationFailure("C1 critical constant mutation rejected")

    return _guard


_assert_runtime_integrity = _make_runtime_guard()
_AUTHORITY_GUARD_CELL[0] = _assert_runtime_integrity
del _AUTHORITY_GUARD_CELL
del _make_authority_api
del _make_runtime_guard


__all__ = [
    "AUTHORITY_LOCATOR",
    "CHECKPOINT_SOURCE_LOCATOR",
    "CLAIM_BOUNDARY",
    "CONFIG_LOCATOR",
    "DECISION_KIND",
    "GENERATOR_SOURCE_LOCATOR",
    "INPUT_ROLE_LOCATORS",
    "LOCATION_DISCLAIMER",
    "REPORT_LOCATOR",
    "REQUIRED_GATES",
    "ROOT",
    "THREAT_MODEL",
    "CheckpointC1Authority",
    "build_checkpoint_c1_bundle",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_checkpoint_c1",
]
