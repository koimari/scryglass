"""Executable structural and semantic validation for the five public v2 outputs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from ..data.source_tree import canonical_source_tree_sha256
from .checks import ValidationFailure
from .types import CONTRACT_TREE_SHA256, canonical_json, canonical_sha256


CONTRACT_ROOT = Path("docs/model-v2/contracts")
CONTRACT_PACK_ROOT = CONTRACT_ROOT.parent
REPOSITORY_ROOT = Path(".")
CONTRACT_SOURCE_TREE_ALLOWLIST = (
    "docs/model-v2/README.md",
    "docs/model-v2/acceptance-gates.md",
    "docs/model-v2/build-order.md",
    "docs/model-v2/contracts/common.schema.json",
    "docs/model-v2/contracts/draft-score.schema.json",
    "docs/model-v2/contracts/examples/draft-score.example.json",
    "docs/model-v2/contracts/examples/partial-draft-state.example.json",
    "docs/model-v2/contracts/examples/player-rating.example.json",
    "docs/model-v2/contracts/examples/team-rating.example.json",
    "docs/model-v2/contracts/examples/tier-list.example.json",
    "docs/model-v2/contracts/model-manifest.schema.json",
    "docs/model-v2/contracts/partial-draft-state.schema.json",
    "docs/model-v2/contracts/player-rating.schema.json",
    "docs/model-v2/contracts/prediction-provenance.schema.json",
    "docs/model-v2/contracts/publication-matrix.schema.json",
    "docs/model-v2/contracts/team-rating.schema.json",
    "docs/model-v2/contracts/tier-list.schema.json",
    "docs/model-v2/data-contract.md",
    "docs/model-v2/estimands.md",
    "docs/model-v2/evaluation-contract.md",
    "docs/model-v2/interface-contract.md",
    "docs/model-v2/mathematical-contract.md",
    "docs/model-v2/product-contract.md",
    "docs/model-v2/research-register.md",
    "docs/model-v2/sources.md",
)
SEMANTIC_ARTIFACT_PATH = Path(
    "data/lol/v2/evaluation/contract-semantic-artifacts.json"
)
CONTRACT_FIXTURE_AUTHORITY_PATH = Path(
    "data/lol/v2/evaluation/contract-fixture-authority.json"
)
CONTRACT_VALIDATION_TRUST_ROOT_PATH = Path(
    "data/lol/v2/evaluation/contract-validation-trust-root.json"
)
EXPECTED_CONTRACT_VALIDATION_TRUST_ROOT_RAW_SHA256 = (
    "2b13b17b6e2f1f1d726c17737576459b754410ca425ae55642fcd21b7caad3f2"
)
EXPECTED_CONTRACT_VALIDATION_TRUST_ROOT_OBJECT_SHA256 = (
    "582f6a1d679cccd8f6f4a6d94ce4b2e9ff601e806d53f3395624f8205c012525"
)
EXPECTED_CONTRACT_FIXTURE_AUTHORITY_RAW_SHA256 = (
    "d7b95f4ddc365f7289f4d81726343d3a0a4c97e66a5f5ce5c844bfae55cffecb"
)
EXPECTED_CONTRACT_FIXTURE_AUTHORITY_OBJECT_SHA256 = (
    "9b938413fd8de6017fdf8a8aed544707436b29844300c7079cb74c30559a89d8"
)
EXPECTED_SEMANTIC_ARTIFACT_SHA256 = (
    "652aab70417ddf3f1727303836f059445dac8a51b2f615bbfad5d6eabe0c09ed"
)
EXPECTED_CONTRACT_CONTENT_SHA256 = (
    "3a9435b05b4a95eae4488f3d232b8974d0d9671a29bb3a9b9710d85c83a431c0"
)
if len(CONTRACT_SOURCE_TREE_ALLOWLIST) != 25:
    raise RuntimeError("frozen C0 allowlist must contain exactly 25 files")


@dataclass(frozen=True)
class ContractValidationAnchors:
    """Explicit byte anchors for one non-mutating contract-validation replay.

    Passing a different anchor set changes only which inputs the replay checks.
    It does not activate a trust root or grant model, probability, publication,
    recommendation, odds, expected-value, or betting authority.
    """

    contract_tree_sha256: str
    schema_sha256: Mapping[str, str]
    example_sha256: Mapping[str, str]
    semantic_artifact_path: Path
    semantic_artifact_raw_sha256: str
    contract_fixture_authority_path: Path
    contract_fixture_authority_raw_sha256: str
    contract_fixture_authority_object_sha256: str
    contract_validation_trust_root_path: Path
    contract_validation_trust_root_raw_sha256: str
    contract_validation_trust_root_object_sha256: str
    contract_content_sha256: str


def _actual_contract_tree_sha256(root: Path = REPOSITORY_ROOT) -> str:
    return canonical_source_tree_sha256(root, CONTRACT_SOURCE_TREE_ALLOWLIST)


def _verify_contract_validation_trust_root(
    root: Path = REPOSITORY_ROOT,
    anchors: ContractValidationAnchors | None = None,
) -> None:
    resolved = anchors or default_contract_validation_anchors()
    raw = (root / resolved.contract_validation_trust_root_path).read_bytes()
    payload = json.loads(raw)
    if (
        hashlib.sha256(raw).hexdigest()
        != resolved.contract_validation_trust_root_raw_sha256
        or canonical_sha256(payload)
        != resolved.contract_validation_trust_root_object_sha256
        or payload["contract_tree_sha256"] != resolved.contract_tree_sha256
    ):
        raise ValidationFailure("contract validation trust root is stale")
    for artifact in payload["artifacts"]:
        artifact_raw = (root / Path(artifact["locator"])).read_bytes()
        if hashlib.sha256(artifact_raw).hexdigest() != artifact["raw_sha256"]:
            raise ValidationFailure(
                f"trusted artifact raw hash mismatch: {artifact['artifact_id']}"
            )
        if artifact["kind"] == "json":
            object_hash = canonical_sha256(json.loads(artifact_raw))
        elif artifact["kind"] == "jsonl":
            object_hash = canonical_sha256(
                [
                    json.loads(line)
                    for line in artifact_raw.decode("utf-8").splitlines()
                    if line
                ]
            )
        else:
            object_hash = hashlib.sha256(artifact_raw).hexdigest()
        if object_hash != artifact["object_sha256"]:
            raise ValidationFailure(
                f"trusted artifact object hash mismatch: {artifact['artifact_id']}"
            )
OUTPUT_SCHEMAS: Mapping[str, str] = {
    "player_rating": "player-rating.schema.json",
    "team_rating": "team-rating.schema.json",
    "draft_score": "draft-score.schema.json",
    "partial_draft_state": "partial-draft-state.schema.json",
    "tier_list": "tier-list.schema.json",
}
SHARED_SCHEMAS = ("common.schema.json", "prediction-provenance.schema.json")
SCHEMA_FILES = tuple(OUTPUT_SCHEMAS.values()) + SHARED_SCHEMAS
EXAMPLE_FILES: Mapping[str, str] = {
    output: f"examples/{schema_name.removesuffix('.schema.json')}.example.json"
    for output, schema_name in OUTPUT_SCHEMAS.items()
}

# These anchors intentionally live in L2. A frozen schema or example change must
# be explicitly reconciled rather than silently changing evaluation behavior.
EXPECTED_SCHEMA_SHA256: Mapping[str, str] = {
    "player-rating.schema.json": "e76f0949dc3f44292edb2b6a3de57d15f7ab4459cc6d87f2f4595d6f84434c41",
    "team-rating.schema.json": "b0284dc172435026c583b5dd7f032661eda380afb085123a2d5d336d34bf8b9c",
    "draft-score.schema.json": "a11dd7bab9613c534d98c5631737f026417fd8286a232be1bf6e85688813106e",
    "partial-draft-state.schema.json": "b33ea32b8ae17b1b193ebf236b68d559126d23b883f738e9285f9cb2f886cde4",
    "tier-list.schema.json": "abc5041e8143cc678a43e363271eec220e45c61ed34aa2430747f53f531088ae",
    "common.schema.json": "3b8791c0af75fdfd7056ad4db728f6fa6bf05548f4f1ec55aba4ecc9366803f7",
    "prediction-provenance.schema.json": "ea74ec5034472a8139615697c6dda6f8d12888d09405477ed413e82af0f40e83",
}
EXPECTED_EXAMPLE_SHA256: Mapping[str, str] = {
    "examples/player-rating.example.json": "25887d7f9d467c2affd188c07d22d0bf09319f03f26199b244b91f1316ab9aa3",
    "examples/team-rating.example.json": "15c2d32a815f1b0833bb873d34cd190c4396d64926aaf25169869fea4771646f",
    "examples/draft-score.example.json": "8e9b8cc3a90b55f0b0d1446926c81a1b9aef723a9f13c5161b766a970df54b8f",
    "examples/partial-draft-state.example.json": "c933e99fcf8615a1123ba316be22868307e1709c4ad65152a8ff9cd38023cf14",
    "examples/tier-list.example.json": "aae28f63b3461d9c04f00c54fb63912301d8ff6e39404e7029346891e622ae5a",
}


def default_contract_validation_anchors() -> ContractValidationAnchors:
    """Return the frozen production-default validation anchors."""

    return ContractValidationAnchors(
        contract_tree_sha256=CONTRACT_TREE_SHA256,
        schema_sha256=dict(EXPECTED_SCHEMA_SHA256),
        example_sha256=dict(EXPECTED_EXAMPLE_SHA256),
        semantic_artifact_path=SEMANTIC_ARTIFACT_PATH,
        semantic_artifact_raw_sha256=EXPECTED_SEMANTIC_ARTIFACT_SHA256,
        contract_fixture_authority_path=CONTRACT_FIXTURE_AUTHORITY_PATH,
        contract_fixture_authority_raw_sha256=(
            EXPECTED_CONTRACT_FIXTURE_AUTHORITY_RAW_SHA256
        ),
        contract_fixture_authority_object_sha256=(
            EXPECTED_CONTRACT_FIXTURE_AUTHORITY_OBJECT_SHA256
        ),
        contract_validation_trust_root_path=(
            CONTRACT_VALIDATION_TRUST_ROOT_PATH
        ),
        contract_validation_trust_root_raw_sha256=(
            EXPECTED_CONTRACT_VALIDATION_TRUST_ROOT_RAW_SHA256
        ),
        contract_validation_trust_root_object_sha256=(
            EXPECTED_CONTRACT_VALIDATION_TRUST_ROOT_OBJECT_SHA256
        ),
        contract_content_sha256=EXPECTED_CONTRACT_CONTENT_SHA256,
    )


EXPECTED_STRUCTURAL_DIAGNOSTIC_SHA256: Mapping[str, str] = {
    "archetype_extrapolation_add_reliability_forbidden": "82e7bcb7411209c0308fe2b2cfd3bff72e50fdf8c53a0d84088d3f597ab26211",
    "archetype_extrapolation_add_score_forbidden": "a143729ee59f9a9c971ce2400793d9575f4362b57ff56c50970659428acb76d0",
    "archetype_extrapolation_add_transform_forbidden": "ce5e49619610187f58024cb5b4a489403b90c705c7951b2c40417147f080e1c2",
    "archetype_extrapolation_nonzero_exact_cell_forbidden": "f9b03a789202264e27e17584f17906b20b5fb85b63a682e6ea4afe8f876721f7",
    "contextual_archetype_exact_residual_forbidden": "7fc3b88c375f619aea398accd1ca249730adfe80ca2b014aaad778a9a1736d97",
    "contextual_archetype_stale_roster_forbidden": "094e6faddab290d307a6feb636888424ec0b9611a03eebe32ec6f2ec4d381797",
    "evidence_aggregate_scalar_forbidden": "a619cc0d2705b51779fb32ce681250bb52467b73dc78b7c74380653c7a2d9aef",
    "evidence_game_count_proxy_forbidden": "fc75eaf2beae79de38332e9ef3ea50ac43678bc4c97b65732c92174aecda3ae0",
    "forecast_simulation_without_replay": "70a05d5fdd88f54e84133ba13941aa9c7979dc9735bcab59ffcd836fc8ad2260",
    "neutral_response_with_contextual_comparison": "0c0fe976e91a50c228dfa71180f5f2c1f82802ee7e43196575bcfedd1c774e44",
    "partial_approximate_search_missing_bound": "cbfc1ef0e4ff14d7846f5c117b45977f97884f84fb9e5f01c75f6a6d9851f3a0",
    "partial_exact_search_partial_coverage": "a3722d7c18b6553d6e573e5d01a0329c39de3478452f566b8977779d4a3a0954",
    "partial_terminal_without_delegation": "edf2978d4aad51c4e51045cbfa4e5eccd3858aefcb40ebf4e383e72897b3ce7f",
    "player_duplicate_writable_rating": "69e42d5189cddf8e04780adc6390fdca264743b97677ad53f580bda806304e6c",
    "player_global_without_structural_eligibility": "45e5a86c47c3a7a15ee82fd7d5f3af68786db99bd6c1671dabc4a1efee5d65cc",
    "reliability_high_missing_baseline": "81023eecdb71a4661f17056d00199204607d60d8c321c502dafaf330078df211",
    "reliability_high_ood": "75913d6e4f47e4c05eac61606a83fc355a2f7f7e21a06f1c2ff0712e40aa639f",
    "reliability_high_unmatched_stratum": "32e3a32db274c1285b50effad42a42c0c443b22486053466d106a0494a2f7bbe",
    "reliability_high_zero_clusters": "b05f3437052890ef09f3630c6be3491bf7b6839960cff898c1d2478f16f69018",
    "settled_probability_equal_boundary": "b4e942d1c7c07ca63ba1b0c0bb168557a6cc14ece2019cfa1b9f7d20e26effcf",
    "team_duplicate_writable_rating": "69e42d5189cddf8e04780adc6390fdca264743b97677ad53f580bda806304e6c",
    "team_regional_league_component_forbidden": "3e8b101da8f6b93384460081aefd3d31dffcadeae1917d3c32d46cc1e1f64a00",
    "terminal_assignment_missing_side_role": "7e02a65c0323d5d622b236dab4fa5d68b249ed70f3bacd3a902b20778bf046e5",
    "tier_list_international_scope_forbidden": "9b1389c54a0076ea889af30485ddc1fef23f497b33325915c77ca91ebf000499",
}


@dataclass(frozen=True)
class ValidationEvidence:
    evidence_id: str
    kind: str
    status: str
    detail: str
    applicable: bool = True

    def to_payload(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "status": self.status,
            "detail": self.detail,
            "applicable": self.applicable,
        }


@dataclass(frozen=True)
class FiveOutputValidationReport:
    contract_tree_sha256: str
    schema_sha256: Mapping[str, str]
    example_sha256: Mapping[str, str]
    semantic_artifact_sha256: str
    contract_fixture_authority_raw_sha256: str
    contract_fixture_authority_object_sha256: str
    contract_validation_trust_root_raw_sha256: str
    contract_validation_trust_root_object_sha256: str
    contract_content_sha256: str
    invariant_ids: tuple[str, ...]
    mutation_ids: tuple[str, ...]
    evidence: tuple[ValidationEvidence, ...]
    invariant_counts: Mapping[str, Mapping[str, int]]
    invariant_pass_count: int
    mutation_pass_count: int
    structural_pass_count: int
    all_pass: bool
    report_sha256: str

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "contract_tree_sha256": self.contract_tree_sha256,
            "schema_sha256": dict(self.schema_sha256),
            "example_sha256": dict(self.example_sha256),
            "semantic_artifact_sha256": self.semantic_artifact_sha256,
            "contract_fixture_authority_raw_sha256": (
                self.contract_fixture_authority_raw_sha256
            ),
            "contract_fixture_authority_object_sha256": (
                self.contract_fixture_authority_object_sha256
            ),
            "contract_validation_trust_root_raw_sha256": (
                self.contract_validation_trust_root_raw_sha256
            ),
            "contract_validation_trust_root_object_sha256": (
                self.contract_validation_trust_root_object_sha256
            ),
            "contract_content_sha256": self.contract_content_sha256,
            "invariant_ids": list(self.invariant_ids),
            "mutation_ids": list(self.mutation_ids),
            "evidence": [item.to_payload() for item in self.evidence],
            "invariant_counts": {
                name: dict(counts)
                for name, counts in self.invariant_counts.items()
            },
            "invariant_pass_count": self.invariant_pass_count,
            "mutation_pass_count": self.mutation_pass_count,
            "structural_pass_count": self.structural_pass_count,
            "all_pass": self.all_pass,
        }

    def to_payload(self) -> dict[str, Any]:
        return {**self.unsigned_payload(), "report_sha256": self.report_sha256}

    def verify_hash(self) -> bool:
        return self.report_sha256 == canonical_sha256(self.unsigned_payload())


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract_content_sha256(contract_pack_root: Path) -> str:
    file_hashes = {
        str(path.relative_to(contract_pack_root)): _raw_sha256(path)
        for path in sorted(contract_pack_root.rglob("*"))
        if path.is_file()
    }
    return hashlib.sha256(
        json.dumps(
            file_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _load_semantic_artifacts() -> Mapping[str, Any]:
    if (
        not SEMANTIC_ARTIFACT_PATH.is_file()
        or _raw_sha256(SEMANTIC_ARTIFACT_PATH)
        != EXPECTED_SEMANTIC_ARTIFACT_SHA256
    ):
        raise ValidationFailure("semantic registry/artifact drift detected")
    return json.loads(SEMANTIC_ARTIFACT_PATH.read_text(encoding="utf-8"))


SEMANTIC_ARTIFACTS = _load_semantic_artifacts()


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _pass(detail: str = "passed") -> tuple[bool, str]:
    return True, detail


def _not_applicable(detail: str = "not applicable to canonical fixture") -> tuple[bool, str]:
    return True, detail


def _fail(detail: str) -> tuple[bool, str]:
    return False, detail


def _status_ok(document: Mapping[str, Any]) -> bool:
    return document.get("status") == "ok"


def _interval_contains(document: Mapping[str, Any], value_key: str) -> tuple[bool, str]:
    if not _status_ok(document):
        return _not_applicable()
    interval = document["interval_95"]
    value = document[value_key]
    return (
        _pass()
        if interval["lower"] <= value <= interval["upper"]
        else _fail("value falls outside interval_95")
    )


def _settled_width(document: Mapping[str, Any]) -> tuple[bool, str]:
    if not _status_ok(document):
        return _not_applicable()
    width = document["interval_95"]["upper"] - document["interval_95"]["lower"]
    return (
        _pass()
        if _close(document["settled"]["interval_width"], width)
        else _fail("settled interval width does not match interval_95")
    )


def _revision_chain(revisions: Sequence[Mapping[str, Any]]) -> bool:
    seen: dict[str, int] = {}
    previous_index: int | None = None
    for revision in revisions:
        index = int(revision["revision_index"])
        if previous_index is not None and index <= previous_index:
            return False
        supersedes = revision.get("supersedes_revision_id")
        if supersedes is not None and (
            supersedes not in seen or seen[supersedes] >= index
        ):
            return False
        seen[str(revision["revision_id"])] = index
        previous_index = index
    return True


def _assignments_match_latest(
    assignments: Sequence[Mapping[str, Any]],
    revisions: Sequence[Mapping[str, Any]],
) -> bool:
    latest: dict[str, Mapping[str, Any]] = {}
    for revision in revisions:
        action_id = str(revision["action_id"])
        current = latest.get(action_id)
        if current is None or int(revision["revision_index"]) > int(current["revision_index"]):
            latest[action_id] = revision
    by_action = {str(item["action_id"]): item for item in assignments}
    return all(
        action_id in by_action
        and by_action[action_id]["role"] == revision["revised_role"]
        for action_id, revision in latest.items()
    )


def _constraints_match_latest(state: Mapping[str, Any]) -> bool:
    latest: dict[str, Mapping[str, Any]] = {}
    for revision in state.get("role_constraint_revisions", []):
        action_id = str(revision["action_id"])
        current = latest.get(action_id)
        if current is None or int(revision["revision_index"]) > int(current["revision_index"]):
            latest[action_id] = revision
    actions = {
        str(item["action_id"]): item
        for item in state.get("actions", [])
        if item.get("kind") == "pick"
    }
    constraints = {
        str(item["action_id"]): item
        for item in state.get("current_role_constraints", [])
    }
    for action_id, action in actions.items():
        expected = (
            latest[action_id]["revised_role_set"]
            if action_id in latest
            else action.get("role_set", [])
        )
        if action_id not in constraints or set(constraints[action_id]["role_set"]) != set(expected):
            return False
    return True


def _source_available(document: Mapping[str, Any], *, partial: bool) -> bool:
    state = document["state"] if partial else document
    as_of = _parse_timestamp(str(document["as_of"]))
    return (
        _parse_timestamp(str(state["side_mapping"]["available_at"])) <= as_of
        and _parse_timestamp(str(state["source_record"]["available_at"])) <= as_of
    )


def _actions_ordered(actions: Sequence[Mapping[str, Any]]) -> bool:
    slots = [int(item["slot"]) for item in actions]
    return slots == sorted(slots) and len(slots) == len(set(slots))


def _scope_valid(document: Mapping[str, Any], *, partial: bool = False) -> bool:
    state = document["state"] if partial else document
    kind = state.get("competition_scope_kind")
    scope_id = str(state.get("competition_scope_id") or "")
    return (
        SEMANTIC_ARTIFACTS["registries"]["competition_scopes"].get(scope_id)
        == kind
    )


def _protocol_actions_legal(state: Mapping[str, Any]) -> bool:
    protocol = SEMANTIC_ARTIFACTS["protocols"].get(state.get("protocol_id"))
    if not isinstance(protocol, Mapping):
        return False
    legal_by_slot = {
        int(item["slot"]): item for item in protocol["action_slots"]
    }
    for action in state.get("actions", []):
        legal = legal_by_slot.get(int(action["slot"]))
        if (
            legal is None
            or action["kind"] != legal["kind"]
            or action["canonical_side"] != legal["canonical_side"]
        ):
            return False
    return True


def _team_component_math(document: Mapping[str, Any]) -> bool:
    policy = SEMANTIC_ARTIFACTS["team_component_policies"].get(
        document.get("policy_snapshot_id")
    )
    if not isinstance(policy, Mapping):
        return False
    c_e = float(policy["c_E"])
    tolerance = float(policy["tolerance"])
    expected_roster = c_e * float(policy["A_q"])
    expected_synergy = c_e * float(policy["gamma_q"])
    expected_league = (
        c_e * float(policy["lambda_L"])
        if document.get("rating_scope") == "global"
        else 0.0
    )
    return (
        document.get("component_unit") == "elo_points"
        and _close(document["roster_strength_component"], expected_roster, tolerance)
        and _close(document["lineup_synergy_component"], expected_synergy, tolerance)
        and _close(document["league_rating_component"], expected_league, tolerance)
    )


def _registered_search(search: Mapping[str, Any]) -> bool:
    policy = SEMANTIC_ARTIFACTS["search_policies"].get(search.get("policy_id"))
    if not isinstance(policy, Mapping):
        return False
    if (
        search.get("method") != policy["method"]
        or search.get("exact") is not policy["exact"]
        or search.get("coverage", {}).get("metric_id")
        != policy["coverage_metric_id"]
    ):
        return False
    coverage = float(search["coverage"]["value"])
    if policy["exact"]:
        return _close(coverage, 1.0) and "approximation_error_bound" not in search
    bound = search.get("approximation_error_bound")
    if not isinstance(bound, Mapping) or bound.get("method_id") != policy["bound_method_id"]:
        return False
    expected_bound = max(
        0.0,
        (1.0 - coverage) * float(search["temperature"]) + 0.0075,
    )
    return _close(bound["value_logit"], expected_bound)


def _tier_order_key(
    entry: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[Any, ...]:
    values: list[Any] = []
    for key in policy["keys"]:
        field = str(key["field"])
        value: Any = entry
        for part in field.split("."):
            value = value[part]
        if key["direction"] == "descending" and isinstance(value, (int, float)):
            value = -float(value)
        values.append(value)
    return tuple(values)


def _tier_rank_valid(document: Mapping[str, Any]) -> bool:
    policy = SEMANTIC_ARTIFACTS["tie_break_policies"].get(
        document.get("ranking_tie_break_policy_id")
    )
    if not isinstance(policy, Mapping):
        return False
    expected = sorted(
        document["entries"],
        key=lambda item: (
            -float(item["tier_value_pp"]),
            *_tier_order_key(item, policy),
        ),
    )
    return (
        [item["champion_id"] for item in document["entries"]]
        == [item["champion_id"] for item in expected]
        and [item["rank"] for item in document["entries"]]
        == list(range(1, len(document["entries"]) + 1))
    )


def _evidence_methods_valid(evidence: Mapping[str, Any]) -> bool:
    forbidden = (
        "game_count",
        "games",
        "play_rate",
        "pick_rate",
        "popularity",
        "volume",
        "sample_count",
        "overall",
        "confidence",
        "evidence_score",
    )
    serialized_keys: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                serialized_keys.append(str(key).lower())
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(evidence)
    if any(any(token in key for token in forbidden) for key in serialized_keys):
        return False
    methods = SEMANTIC_ARTIFACTS["evidence_methods"]
    return (
        evidence.get("posterior_displacement", {}).get("method_id")
        in methods["displacement"]
        and evidence.get("precision", {}).get("method_id")
        in methods["precision"]
        and str(
            evidence.get("source_context_coverage", {}).get("coverage_spec_id", "")
        ).startswith(methods["source_context_coverage_prefix"])
    )


def _reliability_mapping_valid(reliability: Mapping[str, Any]) -> bool:
    mapping = SEMANTIC_ARTIFACTS["reliability_maps"].get(
        reliability.get("stratum_mapping_sha256")
    )
    if not isinstance(mapping, Mapping):
        return False
    if reliability.get("stratum_match_status") == "unmatched":
        return (
            reliability.get("validation_stratum_id")
            == mapping["default"]["validation_stratum_id"]
            and reliability.get("label") == mapping["default"]["label"]
        )
    stratum_id = reliability.get("validation_stratum_id")
    return mapping["strata"].get(stratum_id) == reliability.get("label")


def _terminal_delegation_valid(
    document: Mapping[str, Any],
    *,
    terminal_output_validator: Callable[
        [str, Mapping[str, Any]], Any
    ] | None = None,
) -> bool:
    delegation = document.get("terminal_delegation", {})
    registration = SEMANTIC_ARTIFACTS["terminal_outputs"].get(
        delegation.get("terminal_prediction_id")
    )
    if not isinstance(registration, Mapping):
        return False
    locator = Path(str(registration["locator"]))
    if (
        not locator.is_file()
        or _raw_sha256(locator) != registration["raw_sha256"]
        or delegation.get("terminal_output_sha256") != registration["raw_sha256"]
    ):
        return False
    terminal = json.loads(locator.read_text(encoding="utf-8"))
    try:
        if terminal_output_validator is None:
            validate_output_payload("draft_score", terminal)
        else:
            terminal_output_validator("draft_score", terminal)
    except ValidationFailure:
        return False
    expected = {
        "delegation_method": "canonical_terminal_draft_score",
        "terminal_prediction_id": terminal["provenance"]["prediction_id"],
        "terminal_model_version": terminal["model_version"],
        "terminal_output_sha256": registration["raw_sha256"],
        "score_a": terminal["score_a"],
        "score_b": terminal["score_b"],
        "standardized_map_win_probability_a": terminal[
            "standardized_map_win_probability_a"
        ],
        "interval_95": terminal["interval_95"],
    }
    return dict(delegation) == expected


def _registry_identity_integrity(
    output: str,
    document: Mapping[str, Any],
) -> tuple[bool, str]:
    registries = SEMANTIC_ARTIFACTS["registries"]
    if document.get("schema_version") != "2.0.0":
        return _fail("schema_version is not the anchored schema version")
    season_id = document.get("season_id")
    if (
        not isinstance(season_id, str)
        or season_id not in registries["taxonomy_ids"]
        or not season_id.rsplit("-", 1)[-1].isdigit()
        or document.get("calendar_year") != int(season_id.rsplit("-", 1)[-1])
    ):
        return _fail("season_id/calendar_year is not registered or derived")

    allowed_by_key: dict[str, set[str]] = {}
    fixture_documents: list[Mapping[str, Any]] = []
    for name in EXAMPLE_FILES.values():
        fixture_documents.append(
            json.loads((CONTRACT_ROOT / name).read_text(encoding="utf-8"))
        )
    partial_schema = json.loads(
        (CONTRACT_ROOT / "partial-draft-state.schema.json").read_text(
            encoding="utf-8"
        )
    )
    fixture_documents.extend(partial_schema.get("examples", []))

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if isinstance(child, str) and (
                    key.endswith(("_id", "_sha256", "_version"))
                    or key in {"role", "schema_version", "patch_id"}
                ):
                    allowed_by_key.setdefault(str(key), set()).add(child)
                elif key.endswith("_ids") and isinstance(child, list):
                    allowed_by_key.setdefault(str(key), set()).update(
                        str(item) for item in child if isinstance(item, str)
                    )
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    for fixture in fixture_documents:
        collect(fixture)
    allowed_by_key["source_ids"] = (
        allowed_by_key.get("source_id", set())
        | allowed_by_key.get("input_id", set())
    )

    dynamic_identity_keys = {"output_sha256", "prediction_id"}
    errors: list[str] = []

    def validate_identifiers(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if (
                    isinstance(child, str)
                    and key not in dynamic_identity_keys
                    and (
                        key.endswith(("_id", "_sha256", "_version"))
                        or key in {"role", "schema_version", "patch_id"}
                    )
                    and child not in allowed_by_key.get(str(key), set())
                ):
                    errors.append(f"{key} is not registered")
                elif key.endswith("_ids") and isinstance(child, list):
                    if any(
                        isinstance(item, str)
                        and item not in allowed_by_key.get(str(key), set())
                        for item in child
                    ):
                        errors.append(f"{key} contains an unregistered identity")
                validate_identifiers(child)
        elif isinstance(value, list):
            for child in value:
                validate_identifiers(child)

    validate_identifiers(document)
    if errors:
        return _fail(errors[0])

    state = document.get("state", document)
    if "protocol_id" in state and state["protocol_id"] not in SEMANTIC_ARTIFACTS["protocols"]:
        return _fail("protocol_id is not registered")
    if "competition_scope_id" in state and not _scope_valid(
        document, partial="state" in document
    ):
        return _fail("competition scope/taxonomy ID is not registered")
    champion_ids = set(registries["champion_ids"])
    actions = state.get("actions", [])
    if any(item.get("champion_id") not in champion_ids for item in actions):
        return _fail("champion_id is not registered")
    if output == "tier_list" and any(
        item.get("champion_id") not in champion_ids
        for item in document.get("entries", [])
    ):
        return _fail("tier-list champion_id is not registered")
    roles = set(registries["roles"])
    role_values = [
        document.get("role"),
        *[item.get("role") for item in document.get("roster", [])],
        *[item.get("role") for item in state.get("final_assignments", [])],
    ]
    if any(value is not None and value not in roles for value in role_values):
        return _fail("role ID is not registered")
    league_values = [
        document.get("league_id"),
    ]
    if any(
        value is not None and value not in registries["league_ids"]
        for value in league_values
    ):
        return _fail("league ID is not registered")

    actions = {
        str(item["action_id"]): item
        for item in state.get("actions", [])
        if isinstance(item, Mapping) and "action_id" in item
    }

    def linked_items(value: Any) -> list[Mapping[str, Any]]:
        found: list[Mapping[str, Any]] = []
        if isinstance(value, Mapping):
            if "action_id" in value and (
                "champion_id" in value or "allowed_roles" in value
            ):
                found.append(value)
            for child in value.values():
                found.extend(linked_items(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(linked_items(child))
        return found

    for item in linked_items(state):
        action_id = str(item["action_id"])
        action = actions.get(action_id)
        if action is None:
            return _fail("assignment/constraint references an unknown action")
        if (
            "champion_id" in item
            and action.get("champion_id") != item.get("champion_id")
        ):
            return _fail("assignment/constraint champion differs from pick action")
    return _pass()


def canonical_unsigned_output_sha256(document: Mapping[str, Any]) -> str:
    unsigned = deepcopy(dict(document))
    provenance = dict(unsigned.get("provenance", {}))
    provenance.pop("output_sha256", None)
    provenance.pop("prediction_id", None)
    unsigned["provenance"] = provenance
    return canonical_sha256(unsigned)


def _seal_dynamic_identity(document: dict[str, Any]) -> None:
    digest = canonical_unsigned_output_sha256(document)
    document["provenance"]["output_sha256"] = digest
    document["provenance"]["prediction_id"] = (
        f"scryglass:prediction:sha256:{digest}"
    )


def _canonical_identity_pairs(
    contract_root: Path,
) -> set[tuple[str, str, str]]:
    pairs: set[tuple[str, str, str]] = set()
    for name in EXAMPLE_FILES.values():
        document = json.loads((contract_root / name).read_text(encoding="utf-8"))
        pairs.add(
            (
                canonical_unsigned_output_sha256(document),
                str(document["provenance"]["output_sha256"]),
                str(document["provenance"]["prediction_id"]),
            )
        )
    partial_schema = json.loads(
        (contract_root / "partial-draft-state.schema.json").read_text(
            encoding="utf-8"
        )
    )
    for document in partial_schema.get("examples", []):
        pairs.add(
            (
                canonical_unsigned_output_sha256(document),
                str(document["provenance"]["output_sha256"]),
                str(document["provenance"]["prediction_id"]),
            )
        )
    return pairs


def _registered_unavailable_fixture(
    output: str,
    required_status: str,
    contract_root: Path = CONTRACT_ROOT,
) -> dict[str, Any]:
    schema = json.loads(
        (contract_root / OUTPUT_SCHEMAS[output]).read_text(encoding="utf-8")
    )
    canonical = json.loads(
        (contract_root / EXAMPLE_FILES[output]).read_text(encoding="utf-8")
    )
    document = {
        key: deepcopy(canonical[key])
        for key in schema["required"]
    }
    document["status"] = "unavailable"
    document["error"] = {
        "code": {
            "missing": "missing_required_input",
            "stale": "stale_context",
            "conflict": "patch_conflict",
        }[required_status],
        "message": "Typed unavailable fixture.",
        "retryable": True,
        "missing_fields": (
            ["required_input"] if required_status == "missing" else []
        ),
        "stale_fields": (
            ["required_input"] if required_status == "stale" else []
        ),
    }
    provenance = document["provenance"]
    provenance["required_input_status"] = required_status
    if required_status == "stale":
        provenance["freshness_checks"][0].update(
            {
                "source_updated_at": "2026-07-20T00:00:00Z",
                "limit_seconds": 60,
                "fresh": False,
            }
        )
        provenance["input_conflicts"] = []
    elif required_status == "missing":
        provenance["freshness_checks"] = []
        provenance["input_conflicts"] = []
    else:
        provenance["freshness_checks"] = []
        provenance["input_conflicts"] = [
            {
                "input_id": "scryglass:source:analysis-input",
                "conflict_type": "patch",
                "source_ids": [
                    "scryglass:source:analysis-input",
                    "scryglass:source:oe-example",
                ],
                "detected_at": "2026-07-27T17:59:00Z",
            }
        ]
    _seal_dynamic_identity(document)
    return document


def _contract_fixture_identity_pairs(
    contract_root: Path = CONTRACT_ROOT,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    anchors: ContractValidationAnchors | None = None,
) -> set[tuple[str, str, str]]:
    resolved = anchors or default_contract_validation_anchors()
    raw = (repository_root / resolved.contract_fixture_authority_path).read_bytes()
    authority = json.loads(raw)
    if (
        hashlib.sha256(raw).hexdigest()
        != resolved.contract_fixture_authority_raw_sha256
        or canonical_sha256(authority)
        != resolved.contract_fixture_authority_object_sha256
        or authority["contract_tree_sha256"] != resolved.contract_tree_sha256
    ):
        raise ValidationFailure("contract fixture authority is missing or stale")
    pairs = _canonical_identity_pairs(contract_root)
    derived_unsigned: dict[str, list[dict[str, str]]] = {}
    for output in OUTPUT_SCHEMAS:
        output_entries: list[dict[str, str]] = []
        canonical = json.loads(
            (contract_root / EXAMPLE_FILES[output]).read_text(encoding="utf-8")
        )
        output_entries.append(
            {
                "status": str(canonical["status"]),
                "sha256": canonical_unsigned_output_sha256(canonical),
            }
        )
        if output == "partial_draft_state":
            schema = json.loads(
                (contract_root / OUTPUT_SCHEMAS[output]).read_text(
                    encoding="utf-8"
                )
            )
            for document in schema.get("examples", []):
                output_entries.append(
                    {
                        "status": str(document["status"]),
                        "sha256": canonical_unsigned_output_sha256(document),
                    }
                )
        for status in ("missing", "stale", "conflict"):
            document = _registered_unavailable_fixture(
                output, status, contract_root
            )
            output_entries.append(
                {
                    "status": f"unavailable:{status}",
                    "sha256": canonical_unsigned_output_sha256(document),
                }
            )
            pairs.add(
                (
                    canonical_unsigned_output_sha256(document),
                    str(document["provenance"]["output_sha256"]),
                    str(document["provenance"]["prediction_id"]),
                )
            )
        derived_unsigned[output] = output_entries
    if authority["allowed_unsigned_output_sha256"] != derived_unsigned:
        raise ValidationFailure("contract fixture identity registry is stale")
    return pairs


def _provenance_integrity(
    document: Mapping[str, Any],
    *,
    canonical_identity_pairs: set[tuple[str, str, str]] | None = None,
) -> tuple[bool, str]:
    provenance = document.get("provenance")
    if not isinstance(provenance, Mapping):
        return _fail("provenance is missing")
    for field in ("schema_version", "model_version", "as_of"):
        if document.get(field) != provenance.get(field):
            return _fail(f"top-level and provenance {field} disagree")
    if document.get("lineage") != provenance.get("lineage"):
        return _fail("top-level and provenance lineage disagree")
    status = document.get("status")
    required_status = provenance.get("required_input_status")
    conflicts = provenance.get("input_conflicts", [])
    checks = provenance.get("freshness_checks", [])
    as_of = _parse_timestamp(str(document["as_of"]))
    registered_freshness: set[tuple[str, str, int]] = set()
    for name in EXAMPLE_FILES.values():
        fixture = json.loads((CONTRACT_ROOT / name).read_text(encoding="utf-8"))
        for check in fixture["provenance"].get("freshness_checks", []):
            registered_freshness.add(
                (
                    str(check["input_id"]),
                    str(check["source_updated_at"]),
                    int(check["limit_seconds"]),
                )
            )
    partial_schema = json.loads(
        (CONTRACT_ROOT / "partial-draft-state.schema.json").read_text(
            encoding="utf-8"
        )
    )
    for fixture in partial_schema.get("examples", []):
        for check in fixture["provenance"].get("freshness_checks", []):
            registered_freshness.add(
                (
                    str(check["input_id"]),
                    str(check["source_updated_at"]),
                    int(check["limit_seconds"]),
                )
            )
    registered_freshness.update(
        {
            ("scryglass:source:analysis-input", "2026-07-20T00:00:00Z", 60),
            ("scryglass:source:oe-example", "2026-07-20T00:00:00Z", 60),
        }
    )
    computed_fresh: list[bool] = []
    for check in checks:
        if (
            str(check["input_id"]),
            str(check["source_updated_at"]),
            int(check["limit_seconds"]),
        ) not in registered_freshness:
            return _fail("freshness evidence is not registered to a typed source")
        updated = _parse_timestamp(str(check["source_updated_at"]))
        limit = int(check["limit_seconds"])
        fresh = (
            updated <= as_of
            and (as_of - updated).total_seconds() <= limit
        )
        computed_fresh.append(fresh)
        if check.get("fresh") is not fresh:
            return _fail("declared freshness disagrees with source_updated_at/as_of/SLO")
    if status in {"ok", "research_only"}:
        if required_status != "complete" or conflicts or not checks or not all(computed_fresh):
            return _fail("successful output provenance is incomplete, stale, or conflicted")
    elif status == "unavailable":
        if required_status not in {"missing", "stale", "conflict"}:
            return _fail("unavailable output lacks typed missing/stale/conflict provenance")
        if "reliability" in document:
            return _fail("unavailable output must not contain Reliability")
    if provenance.get("immutable") is not True:
        return _fail("output is not immutable")
    computed = canonical_unsigned_output_sha256(document)
    declared = str(provenance.get("output_sha256"))
    prediction_id = str(provenance.get("prediction_id"))
    registered = canonical_identity_pairs or set()
    if (computed, declared, prediction_id) not in registered and (
        declared != computed
        or prediction_id != f"scryglass:prediction:sha256:{computed}"
    ):
        return _fail("output_sha256/prediction identity does not match unsigned output")
    return _pass()


def _provenance_inputs_ready(document: Mapping[str, Any]) -> bool:
    provenance = document.get("provenance", {})
    as_of = _parse_timestamp(str(document["as_of"]))
    checks = provenance.get("freshness_checks", [])
    return (
        provenance.get("required_input_status") == "complete"
        and not provenance.get("input_conflicts")
        and bool(checks)
        and all(
            _parse_timestamp(str(check["source_updated_at"])) <= as_of
            and (
                as_of - _parse_timestamp(str(check["source_updated_at"]))
            ).total_seconds()
            <= int(check["limit_seconds"])
            for check in checks
        )
    )


InvariantFn = Callable[[Mapping[str, Any]], tuple[bool, str]]
OutputValidatorFn = Callable[[str, Mapping[str, Any]], Any]


def _build_invariant_dispatch(
    *, terminal_output_validator: OutputValidatorFn | None = None
) -> dict[str, InvariantFn]:
    def score_identity(score: str, probability: str) -> InvariantFn:
        return lambda d: (
            _not_applicable()
            if not _status_ok(d) or score not in d
            else (
                _pass()
                if _close(d[score], 100.0 * d[probability])
                else _fail(f"{score} does not equal 100 times probability")
            )
        )

    def complement(score_a: str, score_b: str) -> InvariantFn:
        return lambda d: (
            _not_applicable()
            if not _status_ok(d) or score_a not in d
            else (
                _pass()
                if _close(d[score_b], 100.0 - d[score_a])
                else _fail(f"{score_b} does not complement {score_a}")
            )
        )

    dispatch: dict[str, InvariantFn] = {
        "player_interval_contains_posterior_mean": lambda d: _interval_contains(d, "posterior_mean"),
        "player_settled_width_matches_interval": _settled_width,
        "team_exact_five_unique_players": lambda d: (
            _not_applicable()
            if not _status_ok(d)
            else (
                _pass()
                if len(d["roster"]) == 5
                and len({item["player_id"] for item in d["roster"]}) == 5
                and {item["role"] for item in d["roster"]}
                == {"top", "jungle", "mid", "bot", "support"}
                else _fail("team roster is not one unique player per role")
            )
        ),
        "team_rating_component_identity": lambda d: (
            _not_applicable()
            if not _status_ok(d)
            else (
                _pass()
                if _close(
                    d["posterior_mean"],
                    d["rating_display"]["anchor"]
                    + d["roster_strength_component"]
                    + d["league_rating_component"]
                    + d["lineup_synergy_component"],
                )
                else _fail("team rating components do not reconcile")
            )
        ),
        "team_component_estimand_units": lambda d: (
            _not_applicable()
            if not _status_ok(d)
            else (
                _pass()
                if _team_component_math(d)
                else _fail("team component units or scope semantics are invalid")
            )
        ),
        "team_interval_contains_posterior_mean": lambda d: _interval_contains(d, "posterior_mean"),
        "team_settled_width_matches_interval": _settled_width,
        "terminal_score_probability_identity": score_identity(
            "score_a", "standardized_map_win_probability_a"
        ),
        "terminal_score_complement_identity": complement("score_a", "score_b"),
        "terminal_ledger_reconciliation": lambda d: (
            _not_applicable()
            if not _status_ok(d)
            else (
                _pass()
                if abs(sum(item["signed_logit"] for item in d["ledger"]) - d["ledger_logit_sum"])
                <= d["reconciliation_tolerance"]
                and abs(d["ledger_logit_sum"] - d["uncalibrated_logit_a"])
                <= d["reconciliation_tolerance"]
                else _fail("terminal ledger does not reconcile")
            )
        ),
        "terminal_interval_contains_probability": lambda d: _interval_contains(
            d, "standardized_map_win_probability_a"
        ),
        "terminal_action_slots_unique_and_ordered": lambda d: (
            _pass()
            if _actions_ordered(d["actions"]) and _protocol_actions_legal(d)
            else _fail("terminal action slots or protocol legality are invalid")
        ),
        "terminal_side_mapping_source_available": lambda d: (
            _pass() if _source_available(d, partial=False) else _fail("terminal source is unavailable at as_of")
        ),
        "terminal_competition_scope_taxonomy": lambda d: (
            _pass() if _scope_valid(d) else _fail("terminal competition scope is invalid")
        ),
        "terminal_role_revision_chain_append_only": lambda d: (
            _pass()
            if _revision_chain(d.get("role_constraint_revisions", []))
            else _fail("terminal role revision chain is not append-only")
        ),
        "terminal_assignment_revision_chain_append_only": lambda d: (
            _pass()
            if _revision_chain(d.get("assignment_revisions", []))
            else _fail("terminal assignment revision chain is not append-only")
        ),
        "terminal_assignments_match_latest_revisions": lambda d: (
            _pass()
            if _assignments_match_latest(
                d.get("final_assignments", []), d.get("assignment_revisions", [])
            )
            else _fail("terminal assignments do not match latest revisions")
        ),
        "contextual_neutral_comparison_identity": lambda d: (
            _not_applicable()
            if not _status_ok(d) or d.get("identity_mode") != "contextual"
            else (
                _pass()
                if _close(
                    d["neutral_comparison"]["score_a"],
                    100.0
                    * d["neutral_comparison"]["standardized_map_win_probability_a"],
                )
                and _close(
                    d["neutral_comparison"]["score_b"],
                    100.0 - d["neutral_comparison"]["score_a"],
                )
                else _fail("neutral comparison does not reconcile")
            )
        ),
        "partial_value_decomposition": lambda d: (
            _not_applicable()
            if not _status_ok(d) or d["state"]["terminal"]
            else (
                _pass()
                if _close(
                    d["strategic_value_logit_a"],
                    d["committed_value_logit_a"]
                    + d["strategic_response_adjustment_logit_a"],
                )
                else _fail("partial strategic value does not decompose")
            )
        ),
        "partial_score_probability_identity": score_identity(
            "partial_score_a", "standardized_map_win_probability_a"
        ),
        "partial_score_complement_identity": complement(
            "partial_score_a", "partial_score_b"
        ),
        "partial_interval_contains_probability": lambda d: (
            _not_applicable()
            if not _status_ok(d) or d["state"]["terminal"]
            else _interval_contains(d, "standardized_map_win_probability_a")
        ),
        "partial_current_constraints_cover_picks": lambda d: (
            _not_applicable()
            if d.get("status") not in {"ok", "research_only"}
            else (
                _pass()
                if {
                    item["action_id"]
                    for item in d["state"]["current_role_constraints"]
                }
                == {
                    item["action_id"]
                    for item in d["state"]["actions"]
                    if item["kind"] == "pick"
                }
                else _fail("current role constraints do not cover all picks")
            )
        ),
        "partial_current_constraints_are_latest_revision": lambda d: (
            _not_applicable()
            if d.get("status") not in {"ok", "research_only"}
            else (
                _pass()
                if _constraints_match_latest(d["state"])
                else _fail("current role constraints are not latest")
            )
        ),
        "partial_role_revision_chain_append_only": lambda d: (
            _not_applicable()
            if d.get("status") not in {"ok", "research_only"}
            else (
                _pass()
                if _revision_chain(d["state"].get("role_constraint_revisions", []))
                else _fail("partial role revision chain is not append-only")
            )
        ),
        "partial_assignment_revision_chain_append_only": lambda d: (
            _not_applicable()
            if d.get("status") not in {"ok", "research_only"}
            else (
                _pass()
                if _revision_chain(d["state"].get("assignment_revisions", []))
                else _fail("partial assignment revision chain is not append-only")
            )
        ),
        "partial_terminal_assignments_match_latest_revisions": lambda d: (
            _not_applicable()
            if not _status_ok(d) or not d["state"]["terminal"]
            else (
                _pass()
                if _assignments_match_latest(
                    d["state"]["final_assignments"],
                    d["state"].get("assignment_revisions", []),
                )
                else _fail("partial terminal assignments do not match latest revisions")
            )
        ),
        "partial_source_available": lambda d: (
            _not_applicable()
            if d.get("status") not in {"ok", "research_only"}
            else (
                _pass()
                if _source_available(d, partial=True)
                else _fail("partial source is unavailable at as_of")
            )
        ),
        "partial_action_slots_unique_and_protocol_legal": lambda d: (
            _not_applicable()
            if d.get("status") not in {"ok", "research_only"}
            else (
                _pass()
                if _actions_ordered(d["state"]["actions"])
                and _protocol_actions_legal(d["state"])
                else _fail("partial action slots or protocol legality are invalid")
            )
        ),
        "partial_competition_scope_taxonomy": lambda d: (
            _not_applicable()
            if d.get("status") not in {"ok", "research_only"}
            else (
                _pass()
                if _scope_valid(d, partial=True)
                else _fail("partial competition scope is invalid")
            )
        ),
        "partial_neutral_comparison_identity": lambda d: (
            _not_applicable()
            if not _status_ok(d) or d.get("identity_mode") != "contextual"
            else (
                _pass()
                if _close(
                    d["neutral_comparison"]["partial_score_a"],
                    100.0
                    * d["neutral_comparison"][
                        "standardized_map_win_probability_a"
                    ],
                )
                and _close(
                    d["neutral_comparison"]["partial_score_b"],
                    100.0 - d["neutral_comparison"]["partial_score_a"],
                )
                else _fail("partial neutral comparison does not reconcile")
            )
        ),
        "terminal_delegation_score_identity": lambda d: (
            _not_applicable()
            if not _status_ok(d) or not d["state"]["terminal"]
            else (
                _pass()
                if _close(
                    d["terminal_delegation"]["score_a"],
                    100.0
                    * d["terminal_delegation"][
                        "standardized_map_win_probability_a"
                    ],
                )
                and _close(
                    d["terminal_delegation"]["score_b"],
                    100.0 - d["terminal_delegation"]["score_a"],
                )
                else _fail("terminal delegation score does not reconcile")
            )
        ),
        "terminal_delegation_exact_canonical_output": lambda d: (
            _not_applicable()
            if not _status_ok(d) or not d["state"]["terminal"]
            else (
                _pass()
                if _terminal_delegation_valid(
                    d, terminal_output_validator=terminal_output_validator
                )
                else _fail("terminal delegation does not resolve exact canonical output")
            )
        ),
        "archetype_extrapolation_ordinal_groups": lambda d: (
            _not_applicable()
            if d.get("status") != "research_only"
            else (
                _pass()
                if (
                    (orders := [
                        item["group_order"]
                        for item in d["archetype_extrapolation"][
                            "recommendation_groups"
                        ]
                    ])
                    == sorted(orders)
                    and len(orders) == len(set(orders))
                    and all(
                        item["within_group_ordered"] is False
                        for item in d["archetype_extrapolation"][
                            "recommendation_groups"
                        ]
                    )
                )
                else _fail("archetype recommendation groups are not ordinal")
            )
        ),
        "archetype_extrapolation_support_gap_matches_fallback": lambda d: (
            _not_applicable()
            if d.get("status") != "research_only"
            else (
                _pass()
                if d["archetype_extrapolation"]["support_gap"][
                    "verified_eligible_appearance_count"
                ]
                == 0
                and d["archetype_extrapolation"]["fallback"]["material"] is True
                and bool(
                    d["archetype_extrapolation"]["fallback"]["fallback_levels"]
                )
                and d["archetype_extrapolation"]["uncertainty"][
                    "quantitative_interval_available"
                ]
                is False
                and d["archetype_extrapolation"]["support_gap"][
                    "competition_scope_id"
                ]
                == d["state"]["competition_scope_id"]
                and d["archetype_extrapolation"]["support_gap"][
                    "competition_scope_kind"
                ]
                == d["state"]["competition_scope_kind"]
                else _fail("archetype support gap and fallback disagree")
            )
        ),
        "archetype_extrapolation_noncanonical_boundary": lambda d: (
            _not_applicable()
            if d.get("status") != "research_only"
            else (
                _pass()
                if not {
                    "partial_score_a",
                    "partial_score_b",
                    "standardized_map_win_probability_a",
                    "interval_95",
                    "reliability",
                    "recommendations",
                    "neutral_comparison",
                    "terminal_delegation",
                }
                & set(d)
                and not {
                    "probability_transform",
                    "partial_probability_calibration",
                    "calibration_id",
                }
                & set(d["provenance"])
                and all(
                    "canonical" not in str(action["action_id"]).lower()
                    for group in d["archetype_extrapolation"][
                        "recommendation_groups"
                    ]
                    for action in group["actions"]
                )
                else _fail("research-only output crossed the canonical boundary")
            )
        ),
        "contextual_archetype_exact_context_fresh": lambda d: (
            _not_applicable()
            if d.get("status") != "research_only"
            or d.get("identity_mode") != "contextual"
            else (
                _pass()
                if all(
                    d["context"].get(key)
                    for key in (
                        "roster_a_snapshot_id",
                        "roster_b_snapshot_id",
                        "policy_a_snapshot_id",
                        "policy_b_snapshot_id",
                    )
                )
                and all(
                    d["context"].get(key)
                    in SEMANTIC_ARTIFACTS["registries"]["context_snapshot_ids"]
                    for key in (
                        "roster_a_snapshot_id",
                        "roster_b_snapshot_id",
                        "policy_a_snapshot_id",
                        "policy_b_snapshot_id",
                    )
                )
                and all(
                    d["context"].get(key) is True
                    for key in (
                        "roster_a_fresh",
                        "roster_b_fresh",
                        "policy_a_fresh",
                        "policy_b_fresh",
                    )
                )
                and d["archetype_extrapolation"]["fallback"]["basis"]
                == "registered_team_player_archetype_fit_prior"
                and d["archetype_extrapolation"]["fallback"][
                    "unsupported_exact_residuals_included"
                ]
                is False
                and _provenance_inputs_ready(d)
                else _fail("contextual archetype inputs are incomplete or stale")
            )
        ),
        "partial_approximate_search_reports_coverage_and_bound": lambda d: (
            _not_applicable()
            if "search" not in d or d["search"]["exact"]
            else (
                _pass()
                if _registered_search(d["search"])
                else _fail("approximate search policy/bound does not recompute")
            )
        ),
        "partial_exact_search_full_coverage": lambda d: (
            _not_applicable()
            if "search" not in d or not d["search"]["exact"]
            else (
                _pass()
                if _registered_search(d["search"])
                else _fail("exact search registration/coverage is invalid")
            )
        ),
        "tier_list_regional_league_scope_only": lambda d: (
            _not_applicable()
            if not _status_ok(d)
            else (
                _pass()
                if d["competition_scope_kind"] == "regional_league"
                and _scope_valid(d)
                else _fail("tier list scope is not a regional league")
            )
        ),
        "tier_value_identity": lambda d: (
            _not_applicable()
            if not _status_ok(d)
            else (
                _pass()
                if all(
                    _close(
                        item["tier_value_pp"],
                        item["incremental_value_pp"]
                        - d["counterability_weight"] * item["counterability_pp"],
                    )
                    for item in d["entries"]
                )
                else _fail("tier value arithmetic does not reconcile")
            )
        ),
        "tier_single_manifested_counterability_weight": lambda d: (
            _not_applicable()
            if not _status_ok(d)
            else (
                _pass()
                if d["counterability_policy_id"]
                in SEMANTIC_ARTIFACTS["counterability_policies"]
                and _close(
                    d["counterability_weight"],
                    SEMANTIC_ARTIFACTS["counterability_policies"][
                        d["counterability_policy_id"]
                    ],
                )
                else _fail("counterability weight is not manifested")
            )
        ),
        "tier_rank_and_champion_unique": lambda d: (
            _not_applicable()
            if not _status_ok(d)
            else (
                _pass()
                if len({item["rank"] for item in d["entries"]}) == len(d["entries"])
                and len({item["champion_id"] for item in d["entries"]})
                == len(d["entries"])
                else _fail("tier rank or champion identities are duplicated")
            )
        ),
        "tier_rank_matches_value_and_tie_break": lambda d: (
            _not_applicable()
            if not _status_ok(d)
            else (
                _pass()
                if _tier_rank_valid(d)
                else _fail("tier rank/tie-break policy does not deterministically order entries")
            )
        ),
        "tier_interval_contains_value": lambda d: (
            _not_applicable()
            if not _status_ok(d)
            else (
                _pass()
                if all(
                    item["interval_95"]["lower"]
                    <= item["tier_value_pp"]
                    <= item["interval_95"]["upper"]
                    for item in d["entries"]
                )
                else _fail("tier interval does not contain value")
            )
        ),
        "interval_probability_order": lambda d: (
            _pass()
            if d["interval_95"]["lower"] <= d["interval_95"]["upper"]
            else _fail("probability interval is reversed")
        ),
        "interval_real_order": lambda d: (
            _pass()
            if d["interval_95"]["lower"] <= d["interval_95"]["upper"]
            else _fail("real interval is reversed")
        ),
        "evidence_concepts_remain_separate": lambda d: (
            _pass()
            if set(d["evidence"])
            == {
                "posterior_displacement",
                "precision",
                "source_context_coverage",
            }
            and len(
                {
                    d["evidence"]["posterior_displacement"]["method_id"],
                    d["evidence"]["precision"]["method_id"],
                    d["evidence"]["source_context_coverage"]["coverage_spec_id"],
                }
            )
            == 3
            else _fail("evidence concepts are aggregated or missing")
        ),
        "evidence_not_volume_proxy": lambda d: (
            _pass()
            if _evidence_methods_valid(d["evidence"])
            else _fail("evidence uses a volume proxy")
        ),
        "reliability_log_loss_skill_identity": lambda d: (
            _pass()
            if _close(
                d["reliability"]["log_loss_skill"],
                d["reliability"]["baseline_log_loss"]
                - d["reliability"]["log_loss"],
            )
            else _fail("log-loss skill does not reconcile")
        ),
        "reliability_brier_skill_identity": lambda d: (
            _pass()
            if _close(
                d["reliability"]["brier_skill"],
                d["reliability"]["baseline_brier_score"]
                - d["reliability"]["brier_score"],
            )
            else _fail("Brier skill does not reconcile")
        ),
        "reliability_total_stratum_mapping": lambda d: (
            _pass()
            if _reliability_mapping_valid(d["reliability"])
            else _fail("reliability stratum mapping is incomplete")
        ),
        "settled_interval_width_at_resolution": lambda d: (
            _not_applicable()
            if d["settled"]["value"] is not True
            else (
                _pass()
                if d["settled"]["interval_width"]
                <= 2 * d["settled"]["rating_resolution"]
                else _fail("settled interval exceeds resolution")
            )
        ),
        "settled_stability_change_at_resolution": lambda d: (
            _not_applicable()
            if d["settled"]["value"] is not True
            else (
                _pass()
                if d["settled"]["stability_change"]
                <= d["settled"]["rating_resolution"]
                else _fail("settled stability exceeds resolution")
            )
        ),
        "draft_protocol_mapping_bijective_game_side": lambda d: (
            _pass()
            if d["side_mapping"]["side_a_game_side"]
            != d["side_mapping"]["side_b_game_side"]
            else _fail("game-side mapping is not bijective")
        ),
        "draft_protocol_mapping_bijective_order": lambda d: (
            _pass()
            if d["side_mapping"]["side_a_draft_order"]
            != d["side_mapping"]["side_b_draft_order"]
            else _fail("draft-order mapping is not bijective")
        ),
        "terminal_assignment_unique_champions": lambda d: (
            _pass()
            if len({item["champion_id"] for item in d["final_assignments"]}) == 10
            else _fail("terminal assignments contain duplicate champions")
        ),
        "terminal_assignment_matches_pick_actions": lambda d: (
            _pass()
            if {item["action_id"] for item in d["final_assignments"]}
            == {
                item["action_id"]
                for item in d["actions"]
                if item["kind"] == "pick"
            }
            else _fail("terminal assignments do not match pick actions")
        ),
        "forecast_created_before_event": lambda d: (
            _not_applicable()
            if d["provenance"]["mode"] != "forecast"
            else (
                _pass()
                if _parse_timestamp(d["provenance"]["created_at"])
                < _parse_timestamp(d["provenance"]["event_start"])
                and _parse_timestamp(d["provenance"]["as_of"])
                < _parse_timestamp(d["provenance"]["event_start"])
                else _fail("forecast was not sealed before event")
            )
        ),
        "forecast_simulation_replays_historical_availability": lambda d: (
            _not_applicable()
            if d["provenance"]["mode"] != "forecast_simulation"
            else (
                _pass()
                if _parse_timestamp(d["provenance"]["as_of"])
                < _parse_timestamp(d["provenance"]["event_start"])
                and d["provenance"]["availability_replayed"] is True
                else _fail("forecast simulation did not replay availability")
            )
        ),
        "hindsight_created_after_event": lambda d: (
            _not_applicable()
            if d["provenance"]["mode"] != "hindsight"
            else (
                _pass()
                if _parse_timestamp(d["provenance"]["created_at"])
                >= _parse_timestamp(d["provenance"]["event_start"])
                else _fail("hindsight was created before event")
            )
        ),
        "partial_transform_identity_matches_proof": lambda d: (
            _not_applicable()
            if "partial_probability_calibration" not in d["provenance"]
            or "probability_transform" not in d["provenance"]
            else (
                _pass()
                if d["provenance"]["partial_probability_calibration"][
                    "transform_sha256"
                ]
                == d["provenance"]["probability_transform"]["transform_sha256"]
                else _fail("partial transform identity does not match proof")
            )
        ),
    }
    return dispatch


INVARIANT_DISPATCH = _build_invariant_dispatch()


def _collect_extensions(
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, tuple[str, Mapping[str, Any]]],
    dict[str, tuple[str, Mapping[str, Any]]],
]:
    invariants: dict[str, tuple[str, Mapping[str, Any]]] = {}
    mutations: dict[str, tuple[str, Mapping[str, Any]]] = {}

    def walk(value: Any, schema_name: str) -> None:
        if isinstance(value, Mapping):
            for item in value.get("x-semantic-invariants", []):
                invariant_id = str(item["id"])
                if invariant_id in invariants:
                    raise ValidationFailure(f"duplicate invariant id: {invariant_id}")
                invariants[invariant_id] = (schema_name, item)
            for item in value.get("x-negative-mutation-fixtures", []):
                mutation_id = str(item["id"])
                if mutation_id in mutations:
                    raise ValidationFailure(f"duplicate mutation id: {mutation_id}")
                mutations[mutation_id] = (schema_name, item)
            for child in value.values():
                walk(child, schema_name)
        elif isinstance(value, list):
            for child in value:
                walk(child, schema_name)

    for schema_name, schema in schemas.items():
        walk(schema, schema_name)
    return invariants, mutations


def _pointer_parent(document: Any, pointer: str) -> tuple[Any, str]:
    parts = [
        part.replace("~1", "/").replace("~0", "~")
        for part in pointer.split("/")[1:]
    ]
    current = document
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current, parts[-1]


def _pointer_get(document: Any, pointer: str) -> Any:
    parent, key = _pointer_parent(document, pointer)
    return parent[int(key)] if isinstance(parent, list) else parent[key]


def _apply_patch_op(
    document: Any,
    operation: Mapping[str, Any],
    *,
    canonical_partial: Mapping[str, Any],
) -> None:
    op = operation["op"]
    path = operation["path"]
    parent, key = _pointer_parent(document, path)
    if op == "remove":
        if isinstance(parent, list):
            parent.pop(int(key))
        else:
            parent.pop(key)
        return
    if "value" in operation:
        value = deepcopy(operation["value"])
    elif "from" in operation:
        value = deepcopy(_pointer_get(document, operation["from"]))
    else:
        source = str(operation["value_from"])
        if source == "canonical fixture /reliability":
            value = deepcopy(canonical_partial["reliability"])
        elif source == "canonical fixture /provenance/probability_transform":
            value = deepcopy(canonical_partial["provenance"]["probability_transform"])
        elif " + " in source:
            pointer, increment = source.split(" + ", 1)
            value = _pointer_get(document, pointer) + float(increment)
        else:
            value = deepcopy(_pointer_get(document, source))

    if op in {"add", "replace", "copy"}:
        if isinstance(parent, list):
            index = int(key)
            if op == "add":
                parent.insert(index, value)
            else:
                parent[index] = value
        else:
            parent[key] = value
    else:
        raise ValidationFailure(f"unsupported mutation op: {op}")


def _mutation_base(
    mutation_id: str,
    schema_name: str,
    examples: Mapping[str, Mapping[str, Any]],
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    by_schema = {
        schema_file: output
        for output, schema_file in OUTPUT_SCHEMAS.items()
    }
    if schema_name in by_schema:
        output = by_schema[schema_name]
        if mutation_id.startswith(("archetype_", "contextual_archetype_")):
            return output, deepcopy(schemas[schema_name]["examples"][0])
        return output, deepcopy(examples[output])
    if schema_name == "prediction-provenance.schema.json":
        document = deepcopy(examples["draft_score"])
        provenance = document["provenance"]
        provenance["mode"] = "forecast_simulation"
        provenance["event_start"] = "2026-07-28T18:00:00Z"
        provenance["simulation_run_id"] = "scryglass:simulation:fixture"
        provenance["availability_replayed"] = True
        provenance.pop("sealed_before_event_start", None)
        return "draft_score", document
    if mutation_id.startswith("interval_probability_"):
        return "draft_score", deepcopy(examples["draft_score"])
    if mutation_id.startswith("interval_real_"):
        return "player_rating", deepcopy(examples["player_rating"])
    if mutation_id.startswith("terminal_assignment_"):
        return "draft_score", deepcopy(examples["draft_score"])
    if mutation_id == "settled_probability_equal_boundary":
        document = deepcopy(examples["player_rating"])
        settled = document["settled"]
        settled.update(
            {
                "value": True,
                "rating_resolution": 40.0,
                "posterior_precision_probability": 0.96,
                "posterior_stability_probability": 0.96,
                "stability_within_resolution": True,
            }
        )
        return "player_rating", document
    return "player_rating", deepcopy(examples["player_rating"])


def _invariant_base(
    invariant_id: str,
    schema_name: str,
    examples: Mapping[str, Mapping[str, Any]],
    schemas: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    by_schema = {
        schema_file: output
        for output, schema_file in OUTPUT_SCHEMAS.items()
    }
    if schema_name in by_schema:
        output = by_schema[schema_name]
        if invariant_id.startswith(("archetype_", "contextual_archetype_")):
            return schemas[schema_name]["examples"][0]
        return examples[output]
    if invariant_id.startswith("terminal_assignment_"):
        return examples["draft_score"]
    if invariant_id.startswith("draft_protocol_mapping_"):
        return examples["draft_score"]
    if invariant_id in {
        "forecast_created_before_event",
        "forecast_simulation_replays_historical_availability",
        "hindsight_created_after_event",
    }:
        return examples["draft_score"]
    if invariant_id == "partial_transform_identity_matches_proof":
        return examples["partial_draft_state"]
    if invariant_id == "interval_probability_order":
        return examples["draft_score"]
    return examples["player_rating"]


def _settled_fixture(document: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(document)
    settled = value["settled"]
    settled.update(
        {
            "value": True,
            "rating_resolution": 40.0,
            "posterior_precision_probability": 0.96,
            "interval_width": (
                value["interval_95"]["upper"] - value["interval_95"]["lower"]
            ),
            "interval_contains_posterior_mean": True,
            "stability_change": 39.0,
            "posterior_stability_probability": 0.96,
            "stability_within_resolution": True,
            "entity_active": True,
            "current_for_scope": True,
            "inputs_complete": True,
            "inputs_fresh": True,
            "material_fallback_levels": [],
            "out_of_distribution_flags": [],
            "coverage_gate_passed": True,
        }
    )
    value["rank_eligibility"]["active"] = True
    value["rank_eligibility"]["current"] = True
    _seal_dynamic_identity(value)
    return value


def _timing_fixture(
    document: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    value = deepcopy(document)
    provenance = value["provenance"]
    provenance["mode"] = mode
    provenance["event_start"] = "2026-07-28T18:00:00Z"
    if mode == "forecast":
        provenance["created_at"] = "2026-07-27T18:00:01Z"
        provenance["sealed_before_event_start"] = True
        provenance.pop("simulation_run_id", None)
        provenance.pop("availability_replayed", None)
    elif mode == "forecast_simulation":
        provenance["created_at"] = "2026-07-29T18:00:01Z"
        provenance["simulation_run_id"] = "scryglass:simulation:fixture"
        provenance["availability_replayed"] = True
        provenance.pop("sealed_before_event_start", None)
    elif mode == "hindsight":
        provenance["event_start"] = "2026-07-27T17:00:00Z"
        provenance["created_at"] = "2026-07-27T18:00:01Z"
        provenance["related_forecast_id"] = "scryglass:prediction:prior-forecast"
        provenance.pop("sealed_before_event_start", None)
        provenance.pop("simulation_run_id", None)
        provenance.pop("availability_replayed", None)
    _seal_dynamic_identity(value)
    return value


def _terminal_partial_fixture(
    examples: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = deepcopy(examples["partial_draft_state"])
    terminal = examples["draft_score"]
    state = value["state"]
    state.update(
        {
            "terminal": True,
            "draft_state_id": terminal["draft_state_id"],
            "event_id": terminal["event_id"],
            "competition_scope_id": terminal["competition_scope_id"],
            "competition_scope_kind": terminal["competition_scope_kind"],
            "patch_id": terminal["patch_id"],
            "protocol_id": terminal["protocol_id"],
            "side_mapping": deepcopy(terminal["side_mapping"]),
            "source_record": deepcopy(terminal["source_record"]),
            "actions": deepcopy(terminal["actions"]),
            "current_role_constraints": [],
            "role_constraint_revisions": deepcopy(
                terminal["role_constraint_revisions"]
            ),
            "assignment_revisions": deepcopy(terminal["assignment_revisions"]),
            "final_assignments": deepcopy(terminal["final_assignments"]),
            "side_to_act": None,
        }
    )
    for key in (
        "committed_value_logit_a",
        "strategic_response_adjustment_logit_a",
        "strategic_value_logit_a",
        "partial_score_a",
        "partial_score_b",
        "standardized_map_win_probability_a",
        "interval_95",
        "flex_value_logit_a",
        "search",
        "recommendations",
    ):
        value.pop(key, None)
    registration = SEMANTIC_ARTIFACTS["terminal_outputs"][
        terminal["provenance"]["prediction_id"]
    ]
    value["terminal_delegation"] = {
        "delegation_method": "canonical_terminal_draft_score",
        "terminal_prediction_id": terminal["provenance"]["prediction_id"],
        "terminal_model_version": terminal["model_version"],
        "terminal_output_sha256": registration["raw_sha256"],
        "score_a": terminal["score_a"],
        "score_b": terminal["score_b"],
        "standardized_map_win_probability_a": terminal[
            "standardized_map_win_probability_a"
        ],
        "interval_95": deepcopy(terminal["interval_95"]),
    }
    value["literal_interpretation"] = (
        "Terminal draft state delegated without recomputation to canonical "
        "Terminal Draft Score for identical inputs and terminal model version."
    )
    _seal_dynamic_identity(value)
    return value


def _exact_search_fixture(
    examples: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = deepcopy(examples["partial_draft_state"])
    value["search"].update(
        {
            "policy_id": "scryglass:policy:exact-minimax-v2",
            "method": "hard_minimax",
            "temperature": None,
            "exact": True,
            "coverage": {
                "metric_id": "scryglass:coverage-metric:complete-state-space",
                "value": 1.0,
            },
        }
    )
    value["search"].pop("approximation_error_bound", None)
    _seal_dynamic_identity(value)
    return value


def _tie_fixture(
    examples: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = deepcopy(examples["tier_list"])
    first, second = value["entries"]
    second["incremental_value_pp"] = first["incremental_value_pp"]
    second["counterability_pp"] = first["counterability_pp"]
    second["tier_value_pp"] = first["tier_value_pp"]
    second["interval_95"] = deepcopy(first["interval_95"])
    value["entries"] = sorted(
        value["entries"],
        key=lambda item: item["champion_id"],
    )
    for index, item in enumerate(value["entries"], start=1):
        item["rank"] = index
    _seal_dynamic_identity(value)
    return value


def _contextual_partial_fixture(
    examples: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = deepcopy(examples["partial_draft_state"])
    value["identity_mode"] = "contextual"
    value["identity_intentionally_omitted"] = False
    value["baseline_strength_equalized"] = True
    value["context"] = deepcopy(examples["draft_score"]["context"])
    value["neutral_comparison"] = {
        "partial_score_a": value["partial_score_a"],
        "partial_score_b": value["partial_score_b"],
        "standardized_map_win_probability_a": value[
            "standardized_map_win_probability_a"
        ],
    }
    _seal_dynamic_identity(value)
    return value


def _positive_fixture(
    invariant_id: str,
    schema_name: str,
    examples: Mapping[str, Mapping[str, Any]],
    schemas: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = deepcopy(_invariant_base(invariant_id, schema_name, examples, schemas))
    if invariant_id in {
        "settled_interval_width_at_resolution",
        "settled_stability_change_at_resolution",
    }:
        return _settled_fixture(examples["player_rating"])
    if invariant_id == "forecast_created_before_event":
        return _timing_fixture(examples["draft_score"], "forecast")
    if invariant_id == "forecast_simulation_replays_historical_availability":
        return _timing_fixture(examples["draft_score"], "forecast_simulation")
    if invariant_id == "hindsight_created_after_event":
        return _timing_fixture(examples["draft_score"], "hindsight")
    if invariant_id in {
        "partial_terminal_assignments_match_latest_revisions",
        "terminal_delegation_score_identity",
        "terminal_delegation_exact_canonical_output",
    }:
        return _terminal_partial_fixture(examples)
    if invariant_id == "partial_exact_search_full_coverage":
        return _exact_search_fixture(examples)
    if invariant_id == "tier_rank_matches_value_and_tie_break":
        return _tie_fixture(examples)
    if invariant_id == "partial_neutral_comparison_identity":
        return _contextual_partial_fixture(examples)
    if invariant_id == "partial_assignment_revision_chain_append_only":
        value["state"]["assignment_revisions"] = [
            {
                "revision_id": "scryglass:assignment-revision:partial-positive",
                "revision_index": 1,
                "supersedes_revision_id": None,
                "action_id": "scryglass:draft-action:partial-1",
                "previous_role": None,
                "revised_role": "top",
                "reason": "initial_resolution",
                "source_id": "scryglass:source:analysis-input",
                "recorded_at": "2026-07-27T17:56:00Z",
                "available_at": "2026-07-27T17:56:00Z",
            }
        ]
        _seal_dynamic_identity(value)
    return value


def _counterexample_fixture(
    invariant_id: str,
    positive: Mapping[str, Any],
) -> dict[str, Any]:
    value = deepcopy(positive)
    if invariant_id in {
        "player_interval_contains_posterior_mean",
        "team_interval_contains_posterior_mean",
    }:
        value["interval_95"]["upper"] = value["posterior_mean"] - 0.1
    elif invariant_id in {
        "player_settled_width_matches_interval",
        "team_settled_width_matches_interval",
    }:
        value["settled"]["interval_width"] += 1.0
    elif invariant_id == "team_exact_five_unique_players":
        value["roster"][1]["player_id"] = value["roster"][0]["player_id"]
    elif invariant_id == "team_rating_component_identity":
        value["posterior_mean"] += 1.0
    elif invariant_id == "team_component_estimand_units":
        value["roster_strength_component"] += 1.0
        value["posterior_mean"] += 1.0
    elif invariant_id in {
        "terminal_score_probability_identity",
        "partial_score_probability_identity",
    }:
        key = "score_a" if "terminal_" in invariant_id else "partial_score_a"
        complement_key = "score_b" if key == "score_a" else "partial_score_b"
        value[key] += 0.1
        value[complement_key] -= 0.1
    elif invariant_id in {
        "terminal_score_complement_identity",
        "partial_score_complement_identity",
    }:
        key = "score_b" if "terminal_" in invariant_id else "partial_score_b"
        value[key] += 0.1
    elif invariant_id == "terminal_ledger_reconciliation":
        value["ledger_logit_sum"] += 0.1
    elif invariant_id in {
        "terminal_interval_contains_probability",
        "partial_interval_contains_probability",
    }:
        value["interval_95"]["upper"] = (
            value["standardized_map_win_probability_a"] - 0.001
        )
    elif invariant_id == "terminal_action_slots_unique_and_ordered":
        value["actions"][0]["canonical_side"] = "B"
    elif invariant_id in {"terminal_side_mapping_source_available", "partial_source_available"}:
        state = value["state"] if "partial_" in invariant_id else value
        state["source_record"]["available_at"] = "2026-07-27T18:00:01Z"
    elif invariant_id in {
        "terminal_competition_scope_taxonomy",
        "partial_competition_scope_taxonomy",
        "tier_list_regional_league_scope_only",
    }:
        state = value["state"] if invariant_id.startswith("partial_") else value
        state["competition_scope_id"] = "scryglass:competition-scope:nonexistent"
    elif invariant_id in {
        "terminal_role_revision_chain_append_only",
        "partial_role_revision_chain_append_only",
    }:
        state = value["state"] if invariant_id.startswith("partial_") else value
        duplicate = deepcopy(state["role_constraint_revisions"][0])
        duplicate["revision_id"] += ":duplicate"
        state["role_constraint_revisions"].append(duplicate)
    elif invariant_id in {
        "terminal_assignment_revision_chain_append_only",
        "partial_assignment_revision_chain_append_only",
    }:
        state = value["state"] if invariant_id.startswith("partial_") else value
        duplicate = deepcopy(state["assignment_revisions"][0])
        duplicate["revision_id"] += ":duplicate"
        state["assignment_revisions"].append(duplicate)
    elif invariant_id in {
        "terminal_assignments_match_latest_revisions",
        "partial_terminal_assignments_match_latest_revisions",
    }:
        state = value["state"] if invariant_id.startswith("partial_") else value
        state["final_assignments"][0]["role"], state["final_assignments"][2]["role"] = (
            state["final_assignments"][2]["role"],
            state["final_assignments"][0]["role"],
        )
    elif invariant_id in {
        "contextual_neutral_comparison_identity",
        "partial_neutral_comparison_identity",
    }:
        score = "score_a" if invariant_id.startswith("contextual_") else "partial_score_a"
        value["neutral_comparison"][score] += 0.1
    elif invariant_id == "partial_value_decomposition":
        value["strategic_value_logit_a"] += 0.1
    elif invariant_id == "partial_current_constraints_cover_picks":
        value["state"]["current_role_constraints"].pop()
    elif invariant_id == "partial_current_constraints_are_latest_revision":
        value["state"]["current_role_constraints"][0]["role_set"] = ["top"]
    elif invariant_id == "partial_action_slots_unique_and_protocol_legal":
        value["state"]["protocol_id"] = "scryglass:protocol:nonexistent"
    elif invariant_id == "terminal_delegation_score_identity":
        value["terminal_delegation"]["score_a"] += 0.1
        value["terminal_delegation"]["score_b"] -= 0.1
    elif invariant_id == "terminal_delegation_exact_canonical_output":
        value["terminal_delegation"]["terminal_output_sha256"] = "9" * 64
    elif invariant_id == "archetype_extrapolation_ordinal_groups":
        groups = value["archetype_extrapolation"]["recommendation_groups"]
        if len(groups) == 1:
            groups.append(deepcopy(groups[0]))
        groups[1]["group_order"] = groups[0]["group_order"]
        groups[1]["group_id"] += ":duplicate-order"
    elif invariant_id == "archetype_extrapolation_support_gap_matches_fallback":
        value["archetype_extrapolation"]["support_gap"][
            "competition_scope_id"
        ] = "scryglass:competition-scope:lcs"
    elif invariant_id == "archetype_extrapolation_noncanonical_boundary":
        value["archetype_extrapolation"]["recommendation_groups"][0]["actions"][
            0
        ]["action_id"] = "scryglass:recommendation:canonical:forbidden"
    elif invariant_id == "contextual_archetype_exact_context_fresh":
        value["context"]["roster_a_snapshot_id"] = (
            "scryglass:roster-snapshot:nonexistent"
        )
    elif invariant_id == "partial_approximate_search_reports_coverage_and_bound":
        value["search"]["approximation_error_bound"]["method_id"] = (
            "scryglass:bound:text-only-unregistered"
        )
    elif invariant_id == "partial_exact_search_full_coverage":
        value["search"]["policy_id"] = "scryglass:policy:text-only-exact"
    elif invariant_id == "tier_value_identity":
        value["entries"][0]["tier_value_pp"] += 0.1
    elif invariant_id == "tier_single_manifested_counterability_weight":
        value["counterability_policy_id"] = "scryglass:counterability:nonexistent"
    elif invariant_id == "tier_rank_and_champion_unique":
        value["entries"][1]["champion_id"] = value["entries"][0]["champion_id"]
    elif invariant_id == "tier_rank_matches_value_and_tie_break":
        value["entries"].reverse()
        for index, item in enumerate(value["entries"], start=1):
            item["rank"] = index
    elif invariant_id == "tier_interval_contains_value":
        value["entries"][0]["interval_95"]["upper"] = (
            value["entries"][0]["tier_value_pp"] - 0.1
        )
    elif invariant_id in {"interval_probability_order", "interval_real_order"}:
        value["interval_95"]["lower"] = value["interval_95"]["upper"] + 0.1
    elif invariant_id == "evidence_concepts_remain_separate":
        common = "scryglass:evidence-method:aggregate-confidence"
        value["evidence"]["posterior_displacement"]["method_id"] = common
        value["evidence"]["precision"]["method_id"] = common
        value["evidence"]["source_context_coverage"]["coverage_spec_id"] = common
    elif invariant_id == "evidence_not_volume_proxy":
        value["evidence"]["posterior_displacement"]["method_id"] = (
            "scryglass:evidence-method:game-count-confidence"
        )
    elif invariant_id == "reliability_log_loss_skill_identity":
        value["reliability"]["log_loss_skill"] += 0.1
    elif invariant_id == "reliability_brier_skill_identity":
        value["reliability"]["brier_skill"] += 0.1
    elif invariant_id == "reliability_total_stratum_mapping":
        value["reliability"]["stratum_mapping_sha256"] = "8" * 64
    elif invariant_id == "settled_interval_width_at_resolution":
        value["settled"]["interval_width"] = (
            2 * value["settled"]["rating_resolution"] + 0.1
        )
    elif invariant_id == "settled_stability_change_at_resolution":
        value["settled"]["stability_change"] = (
            value["settled"]["rating_resolution"] + 0.1
        )
    elif invariant_id == "draft_protocol_mapping_bijective_game_side":
        value["side_mapping"]["side_b_game_side"] = value["side_mapping"][
            "side_a_game_side"
        ]
    elif invariant_id == "draft_protocol_mapping_bijective_order":
        value["side_mapping"]["side_b_draft_order"] = value["side_mapping"][
            "side_a_draft_order"
        ]
    elif invariant_id == "terminal_assignment_unique_champions":
        value["final_assignments"][1]["champion_id"] = value["final_assignments"][0][
            "champion_id"
        ]
    elif invariant_id == "terminal_assignment_matches_pick_actions":
        value["final_assignments"][0]["action_id"] = (
            "scryglass:draft-action:not-a-pick"
        )
    elif invariant_id == "forecast_created_before_event":
        value["provenance"]["created_at"] = value["provenance"]["event_start"]
    elif invariant_id == "forecast_simulation_replays_historical_availability":
        value["provenance"]["as_of"] = value["provenance"]["event_start"]
    elif invariant_id == "hindsight_created_after_event":
        value["provenance"]["created_at"] = "2026-07-27T16:59:59Z"
    elif invariant_id == "partial_transform_identity_matches_proof":
        value["provenance"]["partial_probability_calibration"][
            "transform_sha256"
        ] = "9" * 64
    else:
        raise ValidationFailure(
            f"no semantic counterexample builder for invariant {invariant_id}"
        )
    _seal_dynamic_identity(value)
    return value


def _validator_registry(
    schemas: Mapping[str, Mapping[str, Any]],
) -> Registry:
    registry = Registry()
    for schema in schemas.values():
        registry = registry.with_resource(
            str(schema["$id"]), Resource.from_contents(schema)
        )
    return registry


def _schema_errors(
    schema: Mapping[str, Any],
    document: Mapping[str, Any],
    registry: Registry,
) -> tuple[str, ...]:
    validator = Draft202012Validator(schema, registry=registry)
    return tuple(
        error.message
        for error in sorted(validator.iter_errors(document), key=lambda item: list(item.path))
    )


def _schema_error_evidence(
    schema: Mapping[str, Any],
    document: Mapping[str, Any],
    registry: Registry,
) -> tuple[dict[str, Any], ...]:
    validator = Draft202012Validator(schema, registry=registry)
    return tuple(
        {
            "instance_path": "/" + "/".join(str(part) for part in error.path),
            "schema_path": "/" + "/".join(str(part) for part in error.schema_path),
            "keyword": str(error.validator),
            "reason": error.message,
        }
        for error in sorted(
            validator.iter_errors(document),
            key=lambda item: (list(item.path), list(item.schema_path)),
        )
    )


def _is_not_applicable(detail: str) -> bool:
    return detail.startswith("not applicable")


def run_five_output_validation(
    *,
    contract_root: Path = CONTRACT_ROOT,
    invariant_dispatch: Mapping[str, InvariantFn] | None = None,
    repository_root: Path = REPOSITORY_ROOT,
    anchors: ContractValidationAnchors | None = None,
) -> FiveOutputValidationReport:
    resolved = anchors or default_contract_validation_anchors()
    resolved_contract_root = (
        contract_root
        if contract_root.is_absolute()
        else repository_root / contract_root
    )
    _verify_contract_validation_trust_root(repository_root, resolved)
    actual_contract_tree_sha256 = _actual_contract_tree_sha256(repository_root)
    if actual_contract_tree_sha256 != resolved.contract_tree_sha256:
        raise ValidationFailure(
            "canonical source-tree-v1 digest does not match frozen C0"
        )
    if invariant_dispatch is None:
        if anchors is None:
            invariant_dispatch = dict(INVARIANT_DISPATCH)
        else:
            invariant_dispatch = _build_invariant_dispatch(
                terminal_output_validator=lambda output, document: (
                    validate_output_payload(
                        output,
                        document,
                        contract_root=resolved_contract_root,
                        repository_root=repository_root,
                        anchors=resolved,
                    )
                )
            )
    else:
        invariant_dispatch = dict(invariant_dispatch)
    missing_files = [
        name
        for name in (*SCHEMA_FILES, *EXAMPLE_FILES.values())
        if not (resolved_contract_root / name).is_file()
    ]
    if missing_files:
        raise ValidationFailure(f"missing contract validation inputs: {missing_files}")

    schema_hashes = {
        name: _raw_sha256(resolved_contract_root / name) for name in SCHEMA_FILES
    }
    example_hashes = {
        name: _raw_sha256(resolved_contract_root / name)
        for name in EXAMPLE_FILES.values()
    }
    if dict(resolved.schema_sha256) != schema_hashes:
        raise ValidationFailure("schema drift detected against L2 anchors")
    if dict(resolved.example_sha256) != example_hashes:
        raise ValidationFailure("canonical example drift detected against L2 anchors")
    contract_content_sha256 = _contract_content_sha256(
        resolved_contract_root.parent
    )
    if contract_content_sha256 != resolved.contract_content_sha256:
        raise ValidationFailure(
            "loaded docs/model-v2 content does not match independently anchored final C0"
        )
    semantic_artifact_path = repository_root / resolved.semantic_artifact_path
    semantic_artifact_sha256 = _raw_sha256(semantic_artifact_path)
    if semantic_artifact_sha256 != resolved.semantic_artifact_raw_sha256:
        raise ValidationFailure("semantic registry/artifact anchor mismatch")
    if (
        _raw_sha256(repository_root / resolved.contract_fixture_authority_path)
        != resolved.contract_fixture_authority_raw_sha256
    ):
        raise ValidationFailure("contract fixture authority anchor mismatch")

    schemas = {
        name: json.loads(
            (resolved_contract_root / name).read_text(encoding="utf-8")
        )
        for name in SCHEMA_FILES
    }
    examples = {
        output: json.loads(
            (resolved_contract_root / name).read_text(encoding="utf-8")
        )
        for output, name in EXAMPLE_FILES.items()
    }
    invariants, mutations = _collect_extensions(schemas)
    structural_mutation_ids = {
        mutation_id
        for mutation_id, (_, fixture) in mutations.items()
        if fixture.get("expected_schema_failure") is True
    }
    if structural_mutation_ids != set(EXPECTED_STRUCTURAL_DIAGNOSTIC_SHA256):
        raise ValidationFailure(
            "frozen structural diagnostic authority is incomplete or extra"
        )
    if set(invariant_dispatch) != set(invariants):
        missing = sorted(set(invariants) - set(invariant_dispatch))
        extra = sorted(set(invariant_dispatch) - set(invariants))
        raise ValidationFailure(
            f"invariant dispatch mismatch; missing={missing}, extra={extra}"
        )

    registry = _validator_registry(schemas)
    canonical_identity_pairs = _canonical_identity_pairs(resolved_contract_root)
    evidence: list[ValidationEvidence] = []
    structural_pass_count = 0

    for output, schema_name in OUTPUT_SCHEMAS.items():
        errors = _schema_errors(schemas[schema_name], examples[output], registry)
        if errors:
            raise ValidationFailure(
                f"canonical {output} example failed schema validation: {errors}"
            )
        provenance_ok, provenance_detail = _provenance_integrity(
            examples[output],
            canonical_identity_pairs=canonical_identity_pairs,
        )
        if not provenance_ok:
            raise ValidationFailure(
                f"canonical {output} provenance failed: {provenance_detail}"
            )
        structural_pass_count += 1
        evidence.append(
            ValidationEvidence(
                evidence_id=f"structural:{output}",
                kind="structural",
                status="passed",
                detail="canonical example and shared references passed",
                applicable=True,
            )
        )
        identity_ok, identity_detail = _registry_identity_integrity(
            output, examples[output]
        )
        if not identity_ok:
            raise ValidationFailure(
                f"canonical {output} registry identity failed: {identity_detail}"
            )

    # The embedded research-only partial example is canonical contract input too.
    partial_schema = schemas["partial-draft-state.schema.json"]
    for index, embedded in enumerate(partial_schema.get("examples", [])):
        errors = _schema_errors(partial_schema, embedded, registry)
        if errors:
            raise ValidationFailure(
                f"embedded partial example {index} failed schema validation: {errors}"
            )
        provenance_ok, detail = _provenance_integrity(
            embedded,
            canonical_identity_pairs=canonical_identity_pairs,
        )
        if not provenance_ok:
            raise ValidationFailure(
                f"embedded partial example {index} provenance failed: {detail}"
            )
        structural_pass_count += 1
        evidence.append(
            ValidationEvidence(
                evidence_id=f"structural:partial_embedded:{index}",
                kind="structural",
                status="passed",
                detail="embedded canonical example passed",
                applicable=True,
            )
        )

    invariant_pass_count = 0
    invariant_counts: dict[str, dict[str, int]] = {}
    for invariant_id in sorted(invariants):
        schema_name, _ = invariants[invariant_id]
        document = _positive_fixture(
            invariant_id, schema_name, examples, schemas
        )
        output = next(
            (
                name
                for name, candidate_schema in OUTPUT_SCHEMAS.items()
                if candidate_schema == schema_name
            ),
            (
                "partial_draft_state"
                if invariant_id == "partial_transform_identity_matches_proof"
                else "draft_score"
                if invariant_id.startswith(
                    (
                        "terminal_assignment_",
                        "draft_protocol_mapping_",
                        "forecast_",
                        "hindsight_",
                        "interval_probability_",
                    )
                )
                else "player_rating"
            ),
        )
        schema_errors = _schema_errors(
            schemas[OUTPUT_SCHEMAS[output]],
            document,
            registry,
        )
        if schema_errors:
            raise ValidationFailure(
                f"positive invariant fixture is schema-invalid: "
                f"{invariant_id}: {schema_errors}"
            )
        passed, detail = invariant_dispatch[invariant_id](document)
        applicable = not _is_not_applicable(detail)
        evidence.append(
            ValidationEvidence(
                evidence_id=f"invariant:{invariant_id}:positive",
                kind="invariant_positive",
                status="passed" if passed else "failed",
                detail=detail,
                applicable=applicable,
            )
        )
        if not passed or not applicable:
            raise ValidationFailure(
                f"invariant lacks applicable positive execution: "
                f"{invariant_id}: {detail}"
            )
        counterexample = _counterexample_fixture(invariant_id, document)
        counter_errors = _schema_errors(
            schemas[OUTPUT_SCHEMAS[output]],
            counterexample,
            registry,
        )
        if counter_errors:
            raise ValidationFailure(
                f"semantic counterexample is schema-invalid: "
                f"{invariant_id}: {counter_errors}"
            )
        counter_passed, counter_detail = invariant_dispatch[invariant_id](
            counterexample
        )
        counter_applicable = not _is_not_applicable(counter_detail)
        evidence.append(
            ValidationEvidence(
                evidence_id=f"invariant:{invariant_id}:counterexample",
                kind="invariant_counterexample",
                status="passed" if not counter_passed else "failed",
                detail=(
                    f"target rejected schema-valid counterexample: {counter_detail}"
                    if not counter_passed
                    else "target unexpectedly accepted schema-valid counterexample"
                ),
                applicable=counter_applicable,
            )
        )
        if counter_passed or not counter_applicable:
            raise ValidationFailure(
                f"invariant lacks applicable semantic counterexample: "
                f"{invariant_id}: {counter_detail}"
            )
        invariant_counts[invariant_id] = {
            "applicable": 2,
            "passed": 2,
            "not_applicable": 0,
        }
        invariant_pass_count += 1

    mutation_pass_count = 0
    canonical_partial = examples["partial_draft_state"]
    for mutation_id in sorted(mutations):
        schema_name, fixture = mutations[mutation_id]
        output, mutated = _mutation_base(
            mutation_id, schema_name, examples, schemas
        )
        operations = fixture["mutation"]
        if isinstance(operations, Mapping):
            operations = [operations]
        for operation in operations:
            _apply_patch_op(
                mutated,
                operation,
                canonical_partial=canonical_partial,
            )
        if mutation_id == "terminal_assignment_overwrite_without_revision":
            mutated["final_assignments"][2]["role"] = "top"

        target_schema_name = OUTPUT_SCHEMAS[output]
        errors = _schema_error_evidence(
            schemas[target_schema_name], mutated, registry
        )
        expected_violation = fixture.get("expected_violation")
        if fixture.get("expected_schema_failure") is True:
            expected_diagnostic_sha256 = (
                EXPECTED_STRUCTURAL_DIAGNOSTIC_SHA256.get(mutation_id)
            )
            passed = (
                expected_diagnostic_sha256 is not None
                and canonical_sha256(errors) == expected_diagnostic_sha256
            )
            detail = (
                "schema rejected mutation with frozen exact diagnostic set: "
                + canonical_json(errors)
                if passed
                else "schema diagnostic set/path/schema/keyword/reason mismatch"
            )
        elif expected_violation:
            if errors:
                passed = False
                detail = (
                    "semantic mutation was structurally invalid: "
                    + canonical_json(errors)
                )
            else:
                semantic_passed, semantic_detail = invariant_dispatch[
                    str(expected_violation)
                ](mutated)
                passed = not semantic_passed and not _is_not_applicable(
                    semantic_detail
                )
                detail = (
                    f"semantic invariant {expected_violation} alone rejected "
                    f"schema-valid mutation: {semantic_detail}"
                    if passed
                    else f"semantic invariant unexpectedly passed/NA: {semantic_detail}"
                )
        else:
            passed = False
            detail = "mutation fixture lacks an executable expectation"
        evidence.append(
            ValidationEvidence(
                evidence_id=mutation_id,
                kind="mutation",
                status="passed" if passed else "failed",
                detail=detail,
                applicable=True,
            )
        )
        if not passed:
            raise ValidationFailure(f"negative mutation failed: {mutation_id}: {detail}")
        mutation_pass_count += 1

    unsigned = {
        "contract_tree_sha256": actual_contract_tree_sha256,
        "schema_sha256": schema_hashes,
        "example_sha256": example_hashes,
        "semantic_artifact_sha256": semantic_artifact_sha256,
        "contract_fixture_authority_raw_sha256": (
            resolved.contract_fixture_authority_raw_sha256
        ),
        "contract_fixture_authority_object_sha256": (
            resolved.contract_fixture_authority_object_sha256
        ),
        "contract_validation_trust_root_raw_sha256": (
            resolved.contract_validation_trust_root_raw_sha256
        ),
        "contract_validation_trust_root_object_sha256": (
            resolved.contract_validation_trust_root_object_sha256
        ),
        "contract_content_sha256": contract_content_sha256,
        "invariant_ids": sorted(invariants),
        "mutation_ids": sorted(mutations),
        "evidence": [item.to_payload() for item in evidence],
        "invariant_counts": invariant_counts,
        "invariant_pass_count": invariant_pass_count,
        "mutation_pass_count": mutation_pass_count,
        "structural_pass_count": structural_pass_count,
        "all_pass": True,
    }
    return FiveOutputValidationReport(
        contract_tree_sha256=actual_contract_tree_sha256,
        schema_sha256=schema_hashes,
        example_sha256=example_hashes,
        semantic_artifact_sha256=semantic_artifact_sha256,
        contract_fixture_authority_raw_sha256=(
            resolved.contract_fixture_authority_raw_sha256
        ),
        contract_fixture_authority_object_sha256=(
            resolved.contract_fixture_authority_object_sha256
        ),
        contract_validation_trust_root_raw_sha256=(
            resolved.contract_validation_trust_root_raw_sha256
        ),
        contract_validation_trust_root_object_sha256=(
            resolved.contract_validation_trust_root_object_sha256
        ),
        contract_content_sha256=contract_content_sha256,
        invariant_ids=tuple(sorted(invariants)),
        mutation_ids=tuple(sorted(mutations)),
        evidence=tuple(evidence),
        invariant_counts=invariant_counts,
        invariant_pass_count=invariant_pass_count,
        mutation_pass_count=mutation_pass_count,
        structural_pass_count=structural_pass_count,
        all_pass=True,
        report_sha256=canonical_sha256(unsigned),
    )


def verify_five_output_validation_report(
    report: FiveOutputValidationReport,
    *,
    contract_root: Path = CONTRACT_ROOT,
    repository_root: Path = REPOSITORY_ROOT,
    anchors: ContractValidationAnchors | None = None,
) -> None:
    resolved = anchors or default_contract_validation_anchors()
    if report.contract_tree_sha256 != resolved.contract_tree_sha256:
        raise ValidationFailure("five-output report contract hash mismatch")
    if (
        report.contract_fixture_authority_raw_sha256
        != resolved.contract_fixture_authority_raw_sha256
        or report.contract_fixture_authority_object_sha256
        != resolved.contract_fixture_authority_object_sha256
    ):
        raise ValidationFailure("five-output report fixture authority mismatch")
    if (
        report.contract_validation_trust_root_raw_sha256
        != resolved.contract_validation_trust_root_raw_sha256
        or report.contract_validation_trust_root_object_sha256
        != resolved.contract_validation_trust_root_object_sha256
    ):
        raise ValidationFailure("five-output report trust-root mismatch")
    if not report.all_pass or not report.verify_hash():
        raise ValidationFailure("five-output report is not all-pass or is tampered")
    fresh = run_five_output_validation(
        contract_root=contract_root,
        repository_root=repository_root,
        anchors=resolved,
    )
    if report.to_payload() != fresh.to_payload():
        raise ValidationFailure(
            "five-output report differs from fresh anchored execution"
        )
    if set(report.invariant_ids) != set(INVARIANT_DISPATCH):
        raise ValidationFailure("five-output report invariant coverage is incomplete")
    if any(item.status != "passed" for item in report.evidence):
        raise ValidationFailure("five-output report contains failed evidence")
    if any(
        counts != {"applicable": 2, "passed": 2, "not_applicable": 0}
        for counts in report.invariant_counts.values()
    ):
        raise ValidationFailure(
            "five-output report lacks positive and counterexample coverage"
        )


def validate_current_contract_validation_inputs(
    *,
    repository_root: Path = REPOSITORY_ROOT,
    contract_root: Path | None = None,
    anchors: ContractValidationAnchors | None = None,
) -> dict[str, Any]:
    """Verify that current public-output contracts match frozen trust anchors.

    This is a byte/provenance check only.  A successful return does not grant
    model, publication, probability, recommendation, or betting authority.
    """

    resolved = anchors or default_contract_validation_anchors()
    requested_contract_root = (
        contract_root
        if contract_root is not None
        else CONTRACT_ROOT
    )
    resolved_contract_root = (
        requested_contract_root
        if requested_contract_root.is_absolute()
        else repository_root / requested_contract_root
    )
    _verify_contract_validation_trust_root(repository_root, resolved)
    schema_hashes = {
        name: _raw_sha256(resolved_contract_root / name) for name in SCHEMA_FILES
    }
    example_hashes = {
        name: _raw_sha256(resolved_contract_root / name)
        for name in EXAMPLE_FILES.values()
    }
    if (
        _actual_contract_tree_sha256(repository_root)
        != resolved.contract_tree_sha256
        or
        schema_hashes != dict(resolved.schema_sha256)
        or example_hashes != dict(resolved.example_sha256)
        or _contract_content_sha256(resolved_contract_root.parent)
        != resolved.contract_content_sha256
        or _raw_sha256(repository_root / resolved.semantic_artifact_path)
        != resolved.semantic_artifact_raw_sha256
    ):
        raise ValidationFailure(
            "output validation inputs are missing, stale, or unanchored"
        )
    return {
        "contract_tree_sha256": resolved.contract_tree_sha256,
        "schema_sha256": dict(schema_hashes),
        "example_sha256": dict(example_hashes),
        "contract_content_sha256": resolved.contract_content_sha256,
        "semantic_artifact_raw_sha256": resolved.semantic_artifact_raw_sha256,
        "production_model_authority": False,
    }


def validate_output_payload(
    output: str,
    document: Mapping[str, Any],
    *,
    contract_root: Path = CONTRACT_ROOT,
    repository_root: Path = REPOSITORY_ROOT,
    anchors: ContractValidationAnchors | None = None,
) -> tuple[ValidationEvidence, ...]:
    resolved_contract_root = (
        contract_root
        if contract_root.is_absolute()
        else repository_root / contract_root
    )
    validate_current_contract_validation_inputs(
        repository_root=repository_root,
        contract_root=resolved_contract_root,
        anchors=anchors,
    )
    if output not in OUTPUT_SCHEMAS:
        raise ValidationFailure(f"unknown output type: {output}")
    schemas = {
        name: json.loads(
            (resolved_contract_root / name).read_text(encoding="utf-8")
        )
        for name in SCHEMA_FILES
    }
    registry = _validator_registry(schemas)
    schema_name = OUTPUT_SCHEMAS[output]
    errors = _schema_errors(schemas[schema_name], document, registry)
    if errors:
        raise ValidationFailure(f"{output} structural validation failed: {errors}")
    fixture_identity = (
        canonical_unsigned_output_sha256(document),
        str(document["provenance"]["output_sha256"]),
        str(document["provenance"]["prediction_id"]),
    )
    identity_pairs = _contract_fixture_identity_pairs(
        resolved_contract_root,
        repository_root=repository_root,
        anchors=anchors,
    )
    if fixture_identity not in identity_pairs:
        raise ValidationFailure(
            "production model authority is unavailable and payload is not an "
            "exact pinned contract fixture"
        )
    provenance_ok, provenance_detail = _provenance_integrity(
        document,
        canonical_identity_pairs=identity_pairs,
    )
    if not provenance_ok:
        raise ValidationFailure(f"{output} provenance failed: {provenance_detail}")
    identity_ok, identity_detail = _registry_identity_integrity(output, document)
    if not identity_ok:
        raise ValidationFailure(f"{output} registry identity failed: {identity_detail}")

    invariants, _ = _collect_extensions(schemas)
    relevant = {
        invariant_id
        for invariant_id, (owner, _) in invariants.items()
        if owner == schema_name
    }
    if "evidence" in document:
        relevant.update(
            {
                "evidence_concepts_remain_separate",
                "evidence_not_volume_proxy",
            }
        )
    if "reliability" in document:
        relevant.update(
            {
                "reliability_log_loss_skill_identity",
                "reliability_brier_skill_identity",
                "reliability_total_stratum_mapping",
            }
        )
    relevant.update(
        {
            "forecast_created_before_event",
            "forecast_simulation_replays_historical_availability",
            "hindsight_created_after_event",
            "partial_transform_identity_matches_proof",
        }
    )
    if output in {"player_rating", "team_rating"} and "interval_95" in document:
        relevant.update(
            {
                "interval_real_order",
                "settled_interval_width_at_resolution",
                "settled_stability_change_at_resolution",
            }
        )
    if output in {"draft_score", "partial_draft_state"} and "interval_95" in document:
        relevant.add("interval_probability_order")
    if output == "draft_score" and "actions" in document:
        relevant.update(
            {
                "draft_protocol_mapping_bijective_game_side",
                "draft_protocol_mapping_bijective_order",
                "terminal_assignment_unique_champions",
                "terminal_assignment_matches_pick_actions",
            }
        )

    evidence: list[ValidationEvidence] = []
    for invariant_id in sorted(relevant):
        if document.get("status") == "unavailable":
            passed, detail = _not_applicable()
        else:
            passed, detail = INVARIANT_DISPATCH[invariant_id](document)
        applicable = not _is_not_applicable(detail)
        evidence.append(
            ValidationEvidence(
                evidence_id=invariant_id,
                kind="invariant",
                status=(
                    "not_applicable"
                    if not applicable
                    else ("passed" if passed else "failed")
                ),
                detail=detail,
                applicable=applicable,
            )
        )
        if not passed:
            raise ValidationFailure(
                f"{output} semantic validation failed: {invariant_id}: {detail}"
            )
    return tuple(evidence)
