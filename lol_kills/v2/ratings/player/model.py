"""Authenticated synthetic-development dynamic Bayesian Player Rating.

The public Player Rating remains unavailable: this module provides executable
mechanics evidence only.  Content hashes identify bytes and never authorize
production, publication, Reliability, probability wording, PASS-B2, or C2.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from collections.abc import Mapping as MappingABC
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import weakref

from lol_kills.v2.evaluation.checkpoint_c1 import (
    AUTHORITY_LOCATOR as C1_AUTHORITY_LOCATOR,
    load_checkpoint_c1,
)
from lol_kills.v2.evaluation.checks import ValidationFailure


ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = "data/lol/v2/models/player"
ROLES = ("top", "jungle", "mid", "bot", "support")
DISPLAY_ANCHOR = 1500.0
DISPLAY_LOGIT_SCALE = 400.0 / math.log(10.0)
DECAY_CANDIDATES = (
    "random_walk_no_reset",
    "mean_reversion",
    "patch_roster_shock",
    "season_shock",
    "calendar_boundary_shock",
    "full_reset",
)
RESOURCE_CANDIDATES = (
    "joint_resource_to_performance",
    "no_resource",
    "lagged_pre_map_policy",
    "player_policy_double_count_sensitivity",
)
SELECTABLE_RESOURCE_CANDIDATES = (
    "joint_resource_to_performance",
    "no_resource",
    "lagged_pre_map_policy",
)
CLAIM_CEILING = {
    "synthetic_only": True,
    "development_only": True,
    "production_eligible": False,
    "predictive_performance_authorized": False,
    "real_data_fit_authorized": False,
    "calibrated_public_probability_authorized": False,
    "empirical_95_coverage_authorized": False,
    "reliability_authorized": False,
    "probability_wording_authorized": False,
    "promotion_authorized": False,
    "publication_authorized": False,
    "sota_authorized": False,
    "pass_b2": False,
    "c2": False,
}
C1_EXPECTED = {
    "locator": C1_AUTHORITY_LOCATOR,
    "raw_sha256": "918d9d67ba4fc0b7567cdc05c4da6b84d1224b13195cbc3679a6a330f7cdd0a2",
    "artifact_id": "scryglass:c1:foundation-freeze-authority:v1",
    "schema_version": "checkpoint-c1-foundation-freeze-authority-v1",
    "decision_kind": "foundation_freeze",
    "authority_scope": "wave_1_foundation_freeze_only",
    "input_closure_sha256": "eee7b48ec055daf52da3c52a8ff4cd7a694f84134630e2e93de095cc505e3333",
}
INDEPENDENT_L4_AUTHORITY_PRESENT = False
POSTERIOR_PREDICTIVE_DOMAIN = {
    "safety_maximum_absolute_mean": 20.0,
    "safety_maximum_variance": 400.0,
    "guaranteed_maximum_absolute_mean": 5.0,
    "guaranteed_maximum_variance": 25.0,
    "successive_order_tolerance": 1e-7,
    "orders": (32, 64, 128, 256, 512, 1024, 2048, 4096),
}
FROZEN_SCHEMA_DIGESTS = {
    "player_schema": {
        "locator": "docs/model-v2/contracts/player-rating.schema.json",
        "raw_sha256": "af93fc0e143c47636641c071c7c1dd25e1438b86777ee0cb72c6f3b2aa78d25a",
        "canonical_sha256": "637d27182c84dba33770418944971e2202ef9d8568757a1812e0d300f9e3dc09",
    },
    "common_schema": {
        "locator": "docs/model-v2/contracts/common.schema.json",
        "raw_sha256": "1cc34d27a45bb2feb207c57ce0cad6ef75e954e1d67c1eb18702a0f4bd444e6a",
        "canonical_sha256": "b8fddd3b2ce3ef8717d8745a55ba89ce2b44d92e7225568ce775607a318d2ae0",
    },
    "provenance_schema": {
        "locator": "docs/model-v2/contracts/prediction-provenance.schema.json",
        "raw_sha256": "4f33fc9d3f39167aa7469d982926125f24337cd98a6fbb91084256756c0fcc2e",
        "canonical_sha256": "7169b18c435962d58886bb80fdebcfcc7b3f9c633fb4abf2b541e6c457987804",
    },
    "model_manifest_schema": {
        "locator": "docs/model-v2/contracts/model-manifest.schema.json",
        "raw_sha256": "3205e59d6430fb0c4819f4c1c268256795095c4fddf107761423b76614c2e319",
        "canonical_sha256": "3b2f9713bf9d344ea0626024d36d3651aa3aae5d7ea46bd9a5106935ccf9bb3f",
    },
}
INTERPRETATION = (
    "Current role-adjusted player contribution on the selected league-relative "
    "or globally bridged player scale, estimated from information available at "
    "as_of; League Rating is not included."
)

# 20-point Gauss-Hermite nodes and weights for exp(-x^2), mirrored.
_GH_NODES = (
    -5.387480890011233,
    -4.603682449550744,
    -3.944764040115625,
    -3.347854567383216,
    -2.78880605842813,
    -2.254974002089276,
    -1.738537712116586,
    -1.234076215395323,
    -0.737473728545394,
    -0.245340708300901,
    0.245340708300901,
    0.737473728545394,
    1.234076215395323,
    1.738537712116586,
    2.254974002089276,
    2.78880605842813,
    3.347854567383216,
    3.944764040115625,
    4.603682449550744,
    5.387480890011233,
)
_GH_WEIGHTS = (
    2.229393645534151e-13,
    4.39934099227318e-10,
    1.0860693707692817e-7,
    7.802556478532064e-6,
    0.000228338636016353,
    0.003243773342237861,
    0.02481052088746359,
    0.10901720602002332,
    0.2866755053628341,
    0.4622436696006101,
    0.4622436696006101,
    0.2866755053628341,
    0.10901720602002332,
    0.02481052088746359,
    0.003243773342237861,
    0.000228338636016353,
    7.802556478532064e-6,
    1.0860693707692817e-7,
    4.39934099227318e-10,
    2.229393645534151e-13,
)


def _fail(message: str) -> None:
    raise ValidationFailure(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _fail(f"non-finite JSON number: {value}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _thaw(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _parse_time(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail("timestamps must be UTC RFC3339 strings ending in Z")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(timezone.utc)
    except ValueError as exc:
        raise ValidationFailure(f"invalid timestamp: {value}") from exc


def _safe_file(root: Path, locator: str) -> tuple[Path, os.stat_result]:
    if not isinstance(locator, str) or "\\" in locator or locator.startswith("/"):
        _fail("artifact locator must be a relative POSIX path")
    raw_parts = locator.split("/")
    if not raw_parts or any(part in ("", ".", "..") for part in raw_parts):
        _fail(f"noncanonical artifact locator: {locator!r}")
    if locator != "/".join(raw_parts):
        _fail(f"noncanonical artifact locator: {locator!r}")
    root_real = root.resolve(strict=True)
    current = root
    leaf_stat: os.stat_result | None = None
    for part in raw_parts:
        current /= part
        try:
            leaf_stat = current.lstat()
        except FileNotFoundError as exc:
            raise ValidationFailure(f"missing artifact: {locator}") from exc
        if stat.S_ISLNK(leaf_stat.st_mode):
            _fail(f"symlink rejected: {locator}")
    assert leaf_stat is not None
    if current.resolve(strict=True).parent != (root_real / "/".join(raw_parts[:-1])).resolve(strict=True):
        _fail(f"path alias rejected: {locator}")
    if not stat.S_ISREG(leaf_stat.st_mode):
        _fail(f"artifact is not a regular file: {locator}")
    if leaf_stat.st_nlink != 1:
        _fail(f"hard-linked artifact rejected: {locator}")
    return current, leaf_stat


def _read_json(path: Path) -> tuple[Any, bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"invalid JSON: {path}") from exc
    return value, raw


@dataclass(frozen=True)
class Bundle:
    root: Path
    config: Mapping[str, Any]
    fixtures: Mapping[str, Any]
    report: Mapping[str, Any]
    manifest: Mapping[str, Any]
    candidate_identity: Mapping[str, Any]
    c1: Any
    raw_sha256: Mapping[str, str]
    selected_candidate_id: str | None


@dataclass(frozen=True)
class PlayerState:
    player_id: str
    role: str
    scope: str
    league_id: str | None
    mean: float
    variance: float
    last_available_at: str | None = None
    patch_id: str | None = None
    season_id: str | None = None
    calendar_year: int | None = None
    organization_id: str | None = None
    league_context_id: str | None = None
    information: float = 0.0
    source_contexts: tuple[str, ...] = ()
    context_event_start: str | None = None


@dataclass(frozen=True)
class ReplayResult:
    candidate_id: str
    decay_candidate_id: str
    resource_candidate_id: str
    as_of: str
    states: Mapping[tuple[str, str | None, str, str], PlayerState]
    policy_states: Mapping[tuple[str, str], float]
    forecasts: tuple[Mapping[str, Any], ...]
    covariances: Mapping[
        tuple[
            tuple[str, str | None, str, str],
            tuple[str, str | None, str, str],
        ],
        float,
    ] = field(default_factory=lambda: MappingProxyType({}))
    joint_outcome_updates: int = 0
    resource_evidence: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )
    replay_identity_sha256: str | None = None
    replay_input_sha256: str | None = None
    ordered_events_sha256: str | None = None
    replay_source_sha256: str | None = None
    target_context: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )


_ISSUED_RATINGS: weakref.WeakSet[Any] = weakref.WeakSet()
_ISSUED_BUNDLES: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}
_ISSUED_REPLAYS: dict[int, tuple[weakref.ReferenceType[Any], str]] = {}


class DevelopmentRating(MappingABC):
    """Loader-issued immutable development rating; never authorization."""

    __slots__ = ("_payload", "__weakref__")
    __hash__ = object.__hash__

    def __new__(cls, *args: Any, **kwargs: Any) -> "DevelopmentRating":
        raise TypeError("DevelopmentRating instances are loader-issued only")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("DevelopmentRating is read-only")

    def __getitem__(self, key: str) -> Any:
        return self._payload[key]

    def __iter__(self):
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)


def _issue_rating(payload: Mapping[str, Any]) -> DevelopmentRating:
    rating = object.__new__(DevelopmentRating)
    object.__setattr__(rating, "_payload", _freeze(payload))
    _ISSUED_RATINGS.add(rating)
    return rating


def _issue_bundle(bundle: Bundle) -> Bundle:
    _ISSUED_BUNDLES[id(bundle)] = (
        weakref.ref(bundle),
        bundle.raw_sha256["candidate_identity"],
    )
    return bundle


def _require_issued_bundle(bundle: Bundle) -> None:
    issued = _ISSUED_BUNDLES.get(id(bundle))
    if (
        issued is None
        or issued[0]() is not bundle
        or issued[1] != bundle.raw_sha256["candidate_identity"]
    ):
        _fail("bundle was not issued by the authenticated loader")


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        _fail(f"{name} keys differ from frozen development contract")


def _validate_development_manifest(
    manifest: Mapping[str, Any], report: Mapping[str, Any], hashes: Mapping[str, str]
) -> None:
    _exact_keys(
        manifest,
        {
            "artifact_id",
            "schema_version",
            "development_manifest_kind",
            "model_version",
            "principal_estimand",
            "selected_candidate_id",
            "selection_decision_sha256",
            "artifact_hashes",
            "c1_scope",
            "production_manifest_conformance",
            "candidate_identity_only",
            "independent_l4_authority_present",
            "registered_formulas",
            "claim_ceiling",
            "blockers",
        },
        "development manifest",
    )
    if manifest["development_manifest_kind"] != "l4_player_synthetic_development":
        _fail("manifest kind is not the frozen L4 development kind")
    if manifest["production_manifest_conformance"] is not False:
        _fail("synthetic manifest must not imply production-manifest conformance")
    if (
        manifest["candidate_identity_only"] is not True
        or manifest["independent_l4_authority_present"] is not False
    ):
        _fail("development manifest must expose absent independent authority")
    if _thaw(manifest["registered_formulas"]) != {
        "team_outcome_likelihood": "single Bernoulli logistic rank-one Laplace update",
        "design_vector": "+reference role weight blue; -reference role weight red",
        "covariance": "full rank-one roster covariance propagated under decay",
        "posterior_predictive": "adaptive Gauss-Hermite; guaranteed only in frozen central domain; larger safety-domain requests may be unavailable",
        "reference_policy_id": "equal-role-reference-policy-v1",
    }:
        _fail("development manifest formula registry mismatch")
    if _thaw(manifest["claim_ceiling"]) != CLAIM_CEILING:
        _fail("manifest claim ceiling mismatch")
    if manifest["selected_candidate_id"] != report["selected_candidate_id"]:
        _fail("manifest/report selection mismatch")
    if manifest["selection_decision_sha256"] != report["selection_decision_sha256"]:
        _fail("manifest/report decision identity mismatch")
    expected_hashes = {role: hashes[role] for role in ("config", "fixtures", "report")}
    if _thaw(manifest["artifact_hashes"]) != expected_hashes:
        _fail("manifest artifact closure is stale")
    if _thaw(manifest["c1_scope"]) != {
        "artifact_id": C1_EXPECTED["artifact_id"],
        "authority_scope": C1_EXPECTED["authority_scope"],
        "foundation_only": True,
    }:
        _fail("manifest C1 scope mismatch")
    if tuple(manifest["blockers"]) != tuple(report["blockers"]):
        _fail("manifest blockers mismatch")


def _validate_report_semantics(report: Mapping[str, Any]) -> None:
    _exact_keys(
        report,
        {
            "artifact_id",
            "schema_version",
            "evaluation_cutoff",
            "status",
            "selected_candidate_id",
            "selection_decision_sha256",
            "eligible_origin_count",
            "common_origin_ids",
            "common_origin_sha256",
            "verified_selectable_origin_sha256",
            "diagnostics",
            "interval_wording",
            "resource_causal_holding_equal_claim",
            "production_eligible",
            "claim_ceiling",
            "blockers",
        },
        "development report",
    )
    if report["interval_wording"] != "95% model range":
        _fail("development interval wording mismatch")
    if report["resource_causal_holding_equal_claim"] is not False:
        _fail("causal holding-resources-equal claims are forbidden")
    if report["production_eligible"] is not False:
        _fail("development report cannot be production eligible")
    if _thaw(report["claim_ceiling"]) != CLAIM_CEILING:
        _fail("development report claim ceiling mismatch")
    if tuple(report["blockers"]) != (
        "authoritative_observed_player_rows_unavailable",
        "production_player_rating_authority_unavailable",
        "independent_l6_l2_and_root_c2_review_required",
    ):
        _fail("development report blockers mismatch")


def _validate_candidate_identity_semantics(identity: Mapping[str, Any]) -> None:
    _exact_keys(
        identity,
        {
            "artifact_id",
            "schema_version",
            "identity_kind",
            "authorization_status",
            "independent_l4_authority_present",
            "external_authority_expected_sha256",
            "threat_model",
            "claim_ceiling",
            "c1_expected_identity",
            "expected_decay_candidates",
            "expected_resource_candidates",
            "required_blockers",
            "artifacts",
        },
        "candidate identity",
    )
    if identity["identity_kind"] != "l4_player_candidate_content_identity":
        _fail("candidate identity kind mismatch")
    if identity["authorization_status"] != "absent":
        _fail("candidate identity cannot authorize L4")
    if identity["independent_l4_authority_present"] is not False:
        _fail("independent L4 authority must remain absent")
    if identity["external_authority_expected_sha256"] is not None:
        _fail("no external L4 authority digest is registered")
    if _thaw(identity["threat_model"]) != {
        "honest_interpreter": "executing bytes must match this candidate identity",
        "fresh_import_of_attacker_modified_code": "outside local identity record",
        "authorization_requirement": "later independent C2/L2 registrar pins expected digest",
    }:
        _fail("candidate identity threat model mismatch")
    if _thaw(identity["claim_ceiling"]) != CLAIM_CEILING:
        _fail("candidate identity claim ceiling mismatch")
    if _thaw(identity["c1_expected_identity"]) != C1_EXPECTED:
        _fail("candidate identity C1 mismatch")
    if tuple(identity["expected_decay_candidates"]) != DECAY_CANDIDATES:
        _fail("candidate identity decay candidate set mismatch")
    if tuple(identity["expected_resource_candidates"]) != RESOURCE_CANDIDATES:
        _fail("candidate identity resource candidate set mismatch")
    expected_blockers = (
        "authoritative_observed_player_rows_unavailable",
        "production_player_rating_authority_unavailable",
        "independent_l6_l2_and_root_c2_review_required",
    )
    if tuple(identity["required_blockers"]) != expected_blockers:
        _fail("candidate identity blocker set mismatch")
    refs = identity["artifacts"]
    if not isinstance(refs, list):
        _fail("candidate identity artifacts must be a list")
    refs_by_role = {ref.get("role"): ref for ref in refs if isinstance(ref, Mapping)}
    schema_roles = set(FROZEN_SCHEMA_DIGESTS)
    if set(refs_by_role).intersection(schema_roles) != schema_roles:
        _fail("frozen schema role closure mismatch")
    for role, expected_schema in FROZEN_SCHEMA_DIGESTS.items():
        frozen_schema = refs_by_role[role]
        if any(
            frozen_schema.get(key) != value
            for key, value in expected_schema.items()
        ):
            _fail(f"frozen schema identity mismatch: {role}")


def load_bundle(root: Path = ROOT) -> Bundle:
    identity_path, identity_stat = _safe_file(
        root, f"{DATA_ROOT}/player-rating-candidate-identity.json"
    )
    identity, identity_raw = _read_json(identity_path)
    _validate_candidate_identity_semantics(identity)
    refs = identity["artifacts"]
    expected_roles = {
        "config",
        "fixtures",
        "report",
        "manifest",
        "package_source",
        "model_source",
        "generator_source",
        "tests",
        "player_schema",
        "common_schema",
        "provenance_schema",
        "model_manifest_schema",
        "c1_authority",
    }
    if not isinstance(refs, list) or {ref.get("role") for ref in refs} != expected_roles:
        _fail("authority artifact role closure mismatch")
    seen_inodes = {(identity_stat.st_dev, identity_stat.st_ino)}
    loaded: dict[str, Any] = {}
    hashes: dict[str, str] = {"candidate_identity": _sha256(identity_raw)}
    for ref in refs:
        _exact_keys(
            ref,
            {"role", "artifact_id", "locator", "media_type", "raw_sha256", "canonical_sha256"},
            "artifact reference",
        )
        role = ref["role"]
        path, leaf_stat = _safe_file(root, ref["locator"])
        inode = (leaf_stat.st_dev, leaf_stat.st_ino)
        if inode in seen_inodes:
            _fail("duplicate inode or source substitution rejected")
        seen_inodes.add(inode)
        raw = path.read_bytes()
        raw_hash = _sha256(raw)
        if ref["media_type"] == "application/json":
            value, _ = _read_json(path)
            canonical_hash = _sha256(_canonical_bytes(value))
            loaded[role] = value
            if value.get("artifact_id", ref["artifact_id"]) != ref["artifact_id"]:
                _fail(f"artifact identity mismatch: {role}")
        elif ref["media_type"] == "text/x-python":
            value = None
            canonical_hash = raw_hash
        else:
            _fail(f"unsupported artifact media type: {role}")
        if raw_hash != ref["raw_sha256"] or canonical_hash != ref["canonical_sha256"]:
            _fail(f"artifact hash mismatch: {role}")
        hashes[role] = raw_hash
    executing_paths = {
        "model_source": Path(__file__).resolve(),
        "package_source": Path(__file__).with_name("__init__.py").resolve(),
        "generator_source": Path(__file__).with_name("generate_artifacts.py").resolve(),
    }
    refs_by_role = {ref["role"]: ref for ref in refs}
    for role, executing_path in executing_paths.items():
        if _sha256(executing_path.read_bytes()) != refs_by_role[role]["raw_sha256"]:
            _fail(f"executing module digest mismatch: {role}")
    schema_paths = {
        "player_schema": root / "docs/model-v2/contracts/player-rating.schema.json",
        "common_schema": root / "docs/model-v2/contracts/common.schema.json",
        "provenance_schema": root / "docs/model-v2/contracts/prediction-provenance.schema.json",
        "model_manifest_schema": root / "docs/model-v2/contracts/model-manifest.schema.json",
    }
    for role, schema_path in schema_paths.items():
        if _sha256(schema_path.read_bytes()) != refs_by_role[role]["raw_sha256"]:
            _fail(f"executing schema digest mismatch: {role}")
    c1 = load_checkpoint_c1(root)
    c1_payload = c1.authenticate()
    for key in ("artifact_id", "schema_version", "decision_kind", "authority_scope", "input_closure_sha256"):
        if c1_payload[key] != C1_EXPECTED[key]:
            _fail(f"C1 authenticated identity mismatch: {key}")
    if _thaw(c1_payload["claim_boundary"]) != {
        "pass_b2": False,
        "probability_wording_authorized": False,
        "production_authority": False,
        "promotion_decision": None,
        "publication_authorized": False,
        "real_data_evidence": False,
        "reliability_authorized": False,
        "sealed_decision_opened": False,
        "sota_authorized": False,
    }:
        _fail("C1 claim boundary mismatch")
    config = loaded["config"]
    fixtures = loaded["fixtures"]
    report = loaded["report"]
    manifest = loaded["manifest"]
    _validate_config(config)
    _validate_rows(fixtures["players"], fixtures["matches"], config)
    _validate_report_semantics(report)
    _validate_development_manifest(manifest, report, hashes)
    comparison = compare_candidates(config, fixtures, report["evaluation_cutoff"])
    if _thaw(comparison) != {
        "status": report["status"],
        "selected_candidate_id": report["selected_candidate_id"],
        "selection_decision_sha256": report["selection_decision_sha256"],
        "eligible_origin_count": report["eligible_origin_count"],
        "common_origin_ids": _thaw(report["common_origin_ids"]),
        "common_origin_sha256": report["common_origin_sha256"],
        "verified_selectable_origin_sha256": report[
            "verified_selectable_origin_sha256"
        ],
        "diagnostics": _thaw(report["diagnostics"]),
    }:
        _fail("persisted selection report differs from deterministic replay")
    return _issue_bundle(Bundle(
        root=root,
        config=_freeze(config),
        fixtures=_freeze(fixtures),
        report=_freeze(report),
        manifest=_freeze(manifest),
        candidate_identity=_freeze(identity),
        c1=c1,
        raw_sha256=_freeze(hashes),
        selected_candidate_id=report["selected_candidate_id"],
    ))


def load_authorized_bundle(
    root: Path = ROOT, *, external_authority_sha256: str | None = None
) -> Bundle:
    """Fail closed until an independent registrar pins an L4 authority digest."""
    bundle = load_bundle(root)
    expected = bundle.candidate_identity["external_authority_expected_sha256"]
    if (
        bundle.candidate_identity["independent_l4_authority_present"] is not True
        or expected is None
        or external_authority_sha256 != expected
    ):
        _fail("independent L4 authority is absent")
    return bundle


def _validate_config(config: Mapping[str, Any]) -> None:
    if config["rating_display"] != {"anchor": 1500, "logistic_scale": 400}:
        _fail("display contract must remain 1500/400")
    if config["league_rating_included"] is not False:
        _fail("League Rating cannot be player skill")
    if tuple(item["candidate_id"] for item in config["decay_candidates"]) != DECAY_CANDIDATES:
        _fail("decay candidate order/set mismatch")
    if tuple(item["candidate_id"] for item in config["resource_candidates"]) != RESOURCE_CANDIDATES:
        _fail("resource candidate order/set mismatch")
    if (
        config["selection"].get("require_complete_candidate_origin_identity")
        is not True
    ):
        _fail("candidate comparison must require complete origin identity")
    if _thaw(config["claim_ceiling"]) != CLAIM_CEILING:
        _fail("config claim ceiling mismatch")
    if _thaw(config["posterior_predictive"]) != {
        "integration": "adaptive_gauss_hermite",
        "orders": list(POSTERIOR_PREDICTIVE_DOMAIN["orders"]),
        "successive_order_tolerance": POSTERIOR_PREDICTIVE_DOMAIN[
            "successive_order_tolerance"
        ],
        "safety_maximum_absolute_mean": POSTERIOR_PREDICTIVE_DOMAIN[
            "safety_maximum_absolute_mean"
        ],
        "safety_maximum_variance": POSTERIOR_PREDICTIVE_DOMAIN[
            "safety_maximum_variance"
        ],
        "guaranteed_convergence_domain": {
            "maximum_absolute_mean": POSTERIOR_PREDICTIVE_DOMAIN[
                "guaranteed_maximum_absolute_mean"
            ],
            "maximum_variance": POSTERIOR_PREDICTIVE_DOMAIN[
                "guaranteed_maximum_variance"
            ],
        },
        "outside_guaranteed_domain_semantics": "attempt adaptive integration within safety bounds; return unavailable on nonconvergence",
        "evaluated_extreme_counterexamples": [
            {"mean": 5.0, "variance": 100.0},
            {"mean": 20.0, "variance": 400.0},
        ],
        "failure_policy": "unavailable",
        "covariance_assumption": "full registered rank-one roster covariance plus independent resource-state covariance",
    }:
        _fail("posterior-predictive accuracy domain mismatch")
    if _thaw(config["team_outcome_anchor"]) != {
        "anchor_id": "binary-team-outcome-player-anchor-v1",
        "likelihood": "single Bernoulli logistic roster likelihood",
        "design_vector": "+reference_role_weight for blue and -reference_role_weight for red",
        "approximation": "one-step Laplace assumed-density rank-one Gaussian update",
        "information_rank_per_match": 1,
        "minimum_curvature": 1e-9,
        "covariance_update": "Sigma_post = Sigma - W/(1+W*xT*Sigma*x)*(Sigma*x)*(Sigma*x)T",
        "availability_rule": "outcome_available_at must be at or before replay cutoff and strictly after serialized forecast",
    }:
        _fail("team-outcome likelihood registry mismatch")
    channels = config["auxiliary_channels"]
    required_channel_fields = {
        "channel_id",
        "provenance_id",
        "transform",
        "unit",
        "timing",
        "availability_rule",
        "missingness",
        "coverage_rule",
        "leakage_rule",
        "routing",
        "resource_channel",
    }
    if any(set(channel) != required_channel_fields for channel in channels):
        _fail("auxiliary channel provenance contract incomplete")
    if not any(not channel["resource_channel"] and channel["routing"] == "skill" for channel in channels):
        _fail("at least one non-resource impact channel is required")


def _validate_rows(
    players: Sequence[Mapping[str, Any]],
    matches: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    player_ids = {player["player_id"] for player in players}
    if len(player_ids) != len(players):
        _fail("fixture player IDs must be unique")
    channels = {channel["channel_id"]: channel for channel in config["auxiliary_channels"]}
    for player in players:
        if (
            not isinstance(player.get("player_id"), str)
            or not player["player_id"]
            or player.get("role") not in ROLES
        ):
            _fail("player identity and role must be registered")
        eligibility = player["global_eligibility"]
        if set(eligibility) != {
            "version",
            "valid_from",
            "valid_until",
            "international_eligible",
            "connectivity_supported",
            "bridge_support_count",
            "bridge_component_id",
            "current_league_tier",
            "active_status",
            "roster_membership",
            "roster_ambiguous",
        }:
            _fail("global eligibility must be versioned and structurally explicit")
        tier = eligibility["current_league_tier"]
        bridge_count = eligibility["bridge_support_count"]
        if type(tier) is not int or tier not in (1, 2, 3):
            _fail("global eligibility tier must be an exact registered integer")
        if type(bridge_count) is not int or bridge_count < 0:
            _fail("bridge_support_count must be a nonnegative exact integer")
        if eligibility["active_status"] not in {"active", "inactive"}:
            _fail("global eligibility active_status is invalid")
        if eligibility["roster_membership"] not in {"main", "substitute"}:
            _fail("global eligibility roster_membership is invalid")
        if type(eligibility["roster_ambiguous"]) is not bool:
            _fail("global eligibility roster_ambiguous must be boolean")
        if (
            type(eligibility["international_eligible"]) is not bool
            or type(eligibility["connectivity_supported"]) is not bool
        ):
            _fail("global eligibility structural flags must be exact booleans")
        if (
            not isinstance(eligibility["version"], str)
            or not eligibility["version"]
            or not isinstance(eligibility["bridge_component_id"], str)
            or not eligibility["bridge_component_id"]
        ):
            _fail("global eligibility version and bridge component must be nonempty strings")
        valid_from = _parse_time(eligibility["valid_from"])
        if (
            eligibility["valid_until"] is not None
            and _parse_time(eligibility["valid_until"]) <= valid_from
        ):
            _fail("global eligibility validity window is invalid")
    players_by_id = {player["player_id"]: player for player in players}
    match_ids: set[str] = set()
    for match in matches:
        if match["match_id"] in match_ids:
            _fail("match IDs must be unique")
        match_ids.add(match["match_id"])
        start = _parse_time(match["event_start"])
        end = _parse_time(match["event_end"])
        outcome = _parse_time(match["outcome_available_at"])
        if not start < end <= outcome:
            _fail("event_start < event_end <= outcome_available_at is required")
        if (
            not isinstance(match.get("season_id"), str)
            or not match["season_id"]
            or type(match.get("calendar_year")) is not int
            or match["calendar_year"] != start.year
        ):
            _fail("match temporal context must be authoritative and match event_start")
        if type(match.get("blue_win")) is not int or match["blue_win"] not in (0, 1):
            _fail("blue_win must be exactly integer 0 or 1")
        if type(match["league_tier"]) is not int or match["league_tier"] not in (1, 2, 3):
            _fail("league_tier must be registered")
        roster_ids: set[str] = set()
        for side in ("blue", "red"):
            roster = match["rosters"][side]
            if set(roster) != set(ROLES):
                _fail("each roster requires exactly one player per role")
            ids = list(roster.values())
            if len(set(ids)) != len(ids) or any(player_id not in player_ids for player_id in ids):
                _fail("rosters require distinct registered players")
            if roster_ids.intersection(ids):
                _fail("a player cannot appear on both sides")
            for role, player_id in roster.items():
                registered_role = next(
                    player["role"] for player in players if player["player_id"] == player_id
                )
                if registered_role != role:
                    _fail("player metadata role does not match exact roster role")
            roster_ids.update(ids)
        for update in match.get("player_updates", []):
            if set(update) != {
                "player_id",
                "channel_id",
                "feature_provenance_id",
                "value",
                "observation_variance",
                "available_at",
                "source_context",
            }:
                _fail("player feature row fields differ from frozen contract")
            if update["player_id"] not in roster_ids:
                _fail("player update must belong to the exact match roster")
            channel = channels.get(update["channel_id"])
            if channel is None or channel["routing"] != "skill" or channel["resource_channel"]:
                _fail("only registered non-resource skill channels can update player skill")
            if update["feature_provenance_id"] != channel["provenance_id"]:
                _fail("player feature provenance mismatch")
            value = update["value"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                _fail("player feature value must be finite")
            variance = update["observation_variance"]
            if (
                isinstance(variance, bool)
                or not isinstance(variance, (int, float))
                or not math.isfinite(variance)
                or variance <= 0
            ):
                _fail("observation_variance must be finite and strictly positive")
            if _parse_time(update["available_at"]) < end:
                _fail("post-map player measurements cannot predate event_end")
        for update in match.get("policy_updates", []):
            if set(update) != {
                "player_id",
                "channel_id",
                "feature_provenance_id",
                "value",
                "observation_variance",
                "available_at",
            }:
                _fail("policy feature row fields differ from frozen contract")
            channel = channels.get(update["channel_id"])
            if channel is None or channel["routing"] != "policy_only" or not channel["resource_channel"]:
                _fail("same-map resources must route only to policy")
            if update["feature_provenance_id"] != channel["provenance_id"]:
                _fail("policy feature provenance mismatch")
            if update["player_id"] not in roster_ids:
                _fail("policy update must belong to the exact match roster")
            policy_eligibility = players_by_id[update["player_id"]]["global_eligibility"]
            policy_at = _parse_time(match["event_start"])
            policy_window = _parse_time(policy_eligibility["valid_from"]) <= policy_at and (
                policy_eligibility["valid_until"] is None
                or policy_at < _parse_time(policy_eligibility["valid_until"])
            )
            if (
                not policy_window
                or policy_eligibility["active_status"] != "active"
                or policy_eligibility["roster_membership"] != "main"
                or policy_eligibility["roster_ambiguous"] is not False
            ):
                _fail("policy update requires an exact active unambiguous main-roster player")
            value = update["value"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                _fail("policy feature value must be finite")
            variance = update["observation_variance"]
            if (
                isinstance(variance, bool)
                or not isinstance(variance, (int, float))
                or not math.isfinite(variance)
                or variance <= 0
            ):
                _fail("observation_variance must be finite and strictly positive")
            if _parse_time(update["available_at"]) < end:
                _fail("post-map resource measurements cannot predate event_end")


def latent_to_rating(latent: float) -> float:
    return DISPLAY_ANCHOR + DISPLAY_LOGIT_SCALE * latent


def rating_to_latent(rating: float) -> float:
    return (rating - DISPLAY_ANCHOR) / DISPLAY_LOGIT_SCALE


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def plugin_expected_result(logit_mean: float) -> float:
    return _sigmoid(logit_mean)


def posterior_predictive_expected_result(logit_mean: float, logit_variance: float) -> float:
    """Integrate adaptively; larger safe-domain requests may be unavailable."""
    if (
        isinstance(logit_variance, bool)
        or not isinstance(logit_variance, (int, float))
        or logit_variance < 0
        or not math.isfinite(logit_variance)
    ):
        _fail("posterior variance must be finite and nonnegative")
    if (
        isinstance(logit_mean, bool)
        or not isinstance(logit_mean, (int, float))
        or not math.isfinite(logit_mean)
        or abs(logit_mean)
        > POSTERIOR_PREDICTIVE_DOMAIN["safety_maximum_absolute_mean"]
        or logit_variance
        > POSTERIOR_PREDICTIVE_DOMAIN["safety_maximum_variance"]
    ):
        _fail("posterior-predictive request is outside the manifested safety domain")
    if logit_variance == 0:
        return _sigmoid(logit_mean)
    try:
        import numpy as np
        from scipy.special import roots_hermitenorm
    except ImportError as exc:
        raise ValidationFailure("adaptive Gauss-Hermite runtime unavailable") from exc
    previous: float | None = None
    for order in POSTERIOR_PREDICTIVE_DOMAIN["orders"]:
        nodes, weights = roots_hermitenorm(order)
        logits = logit_mean + math.sqrt(logit_variance) * nodes
        values = np.empty_like(logits)
        positive = logits >= 0
        values[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponent = np.exp(logits[~positive])
        values[~positive] = exponent / (1.0 + exponent)
        estimate = float(np.dot(weights, values) / math.sqrt(2.0 * math.pi))
        if (
            previous is not None
            and abs(estimate - previous)
            <= POSTERIOR_PREDICTIVE_DOMAIN["successive_order_tolerance"]
        ):
            return estimate
        previous = estimate
    _fail("adaptive Gauss-Hermite integration did not converge")


def _candidate_parts(candidate_id: str) -> tuple[str, str]:
    parts = candidate_id.split("+")
    if len(parts) != 2 or parts[0] not in DECAY_CANDIDATES or parts[1] not in RESOURCE_CANDIDATES:
        _fail(f"unknown model candidate: {candidate_id}")
    return parts[0], parts[1]


def _transition_factor(
    state: PlayerState,
    candidate: Mapping[str, Any],
    boundary: datetime,
    *,
    patch_id: str | None = None,
    season_id: str | None = None,
    calendar_year: int | None = None,
    organization_id: str | None = None,
    league_id: str | None = None,
    event_start: str | None = None,
) -> tuple[float, float]:
    if state.last_available_at is None:
        return 1.0, 0.0
    elapsed_days = max(
        0.0, (boundary - _parse_time(state.last_available_at)).total_seconds() / 86400.0
    )
    phi = 1.0
    kind = candidate["kind"]
    if kind == "mean_reversion":
        phi = math.exp(-math.log(2.0) * elapsed_days / candidate["half_life_days"])
    elif kind == "patch_roster_shock":
        if patch_id is not None and state.patch_id is not None and patch_id != state.patch_id:
            phi *= candidate["patch_retention"]
        if (
            organization_id is not None
            and state.organization_id is not None
            and organization_id != state.organization_id
        ):
            phi *= candidate["roster_retention"]
    elif kind == "season_shock":
        if season_id is not None and state.season_id is not None and season_id != state.season_id:
            phi = candidate["retention"]
    elif kind == "calendar_boundary_shock":
        if (
            calendar_year is not None
            and state.calendar_year is not None
            and calendar_year != state.calendar_year
        ):
            phi = candidate["retention"]
    elif kind == "full_reset":
        if (
            candidate["reset_boundary"] == "season_change"
            and season_id is not None
            and state.season_id is not None
            and season_id != state.season_id
        ):
            phi = 0.0
    return phi, elapsed_days


def _transition(
    state: PlayerState,
    candidate: Mapping[str, Any],
    boundary: datetime,
    *,
    patch_id: str | None = None,
    season_id: str | None = None,
    calendar_year: int | None = None,
    organization_id: str | None = None,
    league_id: str | None = None,
    event_start: str | None = None,
    advance_context: bool = False,
) -> PlayerState:
    phi, elapsed_days = _transition_factor(
        state,
        candidate,
        boundary,
        patch_id=patch_id,
        season_id=season_id,
        calendar_year=calendar_year,
        organization_id=organization_id,
        league_id=league_id,
    )
    variance = phi * phi * state.variance + candidate["process_variance_per_day"] * elapsed_days
    changes: dict[str, Any] = {
        "mean": phi * state.mean,
        "variance": max(candidate["minimum_variance"], variance),
        "league_context_id": league_id or state.league_context_id,
    }
    if advance_context:
        changes.update(
            {
                "last_available_at": event_start
                or boundary.isoformat().replace("+00:00", "Z"),
                "patch_id": patch_id if patch_id is not None else state.patch_id,
                "season_id": season_id,
                "calendar_year": calendar_year,
                "organization_id": organization_id
                if organization_id is not None
                else state.organization_id,
                "context_event_start": event_start
                or boundary.isoformat().replace("+00:00", "Z"),
            }
        )
    return replace(state, **changes)


def _gaussian_update(
    state: PlayerState,
    value: float,
    observation_variance: float,
    available_at: str,
    context: Mapping[str, Any],
    source_context: str,
    *,
    advance_context: bool = True,
) -> PlayerState:
    gain = state.variance / (state.variance + observation_variance)
    context_changes = (
        {
            "patch_id": context["patch_id"],
            "season_id": context["season_id"],
            "calendar_year": context["calendar_year"],
            "organization_id": context["organization_id"],
            "league_context_id": context["league_id"],
            "context_event_start": context["event_start"],
        }
        if advance_context
        else {}
    )
    return replace(
        state,
        mean=state.mean + gain * (value - state.mean),
        variance=(1.0 - gain) * state.variance,
        last_available_at=available_at,
        information=state.information + 1.0 / observation_variance,
        source_contexts=tuple(sorted(set(state.source_contexts) | {source_context})),
        **context_changes,
    )


def _eligible_global(
    player: Mapping[str, Any],
    at: datetime,
    config: Mapping[str, Any],
    *,
    league_tier: int | None = None,
) -> bool:
    eligibility = player["global_eligibility"]
    valid_until = eligibility["valid_until"]
    in_window = _parse_time(eligibility["valid_from"]) <= at and (
        valid_until is None or at < _parse_time(valid_until)
    )
    return (
        in_window
        and (league_tier if league_tier is not None else eligibility["current_league_tier"])
        == 1
        and eligibility["current_league_tier"] == 1
        and eligibility["active_status"] == "active"
        and eligibility["roster_membership"] == "main"
        and eligibility["roster_ambiguous"] is False
        and eligibility["international_eligible"]
        and eligibility["connectivity_supported"]
        and eligibility["bridge_support_count"]
        >= config["global_bridge"]["minimum_bridge_support"]
    )


def _new_state(
    player_id: str,
    role: str,
    scope: str,
    league_id: str | None,
    prior_variance: float,
    carry: PlayerState | None = None,
    carry_variance: float = 0.0,
    carry_mean_adjustment: float = 0.0,
) -> PlayerState:
    return PlayerState(
        player_id=player_id,
        role=role,
        scope=scope,
        league_id=league_id,
        mean=0.0 if carry is None else carry.mean + carry_mean_adjustment,
        variance=prior_variance if carry is None else carry.variance + carry_variance,
        last_available_at=None if carry is None else carry.last_available_at,
        patch_id=None if carry is None else carry.patch_id,
        season_id=None if carry is None else carry.season_id,
        calendar_year=None if carry is None else carry.calendar_year,
        organization_id=None if carry is None else carry.organization_id,
        league_context_id=None if carry is None else carry.league_context_id,
        information=0.0 if carry is None else carry.information,
        source_contexts=() if carry is None else carry.source_contexts,
        context_event_start=None if carry is None else carry.context_event_start,
    )


def _validate_target_context(
    as_of: str,
    rows: Sequence[Mapping[str, Any]],
    target_context: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    cutoff = _parse_time(as_of)
    if target_context is None:
        eligible = [
            row
            for row in rows
            if _parse_time(row["event_start"]) <= cutoff
            and row["calendar_year"] == cutoff.year
        ]
        if not eligible:
            _fail("authoritative target temporal context is unavailable")
        latest = max(eligible, key=lambda row: (row["event_start"], row["match_id"]))
        resolved = {
            "season_id": latest["season_id"],
            "calendar_year": latest["calendar_year"],
        }
    else:
        resolved = dict(target_context)
    if set(resolved) != {"season_id", "calendar_year"}:
        _fail("target context requires exactly season_id and calendar_year")
    if (
        not isinstance(resolved["season_id"], str)
        or not resolved["season_id"]
        or type(resolved["calendar_year"]) is not int
        or resolved["calendar_year"] != cutoff.year
    ):
        _fail("target temporal context is invalid or inconsistent with as_of")
    return _freeze(resolved)


def replay(
    config: Mapping[str, Any],
    fixtures: Mapping[str, Any],
    candidate_id: str,
    as_of: str,
    *,
    matches: Sequence[Mapping[str, Any]] | None = None,
    target_context: Mapping[str, Any] | None = None,
) -> ReplayResult:
    decay_id, resource_id = _candidate_parts(candidate_id)
    rows = list(matches if matches is not None else fixtures["matches"])
    _validate_rows(fixtures["players"], rows, config)
    cutoff = _parse_time(as_of)
    resolved_target_context = _validate_target_context(as_of, rows, target_context)
    decay = {item["candidate_id"]: item for item in config["decay_candidates"]}[decay_id]
    resource = {item["candidate_id"]: item for item in config["resource_candidates"]}[resource_id]
    players = {player["player_id"]: player for player in fixtures["players"]}
    role_weights = config["reference_policy"]["role_weights"]
    prior_variance = config["prior"]["variance"]
    master: dict[tuple[str, str], PlayerState] = {}
    states: dict[tuple[str, str | None, str, str], PlayerState] = {}
    policy: dict[tuple[str, str], float] = {}
    joint_policy: dict[tuple[str, str], tuple[float, float, int]] = {}
    covariances: dict[
        tuple[
            tuple[str, str | None, str, str],
            tuple[str, str | None, str, str],
        ],
        float,
    ] = {}
    pending: list[tuple[datetime, int, str, Mapping[str, Any]]] = []
    sequence = 0
    joint_outcome_updates = 0
    resource_observation_count = 0
    resource_observation_count_by_player: dict[str, int] = {}

    def context_for(match: Mapping[str, Any], player_id: str) -> dict[str, Any]:
        side = "blue" if player_id in match["rosters"]["blue"].values() else "red"
        return {
            "patch_id": match["patch_id"],
            "season_id": match["season_id"],
            "calendar_year": match["calendar_year"],
            "organization_id": match["organization_ids"][side],
            "league_id": match["league_id"],
            "event_start": match["event_start"],
        }

    def context_advances(state: PlayerState, ctx: Mapping[str, Any]) -> bool:
        return (
            state.context_event_start is None
            or _parse_time(ctx["event_start"]) >= _parse_time(state.context_event_start)
        )

    def safe_transition(
        state: PlayerState,
        boundary: datetime,
        ctx: Mapping[str, Any],
    ) -> tuple[PlayerState, bool]:
        advances = context_advances(state, ctx)
        effective = (
            ctx
            if advances
            else {
                "patch_id": state.patch_id,
                "season_id": state.season_id,
                "calendar_year": state.calendar_year,
                "organization_id": state.organization_id,
                "league_id": state.league_context_id,
                "event_start": state.context_event_start,
            }
        )
        return _transition(state, decay, boundary, **effective), advances

    def safe_transition_factor(
        state: PlayerState,
        boundary: datetime,
        ctx: Mapping[str, Any],
    ) -> float:
        effective = (
            ctx
            if context_advances(state, ctx)
            else {
                "patch_id": state.patch_id,
                "season_id": state.season_id,
                "calendar_year": state.calendar_year,
                "organization_id": state.organization_id,
                "league_id": state.league_context_id,
                "event_start": state.context_event_start,
            }
        )
        return _transition_factor(state, decay, boundary, **effective)[0]

    def ensure_states(match: Mapping[str, Any], player_id: str) -> tuple[PlayerState, PlayerState, PlayerState | None]:
        role = players[player_id]["role"]
        ctx = context_for(match, player_id)
        master_key = (role, player_id)
        base = master.get(master_key, _new_state(player_id, role, "player", None, prior_variance))
        league_key = ("league", match["league_id"], role, player_id)
        league_state = states.get(
            league_key,
            _new_state(
                player_id,
                role,
                "league",
                match["league_id"],
                prior_variance,
                carry=base if base.last_available_at else None,
                carry_variance=config["transfer"]["cross_league_variance"],
                carry_mean_adjustment=-config["league_centers"].get(
                    match["league_id"], 0.0
                ),
            ),
        )
        global_state: PlayerState | None = None
        if _eligible_global(
            players[player_id],
            _parse_time(match["event_start"]),
            config,
            league_tier=match["league_tier"],
        ):
            global_key = ("global", None, role, player_id)
            global_state = states.get(
                global_key,
                _new_state(
                    player_id,
                    role,
                    "global",
                    None,
                    prior_variance + config["global_bridge"]["bridge_variance"],
                ),
            )
        return base, league_state, global_state

    def covariance_key(
        left: tuple[str, str | None, str, str],
        right: tuple[str, str | None, str, str],
    ):
        return (left, right) if left <= right else (right, left)

    def shrink_covariances(
        key: tuple[str, str | None, str, str], factor: float
    ) -> None:
        for pair in list(covariances):
            if key in pair:
                covariances[pair] *= factor

    def joint_roster_update(
        match: Mapping[str, Any],
        available: datetime,
        scope: str,
    ) -> None:
        entries: list[
            tuple[
                tuple[str, str | None, str, str],
                PlayerState,
                float,
                Mapping[str, Any],
            ]
        ] = []
        for side, sign in (("blue", 1.0), ("red", -1.0)):
            for role, player_id in match["rosters"][side].items():
                base, league_state, global_state = ensure_states(match, player_id)
                ctx = context_for(match, player_id)
                if scope == "league":
                    key = ("league", match["league_id"], role, player_id)
                    state = league_state
                else:
                    if global_state is None:
                        continue
                    key = ("global", None, role, player_id)
                    state = global_state
                effective_ctx = (
                    ctx
                    if context_advances(state, ctx)
                    else {
                        "patch_id": state.patch_id,
                        "season_id": state.season_id,
                        "calendar_year": state.calendar_year,
                        "organization_id": state.organization_id,
                        "league_id": state.league_context_id,
                        "event_start": state.context_event_start,
                    }
                )
                phi, _ = _transition_factor(state, decay, available, **effective_ctx)
                shrink_covariances(key, phi)
                state, _ = safe_transition(state, available, ctx)
                states[key] = state
                entries.append((key, state, sign * role_weights[role], ctx))
        if not entries:
            return
        size = len(entries)
        means = [entry[1].mean for entry in entries]
        design = [entry[2] for entry in entries]
        sigma = [[0.0] * size for _ in range(size)]
        for i, (key_i, state_i, _, _) in enumerate(entries):
            for j, (key_j, _, _, _) in enumerate(entries):
                sigma[i][j] = (
                    state_i.variance
                    if i == j
                    else covariances.get(covariance_key(key_i, key_j), 0.0)
                )
        sigma_x = [
            sum(sigma[i][j] * design[j] for j in range(size)) for i in range(size)
        ]
        x_sigma_x = sum(design[i] * sigma_x[i] for i in range(size))
        eta = sum(design[i] * means[i] for i in range(size))
        probability = _sigmoid(eta)
        curvature = max(
            config["team_outcome_anchor"]["minimum_curvature"],
            probability * (1.0 - probability),
        )
        denominator = 1.0 + curvature * x_sigma_x
        residual = float(match["blue_win"]) - probability
        updated_sigma = [
            [
                sigma[i][j]
                - curvature / denominator * sigma_x[i] * sigma_x[j]
                for j in range(size)
            ]
            for i in range(size)
        ]
        for i, (key, state, _, ctx) in enumerate(entries):
            advances = context_advances(state, ctx)
            updated = replace(
                state,
                mean=state.mean + sigma_x[i] * residual / denominator,
                variance=max(decay["minimum_variance"], updated_sigma[i][i]),
                last_available_at=match["outcome_available_at"],
                source_contexts=tuple(
                    sorted(set(state.source_contexts) | {"team_outcome_anchor"})
                ),
                **(
                    {
                        "patch_id": ctx["patch_id"],
                        "season_id": ctx["season_id"],
                        "calendar_year": ctx["calendar_year"],
                        "organization_id": ctx["organization_id"],
                        "league_context_id": ctx["league_id"],
                        "context_event_start": ctx["event_start"],
                    }
                    if advances
                    else {}
                ),
            )
            states[key] = updated
            if scope == "league":
                role, player_id = key[2], key[3]
                center = config["league_centers"].get(match["league_id"], 0.0)
                master[(role, player_id)] = replace(
                    updated,
                    scope="player",
                    league_id=None,
                    mean=updated.mean + center,
                )
        for i, (key_i, _, _, _) in enumerate(entries):
            for j in range(i + 1, size):
                key_j = entries[j][0]
                covariances[covariance_key(key_i, key_j)] = updated_sigma[i][j]

    def apply_observation(item: Mapping[str, Any], available: datetime) -> None:
        nonlocal master, states, policy
        nonlocal joint_outcome_updates, resource_observation_count
        match = item["match"]
        kind = item["kind"]
        if kind == "policy":
            update = item["update"]
            policy_key = (update["player_id"], update["channel_id"])
            policy[policy_key] = update["value"]
            resource_observation_count += 1
            resource_observation_count_by_player[update["player_id"]] = (
                resource_observation_count_by_player.get(update["player_id"], 0) + 1
            )
            if resource["mechanics"] in {
                "joint_latent_resource_performance",
                "double_count_collider_sensitivity",
            }:
                prior_mean, prior_variance_policy, prior_count = joint_policy.get(
                    policy_key,
                    (0.0, resource["policy_prior_variance"], 0),
                )
                observation_variance = update["observation_variance"]
                gain = prior_variance_policy / (
                    prior_variance_policy + observation_variance
                )
                joint_policy[policy_key] = (
                    prior_mean + gain * (update["value"] - prior_mean),
                    (1.0 - gain) * prior_variance_policy,
                    prior_count + 1,
                )
            if resource["double_count_skill_sensitivity"]:
                player_id = update["player_id"]
                role = players[player_id]["role"]
                base, league_state, _ = ensure_states(match, player_id)
                ctx = context_for(match, player_id)
                base, base_advances = safe_transition(base, available, ctx)
                league_state, league_advances = safe_transition(
                    league_state, available, ctx
                )
                sensitivity_value = resource["skill_sensitivity_coefficient"] * update["value"]
                master[(role, player_id)] = _gaussian_update(
                    base,
                    sensitivity_value,
                    resource["skill_sensitivity_variance"],
                    update["available_at"],
                    ctx,
                    "resource_double_count_sensitivity",
                    advance_context=base_advances,
                )
                states[("league", match["league_id"], role, player_id)] = _gaussian_update(
                    league_state,
                    sensitivity_value,
                    resource["skill_sensitivity_variance"],
                    update["available_at"],
                    ctx,
                    "resource_double_count_sensitivity",
                    advance_context=league_advances,
                )
            return
        if kind == "team_anchor":
            joint_roster_update(match, available, "league")
            joint_outcome_updates += 1
            return
        update = item["update"]
        player_id = update["player_id"]
        value = update["value"]
        obs_variance = update["observation_variance"]
        available_at = update["available_at"]
        source_context = update["source_context"]
        role = players[player_id]["role"]
        ctx = context_for(match, player_id)
        base, league_state, global_state = ensure_states(match, player_id)
        base, base_advances = safe_transition(base, available, ctx)
        league_phi = safe_transition_factor(league_state, available, ctx)
        shrink_covariances(
            ("league", match["league_id"], role, player_id), league_phi
        )
        league_state, league_advances = safe_transition(league_state, available, ctx)
        league_value = value - config["league_centers"].get(match["league_id"], 0.0)
        master[(role, player_id)] = _gaussian_update(
            base,
            value,
            obs_variance,
            available_at,
            ctx,
            source_context,
            advance_context=base_advances,
        )
        league_key = ("league", match["league_id"], role, player_id)
        league_gain = league_state.variance / (
            league_state.variance + obs_variance
        )
        shrink_covariances(league_key, 1.0 - league_gain)
        states[league_key] = _gaussian_update(
            league_state,
            league_value,
            obs_variance,
            available_at,
            ctx,
            source_context,
            advance_context=league_advances,
        )
        if global_state is not None:
            global_phi = safe_transition_factor(global_state, available, ctx)
            shrink_covariances(("global", None, role, player_id), global_phi)
            global_state, global_advances = safe_transition(
                global_state, available, ctx
            )
            bridge = config["global_bridge"]
            global_value = bridge["observation_scale"] * value - bridge["global_center"]
            global_variance = obs_variance + bridge["bridge_variance"]
            global_key = ("global", None, role, player_id)
            global_gain = global_state.variance / (
                global_state.variance + global_variance
            )
            shrink_covariances(global_key, 1.0 - global_gain)
            states[global_key] = _gaussian_update(
                global_state,
                global_value,
                global_variance,
                available_at,
                ctx,
                f"bridged_{source_context}",
                advance_context=global_advances,
            )

    def drain(boundary: datetime, inclusive: bool) -> None:
        while pending and (pending[0][0] < boundary or (inclusive and pending[0][0] == boundary)):
            available, _, _, item = heapq.heappop(pending)
            apply_observation(item, available)

    forecasts: list[Mapping[str, Any]] = []
    ordered = sorted(rows, key=lambda row: (row["event_start"], row["match_id"]))
    for match in ordered:
        boundary = _parse_time(match["event_start"])
        if boundary > cutoff:
            break
        drain(boundary, inclusive=False)
        side_mean: dict[str, float] = {}
        side_variance: dict[str, float] = {}
        side_policy: dict[str, float] = {}
        side_policy_variance: dict[str, float] = {}
        forecast_factors: dict[
            tuple[str, str | None, str, str], float
        ] = {}
        for side in ("blue", "red"):
            total_mean = 0.0
            total_variance = 0.0
            policy_total = 0.0
            policy_variance_total = 0.0
            for role, player_id in match["rosters"][side].items():
                base, league_state, global_state = ensure_states(match, player_id)
                ctx = context_for(match, player_id)
                forecast_key = ("league", match["league_id"], role, player_id)
                forecast_factors[forecast_key] = _transition_factor(
                    league_state, decay, boundary, **ctx
                )[0]
                shrink_covariances(
                    forecast_key, forecast_factors[forecast_key]
                )
                forecast_state = _transition(
                    league_state, decay, boundary, **ctx, advance_context=True
                )
                states[forecast_key] = forecast_state
                master[(role, player_id)] = _transition(
                    base, decay, boundary, **ctx, advance_context=True
                )
                if global_state is not None:
                    global_key = ("global", None, role, player_id)
                    global_phi, _ = _transition_factor(
                        global_state, decay, boundary, **ctx
                    )
                    shrink_covariances(global_key, global_phi)
                    states[global_key] = _transition(
                        global_state, decay, boundary, **ctx, advance_context=True
                    )
                weight = role_weights[role]
                total_mean += weight * forecast_state.mean
                total_variance += weight * weight * forecast_state.variance
                policy_key = (player_id, "same_map_resource_share")
                if resource["mechanics"] == "joint_latent_resource_performance":
                    joint_mean, joint_variance_value, _ = joint_policy.get(
                        policy_key, (0.0, resource["policy_prior_variance"], 0)
                    )
                    policy_total += weight * joint_mean
                    policy_variance_total += weight * weight * joint_variance_value
                elif resource["mechanics"] in {
                    "lagged_observed_policy",
                    "double_count_collider_sensitivity",
                }:
                    policy_total += weight * policy.get(policy_key, 0.0)
            side_mean[side] = total_mean
            side_variance[side] = total_variance
            side_policy[side] = policy_total
            side_policy_variance[side] = policy_variance_total
        resource_adjustment = resource["forecast_coefficient"] * (
            side_policy["blue"] - side_policy["red"]
        )
        mean = side_mean["blue"] - side_mean["red"] + resource_adjustment
        roster_keys: list[tuple[tuple[str, str | None, str, str], float]] = []
        for side, sign in (("blue", 1.0), ("red", -1.0)):
            for role, player_id in match["rosters"][side].items():
                roster_keys.append(
                    (
                        ("league", match["league_id"], role, player_id),
                        sign * role_weights[role],
                    )
                )
        covariance_term = 0.0
        for index, (left, left_weight) in enumerate(roster_keys):
            for right, right_weight in roster_keys[index + 1 :]:
                covariance_term += (
                    2.0
                        * left_weight
                        * right_weight
                        * covariances.get(covariance_key(left, right), 0.0)
                        * forecast_factors.get(left, 1.0)
                        * forecast_factors.get(right, 1.0)
                    )
        variance = (
            side_variance["blue"]
            + side_variance["red"]
            + covariance_term
            + resource["forecast_coefficient"] ** 2
            * (side_policy_variance["blue"] + side_policy_variance["red"])
        )
        serialized = {
            "match_id": match["match_id"],
            "event_start": match["event_start"],
            "candidate_id": candidate_id,
            "logit_mean": mean,
            "logit_variance": variance,
            "plugin_expected_result": plugin_expected_result(mean),
            "posterior_predictive_expected_result": posterior_predictive_expected_result(
                mean, variance
            ),
            "covariance_assumption": config["posterior_predictive"]["covariance_assumption"],
            "eligible_resource_observations_pre_origin": sum(
                resource_observation_count_by_player.get(player_id, 0)
                for side in ("blue", "red")
                for player_id in match["rosters"][side].values()
            ),
        }
        forecast_hash = _sha256(_canonical_bytes(serialized))
        label_available = _parse_time(match["outcome_available_at"]) <= cutoff
        forecasts.append(
            _freeze(
                {
                    **serialized,
                    "forecast_sha256": forecast_hash,
                    "label": match["blue_win"] if label_available else None,
                    "label_available_at": match["outcome_available_at"],
                }
            )
        )
        sequence += 1
        heapq.heappush(
            pending,
            (
                _parse_time(match["outcome_available_at"]),
                sequence,
                match["match_id"],
                {"kind": "team_anchor", "match": match},
            ),
        )
        for update in match.get("player_updates", []):
            sequence += 1
            heapq.heappush(
                pending,
                (
                    _parse_time(update["available_at"]),
                    sequence,
                    match["match_id"],
                    {"kind": "skill", "match": match, "update": update},
                ),
            )
        for update in match.get("policy_updates", []):
            sequence += 1
            heapq.heappush(
                pending,
                (
                    _parse_time(update["available_at"]),
                    sequence,
                    match["match_id"],
                    {"kind": "policy", "match": match, "update": update},
                ),
            )
    drain(cutoff, inclusive=True)
    for key, state in list(states.items()):
        phi, _ = _transition_factor(
            state,
            decay,
            cutoff,
            season_id=resolved_target_context["season_id"],
            calendar_year=resolved_target_context["calendar_year"],
        )
        states[key] = _transition(
            state,
            decay,
            cutoff,
            season_id=resolved_target_context["season_id"],
            calendar_year=resolved_target_context["calendar_year"],
            event_start=as_of,
            advance_context=True,
        )
        for pair in list(covariances):
            if key in pair:
                covariances[pair] *= phi
    return ReplayResult(
        candidate_id=candidate_id,
        decay_candidate_id=decay_id,
        resource_candidate_id=resource_id,
        as_of=as_of,
        states=_freeze(states),
        policy_states=_freeze(policy),
        forecasts=tuple(forecasts),
        covariances=_freeze(covariances),
        joint_outcome_updates=joint_outcome_updates,
        resource_evidence=_freeze(
            {
                "candidate_id": resource_id,
                "mechanics": resource["mechanics"],
                "eligible_resource_observations": resource_observation_count,
                "state_path": (
                    "joint_policy_latent"
                    if resource["mechanics"] == "joint_latent_resource_performance"
                    else (
                        "lagged_observed_policy"
                        if resource["mechanics"] == "lagged_observed_policy"
                        else (
                            "excluded"
                            if resource["mechanics"] == "resource_excluded"
                            else "double_count_sensitivity"
                        )
                    )
                ),
            }
        ),
        target_context=resolved_target_context,
    )


def _replay_result_core(
    bundle: Bundle,
    result: ReplayResult,
    replay_input_sha256: str,
    ordered_events_sha256: str,
) -> Mapping[str, Any]:
    state_rows = [
        {
            "key": list(key),
            "state": {
                field_name: getattr(state, field_name)
                for field_name in PlayerState.__dataclass_fields__
            },
        }
        for key, state in sorted(result.states.items())
    ]
    covariance_rows = [
        {"left": list(pair[0]), "right": list(pair[1]), "value": value}
        for pair, value in sorted(result.covariances.items())
    ]
    policy_rows = [
        {"key": list(key), "value": value}
        for key, value in sorted(result.policy_states.items())
    ]
    return {
        "candidate_identity_sha256": bundle.raw_sha256["candidate_identity"],
        "config_sha256": bundle.raw_sha256["config"],
        "fixture_input_sha256": bundle.raw_sha256["fixtures"],
        "replay_input_sha256": replay_input_sha256,
        "ordered_events_sha256": ordered_events_sha256,
        "replay_source_sha256": bundle.raw_sha256["model_source"],
        "candidate_id": result.candidate_id,
        "decay_candidate_id": result.decay_candidate_id,
        "resource_candidate_id": result.resource_candidate_id,
        "as_of": result.as_of,
        "target_context": _thaw(result.target_context),
        "states": state_rows,
        "policy_states": policy_rows,
        "forecasts": _thaw(result.forecasts),
        "covariances": covariance_rows,
        "joint_outcome_updates": result.joint_outcome_updates,
        "resource_evidence": _thaw(result.resource_evidence),
    }


def replay_authenticated(
    bundle: Bundle,
    candidate_id: str,
    as_of: str,
    *,
    matches: Sequence[Mapping[str, Any]] | None = None,
    target_context: Mapping[str, Any] | None = None,
) -> ReplayResult:
    """Issue an immutable replay whose digest closes over inputs and outputs."""
    _require_issued_bundle(bundle)
    rows = list(matches if matches is not None else bundle.fixtures["matches"])
    ordered = sorted(rows, key=lambda row: (row["event_start"], row["match_id"]))
    replay_input_sha256 = _sha256(_canonical_bytes(rows))
    ordered_events_sha256 = _sha256(_canonical_bytes(ordered))
    result = replay(
        bundle.config,
        bundle.fixtures,
        candidate_id,
        as_of,
        matches=rows,
        target_context=target_context,
    )
    provisional = replace(
        result,
        replay_input_sha256=replay_input_sha256,
        ordered_events_sha256=ordered_events_sha256,
        replay_source_sha256=bundle.raw_sha256["model_source"],
    )
    identity = _sha256(
        _canonical_bytes(
            _replay_result_core(
                bundle,
                provisional,
                replay_input_sha256,
                ordered_events_sha256,
            )
        )
    )
    issued = replace(provisional, replay_identity_sha256=identity)
    _ISSUED_REPLAYS[id(issued)] = (weakref.ref(issued), identity)
    return issued


def _require_authenticated_replay(bundle: Bundle, result: ReplayResult) -> None:
    _require_issued_bundle(bundle)
    issued = _ISSUED_REPLAYS.get(id(result))
    if (
        issued is None
        or issued[0]() is not result
        or result.replay_identity_sha256 is None
        or issued[1] != result.replay_identity_sha256
        or result.replay_input_sha256 is None
        or result.ordered_events_sha256 is None
        or result.replay_source_sha256 != bundle.raw_sha256["model_source"]
    ):
        _fail("rating requires a loader-issued authenticated replay")
    expected = _sha256(
        _canonical_bytes(
            _replay_result_core(
                bundle,
                result,
                result.replay_input_sha256,
                result.ordered_events_sha256,
            )
        )
    )
    if expected != result.replay_identity_sha256:
        _fail("authenticated replay identity mismatch")


def compare_candidates(
    config: Mapping[str, Any], fixtures: Mapping[str, Any], as_of: str
) -> Mapping[str, Any]:
    minimum_origins = config["selection"]["minimum_eligible_origins"]
    registered_target_context = config["selection"]["evaluation_target_context"]
    comparison_target_context = (
        registered_target_context
        if registered_target_context["calendar_year"] == _parse_time(as_of).year
        else None
    )
    origin_reference = replay(
        config,
        fixtures,
        "random_walk_no_reset+no_resource",
        as_of,
        target_context=comparison_target_context,
    )
    labeled_reference = [
        row
        for row in origin_reference.forecasts
        if type(row["label"]) is int and row["label"] in (0, 1)
    ]
    supported_origin_ids = tuple(
        row["match_id"]
        for row in labeled_reference
        if row["eligible_resource_observations_pre_origin"] > 0
    )
    resource_comparison_enabled = len(supported_origin_ids) >= minimum_origins
    common_origin_ids = (
        supported_origin_ids
        if resource_comparison_enabled
        else tuple(row["match_id"] for row in labeled_reference)
    )
    common_origin_sha256 = _sha256(_canonical_bytes(common_origin_ids))
    diagnostics: list[dict[str, Any]] = []
    for decay_id in DECAY_CANDIDATES:
        for resource_id in RESOURCE_CANDIDATES:
            candidate_id = f"{decay_id}+{resource_id}"
            result = replay(
                config,
                fixtures,
                candidate_id,
                as_of,
                target_context=comparison_target_context,
            )
            labeled = [
                row
                for row in result.forecasts
                if type(row["label"]) is int and row["label"] in (0, 1)
            ]
            actual_eligible = (
                [
                    row
                    for row in labeled
                    if row["eligible_resource_observations_pre_origin"] > 0
                ]
                if resource_comparison_enabled
                else labeled
            )
            actual_origin_ids = tuple(row["match_id"] for row in actual_eligible)
            canonical_actual_ids = tuple(
                row["match_id"]
                for row in sorted(
                    actual_eligible,
                    key=lambda row: (row["event_start"], row["match_id"]),
                )
            )
            actual_origin_sha256 = _sha256(_canonical_bytes(actual_origin_ids))
            origin_identity_valid = (
                len(actual_origin_ids) == len(set(actual_origin_ids))
                and actual_origin_ids == canonical_actual_ids
                and actual_origin_ids == common_origin_ids
                and actual_origin_sha256 == common_origin_sha256
            )
            scored = actual_eligible if origin_identity_valid else []
            resource_support = sum(
                row["eligible_resource_observations_pre_origin"]
                for row in actual_eligible
            )
            resource_eligible = (
                resource_id == "no_resource"
                or (
                    resource_comparison_enabled
                    and origin_identity_valid
                    and all(
                        row["eligible_resource_observations_pre_origin"] > 0
                        for row in actual_eligible
                    )
                )
            )
            if origin_identity_valid and scored:
                losses = []
                briers = []
                for row in scored:
                    probability = min(
                        max(row["posterior_predictive_expected_result"], 1e-12), 1.0 - 1e-12
                    )
                    label = row["label"]
                    losses.append(
                        -(label * math.log(probability) + (1 - label) * math.log(1 - probability))
                    )
                    briers.append((probability - label) ** 2)
                log_loss = round(sum(losses) / len(losses), 15)
                brier = round(sum(briers) / len(briers), 15)
            else:
                log_loss = None
                brier = None
            diagnostics.append(
                {
                    "candidate_id": candidate_id,
                    "decay_candidate_id": decay_id,
                    "resource_candidate_id": resource_id,
                    "selectable": (
                        origin_identity_valid
                        and resource_id in SELECTABLE_RESOURCE_CANDIDATES
                        and resource_eligible
                    ),
                    "origin_identity_status": (
                        "verified"
                        if origin_identity_valid
                        else "origin_identity_mismatch"
                    ),
                    "eligible_resource_observations": resource_support,
                    "eligible_origin_count": len(actual_origin_ids),
                    "resource_supported_origin_count": sum(
                        row["eligible_resource_observations_pre_origin"] > 0
                        for row in actual_eligible
                    ),
                    "origin_ids": actual_origin_ids,
                    "origin_set_sha256": actual_origin_sha256,
                    "expected_origin_ids": common_origin_ids,
                    "expected_origin_set_sha256": common_origin_sha256,
                    "log_loss": log_loss,
                    "brier": brier,
                }
            )
    comparison_incomplete = (
        config["selection"]["require_complete_candidate_origin_identity"]
        and any(
            row["origin_identity_status"] != "verified"
            for row in diagnostics
            if row["resource_candidate_id"] in SELECTABLE_RESOURCE_CANDIDATES
        )
    )
    if comparison_incomplete:
        for row in diagnostics:
            row["selectable"] = False
    selectable = [
        row
        for row in diagnostics
        if row["selectable"] and row["eligible_origin_count"] >= minimum_origins
    ]
    selectable_origin_digests = {
        row["origin_set_sha256"] for row in selectable
    }
    if selectable_origin_digests and selectable_origin_digests != {
        common_origin_sha256
    }:
        _fail("selectable candidates do not share one verified origin identity")
    verified_selectable_origin_sha256 = (
        common_origin_sha256 if selectable else None
    )
    selected = (
        min(selectable, key=lambda row: (row["log_loss"], row["brier"], row["candidate_id"]))
        if selectable
        else None
    )
    decision_core = {
        "evaluation_cutoff": as_of,
        "minimum_eligible_origins": minimum_origins,
        "common_origin_ids": common_origin_ids,
        "common_origin_sha256": common_origin_sha256,
        "verified_selectable_origin_sha256": verified_selectable_origin_sha256,
        "selected_candidate_id": None if selected is None else selected["candidate_id"],
        "diagnostics": diagnostics,
    }
    return _freeze(
        {
            "status": (
                "origin_identity_incomplete"
                if comparison_incomplete
                else (
                    "synthetic_mechanics_selection"
                    if selected is not None
                    else "insufficient_evidence"
                )
            ),
            "selected_candidate_id": None if selected is None else selected["candidate_id"],
            "selection_decision_sha256": _sha256(_canonical_bytes(decision_core)),
            "eligible_origin_count": 0 if selected is None else selected["eligible_origin_count"],
            "common_origin_ids": common_origin_ids,
            "common_origin_sha256": common_origin_sha256,
            "verified_selectable_origin_sha256": verified_selectable_origin_sha256,
            "diagnostics": diagnostics,
        }
    )


def compare_decay_candidates(
    config: Mapping[str, Any], fixtures: Mapping[str, Any], as_of: str
) -> Mapping[str, Any]:
    """Compatibility alias; now evaluates the full decay x resource registry."""
    return compare_candidates(config, fixtures, as_of)


def reference_roster_logit(
    player_logits: Mapping[str, float],
    policy_weights: Mapping[str, float],
    reference_weights: Mapping[str, float],
) -> float:
    if set(player_logits) != set(ROLES) or set(policy_weights) != set(ROLES):
        _fail("roster aggregation requires every role exactly once")
    if set(reference_weights) != set(ROLES) or any(reference_weights[role] <= 0 for role in ROLES):
        _fail("reference policy requires positive weights for every role")
    return sum(
        policy_weights[role] / reference_weights[role] * player_logits[role]
        for role in ROLES
    )


def replacement_logit_delta(
    role: str,
    incumbent_display_rating: float,
    replacement_display_rating: float,
    policy_weights: Mapping[str, float],
    reference_weights: Mapping[str, float],
) -> float:
    if role not in ROLES:
        _fail("unknown role")
    return (
        policy_weights[role]
        / reference_weights[role]
        * (rating_to_latent(replacement_display_rating) - rating_to_latent(incumbent_display_rating))
    )


def evidence_components(
    state: PlayerState, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    evidence = config["evidence"]
    prior_sd = math.sqrt(config["prior"]["variance"])
    posterior_sd = math.sqrt(state.variance)
    required_contexts = tuple(evidence["required_context_ids"])
    supported = tuple(sorted(set(state.source_contexts).intersection(required_contexts)))
    missing = tuple(sorted(set(required_contexts).difference(supported)))
    return _freeze(
        {
            "selection_status": "selected",
            "evidence_spec_id": evidence["evidence_spec_id"],
            "posterior_displacement": {
                "method_id": evidence["displacement_method_id"],
                "value": abs(state.mean) / prior_sd,
                "unit": "prior_standard_deviations",
            },
            "precision": {
                "method_id": evidence["precision_method_id"],
                "posterior_dispersion": posterior_sd,
                "reference_dispersion": prior_sd,
                "unit": "latent_standard_deviation",
            },
            "source_context_coverage": {
                "method_id": evidence["coverage_method_id"],
                "supported_context_ids": supported,
                "missing_required_context_ids": missing,
                "required_context_count": len(required_contexts),
                "supported_context_count": len(supported),
                "coverage_fraction": (
                    len(supported) / len(required_contexts) if required_contexts else 0.0
                ),
                "identity_terms_status": "unsupported",
                "bridge_path_status": (
                    "weak" if state.scope == "global" else "not_applicable"
                ),
            },
        }
    )


def rating_payload(
    bundle: Bundle,
    result: ReplayResult,
    player_id: str,
    role: str,
    league_id: str,
    scope: str = "league",
) -> Mapping[str, Any]:
    """Return a nonpublic development payload for the one authenticated estimator."""
    _require_authenticated_replay(bundle, result)
    if bundle.selected_candidate_id is None:
        core = {
            "type": "l4_player_rating_development",
            "status": "unavailable",
            "candidate_id": None,
            "player_id": player_id,
            "role": role,
            "league_id": league_id,
            "rating_scope": scope,
            "as_of": result.as_of,
            "reason": "authenticated synthetic selection is insufficient",
            "independent_l4_authority_present": False,
            "claim_ceiling": CLAIM_CEILING,
        }
        return _issue_rating(
            {**core, "output_sha256": _sha256(_canonical_bytes(core))}
        )
    if result.candidate_id != bundle.selected_candidate_id:
        _fail("principal rating may only use the authenticated selected candidate")
    metadata = {player["player_id"]: player for player in bundle.fixtures["players"]}[player_id]
    if scope == "global" and not _eligible_global(
        metadata, _parse_time(result.as_of), bundle.config
    ):
        core = {
            "type": "l4_player_rating_development",
            "status": "unavailable",
            "candidate_id": result.candidate_id,
            "player_id": player_id,
            "role": role,
            "league_id": league_id,
            "rating_scope": scope,
            "as_of": result.as_of,
            "reason": "versioned global eligibility or bridge connectivity unavailable",
            "rank_eligible": False,
            "claim_ceiling": CLAIM_CEILING,
            "identities": {
                "model_version": bundle.config["model_version"],
                "config_sha256": bundle.raw_sha256["config"],
                "manifest_sha256": bundle.raw_sha256["manifest"],
                "input_sha256": bundle.raw_sha256["fixtures"],
                "c1_authority_sha256": C1_EXPECTED["raw_sha256"],
                "candidate_identity_sha256": bundle.raw_sha256[
                    "candidate_identity"
                ],
            },
            "independent_l4_authority_present": False,
        }
        return _issue_rating(
            {**core, "output_sha256": _sha256(_canonical_bytes(core))}
        )
    key = (scope, league_id if scope == "league" else None, role, player_id)
    state = result.states.get(key)
    if state is None:
        core = {
            "type": "l4_player_rating_development",
            "status": "unavailable",
            "candidate_id": result.candidate_id,
            "player_id": player_id,
            "role": role,
            "league_id": league_id,
            "rating_scope": scope,
            "as_of": result.as_of,
            "reason": "zero posterior support",
            "rank_eligible": False,
            "independent_l4_authority_present": False,
            "claim_ceiling": CLAIM_CEILING,
        }
        return _issue_rating(
            {**core, "output_sha256": _sha256(_canonical_bytes(core))}
        )
    weight = bundle.config["reference_policy"]["role_weights"][role]
    latent_mean = weight * state.mean
    latent_sd = weight * math.sqrt(state.variance)
    core = {
        "type": "l4_player_rating_development",
        "status": "development_only",
        "candidate_id": result.candidate_id,
        "decay_candidate_id": result.decay_candidate_id,
        "resource_candidate_id": result.resource_candidate_id,
        "model_version": bundle.config["model_version"],
        "config_sha256": bundle.raw_sha256["config"],
        "manifest_sha256": bundle.raw_sha256["manifest"],
        "input_sha256": bundle.raw_sha256["fixtures"],
        "c1_authority_sha256": C1_EXPECTED["raw_sha256"],
        "candidate_identity_sha256": bundle.raw_sha256["candidate_identity"],
        "replay_identity_sha256": result.replay_identity_sha256,
        "replay_input_sha256": result.replay_input_sha256,
        "ordered_events_sha256": result.ordered_events_sha256,
        "replay_source_sha256": result.replay_source_sha256,
        "independent_l4_authority_present": False,
        "player_id": player_id,
        "display_name": metadata["display_name"],
        "role": role,
        "league_id": league_id,
        "rating_scope": scope,
        "rating_basis": (
            "league_relative_player_contribution"
            if scope == "league"
            else "globally_bridged_player_contribution"
        ),
        "league_rating_included": False,
        "as_of": result.as_of,
        "reference_policy_id": bundle.config["reference_policy"]["policy_id"],
        "reference_population_id": bundle.config["reference_policy"]["population_id"],
        "rating_display": {"anchor": 1500, "logistic_scale": 400},
        "posterior_mean": latent_to_rating(latent_mean),
        "interval_95_model_range": {
            "lower": latent_to_rating(latent_mean - 1.96 * latent_sd),
            "upper": latent_to_rating(latent_mean + 1.96 * latent_sd),
        },
        "evidence": evidence_components(state, bundle.config),
        "reliability": {
            "status": "unavailable",
            "label": "unrated",
            "real_sample_count": 0,
            "real_cluster_count": 0,
            "validation_stratum_id": None,
            "probability_wording_approved": False,
            "reason": "authoritative observed rows unavailable",
        },
        "rank_eligible": False,
        "current": False,
        "inputs_complete_for_production": False,
        "inputs_fresh_for_production": False,
        "literal_interpretation": INTERPRETATION,
        "claim_ceiling": CLAIM_CEILING,
    }
    return _issue_rating(
        {**core, "output_sha256": _sha256(_canonical_bytes(core))}
    )


def public_unavailable_payload(
    bundle: Bundle, result: ReplayResult, player_id: str, role: str, league_id: str
) -> Mapping[str, Any]:
    """Return the only honest public-state semantic: unavailable."""
    lineage = {
        "manifest_id": bundle.manifest["artifact_id"],
        "training_snapshot_id": bundle.fixtures["artifact_id"],
        "source_snapshot_ids": [bundle.fixtures["artifact_id"]],
        "artifact_sha256": bundle.raw_sha256["manifest"],
        "source_tree_sha256": bundle.raw_sha256["fixtures"],
        "calibration_sha256": None,
        "evaluation_report_sha256": bundle.raw_sha256["report"],
        "code_commit": None,
        "environment_lock_sha256": bundle.raw_sha256["config"],
        "train_cutoff": result.as_of,
    }
    identity = {
        "player_id": player_id,
        "role": role,
        "league_id": league_id,
        "as_of": result.as_of,
        "reason": "production authority unavailable",
    }
    core = {
        "schema_version": "2.0.0",
        "model_version": bundle.config["model_version"],
        "as_of": result.as_of,
        "season_id": "synthetic-development",
        "calendar_year": _parse_time(result.as_of).year,
        "status": "unavailable",
        "rating_display": {"anchor": 1500, "logistic_scale": 400},
        "player_id": player_id,
        "role": role,
        "league_id": league_id,
        "rating_scope": "league",
        "rating_basis": "league_relative_player_contribution",
        "league_rating_included": False,
        "error": {
            "code": "model_not_promoted",
            "message": "Synthetic development mechanics do not authorize a public Player Rating.",
            "retryable": False,
            "missing_fields": ["production_player_rating_authority"],
            "stale_fields": [],
        },
        "lineage": lineage,
        "provenance": {
            "schema_version": "2.0.0",
            "model_version": bundle.config["model_version"],
            "as_of": result.as_of,
            "prediction_id": f"player-rating-unavailable-{league_id}-{role}-{player_id}",
            "mode": "state_snapshot",
            "created_at": result.as_of,
            "event_start": None,
            "related_forecast_id": None,
            "input_snapshot_id": bundle.fixtures["artifact_id"],
            "estimator_id": "dynamic-bayesian-player-state-development",
            "calibration_id": None,
            "required_input_status": "missing",
            "freshness_checks": [],
            "input_conflicts": [],
            "fallback_levels": ["F5"],
            "out_of_distribution_flags": ["synthetic_development_only"],
            "immutable": True,
            "lineage": lineage,
        },
    }
    output_sha256 = _sha256(_canonical_bytes(core))
    core["provenance"]["output_sha256"] = output_sha256
    return _freeze(core)


def canonical_public_output_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = _thaw(payload)
    provenance = unsigned.get("provenance")
    if not isinstance(provenance, dict) or "output_sha256" not in provenance:
        _fail("public payload output identity is missing")
    provenance.pop("output_sha256")
    return _sha256(_canonical_bytes(unsigned))


def rank_ratings(payloads: Iterable[DevelopmentRating]) -> list[DevelopmentRating]:
    checked: list[DevelopmentRating] = []
    for payload in payloads:
        if not isinstance(payload, DevelopmentRating) or payload not in _ISSUED_RATINGS:
            _fail("FORGED_RANK: rating was not issued by the authenticated loader")
        if payload.get("status") not in {"development_only", "unavailable"}:
            _fail("FORGED_RANK: invalid development rating status")
        if payload.get("independent_l4_authority_present") is not False:
            _fail("FORGED_RANK: invalid authorization identity")
        unsigned = _thaw(payload)
        output_hash = unsigned.pop("output_sha256", None)
        if output_hash != _sha256(_canonical_bytes(unsigned)):
            _fail("FORGED_RANK: rating output identity mismatch")
        if payload.get("status") == "development_only" and payload.get("rank_eligible") is True:
            checked.append(payload)
    eligible = checked
    return sorted(
        eligible,
        key=lambda payload: (
            -payload["posterior_mean"],
            payload["interval_95"]["upper"] - payload["interval_95"]["lower"],
            payload["player_id"],
        ),
    )
