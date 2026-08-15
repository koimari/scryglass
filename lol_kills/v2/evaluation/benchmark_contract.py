"""Fail-closed validation for the label-blind real-v1 benchmark freeze.

The package has two deliberately separate layers:

* source-independent scientific rules, frozen before a real snapshot is bound;
* source-dependent identities, which remain an explicitly non-authorizing
  template until G1 binds them and an independent reviewer accepts the result.

Nothing in this module reads a model target, fits a model, or opens a holdout.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import fcntl
import hmac
import importlib.util
from itertools import combinations
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import threading
import types
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
import numpy as np


REAL_V1_ROOT = Path("data/lol/v2/evaluation/real-v1")
BENCHMARK_SCHEMA = "benchmark-contract.schema.json"
BASELINE_SCHEMA = "baseline-registry.schema.json"
OPENING_PERMIT_SCHEMA = "opening-permit.schema.json"
AUTHORITY_BUNDLE_SCHEMA = "authority-bundle.schema.json"
CANDIDATE_REGISTRY_SCHEMA = "candidate-registry.schema.json"
PAIR_REGISTRY_SCHEMA = "pair-registry.schema.json"
BENCHMARK_CONTRACT = "benchmark-contract.json"
BASELINE_REGISTRY = "baseline-registry.json"
OPENING_PERMIT_TEMPLATE = "opening-permit-template.json"
AUTHORITY_BUNDLE = "authority-bundle.json"
CANDIDATE_REGISTRY = "candidate-registry.json"
PAIR_REGISTRY = "pair-registry.json"
CONTRACT_MANIFEST = "contract-manifest.json"

_REPO_ROOT = Path(os.path.abspath(__file__)).parents[3]
_PRODUCTION_PACKAGE_ROOT = _REPO_ROOT / REAL_V1_ROOT
_WCR_EXECUTION_ID = "multiway-wcr-cgm-v1.4-promotion-remand"
_WCR_SEED = 2026072901
_WCR_REPLICATES = 9999
_REVIEWER_IDENTITY_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$"
)

_HEX = set("0123456789abcdef")
_PARTIAL_DEPTHS = tuple(range(10))
_TERMINAL_DEPTH = (10,)
_RATING_DEPTH = (-1,)
_PRIMARY_LEAGUES = ("LEC", "LCK", "LPL", "LCS", "LCP")
_CANDIDATE_IDS = (
    "candidate:scryglass-v2",
    "candidate:scryglass-v2-simplest-parent",
)
_OUTPUT_IDS = (
    "player_rating",
    "team_rating",
    "terminal_draft_score",
    "partial_draft_score",
)
_SUPPORTED_SECONDARY_BENEFITS = {
    "player_rating": ("brier", "transfer_new_roster"),
    "team_rating": ("brier", "transfer_new_roster"),
    "terminal_draft_score": (
        "brier",
        "equal_strength_draft_increment",
    ),
    "partial_draft_score": (
        "brier",
        "equal_strength_draft_increment",
    ),
}
_CRITICAL_CELLS_BY_OUTPUT = {
    "player_rating": (
        *(f"league:{league}" for league in _PRIMARY_LEAGUES),
        "patch:each-held-out-major-minor",
        "game_side:BLUE",
        "game_side:RED",
        "roster_change:FIRST_TOURNAMENT_NEW_EXACT_ROSTER",
        "roster_change:STABLE_EXACT_ROSTER",
        "international_event:MSI",
        "international_event:EWC",
        "international_event:OTHER_NAMED_EVENT",
    ),
    "team_rating": (
        *(f"league:{league}" for league in _PRIMARY_LEAGUES),
        "patch:each-held-out-major-minor",
        "game_side:BLUE",
        "game_side:RED",
        "roster_change:FIRST_TOURNAMENT_NEW_EXACT_ROSTER",
        "roster_change:STABLE_EXACT_ROSTER",
        "international_event:MSI",
        "international_event:EWC",
        "international_event:OTHER_NAMED_EVENT",
    ),
    "terminal_draft_score": (
        *(f"league:{league}" for league in _PRIMARY_LEAGUES),
        "patch:each-held-out-major-minor",
        "game_side:BLUE",
        "game_side:RED",
        "international_event:MSI",
        "international_event:EWC",
        "international_event:OTHER_NAMED_EVENT",
        "draft_depth:10",
    ),
    "partial_draft_score": (
        *(f"league:{league}" for league in _PRIMARY_LEAGUES),
        "patch:each-held-out-major-minor",
        "game_side:BLUE",
        "game_side:RED",
        "international_event:MSI",
        "international_event:EWC",
        "international_event:OTHER_NAMED_EVENT",
        *(f"draft_depth:{depth}" for depth in range(10)),
    ),
}
_REQUIRED_RATING_BASELINES = frozenset(
    {
        "rating.constant_league_side_frequency",
        "rating.classical_elo",
        "rating.static_bradley_terry",
        "rating.current_dual_elo",
        "rating.current_hierarchical_bradley_terry",
        "rating.dynamic_uncertainty",
        "rating.reproducible_state_space",
        "rating.conditional_trueskill2",
        "team.player_average_no_policy_synergy",
        "team.exact_roster_no_league_rating",
        "rating.ablation.no_temporal_dynamics_decay",
        "rating.ablation.no_individual_auxiliary_channels",
        "team.ablation.no_team_policy",
        "team.ablation.no_lineup_synergy",
    }
)
_REQUIRED_TERMINAL_BASELINES = frozenset(
    {
        "draft.league_patch_side_frequency",
        "draft.role_aware_additive",
        "draft.same_role_matchup",
        "draft.all_pair_ally_enemy",
        "draft.factorization",
        "draft.current_served_estimator",
        "draft.ablation.no_team_player_context",
        "draft.ablation.no_draft_component",
        "draft.ablation.neutral_no_archetype_transfer",
        "draft.ablation.neutral_no_patch_league_deviation",
        "draft.ablation.contextual_no_player_champion_fit",
        "draft.ablation.contextual_no_team_policy",
        "draft.ablation.no_ontology_priors",
        "draft.ablation.no_champion_residuals",
        "draft.ablation.no_ally_pairs",
        "draft.ablation.no_enemy_pairs",
        "draft.ablation.no_whole_team_residual",
        "draft.ablation.no_cross_team_residual",
        "draft.ablation.no_functional_anova_projection",
        "draft.ablation.no_legal_support_rank_cooccurrence",
        "draft.ablation.no_contextual_h_vs_q_identification",
        "draft.ablation.no_patch_deviation",
        "draft.ablation.no_league_deviation",
        "draft.ablation.no_role_deviation",
        "draft.ablation.no_exact_contextual_equalization",
        "draft.ablation.no_calibration",
        "tier.raw_win_rate",
        "tier.strength_controlled_additive_champion_value",
        "tier.ablation.incremental_without_counterability",
    }
)
_REQUIRED_PARTIAL_BASELINES = frozenset(
    {
        "partial.league_patch_side_frequency",
        "partial.observed_behavior_frequency",
        "partial.role_aware_additive",
        "partial.same_role_matchup",
        "partial.all_pair_ally_enemy",
        "partial.factorization",
        "partial.current_served_search_policy",
        "partial.ablation.no_team_player_context",
        "partial.ablation.no_draft_component",
        "partial.ablation.neutral_no_archetype_transfer",
        "partial.ablation.neutral_no_patch_league_deviation",
        "partial.ablation.contextual_no_player_champion_fit",
        "partial.ablation.contextual_no_team_policy",
        "partial.greedy_terminal_search",
        "partial.hard_minimax",
        "partial.reproducible_published_tree_search",
        "partial.ablation.no_strategic_response_adjustment",
        "partial.ablation.no_flex_handling",
    }
)
_REQUIRED_BASELINES = (
    _REQUIRED_RATING_BASELINES
    | _REQUIRED_TERMINAL_BASELINES
    | _REQUIRED_PARTIAL_BASELINES
)
_REQUIRED_COMPONENTS = frozenset(
    {
        "exact_roster_aggregation",
        "league_rating",
        "temporal_dynamics_decay",
        "individual_auxiliary_channels",
        "team_rating_policy",
        "lineup_synergy",
        "team_player_context",
        "all_draft_information",
        "ontology_priors",
        "champion_residuals",
        "archetype_transfer",
        "patch_league_deviation",
        "ally_pairs",
        "enemy_pairs",
        "whole_team_residual",
        "cross_team_residual",
        "functional_anova_projection",
        "legal_support_rank_cooccurrence",
        "contextual_h_vs_q_identification",
        "patch_deviation",
        "league_deviation",
        "role_deviation",
        "player_champion_fit",
        "draft_team_policy",
        "exact_contextual_equalization",
        "calibration",
        "partial_team_player_context",
        "partial_all_draft_information",
        "partial_archetype_transfer",
        "partial_patch_league_deviation",
        "partial_player_champion_fit",
        "partial_team_policy",
        "partial_search_policy",
        "partial_strategic_response_adjustment",
        "partial_flex_handling",
        "tier_counterability",
    }
)
_REQUIRED_DIAGNOSTICS = frozenset(
    {
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "reliability_diagram",
        "sharpness",
        "auc",
        "interval_behavior",
        "transfer_new_roster",
        "rank_stability",
        "sparse_new_champion",
        "equal_strength_draft_increment",
    }
)
_EXPECTED_OUTPUT_DIAGNOSTICS = {
    "player_rating": (
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "reliability_diagram",
        "sharpness",
        "auc",
        "interval_behavior",
        "transfer_new_roster",
        "rank_stability",
    ),
    "team_rating": (
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "reliability_diagram",
        "sharpness",
        "auc",
        "interval_behavior",
        "transfer_new_roster",
        "rank_stability",
    ),
    "terminal_draft_score": (
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "reliability_diagram",
        "sharpness",
        "auc",
        "equal_strength_draft_increment",
        "sparse_new_champion",
    ),
    "partial_draft_score": (
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "reliability_diagram",
        "sharpness",
        "equal_strength_draft_increment",
        "sparse_new_champion",
    ),
}
_REQUIRED_STRATA = frozenset(
    {"league", "patch", "game_side", "roster_change", "international_event", "draft_depth"}
)
_ALLOWED_UNAVAILABLE_REASONS = frozenset(
    {
        "NO_FROZEN_LABEL_SAFE_ADAPTER",
        "NO_REPRODUCIBLE_PRE_EVENT_IMPLEMENTATION",
        "NO_PRE_EVENT_FORECAST_ADAPTER",
        "MODEL_FAMILY_NOT_YET_BOUND",
        "NO_REPRODUCIBLE_IMPLEMENTATION",
        "CONDITIONAL_TRUESKILL2_NOT_IMPLEMENTED",
        "G1_EXACT_ROSTER_ADAPTER_REQUIRED",
        "NO_AUTHORIZED_SERVING_IDENTITY",
        "CONTEXTUAL_ABLATION_NOT_INSTANTIATED",
        "NO_DRAFT_ABLATION_NOT_INSTANTIATED",
        "ARCHETYPE_ABLATION_NOT_INSTANTIATED",
        "PATCH_LEAGUE_ABLATION_NOT_INSTANTIATED",
        "PLAYER_CHAMPION_ABLATION_NOT_INSTANTIATED",
        "TEAM_POLICY_ABLATION_NOT_INSTANTIATED",
        "PARTIAL_DEPTH_ADAPTER_NOT_IMPLEMENTED",
        "PARTIAL_BEHAVIOR_POLICY_NOT_BOUND",
        "NO_PROSPECTIVE_OR_VALID_OPE_EVIDENCE",
        "PARTIAL_CONTEXTUAL_ABLATION_NOT_INSTANTIATED",
        "PARTIAL_NO_DRAFT_ABLATION_NOT_INSTANTIATED",
        "PARTIAL_ARCHETYPE_ABLATION_NOT_INSTANTIATED",
        "PARTIAL_PATCH_LEAGUE_ABLATION_NOT_INSTANTIATED",
        "PARTIAL_PLAYER_CHAMPION_ABLATION_NOT_INSTANTIATED",
        "PARTIAL_TEAM_POLICY_ABLATION_NOT_INSTANTIATED",
        "REQUIRED_ABLATION_NOT_INSTANTIATED",
        "GREEDY_TERMINAL_SEARCH_NOT_IMPLEMENTED",
        "HARD_MINIMAX_NOT_IMPLEMENTED",
        "PUBLISHED_TREE_SEARCH_NOT_REPRODUCIBLY_BOUND",
        "TIER_COMPARATOR_NOT_IMPLEMENTED",
    }
)
_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "label",
        "labels",
        "outcome",
        "outcomes",
        "target",
        "targets",
        "map_winner",
        "final_label",
        "final_labels",
        "sealed_label",
        "sealed_labels",
    }
)
_FORBIDDEN_EXACT_BASENAMES = frozenset(
    {
        "final-labels.json",
        "final_labels.json",
        "sealed-labels.json",
        "sealed_labels.json",
        "outcomes.json",
        "targets.json",
    }
)


class BenchmarkContractError(ValueError):
    """The freeze is malformed, unauthorised, or scientifically incomplete."""

    def __init__(self, message: str, *, code: str = "BENCHMARK_CONTRACT_INVALID") -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def canonical_json(value: Any) -> str:
    """Stable semantic representation.

    This is intentionally a semantic digest representation, not a claim that
    the source file was byte-canonical. Exact bytes are pinned separately.
    """

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_digest(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def raw_digest(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"{label}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise BenchmarkContractError(f"{label}: JSON root must be an object")
    return value


def load_canonical_json(path: Path) -> dict[str, Any]:
    """Compatibility name for duplicate-safe JSON loading.

    Semantic and raw identities are checked by :func:`validate_real_v1`.
    """

    return _parse_json_bytes(path.read_bytes(), label=str(path))


def _require(
    condition: bool,
    message: str,
    *,
    code: str = "BENCHMARK_CONTRACT_INVALID",
) -> None:
    if not condition:
        raise BenchmarkContractError(message, code=code)


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _schema_validate(
    instance: Mapping[str, Any],
    schema: Mapping[str, Any],
    label: str,
) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        raise BenchmarkContractError(f"{label}: invalid Draft 2020-12 schema") from exc
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(instance),
        key=lambda error: list(error.path),
    )
    if errors:
        error = errors[0]
        where = "/".join(str(part) for part in error.path) or "<root>"
        pending = [error]
        closure_violation = False
        while pending:
            current = pending.pop()
            if current.validator in {
                "additionalProperties",
                "unevaluatedProperties",
            }:
                closure_violation = True
                break
            pending.extend(current.context)
        code = "SCHEMA_CLOSED" if closure_violation else "SCHEMA_INVALID"
        raise BenchmarkContractError(
            f"{label}: schema {where}: {error.message}",
            code=code,
        )


def _as_id_set(items: object, field: str) -> set[str]:
    _require(isinstance(items, list), f"{field} must be an array")
    values = [item.get("id") for item in items if isinstance(item, dict)]
    _require(
        len(values) == len(items) and all(isinstance(value, str) for value in values),
        f"{field} entries require string id",
    )
    _require(len(values) == len(set(values)), f"{field} contains duplicate ids")
    return set(values)


def _walk_items(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_items(child)


def _reject_nulls(value: Any, *, label: str) -> None:
    def visit(current: Any, path: str) -> None:
        if current is None:
            raise BenchmarkContractError(f"{label}: null placeholder at {path}")
        if isinstance(current, dict):
            for key, child in current.items():
                visit(child, f"{path}/{key}")
        elif isinstance(current, list):
            for index, child in enumerate(current):
                visit(child, f"{path}/{index}")

    visit(value, "<root>")


def validate_no_final_labels(value: Any, *, label: str) -> None:
    offending = sorted(
        {key for key, _ in _walk_items(value) if key.lower() in _FORBIDDEN_PAYLOAD_KEYS}
    )
    _require(
        not offending,
        f"{label}: target-like payload keys are forbidden at the label-free boundary: {offending}",
    )


def _normalize_relative(path: str | Path) -> str:
    normalized = str(path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _path_is_forbidden(path: str | Path, contract: Mapping[str, Any]) -> bool:
    normalized = _normalize_relative(path).lower()
    pure = PurePosixPath(normalized)
    parts = tuple(part.lower() for part in pure.parts)
    basename = parts[-1] if parts else ""
    policy = contract["label_access"]
    exact = {str(name).lower() for name in policy["forbidden_exact_basenames"]}
    if basename in exact or basename in _FORBIDDEN_EXACT_BASENAMES:
        return True
    if "final" in parts or "sealed-targets" in parts or "sealed_labels" in parts:
        return True
    if basename == "outcomes.json" and "holdout" in parts:
        return True
    return any(pure.match(str(pattern).lower()) for pattern in policy["forbidden_path_patterns"])


def _canonical_relative_locator(
    locator: str | Path,
    *,
    purpose: str,
    contract: Mapping[str, Any] | None = None,
) -> str:
    text = os.fspath(locator)
    _require(
        isinstance(text, str)
        and text not in {"", "."}
        and "\\" not in text
        and not text.startswith("/")
        and not Path(text).is_absolute(),
        f"{purpose}: locator is not a canonical relative POSIX path",
        code="LABEL_PATH_FORBIDDEN" if contract is not None else "PATH_LEAF_NOT_REGULAR",
    )
    pure = PurePosixPath(text)
    canonical = pure.as_posix()
    _require(
        canonical == text
        and all(part not in {"", ".", ".."} for part in pure.parts),
        f"{purpose}: locator is not canonical: {text}",
        code="LABEL_PATH_FORBIDDEN" if contract is not None else "PATH_LEAF_NOT_REGULAR",
    )
    if contract is not None:
        _require(
            not _path_is_forbidden(canonical, contract),
            f"protected target path refused: {canonical}",
            code="LABEL_PATH_FORBIDDEN",
        )
    return canonical


def _open_root_directory(root: Path, *, purpose: str) -> int:
    """Open a requested root without resolving away a root symlink."""

    requested = Path(os.path.abspath(os.fspath(root)))
    try:
        before = os.lstat(requested)
    except FileNotFoundError:
        raise BenchmarkContractError(
            f"{purpose}: root is missing: {requested}",
            code="PATH_COMPONENT_NOT_DIRECTORY",
        ) from None
    _require(
        not stat.S_ISLNK(before.st_mode),
        f"{purpose}: root symlink refused: {requested}",
        code="PATH_ROOT_SYMLINK",
    )
    _require(
        stat.S_ISDIR(before.st_mode),
        f"{purpose}: root is not a directory: {requested}",
        code="PATH_COMPONENT_NOT_DIRECTORY",
    )
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(requested, flags)
    except OSError as exc:
        raise BenchmarkContractError(
            f"{purpose}: root open failed: {requested}",
            code="PATH_COMPONENT_NOT_DIRECTORY",
        ) from exc
    after = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(after.st_mode)
        or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
    ):
        os.close(descriptor)
        raise BenchmarkContractError(
            f"{purpose}: root identity changed during open",
            code="PATH_COMPONENT_NOT_DIRECTORY",
        )
    return descriptor


def _read_regular_under_root(
    root: Path,
    locator: str | Path,
    *,
    purpose: str,
    contract: Mapping[str, Any] | None = None,
) -> bytes:
    """Descriptor-contained, nofollow read of a single-link regular file."""

    relative = _canonical_relative_locator(
        locator,
        purpose=purpose,
        contract=contract,
    )
    parts = PurePosixPath(relative).parts
    root_descriptor = _open_root_directory(Path(root), purpose=purpose)
    descriptors = [root_descriptor]
    try:
        current = root_descriptor
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        for part in parts[:-1]:
            try:
                before = os.stat(part, dir_fd=current, follow_symlinks=False)
            except FileNotFoundError:
                raise BenchmarkContractError(
                    f"{purpose}: parent component is missing: {part}",
                    code="PATH_COMPONENT_NOT_DIRECTORY",
                ) from None
            _require(
                not stat.S_ISLNK(before.st_mode),
                f"{purpose}: parent symlink refused: {part}",
                code="PATH_COMPONENT_SYMLINK",
            )
            _require(
                stat.S_ISDIR(before.st_mode),
                f"{purpose}: parent is not a directory: {part}",
                code="PATH_COMPONENT_NOT_DIRECTORY",
            )
            try:
                opened = os.open(part, directory_flags, dir_fd=current)
            except OSError as exc:
                raise BenchmarkContractError(
                    f"{purpose}: parent open failed: {part}",
                    code="PATH_COMPONENT_NOT_DIRECTORY",
                ) from exc
            after = os.fstat(opened)
            if (
                not stat.S_ISDIR(after.st_mode)
                or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            ):
                os.close(opened)
                raise BenchmarkContractError(
                    f"{purpose}: parent identity changed during open: {part}",
                    code="PATH_COMPONENT_NOT_DIRECTORY",
                )
            descriptors.append(opened)
            current = opened

        leaf = parts[-1]
        try:
            before = os.stat(leaf, dir_fd=current, follow_symlinks=False)
        except FileNotFoundError:
            raise BenchmarkContractError(
                f"{purpose}: leaf is missing: {relative}",
                code="PATH_LEAF_NOT_REGULAR",
            ) from None
        _require(
            not stat.S_ISLNK(before.st_mode),
            f"{purpose}: leaf symlink refused: {relative}",
            code="PATH_COMPONENT_SYMLINK",
        )
        _require(
            stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
            f"{purpose}: leaf must be a single-link regular file: {relative}",
            code="PATH_LEAF_NOT_REGULAR",
        )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            leaf_descriptor = os.open(leaf, flags, dir_fd=current)
        except OSError as exc:
            raise BenchmarkContractError(
                f"{purpose}: leaf open failed: {relative}",
                code="PATH_LEAF_NOT_REGULAR",
            ) from exc
        try:
            after = os.fstat(leaf_descriptor)
            _require(
                stat.S_ISREG(after.st_mode)
                and after.st_nlink == 1
                and (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino),
                f"{purpose}: leaf identity changed during open: {relative}",
                code="PATH_LEAF_NOT_REGULAR",
            )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(leaf_descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(leaf_descriptor)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


@dataclass(frozen=True)
class VerifiedRead:
    relative_path: str
    raw_sha256: str
    semantic_sha256: str


@dataclass(frozen=True)
class VerifiedAuthority:
    """Immutable audit result derived from fixed, manifest-pinned files.

    This object is output only. No public authorization API accepts it back as
    evidence. It is deliberately not described as cryptographically
    unforgeable inside a cooperative Python process.
    """

    authority_id: str
    authority_bundle_raw_sha256: str
    authority_contract_set_sha256: str
    manifest_contract_set_sha256: str
    expected_reads: tuple[VerifiedRead, ...]
    expected_read_set_sha256: str
    candidate_registry_raw_sha256: str
    candidate_registry_semantic_sha256: str
    pair_registry_raw_sha256: str
    pair_registry_semantic_sha256: str
    evidence_generator_identity: str
    permit_ledger_relative_path: str
    preflight_sha256: str
    threat_ceiling: str


@dataclass(frozen=True)
class _AuthorityContext:
    verified: VerifiedAuthority
    package_root: Path
    repo_root: Path
    contract: dict[str, Any]
    baseline_registry: dict[str, Any]
    candidate_registry: dict[str, Any]
    pair_registry: dict[str, Any]


def _manifest_bound_json(
    package_root: Path,
    manifest: Mapping[str, Any],
    locator: str,
    *,
    purpose: str,
) -> tuple[bytes, dict[str, Any]]:
    records = manifest.get("files")
    _require(
        isinstance(records, Mapping) and locator in records,
        f"{purpose}: file is absent from manifest",
        code="AUTHORITY_BUNDLE_MISMATCH",
    )
    record = records[locator]
    _require(
        isinstance(record, Mapping)
        and set(record) == {"raw_sha256", "semantic_sha256"},
        f"{purpose}: manifest record is malformed",
        code="AUTHORITY_BUNDLE_MISMATCH",
    )
    raw = _read_regular_under_root(package_root, locator, purpose=purpose)
    payload = _parse_json_bytes(raw, label=purpose)
    _require(
        raw_digest(raw) == record["raw_sha256"]
        and stable_digest(payload) == record["semantic_sha256"],
        f"{purpose}: manifest-bound identity mismatch",
        code="AUTHORITY_BUNDLE_MISMATCH",
    )
    return raw, payload


def _verified_read_records(
    reads: Sequence[VerifiedRead],
) -> list[dict[str, str]]:
    return [
        {
            "relative_path": item.relative_path,
            "raw_sha256": item.raw_sha256,
            "semantic_sha256": item.semantic_sha256,
        }
        for item in reads
    ]


def _verify_authoritative_preflight_at(
    package_root: Path,
    repo_root: Path,
) -> _AuthorityContext:
    """Reopen and derive every authority identity from fixed artifact roots."""

    package_root = Path(package_root)
    repo_root = Path(repo_root)
    try:
        validate_real_v1(package_root, repo_root=repo_root)
    except BenchmarkContractError as exc:
        preserved = {
            "CANDIDATE_SLOT_MISSING",
            "CANDIDATE_SLOT_UNEXPECTED",
            "CANDIDATE_SLOT_DUPLICATE",
            "CANDIDATE_SLOT_UNRESOLVED",
            "PAIR_IDENTITY_MISMATCH",
            "FAMILY_DERIVATION_MISMATCH",
            "FAMILY_UNREGISTERED",
        }
        if exc.code in preserved:
            code = exc.code
        elif exc.code.startswith("PATH_"):
            code = "AUTHORITY_BUNDLE_UNAVAILABLE"
        elif "candidate" in str(exc).lower() or "pair registry" in str(exc).lower():
            code = "CANDIDATE_REGISTRY_MISMATCH"
        else:
            code = "AUTHORITY_BUNDLE_MISMATCH"
        raise BenchmarkContractError(
            "manifest-pinned authority package is unavailable or invalid",
            code=code,
        ) from exc

    manifest_raw = _read_regular_under_root(
        package_root,
        CONTRACT_MANIFEST,
        purpose="authority contract manifest",
    )
    manifest = _parse_json_bytes(manifest_raw, label=CONTRACT_MANIFEST)
    bundle_raw, bundle = _manifest_bound_json(
        package_root,
        manifest,
        AUTHORITY_BUNDLE,
        purpose="authority bundle",
    )
    _, bundle_schema = _manifest_bound_json(
        package_root,
        manifest,
        AUTHORITY_BUNDLE_SCHEMA,
        purpose="authority bundle schema",
    )
    _schema_validate(bundle, bundle_schema, AUTHORITY_BUNDLE)

    _, contract = _manifest_bound_json(
        package_root,
        manifest,
        BENCHMARK_CONTRACT,
        purpose="benchmark contract",
    )
    _require(
        bundle["authority_id"] == contract["label_access"]["boundary_id"],
        "authority bundle id differs from the label-access boundary",
        code="AUTHORITY_BUNDLE_MISMATCH",
    )
    _, baseline_registry = _manifest_bound_json(
        package_root,
        manifest,
        BASELINE_REGISTRY,
        purpose="baseline registry",
    )
    candidate_raw, candidate_registry = _manifest_bound_json(
        package_root,
        manifest,
        CANDIDATE_REGISTRY,
        purpose="candidate registry",
    )
    pair_raw, pair_registry = _manifest_bound_json(
        package_root,
        manifest,
        PAIR_REGISTRY,
        purpose="pair registry",
    )

    _require(
        bundle["contract_set_sha256"]
        == manifest.get("authority_contract_set_sha256"),
        "authority bundle is bound to a different authority contract set",
        code="AUTHORITY_BUNDLE_MISMATCH",
    )
    _require(
        candidate_registry["contract_set_sha256"]
        == bundle["contract_set_sha256"]
        and pair_registry["contract_set_sha256"]
        == bundle["contract_set_sha256"],
        "candidate or pair registry is bound to a different contract set",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )

    candidate_binding = bundle["candidate_registry"]
    _require(
        candidate_binding
        == {
            "relative_path": CANDIDATE_REGISTRY,
            "raw_sha256": raw_digest(candidate_raw),
            "semantic_sha256": stable_digest(candidate_registry),
        },
        "candidate registry binding mismatch",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    pair_binding = bundle["pair_registry"]
    _require(
        pair_binding
        == {
            "relative_path": PAIR_REGISTRY,
            "raw_sha256": raw_digest(pair_raw),
            "semantic_sha256": stable_digest(pair_registry),
        },
        "pair registry binding mismatch",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    for candidate in candidate_registry["candidates"]:
        if candidate["status"] != "RESOLVED":
            continue
        relative = _canonical_relative_locator(
            candidate["candidate_artifact_relative_path"],
            purpose=f"candidate artifact {candidate['candidate_id']}",
            contract=contract,
        )
        try:
            raw = _read_regular_under_root(
                package_root,
                relative,
                purpose=f"candidate artifact {candidate['candidate_id']}",
                contract=contract,
            )
        except BenchmarkContractError as exc:
            raise BenchmarkContractError(
                f"registered candidate artifact is unavailable: {relative}",
                code="CANDIDATE_REGISTRY_MISMATCH",
            ) from exc
        payload = _parse_json_bytes(
            raw,
            label=f"candidate artifact {candidate['candidate_id']}",
        )
        try:
            validate_no_final_labels(
                payload,
                label=f"candidate artifact {candidate['candidate_id']}",
            )
        except BenchmarkContractError as exc:
            raise BenchmarkContractError(
                "registered candidate artifact violates the label boundary",
                code="AUTH_LABEL_BOUNDARY_VIOLATION",
            ) from exc
        _require(
            raw_digest(raw)
            == candidate["candidate_artifact_raw_sha256"]
            and stable_digest(payload)
            == candidate["candidate_artifact_semantic_sha256"],
            f"registered candidate artifact binding mismatch: {relative}",
            code="CANDIDATE_REGISTRY_MISMATCH",
        )

    expected = bundle.get("expected_reads")
    _require(
        isinstance(expected, list) and bool(expected),
        "authority expected-read set is empty",
        code="AUTH_EXPECTED_READ_MISSING",
    )
    _require(
        expected
        == sorted(expected, key=lambda item: item.get("relative_path", ""))
        and len({item.get("relative_path") for item in expected}) == len(expected),
        "authority expected-read set is not canonical and unique",
        code="AUTHORITY_BUNDLE_MISMATCH",
    )
    verified_reads: list[VerifiedRead] = []
    for item in expected:
        _require(
            isinstance(item, Mapping)
            and set(item)
            == {"relative_path", "raw_sha256", "semantic_sha256"},
            "authority expected-read record is malformed",
            code="AUTHORITY_BUNDLE_MISMATCH",
        )
        try:
            relative = _canonical_relative_locator(
                item["relative_path"],
                purpose="authority expected read",
                contract=contract,
            )
        except BenchmarkContractError as exc:
            raise BenchmarkContractError(
                "authority expected read violates the label path boundary",
                code="AUTH_LABEL_BOUNDARY_VIOLATION",
            ) from exc
        try:
            raw = _read_regular_under_root(
                package_root,
                relative,
                purpose=f"authority expected read {relative}",
                contract=contract,
            )
        except BenchmarkContractError as exc:
            raise BenchmarkContractError(
                f"authority expected read is missing: {relative}",
                code="AUTH_EXPECTED_READ_MISSING",
            ) from exc
        payload = _parse_json_bytes(raw, label=relative)
        try:
            validate_no_final_labels(payload, label=relative)
        except BenchmarkContractError as exc:
            raise BenchmarkContractError(
                f"authority expected read violates the label boundary: {relative}",
                code="AUTH_LABEL_BOUNDARY_VIOLATION",
            ) from exc
        verified = VerifiedRead(
            relative,
            raw_digest(raw),
            stable_digest(payload),
        )
        _require(
            item
            == {
                "relative_path": verified.relative_path,
                "raw_sha256": verified.raw_sha256,
                "semantic_sha256": verified.semantic_sha256,
            },
            f"authority expected read digest mismatch: {relative}",
            code="AUTH_EXPECTED_READ_DIGEST_MISMATCH",
        )
        verified_reads.append(verified)

    reads_tuple = tuple(verified_reads)
    read_records = _verified_read_records(reads_tuple)
    threat_ceiling = (
        "cooperative-process integrity only; hostile same-process code, same-account "
        "file substitution, and intercepted I/O require a separately owned verifier, "
        "signed authority artifacts, and an append-only separately owned ledger"
    )
    verified = VerifiedAuthority(
        authority_id=bundle["authority_id"],
        authority_bundle_raw_sha256=raw_digest(bundle_raw),
        authority_contract_set_sha256=bundle["contract_set_sha256"],
        manifest_contract_set_sha256=manifest["contract_set_sha256"],
        expected_reads=reads_tuple,
        expected_read_set_sha256=stable_digest(read_records),
        candidate_registry_raw_sha256=raw_digest(candidate_raw),
        candidate_registry_semantic_sha256=stable_digest(candidate_registry),
        pair_registry_raw_sha256=raw_digest(pair_raw),
        pair_registry_semantic_sha256=stable_digest(pair_registry),
        evidence_generator_identity=bundle["evidence_generator_identity"],
        permit_ledger_relative_path=bundle["permit_ledger"]["relative_path"],
        preflight_sha256=stable_digest(
            {
                "authority_bundle_raw_sha256": raw_digest(bundle_raw),
                "contract_set_sha256": bundle["contract_set_sha256"],
                "expected_reads": read_records,
                "candidate_registry_raw_sha256": raw_digest(candidate_raw),
                "pair_registry_raw_sha256": raw_digest(pair_raw),
            }
        ),
        threat_ceiling=threat_ceiling,
    )
    context = _AuthorityContext(
        verified=verified,
        package_root=package_root,
        repo_root=repo_root,
        contract=contract,
        baseline_registry=baseline_registry,
        candidate_registry=candidate_registry,
        pair_registry=pair_registry,
    )
    for candidate in candidate_registry["candidates"]:
        if candidate["status"] == "RESOLVED":
            _validate_resolved_candidate_provenance(context, candidate)
    return context


def verify_authoritative_preflight() -> VerifiedAuthority:
    """Verify the fixed production authority package without caller evidence."""

    return _verify_authoritative_preflight_at(
        _PRODUCTION_PACKAGE_ROOT,
        _REPO_ROOT,
    ).verified


@dataclass(frozen=True)
class _AuthorityFixtureHarness:
    """Separate test harness; production APIs never accept these roots."""

    package_root: Path
    repo_root: Path

    def verify_authoritative_preflight(self) -> VerifiedAuthority:
        return _verify_authoritative_preflight_at(
            self.package_root,
            self.repo_root,
        ).verified

    def context(self) -> _AuthorityContext:
        return _verify_authoritative_preflight_at(
            self.package_root,
            self.repo_root,
        )

    def compute_registered_holm(
        self,
        candidate_id: str,
        family_id: str,
    ) -> HolmReport:
        return _compute_registered_holm_at(
            self.context(),
            candidate_id,
            family_id,
        )

    def compute_registered_pairwise_intervals(
        self,
        pair_family_id: str,
    ) -> PairwiseIntervalReport:
        return _compute_registered_pairwise_intervals_at(
            self.context(),
            pair_family_id,
        )

    def consume_bound_opening_permit(
        self,
        permit_raw: bytes,
    ) -> OpeningReceipt:
        return _consume_bound_opening_permit_at(
            self.context(),
            permit_raw,
        )

    def derive_registered_margin(
        self,
        candidate_id: str,
        baseline_id: str,
    ) -> float:
        return _derive_registered_margin_at(
            self.context(),
            candidate_id,
            baseline_id,
        )


def validate_partition_identity(partitions: Mapping[str, Sequence[str]]) -> None:
    seen: dict[str, str] = {}
    for partition, row_ids in partitions.items():
        local = list(row_ids)
        _require(len(local) == len(set(local)), f"{partition}: duplicate row id within partition")
        for row_id in local:
            _require(isinstance(row_id, str) and row_id, f"{partition}: row IDs must be nonempty strings")
            previous = seen.setdefault(row_id, partition)
            _require(
                previous == partition,
                f"row id {row_id!r} overlaps {previous!r} and {partition!r}",
            )


def _type7_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    _require(bool(ordered), "quantile requires at least one value")
    h = (len(ordered) - 1) * probability
    lower = math.floor(h)
    upper = math.ceil(h)
    if lower == upper:
        return float(ordered[lower])
    fraction = h - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def type7_quantile(values: Sequence[float], probability: float) -> float:
    _require(0.0 <= probability <= 1.0, "quantile probability must be in [0,1]")
    return _type7_quantile(values, probability)


_MARGIN_CONSTRUCTION_ID = "scryglass:deterministic-refresh-refit-pairs:v1.3"
_MARGIN_BASE_SEED = 2026072902


def margin_replicate_payload_sha256(bundle: Mapping[str, Any]) -> str:
    """Digest the frozen replica identity and values, excluding its own digest."""

    return stable_digest(
        {
            key: value
            for key, value in bundle.items()
            if key != "replicate_payload_sha256"
        }
    )


def _derive_margin_from_bundle(
    bundle: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> float:
    """Derive a margin only from a registry-bound development replica set."""

    _require(isinstance(bundle, Mapping), "margin replica bundle must be an object")
    expected_bindings = {
        "development_snapshot_sha256": binding.get(
            "development_snapshot_sha256"
        ),
        "baseline_binding_sha256": binding.get("baseline_binding_sha256"),
        "evaluation_rows_sha256": binding.get("evaluation_rows_sha256"),
        "procedure_sha256": binding.get("procedure_sha256"),
        "replicate_payload_sha256": binding.get(
            "replicate_payload_sha256"
        ),
        "independent_review_id": binding.get("independent_review_id"),
        "replica_construction_id": binding.get("replica_construction_id"),
    }
    for key, expected in expected_bindings.items():
        _require(
            bool(expected),
            f"authoritative margin binding is missing: {key}",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        if key.endswith("_sha256"):
            _require(
                _is_digest(expected),
                f"authoritative margin digest malformed: {key}",
                code="FAMILY_DERIVATION_MISMATCH",
            )
        _require(
            bundle.get(key) == expected,
            f"margin bundle binding mismatch: {key}",
            code="FAMILY_DERIVATION_MISMATCH",
        )
    exact_bundle_keys = {
        "development_snapshot_sha256",
        "baseline_binding_sha256",
        "evaluation_rows_sha256",
        "procedure_sha256",
        "independent_review_id",
        "replica_construction_id",
        "allowed_labels",
        "final_labels_read",
        "metric",
        "algorithm_identity_unchanged",
        "replicates",
        "replicate_payload_sha256",
    }
    _require(
        set(bundle) == exact_bundle_keys,
        "margin replica bundle schema is not exact",
    )
    _require(
        bundle.get("allowed_labels") == "REGISTERED_DEVELOPMENT_ONLY"
        and bundle.get("final_labels_read") is False,
        "margin replicas are not registered development-only evidence",
    )
    _require(
        bundle.get("metric") == "macro_regional_chronological_log_loss"
        and bundle.get("algorithm_identity_unchanged") is True,
        "margin refresh/refit changed metric or algorithm identity",
    )
    _require(
        expected_bindings["replica_construction_id"] == _MARGIN_CONSTRUCTION_ID,
        "margin replica construction identity mismatch",
    )
    replicates = bundle.get("replicates")
    _require(isinstance(replicates, list), "margin replicas must be an array")
    _require(len(replicates) >= 30, "margin derivation requires at least 30 replicates")
    _require(
        margin_replicate_payload_sha256(bundle) == bundle.get("replicate_payload_sha256"),
        "margin replica payload self-digest mismatch",
    )
    paired: list[float] = []
    seen_ids: set[str] = set()
    for index, replica in enumerate(replicates):
        _require(
            isinstance(replica, Mapping)
            and set(replica)
            == {
                "replicate_id",
                "seed",
                "evaluation_rows_sha256",
                "reference_log_loss",
                "refit_log_loss",
            },
            "margin replica record is malformed",
        )
        expected_id = f"refresh-refit-{index:04d}"
        _require(replica["replicate_id"] == expected_id, "margin replica order or identity mismatch")
        _require(replica["replicate_id"] not in seen_ids, "margin replica IDs are not unique")
        seen_ids.add(str(replica["replicate_id"]))
        _require(replica["seed"] == _MARGIN_BASE_SEED + index, "margin replica seed schedule mismatch")
        _require(
            replica["evaluation_rows_sha256"]
            == expected_bindings["evaluation_rows_sha256"],
            "margin replica evaluation-row identity mismatch",
        )
        reference = replica["reference_log_loss"]
        refit = replica["refit_log_loss"]
        _require(
            isinstance(reference, (int, float))
            and not isinstance(reference, bool)
            and isinstance(refit, (int, float))
            and not isinstance(refit, bool)
            and math.isfinite(float(reference))
            and math.isfinite(float(refit))
            and float(reference) >= 0.0
            and float(refit) >= 0.0,
            "margin log losses must be finite, non-boolean, and nonnegative",
            code="LOG_LOSS_INVALID",
        )
        paired.append(abs(float(refit) - float(reference)))
    return _type7_quantile(paired, 0.95)


def effective_cluster_count(cluster_sizes: Sequence[int]) -> float:
    _require(bool(cluster_sizes), "cluster-size vector is empty")
    _require(all(isinstance(size, int) and size > 0 for size in cluster_sizes), "cluster sizes must be positive integers")
    total = float(sum(cluster_sizes))
    return total * total / float(sum(size * size for size in cluster_sizes))


def inference_support_status(
    *,
    resolved_series: int,
    effective_clusters: float,
) -> str:
    _require(isinstance(resolved_series, int) and resolved_series >= 0, "resolved series must be a nonnegative integer")
    _require(isinstance(effective_clusters, (int, float)) and math.isfinite(float(effective_clusters)) and effective_clusters >= 0, "effective clusters must be finite and nonnegative")
    if resolved_series < 30 or float(effective_clusters) < 30.0:
        return "DESCRIPTIVE_ONLY_BLOCK_INFERENCE_AND_PROMOTION"
    return "INFERENCE_ELIGIBLE"


def canonical_intersection_id(cluster_ids: Sequence[str]) -> str:
    _require(
        bool(cluster_ids)
        and all(isinstance(item, str) and item for item in cluster_ids),
        "intersection cluster IDs must be nonempty strings",
    )
    encoded = [item.encode("utf-8") for item in cluster_ids]
    return "|".join(f"{len(item)}:{item.hex()}" for item in encoded)


def cgm_subset_sign(subset_size: int) -> int:
    _require(
        isinstance(subset_size, int) and subset_size > 0,
        "CGM subset size must be positive",
    )
    return 1 if subset_size % 2 == 1 else -1


_WEBB_SUPPORT = (
    -math.sqrt(3.0 / 2.0),
    -1.0,
    -math.sqrt(1.0 / 2.0),
    math.sqrt(1.0 / 2.0),
    1.0,
    math.sqrt(3.0 / 2.0),
)
_RADEMACHER_SUPPORT = (-1.0, 1.0)


def _bootstrap_multiplier_law(effective_clusters: float) -> str:
    _require(
        math.isfinite(effective_clusters) and effective_clusters >= 30.0,
        "selected bootstrap dimension has fewer than 30 effective clusters",
        code="unavailable_dependence_support",
    )
    if effective_clusters < 50.0:
        return "WEBB_SIX_POINT"
    return "RADEMACHER"


def _deterministic_wcr_multiplier(
    *,
    active_dimensions: Sequence[str],
    bootcluster_dimension: str,
    replicate: int,
    canonical_cluster: str,
    law: str,
) -> float:
    _require(
        0 <= replicate < _WCR_REPLICATES,
        "bootstrap replicate index is out of range",
    )
    _require(
        tuple(active_dimensions)
        in {
            ("participant_component_28d", "tournament_or_week"),
            (
                "participant_component_28d",
                "tournament_or_week",
                "patch",
            ),
        }
        and bootcluster_dimension in active_dimensions
        and bool(canonical_cluster),
        "WCR multiplier identity is incomplete",
    )
    support = (
        _WEBB_SUPPORT
        if law == "WEBB_SIX_POINT"
        else _RADEMACHER_SUPPORT
        if law == "RADEMACHER"
        else None
    )
    _require(support is not None, "bootstrap multiplier law is unknown")
    active = ",".join(active_dimensions)
    material = (
        f"mw-wcb-v2|A={active}|bootcluster={bootcluster_dimension}|"
        f"replicate={replicate}|cluster={canonical_cluster}|law={law}"
    ).encode("utf-8")
    digest = hmac.new(
        str(_WCR_SEED).encode("ascii"),
        material,
        sha256,
    ).digest()
    index = int.from_bytes(digest[:8], "big") % len(support)
    return support[index]


def macro_regional_series_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_leagues: Sequence[str] = _PRIMARY_LEAGUES,
) -> dict[str, float]:
    _require(bool(rows), "analysis rows are empty")
    identities = [row.get("series_id") for row in rows]
    _require(
        all(isinstance(item, str) and item for item in identities)
        and len(identities) == len(set(identities)),
        "series IDs must be nonempty and unique",
    )
    folds = sorted({str(row.get("fold_id")) for row in rows})
    _require(bool(folds) and all(fold != "None" for fold in folds), "fold IDs are missing")
    leagues = tuple(required_leagues)
    _require(
        bool(leagues)
        and len(leagues) == len(set(leagues))
        and set(leagues) <= set(_PRIMARY_LEAGUES),
        "macro-regional required-league set is invalid",
    )
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        fold = str(row.get("fold_id"))
        league = row.get("league_id")
        _require(league in leagues, f"unknown required league: {league}")
        counts[(fold, str(league))] = counts.get((fold, str(league)), 0) + 1
    _require(
        all((fold, league) in counts for fold in folds for league in leagues),
        "every chronological fold must contain every required league",
    )
    fold_count = len(folds)
    weights = {
        str(row["series_id"]): 1.0
        / (
            fold_count
            * len(leagues)
            * counts[(str(row["fold_id"]), str(row["league_id"]))]
        )
        for row in rows
    }
    _require(abs(sum(weights.values()) - 1.0) <= 1e-12, "macro-regional weights do not sum to one")
    return weights


def select_largest_cluster(
    rows: Sequence[Mapping[str, Any]],
    dimension: str,
) -> tuple[str, int]:
    counts: dict[str, set[str]] = {}
    for row in rows:
        cluster = row.get(dimension)
        series_id = row.get("series_id")
        _require(isinstance(cluster, str) and cluster, f"{dimension}: cluster ID missing")
        _require(isinstance(series_id, str) and series_id, "series ID missing")
        counts.setdefault(cluster, set()).add(series_id)
    _require(bool(counts), f"{dimension}: no clusters")
    maximum = max(len(series_ids) for series_ids in counts.values())
    selected = min(
        cluster
        for cluster, series_ids in counts.items()
        if len(series_ids) == maximum
    )
    return selected, maximum


_P = "participant_component_28d"
_T = "tournament_or_week"
_H = "patch"
_PRIMARY_ACTIVE = (_P, _T)
_PATCH_ACTIVE = (_P, _T, _H)
_EXECUTION_AUTHORITY_ID = "scryglass:authority-derived-preflight:v1.4"
_G1_SOURCE_BOUND_TRANSITION = (
    "PROHIBITED_IN_V1_4_REQUIRES_G1_UNIFIED_AUTHORITY_BUNDLE"
)
_G1_CANDIDATE_AUTHORITY_HANDOFF = {
    "status": "NOT_IMPLEMENTED_IN_V1_4",
    "transition": _G1_SOURCE_BOUND_TRANSITION,
    "required_boundary": (
        "SEALED_PRE_OUTCOME_INPUT_PLUS_SEPARATELY_OWNED_"
        "REPLAY_FROM_RAW_SCORING_DERIVATION"
    ),
    "authorization_path": (
        "oe_target_evidence.require_exact_human_authority"
        "->representation_rank_private_runner"
    ),
    "reviewer_identity": "KOI_MARI",
    "approval_scope": "private_retrospective_oe_target_v1",
    "authority_artifact": {
        "relative_path": (
            "data/lol/v2/models/draft-interactions/"
            "oe-private-target-authority-2026-07-29.json"
        ),
        "raw_sha256": (
            "b1d0a6e37abb9a74dee8689dc19ab54d"
            "30fd15516bd4ee454906a075d8f20788"
        ),
    },
    "approved_actions": ["model_fit", "rank_selection"],
    "excluded_scopes": [
        "final_temporal_holdout",
        "G9_final_opening",
        "promotion",
        "publication",
        "public_claims",
    ],
}
_G1_PAIR_AUTHORITY_HANDOFF = {
    "status": "NOT_IMPLEMENTED_IN_V1_4",
    "transition": _G1_SOURCE_BOUND_TRANSITION,
    "required_sample_identity": (
        "PAIR_CANDIDATES_SHARE_ONE_AUTHORITY_DERIVED_OUTPUT_SAMPLE"
    ),
    "unresolved_action": (
        "TYPED_UNRESOLVED_PAIR_BLOCKS_COMPLEXITY_SELECTION"
    ),
}


def _reject_source_bound_transition(
    transition: str,
    identifier: object = "",
) -> None:
    """Phase A has no source-bound authority; every such transition is closed."""

    suffix = f": {identifier}" if identifier else ""
    raise BenchmarkContractError(
        f"v1.4 source-independent freeze cannot authorize {transition}{suffix}",
        code="G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED",
    )


def _validate_g1_human_authority_artifact(
    handoff: Mapping[str, Any],
    repo_root: Path,
) -> None:
    """Reopen the exact existing approval metadata without reading target rows."""

    binding = handoff["authority_artifact"]
    raw = _read_regular_under_root(
        Path(repo_root),
        binding["relative_path"],
        purpose="G1 human authority artifact",
    )
    _require(
        raw_digest(raw) == binding["raw_sha256"],
        "G1 human authority artifact raw identity drifted",
        code="G1_HUMAN_AUTHORITY_ARTIFACT_MISMATCH",
    )
    payload = _parse_json_bytes(raw, label="G1 human authority artifact")
    _require(
        isinstance(payload, Mapping)
        and payload.get("reviewer_identity") == handoff["reviewer_identity"]
        and payload.get("approval_scope") == handoff["approval_scope"]
        and payload.get("approved_actions") == handoff["approved_actions"]
        and payload.get("final_temporal_holdout_sealed") is True,
        "G1 human authority artifact scope or sealed boundary changed",
        code="G1_HUMAN_AUTHORITY_ARTIFACT_MISMATCH",
    )


def _load_bound_adapter(
    root: Path,
    *,
    locator: str,
    expected_code_raw_sha256: str,
    entry_point: str,
    purpose: str,
) -> Callable[[dict[str, Any], dict[str, Any]], Any]:
    """Resolve a callable only from the exact reopened adapter bytes."""

    _reject_source_bound_transition("adapter execution", purpose)
    _require(
        isinstance(entry_point, str)
        and entry_point.count(":") == 1,
        f"{purpose}: executable entry point is malformed",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    module_name, attribute = entry_point.split(":", 1)
    expected_module = locator[:-3].replace("/", ".") if locator.endswith(".py") else ""
    _require(
        module_name == expected_module
        and bool(attribute)
        and all(part.isidentifier() for part in attribute.split(".")),
        f"{purpose}: entry point does not name the bound adapter",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    raw = _read_regular_under_root(root, locator, purpose=f"{purpose} adapter")
    _require(
        raw_digest(raw) == expected_code_raw_sha256,
        f"{purpose}: adapter code identity drifted",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    path = Path(root).joinpath(*PurePosixPath(locator).parts)
    synthetic_name = f"_scryglass_bound_{sha256(locator.encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(synthetic_name, path)
    _require(
        spec is not None and spec.loader is not None,
        f"{purpose}: adapter cannot be loaded",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    module = importlib.util.module_from_spec(spec)
    _require(
        isinstance(module, types.ModuleType),
        f"{purpose}: adapter module is invalid",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    try:
        spec.loader.exec_module(module)
        target: Any = module
        for part in attribute.split("."):
            target = getattr(target, part)
    except Exception as exc:
        raise BenchmarkContractError(
            f"{purpose}: adapter entry point cannot be resolved",
            code="CANDIDATE_REGISTRY_MISMATCH",
        ) from exc
    _require(
        callable(target),
        f"{purpose}: adapter entry point is not callable",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    return target


def _invoke_bound_fixture(
    root: Path,
    *,
    locator: str,
    expected_code_raw_sha256: str,
    entry_point: str,
    fixture_input: Mapping[str, Any],
    config: Mapping[str, Any],
    fixture_output: Any,
    fixture_sha256: str,
    execution_authority: Mapping[str, Any],
    purpose: str,
) -> str:
    """Run a label-free reachability fixture; it does not authorize forecasts."""

    _reject_source_bound_transition("fixture execution", purpose)
    target = _load_bound_adapter(
        root,
        locator=locator,
        expected_code_raw_sha256=expected_code_raw_sha256,
        entry_point=entry_point,
        purpose=purpose,
    )
    validate_no_final_labels(fixture_input, label=f"{purpose} fixture input")
    validate_no_final_labels(fixture_output, label=f"{purpose} fixture output")
    _require(
        fixture_sha256
        == stable_digest({"input": fixture_input, "output": fixture_output}),
        f"{purpose}: fixture identity drifted",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    try:
        observed = target(dict(fixture_input), dict(config))
    except Exception as exc:
        raise BenchmarkContractError(
            f"{purpose}: label-free fixture execution failed",
            code="CANDIDATE_REGISTRY_MISMATCH",
        ) from exc
    _require(
        canonical_json(observed) == canonical_json(fixture_output),
        f"{purpose}: fixture output mismatch",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    receipt = stable_digest(
        {
            "authority_id": _EXECUTION_AUTHORITY_ID,
            "code_raw_sha256": expected_code_raw_sha256,
            "entry_point": entry_point,
            "config_sha256": stable_digest(config),
            "fixture_sha256": fixture_sha256,
            "observed_output_sha256": stable_digest(observed),
        }
    )
    _require(
        execution_authority
        == {
            "kind": "AUTHORITY_SIDE_REEXECUTION",
            "authority_id": _EXECUTION_AUTHORITY_ID,
            "receipt_sha256": receipt,
        },
        f"{purpose}: execution authority is absent, self-labelled, or stale",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    return receipt


def _valid_execution_authority_record(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"kind", "authority_id", "receipt_sha256"}
        and value.get("kind") == "AUTHORITY_SIDE_REEXECUTION"
        and value.get("authority_id") == _EXECUTION_AUTHORITY_ID
        and _is_digest(value.get("receipt_sha256"))
    )


_FULL_FORECAST_AUTHORITY_FIELDS = {
    "kind",
    "authority_id",
    "forecast_role",
    "model_id",
    "slot_id",
    "output_id",
    "code_raw_sha256",
    "entry_point",
    "config_sha256",
    "environment_lock_raw_sha256",
    "source_snapshot_raw_sha256",
    "partition_bindings_raw_sha256",
    "batch_input_sha256",
    "batch_output_sha256",
    "receipt_sha256",
}


def _valid_full_forecast_authority_record(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == _FULL_FORECAST_AUTHORITY_FIELDS
        and value.get("kind")
        == "AUTHORITY_SIDE_FULL_FORECAST_REEXECUTION"
        and value.get("authority_id") == _EXECUTION_AUTHORITY_ID
        and value.get("forecast_role") in {"candidate", "baseline"}
        and all(
            isinstance(value.get(field), str) and bool(value[field])
            for field in ("model_id", "slot_id", "output_id", "entry_point")
        )
        and all(
            _is_digest(value.get(field))
            for field in (
                "code_raw_sha256",
                "config_sha256",
                "environment_lock_raw_sha256",
                "source_snapshot_raw_sha256",
                "partition_bindings_raw_sha256",
                "batch_input_sha256",
                "batch_output_sha256",
                "receipt_sha256",
            )
        )
    )


_FULL_FORECAST_PAIR_AUTHORITY_FIELDS = {
    "kind",
    "authority_id",
    "slot_id",
    "output_id",
    "candidate_receipt_sha256",
    "baseline_receipt_sha256",
    "receipt_sha256",
}


def _valid_full_forecast_pair_authority_record(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == _FULL_FORECAST_PAIR_AUTHORITY_FIELDS
        and value.get("kind")
        == "AUTHORITY_SIDE_FULL_FORECAST_PAIR_REEXECUTION"
        and value.get("authority_id") == _EXECUTION_AUTHORITY_ID
        and all(
            isinstance(value.get(field), str) and bool(value[field])
            for field in ("slot_id", "output_id")
        )
        and all(
            _is_digest(value.get(field))
            for field in (
                "candidate_receipt_sha256",
                "baseline_receipt_sha256",
                "receipt_sha256",
            )
        )
    )


def _canonical_forecast_batch_input(
    slot: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    source_snapshot_relative_path: str,
    source_snapshot_raw_sha256: str,
    partition_bindings_relative_path: str,
    partition_bindings_raw_sha256: str,
) -> dict[str, Any]:
    """Derive the sole label-free adapter input from validated WCR rows."""

    row_fields = (
        "series_id",
        "fold_id",
        "registered_fold_ids",
        "league_id",
        "map_ids",
        _P,
        _T,
        _H,
        "game_side",
        "roster_change",
        "international_event",
        "draft_depth",
        "exact_roster_id",
        "series_order_within_exact_roster_tournament",
        "strength_source_id",
        "pre_outcome_candidate_strength",
        "pre_outcome_baseline_strength",
        "resolved",
    )
    ordered_rows = [
        {field: row[field] for field in row_fields}
        for row in rows
    ]
    batch = {
        "schema_version": "g0-full-forecast-batch-input-v1.4",
        "output_id": slot["output_id"],
        "source_snapshot": {
            "relative_path": source_snapshot_relative_path,
            "raw_sha256": source_snapshot_raw_sha256,
        },
        "partition_bindings": {
            "relative_path": partition_bindings_relative_path,
            "raw_sha256": partition_bindings_raw_sha256,
        },
        "row_order_sha256": stable_digest(
            [row["series_id"] for row in ordered_rows]
        ),
        "rows": ordered_rows,
    }
    validate_no_final_labels(batch, label="full forecast batch input")
    return batch


def _expected_forecast_batch_output(
    slot: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> dict[str, Any]:
    _require(
        role in {"candidate", "baseline"},
        "full forecast role is invalid",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    probabilities = f"{role}_probabilities"
    return {
        "schema_version": "g0-full-forecast-batch-output-v1.4",
        "output_id": slot["output_id"],
        "rows": [
            {
                "series_id": row["series_id"],
                "map_ids": row["map_ids"],
                "probabilities": row[probabilities],
            }
            for row in rows
        ],
    }


def _invoke_bound_forecast_batch(
    root: Path,
    *,
    locator: str,
    expected_code_raw_sha256: str,
    entry_point: str,
    config: Mapping[str, Any],
    environment_lock_raw_sha256: str,
    source_snapshot_relative_path: str,
    source_snapshot_raw_sha256: str,
    partition_bindings_relative_path: str,
    partition_bindings_raw_sha256: str,
    slot: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    role: str,
    model_id: str,
    purpose: str,
) -> dict[str, Any]:
    """Reexecute the complete forecast vector and derive its authority receipt."""

    _reject_source_bound_transition("full forecast execution", purpose)
    target = _load_bound_adapter(
        root,
        locator=locator,
        expected_code_raw_sha256=expected_code_raw_sha256,
        entry_point=entry_point,
        purpose=purpose,
    )
    batch_input = _canonical_forecast_batch_input(
        slot,
        rows,
        source_snapshot_relative_path=source_snapshot_relative_path,
        source_snapshot_raw_sha256=source_snapshot_raw_sha256,
        partition_bindings_relative_path=partition_bindings_relative_path,
        partition_bindings_raw_sha256=partition_bindings_raw_sha256,
    )
    expected_output = _expected_forecast_batch_output(
        slot,
        rows,
        role=role,
    )
    try:
        observed = target(dict(batch_input), dict(config))
    except Exception as exc:
        raise BenchmarkContractError(
            f"{purpose}: full forecast reexecution failed",
            code="CANDIDATE_REGISTRY_MISMATCH",
        ) from exc
    validate_no_final_labels(observed, label=f"{purpose} full forecast output")
    _require(
        canonical_json(observed) == canonical_json(expected_output),
        f"{purpose}: full forecast vector differs from registered evidence",
        code="FORECAST_IDENTITY_MISMATCH",
    )
    batch_input_sha256 = stable_digest(batch_input)
    batch_output_sha256 = stable_digest(observed)
    config_sha256 = stable_digest(config)
    receipt_sha256 = stable_digest(
        {
            "authority_id": _EXECUTION_AUTHORITY_ID,
            "kind": "AUTHORITY_SIDE_FULL_FORECAST_REEXECUTION",
            "forecast_role": role,
            "model_id": model_id,
            "slot_id": slot["slot_id"],
            "output_id": slot["output_id"],
            "code_raw_sha256": expected_code_raw_sha256,
            "entry_point": entry_point,
            "config_sha256": config_sha256,
            "environment_lock_raw_sha256": environment_lock_raw_sha256,
            "source_snapshot_raw_sha256": source_snapshot_raw_sha256,
            "partition_bindings_raw_sha256": partition_bindings_raw_sha256,
            "batch_input_sha256": batch_input_sha256,
            "batch_output_sha256": batch_output_sha256,
        }
    )
    return {
        "kind": "AUTHORITY_SIDE_FULL_FORECAST_REEXECUTION",
        "authority_id": _EXECUTION_AUTHORITY_ID,
        "forecast_role": role,
        "model_id": model_id,
        "slot_id": slot["slot_id"],
        "output_id": slot["output_id"],
        "code_raw_sha256": expected_code_raw_sha256,
        "entry_point": entry_point,
        "config_sha256": config_sha256,
        "environment_lock_raw_sha256": environment_lock_raw_sha256,
        "source_snapshot_raw_sha256": source_snapshot_raw_sha256,
        "partition_bindings_raw_sha256": partition_bindings_raw_sha256,
        "batch_input_sha256": batch_input_sha256,
        "batch_output_sha256": batch_output_sha256,
        "receipt_sha256": receipt_sha256,
    }


def _full_forecast_pair_authority(
    slot: Mapping[str, Any],
    candidate_authority: Mapping[str, Any],
    baseline_authority: Mapping[str, Any],
) -> dict[str, Any]:
    _reject_source_bound_transition(
        "full forecast pair receipt",
        slot.get("slot_id"),
    )
    _require(
        _valid_full_forecast_authority_record(candidate_authority)
        and candidate_authority["forecast_role"] == "candidate"
        and _valid_full_forecast_authority_record(baseline_authority)
        and baseline_authority["forecast_role"] == "baseline",
        "full forecast authorities are malformed or swapped",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    receipt_sha256 = stable_digest(
        {
            "authority_id": _EXECUTION_AUTHORITY_ID,
            "slot_id": slot["slot_id"],
            "output_id": slot["output_id"],
            "candidate_authority": dict(candidate_authority),
            "baseline_authority": dict(baseline_authority),
        }
    )
    return {
        "kind": "AUTHORITY_SIDE_FULL_FORECAST_PAIR_REEXECUTION",
        "authority_id": _EXECUTION_AUTHORITY_ID,
        "slot_id": slot["slot_id"],
        "output_id": slot["output_id"],
        "candidate_receipt_sha256": candidate_authority["receipt_sha256"],
        "baseline_receipt_sha256": baseline_authority["receipt_sha256"],
        "receipt_sha256": receipt_sha256,
    }


def _derived_input_sha256(row: Mapping[str, Any]) -> str:
    """Derive the common observation identity; never trust a caller label."""

    return stable_digest(
        {
            "series_id": row["series_id"],
            "fold_id": row["fold_id"],
            "league_id": row["league_id"],
            "output_id": row["output_id"],
            "stratum_id": row["stratum_id"],
            "map_ids": row["map_ids"],
            "outcome": row["outcome"],
            "participant_component_28d": row[_P],
            "tournament_or_week": row[_T],
            "patch": row[_H],
            "game_side": row["game_side"],
            "roster_change": row["roster_change"],
            "international_event": row["international_event"],
            "draft_depth": row["draft_depth"],
            "exact_roster_id": row["exact_roster_id"],
            "series_order_within_exact_roster_tournament": row[
                "series_order_within_exact_roster_tournament"
            ],
            "strength_source_id": row["strength_source_id"],
            "pre_outcome_candidate_strength": row[
                "pre_outcome_candidate_strength"
            ],
            "pre_outcome_baseline_strength": row[
                "pre_outcome_baseline_strength"
            ],
        }
    )


def _derived_prediction_sha256(
    row: Mapping[str, Any],
    probabilities: Sequence[float],
) -> str:
    """Derive a series forecast identity from exact ordered map predictions."""

    return stable_digest(
        {
            "series_id": row["series_id"],
            "map_ids": row["map_ids"],
            "probabilities": list(probabilities),
        }
    )


def _derived_input_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return stable_digest(
        [
            {
                "series_id": row["series_id"],
                "input_sha256": _derived_input_sha256(row),
            }
            for row in rows
        ]
    )


def _derived_prediction_rows_sha256(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> str:
    _require(
        role in {"candidate", "baseline"},
        "forecast role is invalid",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    vector_field = f"{role}_probabilities"
    return stable_digest(
        [
            {
                "series_id": row["series_id"],
                "prediction_sha256": _derived_prediction_sha256(
                    row, row[vector_field]
                ),
            }
            for row in rows
        ]
    )


@dataclass(frozen=True)
class _PreparedWCR:
    rows: tuple[dict[str, Any], ...]
    candidate_id: str
    baseline_id: str
    output_id: str
    stratum_id: str
    score_kind: str
    canonical_series_ids: tuple[str, ...]
    analysis_rows_sha256: str
    differences: np.ndarray
    weights: np.ndarray
    cluster_values: dict[str, tuple[str, ...]]
    effective_clusters: dict[str, float]


@dataclass(frozen=True)
class WCRBootclusterResult:
    execution_id: str
    active_dimensions: tuple[str, ...]
    bootcluster_dimension: str
    point_estimate: float
    threshold: float
    standard_error: float
    observed_t: float
    lower_tail_p: float
    # This is an inversion of the *unadjusted* one-sided test.  It is not a
    # multiplicity-adjusted endpoint and must never be presented as one.
    unadjusted_one_sided_95_upper_bound: float
    # Type-7 quantities are retained strictly as diagnostics.  They are not
    # confidence bounds and do not participate in a promotion decision.
    type7_linearized_upper_bound_diagnostic: float
    type7_bootstrap_t_q05_diagnostic: float
    type7_bootstrap_t_q95_diagnostic: float
    bootstrap_t: tuple[float, ...]
    bootstrap_t_sha256: str
    multiplier_law: str
    replicates: int
    effective_clusters: float
    analysis_rows_sha256: str
    source_code_raw_sha256: str


@dataclass(frozen=True)
class WCRSuiteResult:
    primary: tuple[WCRBootclusterResult, ...]
    patch: tuple[WCRBootclusterResult, ...]
    primary_decisions_agree: bool
    patch_decisions_agree: bool
    patch_matches_primary: bool
    passes: bool


@dataclass(frozen=True)
class WCRConsensusReport:
    full_sample: WCRSuiteResult
    leave_largest: tuple[tuple[str, str, WCRSuiteResult], ...]
    passes: bool


def _valid_log_loss(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _required_leagues_for_wcr_cell(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    stratum_ids = {
        row.get("stratum_id")
        for row in rows
    }
    _require(
        len(stratum_ids) == 1,
        "WCR cell has mixed stratum identities",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    stratum_id = next(iter(stratum_ids))
    if isinstance(stratum_id, str) and stratum_id.startswith("league:"):
        league = stratum_id.split(":", 1)[1]
        _require(
            league in _PRIMARY_LEAGUES
            and {row.get("league_id") for row in rows} == {league},
            "league critical cell does not match its registered league",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        return (league,)
    if (
        isinstance(stratum_id, str)
        and stratum_id.startswith("international_event:")
    ):
        observed = tuple(
            sorted({str(row.get("league_id")) for row in rows})
        )
        _require(
            bool(observed)
            and set(observed) <= set(_PRIMARY_LEAGUES),
            "international critical cell has an invalid source-league set",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        return observed
    return _PRIMARY_LEAGUES


def _prepare_wcr_rows(
    rows: Sequence[Mapping[str, Any]],
) -> _PreparedWCR:
    required = {
        "series_id",
        "fold_id",
        "league_id",
        "candidate_id",
        "baseline_id",
        "output_id",
        "stratum_id",
        "score_kind",
        "map_ids",
        "candidate_probabilities",
        "baseline_probabilities",
        "outcome",
        "macro_weight",
        "registered_fold_ids",
        "game_side",
        "roster_change",
        "international_event",
        "draft_depth",
        "exact_roster_id",
        "series_order_within_exact_roster_tournament",
        "strength_source_id",
        "pre_outcome_candidate_strength",
        "pre_outcome_baseline_strength",
        _P,
        _T,
        _H,
        "resolved",
        "input_sha256",
        "candidate_prediction_sha256",
        "baseline_prediction_sha256",
        "row_order_sha256",
    }
    canonical_rows = tuple(
        dict(row)
        for row in sorted(rows, key=lambda row: str(row.get("series_id")))
    )
    _require(
        len(canonical_rows) >= 30,
        "WCR requires at least 30 resolved series",
        code="unavailable_dependence_support",
    )
    for row in canonical_rows:
        _require(
            set(row) == required,
            "WCR analysis row schema is not exact",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        for identity_field in (
            "series_id",
            "fold_id",
            "league_id",
            "candidate_id",
            "baseline_id",
            "output_id",
            "stratum_id",
            "score_kind",
        ):
            _require(
                isinstance(row[identity_field], str)
                and bool(row[identity_field]),
                f"WCR analysis row identity is malformed: {identity_field}",
                code="FAMILY_DERIVATION_MISMATCH",
            )
        _require(
            row["resolved"] is True,
            "unresolved series cannot enter WCR inference",
            code="unavailable_dependence_support",
        )
        for digest_field in (
            "input_sha256",
            "candidate_prediction_sha256",
            "baseline_prediction_sha256",
            "row_order_sha256",
        ):
            _require(
                _is_digest(row[digest_field]),
                f"WCR analysis row {digest_field} is malformed",
                code="FAMILY_DERIVATION_MISMATCH",
            )
        for dimension in _PATCH_ACTIVE:
            _require(
                isinstance(row[dimension], str) and bool(row[dimension]),
                f"WCR cluster dimension is missing: {dimension}",
                code="unavailable_dependence_support",
            )
        map_ids = row["map_ids"]
        candidate_probabilities = row["candidate_probabilities"]
        baseline_probabilities = row["baseline_probabilities"]
        outcomes = row["outcome"]
        _require(
            isinstance(map_ids, list)
            and isinstance(candidate_probabilities, list)
            and isinstance(baseline_probabilities, list)
            and isinstance(outcomes, list)
            and bool(map_ids)
            and all(
                isinstance(value, str) and bool(value)
                for value in map_ids
            )
            and len(map_ids) == len(set(map_ids))
            and len(candidate_probabilities)
            == len(baseline_probabilities)
            == len(outcomes)
            == len(map_ids)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 < float(value) < 1.0
                for value in (
                    candidate_probabilities + baseline_probabilities
                )
            ),
            "map IDs and probability vectors must be aligned, unique, finite, and strictly inside (0,1)",
            code="PROBABILITY_INVALID",
        )
        _require(
            row["input_sha256"] == _derived_input_sha256(row),
            "WCR input identity is caller-labelled or drifted",
            code="FORECAST_IDENTITY_MISMATCH",
        )
        _require(
            row["candidate_prediction_sha256"]
            == _derived_prediction_sha256(
                row, row["candidate_probabilities"]
            )
            and row["baseline_prediction_sha256"]
            == _derived_prediction_sha256(
                row, row["baseline_probabilities"]
            ),
            "WCR forecast identity is caller-labelled or drifted",
            code="FORECAST_IDENTITY_MISMATCH",
        )
        _require(
            all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value in {0, 1}
                for value in outcomes
            ),
            "WCR outcome vectors must contain only integer 0/1 values",
            code="OUTCOME_SUPPORT_INVALID",
        )
        _require(
            row["score_kind"] in {"LOG_LOSS", "BRIER"},
            "WCR score kind is not registered",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        _require(
            row["game_side"] in {"BLUE", "RED"}
            and row["roster_change"]
            in {
                "FIRST_TOURNAMENT_NEW_EXACT_ROSTER",
                "STABLE_EXACT_ROSTER",
            }
            and row["international_event"]
            in {"NONE", "MSI", "EWC", "OTHER_NAMED_EVENT"}
            and isinstance(row["draft_depth"], int)
            and not isinstance(row["draft_depth"], bool)
            and -1 <= row["draft_depth"] <= 10,
            "WCR critical-cell attributes are malformed",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        _require(
            isinstance(row["exact_roster_id"], str)
            and bool(row["exact_roster_id"])
            and isinstance(
                row["series_order_within_exact_roster_tournament"], int
            )
            and not isinstance(
                row["series_order_within_exact_roster_tournament"], bool
            )
            and row["series_order_within_exact_roster_tournament"] >= 1
            and isinstance(row["strength_source_id"], str)
            and bool(row["strength_source_id"])
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in (
                    row["pre_outcome_candidate_strength"],
                    row["pre_outcome_baseline_strength"],
                )
            ),
            "WCR roster-order or pre-outcome strength authority is malformed",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        _require(
            isinstance(row["registered_fold_ids"], list)
            and bool(row["registered_fold_ids"])
            and row["registered_fold_ids"]
            == sorted(row["registered_fold_ids"])
            and len(row["registered_fold_ids"])
            == len(set(row["registered_fold_ids"]))
            and all(
                isinstance(value, str) and bool(value)
                for value in row["registered_fold_ids"]
            ),
            "WCR registered fold inventory is malformed",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        _require(
            isinstance(row["macro_weight"], (int, float))
            and not isinstance(row["macro_weight"], bool)
            and math.isfinite(float(row["macro_weight"]))
            and float(row["macro_weight"]) > 0.0,
            "WCR macro weight is malformed",
            code="MACRO_WEIGHT_MISMATCH",
        )

    series_ids = tuple(str(row["series_id"]) for row in canonical_rows)
    _require(
        all(series_ids)
        and len(series_ids) == len(set(series_ids)),
        "WCR series IDs must be nonempty and unique",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    candidate_ids = {row["candidate_id"] for row in canonical_rows}
    baseline_ids = {row["baseline_id"] for row in canonical_rows}
    output_ids = {row["output_id"] for row in canonical_rows}
    stratum_ids = {row["stratum_id"] for row in canonical_rows}
    score_kinds = {row["score_kind"] for row in canonical_rows}
    _require(
        len(candidate_ids) == 1
        and len(baseline_ids) == 1
        and len(output_ids) == 1
        and len(stratum_ids) == 1
        and len(score_kinds) == 1
        and all(
            isinstance(value, str) and bool(value)
            for value in (
                candidate_ids
                | baseline_ids
                | output_ids
                | stratum_ids
                | score_kinds
            )
        ),
        "WCR candidate, baseline, output, and stratum identities must be constant",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    registered_fold_sets = {
        tuple(row["registered_fold_ids"]) for row in canonical_rows
    }
    actual_fold_ids = {row["fold_id"] for row in canonical_rows}
    _require(
        len(registered_fold_sets) == 1
        and set(next(iter(registered_fold_sets))) == actual_fold_ids,
        "WCR rows do not preserve the complete registered fold set",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    _require(
        next(iter(output_ids)) in _OUTPUT_IDS,
        "WCR output identity is unregistered",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    stratum_id = next(iter(stratum_ids))
    if stratum_id.startswith("patch:"):
        _require(
            stratum_id != "patch:each-held-out-major-minor",
            "patch template must expand to exact held-out patch cells before inference",
            code="CANDIDATE_SLOT_UNRESOLVED",
        )
        _require(
            {row[_H] for row in canonical_rows}
            == {stratum_id.split(":", 1)[1]},
            "patch critical rows do not match the registered patch",
            code="FAMILY_DERIVATION_MISMATCH",
        )
    elif stratum_id.startswith("game_side:"):
        _require(
            {row["game_side"] for row in canonical_rows}
            == {stratum_id.split(":", 1)[1]},
            "game-side critical rows do not match the registered cell",
            code="FAMILY_DERIVATION_MISMATCH",
        )
    elif stratum_id.startswith("roster_change:"):
        _require(
            {row["roster_change"] for row in canonical_rows}
            == {stratum_id.split(":", 1)[1]},
            "roster critical rows do not match the registered cell",
            code="FAMILY_DERIVATION_MISMATCH",
        )
    elif stratum_id.startswith("international_event:"):
        _require(
            {row["international_event"] for row in canonical_rows}
            == {stratum_id.split(":", 1)[1]},
            "international critical rows do not match the registered event",
            code="FAMILY_DERIVATION_MISMATCH",
        )
    elif stratum_id.startswith("draft_depth:"):
        _require(
            {row["draft_depth"] for row in canonical_rows}
            == {int(stratum_id.split(":", 1)[1])},
            "draft-depth critical rows do not match the registered depth",
            code="FAMILY_DERIVATION_MISMATCH",
        )
    observed_outcomes = {
        outcome
        for row in canonical_rows
        for outcome in row["outcome"]
    }
    _require(
        observed_outcomes == {0, 1},
        "inferential WCR cell must contain both outcomes",
        code="OUTCOME_SUPPORT_INVALID",
    )
    row_order_digest = stable_digest(list(series_ids))
    _require(
        {row["row_order_sha256"] for row in canonical_rows}
        == {row_order_digest},
        "WCR row-order identity does not bind the canonical series order",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    weights_by_id = macro_regional_series_weights(
        canonical_rows,
        required_leagues=_required_leagues_for_wcr_cell(canonical_rows),
    )
    _require(
        all(
            abs(
                float(row["macro_weight"])
                - weights_by_id[str(row["series_id"])]
            )
            <= 1e-12
            for row in canonical_rows
        ),
        "registered macro weights differ from the frozen league-fold estimand",
        code="MACRO_WEIGHT_MISMATCH",
    )
    weights = np.asarray(
        [weights_by_id[series_id] for series_id in series_ids],
        dtype=np.float64,
    )
    score_kind = next(iter(score_kinds))
    differences_list: list[float] = []
    for row in canonical_rows:
        candidate_scores: list[float] = []
        baseline_scores: list[float] = []
        for candidate_p, baseline_p, outcome in zip(
            row["candidate_probabilities"],
            row["baseline_probabilities"],
            row["outcome"],
        ):
            if score_kind == "LOG_LOSS":
                candidate_scores.append(
                    -(
                        outcome * math.log(float(candidate_p))
                        + (1 - outcome)
                        * math.log1p(-float(candidate_p))
                    )
                )
                baseline_scores.append(
                    -(
                        outcome * math.log(float(baseline_p))
                        + (1 - outcome)
                        * math.log1p(-float(baseline_p))
                    )
                )
            else:
                candidate_scores.append(
                    (float(candidate_p) - outcome) ** 2
                )
                baseline_scores.append(
                    (float(baseline_p) - outcome) ** 2
                )
        differences_list.append(
            float(np.mean(candidate_scores))
            - float(np.mean(baseline_scores))
        )
    differences = np.asarray(differences_list, dtype=np.float64)
    cluster_values = {
        dimension: tuple(str(row[dimension]) for row in canonical_rows)
        for dimension in _PATCH_ACTIVE
    }
    effective: dict[str, float] = {}
    for dimension, values in cluster_values.items():
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        effective[dimension] = effective_cluster_count(list(counts.values()))
    return _PreparedWCR(
        rows=canonical_rows,
        candidate_id=next(iter(candidate_ids)),
        baseline_id=next(iter(baseline_ids)),
        output_id=next(iter(output_ids)),
        stratum_id=next(iter(stratum_ids)),
        score_kind=score_kind,
        canonical_series_ids=series_ids,
        analysis_rows_sha256=stable_digest(canonical_rows),
        differences=differences,
        weights=weights,
        cluster_values=cluster_values,
        effective_clusters=effective,
    )


def _build_cgm_plan(
    prepared: _PreparedWCR,
    active_dimensions: Sequence[str],
) -> tuple[tuple[int, float, tuple[np.ndarray, ...]], ...]:
    active = tuple(active_dimensions)
    _require(
        active in {_PRIMARY_ACTIVE, _PATCH_ACTIVE},
        "active WCR dimensions are not a frozen run variant",
        code="unavailable_dependence_support",
    )
    plan: list[tuple[int, float, tuple[np.ndarray, ...]]] = []
    row_count = len(prepared.canonical_series_ids)
    for size in range(1, len(active) + 1):
        for subset in combinations(active, size):
            grouped: dict[str, list[int]] = {}
            for index in range(row_count):
                key = canonical_intersection_id(
                    [
                        prepared.cluster_values[dimension][index]
                        for dimension in subset
                    ]
                )
                grouped.setdefault(key, []).append(index)
            group_count = len(grouped)
            _require(
                group_count >= 2,
                "a required CGM intersection has fewer than two clusters",
                code="unavailable_dependence_support",
            )
            groups = tuple(
                np.asarray(grouped[key], dtype=np.int64)
                for key in sorted(grouped)
            )
            plan.append(
                (
                    cgm_subset_sign(size),
                    group_count / (group_count - 1.0),
                    groups,
                )
            )
    return tuple(plan)


def _cgm_variance(
    weighted_scores: np.ndarray,
    plan: Sequence[tuple[int, float, tuple[np.ndarray, ...]]],
) -> np.ndarray | float:
    scores = np.asarray(weighted_scores, dtype=np.float64)
    _require(
        scores.ndim in {1, 2},
        "CGM scores must be one- or two-dimensional",
        code="unavailable_dependence_support",
    )
    if scores.ndim == 1:
        variance = 0.0
        for sign, correction, groups in plan:
            sums = np.asarray(
                [float(np.sum(scores[group])) for group in groups],
                dtype=np.float64,
            )
            variance += sign * correction * float(np.dot(sums, sums))
        return float(variance)
    variance_rows = np.zeros(scores.shape[0], dtype=np.float64)
    for sign, correction, groups in plan:
        sums = np.stack(
            [np.sum(scores[:, group], axis=1) for group in groups],
            axis=1,
        )
        variance_rows += (
            sign * correction * np.sum(np.square(sums), axis=1)
        )
    return variance_rows


def _wcr_multiplier_matrix(
    prepared: _PreparedWCR,
    active_dimensions: Sequence[str],
    bootcluster_dimension: str,
    law: str,
    *,
    attempts: int,
) -> np.ndarray:
    _require(
        1 <= attempts <= _WCR_REPLICATES,
        "WCR attempt count is outside the frozen range",
        code="unavailable_dependence_support",
    )
    values = prepared.cluster_values[bootcluster_dimension]
    clusters = sorted(set(values))
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    by_cluster = np.empty((attempts, len(clusters)), dtype=np.float64)
    for replicate in range(attempts):
        for cluster, index in cluster_index.items():
            by_cluster[replicate, index] = _deterministic_wcr_multiplier(
                active_dimensions=active_dimensions,
                bootcluster_dimension=bootcluster_dimension,
                replicate=replicate,
                canonical_cluster=cluster,
                law=law,
            )
    indices = np.asarray(
        [cluster_index[value] for value in values],
        dtype=np.int64,
    )
    return by_cluster[:, indices]


def _wcr_distribution_at_threshold(
    prepared: _PreparedWCR,
    active_dimensions: Sequence[str],
    theta0: float,
    multipliers: np.ndarray,
    plan: Sequence[tuple[int, float, tuple[np.ndarray, ...]]],
) -> tuple[float, float, float, np.ndarray]:
    _require(
        isinstance(theta0, (int, float))
        and not isinstance(theta0, bool)
        and math.isfinite(float(theta0)),
        "WCR threshold must be finite",
        code="unavailable_dependence_support",
    )
    theta_hat = float(
        np.sum(prepared.weights * prepared.differences)
    )
    unrestricted = prepared.differences - theta_hat
    observed_variance = float(
        _cgm_variance(prepared.weights * unrestricted, plan)
    )
    _require(
        observed_variance > 1e-12,
        "observed CGM variance is nonpositive or below the frozen floor",
        code="unavailable_dependence_support",
    )
    standard_error = math.sqrt(observed_variance)
    observed_t = (theta_hat - float(theta0)) / standard_error

    restricted = prepared.differences - float(theta0)
    starred = float(theta0) + multipliers * restricted[np.newaxis, :]
    theta_star = np.sum(
        starred * prepared.weights[np.newaxis, :],
        axis=1,
    )
    residual_star = starred - theta_star[:, np.newaxis]
    weighted_star = residual_star * prepared.weights[np.newaxis, :]
    variance_star = np.asarray(
        _cgm_variance(weighted_star, plan),
        dtype=np.float64,
    )
    _require(
        bool(np.all(np.isfinite(variance_star)))
        and bool(np.all(variance_star > 1e-12)),
        "at least one WCR replicate has nonpositive or invalid CGM variance",
        code="unavailable_dependence_support",
    )
    bootstrap_t = (
        theta_star - float(theta0)
    ) / np.sqrt(variance_star)
    _require(
        bool(np.all(np.isfinite(bootstrap_t))),
        "at least one WCR replicate is nonfinite",
        code="unavailable_dependence_support",
    )
    return theta_hat, standard_error, observed_t, bootstrap_t


def _wcr_lower_tail_p(
    prepared: _PreparedWCR,
    active_dimensions: Sequence[str],
    theta0: float,
    multipliers: np.ndarray,
    plan: Sequence[tuple[int, float, tuple[np.ndarray, ...]]],
) -> float:
    _, _, observed_t, bootstrap_t = _wcr_distribution_at_threshold(
        prepared,
        active_dimensions,
        theta0,
        multipliers,
        plan,
    )
    return (
        1.0 + float(np.count_nonzero(bootstrap_t <= observed_t))
    ) / (len(bootstrap_t) + 1.0)


def _invert_wcr_upper_bound(
    prepared: _PreparedWCR,
    active_dimensions: Sequence[str],
    multipliers: np.ndarray,
    plan: Sequence[tuple[int, float, tuple[np.ndarray, ...]]],
    *,
    alpha: float,
) -> float:
    _require(
        0.0 < alpha < 0.5,
        "WCR inversion alpha is outside (0,.5)",
        code="unavailable_dependence_support",
    )
    theta_hat = float(
        np.sum(prepared.weights * prepared.differences)
    )
    scale = max(
        1.0,
        float(np.ptp(prepared.differences)) * 2.0,
    )
    low = theta_hat
    p_low = _wcr_lower_tail_p(
        prepared,
        active_dimensions,
        low,
        multipliers,
        plan,
    )
    _require(
        p_low > alpha,
        "WCR inversion has no verified lower bracket",
        code="unavailable_dependence_support",
    )
    high = low + scale
    p_high = _wcr_lower_tail_p(
        prepared,
        active_dimensions,
        high,
        multipliers,
        plan,
    )
    expansions = 0
    while p_high > alpha and expansions < 12:
        scale *= 2.0
        high = low + scale
        p_high = _wcr_lower_tail_p(
            prepared,
            active_dimensions,
            high,
            multipliers,
            plan,
        )
        expansions += 1
    _require(
        p_high <= alpha,
        "WCR inversion has no verified upper bracket",
        code="unavailable_dependence_support",
    )

    grid = np.linspace(low, high, 17)
    grid_p = [
        _wcr_lower_tail_p(
            prepared,
            active_dimensions,
            float(value),
            multipliers,
            plan,
        )
        for value in grid
    ]
    _require(
        all(
            grid_p[index + 1] <= grid_p[index] + 1e-12
            for index in range(len(grid_p) - 1)
        ),
        "WCR inversion p-value path is numerically nonmonotone",
        code="unavailable_dependence_support",
    )
    for _ in range(40):
        middle = (low + high) / 2.0
        p_middle = _wcr_lower_tail_p(
            prepared,
            active_dimensions,
            middle,
            multipliers,
            plan,
        )
        if p_middle > alpha:
            low = middle
        else:
            high = middle
    return high


def _run_wcr_bootcluster_core(
    prepared: _PreparedWCR,
    *,
    active_dimensions: Sequence[str],
    bootcluster_dimension: str,
    theta0: float,
    attempts: int,
    invert_endpoint: bool,
) -> WCRBootclusterResult:
    active = tuple(active_dimensions)
    _require(
        active in {_PRIMARY_ACTIVE, _PATCH_ACTIVE}
        and bootcluster_dimension in active,
        "WCR must select exactly one active bootstrap dimension",
        code="unavailable_dependence_support",
    )
    values = prepared.cluster_values[bootcluster_dimension]
    _require(
        len(set(values)) >= 2,
        "selected WCR bootstrap dimension has fewer than two clusters",
        code="unavailable_dependence_support",
    )
    effective = prepared.effective_clusters[bootcluster_dimension]
    law = _bootstrap_multiplier_law(effective)
    plan = _build_cgm_plan(prepared, active)
    multipliers = _wcr_multiplier_matrix(
        prepared,
        active,
        bootcluster_dimension,
        law,
        attempts=attempts,
    )
    theta_hat, standard_error, observed_t, bootstrap_t = (
        _wcr_distribution_at_threshold(
            prepared,
            active,
            theta0,
            multipliers,
            plan,
        )
    )
    lower_tail_p = (
        1.0 + float(np.count_nonzero(bootstrap_t <= observed_t))
    ) / (attempts + 1.0)
    q05 = _type7_quantile(bootstrap_t.tolist(), 0.05)
    q95 = _type7_quantile(bootstrap_t.tolist(), 0.95)
    approximate_upper = theta_hat - q05 * standard_error
    inverted = (
        _invert_wcr_upper_bound(
            prepared,
            active,
            multipliers,
            plan,
            alpha=0.05,
        )
        if invert_endpoint
        else math.nan
    )
    draws = tuple(float(value) for value in bootstrap_t)
    source_raw = _read_regular_under_root(
        _REPO_ROOT,
        "lol_kills/v2/evaluation/benchmark_contract.py",
        purpose="WCR source-code identity",
    )
    return WCRBootclusterResult(
        execution_id=_WCR_EXECUTION_ID,
        active_dimensions=active,
        bootcluster_dimension=bootcluster_dimension,
        point_estimate=theta_hat,
        threshold=float(theta0),
        standard_error=standard_error,
        observed_t=observed_t,
        lower_tail_p=lower_tail_p,
        unadjusted_one_sided_95_upper_bound=inverted,
        type7_linearized_upper_bound_diagnostic=approximate_upper,
        type7_bootstrap_t_q05_diagnostic=q05,
        type7_bootstrap_t_q95_diagnostic=q95,
        bootstrap_t=draws,
        bootstrap_t_sha256=stable_digest(list(draws)),
        multiplier_law=law,
        replicates=attempts,
        effective_clusters=effective,
        analysis_rows_sha256=prepared.analysis_rows_sha256,
        source_code_raw_sha256=raw_digest(source_raw),
    )


def _run_wcr_bootcluster(
    prepared: _PreparedWCR,
    *,
    active_dimensions: Sequence[str],
    bootcluster_dimension: str,
    theta0: float,
) -> WCRBootclusterResult:
    return _run_wcr_bootcluster_core(
        prepared,
        active_dimensions=active_dimensions,
        bootcluster_dimension=bootcluster_dimension,
        theta0=theta0,
        attempts=_WCR_REPLICATES,
        invert_endpoint=True,
    )


def _run_wcr_suite(
    prepared: _PreparedWCR,
    *,
    theta0: float,
    decision_kind: str,
    attempts: int = _WCR_REPLICATES,
    invert_endpoint: bool = True,
) -> WCRSuiteResult:
    _require(
        decision_kind in {"SUPERIORITY", "HARM", "NONINFERIORITY"},
        "legacy WCR suite decision kind is not registered",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    _require(
        all(
            prepared.effective_clusters[dimension] >= 30.0
            for dimension in _PATCH_ACTIVE
        ),
        "P, T, and H each require at least 30 effective clusters",
        code="unavailable_dependence_support",
    )
    primary = tuple(
        _run_wcr_bootcluster_core(
            prepared,
            active_dimensions=_PRIMARY_ACTIVE,
            bootcluster_dimension=dimension,
            theta0=theta0,
            attempts=attempts,
            invert_endpoint=invert_endpoint,
        )
        for dimension in _PRIMARY_ACTIVE
    )
    patch = tuple(
        _run_wcr_bootcluster_core(
            prepared,
            active_dimensions=_PATCH_ACTIVE,
            bootcluster_dimension=dimension,
            theta0=theta0,
            attempts=attempts,
            invert_endpoint=invert_endpoint,
        )
        for dimension in _PATCH_ACTIVE
    )
    def endpoint_pass(result: WCRBootclusterResult) -> bool:
        upper = result.unadjusted_one_sided_95_upper_bound
        return upper < float(theta0) if decision_kind == "SUPERIORITY" else upper <= float(theta0)

    primary_decisions = tuple(endpoint_pass(result) for result in primary)
    patch_decisions = tuple(endpoint_pass(result) for result in patch)
    primary_agree = len(set(primary_decisions)) == 1
    patch_agree = len(set(patch_decisions)) == 1
    patch_matches = (
        primary_agree
        and patch_agree
        and primary_decisions[0] is patch_decisions[0]
    )
    return WCRSuiteResult(
        primary=primary,
        patch=patch,
        primary_decisions_agree=primary_agree,
        patch_decisions_agree=patch_agree,
        patch_matches_primary=patch_matches,
        passes=(
            primary_agree
            and patch_agree
            and patch_matches
            and all(primary_decisions)
            and all(patch_decisions)
        ),
    )


def _reweight_reduced_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reduced = [dict(row) for row in rows]
    _require(
        bool(reduced),
        "leave-largest removal emptied the inferential cell",
        code="unavailable_dependence_support",
    )
    reduced.sort(key=lambda row: str(row["series_id"]))
    reduced_order = stable_digest(
        [str(row["series_id"]) for row in reduced]
    )
    weights = macro_regional_series_weights(
        reduced,
        required_leagues=_required_leagues_for_wcr_cell(reduced),
    )
    for row in reduced:
        row["row_order_sha256"] = reduced_order
        row["macro_weight"] = weights[str(row["series_id"])]
    return reduced


def _run_wcr_consensus(
    rows: Sequence[Mapping[str, Any]],
    *,
    theta0: float,
    decision_kind: str,
    attempts: int = _WCR_REPLICATES,
) -> WCRConsensusReport:
    prepared = _prepare_wcr_rows(rows)
    full = _run_wcr_suite(
        prepared,
        theta0=theta0,
        decision_kind=decision_kind,
        attempts=attempts,
    )
    leave: list[tuple[str, str, WCRSuiteResult]] = []
    for dimension in _PATCH_ACTIVE:
        cluster, _ = select_largest_cluster(prepared.rows, dimension)
        reduced = _reweight_reduced_rows(
            [
                row
                for row in prepared.rows
                if row[dimension] != cluster
            ]
        )
        reduced_prepared = _prepare_wcr_rows(reduced)
        leave.append(
            (
                dimension,
                cluster,
                _run_wcr_suite(
                    reduced_prepared,
                    theta0=theta0,
                    decision_kind=decision_kind,
                    attempts=attempts,
                ),
            )
        )
    return WCRConsensusReport(
        full_sample=full,
        leave_largest=tuple(leave),
        passes=full.passes and all(suite.passes for _, _, suite in leave),
    )


def _registered_wcr_plan_sha256() -> str:
    return stable_digest(
        {
            "execution_id": _WCR_EXECUTION_ID,
            "seed": _WCR_SEED,
            "replicates": _WCR_REPLICATES,
            "full_sample_runs": [
                {"active": list(_PRIMARY_ACTIVE), "bootcluster": dimension}
                for dimension in _PRIMARY_ACTIVE
            ]
            + [
                {"active": list(_PATCH_ACTIVE), "bootcluster": dimension}
                for dimension in _PATCH_ACTIVE
            ],
            "leave_largest_dimensions": list(_PATCH_ACTIVE),
            "endpoint": (
                "unadjusted one-sided 95% upper endpoint from inversion of "
                "the null-imposed lower-tail WCR test"
            ),
            "family_recompute": (
                "complete Holm ordering, ranks, rejections, endpoints, and "
                "decision for every full and leave-largest run"
            ),
        }
    )


def _registered_pair_wcr_plan_sha256() -> str:
    return stable_digest(
        {
            "execution_id": _WCR_EXECUTION_ID,
            "seed": _WCR_SEED,
            "replicates": _WCR_REPLICATES,
            "full_sample_runs": [
                {"active": list(_PRIMARY_ACTIVE), "bootcluster": dimension}
                for dimension in _PRIMARY_ACTIVE
            ]
            + [
                {"active": list(_PATCH_ACTIVE), "bootcluster": dimension}
                for dimension in _PATCH_ACTIVE
            ],
            "estimand": (
                "candidate A minus candidate B macro-regional "
                "chronological mean loss"
            ),
            "centering": (
                "each full or leave-largest sample is centered at its own "
                "internally derived macro-weighted point estimate"
            ),
            "simultaneous_interval": (
                "for each separately reported full or P/T/H leave-largest "
                "P/T/H run, replicate-wise maximum absolute centered "
                "bootstrap-t across every registered pair"
            ),
            "leave_largest_dimensions": list(_PATCH_ACTIVE),
            "selection_consensus": (
                "all 20 run-level simultaneous interval conclusions must agree"
            ),
        }
    )


@dataclass(frozen=True)
class _RegisteredSlotRun:
    sample_variant: str
    active_dimensions: tuple[str, ...]
    bootcluster_dimension: str
    removed_dimension: str | None
    removed_cluster_id: str | None
    result: WCRBootclusterResult

    @property
    def variant_id(self) -> str:
        active = ",".join(self.active_dimensions)
        return (
            f"{self.sample_variant}|active={active}|"
            f"bootcluster={self.bootcluster_dimension}"
        )


def _registered_wcr_variants(
    prepared: _PreparedWCR,
    *,
    theta0: float | Callable[[_PreparedWCR], float],
    attempts: int = _WCR_REPLICATES,
    invert_endpoint: bool = True,
) -> tuple[_RegisteredSlotRun, ...]:
    samples: list[
        tuple[str, str | None, str | None, _PreparedWCR]
    ] = [("full", None, None, prepared)]
    for removed_dimension in _PATCH_ACTIVE:
        removed_cluster, _ = select_largest_cluster(
            prepared.rows,
            removed_dimension,
        )
        reduced_rows = _reweight_reduced_rows(
            [
                row
                for row in prepared.rows
                if row[removed_dimension] != removed_cluster
            ]
        )
        samples.append(
            (
                f"leave_largest:{removed_dimension}",
                removed_dimension,
                removed_cluster,
                _prepare_wcr_rows(reduced_rows),
            )
        )

    runs: list[_RegisteredSlotRun] = []
    for (
        sample_variant,
        removed_dimension,
        removed_cluster,
        sample,
    ) in samples:
        run_theta0 = (
            float(theta0(sample))
            if callable(theta0)
            else float(theta0)
        )
        _require(
            math.isfinite(run_theta0),
            "registered WCR threshold is nonfinite",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        _require(
            all(
                sample.effective_clusters[dimension] >= 30.0
                for dimension in _PATCH_ACTIVE
            ),
            "P, T, and H each require 30 effective clusters in every sensitivity",
            code="unavailable_dependence_support",
        )
        for active in (_PRIMARY_ACTIVE, _PATCH_ACTIVE):
            for bootcluster in active:
                runs.append(
                    _RegisteredSlotRun(
                        sample_variant=sample_variant,
                        active_dimensions=active,
                        bootcluster_dimension=bootcluster,
                        removed_dimension=removed_dimension,
                        removed_cluster_id=removed_cluster,
                        result=_run_wcr_bootcluster_core(
                            sample,
                            active_dimensions=active,
                            bootcluster_dimension=bootcluster,
                            theta0=run_theta0,
                            attempts=attempts,
                            invert_endpoint=invert_endpoint,
                        ),
                    )
                )
    _require(
        len(runs) == 20
        and len({run.variant_id for run in runs}) == 20,
        "registered WCR sensitivity inventory is incomplete",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    return tuple(runs)


def _holm_rejections(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> tuple[bool, ...]:
    _require(0.0 < alpha < 1.0, "Holm alpha must be in (0,1)")
    _require(bool(p_values), "Holm family cannot be empty")
    _require(all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in p_values), "Holm p-values must be finite in [0,1]")
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    rejected = [False] * len(p_values)
    for rank, index in enumerate(order):
        threshold = alpha / (len(order) - rank)
        if p_values[index] > threshold:
            break
        rejected[index] = True
    return tuple(rejected)


@dataclass(frozen=True)
class HolmDecision:
    slot_id: str
    raw_one_sided_p: float
    holm_adjusted_p: float
    holm_rank: int
    local_alpha: float
    registered_threshold: float
    unadjusted_one_sided_95_upper_bound: float
    endpoint_label: str
    holm_reject: bool
    unadjusted_endpoint_pass: bool
    passes: bool


@dataclass(frozen=True)
class HolmVariantReport:
    variant_id: str
    sample_variant: str
    active_dimensions: tuple[str, ...]
    bootcluster_dimension: str
    decisions: tuple[HolmDecision, ...]
    all_members_pass: bool
    family_gate_pass: bool


@dataclass(frozen=True)
class HolmReport:
    candidate_id: str
    family_id: str
    family_kind: str
    variants: tuple[HolmVariantReport, ...]
    decisions: tuple[HolmDecision, ...]
    family_gate_pass: bool

    @property
    def all_pass(self) -> bool:
        """Compatibility alias for the explicitly named family gate."""

        return self.family_gate_pass


@dataclass(frozen=True)
class PairwiseInterval:
    pair_id: str
    output_id: str
    candidate_a_id: str
    candidate_b_id: str
    point_estimate: float
    simultaneous_two_sided_95_bootstrap_t_lower: float
    simultaneous_two_sided_95_bootstrap_t_upper: float

    @property
    def lower_bound(self) -> float:
        return self.simultaneous_two_sided_95_bootstrap_t_lower

    @property
    def upper_bound(self) -> float:
        return self.simultaneous_two_sided_95_bootstrap_t_upper


@dataclass(frozen=True)
class PairwiseSensitivityReport:
    sample_variant: str
    active_dimensions: tuple[str, ...]
    bootcluster_dimension: str
    critical_value: float
    intervals: tuple[PairwiseInterval, ...]
    directional_conclusion: str


@dataclass(frozen=True)
class PairwiseIntervalReport:
    pair_family_id: str
    critical_value: float | None
    intervals: tuple[PairwiseInterval, ...]
    sensitivities: tuple[PairwiseSensitivityReport, ...]
    sensitivity_conclusions_agree: bool
    candidate_gate_status: tuple[tuple[str, str], ...]
    selection_status: str
    winner_candidate_id: str | None


def _registry_index(
    records: object,
    key: str,
    *,
    duplicate_code: str,
) -> dict[str, dict[str, Any]]:
    _require(
        isinstance(records, list),
        f"registry {key} collection is malformed",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        _require(
            isinstance(record, dict)
            and isinstance(record.get(key), str)
            and bool(record[key]),
            f"registry {key} record is malformed",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        _require(
            record[key] not in index,
            f"duplicate registry {key}: {record[key]}",
            code=duplicate_code,
        )
        index[record[key]] = record
    return index


def _read_registry_bound_json(
    context: _AuthorityContext,
    binding: Mapping[str, Any],
    *,
    purpose: str,
    mismatch_code: str,
) -> dict[str, Any]:
    _require(
        isinstance(binding, Mapping)
        and {"relative_path", "raw_sha256", "semantic_sha256"}
        <= set(binding),
        f"{purpose}: binding is malformed",
        code=mismatch_code,
    )
    relative = _canonical_relative_locator(
        binding["relative_path"],
        purpose=purpose,
        contract=context.contract,
    )
    raw = _read_regular_under_root(
        context.package_root,
        relative,
        purpose=purpose,
        contract=context.contract,
    )
    payload = _parse_json_bytes(raw, label=purpose)
    _require(
        raw_digest(raw) == binding["raw_sha256"]
        and stable_digest(payload) == binding["semantic_sha256"],
        f"{purpose}: bound artifact digest mismatch",
        code=mismatch_code,
    )
    return payload


def _candidate_record(
    context: _AuthorityContext,
    candidate_id: str,
) -> dict[str, Any]:
    _reject_source_bound_transition("resolved candidate", candidate_id)
    candidates = _registry_index(
        context.candidate_registry["candidates"],
        "candidate_id",
        duplicate_code="CANDIDATE_SLOT_DUPLICATE",
    )
    _require(
        candidate_id in candidates,
        f"candidate is not registered: {candidate_id}",
        code="CANDIDATE_UNREGISTERED",
    )
    candidate = candidates[candidate_id]
    _require(
        candidate["status"] == "RESOLVED",
        f"candidate remains typed unresolved: {candidate_id}",
        code="CANDIDATE_SLOT_UNRESOLVED",
    )
    _validate_resolved_candidate_provenance(context, candidate)
    return candidate


def _validate_resolved_candidate_provenance(
    context: _AuthorityContext,
    candidate: Mapping[str, Any],
) -> str:
    """Reopen all executable candidate inputs; no stored preflight is authority."""

    _reject_source_bound_transition(
        "resolved candidate provenance",
        candidate.get("candidate_id"),
    )
    fields = (
        ("candidate_artifact_relative_path", "candidate_artifact_raw_sha256"),
        ("executable_adapter_relative_path", "executable_adapter_raw_sha256"),
        ("config_relative_path", "config_raw_sha256"),
        ("environment_lock_relative_path", "environment_lock_raw_sha256"),
        ("source_snapshot_relative_path", "source_snapshot_raw_sha256"),
        ("partition_bindings_relative_path", "partition_bindings_raw_sha256"),
    )
    _require(
        isinstance(candidate.get("executable_adapter_entry_point"), str)
        and bool(candidate["executable_adapter_entry_point"])
        and "preflight_sha256" not in candidate,
        "resolved candidate provenance is incomplete or circular",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    _require(
        all(
            key in candidate
            for key in (
                "fixture_input", "fixture_output", "fixture_sha256",
                "execution_authority",
            )
        )
        and _valid_execution_authority_record(
            candidate.get("execution_authority")
        ),
        "resolved candidate executable fixture authority is incomplete",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    for locator_key, digest_key in fields:
        relative = _canonical_relative_locator(
            candidate.get(locator_key),
            purpose=f"candidate provenance {candidate['candidate_id']}:{locator_key}",
            contract=context.contract,
        )
        raw = _read_regular_under_root(
            context.package_root,
            relative,
            purpose=f"candidate provenance {candidate['candidate_id']}:{locator_key}",
            contract=context.contract,
        )
        _require(
            raw_digest(raw) == candidate.get(digest_key),
            f"candidate provenance digest drifted: {locator_key}",
            code="CANDIDATE_REGISTRY_MISMATCH",
        )
    artifact = _read_regular_under_root(
        context.package_root,
        _canonical_relative_locator(
            candidate["candidate_artifact_relative_path"],
            purpose=f"candidate artifact {candidate['candidate_id']}",
            contract=context.contract,
        ),
        purpose=f"candidate artifact {candidate['candidate_id']}",
        contract=context.contract,
    )
    payload = _parse_json_bytes(artifact, label="candidate artifact")
    _require(
        stable_digest(payload) == candidate.get("candidate_artifact_semantic_sha256"),
        "candidate artifact semantic identity drifted",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    validate_no_final_labels(payload, label="candidate artifact")
    config_raw = _read_regular_under_root(
        context.package_root,
        candidate["config_relative_path"],
        purpose=f"candidate config {candidate['candidate_id']}",
        contract=context.contract,
    )
    config = _parse_json_bytes(config_raw, label="candidate config")
    _require(
        isinstance(config, Mapping),
        "candidate config must be an exact JSON object",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    return _invoke_bound_fixture(
        context.package_root,
        locator=candidate["executable_adapter_relative_path"],
        expected_code_raw_sha256=candidate["executable_adapter_raw_sha256"],
        entry_point=candidate["executable_adapter_entry_point"],
        fixture_input=candidate["fixture_input"],
        config=config,
        fixture_output=candidate["fixture_output"],
        fixture_sha256=candidate["fixture_sha256"],
        execution_authority=candidate["execution_authority"],
        purpose=f"candidate {candidate['candidate_id']}",
    )


def _holm_evidence(
    context: _AuthorityContext,
    slot: Mapping[str, Any],
) -> _PreparedWCR:
    _reject_source_bound_transition(
        "resolved Holm evidence",
        slot.get("slot_id"),
    )
    evidence = _read_registry_bound_json(
        context,
        slot["evidence"],
        purpose=f"Holm evidence {slot['slot_id']}",
        mismatch_code="FAMILY_DERIVATION_MISMATCH",
    )
    exact = {
        "schema_version",
        "slot_id",
        "family_id",
        "candidate_id",
        "baseline_id",
        "output_id",
        "stratum_id",
        "decision_kind",
        "threshold_source",
        "wcr_execution_id",
        "analysis_rows_sha256",
        "bootstrap_plan_sha256",
        "source_code_raw_sha256",
        "candidate_artifact_raw_sha256",
        "candidate_execution_code_raw_sha256",
        "candidate_adapter_entry_point",
        "candidate_config_raw_sha256",
        "candidate_environment_lock_raw_sha256",
        "candidate_source_snapshot_raw_sha256",
        "candidate_partition_bindings_raw_sha256",
        "candidate_execution_authority",
        "candidate_execution_receipt_sha256",
        "baseline_code_raw_sha256",
        "baseline_entry_point",
        "baseline_config_schema_sha256",
        "baseline_default_config_sha256",
        "baseline_environment_lock_raw_sha256",
        "baseline_fixture_sha256",
        "baseline_execution_authority",
        "baseline_execution_receipt_sha256",
        "candidate_full_forecast_execution_authority",
        "baseline_full_forecast_execution_authority",
        "forecast_execution_authority",
        "rows",
    }
    if str(slot["stratum_id"]).startswith("benefit:"):
        exact |= {"secondary_primary_binding", "secondary_selector"}
    _require(
        set(evidence) == exact
        and evidence["schema_version"] == "g0-holm-evidence-v1.4",
        "registered Holm evidence schema is not exact",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    for field in (
        "slot_id",
        "family_id",
        "candidate_id",
        "baseline_id",
        "output_id",
        "stratum_id",
        "decision_kind",
        "threshold_source",
    ):
        _require(
            evidence[field] == slot[field],
            f"registered Holm evidence identity mismatch: {field}",
            code="FAMILY_DERIVATION_MISMATCH",
        )
    candidate = _candidate_record(context, slot["candidate_id"])
    candidate_receipt = _validate_resolved_candidate_provenance(
        context, candidate
    )
    candidate_bindings = {
        "candidate_artifact_raw_sha256": candidate["candidate_artifact_raw_sha256"],
        "candidate_execution_code_raw_sha256": candidate["executable_adapter_raw_sha256"],
        "candidate_adapter_entry_point": candidate["executable_adapter_entry_point"],
        "candidate_config_raw_sha256": candidate["config_raw_sha256"],
        "candidate_environment_lock_raw_sha256": candidate["environment_lock_raw_sha256"],
        "candidate_source_snapshot_raw_sha256": candidate["source_snapshot_raw_sha256"],
        "candidate_partition_bindings_raw_sha256": candidate["partition_bindings_raw_sha256"],
        "candidate_execution_authority": candidate["execution_authority"],
        "candidate_execution_receipt_sha256": candidate_receipt,
    }
    _require(
        all(evidence.get(field) == value for field, value in candidate_bindings.items()),
        "Holm evidence does not bind re-opened candidate provenance",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    baselines = _registry_index(
        context.baseline_registry["baselines"],
        "id",
        duplicate_code="FAMILY_DERIVATION_MISMATCH",
    )
    baseline = baselines.get(slot["baseline_id"])
    _require(
        baseline is not None and baseline.get("status") == "EXECUTABLE_PREBOUND",
        "Holm evidence baseline is not executable and prebound",
        code="CANDIDATE_SLOT_UNRESOLVED",
    )
    execution = baseline["execution"]
    baseline_receipt = _invoke_bound_fixture(
        context.repo_root,
        locator=execution["code_locator"],
        expected_code_raw_sha256=execution["code_raw_sha256"],
        entry_point=execution["entry_point"],
        fixture_input=execution["fixture_input"],
        config=execution["default_config"],
        fixture_output=execution["fixture_output"],
        fixture_sha256=execution["fixture_sha256"],
        execution_authority=execution["execution_authority"],
        purpose=f"baseline {slot['baseline_id']}",
    )
    baseline_bindings = {
        "baseline_code_raw_sha256": execution["code_raw_sha256"],
        "baseline_entry_point": execution["entry_point"],
        "baseline_config_schema_sha256": execution["config_schema_sha256"],
        "baseline_default_config_sha256": execution["default_config_sha256"],
        "baseline_environment_lock_raw_sha256": execution["environment_lock_raw_sha256"],
        "baseline_fixture_sha256": execution["fixture_sha256"],
        "baseline_execution_authority": execution["execution_authority"],
        "baseline_execution_receipt_sha256": baseline_receipt,
    }
    _require(
        all(evidence.get(field) == value for field, value in baseline_bindings.items()),
        "Holm evidence does not bind re-executed baseline provenance",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    _require(
        evidence["wcr_execution_id"] == _WCR_EXECUTION_ID,
        "registered Holm evidence uses the wrong WCR execution",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    for field in (
        "analysis_rows_sha256",
        "bootstrap_plan_sha256",
        "source_code_raw_sha256",
    ):
        _require(
            _is_digest(evidence[field]),
            f"registered Holm evidence digest malformed: {field}",
            code="FAMILY_DERIVATION_MISMATCH",
        )
    _require(
        evidence["bootstrap_plan_sha256"]
        == _registered_wcr_plan_sha256(),
        "registered Holm evidence uses an opaque or stale bootstrap plan",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    source_raw = _read_regular_under_root(
        context.repo_root,
        "lol_kills/v2/evaluation/benchmark_contract.py",
        purpose="registered Holm source-code identity",
    )
    _require(
        evidence["source_code_raw_sha256"] == raw_digest(source_raw),
        "registered Holm evidence was produced by different source code",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    rows = evidence["rows"]
    _require(
        isinstance(rows, list) and bool(rows),
        "registered Holm evidence rows are empty",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    prepared = _prepare_wcr_rows(rows)
    _require(
        prepared.candidate_id == slot["candidate_id"]
        and prepared.baseline_id == slot["baseline_id"]
        and prepared.output_id == slot["output_id"]
        and prepared.stratum_id == slot["stratum_id"],
        "registered Holm rows differ from their frozen slot identity",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    expected_score_kind = (
        "BRIER"
        if slot["stratum_id"] == "benefit:brier"
        else "LOG_LOSS"
    )
    _require(
        prepared.score_kind == expected_score_kind,
        "registered Holm score kind differs from its frozen slot",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    _require(
        evidence["analysis_rows_sha256"]
        == prepared.analysis_rows_sha256,
        "registered Holm analysis-row digest mismatch",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    _validate_secondary_evidence(
        context,
        slot,
        evidence,
        prepared,
    )
    candidate_config_raw = _read_regular_under_root(
        context.package_root,
        candidate["config_relative_path"],
        purpose=f"candidate config {candidate['candidate_id']}",
        contract=context.contract,
    )
    candidate_config = _parse_json_bytes(
        candidate_config_raw,
        label=f"candidate config {candidate['candidate_id']}",
    )
    _require(
        isinstance(candidate_config, Mapping),
        "candidate full forecast config is not an object",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    candidate_full_authority = _invoke_bound_forecast_batch(
        context.package_root,
        locator=candidate["executable_adapter_relative_path"],
        expected_code_raw_sha256=candidate[
            "executable_adapter_raw_sha256"
        ],
        entry_point=candidate["executable_adapter_entry_point"],
        config=candidate_config,
        environment_lock_raw_sha256=candidate[
            "environment_lock_raw_sha256"
        ],
        source_snapshot_relative_path=candidate[
            "source_snapshot_relative_path"
        ],
        source_snapshot_raw_sha256=candidate[
            "source_snapshot_raw_sha256"
        ],
        partition_bindings_relative_path=candidate[
            "partition_bindings_relative_path"
        ],
        partition_bindings_raw_sha256=candidate[
            "partition_bindings_raw_sha256"
        ],
        slot=slot,
        rows=prepared.rows,
        role="candidate",
        model_id=slot["candidate_id"],
        purpose=f"candidate full forecast {slot['slot_id']}",
    )
    baseline_full_authority = _invoke_bound_forecast_batch(
        context.repo_root,
        locator=execution["code_locator"],
        expected_code_raw_sha256=execution["code_raw_sha256"],
        entry_point=execution["entry_point"],
        config=execution["default_config"],
        environment_lock_raw_sha256=execution[
            "environment_lock_raw_sha256"
        ],
        source_snapshot_relative_path=candidate[
            "source_snapshot_relative_path"
        ],
        source_snapshot_raw_sha256=candidate[
            "source_snapshot_raw_sha256"
        ],
        partition_bindings_relative_path=candidate[
            "partition_bindings_relative_path"
        ],
        partition_bindings_raw_sha256=candidate[
            "partition_bindings_raw_sha256"
        ],
        slot=slot,
        rows=prepared.rows,
        role="baseline",
        model_id=slot["baseline_id"],
        purpose=f"baseline full forecast {slot['slot_id']}",
    )
    forecast_authority = _full_forecast_pair_authority(
        slot,
        candidate_full_authority,
        baseline_full_authority,
    )
    _require(
        evidence.get("candidate_full_forecast_execution_authority")
        == candidate_full_authority
        and evidence.get("baseline_full_forecast_execution_authority")
        == baseline_full_authority
        and evidence.get("forecast_execution_authority")
        == forecast_authority
        and slot.get("execution_authority") == forecast_authority,
        "Holm evidence or slot does not bind authority-side full forecast reexecution",
        code="CANDIDATE_SLOT_UNRESOLVED",
    )
    return prepared


def _validate_secondary_evidence(
    context: _AuthorityContext,
    slot: Mapping[str, Any],
    evidence: Mapping[str, Any],
    prepared: _PreparedWCR,
) -> _PreparedWCR | None:
    """Secondary claims are projections of an exact registered primary row set."""

    _reject_source_bound_transition(
        "resolved secondary evidence",
        slot.get("slot_id"),
    )
    stratum = str(slot["stratum_id"])
    if not stratum.startswith("benefit:"):
        return None
    benefit = stratum.split(":", 1)[1]
    primary_slots = [
        item
        for item in context.candidate_registry["slots"]
        if item["candidate_id"] == slot["candidate_id"]
        and item["baseline_id"] == slot["baseline_id"]
        and item["output_id"] == slot["output_id"]
        and item["stratum_id"] == "overall"
        and item["decision_kind"] == "SUPERIORITY"
    ]
    _require(
        len(primary_slots) == 1 and primary_slots[0]["status"] == "RESOLVED",
        "secondary evidence has no exact resolved primary comparison",
        code="CANDIDATE_SLOT_UNRESOLVED",
    )
    primary_slot = primary_slots[0]
    primary = _holm_evidence(context, primary_slot)
    binding = evidence.get("secondary_primary_binding")
    _require(
        isinstance(binding, Mapping)
        and binding
        == {
            "primary_slot_id": primary_slot["slot_id"],
            "analysis_rows_sha256": primary.analysis_rows_sha256,
            "input_rows_sha256": _derived_input_rows_sha256(primary.rows),
            "candidate_prediction_rows_sha256": _derived_prediction_rows_sha256(
                primary.rows, role="candidate"
            ),
            "baseline_prediction_rows_sha256": _derived_prediction_rows_sha256(
                primary.rows, role="baseline"
            ),
            "outcomes_sha256": stable_digest(
                [
                    {"series_id": row["series_id"], "outcome": row["outcome"]}
                    for row in primary.rows
                ]
            ),
        },
        "secondary evidence does not bind the exact primary rows/outcomes/forecasts",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    def observation_surface(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            field: row[field]
            for field in (
                "candidate_id", "baseline_id", "output_id", "series_id",
                "map_ids", "fold_id", "registered_fold_ids", "league_id",
                _P, _T, _H, "game_side", "roster_change",
                "international_event", "draft_depth", "resolved",
                "row_order_sha256", "outcome", "candidate_probabilities",
                "baseline_probabilities", "candidate_prediction_sha256",
                "baseline_prediction_sha256", "exact_roster_id",
                "series_order_within_exact_roster_tournament",
                "strength_source_id", "pre_outcome_candidate_strength",
                "pre_outcome_baseline_strength",
            )
        } | {
            "P_T": canonical_intersection_id([row[_P], row[_T]]),
            "P_H": canonical_intersection_id([row[_P], row[_H]]),
            "T_H": canonical_intersection_id([row[_T], row[_H]]),
            "P_T_H": canonical_intersection_id([row[_P], row[_T], row[_H]]),
        }

    primary_by_series = {row["series_id"]: row for row in primary.rows}
    selected_ids = [row["series_id"] for row in prepared.rows]
    _require(
        all(
            row["series_id"] in primary_by_series
            and observation_surface(row)
            == observation_surface(primary_by_series[row["series_id"]])
            for row in prepared.rows
        ),
        "secondary rows drift from their primary rows/outcomes/forecasts",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    selector = evidence.get("secondary_selector")
    if benefit == "brier":
        _require(
            selector == {"kind": "PRIMARY_ROWS_EXACT"}
            and selected_ids == list(primary.canonical_series_ids),
            "Brier secondary benefit must use exactly its primary rows",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        return None
    if benefit == "transfer_new_roster":
        expected = [
            row["series_id"]
            for row in primary.rows
            if row["roster_change"] == "FIRST_TOURNAMENT_NEW_EXACT_ROSTER"
        ]
        first_series = [
            row["series_id"]
            for row in primary.rows
            if row["roster_change"] == "FIRST_TOURNAMENT_NEW_EXACT_ROSTER"
            and row["series_order_within_exact_roster_tournament"] == 1
        ]
        _require(
            selector
            == {
                "kind": "FIRST_TOURNAMENT_NEW_EXACT_ROSTER",
                "rows_sha256": stable_digest(expected),
                "first_series_sensitivity": {
                    "kind": "FIRST_SERIES_PER_EXACT_ROSTER_TOURNAMENT",
                    "rows_sha256": stable_digest(first_series),
                },
            }
            and selected_ids == expected,
            "new-roster transfer secondary must contain only the registered first-tournament rows",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        first_series_rows = [
            row
            for row in prepared.rows
            if row["series_order_within_exact_roster_tournament"] == 1
        ]
        return _prepare_wcr_rows(
            _reweight_reduced_rows(first_series_rows)
        )
    _require(
        benefit == "equal_strength_draft_increment",
        "secondary benefit is unregistered",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    authority = context.candidate_registry.get("secondary_authority", {})
    registered = authority.get("equal_strength_draft_increment") if isinstance(authority, Mapping) else None
    rule_fields = {
        "status", "strength_source_id", "as_of_rule", "distance_metric",
        "tolerance", "missing_policy", "tie_policy",
    }
    derived_membership = (
        sorted(
            row["series_id"]
            for row in primary.rows
            if isinstance(registered, Mapping)
            and row["strength_source_id"] == registered.get("strength_source_id")
            and abs(
                float(row["pre_outcome_candidate_strength"])
                - float(row["pre_outcome_baseline_strength"])
            )
            <= float(registered.get("tolerance", -1.0))
        )
        if isinstance(registered, Mapping)
        else []
    )
    rule_digest = stable_digest(dict(registered)) if isinstance(registered, Mapping) else ""
    _require(
        isinstance(registered, Mapping)
        and set(registered) == rule_fields
        and registered.get("status") == "RESOLVED"
        and registered.get("as_of_rule") == "STRICTLY_BEFORE_SERIES_START"
        and registered.get("distance_metric") == "ABSOLUTE_DIFFERENCE"
        and isinstance(registered.get("tolerance"), (int, float))
        and not isinstance(registered.get("tolerance"), bool)
        and float(registered["tolerance"]) >= 0.0
        and registered.get("missing_policy") == "EXCLUDE_AND_BLOCK_IF_EMPTY"
        and registered.get("tie_policy") == "INCLUDE_AT_OR_BELOW_TOLERANCE"
        and bool(derived_membership)
        and derived_membership == selected_ids
        and selector
        == {
            "kind": "REGISTERED_EQUAL_STRENGTH_OVERLAP",
            "rule_sha256": rule_digest,
            "membership_sha256": stable_digest(derived_membership),
        }
        and all(row["series_id"] in primary_by_series for row in prepared.rows),
        "equal-strength secondary lacks exact registered overlap authority",
        code="CANDIDATE_SLOT_UNRESOLVED",
    )
    return None


def _registered_slot_threshold(
    context: _AuthorityContext,
    slot: Mapping[str, Any],
) -> float:
    source = slot["threshold_source"]
    if source == "ZERO":
        return 0.0
    _require(
        source in {
            "REGISTERED_MARGIN",
            "REGISTERED_MARGIN_OR_ZERO_FALLBACK",
        },
        "registered slot threshold source is unknown",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    primary_slots = [
        item
        for item in context.candidate_registry["slots"]
        if item["candidate_id"] == slot["candidate_id"]
        and item["baseline_id"] == slot["baseline_id"]
        and item["stratum_id"] == "overall"
        and item["decision_kind"] == "SUPERIORITY"
    ]
    _require(
        len(primary_slots) == 1,
        "registered primary margin slot is missing or ambiguous",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    if (
        source == "REGISTERED_MARGIN_OR_ZERO_FALLBACK"
        and "margin_binding" not in primary_slots[0]
    ):
        return 0.0
    margin = _derive_registered_margin_at(
        context,
        slot["candidate_id"],
        slot["baseline_id"],
    )
    _require(
        math.isfinite(margin) and margin >= 0.0,
        "registered margin threshold is invalid",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    return float(margin)


def _holm_variant_decisions(
    *,
    family_kind: str,
    selected: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    runs: Sequence[_RegisteredSlotRun],
) -> HolmVariantReport:
    _require(
        len(selected) == len(thresholds) == len(runs),
        "Holm variant inputs are incomplete",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    variant_ids = {run.variant_id for run in runs}
    _require(
        len(variant_ids) == 1,
        "Holm family variant identities are not aligned",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    raw_p = [run.result.lower_tail_p for run in runs]
    order = sorted(
        range(len(selected)),
        key=lambda index: (raw_p[index], selected[index]["slot_id"]),
    )
    rank_by_index = {index: rank for rank, index in enumerate(order)}
    rejected = [False] * len(selected)
    adjusted = [1.0] * len(selected)
    running = 0.0
    stopped = False
    for rank, index in enumerate(order):
        factor = len(selected) - rank
        running = max(running, min(1.0, factor * raw_p[index]))
        adjusted[index] = running
        if not stopped and raw_p[index] <= 0.05 / factor:
            rejected[index] = True
        else:
            stopped = True

    decisions: list[HolmDecision] = []
    endpoint_label = (
        "UNADJUSTED_ONE_SIDED_95_PERCENT_WCR_INVERTED_UPPER_BOUND"
    )
    for index, slot in enumerate(selected):
        rank = rank_by_index[index]
        threshold = float(thresholds[index])
        upper = float(
            runs[index].result.unadjusted_one_sided_95_upper_bound
        )
        endpoint_pass = (
            upper < threshold
            if slot["decision_kind"] == "SUPERIORITY"
            else upper <= threshold
        )
        holm_reject = rejected[index]
        decisions.append(
            HolmDecision(
                slot_id=slot["slot_id"],
                raw_one_sided_p=raw_p[index],
                holm_adjusted_p=adjusted[index],
                holm_rank=rank + 1,
                local_alpha=0.05 / (len(selected) - rank),
                registered_threshold=threshold,
                unadjusted_one_sided_95_upper_bound=upper,
                endpoint_label=endpoint_label,
                holm_reject=holm_reject,
                unadjusted_endpoint_pass=endpoint_pass,
                passes=holm_reject and endpoint_pass,
            )
        )
    ordered = tuple(
        sorted(decisions, key=lambda item: item.slot_id)
    )
    all_members_pass = all(item.passes for item in ordered)
    family_gate_pass = (
        any(item.passes for item in ordered)
        if family_kind == "SECONDARY_HOLM"
        else all_members_pass
    )
    first = runs[0]
    return HolmVariantReport(
        variant_id=first.variant_id,
        sample_variant=first.sample_variant,
        active_dimensions=first.active_dimensions,
        bootcluster_dimension=first.bootcluster_dimension,
        decisions=ordered,
        all_members_pass=all_members_pass,
        family_gate_pass=family_gate_pass,
    )


def _holm_family_gate_across_variants(
    family_kind: str,
    variants: Sequence[HolmVariantReport],
) -> bool:
    """Apply the family rule across all 20 independently replayed runs."""

    _require(
        bool(variants),
        "Holm sensitivity inventory is empty",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    if family_kind != "SECONDARY_HOLM":
        return all(variant.family_gate_pass for variant in variants)
    slot_ids = {
        decision.slot_id
        for variant in variants
        for decision in variant.decisions
    }
    return any(
        all(
            next(
                (
                    decision.passes
                    for decision in variant.decisions
                    if decision.slot_id == slot_id
                ),
                False,
            )
            for variant in variants
        )
        for slot_id in slot_ids
    )


def _compute_registered_holm_at(
    context: _AuthorityContext,
    candidate_id: str,
    family_id: str,
    *,
    attempts: int = _WCR_REPLICATES,
) -> HolmReport:
    _candidate_record(context, candidate_id)
    families = _registry_index(
        context.candidate_registry["families"],
        "family_id",
        duplicate_code="CANDIDATE_SLOT_DUPLICATE",
    )
    _require(
        family_id in families,
        f"Holm family is not registered: {family_id}",
        code="FAMILY_UNREGISTERED",
    )
    family = families[family_id]
    _require(
        family["candidate_id"] == candidate_id,
        "Holm family belongs to a different candidate",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    slots = _registry_index(
        context.candidate_registry["slots"],
        "slot_id",
        duplicate_code="CANDIDATE_SLOT_DUPLICATE",
    )
    declared = family["slot_ids"]
    _require(
        isinstance(declared, list)
        and bool(declared)
        and declared == sorted(declared)
        and len(declared) == len(set(declared)),
        "Holm family slot membership is malformed",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    actual_members = sorted(
        slot_id
        for slot_id, slot in slots.items()
        if slot["family_id"] == family_id
    )
    missing = sorted(set(declared) - set(actual_members))
    unexpected = sorted(set(actual_members) - set(declared))
    _require(
        not missing,
        f"registered Holm family has missing slots: {missing}",
        code="CANDIDATE_SLOT_MISSING",
    )
    _require(
        not unexpected,
        f"registered Holm family has unexpected slots: {unexpected}",
        code="CANDIDATE_SLOT_UNEXPECTED",
    )
    selected = [slots[slot_id] for slot_id in declared]
    _require(
        all(slot["candidate_id"] == candidate_id for slot in selected),
        "Holm slot candidate identity mismatch",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    baselines = _registry_index(
        context.baseline_registry["baselines"],
        "id",
        duplicate_code="FAMILY_DERIVATION_MISMATCH",
    )
    _require(
        all(
            baselines.get(slot["baseline_id"], {}).get("status")
            == "EXECUTABLE_PREBOUND"
            for slot in selected
        ),
        "a required baseline is not executable and prebound",
        code="CANDIDATE_SLOT_UNRESOLVED",
    )
    unresolved = [
        slot["slot_id"]
        for slot in selected
        if slot["status"] != "RESOLVED"
    ]
    _require(
        not unresolved,
        f"typed-unresolved Holm slots block the family: {unresolved}",
        code="CANDIDATE_SLOT_UNRESOLVED",
    )
    prepared = [_holm_evidence(context, slot) for slot in selected]
    thresholds = [
        _registered_slot_threshold(context, slot) for slot in selected
    ]
    runs_by_slot = [
        _registered_wcr_variants(
            item,
            theta0=threshold,
            attempts=attempts,
        )
        for item, threshold in zip(prepared, thresholds)
    ]
    variant_order = [run.variant_id for run in runs_by_slot[0]]
    _require(
        all(
            [run.variant_id for run in runs] == variant_order
            for runs in runs_by_slot
        ),
        "registered Holm sensitivities differ across family members",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    variants = tuple(
        _holm_variant_decisions(
            family_kind=family["kind"],
            selected=selected,
            thresholds=thresholds,
            runs=[runs[index] for runs in runs_by_slot],
        )
        for index in range(len(variant_order))
    )
    full_primary = next(
        variant
        for variant in variants
        if variant.sample_variant == "full"
        and variant.active_dimensions == _PRIMARY_ACTIVE
        and variant.bootcluster_dimension == _P
    )
    return HolmReport(
        candidate_id=candidate_id,
        family_id=family_id,
        family_kind=family["kind"],
        variants=variants,
        decisions=full_primary.decisions,
        family_gate_pass=_holm_family_gate_across_variants(
            family["kind"],
            variants,
        ),
    )


def compute_registered_holm(
    candidate_id: str,
    family_id: str,
) -> HolmReport:
    """Compute exactly one manifest-derived family; membership is not input."""

    context = _verify_authoritative_preflight_at(
        _PRODUCTION_PACKAGE_ROOT,
        _REPO_ROOT,
    )
    return _compute_registered_holm_at(context, candidate_id, family_id)


def _validated_pair_evidence(
    context: _AuthorityContext,
    pair: Mapping[str, Any],
) -> _PreparedWCR:
    _reject_source_bound_transition(
        "resolved pair evidence",
        pair.get("pair_id"),
    )
    evidence = _read_registry_bound_json(
        context,
        pair["evidence"],
        purpose=f"pairwise evidence {pair['pair_id']}",
        mismatch_code="PAIR_IDENTITY_MISMATCH",
    )
    exact = {
        "schema_version",
        "pair_id",
        "pair_family_id",
        "candidate_a_id",
        "candidate_b_id",
        "output_id",
        "orientation",
        "score_kind",
        "wcr_execution_id",
        "aligned_row_ids_sha256",
        "difference_rows_sha256",
        "bootstrap_plan_sha256",
        "registered_fold_ids_sha256",
        "league_ids_sha256",
        "macro_weights_sha256",
        "outcomes_sha256",
        "pth_assignments_sha256",
        "critical_selectors_sha256",
        "cluster_assignments_sha256",
        "candidate_a_prediction_rows_sha256",
        "candidate_b_prediction_rows_sha256",
        "input_rows_sha256",
        "analysis_rows_sha256",
        "source_code_raw_sha256",
        "rows",
    }
    _require(
        set(evidence) == exact
        and evidence["schema_version"] == "g0-pair-evidence-v1.4",
        "pairwise evidence schema is not exact",
        code="PAIR_IDENTITY_MISMATCH",
    )
    for field in (
        "pair_id",
        "pair_family_id",
        "candidate_a_id",
        "candidate_b_id",
        "output_id",
        "orientation",
    ):
        _require(
            evidence[field] == pair[field],
            f"pairwise evidence identity mismatch: {field}",
            code="PAIR_IDENTITY_MISMATCH",
        )
    _require(
        evidence["orientation"] == "candidate_a_minus_candidate_b",
        "pairwise evidence orientation is wrong",
        code="PAIR_IDENTITY_MISMATCH",
    )
    _require(
        evidence["score_kind"] == "LOG_LOSS"
        and pair["score_kind"] == "LOG_LOSS",
        "pairwise complexity score kind must be exact log loss",
        code="PAIR_IDENTITY_MISMATCH",
    )
    _require(
        evidence["wcr_execution_id"] == _WCR_EXECUTION_ID
        and evidence["bootstrap_plan_sha256"]
        == _registered_pair_wcr_plan_sha256(),
        "pairwise evidence uses a stale or opaque WCR plan",
        code="PAIR_BOOTSTRAP_PLAN_MISMATCH",
    )
    source_raw = _read_regular_under_root(
        context.repo_root,
        "lol_kills/v2/evaluation/benchmark_contract.py",
        purpose="registered pair source-code identity",
    )
    _require(
        evidence["source_code_raw_sha256"] == raw_digest(source_raw),
        "pairwise evidence was produced by different source code",
        code="PAIR_BOOTSTRAP_PLAN_MISMATCH",
    )
    rows = evidence["rows"]
    _require(
        isinstance(rows, list) and bool(rows),
        "pairwise evidence rows are empty",
        code="PAIR_ROWS_UNALIGNED",
    )
    prepared = _prepare_wcr_rows(rows)
    _require(
        prepared.candidate_id == pair["candidate_a_id"]
        and prepared.baseline_id == pair["candidate_b_id"]
        and prepared.output_id == pair["output_id"]
        and prepared.stratum_id == "overall",
        "pairwise WCR rows differ from frozen A/B/output identity",
        code="PAIR_IDENTITY_MISMATCH",
    )
    _require(
        prepared.score_kind == "LOG_LOSS",
        "pairwise complexity score must be log loss",
        code="PAIR_IDENTITY_MISMATCH",
    )
    row_ids = list(prepared.canonical_series_ids)
    differences = [
        {
            "series_id": series_id,
            "candidate_a_minus_candidate_b": float(difference),
        }
        for series_id, difference in zip(
            prepared.canonical_series_ids,
            prepared.differences,
        )
    ]
    macro_weights = [
        {
            "series_id": series_id,
            "macro_weight": float(weight),
        }
        for series_id, weight in zip(
            prepared.canonical_series_ids,
            prepared.weights,
        )
    ]
    registered_folds = [
        {
            "series_id": row["series_id"],
            "fold_id": row["fold_id"],
            "registered_fold_ids": row["registered_fold_ids"],
        }
        for row in prepared.rows
    ]
    leagues = [
        {
            "series_id": row["series_id"],
            "league_id": row["league_id"],
        }
        for row in prepared.rows
    ]
    pth_assignments = [
        {
            "series_id": row["series_id"],
            _P: row[_P],
            _T: row[_T],
            _H: row[_H],
        }
        for row in prepared.rows
    ]
    critical_selectors = [
        {
            "series_id": row["series_id"],
            "game_side": row["game_side"],
            "roster_change": row["roster_change"],
            "international_event": row["international_event"],
            "draft_depth": row["draft_depth"],
        }
        for row in prepared.rows
    ]
    outcomes = [
        {
            "series_id": row["series_id"],
            "outcome": row["outcome"],
        }
        for row in prepared.rows
    ]
    clusters = [
        {
            "series_id": row["series_id"],
            _P: row[_P],
            _T: row[_T],
            _H: row[_H],
            "P_T": canonical_intersection_id([row[_P], row[_T]]),
            "P_H": canonical_intersection_id([row[_P], row[_H]]),
            "T_H": canonical_intersection_id([row[_T], row[_H]]),
            "P_T_H": canonical_intersection_id(
                [row[_P], row[_T], row[_H]]
            ),
        }
        for row in prepared.rows
    ]
    derived = {
        "aligned_row_ids_sha256": stable_digest(row_ids),
        "difference_rows_sha256": stable_digest(differences),
        "registered_fold_ids_sha256": stable_digest(registered_folds),
        "league_ids_sha256": stable_digest(leagues),
        "macro_weights_sha256": stable_digest(macro_weights),
        "outcomes_sha256": stable_digest(outcomes),
        "pth_assignments_sha256": stable_digest(pth_assignments),
        "critical_selectors_sha256": stable_digest(critical_selectors),
        "cluster_assignments_sha256": stable_digest(clusters),
        "candidate_a_prediction_rows_sha256": _derived_prediction_rows_sha256(
            prepared.rows, role="candidate"
        ),
        "candidate_b_prediction_rows_sha256": _derived_prediction_rows_sha256(
            prepared.rows, role="baseline"
        ),
        "input_rows_sha256": _derived_input_rows_sha256(prepared.rows),
        "analysis_rows_sha256": prepared.analysis_rows_sha256,
        "bootstrap_plan_sha256": _registered_pair_wcr_plan_sha256(),
    }
    for field, value in derived.items():
        mismatch_code = {
            "aligned_row_ids_sha256": "PAIR_ROWS_UNALIGNED",
            "difference_rows_sha256": "PAIR_DIFFERENCE_MISMATCH",
            "bootstrap_plan_sha256": "PAIR_BOOTSTRAP_PLAN_MISMATCH",
            "registered_fold_ids_sha256": "PAIR_FOLD_MISMATCH",
            "league_ids_sha256": "PAIR_LEAGUE_MISMATCH",
            "macro_weights_sha256": "PAIR_WEIGHT_MISMATCH",
            "outcomes_sha256": "PAIR_OUTCOME_MISMATCH",
            "pth_assignments_sha256": "PAIR_PTH_MISMATCH",
            "critical_selectors_sha256": "PAIR_CRITICAL_SELECTOR_MISMATCH",
            "cluster_assignments_sha256": "PAIR_CLUSTER_MISMATCH",
            "candidate_a_prediction_rows_sha256": (
                "PAIR_IDENTITY_MISMATCH"
            ),
                "candidate_b_prediction_rows_sha256": (
                    "PAIR_IDENTITY_MISMATCH"
                ),
                "input_rows_sha256": "PAIR_IDENTITY_MISMATCH",
                "analysis_rows_sha256": "PAIR_IDENTITY_MISMATCH",
        }[field]
        _require(
            evidence[field] == value and pair[field] == value,
            f"pairwise frozen binding mismatch: {field}",
            code=mismatch_code,
        )
    return prepared


def _candidate_gate_status_at(
    context: _AuthorityContext,
    candidate_id: str,
    *,
    attempts: int,
) -> str:
    _candidate_record(context, candidate_id)
    candidate_slots = [
        slot
        for slot in context.candidate_registry["slots"]
        if slot["candidate_id"] == candidate_id
    ]
    if not candidate_slots or any(
        slot["status"] != "RESOLVED" for slot in candidate_slots
    ):
        return "BLOCKED_TYPED_UNRESOLVED"
    families = [
        family
        for family in context.candidate_registry["families"]
        if family["candidate_id"] == candidate_id
    ]
    if {family["kind"] for family in families} != {
        "PRIMARY_HOLM",
        "HARM_HOLM",
        "SECONDARY_HOLM",
    } or len(families) != 3:
        return "BLOCKED_INCOMPLETE_FAMILY_INVENTORY"
    try:
        reports = [
            _compute_registered_holm_at(
                context,
                candidate_id,
                family["family_id"],
                attempts=attempts,
            )
            for family in families
        ]
    except BenchmarkContractError:
        return "BLOCKED_INFERENCE"
    return (
        "PASS"
        if all(report.family_gate_pass for report in reports)
        else "BLOCKED_FAMILY_DECISION"
    )


def _pair_wcr_runs(
    prepared: _PreparedWCR,
    *,
    attempts: int,
) -> tuple[_RegisteredSlotRun, ...]:
    return _registered_wcr_variants(
        prepared,
        theta0=lambda sample: float(
            np.sum(sample.weights * sample.differences)
        ),
        attempts=attempts,
        invert_endpoint=False,
    )


def _candidate_primary_prediction_digest_at(
    context: _AuthorityContext,
    candidate_id: str,
    output_id: str,
) -> str:
    families = _registry_index(
        context.candidate_registry["families"],
        "family_id",
        duplicate_code="CANDIDATE_SLOT_DUPLICATE",
    )
    slots = [
        slot
        for slot in context.candidate_registry["slots"]
        if slot["candidate_id"] == candidate_id
        and slot["output_id"] == output_id
        and families[slot["family_id"]]["kind"] == "PRIMARY_HOLM"
    ]
    _require(
        bool(slots) and all(slot["status"] == "RESOLVED" for slot in slots),
        "candidate primary prediction evidence is unresolved",
        code="CANDIDATE_SLOT_UNRESOLVED",
    )
    digests: set[str] = set()
    for slot in slots:
        prepared = _holm_evidence(context, slot)
        digests.add(
            _derived_prediction_rows_sha256(prepared.rows, role="candidate")
        )
    _require(
        len(digests) == 1,
        "candidate primary comparisons do not share one forecast identity",
        code="PAIR_IDENTITY_MISMATCH",
    )
    return next(iter(digests))


def _candidate_primary_input_digest_at(
    context: _AuthorityContext,
    candidate_id: str,
    output_id: str,
) -> str:
    """Return the one re-derived common sample identity for a candidate/output.

    Primary comparisons may differ only in their executable baseline.  Their
    candidate observation population and outcomes must be byte-for-byte the
    same canonical series surface; an opaque row SHA is not authority.
    """

    families = _registry_index(
        context.candidate_registry["families"],
        "family_id",
        duplicate_code="CANDIDATE_SLOT_DUPLICATE",
    )
    slots = [
        slot
        for slot in context.candidate_registry["slots"]
        if slot["candidate_id"] == candidate_id
        and slot["output_id"] == output_id
        and families[slot["family_id"]]["kind"] == "PRIMARY_HOLM"
    ]
    _require(
        bool(slots) and all(slot["status"] == "RESOLVED" for slot in slots),
        "candidate primary sample evidence is unresolved",
        code="CANDIDATE_SLOT_UNRESOLVED",
    )
    identities: set[str] = set()
    for slot in slots:
        prepared = _holm_evidence(context, slot)
        identities.add(_derived_input_rows_sha256(prepared.rows))
    _require(
        len(identities) == 1,
        "candidate primary comparisons do not share one sample identity",
        code="PAIR_IDENTITY_MISMATCH",
    )
    return next(iter(identities))


def _compute_registered_pairwise_intervals_at(
    context: _AuthorityContext,
    pair_family_id: str,
    *,
    attempts: int = _WCR_REPLICATES,
) -> PairwiseIntervalReport:
    families = _registry_index(
        context.pair_registry["families"],
        "pair_family_id",
        duplicate_code="CANDIDATE_SLOT_DUPLICATE",
    )
    _require(
        pair_family_id in families,
        f"pair family is not registered: {pair_family_id}",
        code="FAMILY_UNREGISTERED",
    )
    family = families[pair_family_id]
    pairs = _registry_index(
        context.pair_registry["pairs"],
        "pair_id",
        duplicate_code="CANDIDATE_SLOT_DUPLICATE",
    )
    declared = family["pair_ids"]
    actual = sorted(
        pair_id
        for pair_id, pair in pairs.items()
        if pair["pair_family_id"] == pair_family_id
    )
    _require(
        declared == sorted(declared)
        and len(declared) == len(set(declared))
        and set(declared) == set(actual),
        "pair family membership differs from its frozen registry",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    selected = [pairs[pair_id] for pair_id in declared]
    unresolved = [
        pair["pair_id"]
        for pair in selected
        if pair["status"] != "RESOLVED"
    ]
    _require(
        not unresolved,
        f"typed-unresolved pairs block simultaneous intervals: {unresolved}",
        code="CANDIDATE_SLOT_UNRESOLVED",
    )
    for pair in selected:
        _candidate_record(context, pair["candidate_a_id"])
        _candidate_record(context, pair["candidate_b_id"])
    prepared = [
        _validated_pair_evidence(context, pair) for pair in selected
    ]
    aligned_rows = {
        pair["aligned_row_ids_sha256"] for pair in selected
    }
    plans = {pair["bootstrap_plan_sha256"] for pair in selected}
    weights = {pair["macro_weights_sha256"] for pair in selected}
    folds = {pair["registered_fold_ids_sha256"] for pair in selected}
    leagues = {pair["league_ids_sha256"] for pair in selected}
    outcomes = {pair["outcomes_sha256"] for pair in selected}
    clusters = {
        pair["cluster_assignments_sha256"] for pair in selected
    }
    pth = {pair["pth_assignments_sha256"] for pair in selected}
    critical_selectors = {
        pair["critical_selectors_sha256"] for pair in selected
    }
    _require(
        len(aligned_rows) == 1,
        "pair family rows are not aligned",
        code="PAIR_ROWS_UNALIGNED",
    )
    _require(
        len(plans) == 1,
        "pair family bootstrap plans are not aligned",
        code="PAIR_BOOTSTRAP_PLAN_MISMATCH",
    )
    _require(
        len(weights)
        == len(folds)
        == len(leagues)
        == len(outcomes)
        == len(pth)
        == len(critical_selectors)
        == len(clusters)
        == 1,
        "pair family fold, league, weighting, outcome, PTH, selector, or cluster identities drifted",
        code="PAIR_IDENTITY_MISMATCH",
    )
    candidate_ids = tuple(
        sorted(
            {
                candidate_id
                for pair in selected
                for candidate_id in (
                    pair["candidate_a_id"],
                    pair["candidate_b_id"],
                )
            }
        )
    )
    gate_status = tuple(
        (
            candidate_id,
            _candidate_gate_status_at(
                context,
                candidate_id,
                attempts=attempts,
            ),
        )
        for candidate_id in candidate_ids
    )
    if not all(status == "PASS" for _, status in gate_status):
        return PairwiseIntervalReport(
            pair_family_id=pair_family_id,
            critical_value=None,
            intervals=(),
            sensitivities=(),
            sensitivity_conclusions_agree=False,
            candidate_gate_status=gate_status,
            selection_status="BLOCKED_CANDIDATE_FAMILIES",
            winner_candidate_id=None,
        )
    for pair in selected:
        _require(
            pair["candidate_a_prediction_rows_sha256"]
            == _candidate_primary_prediction_digest_at(
                context,
                pair["candidate_a_id"],
                pair["output_id"],
            )
            and pair["candidate_b_prediction_rows_sha256"]
            == _candidate_primary_prediction_digest_at(
                context,
                pair["candidate_b_id"],
                pair["output_id"],
            ),
            "pair forecasts differ from independently gated candidate forecasts",
            code="PAIR_IDENTITY_MISMATCH",
        )
        _require(
            pair["input_rows_sha256"]
            == _candidate_primary_input_digest_at(
                context,
                pair["candidate_a_id"],
                pair["output_id"],
            )
            == _candidate_primary_input_digest_at(
                context,
                pair["candidate_b_id"],
                pair["output_id"],
            ),
            "pair sample differs from independently gated primary samples",
            code="PAIR_IDENTITY_MISMATCH",
        )

    runs_by_pair = [
        _pair_wcr_runs(item, attempts=attempts)
        for item in prepared
    ]
    variant_order = [run.variant_id for run in runs_by_pair[0]]
    _require(
        all(
            [run.variant_id for run in runs] == variant_order
            for runs in runs_by_pair
        ),
        "pair WCR sensitivity inventories are not aligned",
        code="PAIR_BOOTSTRAP_PLAN_MISMATCH",
    )
    sensitivity_reports: list[PairwiseSensitivityReport] = []
    for variant_index in range(len(variant_order)):
        variant_runs = [
            runs[variant_index] for runs in runs_by_pair
        ]
        _require(
            len(
                {
                    (
                        run.removed_dimension,
                        run.removed_cluster_id,
                    )
                    for run in variant_runs
                }
            )
            == 1,
            "pair WCR removals are not aligned",
            code="PAIR_CLUSTER_MISMATCH",
        )
        maxima = [
            max(
                abs(float(run.result.bootstrap_t[replicate]))
                for run in variant_runs
            )
            for replicate in range(attempts)
        ]
        critical = _type7_quantile(maxima, 0.95)
        intervals = tuple(
            PairwiseInterval(
                pair_id=pair["pair_id"],
                output_id=pair["output_id"],
                candidate_a_id=pair["candidate_a_id"],
                candidate_b_id=pair["candidate_b_id"],
                point_estimate=run.result.point_estimate,
                simultaneous_two_sided_95_bootstrap_t_lower=(
                    run.result.point_estimate
                    - critical * run.result.standard_error
                ),
                simultaneous_two_sided_95_bootstrap_t_upper=(
                    run.result.point_estimate
                    + critical * run.result.standard_error
                ),
            )
            for pair, run in zip(selected, variant_runs)
        )
        if all(interval.upper_bound < 0.0 for interval in intervals):
            conclusion = "CANDIDATE_A_DOMINATES"
        elif all(interval.lower_bound > 0.0 for interval in intervals):
            conclusion = "CANDIDATE_B_DOMINATES"
        elif all(
            interval.lower_bound <= 0.0 <= interval.upper_bound
            for interval in intervals
        ):
            conclusion = "INDISTINGUISHABLE"
        else:
            conclusion = "MIXED"
        first = variant_runs[0]
        sensitivity_reports.append(
            PairwiseSensitivityReport(
                sample_variant=first.sample_variant,
                active_dimensions=first.active_dimensions,
                bootcluster_dimension=first.bootcluster_dimension,
                critical_value=critical,
                intervals=intervals,
                directional_conclusion=conclusion,
            )
        )
    sensitivities = tuple(sensitivity_reports)
    conclusions = {
        report.directional_conclusion for report in sensitivities
    }
    conclusions_agree = len(conclusions) == 1
    canonical = next(
        report
        for report in sensitivities
        if report.sample_variant == "full"
        and report.active_dimensions == _PRIMARY_ACTIVE
        and report.bootcluster_dimension == _P
    )
    winner: str | None = None
    if conclusions_agree and conclusions == {"CANDIDATE_A_DOMINATES"}:
        winner = selected[0]["candidate_a_id"]
        selection_status = "DIRECTIONAL_WINNER"
    elif conclusions_agree and conclusions == {"CANDIDATE_B_DOMINATES"}:
        winner = selected[0]["candidate_b_id"]
        selection_status = "DIRECTIONAL_WINNER"
    elif conclusions_agree and conclusions == {"INDISTINGUISHABLE"}:
        candidates = _registry_index(
            context.candidate_registry["candidates"],
            "candidate_id",
            duplicate_code="CANDIDATE_SLOT_DUPLICATE",
        )
        winner = min(
            candidate_ids,
            key=lambda candidate_id: (
                candidates[candidate_id]["simplicity_rank"],
                candidate_id,
            ),
        )
        selection_status = "SIMPLICITY_WINNER"
    else:
        selection_status = "NO_WINNER_REMAND"
    return PairwiseIntervalReport(
        pair_family_id=pair_family_id,
        critical_value=canonical.critical_value,
        intervals=canonical.intervals,
        sensitivities=sensitivities,
        sensitivity_conclusions_agree=conclusions_agree,
        candidate_gate_status=gate_status,
        selection_status=selection_status,
        winner_candidate_id=winner,
    )


def compute_registered_pairwise_intervals(
    pair_family_id: str,
) -> PairwiseIntervalReport:
    """Compute one frozen A-minus-B pair family from pinned evidence."""

    context = _verify_authoritative_preflight_at(
        _PRODUCTION_PACKAGE_ROOT,
        _REPO_ROOT,
    )
    return _compute_registered_pairwise_intervals_at(
        context,
        pair_family_id,
    )


def _derive_registered_margin_at(
    context: _AuthorityContext,
    candidate_id: str,
    baseline_id: str,
) -> float:
    _reject_source_bound_transition(
        "resolved development margin",
        f"{candidate_id}:{baseline_id}",
    )
    _candidate_record(context, candidate_id)
    families = _registry_index(
        context.candidate_registry["families"],
        "family_id",
        duplicate_code="CANDIDATE_SLOT_DUPLICATE",
    )
    slots = [
        slot
        for slot in context.candidate_registry["slots"]
        if slot["candidate_id"] == candidate_id
        and slot["baseline_id"] == baseline_id
        and slot["stratum_id"] == "overall"
        and families[slot["family_id"]]["kind"] == "PRIMARY_HOLM"
    ]
    _require(
        len(slots) == 1,
        "registered margin slot is missing or ambiguous",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    slot = slots[0]
    _require(
        slot["status"] == "RESOLVED"
        and isinstance(slot.get("margin_binding"), Mapping),
        "registered margin remains typed unresolved",
        code="CANDIDATE_SLOT_UNRESOLVED",
    )
    binding = slot["margin_binding"]
    bundle = _read_registry_bound_json(
        context,
        binding,
        purpose=f"margin bundle {candidate_id} vs {baseline_id}",
        mismatch_code="FAMILY_DERIVATION_MISMATCH",
    )
    return _derive_margin_from_bundle(bundle, binding)


def derive_registered_margin(
    candidate_id: str,
    baseline_id: str,
) -> float:
    """Derive a margin from the sole authority-registered replica bundle."""

    context = _verify_authoritative_preflight_at(
        _PRODUCTION_PACKAGE_ROOT,
        _REPO_ROOT,
    )
    return _derive_registered_margin_at(
        context,
        candidate_id,
        baseline_id,
    )


def validate_benchmark_contract(contract: Mapping[str, Any]) -> None:
    _require(contract.get("artifact_kind") == "PRE_BINDING_BENCHMARK_TEMPLATE", "benchmark artifact kind is wrong")
    _require(contract.get("contract_id") == "scryglass:real-benchmark:v1", "contract_id is wrong")
    _require(contract.get("schema_version") == "1.4", "benchmark schema version is wrong")
    _require(contract.get("final_labels_read") is False, "benchmark must record final_labels_read=false")

    acceptance = contract["acceptance"]
    _require(acceptance["source_independent_rules_status"] == "FROZEN_CANDIDATE", "source-independent rules are not frozen")
    _require(acceptance["source_dependent_binding_status"] == "UNBOUND", "pre-binding template must remain unbound")
    _require(acceptance["status"] == "PENDING_INDEPENDENT_REVIEW", "artifact must not self-approve")
    _require(acceptance["binding_cannot_change_source_independent_rules"] is True, "binding immutability rule missing")

    outputs = _as_id_set(contract["outputs"], "outputs")
    _require(
        {"player_rating", "team_rating", "terminal_draft_score", "partial_draft_score"}
        == outputs,
        "output inventory must be exact",
    )
    population = contract["population"]
    league_ids = tuple(item["league_id"] for item in population["domestic_leagues"])
    _require(league_ids == _PRIMARY_LEAGUES, "eligible Tier-1 league order or identity is wrong")
    _require(population["competition_tier"] == "tier1", "population must be Tier-1")
    _require(population["international_events"] == ["MSI", "EWC", "OTHER_NAMED_EVENT"], "international scopes are wrong")
    _require(population["international_events_excluded_from_domestic_primary"] is True, "international events must be separate")

    outcome = contract["outcome"]
    _require(outcome["name"] == "map_winner", "primary outcome must be map_winner")
    _require(outcome["unit"] == "professional_map", "primary unit must be professional_map")
    _require(outcome["series_atomic"] is True, "series must be atomic")
    _require(outcome["unresolved_series_primary_action"] == "EXCLUDE", "unresolved series must be excluded")

    partition = contract["partition_template"]
    _require(partition["status"] == "UNBOUND", "partition template must remain unbound")
    _require(partition["row_identity_digest"] == "UNBOUND", "unbound partition must not claim a digest")
    _require(partition["source_snapshot_digest"] == "UNBOUND", "unbound source must not claim a digest")

    primary = contract["primary_comparison"]
    _require(primary["metric"] == "log_loss", "primary metric must be log_loss")
    _require(primary["direction"] == "lower_is_better", "log-loss direction is wrong")
    _require(primary["paired_difference"] == "candidate_minus_baseline", "paired difference orientation is wrong")
    aggregation = primary["aggregation"]
    _require("resolved series" in aggregation["level_2"], "league-fold aggregation must use resolved series")
    _require(
        aggregation["level_3"] == "equal-weight mean across exactly LEC,LCK,LPL,LCS,LCP within fold",
        "macro-regional league weighting is wrong",
    )
    _require(aggregation["level_4"] == "equal-weight mean across registered chronological folds", "fold weighting is wrong")
    _require(aggregation["missing_registered_league_in_fold"] == "PRIMARY_UNAVAILABLE", "missing league must block primary")

    diagnostics = _as_id_set(contract["diagnostics"], "diagnostics")
    _require(diagnostics == _REQUIRED_DIAGNOSTICS, "diagnostic inventory is incomplete or expanded without versioning")
    _require(
        set(contract["output_diagnostics"]) == outputs,
        "output-diagnostic mapping must cover every output exactly",
        code="OUTPUT_DIAGNOSTICS_MISSING",
    )
    for output, required in contract["output_diagnostics"].items():
        _require(
            isinstance(required, list) and bool(required),
            f"{output}: diagnostic mapping is empty",
            code="OUTPUT_DIAGNOSTICS_MISSING",
        )
        _require(
            len(required) == len(set(required)),
            f"{output}: diagnostic mapping contains duplicates",
            code="OUTPUT_DIAGNOSTICS_DUPLICATE",
        )
        _require(
            set(required) <= diagnostics,
            f"{output}: diagnostic mapping contains an unknown ID",
            code="OUTPUT_DIAGNOSTICS_UNKNOWN",
        )
        _require(
            tuple(required) == _EXPECTED_OUTPUT_DIAGNOSTICS[output],
            f"{output}: diagnostic mapping is not the registered output mapping",
            code="OUTPUT_DIAGNOSTICS_MISSING",
        )

    strata = _as_id_set(contract["critical_strata"], "critical_strata")
    _require(strata == _REQUIRED_STRATA, "critical-strata inventory is incomplete or expanded without versioning")
    by_stratum = {item["id"]: item for item in contract["critical_strata"]}
    _require(by_stratum["league"]["levels"] == list(_PRIMARY_LEAGUES), "league stratum levels are wrong")
    _require(by_stratum["game_side"]["levels"] == ["BLUE", "RED"], "game-side levels are wrong")
    _require(by_stratum["draft_depth"]["levels"] == list(range(11)), "draft-depth levels must be 0 through 10")
    for item in contract["critical_strata"]:
        support = item["support"]
        _require(support["min_resolved_series"] >= 30, f"{item['id']}: minimum series must be at least 30")
        _require(support["min_effective_clusters"] >= 30, f"{item['id']}: minimum clusters must be at least 30")
        _require(support["both_outcomes"] is True, f"{item['id']}: both outcomes are required")
        _require(
            item["unsupported_action"] == "DESCRIPTIVE_ONLY_BLOCK_INFERENCE_AND_PROMOTION",
            f"{item['id']}: unsupported action must be descriptive-only and fail closed",
        )

    dependence = contract["dependence"]
    _require(dependence["analysis_row"] == "one paired candidate-minus-baseline mean loss difference per resolved series", "dependence analysis row is wrong")
    _require(dependence["bootstrap_replicates"] == 9999, "bootstrap replicate count is not frozen")
    _require(dependence["deterministic_seed"] == 2026072901, "bootstrap seed is not frozen")
    _require(dependence["minimum_effective_clusters"]["minimum_each"] == 30, "primary effective-cluster minimum is wrong")
    _require(dependence["minimum_effective_clusters"]["patch_sensitivity_minimum"] == 30, "patch cluster minimum is wrong")
    _require("Webb six-point" in dependence["small_cluster_correction"], "small-cluster correction is not frozen")
    algorithm = dependence["algorithm"]
    _require(
        algorithm["algorithm_id"] == _WCR_EXECUTION_ID,
        "multiway WCR algorithm identity is wrong",
    )
    _require(
        all(
            field in algorithm["input_schema"]
            for field in (
                "output_id",
                "stratum_id",
                "outcome",
                "macro_weight",
            )
        ),
        "WCR input schema omits v1.4 inferential identities",
    )
    _require("all seven P,T,H,PT,PH,TH,PTH" in algorithm["intersection_sets"], "multiway intersection-set rule is incomplete")
    _require("c_S=G_S/(G_S-1)" in algorithm["finite_sample_correction"], "finite-sample correction is not frozen")
    _require(
        "weighted intercept score" in algorithm["covariance"]
        and "bootstrap residuals" in algorithm["covariance"],
        "multiway covariance rule is wrong",
    )
    _require("modulo support length" in algorithm["multiplier_substream"], "multiplier substream derivation is under-specified")
    _require(
        "HMAC-SHA-256" in algorithm["multiplier_substream"]
        and "no runtime RNG, subset multiplier, candidate ID, or comparison ID"
        in algorithm["multiplier_substream"],
        "multiplier alignment identity is wrong",
    )
    _require(
        algorithm["webb_support"]
        == ["-sqrt(3/2)", "-1", "-sqrt(1/2)", "sqrt(1/2)", "1", "sqrt(3/2)"],
        "Webb multiplier support is wrong",
    )
    _require(algorithm["rademacher_support"] == ["-1", "1"], "Rademacher support is wrong")
    _require(
        "recompute every Q*" in algorithm["studentizer"],
        "bootstrap studentizer must be recomputed from refit residuals",
    )
    _require(algorithm["variance_floor"] == 1e-12, "bootstrap variance floor is wrong")
    _require(algorithm["quantile"] == "Hyndman-Fan type 7", "bootstrap quantile rule is wrong")
    _require(
        "invert the null-imposed lower-tail WCR test"
        in algorithm["one_sided_upper_endpoint"]
        and "diagnostic only" in algorithm["one_sided_upper_endpoint"],
        "one-sided WCR endpoint is wrong",
    )
    _require(
        algorithm["largest_cluster_tie_break"]
        == "largest resolved-series count, then lexicographically smallest canonical cluster ID",
        "largest-cluster tie rule is wrong",
    )
    _require(
        dependence["interval"].startswith(
            "one-sided 95% upper confidence bound obtained by null-imposed WCR"
        ),
        "interval rule is wrong",
    )
    _require(
        "recomputes Holm ordering"
        in dependence["decision_agreement"]
        and "promotion is blocked" in dependence["decision_agreement"],
        "sensitivity disagreement must block",
    )
    _require(dependence["singleton_or_map_independence_fallback"] == "PROHIBITED", "map-independent fallback is prohibited")

    margin = contract["margin_derivation"]
    _require(margin["status"] == "PROCEDURE_FROZEN_NUMBERS_DEFERRED", "margin procedure status is wrong")
    _require(margin["numeric_binding"] == "DEFERRED_UNTIL_DEVELOPMENT_EVIDENCE", "margin numeric binding must be deferred")
    _require(
        margin["entry_point"].endswith(":derive_registered_margin"),
        "margin procedure entry point is wrong",
    )
    _require(margin["minimum_replicates"] == 30, "margin replicate minimum is wrong")
    _require("type-7 95th percentile" in margin["statistic"], "margin statistic is not frozen")
    _require(margin["replica_construction_id"] == _MARGIN_CONSTRUCTION_ID, "margin replica construction identity is wrong")
    _require("2026072902+i" in margin["replica_seed_schedule"], "margin seed schedule is wrong")
    _require(
        set(margin["required_binding"])
        == {
            "development_snapshot_sha256",
            "baseline_binding_sha256",
            "evaluation_rows_sha256",
            "replicate_payload_sha256",
            "procedure_sha256",
            "independent_review_id",
            "replica_construction_id",
        },
        "margin binding inventory is incomplete",
    )
    _require(margin["fallback_when_unbound"] == "STRICT_SUPERIORITY_WITH_ZERO_MARGIN", "unbound margin fallback is wrong")

    promotion = contract["promotion"]
    _require(promotion["familywise_alpha"] == 0.05, "familywise alpha is wrong")
    _require(
        promotion["multiplicity"]["method"].startswith(
            "Holm step-down, one-sided, complete predeclared family"
        ),
        "multiplicity method is wrong",
    )
    _require(promotion["multiplicity"]["missing_or_unsupported_member"] == "BLOCK_RELEVANT_PROMOTION", "missing multiplicity member must block")
    _require(
        promotion["multiplicity"]["family_gate_entry_point"].endswith(
            ":compute_registered_holm"
        ),
        "Holm family-gate implementation is not bound",
    )
    _require(
        "unadjusted one-sided 95%" in promotion["multiplicity"]["pass_coupling"]
        and "Holm rejection" in promotion["multiplicity"]["pass_coupling"],
        "endpoint is mislabeled or uncoupled from Holm rejection",
    )
    _require(
        promotion["noninferiority_secondary_benefit_path"].startswith(
            "UNAVAILABLE_UNTIL_A_SEPARATE_REGISTERED_MARGIN_THRESHOLD"
        ),
        "zero-threshold evidence must not imply a noninferiority route",
    )
    _require("threshold is 0" in promotion["material_harm"], "material-harm fallback threshold is not frozen")
    _require(promotion["unresolved_action"] == "BLOCK_PROMOTION", "unresolved promotion choices must block")

    complexity = contract["complexity"]
    _require(complexity["selection"] == "choose lower numeric simplicity_rank among indistinguishable passing candidates", "complexity selection is wrong")
    _require(
        "simultaneous" in complexity["indistinguishable_rule"]
        and "contain" in complexity["indistinguishable_rule"],
        "indistinguishability rule is not frozen",
    )
    pair_entry_point = complexity["simultaneous_interval_entry_point"]
    _require(
        ":compute_registered_pairwise_intervals" in pair_entry_point
        and "20 separate centered wild-cluster CGM simultaneous max-t"
        in pair_entry_point
        and "two-sided 95% bands" in pair_entry_point
        and "not a pooled generic CI or inverted WCR endpoints"
        in pair_entry_point,
        "simultaneous candidate-interval implementation is not bound",
    )
    _require(
        "Holm rejection" in complexity["ablation_rule"]
        and "unadjusted one-sided 95%" in complexity["ablation_rule"],
        "ablation endpoint is mislabeled or not frozen",
    )
    _require(complexity["unsupported_component_action"] == "COLLAPSE_OR_REMOVE", "unsupported component action is wrong")

    access = contract["label_access"]
    _require(
        access["boundary_id"]
        == "scryglass:authority-derived-preflight:v1.4",
        "label boundary id is wrong",
    )
    _require(
        access["required_loader_entry_point"].endswith(
            ":verify_authoritative_preflight"
        ),
        "required authority-derived preflight is wrong",
    )
    _require(set(access["forbidden_exact_basenames"]) >= _FORBIDDEN_EXACT_BASENAMES, "forbidden filenames are incomplete")
    _require(set(access["forbidden_payload_keys"]) >= (_FORBIDDEN_PAYLOAD_KEYS - {"outcome"}), "forbidden payload keys are incomplete")
    _require(
        access["ordinary_preflight_requires_attestation"] is False,
        "caller attestation must not be an authority input",
    )
    _require(access["opening_requires_independent_permit"] is True, "independent permit is required")
    _require(
        access["required_attestation_fields"]
        == ["NONE_ACCEPTED_AS_AUTHORITY"],
        "caller attestation is still accepted as authority",
    )

    authority = contract["authority_model"]
    _require(
        authority["production_entry_points_accept_caller_authority"] is False
        and authority["production_authority_status"] == "BLOCKED"
        and len(authority["required_external_boundary"]) >= 3,
        "production authority threat ceiling is not fail-closed",
    )

    digest = contract["digest_policy"]
    _require(digest["manifest_requires_both"] is True, "manifest must bind semantic and raw digests")
    _require(digest["authority_code_raw_binding_required"] is True, "authority code must be raw-byte bound")
    _require(contract["opening_permit_schema"] == OPENING_PERMIT_SCHEMA, "permit-schema linkage is wrong")


def validate_baseline_registry(
    registry: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> None:
    _require(registry.get("artifact_kind") == "PRE_BINDING_BASELINE_TEMPLATE", "baseline artifact kind is wrong")
    _require(registry.get("registry_id") == "scryglass:real-baselines:v1", "baseline registry id is wrong")
    _require(registry.get("schema_version") == "1.4", "baseline schema version is wrong")
    _require(registry.get("final_labels_read") is False, "baseline registry must record final_labels_read=false")
    _reject_nulls(registry, label="baseline registry")
    validate_no_final_labels(registry, label="baseline registry")

    acceptance = registry["acceptance"]
    _require(acceptance["source_independent_inventory_status"] == "FROZEN_CANDIDATE", "baseline inventory is not frozen")
    _require(acceptance["status"] == "PENDING_INDEPENDENT_REVIEW", "baseline registry must not self-approve")
    binding_contract = registry["binding_contract"]
    _require(binding_contract["null_placeholders_prohibited"] is True, "null placeholders must be prohibited")
    _require(
        binding_contract.get("source_bound_transition")
        == _G1_SOURCE_BOUND_TRANSITION,
        "v1.4 baseline transition boundary is not hard closed",
        code="G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED",
    )

    entries = registry["baselines"]
    ids = _as_id_set(entries, "baselines")
    _require(ids == _REQUIRED_BASELINES, "required baseline inventory is incomplete or changed")
    if any(entry.get("status") != "TYPED_UNAVAILABLE" for entry in entries):
        _reject_source_bound_transition("executable baseline registry")
    _require(
        acceptance["source_dependent_execution_bindings_status"] == "UNBOUND",
        "v1.4 baseline execution state must remain UNBOUND",
        code="G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED",
    )
    ranks: set[tuple[str, int]] = set()
    for entry in entries:
        baseline_id = entry["id"]
        expected_depths = (
            _PARTIAL_DEPTHS
            if baseline_id in _REQUIRED_PARTIAL_BASELINES
            else _TERMINAL_DEPTH
            if baseline_id in _REQUIRED_TERMINAL_BASELINES
            else _RATING_DEPTH
        )
        _require(tuple(entry["applies_to_draft_depths"]) == expected_depths, f"{baseline_id}: draft-depth coverage is incomplete")
        _require(entry["label_access"] == "NO_FINAL_LABEL_ACCESS", f"{baseline_id}: final-label access is forbidden")
        rank_key = (entry["comparison_output"], entry["simplicity_rank"])
        _require(rank_key not in ranks, f"{baseline_id}: duplicate simplicity rank within output")
        ranks.add(rank_key)
        _require(
            entry["status"] == "TYPED_UNAVAILABLE",
            f"{baseline_id}: v1.4 baseline status must remain unavailable",
            code="G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED",
        )
        unavailable = entry["unavailable"]
        _require(unavailable["reason_code"] in _ALLOWED_UNAVAILABLE_REASONS, f"{baseline_id}: unavailable reason is not permitted")
        _require(bool(unavailable["evidence_locator"]), f"{baseline_id}: unavailable evidence locator is required")
        _require(unavailable["resolution_task"] == "G1-018", f"{baseline_id}: resolution task must be G1-018")
        _require(unavailable["claim_effect"] == "REQUIRED_COMPARISON_BLOCKED_UNTIL_RESOLVED", f"{baseline_id}: unavailable comparator must block claim")

    components = registry["component_ablation_registry"]
    component_ids = {item["component_id"] for item in components}
    _require(len(component_ids) == len(components), "component-to-ablation registry has duplicates")
    _require(component_ids == _REQUIRED_COMPONENTS, "component-to-ablation inventory is incomplete or changed")
    for item in components:
        _require(item["ablation_baseline_id"] in ids, f"{item['component_id']}: ablation baseline is missing")
        _require(
            item["required_decision"]
            == (
                "HOLM_REJECTION_PLUS_UNADJUSTED_ONE_SIDED_95_"
                "WCR_INVERTED_UPPER_BOUND_BELOW_ZERO"
            ),
            f"{item['component_id']}: ablation decision is wrong",
        )


def _expected_candidate_inventory(
    baseline_registry: Mapping[str, Any],
    *,
    held_out_patches: Sequence[str] | None = None,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    baselines_by_output = {
        output: tuple(
            sorted(
                baseline["id"]
                for baseline in baseline_registry["baselines"]
                if baseline["comparison_output"] == output
            )
        )
        for output in _OUTPUT_IDS
    }
    families: dict[str, dict[str, Any]] = {}
    slots: dict[str, dict[str, Any]] = {}
    patch_cells = (
        tuple(f"patch:{patch}" for patch in held_out_patches)
        if held_out_patches is not None
        else ("patch:each-held-out-major-minor",)
    )
    for candidate_id in _CANDIDATE_IDS:
        primary_family = f"primary:{candidate_id}"
        harm_family = f"harm:{candidate_id}"
        secondary_family = f"secondary:{candidate_id}"
        primary_ids: list[str] = []
        harm_ids: list[str] = []
        secondary_ids: list[str] = []
        for output in _OUTPUT_IDS:
            for baseline_id in baselines_by_output[output]:
                primary_id = (
                    f"slot:primary:{candidate_id}:{output}:"
                    f"{baseline_id}:overall"
                )
                primary_ids.append(primary_id)
                slots[primary_id] = {
                    "slot_id": primary_id,
                    "family_id": primary_family,
                    "candidate_id": candidate_id,
                    "baseline_id": baseline_id,
                    "output_id": output,
                    "stratum_id": "overall",
                    "decision_kind": "SUPERIORITY",
                    "threshold_source": "ZERO",
                }
                critical_cells = tuple(
                    cell
                    for cell in _CRITICAL_CELLS_BY_OUTPUT[output]
                    if cell != "patch:each-held-out-major-minor"
                ) + patch_cells
                for cell_id in critical_cells:
                    harm_id = (
                        f"slot:harm:{candidate_id}:{output}:"
                        f"{baseline_id}:{cell_id}"
                    )
                    harm_ids.append(harm_id)
                    slots[harm_id] = {
                        "slot_id": harm_id,
                        "family_id": harm_family,
                        "candidate_id": candidate_id,
                        "baseline_id": baseline_id,
                        "output_id": output,
                        "stratum_id": cell_id,
                        "decision_kind": "HARM",
                        "threshold_source": (
                            "REGISTERED_MARGIN_OR_ZERO_FALLBACK"
                        ),
                    }
                for benefit_id in _SUPPORTED_SECONDARY_BENEFITS[output]:
                    secondary_id = (
                        f"slot:secondary:{candidate_id}:{output}:"
                        f"{baseline_id}:benefit:{benefit_id}"
                    )
                    secondary_ids.append(secondary_id)
                    slots[secondary_id] = {
                        "slot_id": secondary_id,
                        "family_id": secondary_family,
                        "candidate_id": candidate_id,
                        "baseline_id": baseline_id,
                        "output_id": output,
                        "stratum_id": f"benefit:{benefit_id}",
                        "decision_kind": "SUPERIORITY",
                        "threshold_source": "ZERO",
                    }
        families[primary_family] = {
            "family_id": primary_family,
            "kind": "PRIMARY_HOLM",
            "candidate_id": candidate_id,
            "slot_ids": sorted(primary_ids),
        }
        families[harm_family] = {
            "family_id": harm_family,
            "kind": "HARM_HOLM",
            "candidate_id": candidate_id,
            "slot_ids": sorted(harm_ids),
        }
        families[secondary_family] = {
            "family_id": secondary_family,
            "kind": "SECONDARY_HOLM",
            "candidate_id": candidate_id,
            "slot_ids": sorted(secondary_ids),
        }
    return families, slots


def validate_candidate_registry(
    registry: Mapping[str, Any],
    baseline_registry: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> None:
    _require(
        registry.get("schema_version") == "g0-candidate-registry-v1.4"
        and registry.get("registry_id")
        == "scryglass:candidate-registry:v1.4",
        "candidate registry identity is wrong",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    _require(
        registry.get("final_labels_read") is False
        and _is_digest(registry.get("contract_set_sha256")),
        "candidate registry authority identity is malformed",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    _require(
        registry.get("g1_unified_authority_handoff")
        == _G1_CANDIDATE_AUTHORITY_HANDOFF,
        "candidate authority handoff differs from the frozen G1 boundary",
        code="G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED",
    )
    if repo_root is not None:
        _validate_g1_human_authority_artifact(
            registry["g1_unified_authority_handoff"],
            Path(repo_root),
        )
    candidates = _registry_index(
        registry.get("candidates"),
        "candidate_id",
        duplicate_code="CANDIDATE_SLOT_DUPLICATE",
    )
    families = _registry_index(
        registry.get("families"),
        "family_id",
        duplicate_code="CANDIDATE_SLOT_DUPLICATE",
    )
    slots = _registry_index(
        registry.get("slots"),
        "slot_id",
        duplicate_code="CANDIDATE_SLOT_DUPLICATE",
    )
    if any(
        candidate.get("status") != "TYPED_UNRESOLVED"
        for candidate in candidates.values()
    ):
        _reject_source_bound_transition("resolved candidate registry")
    if any(slot.get("status") != "TYPED_UNRESOLVED" for slot in slots.values()):
        _reject_source_bound_transition("resolved candidate slot registry")
    secondary_authority = registry.get("secondary_authority")
    _require(
        isinstance(secondary_authority, Mapping)
        and set(secondary_authority)
        == {
            "equal_strength_draft_increment",
            "transfer_new_roster",
        },
        "secondary authority inventory is incomplete or expanded",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    for authority_id, authority in secondary_authority.items():
        if not (
            isinstance(authority, Mapping)
            and authority.get("status") == "TYPED_UNRESOLVED"
            and set(authority) == {"status", "unavailable"}
        ):
            _reject_source_bound_transition(
                "resolved secondary authority",
                authority_id,
            )
        _require(
            authority
            == {
                "status": "TYPED_UNRESOLVED",
                "unavailable": {
                    "reason_code": (
                        "G0_103_REGISTERED_EQUAL_STRENGTH_OVERLAP_UNRESOLVED"
                        if authority_id == "equal_strength_draft_increment"
                        else (
                            "G0_103_REGISTERED_ROSTER_TIMELINE_"
                            "AUTHORITY_UNRESOLVED"
                        )
                    ),
                    "claim_effect": (
                        "BLOCK_RELEVANT_PROMOTION_AND_G9_OPENING"
                    ),
                },
            },
            f"typed-unresolved secondary authority changed: {authority_id}",
            code="CANDIDATE_SLOT_UNRESOLVED",
        )
    _require(
        "opening_candidate_id" not in registry,
        "opening candidate must be derived from the global complexity report",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    baseline_by_id = {
        entry["id"]: entry for entry in baseline_registry["baselines"]
    }
    _require(
        set(candidates) == set(_CANDIDATE_IDS)
        and {
            candidate_id: candidates[candidate_id]["simplicity_rank"]
            for candidate_id in candidates
        }
        == {
            "candidate:scryglass-v2": 100,
            "candidate:scryglass-v2-simplest-parent": 0,
        }
        ,
        "candidate inventory or simplicity ranks changed",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    for candidate_id, candidate in candidates.items():
        _require(
            candidate.get("unavailable")
            == {
                "reason_code": "G0_103_EXECUTABLE_ADAPTER_UNRESOLVED",
                "claim_effect": (
                    "BLOCK_RELEVANT_PROMOTION_AND_G9_OPENING"
                ),
            },
            f"typed-unresolved candidate reason changed: {candidate_id}",
            code="CANDIDATE_SLOT_UNRESOLVED",
        )
    patch_inventory = registry.get("held_out_patch_inventory")
    held_out_patches: tuple[str, ...] | None = None
    _require(
        patch_inventory is not None,
        "held-out patch inventory is missing",
        code="CANDIDATE_SLOT_UNRESOLVED",
    )
    if patch_inventory is not None:
        _require(
            isinstance(patch_inventory, Mapping),
            "held-out patch inventory is malformed",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        if patch_inventory.get("status") != "TYPED_UNRESOLVED":
            _reject_source_bound_transition(
                "resolved held-out patch inventory"
            )
        if patch_inventory.get("status") == "TYPED_UNRESOLVED":
            _require(
                patch_inventory
                == {
                    "status": "TYPED_UNRESOLVED",
                    "unavailable": {
                        "reason_code": (
                            "G0_103_HELD_OUT_PATCH_INVENTORY_UNRESOLVED"
                        ),
                        "claim_effect": (
                            "BLOCK_RELEVANT_PROMOTION_AND_G9_OPENING"
                        ),
                    },
                },
                "typed-unresolved held-out patch inventory is malformed",
                code="CANDIDATE_SLOT_UNRESOLVED",
            )
        else:
            _require(
                isinstance(patch_inventory, Mapping)
                and set(patch_inventory)
                == {"status", "patches", "patches_sha256"}
                and patch_inventory.get("status") == "RESOLVED"
                and isinstance(patch_inventory.get("patches"), list)
                and bool(patch_inventory["patches"])
                and patch_inventory["patches"] == sorted(patch_inventory["patches"])
                and len(patch_inventory["patches"])
                == len(set(patch_inventory["patches"]))
                and all(
                    isinstance(patch, str) and bool(patch)
                    for patch in patch_inventory["patches"]
                )
                and patch_inventory["patches_sha256"]
                == stable_digest(patch_inventory["patches"]),
                "held-out patch inventory is malformed or mutated",
                code="FAMILY_DERIVATION_MISMATCH",
            )
            held_out_patches = tuple(patch_inventory["patches"])
    expected_families, expected_slots = _expected_candidate_inventory(
        baseline_registry,
        held_out_patches=held_out_patches,
    )
    _require(
        set(families) == set(expected_families),
        "candidate family inventory is incomplete or expanded",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    _require(
        set(slots) == set(expected_slots),
        "candidate slot inventory is incomplete or expanded",
        code="CANDIDATE_SLOT_MISSING",
    )
    for family_id, family in families.items():
        _require(
            family["candidate_id"] in candidates,
            f"family candidate is unregistered: {family_id}",
            code="CANDIDATE_UNREGISTERED",
        )
        declared = family["slot_ids"]
        _require(
            declared == sorted(declared)
            and len(declared) == len(set(declared)),
            f"family slot list is not canonical: {family_id}",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        actual = sorted(
            slot_id
            for slot_id, slot in slots.items()
            if slot["family_id"] == family_id
        )
        _require(
            family == expected_families[family_id]
            and set(declared) == set(actual),
            f"family slots differ from frozen membership: {family_id}",
            code="FAMILY_DERIVATION_MISMATCH",
        )
    for slot_id, slot in slots.items():
        _require(
            slot["family_id"] in families,
            f"slot family is unregistered: {slot_id}",
            code="FAMILY_UNREGISTERED",
        )
        family = families[slot["family_id"]]
        _require(
            slot["candidate_id"] == family["candidate_id"],
            f"slot candidate differs from family: {slot_id}",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        baseline = baseline_by_id.get(slot["baseline_id"])
        _require(
            baseline is not None
            and baseline["comparison_output"] == slot["output_id"],
            f"slot baseline/output identity mismatch: {slot_id}",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        _require(
            baseline["status"] == "EXECUTABLE_PREBOUND"
            or slot["status"] == "TYPED_UNRESOLVED",
            f"unavailable baseline cannot be forged into a resolved slot: {slot_id}",
            code="CANDIDATE_SLOT_UNRESOLVED",
        )
        identity_fields = set(expected_slots[slot_id])
        _require(
            {
                field: slot[field]
                for field in identity_fields
            }
            == expected_slots[slot_id],
            f"slot identity differs from frozen matrix: {slot_id}",
            code="FAMILY_DERIVATION_MISMATCH",
        )
        if slot["status"] == "TYPED_UNRESOLVED":
            _require(
                slot.get("unavailable")
                == {
                    "reason_code": "G0_103_EXECUTABLE_ADAPTER_UNRESOLVED",
                    "claim_effect": "BLOCK_RELEVANT_PROMOTION",
                },
                f"typed-unresolved slot does not block: {slot_id}",
                code="CANDIDATE_SLOT_UNRESOLVED",
            )
        else:
            _require(
                slot["status"] == "RESOLVED"
                and _valid_full_forecast_pair_authority_record(
                    slot.get("execution_authority")
                ),
                f"resolved slot lacks authority-side full forecast receipts: {slot_id}",
                code="CANDIDATE_SLOT_UNRESOLVED",
            )
        if held_out_patches is None and slot["stratum_id"] == "patch:each-held-out-major-minor":
            _require(
                slot["status"] == "TYPED_UNRESOLVED",
                "unexpanded patch template must remain blocking",
                code="CANDIDATE_SLOT_UNRESOLVED",
            )


def validate_pair_registry(
    registry: Mapping[str, Any],
    candidate_registry: Mapping[str, Any],
) -> None:
    _require(
        registry.get("schema_version") == "g0-pair-registry-v1.4"
        and registry.get("registry_id") == "scryglass:pair-registry:v1.4",
        "pair registry identity is wrong",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    _require(
        registry.get("final_labels_read") is False
        and _is_digest(registry.get("contract_set_sha256")),
        "pair registry authority identity is malformed",
        code="CANDIDATE_REGISTRY_MISMATCH",
    )
    _require(
        registry.get("g1_unified_authority_handoff")
        == _G1_PAIR_AUTHORITY_HANDOFF,
        "pair authority handoff differs from the frozen G1 boundary",
        code="G1_UNIFIED_AUTHORITY_BUNDLE_REQUIRED",
    )
    candidate_ids = {
        item["candidate_id"] for item in candidate_registry["candidates"]
    }
    families = _registry_index(
        registry.get("families"),
        "pair_family_id",
        duplicate_code="CANDIDATE_SLOT_DUPLICATE",
    )
    pairs = _registry_index(
        registry.get("pairs"),
        "pair_id",
        duplicate_code="CANDIDATE_SLOT_DUPLICATE",
    )
    if any(pair.get("status") != "TYPED_UNRESOLVED" for pair in pairs.values()):
        _reject_source_bound_transition("resolved pair registry")
    expected_families = {
        "complexity:global": {
            "pair_family_id": "complexity:global",
            "kind": "SIMULTANEOUS_COMPLEXITY",
            "pair_ids": sorted(
                f"pair:{output}:scryglass-v2__minus__simplest-parent"
                for output in _OUTPUT_IDS
            ),
        }
    }
    expected_pairs = {
        f"pair:{output}:scryglass-v2__minus__simplest-parent": {
            "pair_id": (
                f"pair:{output}:scryglass-v2__minus__simplest-parent"
            ),
            "pair_family_id": "complexity:global",
            "candidate_a_id": "candidate:scryglass-v2",
            "candidate_b_id": (
                "candidate:scryglass-v2-simplest-parent"
            ),
            "output_id": output,
            "orientation": "candidate_a_minus_candidate_b",
            "bootstrap_plan_sha256": _registered_pair_wcr_plan_sha256(),
        }
        for output in _OUTPUT_IDS
    }
    _require(
        set(families) == set(expected_families)
        and set(pairs) == set(expected_pairs),
        "pair family or pair inventory is incomplete or expanded",
        code="FAMILY_DERIVATION_MISMATCH",
    )
    for family_id, family in families.items():
        declared = family["pair_ids"]
        actual = sorted(
            pair_id
            for pair_id, pair in pairs.items()
            if pair["pair_family_id"] == family_id
        )
        _require(
            family == expected_families[family_id]
            and declared == sorted(declared)
            and len(declared) == len(set(declared))
            and set(declared) == set(actual),
            f"pair family membership is not exact: {family_id}",
            code="FAMILY_DERIVATION_MISMATCH",
        )
    for pair_id, pair in pairs.items():
        _require(
            pair["pair_family_id"] in families,
            f"pair family is unregistered: {pair_id}",
            code="FAMILY_UNREGISTERED",
        )
        _require(
            pair["candidate_a_id"] in candidate_ids
            and pair["candidate_b_id"] in candidate_ids
            and pair["candidate_a_id"] != pair["candidate_b_id"],
            f"pair candidate identity is invalid: {pair_id}",
            code="PAIR_IDENTITY_MISMATCH",
        )
        _require(
            pair["orientation"] == "candidate_a_minus_candidate_b",
            f"pair orientation is wrong: {pair_id}",
            code="PAIR_IDENTITY_MISMATCH",
        )
        expected_identity = expected_pairs[pair_id]
        _require(
            {
                field: pair[field] for field in expected_identity
            }
            == expected_identity,
            f"pair frozen identity changed: {pair_id}",
            code="PAIR_IDENTITY_MISMATCH",
        )
        _require(
            pair.get("unavailable")
            == {
                "reason_code": "G0_103_PAIR_EVIDENCE_UNRESOLVED",
                "claim_effect": "BLOCK_COMPLEXITY_SELECTION",
            },
            f"typed-unresolved pair reason changed: {pair_id}",
            code="CANDIDATE_SLOT_UNRESOLVED",
        )


def validate_opening_permit_template(
    template: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
) -> None:
    if schema is not None:
        _schema_validate(template, schema, OPENING_PERMIT_TEMPLATE)
    _require(template == {
        "schema_version": "1.4",
        "status": "NOT_REQUESTED",
        "decision_scope": "G9_FINAL_OPENING_ONLY",
        "authorizing": False,
    }, "blank permit template must be exact and nonauthorizing", code="PERMIT_TEMPLATE_NONAUTHORIZING")


def _expected_exact_once_key(
    *,
    candidate_sha256: str,
    partition_sha256: str,
    preflight_sha256: str,
    contract_set_sha256: str,
) -> str:
    return stable_digest(
        {
            "candidate_sha256": candidate_sha256,
            "contract_set_sha256": contract_set_sha256,
            "partition_sha256": partition_sha256,
            "preflight_sha256": preflight_sha256,
        }
    )


_PERMIT_THREAD_LOCKS_GUARD = threading.Lock()
_PERMIT_THREAD_LOCKS: dict[str, threading.Lock] = {}


class _AtomicFileExactOnceStore:
    """Durable, locked single-state transaction indexed by key and permit ID."""

    def __init__(self, root: Path) -> None:
        descriptor = _open_root_directory(Path(root), purpose="exact-once store")
        os.close(descriptor)
        self._root = Path(root)
        key = os.path.abspath(os.fspath(root))
        with _PERMIT_THREAD_LOCKS_GUARD:
            self._thread_lock = _PERMIT_THREAD_LOCKS.setdefault(
                key,
                threading.Lock(),
            )

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "schema_version": "g0-ledger-v1.4",
            "receipts_by_exact_once_key": {},
            "exact_once_key_by_permit_id": {},
        }

    @staticmethod
    def _validate_state(state: object) -> dict[str, Any]:
        if (
            not isinstance(state, dict)
            or set(state)
            != {
                "schema_version",
                "receipts_by_exact_once_key",
                "exact_once_key_by_permit_id",
            }
            or state.get("schema_version") != "g0-ledger-v1.4"
            or not isinstance(state.get("receipts_by_exact_once_key"), dict)
            or not isinstance(state.get("exact_once_key_by_permit_id"), dict)
        ):
            raise BenchmarkContractError(
                "permit store state schema is corrupt",
                code="PERMIT_RECEIPT_CORRUPT",
            )
        receipts = state["receipts_by_exact_once_key"]
        index = state["exact_once_key_by_permit_id"]
        for key, receipt in receipts.items():
            expected_fields = {
                "exact_once_key",
                "permit_id",
                "reviewer_identity",
                "candidate_sha256",
                "partition_sha256",
                "preflight_sha256",
                "contract_set_sha256",
                "approved_permit_raw_sha256",
            }
            if (
                not _is_digest(key)
                or not isinstance(receipt, dict)
                or set(receipt) != expected_fields
                or receipt.get("exact_once_key") != key
                or not all(
                    _is_digest(receipt[field])
                    for field in (
                        "candidate_sha256",
                        "partition_sha256",
                        "preflight_sha256",
                        "contract_set_sha256",
                        "approved_permit_raw_sha256",
                    )
                )
                or not isinstance(receipt.get("permit_id"), str)
                or not receipt["permit_id"].startswith("permit:g9:")
                or not isinstance(receipt.get("reviewer_identity"), str)
                or _REVIEWER_IDENTITY_PATTERN.fullmatch(
                    receipt["reviewer_identity"]
                )
                is None
                or index.get(receipt.get("permit_id")) != key
            ):
                raise BenchmarkContractError(
                    "permit store receipt or permit-id index is corrupt",
                    code="PERMIT_RECEIPT_CORRUPT",
                )
        if len(index) != len(receipts) or any(
            permit_id
            not in {
                receipt["permit_id"] for receipt in receipts.values()
            }
            or key not in receipts
            for permit_id, key in index.items()
        ):
            raise BenchmarkContractError(
                "permit store permit-id index is corrupt",
                code="PERMIT_RECEIPT_CORRUPT",
            )
        return state

    @staticmethod
    def _secure_read_at(directory_descriptor: int, name: str) -> bytes | None:
        try:
            before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise BenchmarkContractError(
                f"permit store entry is not a single-link regular file: {name}",
                code="PERMIT_RECEIPT_CORRUPT",
            )
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
        except OSError as exc:
            raise BenchmarkContractError(
                f"permit store entry cannot be securely opened: {name}",
                code="PERMIT_RECEIPT_CORRUPT",
            ) from exc
        try:
            after = os.fstat(descriptor)
            if (
                not stat.S_ISREG(after.st_mode)
                or after.st_nlink != 1
                or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            ):
                raise BenchmarkContractError(
                    f"permit store entry identity changed: {name}",
                    code="PERMIT_RECEIPT_CORRUPT",
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    @staticmethod
    def _open_lock(directory_descriptor: int) -> int:
        name = ".permit-store.lock"
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=directory_descriptor)
        except OSError as exc:
            raise BenchmarkContractError(
                "permit store lock cannot be opened",
                code="PERMIT_RECEIPT_CORRUPT",
            ) from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise BenchmarkContractError(
                "permit store lock is not a single-link regular file",
                code="PERMIT_RECEIPT_CORRUPT",
            )
        return descriptor

    def consume(self, exact_once_key: str, receipt: Mapping[str, Any]) -> None:
        with self._thread_lock:
            self._consume_locked(exact_once_key, receipt)

    def _consume_locked(
        self,
        exact_once_key: str,
        receipt: Mapping[str, Any],
    ) -> None:
        _require(_is_digest(exact_once_key), "exact-once key is malformed")
        _require(
            receipt.get("exact_once_key") == exact_once_key,
            "exact-once receipt key mismatch",
        )
        root_descriptor = _open_root_directory(
            self._root,
            purpose="exact-once store",
        )
        lock_descriptor = self._open_lock(root_descriptor)
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            raw_state = self._secure_read_at(root_descriptor, "state.json")
            if raw_state is None:
                state = self._empty_state()
            else:
                try:
                    parsed = json.loads(
                        raw_state.decode("utf-8"),
                        object_pairs_hook=_no_duplicate_pairs,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, BenchmarkContractError) as exc:
                    raise BenchmarkContractError(
                        "permit store state is corrupt JSON",
                        code="PERMIT_RECEIPT_CORRUPT",
                    ) from exc
                state = self._validate_state(parsed)
            receipts = state["receipts_by_exact_once_key"]
            permit_index = state["exact_once_key_by_permit_id"]
            if exact_once_key in receipts:
                self._validate_state(state)
                raise BenchmarkContractError(
                    "opening permit exact-once key already consumed",
                    code="PERMIT_ALREADY_CONSUMED",
                )
            permit_id = receipt.get("permit_id")
            if permit_id in permit_index:
                raise BenchmarkContractError(
                    "opening permit ID already consumed under another key",
                    code="PERMIT_ALREADY_CONSUMED",
                )
            next_state = {
                "schema_version": "g0-ledger-v1.4",
                "receipts_by_exact_once_key": {
                    **receipts,
                    exact_once_key: dict(receipt),
                },
                "exact_once_key_by_permit_id": {
                    **permit_index,
                    str(permit_id): exact_once_key,
                },
            }
            self._validate_state(next_state)
            payload = (canonical_json(next_state) + "\n").encode("utf-8")
            stage = (
                f".state.stage.{os.getpid()}.{secrets.token_hex(16)}"
            )
            stage_descriptor: int | None = None
            renamed = False
            try:
                stage_descriptor = os.open(
                    stage,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=root_descriptor,
                )
                offset = 0
                while offset < len(payload):
                    written = os.write(stage_descriptor, payload[offset:])
                    _require(written > 0, "permit store stage write made no progress")
                    offset += written
                os.fsync(stage_descriptor)
                os.close(stage_descriptor)
                stage_descriptor = None
                os.rename(
                    stage,
                    "state.json",
                    src_dir_fd=root_descriptor,
                    dst_dir_fd=root_descriptor,
                )
                renamed = True
                try:
                    os.fsync(root_descriptor)
                except OSError as exc:
                    raise BenchmarkContractError(
                        "permit receipt linked but directory durability is unknown",
                        code="PERMIT_DURABILITY_UNKNOWN",
                    ) from exc
            finally:
                if stage_descriptor is not None:
                    os.close(stage_descriptor)
                if not renamed:
                    try:
                        os.unlink(stage, dir_fd=root_descriptor)
                    except FileNotFoundError:
                        pass
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
            os.close(root_descriptor)


@dataclass(frozen=True)
class OpeningReceipt:
    exact_once_key: str
    permit_id: str
    reviewer_identity: str
    candidate_sha256: str
    partition_sha256: str
    preflight_sha256: str
    contract_set_sha256: str
    approved_permit_raw_sha256: str


def _validate_reviewer_identity(identity: object) -> str:
    _require(
        isinstance(identity, str)
        and bool(identity)
        and identity == identity.strip()
        and _REVIEWER_IDENTITY_PATTERN.fullmatch(identity) is not None,
        "opening permit reviewer identity is malformed",
        code="PERMIT_REVIEWER_IDENTITY_INVALID",
    )
    return identity


def _consume_bound_opening_permit_at(
    context: _AuthorityContext,
    permit_raw: bytes,
) -> OpeningReceipt:
    _reject_source_bound_transition(
        "G9 opening permit consumption",
        context.verified.preflight_sha256,
    )
    _require(
        isinstance(permit_raw, bytes),
        "approved opening permit must be exact bytes",
        code="PERMIT_BOUND_IDENTITY_MISMATCH",
    )
    permit = _parse_json_bytes(permit_raw, label="approved opening permit")
    reviewer = _validate_reviewer_identity(permit.get("reviewer_identity"))
    schema_raw = _read_regular_under_root(
        context.package_root,
        OPENING_PERMIT_SCHEMA,
        purpose="opening permit schema",
    )
    schema = _parse_json_bytes(schema_raw, label=OPENING_PERMIT_SCHEMA)
    _schema_validate(permit, schema, "approved opening permit")
    _require(
        permit.get("status") == "APPROVED"
        and permit.get("authorizing") is True
        and permit.get("decision_scope") == "G9_FINAL_OPENING_ONLY",
        "opening permit is nonauthorizing",
        code="PERMIT_TEMPLATE_NONAUTHORIZING",
    )
    _require(
        reviewer != context.verified.evidence_generator_identity,
        "reviewer identity equals the authority-derived evidence generator",
        code="PERMIT_REVIEWER_NOT_INDEPENDENT",
    )
    approved_at = permit.get("approved_at_utc")
    try:
        parsed_time = datetime.fromisoformat(
            str(approved_at).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise BenchmarkContractError(
            "opening permit approval time is invalid",
            code="PERMIT_BOUND_IDENTITY_MISMATCH",
        ) from exc
    _require(
        parsed_time.tzinfo is not None
        and parsed_time.utcoffset() == timezone.utc.utcoffset(parsed_time),
        "opening permit approval time must be UTC",
        code="PERMIT_BOUND_IDENTITY_MISMATCH",
    )

    report = _compute_registered_pairwise_intervals_at(
        context,
        "complexity:global",
        attempts=_WCR_REPLICATES,
    )
    _require(
        report.selection_status in {"DIRECTIONAL_WINNER", "SIMPLICITY_WINNER"}
        and report.winner_candidate_id is not None,
        "global complexity report has no authoritative opening winner",
        code="NO_WINNER_REMAND",
    )
    candidate_id = report.winner_candidate_id
    candidate = _candidate_record(context, candidate_id)
    _require(
        _candidate_gate_status_at(
            context,
            candidate_id,
            attempts=_WCR_REPLICATES,
        )
        == "PASS",
        "complete registered primary, harm, and secondary families do not authorize opening",
        code="PERMIT_BOUND_IDENTITY_MISMATCH",
    )
    candidate_sha256 = candidate["candidate_artifact_raw_sha256"]
    partition_sha256 = candidate["partition_bindings_raw_sha256"]
    preflight_sha256 = context.verified.preflight_sha256
    contract_set_sha256 = context.verified.manifest_contract_set_sha256
    _require(
        all(
            _is_digest(value)
            for value in (
                candidate_sha256,
                partition_sha256,
                preflight_sha256,
                contract_set_sha256,
            )
        ),
        "authority-derived opening identities are incomplete",
        code="PERMIT_BOUND_IDENTITY_MISMATCH",
    )
    once = _expected_exact_once_key(
        candidate_sha256=candidate_sha256,
        partition_sha256=partition_sha256,
        preflight_sha256=preflight_sha256,
        contract_set_sha256=contract_set_sha256,
    )
    receipt = OpeningReceipt(
        exact_once_key=once,
        permit_id=permit["permit_id"],
        reviewer_identity=reviewer,
        candidate_sha256=candidate_sha256,
        partition_sha256=partition_sha256,
        preflight_sha256=preflight_sha256,
        contract_set_sha256=contract_set_sha256,
        approved_permit_raw_sha256=raw_digest(permit_raw),
    )
    ledger_relative = _canonical_relative_locator(
        context.verified.permit_ledger_relative_path,
        purpose="bound permit ledger",
    )
    ledger_path = context.package_root.joinpath(
        *PurePosixPath(ledger_relative).parts
    )
    try:
        store = _AtomicFileExactOnceStore(ledger_path)
    except BenchmarkContractError as exc:
        raise BenchmarkContractError(
            "authority-bound permit ledger is unavailable or redirected",
            code="PERMIT_LEDGER_BINDING_MISMATCH",
        ) from exc
    store.consume(once, receipt.__dict__)
    return receipt


def consume_bound_opening_permit(permit_raw: bytes) -> OpeningReceipt:
    """Consume a permit using only fixed, internally re-derived authority."""

    context = _verify_authoritative_preflight_at(
        _PRODUCTION_PACKAGE_ROOT,
        _REPO_ROOT,
    )
    return _consume_bound_opening_permit_at(context, permit_raw)


def build_contract_manifest_payload(
    root: Path = REAL_V1_ROOT,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Deterministically compute the noncyclic core and full v1.4 manifest."""

    if repo_root is None:
        repo_root = _REPO_ROOT
    core_names = (
        BENCHMARK_SCHEMA,
        BASELINE_SCHEMA,
        OPENING_PERMIT_SCHEMA,
        AUTHORITY_BUNDLE_SCHEMA,
        CANDIDATE_REGISTRY_SCHEMA,
        PAIR_REGISTRY_SCHEMA,
        BENCHMARK_CONTRACT,
        BASELINE_REGISTRY,
        OPENING_PERMIT_TEMPLATE,
    )
    names = core_names + (
        AUTHORITY_BUNDLE,
        CANDIDATE_REGISTRY,
        PAIR_REGISTRY,
    )
    files: dict[str, dict[str, str]] = {}
    for name in names:
        raw = _read_regular_under_root(
            Path(root),
            name,
            purpose=f"manifest input {name}",
        )
        parsed = _parse_json_bytes(raw, label=name)
        files[name] = {
            "raw_sha256": raw_digest(raw),
            "semantic_sha256": stable_digest(parsed),
        }
    locator = "lol_kills/v2/evaluation/benchmark_contract.py"
    code_raw = _read_regular_under_root(
        Path(repo_root),
        locator,
        purpose=f"manifest authority code {locator}",
    )
    authority_code = {locator: {"raw_sha256": raw_digest(code_raw)}}
    core_files = {name: files[name] for name in core_names}
    authority_contract_set_sha256 = stable_digest(
        {"json_files": core_files, "authority_code": authority_code}
    )
    contract_set_sha256 = stable_digest(
        {"json_files": files, "authority_code": authority_code}
    )
    return {
        "manifest_id": "scryglass:real-benchmark-manifest:v1.4",
        "status": "FROZEN_CANDIDATE_PENDING_INDEPENDENT_REVIEW",
        "digest_intent": {
            "raw_sha256": "exact UTF-8 file bytes",
            "semantic_sha256": "sorted compact JSON semantics",
        },
        "files": files,
        "authority_code": authority_code,
        "authority_contract_set_sha256": authority_contract_set_sha256,
        "contract_set_sha256": contract_set_sha256,
    }


def validate_real_v1(
    root: Path = REAL_V1_ROOT,
    *,
    repo_root: Path | None = None,
) -> dict[str, dict[str, str]]:
    root = Path(root)
    if repo_root is None:
        repo_root = _REPO_ROOT
    core_names = (
        BENCHMARK_SCHEMA,
        BASELINE_SCHEMA,
        OPENING_PERMIT_SCHEMA,
        AUTHORITY_BUNDLE_SCHEMA,
        CANDIDATE_REGISTRY_SCHEMA,
        PAIR_REGISTRY_SCHEMA,
        BENCHMARK_CONTRACT,
        BASELINE_REGISTRY,
        OPENING_PERMIT_TEMPLATE,
    )
    names = core_names + (
        AUTHORITY_BUNDLE,
        CANDIDATE_REGISTRY,
        PAIR_REGISTRY,
    )
    raw_files = {
        name: _read_regular_under_root(
            root,
            name,
            purpose=f"real-v1 package {name}",
        )
        for name in names
    }
    files = {name: _parse_json_bytes(raw, label=name) for name, raw in raw_files.items()}
    manifest = _parse_json_bytes(
        _read_regular_under_root(
            root,
            CONTRACT_MANIFEST,
            purpose="real-v1 manifest",
        ),
        label=CONTRACT_MANIFEST,
    )

    _schema_validate(files[BENCHMARK_CONTRACT], files[BENCHMARK_SCHEMA], BENCHMARK_CONTRACT)
    _schema_validate(files[BASELINE_REGISTRY], files[BASELINE_SCHEMA], BASELINE_REGISTRY)
    _schema_validate(files[OPENING_PERMIT_TEMPLATE], files[OPENING_PERMIT_SCHEMA], OPENING_PERMIT_TEMPLATE)
    _schema_validate(files[AUTHORITY_BUNDLE], files[AUTHORITY_BUNDLE_SCHEMA], AUTHORITY_BUNDLE)
    _schema_validate(files[CANDIDATE_REGISTRY], files[CANDIDATE_REGISTRY_SCHEMA], CANDIDATE_REGISTRY)
    _schema_validate(files[PAIR_REGISTRY], files[PAIR_REGISTRY_SCHEMA], PAIR_REGISTRY)
    validate_benchmark_contract(files[BENCHMARK_CONTRACT])
    validate_baseline_registry(files[BASELINE_REGISTRY], repo_root=repo_root)
    validate_opening_permit_template(files[OPENING_PERMIT_TEMPLATE], files[OPENING_PERMIT_SCHEMA])
    validate_candidate_registry(
        files[CANDIDATE_REGISTRY],
        files[BASELINE_REGISTRY],
        repo_root=Path(repo_root),
    )
    validate_pair_registry(
        files[PAIR_REGISTRY],
        files[CANDIDATE_REGISTRY],
    )

    _require(manifest.get("manifest_id") == "scryglass:real-benchmark-manifest:v1.4", "manifest id is wrong")
    _require(manifest.get("status") == "FROZEN_CANDIDATE_PENDING_INDEPENDENT_REVIEW", "manifest must not self-approve")
    _require(
        manifest.get("digest_intent")
        == {
            "raw_sha256": "exact UTF-8 file bytes",
            "semantic_sha256": "sorted compact JSON semantics",
        },
        "manifest digest intent is ambiguous",
    )
    listed = manifest.get("files")
    _require(isinstance(listed, dict) and set(listed) == set(names), "manifest file inventory is incomplete or changed")
    digests: dict[str, dict[str, str]] = {}
    for name in names:
        record = listed[name]
        _require(isinstance(record, dict), f"manifest record malformed for {name}")
        _require(set(record) == {"raw_sha256", "semantic_sha256"}, f"manifest digest fields wrong for {name}")
        actual = {
            "raw_sha256": raw_digest(raw_files[name]),
            "semantic_sha256": stable_digest(files[name]),
        }
        _require(record == actual, f"manifest digest mismatch for {name}")
        digests[name] = actual
    authority_code = manifest.get("authority_code")
    expected_code_locators = {
        "lol_kills/v2/evaluation/benchmark_contract.py",
    }
    _require(
        isinstance(authority_code, dict)
        and set(authority_code) == expected_code_locators,
        "manifest authority-code inventory is incomplete or changed",
        code="MANIFEST_AUTHORITY_CODE_MISSING",
    )
    actual_code: dict[str, dict[str, str]] = {}
    for locator in sorted(expected_code_locators):
        record = authority_code[locator]
        _require(
            isinstance(record, dict)
            and set(record) == {"raw_sha256"},
            f"manifest authority-code record malformed: {locator}",
            code="MANIFEST_AUTHORITY_CODE_MISSING",
        )
        code_raw = _read_regular_under_root(
            Path(repo_root),
            locator,
            purpose=f"authority code {locator}",
        )
        _require(
            record["raw_sha256"] == raw_digest(code_raw),
            f"authority code raw digest mismatch: {locator}",
            code="MANIFEST_AUTHORITY_CODE_MISMATCH",
        )
        actual_code[locator] = {"raw_sha256": record["raw_sha256"]}
    _require(
        _is_digest(manifest.get("authority_contract_set_sha256")),
        "manifest authority_contract_set_sha256 malformed",
    )
    core_digests = {name: digests[name] for name in core_names}
    _require(
        manifest["authority_contract_set_sha256"]
        == stable_digest(
            {
                "json_files": core_digests,
                "authority_code": actual_code,
            }
        ),
        "manifest authority_contract_set_sha256 mismatch",
    )
    _require(_is_digest(manifest.get("contract_set_sha256")), "manifest contract_set_sha256 malformed")
    _require(
        manifest["contract_set_sha256"]
        == stable_digest({"json_files": digests, "authority_code": actual_code}),
        "manifest contract_set_sha256 mismatch",
    )
    core_identity = manifest["authority_contract_set_sha256"]
    _require(
        files[AUTHORITY_BUNDLE]["contract_set_sha256"] == core_identity
        and files[CANDIDATE_REGISTRY]["contract_set_sha256"] == core_identity
        and files[PAIR_REGISTRY]["contract_set_sha256"] == core_identity,
        "authority artifacts do not bind the manifest core contract set",
        code="AUTHORITY_BUNDLE_MISMATCH",
    )
    return digests


__all__ = [
    "AUTHORITY_BUNDLE",
    "AUTHORITY_BUNDLE_SCHEMA",
    "BASELINE_REGISTRY",
    "BENCHMARK_CONTRACT",
    "BenchmarkContractError",
    "CANDIDATE_REGISTRY",
    "CANDIDATE_REGISTRY_SCHEMA",
    "CONTRACT_MANIFEST",
    "HolmDecision",
    "HolmReport",
    "OPENING_PERMIT_SCHEMA",
    "OPENING_PERMIT_TEMPLATE",
    "OpeningReceipt",
    "PAIR_REGISTRY",
    "PAIR_REGISTRY_SCHEMA",
    "PairwiseInterval",
    "PairwiseIntervalReport",
    "REAL_V1_ROOT",
    "VerifiedAuthority",
    "WCRBootclusterResult",
    "WCRConsensusReport",
    "WCRSuiteResult",
    "build_contract_manifest_payload",
    "canonical_intersection_id",
    "canonical_json",
    "cgm_subset_sign",
    "compute_registered_holm",
    "compute_registered_pairwise_intervals",
    "consume_bound_opening_permit",
    "derive_registered_margin",
    "effective_cluster_count",
    "inference_support_status",
    "load_canonical_json",
    "macro_regional_series_weights",
    "margin_replicate_payload_sha256",
    "raw_digest",
    "select_largest_cluster",
    "stable_digest",
    "type7_quantile",
    "validate_baseline_registry",
    "validate_benchmark_contract",
    "validate_candidate_registry",
    "validate_no_final_labels",
    "validate_opening_permit_template",
    "validate_pair_registry",
    "validate_partition_identity",
    "validate_real_v1",
    "verify_authoritative_preflight",
]
