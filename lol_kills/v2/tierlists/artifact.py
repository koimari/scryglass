"""L9 tier-list artifact builder: one league x current-patch x role cell.

Development-only and fail-closed.  The builder consumes:

- the frozen L7 development terminal artifact (TerminalModel, exact bytes and
  sha256 bound by lol_kills.v2.draft.terminal.model.TerminalModel);
- the frozen L3 champion-id crosswalk vocabulary;
- verified L1-style appearances (OE player-games table);
- an optional CompetitionTaxonomy for the tier profile cross-check.

The artifact carries as_of, patch_id, league_id / international event,
eligibility status, lineage hashes, and fail-closed status.  Unavailable
components serialize as null (never zero).  rank_eligibility stays false.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lol_kills.v2.data.common import ROLES, parse_rfc3339, sha256_bytes, to_rfc3339
from lol_kills.v2.data.competitions import CompetitionTaxonomy
from lol_kills.v2.draft.terminal.model import TerminalDraftError, TerminalModel

from .appearances import AppearanceScope, AppearanceTable, CellAppearances
from .model import (
    APPEARANCE_SOURCE,
    ARTIFACT_KIND,
    _FORBIDDEN_RAW_WIN_RATE_KEYS,
    CLAIM_CEILING,
    COUNTERABILITY_TAIL_ALPHA,
    COUNTERABILITY_WEIGHT_LAMBDA_C,
    COUNTERABILITY_WEIGHT_SELECTION,
    CROSSWALK_ARTIFACT,
    HASH_RE,
    PATCH_RE,
    REFERENCE_MIXTURE_RULE,
    REGIONS,
    SCHEMA_VERSION,
    SOURCE_TREE_ALLOWLIST,
    TERMINAL_MODEL_ARTIFACT,
    TierListError,
    TierListIntegrityError,
    calibrated_probability,
    load_crosswalk_vocabulary,
    reference_mixture_logit,
    response_regret,
    standardized_replacement_probability_points,
)

DEFAULT_ARTIFACT_PATH = Path("data/lol/v2/tierlists/tierlist-lec-16.14-mid-development-v1.json")

_Z95 = 1.959963984540054


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_object(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_frozen_terminal_model(root: Path) -> TerminalModel:
    """Load the frozen development terminal artifact, binding its exact bytes."""
    locator = TERMINAL_MODEL_ARTIFACT["locator"]
    path = root / locator
    if not path.is_file() or path.is_symlink():
        raise TierListError("frozen terminal model artifact is missing or not a regular file")
    raw = path.read_bytes()
    if sha256_bytes(raw) != TERMINAL_MODEL_ARTIFACT["raw_sha256"]:
        raise TierListError("frozen terminal model artifact bytes do not match the pinned sha256")
    try:
        model = TerminalModel.from_artifact_bytes(raw, expected_artifact_sha256=TERMINAL_MODEL_ARTIFACT["raw_sha256"])
    except TerminalDraftError as exc:
        raise TierListError(f"frozen terminal model artifact is not admissible: {exc}") from exc
    if model.model_version != TERMINAL_MODEL_ARTIFACT["model_version"]:
        raise TierListError("frozen terminal model version mismatch")
    return model


def _resolve_champion_id(crosswalk: Mapping[str, str], champion_name: str) -> str:
    stable = crosswalk.get(champion_name)
    if stable is None:
        # tolerate case-only differences through the OE vocabulary key
        lowered = {key.lower(): value for key, value in crosswalk.items()}
        stable = lowered.get(champion_name.lower())
    if stable is None:
        raise TierListError(f"played champion has no resolved identity: {champion_name!r}")
    return stable


def _unavailable(
    *,
    scope: AppearanceScope,
    as_of: str,
    reason: str,
    error_code: str,
    missing_components: Sequence[str] = (),
    message: str = "Tier list is unavailable for this cell or input state.",
    retryable: bool = False,
    lineage: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    created = created_at or to_rfc3339(datetime.now(timezone.utc))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "artifact_id": f"scryglass:tierlist:{scope.scope_id.lower()}:{scope.patch_id}:{scope.role}:unavailable",
        "status": "unavailable",
        "fail_closed_status": "unavailable",
        "development_only": True,
        "rank_eligibility": False,
        "publication_eligible": False,
        "claim_ceiling": dict(CLAIM_CEILING),
        "scope": scope.as_mapping(),
        "role": scope.role,
        "patch_id": scope.patch_id,
        "as_of": to_rfc3339(parse_rfc3339(as_of)),
        "created_at": created,
        "membership": [],
        "rows": [],
        "reference_convention": None,
        "estimand": None,
        "lineage": dict(lineage) if lineage else None,
        "provenance": {
            "schema_version": "2.0.0",
            "mode": "state_snapshot",
            "model_version": TERMINAL_MODEL_ARTIFACT["model_version"],
            "as_of": to_rfc3339(parse_rfc3339(as_of)),
            "created_at": created,
            "required_input_status": "missing" if error_code != "conflict" else "conflict",
            "freshness_checks": [],
            "input_conflicts": [],
            "missing_components": list(missing_components),
            "closed_components": [],
            "out_of_distribution_flags": [],
            "development_only": True,
        },
        "literal_interpretation": "Tier list is unavailable for this cell; no numeric payload is emitted.",
        "error": {
            "code": error_code,
            "reason": reason,
            "message": message,
            "retryable": retryable,
            "missing_fields": list(missing_components),
        },
    }
    payload["artifact_sha256"] = sha256_object({k: v for k, v in payload.items() if k != "artifact_sha256"})
    return payload


def build_tier_list_artifact(
    *,
    scope: AppearanceScope,
    as_of: str,
    terminal_model: TerminalModel,
    crosswalk: Mapping[str, str],
    appearances: CellAppearances,
    appearance_source_sha256: str,
    appearance_source_locator: str | None = None,
    created_at: str | None = None,
    source_tree_sha256: str | None = None,
    taxonomy: CompetitionTaxonomy | None = None,
) -> dict[str, Any]:
    """Build one development tier-list cell payload (numeric or fail-closed)."""

    if not isinstance(appearance_source_sha256, str) or not HASH_RE.fullmatch(appearance_source_sha256):
        raise TierListError("appearance_source_sha256 must be a lowercase sha256")
    appearance_source_locator = appearance_source_locator or APPEARANCE_SOURCE["locator"]
    if terminal_model.model_version != TERMINAL_MODEL_ARTIFACT["model_version"]:
        return _unavailable(
            scope=scope, as_of=as_of, reason="terminal_model_version_mismatch",
            error_code="model_not_promoted", retryable=False,
            missing_components=["terminal_model"],
        )
    as_of_utc = parse_rfc3339(as_of)
    created = created_at or to_rfc3339(datetime.now(timezone.utc))

    if taxonomy is not None and scope.scope_kind == "league" and scope.competition_tier in {"tier1", "tier2"}:
        try:
            resolved_tier, _rule = taxonomy.resolve_league_tier(scope.scope_id, as_of_utc)
        except Exception as exc:  # PatchConflictError etc. -> fail closed
            return _unavailable(
                scope=scope, as_of=as_of, reason=f"taxonomy_tier_conflict: {exc}",
                error_code="conflict", missing_components=["competition_taxonomy"],
            )
        if resolved_tier != scope.competition_tier:
            return _unavailable(
                scope=scope, as_of=as_of,
                reason=f"taxonomy_tier_conflict: requested {scope.competition_tier}, taxonomy says {resolved_tier}",
                error_code="conflict", missing_components=["competition_taxonomy"],
            )

    membership = appearances.membership()
    if not membership:
        return _unavailable(
            scope=scope, as_of=as_of, reason="no_played_champions_in_cell",
            error_code="missing_required_input", missing_components=["played_membership"],
        )

    role_coefficients = terminal_model.champion_role_logit
    key_prefix = f"{scope.role}|"

    unresolved = [name for name in membership if _safe_resolve(crosswalk, name) is None]
    if unresolved:
        return _unavailable(
            scope=scope, as_of=as_of, reason="played_champion_unresolved_identity",
            error_code="missing_required_input", missing_components=sorted(unresolved),
            message="Played champions could not be resolved to stable champion ids.",
        )
    no_coverage = [name for name in membership if f"{key_prefix}{name}" not in role_coefficients]
    if no_coverage:
        return _unavailable(
            scope=scope, as_of=as_of, reason="played_champion_missing_terminal_coverage",
            error_code="missing_required_input", missing_components=sorted(no_coverage),
            message="Played champions have no coefficient in the frozen terminal model; fail closed rather than drop rows.",
        )

    member_logits = {name: float(role_coefficients[f"{key_prefix}{name}"]) for name in membership}
    reference_logit = reference_mixture_logit(list(member_logits.values()))
    reference_probability = calibrated_probability(
        reference_logit,
        calibration_slope=terminal_model.calibration_slope,
        calibration_intercept=terminal_model.calibration_intercept,
    )
    slope = terminal_model.calibration_slope
    intercept = terminal_model.calibration_intercept
    sigma = float(terminal_model.uncertainty_logit_sd)

    rows: list[dict[str, Any]] = []
    for champion_name in sorted(membership):
        champion_id = _resolve_champion_id(crosswalk, champion_name)
        champion_logit = member_logits[champion_name]
        tier_value = standardized_replacement_probability_points(
            champion_logit, reference_logit, calibration_slope=slope, calibration_intercept=intercept
        )
        lower = standardized_replacement_probability_points(
            champion_logit - _Z95 * sigma, reference_logit, calibration_slope=slope, calibration_intercept=intercept
        )
        upper = standardized_replacement_probability_points(
            champion_logit + _Z95 * sigma, reference_logit, calibration_slope=slope, calibration_intercept=intercept
        )
        regret = response_regret(
            role=scope.role,
            champion=champion_name,
            champion_logit=champion_logit,
            reference_logit=reference_logit,
            member_logits=member_logits,
            counter_logit=terminal_model.counter_logit,
            ally_synergy_logit=terminal_model.ally_synergy_logit,
            tail_alpha=COUNTERABILITY_TAIL_ALPHA,
            calibration_slope=slope,
            calibration_intercept=intercept,
        )
        if regret is None:
            counterability = None
            counterability_status = "unavailable"
            counterability_reason = "no_response_specific_counter_logit_distribution"
            support_size = 0
        else:
            counterability = regret["regret"]
            counterability_status = "available"
            counterability_reason = None
            support_size = regret["support_size"]
        weighted = tier_value - COUNTERABILITY_WEIGHT_LAMBDA_C * (counterability or 0.0)
        rows.append(
            {
                "champion_id": champion_id,
                "champion_name": champion_name,
                "role": scope.role,
                "main_effect_logit": champion_logit,
                "reference_logit": reference_logit,
                "tier_value": tier_value,
                "weighted_tier_value": weighted,
                "tier_value_interval_95": {
                    "lower": lower,
                    "upper": upper,
                    "wording": "model_range_not_approved_exact_wording",
                },
                "counterability": counterability,
                "counterability_status": counterability_status,
                "counterability_reason": counterability_reason,
                "counterability_regret_support_size": support_size,
                "verified_appearance_count": int(membership[champion_name]["distinct_maps"]),
                "evidence": {
                    "distinct_maps": int(membership[champion_name]["distinct_maps"]),
                    "earliest_event_end": membership[champion_name]["earliest_event_end"],
                    "latest_event_end": membership[champion_name]["latest_event_end"],
                    "source": appearances.scope.scope_id,
                },
            }
        )

    any_counter_available = any(row["counterability_status"] == "available" for row in rows)
    fail_closed_status = "none" if any_counter_available else "counterability_unavailable"
    closed_components = [] if any_counter_available else ["counterability"]

    membership_payload = [
        {
            "champion_id": row["champion_id"],
            "champion_name": row["champion_name"],
            "role": row["role"],
            "verified_appearance_count": row["verified_appearance_count"],
        }
        for row in rows
    ]

    lineage = {
        "manifest_id": f"scryglass:manifest:{terminal_model.model_version}",
        "training_snapshot_id": "scryglass:training:terminal-development",
        "terminal_model": {
            "model_version": terminal_model.model_version,
            "model_as_of": terminal_model.model_as_of,
            "artifact_locator": TERMINAL_MODEL_ARTIFACT["locator"],
            "artifact_sha256": terminal_model.artifact_sha256,
            "candidate_id": TERMINAL_MODEL_ARTIFACT["candidate_id"],
        },
        "champion_crosswalk": {
            "artifact_locator": CROSSWALK_ARTIFACT["locator"],
            "artifact_sha256": CROSSWALK_ARTIFACT["artifact_sha256"],
            "l1_source_replay": "unavailable_warehouse_snapshot_changed",
        },
        "appearance_source": {
            "locator": appearance_source_locator,
            "raw_sha256": appearance_source_sha256,
            "source_kind": APPEARANCE_SOURCE["source_kind"],
            "availability_approximation": APPEARANCE_SOURCE["availability_approximation"],
        },
        "source_tree_sha256": source_tree_sha256,
        "code_commit": None,
    }
    window = appearances.window()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "artifact_id": f"scryglass:tierlist:{scope.scope_id.lower()}:{scope.patch_id}:{scope.role}:development-v1",
        "status": "development_only",
        "fail_closed_status": fail_closed_status,
        "development_only": True,
        "rank_eligibility": False,
        "publication_eligible": False,
        "claim_ceiling": dict(CLAIM_CEILING),
        "scope": scope.as_mapping(),
        "role": scope.role,
        "patch_id": scope.patch_id,
        "as_of": to_rfc3339(as_of_utc),
        "created_at": created,
        "membership": membership_payload,
        "rows": rows,
        "reference_convention": {
            "role_reference_mixture": REFERENCE_MIXTURE_RULE,
            "reference_mixture_size": len(rows),
            "reference_logit": reference_logit,
            "reference_probability": reference_probability,
            "context_distribution": "degenerate_singleton_empty_context",
            "response_policy": "uniform_over_role_counter_support_dev_convention",
            "counterability_tail_alpha": COUNTERABILITY_TAIL_ALPHA,
            "counterability_weight_lambda_c": COUNTERABILITY_WEIGHT_LAMBDA_C,
            "counterability_weight_selection": COUNTERABILITY_WEIGHT_SELECTION,
        },
        "estimand": {
            "estimand_id": "tier_value_incremental_model_standardized_probability_points",
            "formula": "TV = IV - lambda_C * C; IV = E_{z~G,a~R_ref}[100*(p_D(c,r,z,a) - p_D(c_ref,r,z,a))]; dev artifact: p_D(c,r) = sigmoid(calibration_slope * champion_role_logit[role|c]) with the degenerate empty context",
            "tier_value_units": "model-standardized probability points (pp)",
            "raw_win_rate_target": False,
            "causal_claim": False,
            "pick_order_claim": False,
            "outcome_calibrated_probability": False,
        },
        "lineage": lineage,
        "provenance": {
            "schema_version": "2.0.0",
            "mode": "state_snapshot",
            "model_version": terminal_model.model_version,
            "as_of": to_rfc3339(as_of_utc),
            "created_at": created,
            "required_input_status": "complete",
            "freshness_checks": [
                {"check": "terminal_model_artifact_bytes", "passed": True},
                {"check": "champion_crosswalk_artifact_digest", "passed": True},
                {"check": "appearance_source_regular_file", "passed": True},
            ],
            "input_conflicts": [],
            "missing_components": [],
            "closed_components": closed_components,
            "out_of_distribution_flags": [],
            "appearance_window": window,
            "development_only": True,
        },
        "literal_interpretation": "Incremental model-standardized value, in probability points, of replacing the same-role reference mixture with this champion under equal-strength composition; played-only membership; descriptive counterability with zero weight.",
        "error": None,
    }
    payload["artifact_sha256"] = sha256_object({k: v for k, v in payload.items() if k != "artifact_sha256"})
    verify_tier_list_payload(payload)
    return payload


def _safe_resolve(crosswalk: Mapping[str, str], champion_name: str) -> str | None:
    try:
        return _resolve_champion_id(crosswalk, champion_name)
    except TierListError:
        return None


# ---------------------------------------------------------------------------
# Payload verification and persistence
# ---------------------------------------------------------------------------


_TOP_LEVEL_FIELDS = {
    "schema_version",
    "artifact_kind",
    "artifact_id",
    "artifact_sha256",
    "status",
    "fail_closed_status",
    "development_only",
    "rank_eligibility",
    "publication_eligible",
    "claim_ceiling",
    "scope",
    "role",
    "patch_id",
    "as_of",
    "created_at",
    "membership",
    "rows",
    "reference_convention",
    "estimand",
    "lineage",
    "provenance",
    "literal_interpretation",
    "error",
}

_ROW_FIELDS = {
    "champion_id",
    "champion_name",
    "role",
    "main_effect_logit",
    "reference_logit",
    "tier_value",
    "weighted_tier_value",
    "tier_value_interval_95",
    "counterability",
    "counterability_status",
    "counterability_reason",
    "counterability_regret_support_size",
    "verified_appearance_count",
    "evidence",
}

_MEMBERSHIP_FIELDS = {"champion_id", "champion_name", "role", "verified_appearance_count"}


def verify_tier_list_payload(payload: Mapping[str, Any]) -> None:
    """Structural and semantic verification of a tier-list artifact payload."""

    if not isinstance(payload, Mapping):
        raise TierListIntegrityError("tier-list payload must be an object")
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise TierListIntegrityError(f"tier-list top-level fields are not exact: {sorted(set(payload) ^ _TOP_LEVEL_FIELDS)}")
    submitted = payload.get("artifact_sha256")
    if not isinstance(submitted, str) or not HASH_RE.fullmatch(submitted):
        raise TierListIntegrityError("artifact_sha256 must be a lowercase sha256")
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    if sha256_object(unsigned) != submitted:
        raise TierListIntegrityError("artifact_sha256 does not match the canonical payload")

    if _deep_has_forbidden_key(payload):
        raise TierListIntegrityError("tier-list payload contains a raw-win-rate target field")

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise TierListIntegrityError("schema_version mismatch")
    if payload.get("artifact_kind") != ARTIFACT_KIND:
        raise TierListIntegrityError("artifact_kind mismatch")
    if payload.get("development_only") is not True:
        raise TierListIntegrityError("development_only must be true")
    if payload.get("rank_eligibility") is not False:
        raise TierListIntegrityError("rank eligibility must remain false until L2 promotion")
    if payload.get("publication_eligible") is not False:
        raise TierListIntegrityError("publication eligibility must remain false")
    claim_ceiling = payload.get("claim_ceiling")
    if not isinstance(claim_ceiling, Mapping) or any(claim_ceiling.values()):
        raise TierListIntegrityError("claim ceiling must remain all-false")
    status = payload.get("status")
    fail_closed_status = payload.get("fail_closed_status")
    if status not in {"development_only", "unavailable"}:
        raise TierListIntegrityError("status must be development_only or unavailable")
    if fail_closed_status not in {"none", "counterability_unavailable", "unavailable"}:
        raise TierListIntegrityError("fail_closed_status is invalid")
    role = payload.get("role")
    if role not in ROLES:
        raise TierListIntegrityError("role is not a canonical role")
    patch_id = payload.get("patch_id")
    if not isinstance(patch_id, str) or not PATCH_RE.fullmatch(patch_id):
        raise TierListIntegrityError("patch_id is invalid")
    parse_rfc3339(payload.get("as_of"))
    created_at = payload.get("created_at")
    parse_rfc3339(created_at)
    if parse_rfc3339(created_at) < parse_rfc3339(payload["as_of"]):
        raise TierListIntegrityError("created_at cannot precede as_of")

    scope = payload.get("scope")
    if not isinstance(scope, Mapping) or scope.get("scope_kind") not in {"league", "international"}:
        raise TierListIntegrityError("scope is invalid")
    if status == "unavailable":
        if fail_closed_status != "unavailable":
            raise TierListIntegrityError("unavailable payloads require fail_closed_status=unavailable")
        if payload.get("rows") not in ([], None) or payload.get("membership") not in ([], None):
            raise TierListIntegrityError("unavailable payloads must not carry rows or membership")
        error = payload.get("error")
        if not isinstance(error, Mapping) or not error.get("code") or not error.get("reason"):
            raise TierListIntegrityError("unavailable payloads require an error object")
        return
    if fail_closed_status == "unavailable":
        raise TierListIntegrityError("numeric payloads cannot be fully unavailable")
    if payload.get("error") is not None:
        raise TierListIntegrityError("numeric payloads cannot carry an error object")
    rows = payload.get("rows")
    membership = payload.get("membership")
    if not isinstance(rows, list) or not rows:
        raise TierListIntegrityError("numeric payloads require a non-empty rows list")
    if not isinstance(membership, list) or len(membership) != len(rows):
        raise TierListIntegrityError("membership must mirror rows")

    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise TierListIntegrityError("tier-list row fields are not exact")
        champion_id = row["champion_id"]
        if not isinstance(champion_id, str) or not champion_id.startswith("riot:champion:"):
            raise TierListIntegrityError("champion_id must be a stable riot champion id")
        if champion_id in seen_ids:
            raise TierListIntegrityError("duplicate champion row")
        seen_ids.add(champion_id)
        if row.get("role") != role:
            raise TierListIntegrityError("row role does not match the cell role")
        if row.get("verified_appearance_count") is True or not isinstance(row["verified_appearance_count"], int) or row["verified_appearance_count"] < 1:
            raise TierListIntegrityError("zero-play rows are forbidden")
        for field in ("main_effect_logit", "reference_logit", "tier_value", "weighted_tier_value"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise TierListIntegrityError(f"{field} must be finite numeric")
        counterability = row.get("counterability")
        counterability_status = row.get("counterability_status")
        if counterability_status not in {"available", "unavailable"}:
            raise TierListIntegrityError("counterability_status is invalid")
        if counterability_status == "unavailable":
            if counterability is not None:
                raise TierListIntegrityError("unavailable counterability must serialize as null, never zero")
        else:
            if not isinstance(counterability, (int, float)) or isinstance(counterability, bool) or counterability < 0:
                raise TierListIntegrityError("counterability must be nonnegative")
        if not math.isclose(float(row["weighted_tier_value"]), float(row["tier_value"]) - COUNTERABILITY_WEIGHT_LAMBDA_C * float(counterability or 0.0), abs_tol=1e-9):
            raise TierListIntegrityError("weighted tier value does not reconcile with tier value and lambda_C")
        interval = row.get("tier_value_interval_95")
        if not isinstance(interval, Mapping):
            raise TierListIntegrityError("tier_value_interval_95 must be an object")
        lower_bound = interval.get("lower")
        upper_bound = interval.get("upper")
        if (
            isinstance(lower_bound, bool)
            or not isinstance(lower_bound, (int, float))
            or isinstance(upper_bound, bool)
            or not isinstance(upper_bound, (int, float))
            or not (lower_bound <= row["tier_value"] <= upper_bound)
        ):
            raise TierListIntegrityError("tier_value_interval_95 must be numeric and contain tier_value")
    for entry in membership:
        if not isinstance(entry, Mapping) or set(entry) != _MEMBERSHIP_FIELDS:
            raise TierListIntegrityError("membership entry fields are not exact")
        if entry["champion_id"] not in seen_ids or entry["verified_appearance_count"] < 1:
            raise TierListIntegrityError("membership must mirror the played-only rows")

    reference = payload.get("reference_convention")
    if not isinstance(reference, Mapping):
        raise TierListIntegrityError("reference_convention is missing")
    if reference.get("counterability_weight_lambda_c") != COUNTERABILITY_WEIGHT_LAMBDA_C:
        raise TierListIntegrityError("counterability weight must be zero without L2 validation")
    if reference.get("counterability_weight_selection") != COUNTERABILITY_WEIGHT_SELECTION:
        raise TierListIntegrityError("counterability weight selection is invalid")
    tail_alpha = reference.get("counterability_tail_alpha")
    if not isinstance(tail_alpha, (int, float)) or isinstance(tail_alpha, bool) or not (0.0 < tail_alpha < 1.0):
        raise TierListIntegrityError("counterability tail alpha must be in (0, 1)")

    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping):
        raise TierListIntegrityError("lineage is missing")
    terminal = lineage.get("terminal_model")
    if not isinstance(terminal, Mapping) or terminal.get("model_version") != TERMINAL_MODEL_ARTIFACT["model_version"]:
        raise TierListIntegrityError("lineage terminal model identity mismatch")
    if not isinstance(terminal.get("artifact_sha256"), str) or not HASH_RE.fullmatch(terminal["artifact_sha256"]):
        raise TierListIntegrityError("lineage terminal artifact hash is invalid")
    crosswalk = lineage.get("champion_crosswalk")
    if not isinstance(crosswalk, Mapping) or crosswalk.get("artifact_sha256") != CROSSWALK_ARTIFACT["artifact_sha256"]:
        raise TierListIntegrityError("lineage champion crosswalk identity mismatch")
    appearance = lineage.get("appearance_source")
    if not isinstance(appearance, Mapping) or not isinstance(appearance.get("raw_sha256"), str) or not HASH_RE.fullmatch(appearance["raw_sha256"]):
        raise TierListIntegrityError("lineage appearance source hash is invalid")
    tree_hash = lineage.get("source_tree_sha256")
    if tree_hash is not None and (not isinstance(tree_hash, str) or not HASH_RE.fullmatch(tree_hash)):
        raise TierListIntegrityError("lineage source tree hash is invalid")
    if lineage.get("code_commit") is not None:
        raise TierListIntegrityError("development lineage must not claim a code commit")

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("mode") != "state_snapshot":
        raise TierListIntegrityError("provenance mode must be state_snapshot")
    if provenance.get("required_input_status") != "complete":
        raise TierListIntegrityError("numeric payloads require complete required inputs")


def _deep_has_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any(key in _FORBIDDEN_RAW_WIN_RATE_KEYS for key in value):
            return True
        return any(_deep_has_forbidden_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_deep_has_forbidden_key(item) for item in value)
    return False


def load_tier_list_artifact(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Load a persisted artifact with canonical-byte and digest verification."""
    if not path.is_file() or path.is_symlink():
        raise TierListIntegrityError(f"tier-list artifact is missing or not a regular file: {path}")
    raw = path.read_bytes()
    if expected_sha256 is not None and sha256_bytes(raw) != expected_sha256:
        raise TierListIntegrityError("tier-list artifact bytes do not match the expected sha256")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=lambda pairs: _reject_duplicates(pairs),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise TierListIntegrityError("tier-list artifact must be strict UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise TierListIntegrityError("tier-list artifact must be a JSON object")
    if canonical_bytes(payload) != raw:
        raise TierListIntegrityError("persisted tier-list artifact bytes are not canonical JSON")
    verify_tier_list_payload(payload)
    return dict(payload)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TierListIntegrityError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def write_tier_list_artifact(path: Path, payload: Mapping[str, Any], *, force: bool = False) -> str:
    """Persist a verified payload as canonical JSON; refuse silent overwrite."""
    verify_tier_list_payload(payload)
    raw = canonical_bytes(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise TierListIntegrityError(f"refusing to overwrite existing tier-list artifact: {path}")
    path.write_bytes(raw)
    loaded = load_tier_list_artifact(path, expected_sha256=sha256_bytes(raw))
    if loaded["artifact_sha256"] != payload["artifact_sha256"]:
        raise TierListIntegrityError("persisted artifact digest mismatch after write")
    return payload["artifact_sha256"]


def filter_rows(
    payload: Mapping[str, Any],
    *,
    region: str | None = None,
    league: str | None = None,
    international: str | None = None,
    competition_tier: str | None = None,
    role: str | None = None,
    patch: str | None = None,
    played_maps_min: int = 1,
) -> list[dict[str, Any]]:
    """User-facing filter view over an artifact (region/league/MSI-EWC/tier/role/patch)."""
    verify_tier_list_payload(payload)
    if played_maps_min < 1:
        raise TierListError("played_maps_min must be at least 1")
    rows = payload.get("rows") or []
    out = []
    for row in rows:
        if role is not None and row["role"] != role:
            continue
        if patch is not None and payload["patch_id"] != patch:
            continue
        if competition_tier is not None and payload["scope"]["competition_tier"] != competition_tier:
            continue
        if row["verified_appearance_count"] < played_maps_min:
            continue
        scope = payload["scope"]
        if scope["scope_kind"] == "international":
            if international is not None and scope["scope_id"] != international:
                continue
            if region == "international":
                pass
            elif region is not None:
                continue
            elif league is not None:
                continue
        else:
            if league is not None and scope["scope_id"] != league:
                continue
            if region is not None:
                if region == "international" or scope["scope_id"] not in REGIONS.get(region, ()):
                    continue
            if international is not None:
                continue
        out.append(row)
    return out
