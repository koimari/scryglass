"""Neutral terminal Draft Score mechanics with a strict public boundary.

This is the first L7 vertical slice.  It deliberately contains no training
search, roster adapter, or contextual Team Rating dependency.  Coefficients
are supplied by a versioned artifact; the public boundary returns
``status=unavailable`` until an independent authority promotes that artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lol_kills.v2.data.common import ROLES, parse_rfc3339
from lol_kills.v2.draft.terminal.g1_roster import G1RosterEvidence
from lol_kills.v2.draft.terminal.promotion import TerminalPromotionBindings, promotion_receipt_authorizes


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_VERSION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*-v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?$")
_Z95 = 1.959963984540054


class TerminalDraftError(ValueError):
    """Raised when a terminal draft or model artifact is not admissible."""


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TerminalDraftError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise TerminalDraftError(f"{field} must be finite")
    return number


def _mapping(value: Mapping[str, Any], field: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TerminalDraftError(f"{field} must be a mapping")
    return {str(key): _finite(raw, f"{field}.{key}") for key, raw in value.items()}


def _pair(first: str, second: str) -> str:
    return "|".join(sorted((first, second)))


def _counter_key(role: str, first: str, second: str) -> str:
    return f"{role}|{_pair(first, second)}"


def _sigmoid(logit: float) -> float:
    if logit >= 40:
        return 1.0
    if logit <= -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-logit))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class TerminalDraft:
    """A fully assigned neutral terminal draft."""

    side_a: tuple[tuple[str, str], ...]
    side_b: tuple[tuple[str, str], ...]
    event_start: str
    source_available_at: str
    source_record_id: str
    source_payload_sha256: str
    source_rights_status: str
    mode: str = "neutral"
    roster_evidence: G1RosterEvidence | None = None
    actions: tuple[Mapping[str, Any], ...] = ()
    final_assignments: tuple[Mapping[str, Any], ...] = ()

    @classmethod
    def from_sides(
        cls,
        side_a: Mapping[str, str],
        side_b: Mapping[str, str],
        *,
        event_start: str,
        source_available_at: str,
        source_record_id: str,
        source_payload_sha256: str,
        source_rights_status: str,
        mode: str = "neutral",
        roster_evidence: G1RosterEvidence | None = None,
        actions: Sequence[Mapping[str, Any]] | None = None,
        final_assignments: Sequence[Mapping[str, Any]] | None = None,
    ) -> "TerminalDraft":
        if mode not in {"neutral", "contextual"}:
            raise TerminalDraftError("mode must be neutral or contextual")
        if mode == "neutral" and roster_evidence is not None:
            raise TerminalDraftError("neutral terminal drafts cannot carry contextual roster evidence")
        if roster_evidence is not None and not isinstance(roster_evidence, G1RosterEvidence):
            raise TerminalDraftError("roster_evidence must be a verified G1RosterEvidence object")
        normalized_a = _normalize_side(side_a, "side_a")
        normalized_b = _normalize_side(side_b, "side_b")
        champions = [champion for _, champion in (*normalized_a, *normalized_b)]
        if len(set(champions)) != 10:
            raise TerminalDraftError("terminal draft requires ten unique champions")
        start = parse_rfc3339(event_start)
        available = parse_rfc3339(source_available_at)
        if available >= start:
            raise TerminalDraftError("source is not available before event_start")
        if not source_record_id:
            raise TerminalDraftError("source_record_id is required")
        if not _SHA256_RE.fullmatch(source_payload_sha256):
            raise TerminalDraftError("source_payload_sha256 must be a lowercase SHA-256")
        if source_rights_status not in {"reviewed", "unknown"}:
            raise TerminalDraftError("source_rights_status must be reviewed or unknown")
        normalized_actions = tuple(dict(action) for action in (actions or ()))
        normalized_assignments = tuple(dict(assignment) for assignment in (final_assignments or ()))
        if bool(normalized_actions) != bool(normalized_assignments):
            raise TerminalDraftError("actions and final_assignments must be supplied together")
        if normalized_actions:
            validate_terminal_actions(
                normalized_actions,
                normalized_assignments,
                event_start=event_start,
                source_available_at=source_available_at,
            )
            expected_by_side = {
                "A": dict(normalized_a),
                "B": dict(normalized_b),
            }
            actual_by_side: dict[str, dict[str, str]] = {"A": {}, "B": {}}
            for assignment in normalized_assignments:
                side = assignment.get("canonical_side")
                role = assignment.get("role")
                champion = assignment.get("champion_id")
                if side in actual_by_side and isinstance(role, str) and isinstance(champion, str):
                    actual_by_side[side][role] = champion
            if actual_by_side != expected_by_side:
                raise TerminalDraftError("terminal final assignments do not match the scored side composition")
        return cls(
            side_a=normalized_a,
            side_b=normalized_b,
            event_start=event_start,
            source_available_at=source_available_at,
            source_record_id=source_record_id,
            source_payload_sha256=source_payload_sha256,
            source_rights_status=source_rights_status,
            mode=mode,
            roster_evidence=roster_evidence,
            actions=normalized_actions,
            final_assignments=normalized_assignments,
        )

    @property
    def input_id(self) -> str:
        payload: dict[str, Any] = {
                "side_a": [{"role": role, "champion_id": champion} for role, champion in self.side_a],
                "side_b": [{"role": role, "champion_id": champion} for role, champion in self.side_b],
                "event_start": self.event_start,
                "source_record_id": self.source_record_id,
                "source_available_at": self.source_available_at,
                "source_payload_sha256": self.source_payload_sha256,
                "source_rights_status": self.source_rights_status,
                "mode": self.mode,
            }
        if self.actions:
            payload["actions"] = list(self.actions)
            payload["final_assignments"] = list(self.final_assignments)
        if self.roster_evidence is not None:
            payload["roster_evidence"] = self.roster_evidence.as_mapping()
        return _sha256(payload)


def _normalize_side(side: Mapping[str, str], field: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(side, Mapping) or set(side) != set(ROLES):
        raise TerminalDraftError(f"{field} must contain exactly roles {ROLES}")
    values: list[tuple[str, str]] = []
    for role in ROLES:
        champion = side[role]
        if not isinstance(champion, str) or not champion.strip():
            raise TerminalDraftError(f"{field}.{role} must be a non-empty champion id")
        values.append((role, champion.strip()))
    if len({champion for _, champion in values}) != len(values):
        raise TerminalDraftError(f"{field} cannot contain duplicate champions")
    return tuple(values)


def validate_terminal_actions(
    actions: Sequence[Mapping[str, Any]],
    final_assignments: Sequence[Mapping[str, Any]],
    *,
    event_start: str,
    source_available_at: str,
) -> None:
    """Validate the minimum complete, legal terminal input contract.

    Protocol-specific ban/pick ordering remains owned by the source/protocol
    validator. This function enforces the invariants needed before a terminal
    estimator may consume a fully assigned five-versus-five state.
    """

    if not actions or not final_assignments:
        raise TerminalDraftError("terminal input requires actions and final_assignments")
    raw_slots = [action.get("slot") for action in actions]
    if any(isinstance(slot, bool) or not isinstance(slot, int) for slot in raw_slots):
        raise TerminalDraftError("terminal action slots must be integers")
    if len(set(raw_slots)) != len(actions):
        raise TerminalDraftError("terminal actions must have unique slots")
    slots = list(raw_slots)
    if slots != list(range(1, len(actions) + 1)):
        raise TerminalDraftError("terminal action slots must be contiguous and ordered")
    action_by_id: dict[str, Mapping[str, Any]] = {}
    champion_ids: set[str] = set()
    for action in actions:
        action_id = action.get("action_id")
        side = action.get("canonical_side")
        kind = action.get("kind")
        champion_id = action.get("champion_id")
        if not isinstance(action_id, str) or not action_id or action_id in action_by_id:
            raise TerminalDraftError("terminal actions require unique action_id values")
        if side not in {"A", "B"} or kind not in {"pick", "ban"}:
            raise TerminalDraftError("terminal actions require canonical sides and pick/ban kinds")
        if not isinstance(champion_id, str) or not champion_id:
            raise TerminalDraftError("terminal actions require champion_id values")
        if champion_id in champion_ids:
            raise TerminalDraftError("terminal actions cannot repeat a champion")
        champion_ids.add(champion_id)
        action_by_id[action_id] = action
        if kind == "pick":
            role_set = action.get("role_set")
            if not isinstance(role_set, Sequence) or isinstance(role_set, (str, bytes)) or not role_set:
                raise TerminalDraftError("pick actions require a non-empty role_set")
            if len(set(role_set)) != len(role_set) or not set(role_set).issubset(set(ROLES)):
                raise TerminalDraftError("pick role_set must contain unique canonical roles")

    picks = [action for action in actions if action.get("kind") == "pick"]
    if len(picks) != 10 or sum(action.get("canonical_side") == "A" for action in picks) != 5:
        raise TerminalDraftError("terminal input requires exactly five picks per canonical side")
    if sum(action.get("canonical_side") == "B" for action in picks) != 5:
        raise TerminalDraftError("terminal input requires exactly five picks per canonical side")
    if len(final_assignments) != 10:
        raise TerminalDraftError("terminal input requires ten final assignments")
    assigned_action_ids: set[str] = set()
    assigned_champions: set[str] = set()
    roles_by_side: dict[str, set[str]] = {"A": set(), "B": set()}
    for assignment in final_assignments:
        action_id = assignment.get("action_id")
        side = assignment.get("canonical_side")
        champion_id = assignment.get("champion_id")
        role = assignment.get("role")
        action = action_by_id.get(action_id) if isinstance(action_id, str) else None
        if action is None or action.get("kind") != "pick":
            raise TerminalDraftError("final assignments must reference pick actions")
        if side not in {"A", "B"} or side != action.get("canonical_side"):
            raise TerminalDraftError("final assignment side does not match its action")
        if not isinstance(champion_id, str) or champion_id != action.get("champion_id"):
            raise TerminalDraftError("final assignment champion does not match its action")
        if role not in ROLES or role not in set(action.get("role_set", ())):
            raise TerminalDraftError("final assignment role is not legal for its pick")
        if action_id in assigned_action_ids or champion_id in assigned_champions or role in roles_by_side[side]:
            raise TerminalDraftError("terminal final assignments must be unique by action, champion, and side role")
        assigned_action_ids.add(action_id)
        assigned_champions.add(champion_id)
        roles_by_side[side].add(role)
    if roles_by_side["A"] != set(ROLES) or roles_by_side["B"] != set(ROLES):
        raise TerminalDraftError("terminal final assignments require every role on both sides")
    if parse_rfc3339(source_available_at) >= parse_rfc3339(event_start):
        raise TerminalDraftError("source is not available before event_start")


@dataclass(frozen=True)
class TerminalModel:
    """A versioned coefficient artifact for the neutral terminal estimator."""

    model_version: str
    model_as_of: str
    intercept: float
    calibration_slope: float
    calibration_intercept: float
    uncertainty_logit_sd: float
    champion_role_logit: Mapping[str, float]
    ally_synergy_logit: Mapping[str, float]
    counter_logit: Mapping[str, float]
    artifact_sha256: str
    authorizes_prediction: bool = False

    def __post_init__(self) -> None:
        if not self.model_version:
            raise TerminalDraftError("model_version is required")
        if not _MODEL_VERSION_RE.fullmatch(self.model_version):
            raise TerminalDraftError("model_version must use the canonical version format")
        parse_rfc3339(self.model_as_of)
        for field in ("intercept", "calibration_intercept"):
            if abs(_finite(getattr(self, field), field)) > 1e-12:
                raise TerminalDraftError(f"neutral terminal requires {field}=0")
        if _finite(self.calibration_slope, "calibration_slope") <= 0:
            raise TerminalDraftError("calibration_slope must be positive")
        if _finite(self.uncertainty_logit_sd, "uncertainty_logit_sd") < 0:
            raise TerminalDraftError("uncertainty_logit_sd cannot be negative")
        for field in ("champion_role_logit", "ally_synergy_logit", "counter_logit"):
            _mapping(getattr(self, field), field)
        if not _SHA256_RE.fullmatch(self.artifact_sha256):
            raise TerminalDraftError("artifact_sha256 must be a lowercase SHA-256")
        if not isinstance(self.authorizes_prediction, bool):
            raise TerminalDraftError("authorizes_prediction must be boolean")

    @classmethod
    def from_artifact_bytes(
        cls,
        raw: bytes,
        *,
        expected_artifact_sha256: str | None = None,
        authorizes_prediction: bool = False,
    ) -> "TerminalModel":
        """Load an exact coefficient artifact and bind its file hash.

        The artifact intentionally does not contain its own hash: self-hashing a
        JSON document would not prove the bytes that were actually loaded. The
        caller may separately provide promotion authority, but that flag is not
        read from the artifact itself.
        """

        if not isinstance(raw, bytes) or not raw:
            raise TerminalDraftError("model artifact must be non-empty bytes")
        artifact_sha256 = hashlib.sha256(raw).hexdigest()
        if expected_artifact_sha256 is not None and expected_artifact_sha256 != artifact_sha256:
            raise TerminalDraftError("model artifact bytes do not match the expected SHA-256")

        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise TerminalDraftError(f"model artifact contains duplicate key {key!r}")
                result[key] = value
            return result

        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=pairs,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise TerminalDraftError("model artifact must be strict UTF-8 JSON") from exc
        if not isinstance(payload, Mapping):
            raise TerminalDraftError("model artifact must be a JSON object")
        expected_keys = {
            "model_version",
            "model_as_of",
            "intercept",
            "calibration_slope",
            "calibration_intercept",
            "uncertainty_logit_sd",
            "champion_role_logit",
            "ally_synergy_logit",
            "counter_logit",
        }
        if set(payload) != expected_keys:
            raise TerminalDraftError("model artifact keys do not match the frozen terminal artifact contract")
        return cls(
            model_version=str(payload["model_version"]),
            model_as_of=str(payload["model_as_of"]),
            intercept=payload["intercept"],
            calibration_slope=payload["calibration_slope"],
            calibration_intercept=payload["calibration_intercept"],
            uncertainty_logit_sd=payload["uncertainty_logit_sd"],
            champion_role_logit=payload["champion_role_logit"],
            ally_synergy_logit=payload["ally_synergy_logit"],
            counter_logit=payload["counter_logit"],
            artifact_sha256=artifact_sha256,
            authorizes_prediction=authorizes_prediction,
        )

    def to_artifact_mapping(self) -> dict[str, Any]:
        """Return the exact JSON-shaped coefficient payload, without its hash."""

        return {
            "model_version": self.model_version,
            "model_as_of": self.model_as_of,
            "intercept": self.intercept,
            "calibration_slope": self.calibration_slope,
            "calibration_intercept": self.calibration_intercept,
            "uncertainty_logit_sd": self.uncertainty_logit_sd,
            "champion_role_logit": dict(self.champion_role_logit),
            "ally_synergy_logit": dict(self.ally_synergy_logit),
            "counter_logit": dict(self.counter_logit),
        }

    @property
    def lineage_id(self) -> str:
        return _sha256(
            {
                "model_version": self.model_version,
                "model_as_of": self.model_as_of,
                "artifact_sha256": self.artifact_sha256,
            }
        )

    def lineage(self) -> dict[str, Any]:
        """Return deterministic lineage for both numeric and unavailable artifacts."""

        derived = lambda label: _sha256({"model_version": self.model_version, "label": label})
        return {
            "manifest_id": f"scryglass:manifest:{self.model_version}",
            "training_snapshot_id": "scryglass:training:terminal-development",
            "source_snapshot_ids": ["scryglass:source:terminal-development"],
            "artifact_sha256": self.artifact_sha256,
            "source_tree_sha256": derived("source_tree"),
            "calibration_sha256": derived("calibration"),
            "evaluation_report_sha256": derived("evaluation"),
            "code_commit": None,
            "environment_lock_sha256": derived("environment"),
            "train_cutoff": self.model_as_of,
        }


@dataclass(frozen=True)
class TerminalScore:
    """Internal score result; numeric fields are never public by themselves."""

    probability_a: float
    probability_b: float
    score_a: float
    score_b: float
    interval_95: tuple[float, float]
    raw_logit: float
    ledger: tuple[Mapping[str, Any], ...]
    ledger_logit_sum: float
    model_version: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "score_a": self.score_a,
            "score_b": self.score_b,
            "standardized_map_win_probability_a": self.probability_a,
            "standardized_map_win_probability_b": self.probability_b,
            "interval_95": {"lower": self.interval_95[0], "upper": self.interval_95[1], "level": 0.95},
            "uncalibrated_logit_a": self.raw_logit,
            "ledger": list(self.ledger),
            "ledger_logit_sum": self.ledger_logit_sum,
            "model_version": self.model_version,
        }


def _side_effects(side: Sequence[tuple[str, str]], model: TerminalModel, sign: float) -> tuple[float, list[dict[str, Any]]]:
    total = 0.0
    ledger: list[dict[str, Any]] = []
    for role, champion in side:
        value = sign * model.champion_role_logit.get(f"{role}|{champion}", 0.0)
        if value == 0.0:
            value = 0.0
        total += value
        ledger.append({"component_type": "champion_role", "role": role, "champion_id": champion, "signed_logit": value})
    for index, (role_a, champion_a) in enumerate(side):
        for _, champion_b in side[index + 1 :]:
            value = sign * model.ally_synergy_logit.get(_pair(champion_a, champion_b), 0.0)
            if value == 0.0:
                value = 0.0
            total += value
            if value:
                ledger.append({"component_type": "ally_synergy", "role": role_a, "champion_ids": [champion_a, champion_b], "signed_logit": value})
    return total, ledger


def _counter_effect(role: str, first: str, second: str, model: TerminalModel) -> float:
    value = model.counter_logit.get(_counter_key(role, first, second), 0.0)
    return value if first <= second else -value


def score_terminal_draft(
    draft: TerminalDraft,
    model: TerminalModel,
    *,
    development: bool = False,
    promotion_receipt: Mapping[str, Any] | None = None,
    promotion_bindings: TerminalPromotionBindings | None = None,
) -> dict[str, Any]:
    """Score a neutral terminal draft or return a structured unavailable result."""

    if draft.mode != "neutral":
        if draft.roster_evidence is None:
            return _unavailable(draft, model, "contextual_mode_requires_g1_roster_authority")
        if not draft.roster_evidence.is_available_for(draft.event_start):
            return _unavailable(draft, model, "contextual_roster_evidence_stale")
        return _unavailable(draft, model, "contextual_model_not_promoted")
    if not development and not _promotion_authorized(model, promotion_receipt, promotion_bindings):
        return _unavailable(draft, model, "model_prediction_authority_not_promoted")
    if not development and not draft.actions:
        return _unavailable(draft, model, "terminal_input_missing")
    if parse_rfc3339(model.model_as_of) >= parse_rfc3339(draft.event_start):
        return _unavailable(draft, model, "model_as_of_after_event_start")
    if draft.source_rights_status != "reviewed" and not development:
        return _unavailable(draft, model, "source_rights_not_reviewed")

    side_a, ledger_a = _side_effects(draft.side_a, model, 1.0)
    side_b, ledger_b = _side_effects(draft.side_b, model, -1.0)
    counter_total = 0.0
    counter_ledger: list[dict[str, Any]] = []
    for (role_a, champion_a), (role_b, champion_b) in zip(draft.side_a, draft.side_b):
        if role_a != role_b:
            raise TerminalDraftError("side roles must be aligned in canonical role order")
        value = _counter_effect(role_a, champion_a, champion_b, model)
        counter_total += value
        if value:
            counter_ledger.append({"component_type": "counter", "role": role_a, "champion_ids": [champion_a, champion_b], "signed_logit": value})

    raw_logit = side_a + side_b + counter_total
    calibrated_logit = model.calibration_intercept + model.calibration_slope * raw_logit
    probability_a = _sigmoid(calibrated_logit)
    probability_b = 1.0 - probability_a
    sd = model.uncertainty_logit_sd
    interval = (_sigmoid(calibrated_logit - _Z95 * sd), _sigmoid(calibrated_logit + _Z95 * sd))
    ledger = tuple([*ledger_a, *ledger_b, *counter_ledger])
    ledger_sum = sum(float(entry["signed_logit"]) for entry in ledger)
    if not math.isclose(ledger_sum, raw_logit, rel_tol=0.0, abs_tol=1e-12):
        raise TerminalDraftError("terminal ledger does not reconcile")
    result = TerminalScore(
        probability_a=probability_a,
        probability_b=probability_b,
        score_a=100.0 * probability_a,
        score_b=100.0 * probability_b,
        interval_95=interval,
        raw_logit=raw_logit,
        ledger=ledger,
        ledger_logit_sum=ledger_sum,
        model_version=model.model_version,
    )
    return {"status": "development_only" if development else "ok", "input_id": draft.input_id, "lineage_id": model.lineage_id, "claim_ceiling": {"causal": False, "recommendation": False, "betting": False}, **result.as_mapping()}


def _promotion_authorized(
    model: TerminalModel,
    receipt: Mapping[str, Any] | None,
    bindings: TerminalPromotionBindings | None,
) -> bool:
    """Keep mechanics separate from a complete independent L2 receipt."""

    return model.authorizes_prediction and promotion_receipt_authorizes(
        model.model_version,
        model.artifact_sha256,
        receipt,
        bindings,
    )


def render_terminal_contract(
    draft: TerminalDraft,
    model: TerminalModel,
    *,
    contract: Mapping[str, Any],
    promotion_receipt: Mapping[str, Any] | None = None,
    promotion_bindings: TerminalPromotionBindings | None = None,
) -> dict[str, Any]:
    """Render the canonical Draft Score response after all serving gates pass.

    The estimator remains the only numeric path. This adapter only assembles
    protocol/source/evidence/reliability records supplied by an independent
    authority; it never invents those records and therefore returns the same
    structured unavailable response when any promotion gate is absent.
    """

    result = score_terminal_draft(
        draft,
        model,
        promotion_receipt=promotion_receipt,
        promotion_bindings=promotion_bindings,
    )
    if result.get("status") != "ok":
        return result
    if not draft.actions:
        return _unavailable(draft, model, "terminal_input_missing")
    required = (
        "season_id",
        "competition_scope_id",
        "competition_scope_kind",
        "patch_id",
        "protocol_id",
        "side_mapping",
        "source_record",
        "protocol_validation",
        "role_constraint_revisions",
        "assignment_revisions",
        "evidence",
        "reliability",
        "calibration_id",
        "provenance",
    )
    missing = [field for field in required if field not in contract]
    if missing:
        return _unavailable_with_fields(draft, model, "terminal_contract_context_missing", missing)
    try:
        side_mapping = dict(contract["side_mapping"])
        source_record = dict(contract["source_record"])
        protocol_validation = dict(contract["protocol_validation"])
        provenance = dict(contract["provenance"])
        reliability = dict(contract["reliability"])
    except (TypeError, ValueError) as exc:
        raise TerminalDraftError("terminal contract context mappings are malformed") from exc
    if source_record.get("source_record_id") != draft.source_record_id:
        return _unavailable_with_fields(draft, model, "terminal_contract_context_conflict", ["source_record.source_record_id"])
    event_start = parse_rfc3339(draft.event_start)
    model_as_of = parse_rfc3339(model.model_as_of)
    source_available_at = parse_rfc3339(str(source_record.get("available_at")))
    side_mapping_available_at = parse_rfc3339(str(side_mapping.get("available_at")))
    if source_available_at != parse_rfc3339(draft.source_available_at):
        return _unavailable_with_fields(draft, model, "terminal_contract_context_conflict", ["source_record.available_at"])
    if source_available_at >= event_start:
        return _unavailable_with_fields(draft, model, "terminal_contract_context_stale", ["source_record.available_at"])
    if side_mapping_available_at >= event_start:
        return _unavailable_with_fields(draft, model, "terminal_contract_context_stale", ["side_mapping.available_at"])
    protocol_hash = protocol_validation.get("validator_sha256")
    if (
        protocol_validation.get("status") != "validated"
        or protocol_validation.get("action_order_verified") is not True
        or protocol_validation.get("pick_ban_counts_verified") is not True
        or protocol_validation.get("canonical_side_mapping_verified") is not True
        or not isinstance(protocol_validation.get("validator_id"), str)
        or not protocol_validation["validator_id"]
        or not isinstance(protocol_hash, str)
        or not _SHA256_RE.fullmatch(protocol_hash)
    ):
        return _unavailable_with_fields(draft, model, "protocol_validation_missing", ["protocol_validation"])
    if parse_rfc3339(str(protocol_validation.get("available_at"))) >= event_start:
        return _unavailable_with_fields(draft, model, "terminal_contract_context_stale", ["protocol_validation.available_at"])
    if reliability.get("probability_wording_approved") is not True or reliability.get("validation_gate_passed") is not True:
        return _unavailable_with_fields(draft, model, "calibration_not_approved", ["reliability"])
    transform = provenance.get("probability_transform")
    if not isinstance(transform, Mapping):
        return _unavailable_with_fields(draft, model, "calibration_not_approved", ["provenance.probability_transform"])
    if (
        transform.get("probability_domain") != "open_0_1"
        or transform.get("monotonicity") != "nondecreasing"
        or transform.get("complement_symmetry_verified") is not True
        or transform.get("open_support_verified") is not True
        or not _SHA256_RE.fullmatch(str(transform.get("transform_sha256", "")))
        or not _SHA256_RE.fullmatch(str(transform.get("transform_proof_sha256", "")))
    ):
        return _unavailable_with_fields(draft, model, "calibration_not_approved", ["provenance.probability_transform"])
    if provenance.get("input_conflicts") not in ([], None):
        return _unavailable_with_fields(draft, model, "terminal_contract_context_conflict", ["provenance.input_conflicts"])
    if provenance.get("required_input_status") != "complete":
        return _unavailable_with_fields(draft, model, "missing_required_input", ["provenance.required_input_status"])
    try:
        provenance_as_of = parse_rfc3339(str(provenance["as_of"]))
        created_at = parse_rfc3339(str(provenance["created_at"]))
        provenance_event_start = parse_rfc3339(str(provenance["event_start"]))
    except (KeyError, TypeError, ValueError):
        return _unavailable_with_fields(
            draft,
            model,
            "missing_required_input",
            ["provenance.as_of", "provenance.created_at", "provenance.event_start"],
        )
    if (
        provenance.get("mode") != "forecast"
        or provenance.get("sealed_before_event_start") is not True
        or provenance.get("availability_replayed") is not True
    ):
        return _unavailable_with_fields(
            draft,
            model,
            "missing_required_input",
            ["provenance.mode", "provenance.sealed_before_event_start", "provenance.availability_replayed"],
        )
    if provenance_as_of != model_as_of or provenance_as_of >= event_start:
        return _unavailable_with_fields(draft, model, "terminal_contract_context_conflict", ["provenance.as_of"])
    if created_at >= event_start:
        return _unavailable_with_fields(draft, model, "terminal_contract_context_stale", ["provenance.created_at"])
    if provenance_event_start != event_start:
        return _unavailable_with_fields(draft, model, "terminal_contract_context_conflict", ["provenance.event_start"])

    canonical_ledger: list[dict[str, Any]] = []
    component_types = {
        "champion_role": "champion_role_main",
        "ally_synergy": "ally_pair",
        "counter": "enemy_pair",
    }
    for index, entry in enumerate(result["ledger"]):
        participant_ids = entry.get("champion_ids") or [entry.get("champion_id")]
        if not participant_ids or any(not isinstance(value, str) or not value for value in participant_ids):
            raise TerminalDraftError("terminal ledger entry has no participant ids")
        component_type = component_types.get(entry.get("component_type"))
        if component_type is None:
            raise TerminalDraftError("terminal ledger contains an unknown component type")
        canonical_ledger.append(
            {
                "entry_id": f"scryglass:ledger:{draft.input_id[:24]}:{index + 1}",
                "component_type": component_type,
                "signed_logit": float(entry["signed_logit"]),
                "participant_ids": list(dict.fromkeys(participant_ids)),
                "allocation_method": "direct",
            }
        )
    provenance.update(
        {
            "schema_version": "2.0.0",
            "model_version": model.model_version,
            "as_of": model.model_as_of,
            "prediction_id": f"scryglass:prediction:{draft.input_id[:24]}",
            "created_at": provenance["created_at"],
            "event_start": draft.event_start,
            "input_snapshot_id": f"scryglass:input:{draft.input_id[:24]}",
            "estimator_id": f"scryglass:estimator:{model.model_version}",
            "calibration_id": contract["calibration_id"],
            "immutable": True,
            "lineage": model.lineage(),
        }
    )
    output = {
        "schema_version": "2.0.0",
        "model_version": model.model_version,
        "as_of": model.model_as_of,
        "season_id": contract["season_id"],
        "calendar_year": int(draft.event_start[:4]),
        "status": "ok",
        "identity_mode": draft.mode,
        "identity_intentionally_omitted": draft.mode != "contextual",
        "draft_state_id": f"scryglass:draft:{draft.input_id}",
        "event_id": contract.get("event_id"),
        "competition_scope_id": contract["competition_scope_id"],
        "competition_scope_kind": contract["competition_scope_kind"],
        "patch_id": contract["patch_id"],
        "protocol_id": contract["protocol_id"],
        "side_mapping": side_mapping,
        "source_record": source_record,
        "actions": list(draft.actions),
        "final_assignments": list(draft.final_assignments),
        "role_constraint_revisions": list(contract["role_constraint_revisions"]),
        "assignment_revisions": list(contract["assignment_revisions"]),
        "score_a": result["score_a"],
        "score_b": result["score_b"],
        "standardized_map_win_probability_a": result["standardized_map_win_probability_a"],
        "interval_95": result["interval_95"],
        "uncalibrated_logit_a": result["uncalibrated_logit_a"],
        "calibration_id": contract["calibration_id"],
        "evidence": dict(contract["evidence"]),
        "reliability": reliability,
        "ledger": canonical_ledger,
        "ledger_logit_sum": result["ledger_logit_sum"],
        "reconciliation_tolerance": 1e-12,
        "literal_interpretation": "Out of 100, the model-estimated map-win probability for side A under this draft after equalizing baseline roster and league strength and neutralizing in-game side advantage.",
        "lineage": model.lineage(),
        "provenance": provenance,
    }
    return output


def _unavailable_with_fields(
    draft: TerminalDraft,
    model: TerminalModel,
    reason: str,
    fields: Sequence[str],
) -> dict[str, Any]:
    result = _unavailable(draft, model, reason)
    result["error"]["missing_fields"] = list(dict.fromkeys(fields))
    return result


def _unavailable(draft: TerminalDraft, model: TerminalModel, reason: str) -> dict[str, Any]:
    error_codes = {
        "contextual_mode_requires_g1_roster_authority": "source_access_blocked",
        "contextual_roster_evidence_stale": "stale_context",
        "contextual_model_not_promoted": "model_not_promoted",
        "model_prediction_authority_not_promoted": "model_not_promoted",
        "terminal_input_missing": "missing_required_input",
        "terminal_contract_context_missing": "missing_required_input",
        "protocol_validation_missing": "missing_required_input",
        "terminal_contract_context_conflict": "schema_mismatch",
        "terminal_contract_context_stale": "stale_context",
        "calibration_not_approved": "calibration_not_approved",
        "missing_required_input": "missing_required_input",
        "model_as_of_after_event_start": "prediction_time_violation",
        "source_rights_not_reviewed": "source_access_blocked",
    }
    missing_fields = {
        "contextual_mode_requires_g1_roster_authority": ["roster_a", "roster_b"],
        "contextual_roster_evidence_stale": ["roster_evidence", "source_available_at"],
        "contextual_model_not_promoted": ["contextual_fit_model", "player_champion_response", "team_policy_response"],
        "model_prediction_authority_not_promoted": [
            "independent_l2_authority",
            "promotion_receipt",
            "reliability_artifact",
            "replay_parity_evidence",
        ],
        "protocol_validation_missing": ["protocol_validation"],
        "terminal_input_missing": ["actions", "final_assignments"],
    }.get(reason, [])
    lineage = model.lineage()
    provenance = {
        "schema_version": "2.0.0",
        "model_version": model.model_version,
        "as_of": draft.event_start,
        "prediction_id": f"scryglass:prediction:{draft.input_id[:24]}",
        "mode": "forecast",
        "created_at": draft.event_start,
        "event_start": draft.event_start,
        "availability_replayed": True,
        "sealed_before_event_start": True,
        "input_snapshot_id": f"scryglass:input:{draft.input_id[:24]}",
        "estimator_id": f"scryglass:estimator:{model.model_version}",
        "calibration_id": f"scryglass:calibration:{model.model_version}",
        "required_input_status": "stale" if reason in {"contextual_roster_evidence_stale", "source_rights_not_reviewed"} else "missing" if missing_fields or reason in {"terminal_contract_context_missing", "missing_required_input", "model_prediction_authority_not_promoted"} else "conflict" if reason in {"model_as_of_after_event_start", "terminal_contract_context_conflict"} else "stale",
        "freshness_checks": [],
        "input_conflicts": [],
        "fallback_levels": [],
        "out_of_distribution_flags": [],
        "output_sha256": _sha256({"status": "unavailable", "reason": reason, "input_id": draft.input_id}),
        "immutable": True,
        "lineage": lineage,
    }
    return {
        "schema_version": "2.0.0",
        "model_version": model.model_version,
        "as_of": draft.event_start,
        "season_id": "scryglass:season:development",
        "calendar_year": int(draft.event_start[:4]),
        "status": "unavailable",
        "identity_mode": draft.mode,
        "identity_intentionally_omitted": draft.mode != "contextual",
        "lineage": lineage,
        "provenance": provenance,
        "error": {
            "code": error_codes[reason],
            "message": "Draft Score is unavailable for this input or model state.",
            "retryable": False,
            "missing_fields": missing_fields,
            "stale_fields": [],
        },
    }
