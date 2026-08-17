"""Validate one independent Draft promotion receipt for public publication."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lol_kills.research.public_draft_score_promotion import sha256_path
from lol_kills.research.selective_draft_probability import canonical_sha256
from lol_kills.research.verify_selective_draft_promotion import (
    APPROVED_FIELDS,
RECEIPT_SCHEMA_VERSION,
)


DRAFT_AUTHORITY_SCHEMA = "scryglass:draft-authority:v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RELEASE_ID_PATTERN = re.compile(r"v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}")
UTC_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z"
)
BOUND_HASH_FIELDS = (
    "candidate_artifact_sha256",
    "candidate_receipt_sha256",
    "protocol_file_sha256",
    "evaluation_file_sha256",
    "evaluation_receipt_sha256",
    "decision_file_sha256",
    "decision_receipt_sha256",
    "outcomes_sha256",
)
PROMOTED_RESULTS_SCHEMA = "scryglass:promoted-draft-results:v1"
PUBLIC_RESULT_SCHEMA = "scryglass:public-draft-score-result:v1"
PROMOTED_RESULTS_KEYS = frozenset(
    {"schema_version", "authority", "release_id", "model_version", "receipt_sha256", "results"}
)
PUBLIC_RESULT_KEYS = frozenset(
    {
        "schema_version", "authority", "release_id", "model_version",
        "receipt_sha256", "evidence_window", "match_win_probability",
        "controlled_draft_score", "side_recommendation",
    }
)
PROBABILITY_KEYS = frozenset({"Blue", "Red"})
EVIDENCE_WINDOW_KEYS = frozenset({"start", "end"})
CONTROLLED_SCORE_KEYS = frozenset(
    {
        "model_units", "edge_percentage_points", "stronger_draft", "explanation",
        "method", "intervention_receipt_sha256", "isolated_blue_draft_probability",
        "fixed_strength_blue_win_probability",
    }
)
OWNER_RELEASE_RECEIPT_SCHEMA = "scryglass:public-draft-score-owner-release-receipt:v1"
OWNER_MODEL_VERSION = "public-draft-score-v34"
OWNER_CANDIDATE_PATH = Path(
    "data/lol/v2/evaluation/public-draft-score-selective-candidate-v34.json"
)
OWNER_DEVELOPMENT_EVALUATION_PATH = Path(
    "data/lol/v2/evaluation/public-draft-score-v34-development-evaluation.json"
)
OWNER_SUPPORTING_EVALUATION_PATH = Path(
    "data/lol/v2/evaluation/public-draft-score-v34-supporting-evaluation.json"
)
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {"betting", "odds", "fair_odds", "expected_value", "ev", "stake", "wager"}
)


class PromotedDraftAuthorityError(ValueError):
    """Raised when a promotion receipt cannot authorize publication."""


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in FORBIDDEN_PUBLIC_KEYS
            or _contains_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not UTC_PATTERN.fullmatch(value):
        raise PromotedDraftAuthorityError(f"{label} is invalid")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise PromotedDraftAuthorityError(f"{label} is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PromotedDraftAuthorityError(f"{label} is invalid")
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotedDraftAuthorityError("promotion receipt is unreadable") from error
    if not isinstance(value, dict):
        raise PromotedDraftAuthorityError("promotion receipt is not an object")
    return value


def _valid_owner_release_contract(receipt: Mapping[str, Any]) -> bool:
    evaluation = receipt.get("evaluation")
    window = receipt.get("evidence_window")
    if not isinstance(evaluation, Mapping) or not isinstance(window, Mapping):
        return False
    numeric_fields = (
        "coverage",
        "auc",
        "brier",
        "log_loss",
        "ece_10",
        "series_bootstrap_auc_median",
    )
    return (
        receipt.get("schema_version") == OWNER_RELEASE_RECEIPT_SCHEMA
        and receipt.get("status") == "promoted"
        and receipt.get("authority") == "promoted"
        and receipt.get("model_version") == OWNER_MODEL_VERSION
        and receipt.get("release_basis")
        == "owner_authorized_development_validation"
        and isinstance(receipt.get("owner_identity"), str)
        and bool(str(receipt["owner_identity"]).strip())
        and tuple(receipt.get("approved_public_fields") or ()) == APPROVED_FIELDS
        and receipt.get("public_probability") is True
        and receipt.get("public_recommendation") is True
        and receipt.get("betting_odds_ev_stake") is False
        and all(
            isinstance(receipt.get(field), str)
            and bool(SHA256_PATTERN.fullmatch(str(receipt[field])))
            for field in (
                "candidate_artifact_sha256",
                "candidate_receipt_sha256",
                "development_evaluation_file_sha256",
                "development_evaluation_receipt_sha256",
                "supporting_evaluation_file_sha256",
                "paired_prediction_file_sha256",
                "paired_prediction_receipt_file_sha256",
                "paired_prediction_receipt_sha256",
                "observed_rows_file_sha256",
                "swapped_rows_file_sha256",
            )
        )
        and isinstance(evaluation.get("eligible_rows"), int)
        and isinstance(evaluation.get("selected_rows"), int)
        and evaluation["eligible_rows"] > 0
        and 0 < evaluation["selected_rows"] <= evaluation["eligible_rows"]
        and all(
            isinstance(evaluation.get(field), (int, float))
            and not isinstance(evaluation.get(field), bool)
            and math.isfinite(float(evaluation[field]))
            for field in numeric_fields
        )
        and float(evaluation["auc"]) > 0.710
        and float(evaluation["series_bootstrap_auc_median"]) > 0.710
        and float(evaluation["ece_10"]) <= 0.08
        and isinstance(window.get("start"), str)
        and isinstance(window.get("end_exclusive"), str)
    )


def is_authorized_public_release_receipt(receipt: Mapping[str, Any]) -> bool:
    """Return whether a receipt has one accepted public-release contract."""

    if receipt.get("schema_version") == OWNER_RELEASE_RECEIPT_SCHEMA:
        return _valid_owner_release_contract(receipt)
    return (
        receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION
        and receipt.get("status") == "promoted"
        and receipt.get("authority") == "promoted"
        and tuple(receipt.get("approved_public_fields") or ()) == APPROVED_FIELDS
        and receipt.get("public_probability") is True
        and receipt.get("public_recommendation") is True
        and receipt.get("betting_odds_ev_stake") is False
    )


def _verify_owner_release_evidence(receipt: Mapping[str, Any]) -> None:
    """Bind the owner decision to the checked-in candidate and evaluations."""

    if not _valid_owner_release_contract(receipt):
        raise PromotedDraftAuthorityError("owner release receipt contract is invalid")
    root = Path(__file__).resolve().parents[2]
    candidate_path = root / OWNER_CANDIDATE_PATH
    development_path = root / OWNER_DEVELOPMENT_EVALUATION_PATH
    supporting_path = root / OWNER_SUPPORTING_EVALUATION_PATH
    try:
        candidate = _load_json(candidate_path)
        development = _load_json(development_path)
    except PromotedDraftAuthorityError as error:
        raise PromotedDraftAuthorityError("owner release evidence is unavailable") from error
    if (
        sha256_path(candidate_path) != receipt.get("candidate_artifact_sha256")
        or candidate.get("receipt_sha256") != receipt.get("candidate_receipt_sha256")
        or sha256_path(development_path)
        != receipt.get("development_evaluation_file_sha256")
        or development.get("receipt_sha256")
        != receipt.get("development_evaluation_receipt_sha256")
        or sha256_path(supporting_path)
        != receipt.get("supporting_evaluation_file_sha256")
        or development.get("gates", {}).get("passed") is not True
    ):
        raise PromotedDraftAuthorityError("owner release evidence changed")
    expected = {
        "eligible_rows": development.get("eligible_rows"),
        "selected_rows": development.get("selected_rows"),
        "coverage": development.get("coverage"),
        "auc": development.get("auc"),
        "brier": development.get("brier"),
        "log_loss": development.get("log_loss"),
        "ece_10": development.get("ece_10"),
        "series_bootstrap_auc_median": development.get(
            "series_bootstrap_auc", {}
        ).get("median"),
    }
    if dict(receipt.get("evaluation") or {}) != expected:
        raise PromotedDraftAuthorityError("owner release metrics changed")
    start = _utc(receipt["evidence_window"]["start"], label="evidence start")
    end = _utc(
        receipt["evidence_window"]["end_exclusive"], label="evidence end"
    )
    if start >= end:
        raise PromotedDraftAuthorityError("owner release evidence window is invalid")


def load_promoted_draft_authority(
    *,
    receipt_path: Path,
    expected_file_sha256: str,
    release_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a fixed manifest authority and its verified private receipt."""

    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise PromotedDraftAuthorityError("release ID is invalid")
    if not SHA256_PATTERN.fullmatch(expected_file_sha256):
        raise PromotedDraftAuthorityError("promotion receipt file SHA-256 is invalid")
    if not receipt_path.is_file() or sha256_path(receipt_path) != expected_file_sha256:
        raise PromotedDraftAuthorityError("promotion receipt file changed")

    receipt = _load_json(receipt_path)
    receipt_sha256 = receipt.get("receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        not isinstance(receipt_sha256, str)
        or not SHA256_PATTERN.fullmatch(receipt_sha256)
        or receipt_sha256 != canonical_sha256(unsigned)
    ):
        raise PromotedDraftAuthorityError("promotion receipt digest changed")

    model_version = receipt.get("model_version")
    issued_utc = receipt.get("issued_utc")
    paired_receipts = receipt.get("controlled_intervention_receipt_sha256")
    if receipt.get("schema_version") == OWNER_RELEASE_RECEIPT_SCHEMA:
        _verify_owner_release_evidence(receipt)
        if not isinstance(model_version, str) or not model_version.strip():
            raise PromotedDraftAuthorityError("owner release model version is invalid")
        _utc(issued_utc, label="owner release receipt time")
        authority = {
            "schema_version": DRAFT_AUTHORITY_SCHEMA,
            "status": "promoted",
            "authority": "promoted",
            "release_id": release_id,
            "model_version": model_version.strip(),
            "artifact_sha256": receipt["candidate_artifact_sha256"],
            "receipt_sha256": receipt_sha256,
            "issued_utc": issued_utc,
            "estimand": "prematch_map_win_probability_with_controlled_draft_intervention",
            "probability_authority": True,
            "recommendation_authority": True,
            "betting_authority": False,
            "reason": None,
        }
        return authority, receipt
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "promoted"
        or receipt.get("authority") != "promoted"
        or not isinstance(model_version, str)
        or not model_version.strip()
        or tuple(receipt.get("approved_public_fields") or ()) != APPROVED_FIELDS
        or receipt.get("public_probability") is not True
        or receipt.get("public_recommendation") is not True
        or receipt.get("betting_odds_ev_stake") is not False
        or not isinstance(receipt.get("reviewer_identity"), str)
        or not receipt["reviewer_identity"].strip()
        or any(
            not isinstance(receipt.get(field), str)
            or not SHA256_PATTERN.fullmatch(receipt[field])
            for field in BOUND_HASH_FIELDS
        )
        or not isinstance(paired_receipts, list)
        or not paired_receipts
        or any(
            not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value)
            for value in paired_receipts
        )
    ):
        raise PromotedDraftAuthorityError("promotion receipt contract is invalid")
    _utc(issued_utc, label="promotion receipt time")

    authority = {
        "schema_version": DRAFT_AUTHORITY_SCHEMA,
        "status": "promoted",
        "authority": "promoted",
        "release_id": release_id,
        "model_version": model_version.strip(),
        "artifact_sha256": receipt["candidate_artifact_sha256"],
        "receipt_sha256": receipt_sha256,
        "issued_utc": issued_utc,
        "estimand": "prematch_map_win_probability_with_controlled_draft_intervention",
        "probability_authority": True,
        "recommendation_authority": True,
        "betting_authority": False,
        "reason": None,
    }
    return authority, receipt


def validate_promoted_results_payload(
    payload: object,
    *,
    authority: Mapping[str, Any],
    maximum_results: int = 10_000,
) -> dict[str, Any]:
    """Validate one release-bound promoted result asset before publication."""

    if (
        not isinstance(payload, dict)
        or set(payload) != PROMOTED_RESULTS_KEYS
        or _contains_forbidden_key(payload)
    ):
        raise PromotedDraftAuthorityError("promoted Draft result asset is invalid")
    results = payload.get("results")
    if (
        payload.get("schema_version") != PROMOTED_RESULTS_SCHEMA
        or payload.get("authority") != "promoted"
        or payload.get("release_id") != authority.get("release_id")
        or payload.get("model_version") != authority.get("model_version")
        or payload.get("receipt_sha256") != authority.get("receipt_sha256")
        or not isinstance(results, dict)
        or not results
        or len(results) > maximum_results
    ):
        raise PromotedDraftAuthorityError("promoted Draft result asset is invalid")

    for game_uid, result in results.items():
        if (
            not isinstance(game_uid, str)
            or not game_uid
            or not isinstance(result, dict)
            or set(result) != PUBLIC_RESULT_KEYS
        ):
            raise PromotedDraftAuthorityError("promoted Draft result row is invalid")
        probabilities = result.get("match_win_probability")
        controlled = result.get("controlled_draft_score")
        evidence = result.get("evidence_window")
        if (
            result.get("schema_version") != PUBLIC_RESULT_SCHEMA
            or result.get("authority") != "promoted"
            or result.get("release_id") != authority.get("release_id")
            or result.get("model_version") != authority.get("model_version")
            or result.get("receipt_sha256") != authority.get("receipt_sha256")
            or not isinstance(probabilities, dict)
            or set(probabilities) != PROBABILITY_KEYS
            or not isinstance(controlled, dict)
            or set(controlled) != CONTROLLED_SCORE_KEYS
            or not isinstance(evidence, dict)
            or set(evidence) != EVIDENCE_WINDOW_KEYS
        ):
            raise PromotedDraftAuthorityError("promoted Draft result row is invalid")
        evidence_start = evidence.get("start")
        evidence_end = evidence.get("end")
        if _utc(evidence_start, label="promoted Draft evidence window") >= _utc(
            evidence_end, label="promoted Draft evidence window"
        ):
            raise PromotedDraftAuthorityError("promoted Draft evidence window is invalid")
        blue = probabilities.get("Blue")
        red = probabilities.get("Red")
        model_units = controlled.get("model_units")
        edge = controlled.get("edge_percentage_points")
        if (
            isinstance(blue, bool)
            or not isinstance(blue, (int, float))
            or isinstance(red, bool)
            or not isinstance(red, (int, float))
            or not math.isfinite(float(blue))
            or not math.isfinite(float(red))
            or not 0.0 <= float(blue) <= 1.0
            or not 0.0 <= float(red) <= 1.0
            or abs(float(blue) + float(red) - 1.0) > 1e-9
            or isinstance(model_units, bool)
            or not isinstance(model_units, (int, float))
            or isinstance(edge, bool)
            or not isinstance(edge, (int, float))
            or not math.isfinite(float(model_units))
            or not math.isfinite(float(edge))
            or abs(float(edge)) > 100.0
        ):
            raise PromotedDraftAuthorityError("promoted Draft result numbers are invalid")
        expected_draft_side = (
            "Blue" if model_units > 0 else "Red" if model_units < 0 else "Even"
        )
        expected_recommendation = "Blue" if blue >= red else "Red"
        intervention_receipt = controlled.get("intervention_receipt_sha256")
        isolated_probability = controlled.get("isolated_blue_draft_probability")
        fixed_strength_probability = controlled.get(
            "fixed_strength_blue_win_probability"
        )
        if (
            controlled.get("stronger_draft") != expected_draft_side
            or ((model_units == 0) != (edge == 0))
            or model_units * edge < 0
            or controlled.get("method") != "role_matched_champion_swap"
            or not isinstance(intervention_receipt, str)
            or not SHA256_PATTERN.fullmatch(intervention_receipt)
            or not isinstance(controlled.get("explanation"), str)
            or not controlled["explanation"].strip()
            or isinstance(isolated_probability, bool)
            or not isinstance(isolated_probability, (int, float))
            or not 0.0 <= float(isolated_probability) <= 1.0
            or isinstance(fixed_strength_probability, bool)
            or not isinstance(fixed_strength_probability, (int, float))
            or not 0.0 <= float(fixed_strength_probability) <= 1.0
            or result.get("side_recommendation") != expected_recommendation
        ):
            raise PromotedDraftAuthorityError("promoted Draft result direction is invalid")
    return payload
