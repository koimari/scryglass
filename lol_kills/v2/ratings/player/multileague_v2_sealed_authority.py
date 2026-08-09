"""Independent authority gate for one-time v2 sealed-final evaluation.

The authority receipt can authorize opening one exact outcome cohort for one
evaluation run.  It cannot authorize production ratings, probabilities, odds,
expected value, recommendations, or betting.  The receipt's raw SHA-256 must be
pinned outside the receipt.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import multileague_v2_protocol_equal_series as protocol
from . import multileague_v2_runner_equal_series as runner


SCHEMA_VERSION = "scryglass:multileague-rating-v2-sealed-opening-authority:v1"
RECEIPT_LOCATOR = Path(
    "data/lol/private_rating_authority/multileague-v2-sealed-opening.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_PRIVATE_RATING_SEALED_OPENING_SHA256"
OUTPUT_LOCATOR = (
    "data/lol/v2/models/player/multileague-v2/sealed-final-evaluation-v1.json"
)


class SealedOpeningAuthorityError(ValueError):
    """The sealed-opening receipt is malformed, unpinned, stale, or mismatched."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SealedOpeningAuthorityError(f"{label} must be a lowercase SHA-256")
    return value


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SealedOpeningAuthorityError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealedOpeningAuthorityError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise SealedOpeningAuthorityError(f"{label} must contain an object")
    return value


def _read_current_artifact(
    root: Path,
    locator: Path,
) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = (root / locator).read_bytes()
    except OSError as error:
        raise SealedOpeningAuthorityError(
            f"bound artifact is unavailable: {locator}"
        ) from error
    return raw, _load_object(raw, locator.as_posix())


def current_expected_bindings(
    root: Path | str = Path("."),
) -> dict[str, Any]:
    repo_root = Path(root)
    protocol_raw, protocol_payload = _read_current_artifact(
        repo_root,
        protocol.DEFAULT_OUTPUT,
    )
    candidate_raw, candidate_payload = _read_current_artifact(
        repo_root,
        runner.DEFAULT_OUTPUT,
    )
    try:
        protocol_payload = protocol.validate_equal_series_protocol_lock(
            protocol_payload,
            root=repo_root,
        )
        candidate_payload = runner.validate_equal_series_adaptive_artifact(
            candidate_payload,
            root=repo_root,
        )
    except (
        protocol.EqualSeriesProtocolError,
        runner.EqualSeriesRunnerError,
    ) as error:
        raise SealedOpeningAuthorityError(
            "current protocol or candidate artifact is invalid"
        ) from error
    selected = (candidate_payload.get("selection") or {}).get(
        "selected_candidate_id"
    )
    if not isinstance(selected, str) or not selected:
        raise SealedOpeningAuthorityError("no adaptive candidate is selected")
    if (
        (candidate_payload.get("sealed_final") or {}).get("opened") is not False
        or (candidate_payload.get("sealed_final") or {}).get("targets_accessed")
        is not False
    ):
        raise SealedOpeningAuthorityError("sealed-final isolation is already lost")
    artifact_input = candidate_payload.get("input") or {}
    return {
        "protocol_locator": protocol.DEFAULT_OUTPUT.as_posix(),
        "protocol_raw_sha256": _sha256(protocol_raw),
        "protocol_artifact_sha256": protocol_payload["artifact_sha256"],
        "candidate_artifact_locator": runner.DEFAULT_OUTPUT.as_posix(),
        "candidate_artifact_raw_sha256": _sha256(candidate_raw),
        "candidate_artifact_sha256": candidate_payload["artifact_sha256"],
        "selected_candidate_id": selected,
        "maps_sha256": artifact_input["maps_sha256"],
        "players_sha256": artifact_input["players_sha256"],
        "cluster_partition_sha256": artifact_input["cluster_partition_sha256"],
        "sealed_selected_metadata_sha256": artifact_input[
            "sealed_selected_metadata_sha256"
        ],
        "sealed_series": artifact_input["sealed_metadata_series"],
        "sealed_maps": artifact_input["sealed_metadata_maps"],
        "authorized_output_locator": OUTPUT_LOCATOR,
    }


def validate_sealed_opening_authority(
    payload: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SealedOpeningAuthorityError("authority receipt must be an object")
    value = dict(payload)
    required_top = {
        "schema_version",
        "authority_id",
        "status",
        "scope",
        "reviewer_id",
        "reviewed_at",
        "independence_attestation",
        "bindings",
        "one_time_run",
        "claim_ceiling",
    }
    if set(value) != required_top:
        raise SealedOpeningAuthorityError("authority receipt fields are not exact")
    if value["schema_version"] != SCHEMA_VERSION:
        raise SealedOpeningAuthorityError("authority schema version is unsupported")
    if value["status"] != "APPROVED" or value["scope"] != (
        "ONE_TIME_SEALED_FINAL_EVALUATION_ONLY"
    ):
        raise SealedOpeningAuthorityError("authority status or scope is invalid")
    if not isinstance(value["authority_id"], str) or not value["authority_id"].strip():
        raise SealedOpeningAuthorityError("authority_id is required")
    if not isinstance(value["reviewer_id"], str) or not value["reviewer_id"].strip():
        raise SealedOpeningAuthorityError("reviewer_id is required")
    try:
        reviewed_at = datetime.fromisoformat(str(value["reviewed_at"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise SealedOpeningAuthorityError("reviewed_at is invalid") from error
    if reviewed_at.tzinfo is None:
        raise SealedOpeningAuthorityError("reviewed_at must include a timezone")

    attestation = value["independence_attestation"]
    expected_attestation = {
        "reviewer_not_model_author_or_candidate_selector": True,
        "review_used_only_pinned_adaptive_evidence": True,
        "sealed_final_outcomes_not_accessed_before_approval": True,
        "approval_was_not_generated_by_the_evaluated_system": True,
    }
    if attestation != expected_attestation:
        raise SealedOpeningAuthorityError("independence attestation is incomplete")
    if value["bindings"] != dict(expected_bindings):
        raise SealedOpeningAuthorityError("authority bindings do not match current evidence")
    for key, item in expected_bindings.items():
        if key.endswith("sha256"):
            _require_sha256(item, f"bindings.{key}")

    one_time = value["one_time_run"]
    if not isinstance(one_time, Mapping) or set(one_time) != {
        "run_id",
        "authorized_output_locator",
        "no_clobber_required",
        "second_holdout_opening_prohibited",
    }:
        raise SealedOpeningAuthorityError("one-time run contract is incomplete")
    if not isinstance(one_time["run_id"], str) or not one_time["run_id"].strip():
        raise SealedOpeningAuthorityError("one-time run_id is required")
    if (
        one_time["authorized_output_locator"] != OUTPUT_LOCATOR
        or one_time["no_clobber_required"] is not True
        or one_time["second_holdout_opening_prohibited"] is not True
    ):
        raise SealedOpeningAuthorityError("one-time output contract changed")
    expected_ceiling = {
        "sealed_evaluation_authorized": True,
        "production_rating_authorized": False,
        "match_probability_authorized": False,
        "fair_odds_authorized": False,
        "expected_value_authorized": False,
        "bet_recommendation_authorized": False,
    }
    if value["claim_ceiling"] != expected_ceiling:
        raise SealedOpeningAuthorityError("authority claim ceiling changed")
    return value


def load_pinned_sealed_opening_authority(
    path: Path,
    *,
    expected_bindings: Mapping[str, Any],
    external_sha256: str,
) -> dict[str, Any]:
    expected_sha256 = _require_sha256(external_sha256, "external authority digest")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SealedOpeningAuthorityError("authority receipt is unavailable") from error
    if _sha256(raw) != expected_sha256:
        raise SealedOpeningAuthorityError("authority receipt does not match external pin")
    value = validate_sealed_opening_authority(
        _load_object(raw, "authority receipt"),
        expected_bindings=expected_bindings,
    )
    return {
        "status": "registered",
        "receipt": value,
        "receipt_raw_sha256": expected_sha256,
        "authority_id": value["authority_id"],
        "run_id": value["one_time_run"]["run_id"],
        "sealed_evaluation_authorized": True,
        "production_rating_authorized": False,
        "match_probability_authorized": False,
        "betting_decision_authorized": False,
        "blockers": [
            "sealed_final_gate_not_yet_evaluated",
            "production_rating_not_authorized_by_opening_receipt",
            "probability_not_authorized_by_opening_receipt",
            "betting_not_authorized_by_opening_receipt",
        ],
    }


def inspect_sealed_opening_authority(
    root: Path | str = Path("."),
    *,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    repo_root = Path(root)
    external = environment.get(EXTERNAL_SHA256_ENV)
    if not external:
        return {
            "status": "unavailable",
            "receipt_locator": RECEIPT_LOCATOR.as_posix(),
            "receipt_present": (repo_root / RECEIPT_LOCATOR).is_file(),
            "external_digest_pin_present": False,
            "sealed_evaluation_authorized": False,
            "production_rating_authorized": False,
            "match_probability_authorized": False,
            "betting_decision_authorized": False,
            "blockers": ["external_sealed_opening_authority_digest_missing"],
        }
    try:
        expected = current_expected_bindings(repo_root)
        return load_pinned_sealed_opening_authority(
            repo_root / RECEIPT_LOCATOR,
            expected_bindings=expected,
            external_sha256=external,
        )
    except SealedOpeningAuthorityError as error:
        return {
            "status": "unavailable",
            "receipt_locator": RECEIPT_LOCATOR.as_posix(),
            "receipt_present": (repo_root / RECEIPT_LOCATOR).is_file(),
            "external_digest_pin_present": True,
            "sealed_evaluation_authorized": False,
            "production_rating_authorized": False,
            "match_probability_authorized": False,
            "betting_decision_authorized": False,
            "error": str(error),
            "blockers": ["sealed_opening_authority_invalid_or_mismatched"],
        }


__all__ = [
    "EXTERNAL_SHA256_ENV",
    "OUTPUT_LOCATOR",
    "RECEIPT_LOCATOR",
    "SCHEMA_VERSION",
    "SealedOpeningAuthorityError",
    "current_expected_bindings",
    "inspect_sealed_opening_authority",
    "load_pinned_sealed_opening_authority",
    "validate_sealed_opening_authority",
]
