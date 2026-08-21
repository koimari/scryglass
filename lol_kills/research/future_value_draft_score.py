"""Research-only four-way Draft Score composition contract.

This module gives the future-value work one small downstream interface.  It
does not replace the public Draft Score.  Each variant uses the same atomized
composition values.  V2 adds prior player form.  V3 adds phase curves and
curve-by-atom interactions.  V4 adds both families.

The input is a pre-match map ledger.  Every producer row is bound to one
accepted census and one earlier fold.  Checkpoint observations and final map
values are rejected from the feature matrix.  A result is always marked
research-only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import inspect
import json
import math
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from lol_kills.research.future_phase_curve import (
    PHASE_SHAPE_FEATURES,
    PHASE_SHAPE_AVAILABILITY_FEATURES,
    PHASE_SHAPE_INVARIANT_FEATURES,
    PHASE_SHAPE_SIGNED_FEATURES,
    phase_shape_features,
)
from lol_kills.research.future_value_rating import (
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    FUTURE_PLAYER_FORM_SIDE_FEATURES,
    RatingVariant,
    rating_variant_config_sha256,
)
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


SCHEMA_VERSION = "scryglass:future-value-draft-score:v1"
AUTHORITY = False
SOURCE_KIND = "oracle_elixir_research_only"
ROLES = ("top", "jungle", "mid", "bot", "support")


class FutureValueDraftScoreError(ValueError):
    """Raised when a Draft Score variant input is unsafe or malformed."""


DraftScoreError = FutureValueDraftScoreError


class DraftScoreVariant(str, Enum):
    """Stable names for the four research variants."""

    CURRENT_ONLY = RatingVariant.CURRENT_ONLY.value
    FUTURE_PLAYER_FORM = RatingVariant.FUTURE_PLAYER_FORM.value
    SCALING_CURVE = RatingVariant.SCALING_CURVE.value
    BOTH = RatingVariant.BOTH.value


VARIANT_NAMES = tuple(value.value for value in DraftScoreVariant)


# These are supplied by the atomized composition producer.  They are map
# signed values.  The fields remain in every variant and their values must be
# byte-equivalent after canonical numeric serialisation.
STATIC_COMPOSITION_COMPONENTS = (
    "base",
    "ally_synergy",
    "enemy_counter",
    "same_role",
    "archetype_interactions",
)
STATIC_COMPOSITION_FEATURES = tuple(
    f"composition_{component}_logit" for component in STATIC_COMPOSITION_COMPONENTS
)
ATOMIZED_COMPOSITION_COMPONENTS = STATIC_COMPOSITION_COMPONENTS
ATOMIZED_COMPOSITION_FEATURES = STATIC_COMPOSITION_FEATURES


PHASE_RAW_FEATURES = tuple(
    f"forecast_{metric}_diff_{checkpoint}"
    for checkpoint in (10, 15, 20, 25)
    for metric in ("gold", "xp")
)
PHASE_SHAPE_SIGNED_FEATURES = tuple(PHASE_SHAPE_SIGNED_FEATURES)
PHASE_SHAPE_INVARIANT_FEATURES = tuple(PHASE_SHAPE_INVARIANT_FEATURES)
# Shape invariants describe the producer output.  They are gates and receipt
# fields.  They are not coordinates in the antisymmetric Draft Score matrix.
PHASE_SHAPE_DIAGNOSTIC_FEATURES = PHASE_SHAPE_INVARIANT_FEATURES
PHASE_MODEL_FEATURES = (*PHASE_RAW_FEATURES, *PHASE_SHAPE_SIGNED_FEATURES)
PHASE_FEATURES = (*PHASE_MODEL_FEATURES, *PHASE_SHAPE_DIAGNOSTIC_FEATURES)


CURVE_ATOM_FAMILIES = ("role", "synergy", "counter")
CURVE_ATOM_INTERACTION_FEATURES = tuple(
    f"curve_atom_{family}_{role}"
    for family in CURVE_ATOM_FAMILIES
    for role in ROLES
)
CURVE_INTERACTION_FEATURES = CURVE_ATOM_INTERACTION_FEATURES


SOURCE_RECEIPT_SCHEMA = "scryglass:future-value-rating-source:v1"
STATIC_ATOM_RECEIPT_SCHEMA = "scryglass:atomized-composition-producer:v1"
COEFFICIENT_RECEIPT_SCHEMA = "scryglass:future-value-draft-score-coefficients:v1"
MODEL_RECEIPT_SCHEMA = "scryglass:future-value-draft-score-model-receipt:v1"
MODEL_ARTIFACT_SCHEMA = "scryglass:future-value-draft-score-linear-model:v1"
MODEL_TRAINER_ID = "scryglass.future_value_draft_score.linear_logit.v1"
PREDICTION_FEATURE_ARTIFACT_SCHEMA = (
    "scryglass:future-value-draft-score-feature-matrix:v1"
)
PREDICTION_LEDGER_SCHEMA = "scryglass:future-value-draft-score-prediction-ledger:v2"
_ALLOWED_PRODUCER_TIMINGS = frozenset(
    {"pregame_strict_prior", "cross_fitted_pregame", "strict_prior_pregame"}
)

# These are closed contracts.  A receipt with a valid digest and an unknown
# field is still caller-authored data, so it must not enter the scorer.
_SOURCE_RECEIPT_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "accepted_game_ids",
        "source_files",
        "authority",
        "receipt_sha256",
    }
)
_SOURCE_RECEIPT_ALLOWED_FIELDS = frozenset(
    {
        *_SOURCE_RECEIPT_REQUIRED_FIELDS,
        "model_eligible_game_count",
        "model_eligible_identity_sha256",
        "model_eligible_game_ids",
        "source_rows",
        "source_extra_game_ids",
        "identity_coverage",
        "checkpoint_coverage",
        "model_exclusions",
        "model_contract",
        "release_id",
        "pack_id",
    }
)
_SOURCE_RECEIPT_ALLOWED_STATUSES = frozenset(
    {"accepted_source_bound_development_only", "verified_public_pack_source"}
)
_AUTHORITY_ALLOWED_FIELDS = frozenset(
    {
        "research_only",
        "public_player_rating",
        "public_team_rating",
        "public_probability",
        "promotion",
        "merge",
        "deployment",
    }
)
_SOURCE_FILE_ALLOWED_FIELDS = frozenset({"locator", "path", "bytes", "sha256", "year"})
_TRUSTED_PRODUCER_NAMES = frozenset(
    {
        "public_descriptive_draft_records",
        "public_crossfit_draft_score",
    }
)
_STATIC_ATOM_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "producer_name",
        "producer_family",
        "artifact_locator",
        "artifact_bytes",
        "artifact_sha256",
        "artifact_receipt_locator",
        "artifact_receipt_bytes",
        "artifact_receipt_sha256",
        "source_receipt_sha256",
        "source_identity_sha256",
        "feature_names",
        "component_values_sha256",
        "receipt_sha256",
    }
)
_STATIC_ATOM_ALLOWED_FIELDS = frozenset(
    {
        *_STATIC_ATOM_REQUIRED_FIELDS,
        "authority_receipt_sha256",
        "authority_receipt_locator",
        "authority_receipt_bytes",
        "model_artifact_sha256",
        "recipe_sha256",
        "scorer_code_sha256",
        "release_id",
        "fit_through",
        "chronological_evaluation_suitable",
        "chronological_evaluation_reason",
        "coverage_game_count",
        "coverage_game_ids",
        "coverage_identity_sha256",
    }
)
_PRODUCER_RECEIPT_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "source_receipt_sha256",
        "source_identity_sha256",
        "accepted_game_count",
        "accepted_game_ids",
        "producer_name",
        "producer_family",
        "fit_game_count",
        "fit_game_ids",
        "fit_game_identity_sha256",
        "fit_window_start",
        "fit_window_end",
        "fit_game_dates",
        "fold_id",
        "series_safe_evidence",
        "producer_timing",
        "artifact_locator",
        "artifact_bytes",
        "artifact_sha256",
        "artifact_receipt_locator",
        "artifact_receipt_bytes",
        "artifact_receipt_sha256",
        "receipt_sha256",
    }
)
_PRODUCER_RECEIPT_ALLOWED_FIELDS = frozenset(_PRODUCER_RECEIPT_REQUIRED_FIELDS)
_PREDICTION_LEDGER_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "source_receipt_sha256",
        "source_identity_sha256",
        "game_ids",
        "fold_id",
        "fit_id",
        "model_id",
        "fit_game_ids",
        "fit_game_identity_sha256",
        "coefficient_sha256",
        "model_receipt_locator",
        "model_receipt_bytes",
        "model_receipt_sha256",
        "model_artifact_locator",
        "model_artifact_bytes",
        "model_artifact_sha256",
        "model_implementation_sha256",
        "feature_artifact_locator",
        "feature_artifact_bytes",
        "feature_artifact_sha256",
        "feature_names",
        "feature_rows_sha256",
        "row_digest_sha256",
        "artifact_locator",
        "artifact_bytes",
        "artifact_sha256",
        "receipt_sha256",
    }
)
_PREDICTION_LEDGER_ALLOWED_FIELDS = frozenset(
    {*_PREDICTION_LEDGER_REQUIRED_FIELDS, "authority", "rows"}
)
_MODEL_RECEIPT_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "model_id",
        "model_version",
        "variant",
        "source_receipt_sha256",
        "source_identity_sha256",
        "fold_id",
        "fit_game_ids",
        "fit_game_identity_sha256",
        "fit_id",
        "coefficient_sha256",
        "artifact_locator",
        "artifact_bytes",
        "artifact_sha256",
        "implementation_sha256",
        "authority",
        "receipt_sha256",
    }
)
_MODEL_RECEIPT_ALLOWED_FIELDS = frozenset(_MODEL_RECEIPT_REQUIRED_FIELDS)


_TARGET_TOKENS = frozenset(
    {
        "target",
        "outcome",
        "result",
        "winner",
        "winning_side",
        "actual",
        "observed",
        "label",
        "y_blue_win",
        "blue_win",
        "red_win",
    }
)
_FINAL_TOKENS = frozenset(
    {
        "cspm",
        "cspermin",
        "dpm",
        "damageshare",
        "damagepermin",
        "kills",
        "deaths",
        "assists",
        "totalgold",
        "earnedgold",
        "gamelength",
        "duration",
        "visionscore",
    }
)
_RAW_CHECKPOINT_RE = re.compile(
    r"(?:gold|xp|cs|kills|assists|deaths)(?:at|_at|_diff_|diff_at|diff)(?:10|15|20|25)$",
    re.IGNORECASE,
)
_HEX_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FutureValueDraftScoreError("value is not canonical finite JSON") from error


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _frame_rows_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    rows: list[dict[str, Any]] = []
    for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        canonical: dict[str, Any] = {}
        for name, value in zip(columns, row):
            if pd.isna(value):
                canonical[str(name)] = None
            else:
                canonical[str(name)] = float(value) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
        rows.append(canonical)
    return _sha256(rows)


def _timestamp(value: Any, field: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise FutureValueDraftScoreError(f"{field} is not a timestamp") from error
    if pd.isna(result) or result.tzinfo is None:
        raise FutureValueDraftScoreError(f"{field} must be a timezone-aware timestamp")
    return result.tz_convert("UTC")


def _timestamp_text(value: Any, field: str) -> str:
    return _timestamp(value, field).isoformat().replace("+00:00", "Z")


def _canonical_variant(value: DraftScoreVariant | RatingVariant | str) -> DraftScoreVariant:
    if isinstance(value, DraftScoreVariant):
        return value
    if isinstance(value, RatingVariant):
        return DraftScoreVariant(value.value)
    try:
        return DraftScoreVariant(str(value).strip())
    except ValueError as error:
        raise FutureValueDraftScoreError(f"unknown Draft Score variant: {value!r}") from error


def _require_hash(value: Any, field: str) -> str:
    text = str(value or "").lower()
    if _HEX_RE.fullmatch(text) is None:
        raise FutureValueDraftScoreError(f"{field} must be a 64-character SHA-256")
    return text


def _normalise_ids(values: Iterable[object], field: str) -> tuple[str, ...]:
    ids = canonical_game_ids(values)
    if not ids:
        raise FutureValueDraftScoreError(f"{field} is empty")
    return ids


def _regular_file(path: Path | str, field: str) -> Path:
    """Return one regular, non-symlink file with no symlinked parent."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise FutureValueDraftScoreError(f"{field} is missing or unsafe")
    current = candidate
    # macOS exposes /tmp and /var as system aliases.  They do not represent a
    # caller-controlled symlink chain.  Check every lower path component.
    system_aliases = {Path("/tmp"), Path("/var"), Path("/private/tmp"), Path("/private/var")}
    while True:
        if current.is_symlink() and current not in system_aliases:
            raise FutureValueDraftScoreError(f"{field} uses a symlink path")
        parent = current.parent
        if parent == current:
            break
        if current in system_aliases:
            break
        current = parent
    return candidate.resolve()


def _safe_locator(
    locator: object,
    *,
    base: Path,
    field: str,
) -> Path:
    text = str(locator or "").strip()
    if not text:
        raise FutureValueDraftScoreError(f"{field} locator is required")
    raw = Path(text)
    if ".." in raw.parts:
        raise FutureValueDraftScoreError(f"{field} locator escapes its root")
    candidate = raw if raw.is_absolute() else base / raw
    return _regular_file(candidate, field)


def _verify_file_digest(
    path: Path,
    *,
    expected_bytes: object,
    expected_sha256: object,
    field: str,
) -> str:
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise FutureValueDraftScoreError(f"{field} byte count is invalid")
    expected_hash = _require_hash(expected_sha256, f"{field} sha256")
    raw = path.read_bytes()
    actual_hash = hashlib.sha256(raw).hexdigest()
    if len(raw) != expected_bytes or actual_hash != expected_hash:
        raise FutureValueDraftScoreError(f"{field} bytes or SHA-256 changed")
    return actual_hash


def _load_json_file(path: Path, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureValueDraftScoreError(f"{field} cannot be read") from error
    if not isinstance(value, dict):
        raise FutureValueDraftScoreError(f"{field} must be a JSON object")
    return value, raw


def _linear_model_logits(
    feature_names: Sequence[str],
    coefficients: Mapping[str, float],
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """Apply the pinned linear-logit prediction contract."""

    weights = np.asarray([float(coefficients[name]) for name in feature_names], dtype=float)
    return np.asarray(feature_matrix, dtype=float) @ weights


def draft_score_trainer_implementation_sha256() -> str:
    """Return the pinned identity of the fitted linear-logit contract."""

    source = inspect.getsource(_linear_model_logits).encode("utf-8")
    return hashlib.sha256(MODEL_TRAINER_ID.encode("utf-8") + b"\n" + source).hexdigest()


def _descriptive_artifact_components(
    payload: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    games = payload.get("games")
    if isinstance(games, Mapping):
        raw_rows = [(str(game_id), value) for game_id, value in games.items()]
    elif isinstance(games, list):
        raw_rows = []
        for value in games:
            if not isinstance(value, Mapping):
                continue
            game_id = value.get("game_uid", value.get("game_id", value.get("id")))
            if game_id is not None:
                raw_rows.append((str(game_id), value))
    else:
        raise FutureValueDraftScoreError("descriptive producer artifact games are missing")
    if not raw_rows or len({game_id for game_id, _value in raw_rows}) != len(raw_rows):
        raise FutureValueDraftScoreError("descriptive producer artifact game IDs are invalid")
    parsed: dict[str, dict[str, float]] = {}
    required = {*STATIC_COMPOSITION_COMPONENTS, "total"}
    for game_id, game in raw_rows:
        if not isinstance(game, Mapping):
            raise FutureValueDraftScoreError("descriptive producer artifact row is invalid")
        edge = game.get("edge_components")
        if not isinstance(edge, Mapping) or set(edge) != required:
            raise FutureValueDraftScoreError(
                f"descriptive producer artifact edge components are incomplete: {game_id}"
            )
        values: dict[str, float] = {}
        for component in STATIC_COMPOSITION_COMPONENTS:
            try:
                value = float(edge[component])
            except (TypeError, ValueError) as error:
                raise FutureValueDraftScoreError(
                    f"descriptive producer artifact component is invalid: {game_id}"
                ) from error
            if not math.isfinite(value):
                raise FutureValueDraftScoreError(
                    f"descriptive producer artifact component is invalid: {game_id}"
                )
            values[f"composition_{component}_logit"] = value
        try:
            total = float(edge["total"])
        except (TypeError, ValueError) as error:
            raise FutureValueDraftScoreError(
                f"descriptive producer artifact total is invalid: {game_id}"
            ) from error
        if not math.isfinite(total) or abs(total - sum(values.values())) > 1e-5:
            raise FutureValueDraftScoreError(
                f"descriptive producer artifact total changed: {game_id}"
            )
        parsed[game_id] = values
    return parsed


def _validate_producer_artifact_shape(
    path: Path,
    *,
    producer_name: str,
    source_identity_sha256: str,
    expected_game_ids: Sequence[str],
    expected_component_frame: pd.DataFrame | None = None,
    allow_coverage_subset: bool = False,
) -> None:
    """Check that a trusted producer receipt names its real public artifact."""

    payload, _raw = _load_json_file(path, "producer artifact")
    expected_identity = _require_hash(source_identity_sha256, "source_identity_sha256")
    if producer_name == "public_descriptive_draft_records":
        if (
            payload.get("schema_version") != "scryglass:draft-records:v1"
            or payload.get("authority") != "descriptive"
            or payload.get("estimand") != "composition_only"
        ):
            raise FutureValueDraftScoreError("descriptive producer artifact schema is invalid")
        if str(payload.get("source_identity_sha256") or "").lower() != expected_identity:
            raise FutureValueDraftScoreError("producer artifact source identity changed")
        components = _descriptive_artifact_components(payload)
        artifact_ids = tuple(sorted(components))
    elif producer_name == "public_crossfit_draft_score":
        rows = payload.get("rows", payload.get("predictions", payload.get("results")))
        if isinstance(rows, Mapping):
            artifact_ids = tuple(sorted(str(value) for value in rows))
        elif isinstance(rows, list):
            artifact_ids = tuple(
                sorted(
                    str(value.get("game_uid", value.get("game_id", value.get("id"))))
                    for value in rows
                    if isinstance(value, Mapping)
                    and value.get("game_uid", value.get("game_id", value.get("id"))) is not None
                )
            )
        else:
            raise FutureValueDraftScoreError("cross-fit producer artifact rows are missing")
    else:
        raise FutureValueDraftScoreError("producer is not a trusted Draft Score adapter")
    expected = tuple(sorted(str(value) for value in expected_game_ids))
    if not expected:
        raise FutureValueDraftScoreError("producer artifact game census changed")
    if allow_coverage_subset:
        if not set(artifact_ids).issubset(set(expected)):
            raise FutureValueDraftScoreError("producer artifact game census changed")
    elif not set(expected).issubset(set(artifact_ids)):
        raise FutureValueDraftScoreError("producer artifact game census changed")
    if expected_component_frame is not None:
        if producer_name != "public_descriptive_draft_records":
            raise FutureValueDraftScoreError("component parity requires descriptive artifacts")
        required_columns = {"game_id", *STATIC_COMPOSITION_FEATURES}
        if not required_columns.issubset(expected_component_frame.columns):
            raise FutureValueDraftScoreError("caller atom frame is incomplete")
        caller_ids = tuple(str(value) for value in expected_component_frame["game_id"])
        if len(set(caller_ids)) != len(caller_ids) or set(caller_ids) != set(expected):
            raise FutureValueDraftScoreError("caller atom frame game census changed")
        for row in expected_component_frame.itertuples(index=False):
            game_id = str(row.game_id)
            artifact_row = components.get(game_id)
            if artifact_row is None:
                raise FutureValueDraftScoreError("producer artifact game census changed")
            for feature in STATIC_COMPOSITION_FEATURES:
                try:
                    caller_value = float(getattr(row, feature))
                except (TypeError, ValueError) as error:
                    raise FutureValueDraftScoreError("caller atom value is invalid") from error
                if not math.isfinite(caller_value) or caller_value != artifact_row[feature]:
                    raise FutureValueDraftScoreError(
                        "atomized composition value differs from producer artifact: "
                        f"{game_id} {feature}"
                    )


def _validate_source_receipt_payload(
    source_receipt: Mapping[str, Any] | None,
    *,
    expected_game_ids: Iterable[object] | None = None,
    source_receipt_path: Path | str | None = None,
    source_root: Path | str | None = None,
) -> dict[str, Any]:
    """Verify the canonical accepted-census receipt used by every producer.

    A hash-shaped string is not source evidence.  The complete payload must
    verify before a Draft Score design can be built.
    """

    if not isinstance(source_receipt, Mapping):
        raise FutureValueDraftScoreError("canonical verified source receipt is required")
    unknown = sorted(set(source_receipt) - _SOURCE_RECEIPT_ALLOWED_FIELDS)
    if unknown:
        raise FutureValueDraftScoreError(
            "canonical source receipt has unknown fields: " + ", ".join(unknown)
        )
    missing = sorted(_SOURCE_RECEIPT_REQUIRED_FIELDS - set(source_receipt))
    if missing:
        raise FutureValueDraftScoreError(
            "canonical source receipt is incomplete: " + ", ".join(missing)
        )
    if str(source_receipt["schema_version"]) != SOURCE_RECEIPT_SCHEMA:
        raise FutureValueDraftScoreError("canonical source receipt schema is invalid")
    if str(source_receipt["status"]) not in _SOURCE_RECEIPT_ALLOWED_STATUSES:
        raise FutureValueDraftScoreError("canonical source receipt status is invalid")
    raw_ids = source_receipt["accepted_game_ids"]
    if not isinstance(raw_ids, list) or not all(isinstance(value, str) for value in raw_ids):
        raise FutureValueDraftScoreError("canonical source receipt accepted IDs are invalid")
    accepted = tuple(raw_ids)
    if accepted != canonical_game_ids(accepted):
        raise FutureValueDraftScoreError("canonical source receipt accepted IDs are not sorted and unique")
    if not accepted:
        raise FutureValueDraftScoreError("canonical source receipt accepted IDs are empty")
    count = source_receipt["source_game_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count != len(accepted):
        raise FutureValueDraftScoreError("canonical source receipt game count is invalid")
    expected_identity = identity_sha256(accepted)
    if str(source_receipt["source_identity_sha256"]).lower() != expected_identity:
        raise FutureValueDraftScoreError("canonical source receipt census identity is invalid")
    model_fields = {
        "model_eligible_game_count",
        "model_eligible_identity_sha256",
        "model_eligible_game_ids",
    }
    present_model_fields = model_fields & set(source_receipt)
    if present_model_fields and present_model_fields != model_fields:
        raise FutureValueDraftScoreError(
            "canonical source receipt model-eligible census is incomplete"
        )
    if present_model_fields:
        raw_model_ids = source_receipt["model_eligible_game_ids"]
        if not isinstance(raw_model_ids, list) or not all(
            isinstance(value, str) for value in raw_model_ids
        ):
            raise FutureValueDraftScoreError(
                "canonical source receipt model-eligible IDs are invalid"
            )
        model_ids = tuple(raw_model_ids)
        if (
            model_ids != canonical_game_ids(model_ids)
            or not set(model_ids).issubset(set(accepted))
            or source_receipt["model_eligible_game_count"] != len(model_ids)
            or str(source_receipt["model_eligible_identity_sha256"]).lower()
            != identity_sha256(model_ids)
        ):
            raise FutureValueDraftScoreError(
                "canonical source receipt model-eligible census is invalid"
            )
    receipt_hash = _require_hash(source_receipt["receipt_sha256"], "source receipt_sha256")
    payload = dict(source_receipt)
    payload.pop("receipt_sha256", None)
    if _sha256(payload) != receipt_hash:
        raise FutureValueDraftScoreError("canonical source receipt hash does not match payload")
    _timestamp(source_receipt["source_as_of"], "source_as_of")
    if expected_game_ids is not None and _normalise_ids(expected_game_ids, "expected game IDs") != accepted:
        raise FutureValueDraftScoreError("source receipt accepted census does not match binding")
    authority = source_receipt.get("authority")
    if not isinstance(authority, Mapping) or set(authority) - _AUTHORITY_ALLOWED_FIELDS:
        raise FutureValueDraftScoreError("source receipt authority is invalid")
    if authority.get("research_only") is not True:
        raise FutureValueDraftScoreError("source receipt authority is not research-only")
    if any(value is not False for key, value in authority.items() if key != "research_only"):
        raise FutureValueDraftScoreError("source receipt grants authority")

    receipt_path: Path | None = None
    if source_receipt_path is None:
        raise FutureValueDraftScoreError("durable source receipt path is required")
    receipt_path = _regular_file(source_receipt_path, "source receipt")
    file_payload, _raw = _load_json_file(receipt_path, "source receipt")
    if file_payload != dict(source_receipt):
        raise FutureValueDraftScoreError("source receipt payload differs from its file")

    if source_root is not None and Path(source_root).is_symlink():
        raise FutureValueDraftScoreError("source root uses a symlink path")
    root = (
        _regular_file(source_root, "source root").parent
        if source_root is not None and Path(source_root).is_file()
        else Path(source_root).expanduser().resolve()
        if source_root is not None
        else receipt_path.parent
    )
    source_files = source_receipt["source_files"]
    if not isinstance(source_files, Mapping) or not source_files:
        raise FutureValueDraftScoreError("source receipt source_files are invalid")
    normalized_files: dict[str, dict[str, Any]] = {}
    for label, raw_record in source_files.items():
        if not isinstance(label, str) or not label.strip() or not isinstance(raw_record, Mapping):
            raise FutureValueDraftScoreError("source receipt source file record is invalid")
        unknown_record = sorted(set(raw_record) - _SOURCE_FILE_ALLOWED_FIELDS)
        if unknown_record:
            raise FutureValueDraftScoreError(
                f"source receipt source file has unknown fields: {label}"
            )
        if "year" in raw_record:
            year = raw_record["year"]
            if (
                isinstance(year, bool)
                or not isinstance(year, int)
                or not 2000 <= year <= 2100
                or not label.startswith("annual_")
                or label != f"annual_{year}"
            ):
                raise FutureValueDraftScoreError(
                    f"source receipt source file year is invalid: {label}"
                )
        locator_values = (
            raw_record.get("path"),
            raw_record.get("locator"),
        )
        if sum(isinstance(value, str) and bool(value.strip()) for value in locator_values) != 1:
            raise FutureValueDraftScoreError(
                f"source receipt source file locator is invalid: {label}"
            )
        locator = raw_record.get("locator", raw_record.get("path"))
        if not str(locator or "").strip():
            raise FutureValueDraftScoreError(f"source receipt source file locator is invalid: {label}")
        # A path field is a concrete local evidence path.  A locator is
        # resolved below the supplied source root or beside the receipt.
        candidate = raw_record.get("path", locator)
        path = _safe_locator(candidate, base=root, field=f"source file {label}")
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise FutureValueDraftScoreError(
                f"source file {label} escapes its root"
            ) from error
        _verify_file_digest(
            path,
            expected_bytes=raw_record.get("bytes"),
            expected_sha256=raw_record.get("sha256"),
            field=f"source file {label}",
        )
        normalized_files[label] = {
            **dict(raw_record),
            "locator": str(locator),
            "bytes": int(raw_record["bytes"]),
            "sha256": str(raw_record["sha256"]).lower(),
        }
    result = dict(source_receipt)
    result["source_files"] = normalized_files
    return result


def _validate_series_safe_evidence(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FutureValueDraftScoreError("series-safe evidence is required")
    if value.get("series_safe") is not True:
        raise FutureValueDraftScoreError("series-safe evidence must be true")
    if value.get("fit_validation_disjoint") is not True:
        raise FutureValueDraftScoreError("series-safe fit and validation disjointness is required")
    for name in ("source_type", "series_column"):
        if not str(value.get(name) or "").strip():
            raise FutureValueDraftScoreError(f"series-safe evidence {name} is required")
    cluster_hash = value.get("cluster_identity_sha256")
    _require_hash(cluster_hash, "series-safe cluster_identity_sha256")
    return dict(value)


def _validate_fit_dates(
    fit_game_ids: Sequence[str],
    fit_game_dates: Mapping[str, Any] | None,
    cutoff: pd.Timestamp,
) -> dict[str, str]:
    if not isinstance(fit_game_dates, Mapping):
        raise FutureValueDraftScoreError("fit game dates are required for strict prior binding")
    if set(str(key) for key in fit_game_dates) != set(fit_game_ids):
        raise FutureValueDraftScoreError("fit game dates do not match fit game IDs")
    normalized: dict[str, str] = {}
    for game_id in fit_game_ids:
        stamp = _timestamp(fit_game_dates[game_id], f"fit date {game_id}")
        if stamp >= cutoff:
            raise FutureValueDraftScoreError("fit game date is not strictly before fit cutoff")
        normalized[game_id] = stamp.isoformat().replace("+00:00", "Z")
    return normalized


@dataclass(frozen=True)
class DraftScoreProducerBinding:
    """Source and fold identity for one feature producer."""

    source_receipt_sha256: str
    source_identity_sha256: str
    accepted_game_ids: tuple[str, ...]
    fit_game_ids: tuple[str, ...]
    fit_window_end: str
    producer_receipt_sha256: str
    fold_id: str
    producer_receipt: Mapping[str, Any] | None = None
    source_receipt: Mapping[str, Any] | None = None
    producer_name: str = ""
    producer_family: str = ""
    fit_window_start: str | None = None
    fit_game_dates: Mapping[str, str] | None = None
    series_safe_evidence: Mapping[str, Any] | None = None
    producer_timing: str = ""
    source_receipt_path: Path | str | None = None
    source_root: Path | str | None = None
    producer_receipt_path: Path | str | None = None
    producer_receipt_file_sha256: str | None = None

    def __post_init__(self) -> None:
        accepted = _normalise_ids(self.accepted_game_ids, "accepted_game_ids")
        fit = _normalise_ids(self.fit_game_ids, "fit_game_ids")
        if not set(fit).issubset(set(accepted)):
            raise FutureValueDraftScoreError("fit_game_ids are outside accepted census")
        source = _validate_source_receipt_payload(
            self.source_receipt,
            expected_game_ids=accepted,
            source_receipt_path=self.source_receipt_path,
            source_root=self.source_root,
        )
        source_hash = str(source["receipt_sha256"]).lower()
        if str(self.source_receipt_sha256).lower() != source_hash:
            raise FutureValueDraftScoreError("source receipt hash does not match payload")
        if str(self.source_identity_sha256).lower() != str(source["source_identity_sha256"]).lower():
            raise FutureValueDraftScoreError("source identity does not match verified receipt")
        _require_hash(self.producer_receipt_sha256, "producer_receipt_sha256")
        window = _timestamp_text(self.fit_window_end, "fit_window_end")
        cutoff = _timestamp(window, "fit_window_end")
        fit_dates = _validate_fit_dates(fit, self.fit_game_dates, cutoff)
        start = (
            _timestamp_text(self.fit_window_start, "fit_window_start")
            if self.fit_window_start is not None
            else min(fit_dates.values())
        )
        if _timestamp(start, "fit_window_start") >= cutoff:
            raise FutureValueDraftScoreError("fit_window_start must precede fit_window_end")
        for field, value in (
            ("fold_id", self.fold_id),
            ("producer_name", self.producer_name),
            ("producer_family", self.producer_family),
            ("producer_timing", self.producer_timing),
        ):
            if not str(value or "").strip():
                raise FutureValueDraftScoreError(f"{field} is required")
        if self.producer_timing not in _ALLOWED_PRODUCER_TIMINGS:
            raise FutureValueDraftScoreError("producer timing is not an allowed pregame timing")
        series_evidence = _validate_series_safe_evidence(self.series_safe_evidence)
        object.__setattr__(self, "accepted_game_ids", accepted)
        object.__setattr__(self, "fit_game_ids", fit)
        object.__setattr__(self, "source_receipt", source)
        object.__setattr__(self, "source_receipt_sha256", source_hash)
        object.__setattr__(self, "source_identity_sha256", str(source["source_identity_sha256"]).lower())
        object.__setattr__(self, "producer_receipt_sha256", str(self.producer_receipt_sha256).lower())
        object.__setattr__(self, "fit_window_end", window)
        object.__setattr__(self, "fit_window_start", start)
        object.__setattr__(self, "fit_game_dates", fit_dates)
        object.__setattr__(self, "series_safe_evidence", series_evidence)
        if self.producer_name not in _TRUSTED_PRODUCER_NAMES:
            raise FutureValueDraftScoreError("producer is not a trusted Draft Score adapter")
        if self.producer_receipt is None or self.producer_receipt_path is None:
            raise FutureValueDraftScoreError("durable producer receipt path is required")
        producer_path = _regular_file(self.producer_receipt_path, "producer receipt")
        producer_file_payload, producer_raw = _load_json_file(producer_path, "producer receipt")
        if producer_file_payload != dict(self.producer_receipt):
            raise FutureValueDraftScoreError("producer receipt payload differs from its file")
        producer_file_hash = hashlib.sha256(producer_raw).hexdigest()
        if self.producer_receipt_file_sha256 is not None:
            expected_file_hash = _require_hash(
                self.producer_receipt_file_sha256,
                "producer_receipt_file_sha256",
            )
            if producer_file_hash != expected_file_hash:
                raise FutureValueDraftScoreError("producer receipt file changed")
        unknown_producer_fields = sorted(
            set(self.producer_receipt) - _PRODUCER_RECEIPT_ALLOWED_FIELDS
        )
        if unknown_producer_fields:
            raise FutureValueDraftScoreError(
                "producer receipt has unknown fields: " + ", ".join(unknown_producer_fields)
            )
        missing_producer_fields = sorted(
            _PRODUCER_RECEIPT_REQUIRED_FIELDS - set(self.producer_receipt)
        )
        if missing_producer_fields:
            raise FutureValueDraftScoreError(
                "producer receipt is incomplete: " + ", ".join(missing_producer_fields)
            )
        artifact_root = producer_path.parent
        artifact_path = _safe_locator(
            self.producer_receipt["artifact_locator"],
            base=artifact_root,
            field="producer artifact",
        )
        artifact_hash = _verify_file_digest(
            artifact_path,
            expected_bytes=self.producer_receipt["artifact_bytes"],
            expected_sha256=self.producer_receipt["artifact_sha256"],
            field="producer artifact",
        )
        artifact_receipt_path = _safe_locator(
            self.producer_receipt["artifact_receipt_locator"],
            base=artifact_root,
            field="producer artifact receipt",
        )
        artifact_receipt_hash = _verify_file_digest(
            artifact_receipt_path,
            expected_bytes=self.producer_receipt["artifact_receipt_bytes"],
            expected_sha256=self.producer_receipt["artifact_receipt_sha256"],
            field="producer artifact receipt",
        )
        artifact_receipt_payload, _artifact_receipt_raw = _load_json_file(
            artifact_receipt_path,
            "producer artifact receipt",
        )
        if (
            artifact_receipt_payload.get("artifact_locator") != str(
                self.producer_receipt["artifact_locator"]
            )
            or artifact_receipt_payload.get("artifact_bytes")
            != self.producer_receipt["artifact_bytes"]
            or artifact_receipt_payload.get("artifact_sha256") != artifact_hash
        ):
            raise FutureValueDraftScoreError("producer artifact receipt binding changed")
        if (
            artifact_receipt_payload.get("source_receipt_sha256")
            and str(artifact_receipt_payload["source_receipt_sha256"]).lower()
            != source_hash
        ) or (
            artifact_receipt_payload.get("source_identity_sha256")
            and str(artifact_receipt_payload["source_identity_sha256"]).lower()
            != str(source["source_identity_sha256"]).lower()
        ):
            raise FutureValueDraftScoreError("producer artifact source binding changed")
        _validate_producer_artifact_shape(
            artifact_path,
            producer_name=str(self.producer_name),
            source_identity_sha256=str(source["source_identity_sha256"]),
            expected_game_ids=accepted,
            allow_coverage_subset=str(self.producer_name)
            == "public_descriptive_draft_records",
        )
        payload = dict(self.producer_receipt)
        claimed = str(payload.pop("receipt_sha256", "")).lower()
        if claimed != self.producer_receipt_sha256 or _sha256(payload) != claimed:
            raise FutureValueDraftScoreError("producer receipt hash does not match payload")
        expected_payload = {
            "schema_version": SCHEMA_VERSION,
            "source_receipt_sha256": source_hash,
            "source_identity_sha256": str(source["source_identity_sha256"]).lower(),
            "accepted_game_count": len(accepted),
            "accepted_game_ids": list(accepted),
            "producer_name": str(self.producer_name),
            "producer_family": str(self.producer_family),
            "fit_game_count": len(fit),
            "fit_game_ids": list(fit),
            "fit_game_identity_sha256": identity_sha256(fit),
            "fit_window_start": start,
            "fit_window_end": window,
            "fit_game_dates": fit_dates,
            "fold_id": str(self.fold_id),
            "series_safe_evidence": series_evidence,
            "producer_timing": str(self.producer_timing),
            "artifact_locator": str(self.producer_receipt["artifact_locator"]),
            "artifact_bytes": self.producer_receipt["artifact_bytes"],
            "artifact_sha256": artifact_hash,
            "artifact_receipt_locator": str(
                self.producer_receipt["artifact_receipt_locator"]
            ),
            "artifact_receipt_bytes": self.producer_receipt["artifact_receipt_bytes"],
            "artifact_receipt_sha256": artifact_receipt_hash,
        }
        if payload != expected_payload:
            raise FutureValueDraftScoreError("producer receipt payload does not match binding")
        object.__setattr__(self, "source_receipt_path", str(_regular_file(self.source_receipt_path, "source receipt")))
        object.__setattr__(self, "producer_receipt_path", str(producer_path))
        object.__setattr__(self, "producer_receipt_file_sha256", producer_file_hash)

    @classmethod
    def create(
        cls,
        *,
        source_receipt: Mapping[str, Any] | None = None,
        source_receipt_sha256: str | None = None,
        accepted_game_ids: Iterable[object],
        fit_game_ids: Iterable[object],
        fit_window_end: Any,
        fold_id: str,
        producer: str,
        producer_family: str | None = None,
        fit_window_start: Any | None = None,
        fit_game_dates: Mapping[str, Any] | None = None,
        series_safe_evidence: Mapping[str, Any] | None = None,
        producer_timing: str = "pregame_strict_prior",
        producer_version: str = "v1",
        source_receipt_path: Path | str | None = None,
        source_root: Path | str | None = None,
        producer_receipt_path: Path | str | None = None,
        producer_receipt_file_sha256: str | None = None,
    ) -> "DraftScoreProducerBinding":
        accepted = _normalise_ids(accepted_game_ids, "accepted_game_ids")
        fit = _normalise_ids(fit_game_ids, "fit_game_ids")
        source = _validate_source_receipt_payload(
            source_receipt,
            expected_game_ids=accepted,
            source_receipt_path=source_receipt_path,
            source_root=source_root,
        )
        source_hash = str(source["receipt_sha256"]).lower()
        if source_receipt_sha256 is not None and str(source_receipt_sha256).lower() != source_hash:
            raise FutureValueDraftScoreError("source_receipt_sha256 does not match verified receipt")
        cutoff = _timestamp(fit_window_end, "fit_window_end")
        normalized_dates = _validate_fit_dates(fit, fit_game_dates, cutoff)
        normalized_start = (
            _timestamp_text(fit_window_start, "fit_window_start")
            if fit_window_start is not None
            else min(normalized_dates.values())
        )
        series_evidence = _validate_series_safe_evidence(series_safe_evidence)
        family = str(producer_family or producer).strip()
        if str(producer) not in _TRUSTED_PRODUCER_NAMES:
            raise FutureValueDraftScoreError("producer is not a trusted Draft Score adapter")
        if producer_receipt_path is None:
            raise FutureValueDraftScoreError("durable producer receipt path is required")
        producer_path = _regular_file(producer_receipt_path, "producer receipt")
        producer_payload, producer_raw = _load_json_file(producer_path, "producer receipt")
        producer_file_hash = hashlib.sha256(producer_raw).hexdigest()
        if producer_receipt_file_sha256 is not None and producer_file_hash != _require_hash(
            producer_receipt_file_sha256,
            "producer_receipt_file_sha256",
        ):
            raise FutureValueDraftScoreError("producer receipt file changed")
        claimed = str(producer_payload.get("receipt_sha256") or "").lower()
        if not claimed:
            raise FutureValueDraftScoreError("producer receipt digest is required")
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "source_receipt_sha256": source_hash,
            "source_identity_sha256": str(source["source_identity_sha256"]).lower(),
            "accepted_game_count": len(accepted),
            "accepted_game_ids": list(accepted),
            "producer_name": str(producer),
            "producer_family": family,
            "fit_game_count": len(fit),
            "fit_game_ids": list(fit),
            "fit_game_identity_sha256": identity_sha256(fit),
            "fit_window_start": normalized_start,
            "fit_window_end": _timestamp_text(fit_window_end, "fit_window_end"),
            "fit_game_dates": normalized_dates,
            "fold_id": str(fold_id),
            "series_safe_evidence": series_evidence,
            "producer_timing": str(producer_timing),
        }
        receipt_hash = _require_hash(claimed, "producer_receipt_sha256")
        if producer_payload.get("receipt_sha256") != receipt_hash:
            raise FutureValueDraftScoreError("producer receipt digest changed")
        return cls(
            source_receipt_sha256=source_hash,
            source_identity_sha256=str(source["source_identity_sha256"]).lower(),
            accepted_game_ids=accepted,
            fit_game_ids=fit,
            fit_window_end=payload["fit_window_end"],
            producer_receipt_sha256=receipt_hash,
            fold_id=str(fold_id),
            producer_receipt=producer_payload,
            source_receipt=source,
            producer_name=str(producer),
            producer_family=family,
            fit_window_start=normalized_start,
            fit_game_dates=normalized_dates,
            series_safe_evidence=series_evidence,
            producer_timing=str(producer_timing),
            source_receipt_path=source_receipt_path,
            source_root=source_root,
            producer_receipt_path=producer_path,
            producer_receipt_file_sha256=producer_file_hash,
        )

    @property
    def verified(self) -> bool:
        return self.producer_receipt is not None and self.source_receipt is not None

    def receipt(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "source_kind": SOURCE_KIND,
            "source_receipt_sha256": self.source_receipt_sha256,
            "source_identity_sha256": self.source_identity_sha256,
            "accepted_game_count": len(self.accepted_game_ids),
            "accepted_game_ids": list(self.accepted_game_ids),
            "producer_name": self.producer_name,
            "producer_family": self.producer_family,
            "fit_game_count": len(self.fit_game_ids),
            "fit_game_ids": list(self.fit_game_ids),
            "fit_game_identity_sha256": identity_sha256(self.fit_game_ids),
            "fit_window_start": self.fit_window_start,
            "fit_window_end": self.fit_window_end,
            "fit_game_dates": dict(self.fit_game_dates or {}),
            "fold_id": self.fold_id,
            "series_safe_evidence": dict(self.series_safe_evidence or {}),
            "producer_timing": self.producer_timing,
            "producer_receipt_sha256": self.producer_receipt_sha256,
        }
        payload["receipt_sha256"] = _sha256(payload)
        return payload


def _binding_from_any(binding: DraftScoreProducerBinding | Mapping[str, Any]) -> DraftScoreProducerBinding:
    if isinstance(binding, DraftScoreProducerBinding):
        return binding
    if not isinstance(binding, Mapping):
        raise FutureValueDraftScoreError("producer binding is required")
    try:
        return DraftScoreProducerBinding(
            source_receipt_sha256=str(binding["source_receipt_sha256"]),
            source_identity_sha256=str(binding["source_identity_sha256"]),
            accepted_game_ids=tuple(binding["accepted_game_ids"]),
            fit_game_ids=tuple(binding["fit_game_ids"]),
            fit_window_end=str(binding["fit_window_end"]),
            producer_receipt_sha256=str(binding["producer_receipt_sha256"]),
            fold_id=str(binding["fold_id"]),
            producer_receipt=binding.get("producer_receipt"),
            source_receipt=binding.get("source_receipt"),
            producer_name=str(binding.get("producer_name") or ""),
            producer_family=str(binding.get("producer_family") or ""),
            fit_window_start=binding.get("fit_window_start"),
            fit_game_dates=binding.get("fit_game_dates"),
            series_safe_evidence=binding.get("series_safe_evidence"),
            producer_timing=str(binding.get("producer_timing") or ""),
            source_receipt_path=binding.get("source_receipt_path"),
            source_root=binding.get("source_root"),
            producer_receipt_path=binding.get("producer_receipt_path"),
            producer_receipt_file_sha256=binding.get("producer_receipt_file_sha256"),
        )
    except KeyError as error:
        raise FutureValueDraftScoreError(f"producer binding is missing {error.args[0]}") from error


def validate_producer_binding(
    frame: pd.DataFrame,
    binding: DraftScoreProducerBinding | Mapping[str, Any],
    *,
    require_exact_census: bool = False,
) -> DraftScoreProducerBinding:
    """Validate census, fold, and strict-prior producer identity."""

    bound = _binding_from_any(binding)
    if not bound.verified:
        raise FutureValueDraftScoreError("producer receipt payload is required")
    if "game_id" not in frame.columns or "date" not in frame.columns:
        raise FutureValueDraftScoreError("feature frame requires game_id and date")
    raw_game_ids = frame["game_id"].astype(str)
    if raw_game_ids.eq("").any() or raw_game_ids.duplicated().any():
        raise FutureValueDraftScoreError("feature frame game IDs must be unique")
    game_ids = _normalise_ids(raw_game_ids, "frame game_id")
    accepted = set(bound.accepted_game_ids)
    if require_exact_census and tuple(game_ids) != bound.accepted_game_ids:
        raise FutureValueDraftScoreError("frame does not match accepted census exactly")
    if not set(game_ids).issubset(accepted):
        raise FutureValueDraftScoreError("frame contains game IDs outside accepted census")
    fit_overlap = set(game_ids) & set(bound.fit_game_ids)
    if fit_overlap:
        raise FutureValueDraftScoreError("fit and validation game IDs overlap")
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    if dates.isna().any():
        raise FutureValueDraftScoreError("feature frame contains an invalid date")
    cutoff = _timestamp(bound.fit_window_end, "fit_window_end")
    if (dates <= cutoff).any():
        raise FutureValueDraftScoreError("producer fit window is not strictly prior to every row")
    source_as_of = _timestamp(bound.source_receipt["source_as_of"], "source_as_of")
    if dates.gt(source_as_of).any():
        raise FutureValueDraftScoreError("feature frame contains rows after source_as_of")
    if "source_receipt_sha256" in frame.columns:
        values = frame["source_receipt_sha256"].astype(str).str.lower()
        if not values.eq(bound.source_receipt_sha256).all():
            raise FutureValueDraftScoreError("frame source receipt binding changed")
    if "source_identity_sha256" in frame.columns:
        values = frame["source_identity_sha256"].astype(str).str.lower()
        if not values.eq(bound.source_identity_sha256).all():
            raise FutureValueDraftScoreError("frame source census identity changed")
    if "producer_receipt_sha256" in frame.columns:
        values = frame["producer_receipt_sha256"].astype(str).str.lower()
        if not values.eq(bound.producer_receipt_sha256).all():
            raise FutureValueDraftScoreError("frame producer receipt binding changed")
    for column, expected, label in (
        ("producer_name", bound.producer_name, "producer name"),
        ("producer_family", bound.producer_family, "producer family"),
        ("fold_id", bound.fold_id, "fold ID"),
        ("producer_timing", bound.producer_timing, "producer timing"),
    ):
        if column in frame.columns and not frame[column].astype(str).eq(str(expected)).all():
            raise FutureValueDraftScoreError(f"frame {label} binding changed")
    if "fit_window_end" in frame.columns:
        values = pd.to_datetime(frame["fit_window_end"], utc=True, errors="coerce")
        if values.isna().any() or not values.eq(cutoff).all():
            raise FutureValueDraftScoreError("frame fit window binding changed")
    if "series_id" in frame.columns and frame["series_id"].isna().any():
        raise FutureValueDraftScoreError("series-safe frame contains a missing series ID")
    return bound


def _forbidden_feature_reason(name: str) -> str | None:
    text = str(name).strip()
    normal = re.sub(r"[^a-z0-9_]", "", text.casefold())
    compact = normal.replace("_", "")
    target_tokens = tuple(token.replace("_", "") for token in _TARGET_TOKENS)
    if compact in target_tokens or any(token in compact for token in target_tokens):
        return "target or observed outcome"
    if compact in _FINAL_TOKENS:
        return "final whole-map metric"
    if _RAW_CHECKPOINT_RE.search(normal) and not normal.startswith("forecast_"):
        return "current checkpoint observation"
    return None


def validate_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Reject target, final-state, and current-checkpoint feature names."""

    names = tuple(str(name) for name in feature_names)
    if len(set(names)) != len(names):
        raise FutureValueDraftScoreError("feature names contain duplicates")
    for name in names:
        reason = _forbidden_feature_reason(name)
        if reason is not None:
            raise FutureValueDraftScoreError(f"forbidden Draft Score feature {name}: {reason}")
    return names


def _validate_static_atom_receipt(
    receipt: Mapping[str, Any] | None,
    frame: pd.DataFrame,
    *,
    source_receipt_sha256: str | None = None,
    source_identity_sha256: str | None = None,
    expected_artifact_sha256: str | None = None,
    expected_receipt_sha256: str | None = None,
    side_swap_source_frame: pd.DataFrame | None = None,
    receipt_path: Path | str | None = None,
    artifact_root: Path | str | None = None,
    require_independent_authority: bool = False,
    authority_receipt_path: Path | str | None = None,
    authority_root: Path | str | None = None,
) -> dict[str, Any]:
    """Bind canonical atom values to an existing producer artifact.

    The receipt is supplied by the atomized producer.  This function never
    creates a receipt from the input frame.
    """

    if not isinstance(receipt, Mapping):
        raise FutureValueDraftScoreError("verified atomized composition receipt is required")
    unknown = sorted(set(receipt) - _STATIC_ATOM_ALLOWED_FIELDS)
    if unknown:
        raise FutureValueDraftScoreError(
            "atomized composition receipt has unknown fields: " + ", ".join(unknown)
        )
    missing = sorted(_STATIC_ATOM_REQUIRED_FIELDS - set(receipt))
    if missing:
        raise FutureValueDraftScoreError(
            "atomized composition receipt is incomplete: " + ", ".join(missing)
        )
    if str(receipt["schema_version"]) != STATIC_ATOM_RECEIPT_SCHEMA:
        raise FutureValueDraftScoreError("atomized composition receipt schema is invalid")
    if not str(receipt["producer_name"]).strip() or not str(receipt["producer_family"]).strip():
        raise FutureValueDraftScoreError("atomized composition producer identity is required")
    if str(receipt["producer_name"]) not in _TRUSTED_PRODUCER_NAMES:
        raise FutureValueDraftScoreError("atomized composition producer is not trusted")
    if receipt_path is None:
        raise FutureValueDraftScoreError("durable atomized composition receipt path is required")
    atom_receipt_path = _regular_file(receipt_path, "atomized composition receipt")
    file_payload, _raw = _load_json_file(atom_receipt_path, "atomized composition receipt")
    if file_payload != dict(receipt):
        raise FutureValueDraftScoreError("atomized composition receipt payload differs from its file")
    artifact_hash = _require_hash(receipt["artifact_sha256"], "atomized artifact_sha256")
    artifact_receipt_hash = _require_hash(receipt["artifact_receipt_sha256"], "atomized artifact_receipt_sha256")
    root = Path(artifact_root).expanduser().resolve() if artifact_root is not None else atom_receipt_path.parent
    artifact_path = _safe_locator(
        receipt["artifact_locator"],
        base=root,
        field="atomized artifact",
    )
    _verify_file_digest(
        artifact_path,
        expected_bytes=receipt["artifact_bytes"],
        expected_sha256=artifact_hash,
        field="atomized artifact",
    )
    artifact_receipt_path = _safe_locator(
        receipt["artifact_receipt_locator"],
        base=root,
        field="atomized artifact receipt",
    )
    _verify_file_digest(
        artifact_receipt_path,
        expected_bytes=receipt["artifact_receipt_bytes"],
        expected_sha256=artifact_receipt_hash,
        field="atomized artifact receipt",
    )
    artifact_payload, _artifact_receipt_raw = _load_json_file(
        artifact_receipt_path,
        "atomized artifact receipt",
    )
    if (
        artifact_payload.get("artifact_locator") != str(receipt["artifact_locator"])
        or artifact_payload.get("artifact_sha256") != artifact_hash
        or artifact_payload.get("artifact_bytes") != receipt["artifact_bytes"]
    ):
        raise FutureValueDraftScoreError("atomized artifact receipt binding changed")
    if (
        artifact_payload.get("source_receipt_sha256")
        and str(artifact_payload["source_receipt_sha256"]).lower()
        != str(receipt["source_receipt_sha256"]).lower()
    ) or (
        artifact_payload.get("source_identity_sha256")
        and str(artifact_payload["source_identity_sha256"]).lower()
        != str(receipt["source_identity_sha256"]).lower()
    ):
        raise FutureValueDraftScoreError("atomized artifact source binding changed")
    artifact_component_frame = (
        side_swap_source_frame if side_swap_source_frame is not None else frame
    )
    _validate_producer_artifact_shape(
        artifact_path,
        producer_name=str(receipt["producer_name"]),
        source_identity_sha256=str(receipt["source_identity_sha256"]),
        expected_game_ids=tuple(
            str(value) for value in artifact_component_frame["game_id"]
        ),
        expected_component_frame=artifact_component_frame,
    )
    coverage_ids = tuple(str(value) for value in receipt.get("coverage_game_ids", ()))
    if coverage_ids:
        canonical_coverage = _normalise_ids(coverage_ids, "atomized coverage game IDs")
        frame_ids = _normalise_ids(frame["game_id"].astype(str), "atomized frame game IDs")
        if canonical_coverage != frame_ids:
            raise FutureValueDraftScoreError("atomized coverage census changed")
        if receipt.get("coverage_game_count") != len(canonical_coverage):
            raise FutureValueDraftScoreError("atomized coverage count changed")
        if str(receipt.get("coverage_identity_sha256") or "").lower() != identity_sha256(
            canonical_coverage
        ):
            raise FutureValueDraftScoreError("atomized coverage identity changed")
    if source_receipt_sha256 is not None and str(receipt["source_receipt_sha256"]).lower() != _require_hash(source_receipt_sha256, "source_receipt_sha256"):
        raise FutureValueDraftScoreError("atomized source receipt binding changed")
    if source_identity_sha256 is not None and str(receipt["source_identity_sha256"]).lower() != _require_hash(source_identity_sha256, "source_identity_sha256"):
        raise FutureValueDraftScoreError("atomized source census binding changed")
    if expected_artifact_sha256 is not None and artifact_hash != _require_hash(expected_artifact_sha256, "expected atomized artifact_sha256"):
        raise FutureValueDraftScoreError("atomized artifact hash changed")
    feature_names = tuple(str(value) for value in receipt["feature_names"])
    if feature_names != STATIC_COMPOSITION_FEATURES:
        raise FutureValueDraftScoreError("atomized composition feature names are not canonical")
    receipt_hash = _require_hash(receipt["receipt_sha256"], "atomized receipt_sha256")
    if expected_receipt_sha256 is not None and receipt_hash != _require_hash(expected_receipt_sha256, "expected atomized receipt_sha256"):
        raise FutureValueDraftScoreError("atomized receipt hash changed")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    if _sha256(payload) != receipt_hash:
        raise FutureValueDraftScoreError("atomized composition receipt hash does not match payload")
    actual_values_hash = static_composition_parity_hash(frame)
    expected_values_hashes = {str(receipt["component_values_sha256"]).lower()}
    if side_swap_source_frame is not None:
        transformed = side_swap_source_frame[["game_id", *STATIC_COMPOSITION_FEATURES]].copy()
        transformed.loc[:, list(STATIC_COMPOSITION_FEATURES)] = -transformed.loc[:, list(STATIC_COMPOSITION_FEATURES)]
        expected_values_hashes.add(static_composition_parity_hash(transformed))
    if actual_values_hash not in expected_values_hashes:
        raise FutureValueDraftScoreError("atomized composition values do not match producer receipt")
    if require_independent_authority:
        required_authority = {
            "authority_receipt_sha256",
            "authority_receipt_locator",
            "authority_receipt_bytes",
            "model_artifact_sha256",
            "recipe_sha256",
            "scorer_code_sha256",
        }
        missing_authority = sorted(required_authority - set(receipt))
        if missing_authority:
            raise FutureValueDraftScoreError(
                "independent atom authority is incomplete: " + ", ".join(missing_authority)
            )
        if authority_receipt_path is None:
            raise FutureValueDraftScoreError("independent atom authority path is required")
        authority_path = _regular_file(authority_receipt_path, "atom authority receipt")
        authority_payload, authority_raw = _load_json_file(
            authority_path,
            "atom authority receipt",
        )
        expected_authority_hash = _require_hash(
            receipt["authority_receipt_sha256"],
            "authority_receipt_sha256",
        )
        if hashlib.sha256(authority_raw).hexdigest() != expected_authority_hash:
            raise FutureValueDraftScoreError("atom authority receipt bytes changed")
        if receipt["authority_receipt_bytes"] != len(authority_raw):
            raise FutureValueDraftScoreError("atom authority receipt byte count changed")
        if authority_path != _safe_locator(
            receipt["authority_receipt_locator"],
            base=Path(authority_root).expanduser().resolve()
            if authority_root is not None
            else authority_path.parent,
            field="atom authority receipt",
        ):
            raise FutureValueDraftScoreError("atom authority receipt locator changed")
        if authority_payload.get("schema_version") != "scryglass:draft-authority:v1" or authority_payload.get("status") != "descriptive" or authority_payload.get("estimand") != "composition_only":
            raise FutureValueDraftScoreError("atom authority receipt contract is invalid")
        if any(authority_payload.get(field) is not False for field in ("probability_authority", "recommendation_authority", "betting_authority")):
            raise FutureValueDraftScoreError("atom authority receipt grants a prohibited output")
        if str(authority_payload.get("artifact_sha256") or "").lower() != str(receipt["model_artifact_sha256"]).lower():
            raise FutureValueDraftScoreError("atom authority model binding changed")
        if str(authority_payload.get("recipe_sha256") or "").lower() != str(receipt["recipe_sha256"]).lower():
            raise FutureValueDraftScoreError("atom authority recipe binding changed")
        if str(authority_payload.get("scorer_code_sha256") or "").lower() != str(receipt["scorer_code_sha256"]).lower():
            raise FutureValueDraftScoreError("atom authority scorer binding changed")
    return {
        **dict(receipt),
        "artifact_sha256": artifact_hash,
        "artifact_receipt_sha256": artifact_receipt_hash,
        "receipt_sha256": receipt_hash,
        "artifact_locator": str(receipt["artifact_locator"]),
        "artifact_receipt_locator": str(receipt["artifact_receipt_locator"]),
        **{
            key: receipt[key]
            for key in (
                "authority_receipt_sha256",
                "authority_receipt_locator",
                "authority_receipt_bytes",
                "model_artifact_sha256",
                "recipe_sha256",
                "scorer_code_sha256",
            )
            if key in receipt
        },
    }


def _feature_columns_for_variant(variant: DraftScoreVariant) -> tuple[str, ...]:
    return VARIANT_CONFIGS[variant].feature_names


def _resolve_static_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for component, canonical in zip(STATIC_COMPOSITION_COMPONENTS, STATIC_COMPOSITION_FEATURES):
        if canonical not in frame.columns:
            raise FutureValueDraftScoreError(f"missing static composition component: {component}")
        values = pd.to_numeric(frame[canonical], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise FutureValueDraftScoreError(f"static component {component} is not finite")
        output[canonical] = values.astype(float)
    return output


def static_composition_parity_hash(
    frame_or_components: pd.DataFrame,
    *,
    game_id_column: str = "game_id",
) -> str:
    """Hash canonical atomized composition values in game-ID order."""

    if game_id_column not in frame_or_components.columns:
        raise FutureValueDraftScoreError("static composition frame requires game_id")
    components = (
        frame_or_components[list(STATIC_COMPOSITION_FEATURES)].copy()
        if set(STATIC_COMPOSITION_FEATURES).issubset(frame_or_components.columns)
        else _resolve_static_columns(frame_or_components)
    )
    ids = frame_or_components[game_id_column].astype(str)
    rows: list[dict[str, Any]] = []
    for index, game_id in zip(frame_or_components.index, ids):
        row: dict[str, Any] = {"game_id": str(game_id)}
        for column in STATIC_COMPOSITION_FEATURES:
            value = float(components.loc[index, column])
            if not math.isfinite(value):
                raise FutureValueDraftScoreError("static composition contains a non-finite value")
            row[column] = value
        rows.append(row)
    rows.sort(key=lambda row: row["game_id"])
    return _sha256(rows)


def assert_static_composition_parity(
    designs: Mapping[DraftScoreVariant | str, "DraftScoreVariantDesign"] | Sequence["DraftScoreVariantDesign"],
) -> str:
    """Require identical map IDs and atomized composition hashes."""

    values = list(designs.values()) if isinstance(designs, Mapping) else list(designs)
    if not values:
        raise FutureValueDraftScoreError("no designs supplied for parity")
    expected_ids = tuple(values[0].game_ids)
    expected_hash = values[0].static_composition_sha256
    expected_receipt = str(values[0].static_atom_receipt["receipt_sha256"])
    expected_artifact = str(values[0].static_atom_receipt["artifact_sha256"])
    expected_artifact_receipt = str(values[0].static_atom_receipt["artifact_receipt_sha256"])
    for design in values[1:]:
        if tuple(design.game_ids) != expected_ids:
            raise FutureValueDraftScoreError("variant game IDs differ")
        if design.static_composition_sha256 != expected_hash:
            raise FutureValueDraftScoreError("static composition changed between variants")
        if str(design.static_atom_receipt["receipt_sha256"]) != expected_receipt:
            raise FutureValueDraftScoreError("atomized producer receipt changed between variants")
        if str(design.static_atom_receipt["artifact_sha256"]) != expected_artifact:
            raise FutureValueDraftScoreError("atomized producer artifact changed between variants")
        if str(design.static_atom_receipt["artifact_receipt_sha256"]) != expected_artifact_receipt:
            raise FutureValueDraftScoreError("atomized producer artifact receipt changed between variants")
    return expected_hash


def _side_level_column_candidates(prefix: str, side: str, family: str, role: str) -> tuple[str, ...]:
    aliases = {
        "role": ("role",),
        "synergy": ("synergy", "ally_synergy"),
        "counter": ("counter", "enemy_counter"),
    }[family]
    return tuple(
        f"{prefix}_{side}_{alias}_{role}"
        for alias in aliases
    ) + tuple(
        f"{prefix}_{alias}_{side}_{role}"
        for alias in aliases
    )


def _find_column(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    matches = [candidate for candidate in candidates if candidate in frame.columns]
    if not matches:
        raise FutureValueDraftScoreError(f"missing curve/atom input: {label}")
    return matches[0]


def build_curve_atom_interactions(
    frame: pd.DataFrame,
    *,
    families: Sequence[str] = CURVE_ATOM_FAMILIES,
    roles: Sequence[str] = ROLES,
) -> pd.DataFrame:
    """Build signed side-level curve-by-atom difference-of-products.

    For each family and role the value is
    ``curve_blue * atom_blue - curve_red * atom_red``.  A product of two
    blue-minus-red values is not used.
    """

    output = pd.DataFrame(index=frame.index)
    for family in families:
        if family not in CURVE_ATOM_FAMILIES:
            raise FutureValueDraftScoreError(f"unknown curve interaction family: {family}")
        for role in roles:
            if role not in ROLES:
                raise FutureValueDraftScoreError(f"unknown curve interaction role: {role}")
            curve_blue = frame[_find_column(
                frame,
                _side_level_column_candidates("curve", "blue", family, role),
                f"curve blue {family} {role}",
            )]
            curve_red = frame[_find_column(
                frame,
                _side_level_column_candidates("curve", "red", family, role),
                f"curve red {family} {role}",
            )]
            atom_blue = frame[_find_column(
                frame,
                _side_level_column_candidates("atom", "blue", family, role),
                f"atom blue {family} {role}",
            )]
            atom_red = frame[_find_column(
                frame,
                _side_level_column_candidates("atom", "red", family, role),
                f"atom red {family} {role}",
            )]
            values = [
                pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                for series in (curve_blue, curve_red, atom_blue, atom_red)
            ]
            if any(not np.isfinite(value).all() for value in values):
                raise FutureValueDraftScoreError(f"curve interaction {family} {role} is not finite")
            output[f"curve_atom_{family}_{role}"] = values[0] * values[2] - values[1] * values[3]
    return output


def _ensure_phase_shape_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    raw_gold = [f"forecast_gold_diff_{checkpoint}" for checkpoint in (10, 15, 20, 25)]
    raw_xp = [f"forecast_xp_diff_{checkpoint}" for checkpoint in (10, 15, 20, 25)]
    if any(name not in output.columns for name in (*raw_gold, *raw_xp)):
        raise FutureValueDraftScoreError("phase shape features are missing")
    for name in PHASE_SHAPE_AVAILABILITY_FEATURES:
        if name not in output.columns:
            raise FutureValueDraftScoreError(f"phase availability field is missing: {name}")
    shape_rows: list[dict[str, float | None]] = []
    for index, row in output.iterrows():
        available = row["forecast_curve_available"]
        missing = row["forecast_curve_missing"]
        if isinstance(available, (bool, np.bool_)):
            available_value = bool(available)
        elif isinstance(available, (int, float, np.integer, np.floating)) and float(available) in (0.0, 1.0):
            available_value = bool(available)
        else:
            raise FutureValueDraftScoreError("phase curve available flag must be boolean")
        if isinstance(missing, (bool, np.bool_)):
            missing_value = bool(missing)
        elif isinstance(missing, (int, float, np.integer, np.floating)) and float(missing) in (0.0, 1.0):
            missing_value = bool(missing)
        else:
            raise FutureValueDraftScoreError("phase curve missing flag must be boolean")
        if available_value == missing_value:
            raise FutureValueDraftScoreError("phase curve availability flags are contradictory")
        shape_value = phase_shape_features(
            [row[name] for name in raw_gold],
            [row[name] for name in raw_xp],
            available=available_value,
        )
        if available_value and any(shape_value[name] is None for name in PHASE_SHAPE_SIGNED_FEATURES):
            raise FutureValueDraftScoreError("available phase curve has missing signed shape")
        for name, expected in shape_value.items():
            if name not in output.columns:
                continue
            actual = row[name]
            if expected is None:
                if not pd.isna(actual):
                    raise FutureValueDraftScoreError(f"phase shape field changed: {name}")
            else:
                try:
                    matches = math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
                except (TypeError, ValueError):
                    matches = False
                if not matches:
                    raise FutureValueDraftScoreError(f"phase shape field changed: {name}")
        shape_rows.append(shape_value)
    shape = pd.DataFrame(shape_rows, index=output.index)
    for name in PHASE_SHAPE_SIGNED_FEATURES:
        output[name] = shape[name]
    for name in PHASE_SHAPE_INVARIANT_FEATURES:
        output[name] = shape[name]
    return output


@dataclass(frozen=True)
class DraftScoreVariantConfig:
    """Immutable feature registry for one downstream variant."""

    variant: DraftScoreVariant
    feature_names: tuple[str, ...]
    static_features: tuple[str, ...]
    current_rating_features: tuple[str, ...]
    future_player_form_features: tuple[str, ...]
    phase_raw_features: tuple[str, ...]
    phase_shape_features: tuple[str, ...]
    phase_diagnostic_features: tuple[str, ...]
    curve_interaction_features: tuple[str, ...]

    def __post_init__(self) -> None:
        resolved = _canonical_variant(self.variant)
        object.__setattr__(self, "variant", resolved)
        all_fields = (
            *self.static_features,
            *self.current_rating_features,
            *self.future_player_form_features,
            *self.phase_raw_features,
            *self.phase_shape_features,
            *self.curve_interaction_features,
        )
        if len(set(all_fields)) != len(all_fields) or tuple(self.feature_names) != all_fields:
            raise FutureValueDraftScoreError("variant feature registry is not canonical")
        validate_feature_names(all_fields)
        if tuple(self.static_features) != STATIC_COMPOSITION_FEATURES:
            raise FutureValueDraftScoreError("static composition feature family changed")
        if tuple(self.current_rating_features) != CURRENT_RATING_SIGNED_MAP_FEATURES:
            raise FutureValueDraftScoreError("current rating feature family changed")
        expected_future = (
            tuple(FUTURE_PLAYER_FORM_SIDE_FEATURES)
            if resolved in (DraftScoreVariant.FUTURE_PLAYER_FORM, DraftScoreVariant.BOTH)
            else ()
        )
        expected_raw = (
            PHASE_RAW_FEATURES
            if resolved in (DraftScoreVariant.SCALING_CURVE, DraftScoreVariant.BOTH)
            else ()
        )
        expected_shape = (
            PHASE_SHAPE_SIGNED_FEATURES
            if resolved in (DraftScoreVariant.SCALING_CURVE, DraftScoreVariant.BOTH)
            else ()
        )
        expected_diagnostics = (
            PHASE_SHAPE_DIAGNOSTIC_FEATURES
            if resolved in (DraftScoreVariant.SCALING_CURVE, DraftScoreVariant.BOTH)
            else ()
        )
        expected_interactions = (
            CURVE_ATOM_INTERACTION_FEATURES
            if resolved in (DraftScoreVariant.SCALING_CURVE, DraftScoreVariant.BOTH)
            else ()
        )
        if (
            tuple(self.future_player_form_features) != expected_future
            or tuple(self.phase_raw_features) != expected_raw
            or tuple(self.phase_shape_features) != expected_shape
            or tuple(self.phase_diagnostic_features) != expected_diagnostics
            or tuple(self.curve_interaction_features) != expected_interactions
        ):
            raise FutureValueDraftScoreError("variant feature family selection changed")

    @property
    def name(self) -> str:
        return self.variant.value

    @property
    def config_sha256(self) -> str:
        return _sha256(self.receipt_payload())

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "variant": self.variant.value,
            "feature_names": list(self.feature_names),
            "static_features": list(self.static_features),
            "current_rating_features": list(self.current_rating_features),
            "future_player_form_features": list(self.future_player_form_features),
            "phase_raw_features": list(self.phase_raw_features),
            "phase_shape_features": list(self.phase_shape_features),
            "phase_diagnostic_features": list(self.phase_diagnostic_features),
            "curve_interaction_features": list(self.curve_interaction_features),
            "rating_variant_config_sha256": rating_variant_config_sha256(self.variant.value),
            "authority": AUTHORITY,
        }

    def receipt(self) -> dict[str, Any]:
        payload = self.receipt_payload()
        payload["config_sha256"] = _sha256(payload)
        return payload


def _make_config(variant: DraftScoreVariant) -> DraftScoreVariantConfig:
    future = tuple(FUTURE_PLAYER_FORM_SIDE_FEATURES) if variant in (DraftScoreVariant.FUTURE_PLAYER_FORM, DraftScoreVariant.BOTH) else ()
    phase_raw = PHASE_RAW_FEATURES if variant in (DraftScoreVariant.SCALING_CURVE, DraftScoreVariant.BOTH) else ()
    phase_shape = PHASE_SHAPE_SIGNED_FEATURES if variant in (DraftScoreVariant.SCALING_CURVE, DraftScoreVariant.BOTH) else ()
    phase_diagnostics = PHASE_SHAPE_DIAGNOSTIC_FEATURES if variant in (DraftScoreVariant.SCALING_CURVE, DraftScoreVariant.BOTH) else ()
    interactions = CURVE_ATOM_INTERACTION_FEATURES if variant in (DraftScoreVariant.SCALING_CURVE, DraftScoreVariant.BOTH) else ()
    features = (*STATIC_COMPOSITION_FEATURES, *CURRENT_RATING_SIGNED_MAP_FEATURES, *future, *phase_raw, *phase_shape, *interactions)
    return DraftScoreVariantConfig(
        variant=variant,
        feature_names=features,
        static_features=STATIC_COMPOSITION_FEATURES,
        current_rating_features=CURRENT_RATING_SIGNED_MAP_FEATURES,
        future_player_form_features=future,
        phase_raw_features=phase_raw,
        phase_shape_features=phase_shape,
        phase_diagnostic_features=phase_diagnostics,
        curve_interaction_features=interactions,
    )


VARIANT_CONFIGS: Mapping[DraftScoreVariant, DraftScoreVariantConfig] = MappingProxyType(
    {variant: _make_config(variant) for variant in DraftScoreVariant}
)
DRAFT_SCORE_VARIANT_CONFIGS = VARIANT_CONFIGS
DRAFT_SCORE_VARIANTS = tuple(DraftScoreVariant)


def draft_score_variant_config(variant: DraftScoreVariant | RatingVariant | str) -> DraftScoreVariantConfig:
    return VARIANT_CONFIGS[_canonical_variant(variant)]


def variant_config(variant: DraftScoreVariant | RatingVariant | str) -> DraftScoreVariantConfig:
    return draft_score_variant_config(variant)


def validate_variant_feature_names(
    variant: DraftScoreVariant | RatingVariant | str,
    feature_names: Iterable[str],
) -> tuple[str, ...]:
    config = draft_score_variant_config(variant)
    names = tuple(feature_names)
    if names != config.feature_names:
        raise FutureValueDraftScoreError("feature names do not match registered variant")
    return names


def _normalise_feature_frame(frame: pd.DataFrame, config: DraftScoreVariantConfig) -> pd.DataFrame:
    work = _ensure_phase_shape_features(frame) if config.phase_shape_features else frame.copy()
    static = _resolve_static_columns(work)
    output = pd.DataFrame(index=work.index)
    for column in config.feature_names:
        if column in static.columns:
            output[column] = static[column]
        elif column in work.columns:
            values = pd.to_numeric(work[column], errors="coerce")
            if values.isna().any() and column in PHASE_SHAPE_INVARIANT_FEATURES:
                # No crossover is a valid shape result.  The explicit curve
                # availability/missing fields remain in the matrix, so zero
                # is only the coordinate used for this nullable diagnostic.
                values = values.fillna(0.0)
            if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                raise FutureValueDraftScoreError(f"feature {column} is not finite")
            output[column] = values.astype(float)
        else:
            raise FutureValueDraftScoreError(f"missing variant feature: {column}")
    return output


def _append_curve_interactions(work: pd.DataFrame, config: DraftScoreVariantConfig) -> pd.DataFrame:
    if not config.curve_interaction_features:
        return work
    interactions = build_curve_atom_interactions(work)
    output = work.copy()
    for column in config.curve_interaction_features:
        output[column] = interactions[column]
    return output


@dataclass(frozen=True)
class DraftScoreVariantDesign:
    """Pre-match design matrix and provenance for one variant."""

    variant: DraftScoreVariant
    game_ids: tuple[str, ...]
    feature_frame: pd.DataFrame
    source_binding: DraftScoreProducerBinding
    static_composition_sha256: str
    static_atom_receipt: Mapping[str, Any]
    phase_diagnostics: pd.DataFrame | None = None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.feature_frame.columns)

    @property
    def authority(self) -> bool:
        return AUTHORITY

    def receipt(self) -> dict[str, Any]:
        config = draft_score_variant_config(self.variant)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "variant": self.variant.value,
            "variant_config_sha256": config.config_sha256,
            "source_receipt_sha256": self.source_binding.source_receipt_sha256,
            "source_identity_sha256": self.source_binding.source_identity_sha256,
            "producer_receipt_sha256": self.source_binding.producer_receipt_sha256,
            "fold_id": self.source_binding.fold_id,
            "game_count": len(self.game_ids),
            "game_ids": list(self.game_ids),
            "static_composition_sha256": self.static_composition_sha256,
            "static_atom_receipt_sha256": str(self.static_atom_receipt["receipt_sha256"]),
            "static_atom_artifact_sha256": str(self.static_atom_receipt["artifact_sha256"]),
            "static_atom_artifact_receipt_sha256": str(self.static_atom_receipt["artifact_receipt_sha256"]),
            "phase_diagnostics_sha256": (
                _frame_rows_sha256(self.phase_diagnostics, PHASE_SHAPE_DIAGNOSTIC_FEATURES)
                if self.phase_diagnostics is not None
                else None
            ),
            "authority": AUTHORITY,
        }
        payload["receipt_sha256"] = _sha256(payload)
        return payload


def build_draft_score_variant_design(
    frame: pd.DataFrame,
    variant: DraftScoreVariant | RatingVariant | str,
    binding: DraftScoreProducerBinding | Mapping[str, Any],
    *,
    require_exact_census: bool = False,
    static_atom_receipt: Mapping[str, Any] | None = None,
    atom_receipt: Mapping[str, Any] | None = None,
    static_atom_artifact_sha256: str | None = None,
    static_atom_receipt_sha256: str | None = None,
    static_atom_side_swap_source_frame: pd.DataFrame | None = None,
    static_atom_receipt_path: Path | str | None = None,
    static_atom_artifact_root: Path | str | None = None,
    require_independent_static_authority: bool = False,
    static_atom_authority_path: Path | str | None = None,
    static_atom_authority_root: Path | str | None = None,
) -> DraftScoreVariantDesign:
    """Build one immutable, source-bound research matrix."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise FutureValueDraftScoreError("feature frame is empty")
    resolved = _canonical_variant(variant)
    config = draft_score_variant_config(resolved)
    validate_feature_names(config.feature_names)
    bound = validate_producer_binding(frame, binding, require_exact_census=require_exact_census)
    atom_binding = static_atom_receipt if static_atom_receipt is not None else atom_receipt
    atom_verified = _validate_static_atom_receipt(
        atom_binding,
        frame,
        source_receipt_sha256=bound.source_receipt_sha256,
        source_identity_sha256=bound.source_identity_sha256,
        expected_artifact_sha256=static_atom_artifact_sha256,
        expected_receipt_sha256=static_atom_receipt_sha256,
        side_swap_source_frame=static_atom_side_swap_source_frame,
        receipt_path=static_atom_receipt_path,
        artifact_root=static_atom_artifact_root,
        require_independent_authority=require_independent_static_authority,
        authority_receipt_path=static_atom_authority_path,
        authority_root=static_atom_authority_root,
    )
    raw_ids = frame["game_id"].astype(str)
    if raw_ids.eq("").any() or raw_ids.duplicated().any():
        raise FutureValueDraftScoreError("one design row is required per unique game")
    order = pd.Series(raw_ids.to_numpy(), index=frame.index).sort_values(kind="stable").index
    ordered_frame = frame.loc[order].copy()
    if config.phase_shape_features:
        ordered_frame = _ensure_phase_shape_features(ordered_frame)
        if bound.producer_timing not in _ALLOWED_PRODUCER_TIMINGS:
            raise FutureValueDraftScoreError("scaling producer timing is not pregame")
        if "forecast_producer_timing" in ordered_frame.columns and not ordered_frame["forecast_producer_timing"].astype(str).eq(bound.producer_timing).all():
            raise FutureValueDraftScoreError("phase forecast producer timing changed")
    work = _append_curve_interactions(ordered_frame, config)
    values = _normalise_feature_frame(work, config)
    game_ids = tuple(ordered_frame["game_id"].astype(str))
    static_hash = static_composition_parity_hash(
        pd.concat([ordered_frame[["game_id"]], _resolve_static_columns(ordered_frame)], axis=1)
    )
    diagnostics = (
        ordered_frame[list(PHASE_SHAPE_DIAGNOSTIC_FEATURES)].reset_index(drop=True).copy()
        if config.phase_shape_features
        else None
    )
    if diagnostics is not None:
        if not diagnostics["forecast_curve_available"].astype(bool).all():
            raise FutureValueDraftScoreError("phase curve availability gate is not satisfied")
    return DraftScoreVariantDesign(
        variant=resolved,
        game_ids=game_ids,
        feature_frame=values.reset_index(drop=True),
        source_binding=bound,
        static_composition_sha256=static_hash,
        static_atom_receipt=atom_verified,
        phase_diagnostics=diagnostics,
    )


def _component_group(feature: str, config: DraftScoreVariantConfig) -> str:
    if feature in config.static_features:
        return f"composition_{feature.removeprefix('composition_').removesuffix('_logit')}_logit"
    if feature in config.current_rating_features:
        return "current_rating_logit"
    if feature in config.future_player_form_features:
        return "future_player_form_logit"
    if feature in config.phase_raw_features:
        return "scaling_raw_logit"
    if feature in config.phase_shape_features:
        return "scaling_shape_logit"
    if feature in config.curve_interaction_features:
        return "curve_atom_interaction_logit"
    raise FutureValueDraftScoreError(f"feature is outside variant component groups: {feature}")


@dataclass(frozen=True)
class DraftScoreVariantScore:
    """Component logits and reconstruction evidence for one variant."""

    variant: DraftScoreVariant
    game_ids: tuple[str, ...]
    components: pd.DataFrame
    coefficients: Mapping[str, float]
    source_binding: DraftScoreProducerBinding
    static_composition_sha256: str
    component_reconstruction_error_max: float
    independent_prediction_error_max: float
    coefficient_receipt: Mapping[str, Any]
    prediction_ledger_sha256: str
    authority: bool = AUTHORITY

    @property
    def composite_logit(self) -> pd.Series:
        return self.components["composite_logit"]

    def receipt(self) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "variant": self.variant.value,
            "source_receipt_sha256": self.source_binding.source_receipt_sha256,
            "source_identity_sha256": self.source_binding.source_identity_sha256,
            "producer_receipt_sha256": self.source_binding.producer_receipt_sha256,
            "static_composition_sha256": self.static_composition_sha256,
            "component_reconstruction_error_max": self.component_reconstruction_error_max,
            "independent_prediction_error_max": self.independent_prediction_error_max,
            "coefficient_receipt_sha256": str(self.coefficient_receipt["receipt_sha256"]),
            "prediction_ledger_sha256": self.prediction_ledger_sha256,
            "authority": AUTHORITY,
        }
        payload["receipt_sha256"] = _sha256(payload)
        return payload


def _coefficient_values_sha256(feature_names: Sequence[str], coefficients: Mapping[str, float]) -> str:
    return _sha256(
        {
            "feature_names": list(feature_names),
            "coefficients": {name: float(coefficients[name]) for name in feature_names},
        }
    )


def _validate_coefficient_receipt(
    receipt: Mapping[str, Any] | None,
    design: DraftScoreVariantDesign,
    coefficients: Mapping[str, float] | None,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise FutureValueDraftScoreError("fitted coefficient receipt is required")
    required = {
        "schema_version",
        "variant",
        "feature_names",
        "coefficient_sha256",
        "source_receipt_sha256",
        "fit_game_ids",
        "fit_game_identity_sha256",
        "fit_window_start",
        "fit_window_end",
        "fit_game_dates",
        "producer_name",
        "producer_family",
        "model_id",
        "fit_id",
        "receipt_sha256",
    }
    missing = sorted(required - set(receipt))
    if missing:
        raise FutureValueDraftScoreError("coefficient receipt is incomplete: " + ", ".join(missing))
    if str(receipt["schema_version"]) != COEFFICIENT_RECEIPT_SCHEMA:
        raise FutureValueDraftScoreError("coefficient receipt schema is invalid")
    config = draft_score_variant_config(design.variant)
    if str(receipt["variant"]) != design.variant.value:
        raise FutureValueDraftScoreError("coefficient receipt variant changed")
    if tuple(str(value) for value in receipt["feature_names"]) != config.feature_names:
        raise FutureValueDraftScoreError("coefficient receipt feature names changed")
    if str(receipt["source_receipt_sha256"]).lower() != design.source_binding.source_receipt_sha256:
        raise FutureValueDraftScoreError("coefficient receipt source changed")
    fit_ids = _normalise_ids(receipt["fit_game_ids"], "coefficient fit game IDs")
    if fit_ids != design.source_binding.fit_game_ids:
        raise FutureValueDraftScoreError("coefficient receipt fit IDs changed")
    if str(receipt["fit_game_identity_sha256"]).lower() != identity_sha256(fit_ids):
        raise FutureValueDraftScoreError("coefficient receipt fit identity changed")
    if dict(receipt["fit_game_dates"]) != dict(design.source_binding.fit_game_dates or {}):
        raise FutureValueDraftScoreError("coefficient receipt fit dates changed")
    if _timestamp_text(receipt["fit_window_start"], "coefficient fit_window_start") != design.source_binding.fit_window_start:
        raise FutureValueDraftScoreError("coefficient receipt fit start changed")
    if _timestamp_text(receipt["fit_window_end"], "coefficient fit_window_end") != design.source_binding.fit_window_end:
        raise FutureValueDraftScoreError("coefficient receipt fit cutoff changed")
    if str(receipt["producer_name"]) != design.source_binding.producer_name or str(receipt["producer_family"]) != design.source_binding.producer_family:
        raise FutureValueDraftScoreError("coefficient producer binding changed")
    if not str(receipt["model_id"]).strip() or not str(receipt["fit_id"]).strip():
        raise FutureValueDraftScoreError("coefficient model and fit IDs are required")
    if coefficients is None:
        raise FutureValueDraftScoreError("fitted coefficients are required")
    if set(coefficients) != set(config.feature_names):
        raise FutureValueDraftScoreError("coefficient registry does not match variant")
    values = {name: float(coefficients[name]) for name in config.feature_names}
    if not all(math.isfinite(value) for value in values.values()):
        raise FutureValueDraftScoreError("coefficients are not finite")
    expected_coefficient_hash = _coefficient_values_sha256(config.feature_names, values)
    if str(receipt["coefficient_sha256"]).lower() != expected_coefficient_hash:
        raise FutureValueDraftScoreError("coefficient receipt hash does not match coefficients")
    payload = dict(receipt)
    receipt_hash = _require_hash(payload.pop("receipt_sha256"), "coefficient receipt_sha256")
    if _sha256(payload) != receipt_hash:
        raise FutureValueDraftScoreError("coefficient receipt hash does not match payload")
    return {**dict(receipt), "receipt_sha256": receipt_hash, "coefficient_sha256": expected_coefficient_hash}


def make_coefficient_receipt(
    design: DraftScoreVariantDesign,
    coefficients: Mapping[str, float],
    *,
    producer_name: str | None = None,
    producer_family: str | None = None,
) -> dict[str, Any]:
    """Serialize a fitted coefficient set for a later scoring call.

    The caller supplies coefficients from a fitted model.  The function only
    records their identity and never chooses defaults.
    """

    config = draft_score_variant_config(design.variant)
    if set(coefficients) != set(config.feature_names):
        raise FutureValueDraftScoreError("coefficient registry does not match variant")
    values = {name: float(coefficients[name]) for name in config.feature_names}
    if not all(math.isfinite(value) for value in values.values()):
        raise FutureValueDraftScoreError("coefficients are not finite")
    payload: dict[str, Any] = {
        "schema_version": COEFFICIENT_RECEIPT_SCHEMA,
        "variant": design.variant.value,
        "feature_names": list(config.feature_names),
        "coefficient_sha256": _coefficient_values_sha256(config.feature_names, values),
        "source_receipt_sha256": design.source_binding.source_receipt_sha256,
        "source_identity_sha256": design.source_binding.source_identity_sha256,
        "fit_game_ids": list(design.source_binding.fit_game_ids),
        "fit_game_identity_sha256": identity_sha256(design.source_binding.fit_game_ids),
        "fit_id": identity_sha256(design.source_binding.fit_game_ids),
        "fit_window_start": design.source_binding.fit_window_start,
        "fit_window_end": design.source_binding.fit_window_end,
        "producer_name": producer_name or design.source_binding.producer_name,
        "producer_family": producer_family or design.source_binding.producer_family,
        "model_id": producer_name or design.source_binding.producer_name,
        "fit_game_dates": dict(design.source_binding.fit_game_dates or {}),
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload


def write_fitted_prediction_model(
    output_path: Path | str,
    design: DraftScoreVariantDesign,
    coefficients: Mapping[str, float],
    *,
    coefficient_receipt: Mapping[str, Any],
    model_version: str,
) -> tuple[Path, Path]:
    """Write fitted parameters and a receipt for the pinned linear model."""

    binding = _validate_coefficient_receipt(coefficient_receipt, design, coefficients)
    if not str(model_version).strip():
        raise FutureValueDraftScoreError("prediction model version is required")
    config = draft_score_variant_config(design.variant)
    implementation_hash = draft_score_trainer_implementation_sha256()
    artifact_path = Path(output_path).expanduser()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload: dict[str, Any] = {
        "schema_version": MODEL_ARTIFACT_SCHEMA,
        "trainer_id": MODEL_TRAINER_ID,
        "implementation_sha256": implementation_hash,
        "model_id": str(binding["model_id"]),
        "model_version": str(model_version),
        "variant": design.variant.value,
        "source_receipt_sha256": design.source_binding.source_receipt_sha256,
        "source_identity_sha256": design.source_binding.source_identity_sha256,
        "fold_id": design.source_binding.fold_id,
        "fit_game_ids": list(design.source_binding.fit_game_ids),
        "fit_game_identity_sha256": identity_sha256(design.source_binding.fit_game_ids),
        "fit_id": str(binding["fit_id"]),
        "feature_names": list(config.feature_names),
        "coefficients": {
            name: float(coefficients[name]) for name in config.feature_names
        },
        "coefficient_sha256": str(binding["coefficient_sha256"]),
    }
    artifact_raw = _canonical_json_bytes(artifact_payload) + b"\n"
    artifact_path.write_bytes(artifact_raw)
    receipt_payload: dict[str, Any] = {
        "schema_version": MODEL_RECEIPT_SCHEMA,
        "model_id": str(binding["model_id"]),
        "model_version": str(model_version),
        "variant": design.variant.value,
        "source_receipt_sha256": design.source_binding.source_receipt_sha256,
        "source_identity_sha256": design.source_binding.source_identity_sha256,
        "fold_id": design.source_binding.fold_id,
        "fit_game_ids": list(design.source_binding.fit_game_ids),
        "fit_game_identity_sha256": identity_sha256(design.source_binding.fit_game_ids),
        "fit_id": str(binding["fit_id"]),
        "coefficient_sha256": str(binding["coefficient_sha256"]),
        "artifact_locator": artifact_path.name,
        "artifact_bytes": len(artifact_raw),
        "artifact_sha256": hashlib.sha256(artifact_raw).hexdigest(),
        "implementation_sha256": implementation_hash,
        "authority": {"research_only": True},
    }
    receipt_payload["receipt_sha256"] = _sha256(receipt_payload)
    receipt_stem = artifact_path.stem.removesuffix("-artifact")
    receipt_path = artifact_path.with_name(receipt_stem + "-receipt.json")
    receipt_path.write_bytes(_canonical_json_bytes(receipt_payload) + b"\n")
    return artifact_path, receipt_path


def write_prediction_feature_artifact(
    output_path: Path | str,
    design: DraftScoreVariantDesign,
) -> Path:
    """Write the source-bound feature matrix used by an independent scorer."""

    config = draft_score_variant_config(design.variant)
    if tuple(design.feature_frame.columns) != config.feature_names:
        raise FutureValueDraftScoreError("prediction feature matrix columns changed")
    rows: list[dict[str, Any]] = []
    for game_id, values in zip(
        design.game_ids,
        design.feature_frame.to_numpy(dtype=float),
    ):
        if not np.isfinite(values).all():
            raise FutureValueDraftScoreError("prediction feature matrix is not finite")
        rows.append(
            {
                "game_id": str(game_id),
                "features": {
                    name: float(value)
                    for name, value in zip(config.feature_names, values)
                },
            }
        )
    payload = {
        "schema_version": PREDICTION_FEATURE_ARTIFACT_SCHEMA,
        "authority": {"research_only": True},
        "variant": design.variant.value,
        "source_receipt_sha256": design.source_binding.source_receipt_sha256,
        "source_identity_sha256": design.source_binding.source_identity_sha256,
        "fold_id": design.source_binding.fold_id,
        "game_ids": list(design.game_ids),
        "feature_names": list(config.feature_names),
        "feature_rows_sha256": _sha256(rows),
        "rows": rows,
    }
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_json_bytes(payload) + b"\n")
    return output


def _load_prediction_feature_artifact(
    path: Path,
    *,
    expected_source_receipt_sha256: str,
    expected_source_identity_sha256: str,
    expected_fold_id: str,
    expected_variant: DraftScoreVariant,
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray, str, int, str]:
    payload, raw = _load_json_file(path, "prediction feature artifact")
    fields = {
        "schema_version",
        "authority",
        "variant",
        "source_receipt_sha256",
        "source_identity_sha256",
        "fold_id",
        "game_ids",
        "feature_names",
        "feature_rows_sha256",
        "rows",
    }
    if set(payload) != fields or payload.get("schema_version") != PREDICTION_FEATURE_ARTIFACT_SCHEMA:
        raise FutureValueDraftScoreError("prediction feature artifact schema is invalid")
    if payload.get("authority") != {"research_only": True}:
        raise FutureValueDraftScoreError("prediction feature artifact authority is invalid")
    if str(payload.get("variant")) != expected_variant.value:
        raise FutureValueDraftScoreError("prediction feature artifact variant changed")
    if str(payload.get("source_receipt_sha256")).lower() != _require_hash(
        expected_source_receipt_sha256, "source_receipt_sha256"
    ):
        raise FutureValueDraftScoreError("prediction feature artifact source changed")
    if str(payload.get("source_identity_sha256")).lower() != _require_hash(
        expected_source_identity_sha256, "source_identity_sha256"
    ):
        raise FutureValueDraftScoreError("prediction feature artifact census changed")
    if str(payload.get("fold_id")) != str(expected_fold_id):
        raise FutureValueDraftScoreError("prediction feature artifact fold changed")
    raw_ids = payload.get("game_ids")
    if not isinstance(raw_ids, list):
        raise FutureValueDraftScoreError("prediction feature artifact game IDs are invalid")
    game_ids = _normalise_ids(raw_ids, "prediction feature artifact game IDs")
    if tuple(str(value) for value in raw_ids) != game_ids:
        raise FutureValueDraftScoreError("prediction feature artifact game IDs are not canonical")
    config = draft_score_variant_config(expected_variant)
    feature_names = tuple(str(value) for value in payload.get("feature_names", ()))
    if feature_names != config.feature_names:
        raise FutureValueDraftScoreError("prediction feature artifact features changed")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(game_ids):
        raise FutureValueDraftScoreError("prediction feature artifact rows are invalid")
    parsed_rows: list[dict[str, Any]] = []
    matrix: list[list[float]] = []
    for expected_game_id, row in zip(game_ids, rows):
        if not isinstance(row, Mapping) or set(row) != {"game_id", "features"}:
            raise FutureValueDraftScoreError("prediction feature artifact row schema is invalid")
        if str(row["game_id"]) != expected_game_id:
            raise FutureValueDraftScoreError("prediction feature artifact row order changed")
        features = row.get("features")
        if not isinstance(features, Mapping) or set(features) != set(feature_names):
            raise FutureValueDraftScoreError("prediction feature artifact row features are invalid")
        try:
            values = [float(features[name]) for name in feature_names]
        except (TypeError, ValueError) as error:
            raise FutureValueDraftScoreError(
                "prediction feature artifact row values are invalid"
            ) from error
        if not all(math.isfinite(value) for value in values):
            raise FutureValueDraftScoreError("prediction feature artifact row values are invalid")
        parsed_rows.append(
            {
                "game_id": expected_game_id,
                "features": {
                    name: value for name, value in zip(feature_names, values)
                },
            }
        )
        matrix.append(values)
    rows_hash = _sha256(parsed_rows)
    if rows_hash != _require_hash(
        payload.get("feature_rows_sha256"), "prediction feature_rows_sha256"
    ):
        raise FutureValueDraftScoreError("prediction feature artifact rows changed")
    return (
        game_ids,
        feature_names,
        np.asarray(matrix, dtype=float),
        hashlib.sha256(raw).hexdigest(),
        len(raw),
        rows_hash,
    )


def _validate_prediction_model_receipt(
    receipt_path: Path,
    *,
    expected_source_receipt_sha256: str,
    expected_source_identity_sha256: str,
    expected_fold_id: str,
    expected_fit_game_ids: Sequence[str],
    expected_fit_id: str,
    expected_model_id: str,
    expected_coefficient_sha256: str,
    expected_variant: str | None,
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    """Verify the independent fitted-model receipt named by a ledger."""

    payload, raw = _load_json_file(receipt_path, "prediction model receipt")
    unknown = sorted(set(payload) - _MODEL_RECEIPT_ALLOWED_FIELDS)
    if unknown:
        raise FutureValueDraftScoreError(
            "prediction model receipt has unknown fields: " + ", ".join(unknown)
        )
    missing = sorted(_MODEL_RECEIPT_REQUIRED_FIELDS - set(payload))
    if missing:
        raise FutureValueDraftScoreError(
            "prediction model receipt is incomplete: " + ", ".join(missing)
        )
    if payload.get("schema_version") != MODEL_RECEIPT_SCHEMA:
        raise FutureValueDraftScoreError("prediction model receipt schema is invalid")
    authority = payload.get("authority")
    if not isinstance(authority, Mapping) or authority.get("research_only") is not True:
        raise FutureValueDraftScoreError("prediction model receipt authority is invalid")
    if any(value is not False for key, value in authority.items() if key != "research_only"):
        raise FutureValueDraftScoreError("prediction model receipt grants authority")
    receipt_hash = _require_hash(payload["receipt_sha256"], "prediction model receipt_sha256")
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256", None)
    if _sha256(unsigned) != receipt_hash:
        raise FutureValueDraftScoreError("prediction model receipt hash does not match payload")
    if str(payload["source_receipt_sha256"]).lower() != _require_hash(
        expected_source_receipt_sha256, "source_receipt_sha256"
    ):
        raise FutureValueDraftScoreError("prediction model receipt source changed")
    if str(payload["source_identity_sha256"]).lower() != _require_hash(
        expected_source_identity_sha256, "source_identity_sha256"
    ):
        raise FutureValueDraftScoreError("prediction model receipt census changed")
    if str(payload["fold_id"]) != str(expected_fold_id):
        raise FutureValueDraftScoreError("prediction model receipt fold changed")
    fit_ids = _normalise_ids(payload["fit_game_ids"], "prediction model fit game IDs")
    expected_fit = _normalise_ids(expected_fit_game_ids, "expected fit game IDs")
    if fit_ids != expected_fit:
        raise FutureValueDraftScoreError("prediction model receipt fit IDs changed")
    if str(payload["fit_id"]) != str(expected_fit_id):
        raise FutureValueDraftScoreError("prediction model receipt fit identity changed")
    if str(payload["fit_game_identity_sha256"]).lower() != identity_sha256(fit_ids):
        raise FutureValueDraftScoreError("prediction model receipt fit identity changed")
    if str(payload["model_id"]) != str(expected_model_id):
        raise FutureValueDraftScoreError("prediction model receipt model identity changed")
    if str(payload["coefficient_sha256"]).lower() != _require_hash(
        expected_coefficient_sha256, "coefficient_sha256"
    ):
        raise FutureValueDraftScoreError("prediction model receipt coefficients changed")
    if str(payload["fit_game_identity_sha256"]).lower() != identity_sha256(expected_fit):
        raise FutureValueDraftScoreError("prediction model receipt fit identity changed")
    if expected_variant is not None and str(payload["variant"]) != str(expected_variant):
        raise FutureValueDraftScoreError("prediction model receipt variant changed")
    if not str(payload["model_version"]).strip():
        raise FutureValueDraftScoreError("prediction model receipt implementation identity is required")
    implementation_hash = _require_hash(
        payload["implementation_sha256"], "prediction model implementation_sha256"
    )
    if implementation_hash != draft_score_trainer_implementation_sha256():
        raise FutureValueDraftScoreError("prediction model trainer implementation changed")
    artifact_path = _safe_locator(
        payload["artifact_locator"],
        base=receipt_path.parent,
        field="prediction model artifact",
    )
    artifact_hash = _verify_file_digest(
        artifact_path,
        expected_bytes=payload["artifact_bytes"],
        expected_sha256=payload["artifact_sha256"],
        field="prediction model artifact",
    )
    artifact_payload, _artifact_raw = _load_json_file(
        artifact_path, "prediction model artifact"
    )
    artifact_fields = {
        "schema_version",
        "trainer_id",
        "implementation_sha256",
        "model_id",
        "model_version",
        "variant",
        "source_receipt_sha256",
        "source_identity_sha256",
        "fold_id",
        "fit_game_ids",
        "fit_game_identity_sha256",
        "fit_id",
        "feature_names",
        "coefficients",
        "coefficient_sha256",
    }
    if set(artifact_payload) != artifact_fields:
        raise FutureValueDraftScoreError("prediction model artifact schema is invalid")
    artifact_variant = _canonical_variant(str(payload["variant"]))
    config = draft_score_variant_config(artifact_variant)
    if artifact_payload.get("schema_version") != MODEL_ARTIFACT_SCHEMA:
        raise FutureValueDraftScoreError("prediction model artifact schema is invalid")
    if artifact_payload.get("trainer_id") != MODEL_TRAINER_ID:
        raise FutureValueDraftScoreError("prediction model artifact trainer changed")
    expected_artifact_bindings = {
        "implementation_sha256": implementation_hash,
        "model_id": str(payload["model_id"]),
        "model_version": str(payload["model_version"]),
        "variant": str(payload["variant"]),
        "source_receipt_sha256": str(payload["source_receipt_sha256"]).lower(),
        "source_identity_sha256": str(payload["source_identity_sha256"]).lower(),
        "fold_id": str(payload["fold_id"]),
        "fit_game_identity_sha256": identity_sha256(fit_ids),
        "fit_id": str(payload["fit_id"]),
        "coefficient_sha256": str(payload["coefficient_sha256"]).lower(),
    }
    for field, expected in expected_artifact_bindings.items():
        value = artifact_payload.get(field)
        actual = str(value).lower() if field.endswith("sha256") else str(value)
        if actual != expected:
            raise FutureValueDraftScoreError(
                f"prediction model artifact {field} changed"
            )
    artifact_fit_ids = _normalise_ids(
        artifact_payload["fit_game_ids"], "prediction model artifact fit game IDs"
    )
    if artifact_fit_ids != fit_ids:
        raise FutureValueDraftScoreError("prediction model artifact fit IDs changed")
    feature_names = tuple(str(value) for value in artifact_payload["feature_names"])
    if feature_names != config.feature_names:
        raise FutureValueDraftScoreError("prediction model artifact features changed")
    artifact_coefficients = artifact_payload.get("coefficients")
    if not isinstance(artifact_coefficients, Mapping) or set(artifact_coefficients) != set(
        feature_names
    ):
        raise FutureValueDraftScoreError("prediction model artifact coefficients are invalid")
    try:
        coefficients = {
            name: float(artifact_coefficients[name]) for name in feature_names
        }
    except (TypeError, ValueError) as error:
        raise FutureValueDraftScoreError(
            "prediction model artifact coefficients are invalid"
        ) from error
    if not all(math.isfinite(value) for value in coefficients.values()):
        raise FutureValueDraftScoreError("prediction model artifact coefficients are invalid")
    if _coefficient_values_sha256(feature_names, coefficients) != str(
        payload["coefficient_sha256"]
    ).lower():
        raise FutureValueDraftScoreError(
            "prediction model artifact parameters differ from coefficient receipt"
        )
    return payload, hashlib.sha256(raw).hexdigest(), artifact_hash, artifact_payload


def write_independent_prediction_ledger(
    output_path: Path | str,
    feature_artifact_path: Path | str,
    *,
    source_receipt_sha256: str,
    source_identity_sha256: str,
    fold_id: str,
    fit_game_ids: Sequence[object],
    fit_id: str,
    model_id: str,
    coefficient_sha256: str,
    model_receipt_path: Path | str,
    variant: DraftScoreVariant | RatingVariant | str,
) -> tuple[Path, Path]:
    """Apply a verified model to a durable independent feature artifact.

    Caller-supplied logits are outside this interface.  This function loads
    exact feature bytes, applies the pinned linear model, and binds both input
    artifacts into the prediction receipt.
    """

    fit_ids = _normalise_ids(fit_game_ids, "prediction ledger fit game IDs")
    source_hash = _require_hash(source_receipt_sha256, "source_receipt_sha256")
    source_identity = _require_hash(source_identity_sha256, "source_identity_sha256")
    coefficient_hash = _require_hash(coefficient_sha256, "coefficient_sha256")
    if not str(fold_id).strip() or not str(fit_id).strip() or not str(model_id).strip():
        raise FutureValueDraftScoreError("prediction ledger model identity is required")
    canonical_variant = _canonical_variant(variant)
    model_path = _regular_file(model_receipt_path, "prediction model receipt")
    (
        model_payload,
        model_receipt_hash,
        model_artifact_hash,
        model_artifact_payload,
    ) = _validate_prediction_model_receipt(
        model_path,
        expected_source_receipt_sha256=source_hash,
        expected_source_identity_sha256=source_identity,
        expected_fold_id=str(fold_id),
        expected_fit_game_ids=fit_ids,
        expected_fit_id=str(fit_id),
        expected_model_id=str(model_id),
        expected_coefficient_sha256=coefficient_hash,
        expected_variant=canonical_variant.value,
    )
    feature_path = _regular_file(feature_artifact_path, "prediction feature artifact")
    (
        ids,
        feature_names,
        feature_matrix,
        feature_artifact_hash,
        feature_artifact_bytes,
        feature_rows_hash,
    ) = _load_prediction_feature_artifact(
        feature_path,
        expected_source_receipt_sha256=source_hash,
        expected_source_identity_sha256=source_identity,
        expected_fold_id=str(fold_id),
        expected_variant=canonical_variant,
    )
    if set(ids) & set(fit_ids):
        raise FutureValueDraftScoreError(
            "prediction feature and model fit game IDs overlap"
        )
    coefficients = {
        name: float(model_artifact_payload["coefficients"][name])
        for name in feature_names
    }
    values = _linear_model_logits(feature_names, coefficients, feature_matrix)
    if values.shape != (len(ids),) or not np.isfinite(values).all():
        raise FutureValueDraftScoreError("prediction model produced invalid logits")
    implementation_hash = _require_hash(
        model_payload["implementation_sha256"], "model_implementation_sha256"
    )
    rows = [
        {"game_id": game_id, "model_logit": value}
        for game_id, value in zip(ids, values.tolist())
    ]
    row_digest = _sha256(rows)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_raw = _canonical_json_bytes({"rows": rows}) + b"\n"
    output.write_bytes(artifact_raw)
    receipt_payload: dict[str, Any] = {
        "schema_version": PREDICTION_LEDGER_SCHEMA,
        "authority": {"research_only": True},
        "source_receipt_sha256": source_hash,
        "source_identity_sha256": source_identity,
        "game_ids": list(ids),
        "fold_id": str(fold_id),
        "fit_id": str(fit_id),
        "model_id": str(model_id),
        "fit_game_ids": list(fit_ids),
        "fit_game_identity_sha256": identity_sha256(fit_ids),
        "coefficient_sha256": coefficient_hash,
        "model_receipt_locator": str(model_path),
        "model_receipt_bytes": model_path.stat().st_size,
        "model_receipt_sha256": model_receipt_hash,
        "model_artifact_locator": str(model_payload["artifact_locator"]),
        "model_artifact_bytes": model_payload["artifact_bytes"],
        "model_artifact_sha256": model_artifact_hash,
        "model_implementation_sha256": implementation_hash,
        "feature_artifact_locator": str(feature_path),
        "feature_artifact_bytes": feature_artifact_bytes,
        "feature_artifact_sha256": feature_artifact_hash,
        "feature_names": list(feature_names),
        "feature_rows_sha256": feature_rows_hash,
        "row_digest_sha256": row_digest,
        "artifact_locator": output.name,
        "artifact_bytes": len(artifact_raw),
        "artifact_sha256": hashlib.sha256(artifact_raw).hexdigest(),
    }
    receipt_payload["receipt_sha256"] = _sha256(receipt_payload)
    receipt_path = output.with_name(output.stem + "-receipt.json")
    receipt_path.write_bytes(_canonical_json_bytes(receipt_payload) + b"\n")
    return output, receipt_path


def _prediction_ledger_values(
    ledger: Path | str | None,
    game_ids: Sequence[str],
    *,
    receipt: Mapping[str, Any] | Path | str | None = None,
    source_receipt_sha256: str | None = None,
    source_identity_sha256: str | None = None,
    fold_id: str | None = None,
    fit_game_ids: Sequence[str] | None = None,
    fit_id: str | None = None,
    model_id: str | None = None,
    coefficient_sha256: str | None = None,
    variant: str | None = None,
) -> tuple[np.ndarray, str]:
    """Read one durable, receipt-bound independent model-logit ledger.

    DataFrames and mappings are deliberately rejected.  The independent gate
    needs an artifact that another process can inspect and hash.
    """

    if ledger is None or isinstance(ledger, (pd.DataFrame, Mapping, list, tuple)):
        raise FutureValueDraftScoreError("independent prediction ledger must be a durable path")
    ledger_path = _regular_file(ledger, "prediction ledger")
    ledger_payload, _ledger_raw = _load_json_file(ledger_path, "prediction ledger")
    rows = ledger_payload.get("rows")
    embedded_receipt = ledger_payload.get("receipt")
    if not isinstance(rows, list):
        raise FutureValueDraftScoreError("prediction ledger rows are missing")

    if receipt is None:
        if isinstance(embedded_receipt, Mapping):
            receipt_payload = dict(embedded_receipt)
        elif "receipt_sha256" in ledger_payload:
            receipt_payload = dict(ledger_payload)
        else:
            raise FutureValueDraftScoreError("prediction ledger receipt is required")
        receipt_path = ledger_path
    elif isinstance(receipt, Mapping):
        raise FutureValueDraftScoreError("prediction ledger receipt must be a durable path")
    else:
        receipt_path = _regular_file(receipt, "prediction ledger receipt")
        receipt_payload, _receipt_raw = _load_json_file(receipt_path, "prediction ledger receipt")

    unknown = sorted(set(receipt_payload) - _PREDICTION_LEDGER_ALLOWED_FIELDS)
    if unknown:
        raise FutureValueDraftScoreError(
            "prediction ledger receipt has unknown fields: " + ", ".join(unknown)
        )
    missing = sorted(_PREDICTION_LEDGER_REQUIRED_FIELDS - set(receipt_payload))
    if missing:
        raise FutureValueDraftScoreError(
            "prediction ledger receipt is incomplete: " + ", ".join(missing)
        )
    if receipt_payload.get("schema_version") != PREDICTION_LEDGER_SCHEMA:
        raise FutureValueDraftScoreError("prediction ledger receipt schema is invalid")
    authority = receipt_payload.get("authority", {"research_only": True})
    if not isinstance(authority, Mapping) or authority.get("research_only") is not True:
        raise FutureValueDraftScoreError("prediction ledger authority is invalid")
    if any(value is not False for key, value in authority.items() if key != "research_only"):
        raise FutureValueDraftScoreError("prediction ledger grants authority")
    claimed_receipt_hash = _require_hash(receipt_payload["receipt_sha256"], "prediction ledger receipt_sha256")
    unsigned = dict(receipt_payload)
    unsigned.pop("receipt_sha256", None)
    if _sha256(unsigned) != claimed_receipt_hash:
        raise FutureValueDraftScoreError("prediction ledger receipt hash does not match payload")

    artifact_locator = receipt_payload["artifact_locator"]
    artifact_base = receipt_path.parent
    artifact_path = _safe_locator(artifact_locator, base=artifact_base, field="prediction ledger artifact")
    artifact_hash = _verify_file_digest(
        artifact_path,
        expected_bytes=receipt_payload["artifact_bytes"],
        expected_sha256=receipt_payload["artifact_sha256"],
        field="prediction ledger artifact",
    )
    if artifact_path != ledger_path:
        raise FutureValueDraftScoreError("prediction ledger artifact path changed")

    model_receipt_path = _safe_locator(
        receipt_payload["model_receipt_locator"],
        base=receipt_path.parent,
        field="prediction model receipt",
    )
    model_receipt_hash = _verify_file_digest(
        model_receipt_path,
        expected_bytes=receipt_payload["model_receipt_bytes"],
        expected_sha256=receipt_payload["model_receipt_sha256"],
        field="prediction model receipt",
    )

    expected_ids = tuple(str(value) for value in game_ids)
    receipt_ids = receipt_payload["game_ids"]
    if not isinstance(receipt_ids, list) or tuple(str(value) for value in receipt_ids) != expected_ids:
        raise FutureValueDraftScoreError("prediction ledger game IDs changed")
    if source_receipt_sha256 is not None and str(receipt_payload["source_receipt_sha256"]).lower() != _require_hash(source_receipt_sha256, "source_receipt_sha256"):
        raise FutureValueDraftScoreError("prediction ledger source receipt binding changed")
    if source_identity_sha256 is not None and str(receipt_payload["source_identity_sha256"]).lower() != _require_hash(source_identity_sha256, "source_identity_sha256"):
        raise FutureValueDraftScoreError("prediction ledger source census binding changed")
    if fold_id is not None and str(receipt_payload["fold_id"]) != str(fold_id):
        raise FutureValueDraftScoreError("prediction ledger fold binding changed")
    if fit_id is not None and str(receipt_payload["fit_id"]) != str(fit_id):
        raise FutureValueDraftScoreError("prediction ledger fit ID changed")
    if model_id is not None and str(receipt_payload["model_id"]) != str(model_id):
        raise FutureValueDraftScoreError("prediction ledger model identity changed")
    if fit_game_ids is not None:
        expected_fit_ids = _normalise_ids(fit_game_ids, "fit game IDs")
        if tuple(str(value) for value in receipt_payload["fit_game_ids"]) != expected_fit_ids:
            raise FutureValueDraftScoreError("prediction ledger fit IDs changed")
        if str(receipt_payload["fit_game_identity_sha256"]).lower() != identity_sha256(expected_fit_ids):
            raise FutureValueDraftScoreError("prediction ledger fit identity changed")
    if coefficient_sha256 is not None and str(receipt_payload["coefficient_sha256"]).lower() != _require_hash(coefficient_sha256, "coefficient_sha256"):
        raise FutureValueDraftScoreError("prediction ledger coefficient binding changed")
    for field in ("fold_id", "model_id"):
        if not str(receipt_payload.get(field) or "").strip():
            raise FutureValueDraftScoreError(f"prediction ledger {field} is required")
    fit_ids = receipt_payload["fit_game_ids"]
    if not isinstance(fit_ids, list) or tuple(str(value) for value in fit_ids) != _normalise_ids(fit_ids, "prediction ledger fit IDs"):
        raise FutureValueDraftScoreError("prediction ledger fit IDs are not canonical")
    if receipt_payload["fit_game_identity_sha256"] != identity_sha256(tuple(str(value) for value in fit_ids)):
        raise FutureValueDraftScoreError("prediction ledger fit identity changed")
    if set(str(value) for value in fit_ids) & set(expected_ids):
        raise FutureValueDraftScoreError("prediction ledger fit and scored game IDs overlap")
    (
        model_payload,
        expected_model_receipt_hash,
        model_artifact_hash,
        model_artifact_payload,
    ) = _validate_prediction_model_receipt(
        model_receipt_path,
        expected_source_receipt_sha256=str(receipt_payload["source_receipt_sha256"]),
        expected_source_identity_sha256=str(receipt_payload["source_identity_sha256"]),
        expected_fold_id=str(receipt_payload["fold_id"]),
        expected_fit_game_ids=tuple(str(value) for value in fit_ids),
        expected_fit_id=str(receipt_payload["fit_id"]),
        expected_model_id=str(receipt_payload["model_id"]),
        expected_coefficient_sha256=str(receipt_payload["coefficient_sha256"]),
        expected_variant=variant,
    )
    if model_receipt_hash != expected_model_receipt_hash:
        raise FutureValueDraftScoreError("prediction model receipt file changed")
    if str(receipt_payload["model_artifact_locator"]) != str(model_payload["artifact_locator"]):
        raise FutureValueDraftScoreError("prediction model artifact locator changed")
    if receipt_payload["model_artifact_bytes"] != model_payload["artifact_bytes"] or str(receipt_payload["model_artifact_sha256"]).lower() != model_artifact_hash:
        raise FutureValueDraftScoreError("prediction model artifact binding changed")
    if str(receipt_payload["model_implementation_sha256"]).lower() != str(model_payload["implementation_sha256"]).lower():
        raise FutureValueDraftScoreError("prediction model implementation changed")
    if variant is None:
        raise FutureValueDraftScoreError("prediction ledger variant binding is required")
    canonical_variant = _canonical_variant(variant)
    feature_path = _safe_locator(
        receipt_payload["feature_artifact_locator"],
        base=receipt_path.parent,
        field="prediction feature artifact",
    )
    feature_hash = _verify_file_digest(
        feature_path,
        expected_bytes=receipt_payload["feature_artifact_bytes"],
        expected_sha256=receipt_payload["feature_artifact_sha256"],
        field="prediction feature artifact",
    )
    (
        feature_ids,
        feature_names,
        feature_matrix,
        expected_feature_hash,
        _feature_bytes,
        feature_rows_hash,
    ) = _load_prediction_feature_artifact(
        feature_path,
        expected_source_receipt_sha256=str(receipt_payload["source_receipt_sha256"]),
        expected_source_identity_sha256=str(receipt_payload["source_identity_sha256"]),
        expected_fold_id=str(receipt_payload["fold_id"]),
        expected_variant=canonical_variant,
    )
    if feature_hash != expected_feature_hash or feature_ids != expected_ids:
        raise FutureValueDraftScoreError("prediction feature artifact binding changed")
    if list(feature_names) != receipt_payload["feature_names"]:
        raise FutureValueDraftScoreError("prediction feature names changed")
    if feature_rows_hash != str(receipt_payload["feature_rows_sha256"]).lower():
        raise FutureValueDraftScoreError("prediction feature rows changed")
    model_coefficients = {
        name: float(model_artifact_payload["coefficients"][name])
        for name in feature_names
    }
    expected_logits = _linear_model_logits(
        feature_names, model_coefficients, feature_matrix
    )

    parsed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"game_id", "model_logit"}:
            raise FutureValueDraftScoreError("prediction ledger row schema is invalid")
        game_id = str(row["game_id"])
        try:
            value = float(row["model_logit"])
        except (TypeError, ValueError) as error:
            raise FutureValueDraftScoreError("prediction ledger value is invalid") from error
        if not math.isfinite(value):
            raise FutureValueDraftScoreError("prediction ledger values are not finite")
        parsed.append({"game_id": game_id, "model_logit": value})
    if [row["game_id"] for row in parsed] != list(expected_ids):
        raise FutureValueDraftScoreError("prediction ledger row and game ID order changed")
    row_digest = _sha256(parsed)
    if row_digest != _require_hash(receipt_payload["row_digest_sha256"], "prediction ledger row_digest_sha256"):
        raise FutureValueDraftScoreError("prediction ledger row digest changed")
    values = np.asarray([row["model_logit"] for row in parsed], dtype=float)
    if values.shape != expected_logits.shape or not np.array_equal(values, expected_logits):
        raise FutureValueDraftScoreError(
            "prediction ledger logits differ from verified model output"
        )
    return values, artifact_hash


def score_draft_score_variant(
    design: DraftScoreVariantDesign,
    coefficients: Mapping[str, float] | None = None,
    *,
    coefficient_receipt: Mapping[str, Any] | None = None,
    fitted_coefficient_receipt: Mapping[str, Any] | None = None,
    independent_prediction_ledger: Path | str | None = None,
    prediction_ledger: Path | str | None = None,
    independent_prediction_ledger_receipt: Mapping[str, Any] | Path | str | None = None,
    prediction_ledger_receipt: Mapping[str, Any] | Path | str | None = None,
) -> DraftScoreVariantScore:
    """Score a fitted design and compare it with an independent ledger."""

    config = draft_score_variant_config(design.variant)
    if tuple(design.feature_frame.columns) != config.feature_names:
        raise FutureValueDraftScoreError("design feature columns changed")
    receipt = coefficient_receipt if coefficient_receipt is not None else fitted_coefficient_receipt
    external_ledger = independent_prediction_ledger if independent_prediction_ledger is not None else prediction_ledger
    external_ledger_receipt = (
        independent_prediction_ledger_receipt
        if independent_prediction_ledger_receipt is not None
        else prediction_ledger_receipt
    )
    coefficient_binding = _validate_coefficient_receipt(receipt, design, coefficients)
    weights = {feature: float(coefficients[feature]) for feature in config.feature_names}  # type: ignore[index]
    contribution = pd.DataFrame(index=design.feature_frame.index)
    grouped: dict[str, list[np.ndarray]] = {}
    for feature in config.feature_names:
        values = design.feature_frame[feature].to_numpy(dtype=float) * weights[feature]
        grouped.setdefault(_component_group(feature, config), []).append(values)
    for group, arrays in grouped.items():
        contribution[group] = np.sum(np.column_stack(arrays), axis=1)
    # Sum component columns in their insertion order.  The same values feed
    # both sides of the reconstruction gate.
    component_names = tuple(grouped)
    contribution["composite_logit"] = contribution.loc[:, list(component_names)].sum(axis=1)
    reconstruction = contribution["composite_logit"] - contribution.loc[:, list(component_names)].sum(axis=1)
    internal_error = float(np.max(np.abs(reconstruction.to_numpy(dtype=float))))
    independent_values, ledger_hash = _prediction_ledger_values(
        external_ledger,
        design.game_ids,
        receipt=external_ledger_receipt,
        source_receipt_sha256=design.source_binding.source_receipt_sha256,
        source_identity_sha256=design.source_binding.source_identity_sha256,
        fold_id=design.source_binding.fold_id,
        fit_game_ids=design.source_binding.fit_game_ids,
        fit_id=coefficient_binding["fit_id"],
        model_id=coefficient_binding["model_id"],
        coefficient_sha256=coefficient_binding["coefficient_sha256"],
        variant=design.variant.value,
    )
    external_error = float(
        np.max(np.abs(contribution["composite_logit"].to_numpy(dtype=float) - independent_values))
    )
    max_error = max(internal_error, external_error)
    if max_error > 1e-12:
        raise FutureValueDraftScoreError("component logits do not reconstruct composite logit")
    return DraftScoreVariantScore(
        variant=design.variant,
        game_ids=design.game_ids,
        components=contribution,
        coefficients=MappingProxyType(weights),
        source_binding=design.source_binding,
        static_composition_sha256=design.static_composition_sha256,
        component_reconstruction_error_max=max_error,
        independent_prediction_error_max=external_error,
        coefficient_receipt=coefficient_binding,
        prediction_ledger_sha256=ledger_hash,
    )


score_variant = score_draft_score_variant


def swap_variant_feature_frame(
    feature_frame: pd.DataFrame,
    variant: DraftScoreVariant | RatingVariant | str,
) -> pd.DataFrame:
    """Swap blue and red values in a canonical variant matrix."""

    config = draft_score_variant_config(variant)
    if tuple(feature_frame.columns) != config.feature_names:
        raise FutureValueDraftScoreError("feature frame does not match variant")
    output = feature_frame.copy()
    invariant = set(PHASE_SHAPE_INVARIANT_FEATURES)
    signed = [feature for feature in config.feature_names if feature not in invariant]
    output.loc[:, signed] = -output.loc[:, signed]
    return output


def validate_side_swap(
    original: pd.DataFrame,
    swapped: pd.DataFrame,
    variant: DraftScoreVariant | RatingVariant | str,
    *,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Check signed negation and invariant preservation after a side swap."""

    config = draft_score_variant_config(variant)
    if tuple(original.columns) != config.feature_names or tuple(swapped.columns) != config.feature_names:
        raise FutureValueDraftScoreError("side-swap frames do not match variant")
    invariant = set(PHASE_SHAPE_INVARIANT_FEATURES)
    expected = swap_variant_feature_frame(original, config.variant)
    differences = (expected - swapped).abs().to_numpy(dtype=float)
    max_error = float(np.nanmax(differences)) if differences.size else 0.0
    passed = bool(np.isfinite(max_error) and max_error <= tolerance)
    if not passed:
        raise FutureValueDraftScoreError(f"side-swap validation failed: max error {max_error}")
    return {
        "passed": True,
        "max_abs_error": max_error,
        "signed_feature_count": len(config.feature_names),
        "invariant_features": sorted(config.phase_diagnostic_features),
    }


validate_variant_side_swap = validate_side_swap


def swap_raw_blue_red_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Relabel raw blue and red inputs before rebuilding interactions."""

    output = frame.copy()
    columns = set(output.columns)
    swapped: set[str] = set()
    for column in tuple(output.columns):
        if "_blue_" in column:
            counterpart = column.replace("_blue_", "_red_", 1)
        elif "_red_" in column:
            counterpart = column.replace("_red_", "_blue_", 1)
        else:
            continue
        if counterpart not in columns or column in swapped or counterpart in swapped:
            continue
        left = output[column].copy()
        output[column] = output[counterpart].to_numpy()
        output[counterpart] = left.to_numpy()
        swapped.update({column, counterpart})
    signed_prefixes = (*STATIC_COMPOSITION_FEATURES, *CURRENT_RATING_SIGNED_MAP_FEATURES, *FUTURE_PLAYER_FORM_SIDE_FEATURES, *PHASE_RAW_FEATURES)
    for column in signed_prefixes:
        if column in output.columns:
            values = pd.to_numeric(output[column], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                raise FutureValueDraftScoreError(f"raw side-swap feature is not finite: {column}")
            output[column] = -values
    # Derived interaction columns must be rebuilt from the swapped raw sides.
    for column in CURVE_ATOM_INTERACTION_FEATURES:
        if column in output.columns:
            output = output.drop(columns=[column])
    return output


def validate_raw_side_swap(
    frame: pd.DataFrame,
    variant: DraftScoreVariant | RatingVariant | str,
    binding: DraftScoreProducerBinding | Mapping[str, Any],
    *,
    static_atom_receipt: Mapping[str, Any] | None = None,
    atom_receipt: Mapping[str, Any] | None = None,
    static_atom_receipt_path: Path | str | None = None,
    static_atom_artifact_root: Path | str | None = None,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Rebuild the variant after a raw side relabel and check antisymmetry."""

    original = build_draft_score_variant_design(
        frame,
        variant,
        binding,
        static_atom_receipt=static_atom_receipt,
        atom_receipt=atom_receipt,
        static_atom_receipt_path=static_atom_receipt_path,
        static_atom_artifact_root=static_atom_artifact_root,
    )
    swapped = build_draft_score_variant_design(
        swap_raw_blue_red_frame(frame),
        variant,
        binding,
        static_atom_receipt=static_atom_receipt,
        atom_receipt=atom_receipt,
        static_atom_receipt_path=static_atom_receipt_path,
        static_atom_artifact_root=static_atom_artifact_root,
        static_atom_side_swap_source_frame=frame,
    )
    expected = swap_variant_feature_frame(original.feature_frame, variant)
    max_error = float(np.max(np.abs((expected - swapped.feature_frame).to_numpy(dtype=float))))
    if not math.isfinite(max_error) or max_error > tolerance:
        raise FutureValueDraftScoreError(f"raw side-swap validation failed: max error {max_error}")
    if original.phase_diagnostics is not None and swapped.phase_diagnostics is not None:
        for name in PHASE_SHAPE_DIAGNOSTIC_FEATURES:
            if name in {"forecast_curve_available", "forecast_curve_missing", "forecast_gold_crossover_count", "forecast_xp_crossover_count", "forecast_gold_first_crossover_minute", "forecast_xp_first_crossover_minute"}:
                left = original.phase_diagnostics[name].to_numpy()
                right = swapped.phase_diagnostics[name].to_numpy()
                if list(left) != list(right):
                    raise FutureValueDraftScoreError(f"raw side-swap invariant changed: {name}")
    return {"passed": True, "max_abs_error": max_error}


def variant_registry_receipt() -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "variants": {variant.value: VARIANT_CONFIGS[variant].receipt() for variant in DraftScoreVariant},
        "static_composition_features": list(STATIC_COMPOSITION_FEATURES),
        "authority": AUTHORITY,
    }
    payload["registry_sha256"] = _sha256(payload)
    return payload


__all__ = [
    "AUTHORITY",
    "ATOMIZED_COMPOSITION_COMPONENTS",
    "ATOMIZED_COMPOSITION_FEATURES",
    "CURVE_ATOM_FAMILIES",
    "CURVE_ATOM_INTERACTION_FEATURES",
    "CURVE_INTERACTION_FEATURES",
    "DraftScoreError",
    "DraftScoreProducerBinding",
    "DraftScoreVariant",
    "DraftScoreVariantConfig",
    "DraftScoreVariantDesign",
    "DraftScoreVariantScore",
    "DRAFT_SCORE_VARIANT_CONFIGS",
    "DRAFT_SCORE_VARIANTS",
    "FutureValueDraftScoreError",
    "PHASE_FEATURES",
    "PHASE_MODEL_FEATURES",
    "PHASE_RAW_FEATURES",
    "PHASE_SHAPE_FEATURES",
    "PHASE_SHAPE_DIAGNOSTIC_FEATURES",
    "PHASE_SHAPE_AVAILABILITY_FEATURES",
    "PHASE_SHAPE_INVARIANT_FEATURES",
    "PHASE_SHAPE_SIGNED_FEATURES",
    "SCHEMA_VERSION",
    "PREDICTION_LEDGER_SCHEMA",
    "PREDICTION_FEATURE_ARTIFACT_SCHEMA",
    "MODEL_RECEIPT_SCHEMA",
    "MODEL_ARTIFACT_SCHEMA",
    "MODEL_TRAINER_ID",
    "STATIC_COMPOSITION_COMPONENTS",
    "STATIC_COMPOSITION_FEATURES",
    "VARIANT_CONFIGS",
    "assert_static_composition_parity",
    "build_curve_atom_interactions",
    "build_draft_score_variant_design",
    "draft_score_variant_config",
    "draft_score_trainer_implementation_sha256",
    "score_draft_score_variant",
    "score_variant",
    "make_coefficient_receipt",
    "write_independent_prediction_ledger",
    "write_fitted_prediction_model",
    "write_prediction_feature_artifact",
    "static_composition_parity_hash",
    "swap_variant_feature_frame",
    "swap_raw_blue_red_frame",
    "validate_feature_names",
    "validate_producer_binding",
    "validate_side_swap",
    "validate_variant_side_swap",
    "validate_raw_side_swap",
    "variant_config",
    "variant_registry_receipt",
]
