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
import json
import math
import re
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
_ALLOWED_PRODUCER_TIMINGS = frozenset(
    {"pregame_strict_prior", "cross_fitted_pregame", "strict_prior_pregame"}
)


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


def _validate_source_receipt_payload(
    source_receipt: Mapping[str, Any] | None,
    *,
    expected_game_ids: Iterable[object] | None = None,
) -> dict[str, Any]:
    """Verify the canonical accepted-census receipt used by every producer.

    A hash-shaped string is not source evidence.  The complete payload must
    verify before a Draft Score design can be built.
    """

    if not isinstance(source_receipt, Mapping):
        raise FutureValueDraftScoreError("canonical verified source receipt is required")
    required = {
        "schema_version",
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "accepted_game_ids",
        "receipt_sha256",
    }
    missing = sorted(required - set(source_receipt))
    if missing:
        raise FutureValueDraftScoreError(
            "canonical source receipt is incomplete: " + ", ".join(missing)
        )
    if str(source_receipt["schema_version"]) != SOURCE_RECEIPT_SCHEMA:
        raise FutureValueDraftScoreError("canonical source receipt schema is invalid")
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
    receipt_hash = _require_hash(source_receipt["receipt_sha256"], "source receipt_sha256")
    payload = dict(source_receipt)
    payload.pop("receipt_sha256", None)
    if _sha256(payload) != receipt_hash:
        raise FutureValueDraftScoreError("canonical source receipt hash does not match payload")
    _timestamp(source_receipt["source_as_of"], "source_as_of")
    if expected_game_ids is not None and _normalise_ids(expected_game_ids, "expected game IDs") != accepted:
        raise FutureValueDraftScoreError("source receipt accepted census does not match binding")
    authority = source_receipt.get("authority")
    if authority is not None:
        if not isinstance(authority, Mapping) or authority.get("research_only") is not True:
            raise FutureValueDraftScoreError("source receipt authority is not research-only")
        forbidden = (
            "deployment",
            "merge",
            "promotion",
            "public_player_rating",
            "public_team_rating",
            "public_probability",
        )
        if any(bool(authority.get(key)) for key in forbidden):
            raise FutureValueDraftScoreError("source receipt grants public authority")
    source_files = source_receipt.get("source_files")
    if source_files is not None:
        if not isinstance(source_files, Mapping) or not source_files:
            raise FutureValueDraftScoreError("source receipt source_files are invalid")
        for label, record in source_files.items():
            if not isinstance(label, str) or not label.strip() or not isinstance(record, Mapping):
                raise FutureValueDraftScoreError("source receipt source file record is invalid")
            if re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256") or ""), re.I) is None:
                raise FutureValueDraftScoreError(f"source receipt file hash is invalid: {label}")
    return dict(source_receipt)


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

    def __post_init__(self) -> None:
        accepted = _normalise_ids(self.accepted_game_ids, "accepted_game_ids")
        fit = _normalise_ids(self.fit_game_ids, "fit_game_ids")
        if not set(fit).issubset(set(accepted)):
            raise FutureValueDraftScoreError("fit_game_ids are outside accepted census")
        source = _validate_source_receipt_payload(
            self.source_receipt,
            expected_game_ids=accepted,
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
        if self.producer_timing in {"observed", "postgame", "final", "checkpoint_observed"}:
            raise FutureValueDraftScoreError("producer timing is not pregame")
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
        if self.producer_receipt is None:
            raise FutureValueDraftScoreError("producer receipt payload is required")
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
        }
        if payload != expected_payload:
            raise FutureValueDraftScoreError("producer receipt payload does not match binding")

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
    ) -> "DraftScoreProducerBinding":
        accepted = _normalise_ids(accepted_game_ids, "accepted_game_ids")
        fit = _normalise_ids(fit_game_ids, "fit_game_ids")
        source = _validate_source_receipt_payload(source_receipt, expected_game_ids=accepted)
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
        receipt_hash = _sha256(payload)
        return cls(
            source_receipt_sha256=source_hash,
            source_identity_sha256=str(source["source_identity_sha256"]).lower(),
            accepted_game_ids=accepted,
            fit_game_ids=fit,
            fit_window_end=payload["fit_window_end"],
            producer_receipt_sha256=receipt_hash,
            fold_id=str(fold_id),
            producer_receipt={**payload, "receipt_sha256": receipt_hash},
            source_receipt=source,
            producer_name=str(producer),
            producer_family=family,
            fit_window_start=normalized_start,
            fit_game_dates=normalized_dates,
            series_safe_evidence=series_evidence,
            producer_timing=str(producer_timing),
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
    expected_artifact_sha256: str | None = None,
    expected_receipt_sha256: str | None = None,
    side_swap_source_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Bind canonical atom values to an existing producer artifact.

    The receipt is supplied by the atomized producer.  This function never
    creates a receipt from the input frame.
    """

    if not isinstance(receipt, Mapping):
        raise FutureValueDraftScoreError("verified atomized composition receipt is required")
    required = {
        "schema_version",
        "producer_name",
        "producer_family",
        "artifact_locator",
        "artifact_sha256",
        "artifact_receipt_sha256",
        "source_receipt_sha256",
        "feature_names",
        "component_values_sha256",
        "receipt_sha256",
    }
    missing = sorted(required - set(receipt))
    if missing:
        raise FutureValueDraftScoreError(
            "atomized composition receipt is incomplete: " + ", ".join(missing)
        )
    if str(receipt["schema_version"]) != STATIC_ATOM_RECEIPT_SCHEMA:
        raise FutureValueDraftScoreError("atomized composition receipt schema is invalid")
    if not str(receipt["producer_name"]).strip() or not str(receipt["producer_family"]).strip():
        raise FutureValueDraftScoreError("atomized composition producer identity is required")
    if not str(receipt["artifact_locator"]).strip():
        raise FutureValueDraftScoreError("atomized composition artifact locator is required")
    artifact_hash = _require_hash(receipt["artifact_sha256"], "atomized artifact_sha256")
    artifact_receipt_hash = _require_hash(receipt["artifact_receipt_sha256"], "atomized artifact_receipt_sha256")
    if source_receipt_sha256 is not None and str(receipt["source_receipt_sha256"]).lower() != _require_hash(source_receipt_sha256, "source_receipt_sha256"):
        raise FutureValueDraftScoreError("atomized source receipt binding changed")
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
    return {
        **dict(receipt),
        "artifact_sha256": artifact_hash,
        "artifact_receipt_sha256": artifact_receipt_hash,
        "receipt_sha256": receipt_hash,
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
        expected_artifact_sha256=static_atom_artifact_sha256,
        expected_receipt_sha256=static_atom_receipt_sha256,
        side_swap_source_frame=static_atom_side_swap_source_frame,
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
        "fit_window_start": design.source_binding.fit_window_start,
        "fit_window_end": design.source_binding.fit_window_end,
        "producer_name": producer_name or design.source_binding.producer_name,
        "producer_family": producer_family or design.source_binding.producer_family,
        "fit_game_dates": dict(design.source_binding.fit_game_dates or {}),
    }
    payload["receipt_sha256"] = _sha256(payload)
    return payload


def _prediction_ledger_values(
    ledger: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    game_ids: Sequence[str],
) -> tuple[np.ndarray, str]:
    if ledger is None:
        raise FutureValueDraftScoreError("independent prediction ledger is required")
    if isinstance(ledger, pd.DataFrame):
        work = ledger.copy()
    elif isinstance(ledger, Mapping):
        work = pd.DataFrame(ledger)
    else:
        work = pd.DataFrame(list(ledger))
    if "game_id" not in work.columns:
        raise FutureValueDraftScoreError("prediction ledger requires game_id")
    value_columns = [name for name in ("model_logit", "prediction") if name in work.columns]
    if len(value_columns) != 1:
        raise FutureValueDraftScoreError("prediction ledger requires one independent model_logit")
    if work["game_id"].astype(str).tolist() != list(game_ids):
        raise FutureValueDraftScoreError("prediction ledger row and game ID order changed")
    values = pd.to_numeric(work[value_columns[0]], errors="coerce").to_numpy(dtype=float)
    if len(values) != len(game_ids) or not np.isfinite(values).all():
        raise FutureValueDraftScoreError("prediction ledger values are not finite")
    rows = [
        {"game_id": str(game_id), "model_logit": float(value)}
        for game_id, value in zip(game_ids, values)
    ]
    return values, _sha256(rows)


def score_draft_score_variant(
    design: DraftScoreVariantDesign,
    coefficients: Mapping[str, float] | None = None,
    *,
    coefficient_receipt: Mapping[str, Any] | None = None,
    fitted_coefficient_receipt: Mapping[str, Any] | None = None,
    independent_prediction_ledger: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    prediction_ledger: pd.DataFrame | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> DraftScoreVariantScore:
    """Score a fitted design and compare it with an independent ledger."""

    config = draft_score_variant_config(design.variant)
    if tuple(design.feature_frame.columns) != config.feature_names:
        raise FutureValueDraftScoreError("design feature columns changed")
    receipt = coefficient_receipt if coefficient_receipt is not None else fitted_coefficient_receipt
    external_ledger = independent_prediction_ledger if independent_prediction_ledger is not None else prediction_ledger
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
    independent_values, ledger_hash = _prediction_ledger_values(external_ledger, design.game_ids)
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
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Rebuild the variant after a raw side relabel and check antisymmetry."""

    original = build_draft_score_variant_design(
        frame,
        variant,
        binding,
        static_atom_receipt=static_atom_receipt,
        atom_receipt=atom_receipt,
    )
    swapped = build_draft_score_variant_design(
        swap_raw_blue_red_frame(frame),
        variant,
        binding,
        static_atom_receipt=static_atom_receipt,
        atom_receipt=atom_receipt,
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
    "STATIC_COMPOSITION_COMPONENTS",
    "STATIC_COMPOSITION_FEATURES",
    "VARIANT_CONFIGS",
    "assert_static_composition_parity",
    "build_curve_atom_interactions",
    "build_draft_score_variant_design",
    "draft_score_variant_config",
    "score_draft_score_variant",
    "score_variant",
    "make_coefficient_receipt",
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
