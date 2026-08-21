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
    "synergy",
    "counter",
    "same_role",
    "ally_synergy",
    "enemy_counter",
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
PHASE_FEATURES = (*PHASE_RAW_FEATURES, *PHASE_SHAPE_FEATURES)


CURVE_ATOM_FAMILIES = ("role", "synergy", "counter")
CURVE_ATOM_INTERACTION_FEATURES = tuple(
    f"curve_atom_{family}_{role}"
    for family in CURVE_ATOM_FAMILIES
    for role in ROLES
)
CURVE_INTERACTION_FEATURES = CURVE_ATOM_INTERACTION_FEATURES


_STATIC_ALIASES: dict[str, tuple[str, ...]] = {
    component: (
        f"composition_{component}_logit",
        f"composition_{component}",
        f"atom_{component}_logit",
        f"atom_{component}",
        f"draft_{component}_logit",
        f"draft_{component}",
        f"component_{component}_logit",
        f"{component}_logit",
    )
    for component in STATIC_COMPOSITION_COMPONENTS
}


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

    def __post_init__(self) -> None:
        accepted = _normalise_ids(self.accepted_game_ids, "accepted_game_ids")
        fit = _normalise_ids(self.fit_game_ids, "fit_game_ids")
        if not set(fit).issubset(set(accepted)):
            raise FutureValueDraftScoreError("fit_game_ids are outside accepted census")
        if str(self.source_identity_sha256).lower() != identity_sha256(accepted):
            raise FutureValueDraftScoreError("source identity does not match accepted census")
        _require_hash(self.source_receipt_sha256, "source_receipt_sha256")
        _require_hash(self.producer_receipt_sha256, "producer_receipt_sha256")
        window = _timestamp_text(self.fit_window_end, "fit_window_end")
        if not str(self.fold_id).strip():
            raise FutureValueDraftScoreError("fold_id is required")
        object.__setattr__(self, "accepted_game_ids", accepted)
        object.__setattr__(self, "fit_game_ids", fit)
        object.__setattr__(self, "source_receipt_sha256", str(self.source_receipt_sha256).lower())
        object.__setattr__(self, "source_identity_sha256", identity_sha256(accepted))
        object.__setattr__(self, "producer_receipt_sha256", str(self.producer_receipt_sha256).lower())
        object.__setattr__(self, "fit_window_end", window)
        if self.producer_receipt is not None:
            payload = dict(self.producer_receipt)
            claimed = str(payload.pop("receipt_sha256", "")).lower()
            if claimed != self.producer_receipt_sha256 or _sha256(payload) != claimed:
                raise FutureValueDraftScoreError("producer receipt hash does not match payload")

    @classmethod
    def create(
        cls,
        *,
        source_receipt_sha256: str,
        accepted_game_ids: Iterable[object],
        fit_game_ids: Iterable[object],
        fit_window_end: Any,
        fold_id: str,
        producer: str,
        producer_version: str = "v1",
    ) -> "DraftScoreProducerBinding":
        accepted = _normalise_ids(accepted_game_ids, "accepted_game_ids")
        fit = _normalise_ids(fit_game_ids, "fit_game_ids")
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "producer": str(producer),
            "producer_version": str(producer_version),
            "source_receipt_sha256": _require_hash(source_receipt_sha256, "source_receipt_sha256"),
            "source_identity_sha256": identity_sha256(accepted),
            "accepted_game_ids": list(accepted),
            "fit_game_ids": list(fit),
            "fit_window_end": _timestamp_text(fit_window_end, "fit_window_end"),
            "fold_id": str(fold_id),
        }
        receipt_hash = _sha256(payload)
        return cls(
            source_receipt_sha256=payload["source_receipt_sha256"],
            source_identity_sha256=payload["source_identity_sha256"],
            accepted_game_ids=accepted,
            fit_game_ids=fit,
            fit_window_end=payload["fit_window_end"],
            producer_receipt_sha256=receipt_hash,
            fold_id=str(fold_id),
            producer_receipt={**payload, "receipt_sha256": receipt_hash},
        )

    @property
    def verified(self) -> bool:
        return self.producer_receipt is not None

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_kind": SOURCE_KIND,
            "source_receipt_sha256": self.source_receipt_sha256,
            "source_identity_sha256": self.source_identity_sha256,
            "accepted_game_count": len(self.accepted_game_ids),
            "accepted_game_ids": list(self.accepted_game_ids),
            "fit_game_count": len(self.fit_game_ids),
            "fit_game_ids": list(self.fit_game_ids),
            "fit_window_end": self.fit_window_end,
            "fold_id": self.fold_id,
            "producer_receipt_sha256": self.producer_receipt_sha256,
        }


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
    game_ids = _normalise_ids(frame["game_id"].astype(str), "frame game_id")
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
    if "fit_window_end" in frame.columns:
        values = pd.to_datetime(frame["fit_window_end"], utc=True, errors="coerce")
        if values.isna().any() or not values.eq(cutoff).all():
            raise FutureValueDraftScoreError("frame fit window binding changed")
    return bound


def _forbidden_feature_reason(name: str) -> str | None:
    text = str(name).strip()
    normal = re.sub(r"[^a-z0-9_]", "", text.casefold())
    compact = normal.replace("_", "")
    if compact in _TARGET_TOKENS or any(token in compact for token in _TARGET_TOKENS):
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


def _feature_columns_for_variant(variant: DraftScoreVariant) -> tuple[str, ...]:
    return VARIANT_CONFIGS[variant].feature_names


def _resolve_static_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for component, canonical in zip(STATIC_COMPOSITION_COMPONENTS, STATIC_COMPOSITION_FEATURES):
        matches = [column for column in _STATIC_ALIASES[component] if column in frame.columns]
        if not matches:
            raise FutureValueDraftScoreError(f"missing static composition component: {component}")
        selected = frame[matches[0]]
        for other in matches[1:]:
            left = pd.to_numeric(selected, errors="coerce")
            right = pd.to_numeric(frame[other], errors="coerce")
            if not np.allclose(left.to_numpy(), right.to_numpy(), equal_nan=True, atol=0.0, rtol=0.0):
                raise FutureValueDraftScoreError(f"conflicting aliases for static component: {component}")
        values = pd.to_numeric(selected, errors="coerce")
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
    for design in values[1:]:
        if tuple(design.game_ids) != expected_ids:
            raise FutureValueDraftScoreError("variant game IDs differ")
        if design.static_composition_sha256 != expected_hash:
            raise FutureValueDraftScoreError("static composition changed between variants")
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
    missing = [name for name in PHASE_SHAPE_FEATURES if name not in output.columns]
    if not missing:
        return output
    raw_gold = [f"forecast_gold_diff_{checkpoint}" for checkpoint in (10, 15, 20, 25)]
    raw_xp = [f"forecast_xp_diff_{checkpoint}" for checkpoint in (10, 15, 20, 25)]
    if any(name not in output.columns for name in (*raw_gold, *raw_xp)):
        raise FutureValueDraftScoreError("phase shape features are missing")
    shape_rows: list[dict[str, float | None]] = []
    for _, row in output.iterrows():
        shape_rows.append(
            phase_shape_features(
                [row[name] for name in raw_gold],
                [row[name] for name in raw_xp],
                available=True,
            )
        )
    shape = pd.DataFrame(shape_rows, index=output.index)
    for name in PHASE_SHAPE_FEATURES:
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
            PHASE_SHAPE_FEATURES
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
    phase_shape = PHASE_SHAPE_FEATURES if variant in (DraftScoreVariant.SCALING_CURVE, DraftScoreVariant.BOTH) else ()
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
) -> DraftScoreVariantDesign:
    """Build one immutable, source-bound research matrix."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise FutureValueDraftScoreError("feature frame is empty")
    resolved = _canonical_variant(variant)
    config = draft_score_variant_config(resolved)
    validate_feature_names(config.feature_names)
    validate_producer_binding(frame, binding, require_exact_census=require_exact_census)
    work = _append_curve_interactions(frame, config)
    values = _normalise_feature_frame(work, config)
    game_ids = _normalise_ids(frame["game_id"].astype(str), "frame game_id")
    if len(game_ids) != len(frame) or len(set(frame["game_id"].astype(str))) != len(frame):
        raise FutureValueDraftScoreError("one design row is required per unique game")
    static_hash = static_composition_parity_hash(
        pd.concat([frame[["game_id"]], _resolve_static_columns(frame)], axis=1)
    )
    return DraftScoreVariantDesign(
        variant=resolved,
        game_ids=game_ids,
        feature_frame=values.reset_index(drop=True),
        source_binding=_binding_from_any(binding),
        static_composition_sha256=static_hash,
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
            "authority": AUTHORITY,
        }
        payload["receipt_sha256"] = _sha256(payload)
        return payload


def score_draft_score_variant(
    design: DraftScoreVariantDesign,
    coefficients: Mapping[str, float] | None = None,
) -> DraftScoreVariantScore:
    """Score a design and emit exact per-component reconstruction."""

    config = draft_score_variant_config(design.variant)
    if tuple(design.feature_frame.columns) != config.feature_names:
        raise FutureValueDraftScoreError("design feature columns changed")
    if coefficients is None:
        weights = {feature: 1.0 for feature in config.feature_names}
    else:
        if set(coefficients) != set(config.feature_names):
            raise FutureValueDraftScoreError("coefficient registry does not match variant")
        weights = {feature: float(coefficients[feature]) for feature in config.feature_names}
        if not all(math.isfinite(value) for value in weights.values()):
            raise FutureValueDraftScoreError("coefficients are not finite")
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
    max_error = float(np.max(np.abs(reconstruction.to_numpy(dtype=float))))
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
        "signed_feature_count": len(config.feature_names) - len(set(config.phase_shape_features) & invariant),
        "invariant_features": sorted(set(config.phase_shape_features) & invariant),
    }


validate_variant_side_swap = validate_side_swap


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
    "PHASE_RAW_FEATURES",
    "PHASE_SHAPE_FEATURES",
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
    "static_composition_parity_hash",
    "swap_variant_feature_frame",
    "validate_feature_names",
    "validate_producer_binding",
    "validate_side_swap",
    "validate_variant_side_swap",
    "variant_config",
    "variant_registry_receipt",
]
