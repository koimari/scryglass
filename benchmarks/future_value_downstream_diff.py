"""Compare the four future-value variants on one frozen downstream census.

This module is a small, file-only comparison harness.  It does not fit a
model, start a runtime, or publish an artifact.  The four roots are expected
to contain the same downstream rows.  The comparison then measures the
numeric and rank changes that each allowed model family causes.

The current rating is V1.  V2 adds future player form.  V3 adds the scaling
curve.  V4 adds both families.  Composition evidence is a direct Draft Score
consumer and remains fixed across all four roots.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import math
import re

from lol_kills.research.future_value_downstream import (
    DownstreamArtifactSpec,
    FutureValueDownstreamError,
    SOURCE_FIELDS,
    SOURCE_RECEIPT_SCHEMA_VERSION,
    _validate_source_receipt,
    required_artifact_specs,
)
from lol_kills.research.future_value_rating import RatingVariant
from lol_kills.research.future_value_rating import FUTURE_PLAYER_FORM_SIDE_FEATURES
from lol_kills.research.future_value_draft_score import (
    CURVE_ATOM_INTERACTION_FEATURES,
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    PHASE_RAW_FEATURES,
    PHASE_SHAPE_FEATURES,
    STATIC_COMPOSITION_FEATURES,
)


SCHEMA_VERSION = "scryglass:future-value-downstream-diff:v1"
DIFF_SCHEMA_VERSION = SCHEMA_VERSION
BASELINE_VARIANT = "current_only"
VARIANT_NAMES = (
    "current_only",
    "future_player_form",
    "scaling_curve",
    "both",
)

FUTURE_FORM_FAMILY = "future_player_form"
SCALING_FAMILY = "scaling_curve"
STATIC_COMPOSITION_COMPONENTS = (
    "base",
    "synergy",
    "counter",
    "same_role",
    "ally_synergy",
    "enemy_counter",
    "archetype_interactions",
    "composition",
)
# These names are the serialized Draft Score component ledger.  The
# component registry is explicit so a new field cannot enter a comparison by
# matching a loose keyword.
DRAFT_SCORE_COMPONENTS = (
    "composition_base_logit",
    "composition_synergy_logit",
    "composition_counter_logit",
    "composition_same_role_logit",
    "composition_ally_synergy_logit",
    "composition_enemy_counter_logit",
    "composition_archetype_interactions_logit",
    "current_rating_logit",
    "future_player_form_logit",
    "scaling_raw_logit",
    "scaling_shape_logit",
    "curve_atom_interaction_logit",
    "crossfit_composition_total",
    "crossfit_champion_main",
    "crossfit_role_champion",
    "crossfit_ally_synergy",
    "crossfit_archetype_synergy",
    "crossfit_enemy_counter",
    "crossfit_archetype_counter",
    "crossfit_same_role",
    "composite_logit",
)
REQUIRED_DRAFT_SCORE_COMPONENT_FIELDS = (
    "base",
    "synergy",
    "counter",
    "same_role",
    "ally_synergy",
    "enemy_counter",
    "archetype_interactions",
    "current_rating_logit",
    "future_player_form_logit",
    "scaling_raw_logit",
    "scaling_shape_logit",
    "curve_atom_interaction_logit",
    "crossfit_composition_total",
    "crossfit_champion_main",
    "crossfit_role_champion",
    "crossfit_ally_synergy",
    "crossfit_archetype_synergy",
    "crossfit_enemy_counter",
    "crossfit_archetype_counter",
    "crossfit_same_role",
    "composite_logit",
)
_RECONSTRUCTION_COMPONENT_FIELDS = tuple(
    field
    for field in REQUIRED_DRAFT_SCORE_COMPONENT_FIELDS
    if field != "composite_logit"
    and (
        not field.startswith("crossfit_")
    or field == "crossfit_composition_total"
    )
)
_DRAFT_COMPONENT_FIELD_MAP = {
    **{
        name: name.removeprefix("composition_").removesuffix("_logit")
        for name in STATIC_COMPOSITION_FEATURES
    },
    **{name: name for name in REQUIRED_DRAFT_SCORE_COMPONENT_FIELDS},
    "base": "base",
    "synergy": "synergy",
    "counter": "counter",
    "same_role": "same_role",
    "ally_synergy": "ally_synergy",
    "enemy_counter": "enemy_counter",
    "archetype_interactions": "archetype_interactions",
}

_TARGET_KEYS = frozenset(
    {
        "target",
        "target_value",
        "target_label",
        "y",
        "y_blue_win",
        "y_red_win",
        "observed_result",
        "observed_outcome",
        "outcome",
        "result",
        "winner",
        "winning_side",
        "actual",
        "actual_result",
        "actual_winner",
        "blue_result",
        "red_result",
        "blue_win",
        "red_win",
        "win",
        "won",
    }
)
# Legacy aliases are kept only for the exact atomized fields that have a
# declared producer.  Free-form component name matching is unsafe here.
_COMPONENT_ALIASES = {
    **{name.removeprefix("composition_").removesuffix("_logit"): name.removeprefix("composition_").removesuffix("_logit") for name in STATIC_COMPOSITION_FEATURES},
    **{name: name.removeprefix("composition_").removesuffix("_logit") for name in STATIC_COMPOSITION_FEATURES},
    "archetype": "archetype_interactions",
    "archetype_interaction": "archetype_interactions",
    "archetype_interactions": "archetype_interactions",
    "ally": "ally_synergy",
    "ally_synergy": "ally_synergy",
    "synergy": "synergy",
    "enemy": "enemy_counter",
    "enemy_counter": "enemy_counter",
    "counter": "counter",
    "same_role": "same_role",
    "base": "base",
}
_AGGREGATE_KEYS = frozenset(
    {"score", "draft_score", "draft_edge", "edge", "component_total", "reconstructed_score", "composite_logit"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_PROVENANCE_FIELDS = frozenset(
    {
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "source_receipt_sha256",
        "authority_receipt_sha256",
        "receipt_sha256",
        "schema_version",
        "status",
        "authority",
        "research_only",
        "public_authority",
        "public_player_rating",
        "public_team_rating",
        "public_probability",
        "promotion",
        "deployment",
        "odds",
        "expected_value",
        "recommendation",
        "betting",
        "accepted_game_ids",
        "source_files",
        "model_eligible_game_count",
        "model_eligible_identity_sha256",
        "model_eligible_game_ids",
        "bytes",
        "sha256",
        "locator",
        "path",
    }
)
_AUTHORITY_KEYWORDS = (
    "public",
    "promotion",
    "deployment",
    "probability",
    "odds",
    "expected_value",
    "recommendation",
    "betting",
)


def _registry(*fields: str, future: Iterable[str] = (), scaling: Iterable[str] = ()) -> Mapping[str, frozenset[str]]:
    """Build one explicit artifact field registry."""

    return MappingProxyType(
        {
            "common": frozenset(fields) | _PROVENANCE_FIELDS,
            FUTURE_FORM_FAMILY: frozenset(future),
            SCALING_FAMILY: frozenset(scaling),
        }
    )


_IDENTITY_FIELDS = frozenset(
    {
        "player_id", "player", "id", "name", "team_key", "team_id", "team",
        "game_uid", "game_id", "match_id", "champion", "role", "position",
        "scope", "patch", "tier", "tier_label", "rank", "rank_tier", "band",
        "league", "competition_tier", "pack_id", "release_id",
    }
)
_RATING_FIELDS = frozenset(
    {
        "mu_total", "mu_effective", "mu_regional", "mu_meta", "sigma", "rating_p10",
        "rating", "score", "win_rate", "delta", "games", "n_maps",
    }
)
_FUTURE_FIELDS = frozenset(
    {
        "future_value", "future_player_value", "future_team_value", "team_value",
        "future_player_form_logit", "future_team_context_logit", "future_quality_logit",
        *FUTURE_PLAYER_FORM_SIDE_FEATURES,
    }
)
_SCALING_FIELDS = frozenset(
    {
        *PHASE_RAW_FEATURES,
        *PHASE_SHAPE_FEATURES,
        *CURVE_ATOM_INTERACTION_FEATURES,
        "forecast_gold_diff_10", "forecast_gold_diff_15", "forecast_gold_diff_20", "forecast_gold_diff_25",
        "forecast_xp_diff_10", "forecast_xp_diff_15", "forecast_xp_diff_20", "forecast_xp_diff_25",
        "scaling_curve", "snowball_index", "comeback_resilience", "curve_available", "curve_missing",
    }
)
_DRAFT_FIELDS = frozenset(
    {
        *REQUIRED_DRAFT_SCORE_COMPONENT_FIELDS,
        *DRAFT_SCORE_COMPONENTS,
        "draft_score", "draft_edge", "score", "edge", "target", "y", "y_blue_win",
        "blue_result", "red_result", "result", "winner", "winning_side",
    }
)
_ARTIFACT_FIELD_REGISTRIES: Mapping[str, Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "player_ratings": _registry(*(_IDENTITY_FIELDS | _RATING_FIELDS), future=_FUTURE_FIELDS, scaling=_SCALING_FIELDS),
        "team_ratings": _registry(*(_IDENTITY_FIELDS | _RATING_FIELDS), future=_FUTURE_FIELDS, scaling=_SCALING_FIELDS),
        # Tier identity is deliberately exact.  ``scope`` is the canonical
        # competition scope, and aliases such as league are not identity.
        "tierlists": _registry(
            "champion", "role", "scope", "patch", "tier", "tier_label", "rank", "rank_tier", "band",
            "score", "rating", "win_rate", "delta", "games",
        ),
        "draft_score": _registry(*(_IDENTITY_FIELDS | _DRAFT_FIELDS), future=_FUTURE_FIELDS, scaling=_SCALING_FIELDS),
        "profiles": _registry(*(_IDENTITY_FIELDS | _RATING_FIELDS), future=_FUTURE_FIELDS, scaling=_SCALING_FIELDS),
        "matches": _registry(*(_IDENTITY_FIELDS | _RATING_FIELDS | _DRAFT_FIELDS), future=_FUTURE_FIELDS, scaling=_SCALING_FIELDS),
        "public_manifest": _registry(
            "pack_id", "release_id", "id", "source_game_count", "team_rating_rows", "player_rating_rows",
            "total_files", "total_bytes", "source_as_of", "source_identity_sha256",
        ),
    }
)
ARTIFACT_FIELD_REGISTRIES = _ARTIFACT_FIELD_REGISTRIES


class DownstreamDiffError(FutureValueDownstreamError):
    """The multi-variant comparison input is unsafe or malformed."""


class VariantName(str, Enum):
    """Stable names for the four registered downstream variants."""

    V1 = "current_only"
    V2 = "future_player_form"
    V3 = "scaling_curve"
    V4 = "both"


@dataclass(frozen=True)
class VariantSpec:
    """Immutable contract for one of the four allowed model variants."""

    name: str
    variant: RatingVariant
    ordinal: int
    changed_families: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            resolved = self.variant if isinstance(self.variant, RatingVariant) else RatingVariant(str(self.variant))
        except ValueError as error:
            raise DownstreamDiffError(f"unknown downstream variant: {self.variant!r}") from error
        canonical_name = str(self.name).strip()
        if canonical_name != resolved.value or canonical_name not in VARIANT_NAMES:
            raise DownstreamDiffError("VariantSpec name and rating variant are not canonical")
        if not isinstance(self.ordinal, int) or isinstance(self.ordinal, bool) or self.ordinal not in (1, 2, 3, 4):
            raise DownstreamDiffError("VariantSpec ordinal is invalid")
        expected = {
            "current_only": (),
            "future_player_form": (FUTURE_FORM_FAMILY,),
            "scaling_curve": (SCALING_FAMILY,),
            "both": (FUTURE_FORM_FAMILY, SCALING_FAMILY),
        }[canonical_name]
        families = tuple(str(value) for value in self.changed_families)
        if families != expected:
            raise DownstreamDiffError(f"VariantSpec changed families are invalid for {canonical_name}")
        object.__setattr__(self, "name", canonical_name)
        object.__setattr__(self, "variant", resolved)
        object.__setattr__(self, "changed_families", families)

    @property
    def key(self) -> str:
        return self.name

    @property
    def label(self) -> str:
        return f"V{self.ordinal}"

    @property
    def allowed_changed_families(self) -> tuple[str, ...]:
        return self.changed_families

    @property
    def family(self) -> tuple[str, ...]:
        return self.changed_families


_VARIANT_SPEC_VALUES = (
    VariantSpec(BASELINE_VARIANT, RatingVariant.CURRENT_ONLY, 1, ()),
    VariantSpec(FUTURE_FORM_FAMILY, RatingVariant.FUTURE_PLAYER_FORM, 2, (FUTURE_FORM_FAMILY,)),
    VariantSpec(SCALING_FAMILY, RatingVariant.SCALING_CURVE, 3, (SCALING_FAMILY,)),
    VariantSpec("both", RatingVariant.BOTH, 4, (FUTURE_FORM_FAMILY, SCALING_FAMILY)),
)
VARIANT_SPECS: Mapping[str, VariantSpec] = MappingProxyType(
    {spec.name: spec for spec in _VARIANT_SPEC_VALUES}
)
VARIANT_REGISTRY = VARIANT_SPECS
ALL_VARIANT_SPECS = _VARIANT_SPEC_VALUES
VARIANT_SPECS_BY_LABEL: Mapping[str, VariantSpec] = MappingProxyType(
    {spec.label: spec for spec in _VARIANT_SPEC_VALUES}
)
V1_SPEC, V2_SPEC, V3_SPEC, V4_SPEC = _VARIANT_SPEC_VALUES


def variant_specs() -> tuple[VariantSpec, ...]:
    """Return the exact four immutable variant specifications."""

    return _VARIANT_SPEC_VALUES


def required_variant_specs() -> tuple[VariantSpec, ...]:
    """Compatibility alias for :func:`variant_specs`."""

    return variant_specs()


def get_variant_spec(value: VariantSpec | VariantName | RatingVariant | str) -> VariantSpec:
    """Resolve V1/V2/V3/V4 and canonical variant names."""

    if isinstance(value, VariantSpec):
        return VARIANT_SPECS[value.name]
    if isinstance(value, VariantName):
        value = value.value
    if isinstance(value, RatingVariant):
        text = value.value
    else:
        text = str(value).strip().casefold()
    aliases = {
        "v1": BASELINE_VARIANT,
        "v1_current_only": BASELINE_VARIANT,
        "current": BASELINE_VARIANT,
        "v2": FUTURE_FORM_FAMILY,
        "v2_future_player_form": FUTURE_FORM_FAMILY,
        "future_form": FUTURE_FORM_FAMILY,
        "v3": SCALING_FAMILY,
        "v3_scaling_curve": SCALING_FAMILY,
        "scaling": SCALING_FAMILY,
        "v4": "both",
        "v4_both": "both",
    }
    text = aliases.get(text, text)
    if text not in VARIANT_SPECS:
        raise DownstreamDiffError(f"unknown downstream variant: {value!r}")
    return VARIANT_SPECS[text]


@dataclass
class _ArtifactData:
    spec: DownstreamArtifactSpec
    paths: tuple[Path, ...]
    payloads: tuple[Any, ...]
    rows: dict[str, dict[str, Any]]
    raw_rows: tuple[dict[str, Any], ...]
    duplicate_ids: tuple[str, ...]
    missing_identity_count: int
    field_paths: frozenset[str]
    source_bindings: tuple[dict[str, Any], ...]
    source_receipt_hashes: tuple[str, ...]
    bytes: int
    sha256: str


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DownstreamDiffError("value is not canonical JSON") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_finite(value: Any, *, path: str = "value") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DownstreamDiffError(f"nonfinite value at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, path=f"{path}[{index}]")


def _read_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DownstreamDiffError(f"artifact JSON cannot be read: {path}") from error
    _assert_finite(value, path=str(path))
    return value


def _has_symlink_component(path: Path) -> bool:
    current = Path(path)
    while True:
        # macOS exposes the standard temporary roots through these stable
        # system aliases.  A caller-created link below them remains unsafe.
        if current.is_symlink() and current not in {Path("/var"), Path("/tmp")}:
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _safe_root(path: Path) -> Path:
    path = Path(path)
    if _has_symlink_component(path) or not path.is_dir():
        raise DownstreamDiffError(f"artifact root is missing or unsafe: {path}")
    resolved = path.resolve()
    if resolved != path and path.is_symlink():
        raise DownstreamDiffError(f"artifact root is a symlink: {path}")
    return resolved


def _safe_artifact_paths(root: Path, spec: DownstreamArtifactSpec) -> tuple[Path, ...]:
    if spec.glob:
        paths = sorted(root.glob(spec.path))
    else:
        paths = [root / spec.path]
    if not paths:
        raise DownstreamDiffError(f"required artifact missing: {spec.name}")
    safe: list[Path] = []
    for raw_path in paths:
        if _has_symlink_component(raw_path) or not raw_path.is_file():
            raise DownstreamDiffError(f"required artifact missing or unsafe: {spec.name}")
        resolved = raw_path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise DownstreamDiffError(f"artifact path escapes root: {raw_path}") from error
        if resolved.is_symlink():
            raise DownstreamDiffError(f"artifact path is a symlink: {raw_path}")
        safe.append(raw_path)
    return tuple(safe)


def _identity_part(row: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    for field in fields:
        value = row.get(field)
        if value is None or isinstance(value, (Mapping, list, tuple)):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _row_identity(row: Mapping[str, Any], spec: DownstreamArtifactSpec) -> str | None:
    if spec.name == "tierlists":
        # Tier rows must remain comparable across variants.  Scope, patch,
        # role, and champion form the complete identity.  League aliases and
        # display names cannot silently change a row's key.
        exact_groups: tuple[tuple[str, ...], ...] = (
            ("scope",),
            ("patch",),
            ("role",),
            ("champion",),
        )
    else:
        exact_groups = spec.identity_groups
    parts: list[str] = []
    for group in exact_groups:
        value = _identity_part(row, group)
        if value is None and len(exact_groups) == 1 and row.get("_container_key") is not None:
            value = str(row["_container_key"]).strip() or None
        if value is None:
            return None
        parts.append(value)
    return "|".join(parts)


def _collect_rows(value: Any, spec: DownstreamArtifactSpec, *, container_key: str | None = None, depth: int = 0) -> list[dict[str, Any]]:
    if depth > 12:
        return []
    if isinstance(value, list):
        list_rows: list[dict[str, Any]] = []
        for child in value:
            list_rows.extend(_collect_rows(child, spec, container_key=container_key, depth=depth + 1))
        return list_rows
    if not isinstance(value, Mapping):
        return []
    row = dict(value)
    if container_key is not None:
        row.setdefault("_container_key", container_key)
    rows: list[dict[str, Any]] = []
    identity = _row_identity(row, spec)
    has_direct_value = any(field in value for field in spec.value_fields)
    if has_direct_value or spec.name == "public_manifest":
        rows.append(row)
    if spec.name == "public_manifest":
        return rows
    for key, child in value.items():
        if not isinstance(child, (Mapping, list)):
            continue
        use_key = str(key) if isinstance(child, Mapping) and key in {
            "games", "records", "by_game", "by_player", "by_team", "matches", "rows"
        } else None
        rows.extend(_collect_rows(child, spec, container_key=use_key, depth=depth + 1))
    return rows


def _payload_rows(payload: Any, spec: DownstreamArtifactSpec) -> list[dict[str, Any]]:
    if spec.name in {"draft_score", "profiles", "matches"} and isinstance(payload, Mapping):
        games = payload.get("games")
        if isinstance(games, Mapping):
            return [
                {**dict(value), "_container_key": str(key)}
                for key, value in games.items()
                if isinstance(value, Mapping)
            ]
        if isinstance(games, list):
            return [dict(value) for value in games if isinstance(value, Mapping)]
    return _collect_rows(payload, spec)


def _load_artifact(
    root: Path,
    spec: DownstreamArtifactSpec,
    *,
    expected_source: Mapping[str, Any] | None = None,
) -> _ArtifactData:
    paths = _safe_artifact_paths(root, spec)
    payloads = tuple(_read_json(path) for path in paths)
    raw_rows = tuple(row for payload in payloads for row in _payload_rows(payload, spec))
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    missing_identity = 0
    for row in raw_rows:
        identity = _row_identity(row, spec)
        if identity is None:
            missing_identity += 1
            continue
        if identity in by_id:
            duplicates.add(identity)
        else:
            by_id[identity] = row
    digest = hashlib.sha256(b"".join(_sha256(path).encode("ascii") for path in paths)).hexdigest()
    field_paths = frozenset().union(*(_field_paths(payload) for payload in payloads))
    source_bindings: tuple[dict[str, Any], ...] = ()
    source_receipt_hashes: tuple[str, ...] = ()
    if expected_source is not None:
        source_bindings, source_receipt_hashes, provenance_blockers = _artifact_provenance(
            payloads,
            expected=expected_source,
            artifact_name=spec.name,
        )
        if provenance_blockers:
            raise DownstreamDiffError(";".join(provenance_blockers))
    return _ArtifactData(
        spec=spec,
        paths=paths,
        payloads=payloads,
        rows=by_id,
        raw_rows=raw_rows,
        duplicate_ids=tuple(sorted(duplicates)),
        missing_identity_count=missing_identity,
        field_paths=field_paths,
        source_bindings=source_bindings,
        source_receipt_hashes=source_receipt_hashes,
        bytes=sum(path.stat().st_size for path in paths),
        sha256=digest,
    )


def _source_binding(value: Mapping[str, Any]) -> dict[str, Any] | None:
    if not all(field in value for field in SOURCE_FIELDS):
        return None
    as_of = value.get("source_as_of")
    count = value.get("source_game_count")
    identity = value.get("source_identity_sha256")
    if not isinstance(as_of, str) or not as_of.strip():
        return None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    if not isinstance(identity, str) or _SHA256_RE.fullmatch(identity) is None:
        return None
    try:
        parsed = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if canonical != as_of:
        return None
    return {
        "source_as_of": as_of,
        "source_game_count": count,
        "source_identity_sha256": identity,
    }


def _find_source_bindings(value: Any, *, depth: int = 0) -> list[dict[str, Any]]:
    if depth > 10:
        return []
    if isinstance(value, Mapping):
        found: list[dict[str, Any]] = []
        direct = _source_binding(value)
        if direct is not None:
            found.append(direct)
        for child in value.values():
            if isinstance(child, (Mapping, list)):
                found.extend(_find_source_bindings(child, depth=depth + 1))
        return found
    if isinstance(value, (list, tuple)):
        list_found: list[dict[str, Any]] = []
        for child in value:
            list_found.extend(_find_source_bindings(child, depth=depth + 1))
        return list_found
    return []


def _find_receipt_hashes(value: Any, *, depth: int = 0) -> list[str]:
    if depth > 10:
        return []
    if isinstance(value, Mapping):
        found: list[str] = []
        for key, child in value.items():
            if key in {"source_receipt_sha256", "authority_receipt_sha256"} and isinstance(child, str):
                found.append(child)
            elif key == "receipt_sha256" and isinstance(child, str):
                found.append(child)
            if isinstance(child, (Mapping, list, tuple)):
                found.extend(_find_receipt_hashes(child, depth=depth + 1))
        return found
    if isinstance(value, (list, tuple)):
        list_found: list[str] = []
        for child in value:
            list_found.extend(_find_receipt_hashes(child, depth=depth + 1))
        return list_found
    return []


def _find_accepted_game_id_lists(value: Any, *, depth: int = 0) -> list[tuple[str, ...]]:
    if depth > 16:
        return []
    found: list[tuple[str, ...]] = []
    if isinstance(value, Mapping):
        raw_ids = value.get("accepted_game_ids")
        if isinstance(raw_ids, list) and all(isinstance(item, str) for item in raw_ids):
            found.append(tuple(raw_ids))
        for child in value.values():
            if isinstance(child, (Mapping, list, tuple)):
                found.extend(_find_accepted_game_id_lists(child, depth=depth + 1))
    elif isinstance(value, (list, tuple)):
        for child in value:
            if isinstance(child, (Mapping, list, tuple)):
                found.extend(_find_accepted_game_id_lists(child, depth=depth + 1))
    return found


def _validate_embedded_receipts(value: Any, *, depth: int = 0) -> None:
    """Validate durable full receipts nested inside an artifact payload."""

    if depth > 10:
        return
    if isinstance(value, Mapping):
        if value.get("schema_version") == SOURCE_RECEIPT_SCHEMA_VERSION:
            try:
                _validate_source_receipt(value)
            except FutureValueDownstreamError as error:
                raise DownstreamDiffError(str(error)) from error
        for child in value.values():
            if isinstance(child, (Mapping, list, tuple)):
                _validate_embedded_receipts(child, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_embedded_receipts(child, depth=depth + 1)


def _coerce_expected_source(value: Mapping[str, Any] | Path) -> dict[str, Any]:
    receipt_path: Path | None = None
    if isinstance(value, Mapping):
        raw = dict(value)
    else:
        receipt_path = Path(value)
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise DownstreamDiffError("canonical source receipt is missing or unsafe")
        raw_value = _read_json(receipt_path)
        if not isinstance(raw_value, Mapping):
            raise DownstreamDiffError("canonical source receipt is not an object")
        raw = dict(raw_value)
    if raw.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION:
        raise DownstreamDiffError("canonical verified source receipt is required")
    try:
        validated = _validate_source_receipt(raw, receipt_path=receipt_path)
    except FutureValueDownstreamError as error:
        raise DownstreamDiffError(str(error)) from error
    result = {
        field: validated[field]
        for field in SOURCE_FIELDS
    }
    result["accepted_game_ids"] = list(validated["accepted_game_ids"])
    result["receipt_sha256"] = validated["receipt_sha256"]
    return result


def _same_source(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(field) == right.get(field) for field in SOURCE_FIELDS)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(child) for key, child in sorted(value.items(), key=lambda item: str(item[0])) if key != "_container_key"}
    if isinstance(value, list):
        return [_canonical_value(child) for child in value]
    if isinstance(value, tuple):
        return [_canonical_value(child) for child in value]
    return value


def _flatten_leaves(value: Any, *, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if key == "_container_key":
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_leaves(child, prefix=path))
        return result
    if isinstance(value, list):
        list_result: dict[str, Any] = {}
        for index, child in enumerate(value):
            list_result.update(_flatten_leaves(child, prefix=f"{prefix}[{index}]"))
        return list_result
    return {prefix: value} if prefix else {}


def _canonical_field_path(path: str) -> str:
    """Return a list-index-independent field path for registry checks."""

    return re.sub(r"\[\d+\]", "[]", path)


def _field_name(path: str) -> str:
    return _canonical_field_path(path).rsplit(".", 1)[-1].removesuffix("[]").casefold()


def _registry_for(spec: DownstreamArtifactSpec) -> Mapping[str, frozenset[str]]:
    try:
        return _ARTIFACT_FIELD_REGISTRIES[spec.name]
    except KeyError as error:
        raise DownstreamDiffError(f"artifact has no explicit field registry: {spec.name}") from error


def _registered_field_names(spec: DownstreamArtifactSpec) -> frozenset[str]:
    registry = _registry_for(spec)
    return frozenset().union(*registry.values())


def _field_family(spec: DownstreamArtifactSpec, field: str) -> str | None:
    registry = _registry_for(spec)
    for family in (FUTURE_FORM_FAMILY, SCALING_FAMILY):
        if field in registry[family]:
            return family
    return None


def _field_paths(value: Any) -> frozenset[str]:
    return frozenset(_canonical_field_path(path) for path in _flatten_leaves(value))


def _validate_authority_flags(value: Any, *, path: str = "artifact", depth: int = 0) -> list[str]:
    """Find every forbidden true authority flag in a JSON artifact."""

    if depth > 32:
        return [f"{path}:authority_scan_depth_exceeded"]
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).casefold().replace("-", "_")
            child_path = f"{path}.{raw_key}"
            if any(token in key for token in _AUTHORITY_KEYWORDS):
                true_value = (isinstance(child, bool) and child) or (
                    isinstance(child, str) and child.casefold().strip() in {"true", "yes", "1"}
                )
                if true_value:
                    found.append(child_path)
            if isinstance(child, (Mapping, list, tuple)):
                found.extend(_validate_authority_flags(child, path=child_path, depth=depth + 1))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            if isinstance(child, (Mapping, list, tuple)):
                found.extend(_validate_authority_flags(child, path=f"{path}[{index}]", depth=depth + 1))
    return found


def _artifact_provenance(
    payloads: Sequence[Any],
    *,
    expected: Mapping[str, Any],
    artifact_name: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], list[str]]:
    """Require one matching source binding and receipt hash per artifact."""

    bindings = tuple(binding for payload in payloads for binding in _find_source_bindings(payload))
    receipt_hashes = tuple(
        receipt_hash
        for payload in payloads
        for receipt_hash in _find_receipt_hashes(payload)
        if isinstance(receipt_hash, str)
    )
    census_lists = tuple(
        ids
        for payload in payloads
        for ids in _find_accepted_game_id_lists(payload)
    )
    blockers: list[str] = []
    if not bindings:
        blockers.append(f"{artifact_name}_source_binding_missing")
    elif any(not _same_source(binding, expected) for binding in bindings):
        blockers.append(f"{artifact_name}_source_binding_mismatch")
    expected_ids = tuple(str(item) for item in expected.get("accepted_game_ids", ()))
    if not census_lists:
        blockers.append(f"{artifact_name}_accepted_game_ids_missing")
    elif any(ids != expected_ids for ids in census_lists):
        blockers.append(f"{artifact_name}_accepted_game_ids_mismatch")
    expected_hash = expected.get("receipt_sha256")
    if not isinstance(expected_hash, str) or _SHA256_RE.fullmatch(expected_hash) is None:
        blockers.append(f"{artifact_name}_canonical_receipt_hash_missing")
    elif not receipt_hashes:
        blockers.append(f"{artifact_name}_source_receipt_hash_missing")
    elif any(receipt_hash != expected_hash for receipt_hash in receipt_hashes):
        blockers.append(f"{artifact_name}_source_receipt_hash_mismatch")
    return bindings, receipt_hashes, blockers


def _field_registry_checks(
    baseline: _ArtifactData,
    candidate: _ArtifactData,
    candidate_spec: VariantSpec,
) -> dict[str, list[str]]:
    """Check the full field union and the exact family allowance."""

    allowed = _registered_field_names(baseline.spec)
    unknown = sorted(
        path
        for path in (baseline.field_paths | candidate.field_paths)
        if _field_name(path) not in allowed
    )
    added = sorted(candidate.field_paths - baseline.field_paths)
    removed = sorted(baseline.field_paths - candidate.field_paths)
    disallowed_added: list[str] = []
    disallowed_removed: list[str] = []
    for path in added:
        family = _field_family(baseline.spec, _field_name(path))
        if family is None or family not in candidate_spec.changed_families:
            disallowed_added.append(path)
    for path in removed:
        family = _field_family(baseline.spec, _field_name(path))
        if family is None or family not in candidate_spec.changed_families:
            disallowed_removed.append(path)
    return {
        "unknown_fields": unknown,
        "added_fields": added,
        "removed_fields": removed,
        "disallowed_added_fields": disallowed_added,
        "disallowed_removed_fields": disallowed_removed,
    }


def _leaf_key(path: str) -> str:
    return re.sub(r"\[\d+\]", "", path.rsplit(".", 1)[-1]).casefold()


def _target_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    leaves = _flatten_leaves(row)
    selected: dict[str, Any] = {}
    for path, value in leaves.items():
        key = _leaf_key(path)
        if key in _TARGET_KEYS:
            selected[path] = _canonical_value(value)
    return selected


def _component_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    leaves = _flatten_leaves(row)
    selected: dict[str, Any] = {}
    for path, value in leaves.items():
        key = _leaf_key(path)
        canonical = _COMPONENT_ALIASES.get(key)
        if canonical in STATIC_COMPOSITION_COMPONENTS:
            selected[path] = _canonical_value(value)
    return selected


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def numeric_delta_stats(old: Sequence[float], candidate: Sequence[float], *, tolerance: float = 1e-12) -> dict[str, Any]:
    """Return deterministic statistics for paired finite numeric values."""

    if len(old) != len(candidate):
        raise DownstreamDiffError("numeric delta vectors have different lengths")
    deltas = [float(right) - float(left) for left, right in zip(old, candidate)]
    if any(not math.isfinite(value) for value in deltas):
        raise DownstreamDiffError("numeric delta contains a nonfinite value")
    if not deltas:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "mean_abs": None,
            "p95_abs": None,
            "max_abs": None,
            "changed_count": 0,
            "changed_fraction": 0.0,
        }
    ordered = sorted(deltas)
    absolute = sorted(abs(value) for value in deltas)
    p95_index = min(len(absolute) - 1, max(0, math.ceil(0.95 * len(absolute)) - 1))
    median = ordered[len(ordered) // 2] if len(ordered) % 2 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2.0
    changed = sum(abs(value) > tolerance for value in deltas)
    return {
        "count": len(deltas),
        "mean": sum(deltas) / len(deltas),
        "median": median,
        "mean_abs": sum(absolute) / len(absolute),
        "p95_abs": absolute[p95_index],
        "max_abs": absolute[-1],
        "changed_count": changed,
        "changed_fraction": changed / len(deltas),
    }


def _row_id(row: Mapping[str, Any], identity_fields: Sequence[str] | None = None) -> str | None:
    if identity_fields:
        value = _identity_part(row, identity_fields)
        return value
    for field in (
        "player_id", "player", "team_key", "team_id", "team", "game_uid", "game_id", "match_id", "champion_id", "champion", "id", "name"
    ):
        value = row.get(field)
        if value is not None and not isinstance(value, (Mapping, list, tuple)) and str(value).strip():
            return str(value).strip()
    return None


def _rank_rows(rows: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]], value_field: str, *, identity_fields: Sequence[str] | None = None, descending: bool = True) -> dict[str, int]:
    if isinstance(rows, Mapping):
        source: list[tuple[str | None, Mapping[str, Any]]] = [
            (str(identity), row) for identity, row in rows.items()
        ]
    else:
        source = [(None, row) for row in rows]
    values: list[tuple[str, float]] = []
    for mapped_identity, row in source:
        identity = mapped_identity if mapped_identity is not None and identity_fields is None else _row_id(row, identity_fields)
        value = _numeric(row.get(value_field))
        if identity is None or value is None:
            continue
        values.append((identity, value))
    values.sort(key=lambda item: ((-item[1] if descending else item[1]), item[0]))
    return {identity: rank for rank, (identity, _value) in enumerate(values, start=1)}


def rank_movement_metrics(
    old_rows: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    value_field: str,
    *,
    identity_fields: Sequence[str] | None = None,
    descending: bool = True,
    top_k_values: Sequence[int] = (5, 10, 25),
) -> dict[str, Any]:
    """Measure deterministic rank movement for one numeric field."""

    old_rank = _rank_rows(old_rows, value_field, identity_fields=identity_fields, descending=descending)
    candidate_rank = _rank_rows(candidate_rows, value_field, identity_fields=identity_fields, descending=descending)
    common = sorted(set(old_rank) & set(candidate_rank))
    movement = [candidate_rank[key] - old_rank[key] for key in common]
    if len(common) >= 2:
        squared = sum(float(candidate_rank[key] - old_rank[key]) ** 2 for key in common)
        spearman = 1.0 - (6.0 * squared) / (len(common) * (len(common) ** 2 - 1))
    else:
        spearman = None
    changed = sum(abs(value) > 0 for value in movement)
    rank_stats = numeric_delta_stats([old_rank[key] for key in common], [candidate_rank[key] for key in common])
    rank_stats["changed_fraction"] = changed / len(common) if common else 0.0
    top_k: dict[str, dict[str, Any]] = {}
    for requested_k in top_k_values:
        k = int(requested_k)
        if k <= 0:
            continue
        old_top = set(key for key, _rank in sorted(old_rank.items(), key=lambda item: item[1])[:k])
        candidate_top = set(key for key, _rank in sorted(candidate_rank.items(), key=lambda item: item[1])[:k])
        overlap = old_top & candidate_top
        entrants = sorted(candidate_top - old_top, key=lambda key: (candidate_rank[key], key))
        exits = sorted(old_top - candidate_top, key=lambda key: (old_rank[key], key))
        denominator = max(min(k, len(old_rank), len(candidate_rank)), 1)
        top_k[str(k)] = {
            "requested": k,
            "old_count": len(old_top),
            "candidate_count": len(candidate_top),
            "overlap_count": len(overlap),
            "overlap_fraction": len(overlap) / denominator,
            "overlap": sorted(overlap, key=lambda key: (old_rank.get(key, candidate_rank.get(key, 0)), key)),
            "entrants": entrants,
            "exits": exits,
        }
    # The deterministic tie break above gives each common row a unique rank.
    # Count inversions with a Fenwick tree so a full leaderboard stays
    # O(n log n) instead of quadratic in the number of rows.
    ordered_common = sorted(common, key=lambda key: old_rank[key])
    seen = [0] * (len(common) + 2)
    inversions = 0
    for index, identity in enumerate(ordered_common):
        rank = candidate_rank[identity]
        cursor = rank
        prior = 0
        while cursor:
            prior += seen[cursor]
            cursor -= cursor & -cursor
        inversions += index - prior
        cursor = rank
        while cursor < len(seen):
            seen[cursor] += 1
            cursor += cursor & -cursor
    pair_count = len(common) * (len(common) - 1) // 2
    top25 = top_k.get("25", {"entrants": [], "exits": []})
    return {
        "value_field": value_field,
        "matched_rows": len(common),
        "old_rows": len(old_rank),
        "candidate_rows": len(candidate_rank),
        "rank_correlation": spearman,
        "spearman": spearman,
        "mean_movement": rank_stats["mean"],
        "mean_abs_movement": rank_stats["mean_abs"],
        "p95_abs_movement": rank_stats["p95_abs"],
        "max_abs_movement": rank_stats["max_abs"],
        "changed_count": changed,
        "changed_fraction": changed / len(common) if common else 0.0,
        "rank_delta_stats": rank_stats,
        "top_k": top_k,
        "top_5_overlap": top_k.get("5", {}).get("overlap_fraction"),
        "top_10_overlap": top_k.get("10", {}).get("overlap_fraction"),
        "top_25_overlap": top_k.get("25", {}).get("overlap_fraction"),
        "entrants": list(top25.get("entrants", [])),
        "exits": list(top25.get("exits", [])),
        "pairwise_inversions": inversions,
        "pairwise_inversion_fraction": inversions / pair_count if pair_count else 0.0,
        "pairwise_comparisons": pair_count,
    }


def tier_transition_metrics(
    old_rows: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    identity_fields: Sequence[str] | None = None,
    tier_fields: Sequence[str] = ("tier", "tier_label", "rank_tier", "band"),
) -> dict[str, Any]:
    """Return tier transitions for matched rows."""

    def keyed(rows: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        source = rows.items() if isinstance(rows, Mapping) else ((None, row) for row in rows)
        for mapped_identity, row in source:
            identity = (
                str(mapped_identity)
                if mapped_identity is not None and identity_fields is None
                else _row_id(row, identity_fields)
            )
            if identity is not None:
                result[identity] = row
        return result

    def tier(row: Mapping[str, Any]) -> str | None:
        for field in tier_fields:
            value = row.get(field)
            if value is not None and str(value).strip():
                return str(value).strip()
        return None

    old = keyed(old_rows)
    candidate = keyed(candidate_rows)
    common = sorted(set(old) & set(candidate))
    transitions: Counter[tuple[str, str]] = Counter()
    missing = 0
    changed = 0
    records: list[dict[str, str]] = []
    for identity in common:
        left = tier(old[identity])
        right = tier(candidate[identity])
        if left is None or right is None:
            missing += 1
            continue
        transitions[(left, right)] += 1
        if left != right:
            changed += 1
            records.append({"identity": identity, "old": left, "candidate": right})
    matrix = {
        left: {right: count for (source, right), count in transitions.items() if source == left}
        for left in sorted({source for source, _right in transitions})
    }
    return {
        "matched_rows": len(common),
        "old_rows": len(old),
        "candidate_rows": len(candidate),
        "missing_tier_count": missing,
        "changed_count": changed,
        "changed_fraction": changed / max(len(common) - missing, 1),
        "transitions": {f"{left}->{right}": count for (left, right), count in sorted(transitions.items())},
        "transition_matrix": matrix,
        "records": records,
    }


def _numeric_fields(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[float]]:
    fields: dict[str, list[float]] = {}
    for row in rows:
        for path, value in _flatten_leaves(row).items():
            number = _numeric(value)
            if number is None:
                continue
            fields.setdefault(path, []).append(number)
    return fields


def _paired_numeric_deltas(old_rows: Mapping[str, Mapping[str, Any]], candidate_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    common = sorted(set(old_rows) & set(candidate_rows))
    fields = set(_numeric_fields(old_rows.values())) | set(_numeric_fields(candidate_rows.values()))
    deltas: dict[str, dict[str, Any]] = {}
    for field in sorted(fields):
        old_values: list[float] = []
        candidate_values: list[float] = []
        old_missing = 0
        candidate_missing = 0
        for identity in common:
            left = _numeric(_flatten_leaves(old_rows[identity]).get(field))
            right = _numeric(_flatten_leaves(candidate_rows[identity]).get(field))
            if left is None:
                old_missing += 1
            if right is None:
                candidate_missing += 1
            if left is not None and right is not None:
                old_values.append(left)
                candidate_values.append(right)
        stats = numeric_delta_stats(old_values, candidate_values)
        stats["old_missing_count"] = old_missing
        stats["candidate_missing_count"] = candidate_missing
        deltas[field] = stats
    return deltas


def _component_values(row: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for path, value in _flatten_leaves(row).items():
        key = _leaf_key(path)
        component = _DRAFT_COMPONENT_FIELD_MAP.get(key)
        if component is None:
            component = _COMPONENT_ALIASES.get(key)
        number = _numeric(value)
        if component is not None and component != "composite_logit" and number is not None:
            values[component] = number
    return values


def _aggregate_values(row: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for path, value in _flatten_leaves(row).items():
        key = _leaf_key(path)
        number = _numeric(value)
        if key in _AGGREGATE_KEYS and number is not None:
            values[path] = number
    return values


def _reconstruction_report(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    checked = 0
    failures: list[dict[str, Any]] = []
    missing_components: list[dict[str, Any]] = []
    maximum = 0.0
    for identity, row in sorted(rows.items()):
        components = _component_values(row)
        aggregates = _aggregate_values(row)
        missing = sorted(
            field
            for field in REQUIRED_DRAFT_SCORE_COMPONENT_FIELDS
            if field not in components and field != "composite_logit"
        )
        if "composite_logit" not in _aggregate_values(row):
            missing.append("composite_logit")
        if missing:
            missing_components.append({"identity": identity, "fields": missing})
            continue
        if not aggregates:
            continue
        expected = sum(components[field] for field in _RECONSTRUCTION_COMPONENT_FIELDS)
        for aggregate_path, aggregate in aggregates.items():
            checked += 1
            error = aggregate - expected
            maximum = max(maximum, abs(error))
            if abs(error) > 1e-9:
                failures.append({"identity": identity, "aggregate": aggregate_path, "error": error})
    return {
        "status": "passed" if not failures and not missing_components else "failed",
        "rows_checked": checked,
        "rows_missing_components": len(missing_components),
        "missing_components": missing_components,
        "maximum_absolute_error": maximum,
        "failures": failures,
    }


def draft_score_component_diff_metrics(
    old_rows: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    candidate_rows: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    *,
    identity_fields: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compare Draft Score components and verify score reconstruction."""

    def keyed(rows: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        source = rows.items() if isinstance(rows, Mapping) else ((None, row) for row in rows)
        for mapped_identity, row in source:
            identity = (
                str(mapped_identity)
                if mapped_identity is not None and identity_fields is None
                else _row_id(row, identity_fields)
            )
            if identity is not None:
                result[identity] = row
        return result

    old = keyed(old_rows)
    candidate = keyed(candidate_rows)
    common = sorted(set(old) & set(candidate))
    component_names: set[str] = set()
    for row in (*old.values(), *candidate.values()):
        component_names.update(_component_values(row))
    component_deltas: dict[str, dict[str, Any]] = {}
    for component in sorted(component_names):
        left: list[float] = []
        right: list[float] = []
        for identity in common:
            old_value = _component_values(old[identity]).get(component)
            candidate_value = _component_values(candidate[identity]).get(component)
            if old_value is not None and candidate_value is not None:
                left.append(old_value)
                right.append(candidate_value)
        component_deltas[component] = numeric_delta_stats(left, right)
    reconstruction = {
        "old": _reconstruction_report(old),
        "candidate": _reconstruction_report(candidate),
    }
    static_identical = True
    changed_static: list[str] = []
    for identity in common:
        static_left = _component_signature(old[identity])
        static_right = _component_signature(candidate[identity])
        if static_left != static_right:
            static_identical = False
            changed_static.append(identity)
    return {
        "matched_rows": len(common),
        "old_rows": len(old),
        "candidate_rows": len(candidate),
        "component_deltas": component_deltas,
        "numeric_deltas": _paired_numeric_deltas(old, candidate),
        "static_components_identical": static_identical,
        "static_component_changed_rows": changed_static,
        "reconstruction": reconstruction,
    }


def _rank_fields(spec: DownstreamArtifactSpec, rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    available = set(_numeric_fields(rows))
    preferred = {
        "player_ratings": ("rating_p10", "mu_effective", "mu_total"),
        "team_ratings": ("rating_p10", "mu_effective", "mu_total"),
        "tierlists": ("rank", "score", "rating", "win_rate"),
        "profiles": ("future_value", "team_value", "mu_effective", "mu_total"),
        "draft_score": ("score", "draft_edge", "draft_score"),
        "matches": ("future_team_value", "future_player_value", "draft_edge"),
        "public_manifest": (),
    }.get(spec.name, ())
    result: list[str] = []
    for field in preferred:
        if field in available or any(path.endswith(f".{field}") for path in available):
            result.append(field)
    return tuple(result)


def _row_target_and_component_checks(
    baseline: _ArtifactData,
    current: _ArtifactData,
    spec: VariantSpec,
    blockers: list[str],
) -> dict[str, Any]:
    common = sorted(set(baseline.rows) & set(current.rows))
    target_changed: list[str] = []
    static_changed: list[str] = []
    disallowed_changed: list[str] = []
    for identity in common:
        left = baseline.rows[identity]
        right = current.rows[identity]
        if _target_signature(left) != _target_signature(right):
            target_changed.append(identity)
        if _component_signature(left) != _component_signature(right):
            static_changed.append(identity)
        left_leaves = _flatten_leaves(left)
        right_leaves = _flatten_leaves(right)
        for path in sorted(set(left_leaves) & set(right_leaves)):
            family = _field_family(baseline.spec, _field_name(path))
            if family is None or family in spec.changed_families:
                continue
            if _canonical_value(left_leaves[path]) != _canonical_value(right_leaves[path]):
                disallowed_changed.append(f"{identity}:{path}")
    if target_changed:
        blockers.append(f"{spec.name}_target_changed")
    if static_changed:
        blockers.append(f"{spec.name}_static_composition_changed")
    if disallowed_changed:
        blockers.append(f"{spec.name}_disallowed_family_changed")
    return {
        "target_changed_rows": target_changed,
        "static_component_changed_rows": static_changed,
        "disallowed_family_changes": disallowed_changed,
    }


def _normalise_variant_roots(value: Mapping[Any, Path] | Sequence[Path]) -> tuple[dict[str, Path], list[str]]:
    blockers: list[str] = []
    result: dict[str, Path] = {}
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(zip(VARIANT_NAMES, value))
    else:
        return {}, ["variant_roots_invalid"]
    for raw_key, raw_path in items:
        try:
            spec = get_variant_spec(raw_key)
        except DownstreamDiffError:
            blockers.append(f"unknown_variant_{raw_key}")
            continue
        if spec.name in result:
            blockers.append(f"duplicate_variant_{spec.name}")
            continue
        result[spec.name] = Path(raw_path)
    missing = [name for name in VARIANT_NAMES if name not in result]
    if missing:
        blockers.append("missing_variants_" + "_".join(missing))
    extras = len(items) - len(result)
    if extras > 0 and not any(item.startswith(("unknown_variant_", "duplicate_variant_")) for item in blockers):
        blockers.append("variant_roots_extra")
    return result, blockers


def _runtime_root_map(
    roots: Mapping[str, Path],
    supplied: Mapping[Any, Path] | None,
) -> tuple[dict[str, Path], list[str]]:
    result: dict[str, Path] = {}
    blockers: list[str] = []
    if supplied is not None:
        for raw_key, path in supplied.items():
            try:
                name = get_variant_spec(raw_key).name
            except DownstreamDiffError:
                blockers.append(f"unknown_runtime_variant_{raw_key}")
                continue
            result[name] = Path(path)
    for name, root in roots.items():
        if name in result:
            continue
        candidate = root / "runtime"
        result[name] = candidate if candidate.exists() else root
    resolved: dict[str, Path] = {}
    for name, path in result.items():
        if _has_symlink_component(path) or not path.is_dir():
            blockers.append(f"{name}_runtime_root_symlink_or_missing")
            continue
        resolved[name] = path.resolve()
        owner_files = (
            ".future-value-runtime-owner.json",
            ".future_value_runtime_owner.json",
            ".rating-autoresearch-runtime-owner.json",
            "runtime-owner.json",
            "runtime_owner.json",
        )
        for owner_name in owner_files:
            owner_path = path / owner_name
            if not owner_path.exists():
                continue
            if _has_symlink_component(owner_path) or not owner_path.is_file():
                blockers.append(f"{name}_runtime_owner_symlink_or_invalid")
                continue
            try:
                owner_payload = _read_json(owner_path)
            except DownstreamDiffError:
                blockers.append(f"{name}_runtime_owner_invalid")
                continue
            if not isinstance(owner_payload, Mapping):
                blockers.append(f"{name}_runtime_owner_invalid")
                continue
            owner_variant = owner_payload.get("variant")
            if owner_variant is not None:
                try:
                    owner_name_value = get_variant_spec(owner_variant).name
                except DownstreamDiffError:
                    blockers.append(f"{name}_runtime_owner_variant_invalid")
                else:
                    if owner_name_value != name:
                        blockers.append(f"{name}_runtime_owner_mismatch")
    names = sorted(resolved)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = resolved[left_name]
            right = resolved[right_name]
            try:
                left.relative_to(right)
                shared = True
            except ValueError:
                try:
                    right.relative_to(left)
                    shared = True
                except ValueError:
                    shared = False
            if shared:
                blockers.append(f"shared_runtime_root_{left_name}_{right_name}")
    return resolved, blockers


def _authority() -> dict[str, bool]:
    return {
        "public_player_rating": False,
        "public_team_rating": False,
        "public_probability": False,
        "odds": False,
        "expected_value": False,
        "recommendation": False,
        "betting": False,
        "promotion": False,
        "deployment": False,
        "merge": False,
        "release": False,
        "integration": False,
    }


def validate_runtime_roots(
    variant_roots: Mapping[Any, Path] | Sequence[Path],
    runtime_roots: Mapping[Any, Path] | None = None,
) -> dict[str, Any]:
    """Validate ownership and isolation of the four runtime roots."""

    roots, blockers = _normalise_variant_roots(variant_roots)
    runtime_map, runtime_blockers = _runtime_root_map(roots, runtime_roots)
    blockers.extend(runtime_blockers)
    return {
        "status": "valid" if not blockers else "blocked",
        "runtime_roots": {name: str(path) for name, path in runtime_map.items()},
        "blockers": sorted(set(blockers)),
    }


def compare_downstream_variants(
    variant_roots: Mapping[Any, Path] | Sequence[Path],
    *,
    source_binding: Mapping[str, Any] | Path | None = None,
    source_receipt: Mapping[str, Any] | Path | None = None,
    runtime_roots: Mapping[Any, Path] | None = None,
    artifact_specs: Sequence[DownstreamArtifactSpec] | None = None,
) -> dict[str, Any]:
    """Compare V1 through V4 with strict source and row identity checks."""

    if source_binding is not None and source_receipt is not None:
        raise DownstreamDiffError("source_binding and source_receipt cannot both be supplied")
    supplied_source = source_receipt if source_receipt is not None else source_binding
    if supplied_source is None:
        raise DownstreamDiffError("canonical source binding is required")
    expected = _coerce_expected_source(supplied_source)
    roots, blockers = _normalise_variant_roots(variant_roots)
    runtime_map, runtime_blockers = _runtime_root_map(roots, runtime_roots)
    blockers.extend(runtime_blockers)
    resolved_roots: dict[str, Path] = {}
    for name, root in roots.items():
        try:
            resolved_roots[name] = _safe_root(root)
        except DownstreamDiffError:
            blockers.append(f"{name}_artifact_root_symlink_or_missing")
    root_paths = list(resolved_roots.values())
    if len(set(root_paths)) != len(root_paths):
        blockers.append("shared_variant_root")
    specs = tuple(artifact_specs or required_artifact_specs())
    spec_names = tuple(spec.name for spec in specs)
    if len(set(spec_names)) != len(spec_names):
        raise DownstreamDiffError("artifact specifications contain duplicates")
    unknown_specs = sorted(set(spec_names) - set(_ARTIFACT_FIELD_REGISTRIES))
    if unknown_specs:
        raise DownstreamDiffError("artifact has no explicit field registry: " + ",".join(unknown_specs))
    loaded: dict[str, dict[str, _ArtifactData]] = {}
    source_counts: dict[str, int] = {}
    for name, root in resolved_roots.items():
        loaded[name] = {}
        all_bindings: list[dict[str, Any]] = []
        receipt_path = root / "future-value-source-receipt.json"
        if receipt_path.is_symlink():
            blockers.append(f"{name}_source_receipt_symlink")
        elif receipt_path.is_file():
            try:
                receipt = _read_json(receipt_path)
                if not isinstance(receipt, Mapping):
                    raise DownstreamDiffError("source receipt is not an object")
                if receipt.get("schema_version") == SOURCE_RECEIPT_SCHEMA_VERSION:
                    try:
                        receipt = _validate_source_receipt(receipt, receipt_path=receipt_path)
                    except FutureValueDownstreamError as error:
                        raise DownstreamDiffError(str(error)) from error
                    if list(receipt.get("accepted_game_ids", ())) != list(expected.get("accepted_game_ids", ())):
                        blockers.append(f"{name}_accepted_game_ids_mismatch")
                    if receipt.get("receipt_sha256") != expected.get("receipt_sha256"):
                        blockers.append(f"{name}_source_receipt_mismatch")
                root_bindings = _find_source_bindings(receipt)
                all_bindings.extend(root_bindings)
                if not root_bindings:
                    blockers.append(f"{name}_source_binding_missing")
            except DownstreamDiffError:
                blockers.append(f"{name}_source_receipt_invalid")
        else:
            blockers.append(f"{name}_source_binding_missing")
        if receipt_path.is_file() and not receipt_path.is_symlink():
            try:
                receipt_payload = _read_json(receipt_path)
                authority_paths = _validate_authority_flags(
                    receipt_payload,
                    path=f"{name}.source_receipt",
                )
                blockers.extend(
                    f"{name}_forbidden_authority_{path}" for path in authority_paths
                )
            except DownstreamDiffError:
                pass
        for spec in specs:
            try:
                artifact = _load_artifact(root, spec, expected_source=expected)
            except DownstreamDiffError as error:
                message = str(error)
                if "nonfinite" in message:
                    blockers.append(f"{name}_{spec.name}_nonfinite")
                elif "symlink" in message or "unsafe" in message:
                    blockers.append(f"{name}_{spec.name}_symlink_or_unsafe")
                else:
                    blockers.append(f"{name}_{spec.name}_missing_or_invalid")
                for reason in (
                    "source_binding_mismatch",
                    "accepted_game_ids_mismatch",
                    "source_receipt_hash_mismatch",
                    "source_receipt_hash_missing",
                    "source_binding_missing",
                ):
                    if reason in message:
                        blockers.append(f"{name}_{spec.name}_{reason}")
                continue
            loaded[name][spec.name] = artifact
            try:
                for payload in artifact.payloads:
                    _validate_embedded_receipts(payload)
            except DownstreamDiffError as error:
                if "nonfinite" in str(error):
                    blockers.append(f"{name}_{spec.name}_nonfinite")
                else:
                    blockers.append(f"{name}_{spec.name}_source_receipt_invalid")
            for payload in artifact.payloads:
                authority_paths = _validate_authority_flags(
                    payload,
                    path=f"{name}.{spec.name}",
                )
                blockers.extend(
                    f"{name}_{spec.name}_forbidden_authority_{path}"
                    for path in authority_paths
                )
            source_counts[name] = source_counts.get(name, 0) + len(artifact.source_bindings)
            all_bindings.extend(artifact.source_bindings)
            if artifact.duplicate_ids:
                blockers.append(f"{name}_{spec.name}_duplicate_identity_rows")
            if artifact.missing_identity_count:
                blockers.append(f"{name}_{spec.name}_row_identity_missing")
            if not artifact.rows:
                blockers.append(f"{name}_{spec.name}_rows_missing")
        if not all_bindings:
            blockers.append(f"{name}_source_binding_missing")
        elif any(not _same_source(binding, expected) for binding in all_bindings):
            blockers.append(f"{name}_source_binding_mismatch")
        if "receipt_sha256" in expected:
            # A full receipt is compared only when a full receipt was supplied.
            if receipt_path.is_file() and not receipt_path.is_symlink():
                try:
                    receipt_value = _read_json(receipt_path)
                    if isinstance(receipt_value, Mapping) and receipt_value.get("receipt_sha256") != expected["receipt_sha256"]:
                        blockers.append(f"{name}_source_receipt_mismatch")
                except DownstreamDiffError:
                    pass
            for spec in specs:
                candidate_artifact = loaded.get(name, {}).get(spec.name)
                if candidate_artifact is None:
                    continue
                hashes = list(candidate_artifact.source_receipt_hashes)
                if any(receipt_hash != expected["receipt_sha256"] for receipt_hash in hashes):
                    blockers.append(f"{name}_source_receipt_hash_mismatch")
    baseline = loaded.get(BASELINE_VARIANT, {})
    artifacts_report: dict[str, Any] = {}
    rank_report: dict[str, Any] = {}
    tier_report: dict[str, Any] = {}
    draft_report: dict[str, Any] = {}
    component_report: dict[str, Any] = {}
    for spec in specs:
        baseline_data = baseline.get(spec.name)
        if baseline_data is None:
            continue
        artifact_entry: dict[str, Any] = {
            "consumers": list(spec.consumers),
            "paths": {BASELINE_VARIANT: [str(path) for path in baseline_data.paths]},
            "bytes": {BASELINE_VARIANT: baseline_data.bytes},
            "sha256": {BASELINE_VARIANT: baseline_data.sha256},
            "comparisons": {},
        }
        for name in VARIANT_NAMES:
            if name == BASELINE_VARIANT:
                continue
            current = loaded.get(name, {}).get(spec.name)
            if current is None:
                continue
            artifact_entry["paths"][name] = [str(path) for path in current.paths]
            artifact_entry["bytes"][name] = current.bytes
            artifact_entry["sha256"][name] = current.sha256
            old_ids = set(baseline_data.rows)
            current_ids = set(current.rows)
            added = sorted(current_ids - old_ids)
            removed = sorted(old_ids - current_ids)
            if added or removed:
                blockers.append(f"{name}_{spec.name}_row_loss")
            row_checks = _row_target_and_component_checks(baseline_data, current, get_variant_spec(name), blockers)
            registry_checks = _field_registry_checks(
                baseline_data,
                current,
                get_variant_spec(name),
            )
            if registry_checks["unknown_fields"]:
                blockers.append(f"{name}_{spec.name}_unknown_fields")
            if registry_checks["disallowed_added_fields"] or registry_checks["disallowed_removed_fields"]:
                blockers.append(f"{name}_{spec.name}_field_union_changed")
            deltas = _paired_numeric_deltas(baseline_data.rows, current.rows)
            comparison = {
                "old_rows": len(old_ids),
                "candidate_rows": len(current_ids),
                "matched_rows": len(old_ids & current_ids),
                "added_rows": added,
                "removed_rows": removed,
                "deltas": deltas,
                "numeric_delta_stats": deltas,
                "field_registry": registry_checks,
                **row_checks,
            }
            fields = _rank_fields(spec, (*baseline_data.rows.values(), *current.rows.values()))
            rank_metrics = {
                field: rank_movement_metrics(
                    baseline_data.rows,
                    current.rows,
                    field,
                    descending=not (spec.name == "tierlists" and field == "rank"),
                )
                for field in fields
            }
            if rank_metrics:
                comparison["rank_movement"] = rank_metrics
                rank_report.setdefault(spec.name, {})[name] = rank_metrics
            artifact_entry["comparisons"][name] = comparison
            if spec.name == "tierlists":
                tier = tier_transition_metrics(baseline_data.rows, current.rows)
                comparison["tier_transitions"] = tier
                tier_report.setdefault(spec.name, {})[name] = tier
            if spec.name == "draft_score":
                draft = draft_score_component_diff_metrics(baseline_data.rows, current.rows)
                comparison["draft_score_components"] = draft
                draft_report[name] = draft
                component_report.setdefault(name, {})["draft_score"] = draft["reconstruction"]
                if not draft["static_components_identical"]:
                    blockers.append(f"{name}_draft_score_static_components_changed")
                if draft["reconstruction"]["old"]["status"] == "failed" or draft["reconstruction"]["candidate"]["status"] == "failed":
                    blockers.append(f"{name}_draft_score_component_reconstruction_failed")
        artifacts_report[spec.name] = artifact_entry
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_research_only" if not blockers else "blocked",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "baseline_variant": BASELINE_VARIANT,
        "variants": {
            spec.name: {
                "label": spec.label,
                "ordinal": spec.ordinal,
                "rating_variant": spec.variant.value,
                "changed_families": list(spec.changed_families),
                "root": str(roots.get(spec.name, "")),
                "runtime_root": str(runtime_map.get(spec.name, "")),
            }
            for spec in variant_specs()
        },
        "source": expected,
        "runtime_roots": {name: str(path) for name, path in runtime_map.items()},
        "source_binding_counts": source_counts,
        "artifacts": artifacts_report,
        "rank_movement": rank_report,
        "tier_transitions": tier_report,
        "draft_score": draft_report,
        "component_reconstruction": component_report,
        "blockers": sorted(set(blockers)),
        "authority": _authority(),
        "claim_ceiling": "source-bound research comparison only; no public rating or prediction authority",
    }
    return report


def evaluate_downstream_diff(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for :func:`compare_downstream_variants`."""

    return compare_downstream_variants(*args, **kwargs)


def build_downstream_diff_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility alias for :func:`compare_downstream_variants`."""

    return compare_downstream_variants(*args, **kwargs)


def write_downstream_diff_report(path: Path, report: Mapping[str, Any]) -> None:
    """Write one deterministic research-only comparison report."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise DownstreamDiffError("downstream diff report schema is invalid")
    authority_paths = _validate_authority_flags(report, path="report")
    if authority_paths:
        raise DownstreamDiffError(
            "downstream diff report grants forbidden authority: "
            + ",".join(authority_paths)
        )
    destination = Path(path)
    if _has_symlink_component(destination):
        raise DownstreamDiffError(
            "downstream diff report destination or parent is a symlink"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _has_symlink_component(destination.parent) or destination.parent.is_symlink():
        raise DownstreamDiffError(
            "downstream diff report destination parent is a symlink"
        )
    if destination.exists() and not destination.is_file():
        raise DownstreamDiffError("downstream diff report destination is not a file")
    destination.write_text(
        json.dumps(dict(report), ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ARTIFACT_FIELD_REGISTRIES",
    "ALL_VARIANT_SPECS",
    "BASELINE_VARIANT",
    "DIFF_SCHEMA_VERSION",
    "DRAFT_SCORE_COMPONENTS",
    "REQUIRED_DRAFT_SCORE_COMPONENT_FIELDS",
    "DownstreamDiffError",
    "FUTURE_FORM_FAMILY",
    "SCALING_FAMILY",
    "SCHEMA_VERSION",
    "STATIC_COMPOSITION_COMPONENTS",
    "VARIANT_NAMES",
    "VARIANT_REGISTRY",
    "VARIANT_SPECS",
    "VARIANT_SPECS_BY_LABEL",
    "V1_SPEC",
    "V2_SPEC",
    "V3_SPEC",
    "V4_SPEC",
    "VariantName",
    "VariantSpec",
    "build_downstream_diff_report",
    "compare_downstream_variants",
    "draft_score_component_diff_metrics",
    "evaluate_downstream_diff",
    "get_variant_spec",
    "numeric_delta_stats",
    "rank_movement_metrics",
    "required_variant_specs",
    "tier_transition_metrics",
    "validate_runtime_roots",
    "variant_specs",
    "write_downstream_diff_report",
]
