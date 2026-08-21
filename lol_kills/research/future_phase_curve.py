"""Development-only OE phase curves for future player and team value.

This module models blue-minus-red checkpoint state from information available
before the map.  It has one strict boundary: a checkpoint value describes the
target for the current map.  A final whole-map metric may enter a later map
only through a strict prior history.

The module has no GRID input and it does not emit win probabilities, odds,
expected value, recommendations, or betting data.  A fitted artifact remains
``development_only`` until an independent promotion process approves it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import weakref
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.research.atomized_rf_composite import (
    CHECKPOINTS as ATOM_CHECKPOINTS,
    GROUP_COLUMNS as ATOM_GROUP_COLUMNS,
)
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


PHASES = tuple(int(value) for value in ATOM_CHECKPOINTS)
PHASE_KEYS = tuple(str(value) for value in PHASES)
SCHEMA_VERSION = "scryglass:future-phase-curve:v1"
MODEL_VERSION = "future-phase-curve-v1"
SOURCE = "oracle_elixir_only"
PHASE_FEATURE_FAMILY = "checkpoint_forecasts"
PHASE_FEATURE_DECLARATION = tuple(ATOM_GROUP_COLUMNS[PHASE_FEATURE_FAMILY])
# This threshold is part of the phase-shape contract.  It is a signed
# gold/XP-difference unit, not a probability or a model decision boundary.
PHASE_SHAPE_MATERIAL_THRESHOLD = 250.0
MATERIAL_ADVANTAGE_THRESHOLD = PHASE_SHAPE_MATERIAL_THRESHOLD


def _phase_shape_metric_names(prefix: str) -> tuple[str, ...]:
    """Return the fixed shape names for one signed checkpoint curve."""

    return (
        *(f"forecast_{prefix}_slope_{first}_{second}" for first, second in zip(PHASES, PHASES[1:])),
        f"forecast_{prefix}_early_mean",
        f"forecast_{prefix}_late_mean",
        f"forecast_{prefix}_late_minus_early",
        f"forecast_{prefix}_late_minus_early_slope",
        f"forecast_{prefix}_late_minus_early_acceleration",
        f"forecast_{prefix}_signed_area",
        f"forecast_{prefix}_first_material_advantage_minute_signed",
    )


# These tuples are a closed feature contract.  Checkpoint target names such
# as ``forecast_gold_diff_10`` are intentionally absent.
PHASE_SHAPE_SIGNED_FEATURES = (
    *_phase_shape_metric_names("gold"),
    *_phase_shape_metric_names("xp"),
)
PHASE_SHAPE_INVARIANT_FEATURES = (
    *(f"forecast_{prefix}_first_crossover_minute" for prefix in ("gold", "xp")),
    *(f"forecast_{prefix}_crossover_count" for prefix in ("gold", "xp")),
    "forecast_curve_available",
    "forecast_curve_missing",
)
PHASE_SHAPE_AVAILABILITY_FEATURES = (
    "forecast_curve_available",
    "forecast_curve_missing",
)
PHASE_SHAPE_FEATURES = (
    *PHASE_SHAPE_SIGNED_FEATURES,
    *PHASE_SHAPE_INVARIANT_FEATURES,
)

# Short aliases keep the registry discoverable for callers that use the
# phase-curve terminology rather than the forecast terminology.
SIGNED_PHASE_SHAPE_FEATURES = PHASE_SHAPE_SIGNED_FEATURES
INVARIANT_PHASE_SHAPE_FEATURES = PHASE_SHAPE_INVARIANT_FEATURES
SOURCE_RECEIPT_SCHEMA = "scryglass:future-value-rating-source:v1"
SOURCE_TRANSPORT = (
    "official_public_oracles_elixir_annual_exports_plus_oe_api_bridge"
)
MIXED_SERIES_PARTITION_SOURCE = (
    "mixed:leaguepedia_crosswalk+conservative_series_superset"
)
MIXED_SERIES_PARTITION_KEY_FIELDS = (
    "league",
    "tournament",
    "unordered_team_pair",
)
_REQUIRED_ANNUAL_SOURCE_FILES = frozenset({"annual_2025", "annual_2026"})
_REQUIRED_BRIDGE_SOURCE_FILES = frozenset(
    {
        "bridge_oe_api_meta.json",
        "bridge_oe_api_player_games.parquet",
        "bridge_oe_api_team_games.parquet",
    }
)

# These are final map metrics.  The function accepts aliases that occur in
# OE exports, but checkpoint columns never enter this list.
FINAL_METRIC_ALIASES = (
    "cspm",
    "cspermin",
    "earnedgpm",
    "earned_gpm",
    "earnedgoldshare",
    "dpm",
    "damageshare",
    "damage_share",
    "kills",
    "deaths",
    "assists",
    "visionscore",
    "vision_score",
    "wardsplaced",
    "wardskilled",
    "damagetochampions",
    "damagetowers",
    "damagetotowers",
    "damagetakenperminute",
    "damagemitigatedperminute",
    "monsterkillsownjungle",
    "monsterkillsenemyjungle",
    "earnedgold",
    "totalgold",
)

class FuturePhaseCurveError(ValueError):
    """Raised when phase inputs violate the source or time contract."""


@dataclass(frozen=True)
class BoundPhaseSource:
    """A phase frame bound to one accepted source census."""

    frame: pd.DataFrame
    receipt: Mapping[str, Any]


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FuturePhaseCurveError("source receipt is not canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path | str) -> str:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise FuturePhaseCurveError("series crosswalk file is missing or unsafe")
    digest = hashlib.sha256()
    with value.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt_file_reference(
    path_value: Path | str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read and bind a durable source-receipt file."""

    path = Path(path_value)
    if path.is_symlink() or not path.is_file():
        raise FuturePhaseCurveError("source receipt file is missing or unsafe")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise FuturePhaseCurveError("source receipt file cannot be read") from error
    actual_sha256 = _sha256_bytes(raw)
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256).lower():
        raise FuturePhaseCurveError("source receipt file hash does not match")
    return {
        "locator": str(path),
        "bytes": len(raw),
        "sha256": actual_sha256,
    }


def _source_lineage(source_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the verified annual-export and OE bridge lineage."""

    source_files = source_receipt.get("source_files")
    if not isinstance(source_files, Mapping):
        raise FuturePhaseCurveError("source receipt source_files are missing")
    labels = {str(label) for label in source_files}
    missing_annual = sorted(_REQUIRED_ANNUAL_SOURCE_FILES - labels)
    missing_bridge = sorted(_REQUIRED_BRIDGE_SOURCE_FILES - labels)
    if missing_annual or missing_bridge:
        missing = [*missing_annual, *missing_bridge]
        raise FuturePhaseCurveError(
            "source receipt source_files do not prove the required OE transport: "
            + ", ".join(missing)
        )
    annual = sorted(label for label in labels if label in _REQUIRED_ANNUAL_SOURCE_FILES)
    bridge = sorted(label for label in labels if label in _REQUIRED_BRIDGE_SOURCE_FILES)
    return {
        "transport": SOURCE_TRANSPORT,
        "annual_exports": annual,
        "oe_api_bridge": bridge,
        "bridge_lineage": {
            "source": "oracle_elixir_api_bridge",
            "file_labels": bridge,
            "identity_binding": "accepted_game_ids and source-file hashes",
            "research_only": True,
        },
    }


def _validate_source_receipt(
    source_receipt: Mapping[str, Any],
    *,
    source_receipt_path: Path | str | None = None,
    source_receipt_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify the canonical source receipt before phase fitting."""

    if not isinstance(source_receipt, Mapping):
        raise FuturePhaseCurveError("verified source receipt is required")
    required = (
        "schema_version",
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "accepted_game_ids",
        "source_files",
        "receipt_sha256",
    )
    missing = [field for field in required if field not in source_receipt]
    if missing:
        raise FuturePhaseCurveError(
            "verified source receipt is incomplete: " + ", ".join(missing)
        )
    if source_receipt.get("schema_version") != SOURCE_RECEIPT_SCHEMA:
        raise FuturePhaseCurveError("verified source receipt schema is invalid")
    receipt_hash = str(source_receipt.get("receipt_sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", receipt_hash):
        raise FuturePhaseCurveError("verified source receipt hash is invalid")
    payload = dict(source_receipt)
    payload.pop("receipt_sha256", None)
    if _sha256_bytes(_canonical_json_bytes(payload)) != receipt_hash:
        raise FuturePhaseCurveError("verified source receipt hash does not match payload")
    raw_ids = source_receipt.get("accepted_game_ids")
    if not isinstance(raw_ids, list) or not all(isinstance(value, str) for value in raw_ids):
        raise FuturePhaseCurveError("verified source receipt accepted IDs are invalid")
    accepted = tuple(raw_ids)
    if not accepted or accepted != canonical_game_ids(accepted):
        raise FuturePhaseCurveError(
            "verified source receipt accepted IDs are not canonical and unique"
        )
    try:
        source_game_count = int(source_receipt["source_game_count"])
    except (TypeError, ValueError) as error:
        raise FuturePhaseCurveError("verified source receipt game count is invalid") from error
    if isinstance(source_receipt["source_game_count"], bool) or source_game_count != len(accepted):
        raise FuturePhaseCurveError("verified source receipt game count is invalid")
    expected_identity = identity_sha256(accepted)
    if str(source_receipt["source_identity_sha256"]).lower() != expected_identity:
        raise FuturePhaseCurveError("verified source receipt census identity is invalid")
    _as_timestamp(source_receipt["source_as_of"], "source_as_of")
    authority = source_receipt.get("authority")
    if not isinstance(authority, Mapping) or authority.get("research_only") is not True:
        raise FuturePhaseCurveError("verified source receipt authority is invalid")
    if any(
        bool(authority.get(name))
        for name in (
            "deployment",
            "merge",
            "promotion",
            "public_player_rating",
            "public_team_rating",
            "public_probability",
        )
    ):
        raise FuturePhaseCurveError("verified source receipt grants public authority")
    source_files = source_receipt.get("source_files")
    if not isinstance(source_files, Mapping) or not source_files:
        raise FuturePhaseCurveError("verified source receipt has no source file hashes")
    for label, record in source_files.items():
        if not isinstance(label, str) or not label.strip():
            raise FuturePhaseCurveError("verified source file label is invalid")
        if not isinstance(record, Mapping):
            raise FuturePhaseCurveError(f"verified source file record is invalid: {label}")
        locator = record.get("locator")
        if not isinstance(locator, str) or not locator.strip():
            raise FuturePhaseCurveError(f"verified source file locator is invalid: {label}")
        locator_path = Path(locator)
        if locator_path.is_absolute() or ".." in locator_path.parts:
            raise FuturePhaseCurveError(f"verified source file locator is unsafe: {label}")
        bytes_value = record.get("bytes")
        if isinstance(bytes_value, bool) or not isinstance(bytes_value, int) or bytes_value < 0:
            raise FuturePhaseCurveError(f"verified source file byte count is invalid: {label}")
        if re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256") or ""), re.I) is None:
            raise FuturePhaseCurveError(f"verified source file hash is invalid: {label}")
    _source_lineage(source_receipt)
    if (source_receipt_path is None) != (source_receipt_file_sha256 is None):
        raise FuturePhaseCurveError(
            "source receipt path and file hash must be provided together"
        )
    if source_receipt_path is not None:
        reference = _receipt_file_reference(
            source_receipt_path,
            expected_sha256=source_receipt_file_sha256,
        )
        try:
            on_disk = json.loads(Path(source_receipt_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FuturePhaseCurveError("source receipt file is not valid JSON") from error
        if not isinstance(on_disk, Mapping) or dict(on_disk) != dict(source_receipt):
            raise FuturePhaseCurveError("source receipt file payload does not match")
        if reference["sha256"] != str(source_receipt_file_sha256).lower():
            raise FuturePhaseCurveError("source receipt file hash does not match")
    return dict(source_receipt)


def verify_source_receipt_artifact(
    reference: Mapping[str, Any],
    *,
    runtime_root: Path | str = Path("."),
    expected_source_game_count: int | None = None,
    expected_source_identity_sha256: str | None = None,
    expected_source_as_of: str | None = None,
) -> dict[str, Any]:
    """Verify a durable source receipt referenced by a phase artifact."""

    locator = str(reference.get("locator") or "").strip()
    if not locator:
        raise FuturePhaseCurveError("source receipt artifact has no locator")
    path = Path(locator)
    if not path.is_absolute():
        path = Path(runtime_root) / path
    expected_bytes = reference.get("bytes")
    expected_sha256 = str(reference.get("sha256") or "").lower()
    expected_receipt_sha256 = str(
        reference.get("source_receipt_sha256") or ""
    ).lower()
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise FuturePhaseCurveError("source receipt artifact byte count is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise FuturePhaseCurveError("source receipt artifact hash is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_receipt_sha256):
        raise FuturePhaseCurveError("source receipt artifact payload hash is invalid")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise FuturePhaseCurveError("source receipt artifact is unavailable") from error
    if len(raw) != expected_bytes or _sha256_bytes(raw) != expected_sha256:
        raise FuturePhaseCurveError("source receipt artifact bytes or hash do not match")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FuturePhaseCurveError("source receipt artifact is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise FuturePhaseCurveError("source receipt artifact is not an object")
    verified = _validate_source_receipt(
        payload,
        source_receipt_path=path,
        source_receipt_file_sha256=expected_sha256,
    )
    if str(verified["receipt_sha256"]).lower() != expected_receipt_sha256:
        raise FuturePhaseCurveError(
            "source receipt artifact payload hash does not match"
        )
    if expected_source_game_count is not None and int(verified["source_game_count"]) != int(
        expected_source_game_count
    ):
        raise FuturePhaseCurveError("source receipt artifact count differs from source contract")
    if (
        expected_source_identity_sha256 is not None
        and str(verified["source_identity_sha256"]).lower()
        != str(expected_source_identity_sha256).lower()
    ):
        raise FuturePhaseCurveError(
            "source receipt artifact identity differs from source contract"
        )
    if expected_source_as_of is not None:
        if _as_timestamp(verified["source_as_of"], "source_as_of") != _as_timestamp(
            expected_source_as_of, "expected_source_as_of"
        ):
            raise FuturePhaseCurveError(
                "source receipt artifact source_as_of differs from source contract"
            )
    lineage = _source_lineage(verified)
    return {
        "status": "verified",
        "locator": locator,
        "bytes": expected_bytes,
        "sha256": expected_sha256,
        "source_receipt_sha256": str(verified["receipt_sha256"]),
        "source_game_count": int(verified["source_game_count"]),
        "source_identity_sha256": str(verified["source_identity_sha256"]),
        "source_as_of": str(verified["source_as_of"]),
        "transport": lineage["transport"],
        "lineage": lineage,
    }


def _as_timestamp(value: Any, field: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise FuturePhaseCurveError(f"{field} must be a timezone-aware timestamp")
    return stamp.tz_convert("UTC")


def _game_series(frame: pd.DataFrame, label: str = "phase frame") -> pd.Series:
    column = next(
        (name for name in ("game_uid", "gameid", "game_id") if name in frame.columns),
        None,
    )
    if column is None:
        raise FuturePhaseCurveError(f"{label} has no game identity column")
    fallback = frame["gameid"] if column == "game_uid" and "gameid" in frame.columns else None
    values = [
        canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
        for index, value in frame[column].items()
    ]
    result = pd.Series(values, index=frame.index, dtype="string")
    if result.eq("").any() or result.isna().any():
        raise FuturePhaseCurveError(f"{label} contains an empty game identity")
    return result


def _date_series(frame: pd.DataFrame, label: str = "phase frame") -> pd.Series:
    column = next(
        (name for name in ("date", "played_at", "game_date", "start_time") if name in frame.columns),
        None,
    )
    if column is None:
        raise FuturePhaseCurveError(f"{label} has no date column")
    result = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if result.isna().any():
        raise FuturePhaseCurveError(f"{label} contains an invalid date")
    return result


def _timestamp_text(value: pd.Timestamp) -> str:
    """Serialize a UTC timestamp with the artifact's stable Z suffix."""

    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _series_cluster_labels(metadata: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Build outcome-free series clusters and record their provenance.

    Numeric annual IDs provide a stable series proxy.  Other rows use the
    competition label, tournament, and unordered stable team keys.  The
    conservative proxy has no date.  It keeps the same matchup and event
    together when a series crosses a date boundary.  It can merge separate
    series with the same matchup and event.
    A row with incomplete identity keeps a game-level fallback and remains a
    promotion blocker.
    """

    labels: list[str] = []
    sources: list[str] = []
    numeric_pattern = re.compile(r"^(\d+-\d+_game)(?:_\d+)?$")

    def token(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    for _, row in metadata.iterrows():
        game_uid = token(row.get("game_uid"))
        if game_uid is None:
            raise FuturePhaseCurveError("series cluster input has an empty game identity")
        existing = token(row.get("series_id"))
        numeric_match = numeric_pattern.fullmatch(game_uid)
        if existing is not None:
            labels.append(existing)
            sources.append("exact_id_proxy")
            continue
        if numeric_match is not None:
            labels.append(numeric_match.group(1))
            sources.append("exact_id_proxy")
            continue
        league = token(row.get("league_source")) or token(row.get("league"))
        tournament = token(row.get("tournament")) or "<missing>"
        blue_team_id = token(row.get("blue_team_id"))
        red_team_id = token(row.get("red_team_id"))
        if blue_team_id is not None and red_team_id is not None:
            blue_team, red_team = blue_team_id, red_team_id
        else:
            blue_team = token(row.get("blue_team_key"))
            red_team = token(row.get("red_team_key"))
        team_keys = sorted(value for value in (blue_team, red_team) if value is not None)
        if league is not None and len(team_keys) == 2:
            labels.append(
                "team-tournament:"
                + "|".join((league, tournament, team_keys[0], team_keys[1]))
            )
            sources.append("team_tournament_proxy")
            continue
        labels.append("game-fallback:" + game_uid)
        sources.append("game_fallback")
    return (
        pd.Series(labels, index=metadata.index, dtype="string"),
        pd.Series(sources, index=metadata.index, dtype="string"),
    )


def _series_identity_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Report whether the frame has authoritative whole-series identity."""

    if "series_id_source" not in frame.columns:
        return {
            "status": "blocked",
            "authoritative": False,
            "source_counts": {},
            "blockers": ["series identity provenance is unavailable"],
        }
    values = frame["series_id_source"].astype("string")
    counts = {
        str(key): int(value)
        for key, value in values.value_counts(dropna=False).items()
    }
    blockers: list[str] = []
    if counts.get("team_tournament_proxy", 0):
        blockers.append(
            "team_tournament_proxy identities keep cross-date matchup events together but can merge separate series"
        )
    if counts.get("game_fallback", 0):
        blockers.append("game-level fallback identities cannot prove whole-series membership")
    if counts.get("leaguepedia_crosswalk", 0) and frame.attrs.get(
        "series_partition_authoritative"
    ) is False:
        blockers.append(
            "verified crosswalk coverage remains research-only and cannot grant authoritative series identity"
        )
    if frame.attrs.get("series_partition_proxy_authority_blocker") is True:
        blockers.append("authoritative_series_id_missing_proxy_cluster_used")
    unknown = sorted(
        key
        for key in counts
        if key
        not in {
            "exact_id_proxy",
            "leaguepedia_crosswalk",
            "team_tournament_proxy",
            "game_fallback",
        }
    )
    if unknown:
        blockers.append("unknown series identity provenance: " + ", ".join(unknown))
    cluster_frame = frame.loc[:, ["series_id", "series_id_source"]].copy()
    cluster_frame["date"] = _date_series(frame, "phase frame")
    cluster_frame["calendar_date"] = cluster_frame["date"].dt.floor("D")
    proxy = cluster_frame["series_id_source"].eq("team_tournament_proxy")
    proxy_clusters = cluster_frame.loc[proxy].groupby("series_id", dropna=False).agg(
        rows=("series_id", "size"), dates=("calendar_date", "nunique")
    )
    possible_collision_clusters = int((proxy_clusters["rows"] > 1).sum())
    possible_collision_rows = int(
        proxy_clusters.loc[proxy_clusters["rows"] > 1, "rows"].sum()
    )
    cross_date_clusters = int((proxy_clusters["dates"] > 1).sum())
    cross_date_rows = int(
        proxy_clusters.loc[proxy_clusters["dates"] > 1, "rows"].sum()
    )
    authoritative = not blockers and bool(counts)
    partition_binding = frame.attrs.get("series_partition")
    if isinstance(partition_binding, Mapping):
        assignment_sha256 = partition_binding.get("eligible_assignment_sha256")
        reference_assignment_sha256 = partition_binding.get(
            "reference_assignment_sha256"
        )
        status = str(
            partition_binding.get("cross_model_partition_status") or "non_comparable"
        )
        if status == "comparable" and assignment_sha256 != reference_assignment_sha256:
            status = "non_comparable"
        cross_model_partition = {
            "status": status,
            "source": partition_binding.get("source"),
            "mapping_sha256": partition_binding.get("mapping_sha256"),
            "crosswalk_sha256": partition_binding.get("crosswalk_sha256"),
            "receipt_sha256": partition_binding.get("receipt_sha256"),
            "receipt_file_sha256": partition_binding.get("receipt_file_sha256"),
            "eligible_game_count": partition_binding.get("eligible_game_count"),
            "eligible_identity_sha256": partition_binding.get(
                "eligible_identity_sha256"
            ),
            "eligible_assignment_sha256": assignment_sha256,
            "reference_assignment_sha256": reference_assignment_sha256,
            "reference_assignment_match": partition_binding.get(
                "reference_assignment_match", False
            ),
            "key_fields": list(
                partition_binding.get("key_fields")
                or MIXED_SERIES_PARTITION_KEY_FIELDS
            ),
            "reason": partition_binding.get(
                "cross_model_partition_reason"
            )
            or "phase and future-value evaluation need the same full-reference assignments",
        }
    else:
        cross_model_partition = {
            "status": "non_comparable",
            "phase_key_fields": [
                "league",
                "tournament",
                "unordered_stable_oe_team_pair",
            ],
            "other_model_key_fields": [
                "league",
                "tournament",
                "unordered_team_pair_alias_key",
            ],
            "reason": "phase and future-value evaluation must use one shared team crosswalk before their cluster metrics can be compared",
        }
    return {
        "status": "verified" if authoritative else "blocked",
        "authoritative": authoritative,
        "source_counts": counts,
        "cluster_counts": {
            str(source): int(cluster_frame.loc[cluster_frame["series_id_source"].eq(source), "series_id"].nunique())
            for source in counts
        },
        "unique_clusters": int(cluster_frame["series_id"].nunique()),
        "possible_collisions": {
            "source": "team_tournament_proxy",
            "clusters": possible_collision_clusters,
            "rows": possible_collision_rows,
            "cross_date_clusters": cross_date_clusters,
            "cross_date_rows": cross_date_rows,
            "definition": "proxy clusters with multiple maps may contain separate series because authoritative series IDs are unavailable",
        },
        "cross_model_partition": cross_model_partition,
        "blockers": blockers,
    }


def _phase_partition_map_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the outcome-free map view needed by the verified crosswalk.

    The Leaguepedia join uses team identity, competition, and time.  It does
    not use a match result.  A fixed zero target satisfies the shared map-frame
    validator while keeping phase targets outside the partition decision.
    """

    game_ids = _game_series(frame, "phase frame")
    dates = _date_series(frame, "phase frame")
    result = pd.DataFrame(
        {
            "game_id": game_ids.astype(str).to_numpy(),
            "date": dates.to_numpy(),
            "y_blue_win": np.zeros(len(frame), dtype=float),
        },
        index=frame.index,
    )
    for name in (
        "league",
        "league_source",
        "tournament",
        "blue_team_key",
        "red_team_key",
        "blue_team",
        "red_team",
        "blue_teamid",
        "red_teamid",
        "blue_team_id",
        "red_team_id",
    ):
        if name in frame.columns:
            result[name] = frame[name].to_numpy(copy=False)
    team_pairs = (
        ("blue_team_id", "red_team_id"),
        ("blue_teamid", "red_teamid"),
        ("blue_team_key", "red_team_key"),
        ("blue_team", "red_team"),
    )
    if not any(left in result.columns and right in result.columns for left, right in team_pairs):
        raise FuturePhaseCurveError(
            "verified phase series partition needs both blue and red team identities"
        )
    return result


def phase_series_assignment_sha256(
    frame: pd.DataFrame,
    *,
    game_column: str | None = None,
) -> str:
    """Hash the final game-to-series assignments in canonical game order."""

    if not isinstance(frame, pd.DataFrame) or "series_id" not in frame.columns:
        raise FuturePhaseCurveError("series assignment frame is incomplete")
    if game_column is None:
        game_column = next(
            (name for name in ("game_id", "game_uid", "gameid") if name in frame.columns),
            None,
        )
    if game_column is None:
        raise FuturePhaseCurveError("series assignment frame has no game identity")
    game_ids = frame[game_column].astype("string").str.strip()
    series_ids = frame["series_id"].astype("string").str.strip()
    if game_ids.isna().any() or game_ids.eq("").any():
        raise FuturePhaseCurveError("series assignment frame has an empty game identity")
    if series_ids.isna().any() or series_ids.eq("").any():
        raise FuturePhaseCurveError("series assignment frame has an empty series identity")
    if game_ids.duplicated().any():
        raise FuturePhaseCurveError("series assignment frame has duplicate game IDs")
    rows = sorted(
        (
            {"game_id": str(game_id), "series_id": str(series_id)}
            for game_id, series_id in zip(game_ids, series_ids)
        ),
        key=lambda row: row["game_id"],
    )
    return _sha256_bytes(_canonical_json_bytes(rows))


_REFERENCE_FACTORY_TOKEN = object()


@dataclass(frozen=True)
class _VerifiedPhaseSeriesReference:
    """Private, source-bound full-frame partition cache."""

    frame: pd.DataFrame
    source_game_count: int
    source_identity_sha256: str
    source_receipt_sha256: str
    crosswalk_artifact_sha256: str
    crosswalk_sha256: str
    crosswalk_assignment_sha256: str
    crosswalk_receipt_sha256: str
    crosswalk_receipt_file_sha256: str
    eligible_assignment_sha256: str
    reference_game_count: int
    reference_identity_sha256: str
    _factory_token: object

    def __post_init__(self) -> None:
        if self._factory_token is not _REFERENCE_FACTORY_TOKEN:
            raise FuturePhaseCurveError("series reference cache is not verified")


# This cache stores only a canonical remap result.  Receipt, file, frame and
# metadata checks still run on every reference use.  A weak reference avoids
# retaining a benchmark frame after its fit/evaluation call ends.
_REFERENCE_REMAP_CACHE: dict[int, tuple[Any, str, dict[str, str], dict[str, bool]]] = {}


def _phase_reference_raw_fingerprint(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return the raw crosswalk input and a cheap mutation fingerprint."""

    raw = _phase_partition_map_frame(frame)
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {
                "columns": list(raw.columns),
                "dtypes": [str(value) for value in raw.dtypes],
            }
        )
    )
    digest.update(
        pd.util.hash_pandas_object(raw, index=True)
        .to_numpy(dtype="uint64", copy=False)
        .tobytes()
    )
    mapped = frame["_series_crosswalk_mapped"]
    digest.update(
        pd.util.hash_pandas_object(mapped, index=True)
        .to_numpy(dtype="uint64", copy=False)
        .tobytes()
    )
    return raw, digest.hexdigest()


def _make_verified_phase_series_reference(
    frame: pd.DataFrame,
    *,
    source_receipt: Mapping[str, Any],
    crosswalk_path: Path | str,
    crosswalk_receipt_path: Path | str,
    crosswalk_receipt_file_sha256: str,
    eligible_ids: Sequence[str],
    eligible_assignment_sha256: str,
) -> _VerifiedPhaseSeriesReference:
    """Create the private cache only from a verified rating map frame."""

    if not isinstance(frame, pd.DataFrame) or not {
        "game_id",
        "series_id",
        "_series_crosswalk_mapped",
    }.issubset(frame.columns):
        raise FuturePhaseCurveError("verified series reference frame is incomplete")
    if frame.attrs.get("series_cluster_source") != MIXED_SERIES_PARTITION_SOURCE:
        raise FuturePhaseCurveError("verified series reference source changed")
    mapped = frame["_series_crosswalk_mapped"]
    if not pd.api.types.is_bool_dtype(mapped.dtype) or mapped.isna().any():
        raise FuturePhaseCurveError("verified series reference mapped flags are invalid")
    verified_receipt = _validate_source_receipt(source_receipt)
    source_receipt_sha256 = str(verified_receipt["receipt_sha256"]).lower()
    source_ids = tuple(canonical_game_ids(verified_receipt["accepted_game_ids"]))
    extras = verified_receipt.get("source_extra_game_ids")
    extra_ids = tuple(
        canonical_game_ids(
            extras.get("maps", ()) if isinstance(extras, Mapping) else ()
        )
    )
    expected_ids = tuple(canonical_game_ids((*source_ids, *extra_ids)))
    frame_ids = frame["game_id"].astype("string").str.strip()
    if frame_ids.isna().any() or frame_ids.eq("").any() or frame_ids.duplicated().any():
        raise FuturePhaseCurveError("verified series reference IDs are invalid")
    if tuple(canonical_game_ids(frame_ids.astype(str))) != expected_ids:
        raise FuturePhaseCurveError("verified series reference IDs differ from source")
    series = frame["series_id"].astype("string").str.strip()
    if series.isna().any() or series.eq("").any():
        raise FuturePhaseCurveError("verified series reference assignments are incomplete")
    audit = frame.attrs.get("series_cluster_audit")
    if not isinstance(audit, Mapping):
        raise FuturePhaseCurveError("verified series reference audit is missing")
    if str(audit.get("source_receipt_sha256") or "").lower() != source_receipt_sha256:
        raise FuturePhaseCurveError("verified series reference source receipt changed")
    crosswalk_path = Path(crosswalk_path)
    crosswalk_receipt_path = Path(crosswalk_receipt_path)
    crosswalk_artifact_sha256 = str(audit.get("crosswalk_artifact_sha256") or "").lower()
    crosswalk_sha256 = str(audit.get("crosswalk_sha256") or "").lower()
    crosswalk_assignment_sha256 = str(
        audit.get("crosswalk_assignment_sha256") or ""
    ).lower()
    crosswalk_receipt_sha256 = str(audit.get("crosswalk_receipt_sha256") or "").lower()
    crosswalk_receipt_file_sha256 = str(crosswalk_receipt_file_sha256).lower()
    if _sha256_file(crosswalk_path) != crosswalk_artifact_sha256:
        raise FuturePhaseCurveError("verified series crosswalk artifact changed")
    if _sha256_file(crosswalk_receipt_path) != crosswalk_receipt_file_sha256:
        raise FuturePhaseCurveError("verified series crosswalk receipt file changed")
    for value, label in (
        (crosswalk_artifact_sha256, "artifact"),
        (crosswalk_sha256, "crosswalk"),
        (crosswalk_assignment_sha256, "assignment"),
        (crosswalk_receipt_sha256, "receipt"),
        (crosswalk_receipt_file_sha256, "receipt file"),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise FuturePhaseCurveError(
                f"verified series crosswalk {label} hash is invalid"
            )
    eligible_set = set(str(value) for value in eligible_ids)
    if not eligible_set.issubset(set(frame_ids.astype(str))):
        raise FuturePhaseCurveError("verified series reference is missing eligible IDs")
    eligible_frame = frame.loc[frame_ids.astype(str).isin(eligible_set)]
    actual_assignment_sha256 = phase_series_assignment_sha256(
        eligible_frame,
        game_column="game_id",
    )
    expected_assignment = str(eligible_assignment_sha256).lower()
    if actual_assignment_sha256 != expected_assignment:
        raise FuturePhaseCurveError("verified series reference assignment digest changed")
    reference_identity_sha256 = identity_sha256(expected_ids)
    return _VerifiedPhaseSeriesReference(
        frame=frame.copy(deep=True),
        source_game_count=int(verified_receipt["source_game_count"]),
        source_identity_sha256=str(verified_receipt["source_identity_sha256"]),
        source_receipt_sha256=source_receipt_sha256,
        crosswalk_artifact_sha256=crosswalk_artifact_sha256,
        crosswalk_sha256=crosswalk_sha256,
        crosswalk_assignment_sha256=crosswalk_assignment_sha256,
        crosswalk_receipt_sha256=crosswalk_receipt_sha256,
        crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
        eligible_assignment_sha256=actual_assignment_sha256,
        reference_game_count=len(expected_ids),
        reference_identity_sha256=reference_identity_sha256,
        _factory_token=_REFERENCE_FACTORY_TOKEN,
    )


def _revalidate_verified_phase_series_reference(
    reference: _VerifiedPhaseSeriesReference,
    *,
    source_receipt: Mapping[str, Any],
    crosswalk_path: Path | str,
    crosswalk_receipt_path: Path | str,
    crosswalk_receipt_file_sha256: str,
    eligible_ids: Sequence[str],
    expected_assignment_sha256: str,
) -> _VerifiedPhaseSeriesReference:
    """Recheck a cached reference before it can affect a fit.

    The cache is an in-process optimisation.  Its DataFrame and metadata are
    mutable Python objects, so the object itself is never a provenance proof.
    Receipt, file, audit and assignment checks run on every use.  The exact
    crosswalk remap is cached only while the raw frame and source-file
    fingerprint stay unchanged.
    """

    if not isinstance(reference, _VerifiedPhaseSeriesReference):
        raise FuturePhaseCurveError("phase series reference cache is invalid")
    expected_assignment = str(expected_assignment_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_assignment):
        raise FuturePhaseCurveError("expected phase series assignment hash is invalid")
    verified = _make_verified_phase_series_reference(
        reference.frame,
        source_receipt=source_receipt,
        crosswalk_path=crosswalk_path,
        crosswalk_receipt_path=crosswalk_receipt_path,
        crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
        eligible_ids=eligible_ids,
        eligible_assignment_sha256=expected_assignment,
    )
    try:
        canonical_raw, raw_fingerprint = _phase_reference_raw_fingerprint(
            reference.frame
        )
        raw_fingerprint = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "raw_frame": raw_fingerprint,
                    "source_receipt_sha256": str(
                        source_receipt.get("receipt_sha256") or ""
                    ).lower(),
                    "crosswalk_artifact_sha256": _sha256_file(crosswalk_path),
                    "crosswalk_receipt_file_sha256": _sha256_file(
                        crosswalk_receipt_path
                    ),
                }
            )
        )
    except FuturePhaseCurveError:
        raise
    except Exception as error:
        raise FuturePhaseCurveError(
            "phase series reference raw frame is invalid"
        ) from error
    cached_remap = _REFERENCE_REMAP_CACHE.get(id(reference))
    canonical_series: dict[str, str] | None = None
    canonical_mapped: dict[str, bool] | None = None
    if (
        cached_remap is not None
        and cached_remap[0]() is reference
        and cached_remap[1] == raw_fingerprint
    ):
        canonical_series = cached_remap[2]
        canonical_mapped = cached_remap[3]
    else:
        try:
            from lol_kills.research.future_value_rating import (
                _map_model_frame,
                bind_verified_leaguepedia_series_crosswalk,
            )
            canonical_bound = bind_verified_leaguepedia_series_crosswalk(
                canonical_raw,
                crosswalk_path=crosswalk_path,
                receipt_path=crosswalk_receipt_path,
                source_receipt=source_receipt,
                expected_receipt_file_sha256=str(crosswalk_receipt_file_sha256),
            )
            canonical = _map_model_frame(
                canonical_bound,
                verified_source_receipt_sha256=str(source_receipt["receipt_sha256"]),
                verified_source_receipt=source_receipt,
                verified_crosswalk_receipt_file_sha256=str(
                    crosswalk_receipt_file_sha256
                ),
            )
        except FuturePhaseCurveError:
            raise
        except Exception as error:
            raise FuturePhaseCurveError(
                "phase series reference cannot be remapped from the verified crosswalk"
            ) from error
        canonical_ids = canonical["game_id"].astype(str)
        if canonical_ids.duplicated().any():
            raise FuturePhaseCurveError(
                "phase series reference remap has duplicate game IDs"
            )
        if "_series_crosswalk_mapped" not in canonical:
            raise FuturePhaseCurveError(
                "phase series reference remap has no crosswalk binding"
            )
        mapped = canonical["_series_crosswalk_mapped"]
        if not pd.api.types.is_bool_dtype(mapped.dtype) or mapped.isna().any():
            raise FuturePhaseCurveError(
                "phase series reference remap has invalid crosswalk flags"
            )
        canonical_series = dict(
            zip(canonical_ids, canonical["series_id"].astype(str))
        )
        canonical_mapped = dict(
            zip(canonical_ids, mapped.astype(bool))
        )
        _REFERENCE_REMAP_CACHE[id(reference)] = (
            weakref.ref(reference),
            raw_fingerprint,
            canonical_series,
            canonical_mapped,
        )
    assert canonical_series is not None
    assert canonical_mapped is not None
    canonical_ids = pd.Index(canonical_series)
    cached_ids = reference.frame["game_id"].astype(str)
    if tuple(canonical_game_ids(canonical_ids)) != tuple(canonical_game_ids(cached_ids)):
        raise FuturePhaseCurveError(
            "phase series reference game IDs differ from verified crosswalk"
        )
    cached_series = dict(
        zip(cached_ids, reference.frame["series_id"].astype(str))
    )
    if canonical_series != cached_series:
        raise FuturePhaseCurveError(
            "phase series reference assignments differ from verified crosswalk"
        )
    cached_mapped = dict(
        zip(cached_ids, reference.frame["_series_crosswalk_mapped"].astype(bool))
    )
    if canonical_mapped != cached_mapped:
        raise FuturePhaseCurveError(
            "phase series reference crosswalk flags differ from verified crosswalk"
        )
    for field in (
        "source_game_count",
        "source_identity_sha256",
        "source_receipt_sha256",
        "crosswalk_artifact_sha256",
        "crosswalk_sha256",
        "crosswalk_assignment_sha256",
        "crosswalk_receipt_sha256",
        "crosswalk_receipt_file_sha256",
        "eligible_assignment_sha256",
        "reference_game_count",
        "reference_identity_sha256",
    ):
        if getattr(reference, field) != getattr(verified, field):
            raise FuturePhaseCurveError(
                f"phase series reference cache {field} changed"
            )
    if reference.source_receipt_sha256 != str(
        source_receipt.get("receipt_sha256") or ""
    ).lower():
        raise FuturePhaseCurveError("pre-bound phase series source receipt changed")
    return verified


def bind_phase_series_partition(
    frame: pd.DataFrame,
    source_receipt: Mapping[str, Any],
    *,
    crosswalk_path: Path | str,
    crosswalk_receipt_path: Path | str,
    crosswalk_receipt_file_sha256: str,
    require_full_eligible: bool,
    reference_frame: pd.DataFrame | None = None,
    _verified_reference: _VerifiedPhaseSeriesReference | None = None,
    expected_assignment_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify and attach the future-value mixed series partition.

    This adapter intentionally delegates receipt and assignment validation to
    ``future_value_rating``.  That keeps phase and rating models on one
    crosswalk contract.  The import is local because the rating module imports
    this phase module for shared feature names.
    """

    if not isinstance(source_receipt, Mapping):
        raise FuturePhaseCurveError("verified source receipt is required for a series crosswalk")
    eligible_raw = source_receipt.get("model_eligible_game_ids")
    if not isinstance(eligible_raw, list) or not all(
        isinstance(value, str) for value in eligible_raw
    ):
        raise FuturePhaseCurveError(
            "source receipt has no model-eligible census for the series crosswalk"
        )
    eligible_ids = tuple(canonical_game_ids(eligible_raw))
    if int(source_receipt.get("model_eligible_game_count", -1)) != len(eligible_ids):
        raise FuturePhaseCurveError("source receipt model-eligible count changed")
    expected_eligible_identity = identity_sha256(eligible_ids)
    if str(source_receipt.get("model_eligible_identity_sha256") or "").lower() != expected_eligible_identity:
        raise FuturePhaseCurveError("source receipt model-eligible identity changed")
    try:
        from lol_kills.research.future_value_rating import (
            _map_model_frame,
            bind_verified_leaguepedia_series_crosswalk,
        )
    except ImportError as error:
        raise FuturePhaseCurveError("verified phase series partition is unavailable") from error
    if reference_frame is not None and _verified_reference is not None:
        raise FuturePhaseCurveError(
            "phase series partition accepts one reference frame"
        )
    if _verified_reference is not None and not isinstance(
        _verified_reference, _VerifiedPhaseSeriesReference
    ):
        raise FuturePhaseCurveError("phase series reference cache is invalid")
    if _verified_reference is not None and expected_assignment_sha256 is None:
        raise FuturePhaseCurveError(
            "pre-bound phase series assignment digest is required"
        )
    if _verified_reference is not None:
        _verified_reference = _revalidate_verified_phase_series_reference(
            _verified_reference,
            source_receipt=source_receipt,
            crosswalk_path=crosswalk_path,
            crosswalk_receipt_path=crosswalk_receipt_path,
            crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
            eligible_ids=eligible_ids,
            expected_assignment_sha256=str(expected_assignment_sha256),
        )
    reference = (
        _verified_reference.frame
        if _verified_reference is not None
        else reference_frame
    )
    if reference is None:
        reference = frame
    if not isinstance(reference, pd.DataFrame):
        raise FuturePhaseCurveError("phase series reference frame is invalid")
    try:
        if _verified_reference is not None:
            model_frame = reference.copy()
            model_frame.attrs = dict(reference.attrs)
            if not {"game_id", "series_id"}.issubset(model_frame.columns):
                raise FuturePhaseCurveError(
                    "pre-bound phase series reference frame is incomplete"
                )
            if model_frame.attrs.get("series_cluster_source") != (
                MIXED_SERIES_PARTITION_SOURCE
            ):
                raise FuturePhaseCurveError(
                    "pre-bound phase series reference source changed"
                )
            if _verified_reference.source_receipt_sha256 != str(
                source_receipt["receipt_sha256"]
            ).lower():
                raise FuturePhaseCurveError(
                    "pre-bound phase series source receipt changed"
                )
        else:
            raw_maps = _phase_partition_map_frame(reference)
            bound_maps = bind_verified_leaguepedia_series_crosswalk(
                raw_maps,
                crosswalk_path=crosswalk_path,
                receipt_path=crosswalk_receipt_path,
                source_receipt=source_receipt,
                expected_receipt_file_sha256=str(crosswalk_receipt_file_sha256),
            )
            model_frame = _map_model_frame(
                bound_maps,
                verified_source_receipt_sha256=str(source_receipt["receipt_sha256"]),
                verified_source_receipt=source_receipt,
                verified_crosswalk_receipt_file_sha256=str(crosswalk_receipt_file_sha256),
            )
    except Exception as error:
        # The source loader uses its own exception type.  Keep the phase API
        # fail-closed without exposing a second provenance exception family.
        if isinstance(error, FuturePhaseCurveError):
            raise
        raise FuturePhaseCurveError(
            "verified phase series crosswalk could not be bound"
        ) from error
    phase_ids = _game_series(frame, "phase frame").astype(str)
    phase_id_set = set(phase_ids)
    eligible_set = set(eligible_ids)
    if not phase_id_set.issubset(eligible_set):
        raise FuturePhaseCurveError(
            "phase frame contains accepted but model-ineligible games"
        )
    if require_full_eligible and phase_id_set != eligible_set:
        raise FuturePhaseCurveError(
            "phase evaluation does not match the model-eligible census"
        )
    model_ids = model_frame["game_id"].astype(str)
    reference_id_set = set(model_ids)
    reference_ids = _game_series(reference, "phase series reference frame").astype(str)
    if reference_ids.duplicated().any() or reference_id_set != set(reference_ids):
        raise FuturePhaseCurveError(
            "phase series reference changed game IDs"
        )
    if not phase_id_set.issubset(reference_id_set):
        raise FuturePhaseCurveError("phase series crosswalk changed phase game IDs")
    series_by_game = pd.Series(
        model_frame["series_id"].astype(str).to_numpy(), index=model_ids
    )
    eligible_series = series_by_game.reindex(list(eligible_ids))
    assignment_frame = model_frame
    if eligible_series.isna().any() and require_full_eligible:
        raise FuturePhaseCurveError(
            "phase series reference is missing model-eligible assignments"
        )
    if eligible_series.isna().any():
        # A fold-local training fit may contain only a subset of the eligible
        # census. The complete evaluation and the final fit use the full
        # reference frame above. Keep this internal subset provenance explicit.
        assignment_frame = model_frame.loc[model_ids.isin(phase_id_set)].copy()
    else:
        assignment_frame = pd.DataFrame(
            {
                "game_id": list(eligible_series.index),
                "series_id": eligible_series.to_numpy(),
            }
        )
    eligible_assignment_sha256 = phase_series_assignment_sha256(
        assignment_frame,
        game_column="game_id",
    )
    expected_assignment = (
        str(expected_assignment_sha256).lower()
        if expected_assignment_sha256 is not None
        else None
    )
    if _verified_reference is not None:
        cached_assignment = _verified_reference.eligible_assignment_sha256
        if expected_assignment is None:
            expected_assignment = cached_assignment
        elif expected_assignment != cached_assignment:
            raise FuturePhaseCurveError(
                "pre-bound phase series assignment digest changed"
            )
    if expected_assignment is not None and not re.fullmatch(
        r"[0-9a-f]{64}", expected_assignment
    ):
        raise FuturePhaseCurveError("expected phase series assignment hash is invalid")
    assignment_matches_reference = bool(
        expected_assignment is not None
        and eligible_assignment_sha256 == expected_assignment
    )
    source_by_game = series_by_game.map(
        lambda value: (
            "leaguepedia_crosswalk"
            if str(value).startswith("leaguepedia:")
            else "game_fallback"
            if str(value).startswith("game-fallback:")
            else "team_tournament_proxy"
        )
    )
    result = frame.copy()
    result["series_id"] = phase_ids.map(series_by_game).to_numpy()
    result["series_id_source"] = phase_ids.map(source_by_game).to_numpy()
    base_audit = dict(model_frame.attrs.get("series_cluster_audit") or {})
    verified_binding = model_frame.attrs.get(
        "verified_leaguepedia_series_crosswalk"
    )
    if isinstance(verified_binding, Mapping):
        mapped_ids = {
            str(value) for value in verified_binding.get("mapped_game_ids", ())
        }
    elif "_series_crosswalk_mapped" in model_frame.columns:
        mapped_mask = model_frame["_series_crosswalk_mapped"]
        if not pd.api.types.is_bool_dtype(mapped_mask.dtype) or mapped_mask.isna().any():
            raise FuturePhaseCurveError(
                "verified phase crosswalk mapped flags are invalid"
            )
        mapped_ids = set(
            model_frame.loc[mapped_mask, "game_id"].astype(str)
        )
    else:
        raise FuturePhaseCurveError("verified phase series crosswalk binding is missing")
    mapped_ids &= eligible_set
    # Recompute the audit on the exact phase rows.  The shared loader's audit
    # is scoped to its input frame, which is a fold subset during fitting.
    partition = {
        "source": MIXED_SERIES_PARTITION_SOURCE,
        "key_fields": list(
            base_audit.get("key_fields") or MIXED_SERIES_PARTITION_KEY_FIELDS
        ),
        "mapping_sha256": str(base_audit.get("crosswalk_assignment_sha256") or ""),
        "crosswalk_sha256": str(base_audit.get("crosswalk_sha256") or ""),
        "artifact_sha256": str(base_audit.get("crosswalk_artifact_sha256") or ""),
        "receipt_sha256": str(base_audit.get("crosswalk_receipt_sha256") or ""),
        "receipt_file_sha256": str(crosswalk_receipt_file_sha256).lower(),
        "eligible_game_count": len(eligible_ids),
        "eligible_identity_sha256": expected_eligible_identity,
        "eligible_game_ids": list(eligible_ids),
        "eligible_assignment_sha256": eligible_assignment_sha256,
        "reference_game_count": len(reference_id_set),
        "reference_assignment_sha256": expected_assignment,
        "reference_assignment_match": assignment_matches_reference,
        "authoritative": False,
        "proxy_authority_blocker": bool(
            len(mapped_ids & eligible_set) < len(eligible_set)
            or base_audit.get("partial_series_blocker")
        ),
        "audit": base_audit,
    }
    if not partition["mapping_sha256"] or not partition["crosswalk_sha256"]:
        raise FuturePhaseCurveError("verified phase series mapping hash is missing")
    if expected_assignment is None:
        partition["cross_model_partition_status"] = "non_comparable"
        partition["cross_model_partition_reason"] = (
            "independent reference eligible assignment digest is required"
        )
    elif not assignment_matches_reference:
        raise FuturePhaseCurveError(
            "phase series eligible assignments differ from the reference partition"
        )
    else:
        partition["cross_model_partition_status"] = "comparable"
        partition["cross_model_partition_reason"] = (
            "phase and rating use the same full-reference eligible game-to-series assignments"
        )
    result.attrs["series_partition"] = partition
    result.attrs["series_partition_source"] = MIXED_SERIES_PARTITION_SOURCE
    result.attrs["series_partition_key_fields"] = tuple(partition["key_fields"])
    result.attrs["series_partition_authoritative"] = False
    result.attrs["series_partition_proxy_authority_blocker"] = partition[
        "proxy_authority_blocker"
    ]
    return result, partition


def _normalised_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().casefold())


def _is_checkpoint_name(value: Any) -> bool:
    name = _normalised_name(value)
    return bool(
        re.search(
            r"(?:gold|xp|cs|kills|assists|deaths)(?:at|diffat|differenceat)(?:10|15|20|25)$",
            name,
        )
        or re.search(r"(?:gold|xp|cs|kills|assists|deaths)(?:10|15|20|25)$", name)
    )


def _is_forbidden_pregame_name(value: Any) -> bool:
    name = _normalised_name(value)
    if _is_checkpoint_name(value):
        return True
    # A historical value is safe only when its name declares the lagged
    # boundary.  Bare final-map fields can otherwise enter a pregame vector.
    history_prefix = (
        "prior_",
        "history_",
        "rolling_",
        "lag_",
        "form_",
        "rating_",
        "atom_",
        "continuity_",
        "roster_",
    )
    if name in {_normalised_name(alias) for alias in FINAL_METRIC_ALIASES} and not name.startswith(
        history_prefix
    ):
        return True
    forbidden = (
        "target",
        "observed",
        "current",
        "result",
        "winner",
        "bluewin",
        "finalresult",
        "gamelength",
        "duration",
        "gameclock",
        "objectives",
        "firstblood",
        "firstdragon",
        "firsttower",
        "baron",
        "inhibitor",
        "gold30",
        "xp30",
    )
    return any(token in name for token in forbidden)


def assert_pregame_feature_names(feature_names: Iterable[str]) -> None:
    """Reject current checkpoint, final-outcome, and censoring fields."""

    forbidden = sorted({str(name) for name in feature_names if _is_forbidden_pregame_name(name)})
    if forbidden:
        raise FuturePhaseCurveError(
            "pregame phase features contain current-map or final-state fields: "
            + ", ".join(forbidden)
        )


def _side(value: Any) -> str | None:
    name = str(value or "").strip().casefold()
    return {"blue": "blue", "b": "blue", "red": "red", "r": "red"}.get(name)


def _target_value(row: Mapping[str, Any], kind: str, phase: int) -> float | None:
    names = (
        f"{kind}diffat{phase}",
        f"{kind}_diff_{phase}",
        f"{kind}_diffat{phase}",
    )
    for name in names:
        value = row.get(name)
        if value is not None and pd.notna(value):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return None


def _state_value(row: Mapping[str, Any], kind: str, phase: int) -> float | None:
    names = (f"{kind}at{phase}", f"{kind}_at_{phase}", f"{kind}At{phase}")
    for name in names:
        value = row.get(name)
        if value is not None and pd.notna(value):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                return number
    return None


def _duration_seconds(row: Mapping[str, Any]) -> float | None:
    for name in ("gamelength", "game_length", "duration_seconds", "duration"):
        value = row.get(name)
        if value is None or pd.isna(value):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            # OE gamelength is in seconds. A small duration value is treated
            # as seconds because phase rows use minute thresholds.
            return number
    return None


def prepare_phase_frame(
    maps: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    pregame_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one blue-oriented phase target row per accepted map.

    Team checkpoint fields become targets only.  Current checkpoint values in
    either input frame never become feature columns.  A short game produces a
    censored missing target when duration proves that the checkpoint was not
    reached.
    """

    maps_value = maps.copy()
    teams_value = teams.copy()
    maps_value["_game_id"] = _game_series(maps_value, "maps")
    teams_value["_game_id"] = _game_series(teams_value, "teams")
    maps_value["_date"] = _date_series(maps_value, "maps")
    teams_value["_date"] = _date_series(teams_value, "teams")
    if maps_value["_game_id"].duplicated().any():
        raise FuturePhaseCurveError("maps must contain one row per game")
    side_column = next(
        (name for name in ("side", "teamcolor", "team_color") if name in teams_value.columns),
        None,
    )
    if side_column is None:
        raise FuturePhaseCurveError("teams has no side column")
    teams_value["_side"] = teams_value[side_column].map(_side)
    if teams_value["_side"].isna().any():
        raise FuturePhaseCurveError("teams contains an unknown side")
    counts = teams_value.groupby("_game_id", sort=False, observed=True)["_side"].agg(
        rows="size", sides="nunique"
    )
    invalid = counts[(counts["rows"] != 2) | (counts["sides"] != 2)]
    if not invalid.empty:
        raise FuturePhaseCurveError(
            f"game {str(invalid.index[0])} does not have two team rows"
        )
    blue = teams_value.loc[teams_value["_side"].eq("blue")].set_index("_game_id", drop=False)
    red = teams_value.loc[teams_value["_side"].eq("red")].set_index("_game_id", drop=False)
    map_index = maps_value.set_index("_game_id", drop=False)
    if not set(map_index.index).issubset(blue.index) or not set(map_index.index).issubset(red.index):
        missing = sorted(set(map_index.index) - set(blue.index) - set(red.index))
        raise FuturePhaseCurveError(f"phase source is missing team rows: {missing[:3]}")

    def numeric_coalesce(source: pd.DataFrame, names: Sequence[str]) -> pd.Series:
        result = pd.Series(np.nan, index=source.index, dtype=float)
        for name in names:
            if name in source.columns:
                result = result.fillna(pd.to_numeric(source[name], errors="coerce"))
        return result

    team_id_column = next(
        (name for name in ("teamid", "team_id") if name in teams_value.columns),
        None,
    )
    result = pd.DataFrame(index=map_index.index)
    result["game_uid"] = map_index["_game_id"].astype(str)
    result["date"] = map_index["_date"]
    for output, names in (
        ("league", ("league",)),
        ("region", ("region", "league_source")),
        ("patch", ("patch", "oe_patch_token")),
        ("series_id", ("series_id", "seriesid")),
        ("tournament", ("tournament",)),
        ("blue_team_key", ("blue_team_key",)),
        ("red_team_key", ("red_team_key",)),
        ("blue_team", ("blue_team",)),
        ("red_team", ("red_team",)),
    ):
        source = next((name for name in names if name in map_index.columns), None)
        if source is not None:
            result[output] = map_index[source]
        else:
            fallback = next((name for name in names if name in blue.columns), None)
            result[output] = blue.reindex(map_index.index)[fallback] if fallback else pd.NA
    for output, source in (
        ("blue_team_id", "blue"),
        ("red_team_id", "red"),
    ):
        if team_id_column is None:
            result[output] = pd.NA
        else:
            side_frame = blue if source == "blue" else red
            result[output] = side_frame.reindex(map_index.index)[team_id_column]
    cluster_metadata = pd.DataFrame(index=map_index.index)
    cluster_metadata["game_uid"] = result["game_uid"]
    cluster_metadata["date"] = result["date"]
    cluster_metadata["series_id"] = result["series_id"]
    for name in ("league", "league_source", "tournament", "blue_team_key", "red_team_key"):
        if name in map_index.columns:
            cluster_metadata[name] = map_index[name]
        else:
            cluster_metadata[name] = pd.NA
    if team_id_column is not None:
        cluster_metadata["blue_team_id"] = blue.reindex(map_index.index)[team_id_column]
        cluster_metadata["red_team_id"] = red.reindex(map_index.index)[team_id_column]
    else:
        cluster_metadata["blue_team_id"] = pd.NA
        cluster_metadata["red_team_id"] = pd.NA
    series_labels, series_sources = _series_cluster_labels(cluster_metadata)
    result["series_id"] = series_labels
    result["series_id_source"] = series_sources

    duration = numeric_coalesce(map_index, ("gamelength", "game_length", "duration_seconds", "duration"))
    duration = duration.combine_first(
        numeric_coalesce(blue.reindex(map_index.index), ("gamelength", "game_length", "duration_seconds", "duration"))
    )
    duration = duration.combine_first(
        numeric_coalesce(red.reindex(map_index.index), ("gamelength", "game_length", "duration_seconds", "duration"))
    )
    result["duration_seconds"] = duration

    if pregame_features is not None:
        feature_value = pregame_features.copy()
        feature_value["_game_id"] = _game_series(feature_value, "pregame features")
        if feature_value["_game_id"].duplicated().any():
            raise FuturePhaseCurveError("pregame features must contain one row per game")
        feature_names = [
            str(name)
            for name in feature_value.columns
            if name not in {"_game_id", "game_uid", "gameid", "game_id"}
        ]
        assert_pregame_feature_names(feature_names)
        feature_value = feature_value.set_index("_game_id").drop(
            columns=[name for name in ("game_uid", "gameid", "game_id") if name in feature_value],
            errors="ignore",
        )
        result = result.join(feature_value, how="left")

    for phase in PHASES:
        for kind in ("gold", "xp"):
            direct = numeric_coalesce(
                blue.reindex(map_index.index),
                (f"{kind}diffat{phase}", f"{kind}_diff_{phase}", f"{kind}_diffat{phase}"),
            )
            left = numeric_coalesce(
                blue.reindex(map_index.index),
                (f"{kind}at{phase}", f"{kind}_at_{phase}", f"{kind}At{phase}"),
            )
            right = numeric_coalesce(
                red.reindex(map_index.index),
                (f"{kind}at{phase}", f"{kind}_at_{phase}", f"{kind}At{phase}"),
            )
            target = direct.combine_first(left - right)
            censored = duration.notna() & duration.lt(float(phase * 60))
            target = target.mask(censored)
            target_name = f"{kind}_diff_{phase}"
            result[target_name] = target
            result[f"{target_name}_missing"] = target.isna()
            result[f"{target_name}_censored"] = censored.astype(bool)
    if result.empty:
        raise FuturePhaseCurveError("phase source has no maps")
    return result.reset_index(drop=True)


def _infer_final_metrics(frame: pd.DataFrame) -> tuple[str, ...]:
    available = {_normalised_name(name): str(name) for name in frame.columns}
    result: list[str] = []
    for alias in FINAL_METRIC_ALIASES:
        name = available.get(_normalised_name(alias))
        if name and name not in result and not _is_checkpoint_name(name):
            result.append(name)
    return tuple(result)


def strict_prior_final_history(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    date_column: str,
    metric_columns: Sequence[str] | None = None,
    output_prefix: str = "prior_form_",
) -> pd.DataFrame:
    """Return final-map metrics from earlier timestamp blocks only.

    Every row at one entity and timestamp receives the same history.  The
    current row and all same-timestamp rows stay out of the history.  This
    preserves batch independence and blocks current-map checkpoint leakage.
    """

    required = {entity_column, date_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise FuturePhaseCurveError("prior history input is missing: " + ", ".join(missing))
    metrics = tuple(metric_columns or _infer_final_metrics(frame))
    if not metrics:
        raise FuturePhaseCurveError("prior history has no approved final metrics")
    forbidden_history = sorted(
        name
        for name in metrics
        if _is_checkpoint_name(name)
        or (
            _is_forbidden_pregame_name(name)
            and _normalised_name(name) in {"result", "winner", "ybluewin"}
        )
    )
    if forbidden_history:
        raise FuturePhaseCurveError(
            "prior history contains current-map fields: " + ", ".join(forbidden_history)
        )
    work = frame[[entity_column, date_column, *metrics]].copy()
    work[date_column] = pd.to_datetime(work[date_column], utc=True, errors="coerce")
    if work[entity_column].isna().any() or work[date_column].isna().any():
        raise FuturePhaseCurveError("prior history identity or date is missing")
    key_frame = work[[entity_column, date_column]].copy()
    output = key_frame.copy()
    for metric in metrics:
        values = pd.to_numeric(work[metric], errors="coerce")
        values = values.where(np.isfinite(values))
        metric_frame = key_frame.copy()
        metric_frame["_value"] = values.to_numpy(dtype=float)
        aggregate = (
            metric_frame.groupby([entity_column, date_column], sort=False, observed=True)["_value"]
            .agg(["sum", "count"])
            .reset_index()
            .sort_values([entity_column, date_column], kind="stable")
        )
        aggregate["_prior_sum"] = (
            aggregate.groupby(entity_column, sort=False, observed=True)["sum"].cumsum()
            - aggregate["sum"]
        )
        aggregate["_prior_count"] = (
            aggregate.groupby(entity_column, sort=False, observed=True)["count"].cumsum()
            - aggregate["count"]
        )
        aggregate[f"{output_prefix}{metric}"] = aggregate["_prior_sum"] / aggregate[
            "_prior_count"
        ].where(aggregate["_prior_count"] > 0)
        aggregate[f"{output_prefix}{metric}_support"] = aggregate["_prior_count"].astype(int)
        output = output.merge(
            aggregate[
                [
                    entity_column,
                    date_column,
                    f"{output_prefix}{metric}",
                    f"{output_prefix}{metric}_support",
                ]
            ],
            on=[entity_column, date_column],
            how="left",
            validate="many_to_one",
        )
    if len(output) != len(frame):
        raise FuturePhaseCurveError("prior history changed row grain")
    return output


def build_strict_prior_team_features(
    teams: pd.DataFrame,
    *,
    metric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build side-neutral team final-form differences for each map."""

    work = teams.copy()
    work["_game_id"] = _game_series(work, "teams")
    work["_date"] = _date_series(work, "teams")
    identity = next((name for name in ("teamid", "team_id") if name in work.columns), None)
    if identity is None:
        raise FuturePhaseCurveError("team final history has no stable team identity")
    side_column = next((name for name in ("side", "teamcolor", "team_color") if name in work.columns), None)
    if side_column is None:
        raise FuturePhaseCurveError("team final history has no side column")
    work["_side"] = work[side_column].map(_side)
    if work["_side"].isna().any():
        raise FuturePhaseCurveError("team final history has an unknown side")
    history = strict_prior_final_history(
        work,
        entity_column=identity,
        date_column="_date",
        metric_columns=metric_columns,
    )
    history_columns = [
        name
        for name in history.columns
        if name.startswith("prior_form_")
    ]
    history = pd.concat(
        [work[["_game_id", "_date", "_side"]].reset_index(drop=True), history[history_columns].reset_index(drop=True)],
        axis=1,
    )
    rows: list[dict[str, Any]] = []
    for game_id, group in history.groupby("_game_id", sort=False):
        if set(group["_side"]) != {"blue", "red"}:
            raise FuturePhaseCurveError(f"team history game {game_id} has invalid sides")
        blue = group.loc[group["_side"].eq("blue")].iloc[0]
        red = group.loc[group["_side"].eq("red")].iloc[0]
        row: dict[str, Any] = {"game_uid": str(game_id), "date": blue["_date"]}
        for name in history_columns:
            if name.endswith("_support"):
                row[f"{name}_min"] = int(min(int(blue[name]), int(red[name])))
                continue
            left = blue[name]
            right = red[name]
            row[f"{name}_diff"] = (
                float(left) - float(right)
                if pd.notna(left) and pd.notna(right)
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def bind_phase_source(
    frame: pd.DataFrame,
    source_receipt: Mapping[str, Any],
    *,
    allow_subset: bool = False,
    source_receipt_path: Path | str | None = None,
    source_receipt_file_sha256: str | None = None,
) -> BoundPhaseSource:
    """Bind one phase frame to the accepted census and source cutoff."""

    verified_receipt = _validate_source_receipt(
        source_receipt,
        source_receipt_path=source_receipt_path,
        source_receipt_file_sha256=source_receipt_file_sha256,
    )
    accepted = tuple(verified_receipt["accepted_game_ids"])
    expected_hash = str(verified_receipt["source_identity_sha256"]).lower()
    cutoff = _as_timestamp(verified_receipt["source_as_of"], "source_as_of")
    value = frame.copy()
    value["_game_id"] = _game_series(value, "phase frame")
    value["_date"] = _date_series(value, "phase frame")
    accepted_set = set(accepted)
    available = set(value["_game_id"])
    extra_ids = sorted(available - accepted_set)
    if extra_ids:
        raise FuturePhaseCurveError(
            "phase frame contains game IDs outside the accepted census; "
            f"extra_games={len(extra_ids)} sample={extra_ids[:5]}"
        )
    missing_ids = sorted(accepted_set - available)
    if missing_ids and not allow_subset:
        raise FuturePhaseCurveError(f"phase frame is missing {len(missing_ids)} accepted games")
    if value["_game_id"].duplicated().any():
        raise FuturePhaseCurveError("phase frame must contain one row per accepted game")
    selected = value.loc[value["_game_id"].isin(accepted_set)].copy()
    if not allow_subset and len(selected) != len(accepted):
        raise FuturePhaseCurveError("phase frame grain does not match accepted census")
    if selected["_date"].gt(cutoff).any():
        raise FuturePhaseCurveError("phase frame contains rows after source_as_of")
    selected = selected.drop(columns=["_game_id", "_date"])
    receipt = dict(verified_receipt)
    receipt["source_identity_sha256"] = expected_hash
    receipt["source_game_count"] = len(accepted)
    return BoundPhaseSource(frame=selected, receipt=receipt)


def verify_accepted_census_artifact(
    reference: Mapping[str, Any],
    *,
    runtime_root: Path | str = Path("."),
    expected_source_game_count: int | None = None,
    expected_source_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a hash-bound accepted-game census referenced by an artifact."""

    locator = str(reference.get("locator") or "").strip()
    if not locator:
        raise FuturePhaseCurveError("accepted census artifact has no locator")
    path = Path(locator)
    if not path.is_absolute():
        path = Path(runtime_root) / path
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise FuturePhaseCurveError("accepted census artifact is unavailable") from error
    try:
        expected_bytes = int(reference["bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise FuturePhaseCurveError("accepted census artifact byte count is invalid") from error
    expected_sha = str(reference.get("sha256") or "").lower()
    if len(expected_sha) != 64 or any(value not in "0123456789abcdef" for value in expected_sha):
        raise FuturePhaseCurveError("accepted census artifact hash is invalid")
    if len(raw) != expected_bytes or _sha256_bytes(raw) != expected_sha:
        raise FuturePhaseCurveError("accepted census artifact bytes or hash do not match")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FuturePhaseCurveError("accepted census artifact is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise FuturePhaseCurveError("accepted census artifact is not an object")
    schema_version = str(reference.get("schema_version") or "")
    if schema_version and payload.get("schema_version") != schema_version:
        raise FuturePhaseCurveError("accepted census artifact schema does not match")
    raw_ids = payload.get(str(reference.get("game_ids_field") or "game_ids"))
    if not isinstance(raw_ids, list):
        raise FuturePhaseCurveError("accepted census artifact has no game ID list")
    accepted = tuple(str(value) for value in raw_ids)
    canonical = canonical_game_ids(accepted)
    if tuple(accepted) != canonical:
        raise FuturePhaseCurveError("accepted census artifact game IDs are not canonical")
    game_count = len(accepted)
    identity = identity_sha256(accepted)
    if payload.get("game_count") != game_count:
        raise FuturePhaseCurveError("accepted census artifact game count is invalid")
    if str(payload.get("source_identity_sha256") or "").lower() != identity:
        raise FuturePhaseCurveError("accepted census artifact identity hash is invalid")
    if expected_source_game_count is not None and game_count != int(expected_source_game_count):
        raise FuturePhaseCurveError("accepted census artifact count differs from source contract")
    if (
        expected_source_identity_sha256 is not None
        and identity != str(expected_source_identity_sha256).lower()
    ):
        raise FuturePhaseCurveError("accepted census artifact identity differs from source contract")
    return {
        "status": "verified",
        "locator": locator,
        "bytes": expected_bytes,
        "sha256": expected_sha,
        "schema_version": schema_version or payload.get("schema_version"),
        "game_count": game_count,
        "source_identity_sha256": identity,
    }


def _default_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    excluded = {
        "game_uid",
        "gameid",
        "game_id",
        "date",
        "played_at",
        "game_date",
        "start_time",
        "league",
        "region",
        "patch",
        "oe_patch_token",
        "public_patch",
        "series_id",
    }
    candidates: list[str] = []
    for name in frame.columns:
        text = str(name)
        if text in excluded or text.endswith("_missing") or text.endswith("_censored"):
            continue
        if text.startswith(("prior_", "form_", "rating_", "atom_", "continuity_", "roster_")):
            candidates.append(text)
    assert_pregame_feature_names(candidates)
    return tuple(candidates)


def _design(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    names: list[str] = []
    columns: list[np.ndarray] = []
    for name in feature_columns:
        if name not in frame.columns:
            raise FuturePhaseCurveError(f"phase feature is missing: {name}")
        assert_pregame_feature_names([name])
        values = pd.to_numeric(frame[name], errors="coerce")
        raw = values.to_numpy(dtype=float)
        missing = (~np.isfinite(raw)).astype(float)
        numeric = np.where(np.isfinite(raw), raw, 0.0)
        # OE metrics have different native units.  Fixed source units keep
        # Ridge conditioning stable and make train/test scoring reproducible.
        normalized_name = _normalised_name(name)
        if "gold" in normalized_name or "xp" in normalized_name:
            numeric = numeric / 10000.0
        elif "dpm" in normalized_name or "damage" in normalized_name:
            numeric = numeric / 1000.0
        elif "vision" in normalized_name or "ward" in normalized_name:
            numeric = numeric / 100.0
        elif (
            "cs" in normalized_name
            or "kill" in normalized_name
            or "assist" in normalized_name
        ):
            numeric = numeric / 100.0
        names.append(str(name))
        columns.append(numeric)
        names.append(f"{name}__missing")
        columns.append(missing)
    if not columns:
        raise FuturePhaseCurveError("phase model has no pregame features")
    return np.column_stack(columns), tuple(names)


def _target(frame: pd.DataFrame, kind: str, phase: int) -> np.ndarray:
    name = f"{kind}_diff_{phase}"
    if name not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(values), values, np.nan)


def _target_coverage(frame: pd.DataFrame, kind: str, phase: int) -> dict[str, Any]:
    """Describe target support while keeping short maps out of the denominator.

    A target is at risk when its duration reaches the checkpoint.  Unknown
    duration stays in the source denominator because an observed checkpoint is
    still usable.  A missing value at an at-risk row is ordinary source
    missingness, not censoring.
    """

    target = _target(frame, kind, phase)
    target_name = f"{kind}_diff_{phase}"
    observed = np.isfinite(target)
    censored = (
        frame[f"{target_name}_censored"].fillna(False).astype(bool).to_numpy()
        if f"{target_name}_censored" in frame
        else np.zeros(len(frame), dtype=bool)
    )
    missing = ~observed
    duration_known = None
    if "duration_seconds" in frame:
        duration_known = pd.to_numeric(frame["duration_seconds"], errors="coerce").to_numpy(
            dtype=float
        )
    elif "gamelength" in frame:
        duration_known = pd.to_numeric(frame["gamelength"], errors="coerce").to_numpy(dtype=float)
    if duration_known is None:
        at_risk = ~censored
    else:
        at_risk = (~np.isfinite(duration_known)) | (duration_known >= float(phase * 60))
    return {
        "rows": int(len(frame)),
        "at_risk_rows": int(at_risk.sum()),
        "observed_rows": int(observed.sum()),
        "coverage": float(observed.mean()) if len(frame) else 0.0,
        "at_risk_coverage": float(observed[at_risk].mean()) if at_risk.any() else None,
        "missing_rows": int(missing.sum()),
        "censored_rows": int(censored.sum()),
        "uncensored_missing_rows": int((missing & ~censored).sum()),
    }


def phase_curve_measures(
    gold_values: Sequence[float | None],
    xp_values: Sequence[float | None],
) -> dict[str, float | None]:
    """Derive signed curve measures from four checkpoint predictions."""

    if len(gold_values) != len(PHASES) or len(xp_values) != len(PHASES):
        raise FuturePhaseCurveError("phase measures need all four checkpoints")
    gold = [float(value) if value is not None and math.isfinite(float(value)) else None for value in gold_values]
    xp = [float(value) if value is not None and math.isfinite(float(value)) else None for value in xp_values]
    scaling = (
        (xp[3] - xp[2]) - (xp[1] - xp[0])
        if all(value is not None for value in xp)
        else None
    )
    snowball = (
        (gold[1] - gold[0]) - (gold[3] - gold[2])
        if all(value is not None for value in gold)
        else None
    )
    return {
        "scaling_index": float(scaling) if scaling is not None else None,
        "snowball_index": float(snowball) if snowball is not None else None,
    }


def _phase_shape_values(
    values: Sequence[float | None] | Mapping[int | str, float | None],
    *,
    name: str,
) -> list[float | None]:
    """Normalize one four-checkpoint curve without filling missing values."""

    if isinstance(values, Mapping):
        normalized: list[float | None] = []
        for phase in PHASES:
            raw = values.get(phase, values.get(str(phase)))
            if raw is None:
                normalized.append(None)
                continue
            try:
                number = float(raw)
            except (TypeError, ValueError) as error:
                raise FuturePhaseCurveError(f"{name} contains a non-numeric value") from error
            normalized.append(number if math.isfinite(number) else None)
        return normalized
    try:
        raw_values = list(values)
    except TypeError as error:
        raise FuturePhaseCurveError(f"{name} must contain four checkpoints") from error
    if len(raw_values) != len(PHASES):
        raise FuturePhaseCurveError(f"{name} must contain four checkpoints")
    normalized = []
    for raw in raw_values:
        if raw is None:
            normalized.append(None)
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError) as error:
            raise FuturePhaseCurveError(f"{name} contains a non-numeric value") from error
        normalized.append(number if math.isfinite(number) else None)
    return normalized


def _phase_shape_availability(
    available: bool | Mapping[str, bool] | Sequence[bool] | None,
    *,
    gold_available: bool | None,
    xp_available: bool | None,
    curve_available: bool | None,
    complete: bool,
) -> bool:
    """Resolve explicit availability while keeping incomplete curves closed."""

    explicit: bool | None = None
    if available is not None:
        if isinstance(available, Mapping):
            values = [
                available.get("gold", available.get("gold_available", True)),
                available.get("xp", available.get("xp_available", True)),
                available.get("curve", available.get("curve_available", True)),
            ]
            if not all(isinstance(value, (bool, np.bool_)) for value in values):
                raise FuturePhaseCurveError("phase shape availability must be boolean")
            explicit = all(bool(value) for value in values)
        elif isinstance(available, (bool, np.bool_)):
            explicit = bool(available)
        else:
            try:
                values = list(available)
            except TypeError as error:
                raise FuturePhaseCurveError("phase shape availability must be boolean") from error
            if len(values) != 2 or not all(isinstance(value, (bool, np.bool_)) for value in values):
                raise FuturePhaseCurveError(
                    "phase shape availability must be one boolean or two booleans"
                )
            explicit = all(bool(value) for value in values)
    for value, label in (
        (gold_available, "gold_available"),
        (xp_available, "xp_available"),
        (curve_available, "curve_available"),
    ):
        if value is not None and not isinstance(value, (bool, np.bool_)):
            raise FuturePhaseCurveError(f"{label} must be boolean")
        if value is not None:
            explicit = bool(value) if explicit is None else explicit and bool(value)
    return complete if explicit is None else bool(explicit) and complete


def _phase_shape_slopes(values: Sequence[float]) -> list[float]:
    return [
        (float(second) - float(first)) / float(second_phase - first_phase)
        for first, second, first_phase, second_phase in zip(
            values,
            values[1:],
            PHASES,
            PHASES[1:],
        )
    ]


def _phase_shape_signed_area(values: Sequence[float]) -> float:
    return float(
        sum(
            (float(first) + float(second)) * float(second_phase - first_phase) / 2.0
            for first, second, first_phase, second_phase in zip(
                values,
                values[1:],
                PHASES,
                PHASES[1:],
            )
        )
    )


def _phase_shape_material_minute(values: Sequence[float]) -> float:
    """Return the first threshold time, signed by the advantaged side."""

    threshold = float(PHASE_SHAPE_MATERIAL_THRESHOLD)
    candidates: list[tuple[float, float]] = []
    for index, value in enumerate(values):
        minute = float(PHASES[index])
        if value >= threshold:
            candidates.append((minute, minute))
        if value <= -threshold:
            candidates.append((minute, -minute))
    for index, (first, second) in enumerate(zip(values, values[1:])):
        first_minute = float(PHASES[index])
        span = float(PHASES[index + 1] - PHASES[index])
        if first < threshold <= second and second != first:
            fraction = (threshold - first) / (second - first)
            candidates.append((first_minute + span * fraction, 1.0))
        if first > -threshold >= second and second != first:
            fraction = (-threshold - first) / (second - first)
            candidates.append((first_minute + span * fraction, -1.0))
    if not candidates:
        # Zero is registered as the no-threshold-crossing value.  It is
        # distinct from unavailable data, which remains None.
        return 0.0
    first_minute, sign = min(candidates, key=lambda value: value[0])
    return float(math.copysign(first_minute, sign))


def _phase_shape_crossovers(values: Sequence[float]) -> tuple[float | None, float]:
    """Count sign changes after removing zeros and interpolate their times."""

    nonzero = [
        (float(PHASES[index]), float(value))
        for index, value in enumerate(values)
        if value != 0.0
    ]
    crossings: list[float] = []
    for (first_minute, first), (second_minute, second) in zip(nonzero, nonzero[1:]):
        if (first < 0.0 and second > 0.0) or (first > 0.0 and second < 0.0):
            if second == first:
                crossing = second_minute
            else:
                crossing = first_minute + (0.0 - first) * (second_minute - first_minute) / (
                    second - first
                )
            crossings.append(float(crossing))
    return (
        float(crossings[0]) if crossings else None,
        float(len(crossings)),
    )


def phase_shape_features(
    gold_values: Sequence[float | None] | Mapping[int | str, float | None],
    xp_values: Sequence[float | None] | Mapping[int | str, float | None],
    available: bool | Mapping[str, bool] | Sequence[bool] | None = None,
    *,
    gold_available: bool | None = None,
    xp_available: bool | None = None,
    curve_available: bool | None = None,
) -> dict[str, float | None]:
    """Return deterministic signed and side-invariant phase-shape features.

    Both signed curves must contain all four finite checkpoints.  Missing or
    partial input yields only the availability flags and ``None`` shape
    values.  An explicit availability value can disable a complete curve.
    The first material threshold time uses a fixed 250-unit threshold.
    """

    gold = _phase_shape_values(gold_values, name="gold_values")
    xp = _phase_shape_values(xp_values, name="xp_values")
    complete = all(value is not None for value in (*gold, *xp))
    is_available = _phase_shape_availability(
        available,
        gold_available=gold_available,
        xp_available=xp_available,
        curve_available=curve_available,
        complete=complete,
    )
    output: dict[str, float | None] = {
        name: None
        for name in (*PHASE_SHAPE_SIGNED_FEATURES, *PHASE_SHAPE_INVARIANT_FEATURES)
    }
    output.update(
        {
            "forecast_curve_available": 1.0 if is_available else 0.0,
            "forecast_curve_missing": 0.0 if is_available else 1.0,
        }
    )
    if not is_available:
        return output

    for prefix, values in (("gold", gold), ("xp", xp)):
        finite_values = [float(value) for value in values]
        slopes = _phase_shape_slopes(finite_values)
        early_mean = float(np.mean(finite_values[:2]))
        late_mean = float(np.mean(finite_values[2:]))
        late_minus_early = late_mean - early_mean
        early_slope = slopes[0]
        late_slope = slopes[-1]
        first_crossover, crossover_count = _phase_shape_crossovers(finite_values)
        for index, slope in enumerate(slopes):
            first_phase = PHASES[index]
            second_phase = PHASES[index + 1]
            output[f"forecast_{prefix}_slope_{first_phase}_{second_phase}"] = float(slope)
        output.update(
            {
                f"forecast_{prefix}_early_mean": early_mean,
                f"forecast_{prefix}_late_mean": late_mean,
                f"forecast_{prefix}_late_minus_early": late_minus_early,
                f"forecast_{prefix}_late_minus_early_slope": late_minus_early / 10.0,
                f"forecast_{prefix}_late_minus_early_acceleration": late_slope - early_slope,
                f"forecast_{prefix}_signed_area": _phase_shape_signed_area(finite_values),
                f"forecast_{prefix}_first_material_advantage_minute_signed": _phase_shape_material_minute(
                    finite_values
                ),
                f"forecast_{prefix}_first_crossover_minute": first_crossover,
                f"forecast_{prefix}_crossover_count": crossover_count,
            }
        )
    return output


def side_swap_phase_shape_features(
    gold_values: Sequence[float | None] | Mapping[int | str, float | None],
    xp_values: Sequence[float | None] | Mapping[int | str, float | None],
    available: bool | Mapping[str, bool] | Sequence[bool] | None = None,
    *,
    gold_available: bool | None = None,
    xp_available: bool | None = None,
    curve_available: bool | None = None,
) -> dict[str, float | None]:
    """Return phase-shape features after relabeling blue and red."""

    gold_normalized = _phase_shape_values(gold_values, name="gold_values")
    xp_normalized = _phase_shape_values(xp_values, name="xp_values")
    return phase_shape_features(
        [None if value is None else -value for value in gold_normalized],
        [None if value is None else -value for value in xp_normalized],
        available,
        gold_available=gold_available,
        xp_available=xp_available,
        curve_available=curve_available,
    )


def phase_shape_side_swap(
    features: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Swap a previously computed feature mapping across the two sides."""

    output = dict(features)
    for name in PHASE_SHAPE_SIGNED_FEATURES:
        value = output.get(name)
        if value is not None:
            output[name] = -float(value)
    return output


def validate_phase_shape_side_swap(
    original: Mapping[str, float | None],
    swapped: Mapping[str, float | None],
    *,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Validate signed antisymmetry and invariant preservation."""

    signed_errors: dict[str, float | None] = {}
    invariant_errors: dict[str, float | None] = {}
    for name in PHASE_SHAPE_SIGNED_FEATURES:
        left = original.get(name)
        right = swapped.get(name)
        if left is None or right is None:
            signed_errors[name] = None if left is None and right is None else math.inf
        else:
            signed_errors[name] = abs(float(left) + float(right))
    for name in PHASE_SHAPE_INVARIANT_FEATURES:
        left = original.get(name)
        right = swapped.get(name)
        if left is None or right is None:
            invariant_errors[name] = None if left is None and right is None else math.inf
        else:
            try:
                invariant_errors[name] = abs(float(left) - float(right))
            except (TypeError, ValueError):
                invariant_errors[name] = 0.0 if left == right else math.inf
    finite_signed = [value for value in signed_errors.values() if value is not None]
    finite_invariant = [value for value in invariant_errors.values() if value is not None]
    max_signed = max(finite_signed, default=None)
    max_invariant = max(finite_invariant, default=None)
    return {
        "passed": bool(
            max_signed is not None
            and max_invariant is not None
            and max_signed <= tolerance
            and max_invariant <= tolerance
        ),
        "max_signed_error": max_signed,
        "max_invariant_error": max_invariant,
        "signed_errors": signed_errors,
        "invariant_errors": invariant_errors,
        "definition": "signed phase fields negate and crossover fields remain unchanged",
    }


def _fit_one(matrix: np.ndarray, target: np.ndarray, alpha: float) -> dict[str, Any] | None:
    valid = np.isfinite(target)
    if int(valid.sum()) < 2:
        return None
    # Phase targets are blue-minus-red quantities.  A zero intercept keeps the
    # fitted curve antisymmetric under a blue/red relabeling.
    model = Ridge(alpha=float(alpha), fit_intercept=False, solver="lsqr")
    model.fit(matrix[valid], target[valid])
    predicted = _finite_linear_predict(
        matrix[valid],
        np.asarray(model.coef_, dtype=float),
        float(model.intercept_),
    )
    residuals = target[valid] - predicted
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    return {
        "intercept": float(model.intercept_),
        "coefficients": [float(value) for value in model.coef_],
        "train_rows": int(valid.sum()),
        "residual_sd": sigma if math.isfinite(sigma) else None,
        "rmse": float(np.sqrt(np.mean(residuals * residuals))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def _finite_linear_predict(
    matrix: np.ndarray,
    coefficients: np.ndarray,
    intercept: float = 0.0,
) -> np.ndarray:
    """Predict a finite phase linear model without warning-prone matmul."""

    values = np.asarray(matrix, dtype=float)
    weights = np.asarray(coefficients, dtype=float)
    if values.ndim != 2 or weights.ndim != 1 or values.shape[1] != len(weights):
        return np.full(values.shape[0] if values.ndim else 0, np.nan, dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        return np.full(len(values), np.nan, dtype=float)
    try:
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            prediction = np.einsum("ij,j->i", values, weights) + float(intercept)
    except (FloatingPointError, TypeError, ValueError):
        return np.full(len(values), np.nan, dtype=float)
    if not np.isfinite(prediction).all():
        return np.full(len(values), np.nan, dtype=float)
    return prediction


def _observed_comeback(frame: pd.DataFrame) -> dict[str, Any]:
    by_window: dict[str, dict[str, Any]] = {}
    for start, end in ((10, 15), (10, 20), (10, 25), (15, 20), (15, 25)):
        early = _target(frame, "gold", start)
        late = _target(frame, "gold", end)
        valid = np.isfinite(early) & np.isfinite(late) & (early < 0.0)
        key = f"{start}_to_{end}"
        if not valid.any():
            by_window[key] = {"value": None, "support": 0}
            continue
        recovered = (late[valid] > early[valid]).astype(float)
        by_window[key] = {
            "value": float(np.mean(recovered)),
            "support": int(valid.sum()),
        }
    primary = by_window["10_to_25"]
    return {
        "value": primary["value"],
        "support": primary["support"],
        "by_window": by_window,
        "definition": "share of early-behind maps with a smaller late gold deficit",
    }


def fit_phase_curve(
    frame: pd.DataFrame,
    *,
    source_receipt: Mapping[str, Any],
    source_receipt_path: Path | str | None = None,
    source_receipt_file_sha256: str | None = None,
    feature_columns: Sequence[str] | None = None,
    alpha: float = 10.0,
    model_version: str = MODEL_VERSION,
    crosswalk_path: Path | str | None = None,
    crosswalk_receipt_path: Path | str | None = None,
    crosswalk_receipt_file_sha256: str | None = None,
    series_partition_reference_frame: pd.DataFrame | None = None,
    series_partition_assignment_sha256: str | None = None,
    _series_partition_reference: _VerifiedPhaseSeriesReference | None = None,
) -> dict[str, Any]:
    """Fit OE-only gold and XP curves from strictly pregame features.

    The returned artifact stays development-only.  It has no probability or
    public-authority field.  Evaluation folds must call this function on the
    training rows only.
    """

    bound = bind_phase_source(
        frame,
        source_receipt,
        allow_subset=True,
        source_receipt_path=source_receipt_path,
        source_receipt_file_sha256=source_receipt_file_sha256,
    )
    crosswalk_values = (
        crosswalk_path,
        crosswalk_receipt_path,
        crosswalk_receipt_file_sha256,
    )
    if any(value is not None for value in crosswalk_values) and not all(
        value is not None for value in crosswalk_values
    ):
        raise FuturePhaseCurveError("phase series crosswalk inputs must be supplied together")
    value = bound.frame.copy()
    partition: dict[str, Any] | None = None
    if crosswalk_path is not None:
        value, partition = bind_phase_series_partition(
            value,
            bound.receipt,
            crosswalk_path=crosswalk_path,
            crosswalk_receipt_path=crosswalk_receipt_path,
            crosswalk_receipt_file_sha256=str(crosswalk_receipt_file_sha256),
            require_full_eligible=False,
            reference_frame=series_partition_reference_frame,
            _verified_reference=_series_partition_reference,
            expected_assignment_sha256=series_partition_assignment_sha256,
        )
    selected_features = tuple(feature_columns or _default_feature_columns(value))
    assert_pregame_feature_names(selected_features)
    matrix, design_names = _design(value, selected_features)
    series_identity = _series_identity_report(value)
    models: dict[str, dict[str, dict[str, Any] | None]] = {"gold": {}, "xp": {}}
    coverage: dict[str, dict[str, Any]] = {"gold": {}, "xp": {}}
    for kind in ("gold", "xp"):
        for phase in PHASES:
            target = _target(value, kind, phase)
            coverage[kind][str(phase)] = _target_coverage(value, kind, phase)
            models[kind][str(phase)] = _fit_one(matrix, target, alpha)
    observed_gold = [
        float(value[f"gold_diff_{phase}"].mean())
        if f"gold_diff_{phase}" in value and value[f"gold_diff_{phase}"].notna().any()
        else None
        for phase in PHASES
    ]
    observed_xp = [
        float(value[f"xp_diff_{phase}"].mean())
        if f"xp_diff_{phase}" in value and value[f"xp_diff_{phase}"].notna().any()
        else None
        for phase in PHASES
    ]
    observed_measures = phase_curve_measures(observed_gold, observed_xp)
    lineage = _source_lineage(bound.receipt)
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "authority": "development_only",
        "source": SOURCE,
        "source_as_of": bound.receipt["source_as_of"],
        "source_game_count": int(bound.receipt["source_game_count"]),
        "source_identity_sha256": str(bound.receipt["source_identity_sha256"]),
        "accepted_game_ids": list(bound.receipt["accepted_game_ids"]),
        "source_receipt_sha256": str(bound.receipt["receipt_sha256"]),
        "source_transport": lineage["transport"],
        "source_lineage": lineage,
        "evaluation_game_count": int(len(value)),
        "evaluation_identity_sha256": identity_sha256(_game_series(value, "phase frame")),
        "series_identity": series_identity,
        "feature_columns": list(selected_features),
        "design_columns": list(design_names),
        "feature_family": PHASE_FEATURE_FAMILY,
        "feature_declaration": list(PHASE_FEATURE_DECLARATION),
        "models": models,
        "coverage": coverage,
        "support": {
            kind: {
                phase: (models[kind][phase] or {}).get("train_rows", 0)
                for phase in PHASE_KEYS
            }
            for kind in ("gold", "xp")
        },
        "uncertainty": {
            kind: {
                phase: (models[kind][phase] or {}).get("residual_sd")
                for phase in PHASE_KEYS
            }
            for kind in ("gold", "xp")
        },
        "curve_definitions": {
            "scaling_index": "XP slope acceleration: (xp25-xp15) - (xp15-xp10)",
            "snowball_index": "gold slope acceleration: (gold15-gold10) - (gold25-gold20)",
            "comeback_resilience": "descriptive conditional recovery share, not a win probability",
        },
        "observed_curve_measures": observed_measures,
        "comeback_resilience": _observed_comeback(value),
        "leakage_contract": {
            "source": "OE only",
            "current_checkpoint_targets": "target_only",
            "final_metrics": "strict_prior_timestamp_blocks_only",
            "same_timestamp_batch": "excluded from prior history",
            "censoring": "checkpoint not reached is missing target",
            "grid_dependency": False,
        },
        "authority_gates": {
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
            "promotion": False,
        },
    }
    if partition is not None:
        output.update(
            {
                "series_partition_source": partition["source"],
                "series_partition_key_fields": list(partition["key_fields"]),
                "series_partition_mapping_sha256": partition["mapping_sha256"],
                "series_partition_crosswalk_sha256": partition["crosswalk_sha256"],
                "series_partition_artifact_sha256": partition["artifact_sha256"],
                "series_partition_receipt_sha256": partition["receipt_sha256"],
                "series_partition_receipt_file_sha256": partition[
                    "receipt_file_sha256"
                ],
                "series_partition_eligible_game_count": partition[
                    "eligible_game_count"
                ],
                "series_partition_eligible_identity_sha256": partition[
                    "eligible_identity_sha256"
                ],
                "series_partition_eligible_assignment_sha256": partition[
                    "eligible_assignment_sha256"
                ],
                "series_partition_reference_game_count": partition[
                    "reference_game_count"
                ],
                "series_partition_reference_assignment_sha256": partition[
                    "reference_assignment_sha256"
                ],
                "series_partition_eligible_game_ids": list(
                    partition["eligible_game_ids"]
                ),
                "series_partition": partition,
                "cross_model_series_partition": series_identity[
                    "cross_model_partition"
                ],
                "series_partition_proxy_authority_blocker": partition[
                    "proxy_authority_blocker"
                ],
            }
        )
    if source_receipt_path is not None:
        output["source_receipt_artifact"] = _receipt_file_reference(
            source_receipt_path,
            expected_sha256=source_receipt_file_sha256,
        )
    return output


def _predict_one(model: Mapping[str, Any] | None, vector: np.ndarray) -> tuple[float | None, float | None]:
    if not isinstance(model, Mapping):
        return None, None
    coefficients = np.asarray(model.get("coefficients") or [], dtype=float)
    if len(coefficients) != len(vector):
        return None, None
    prediction = _finite_linear_predict(
        np.asarray(vector, dtype=float).reshape(1, -1),
        coefficients,
        float(model.get("intercept") or 0.0),
    )
    value = float(prediction[0]) if len(prediction) and np.isfinite(prediction[0]) else math.nan
    residual_sd = model.get("residual_sd")
    try:
        uncertainty = float(residual_sd) if residual_sd is not None else None
    except (TypeError, ValueError):
        uncertainty = None
    return value if math.isfinite(value) else None, uncertainty


def score_phase_curve(
    artifact: Mapping[str, Any],
    features: Mapping[str, Any],
) -> dict[str, Any]:
    """Score an artifact for research inspection.

    The result remains marked ``development_only``.  It contains no win
    probability and cannot authorize public model output.
    """

    if artifact.get("authority") != "development_only":
        raise FuturePhaseCurveError("phase artifact authority is not development_only")
    feature_columns = tuple(str(name) for name in artifact.get("feature_columns") or ())
    assert_pregame_feature_names(feature_columns)
    vector_values: list[float] = []
    missing_features: list[str] = []
    for name in feature_columns:
        raw = features.get(name)
        try:
            number = float(raw)
        except (TypeError, ValueError):
            number = math.nan
        if not math.isfinite(number):
            missing_features.append(name)
            number = 0.0
        vector_values.extend((number, 1.0 if name in missing_features else 0.0))
    vector = np.asarray(vector_values, dtype=float)
    expected_gold: dict[str, float | None] = {}
    expected_xp: dict[str, float | None] = {}
    uncertainty_gold: dict[str, float | None] = {}
    uncertainty_xp: dict[str, float | None] = {}
    for phase in PHASES:
        value, sigma = _predict_one((artifact.get("models") or {}).get("gold", {}).get(str(phase)), vector)
        expected_gold[str(phase)] = round(value, 4) if value is not None else None
        uncertainty_gold[str(phase)] = round(sigma, 4) if sigma is not None else None
        value, sigma = _predict_one((artifact.get("models") or {}).get("xp", {}).get(str(phase)), vector)
        expected_xp[str(phase)] = round(value, 4) if value is not None else None
        uncertainty_xp[str(phase)] = round(sigma, 4) if sigma is not None else None
    gold_values = [expected_gold[str(phase)] for phase in PHASES]
    xp_values = [expected_xp[str(phase)] for phase in PHASES]
    measures = phase_curve_measures(gold_values, xp_values)
    gold_sigma = [uncertainty_gold[str(phase)] for phase in PHASES]
    xp_sigma = [uncertainty_xp[str(phase)] for phase in PHASES]
    scaling_sigma = (
        float(math.sqrt(sum(float(xp_sigma[index]) ** 2 for index in (0, 1, 2, 3))))
        if all(value is not None for value in xp_sigma)
        else None
    )
    snowball_sigma = (
        float(math.sqrt(sum(float(gold_sigma[index]) ** 2 for index in (0, 1, 2, 3))))
        if all(value is not None for value in gold_sigma)
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": artifact.get("model_version"),
        "authority": "development_only",
        "source": artifact.get("source", SOURCE),
        "expected_gold_curve": expected_gold,
        "expected_xp_curve": expected_xp,
        "uncertainty_gold": uncertainty_gold,
        "uncertainty_xp": uncertainty_xp,
        "support": artifact.get("support", {}),
        "coverage": artifact.get("coverage", {}),
        "scaling_index": (
            round(float(measures["scaling_index"]), 4)
            if measures["scaling_index"] is not None
            else None
        ),
        "snowball_index": (
            round(float(measures["snowball_index"]), 4)
            if measures["snowball_index"] is not None
            else None
        ),
        "uncertainty_scaling_index": round(scaling_sigma, 4) if scaling_sigma is not None else None,
        "uncertainty_snowball_index": round(snowball_sigma, 4) if snowball_sigma is not None else None,
        "comeback_resilience": artifact.get("comeback_resilience"),
        "missing_features": missing_features,
        "authority_gates": artifact.get("authority_gates", {}),
    }


def chronological_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 3,
    min_train_rows: int = 1,
    cluster_column: str | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return chronological, timestamp-blocked train/test indices.

    With ``cluster_column`` set, a cluster appears in one split only.  This
    supports series-cluster-safe evaluation when the source has series IDs.
    """

    if n_splits < 1:
        raise FuturePhaseCurveError("n_splits must be positive")
    dates = _date_series(frame)
    # A series is one evaluation unit.  A series can span several timestamps,
    # so splitting by timestamp would put one series in several test folds.
    if cluster_column and cluster_column in frame.columns:
        cluster_values = frame[cluster_column].astype("string")
    else:
        cluster_values = _game_series(frame, "phase frame")
    fallback_values = _game_series(frame, "phase frame")
    cluster_values = cluster_values.fillna(fallback_values)
    cluster_frame = pd.DataFrame({"cluster": cluster_values, "date": dates})
    cluster_dates = (
        cluster_frame.groupby("cluster", sort=False, observed=True)["date"]
        .agg(first="min", last="max")
        .sort_values(["first", "last"], kind="stable")
    )
    if len(cluster_dates) < 2:
        return ()
    boundaries = np.array_split(cluster_dates.index.to_numpy(dtype=object), n_splits)
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for block in boundaries:
        if len(block) == 0:
            continue
        test_clusters = set(block.tolist())
        test_mask = cluster_values.isin(test_clusters)
        test_start = dates.loc[test_mask].min()
        cluster_last = cluster_values.map(cluster_dates["last"])
        train_mask = (
            dates.lt(test_start)
            & ~cluster_values.isin(test_clusters)
            & cluster_last.lt(test_start)
        )
        train = np.flatnonzero(train_mask.to_numpy())
        test = np.flatnonzero(test_mask.to_numpy())
        if len(train) < min_train_rows or len(test) == 0:
            continue
        output.append((train, test))
    return tuple(output)


def _cluster_boundary_diagnostics(
    frame: pd.DataFrame,
    test_indices: np.ndarray,
    cluster_column: str | None,
) -> dict[str, Any]:
    """Report rows kept out because their cluster continues into the future."""

    dates = _date_series(frame)
    if cluster_column and cluster_column in frame.columns:
        cluster_values = frame[cluster_column].astype("string")
    else:
        cluster_values = _game_series(frame, "phase frame")
    cluster_values = cluster_values.fillna(_game_series(frame, "phase frame"))
    test_mask = pd.Series(False, index=frame.index)
    test_mask.iloc[test_indices] = True
    test_start = dates.loc[test_mask].min()
    cluster_last = pd.DataFrame(
        {"cluster": cluster_values, "date": dates}, index=frame.index
    ).groupby("cluster", sort=False, observed=True)["date"].transform("max")
    boundary_mask = (
        dates.lt(test_start)
        & ~test_mask
        & ~cluster_values.isin(set(cluster_values.loc[test_mask].tolist()))
        & cluster_last.ge(test_start)
    )
    boundary_clusters = sorted(str(value) for value in cluster_values.loc[boundary_mask].unique())
    test_cluster_prior_rows = int(
        (
            dates.lt(test_start)
            & ~test_mask
            & cluster_values.isin(set(cluster_values.loc[test_mask].tolist()))
        ).sum()
    )
    return {
        "test_start": test_start.isoformat() if pd.notna(test_start) else None,
        "test_clusters": int(cluster_values.loc[test_mask].nunique()),
        "boundary_excluded_rows": int(boundary_mask.sum()),
        "boundary_excluded_clusters": len(boundary_clusters),
        "boundary_cluster_ids": boundary_clusters,
        "test_cluster_prior_rows": test_cluster_prior_rows,
        "definition": "rows before test start whose cluster has a row at or after test start",
    }


def _prediction_errors(
    artifact: Mapping[str, Any],
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    matrix, _ = _design(frame, feature_columns)
    missing_count = np.zeros(len(frame), dtype=int)
    for name in feature_columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        missing_count += (~np.isfinite(values.to_numpy(dtype=float))).astype(int)
    errors: dict[str, dict[str, np.ndarray]] = {
        kind: {} for kind in ("gold", "xp")
    }
    for kind in ("gold", "xp"):
        for phase in PHASES:
            model = (artifact.get("models") or {}).get(kind, {}).get(str(phase))
            target = _target(frame, kind, phase)
            if isinstance(model, Mapping):
                coefficients = np.asarray(model.get("coefficients") or [], dtype=float)
                prediction = _finite_linear_predict(
                    matrix,
                    coefficients,
                    float(model.get("intercept") or 0.0),
                )
            else:
                prediction = np.full(len(frame), np.nan)
            errors[kind][str(phase)] = target - prediction
    return errors, missing_count


def side_swap_invariance_report(
    artifact: Mapping[str, Any],
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    """Check that model outputs change sign after a blue/red swap."""

    swapped = side_swap_frame(frame)
    original_matrix, _ = _design(frame, feature_columns)
    swapped_matrix, _ = _design(swapped, feature_columns)
    complete = np.ones(len(frame), dtype=bool)
    for name in feature_columns:
        complete &= np.isfinite(
            pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        )
    report: dict[str, Any] = {}
    for kind in ("gold", "xp"):
        for phase in PHASES:
            model = (artifact.get("models") or {}).get(kind, {}).get(str(phase))
            key = f"{kind}_{phase}"
            if not isinstance(model, Mapping):
                report[key] = {"rows": 0, "max_abs_sum": None, "passed": False}
                continue
            coefficients = np.asarray(model.get("coefficients") or [], dtype=float)
            original = _finite_linear_predict(
                original_matrix,
                coefficients,
                float(model.get("intercept") or 0.0),
            )
            swapped_values = _finite_linear_predict(
                swapped_matrix,
                coefficients,
                float(model.get("intercept") or 0.0),
            )
            finite = np.isfinite(original) & np.isfinite(swapped_values) & complete
            max_abs_sum = (
                float(np.max(np.abs(original[finite] + swapped_values[finite])))
                if finite.any()
                else None
            )
            report[key] = {
                "rows": int(finite.sum()),
                "excluded_missing_rows": int((~complete).sum()),
                "max_abs_sum": max_abs_sum,
                "passed": bool(max_abs_sum is not None and max_abs_sum <= 1e-8),
            }
    return {
        "passed": bool(report) and all(bool(item["passed"]) for item in report.values()),
        "metrics": report,
        "definition": "predicted blue-minus-red curve plus swapped red-minus-blue curve",
    }


def _error_summary(values: Sequence[float]) -> dict[str, Any]:
    residual = np.asarray(values, dtype=float)
    valid = residual[np.isfinite(residual)]
    return {
        "rows": int(len(valid)),
        "rmse": float(np.sqrt(np.mean(valid * valid))) if len(valid) else None,
        "mae": float(np.mean(np.abs(valid))) if len(valid) else None,
        "bias": float(np.mean(valid)) if len(valid) else None,
    }


def _evaluate_transfer_slices(
    frame: pd.DataFrame,
    *,
    source_receipt: Mapping[str, Any],
    source_receipt_path: Path | str | None = None,
    source_receipt_file_sha256: str | None = None,
    feature_columns: Sequence[str],
    columns: Sequence[str],
    alpha: float,
    max_groups_per_column: int | None = None,
    crosswalk_path: Path | str | None = None,
    crosswalk_receipt_path: Path | str | None = None,
    crosswalk_receipt_file_sha256: str | None = None,
    series_partition_reference_frame: pd.DataFrame | None = None,
    series_partition_assignment_sha256: str | None = None,
    _series_partition_reference: _VerifiedPhaseSeriesReference | None = None,
) -> dict[str, Any]:
    """Evaluate earlier rows from other groups against each transfer group."""

    dates = _date_series(frame)
    output: dict[str, Any] = {}
    for column in columns:
        if column not in frame.columns:
            output[column] = {"available": False, "reason": "column_missing", "groups": {}}
            continue
        groups = frame[column].astype("string")
        groups = groups.where(groups.notna() & groups.str.strip().ne(""), "__missing__")
        reports: dict[str, Any] = {}
        unique_groups = sorted(str(value) for value in groups.unique())
        if max_groups_per_column is not None:
            unique_groups = unique_groups[: max(0, int(max_groups_per_column))]
        for group in unique_groups:
            test_mask = groups.eq(group)
            if not test_mask.any():
                continue
            test_start = dates.loc[test_mask].min()
            train_mask = dates.lt(test_start) & ~test_mask
            train = frame.loc[train_mask].copy()
            test = frame.loc[test_mask & dates.ge(test_start)].copy()
            if len(train) < max(1, len(feature_columns) + 1) or test.empty:
                reports[group] = {
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "available": False,
                    "reason": "insufficient_chronological_support",
                }
                continue
            artifact = fit_phase_curve(
                train,
                source_receipt=source_receipt,
                source_receipt_path=source_receipt_path,
                source_receipt_file_sha256=source_receipt_file_sha256,
                feature_columns=feature_columns,
                alpha=alpha,
                crosswalk_path=crosswalk_path,
                crosswalk_receipt_path=crosswalk_receipt_path,
                crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
                series_partition_reference_frame=series_partition_reference_frame,
                series_partition_assignment_sha256=series_partition_assignment_sha256,
                _series_partition_reference=_series_partition_reference,
            )
            residuals, _missing = _prediction_errors(artifact, test, feature_columns)
            metric_report: dict[str, Any] = {}
            for kind in ("gold", "xp"):
                metric_report[kind] = {}
                for phase in PHASES:
                    residual = residuals[kind][str(phase)]
                    target = _target(test, kind, phase)
                    valid = np.isfinite(residual) & np.isfinite(target)
                    model_summary = _error_summary(residual[valid])
                    baseline_summary = _error_summary(target[valid])
                    metric_report[kind][str(phase)] = {
                        **model_summary,
                        "baseline_zero": baseline_summary,
                        "baseline_rows_match": bool(
                            model_summary["rows"] == baseline_summary["rows"]
                        ),
                    }
            reports[group] = {
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "available": True,
                "metrics": metric_report,
            }
        output[column] = {
            "available": bool(reports),
            "groups": reports,
            "definition": "train on earlier rows from other groups; test on later held-out group",
        }
    return output


def evaluate_phase_curve(
    frame: pd.DataFrame,
    *,
    source_receipt: Mapping[str, Any],
    source_receipt_path: Path | str | None = None,
    source_receipt_file_sha256: str | None = None,
    feature_columns: Sequence[str],
    n_splits: int = 3,
    required_validation_folds: int | None = None,
    cluster_column: str | None = None,
    alpha: float = 10.0,
    transfer_columns: Sequence[str] = ("region", "patch"),
    max_transfer_groups: int | None = None,
    crosswalk_path: Path | str | None = None,
    crosswalk_receipt_path: Path | str | None = None,
    crosswalk_receipt_file_sha256: str | None = None,
    series_partition_reference_frame: pd.DataFrame | None = None,
    series_partition_assignment_sha256: str | None = None,
    _series_partition_reference: _VerifiedPhaseSeriesReference | None = None,
) -> dict[str, Any]:
    """Evaluate each phase on future rows with fold-internal fitting."""

    if required_validation_folds is not None and int(required_validation_folds) < 1:
        raise FuturePhaseCurveError("required_validation_folds must be positive")

    bound = bind_phase_source(
        frame,
        source_receipt,
        allow_subset=True,
        source_receipt_path=source_receipt_path,
        source_receipt_file_sha256=source_receipt_file_sha256,
    )
    crosswalk_values = (
        crosswalk_path,
        crosswalk_receipt_path,
        crosswalk_receipt_file_sha256,
    )
    if any(item is not None for item in crosswalk_values) and not all(
        item is not None for item in crosswalk_values
    ):
        raise FuturePhaseCurveError("phase series crosswalk inputs must be supplied together")
    value = bound.frame.copy()
    partition: dict[str, Any] | None = None
    if crosswalk_path is not None:
        value, partition = bind_phase_series_partition(
            value,
            bound.receipt,
            crosswalk_path=crosswalk_path,
            crosswalk_receipt_path=crosswalk_receipt_path,
            crosswalk_receipt_file_sha256=str(crosswalk_receipt_file_sha256),
            require_full_eligible=True,
            reference_frame=series_partition_reference_frame,
            _verified_reference=_series_partition_reference,
            expected_assignment_sha256=series_partition_assignment_sha256,
        )
    effective_cluster_column = cluster_column or (
        "series_id" if partition is not None else None
    )

    folds = chronological_folds(
        value,
        n_splits=n_splits,
        min_train_rows=max(1, len(feature_columns) + 1),
        cluster_column=effective_cluster_column,
    )
    errors: dict[str, dict[str, list[float]]] = {
        kind: {str(phase): [] for phase in PHASES} for kind in ("gold", "xp")
    }
    baseline_errors: dict[str, dict[str, list[float]]] = {
        kind: {str(phase): [] for phase in PHASES} for kind in ("gold", "xp")
    }
    missingness_errors: dict[str, dict[str, dict[str, list[float]]]] = {
        kind: {
            str(phase): {"complete": [], "any_missing": []}
            for phase in PHASES
        }
        for kind in ("gold", "xp")
    }
    fold_rows: list[dict[str, Any]] = []
    side_swap_checks: list[dict[str, Any]] = []
    for fold_number, (train_indices, test_indices) in enumerate(folds):
        train = value.iloc[train_indices].copy()
        test = value.iloc[test_indices].copy()
        artifact = fit_phase_curve(
            train,
            source_receipt=source_receipt,
            source_receipt_path=source_receipt_path,
            source_receipt_file_sha256=source_receipt_file_sha256,
            feature_columns=feature_columns,
            alpha=alpha,
            crosswalk_path=crosswalk_path,
            crosswalk_receipt_path=crosswalk_receipt_path,
            crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
            series_partition_reference_frame=series_partition_reference_frame,
            series_partition_assignment_sha256=series_partition_assignment_sha256,
            _series_partition_reference=_series_partition_reference,
        )
        prediction_errors, missing_count = _prediction_errors(artifact, test, feature_columns)
        side_swap_checks.append(side_swap_invariance_report(artifact, test, feature_columns))
        boundary = _cluster_boundary_diagnostics(
            value, test_indices, effective_cluster_column
        )
        row: dict[str, Any] = {
            "fold": fold_number,
            "train_rows": len(train),
            "test_rows": len(test),
            "cluster_boundary_exclusions": boundary,
        }
        for kind in ("gold", "xp"):
            for phase in PHASES:
                residual = prediction_errors[kind][str(phase)]
                target = _target(test, kind, phase)
                valid = np.isfinite(residual)
                if valid.any():
                    errors[kind][str(phase)].extend(float(value) for value in residual)
                    baseline_errors[kind][str(phase)].extend(
                        float(value) for value in target[valid]
                    )
                    missingness_errors[kind][str(phase)]["complete"].extend(
                        float(value)
                        for value in residual[valid & (missing_count == 0)]
                    )
                    missingness_errors[kind][str(phase)]["any_missing"].extend(
                        float(value)
                        for value in residual[valid & (missing_count > 0)]
                    )
                row[f"{kind}_{phase}_rows"] = int(valid.sum())
        fold_rows.append(row)
    metrics: dict[str, dict[str, Any]] = {kind: {} for kind in ("gold", "xp")}
    for kind in ("gold", "xp"):
        for phase in PHASES:
            metrics[kind][str(phase)] = _error_summary(errors[kind][str(phase)])
            baseline = _error_summary(baseline_errors[kind][str(phase)])
            metrics[kind][str(phase)]["baseline_zero"] = baseline
            model_rows = metrics[kind][str(phase)]["rows"]
            metrics[kind][str(phase)]["baseline_rows_match"] = bool(
                baseline["rows"] == model_rows
            )
            if (
                baseline["rmse"] is not None
                and metrics[kind][str(phase)]["rmse"] is not None
            ):
                metrics[kind][str(phase)]["rmse_gain_vs_zero"] = float(
                    baseline["rmse"] - metrics[kind][str(phase)]["rmse"]
                )
            if (
                baseline["mae"] is not None
                and metrics[kind][str(phase)]["mae"] is not None
            ):
                metrics[kind][str(phase)]["mae_gain_vs_zero"] = float(
                    baseline["mae"] - metrics[kind][str(phase)]["mae"]
                )
            metrics[kind][str(phase)]["missingness"] = {
                key: _error_summary(value)
                for key, value in missingness_errors[kind][str(phase)].items()
            }
    transfer = _evaluate_transfer_slices(
        value,
        source_receipt=bound.receipt,
        source_receipt_path=source_receipt_path,
        source_receipt_file_sha256=source_receipt_file_sha256,
        feature_columns=feature_columns,
        columns=transfer_columns,
        alpha=alpha,
        max_groups_per_column=max_transfer_groups,
        crosswalk_path=crosswalk_path,
        crosswalk_receipt_path=crosswalk_receipt_path,
        crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
        series_partition_reference_frame=series_partition_reference_frame,
        series_partition_assignment_sha256=series_partition_assignment_sha256,
        _series_partition_reference=_series_partition_reference,
    )
    side_swap = {
        "passed": bool(side_swap_checks) and all(item["passed"] for item in side_swap_checks),
        "folds": side_swap_checks,
        "definition": "predicted blue-minus-red curve plus swapped red-minus-blue curve",
    }
    fallback_rows = int(
        value["series_id_source"].astype("string").eq("game_fallback").sum()
        if "series_id_source" in value.columns
        else 0
    )
    series_identity = _series_identity_report(value)
    evaluation_dates = _date_series(value, "bound phase frame")
    lineage = _source_lineage(bound.receipt)
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "method": "chronological_fold_internal_ridge",
        "source": SOURCE,
        "source_as_of": bound.receipt["source_as_of"],
        "source_game_count": int(bound.receipt["source_game_count"]),
        "source_identity_sha256": str(bound.receipt["source_identity_sha256"]),
        "accepted_game_ids": list(bound.receipt["accepted_game_ids"]),
        "source_receipt_sha256": str(bound.receipt["receipt_sha256"]),
        "source_transport": lineage["transport"],
        "source_lineage": lineage,
        "evaluation_game_count": int(len(value)),
        "evaluation_identity_sha256": identity_sha256(_game_series(value, "phase frame")),
        "evaluation_window": {
            "date_start": _timestamp_text(evaluation_dates.min()),
            "date_end": _timestamp_text(evaluation_dates.max()),
            "definition": "UTC bounds of the exact accepted rows evaluated",
        },
        "folds": fold_rows,
        "metrics": metrics,
        "chronological_blocks_requested": int(n_splits),
        "validation_folds_required": (
            int(required_validation_folds)
            if required_validation_folds is not None
            else None
        ),
        "validation_folds_valid": len(fold_rows),
        "fold_count": len(fold_rows),
        "cluster_safe": bool(series_identity["authoritative"]),
        "cluster_column": effective_cluster_column or "game_uid",
        "cluster_fallback_rows": fallback_rows,
        "series_identity": series_identity,
        "cluster_boundary_exclusions": {
            "rows": int(
                sum(
                    int(row["cluster_boundary_exclusions"]["boundary_excluded_rows"])
                    for row in fold_rows
                )
            ),
            "clusters": int(
                sum(
                    int(row["cluster_boundary_exclusions"]["boundary_excluded_clusters"])
                    for row in fold_rows
                )
            ),
            "folds": [row["cluster_boundary_exclusions"] for row in fold_rows],
            "definition": "test clusters are ordered by first date; a train cluster must end before test start",
        },
        "missingness": {
            kind: {
                phase: metrics[kind][phase]["missingness"]
                for phase in PHASE_KEYS
            }
            for kind in ("gold", "xp")
        },
        "transfer": transfer,
        "regional_transfer": transfer.get("region", {}),
        "patch_transfer": transfer.get("patch", {}),
        "side_swap_invariance": side_swap,
        "authority": "development_only",
    }
    if partition is not None:
        output.update(
            {
                "series_partition_source": partition["source"],
                "series_partition_key_fields": list(partition["key_fields"]),
                "series_partition_mapping_sha256": partition["mapping_sha256"],
                "series_partition_crosswalk_sha256": partition["crosswalk_sha256"],
                "series_partition_artifact_sha256": partition["artifact_sha256"],
                "series_partition_receipt_sha256": partition["receipt_sha256"],
                "series_partition_receipt_file_sha256": partition[
                    "receipt_file_sha256"
                ],
                "series_partition_eligible_game_count": partition[
                    "eligible_game_count"
                ],
                "series_partition_eligible_identity_sha256": partition[
                    "eligible_identity_sha256"
                ],
                "series_partition_eligible_assignment_sha256": partition[
                    "eligible_assignment_sha256"
                ],
                "series_partition_reference_game_count": partition[
                    "reference_game_count"
                ],
                "series_partition_reference_assignment_sha256": partition[
                    "reference_assignment_sha256"
                ],
                "series_partition_eligible_game_ids": list(
                    partition["eligible_game_ids"]
                ),
                "series_partition": partition,
                "cross_model_series_partition": series_identity[
                    "cross_model_partition"
                ],
                "series_partition_proxy_authority_blocker": partition[
                    "proxy_authority_blocker"
                ],
            }
        )
    if source_receipt_path is not None:
        output["source_receipt_artifact"] = _receipt_file_reference(
            source_receipt_path,
            expected_sha256=source_receipt_file_sha256,
        )
    return output


def side_swap_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the same phase rows from the opposing side's perspective."""

    result = frame.copy()
    for name in result.columns:
        text = str(name).casefold()
        if ("_diff_" in text or text.endswith("_diff")) and not (
            text.endswith("_missing") or text.endswith("_censored")
        ):
            values = pd.to_numeric(result[name], errors="coerce")
            result[name] = -values
    if "y_blue_win" in result.columns:
        labels = pd.to_numeric(result["y_blue_win"], errors="coerce")
        result["y_blue_win"] = labels.where(labels.isna(), 1.0 - labels)
    return result


__all__ = [
    "BoundPhaseSource",
    "FINAL_METRIC_ALIASES",
    "FuturePhaseCurveError",
    "MODEL_VERSION",
    "MATERIAL_ADVANTAGE_THRESHOLD",
    "PHASE_FEATURE_DECLARATION",
    "PHASE_FEATURE_FAMILY",
    "PHASE_SHAPE_AVAILABILITY_FEATURES",
    "PHASE_SHAPE_FEATURES",
    "PHASE_SHAPE_INVARIANT_FEATURES",
    "PHASE_SHAPE_MATERIAL_THRESHOLD",
    "PHASE_SHAPE_SIGNED_FEATURES",
    "PHASES",
    "INVARIANT_PHASE_SHAPE_FEATURES",
    "SIGNED_PHASE_SHAPE_FEATURES",
    "SCHEMA_VERSION",
    "assert_pregame_feature_names",
    "bind_phase_source",
    "bind_phase_series_partition",
    "build_strict_prior_team_features",
    "chronological_folds",
    "evaluate_phase_curve",
    "fit_phase_curve",
    "phase_curve_measures",
    "phase_series_assignment_sha256",
    "phase_shape_features",
    "phase_shape_side_swap",
    "prepare_phase_frame",
    "score_phase_curve",
    "side_swap_phase_shape_features",
    "side_swap_invariance_report",
    "side_swap_frame",
    "strict_prior_final_history",
    "validate_phase_shape_side_swap",
    "verify_accepted_census_artifact",
    "verify_source_receipt_artifact",
]
