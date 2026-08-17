"""Publish one validated Scryglass release through Supabase Postgres.

The worker uploads page-ready JSON into staging rows. One database function
checks the complete asset set and changes the active release in one transaction.
Readers can therefore see either the previous complete release or the new one.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lol_kills.export.pack_spec import (
    OPTIONAL_PUBLIC_FILES,
    PUBLIC_ASSET_PATHS,
    PUBLIC_RATING_REQUIRED_FILES,
)
from lol_kills.export.public_query_projection import (
    QUERY_API_SCHEMA,
    QUERY_DATASETS,
    QUERY_PROJECTION_PATH,
    TIER_QUERY_DATASETS,
    PublicQueryProjectionError,
    build_tier_query_datasets,
    canonical_query_bytes,
    query_dataset_receipt,
    validate_public_query_projection,
)
from lol_kills.export.promoted_draft_authority import (
    PromotedDraftAuthorityError,
    validate_promoted_results_payload,
)
from lol_kills.v2.patch_identity import PatchIdentityError, public_patch


TIER_ASSET_PATH = "rankings/tierlists.json"
TIER_LATEST_ASSET_PATH = "rankings/tierlists-latest.json"
DRAFT_ASSET_PATH = "features/draft_records.json"
PROMOTED_DRAFT_RESULTS_PATH = "features/promoted_draft_results.json"
DRAFT_AUTHORITY_SCHEMA = "scryglass:draft-authority:v1"
DRAFT_RECORDS_SCHEMA = "scryglass:draft-records:v1"
PUBLIC_ASSET_PATH_SET = frozenset(PUBLIC_ASSET_PATHS)
PACK_ID_RE = re.compile(r"^v\d{4}\.\d{2}\.\d{2}\.\d{6}$")
REQUEST_TIMEOUT_SECONDS = 150.0
MAX_RETENTION_PRUNE_CALLS = 20
MAX_QUERY_STAGE_ATTEMPTS = 4
RETRYABLE_QUERY_STAGE_HTTP_CODES = frozenset(
    {408, 425, 429, 500, 502, 503, 504, 520, 522, 524}
)
RESUMABLE_UPLOAD_THRESHOLD_BYTES = 6 * 1024 * 1024
RESUMABLE_UPLOAD_CHUNK_BYTES = 6 * 1024 * 1024
MAX_STORAGE_UPLOAD_ATTEMPTS = 4
RETRYABLE_STORAGE_HTTP_CODES = frozenset(
    {408, 425, 429, 500, 502, 503, 504, 520, 522, 524, 544}
)
QUERY_STAGE_BATCH_ROWS = 500
QUERY_STAGE_BATCH_BYTES = 3_200_000
PUBLIC_ASSET_CONTENT_TYPE = "application/json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DRAFT_ISSUED_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
QUERY_TABLE = "scryglass_public_query_rows"
QUERY_RECEIPT_TABLE = "scryglass_public_query_receipts"
QUERY_ROW_FIELDS = frozenset(
    {
        "row_key",
        "player_id",
        "team_id",
        "champion_id",
        "identity_id",
        "name",
        "search_key",
        "kind",
        "alias_key",
        "role",
        "team",
        "league",
        "tier",
        "active",
        "rating",
        "adjusted_rating",
        "movement",
        "games",
        "wins",
        "win_rate",
        "grade_a_games",
        "grade_games",
        "champion",
        "champion_key",
        "score",
        "game_id",
        "played_at",
        "year",
        "blue_team",
        "red_team",
        "blue_team_id",
        "red_team_id",
        "blue_win",
        "champions",
        "ordinal",
        "image_url",
        "patch",
        "region",
        "rank",
        "played_maps",
        "scope_id",
        "reference_id",
        "payload",
        "source_bytes",
        "source_sha256",
        "row_sha256",
    }
)


class SupabasePublicationError(RuntimeError):
    """A Supabase release could not be staged, activated, or verified."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# QRDBG: TEMPORARY query-row staging instrumentation. Diagnosis only.
#
# Every helper below is strictly additive and inert unless SCRYGLASS_QRDBG_PATH
# names a writable file. Nothing here changes a validation, a digest, a gate or
# a threshold; the probes only observe what the publisher already built.
# ---------------------------------------------------------------------------

# Mirrors public.scryglass_json_has_draft_fields as redefined by
# supabase/migrations/20260815060001_descriptive_draft_query_api.sql:219.
# Reported for comparison only; never used to accept or reject a row.
_QRDBG_BANNED_KEYS = frozenset(
    {
        "average_win_share", "best_available", "betting", "development_composite",
        "draft_authority", "draft_edge", "draft_pool", "draft_probability",
        "draft_score", "draft_win_share", "elo", "ev", "expected_value",
        "fair_odds", "gold", "live_state", "match_probability",
        "match_win_expectation", "momentum", "mu_diff", "objectives", "odds",
        "p_blue", "p_red", "phase_curve", "player_elo", "player_rating",
        "probability", "r9e", "r9e_state_space", "rating_uncertainty",
        "recommendation", "sigma_pair", "strength", "team_elo", "team_rating",
        "win_probability",
    }
)
_QRDBG_BANNED_PREFIXES = ("r9e_", "control_", "strength_", "phase_", "live_")
_QRDBG_CONDITIONAL_KEYS = ("draft_contribution", "draft_metric")
_QRDBG_MAX_BISECT_REJECTS = 25


def _qrdbg_path() -> str | None:
    """Return the probe log path, or None when the probe is disabled."""

    raw = os.environ.get("SCRYGLASS_QRDBG_PATH")
    if not isinstance(raw, str):
        return None
    path = raw.strip()
    return path or None


def _qrdbg(event: str, **fields: Any) -> None:
    """Append one JSON line to the probe log. Never raises."""

    path = _qrdbg_path()
    if not path:
        return
    try:
        record: dict[str, Any] = {
            "tag": "QRDBG",
            "utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
    except Exception:  # noqa: BLE001 - a probe must never break a refresh
        pass


def _qrdbg_draft_key_hits(value: object, prefix: str = "$") -> dict[str, list[str]]:
    """Report the key paths the server-side draft-field guard would look at."""

    banned: list[str] = []
    conditional: list[str] = []

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                normalized = str(key).lower()
                child_path = f"{path}.{key}"
                if normalized in _QRDBG_CONDITIONAL_KEYS:
                    conditional.append(child_path)
                    continue
                if normalized in _QRDBG_BANNED_KEYS or any(
                    normalized.startswith(item) for item in _QRDBG_BANNED_PREFIXES
                ):
                    banned.append(child_path)
                    continue
                walk(child, child_path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path}[{index}]")

    walk(value, prefix)
    return {"banned": banned[:40], "conditional": conditional[:40]}


def _qrdbg_row_diagnosis(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute, read-only, every clause the server digest gate evaluates."""

    diagnosis: dict[str, Any] = {}
    source_text = row.get("source_json")
    payload_text = row.get("payload_json")
    expected = row.get("row_sha256")
    try:
        diagnosis["source_json_bytes"] = len(str(source_text).encode("utf-8"))
        diagnosis["payload_json_bytes"] = len(str(payload_text).encode("utf-8"))
        diagnosis["computed_source_json_sha256"] = _sha256(
            str(source_text).encode("utf-8")
        )
        diagnosis["client_row_sha256"] = expected
        diagnosis["clause1_row_digest_matches"] = (
            diagnosis["computed_source_json_sha256"] == expected
        )
    except Exception as error:  # noqa: BLE001
        diagnosis["digest_error"] = repr(error)
        return diagnosis
    try:
        source = json.loads(str(source_text))
    except Exception as error:  # noqa: BLE001
        diagnosis["source_json_parse_error"] = repr(error)
        return diagnosis
    payload = source.get("payload") if isinstance(source, dict) else None
    diagnosis["row_key"] = source.get("row_key") if isinstance(source, dict) else None
    diagnosis["clause2_payload_is_object"] = isinstance(payload, dict)
    try:
        diagnosis["clause3_payload_json_matches_source_payload"] = (
            json.loads(str(payload_text)) == payload
        )
    except Exception as error:  # noqa: BLE001
        diagnosis["clause3_payload_json_matches_source_payload"] = f"error: {error!r}"
    computed_payload_sha = _sha256(str(payload_text).encode("utf-8"))
    diagnosis["computed_payload_json_sha256"] = computed_payload_sha
    diagnosis["client_source_sha256"] = (
        source.get("source_sha256") if isinstance(source, dict) else None
    )
    diagnosis["clause4_payload_digest_matches"] = (
        computed_payload_sha == diagnosis["client_source_sha256"]
    )
    row_key = diagnosis.get("row_key")
    diagnosis["clause5_row_key_length"] = len(str(row_key)) if row_key is not None else 0
    diagnosis["clause5_row_key_in_range"] = 1 <= diagnosis["clause5_row_key_length"] <= 200
    diagnosis["clause6_draft_key_hits"] = _qrdbg_draft_key_hits(source)
    if isinstance(payload, dict):
        for key in _QRDBG_CONDITIONAL_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                diagnosis[f"{key}_keys"] = sorted(str(item) for item in candidate)
    return diagnosis


def _release_id(value: object) -> str:
    release_id = str(value or "")
    if not PACK_ID_RE.fullmatch(release_id):
        raise SupabasePublicationError("release ID is invalid")
    return release_id


def _rollback_release_id(value: object) -> str:
    try:
        return _release_id(value)
    except SupabasePublicationError as error:
        raise SupabasePublicationError("rollback release ID is invalid") from error


def _public_asset_path(value: object) -> str:
    path = str(value or "")
    if path not in PUBLIC_ASSET_PATH_SET:
        raise SupabasePublicationError("public asset path is not allowed")
    return path


def _storage_path(value: object) -> str:
    storage_path = str(value or "")
    release_id, separator, path = storage_path.partition("/")
    if not separator or _release_id(release_id) + "/" + _public_asset_path(path) != storage_path:
        raise SupabasePublicationError("Supabase Storage path is invalid")
    return storage_path


def _retention_storage_path(value: object) -> str:
    storage_path = str(value or "")
    release_id, separator, path = storage_path.partition("/")
    if (
        not separator
        or not (
            PACK_ID_RE.fullmatch(release_id)
        )
        or f"{release_id}/{_public_asset_path(path)}" != storage_path
    ):
        raise SupabasePublicationError("retained Supabase Storage path is invalid")
    return storage_path


def _draft_authority(manifest: dict[str, Any], release_id: str) -> dict[str, Any]:
    """Validate the descriptive Draft receipt at the publication boundary."""

    candidate = manifest.get("draft_authority")
    if not isinstance(candidate, dict):
        return {
            "schema_version": DRAFT_AUTHORITY_SCHEMA,
            "status": "unavailable",
            "release_id": release_id,
            "model_version": None,
            "artifact_sha256": None,
            "receipt_sha256": None,
            "issued_utc": None,
            "reason": "model_not_promoted",
        }
    status = candidate.get("status")
    if status == "promoted":
        artifact_sha256 = candidate.get("artifact_sha256")
        receipt_sha256 = candidate.get("receipt_sha256")
        issued_utc = candidate.get("issued_utc")
        model_version = candidate.get("model_version")
        descriptive_authority = _draft_authority(
            {"draft_authority": candidate.get("descriptive_authority")},
            release_id,
        )
        if (
            candidate.get("schema_version") != DRAFT_AUTHORITY_SCHEMA
            or candidate.get("release_id") != release_id
            or candidate.get("authority") != "promoted"
            or candidate.get("estimand")
            != "prematch_map_win_probability_with_controlled_draft_intervention"
            or not isinstance(model_version, str)
            or not model_version.strip()
            or not isinstance(artifact_sha256, str)
            or not SHA256_RE.fullmatch(artifact_sha256)
            or not isinstance(receipt_sha256, str)
            or not SHA256_RE.fullmatch(receipt_sha256)
            or not _valid_issued_utc(issued_utc)
            or candidate.get("probability_authority") is not True
            or candidate.get("recommendation_authority") is not True
            or candidate.get("betting_authority") is not False
            or candidate.get("reason") is not None
            or descriptive_authority.get("status") != "descriptive"
        ):
            raise SupabasePublicationError("promoted Draft Score authority is invalid")
        return {
            "schema_version": DRAFT_AUTHORITY_SCHEMA,
            "status": "promoted",
            "authority": "promoted",
            "release_id": release_id,
            "model_version": model_version.strip(),
            "artifact_sha256": artifact_sha256,
            "receipt_sha256": receipt_sha256,
            "issued_utc": issued_utc,
            "estimand": candidate["estimand"],
            "probability_authority": True,
            "recommendation_authority": True,
            "betting_authority": False,
            "reason": None,
            "descriptive_authority": descriptive_authority,
        }
    if status == "descriptive":
        artifact_sha256 = candidate.get("artifact_sha256")
        receipt_sha256 = candidate.get("receipt_sha256")
        issued_utc = candidate.get("issued_utc")
        if (
            candidate.get("schema_version") != DRAFT_AUTHORITY_SCHEMA
            or candidate.get("release_id") != release_id
            or candidate.get("authority") != "descriptive"
            or candidate.get("estimand") != "composition_only"
            or candidate.get("model_version")
            != "draft-recommendation-static-v2"
            or not isinstance(artifact_sha256, str)
            or not SHA256_RE.fullmatch(artifact_sha256)
            or not isinstance(receipt_sha256, str)
            or not SHA256_RE.fullmatch(receipt_sha256)
            or not _valid_issued_utc(issued_utc)
            or candidate.get("probability_authority") is not False
            or candidate.get("recommendation_authority") is not False
            or candidate.get("betting_authority") is not False
            or candidate.get("reason") is not None
        ):
            raise SupabasePublicationError("descriptive Draft Score authority is invalid")
        return {
            "schema_version": DRAFT_AUTHORITY_SCHEMA,
            "status": "descriptive",
            "authority": "descriptive",
            "release_id": release_id,
            "model_version": str(candidate["model_version"]),
            "artifact_sha256": artifact_sha256,
            "receipt_sha256": receipt_sha256,
            "issued_utc": issued_utc,
            "estimand": "composition_only",
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
            "reason": None,
        }
    if status not in {None, "unavailable"}:
        raise SupabasePublicationError("Draft Score authority status is invalid")
    reason = candidate.get("reason")
    return {
        "schema_version": DRAFT_AUTHORITY_SCHEMA,
        "status": "unavailable",
        "release_id": release_id,
        "model_version": None,
        "artifact_sha256": None,
        "receipt_sha256": None,
        "issued_utc": None,
        "reason": str(reason or "model_not_promoted"),
    }


def _valid_issued_utc(value: object) -> bool:
    if not isinstance(value, str) or not DRAFT_ISSUED_UTC_RE.fullmatch(value):
        return False
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _validate_descriptive_draft_records(payload: object) -> None:
    """Reject predictive fields from the descriptive Draft asset."""

    if not isinstance(payload, dict):
        raise SupabasePublicationError("descriptive Draft asset is malformed")
    if (
        payload.get("schema_version") != DRAFT_RECORDS_SCHEMA
        or payload.get("authority") != "descriptive"
        or payload.get("estimand") != "composition_only"
        or payload.get("model_version") != "draft-recommendation-static-v2"
        or not isinstance(payload.get("games"), dict)
    ):
        raise SupabasePublicationError("descriptive Draft asset binding is invalid")
    if not payload["games"]:
        raise SupabasePublicationError("descriptive Draft asset has no usable games")
    forbidden = {
        "probability",
        "win_probability",
        "draft_probability",
        "draft_win_share",
        "average_win_share",
        "p_blue",
        "p_red",
        "odds",
        "ev",
        "r9e",
        "r9e_state_space",
        "development_composite",
        "match_probability",
        "match_win_expectation",
        "team_rating",
        "player_rating",
        "elo",
        "mu_diff",
        "sigma_pair",
        "momentum",
        "gold",
        "objectives",
        "recommendation",
        "betting",
    }

    def scan(value: object) -> None:
        if isinstance(value, dict):
            normalized_keys = {str(key).strip().casefold() for key in value}
            if forbidden.intersection(normalized_keys) or any(
                key.startswith("r9e_") or key.startswith("r9e|")
                for key in normalized_keys
            ):
                raise SupabasePublicationError(
                    "descriptive Draft asset contains predictive fields"
                )
            for nested in value.values():
                scan(nested)
        elif isinstance(value, list):
            for nested in value:
                scan(nested)

    scan(payload)
    for game in payload["games"].values():
        if not isinstance(game, dict):
            raise SupabasePublicationError("descriptive Draft game is malformed")
        edge = game.get("draft_edge")
        if not isinstance(edge, (int, float)) or isinstance(edge, bool):
            raise SupabasePublicationError("descriptive Draft edge is invalid")
        pool = game.get("draft_pool")
        if not isinstance(pool, dict) or pool.get("status") != "complete":
            raise SupabasePublicationError("descriptive Draft pool is incomplete")
        if not isinstance(pool.get("patch"), str) or not pool["patch"].strip():
            raise SupabasePublicationError("descriptive Draft pool patch is missing")
        bans = pool.get("bans")
        if not isinstance(bans, dict):
            raise SupabasePublicationError("descriptive Draft bans are incomplete")
        blue_bans = bans.get("Blue")
        red_bans = bans.get("Red")
        if (
            not isinstance(blue_bans, list)
            or not isinstance(red_bans, list)
            or len(blue_bans) != 5
            or len(red_bans) != 5
            or len({str(item).strip().casefold() for item in (*blue_bans, *red_bans)}) != 10
            or any(not isinstance(item, str) or not item.strip() for item in (*blue_bans, *red_bans))
        ):
            raise SupabasePublicationError("descriptive Draft bans are incomplete")
        picked = pool.get("picked")
        if not isinstance(picked, list) or len(picked) != 10:
            raise SupabasePublicationError("descriptive Draft picks are incomplete")
        identities: set[tuple[str, str]] = set()
        champions: set[str] = set()
        orders: set[int] = set()
        for pick in picked:
            if not isinstance(pick, dict):
                raise SupabasePublicationError("descriptive Draft pick is malformed")
            side = str(pick.get("side") or "").strip().title()
            role = str(pick.get("role") or "").strip().casefold()
            champion = str(pick.get("champion") or "").strip().casefold()
            order = pick.get("order")
            if (
                side not in {"Blue", "Red"}
                or not role
                or not champion
                or not isinstance(order, int)
                or isinstance(order, bool)
                or order not in range(1, 11)
                or (side, role) in identities
                or champion in champions
                or order in orders
                or not isinstance(pick.get("best_available"), bool)
                or not isinstance(pick.get("tier_rank"), int)
                or isinstance(pick.get("tier_rank"), bool)
                or not isinstance(pick.get("available_count"), int)
                or isinstance(pick.get("available_count"), bool)
                or pick.get("available_count") < 1
            ):
                raise SupabasePublicationError("descriptive Draft pick evidence is incomplete")
            identities.add((side, role))
            champions.add(champion)
            orders.add(order)
        if len(identities) != 10 or orders != set(range(1, 11)):
            raise SupabasePublicationError("descriptive Draft pick evidence is incomplete")
        if pool.get("evaluated_picks") != 10:
            raise SupabasePublicationError("descriptive Draft pool evaluation is incomplete")


def _pack_asset_path(pack_dir: Path, path: str) -> Path:
    root = pack_dir.resolve(strict=True)
    candidate = (root / _public_asset_path(path)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise SupabasePublicationError("public asset path leaves the pack") from error
    if not candidate.is_file():
        raise SupabasePublicationError(f"public asset is not a file: {path}")
    return candidate


def _query_projection_path(pack_dir: Path) -> Path:
    root = pack_dir.resolve(strict=True)
    candidate = (root / QUERY_PROJECTION_PATH).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise SupabasePublicationError("query projection path leaves the pack") from error
    if not candidate.is_file():
        raise SupabasePublicationError("query projection is missing")
    return candidate


def _prepare_query_datasets(
    pack_dir: Path,
    manifest: dict[str, Any],
    tier_body: dict[str, Any],
    release_id: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any] | None]:
    """Read the internal projection and bind its receipts to publication."""

    query_api = manifest.get("query_api")
    if query_api is None:
        return {}, None
    projection_metadata = query_api.get("projection") if isinstance(query_api, dict) else None
    if (
        not isinstance(query_api, dict)
        or query_api.get("schema_version") != QUERY_API_SCHEMA
        or query_api.get("status") != "available"
        or not isinstance(projection_metadata, dict)
        or projection_metadata.get("path") != QUERY_PROJECTION_PATH
    ):
        raise SupabasePublicationError("query API manifest is invalid")
    path = _query_projection_path(pack_dir)
    raw = path.read_bytes()
    if (
        projection_metadata.get("bytes") != len(raw)
        or projection_metadata.get("sha256") != _sha256(raw)
    ):
        raise SupabasePublicationError("query projection checksum failed")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupabasePublicationError("query projection is invalid JSON") from error
    if not isinstance(parsed, dict):
        raise SupabasePublicationError("query projection is malformed")
    try:
        datasets = validate_public_query_projection(parsed, release_id=release_id)
        tier_datasets = build_tier_query_datasets(tier_body)
    except PublicQueryProjectionError as error:
        raise SupabasePublicationError(str(error)) from error
    datasets.update(tier_datasets)
    receipts = {
        dataset: query_dataset_receipt(dataset, rows)
        for dataset, rows in datasets.items()
    }
    if set(receipts) != {*QUERY_DATASETS, *TIER_QUERY_DATASETS}:
        raise SupabasePublicationError("query dataset inventory is not exact")
    return datasets, {
        "schema_version": QUERY_API_SCHEMA,
        "status": "available",
        "projection_sha256": _sha256(raw),
        "datasets": receipts,
    }


def _project_url(value: str) -> str:
    raw = value.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
        or not (parsed.hostname or "").endswith(".supabase.co")
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Supabase URL must be an HTTPS project URL")
    return raw


def _secret_key(value: str) -> str:
    key = value.strip()
    if not key.startswith("sb_secret_") or len(key) < 24 or any(character.isspace() for character in key):
        raise ValueError("Supabase secret key is malformed")
    return key


class SupabasePublicData:
    """Small PostgREST client for the Scryglass public release tables."""

    def __init__(
        self,
        project_url: str,
        api_key: str,
        *,
        opener: Any | None = None,
    ) -> None:
        self.project_url = _project_url(project_url)
        self._api_key = _secret_key(api_key)
        self._opener = opener or urllib.request.build_opener()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(project_url={self.project_url!r}, api_key=<redacted>)"

    def _request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
        *,
        prefer: str | None = None,
    ) -> Any:
        body = None
        headers = {
            "apikey": self._api_key,
            "Accept": "application/json",
        }
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if prefer:
            headers["Prefer"] = prefer
        request = urllib.request.Request(
            f"{self.project_url}/rest/v1/{path.lstrip('/')}",
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = ""
            try:
                parsed = json.loads(error.read().decode("utf-8"))
                detail = str(parsed.get("message") or parsed.get("hint") or "")[:300]
            except Exception:  # noqa: BLE001
                detail = ""
            suffix = f": {detail}" if detail else ""
            raise SupabasePublicationError(
                f"Supabase request failed with HTTP {error.code} for {path.split('?', 1)[0]}{suffix}"
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise SupabasePublicationError(
                f"Supabase request failed for {path.split('?', 1)[0]}"
            ) from error
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SupabasePublicationError("Supabase returned invalid JSON") from error

    def release(self, release_id: str) -> dict[str, Any] | None:
        release_id = _release_id(release_id)
        encoded = urllib.parse.quote(release_id, safe="")
        rows = self._request(
            "GET",
            "scryglass_public_releases"
            f"?release_id=eq.{encoded}&select=release_id,status,manifest,source_as_of,activated_at&limit=1",
        )
        if not isinstance(rows, list):
            raise SupabasePublicationError("Supabase release response is malformed")
        return rows[0] if rows and isinstance(rows[0], dict) else None

    def active_release(self) -> dict[str, Any] | None:
        rows = self._request(
            "GET",
            "scryglass_public_releases"
            "?status=eq.active&select=release_id,status,manifest,source_as_of,activated_at&limit=1",
        )
        if not isinstance(rows, list):
            raise SupabasePublicationError("Supabase active release response is malformed")
        return rows[0] if rows and isinstance(rows[0], dict) else None

    def asset(self, release_id: str, path: str) -> dict[str, Any] | None:
        release_id = _release_id(release_id)
        path = _public_asset_path(path)
        release = urllib.parse.quote(release_id, safe="")
        asset_path = urllib.parse.quote(path, safe="")
        rows = self._request(
            "GET",
            "scryglass_public_assets"
            f"?release_id=eq.{release}&path=eq.{asset_path}"
            "&select=path,body,storage_path,bytes,sha256,content_type&limit=1",
        )
        if not isinstance(rows, list):
            raise SupabasePublicationError("Supabase asset response is malformed")
        return rows[0] if rows and isinstance(rows[0], dict) else None

    def asset_metadata(self, release_id: str) -> dict[str, dict[str, Any]]:
        release_id = _release_id(release_id)
        release = urllib.parse.quote(release_id, safe="")
        rows = self._request(
            "GET",
            "scryglass_public_assets"
            f"?release_id=eq.{release}&select=path,bytes,sha256,content_type,storage_path",
        )
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise SupabasePublicationError("Supabase asset inventory is malformed")
        return {
            str(row["path"]): row
            for row in rows
            if isinstance(row.get("path"), str)
        }

    def query_receipts(self, release_id: str) -> dict[str, dict[str, Any]]:
        release_id = _release_id(release_id)
        encoded = urllib.parse.quote(release_id, safe="")
        rows = self._request(
            "GET",
            f"{QUERY_RECEIPT_TABLE}?release_id=eq.{encoded}"
            "&select=dataset,row_count,source_bytes,source_sha256,row_digest_sha256,storage_bytes,storage_sha256",
        )
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise SupabasePublicationError("Supabase query receipt response is malformed")
        return {
            str(row["dataset"]): row
            for row in rows
            if isinstance(row.get("dataset"), str)
        }

    def write_refresh_run(self, payload: dict[str, Any]) -> None:
        fields = {
            key: payload.get(key)
            for key in (
                "run_id",
                "scheduled_for",
                "retry_of",
                "status",
                "stage",
                "input_fingerprint",
                "worker_commit",
                "requirements_lock_sha256",
                "source_file_sha256",
                "source_observed_through",
                "release_id",
                "accepted_games",
                "new_games",
                "corrected_games",
                "unchanged_games",
                "quarantined_games",
                "stage_durations",
                "failure_code",
                "failure_detail",
                "started_at",
                "completed_at",
            )
        }
        fields["updated_at"] = payload.get("updated_at")
        self._request(
            "POST",
            "scryglass_refresh_runs?on_conflict=run_id",
            [fields],
            prefer="resolution=merge-duplicates,return=minimal",
        )

    def write_public_health(self, payload: dict[str, Any]) -> None:
        row = {
            "health_id": "public-refresh",
            "status": payload["status"],
            "refresh_status": payload["refresh_status"],
            "checked_at": payload["checked_at"],
            "last_success_at": payload.get("last_success_at"),
            "source_as_of": payload.get("source_as_of"),
            "active_release_id": payload.get("active_release_id"),
            "last_run_id": payload.get("last_run_id"),
            "worker_commit": payload.get("worker_commit"),
            "stale": bool(payload.get("stale")),
            "updated_at": payload["checked_at"],
        }
        self._request(
            "POST",
            "scryglass_public_health?on_conflict=health_id",
            [row],
            prefer="resolution=merge-duplicates,return=minimal",
        )

    def write_diagnostic_credential(self, token: str) -> None:
        if not 32 <= len(token) <= 512 or any(character.isspace() for character in token):
            raise SupabasePublicationError("diagnostic token is malformed")
        self._request(
            "POST",
            "scryglass_diagnostic_credentials?on_conflict=credential_id",
            [
                {
                    "credential_id": "public-health",
                    "token_sha256": _sha256(token.encode("utf-8")),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            ],
            prefer="resolution=merge-duplicates,return=minimal",
        )

    def storage_object(self, storage_path: str) -> bytes:
        storage_path = _storage_path(storage_path)
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in storage_path.split("/"))
        request = urllib.request.Request(
            f"{self.project_url}/storage/v1/object/authenticated/scryglass-public/{encoded}",
            method="GET",
            headers={
                "apikey": self._api_key,
            },
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    return response.read()
            except Exception as error:  # noqa: BLE001
                last_error = error
                time.sleep(0.5 * (attempt + 1))
        raise SupabasePublicationError("Supabase Storage read failed") from last_error

    def storage_object_metadata(self, storage_path: str) -> dict[str, Any]:
        """Object metadata (size, etag) without downloading the body.

        Used for post-publish verification so a 140MB release is verified by
        metadata instead of being downloaded back (cached-egress cost).
        """
        storage_path = _storage_path(storage_path)
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in storage_path.split("/"))
        request = urllib.request.Request(
            f"{self.project_url}/storage/v1/object/info/authenticated/scryglass-public/{encoded}",
            method="GET",
            headers={
                "apikey": self._api_key,
            },
        )
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as error:  # noqa: BLE001
            raise SupabasePublicationError("Supabase Storage metadata read failed") from error
        if not isinstance(payload, dict):
            raise SupabasePublicationError("Supabase Storage metadata is malformed")
        return payload

    def put_storage_object(
        self,
        storage_path: str,
        raw: bytes,
        *,
        sha256: str,
        content_type: str,
    ) -> None:
        storage_path = _storage_path(storage_path)
        if sha256 != _sha256(raw):
            raise SupabasePublicationError("Supabase Storage upload digest is invalid")
        if content_type != PUBLIC_ASSET_CONTENT_TYPE:
            raise SupabasePublicationError("Supabase Storage content type is invalid")
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in storage_path.split("/"))
        custom_metadata = {
            "sha256": sha256,
            "bytes": len(raw),
            "content_type": content_type,
        }
        if len(raw) > RESUMABLE_UPLOAD_THRESHOLD_BYTES:
            self._put_resumable_storage_object(
                storage_path,
                raw,
                custom_metadata=custom_metadata,
                content_type=content_type,
            )
            return
        metadata = base64.b64encode(
            json.dumps(custom_metadata, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        request = urllib.request.Request(
            f"{self.project_url}/storage/v1/object/scryglass-public/{encoded}",
            data=raw,
            method="POST",
            headers={
                "apikey": self._api_key,
                "Content-Type": "application/json",
                "Cache-Control": "public, max-age=31536000, immutable",
                "x-upsert": "false",
                "x-metadata": metadata,
            },
        )
        try:
            with self._opener.open(request, timeout=180.0) as response:
                response.read()
            return
        except urllib.error.HTTPError as error:
            if error.code not in {400, 409}:
                raise SupabasePublicationError(
                    f"Supabase Storage upload failed with HTTP {error.code}"
                ) from error
        existing_metadata = self.storage_object_metadata(storage_path)
        existing_custom = existing_metadata.get("metadata")
        if isinstance(existing_custom, str):
            try:
                existing_custom = json.loads(existing_custom)
            except json.JSONDecodeError:
                existing_custom = None
        if (
            int(existing_metadata.get("size") or -1) != len(raw)
            or not isinstance(existing_custom, dict)
            or existing_custom.get("sha256") != sha256
            or existing_custom.get("bytes") != len(raw)
            or existing_custom.get("content_type") != content_type
        ):
            raise SupabasePublicationError("existing Supabase Storage metadata is different")
        existing = self.storage_object(storage_path)
        if len(existing) != len(raw) or _sha256(existing) != _sha256(raw):
            raise SupabasePublicationError("existing Supabase Storage object has different content")

    def _put_resumable_storage_object(
        self,
        storage_path: str,
        raw: bytes,
        *,
        custom_metadata: dict[str, Any],
        content_type: str,
    ) -> None:
        """Upload one large immutable object through Supabase's TUS endpoint."""

        project = urllib.parse.urlsplit(self.project_url)
        project_ref = (project.hostname or "").removesuffix(".supabase.co")
        endpoint = (
            f"https://{project_ref}.storage.supabase.co/storage/v1/upload/resumable"
        )
        upload_metadata = {
            "bucketName": "scryglass-public",
            "objectName": storage_path,
            "contentType": content_type,
            "cacheControl": "31536000",
            "metadata": json.dumps(custom_metadata, separators=(",", ":")),
        }
        metadata_header = ",".join(
            f"{key} {base64.b64encode(value.encode('utf-8')).decode('ascii')}"
            for key, value in upload_metadata.items()
        )
        location = None
        for attempt in range(MAX_STORAGE_UPLOAD_ATTEMPTS):
            create = urllib.request.Request(
                endpoint,
                data=b"",
                method="POST",
                headers={
                    "apikey": self._api_key,
                    "Tus-Resumable": "1.0.0",
                    "Upload-Length": str(len(raw)),
                    "Upload-Metadata": metadata_header,
                    "x-upsert": "false",
                },
            )
            try:
                with self._opener.open(
                    create, timeout=REQUEST_TIMEOUT_SECONDS
                ) as response:
                    location = response.headers.get("Location")
                break
            except urllib.error.HTTPError as error:
                if error.code in {400, 409}:
                    self._verify_existing_storage_object(
                        storage_path, raw, custom_metadata
                    )
                    return
                retryable = error.code in RETRYABLE_STORAGE_HTTP_CODES
                if not retryable or attempt + 1 >= MAX_STORAGE_UPLOAD_ATTEMPTS:
                    raise SupabasePublicationError(
                        f"Supabase resumable upload creation failed with HTTP {error.code}"
                    ) from error
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt + 1 >= MAX_STORAGE_UPLOAD_ATTEMPTS:
                    raise SupabasePublicationError(
                        "Supabase resumable upload creation failed"
                    ) from error
            time.sleep(0.5 * (2**attempt))
        if not isinstance(location, str) or not location:
            raise SupabasePublicationError("Supabase resumable upload URL is missing")
        upload_url = urllib.parse.urljoin(endpoint, location)
        parsed_upload = urllib.parse.urlsplit(upload_url)
        if (
            parsed_upload.scheme != "https"
            or parsed_upload.hostname != f"{project_ref}.storage.supabase.co"
            or parsed_upload.port not in {None, 443}
            or parsed_upload.username is not None
            or parsed_upload.password is not None
            or not parsed_upload.path.startswith("/storage/v1/upload/resumable/")
        ):
            raise SupabasePublicationError("Supabase resumable upload URL is invalid")

        offset = 0
        while offset < len(raw):
            chunk = raw[offset : offset + RESUMABLE_UPLOAD_CHUNK_BYTES]
            for attempt in range(MAX_STORAGE_UPLOAD_ATTEMPTS):
                request = urllib.request.Request(
                    upload_url,
                    data=chunk,
                    method="PATCH",
                    headers={
                        "apikey": self._api_key,
                        "Tus-Resumable": "1.0.0",
                        "Upload-Offset": str(offset),
                        "Content-Type": "application/offset+octet-stream",
                    },
                )
                try:
                    with self._opener.open(
                        request, timeout=REQUEST_TIMEOUT_SECONDS
                    ) as response:
                        next_offset = response.headers.get("Upload-Offset")
                except Exception as error:  # noqa: BLE001
                    retryable = isinstance(error, (urllib.error.URLError, TimeoutError))
                    if isinstance(error, urllib.error.HTTPError):
                        retryable = error.code in RETRYABLE_STORAGE_HTTP_CODES
                    if not retryable or attempt + 1 >= MAX_STORAGE_UPLOAD_ATTEMPTS:
                        raise SupabasePublicationError(
                            "Supabase resumable upload chunk failed"
                        ) from error
                    time.sleep(0.5 * (2**attempt))
                    offset = self._resumable_upload_offset(upload_url)
                    if offset >= len(raw):
                        break
                    chunk = raw[offset : offset + RESUMABLE_UPLOAD_CHUNK_BYTES]
                    continue
                expected_offset = offset + len(chunk)
                if next_offset != str(expected_offset):
                    raise SupabasePublicationError(
                        "Supabase resumable upload offset is invalid"
                    )
                offset = expected_offset
                break
            else:
                raise AssertionError("resumable upload retry loop did not finish")
        self._verify_existing_storage_object(storage_path, raw, custom_metadata)

    def _resumable_upload_offset(self, upload_url: str) -> int:
        request = urllib.request.Request(
            upload_url,
            method="HEAD",
            headers={
                "apikey": self._api_key,
                "Tus-Resumable": "1.0.0",
            },
        )
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                offset = response.headers.get("Upload-Offset")
        except Exception as error:  # noqa: BLE001
            raise SupabasePublicationError(
                "Supabase resumable upload offset read failed"
            ) from error
        if not isinstance(offset, str) or not offset.isdigit():
            raise SupabasePublicationError("Supabase resumable upload offset is invalid")
        return int(offset)

    def _verify_existing_storage_object(
        self,
        storage_path: str,
        raw: bytes,
        custom_metadata: dict[str, Any],
    ) -> None:
        existing_metadata = self.storage_object_metadata(storage_path)
        existing_custom = existing_metadata.get("metadata")
        if isinstance(existing_custom, str):
            try:
                existing_custom = json.loads(existing_custom)
            except json.JSONDecodeError:
                existing_custom = None
        if (
            int(existing_metadata.get("size") or -1) != len(raw)
            or not isinstance(existing_custom, dict)
            or existing_custom != custom_metadata
        ):
            raise SupabasePublicationError("existing Supabase Storage metadata is different")
        existing = self.storage_object(storage_path)
        if len(existing) != len(raw) or _sha256(existing) != _sha256(raw):
            raise SupabasePublicationError("existing Supabase Storage object has different content")

    def delete_storage_objects(self, storage_paths: list[str]) -> None:
        prefixes = [_retention_storage_path(path) for path in storage_paths]
        if not prefixes:
            return
        body = json.dumps({"prefixes": prefixes}, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self.project_url}/storage/v1/object/scryglass-public",
            data=body,
            method="DELETE",
            headers={
                "apikey": self._api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                response.read()
        except Exception as error:  # noqa: BLE001
            raise SupabasePublicationError("Supabase Storage retention delete failed") from error

    def ack_storage_cleanup(self, storage_paths: list[str]) -> int:
        paths = [_retention_storage_path(path) for path in storage_paths]
        result = self._request(
            "POST",
            "rpc/ack_scryglass_storage_cleanup",
            {"p_storage_paths": paths},
        )
        if type(result) is not int:
            raise SupabasePublicationError(
                "Supabase Storage cleanup acknowledgement is malformed"
            )
        return result

    def discard_staging_release(self, release_id: str) -> int:
        """Remove one failed staging release and drain its Storage inventory.

        The database RPC queues exact Storage paths before deleting the rows.
        Storage deletion is retried independently, so a transient Storage
        failure cannot leave the database release rows behind.
        """

        release_id = _release_id(release_id)
        result = self._request(
            "POST",
            "rpc/discard_scryglass_staging_release",
            {"p_release_id": release_id},
        )
        if not isinstance(result, dict) or result.get("release_id") != release_id:
            raise SupabasePublicationError(
                "Supabase staging discard response is malformed"
            )
        status = result.get("status")
        if status not in {"discarded", "absent"}:
            raise SupabasePublicationError(
                "Supabase staging discard status is malformed"
            )
        storage_paths = result.get("storage_paths")
        if not isinstance(storage_paths, list) or any(
            not isinstance(path, str) for path in storage_paths
        ):
            raise SupabasePublicationError(
                "Supabase staging discard Storage inventory is malformed"
            )
        if storage_paths:
            self.delete_storage_objects(storage_paths)
            self.ack_storage_cleanup(storage_paths)
        return len(storage_paths)

    def drain_staging_cleanup(self, release_id: str) -> int:
        """Drain old cleanup paths without touching a live staging release."""

        release_id = _release_id(release_id)
        result = self._request(
            "POST",
            "rpc/drain_scryglass_staging_cleanup",
            {"p_release_id": release_id},
        )
        if not isinstance(result, dict) or result.get("release_id") != release_id:
            raise SupabasePublicationError(
                "Supabase staging cleanup response is malformed"
            )
        if result.get("status") == "staging":
            raise SupabasePublicationError(
                "Supabase release became staging while cleanup was checked"
            )
        if result.get("status") != "ready":
            raise SupabasePublicationError(
                "Supabase staging cleanup status is malformed"
            )
        storage_paths = result.get("storage_paths")
        if not isinstance(storage_paths, list) or any(
            not isinstance(path, str) for path in storage_paths
        ):
            raise SupabasePublicationError(
                "Supabase staging cleanup inventory is malformed"
            )
        if storage_paths:
            self.delete_storage_objects(storage_paths)
            self.ack_storage_cleanup(storage_paths)
        return len(storage_paths)

    def discard_stale_staging_releases(
        self,
        *,
        min_age_minutes: int = 360,
        limit: int = 10,
    ) -> int:
        result = self._request(
            "POST",
            "rpc/discard_stale_scryglass_staging_releases",
            {
                "p_min_age_minutes": min_age_minutes,
                "p_limit": limit,
            },
        )
        if not isinstance(result, dict) or result.get("status") != "complete":
            raise SupabasePublicationError(
                "Supabase stale staging discard response is malformed"
            )
        release_ids = result.get("release_ids")
        storage_paths = result.get("storage_paths")
        if not isinstance(release_ids, list) or any(
            not isinstance(release_id, str) for release_id in release_ids
        ):
            raise SupabasePublicationError(
                "Supabase stale staging release inventory is malformed"
            )
        if not isinstance(storage_paths, list) or any(
            not isinstance(path, str) for path in storage_paths
        ):
            raise SupabasePublicationError(
                "Supabase stale staging Storage inventory is malformed"
            )
        if storage_paths:
            self.delete_storage_objects(storage_paths)
            self.ack_storage_cleanup(storage_paths)
        return len(release_ids)

    def stage_release(
        self,
        release: dict[str, Any],
        assets: list[dict[str, Any]],
        *,
        storage_objects: dict[str, bytes] | None = None,
    ) -> int:
        release_id = _release_id(release.get("release_id"))
        self.create_release(release)
        return self.stage_assets(
            release_id,
            assets,
            storage_objects=storage_objects,
        )

    def create_release(self, release: dict[str, Any]) -> None:
        _release_id(release.get("release_id"))
        self._request(
            "POST",
            "scryglass_public_releases",
            [release],
            prefer="return=minimal",
        )

    def stage_assets(
        self,
        release_id: str,
        assets: list[dict[str, Any]],
        *,
        storage_objects: dict[str, bytes] | None = None,
    ) -> int:
        """Add immutable assets to an existing release and reuse exact matches."""

        release_id = _release_id(release_id)
        existing = self.asset_metadata(release_id)
        reused = 0
        pending: list[dict[str, Any]] = []
        for asset in assets:
            if _release_id(asset.get("release_id")) != release_id:
                raise SupabasePublicationError("public asset release ID does not match")
            _public_asset_path(asset.get("path"))
            storage_path = asset.get("storage_path")
            if storage_path is not None:
                expected_storage_path = f"{release_id}/{asset['path']}"
                if _storage_path(storage_path) != expected_storage_path:
                    raise SupabasePublicationError("public asset Storage path does not match")
            prior = existing.get(str(asset["path"]))
            if (
                prior
                and prior.get("bytes") == asset["bytes"]
                and prior.get("sha256") == asset["sha256"]
                and prior.get("content_type") == asset["content_type"]
                and prior.get("storage_path") == asset.get("storage_path")
            ):
                reused += 1
                continue
            if prior:
                raise SupabasePublicationError(
                    f"existing public asset has different content: {asset['path']}"
                )
            pending.append(asset)
        # Insert every intended Storage asset before uploading any bytes. A
        # failed or interrupted upload therefore leaves a complete database
        # inventory for the locked staging discard path to clean up.
        for asset in pending:
            self._request(
                "POST",
                "scryglass_public_assets",
                [asset],
                prefer="return=minimal",
            )

        # Phase 1: upload all large storage objects concurrently (bounded workers).
        uploads = [
            (asset, (storage_objects or {}).get(str(asset.get("storage_path"))))
            for asset in pending
            if isinstance(asset.get("storage_path"), str)
        ]
        for asset, raw in uploads:
            if raw is None:
                raise SupabasePublicationError("large public asset has no storage payload")
        if uploads:
            with ThreadPoolExecutor(max_workers=min(6, len(uploads))) as executor:
                list(executor.map(
                    lambda item: self.put_storage_object(
                        str(item[0]["storage_path"]),
                        item[1],
                        sha256=str(item[0]["sha256"]),
                        content_type=str(item[0]["content_type"]),
                    ),
                    uploads,
                ))
        return reused

    def stage_query_datasets(
        self,
        release_id: str,
        datasets: dict[str, list[dict[str, Any]]],
        receipts: dict[str, Any],
    ) -> int:
        """Insert immutable bounded rows, then seal each dataset receipt."""

        release_id = _release_id(release_id)
        reused = 0
        existing = self.query_receipts(release_id)
        for dataset in (*QUERY_DATASETS, *TIER_QUERY_DATASETS):
            rows = datasets.get(dataset)
            receipt = receipts.get(dataset)
            if not isinstance(rows, list) or not isinstance(receipt, dict):
                raise SupabasePublicationError(f"query dataset is missing: {dataset}")
            prior = existing.get(dataset)
            if prior:
                if (
                    prior.get("row_count") != receipt.get("rows")
                    or prior.get("source_bytes") != receipt.get("bytes")
                    or prior.get("source_sha256") != receipt.get("sha256")
                ):
                    raise SupabasePublicationError(
                        f"existing query receipt has different content: {dataset}"
                    )
                reused += len(rows)
                _qrdbg(
                    "dataset",
                    dataset=dataset,
                    release_id=release_id,
                    rows=len(rows),
                    disposition="reused",
                    receipt_rows=receipt.get("rows"),
                    receipt_bytes=receipt.get("bytes"),
                )
                continue
            _qrdbg(
                "dataset",
                dataset=dataset,
                release_id=release_id,
                rows=len(rows),
                disposition="pending",
                receipt_rows=receipt.get("rows"),
                receipt_bytes=receipt.get("bytes"),
            )
            pending: list[dict[str, Any]] = []
            pending_bytes = 0
            pending_keys: list[str] = []
            batch_index = 0
            batch_start = 0
            for row_ordinal, source in enumerate(rows):
                unknown = set(source) - QUERY_ROW_FIELDS
                if unknown:
                    raise SupabasePublicationError(
                        f"query row has unsupported fields: {dataset}"
                    )
                source_without_digest = {
                    key: value for key, value in source.items() if key != "row_sha256"
                }
                row = {
                    "release_id": release_id,
                    "dataset": dataset,
                    "row_sha256": source.get("row_sha256"),
                    "source_json": canonical_query_bytes(source_without_digest).decode("utf-8"),
                    "payload_json": canonical_query_bytes(source.get("payload")).decode("utf-8"),
                }
                row_bytes = len(canonical_query_bytes(row))
                if pending and (
                    len(pending) >= QUERY_STAGE_BATCH_ROWS
                    or pending_bytes + row_bytes > QUERY_STAGE_BATCH_BYTES
                ):
                    _qrdbg(
                        "batch_boundary",
                        dataset=dataset,
                        batch_index=batch_index,
                        first_row_ordinal=batch_start,
                        last_row_ordinal=row_ordinal - 1,
                        rows=len(pending),
                        bytes=pending_bytes,
                    )
                    self._stage_query_rows(
                        release_id,
                        dataset,
                        pending,
                        batch_index=batch_index,
                        row_keys=pending_keys,
                        first_row_ordinal=batch_start,
                    )
                    batch_index += 1
                    batch_start = row_ordinal
                    pending = []
                    pending_bytes = 0
                    pending_keys = []
                pending.append(row)
                pending_bytes += row_bytes
                pending_keys.append(str(source.get("row_key")))
            if pending:
                _qrdbg(
                    "batch_boundary",
                    dataset=dataset,
                    batch_index=batch_index,
                    first_row_ordinal=batch_start,
                    last_row_ordinal=len(rows) - 1,
                    rows=len(pending),
                    bytes=pending_bytes,
                    final=True,
                )
                self._stage_query_rows(
                    release_id,
                    dataset,
                    pending,
                    batch_index=batch_index,
                    row_keys=pending_keys,
                    first_row_ordinal=batch_start,
                )
            sealed = self._request(
                "POST",
                "rpc/seal_scryglass_query_dataset",
                {
                    "p_release_id": release_id,
                    "p_dataset": dataset,
                    "p_source_rows": receipt.get("rows"),
                    "p_source_bytes": receipt.get("bytes"),
                    "p_source_sha256": receipt.get("sha256"),
                    "p_row_digest_sha256": receipt.get("row_digest_sha256"),
                },
            )
            if not isinstance(sealed, dict) or sealed.get("dataset") != dataset:
                raise SupabasePublicationError(
                    f"Supabase query receipt seal failed: {dataset}"
                )
        return reused

    def _stage_query_rows(
        self,
        release_id: str,
        dataset: str,
        rows: list[dict[str, Any]],
        *,
        batch_index: int = 0,
        row_keys: list[str] | None = None,
        first_row_ordinal: int = 0,
    ) -> int:
        """Retry one idempotent query-row batch after transient transport faults."""

        payload = {
            "p_release_id": release_id,
            "p_dataset": dataset,
            "p_rows": rows,
        }
        if _qrdbg_path():
            _qrdbg(
                "batch",
                dataset=dataset,
                release_id=release_id,
                batch_index=batch_index,
                first_row_ordinal=first_row_ordinal,
                rows=len(rows),
                payload_bytes=len(
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ),
                row_keys=list(row_keys or []),
            )
        for attempt in range(MAX_QUERY_STAGE_ATTEMPTS):
            try:
                staged = self._request(
                    "POST",
                    "rpc/stage_scryglass_query_rows",
                    payload,
                )
            except SupabasePublicationError as error:
                cause = error.__cause__
                retryable = isinstance(cause, (urllib.error.URLError, TimeoutError))
                if isinstance(cause, urllib.error.HTTPError):
                    retryable = cause.code in RETRYABLE_QUERY_STAGE_HTTP_CODES
                if not retryable or attempt + 1 >= MAX_QUERY_STAGE_ATTEMPTS:
                    if isinstance(cause, urllib.error.HTTPError) and cause.code == 400:
                        self._qrdbg_failing_batch(
                            release_id,
                            dataset,
                            rows,
                            batch_index=batch_index,
                            first_row_ordinal=first_row_ordinal,
                            row_keys=row_keys,
                            error=error,
                        )
                    raise
                time.sleep(0.5 * (2**attempt))
                continue
            if type(staged) is not int:
                raise SupabasePublicationError(
                    "Supabase query row staging is malformed"
                )
            return staged
        raise AssertionError("query staging retry loop did not return")

    def _qrdbg_failing_batch(
        self,
        release_id: str,
        dataset: str,
        rows: list[dict[str, Any]],
        *,
        batch_index: int,
        first_row_ordinal: int,
        row_keys: list[str] | None,
        error: Exception,
    ) -> None:
        """TEMPORARY probe: dump one rejected batch, then bisect it row by row.

        Inert unless SCRYGLASS_QRDBG_PATH is set. Reads only; the single-row
        replays use exactly the payload the batch already submitted, and the
        original error is always re-raised by the caller.
        """

        path = _qrdbg_path()
        if not path:
            return
        try:
            _qrdbg(
                "batch_rejected",
                dataset=dataset,
                release_id=release_id,
                batch_index=batch_index,
                first_row_ordinal=first_row_ordinal,
                rows=len(rows),
                error=str(error),
                row_keys=list(row_keys or []),
            )
            dump = {
                "release_id": release_id,
                "dataset": dataset,
                "batch_index": batch_index,
                "first_row_ordinal": first_row_ordinal,
                "rows": len(rows),
                "error": str(error),
                "row_dumps": [
                    {
                        "index_in_batch": index,
                        "row_ordinal": first_row_ordinal + index,
                        "row": row,
                        "diagnosis": _qrdbg_row_diagnosis(row),
                    }
                    for index, row in enumerate(rows)
                ],
            }
            with open(f"{path}.failing-batch.json", "w", encoding="utf-8") as handle:
                json.dump(dump, handle, ensure_ascii=False, default=str)
                handle.flush()
        except Exception as dump_error:  # noqa: BLE001 - probe must not mask the fault
            _qrdbg("batch_dump_failed", dataset=dataset, error=repr(dump_error))

        rejected = 0
        for index, row in enumerate(rows):
            if rejected >= _QRDBG_MAX_BISECT_REJECTS:
                _qrdbg(
                    "bisect_stopped",
                    dataset=dataset,
                    batch_index=batch_index,
                    reason="reject cap reached",
                    cap=_QRDBG_MAX_BISECT_REJECTS,
                    inspected=index,
                )
                break
            try:
                staged = self._request(
                    "POST",
                    "rpc/stage_scryglass_query_rows",
                    {
                        "p_release_id": release_id,
                        "p_dataset": dataset,
                        "p_rows": [row],
                    },
                )
            except Exception as row_error:  # noqa: BLE001
                rejected += 1
                _qrdbg(
                    "bisect_row",
                    dataset=dataset,
                    batch_index=batch_index,
                    index_in_batch=index,
                    row_ordinal=first_row_ordinal + index,
                    row_key=(row_keys or [None] * len(rows))[index]
                    if index < len(row_keys or [])
                    else None,
                    row_sha256=row.get("row_sha256"),
                    accepted=False,
                    error=str(row_error),
                    diagnosis=_qrdbg_row_diagnosis(row),
                )
                continue
            _qrdbg(
                "bisect_row",
                dataset=dataset,
                batch_index=batch_index,
                index_in_batch=index,
                row_ordinal=first_row_ordinal + index,
                row_key=(row_keys or [])[index] if index < len(row_keys or []) else None,
                row_sha256=row.get("row_sha256"),
                accepted=True,
                staged=staged,
            )
        _qrdbg(
            "bisect_done",
            dataset=dataset,
            batch_index=batch_index,
            rejected=rejected,
            rows=len(rows),
        )

    def activate(self, release_id: str) -> dict[str, Any]:
        release_id = _release_id(release_id)
        result = self._request(
            "POST",
            "rpc/activate_scryglass_public_release",
            {"p_release_id": release_id},
        )
        if not isinstance(result, dict) or result.get("release_id") != release_id:
            raise SupabasePublicationError("Supabase activation response is malformed")
        return result

    def restore(self, release_id: str, *, _recover: bool = True) -> dict[str, Any]:
        release_id = _rollback_release_id(release_id)
        candidate = self.release(release_id)
        manifest = candidate.get("manifest") if isinstance(candidate, dict) else None
        release_metadata = manifest.get("release") if isinstance(manifest, dict) else None
        draft_authority = manifest.get("draft_authority") if isinstance(manifest, dict) else None
        query_api = manifest.get("query_api") if isinstance(manifest, dict) else None
        raw_files = manifest.get("files") if isinstance(manifest, dict) else None
        artifact_hashes = (
            release_metadata.get("artifact_hashes")
            if isinstance(release_metadata, dict)
            else None
        )
        if (
            not isinstance(candidate, dict)
            or candidate.get("status") not in {"active", "superseded"}
            or not isinstance(manifest, dict)
            or manifest.get("pack_id") != release_id
            or not isinstance(release_metadata, dict)
            or release_metadata.get("release_id") != release_id
            or not isinstance(draft_authority, dict)
            or draft_authority.get("schema_version") != DRAFT_AUTHORITY_SCHEMA
            or draft_authority.get("status") not in {"unavailable", "descriptive"}
            or draft_authority.get("release_id") != release_id
            or not isinstance(raw_files, list)
            or not isinstance(artifact_hashes, dict)
        ):
            raise SupabasePublicationError("Supabase rollback release binding is invalid")
        inventory = self.asset_metadata(release_id)
        expected_assets: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw_files:
            if not isinstance(item, dict):
                raise SupabasePublicationError("Supabase rollback manifest inventory is invalid")
            path = _public_asset_path(item.get("path"))
            if path in seen or (
                path == DRAFT_ASSET_PATH
                and draft_authority.get("status") != "descriptive"
            ):
                raise SupabasePublicationError("Supabase rollback manifest inventory is invalid")
            seen.add(path)
            raw_bytes = item.get("bytes")
            sha256 = item.get("sha256")
            if (
                type(raw_bytes) is not int
                or raw_bytes < 0
                or not isinstance(sha256, str)
                or not SHA256_RE.fullmatch(sha256)
                or artifact_hashes.get(path) != sha256
            ):
                raise SupabasePublicationError("Supabase rollback manifest inventory is invalid")
            expected_assets.append(
                {
                    "release_id": release_id,
                    "path": path,
                    "body": None,
                    "storage_path": f"{release_id}/{path}",
                    "bytes": raw_bytes,
                    "sha256": sha256,
                    "content_type": PUBLIC_ASSET_CONTENT_TYPE,
                }
            )
        if set(inventory) != seen or set(artifact_hashes) != seen:
            raise SupabasePublicationError("Supabase rollback manifest inventory is not exact")
        _verify_release_assets(
            self,
            release_id,
            expected_assets,
            verify_storage_content=True,
        )
        _verify_query_receipts(
            self,
            release_id,
            query_api if isinstance(query_api, dict) else None,
        )
        result = self._request(
            "POST",
            "rpc/restore_scryglass_public_release",
            {"p_release_id": release_id},
        )
        if not isinstance(result, dict) or result.get("release_id") != release_id:
            raise SupabasePublicationError("Supabase rollback response is malformed")
        try:
            # The target was checked before the transition. Check the bytes
            # again after it becomes active. A superseded Storage object may
            # have changed while the transition waited for its database lock.
            _verify_active_release(
                self,
                release_id,
                expected_assets,
                verify_storage_content=True,
            )
            _verify_query_receipts(
                self,
                release_id,
                query_api if isinstance(query_api, dict) else None,
            )
        except Exception:
            replaced_release_id = result.get("replaced_release_id") or result.get(
                "previous_release_id"
            )
            if _recover and isinstance(replaced_release_id, str) and replaced_release_id:
                try:
                    self.restore(replaced_release_id, _recover=False)
                except Exception as recovery_error:
                    raise SupabasePublicationError(
                        "Supabase rollback readback failed and recovery failed"
                    ) from recovery_error
            raise
        return result

    def prune(self, keep: int = 3) -> int:
        if keep < 1 or keep > 10:
            raise SupabasePublicationError("Supabase retention must be between one and ten")
        deleted_total = 0
        for _attempt in range(MAX_RETENTION_PRUNE_CALLS):
            result = self._request(
                "POST",
                "rpc/prune_scryglass_public_releases_v2",
                {"p_keep": keep},
            )
            deleted_count = (
                result.get("deleted_count") if isinstance(result, dict) else None
            )
            if type(deleted_count) is not int or deleted_count not in (0, 1):
                raise SupabasePublicationError("Supabase retention response is malformed")
            has_more = result.get("has_more")
            if type(has_more) is not bool:
                raise SupabasePublicationError("Supabase retention response is malformed")
            storage_paths = result.get("storage_paths")
            if not isinstance(storage_paths, list) or any(
                not isinstance(path, str) for path in storage_paths
            ):
                raise SupabasePublicationError(
                    "Supabase retention Storage inventory is malformed"
                )
            self.delete_storage_objects(storage_paths)
            self.ack_storage_cleanup(storage_paths)
            deleted_total += deleted_count
            if not has_more:
                return deleted_total
        raise SupabasePublicationError(
            "Supabase retention exceeded the bounded prune calls"
        )


def _asset(path: str, raw: bytes, release_id: str) -> dict[str, Any]:
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupabasePublicationError(f"public asset is invalid JSON: {path}") from error
    return {
        "release_id": release_id,
        "path": path,
        "body": body,
        "bytes": len(raw),
        "sha256": _sha256(raw),
        "content_type": PUBLIC_ASSET_CONTENT_TYPE,
    }


def _patch_order(value: object) -> tuple[int, int]:
    parts = str(value or "").split(".", 1)
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 0, 0


def _public_patch_label(value: object) -> str:
    """Return the public Riot label for a source or client patch token."""

    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return public_patch(text)
    except PatchIdentityError:
        return text


def _public_scope_id(
    scope_id: object,
    source_patch: object,
    patch: str,
    role: object | None = None,
) -> str:
    value = str(scope_id or "").strip()
    source = str(source_patch or "").strip()
    source_prefix = f"patch:{source}" if source else ""
    public_prefix = f"patch:{patch}"
    if value == public_prefix or (source_prefix and value == source_prefix):
        value = patch
    elif value.startswith(f"{public_prefix}-"):
        value = f"{patch}-{value[len(public_prefix) + 1:]}"
    elif source_prefix and value.startswith(f"{source_prefix}-"):
        value = f"{patch}-{value[len(source_prefix) + 1:]}"
    elif value == source:
        value = patch
    elif source and value.startswith(f"{source}-"):
        value = f"{patch}-{value[len(source) + 1:]}"
    role_text = str(role or "").strip().casefold()
    if role_text and value == patch:
        return f"{patch}-{role_text}"
    return value


def _normalized_tier_payload(tier_body: dict[str, Any]) -> dict[str, Any]:
    """Normalize every published patch and role-scoped identity."""

    options = tier_body.get("options")
    patches = options.get("patches") if isinstance(options, dict) else None
    if not isinstance(patches, list) or not patches:
        raise SupabasePublicationError("tier-list asset has no patch options")
    normalized_patches = [_public_patch_label(value) for value in patches]
    rows = tier_body.get("rows")
    scopes = tier_body.get("scopes")
    if not isinstance(rows, list) or not isinstance(scopes, list):
        raise SupabasePublicationError("tier-list asset has invalid rows or scopes")
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        source_patch = item.get("patch")
        item["patch"] = _public_patch_label(source_patch)
        if item.get("scope_id") is not None:
            item["scope_id"] = _public_scope_id(
                item.get("scope_id"), source_patch, item["patch"], item.get("role")
            )
        normalized_rows.append(item)
    normalized_scopes: list[dict[str, Any]] = []
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        item = dict(scope)
        source_patch = item.get("patch")
        item["patch"] = _public_patch_label(source_patch)
        if item.get("scope_id") is not None:
            item["scope_id"] = _public_scope_id(
                item.get("scope_id"), source_patch, item["patch"], item.get("role")
            )
        normalized_scopes.append(item)
    normalized = dict(tier_body)
    normalized_options = dict(options)
    normalized_options["patches"] = sorted(set(normalized_patches), key=_patch_order)
    normalized["options"] = normalized_options
    normalized["rows"] = normalized_rows
    normalized["scopes"] = normalized_scopes
    return normalized


def latest_tier_payload(tier_body: dict[str, Any]) -> dict[str, Any]:
    """Keep the newest patch while preserving every view for that patch."""

    normalized = _normalized_tier_payload(tier_body)
    options = normalized["options"]
    normalized_patches = options["patches"]
    latest_patch = max(normalized_patches, key=_patch_order)
    latest = dict(normalized)
    latest["rows"] = [row for row in normalized["rows"] if row.get("patch") == latest_patch]
    latest["scopes"] = [scope for scope in normalized["scopes"] if scope.get("patch") == latest_patch]
    latest["latest_patch"] = latest_patch
    if not latest["rows"] or not latest["scopes"]:
        raise SupabasePublicationError("latest tier-list patch has no public data")
    return latest


def prepare_release(
    pack_dir: Path,
    manifest: dict[str, Any],
    tier_path: Path,
    *,
    project_url: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bytes]]:
    release_id = _release_id(manifest.get("pack_id"))
    draft_authority = _draft_authority(manifest, release_id)
    if not pack_dir.is_dir():
        raise SupabasePublicationError("pack directory is missing")
    manifest_entries = manifest.get("files")
    if not isinstance(manifest_entries, list) or any(
        not isinstance(item, dict) for item in manifest_entries
    ):
        raise SupabasePublicationError("manifest file inventory is malformed")
    manifest_files: dict[str, dict[str, Any]] = {}
    for item in manifest_entries:
        path = _public_asset_path(item.get("path"))
        if path in manifest_files:
            raise SupabasePublicationError(f"manifest repeats public asset: {path}")
        manifest_files[path] = item
    if manifest.get("total_files") != len(manifest_entries):
        raise SupabasePublicationError("manifest file total does not match its inventory")
    try:
        manifest_total_bytes = sum(int(item["bytes"]) for item in manifest_entries)
    except (KeyError, TypeError, ValueError) as error:
        raise SupabasePublicationError("manifest byte inventory is malformed") from error
    if manifest.get("total_bytes") != manifest_total_bytes:
        raise SupabasePublicationError("manifest byte total does not match its inventory")

    release_metadata = manifest.get("release")
    if not isinstance(release_metadata, dict):
        release_metadata = {}
    bound_release_id = release_metadata.get("release_id")
    if bound_release_id is not None and bound_release_id != release_id:
        raise SupabasePublicationError("manifest release ID does not match its pack ID")
    bound_hashes = release_metadata.get("artifact_hashes")
    if bound_hashes is not None and not isinstance(bound_hashes, dict):
        raise SupabasePublicationError("manifest artifact hash inventory is malformed")
    if isinstance(bound_hashes, dict) and not set(manifest_files).issubset(
        set(map(str, bound_hashes))
    ):
        raise SupabasePublicationError("manifest artifact hashes do not match its file inventory")

    assets: list[dict[str, Any]] = []
    storage_objects: dict[str, bytes] = {}
    for path in PUBLIC_RATING_REQUIRED_FILES:
        metadata = manifest_files.get(path)
        if not isinstance(metadata, dict):
            raise SupabasePublicationError(f"manifest is missing public asset: {path}")
        raw = _pack_asset_path(pack_dir, path).read_bytes()
        if len(raw) != metadata.get("bytes") or _sha256(raw) != metadata.get("sha256"):
            raise SupabasePublicationError(f"public asset checksum failed: {path}")
        if isinstance(bound_hashes, dict) and bound_hashes.get(path) != _sha256(raw):
            raise SupabasePublicationError(f"manifest release hash conflicts with file: {path}")
        asset = _asset(path, raw, release_id)
        storage_path = f"{release_id}/{path}"
        asset["body"] = None
        asset["storage_path"] = storage_path
        storage_objects[storage_path] = raw
        assets.append(asset)

    for path in OPTIONAL_PUBLIC_FILES:
        metadata = manifest_files.get(path)
        if not isinstance(metadata, dict):
            continue
        if path == DRAFT_ASSET_PATH:
            if draft_authority["status"] not in {"descriptive", "promoted"}:
                continue
            raw = _pack_asset_path(pack_dir, path).read_bytes()
            try:
                draft_payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SupabasePublicationError("descriptive Draft asset is invalid JSON") from error
            _validate_descriptive_draft_records(draft_payload)
        elif path == PROMOTED_DRAFT_RESULTS_PATH:
            if draft_authority["status"] != "promoted":
                continue
            raw = _pack_asset_path(pack_dir, path).read_bytes()
            try:
                promoted_payload = json.loads(raw.decode("utf-8"))
                validate_promoted_results_payload(
                    promoted_payload,
                    authority=draft_authority,
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                PromotedDraftAuthorityError,
            ) as error:
                raise SupabasePublicationError(
                    "promoted Draft result asset is invalid"
                ) from error
        else:
            raw = _pack_asset_path(pack_dir, path).read_bytes()
        if len(raw) != metadata.get("bytes") or _sha256(raw) != metadata.get("sha256"):
            raise SupabasePublicationError(f"optional public asset checksum failed: {path}")
        if isinstance(bound_hashes, dict) and bound_hashes.get(path) != _sha256(raw):
            raise SupabasePublicationError(f"manifest release hash conflicts with file: {path}")
        asset = _asset(path, raw, release_id)
        storage_path = f"{release_id}/{path}"
        asset["body"] = None
        asset["storage_path"] = storage_path
        storage_objects[storage_path] = raw
        assets.append(asset)

    if draft_authority["status"] in {"descriptive", "promoted"} and not any(
        str(asset.get("path")) == DRAFT_ASSET_PATH for asset in assets
    ):
        raise SupabasePublicationError("published Draft authority has no Draft asset")
    if draft_authority["status"] == "promoted" and not any(
        str(asset.get("path")) == PROMOTED_DRAFT_RESULTS_PATH for asset in assets
    ):
        raise SupabasePublicationError(
            "promoted Draft authority has no promoted result asset"
        )

    tier_source_raw = tier_path.read_bytes()
    try:
        tier_body = json.loads(tier_source_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupabasePublicationError("tier-list asset is invalid JSON") from error
    if not isinstance(tier_body, dict) or tier_body.get("status") != "available":
        raise SupabasePublicationError("tier-list asset is unavailable")
    tier_body = _normalized_tier_payload(tier_body)
    tier_raw = (
        json.dumps(tier_body, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    tier_asset = _asset(TIER_ASSET_PATH, tier_raw, release_id)
    _query_datasets, published_query_api = _prepare_query_datasets(
        pack_dir,
        manifest,
        tier_body,
        release_id,
    )
    tier_storage_path = f"{release_id}/{TIER_ASSET_PATH}"
    tier_asset["body"] = None
    tier_asset["storage_path"] = tier_storage_path
    assets.append(tier_asset)
    storage_objects[tier_storage_path] = tier_raw

    latest_tier_raw = (
        json.dumps(
            latest_tier_payload(tier_body),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    latest_tier_asset = _asset(TIER_LATEST_ASSET_PATH, latest_tier_raw, release_id)
    latest_tier_storage_path = f"{release_id}/{TIER_LATEST_ASSET_PATH}"
    latest_tier_asset["body"] = None
    latest_tier_asset["storage_path"] = latest_tier_storage_path
    assets.append(latest_tier_asset)
    storage_objects[latest_tier_storage_path] = latest_tier_raw

    published_manifest = dict(manifest)
    published_manifest["base_url"] = _project_url(project_url)
    published_manifest["data_backend"] = "supabase"
    published_manifest["draft_authority"] = draft_authority
    published_manifest["tier"] = {
        "status": "available",
        "as_of": tier_body.get("as_of"),
        "latest_path": TIER_LATEST_ASSET_PATH,
    }
    if published_query_api is not None:
        published_manifest["query_api"] = published_query_api
    else:
        published_manifest.pop("query_api", None)
    artifact_hashes = {
        str(asset["path"]): str(asset["sha256"])
        for asset in assets
    }
    published_manifest["release"] = {
        **release_metadata,
        "release_id": release_id,
        "tier_list_version": str(tier_body.get("schema_version") or "rankings-tierlists-v2"),
        "artifact_hashes": artifact_hashes,
    }
    public_files = [
        {
            "path": str(asset["path"]),
            "bytes": int(asset["bytes"]),
            "sha256": str(asset["sha256"]),
        }
        for asset in assets
    ]
    published_manifest["files"] = public_files
    published_manifest["total_files"] = len(public_files)
    published_manifest["total_bytes"] = sum(int(asset["bytes"]) for asset in assets)
    release = {
        "release_id": release_id,
        "status": "staging",
        "manifest": published_manifest,
        "source_as_of": (manifest.get("ratings") or {}).get("source_as_of"),
    }
    return release, assets, storage_objects


def _verify_release_assets(
    client: SupabasePublicData,
    release_id: str,
    assets: list[dict[str, Any]],
    *,
    verify_storage_content: bool,
) -> None:
    for expected in assets:
        actual = client.asset(release_id, str(expected["path"]))
        if (
            not actual
            or actual.get("bytes") != expected["bytes"]
            or actual.get("sha256") != expected["sha256"]
        ):
            raise SupabasePublicationError(
                f"Supabase asset readback failed: {expected['path']}"
            )
        storage_path = expected.get("storage_path")
        if isinstance(storage_path, str):
            metadata = client.storage_object_metadata(storage_path)
            custom = metadata.get("metadata")
            if isinstance(custom, str):
                try:
                    custom = json.loads(custom)
                except json.JSONDecodeError:
                    custom = None
            if (
                int(metadata.get("size") or -1) != expected["bytes"]
                or not isinstance(custom, dict)
                or custom.get("sha256") != expected["sha256"]
                or custom.get("bytes") != expected["bytes"]
                or custom.get("content_type") != expected["content_type"]
            ):
                raise SupabasePublicationError(
                    f"Supabase Storage readback failed: {expected['path']}"
                )
            if verify_storage_content:
                raw = client.storage_object(storage_path)
                if len(raw) != expected["bytes"] or _sha256(raw) != expected["sha256"]:
                    raise SupabasePublicationError(
                        f"Supabase Storage checksum failed: {expected['path']}"
                    )
        elif actual.get("body") != expected["body"]:
            raise SupabasePublicationError(
                f"Supabase asset readback failed: {expected['path']}"
            )


def _verify_query_receipts(
    client: SupabasePublicData,
    release_id: str,
    query_api: dict[str, Any] | None,
) -> None:
    if query_api is None:
        return
    expected = query_api.get("datasets")
    if not isinstance(expected, dict):
        raise SupabasePublicationError("query API receipt inventory is missing")
    actual = client.query_receipts(release_id)
    if set(actual) != set(expected):
        raise SupabasePublicationError("Supabase query receipt inventory is not exact")
    for dataset, receipt in expected.items():
        row = actual.get(dataset)
        if (
            not isinstance(receipt, dict)
            or not isinstance(row, dict)
            or row.get("row_count") != receipt.get("rows")
            or row.get("source_bytes") != receipt.get("bytes")
            or row.get("source_sha256") != receipt.get("sha256")
            or row.get("row_digest_sha256") != receipt.get("row_digest_sha256")
            or type(row.get("storage_bytes")) is not int
            or not isinstance(row.get("storage_sha256"), str)
            or not SHA256_RE.fullmatch(str(row.get("storage_sha256")))
        ):
            raise SupabasePublicationError(
                f"Supabase query receipt readback failed: {dataset}"
            )


def _verify_active_release(
    client: SupabasePublicData,
    release_id: str,
    assets: list[dict[str, Any]],
    *,
    verify_storage_content: bool,
) -> None:
    active = client.active_release()
    if not active or active.get("release_id") != release_id:
        raise SupabasePublicationError("Supabase active release readback failed")
    manifest = active.get("manifest")
    release_metadata = manifest.get("release") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("pack_id") != release_id
        or not isinstance(release_metadata, dict)
        or release_metadata.get("release_id") != release_id
        or release_metadata.get("artifact_hashes")
        != {str(asset["path"]): str(asset["sha256"]) for asset in assets}
    ):
        raise SupabasePublicationError("Supabase active manifest readback failed")
    _verify_release_assets(
        client,
        release_id,
        assets,
        verify_storage_content=verify_storage_content,
    )


def publish_release(
    pack_dir: Path,
    manifest: dict[str, Any],
    tier_path: Path,
    *,
    project_url: str,
    secret_key: str,
    retention: int = 3,
    client: SupabasePublicData | None = None,
) -> dict[str, Any]:
    database = client or SupabasePublicData(project_url, secret_key)
    # A stale release can contain a large query index. Remove one per
    # publication so retention stays inside the database statement budget.
    database.discard_stale_staging_releases(limit=1)
    release, assets, storage_objects = prepare_release(
        pack_dir,
        manifest,
        tier_path,
        project_url=database.project_url,
    )
    release_id = str(release["release_id"])
    try:
        raw_tier = json.loads(tier_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupabasePublicationError("tier-list asset is invalid JSON") from error
    if not isinstance(raw_tier, dict):
        raise SupabasePublicationError("tier-list asset is malformed")
    query_datasets, query_api = _prepare_query_datasets(
        pack_dir,
        manifest,
        raw_tier,
        release_id,
    )
    existing = database.release(release_id)
    if existing and existing.get("manifest") != release["manifest"]:
        raise SupabasePublicationError("existing release has a different manifest")
    if existing and existing.get("status") == "superseded":
        raise SupabasePublicationError("a superseded release ID cannot be changed")
    if existing and existing.get("status") == "active":
        _verify_active_release(
            database,
            release_id,
            assets,
            verify_storage_content=True,
        )
        _verify_query_receipts(database, release_id, query_api)
        return {
            "status": "already_active",
            "release_id": release_id,
            "previous_release_id": None,
            "assets": len(assets),
            "reused_assets": len(assets),
            "query_rows": sum(len(rows) for rows in query_datasets.values()),
            "reused_query_rows": sum(len(rows) for rows in query_datasets.values()),
            "bytes": sum(int(asset["bytes"]) for asset in assets),
            "retained_releases": retention,
        }

    if existing and existing.get("status") == "staging":
        # A staging row may belong to another worker. Leave it in place and
        # let bounded stale cleanup remove it after its retention window.
        raise SupabasePublicationError(
            "Supabase release is already being staged; retry after stale cleanup"
        )

    if existing is None:
        # A previous discard may have removed the rows before Storage cleanup
        # completed. Drain that exact queue before reusing this release ID.
        database.drain_staging_cleanup(release_id)

    staged_by_this_call = False
    try:
        if existing:
            raise SupabasePublicationError(
                "Supabase release has an unsupported existing status"
            )
        database.create_release(release)
        staged_by_this_call = True
        reused_assets = database.stage_assets(
            release_id,
            assets,
            storage_objects=storage_objects,
        )
        staged = database.release(release_id)
        if not staged or staged.get("status") != "staging":
            raise SupabasePublicationError("Supabase staged release readback failed")
        reused_query_rows = 0
        if query_api is not None:
            receipts = query_api.get("datasets")
            if not isinstance(receipts, dict):
                raise SupabasePublicationError("query API receipt inventory is missing")
            reused_query_rows = database.stage_query_datasets(
                release_id,
                query_datasets,
                receipts,
            )
            _verify_query_receipts(database, release_id, query_api)
        _verify_release_assets(
            database,
            release_id,
            assets,
            verify_storage_content=True,
        )
    except Exception:
        if staged_by_this_call:
            try:
                staged_after_failure = database.release(release_id)
                if (
                    isinstance(staged_after_failure, dict)
                    and staged_after_failure.get("status") == "staging"
                ):
                    database.discard_staging_release(release_id)
            except Exception:
                # Preserve the original publication failure. Retention will
                # retry the locked cleanup path if this best-effort cleanup
                # cannot reach the database or Storage service.
                pass
        raise
    activation = database.activate(release_id)
    try:
        _verify_active_release(
            database,
            release_id,
            assets,
            verify_storage_content=False,
        )
    except Exception:
        previous_release_id = activation.get("previous_release_id")
        if isinstance(previous_release_id, str) and previous_release_id:
            try:
                database.restore(previous_release_id)
            except Exception as rollback_error:
                raise SupabasePublicationError(
                    "active release verification and rollback failed"
                ) from rollback_error
        raise
    try:
        pruned = database.prune(retention)
    except Exception:
        previous_release_id = activation.get("previous_release_id")
        if isinstance(previous_release_id, str) and previous_release_id:
            try:
                database.restore(previous_release_id)
            except Exception as rollback_error:
                raise SupabasePublicationError(
                    "Supabase retention and rollback failed"
                ) from rollback_error
        raise
    return {
        "status": "activated_pending_health",
        "release_id": release_id,
        "previous_release_id": activation.get("previous_release_id"),
        "assets": len(assets),
        "reused_assets": reused_assets,
        "query_rows": sum(len(rows) for rows in query_datasets.values()),
        "reused_query_rows": reused_query_rows,
        "bytes": sum(int(asset["bytes"]) for asset in assets),
        "pruned_releases": pruned,
        "retained_releases": retention,
    }


def restore_release(
    release_id: str,
    *,
    project_url: str,
    secret_key: str,
) -> dict[str, Any]:
    return SupabasePublicData(project_url, secret_key).restore(release_id)
