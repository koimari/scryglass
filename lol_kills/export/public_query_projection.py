"""Build the bounded, release-bound public query projection.

The projection is an internal publication input.  It is never a downloadable
public asset.  Supabase stores its rows behind active-release RPCs so web
requests do not need to read the large profile and match collection files.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
import urllib.parse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lol_kills.etl.aliases import CHAMP_ALIASES, TEAM_ALIASES
from lol_kills.etl.competition import competition_tier
from lol_kills.v2.patch_identity import public_patch
from lol_kills.v2.tierlists.patch_mapping import normalize_oe_token

QUERY_API_SCHEMA = "scryglass:query-api:v1"
QUERY_PROJECTION_PATH = "query/public_query_v1.json"
QUERY_DATASETS = (
    "players",
    "teams",
    "player_champions",
    "games",
    "identity_games",
    "champions",
    "aliases",
)
TIER_QUERY_DATASETS = (
    "tier_rows",
    "tier_scopes",
    "tier_matrix_rows",
    "tier_similarity_champions",
    "tier_similarity_edges",
)
TIER_QUERY_DATASET = TIER_QUERY_DATASETS[0]
MAX_QUERY_ROW_BYTES = 64 * 1024
MAX_QUERY_SOURCE_ROW_BYTES = 70_000
DESCRIPTIVE_SIGNAL_SCHEMA = "scryglass:draft-descriptive-signal:v1"
DESCRIPTIVE_SUMMARY_SCHEMA = "scryglass:draft-descriptive-summary:v1"
DESCRIPTIVE_AUTHORITY_SCHEMA = "scryglass:draft-authority:v1"
_SHA256_LENGTH = 64
_DRAFT_KEYS = frozenset(
    {
        "authority_receipt_sha256",
        "best_available",
        "draft_authority",
        "draft_contribution",
        "draft_edge",
        "draft_pool",
        "draft_probability",
        "draft_score",
        "draft_win_share",
    }
)
_FORBIDDEN_DRAFT_KEYS = frozenset(
    {
        "average_win_share",
        "betting",
        "development_composite",
        "draft_probability",
        "draft_win_share",
        "elo",
        "ev",
        "expected_value",
        "fair_odds",
        "gold",
        "live_state",
        "match_probability",
        "match_win_expectation",
        "momentum",
        "mu_diff",
        "objectives",
        "odds",
        "p_blue",
        "p_red",
        "phase_curve",
        "player_elo",
        "player_rating",
        "probability",
        "r9e",
        "r9e_state_space",
        "rating_uncertainty",
        "recommendation",
        "sigma_pair",
        "strength",
        "team_elo",
        "team_rating",
        "win_probability",
    }
)
_COMPONENT_KEYS = (
    "base",
    "archetype_interactions",
    "ally_synergy",
    "enemy_counter",
    "same_role",
)
_PUBLIC_IMAGE_HOSTS = frozenset({"cdn.communitydragon.org"})


class PublicQueryProjectionError(RuntimeError):
    """The bounded public query projection is invalid."""


def normalize_public_key(value: object) -> str:
    """Return the shared case-insensitive public identity key."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    text = text.replace("’", "'").replace("`", "'")
    return " ".join(text.split())


def _public_patch_label(value: object) -> str | None:
    """Expose Riot's public 25/26.x patch label in query data.

    OE and client feeds use 15/16.x tokens. They remain valid source inputs,
    but they must not leak into public tier or profile output.
    """

    text = str(value or "").strip()
    if not text:
        return None
    try:
        return public_patch(normalize_oe_token(text))
    except (TypeError, ValueError):
        try:
            return public_patch(text)
        except (TypeError, ValueError):
            return None


def _public_scope_id(
    scope_id: str,
    raw_patch: object,
    patch: str | None,
    role: str | None = None,
) -> str:
    if not patch:
        return scope_id
    source = str(raw_patch or "").strip()
    source_prefix = f"patch:{source}" if source else ""
    public_prefix = f"patch:{patch}"
    if scope_id == public_prefix or (source_prefix and scope_id == source_prefix):
        scope_id = patch
    elif scope_id.startswith(f"{public_prefix}-"):
        scope_id = f"{patch}-{scope_id[len(public_prefix) + 1:]}"
    elif source_prefix and scope_id.startswith(f"{source_prefix}-"):
        scope_id = f"{patch}-{scope_id[len(source_prefix) + 1:]}"
    elif source and scope_id == source:
        scope_id = patch
    elif source and scope_id.startswith(f"{source}-"):
        scope_id = f"{patch}-{scope_id[len(source) + 1:]}"
    if role and scope_id == patch:
        return f"{patch}-{role}"
    return scope_id


def _source_identity_key(value: object) -> str:
    """Keep source case when it separates two accepted identities."""

    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _stable_id(kind: str, value: object) -> str:
    key = normalize_public_key(value)
    if not key:
        raise PublicQueryProjectionError(f"{kind} identity is empty")
    return hashlib.sha256(f"{kind}\0{key}".encode("utf-8")).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_query_bytes(value: object) -> bytes:
    """Return bytes used by every projection and receipt digest."""

    return _canonical_bytes(value)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _contains_draft_field(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _DRAFT_KEYS or _contains_draft_field(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_draft_field(child) for child in value)
    return False


def reject_draft_fields(value: object) -> None:
    """Reject predictive Draft fields at the publication boundary."""

    if _contains_draft_field(value):
        raise PublicQueryProjectionError("query data contains Draft fields")


def _sha256_text(value: object) -> str | None:
    text = str(value or "").strip()
    if len(text) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        return None
    return text


def _descriptive_authority(
    authority: Mapping[str, Any] | None,
    *,
    release_id: str,
) -> dict[str, Any] | None:
    """Return the exact release-bound descriptive authority or fail closed."""

    if authority is None or authority.get("status") == "unavailable":
        return None
    normalized = {
        "schema_version": authority.get("schema_version"),
        "status": authority.get("status"),
        "authority": authority.get("authority"),
        "estimand": authority.get("estimand"),
        "release_id": authority.get("release_id"),
        "model_version": str(authority.get("model_version") or "").strip(),
        "artifact_sha256": _sha256_text(authority.get("artifact_sha256")),
        "receipt_sha256": _sha256_text(
            authority.get("receipt_sha256")
            or authority.get("authority_receipt_sha256")
        ),
        "probability_authority": authority.get("probability_authority"),
        "recommendation_authority": authority.get("recommendation_authority"),
        "betting_authority": authority.get("betting_authority"),
    }
    if (
        normalized["schema_version"] != DESCRIPTIVE_AUTHORITY_SCHEMA
        or normalized["status"] != "descriptive"
        or normalized["authority"] != "descriptive"
        or normalized["estimand"] != "composition_only"
        or normalized["release_id"] != release_id
        or not normalized["model_version"]
        or len(normalized["model_version"]) > 100
        or normalized["artifact_sha256"] is None
        or normalized["receipt_sha256"] is None
        or normalized["probability_authority"] is not False
        or normalized["recommendation_authority"] is not False
        or normalized["betting_authority"] is not False
    ):
        raise PublicQueryProjectionError(
            "descriptive Draft Score authority is not release-bound"
        )
    return normalized


def _reject_forbidden_draft_content(value: object, path: str = "draft") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().casefold()
            if normalized in _FORBIDDEN_DRAFT_KEYS or normalized.startswith(
                ("r9e_", "control_", "strength_", "phase_", "live_")
            ):
                raise PublicQueryProjectionError(
                    f"descriptive Draft Score contains forbidden field: {path}.{key}"
                )
            _reject_forbidden_draft_content(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_draft_content(child, f"{path}[{index}]")


def _strict_number(value: object, *, field: str, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    number = _finite(value)
    if number is None:
        raise PublicQueryProjectionError(
            f"descriptive Draft Score field is not finite: {field}"
        )
    return round(number, 6)


def _strict_nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise PublicQueryProjectionError(
            f"descriptive Draft Score field is not an integer: {field}"
        )
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise PublicQueryProjectionError(
            f"descriptive Draft Score field is not an integer: {field}"
        ) from error
    if number < 0 or number != value:
        raise PublicQueryProjectionError(
            f"descriptive Draft Score field is not a nonnegative integer: {field}"
        )
    return number


def _strict_components(
    value: object,
    *,
    include_total: bool,
    field: str,
) -> dict[str, float]:
    expected = (*_COMPONENT_KEYS, "total") if include_total else _COMPONENT_KEYS
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise PublicQueryProjectionError(
            f"descriptive Draft Score component ledger is invalid: {field}"
        )
    return {
        key: _strict_number(value.get(key), field=f"{field}.{key}")  # type: ignore[dict-item]
        for key in expected
    }


def _sanitize_descriptive_signal(
    signal: object,
    *,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the bounded component-only subset of one descriptive signal."""

    if not isinstance(signal, Mapping):
        raise PublicQueryProjectionError("descriptive Draft Score signal is malformed")
    _reject_forbidden_draft_content(signal)
    if (
        signal.get("schema_version") != DESCRIPTIVE_SIGNAL_SCHEMA
        or signal.get("status") != "available"
        or signal.get("authority") != "descriptive"
        or signal.get("estimand") != "composition_only"
        or signal.get("model_version") != authority["model_version"]
        or signal.get("artifact_sha256") != authority["artifact_sha256"]
        or signal.get("fit_through") is not None
    ):
        raise PublicQueryProjectionError(
            "descriptive Draft Score signal does not match its authority"
        )
    sides: dict[str, dict[str, Any]] = {}
    for side in ("blue", "red"):
        value = signal.get(side)
        if not isinstance(value, Mapping):
            raise PublicQueryProjectionError(
                f"descriptive Draft Score {side} ledger is malformed"
            )
        components = _strict_components(
            value.get("components"), include_total=False, field=f"{side}.components"
        )
        side_signal = _strict_number(value.get("signal"), field=f"{side}.signal")
        if not math.isclose(side_signal, sum(components.values()), abs_tol=1e-5):
            raise PublicQueryProjectionError(
                f"descriptive Draft Score {side} total does not reconcile"
            )
        sides[side] = {
            "signal": side_signal,
            "prior_role_games": _strict_nonnegative_integer(
                value.get("prior_role_games"), field=f"{side}.prior_role_games"
            ),
            "components": components,
        }
    edge_components = _strict_components(
        signal.get("edge_components"),
        include_total=True,
        field="edge_components",
    )
    for key in _COMPONENT_KEYS:
        expected_edge = sides["blue"]["components"][key] - sides["red"]["components"][key]
        if not math.isclose(edge_components[key], expected_edge, abs_tol=1e-5):
            raise PublicQueryProjectionError(
                f"descriptive Draft Score edge component does not reconcile: {key}"
            )
    if not math.isclose(
        edge_components["total"],
        sum(edge_components[key] for key in _COMPONENT_KEYS),
        abs_tol=1e-5,
    ):
        raise PublicQueryProjectionError(
            "descriptive Draft Score total edge does not reconcile"
        )

    raw_picks = signal.get("picks")
    if not isinstance(raw_picks, list) or len(raw_picks) != 10:
        raise PublicQueryProjectionError("descriptive Draft Score needs ten picks")
    picks: list[dict[str, Any]] = []
    slots: set[tuple[str, str]] = set()
    for index, value in enumerate(raw_picks):
        if not isinstance(value, Mapping):
            raise PublicQueryProjectionError("descriptive Draft Score pick is malformed")
        side = str(value.get("side") or "").strip().title()
        role = str(value.get("role") or "").strip().casefold()
        champion = str(value.get("champion") or "").strip()
        slot = (side, role)
        if side not in {"Blue", "Red"} or role not in {"top", "jng", "mid", "bot", "sup"}:
            raise PublicQueryProjectionError("descriptive Draft Score pick slot is invalid")
        if not champion or len(champion) > 100 or slot in slots:
            raise PublicQueryProjectionError("descriptive Draft Score pick identity is invalid")
        slots.add(slot)
        evidence_status = str(value.get("evidence_status") or "")
        if evidence_status != "available":
            raise PublicQueryProjectionError(
                "descriptive Draft Score pick evidence is unavailable"
            )
        picks.append(
            {
                "side": side,
                "role": role,
                "champion": champion,
                "contribution": _strict_number(
                    value.get("contribution"), field=f"picks[{index}].contribution"
                ),
                "prior_role_games": _strict_nonnegative_integer(
                    value.get("prior_role_games"),
                    field=f"picks[{index}].prior_role_games",
                ),
                "evidence_status": evidence_status,
            }
        )
    expected_slots = {
        (side, role)
        for side in ("Blue", "Red")
        for role in ("top", "jng", "mid", "bot", "sup")
    }
    if slots != expected_slots:
        raise PublicQueryProjectionError("descriptive Draft Score pick set is incomplete")
    return {
        "schema_version": DESCRIPTIVE_SIGNAL_SCHEMA,
        "status": "available",
        "authority": "descriptive",
        "estimand": "composition_only",
        "release_id": authority["release_id"],
        "model_version": authority["model_version"],
        "artifact_sha256": authority["artifact_sha256"],
        "authority_receipt_sha256": authority["receipt_sha256"],
        "blue": sides["blue"],
        "red": sides["red"],
        "edge_components": edge_components,
        "picks": picks,
        "note": "Descriptive composition score in model units.",
    }


def _validate_signal_game_binding(
    signal: Mapping[str, Any],
    game: Mapping[str, Any],
) -> None:
    participants = game.get("players")
    if not isinstance(participants, list) or len(participants) != 10:
        raise PublicQueryProjectionError(
            "descriptive Draft Score game does not have ten players"
        )
    expected: dict[tuple[str, str], str] = {}
    champions: set[str] = set()
    for participant in participants:
        if not isinstance(participant, Mapping):
            raise PublicQueryProjectionError(
                "descriptive Draft Score game participant is malformed"
            )
        side = str(participant.get("side") or "").strip().title()
        role = str(participant.get("role") or "").strip().casefold()
        role = {"jungle": "jng", "support": "sup"}.get(role, role)
        champion = normalize_public_key(participant.get("champion"))
        slot = (side, role)
        if (
            side not in {"Blue", "Red"}
            or role not in {"top", "jng", "mid", "bot", "sup"}
            or not champion
            or slot in expected
            or champion in champions
        ):
            raise PublicQueryProjectionError(
                "descriptive Draft Score game draft is incomplete"
            )
        expected[slot] = champion
        champions.add(champion)
    actual = {
        (
            str(pick.get("side") or "").strip().title(),
            str(pick.get("role") or "").strip().casefold(),
        ): normalize_public_key(pick.get("champion"))
        for pick in signal["picks"]
        if isinstance(pick, Mapping)
    }
    if expected != actual:
        raise PublicQueryProjectionError(
            "descriptive Draft Score picks do not match the game"
        )


def _validated_pool_picks(
    pool: object,
    signal: Mapping[str, Any],
) -> dict[tuple[str, str], tuple[str, bool]] | None:
    if not isinstance(pool, Mapping):
        return None
    pool_picks = pool.get("picked")
    bans = pool.get("bans")
    if (
        pool.get("status") != "complete"
        or pool.get("evaluated_picks") != 10
        or not isinstance(pool_picks, list)
        or len(pool_picks) != 10
        or not isinstance(bans, Mapping)
        or not all(
            isinstance(bans.get(side), list) and len(bans.get(side)) == 5
            for side in ("Blue", "Red")
        )
    ):
        return None
    normalized_bans = [
        normalize_public_key(champion)
        for side in ("Blue", "Red")
        for champion in bans[side]
    ]
    if any(not champion for champion in normalized_bans) or len(set(normalized_bans)) != 10:
        return None
    signal_champions = {
        (
            str(pick.get("side") or "").strip().title(),
            str(pick.get("role") or "").strip().casefold(),
        ): normalize_public_key(pick.get("champion"))
        for pick in signal.get("picks", [])
        if isinstance(pick, Mapping)
    }
    output: dict[tuple[str, str], tuple[str, bool]] = {}
    pick_orders: set[int] = set()
    for pool_pick in pool_picks:
        if not isinstance(pool_pick, Mapping):
            return None
        side = str(pool_pick.get("side") or "").strip().title()
        role = str(pool_pick.get("role") or "").strip().casefold()
        role = {"jungle": "jng", "support": "sup"}.get(role, role)
        champion = normalize_public_key(pool_pick.get("champion"))
        best = pool_pick.get("best_available")
        order = pool_pick.get("order")
        slot = (side, role)
        if (
            best not in (True, False)
            or isinstance(order, bool)
            or not isinstance(order, int)
            or order not in range(1, 11)
            or order in pick_orders
            or slot in output
            or champion != signal_champions.get(slot)
            or champion in normalized_bans
        ):
            return None
        output[slot] = (champion, bool(best))
        pick_orders.add(order)
    return output if len(output) == 10 else None


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _movement_delta(value: Mapping[str, Any], scope: object = None) -> float | None:
    """Read a typed rank delta from either weekly-rank shape.

    Team weekly ranks store ``delta`` at the top level. Player weekly ranks
    store one rank and delta object per competition scope. Keep the complete
    nested weekly payload in the row while exposing one typed value for table
    sorting and the public movement column.
    """

    direct = _finite(
        value.get("movement")
        if value.get("movement") is not None
        else value.get("delta")
    )
    if direct is not None:
        return direct
    requested = str(scope or "").strip().casefold()
    scopes = [requested] if requested else []
    scopes.extend(scope_name for scope_name in ("all", "tier1", "tier2", "tier3") if scope_name not in scopes)
    for scope_name in scopes:
        scoped = value.get(scope_name)
        if not isinstance(scoped, Mapping):
            continue
        scoped_delta = _finite(
            scoped.get("movement")
            if scoped.get("movement") is not None
            else scoped.get("delta")
        )
        if scoped_delta is not None:
            return scoped_delta
    return None


def _public_image_url(value: object) -> str | None:
    url = str(value or "").strip()
    if not url:
        return None
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in _PUBLIC_IMAGE_HOSTS
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
        or parsed.fragment
    ):
        raise PublicQueryProjectionError("public champion image URL is not allowed")
    return url


def _integer(value: object, default: int = 0) -> int:
    number = _finite(value)
    return int(number) if number is not None else default


def _record_by_key(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        name = str(row.get(field) or "").strip()
        key = _source_identity_key(name)
        if not key:
            continue
        if key in output:
            raise PublicQueryProjectionError(f"{field} identity is duplicated: {name}")
        output[key] = row
    return output


def _mapping_by_key(rows: Mapping[str, Any]) -> dict[str, tuple[str, Any]]:
    output: dict[str, tuple[str, Any]] = {}
    for name, payload in rows.items():
        clean = str(name).strip()
        key = _source_identity_key(clean)
        if not key:
            continue
        if key in output:
            raise PublicQueryProjectionError(f"published identity is duplicated: {clean}")
        output[key] = (clean, payload)
    return output


def _search_keys(names: Iterable[str]) -> dict[str, str]:
    """Return unique search keys and disambiguate accepted name collisions."""

    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(normalize_public_key(name), []).append(name)
    output: dict[str, str] = {}
    for base_key, members in grouped.items():
        if len(members) == 1:
            output[members[0]] = base_key
            continue
        for name in sorted(members):
            suffix = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
            output[name] = f"{base_key}#{suffix}"
    return output


def _row(dataset: str, row_key: str, payload: Mapping[str, Any], **columns: Any) -> dict[str, Any]:
    if dataset not in (*QUERY_DATASETS, *TIER_QUERY_DATASETS):
        raise PublicQueryProjectionError(f"query dataset is invalid: {dataset}")
    raw = _canonical_bytes(payload)
    if len(raw) > MAX_QUERY_ROW_BYTES:
        raise PublicQueryProjectionError(
            f"query row exceeds {MAX_QUERY_ROW_BYTES} bytes: {dataset}/{row_key}"
        )
    row = {
        "row_key": row_key,
        **columns,
        "payload": dict(payload),
        "source_bytes": len(raw),
        "source_sha256": _sha256(raw),
    }
    source_raw = _canonical_bytes(row)
    if len(source_raw) > MAX_QUERY_SOURCE_ROW_BYTES:
        raise PublicQueryProjectionError(
            f"query source row exceeds {MAX_QUERY_SOURCE_ROW_BYTES} bytes: "
            f"{dataset}/{row_key}"
        )
    row["row_sha256"] = _sha256(source_raw)
    return row


def _refresh_row_digest(row: dict[str, Any]) -> None:
    """Rebind a row digest after a final typed-column join."""

    source = {key: value for key, value in row.items() if key != "row_sha256"}
    row["row_sha256"] = _sha256(_canonical_bytes(source))


def _refresh_payload_digest(row: dict[str, Any]) -> None:
    payload_raw = _canonical_bytes(row["payload"])
    if len(payload_raw) > MAX_QUERY_ROW_BYTES:
        raise PublicQueryProjectionError("query payload exceeds its byte budget")
    row["source_bytes"] = len(payload_raw)
    row["source_sha256"] = _sha256(payload_raw)
    _refresh_row_digest(row)


def _wilson_lower_bound(wins: int, games: int, z: float = 1.96) -> float | None:
    if games <= 0 or wins < 0 or wins > games:
        return None
    proportion = wins / games
    z_squared = z * z
    denominator = 1 + z_squared / games
    centre = proportion + z_squared / (2 * games)
    margin = z * math.sqrt(
        (proportion * (1 - proportion) + z_squared / (4 * games)) / games
    )
    return max(0.0, (centre - margin) / denominator)


def _player_rows(
    ratings: Sequence[Mapping[str, Any]],
    records: Mapping[str, Any],
    weekly: Mapping[str, Any],
    metadata: Mapping[str, Any],
    leaderboards: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rating_by_key = _record_by_key(ratings, "player")
    record_by_key = _mapping_by_key(records)
    metadata_by_key = _mapping_by_key(metadata)
    board_by_key = _mapping_by_key(
        leaderboards.get("players", {})
        if isinstance(leaderboards.get("players"), Mapping)
        else {}
    )
    weekly_rows = weekly.get("by_player", {}) if isinstance(weekly, Mapping) else {}
    weekly_by_key = _mapping_by_key(weekly_rows if isinstance(weekly_rows, Mapping) else {})
    names: dict[str, str] = {}
    for source in (rating_by_key, record_by_key, metadata_by_key, board_by_key, weekly_by_key):
        for key, value in source.items():
            if isinstance(value, tuple):
                names.setdefault(key, value[0])
            else:
                names.setdefault(key, str(value.get("player") or "").strip())

    rows: list[dict[str, Any]] = []
    ids: dict[str, str] = {}
    search_keys = _search_keys(names.values())
    for key in sorted(names, key=lambda value: (normalize_public_key(names[value]), names[value])):
        name = names[key]
        if not name:
            continue
        rating = dict(rating_by_key.get(key, {}))
        record = dict(record_by_key.get(key, (name, {}))[1] or {})
        player_metadata = dict(metadata_by_key.get(key, (name, {}))[1] or {})
        board = dict(board_by_key.get(key, (name, {}))[1] or {})
        movement = dict(weekly_by_key.get(key, (name, {}))[1] or {})
        movement_value = _movement_delta(movement, record.get("current_tier"))
        search_key = search_keys[name]
        player_id = hashlib.sha256(
            f"player\0{search_key}".encode("utf-8")
        ).hexdigest()
        ids[key] = player_id
        rating_value = _finite(rating.get("mu_total"))
        sigma = _finite(rating.get("sigma"))
        adjusted = (
            rating_value - max(0.0, sigma - 28.0)
            if rating_value is not None and sigma is not None
            else rating_value
        )
        games = _integer(record.get("games"), _integer(rating.get("n_maps")))
        wins = _finite(record.get("wins"))
        win_rate = _finite(record.get("wr"))
        if win_rate is None and wins is not None and games > 0:
            win_rate = wins / games
        payload = {
            "schema_version": "scryglass:player-query-row:v1",
            "player": name,
            "rating": rating,
            "record": record,
            "weekly": movement,
            "metadata": player_metadata,
            "grade_a_games": _integer(board.get("grade_a_games")),
            "grade_games": _integer(board.get("grade_games")),
            "recent_form": _finite(board.get("recent_form")),
        }
        rows.append(
            _row(
                "players",
                player_id,
                payload,
                player_id=player_id,
                name=name,
                search_key=search_key,
                role=str(record.get("primary_role") or "").strip().casefold() or None,
                team=str(
                    record.get("current_team") or rating.get("last_team") or ""
                ).strip()
                or None,
                league=str(
                    record.get("current_league")
                    or rating.get("home_league")
                    or ""
                ).strip()
                or None,
                tier=str(record.get("current_tier") or "").strip().casefold() or None,
                active=rating.get("evidence_active") == 1,
                rating=rating_value,
                adjusted_rating=adjusted,
                movement=movement_value,
                games=games,
                wins=int(wins) if wins is not None else None,
                win_rate=win_rate,
                grade_a_games=_integer(board.get("grade_a_games")),
                grade_games=_integer(board.get("grade_games")),
            )
        )
    return rows, ids


def _team_rows(
    ratings: Sequence[Mapping[str, Any]],
    records: Mapping[str, Any],
    weekly: Mapping[str, Any],
    leaderboards: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rating_by_key = _record_by_key(ratings, "team")
    record_by_key = _mapping_by_key(records)
    board_rows = leaderboards.get("teams", []) if isinstance(leaderboards, Mapping) else []
    board_by_key = _record_by_key(
        [row for row in board_rows if isinstance(row, Mapping)],
        "team",
    )
    weekly_rows = weekly.get("by_team", {}) if isinstance(weekly, Mapping) else {}
    weekly_by_key = _mapping_by_key(weekly_rows if isinstance(weekly_rows, Mapping) else {})
    names: dict[str, str] = {}
    for key, row in rating_by_key.items():
        names[key] = str(row.get("team") or "").strip()
    for source in (record_by_key, weekly_by_key):
        for key, value in source.items():
            names.setdefault(key, value[0])

    rows: list[dict[str, Any]] = []
    ids: dict[str, str] = {}
    search_keys = _search_keys(names.values())
    for key in sorted(names, key=lambda value: (normalize_public_key(names[value]), names[value])):
        name = names[key]
        if not name:
            continue
        rating = dict(rating_by_key.get(key, {}))
        record = dict(record_by_key.get(key, (name, {}))[1] or {})
        movement = dict(weekly_by_key.get(key, (name, {}))[1] or {})
        movement_value = _finite(
            movement.get("movement")
            if movement.get("movement") is not None
            else movement.get("delta")
        )
        board = dict(board_by_key.get(key, {}))
        search_key = search_keys[name]
        team_id = hashlib.sha256(
            f"team\0{search_key}".encode("utf-8")
        ).hexdigest()
        ids[key] = team_id
        rating_value = _finite(rating.get("mu_total"))
        sigma = _finite(rating.get("sigma"))
        rating_p10 = _finite(rating.get("rating_p10"))
        adjusted = rating_p10
        if adjusted is None and rating_value is not None and sigma is not None:
            adjusted = rating_value - max(0.0, sigma - 25.0)
        games = _integer(record.get("games"), _integer(rating.get("n_maps")))
        wins = _finite(record.get("wins"))
        win_rate = _finite(record.get("wr"))
        if win_rate is None and wins is not None and games > 0:
            win_rate = wins / games
        payload = {
            "schema_version": "scryglass:team-query-row:v1",
            "team": name,
            "rating": rating,
            "record": record,
            "weekly": movement,
            "recent": board.get("recent", []),
        }
        rows.append(
            _row(
                "teams",
                team_id,
                payload,
                team_id=team_id,
                name=name,
                search_key=search_key,
                league=str(
                    record.get("current_league")
                    or rating.get("home_league")
                    or ""
                ).strip()
                or None,
                tier=str(record.get("current_tier") or "").strip().casefold() or None,
                active=rating.get("evidence_active") == 1,
                rating=rating_value,
                adjusted_rating=adjusted,
                movement=movement_value,
                games=games,
                wins=int(wins) if wins is not None else None,
                win_rate=win_rate,
            )
        )
    return rows, ids


def _champion_rows(
    records: Mapping[str, Any],
    player_ids: Mapping[str, str],
    images: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    image_by_key = {
        normalize_public_key(name): (str(name), _public_image_url(url))
        for name, url in images.items()
        if normalize_public_key(name) and str(url).strip()
    }
    champion_names: dict[str, str] = {
        key: name
        for key, (name, _url) in image_by_key.items()
    }
    player_champions: list[dict[str, Any]] = []
    player_records = _mapping_by_key(records)
    pending: list[tuple[str, str, Mapping[str, Any]]] = []
    for player_key, (player_name, values) in player_records.items():
        player_id = player_ids.get(player_key)
        if not player_id or not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                continue
            champion = str(value.get("champion") or "").strip()
            champion_key = normalize_public_key(champion)
            if not champion_key:
                continue
            champion_names.setdefault(champion_key, champion)
            pending.append((player_name, player_id, value))

    champion_ids = {
        key: _stable_id("champion", name)
        for key, name in champion_names.items()
    }
    champions: list[dict[str, Any]] = []
    for key in sorted(champion_names):
        name = champion_names[key]
        champion_id = champion_ids[key]
        image_url = image_by_key.get(key, (name, None))[1]
        payload = {
            "schema_version": "scryglass:champion-query-row:v1",
            "champion": name,
            "image_url": image_url,
        }
        champions.append(
            _row(
                "champions",
                champion_id,
                payload,
                champion_id=champion_id,
                name=name,
                search_key=key,
                image_url=image_url,
            )
        )

    for player_name, player_id, value in pending:
        champion = str(value.get("champion") or "").strip()
        champion_key = normalize_public_key(champion)
        champion_id = champion_ids[champion_key]
        games = _integer(value.get("games"))
        wins = _integer(value.get("wins"))
        row_key = hashlib.sha256(
            f"player_champion\0{player_id}\0{champion_id}".encode("utf-8")
        ).hexdigest()
        payload = {
            "schema_version": "scryglass:player-champion-query-row:v1",
            "player": player_name,
            "champion": champion,
            "record": dict(value),
            "champion_image_url": image_by_key.get(champion_key, (champion, None))[1],
        }
        player_champions.append(
            _row(
                "player_champions",
                row_key,
                payload,
                player_id=player_id,
                champion_id=champion_id,
                champion=champion,
                champion_key=champion_key,
                games=games,
                wins=wins,
                win_rate=_finite(value.get("wr")) if games else None,
                score=_wilson_lower_bound(wins, games),
            )
        )
    player_champions.sort(key=lambda row: row["row_key"])
    return player_champions, champions, champion_ids


def _game_rows(
    games: Mapping[str, Any],
    team_ids: Mapping[str, str],
    champion_images: Mapping[str, Any] | None = None,
    draft_authority: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    image_by_key = {
        normalize_public_key(name): _public_image_url(url)
        for name, url in (champion_images or {}).items()
        if normalize_public_key(name) and str(url).strip()
    }
    for raw_game_id in sorted(games):
        raw = games[raw_game_id]
        if not isinstance(raw, Mapping):
            continue
        game = dict(raw)
        raw_signal = game.pop("draft_contribution", None)
        raw_pool = game.pop("draft_pool", None)
        for key in game:
            normalized = str(key).strip().casefold()
            if normalized in _DRAFT_KEYS:
                raise PublicQueryProjectionError(
                    f"game contains an unapproved Draft field: {key}"
                )
        if draft_authority is not None and raw_signal is not None:
            sanitized_signal = _sanitize_descriptive_signal(
                raw_signal,
                authority=draft_authority,
            )
            _validate_signal_game_binding(sanitized_signal, game)
            if _validated_pool_picks(raw_pool, sanitized_signal) is not None:
                game["draft_contribution"] = sanitized_signal
        elif raw_signal is not None:
            _reject_forbidden_draft_content(raw_signal)
        game_id = str(game.get("game_id") or raw_game_id).strip()
        if not game_id:
            continue
        blue_team = str(game.get("blue_team") or "").strip()
        red_team = str(game.get("red_team") or "").strip()
        players: list[dict[str, Any]] = []
        for raw_player in game.get("players", []):
            if not isinstance(raw_player, Mapping):
                continue
            player = dict(raw_player)
            champion = str(player.get("champion") or "").strip()
            if champion:
                player["champion_image_url"] = image_by_key.get(
                    normalize_public_key(champion)
                )
            players.append(player)
        game["players"] = players
        champions = [
            str(player.get("champion") or "").strip()
            for player in players
            if str(player.get("champion") or "").strip()
        ]
        played_at = str(game.get("date") or "").strip()
        year = _integer(played_at[:4]) if len(played_at) >= 4 else 0
        payload = {
            "schema_version": "scryglass:game-query-row:v1",
            **game,
        }
        rows.append(
            _row(
                "games",
                game_id,
                payload,
                game_id=game_id,
                played_at=played_at,
                year=year,
                league=str(game.get("league") or "").strip() or None,
                tier=str(game.get("competition_tier") or "").strip().casefold() or None,
                blue_team=blue_team,
                red_team=red_team,
                blue_team_id=team_ids.get(_source_identity_key(blue_team)),
                red_team_id=team_ids.get(_source_identity_key(red_team)),
                blue_win=_integer(game.get("blue_win")),
                champions=champions,
            )
        )
    return rows


def _draft_summary(
    *,
    authority: Mapping[str, Any],
    values: Sequence[float],
    kind: str,
    best_available_rate: float | None = None,
    ban_coverage: float | None = None,
) -> dict[str, Any] | None:
    if not values:
        return None
    average = round(sum(values) / len(values), 6)
    summary: dict[str, Any] = {
        "schema_version": DESCRIPTIVE_SUMMARY_SCHEMA,
        "authority": "descriptive",
        "estimand": "composition_only",
        "release_id": authority["release_id"],
        "model_version": authority["model_version"],
        "artifact_sha256": authority["artifact_sha256"],
        "authority_receipt_sha256": authority["receipt_sha256"],
        "games": len(values),
        "scope": "whole_archive",
    }
    if kind == "team":
        summary.update(
            {
                "draft_edge": average,
                "positive_edge_rate": round(
                    sum(value > 0 for value in values) / len(values), 6
                ),
            }
        )
    elif kind == "player":
        if best_available_rate is None or ban_coverage is None:
            return None
        summary.update(
            {
                "best_available_rate": round(best_available_rate, 6),
                "pick_contribution": average,
                "pool_definition": (
                    "Best available champion in the published unbanned role pool"
                ),
                "ban_coverage": round(ban_coverage, 6),
            }
        )
    else:
        raise PublicQueryProjectionError("descriptive Draft summary kind is invalid")
    return summary


def _attach_draft_summaries(
    *,
    players: list[dict[str, Any]],
    teams: list[dict[str, Any]],
    games: Sequence[Mapping[str, Any]],
    source_games: Mapping[str, Any],
    authority: Mapping[str, Any] | None,
) -> None:
    if authority is None:
        return
    team_values: dict[str, list[float]] = {}
    player_values: dict[str, list[float]] = {}
    player_best: dict[str, int] = {}
    player_scored_picks: dict[str, int] = {}
    for row in games:
        payload = row.get("payload")
        if not isinstance(payload, Mapping):
            continue
        signal = payload.get("draft_contribution")
        if not isinstance(signal, Mapping) or signal.get("status") != "available":
            continue
        blue = signal.get("blue")
        red = signal.get("red")
        if not isinstance(blue, Mapping) or not isinstance(red, Mapping):
            continue
        blue_signal = _finite(blue.get("signal"))
        red_signal = _finite(red.get("signal"))
        if blue_signal is None or red_signal is None:
            continue
        edge = blue_signal - red_signal
        blue_team = _source_identity_key(payload.get("blue_team"))
        red_team = _source_identity_key(payload.get("red_team"))
        if blue_team:
            team_values.setdefault(blue_team, []).append(edge)
        if red_team:
            team_values.setdefault(red_team, []).append(-edge)

    for game_id, raw_game in source_games.items():
        if not isinstance(raw_game, Mapping):
            continue
        raw_signal = raw_game.get("draft_contribution")
        if raw_signal is None:
            continue
        signal = _sanitize_descriptive_signal(raw_signal, authority=authority)
        _validate_signal_game_binding(signal, raw_game)
        pool_values = _validated_pool_picks(raw_game.get("draft_pool"), signal)
        pick_values = {
            (
                str(pick.get("side") or "").strip().title(),
                str(pick.get("role") or "").strip().casefold(),
            ): _strict_number(
                pick.get("contribution"),
                field=f"source_games.{game_id}.pick_contribution",
            )
            for pick in signal.get("picks", [])
            if isinstance(pick, Mapping)
        }
        for participant in raw_game.get("players", []):
            if not isinstance(participant, Mapping):
                continue
            player_name = _source_identity_key(participant.get("player"))
            role = str(participant.get("role") or "").strip().casefold()
            role = {"jungle": "jng", "support": "sup"}.get(role, role)
            slot = (str(participant.get("side") or "").strip().title(), role)
            contribution = pick_values.get(slot)
            if player_name and contribution is not None:
                player_scored_picks[player_name] = (
                    player_scored_picks.get(player_name, 0) + 1
                )
                pool_value = pool_values.get(slot) if pool_values is not None else None
                if (
                    pool_value is not None
                    and pool_value[0]
                    == normalize_public_key(participant.get("champion"))
                ):
                    player_values.setdefault(player_name, []).append(contribution)
                    player_best[player_name] = (
                        player_best.get(player_name, 0) + int(pool_value[1])
                    )

    for row in players:
        player_key = _source_identity_key(row.get("name"))
        pool_picks = len(player_values.get(player_key, []))
        scored_picks = player_scored_picks.get(player_key, 0)
        summary = _draft_summary(
            authority=authority,
            values=player_values.get(player_key, []),
            kind="player",
            best_available_rate=(
                player_best.get(player_key, 0) / pool_picks if pool_picks else None
            ),
            ban_coverage=(
                pool_picks / scored_picks
                if scored_picks
                else None
            ),
        )
        if summary is not None:
            row["payload"]["draft_metric"] = summary
            _refresh_payload_digest(row)
    for row in teams:
        summary = _draft_summary(
            authority=authority,
            values=team_values.get(_source_identity_key(row.get("name")), []),
            kind="team",
        )
        if summary is not None:
            row["payload"]["draft_metric"] = summary
            _refresh_payload_digest(row)


def _sanitize_draft_summary(
    value: object,
    *,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicQueryProjectionError("descriptive Draft summary is malformed")
    _reject_forbidden_draft_content(value, "draft_metric")
    common = {
        "schema_version": DESCRIPTIVE_SUMMARY_SCHEMA,
        "authority": "descriptive",
        "estimand": "composition_only",
        "release_id": authority["release_id"],
        "model_version": authority["model_version"],
        "artifact_sha256": authority["artifact_sha256"],
        "authority_receipt_sha256": authority["receipt_sha256"],
        "games": _strict_nonnegative_integer(value.get("games"), field="draft_metric.games"),
        "scope": "whole_archive",
    }
    if common["games"] <= 0:
        raise PublicQueryProjectionError("descriptive Draft summary has no evidence")
    if "draft_edge" in value:
        expected = set(common) | {"draft_edge", "positive_edge_rate"}
        if set(value) != expected:
            raise PublicQueryProjectionError("team Draft summary keys are not exact")
        positive_rate = _strict_number(
            value.get("positive_edge_rate"), field="draft_metric.positive_edge_rate"
        )
        if positive_rate < 0 or positive_rate > 1:
            raise PublicQueryProjectionError("team Draft positive-edge rate is invalid")
        return {
            **common,
            "draft_edge": _strict_number(
                value.get("draft_edge"), field="draft_metric.draft_edge"
            ),
            "positive_edge_rate": positive_rate,
        }
    expected = set(common) | {
        "best_available_rate",
        "pick_contribution",
        "pool_definition",
        "ban_coverage",
    }
    if set(value) != expected:
        raise PublicQueryProjectionError("player Draft summary keys are not exact")
    best_available_rate = _strict_number(
        value.get("best_available_rate"),
        field="draft_metric.best_available_rate",
    )
    ban_coverage = _strict_number(
        value.get("ban_coverage"), field="draft_metric.ban_coverage"
    )
    if not 0 <= best_available_rate <= 1 or not 0 <= ban_coverage <= 1:
        raise PublicQueryProjectionError("player Draft evidence rate is invalid")
    pool_definition = str(value.get("pool_definition") or "")
    if pool_definition != "Best available champion in the published unbanned role pool":
        raise PublicQueryProjectionError("player Draft pool definition is invalid")
    return {
        **common,
        "best_available_rate": best_available_rate,
        "pick_contribution": _strict_number(
            value.get("pick_contribution"),
            field="draft_metric.pick_contribution",
        ),
        "pool_definition": pool_definition,
        "ban_coverage": ban_coverage,
    }


def _validate_projected_draft_value(
    value: object,
    *,
    authority: Mapping[str, Any] | None,
    path: str = "query",
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().casefold()
            if normalized == "draft_contribution":
                if authority is None or _sanitize_descriptive_signal(
                    child, authority=authority
                ) != child:
                    raise PublicQueryProjectionError(
                        "query Draft signal is unavailable or not canonical"
                    )
                continue
            if normalized == "draft_metric":
                if authority is None or _sanitize_draft_summary(
                    child, authority=authority
                ) != child:
                    raise PublicQueryProjectionError(
                        "query Draft summary is unavailable or not canonical"
                    )
                continue
            if normalized in _DRAFT_KEYS:
                raise PublicQueryProjectionError(
                    f"query data contains an unapproved Draft field: {path}.{key}"
                )
            _validate_projected_draft_value(
                child, authority=authority, path=f"{path}.{key}"
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_projected_draft_value(
                child, authority=authority, path=f"{path}[{index}]"
            )
def _identity_game_rows(
    profile_records: Mapping[str, Any],
    player_ids: Mapping[str, str],
    team_ids: Mapping[str, str],
    available_game_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, index_name, identities in (
        ("player", "players", player_ids),
        ("team", "teams", team_ids),
    ):
        index = profile_records.get(index_name)
        if not isinstance(index, Mapping):
            continue
        for display_name, game_ids in index.items():
            identity_id = identities.get(_source_identity_key(display_name))
            if not identity_id or not isinstance(game_ids, list):
                continue
            for ordinal, game_id in enumerate(game_ids[:10]):
                clean_game_id = str(game_id).strip()
                if clean_game_id not in available_game_ids:
                    continue
                row_key = hashlib.sha256(
                    (
                        f"identity_game\0{kind}\0{identity_id}\0{ordinal}\0"
                        f"{clean_game_id}"
                    ).encode("utf-8")
                ).hexdigest()
                payload = {
                    "schema_version": "scryglass:identity-game-query-row:v1",
                    "kind": kind,
                    "identity_id": identity_id,
                    "ordinal": ordinal,
                    "game_id": clean_game_id,
                }
                rows.append(
                    _row(
                        "identity_games",
                        row_key,
                        payload,
                        kind=kind,
                        identity_id=identity_id,
                        ordinal=ordinal,
                        game_id=clean_game_id,
                    )
                )
    rows.sort(key=lambda row: row["row_key"])
    return rows


def _alias_rows(
    players: Sequence[Mapping[str, Any]],
    teams: Sequence[Mapping[str, Any]],
    champions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    canonical_members: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for kind, rows, id_field in (
        ("player", players, "player_id"),
        ("team", teams, "team_id"),
        ("champion", champions, "champion_id"),
    ):
        for row in rows:
            normalized_name = normalize_public_key(row.get("name"))
            canonical_members.setdefault((kind, normalized_name), []).append(
                (str(row[id_field]), str(row.get("search_key") or ""))
            )

    candidates: list[tuple[str, str, str]] = []
    for (kind, _normalized_name), members in canonical_members.items():
        for identity_id, search_key in members:
            candidates.append((kind, search_key, identity_id))
    for alias, name in TEAM_ALIASES.items():
        members = canonical_members.get(("team", normalize_public_key(name)), [])
        if len(members) == 1:
            candidates.append(("team", normalize_public_key(alias), members[0][0]))
    for alias, name in CHAMP_ALIASES.items():
        members = canonical_members.get(("champion", normalize_public_key(name)), [])
        if len(members) == 1:
            candidates.append(("champion", normalize_public_key(alias), members[0][0]))

    aliases: dict[tuple[str, str], str] = {}
    for kind, alias_key, identity_id in candidates:
        if not alias_key:
            continue
        prior = aliases.get((kind, alias_key))
        if prior and prior != identity_id:
            raise PublicQueryProjectionError(f"public alias is ambiguous: {kind}/{alias_key}")
        aliases[(kind, alias_key)] = identity_id

    rows: list[dict[str, Any]] = []
    for (kind, alias_key), identity_id in sorted(aliases.items()):
        row_key = hashlib.sha256(
            f"alias\0{kind}\0{alias_key}".encode("utf-8")
        ).hexdigest()
        payload = {
            "schema_version": "scryglass:alias-query-row:v1",
            "kind": kind,
            "alias_key": alias_key,
            "identity_id": identity_id,
        }
        rows.append(
            _row(
                "aliases",
                row_key,
                payload,
                kind=kind,
                alias_key=alias_key,
                identity_id=identity_id,
            )
        )
    return rows


def query_dataset_receipt(dataset: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row.get("row_key") or ""))
    raw = _canonical_bytes(ordered)
    row_digest = _sha256(
        "\n".join(
            f"{row.get('row_key')}:{row.get('row_sha256')}"
            for row in ordered
        ).encode("utf-8")
    )
    return {
        "schema_version": QUERY_API_SCHEMA,
        "dataset": dataset,
        "rows": len(ordered),
        "bytes": len(raw),
        "sha256": _sha256(raw),
        "row_digest_sha256": row_digest,
    }


def build_public_query_projection(
    *,
    release_id: str,
    player_ratings: Sequence[Mapping[str, Any]],
    team_ratings: Sequence[Mapping[str, Any]],
    player_records: Mapping[str, Any],
    team_records: Mapping[str, Any],
    player_champion_records: Mapping[str, Any],
    profile_records: Mapping[str, Any],
    archive_games: Mapping[str, Any],
    player_weekly_ranks: Mapping[str, Any],
    team_weekly_ranks: Mapping[str, Any],
    player_metadata: Mapping[str, Any],
    leaderboards: Mapping[str, Any],
    draft_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return every row needed by the bounded public read APIs."""

    authority = _descriptive_authority(draft_authority, release_id=release_id)
    profile_games = profile_records.get("games")
    if isinstance(profile_games, Mapping):
        for game_id, raw_game in profile_games.items():
            if not isinstance(raw_game, Mapping):
                continue
            signal = raw_game.get("draft_contribution")
            if signal is not None:
                if authority is None:
                    _reject_forbidden_draft_content(
                        signal, f"profile.games.{game_id}"
                    )
                else:
                    _sanitize_descriptive_signal(signal, authority=authority)
    players, player_ids = _player_rows(
        player_ratings,
        player_records,
        player_weekly_ranks,
        player_metadata,
        leaderboards,
    )
    teams, team_ids = _team_rows(
        team_ratings,
        team_records,
        team_weekly_ranks,
        leaderboards,
    )
    for player in players:
        player["team_id"] = team_ids.get(_source_identity_key(player.get("team")))
        _refresh_row_digest(player)
    images = profile_records.get("champion_images", {})
    if not isinstance(images, Mapping):
        images = {}
    player_champions, champions, _champion_ids = _champion_rows(
        player_champion_records,
        player_ids,
        images,
    )
    games = _game_rows(
        archive_games,
        team_ids,
        images,
        draft_authority=authority,
    )
    _attach_draft_summaries(
        players=players,
        teams=teams,
        games=games,
        source_games=archive_games,
        authority=authority,
    )
    game_ids = {str(row["game_id"]) for row in games}
    identity_games = _identity_game_rows(
        profile_records,
        player_ids,
        team_ids,
        game_ids,
    )
    aliases = _alias_rows(players, teams, champions)
    datasets = {
        "players": players,
        "teams": teams,
        "player_champions": player_champions,
        "games": games,
        "identity_games": identity_games,
        "champions": champions,
        "aliases": aliases,
    }
    receipts = {
        dataset: query_dataset_receipt(dataset, rows)
        for dataset, rows in datasets.items()
    }
    return {
        "schema_version": QUERY_API_SCHEMA,
        "release_id": release_id,
        "draft_authority": authority or {
            "schema_version": DESCRIPTIVE_AUTHORITY_SCHEMA,
            "status": "unavailable",
            "release_id": release_id,
        },
        "datasets": datasets,
        "receipts": receipts,
    }


def write_public_query_projection(
    projection: Mapping[str, Any],
    pack_dir: Path,
) -> dict[str, Any]:
    """Write the internal projection and return its manifest metadata."""

    if projection.get("schema_version") != QUERY_API_SCHEMA:
        raise PublicQueryProjectionError("query projection schema is invalid")
    destination = pack_dir / QUERY_PROJECTION_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_bytes(projection) + b"\n"
    destination.write_bytes(raw)
    receipts = projection.get("receipts")
    if not isinstance(receipts, Mapping):
        raise PublicQueryProjectionError("query projection receipts are missing")
    return {
        "schema_version": QUERY_API_SCHEMA,
        "status": "available",
        "projection": {
            "path": QUERY_PROJECTION_PATH,
            "bytes": len(raw),
            "sha256": _sha256(raw),
        },
        "datasets": dict(receipts),
    }


def iter_query_rows(projection: Mapping[str, Any]) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    """Yield validated query dataset rows in the stable publication order."""

    if projection.get("schema_version") != QUERY_API_SCHEMA:
        raise PublicQueryProjectionError("query projection schema is invalid")
    datasets = projection.get("datasets")
    if not isinstance(datasets, Mapping):
        raise PublicQueryProjectionError("query projection datasets are missing")
    for dataset in QUERY_DATASETS:
        rows = datasets.get(dataset)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise PublicQueryProjectionError(f"query dataset is invalid: {dataset}")
        yield dataset, rows


def validate_public_query_projection(
    projection: Mapping[str, Any],
    *,
    release_id: str,
) -> dict[str, list[dict[str, Any]]]:
    """Validate a projection before database staging."""

    if projection.get("release_id") != release_id:
        raise PublicQueryProjectionError("query projection release ID is invalid")
    authority_value = projection.get("draft_authority")
    if not isinstance(authority_value, Mapping):
        raise PublicQueryProjectionError("query Draft authority is missing")
    authority = _descriptive_authority(authority_value, release_id=release_id)
    if authority is None and authority_value != {
        "schema_version": DESCRIPTIVE_AUTHORITY_SCHEMA,
        "status": "unavailable",
        "release_id": release_id,
    }:
        raise PublicQueryProjectionError(
            "unavailable query Draft authority is malformed"
        )
    receipts = projection.get("receipts")
    if not isinstance(receipts, Mapping):
        raise PublicQueryProjectionError("query projection receipts are missing")
    datasets: dict[str, list[dict[str, Any]]] = {}
    for dataset, rows in iter_query_rows(projection):
        _validate_projected_draft_value(rows, authority=authority)
        expected = query_dataset_receipt(dataset, rows)
        if receipts.get(dataset) != expected:
            raise PublicQueryProjectionError(f"query receipt is invalid: {dataset}")
        datasets[dataset] = rows
    if set(receipts) != set(QUERY_DATASETS):
        raise PublicQueryProjectionError("query receipt inventory is not exact")
    return datasets


def build_tier_query_datasets(
    tier_body: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Normalize the public tier artifact into bounded query datasets."""

    reject_draft_fields(tier_body)
    raw_rows = tier_body.get("rows")
    if not isinstance(raw_rows, list):
        raise PublicQueryProjectionError("tier rows are invalid")
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(raw_rows):
        if not isinstance(value, Mapping):
            raise PublicQueryProjectionError("tier row is invalid")
        payload = dict(value)
        kind = str(payload.get("kind") or "champion").strip().casefold()
        name = str(
            payload.get("name")
            or payload.get("champion_name")
            or payload.get("champion")
            or payload.get("player")
            or payload.get("team")
            or ""
        ).strip()
        if not name:
            raise PublicQueryProjectionError("tier row name is empty")
        scope = payload.get("scope") if isinstance(payload.get("scope"), Mapping) else {}
        raw_patch = payload.get("patch") or scope.get("patch")
        patch = _public_patch_label(raw_patch)
        role = str(payload.get("role") or scope.get("role") or "").strip().casefold() or None
        raw_scope_id = str(payload.get("scope_id") or scope.get("scope_id") or "").strip()
        scope_id = _public_scope_id(raw_scope_id, raw_patch, patch, role)
        if patch:
            payload["patch"] = patch
        if scope_id:
            payload["scope_id"] = scope_id
        region = str(payload.get("region") or scope.get("region") or "").strip() or None
        league = str(payload.get("league") or scope.get("league") or "").strip() or None
        tier = str(payload.get("tier") or scope.get("tier") or "").strip().casefold() or None
        rank = _integer(payload.get("rank"), index + 1)
        score = _finite(
            payload.get("score")
            if payload.get("score") is not None
            else (
                payload.get("tier_value_pp")
                if payload.get("tier_value_pp") is not None
                else payload.get("tier_value")
            )
        )
        row_key = hashlib.sha256(
            (
                f"tier\0{kind}\0{normalize_public_key(name)}\0{patch}\0"
                f"{region}\0{league}\0{tier}\0{role}\0{index}"
            ).encode("utf-8")
        ).hexdigest()
        rows.append(
            _row(
                TIER_QUERY_DATASET,
                row_key,
                payload,
                champion_id=str(payload.get("champion_id") or "") or None,
                kind=kind,
                name=name,
                search_key=normalize_public_key(name),
                patch=patch,
                region=region,
                league=league,
                tier=tier,
                role=role,
                scope_id=scope_id or None,
                rank=rank,
                score=score,
                played_maps=_integer(
                    payload.get("played_maps")
                    if payload.get("played_maps") is not None
                    else payload.get("verified_appearance_count")
                ),
            )
        )
    rows.sort(key=lambda row: row["row_key"])
    base_rows = {
        (
            str(row.get("scope_id") or ""),
            str(row.get("champion_id") or ""),
        ): row
        for row in rows
        if row.get("region") is None
        and row.get("league") is None
        and row.get("tier") is None
    }

    scope_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    raw_scopes = tier_body.get("scopes")
    if not isinstance(raw_scopes, list):
        raise PublicQueryProjectionError("tier scopes are invalid")
    for source in raw_scopes:
        if not isinstance(source, Mapping):
            raise PublicQueryProjectionError("tier scope is invalid")
        scope = dict(source)
        response_matrix = scope.pop("response_matrix", None)
        scope_id = str(scope.get("scope_id") or "").strip()
        raw_scope_patch = scope.get("patch")
        patch = _public_patch_label(raw_scope_patch) or ""
        role = str(scope.get("role") or "").strip().casefold()
        scope_id = _public_scope_id(scope_id, raw_scope_patch, patch, role)
        scope["scope_id"] = scope_id
        scope["patch"] = patch
        if not scope_id or not patch or not role:
            raise PublicQueryProjectionError("tier scope identity is invalid")
        raw_regional_views = scope.get("regional_views", [])
        if not isinstance(raw_regional_views, list):
            raise PublicQueryProjectionError("tier regional views are invalid")
        regional_view_metadata: list[dict[str, Any]] = []
        tier_members: dict[str, dict[str, dict[str, Any]]] = {}
        for view in raw_regional_views:
            if not isinstance(view, Mapping):
                raise PublicQueryProjectionError("tier regional view is invalid")
            region_id = str(view.get("id") or "").strip().upper()
            regional_rows = view.get("rows")
            if (
                not region_id
                or len(region_id) > 50
                or not isinstance(regional_rows, list)
            ):
                raise PublicQueryProjectionError("tier regional view identity is invalid")
            league = region_id if region_id != "INTERNATIONAL" else None
            derived_tier = (
                "international"
                if region_id == "INTERNATIONAL"
                else competition_tier(region_id)
            )
            if derived_tier not in {"tier1", "tier2", "tier3", "international"}:
                derived_tier = None
            metadata = {
                key: value for key, value in view.items() if key != "rows"
            }
            metadata["id"] = region_id
            metadata["rows"] = []
            regional_view_metadata.append(metadata)
            for regional_index, regional in enumerate(regional_rows):
                if not isinstance(regional, Mapping):
                    raise PublicQueryProjectionError("tier regional row is invalid")
                champion_id = str(regional.get("champion_id") or "").strip()
                base = base_rows.get((scope_id, champion_id))
                if base is None:
                    raise PublicQueryProjectionError(
                        "tier regional row has no patch-wide champion"
                    )
                base_payload = base["payload"]
                if not isinstance(base_payload, Mapping):
                    raise PublicQueryProjectionError("tier base row payload is invalid")
                payload = {
                    key: base_payload.get(key)
                    for key in (
                        "scope_id", "role", "patch", "champion",
                        "champion_id", "champion_image_url", "rank_delta",
                        "movement", "tier_bucket", "tier_value_pp",
                        "counterability_status", "matchup_maps",
                        "matchup_opponents", "blind_score_pp", "counter_score",
                        "countered_opponent_count", "countered_opponent_share",
                        "expected_counter_breadth",
                    )
                    if key in base_payload
                }
                payload["scope_id"] = scope_id
                payload["patch"] = patch
                regional_rank = _integer(
                    regional.get("regional_rank"), regional_index + 1
                )
                played_maps = _integer(regional.get("played_maps"))
                score = _finite(regional.get("strength_score_pp"))
                payload.update(
                    {
                        "rank": regional_rank,
                        "played_maps": played_maps,
                        "tier_value_pp": score,
                        "region": region_id,
                        "league": league,
                        "competition_tier": derived_tier,
                        "global_rank": _integer(regional.get("global_rank")),
                        "sample_status": regional.get("sample_status"),
                    }
                )
                row_key = hashlib.sha256(
                    f"tier_region\0{scope_id}\0{region_id}\0{champion_id}".encode(
                        "utf-8"
                    )
                ).hexdigest()
                rows.append(
                    _row(
                        TIER_QUERY_DATASET,
                        row_key,
                        payload,
                        champion_id=champion_id,
                        kind="champion",
                        name=str(regional.get("champion") or base.get("name") or ""),
                        search_key=normalize_public_key(
                            regional.get("champion") or base.get("name")
                        ),
                        patch=patch,
                        region=region_id,
                        league=league,
                        tier=derived_tier,
                        role=role,
                        rank=regional_rank,
                        score=score,
                        played_maps=played_maps,
                        scope_id=scope_id,
                    )
                )
                if derived_tier:
                    member = tier_members.setdefault(derived_tier, {}).setdefault(
                        champion_id,
                        {
                            "base": base,
                            "played_maps": 0,
                            "global_rank": _integer(regional.get("global_rank")),
                            "score": score,
                        },
                    )
                    member["played_maps"] += played_maps
                    member["global_rank"] = min(
                        int(member["global_rank"] or 1_000_000),
                        _integer(regional.get("global_rank"), 1_000_000),
                    )
        scope["regional_views"] = regional_view_metadata
        for tier_name, members in tier_members.items():
            ordered_members = sorted(
                members.items(),
                key=lambda item: (
                    int(item[1]["global_rank"] or 1_000_000),
                    -float(item[1]["score"] or 0.0),
                    normalize_public_key(item[1]["base"].get("name")),
                ),
            )
            for tier_rank, (champion_id, member) in enumerate(
                ordered_members, start=1
            ):
                base = member["base"]
                base_payload = base["payload"]
                payload = {
                    key: base_payload.get(key)
                    for key in (
                        "scope_id", "role", "patch", "champion",
                        "champion_id", "champion_image_url", "rank_delta",
                        "movement", "tier_bucket", "tier_value_pp",
                        "counterability_status", "matchup_maps",
                        "matchup_opponents", "blind_score_pp", "counter_score",
                        "countered_opponent_count", "countered_opponent_share",
                        "expected_counter_breadth",
                    )
                    if key in base_payload
                }
                payload["scope_id"] = scope_id
                payload["patch"] = patch
                payload.update(
                    {
                        "rank": tier_rank,
                        "played_maps": int(member["played_maps"]),
                        "competition_tier": tier_name,
                        "global_rank": int(member["global_rank"]),
                    }
                )
                row_key = hashlib.sha256(
                    f"tier_aggregate\0{scope_id}\0{tier_name}\0{champion_id}".encode(
                        "utf-8"
                    )
                ).hexdigest()
                rows.append(
                    _row(
                        TIER_QUERY_DATASET,
                        row_key,
                        payload,
                        champion_id=champion_id,
                        kind="champion",
                        name=str(base.get("name") or ""),
                        search_key=str(base.get("search_key") or ""),
                        patch=patch,
                        region=None,
                        league=None,
                        tier=tier_name,
                        role=role,
                        rank=tier_rank,
                        score=_finite(member["score"]),
                        played_maps=int(member["played_maps"]),
                        scope_id=scope_id,
                    )
                )
        scope_rows.append(
            _row(
                "tier_scopes",
                scope_id,
                scope,
                scope_id=scope_id,
                patch=patch,
                role=role,
                region=None,
                league=None,
                tier=None,
                rank=_integer(scope.get("row_count")),
            )
        )
        if response_matrix is None:
            continue
        if not isinstance(response_matrix, Mapping):
            raise PublicQueryProjectionError("tier response matrix is invalid")
        champions = response_matrix.get("champions")
        if not isinstance(champions, list):
            raise PublicQueryProjectionError("tier response matrix champions are invalid")
        matrix_fields = (
            "edge_pp",
            "interval_low_pp",
            "interval_high_pp",
            "evidence",
            "effective_maps",
            "basis",
        )
        for index, champion in enumerate(champions):
            if not isinstance(champion, Mapping):
                raise PublicQueryProjectionError("tier response champion is invalid")
            matrix_payload: dict[str, Any] = {
                "schema_version": "scryglass:tier-matrix-row:v1",
                "scope_id": scope_id,
                "index": index,
                "champion": dict(champion),
            }
            for field in matrix_fields:
                matrix = response_matrix.get(field)
                if matrix is not None:
                    if not isinstance(matrix, list) or index >= len(matrix):
                        raise PublicQueryProjectionError("tier response matrix is ragged")
                    matrix_payload[field] = matrix[index]
            if index == 0 and isinstance(response_matrix.get("grade_thresholds_pp"), Mapping):
                matrix_payload["grade_thresholds_pp"] = dict(
                    response_matrix["grade_thresholds_pp"]
                )
            row_key = hashlib.sha256(
                f"tier_matrix\0{scope_id}\0{index}".encode("utf-8")
            ).hexdigest()
            matrix_rows.append(
                _row(
                    "tier_matrix_rows",
                    row_key,
                    matrix_payload,
                    scope_id=scope_id,
                    ordinal=index,
                    champion_id=str(champion.get("champion_id") or "") or None,
                    champion=str(champion.get("champion") or "") or None,
                )
            )

    similarity = tier_body.get("structural_similarity")
    similarity_champions: list[dict[str, Any]] = []
    similarity_edges: list[dict[str, Any]] = []
    if similarity is not None:
        if not isinstance(similarity, Mapping):
            raise PublicQueryProjectionError("tier structural similarity is invalid")
        champions = similarity.get("champions")
        matrix = similarity.get("similarity")
        if not isinstance(champions, list) or not isinstance(matrix, list):
            raise PublicQueryProjectionError("tier structural similarity matrix is invalid")
        metadata = {
            key: value
            for key, value in similarity.items()
            if key not in {"champions", "similarity"}
        }
        for index, champion in enumerate(champions):
            if not isinstance(champion, Mapping):
                raise PublicQueryProjectionError("tier structural champion is invalid")
            champion_id = str(champion.get("champion_id") or "").strip()
            if not champion_id:
                raise PublicQueryProjectionError("tier structural champion ID is empty")
            payload = dict(champion)
            if index == 0:
                payload["library"] = metadata
            similarity_champions.append(
                _row(
                    "tier_similarity_champions",
                    champion_id,
                    payload,
                    champion_id=champion_id,
                    champion=str(champion.get("champion") or "") or None,
                    ordinal=index,
                    image_url=_public_image_url(champion.get("champion_image_url")),
                )
            )
            if index >= len(matrix) or not isinstance(matrix[index], list):
                raise PublicQueryProjectionError("tier structural similarity matrix is ragged")
            if len(matrix[index]) != len(champions):
                raise PublicQueryProjectionError("tier structural similarity matrix is not square")
            for right_index, value in enumerate(matrix[index]):
                score = _finite(value)
                if score is None:
                    raise PublicQueryProjectionError("tier similarity score is invalid")
                reference_id = str(champions[right_index].get("champion_id") or "").strip()
                row_key = hashlib.sha256(
                    f"tier_similarity\0{champion_id}\0{reference_id}".encode("utf-8")
                ).hexdigest()
                similarity_edges.append(
                    _row(
                        "tier_similarity_edges",
                        row_key,
                        {
                            "schema_version": "scryglass:tier-similarity-edge:v1",
                            "champion_id": champion_id,
                            "reference_id": reference_id,
                            "similarity": score,
                        },
                        champion_id=champion_id,
                        reference_id=reference_id,
                        score=score,
                    )
                )
    rows.sort(key=lambda row: row["row_key"])
    return {
        "tier_rows": rows,
        "tier_scopes": scope_rows,
        "tier_matrix_rows": matrix_rows,
        "tier_similarity_champions": similarity_champions,
        "tier_similarity_edges": similarity_edges,
    }


def build_tier_query_rows(tier_body: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return tier rows for callers that do not need the scope datasets."""

    return build_tier_query_datasets(tier_body)[TIER_QUERY_DATASET]
