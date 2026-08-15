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
    if _contains_draft_field(payload):
        raise PublicQueryProjectionError(f"query row contains Draft fields: {dataset}/{row_key}")
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
) -> dict[str, Any]:
    """Return every row needed by the bounded public read APIs."""

    if _contains_draft_field(profile_records) or _contains_draft_field(archive_games):
        raise PublicQueryProjectionError("Draft fields must be removed before query projection")
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
    games = _game_rows(archive_games, team_ids, images)
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
    receipts = projection.get("receipts")
    if not isinstance(receipts, Mapping):
        raise PublicQueryProjectionError("query projection receipts are missing")
    datasets: dict[str, list[dict[str, Any]]] = {}
    for dataset, rows in iter_query_rows(projection):
        reject_draft_fields(rows)
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
