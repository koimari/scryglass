"""Fold-local current-rating feature ledger for future-value research.

This module replays the existing sequential team and player Elo state on one
frozen source frame.  It emits the four registered current-rating features for
every model-eligible map.  The replay is deliberately separate from the
production rating builders.  It keeps state in memory and never writes a
production artifact.

The replay scores every map in one UTC timestamp batch before it applies any
training outcome from that batch.  Validation outcomes and map/player result
metrics are masked before replay.  This gives all maps at one timestamp the
same prior state and prevents a validation row from changing a later feature.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd

from lol_kills.etl.aliases import normalize_team
from lol_kills.etl.competition import canonicalize_competition_frame
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.ratings.dual_elo import (
    DualEloConfig,
    TeamState,
    _append_momentum as _append_team_momentum,
    _is_intl,
    _momentum_residual as _team_momentum_residual,
    expected_score,
    total_mu,
)
from lol_kills.ratings.player_elo import (
    PlayerEloConfig,
    PlayerState,
    _aggregate,
    _append_momentum as _append_player_momentum,
    _lineups_by_game,
    _norm_role,
    is_team_affiliation_league,
    player_attribution_multipliers,
)
from lol_kills.research.future_value_rating import (
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    AcceptedFutureValueSource,
    FutureValueSourceError,
    _canonical_json_bytes,
    _sha256_path,
    _utc_text,
    _utc_timestamp,
    _map_model_frame,
    validate_future_value_source_receipt_payload,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SCHEMA_VERSION = "scryglass:future-value-current-rating-ledger:v2"
RECEIPT_SCHEMA_VERSION = "scryglass:future-value-current-rating-ledger-receipt:v2"
IMPLEMENTATION_LOCATOR = "lol_kills/research/future_value_rating_ledger.py"

_MAP_ID_COLUMNS = ("game_uid", "gameid", "game_id")
_PLAYER_ID_COLUMNS = ("game_uid", "gameid", "game_id")
_STRUCTURAL_MAP_COLUMNS = frozenset(
    {
        "game_uid",
        "gameid",
        "game_id",
        "date",
        "blue_team",
        "red_team",
        "blue_teamname",
        "red_teamname",
        "league",
        "league_source",
        "tournament",
        "competition_tier",
        "patch",
        "series_id",
        "seriesid",
        "match_id",
        "matchid",
    }
)
_STRUCTURAL_PLAYER_COLUMNS = frozenset(
    {
        "game_uid",
        "gameid",
        "game_id",
        "date",
        "side",
        "position",
        "playername",
        "playerid",
        "teamid",
        "teamname",
        "team",
        "champion",
        "league",
        "league_source",
        "tournament",
        "competition_tier",
        "patch",
    }
)
_EXPECTED_ROLES = {"top", "jng", "mid", "bot", "sup"}
_ROLE_ORDER = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
_RECEIPT_STATE_KEY_POLICY = {
    "team": "stable_oe_team_id",
    "player": "stable_oe_player_id",
    "display_names": "metadata_only",
    "alias_collision": "fail_closed",
}


class CurrentRatingLedgerError(FutureValueSourceError):
    """The fold-local rating replay cannot prove a safe feature ledger."""


def _as_ids(values: Iterable[Any] | Any, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        values = [values]
    try:
        raw = list(values)
    except TypeError as error:
        raise CurrentRatingLedgerError(f"{label} IDs are not iterable") from error
    result = tuple(sorted({str(gid) for value in raw if (gid := canonical_source_game_key(value))}))
    if not result:
        raise CurrentRatingLedgerError(f"{label} IDs are empty")
    if len(result) != len(raw):
        raise CurrentRatingLedgerError(f"{label} IDs contain duplicates or invalid identities")
    return result


def _game_ids(frame: pd.DataFrame, label: str) -> pd.Series:
    column = next((name for name in _MAP_ID_COLUMNS if name in frame.columns), None)
    if column is None:
        raise CurrentRatingLedgerError(f"{label} has no game identity column")
    fallback = frame["gameid"] if column == "game_uid" and "gameid" in frame.columns else None
    values = [
        canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
        for index, value in frame[column].items()
    ]
    result = pd.Series(values, index=frame.index, dtype="string")
    if result.isna().any() or result.str.strip().eq("").any():
        raise CurrentRatingLedgerError(f"{label} contains an empty game identity")
    return result


def _frame_digest(frame: pd.DataFrame, label: str) -> str:
    """Hash frame schema and values after a deterministic identity sort."""

    work = frame.copy()
    if not work.empty:
        try:
            gid = _game_ids(work, label)
        except CurrentRatingLedgerError:
            gid = pd.Series([str(index) for index in work.index], index=work.index)
        work["__canonical_game_id"] = gid.astype(str).to_numpy()
        sort_columns = ["__canonical_game_id"]
        for column in ("side", "position", "playerid", "teamid", "date"):
            if column in work.columns:
                sort_columns.append(column)
        work = work.sort_values(sort_columns, kind="mergesort")
        work = work.drop(columns=["__canonical_game_id"])
    columns = [str(column) for column in work.columns]
    header = _canonical_json_bytes(
        {
            "label": label,
            "columns": columns,
            "dtypes": [str(work[column].dtype) for column in columns],
            "rows": int(len(work)),
        }
    )
    digest = hashlib.sha256(header)
    if not work.empty:
        try:
            values = pd.util.hash_pandas_object(work[columns], index=False).to_numpy(
                dtype="uint64", copy=False
            )
            digest.update(values.tobytes())
        except (TypeError, ValueError):
            # Object columns can contain extension values that pandas cannot
            # hash.  JSON remains deterministic for the accepted OE schema.
            rows = work[columns].astype(object).where(work[columns].notna(), None).to_dict(
                orient="records"
            )
            digest.update(_canonical_json_bytes(rows))
    return digest.hexdigest()


def _artifact_digest(frame: pd.DataFrame, feature_names: Sequence[str]) -> str:
    rows: list[dict[str, Any]] = []
    ordered = frame.sort_values(["date", "game_id"], kind="mergesort")
    for row in ordered[["game_id", "date", "series_id", *feature_names]].to_dict(
        orient="records"
    ):
        item: dict[str, Any] = {
            "game_id": str(row["game_id"]),
            "date": _utc_text(row["date"]),
            "series_id": str(row["series_id"]),
        }
        for name in feature_names:
            value = float(row[name])
            if not math.isfinite(value):
                raise CurrentRatingLedgerError(f"current rating feature is non-finite: {name}")
            item[name] = value
        rows.append(item)
    return hashlib.sha256(_canonical_json_bytes(rows)).hexdigest()


def _implementation_hash() -> str:
    path = Path(__file__).resolve()
    if not path.is_file() or path.is_symlink():
        raise CurrentRatingLedgerError("current rating ledger implementation is unavailable")
    return _sha256_path(path)


def _series_ids(frame: pd.DataFrame, game_ids: pd.Series) -> pd.Series:
    """Use the evaluator's conservative partition for every gate-grade fold.

    A raw OE map row does not carry a verified series assignment.  The
    evaluator therefore uses league, tournament, and the unordered team pair
    as a conservative proxy.  A map-ID fallback would make a split series look
    safe, so this producer rejects it.
    """

    model_frame = _map_model_frame(frame)
    source = model_frame.attrs.get("series_cluster_source")
    if source != "conservative_series_superset":
        raise CurrentRatingLedgerError(
            "maps have no safe conservative series partition"
        )
    if "game_id" not in model_frame.columns or "series_id" not in model_frame.columns:
        raise CurrentRatingLedgerError("conservative series partition is incomplete")
    by_game = pd.Series(
        model_frame["series_id"].astype("string").to_numpy(),
        index=model_frame["game_id"].astype(str),
    )
    requested = game_ids.astype(str)
    if not set(requested).issubset(set(by_game.index)):
        raise CurrentRatingLedgerError(
            "conservative series partition is missing map identities"
        )
    result = requested.map(by_game)
    if result.isna().any() or result.str.strip().eq("").any():
        raise CurrentRatingLedgerError("conservative series identity is incomplete")
    return result.astype(str)


def _identity_text(value: Any) -> str:
    """Return one non-null source identity or an empty string."""

    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        return ""
    return str(value).strip()


def _validate_display_identity_mappings(
    players: pd.DataFrame,
    teams: pd.DataFrame,
    maps: pd.DataFrame | None = None,
) -> None:
    """Reject one display alias that names more than one stable entity.

    A stable ID may receive a new display name over time.  A display name may
    not resolve to two stable IDs in one replay.  State keys therefore remain
    stable when an OE display name changes, while ambiguous aliases fail closed.
    """

    player_aliases: dict[str, set[str]] = defaultdict(set)
    team_aliases: dict[str, set[str]] = defaultdict(set)
    for frame, id_column, name_column, aliases, prefix, label in (
        (players, "playerid", "playername", player_aliases, "oe:player:", "player"),
        (players, "teamid", "teamname", team_aliases, "oe:team:", "team"),
        (teams, "teamid", "teamname", team_aliases, "oe:team:", "team"),
    ):
        if id_column not in frame.columns or name_column not in frame.columns:
            continue
        for identity, display in zip(frame[id_column], frame[name_column]):
            stable = _identity_text(identity)
            alias = _identity_text(display)
            if not stable.startswith(prefix) or not alias:
                continue
            if label == "team":
                alias = normalize_team(alias).casefold()
            else:
                alias = alias.casefold()
            if alias and alias not in {"nan", "none", "<na>"}:
                aliases[alias].add(stable)
    if maps is not None:
        team_ids_by_game_side = _stable_team_ids_by_game_side(players)
        map_ids = _game_ids(maps, "maps").astype(str)
        blue_column = "blue_team" if "blue_team" in maps.columns else "blue_teamname"
        red_column = "red_team" if "red_team" in maps.columns else "red_teamname"
        for gid, row in zip(map_ids, maps.to_dict(orient="records")):
            for side, column in (("Blue", blue_column), ("Red", red_column)):
                stable = team_ids_by_game_side.get((str(gid), side), "")
                alias = _identity_text(row.get(column))
                if stable and alias:
                    team_aliases[normalize_team(alias).casefold()].add(stable)
    collisions = [
        (alias, sorted(ids))
        for alias, ids in (*player_aliases.items(), *team_aliases.items())
        if len(ids) > 1
    ]
    if collisions:
        alias, identities = sorted(collisions, key=lambda item: item[0])[0]
        raise CurrentRatingLedgerError(
            f"stable identity alias collision: {alias}: {', '.join(identities)}"
        )


def _stable_team_ids_by_game_side(players: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Map each map-side to its stable OE team ID."""

    if "teamid" not in players.columns or "side" not in players.columns:
        raise CurrentRatingLedgerError("stable team identity columns are missing")
    gids = _game_ids(players, "players").astype(str)
    work = players.copy()
    work["__gid"] = gids.to_numpy()
    work["__side"] = work["side"].astype("string").str.strip().str.title()
    result: dict[tuple[str, str], str] = {}
    for (gid, side), group in work.groupby(["__gid", "__side"], sort=False):
        values = {_identity_text(value) for value in group["teamid"]}
        values.discard("")
        if len(values) != 1:
            raise CurrentRatingLedgerError(
                f"stable team identity is not one-to-one for {gid} {side}"
            )
        result[(str(gid), str(side))] = next(iter(values))
    return result


def _stable_player_lineups(
    players: pd.DataFrame,
) -> dict[str, dict[str, list[tuple[str, str]]]]:
    """Build the exact-five lineups with stable player IDs as state keys."""

    required = {"side", "position", "playerid"}
    if not required.issubset(players.columns):
        raise CurrentRatingLedgerError(
            "stable player lineup columns are missing: "
            + ", ".join(sorted(required - set(players.columns)))
        )
    gids = _game_ids(players, "players").astype(str)
    work = players.copy()
    work["__gid"] = gids.to_numpy()
    work["__side"] = work["side"].astype("string").str.strip().str.title()
    work["__role"] = work["position"].map(_norm_role)
    work["__player_id"] = work["playerid"].map(_identity_text)
    result: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: {"Blue": [], "Red": []}
    )
    for (gid, side), group in work.groupby(["__gid", "__side"], sort=False):
        if side not in ("Blue", "Red"):
            continue
        rows = sorted(
            group[["__player_id", "__role"]].itertuples(index=False, name=None),
            key=lambda item: _ROLE_ORDER.get(str(item[1]), 9),
        )
        seen: set[str] = set()
        lineup: list[tuple[str, str]] = []
        for player_id, role in rows:
            if not player_id or role in seen:
                continue
            seen.add(str(role))
            lineup.append((str(player_id), str(role)))
        if len(lineup) != 5 or {role for _player, role in lineup} != _EXPECTED_ROLES:
            raise CurrentRatingLedgerError(
                f"stable player lineup is not exact five for {gid} {side}"
            )
        result[str(gid)][str(side)] = lineup
    return result


def _stable_lineup_hashes(players: pd.DataFrame) -> dict[str, str]:
    """Hash roster membership with stable player IDs and stable team IDs."""

    if not {"playerid", "teamid"}.issubset(players.columns):
        raise CurrentRatingLedgerError("stable lineup identity columns are missing")
    gids = _game_ids(players, "players").astype(str)
    work = players.copy()
    work["__gid"] = gids.to_numpy()
    work["__player_id"] = work["playerid"].map(_identity_text)
    work["__team_id"] = work["teamid"].map(_identity_text)
    result: dict[str, str] = {}
    for (gid, team_id), group in work.groupby(["__gid", "__team_id"], sort=False):
        player_ids = sorted(value for value in group["__player_id"] if value)
        if len(player_ids) != 5 or len(set(player_ids)) != 5:
            raise CurrentRatingLedgerError(
                f"stable roster identity is not exact five for {gid} {team_id}"
            )
        result[f"{gid}|{team_id}"] = "|".join(player_ids)
    return result


def _validate_source_frames(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    eligible_ids: Sequence[str],
    requested_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate exact map/player/team grain before the replay."""

    if not isinstance(maps, pd.DataFrame) or not isinstance(players, pd.DataFrame) or not isinstance(teams, pd.DataFrame):
        raise CurrentRatingLedgerError("maps, players, and teams must be DataFrames")
    map_ids = _game_ids(maps, "maps")
    player_ids = _game_ids(players, "players")
    team_ids = _game_ids(teams, "teams")
    if map_ids.duplicated().any():
        raise CurrentRatingLedgerError("maps contain duplicate game identities")
    eligible = set(eligible_ids)
    requested = set(requested_ids)
    if not requested.issubset(eligible):
        raise CurrentRatingLedgerError("fold IDs are outside the model-eligible census")
    if not requested.issubset(set(map_ids.astype(str))):
        raise CurrentRatingLedgerError("maps are missing fold rows")
    if "date" not in maps.columns:
        raise CurrentRatingLedgerError("maps have no date column")
    dates = pd.to_datetime(maps["date"], utc=True, errors="coerce")
    if dates.isna().any():
        raise CurrentRatingLedgerError("maps contain invalid dates")
    map_work = maps.copy()
    map_work["__game_id"] = map_ids.astype(str).to_numpy()
    map_work["date"] = dates
    player_work = players.copy()
    player_work["__game_id"] = player_ids.astype(str).to_numpy()
    team_work = teams.copy()
    team_work["__game_id"] = team_ids.astype(str).to_numpy()
    # The accepted source contains excluded maps.  Keep the model population
    # exact while allowing the frozen source frames to retain those rows.
    map_work = map_work.loc[map_work["__game_id"].isin(requested)].copy()
    player_work = player_work.loc[player_work["__game_id"].isin(requested)].copy()
    team_work = team_work.loc[team_work["__game_id"].isin(requested)].copy()
    _validate_display_identity_mappings(player_work, team_work, map_work)
    if not player_work.groupby("__game_id", sort=False).size().reindex(sorted(requested), fill_value=0).eq(10).all():
        raise CurrentRatingLedgerError("players do not contain exactly ten rows per eligible map")
    if not team_work.groupby("__game_id", sort=False).size().reindex(sorted(requested), fill_value=0).eq(2).all():
        raise CurrentRatingLedgerError("teams do not contain exactly two rows per eligible map")
    missing = {"side", "position", "playername", "playerid", "teamid"} - set(player_work.columns)
    if missing:
        raise CurrentRatingLedgerError("player identity columns are missing: " + ", ".join(sorted(missing)))
    for gid, group in player_work.groupby("__game_id", sort=False):
        names = group["playername"].astype("string").str.strip()
        ids = group["playerid"].astype("string").str.strip()
        tids = group["teamid"].astype("string").str.strip()
        if names.isna().any() or names.eq("").any() or names.str.casefold().isin({"nan", "none", "<na>"}).any() or names.nunique() != 10:
            raise CurrentRatingLedgerError(f"player names are incomplete for {gid}")
        if (
            ids.isna().any()
            or not ids.str.startswith("oe:player:").all()
            or ids.nunique() != 10
        ):
            raise CurrentRatingLedgerError(f"stable player identities are incomplete for {gid}")
        if tids.isna().any() or not tids.str.startswith("oe:team:").all():
            raise CurrentRatingLedgerError(f"stable team identities are incomplete for {gid}")
        side = group["side"].astype("string").str.strip().str.casefold()
        role = group["position"].map(_norm_role)
        if set(side) != {"blue", "red"}:
            raise CurrentRatingLedgerError(f"player side closure is invalid for {gid}")
        for value in ("blue", "red"):
            rows = group.loc[side.eq(value)]
            if len(rows) != 5 or set(role.loc[rows.index]) != _EXPECTED_ROLES:
                raise CurrentRatingLedgerError(f"exact five-player role closure is invalid for {gid} {value}")
            if rows["playerid"].astype(str).nunique() != 5 or rows["teamid"].astype(str).nunique() != 1:
                raise CurrentRatingLedgerError(f"player identity closure is invalid for {gid} {value}")
        if tids.nunique() != 2:
            raise CurrentRatingLedgerError(f"team identity closure is invalid for {gid}")
        player_team_by_side = {
            str(side_value).casefold(): str(team_id)
            for side_value, team_id in zip(side, tids)
        }
        team_rows = team_work.loc[team_work["__game_id"].eq(gid)]
        team_side = team_rows["side"].astype("string").str.strip().str.casefold()
        team_ids_by_side = {
            str(side_value).casefold(): str(team_id)
            for side_value, team_id in zip(team_side, team_rows["teamid"])
        }
        if player_team_by_side != team_ids_by_side:
            raise CurrentRatingLedgerError(f"player and team identity rows disagree for {gid}")
    if "side" not in team_work.columns or "teamid" not in team_work.columns:
        raise CurrentRatingLedgerError("team identity columns are missing")
    for gid, group in team_work.groupby("__game_id", sort=False):
        sides = group["side"].astype("string").str.strip().str.casefold()
        tids = group["teamid"].astype("string").str.strip()
        if set(sides) != {"blue", "red"} or tids.isna().any() or not tids.str.startswith("oe:team:").all() or tids.nunique() != 2:
            raise CurrentRatingLedgerError(f"team row identity closure is invalid for {gid}")
    blue_column = "blue_team" if "blue_team" in map_work.columns else "blue_teamname"
    red_column = "red_team" if "red_team" in map_work.columns else "red_teamname"
    if blue_column not in map_work.columns or red_column not in map_work.columns:
        raise CurrentRatingLedgerError("maps have incomplete team identity")
    for _, row in map_work.iterrows():
        blue = str(row.get(blue_column) or "").strip()
        red = str(row.get(red_column) or "").strip()
        if not blue or not red or normalize_team(blue).casefold() == normalize_team(red).casefold():
            raise CurrentRatingLedgerError(f"maps have incomplete team identity: {row['__game_id']}")
    return map_work, player_work, team_work


def _mask_nontraining(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    train_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mask every outcome and final metric outside the training partition."""

    map_work = maps.copy()
    player_work = players.copy()
    map_mask = ~map_work["__game_id"].astype(str).isin(train_ids)
    player_mask = ~player_work["__game_id"].astype(str).isin(train_ids)
    for column in map_work.columns:
        if column not in _STRUCTURAL_MAP_COLUMNS and column != "__game_id":
            map_work[column] = map_work[column].astype(object)
            map_work.loc[map_mask, column] = np.nan
    for column in player_work.columns:
        if column not in _STRUCTURAL_PLAYER_COLUMNS and column != "__game_id":
            player_work[column] = player_work[column].astype(object)
            player_work.loc[player_mask, column] = np.nan
    return map_work, player_work


def _team_replay(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    train_ids: set[str],
    cfg: DualEloConfig,
) -> pd.DataFrame:
    """Replay the existing Dual Elo equations with timestamp batching."""

    frame = canonicalize_competition_frame(maps.drop(columns=["__game_id"], errors="ignore")).copy()
    frame["__game_id"] = _game_ids(frame, "maps").astype(str).to_numpy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.tz_localize(None)
    frame = frame.sort_values(["date", "__game_id"], kind="mergesort").reset_index(drop=True)
    source_players = players.drop(columns=["__game_id"], errors="ignore")
    team_ids = _stable_team_ids_by_game_side(source_players)
    lineups = _stable_lineup_hashes(source_players)
    states: dict[str, TeamState] = defaultdict(lambda: TeamState(sigma=cfg.sigma0))
    rows: list[dict[str, Any]] = []
    blue_col = "blue_team" if "blue_team" in frame.columns else "blue_teamname"
    red_col = "red_team" if "red_team" in frame.columns else "red_teamname"
    for stamp, batch in frame.groupby("date", sort=False, dropna=False):
        # Prepare uncertainty and roster state once for the timestamp.  A team
        # appearing twice in one timestamp would have two simultaneous lineups
        # and cannot be assigned a single pre-map state.
        seen_teams: set[str] = set()
        for raw in batch.to_dict(orient="records"):
            gid = str(raw["__game_id"])
            blue = team_ids.get((gid, "Blue"), "")
            red = team_ids.get((gid, "Red"), "")
            if not blue or not red or blue == red:
                raise CurrentRatingLedgerError(
                    f"stable team identity is incomplete for {gid}"
                )
            for team, side in ((blue, "blue"), (red, "red")):
                if team in seen_teams:
                    raise CurrentRatingLedgerError(f"team appears in multiple maps at one timestamp: {team}")
                seen_teams.add(team)
                state = states[team]
                if pd.notna(stamp) and state.last_date is not None:
                    months = max((pd.Timestamp(stamp) - state.last_date).days / 30.0, 0.0)
                    state.sigma = min(150.0, state.sigma + cfg.sigma_month_inflate * months)
                key = f"{gid}|{team}"
                lineup_hash = lineups.get(key)
                if lineup_hash and state.lineup_hash and lineup_hash != state.lineup_hash:
                    state.sigma = min(150.0, state.sigma + cfg.roster_sigma_bump)
                if lineup_hash:
                    state.lineup_hash = lineup_hash
                states[team] = state
        pending: list[tuple[dict[str, Any], str, str, float, float, float, float]] = []
        for raw in batch.to_dict(orient="records"):
            gid = str(raw["__game_id"])
            blue = team_ids.get((gid, "Blue"), "")
            red = team_ids.get((gid, "Red"), "")
            if not blue or not red or blue == red:
                raise CurrentRatingLedgerError(
                    f"stable team identity is incomplete for {gid}"
                )
            sb, sr = states[blue], states[red]
            base_b, base_r = total_mu(sb), total_mu(sr)
            momentum_b = cfg.momentum_scale * _team_momentum_residual(sb, cfg)
            momentum_r = cfg.momentum_scale * _team_momentum_residual(sr, cfg)
            mu_b, mu_r = base_b + momentum_b, base_r + momentum_r
            sig = math.hypot(sb.sigma, sr.sigma)
            p_base = expected_score(base_b, base_r)
            p = expected_score(mu_b, mu_r)
            shrink = 1.0 / (1.0 + (sig / 120.0) ** 2)
            p_shrunk = 0.5 + (p - 0.5) * shrink
            rows.append(
                {
                    "game_id": gid,
                    "date": pd.Timestamp(raw["date"], tz="UTC"),
                    "series_id": str(raw.get("series_id") or f"map:{gid}"),
                    "base_team_logit": float(np.log(p_shrunk / (1.0 - p_shrunk))),
                    "team_rating_diff_scaled": float((mu_b - mu_r) / 400.0),
                }
            )
            y = raw.get("y_blue_win")
            if gid not in train_ids or pd.isna(y):
                continue
            y_value = float(y)
            if y_value not in (0.0, 1.0):
                raise CurrentRatingLedgerError(f"training outcome is invalid: {gid}")
            gdiff = raw.get("blue_golddiffat15")
            if pd.isna(gdiff):
                gdiff = raw.get("blue_golddiffat10")
            raw_length = raw.get("length_min")
            if pd.notna(raw_length):
                length = float(raw_length)
            elif pd.notna(raw.get("gamelength")):
                length = float(raw["gamelength"]) / 60.0
            else:
                length = 30.0
            mov = 1.0
            if pd.notna(gdiff) and length:
                mov = 1.0 + cfg.mov_scale * math.tanh(float(gdiff) / (200.0 * max(length, 1.0)))
            pending.append((raw, blue, red, p, p_base, mov, y_value))
        # Apply every training update only after all features in the batch exist.
        for raw, blue, red, p, p_base, mov, y_value in pending:
            sb, sr = states[blue], states[red]
            intl = _is_intl(str(raw.get("league") or ""), raw.get("tournament"))
            if intl:
                sb.mu_meta += cfg.k_meta * mov * (y_value - p)
                sr.mu_meta += cfg.k_meta * mov * ((1.0 - y_value) - (1.0 - p))
            else:
                sb.mu_regional += cfg.k_regional * mov * (y_value - p)
                sr.mu_regional += cfg.k_regional * mov * ((1.0 - y_value) - (1.0 - p))
            _append_team_momentum(sb, y_value - p_base, cfg)
            _append_team_momentum(sr, (1.0 - y_value) - (1.0 - p_base), cfg)
            sb.sigma = max(cfg.sigma_min, sb.sigma * 0.98)
            sr.sigma = max(cfg.sigma_min, sr.sigma * 0.98)
            if pd.notna(raw.get("date")):
                date = pd.Timestamp(raw["date"])
                sb.last_date = date
                sr.last_date = date
            states[blue], states[red] = sb, sr
    return pd.DataFrame(rows)


def _player_replay(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    train_ids: set[str],
    cfg: PlayerEloConfig,
) -> pd.DataFrame:
    """Replay the existing player Elo equations with timestamp batching."""

    source_maps = maps.drop(columns=["__game_id"], errors="ignore")
    source_players = players.drop(columns=["__game_id"], errors="ignore")
    frame = canonicalize_competition_frame(source_maps).copy()
    frame["__game_id"] = _game_ids(frame, "maps").astype(str).to_numpy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.tz_localize(None)
    frame = frame.sort_values(["date", "__game_id"], kind="mergesort").reset_index(drop=True)
    _name_lineups, attribution_metrics = _lineups_by_game(
        source_players, with_metrics=True
    )
    stable_lineups = _stable_player_lineups(source_players)
    team_ids = _stable_team_ids_by_game_side(source_players)
    attribution_by_name, _ = player_attribution_multipliers(
        attribution_metrics, cfg, baseline_cache=None
    )
    name_to_player_id: dict[tuple[str, str, str], str] = {}
    player_gids = _game_ids(source_players, "players").astype(str)
    player_work = source_players.copy()
    player_work["__gid"] = player_gids.to_numpy()
    player_work["__side"] = player_work["side"].astype("string").str.strip().str.title()
    player_work["__name"] = player_work["playername"].map(_identity_text)
    player_work["__player_id"] = player_work["playerid"].map(_identity_text)
    for row in player_work[["__gid", "__side", "__name", "__player_id"]].itertuples(
        index=False, name=None
    ):
        gid, side, name, player_id = (str(value) for value in row)
        key = (gid, side, name)
        if not name or not player_id or (key in name_to_player_id and name_to_player_id[key] != player_id):
            raise CurrentRatingLedgerError(
                f"player display identity is unstable for {gid} {side} {name}"
            )
        name_to_player_id[key] = player_id
    attribution: dict[tuple[str, str, str], float] = {}
    for (gid, side, name), multiplier in attribution_by_name.items():
        player_id = name_to_player_id.get((str(gid), str(side), str(name)))
        if not player_id:
            raise CurrentRatingLedgerError(
                f"player attribution identity is unresolved for {gid} {side} {name}"
            )
        attribution[(str(gid), str(side), player_id)] = float(multiplier)
    states: dict[str, PlayerState] = {}
    rows: list[dict[str, Any]] = []
    for stamp, batch in frame.groupby("date", sort=False, dropna=False):
        seen_players: set[str] = set()
        for raw in batch.to_dict(orient="records"):
            gid = str(raw["__game_id"])
            blue_lu = stable_lineups.get(gid, {}).get("Blue") or []
            red_lu = stable_lineups.get(gid, {}).get("Red") or []
            blue = team_ids.get((gid, "Blue"), "")
            red = team_ids.get((gid, "Red"), "")
            if not blue or not red or blue == red:
                raise CurrentRatingLedgerError(
                    f"stable team identity is incomplete for {gid}"
                )
            for name, _role in [*blue_lu, *red_lu]:
                if name in seen_players:
                    raise CurrentRatingLedgerError(f"player appears in multiple maps at one timestamp: {name}")
                seen_players.add(name)
                state = states.setdefault(name, PlayerState(sigma=cfg.sigma0))
                if pd.notna(stamp) and state.last_date is not None:
                    months = max((pd.Timestamp(stamp) - state.last_date).days / 30.0, 0.0)
                    state.sigma = min(160.0, state.sigma + cfg.sigma_month_inflate * months)
                team_now = blue if any(item[0] == name for item in blue_lu) else red
                if state.last_team and team_now and state.last_team != team_now:
                    state.sigma = min(160.0, state.sigma + cfg.team_switch_sigma_bump)
                states[name] = state
        pending: list[tuple[dict[str, Any], str, str, float, float, float, float]] = []
        for raw in batch.to_dict(orient="records"):
            gid = str(raw["__game_id"])
            blue_lu = stable_lineups.get(gid, {}).get("Blue") or []
            red_lu = stable_lineups.get(gid, {}).get("Red") or []
            blue = team_ids.get((gid, "Blue"), "")
            red = team_ids.get((gid, "Red"), "")
            if not blue or not red or blue == red:
                raise CurrentRatingLedgerError(
                    f"stable team identity is incomplete for {gid}"
                )
            base_b, sig_b, known_b, _ = _aggregate(states, blue_lu, cfg, include_momentum=False)
            base_r, sig_r, known_r, _ = _aggregate(states, red_lu, cfg, include_momentum=False)
            mu_b, _, _, details_b = _aggregate(states, blue_lu, cfg)
            mu_r, _, _, details_r = _aggregate(states, red_lu, cfg)
            sig = math.hypot(sig_b, sig_r)
            p_base = expected_score(base_b, base_r)
            p = expected_score(mu_b, mu_r)
            shrink = 1.0 / (1.0 + (sig / 130.0) ** 2)
            p_shrunk = 0.5 + (p - 0.5) * shrink
            rows.append(
                {
                    "game_id": gid,
                    "date": pd.Timestamp(raw["date"], tz="UTC"),
                    "series_id": str(raw.get("series_id") or f"map:{gid}"),
                    "base_player_logit": float(np.log(p_shrunk / (1.0 - p_shrunk))),
                    "player_rating_diff_scaled": float((mu_b - mu_r) / 400.0),
                }
            )
            y = raw.get("y_blue_win")
            if gid not in train_ids or pd.isna(y):
                continue
            y_value = float(y)
            if y_value not in (0.0, 1.0):
                raise CurrentRatingLedgerError(f"training outcome is invalid: {gid}")
            gdiff = raw.get("blue_golddiffat15")
            if pd.isna(gdiff):
                gdiff = raw.get("blue_golddiffat10")
            raw_length = raw.get("length_min")
            if pd.notna(raw_length):
                length = float(raw_length)
            elif pd.notna(raw.get("gamelength")):
                length = float(raw["gamelength"]) / 60.0
            else:
                length = 30.0
            mov = 1.0
            if pd.notna(gdiff) and length:
                mov = 1.0 + cfg.mov_scale * math.tanh(float(gdiff) / (200.0 * max(length, 1.0)))
            pending.append((raw, blue, red, p, p_base, mov, y_value))
        for raw, blue, red, p, p_base, mov, y_value in pending:
            gid = str(raw["__game_id"])
            blue_lu = stable_lineups.get(gid, {}).get("Blue") or []
            red_lu = stable_lineups.get(gid, {}).get("Red") or []
            intl = _is_intl(str(raw.get("league") or ""), raw.get("tournament"))
            for name, _role in blue_lu:
                state = states.setdefault(name, PlayerState(sigma=cfg.sigma0))
                k_scale = state.sigma / cfg.sigma0
                multiplier = attribution.get((gid, "Blue", name), 1.0)
                if intl:
                    state.mu_meta += cfg.k_meta * k_scale * mov * (y_value - p) * multiplier
                else:
                    state.mu_regional += cfg.k_regional * k_scale * mov * (y_value - p) * multiplier
                state.sigma = max(cfg.sigma_min, state.sigma * 0.985)
                state.n_maps += 1
                state.last_date = pd.Timestamp(raw["date"]) if pd.notna(raw.get("date")) else state.last_date
                state.last_team = blue
                league = str(raw.get("league") or "")
                if is_team_affiliation_league(league):
                    state.home_league = league
                _append_player_momentum(state, y_value - p_base, cfg)
                states[name] = state
            for name, _role in red_lu:
                state = states.setdefault(name, PlayerState(sigma=cfg.sigma0))
                k_scale = state.sigma / cfg.sigma0
                multiplier = attribution.get((gid, "Red", name), 1.0)
                if intl:
                    state.mu_meta += cfg.k_meta * k_scale * mov * ((1.0 - y_value) - (1.0 - p)) * multiplier
                else:
                    state.mu_regional += cfg.k_regional * k_scale * mov * ((1.0 - y_value) - (1.0 - p)) * multiplier
                state.sigma = max(cfg.sigma_min, state.sigma * 0.985)
                state.n_maps += 1
                state.last_date = pd.Timestamp(raw["date"]) if pd.notna(raw.get("date")) else state.last_date
                state.last_team = red
                league = str(raw.get("league") or "")
                if is_team_affiliation_league(league):
                    state.home_league = league
                _append_player_momentum(state, (1.0 - y_value) - (1.0 - p_base), cfg)
                states[name] = state
    return pd.DataFrame(rows)


def build_fold_current_rating_feature_ledger(
    maps: pd.DataFrame | AcceptedFutureValueSource,
    players: pd.DataFrame | None = None,
    teams: pd.DataFrame | None = None,
    *,
    source_receipt: Mapping[str, Any] | None = None,
    train_game_ids: Iterable[Any],
    validation_game_ids: Iterable[Any],
    fit_window_end: Any,
    destination: Path | None = None,
    source_frame_sha256: Mapping[str, str] | None = None,
    dual_config: DualEloConfig | None = None,
    player_config: PlayerEloConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one strict fold-local current-rating ledger.

    ``maps`` may be an ``AcceptedFutureValueSource``.  In that form its three
    source frames and canonical receipt are consumed directly.  The explicit
    frame form is useful for small fixtures and still requires the same source
    receipt validation.
    """

    if isinstance(maps, AcceptedFutureValueSource):
        source = maps
        if source_receipt is not None and dict(source_receipt) != dict(source.receipt):
            raise CurrentRatingLedgerError("explicit source receipt does not match accepted source")
        maps, players, teams = source.maps, source.players, source.teams
        source_receipt = source.receipt
    if players is None or teams is None or source_receipt is None:
        raise CurrentRatingLedgerError("maps, players, teams, and source receipt are required")
    _, eligible_ids = validate_future_value_source_receipt_payload(source_receipt)
    train_ids = _as_ids(train_game_ids, "training")
    validation_ids = _as_ids(validation_game_ids, "validation")
    output_ids = tuple(sorted(set(train_ids) | set(validation_ids)))
    eligible_set = set(eligible_ids)
    if set(train_ids) & set(validation_ids):
        raise CurrentRatingLedgerError("training and validation IDs overlap")
    if not set(output_ids).issubset(eligible_set):
        raise CurrentRatingLedgerError("training and validation IDs are outside the eligible census")
    if not output_ids:
        raise CurrentRatingLedgerError("training and validation IDs do not cover a fold")
    raw_map_ids_series = _game_ids(maps, "maps").astype(str)
    raw_map_ids = set(raw_map_ids_series)
    raw_player_ids = set(_game_ids(players, "players").astype(str))
    raw_team_ids = set(_game_ids(teams, "teams").astype(str))
    if not eligible_set.issubset(raw_map_ids) or not eligible_set.issubset(raw_player_ids) or not eligible_set.issubset(raw_team_ids):
        raise CurrentRatingLedgerError(
            "full source frames are required for the accepted eligible census"
        )
    computed_source_frames = {
        "maps": _frame_digest(maps, "maps"),
        "players": _frame_digest(players, "players"),
        "teams": _frame_digest(teams, "teams"),
    }
    if source_frame_sha256 is not None:
        claimed_source_frames = {
            str(label): str(value).lower()
            for label, value in source_frame_sha256.items()
        }
        if set(claimed_source_frames) != set(computed_source_frames) or any(
            not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in claimed_source_frames.values()
        ):
            raise CurrentRatingLedgerError("full source frame hashes are invalid")
        if claimed_source_frames != computed_source_frames:
            raise CurrentRatingLedgerError("full source frame hashes do not match source frames")
    source_frames = computed_source_frames
    map_frame, player_frame, team_frame = _validate_source_frames(
        maps, players, teams, eligible_ids, output_ids
    )
    full_series_by_id = pd.Series(
        _series_ids(maps, raw_map_ids_series).astype(str).to_numpy(),
        index=raw_map_ids_series,
    )
    map_frame["series_id"] = full_series_by_id.loc[
        map_frame["__game_id"].astype(str)
    ].to_numpy()
    series_by_id = pd.Series(
        map_frame["series_id"].astype(str).to_numpy(),
        index=map_frame["__game_id"].astype(str),
    )
    train_series_ids = tuple(sorted(set(series_by_id.loc[list(train_ids)].astype(str))))
    validation_series_ids = tuple(
        sorted(set(series_by_id.loc[list(validation_ids)].astype(str)))
    )
    overlap_series = set(train_series_ids) & set(validation_series_ids)
    if overlap_series:
        raise CurrentRatingLedgerError(
            "training and validation series overlap: "
            + ", ".join(sorted(overlap_series))
        )
    cutoff = _utc_timestamp(fit_window_end, "fit_window_end")
    map_dates = pd.to_datetime(map_frame["date"], utc=True, errors="coerce")
    date_by_id = pd.Series(map_dates.to_numpy(), index=map_frame["__game_id"].astype(str))
    train_dates = date_by_id.loc[list(train_ids)]
    validation_dates = date_by_id.loc[list(validation_ids)]
    if not bool((train_dates < cutoff).all()) or not bool((validation_dates >= cutoff).all()):
        raise CurrentRatingLedgerError("training or validation dates violate the strict cutoff")
    if train_dates.max() >= validation_dates.min():
        raise CurrentRatingLedgerError("training and validation dates do not have a strict boundary")
    masked_maps, masked_players = _mask_nontraining(map_frame, player_frame, set(train_ids))
    team_features = _team_replay(masked_maps, masked_players, set(train_ids), dual_config or DualEloConfig())
    player_features = _player_replay(masked_maps, masked_players, set(train_ids), player_config or PlayerEloConfig())
    if team_features.empty or player_features.empty:
        raise CurrentRatingLedgerError("rating replay returned no features")
    ledger = team_features.merge(
        player_features,
        on=["game_id", "date", "series_id"],
        how="inner",
        validate="one_to_one",
    )
    ledger = ledger.sort_values(["date", "game_id"], kind="mergesort").reset_index(drop=True)
    if tuple(sorted(ledger["game_id"].astype(str))) != output_ids:
        raise CurrentRatingLedgerError("rating replay coverage does not match fold IDs")
    for feature in CURRENT_RATING_SIGNED_MAP_FEATURES:
        values = pd.to_numeric(ledger[feature], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise CurrentRatingLedgerError(f"rating replay produced non-finite feature: {feature}")
        ledger[feature] = values
    implementation_sha = _implementation_hash()
    artifact_digest = _artifact_digest(ledger, CURRENT_RATING_SIGNED_MAP_FEATURES)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "ledger_schema_version": SCHEMA_VERSION,
        "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
        "source_as_of": str(source_receipt["source_as_of"]),
        "source_game_count": int(source_receipt["source_game_count"]),
        "source_identity_sha256": str(source_receipt["source_identity_sha256"]),
        "model_eligible_game_count": int(len(eligible_ids)),
        "model_eligible_identity_sha256": str(source_receipt["model_eligible_identity_sha256"]),
        "model_eligible_game_ids": list(eligible_ids),
        "output_game_ids": list(output_ids),
        "output_game_count": len(output_ids),
        "output_game_identity_sha256": identity_sha256(output_ids),
        "train_game_ids": list(train_ids),
        "train_game_count": len(train_ids),
        "train_game_identity_sha256": identity_sha256(train_ids),
        "validation_game_ids": list(validation_ids),
        "validation_game_count": len(validation_ids),
        "validation_game_identity_sha256": identity_sha256(validation_ids),
        "train_series_ids": list(train_series_ids),
        "train_series_count": len(train_series_ids),
        "train_series_identity_sha256": identity_sha256(train_series_ids),
        "validation_series_ids": list(validation_series_ids),
        "validation_series_count": len(validation_series_ids),
        "validation_series_identity_sha256": identity_sha256(validation_series_ids),
        "series_disjoint": True,
        "series_partition_source": "conservative_series_superset",
        "series_partition_key_fields": [
            "league",
            "tournament",
            "unordered_team_pair",
        ],
        "state_key_policy": dict(_RECEIPT_STATE_KEY_POLICY),
        "fit_window_end": _utc_text(cutoff),
        "strict_prior_timing": "train_outcomes_only_strictly_before_cutoff",
        "same_timestamp_policy": "score_full_utc_timestamp_batch_before_training_updates",
        "masked_nontraining_map_columns": sorted(
            str(column) for column in map_frame.columns if column not in _STRUCTURAL_MAP_COLUMNS and column != "__game_id"
        ),
        "masked_nontraining_player_columns": sorted(
            str(column) for column in player_frame.columns if column not in _STRUCTURAL_PLAYER_COLUMNS and column != "__game_id"
        ),
        "source_frame_sha256": source_frames,
        "feature_names": list(CURRENT_RATING_SIGNED_MAP_FEATURES),
        "ledger_rows_sha256": artifact_digest,
        "implementation_locator": IMPLEMENTATION_LOCATOR,
        "implementation_sha256": implementation_sha,
        "artifact": {"path": None, "bytes": None, "sha256": None},
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "merge": False,
            "deployment": False,
            "betting": False,
        },
    }
    if destination is not None:
        destination = Path(destination)
        if destination.exists() and destination.is_symlink():
            raise CurrentRatingLedgerError("ledger destination must not be a symlink")
        destination.mkdir(parents=True, exist_ok=True)
        artifact_path = destination / "current-rating-feature-ledger.parquet"
        ledger.to_parquet(artifact_path, index=False)
        receipt["artifact"] = {
            "path": str(artifact_path.resolve()),
            "bytes": int(artifact_path.stat().st_size),
            "sha256": _sha256_path(artifact_path),
        }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest()
    ledger.attrs.update(
        {
            "schema_version": SCHEMA_VERSION,
            "receipt": receipt,
            "receipt_sha256": receipt["receipt_sha256"],
            "source_receipt_sha256": receipt["source_receipt_sha256"],
            "implementation_sha256": implementation_sha,
        }
    )
    if destination is not None:
        receipt_path = Path(destination) / "current-rating-feature-ledger.receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return ledger, receipt


def validate_fold_current_rating_feature_ledger(
    ledger: pd.DataFrame,
    receipt: Mapping[str, Any],
    *,
    source_receipt: Mapping[str, Any],
    train_game_ids: Iterable[Any],
    validation_game_ids: Iterable[Any],
    fit_window_end: Any,
    source_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Validate a produced ledger and its durable artifact binding."""

    if not isinstance(receipt, Mapping) or set(receipt) != {
        "schema_version", "ledger_schema_version", "source_receipt_sha256", "source_as_of",
        "source_game_count", "source_identity_sha256", "model_eligible_game_count",
        "model_eligible_identity_sha256", "model_eligible_game_ids", "output_game_ids",
        "output_game_count", "output_game_identity_sha256", "train_game_ids",
        "train_game_count", "train_game_identity_sha256", "validation_game_ids",
        "validation_game_count", "validation_game_identity_sha256", "fit_window_end",
        "train_series_ids", "train_series_count", "train_series_identity_sha256",
        "validation_series_ids", "validation_series_count",
        "validation_series_identity_sha256", "series_disjoint", "state_key_policy",
        "series_partition_source", "series_partition_key_fields",
        "strict_prior_timing", "same_timestamp_policy", "masked_nontraining_map_columns",
        "masked_nontraining_player_columns", "source_frame_sha256", "feature_names",
        "ledger_rows_sha256", "implementation_locator", "implementation_sha256", "artifact",
        "authority", "receipt_sha256",
    }:
        raise CurrentRatingLedgerError("current rating receipt schema is not canonical")
    claimed = receipt.get("receipt_sha256")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    if not isinstance(claimed, str) or hashlib.sha256(_canonical_json_bytes(payload)).hexdigest() != claimed:
        raise CurrentRatingLedgerError("current rating receipt hash does not match payload")
    validate_future_value_source_receipt_payload(source_receipt)
    if receipt.get("source_receipt_sha256") != source_receipt.get("receipt_sha256") or receipt.get("source_identity_sha256") != source_receipt.get("source_identity_sha256"):
        raise CurrentRatingLedgerError("current rating source binding changed")
    if receipt.get("implementation_sha256") != _implementation_hash():
        raise CurrentRatingLedgerError("current rating implementation binding changed")
    source_frame_hashes = receipt.get("source_frame_sha256")
    if (
        not isinstance(source_frame_hashes, Mapping)
        or set(source_frame_hashes) != {"maps", "players", "teams"}
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value, re.I) is None
            for value in source_frame_hashes.values()
        )
    ):
        raise CurrentRatingLedgerError("current rating source frame hashes are invalid")
    if source_frames is not None:
        if set(source_frames) != {"maps", "players", "teams"} or any(
            not isinstance(frame, pd.DataFrame) for frame in source_frames.values()
        ):
            raise CurrentRatingLedgerError("current rating source frames are invalid")
        computed_source_frames = {
            label: _frame_digest(source_frames[label], label)
            for label in ("maps", "players", "teams")
        }
        if dict(source_frame_hashes) != computed_source_frames:
            raise CurrentRatingLedgerError("current rating source frame hashes changed")
    train_ids = _as_ids(train_game_ids, "training")
    validation_ids = _as_ids(validation_game_ids, "validation")
    if tuple(receipt.get("train_game_ids", ())) != train_ids or tuple(receipt.get("validation_game_ids", ())) != validation_ids:
        raise CurrentRatingLedgerError("current rating fold IDs changed")
    if receipt.get("fit_window_end") != _utc_text(_utc_timestamp(fit_window_end, "fit_window_end")):
        raise CurrentRatingLedgerError("current rating cutoff changed")
    if tuple(receipt.get("feature_names", ())) != CURRENT_RATING_SIGNED_MAP_FEATURES:
        raise CurrentRatingLedgerError("current rating feature names changed")
    if receipt.get("state_key_policy") != _RECEIPT_STATE_KEY_POLICY:
        raise CurrentRatingLedgerError("current rating state key policy changed")
    if (
        receipt.get("series_partition_source") != "conservative_series_superset"
        or tuple(receipt.get("series_partition_key_fields", ()))
        != ("league", "tournament", "unordered_team_pair")
    ):
        raise CurrentRatingLedgerError("current rating series partition changed")
    authority = receipt.get("authority")
    if not isinstance(authority, Mapping) or authority.get("research_only") is not True or any(authority.get(key) is not False for key in ("public_player_rating", "public_team_rating", "public_probability", "promotion", "merge", "deployment", "betting")):
        raise CurrentRatingLedgerError("current rating receipt authority is not research-only")
    required = {"game_id", "date", "series_id", *CURRENT_RATING_SIGNED_MAP_FEATURES}
    if not required.issubset(ledger.columns) or ledger["game_id"].astype(str).duplicated().any():
        raise CurrentRatingLedgerError("current rating ledger columns or grain are invalid")
    eligible_ids = tuple(sorted(str(value) for value in receipt["model_eligible_game_ids"]))
    if receipt.get("model_eligible_game_count") != len(eligible_ids) or receipt.get("model_eligible_identity_sha256") != identity_sha256(eligible_ids):
        raise CurrentRatingLedgerError("current rating eligible census binding changed")
    expected_ids = tuple(sorted(set(train_ids) | set(validation_ids)))
    output_ids = tuple(sorted(str(value) for value in receipt["output_game_ids"]))
    if output_ids != expected_ids or receipt.get("output_game_count") != len(output_ids) or receipt.get("output_game_identity_sha256") != identity_sha256(output_ids):
        raise CurrentRatingLedgerError("current rating fold output identity changed")
    if tuple(sorted(ledger["game_id"].astype(str))) != expected_ids:
        raise CurrentRatingLedgerError("current rating ledger coverage changed")
    ledger_series = ledger.set_index(ledger["game_id"].astype(str))["series_id"].astype("string")
    if ledger_series.isna().any() or ledger_series.str.strip().eq("").any():
        raise CurrentRatingLedgerError("current rating ledger series identity is incomplete")
    actual_train_series = tuple(
        sorted(set(ledger_series.loc[list(train_ids)].astype(str)))
    )
    actual_validation_series = tuple(
        sorted(set(ledger_series.loc[list(validation_ids)].astype(str)))
    )
    if set(actual_train_series) & set(actual_validation_series):
        raise CurrentRatingLedgerError("current rating train and validation series overlap")
    if receipt.get("series_disjoint") is not True:
        raise CurrentRatingLedgerError("current rating series disjointness is not bound")
    if (
        tuple(receipt.get("train_series_ids", ())) != actual_train_series
        or tuple(receipt.get("validation_series_ids", ())) != actual_validation_series
        or receipt.get("train_series_count") != len(actual_train_series)
        or receipt.get("validation_series_count") != len(actual_validation_series)
        or receipt.get("train_series_identity_sha256") != identity_sha256(actual_train_series)
        or receipt.get("validation_series_identity_sha256")
        != identity_sha256(actual_validation_series)
    ):
        raise CurrentRatingLedgerError("current rating series binding changed")
    if _artifact_digest(ledger, CURRENT_RATING_SIGNED_MAP_FEATURES) != receipt["ledger_rows_sha256"]:
        raise CurrentRatingLedgerError("current rating ledger values changed")
    artifact = receipt.get("artifact")
    if not isinstance(artifact, Mapping):
        raise CurrentRatingLedgerError("current rating artifact binding is missing")
    artifact_path = artifact.get("path")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise CurrentRatingLedgerError("current rating artifact path is missing")
    path = Path(artifact_path)
    if path.is_symlink() or not path.is_file() or int(artifact.get("bytes") or -1) != path.stat().st_size or str(artifact.get("sha256") or "") != _sha256_path(path):
        raise CurrentRatingLedgerError("current rating artifact bytes changed")
    return ledger


# Short aliases for orchestration code.
build_current_rating_feature_ledger = build_fold_current_rating_feature_ledger
validate_current_rating_feature_ledger = validate_fold_current_rating_feature_ledger


__all__ = [
    "CurrentRatingLedgerError",
    "IMPLEMENTATION_LOCATOR",
    "RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_current_rating_feature_ledger",
    "build_fold_current_rating_feature_ledger",
    "validate_current_rating_feature_ledger",
    "validate_fold_current_rating_feature_ledger",
]
