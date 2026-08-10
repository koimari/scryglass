"""Build the public OE source from the validated annual CSV cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from lol_kills.export.pack_records import build_maps_frame_from_team_games
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.ratings.player_map_grades import CORE_INPUTS

LIVE_ROOT = Path("data/lol/warehouse/parquet/oe_live")
LIVE_PLAYER_OUTPUT = LIVE_ROOT / "oe_player_games.parquet"
LIVE_TEAM_OUTPUT = LIVE_ROOT / "oe_team_games.parquet"
LIVE_MAP_OUTPUT = LIVE_ROOT / "maps.parquet"
LIVE_META_OUTPUT = LIVE_ROOT / "meta.json"
API_PLAYER_INPUT = Path("data/lol/warehouse/parquet/oe_api_player_games.parquet")
API_TEAM_INPUT = Path("data/lol/warehouse/parquet/oe_api_team_games.parquet")
API_META_INPUT = Path("data/lol/warehouse/parquet/oe_api_meta.json")


class OeLiveSourceError(RuntimeError):
    """Raised when the annual OE source cannot form a complete source."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normal(value: object) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


def _stamp(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).isoformat()


def _role(value: object) -> str:
    token = str(value or "").casefold().strip()
    return {
        "top": "top",
        "jng": "jng",
        "jungle": "jng",
        "mid": "mid",
        "bot": "bot",
        "adc": "bot",
        "sup": "sup",
        "support": "sup",
    }.get(token, token)


def _game_keys(frame: pd.DataFrame) -> pd.Series:
    fallback = frame["gameid"] if "gameid" in frame.columns else None
    source = frame["game_uid"] if "game_uid" in frame.columns else fallback
    if source is None:
        raise OeLiveSourceError("OE rows have no game identifier")

    def clean(values: pd.Series) -> pd.Series:
        text = values.astype("string").str.strip()
        text = text.mask(
            text.isna()
            | text.str.casefold().isin({"", "nan", "nat", "none", "<na>"})
        )
        return text.str.replace(
            r"^(?:(?:oe-api|oracle-elixir-api):\s*)+",
            "",
            regex=True,
            case=False,
        ).str.strip()

    keys = clean(source)
    if fallback is not None and source is not fallback:
        keys = keys.fillna(clean(fallback))
    return keys.fillna("").astype("string")


def _patch_tokens(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep OE patch tokens as source text instead of inferred floats."""

    if "patch" not in frame.columns:
        return frame
    result = frame.copy()
    tokens = result["patch"].astype("string").str.strip()
    result["patch"] = tokens.mask(
        tokens.isna() | tokens.str.casefold().isin({"", "nan", "nat", "none", "<na>"})
    )
    return result


def _identity_complete_player_game_ids(frame: pd.DataFrame) -> set[str]:
    """Return games with complete player identities and canonical roles."""

    required = {"side", "position", "playername"}
    if frame.empty or not required.issubset(frame.columns):
        return set()
    work = pd.DataFrame(index=frame.index)
    work["_source_game_key"] = _game_keys(frame)
    work["_side"] = frame["side"].astype(str).str.title()
    work["_role"] = frame["position"].map(_role)
    work["_name"] = frame["playername"].astype("string").str.strip()
    work["_name"] = work["_name"].mask(work["_name"].eq(""))
    placeholders = {"unknown", "unknown player", "tbd", "none", "nan"}
    valid_sides = {"Blue", "Red"}
    valid_roles = {"top", "jng", "mid", "bot", "sup"}
    work["_valid_slot"] = work["_side"].isin(valid_sides) & work["_role"].isin(valid_roles)
    work["_slot"] = work["_side"].astype(str) + ":" + work["_role"].astype(str)
    work["_bad_name"] = work["_name"].str.casefold().isin(placeholders).fillna(True)
    summary = work.groupby("_source_game_key", sort=False).agg(
        rows=("_source_game_key", "size"),
        names=("_name", "count"),
        unique_names=("_name", lambda values: values.str.casefold().nunique()),
        valid_slots=("_valid_slot", "sum"),
        unique_slots=("_slot", "nunique"),
        bad_names=("_bad_name", "sum"),
    )
    valid = summary[
        summary["rows"].eq(10)
        & summary["names"].eq(10)
        & summary["unique_names"].eq(10)
        & summary["valid_slots"].eq(10)
        & summary["unique_slots"].eq(10)
        & summary["bad_names"].eq(0)
    ]
    return set(valid.index.astype(str))


def _complete_player_game_ids(frame: pd.DataFrame) -> set[str]:
    """Return identity-complete games with complete public statistics."""

    if frame.empty or any(column not in frame.columns for column in CORE_INPUTS):
        return set()
    identity_complete = _identity_complete_player_game_ids(frame)
    if not identity_complete:
        return set()
    keys = _game_keys(frame)
    statistics = frame[list(CORE_INPUTS)].apply(pd.to_numeric, errors="coerce")
    complete_rows = statistics.notna().all(axis=1)
    complete_games = complete_rows.groupby(keys, sort=False).all()
    return {
        str(game_id)
        for game_id, complete in complete_games.items()
        if bool(complete) and str(game_id) in identity_complete
    }


def _signature(group: pd.DataFrame, *, with_players: bool) -> tuple[Any, ...] | None:
    if group.empty or "side" not in group or "date" not in group:
        return None
    sides = {}
    for side in ("Blue", "Red"):
        rows = group[group["side"].astype(str).str.casefold().eq(side.casefold())]
        if rows.empty:
            return None
        first = rows.iloc[0]
        team = str(first.get("teamname") or "").strip()
        result = pd.to_numeric(first.get("result"), errors="coerce")
        if not team or pd.isna(result):
            return None
        sides[side] = {
            "team": _normal(team),
            "result": int(result),
        }
        if with_players:
            role_champions: dict[str, str] = {}
            player_rows = rows[rows.get("position", pd.Series("", index=rows.index)).astype(str).str.casefold().ne("team")]
            for _, row in player_rows.iterrows():
                role = _role(row.get("position"))
                champion = str(row.get("champion") or "").strip()
                if role in {"top", "jng", "mid", "bot", "sup"} and champion:
                    role_champions[role] = _normal(champion)
            if len(role_champions) != 5:
                return None
            sides[side]["champions"] = tuple(role_champions[role] for role in ("top", "jng", "mid", "bot", "sup"))
    return (
        _stamp(group.iloc[0]["date"]),
        _normal(group.iloc[0].get("league")),
        sides["Blue"]["team"],
        sides["Red"]["team"],
        sides["Blue"]["result"],
        sides["Red"]["result"],
        sides["Blue"].get("champions") if with_players else None,
        sides["Red"].get("champions") if with_players else None,
    )


def _merge(primary: pd.DataFrame, supplement: pd.DataFrame, *, with_players: bool) -> pd.DataFrame:
    """Merge the cached bridge while keeping its published game identities stable.

    A bridge map can later arrive in the annual OE file under Riot's official
    game ID.  The exact map signature proves that both IDs describe one map.
    Keep the older bridge ID in that case so an annual refresh does not make a
    previously published map appear to disappear.
    """

    def add_game_key(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        if "game_uid" in frame.columns:
            fallback = frame["gameid"] if "gameid" in frame.columns else None
            frame["_source_game_key"] = [
                canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
                for index, value in frame["game_uid"].items()
            ]
        elif "gameid" in frame.columns:
            frame["_source_game_key"] = frame["gameid"].map(canonical_source_game_key)
        else:
            raise OeLiveSourceError("OE rows have no game identifier")
        frame = frame[frame["_source_game_key"].str.strip().ne("")].copy()
        frame["game_uid"] = frame["_source_game_key"]
        if "gameid" in frame.columns:
            frame["gameid"] = frame["_source_game_key"]
        return frame

    primary = add_game_key(primary) if not primary.empty else primary.copy()
    supplement = add_game_key(supplement) if not supplement.empty else supplement.copy()
    if primary.empty:
        return supplement.drop(columns=["_source_game_key"], errors="ignore")
    if supplement.empty:
        return primary.drop(columns=["_source_game_key"], errors="ignore")
    for column in primary.columns.intersection(supplement.columns):
        dtype = primary[column].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            supplement[column] = pd.to_numeric(supplement[column], errors="coerce")
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            supplement[column] = pd.to_datetime(
                supplement[column], errors="coerce", utc=True
            )
    seen_ids: set[str] = set()
    signature_ids: dict[tuple[Any, ...], set[str]] = {}
    for game_id, group in primary.groupby("_source_game_key", sort=False):
        seen_ids.add(str(game_id))
        signature = _signature(group, with_players=with_players)
        if signature is not None:
            signature_ids.setdefault(signature, set()).add(str(game_id))
    accepted: list[pd.DataFrame] = []
    preserved_ids: dict[str, str] = {}
    for game_id, group in supplement.groupby("_source_game_key", sort=False):
        if str(game_id) in seen_ids:
            continue
        signature = _signature(group, with_players=with_players)
        if signature is not None:
            matching_primary_ids = signature_ids.get(signature, set())
            if len(matching_primary_ids) == 1:
                preserved_ids[next(iter(matching_primary_ids))] = str(game_id)
                continue
        seen_ids.add(str(game_id))
        if signature is not None:
            signature_ids.setdefault(signature, set()).add(str(game_id))
        accepted.append(group)
    for annual_id, published_id in preserved_ids.items():
        mask = primary["_source_game_key"].eq(annual_id)
        primary.loc[mask, "_source_game_key"] = published_id
        primary.loc[mask, "game_uid"] = published_id
        if "gameid" in primary.columns:
            primary.loc[mask, "gameid"] = published_id
    if not accepted:
        return primary.drop(columns=["_source_game_key"], errors="ignore")
    return pd.concat([primary, *accepted], ignore_index=True, sort=False).drop(columns=["_source_game_key"], errors="ignore")


def build_live_source(root: Path | str = Path(".")) -> dict[str, Any]:
    repo_root = Path(root).resolve()
    parquet_root = repo_root / "data/lol/warehouse/parquet"
    primary_player_path = parquet_root / "oe_player_games.parquet"
    primary_team_path = parquet_root / "oe_team_games.parquet"
    annual_meta_path = parquet_root / "oe_meta.json"
    api_player_path = repo_root / API_PLAYER_INPUT
    api_team_path = repo_root / API_TEAM_INPUT
    api_meta_path = repo_root / API_META_INPUT
    for path in (primary_player_path, primary_team_path):
        if not path.is_file():
            raise OeLiveSourceError(f"required OE source is missing: {path}")

    primary_player = _patch_tokens(pd.read_parquet(primary_player_path))
    primary_team = _patch_tokens(pd.read_parquet(primary_team_path))
    player = primary_player.copy()
    team = primary_team.copy()
    bridge_game_count = 0
    bridge_identity_complete_count = 0
    bridge_statistics_complete_count = 0
    bridge_used = False
    if api_player_path.is_file() and api_team_path.is_file():
        try:
            api_player = _patch_tokens(pd.read_parquet(api_player_path))
            api_team = _patch_tokens(pd.read_parquet(api_team_path))
            bridge_game_count = len(set(_game_keys(api_player).dropna().astype(str)))
            identity_complete_api_ids = _identity_complete_player_game_ids(api_player)
            statistics_complete_api_ids = _complete_player_game_ids(api_player)
            api_player_keys = _game_keys(api_player)
            api_team_keys = _game_keys(api_team)
            api_player = api_player[api_player_keys.isin(identity_complete_api_ids)].copy()
            api_team = api_team[api_team_keys.isin(identity_complete_api_ids)].copy()
            bridge_identity_complete_count = len(identity_complete_api_ids)
            bridge_statistics_complete_count = len(statistics_complete_api_ids)
            player = _merge(player, api_player, with_players=True)
            team = _merge(team, api_team, with_players=False)
            bridge_used = bool(identity_complete_api_ids)
        except (OSError, ValueError, TypeError, KeyError, OeLiveSourceError):
            bridge_game_count = 0
            bridge_identity_complete_count = 0
            bridge_statistics_complete_count = 0
            bridge_used = False
    maps = build_maps_frame_from_team_games(team)
    if player.empty or team.empty or maps.empty:
        raise OeLiveSourceError("OE live source has no complete player, team, and map frames")

    identity_complete_ids = _identity_complete_player_game_ids(player)
    statistics_complete_ids = _complete_player_game_ids(player)
    player_keys = _game_keys(player)
    source_game_ids = sorted(set(player_keys.dropna().astype(str)))
    statistics_complete_source_latest: str | None = None
    if statistics_complete_ids and "date" in player.columns:
        complete_dates = pd.to_datetime(
            player.loc[player_keys.isin(statistics_complete_ids), "date"],
            errors="coerce",
            utc=True,
        ).dropna()
        if not complete_dates.empty:
            statistics_complete_source_latest = pd.Timestamp(
                complete_dates.max()
            ).isoformat()
    source_latest_candidates: list[pd.Timestamp] = []
    try:
        annual_meta = json.loads(annual_meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        annual_meta = {}
    source_files = annual_meta.get("source_files") if isinstance(annual_meta, dict) else None
    if isinstance(source_files, list):
        source_latest_candidates.extend(
            timestamp
            for item in source_files
            if isinstance(item, dict) and item.get("date_max_utc")
            if not pd.isna(timestamp := pd.to_datetime(item.get("date_max_utc"), errors="coerce", utc=True))
        )
    if bridge_used and api_meta_path.is_file():
        try:
            api_meta = json.loads(api_meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            api_meta = {}
        api_latest = pd.to_datetime(api_meta.get("source_latest"), errors="coerce", utc=True)
        if not pd.isna(api_latest):
            source_latest_candidates.append(api_latest)
    if "date" in team.columns:
        dates = pd.to_datetime(team["date"], errors="coerce", utc=True).dropna()
        if not dates.empty:
            source_latest_candidates.append(pd.Timestamp(dates.max()))
    source_latest = (
        pd.Timestamp(max(source_latest_candidates)).isoformat()
        if source_latest_candidates
        else None
    )

    live_root = repo_root / "data/lol/warehouse/parquet/oe_live"
    live_root.mkdir(parents=True, exist_ok=True)
    live_player_output = live_root / "oe_player_games.parquet"
    live_team_output = live_root / "oe_team_games.parquet"
    live_map_output = live_root / "maps.parquet"
    live_meta_output = live_root / "meta.json"
    player.to_parquet(live_player_output, index=False)
    team.to_parquet(live_team_output, index=False)
    maps.to_parquet(live_map_output, index=False)
    meta = {
        "schema_version": "scryglass:oe-live-source:v1",
        "source_mode": "oe_only",
        "sources": [
            {"locator": str(primary_player_path.relative_to(repo_root)), "raw_sha256": _sha256(primary_player_path)},
            {"locator": str(primary_team_path.relative_to(repo_root)), "raw_sha256": _sha256(primary_team_path)},
            *(
                [{"locator": str(annual_meta_path.relative_to(repo_root)), "raw_sha256": _sha256(annual_meta_path)}]
                if annual_meta_path.is_file()
                else []
            ),
            *(
                [
                    {"locator": str(api_player_path.relative_to(repo_root)), "raw_sha256": _sha256(api_player_path)},
                    {"locator": str(api_team_path.relative_to(repo_root)), "raw_sha256": _sha256(api_team_path)},
                ]
                if bridge_used
                else []
            ),
            *(
                [{"locator": str(api_meta_path.relative_to(repo_root)), "raw_sha256": _sha256(api_meta_path)}]
                if bridge_used and api_meta_path.is_file()
                else []
            ),
        ],
        "source_latest": source_latest,
        "source_transport": "public_google_drive_file",
        "cached_bridge_used": bridge_used,
        "cached_bridge_game_count": bridge_game_count,
        "cached_bridge_identity_complete_count": bridge_identity_complete_count,
        "cached_bridge_statistics_complete_count": bridge_statistics_complete_count,
        "source_game_count": len(source_game_ids),
        "identity_complete_game_count": len(identity_complete_ids),
        "statistics_complete_game_count": len(statistics_complete_ids),
        "statistics_incomplete_game_count": len(
            identity_complete_ids.difference(statistics_complete_ids)
        ),
        "statistics_complete_source_latest": statistics_complete_source_latest,
        "player_statistics_complete": identity_complete_ids.issubset(statistics_complete_ids),
        "player_rows": len(player),
        "player_rows_with_names": int(player["playername"].notna().sum()) if "playername" in player else 0,
        "team_rows": len(team),
        "maps": len(maps),
        "outputs": {
            "player": str(live_player_output.relative_to(repo_root)),
            "team": str(live_team_output.relative_to(repo_root)),
            "maps": str(live_map_output.relative_to(repo_root)),
        },
    }
    live_meta_output.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(build_live_source(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
