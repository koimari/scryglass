"""Build one deduplicated OE source from the annual file and the API bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from lol_kills.export.pack_records import build_maps_frame_from_team_games
from lol_kills.etl.paths import PARQUET_DIR

LIVE_ROOT = PARQUET_DIR / "oe_live"
LIVE_PLAYER_OUTPUT = LIVE_ROOT / "oe_player_games.parquet"
LIVE_TEAM_OUTPUT = LIVE_ROOT / "oe_team_games.parquet"
LIVE_MAP_OUTPUT = LIVE_ROOT / "maps.parquet"
LIVE_META_OUTPUT = LIVE_ROOT / "meta.json"


class OeLiveSourceError(RuntimeError):
    """Raised when the annual and API sources cannot form a complete source."""


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
    if primary.empty:
        return supplement.copy()
    if supplement.empty:
        return primary.copy()
    primary = primary.copy()
    supplement = supplement.copy()
    def add_game_key(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        if "game_uid" in frame.columns:
            game_uid = frame["game_uid"].astype("string")
            fallback = frame["gameid"].astype("string") if "gameid" in frame.columns else pd.Series("", index=frame.index, dtype="string")
            frame["_source_game_key"] = game_uid.where(game_uid.notna() & game_uid.str.strip().ne(""), fallback)
        elif "gameid" in frame.columns:
            frame["_source_game_key"] = frame["gameid"].astype("string")
        else:
            raise OeLiveSourceError("OE rows have no game identifier")
        return frame[frame["_source_game_key"].notna() & frame["_source_game_key"].str.strip().ne("")]

    primary = add_game_key(primary)
    supplement = add_game_key(supplement)
    seen: set[tuple[Any, ...]] = set()
    for _, group in primary.groupby("_source_game_key", sort=False):
        signature = _signature(group, with_players=with_players)
        if signature is not None:
            seen.add(signature)
    accepted: list[pd.DataFrame] = []
    for _, group in supplement.groupby("_source_game_key", sort=False):
        signature = _signature(group, with_players=with_players)
        if signature is not None and signature in seen:
            continue
        if signature is not None:
            seen.add(signature)
        accepted.append(group)
    if not accepted:
        return primary.drop(columns=["_source_game_key"], errors="ignore")
    return pd.concat([primary, *accepted], ignore_index=True, sort=False).drop(columns=["_source_game_key"], errors="ignore")


def build_live_source(root: Path | str = Path(".")) -> dict[str, Any]:
    repo_root = Path(root).resolve()
    primary_player_path = repo_root / PARQUET_DIR / "oe_player_games.parquet"
    primary_team_path = repo_root / PARQUET_DIR / "oe_team_games.parquet"
    api_player_path = repo_root / PARQUET_DIR / "oe_api_player_games.parquet"
    api_team_path = repo_root / PARQUET_DIR / "oe_api_team_games.parquet"
    api_meta_path = repo_root / PARQUET_DIR / "oe_api_meta.json"
    for path in (primary_player_path, primary_team_path, api_player_path, api_team_path, api_meta_path):
        if not path.is_file():
            raise OeLiveSourceError(f"required OE source is missing: {path}")

    primary_player = pd.read_parquet(primary_player_path)
    primary_team = pd.read_parquet(primary_team_path)
    api_player = pd.read_parquet(api_player_path)
    api_team = pd.read_parquet(api_team_path)
    player = _merge(primary_player, api_player, with_players=True)
    team = _merge(primary_team, api_team, with_players=False)
    maps = build_maps_frame_from_team_games(team)
    if player.empty or team.empty or maps.empty:
        raise OeLiveSourceError("OE live source has no complete player, team, and map frames")

    live_root = repo_root / LIVE_ROOT
    live_root.mkdir(parents=True, exist_ok=True)
    player.to_parquet(repo_root / LIVE_PLAYER_OUTPUT, index=False)
    team.to_parquet(repo_root / LIVE_TEAM_OUTPUT, index=False)
    maps.to_parquet(repo_root / LIVE_MAP_OUTPUT, index=False)
    api_meta = json.loads(api_meta_path.read_text(encoding="utf-8"))
    meta = {
        "schema_version": "scryglass:oe-live-source:v1",
        "source_mode": "oe_only",
        "sources": [
            {"locator": str(primary_player_path.relative_to(repo_root)), "raw_sha256": _sha256(primary_player_path)},
            {"locator": str(primary_team_path.relative_to(repo_root)), "raw_sha256": _sha256(primary_team_path)},
            {"locator": str(api_player_path.relative_to(repo_root)), "raw_sha256": _sha256(api_player_path)},
            {"locator": str(api_team_path.relative_to(repo_root)), "raw_sha256": _sha256(api_team_path)},
            {"locator": str(api_meta_path.relative_to(repo_root)), "raw_sha256": _sha256(api_meta_path)},
        ],
        "source_latest": api_meta.get("source_latest"),
        "player_rows": len(player),
        "player_rows_with_names": int(player["playername"].notna().sum()) if "playername" in player else 0,
        "team_rows": len(team),
        "maps": len(maps),
        "outputs": {
            "player": str((repo_root / LIVE_PLAYER_OUTPUT).relative_to(repo_root)),
            "team": str((repo_root / LIVE_TEAM_OUTPUT).relative_to(repo_root)),
            "maps": str((repo_root / LIVE_MAP_OUTPUT).relative_to(repo_root)),
        },
    }
    (repo_root / LIVE_META_OUTPUT).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(build_live_source(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
