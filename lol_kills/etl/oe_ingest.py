"""Download / parse Oracle's Elixir annual CSVs into team-game + player-game parquet.

Keeps the **full** OE schema on both team and player rows (no column allowlist).
Downstream models can select what they need; missing fields across years are NaN."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from lol_kills.etl.aliases import normalize_champ, normalize_team
from lol_kills.etl.paths import OE_DRIVE_IDS, OE_FOLDER, PARQUET_DIR, RAW_OE_DIR


def oe_csv_path(year: str | int) -> Path:
    y = str(year)
    return RAW_OE_DIR / f"{y}_LoL_esports_match_data_from_OraclesElixir.csv"


def list_local_oe_csvs() -> list[Path]:
    RAW_OE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(RAW_OE_DIR.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))


def load_cached_oe(
    years: Iterable[str | int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the last successful normalized OE cache, if one exists.

    CI workers do not retain the raw annual CSVs between runs.  The normalized
    parquet cache is the durable hand-off between the fast GRID refresh and a
    slower OE reconciliation.  An empty pair is returned when no cache exists.
    """
    team_path = PARQUET_DIR / "oe_team_games.parquet"
    player_path = PARQUET_DIR / "oe_player_games.parquet"
    if not team_path.exists() or not player_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    try:
        team = pd.read_parquet(team_path)
        players = pd.read_parquet(player_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[oe] cached parquet could not be read: {exc}")
        return pd.DataFrame(), pd.DataFrame()

    # This compatibility path can contain a previously reconciled OE+GRID
    # frame. A fast GRID refresh must not feed those GRID rows back into the
    # primary side of the next merge.
    def oe_rows_only(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "source" not in frame.columns:
            return frame
        source = (
            frame["source"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.casefold()
        )
        return frame.loc[source.eq("oe")].copy()

    team = oe_rows_only(team)
    players = oe_rows_only(players)

    if years:
        wanted = {int(y) for y in years}

        def keep_years(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return frame
            year_col = next(
                (c for c in ("oe_year", "year") if c in frame.columns),
                None,
            )
            if year_col is None:
                return frame
            return frame[pd.to_numeric(frame[year_col], errors="coerce").isin(wanted)].copy()

        team = keep_years(team)
        players = keep_years(players)
    return team, players


def download_oe_years(years: Iterable[str | int], force: bool = False) -> list[Path]:
    """
    Attempt Google Drive download via gdown.
    Drive often rate-limits; place CSVs manually into data/lol/warehouse/raw/ if this fails.
    """
    RAW_OE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError as e:
        raise SystemExit(
            "gdown required for OE download: pip install gdown\n"
            f"Or manually download from {OE_FOLDER} into {RAW_OE_DIR}"
        ) from e

    out_paths: list[Path] = []
    for year in years:
        y = str(year)
        dest = oe_csv_path(y)
        if dest.exists() and dest.stat().st_size > 1_000_000 and not force:
            print(f"[oe] skip existing {dest.name}")
            out_paths.append(dest)
            continue
        fid = OE_DRIVE_IDS.get(y)
        if not fid:
            print(f"[oe] no known Drive id for {y}")
            continue
        url = f"https://drive.google.com/uc?id={fid}"
        print(f"[oe] downloading {y} → {dest.name}")
        try:
            gdown.download(url, str(dest), quiet=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[oe] download failed for {y}: {exc}")
            print(f"     Place file manually: {dest}")
            if dest.exists() and dest.stat().st_size < 10_000:
                dest.unlink(missing_ok=True)
            continue
        if not dest.exists() or dest.stat().st_size < 10_000:
            print(f"[oe] download looks empty/blocked for {y}")
            dest.unlink(missing_ok=True)
            continue
        # reject HTML quota pages
        head = dest.read_bytes()[:200].lower()
        if b"<!doctype html" in head or b"<html" in head:
            print(f"[oe] got HTML (quota?) for {y}; remove and retry later")
            dest.unlink(missing_ok=True)
            continue
        out_paths.append(dest)
    return out_paths


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve OE column names (stripped); drop accidental duplicate headers."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


def _normalize_identity(frame: pd.DataFrame, *, players: bool) -> pd.DataFrame:
    """Team/champ aliases + dates only — do not drop or invent columns."""
    out = frame
    for column in ("playername", "playerid", "teamid"):
        if column in out.columns:
            out[column] = out[column].map(
                lambda value: str(value).strip() if pd.notna(value) else value
            )
    if "teamname" in out.columns:
        out["teamname"] = out["teamname"].map(
            lambda x: normalize_team(str(x)) if pd.notna(x) else x
        )
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if players and "champion" in out.columns:
        out["champion"] = out["champion"].map(
            lambda x: normalize_champ(str(x)) if pd.notna(x) else x
        )
    return out


def parse_oe_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (team_games, player_games) with the full OE schema on each."""
    print(f"[oe] parsing {path.name}")
    df = _strip_columns(pd.read_csv(path, low_memory=False))
    # position/participant: team rows have position == 'team'
    pos_col = next((c for c in df.columns if c.lower() == "position"), None)
    if pos_col is None:
        raise ValueError(f"No position column in {path}")

    team = _normalize_identity(
        df[df[pos_col].astype(str).str.lower() == "team"].copy(), players=False
    )
    players = _normalize_identity(
        df[df[pos_col].astype(str).str.lower() != "team"].copy(), players=True
    )

    year = _year_from_name(path.name)
    for frame in (team, players):
        frame["source"] = "oe"
        frame["oe_year"] = year
    print(
        f"[oe]   {path.name}: team={len(team)} player={len(players)} "
        f"cols={len(df.columns)}"
    )
    return team, players


def _year_from_name(name: str) -> int | None:
    m = re.match(r"(\d{4})_", name)
    return int(m.group(1)) if m else None


def ingest_oe(
    years: list[str] | None = None,
    download: bool = False,
    force_download: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all local OE CSVs (optionally download first).
    Writes parquet caches under warehouse/parquet/.
    """
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    if download:
        ys = years or list(OE_DRIVE_IDS.keys())[-4:]  # recent by default
        download_oe_years(ys, force=force_download)

    paths = list_local_oe_csvs()
    if years:
        want = {str(y) for y in years}
        paths = [p for p in paths if any(p.name.startswith(y) for y in want)]

    if not paths:
        cached_team, cached_players = load_cached_oe(years)
        if not cached_team.empty or not cached_players.empty:
            print(
                "[oe] no annual CSV available; preserving the last normalized "
                f"cache (team={len(cached_team)} player={len(cached_players)})"
            )
            return cached_team, cached_players
        print(
            "[oe] no CSVs found. Put annual files in "
            f"{RAW_OE_DIR} (from {OE_FOLDER}) or pass --download-oe"
        )
        empty_t = pd.DataFrame()
        empty_p = pd.DataFrame()
        empty_t.to_parquet(PARQUET_DIR / "oe_team_games.parquet", index=False)
        empty_p.to_parquet(PARQUET_DIR / "oe_player_games.parquet", index=False)
        return empty_t, empty_p

    teams, players = [], []
    for p in paths:
        t, pl = parse_oe_csv(p)
        teams.append(t)
        players.append(pl)

    # A partial Google Drive recovery must not erase years whose annual file
    # was still quota-blocked. Keep cached rows only for those missing years;
    # newly parsed OE rows remain authoritative for years we did download.
    cached_team, cached_players = load_cached_oe()
    parsed_years = {_year_from_name(p.name) for p in paths}
    parsed_years.discard(None)
    if parsed_years:
        for cached, target in (
            (cached_team, teams),
            (cached_players, players),
        ):
            if cached.empty:
                continue
            year_col = "oe_year" if "oe_year" in cached.columns else "year"
            if year_col not in cached.columns:
                continue
            missing = cached[
                ~pd.to_numeric(cached[year_col], errors="coerce").isin(parsed_years)
            ]
            if not missing.empty:
                target.insert(0, missing)

    # Outer-union columns across years (older dumps may lack newer fields).
    team_df = pd.concat(teams, ignore_index=True, sort=False)
    player_df = pd.concat(players, ignore_index=True, sort=False)

    # Dedup by gameid+side (later years may overlap)
    if not team_df.empty and "gameid" in team_df.columns:
        team_df = team_df.sort_values("date").drop_duplicates(["gameid", "side"], keep="last")
    if not player_df.empty and "gameid" in player_df.columns:
        player_df = player_df.sort_values("date").drop_duplicates(
            ["gameid", "side", "position"], keep="last"
        )

    team_path = PARQUET_DIR / "oe_team_games.parquet"
    player_path = PARQUET_DIR / "oe_player_games.parquet"
    team_df.to_parquet(team_path, index=False)
    player_df.to_parquet(player_path, index=False)
    meta = {
        "n_team_rows": int(len(team_df)),
        "n_player_rows": int(len(player_df)),
        "n_team_cols": int(len(team_df.columns)),
        "n_player_cols": int(len(player_df.columns)),
        "n_games": int(team_df["gameid"].nunique()) if len(team_df) and "gameid" in team_df else 0,
        "files": [p.name for p in paths],
        "schema": "full_oe",
        "team_columns": list(team_df.columns),
        "player_columns": list(player_df.columns),
    }
    (PARQUET_DIR / "oe_meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[oe] wrote {team_path.name} rows={len(team_df)} cols={meta['n_team_cols']} "
        f"games={meta['n_games']}"
    )
    print(
        f"[oe] wrote {player_path.name} rows={len(player_df)} cols={meta['n_player_cols']}"
    )
    return team_df, player_df


if __name__ == "__main__":
    ingest_oe(download=False)
