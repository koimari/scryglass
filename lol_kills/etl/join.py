"""Join OE ↔ Leaguepedia — OE-primary maps with LP draft enrichment."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lol_kills.etl.aliases import normalize_team
from lol_kills.etl.competition import canonicalize_competition_frame
from lol_kills.etl.paths import PARQUET_DIR, SCHEMA_PATH, WAREHOUSE_DIR

# Focus leagues for research models (still OE-primary, not LP-capped)
MAJOR_LEAGUES = {
    "LCK",
    "LPL",
    "LEC",
    "LCS",
    "CBLOL",
    "PCS",
    "VCS",
    "LJL",
    "TCL",
    "LCP",
    "WORLDS",
    "MSI",
    "EWC",
    "FST",
    "INTL",
    "WLDs",
    "IWCs",
}


def _oe_wide(oe_team: pd.DataFrame) -> pd.DataFrame:
    """Collapse OE team rows (2 per game) into one map row with blue/red columns."""
    if oe_team.empty:
        return pd.DataFrame()

    df = oe_team.copy()
    df["teamname"] = df["teamname"].map(lambda x: normalize_team(str(x)) if pd.notna(x) else x)
    df["side"] = df["side"].astype(str).str.strip().str.title()
    df.loc[df["side"].str.lower().isin(["blue", "1"]), "side"] = "Blue"
    df.loc[df["side"].str.lower().isin(["red", "2"]), "side"] = "Red"

    blue = df[df["side"] == "Blue"].copy()
    red = df[df["side"] == "Red"].copy()
    if blue.empty or red.empty:
        return pd.DataFrame()

    # Game-level fields stay unprefixed; everything else becomes blue_/red_.
    meta = [
        "gameid",
        "date",
        "league",
        "league_source",
        "competition_scope",
        "event_kind",
        "is_international",
        "patch",
        "year",
        "split",
        "playoffs",
        "game",
        "datacompleteness",
        "url",
        "source",
        "oe_year",
        "game_uid",
        "tournament",
        "grid_series_id",
        "grid_game_id",
        "grid_game_index",
    ]
    rename_b = {c: f"blue_{c}" for c in blue.columns if c not in meta}
    rename_r = {c: f"red_{c}" for c in red.columns if c not in meta}
    blue = blue.rename(columns=rename_b)
    red = red.rename(columns=rename_r)

    merged = blue.merge(red, on=["gameid"], how="inner", suffixes=("", "_r"))
    source_blue = merged.get("source")
    source_red = merged.get("source_r")
    for col in meta:
        if col == "gameid":
            continue
        if col in merged.columns and f"{col}_r" in merged.columns:
            merged[col] = merged[col].fillna(merged[f"{col}_r"])
            merged.drop(columns=[f"{col}_r"], inplace=True, errors="ignore")
        elif f"{col}_r" in merged.columns and col not in merged.columns:
            merged[col] = merged[f"{col}_r"]
            merged.drop(columns=[f"{col}_r"], inplace=True, errors="ignore")

    merged = merged.rename(columns={"gameid": "oe_gameid"})
    merged["blue_team"] = merged.get("blue_teamname", pd.Series(dtype=object)).map(
        lambda x: normalize_team(str(x)) if pd.notna(x) else x
    )
    merged["red_team"] = merged.get("red_teamname", pd.Series(dtype=object)).map(
        lambda x: normalize_team(str(x)) if pd.notna(x) else x
    )
    # Preserve provenance when GRID temporarily fills the OE freshness gap.
    # OE remains the primary source when both contain the same game, but a
    # current GRID row must not be mislabeled as Oracle's Elixir data. A
    # partially overlapping game is explicitly marked mixed.
    if source_blue is None:
        merged["source_oe"] = True
        merged["source_grid"] = False
    else:
        blue_is_oe = source_blue.astype(str).str.lower().eq("oe")
        blue_is_grid = source_blue.astype(str).str.lower().eq("grid")
        if source_red is None:
            merged["source_oe"] = blue_is_oe
            merged["source_grid"] = blue_is_grid
        else:
            red_is_oe = source_red.astype(str).str.lower().eq("oe")
            red_is_grid = source_red.astype(str).str.lower().eq("grid")
            merged["source_oe"] = blue_is_oe & red_is_oe
            merged["source_grid"] = blue_is_grid | red_is_grid
            merged["source"] = np.select(
                [merged["source_oe"], blue_is_grid & red_is_grid, merged["source_grid"]],
                ["oe", "grid", "mixed"],
                default=source_blue,
            )
    return merged


def _normalize_oe_maps(oe_w: pd.DataFrame) -> pd.DataFrame:
    """Standard schema from OE-wide frame (labels from OE only — no same-game early as features)."""
    if oe_w.empty:
        return oe_w
    m = oe_w.copy()
    m["game_uid"] = m["oe_gameid"].astype(str)
    m = canonicalize_competition_frame(m)

    # kills / result
    bk = m.get("blue_kills", m.get("blue_teamkills"))
    rk = m.get("red_kills", m.get("red_teamkills"))
    m["blue_kills"] = pd.to_numeric(bk, errors="coerce")
    m["red_kills"] = pd.to_numeric(rk, errors="coerce")
    m["total_kills"] = m["blue_kills"] + m["red_kills"]

    br = pd.to_numeric(m.get("blue_result"), errors="coerce")
    m["blue_result"] = br
    m["y_blue_win"] = br
    m["y_total_kills"] = m["total_kills"]

    fb = pd.to_numeric(m.get("blue_firstblood"), errors="coerce")
    m["blue_firstblood"] = fb
    m["y_blue_firstblood"] = fb

    # first inhib proxy from inhibitor counts if needed
    bi = pd.to_numeric(m.get("blue_inhibitors"), errors="coerce")
    ri = pd.to_numeric(m.get("red_inhibitors"), errors="coerce")
    # OE may not have first-inhib; leave NaN unless we can infer asymmetry early — skip inference
    if "blue_first_inhib" not in m.columns:
        m["blue_first_inhib"] = np.nan
    m["y_blue_first_inhib"] = pd.to_numeric(m["blue_first_inhib"], errors="coerce")

    gl = pd.to_numeric(m.get("blue_gamelength"), errors="coerce")
    m["gamelength"] = gl
    m["length_min"] = gl / 60.0 if gl is not None else np.nan
    ck = pd.to_numeric(m.get("blue_ckpm"), errors="coerce")
    m["ckpm"] = ck

    # Keep OE early columns for rolling priors only (feature store must not use same-game values)
    for c in (
        "blue_golddiffat10",
        "blue_xpdiffat10",
        "blue_golddiffat15",
        "blue_xpdiffat15",
        "blue_firstdragon",
        "blue_firsttower",
        "blue_firstbaron",
    ):
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")

    m["source_lp"] = False
    m["lp_matched"] = False
    m["lp_game_id"] = None
    return m


def _lp_wide(lp_team: pd.DataFrame) -> pd.DataFrame:
    if lp_team.empty:
        return pd.DataFrame()
    source = canonicalize_competition_frame(lp_team)
    blue = source[source["side"] == "Blue"].copy()
    red = source[source["side"] == "Red"].copy()
    b = blue.rename(
        columns={
            "teamname": "blue_team",
            "kills": "blue_kills",
            "result": "blue_result",
            "firstblood": "blue_firstblood",
            "first_inhib": "blue_first_inhib",
            "towers": "blue_towers",
            "inhibitors": "blue_inhibitors",
            "dragons": "blue_dragons",
            "barons": "blue_barons",
            "ckpm": "ckpm",
            "length_min": "length_min",
            "total_kills": "total_kills",
        }
    )
    r = red.rename(
        columns={
            "teamname": "red_team",
            "kills": "red_kills",
            "result": "red_result",
            "firstblood": "red_firstblood",
            "first_inhib": "red_first_inhib",
            "towers": "red_towers",
            "inhibitors": "red_inhibitors",
        }
    )
    cols_b = [c for c in [
        "lp_game_id", "date", "league", "tournament", "blue_team", "blue_kills",
        "blue_result", "blue_firstblood", "blue_first_inhib", "blue_towers",
        "blue_inhibitors", "blue_dragons", "blue_barons", "ckpm", "length_min", "total_kills",
    ] if c in b.columns]
    cols_r = [c for c in [
        "lp_game_id", "red_team", "red_kills", "red_result", "red_firstblood",
        "red_first_inhib", "red_towers", "red_inhibitors",
    ] if c in r.columns]
    m = b[cols_b].merge(r[cols_r], on="lp_game_id", how="inner")
    m["game_uid"] = m["lp_game_id"]
    m["source_lp"] = True
    return canonicalize_competition_frame(m)


def _attach_lp_to_oe(oe_maps: pd.DataFrame, lp_w: pd.DataFrame, window_hours: float = 18.0) -> pd.DataFrame:
    """Enrich OE-primary maps with LP game ids / first_inhib / tournament when matched."""
    out = oe_maps.copy()
    if lp_w.empty:
        out["lp_matched"] = False
        return out

    lp = lp_w.copy()
    lp = canonicalize_competition_frame(lp)
    out = canonicalize_competition_frame(out)
    lp["blue_team"] = lp["blue_team"].map(normalize_team)
    lp["red_team"] = lp["red_team"].map(normalize_team)
    out["_league"] = out["league"].astype(str).str.upper()
    out["_bt"] = out["blue_team"].map(normalize_team)
    out["_rt"] = out["red_team"].map(normalize_team)
    out["_key"] = out["_league"] + "|" + out["_bt"] + "|" + out["_rt"]
    lp["_key"] = lp["league"] + "|" + lp["blue_team"] + "|" + lp["red_team"]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    lp["date"] = pd.to_datetime(lp["date"], errors="coerce")
    # avoid suffix collisions on merge_asof
    for c in ("lp_game_id", "tournament"):
        if c in out.columns:
            out = out.drop(columns=[c])

    pieces = []
    matched = 0
    for key, g in out.groupby("_key", sort=False):
        cand = lp[lp["_key"] == key]
        g = g.sort_values("date").copy()
        if cand.empty:
            g["lp_game_id"] = None
            g["lp_matched"] = False
            pieces.append(g)
            continue
        right = (
            cand[["date", "lp_game_id", "blue_first_inhib", "tournament"]]
            .rename(columns={"blue_first_inhib": "lp_first_inhib", "tournament": "lp_tournament"})
            .sort_values("date")
            .drop_duplicates("date", keep="last")
        )
        try:
            m = pd.merge_asof(
                g,
                right,
                on="date",
                direction="nearest",
                tolerance=pd.Timedelta(hours=window_hours),
            )
        except Exception:
            g["lp_game_id"] = None
            g["lp_matched"] = False
            pieces.append(g)
            continue
        if "lp_game_id" not in m.columns:
            # suffix fallback
            col = next((c for c in m.columns if c.startswith("lp_game_id")), None)
            if col:
                m = m.rename(columns={col: "lp_game_id"})
            else:
                m["lp_game_id"] = None
        m["lp_matched"] = m["lp_game_id"].notna()
        matched += int(m["lp_matched"].sum())
        if "lp_first_inhib" in m.columns:
            m["y_blue_first_inhib"] = m["lp_first_inhib"].fillna(m.get("y_blue_first_inhib"))
            m["blue_first_inhib"] = m["y_blue_first_inhib"]
        if "lp_tournament" in m.columns:
            if "tournament" not in m.columns:
                m["tournament"] = m["lp_tournament"]
            else:
                m["tournament"] = m["lp_tournament"].fillna(m["tournament"])
        pieces.append(m)

    out2 = pd.concat(pieces, ignore_index=True)
    out2["source_lp"] = out2.get("lp_matched", False)
    drop_cols = [c for c in ("_league", "_bt", "_rt", "_key", "lp_first_inhib", "lp_tournament") if c in out2.columns]
    out2.drop(columns=drop_cols, inplace=True, errors="ignore")
    print(f"[join] LP enriched {matched}/{len(out2)} OE maps")
    return out2

def build_map_warehouse(
    lp_team: pd.DataFrame | None = None,
    oe_team: pd.DataFrame | None = None,
    lp_players: pd.DataFrame | None = None,
    majors_only: bool = True,
) -> pd.DataFrame:
    """OE-primary maps.parquet (+ players). LP drafts attached when matched."""
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

    if lp_team is None:
        p = PARQUET_DIR / "lp_team_games.parquet"
        lp_team = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if oe_team is None:
        p = PARQUET_DIR / "oe_team_games.parquet"
        oe_team = pd.read_parquet(p) if p.exists() else pd.DataFrame()
    if lp_players is None:
        p = PARQUET_DIR / "lp_player_games.parquet"
        lp_players = pd.read_parquet(p) if p.exists() else pd.DataFrame()

    oe_w = _oe_wide(oe_team) if oe_team is not None and not oe_team.empty else pd.DataFrame()
    maps = _normalize_oe_maps(oe_w)

    if majors_only and not maps.empty:
        # Keep major + anything containing Worlds/MSI/EWC
        def is_major(lg: str) -> bool:
            u = str(lg).upper()
            if u in MAJOR_LEAGUES:
                return True
            return any(x in u for x in ("WORLD", "MSI", "EWC", "FIRST STAND"))

        mask = maps["league"].map(is_major)
        maps = maps[mask].copy()
        print(f"[join] majors filter → {len(maps)} OE maps")

    lp_w = _lp_wide(lp_team) if lp_team is not None and not lp_team.empty else pd.DataFrame()
    if not maps.empty:
        maps = _attach_lp_to_oe(maps, lp_w)
    elif not lp_w.empty:
        # Fallback if no OE
        maps = canonicalize_competition_frame(lp_w.copy())
        maps["y_blue_win"] = maps["blue_result"]
        maps["y_total_kills"] = maps["total_kills"]
        maps["y_blue_firstblood"] = maps.get("blue_firstblood")
        maps["y_blue_first_inhib"] = maps.get("blue_first_inhib")
        maps["oe_gameid"] = None
        maps["lp_matched"] = True
        print("[join] WARNING: no OE data — falling back to LP-primary")

    maps_path = PARQUET_DIR / "maps.parquet"
    maps.to_parquet(maps_path, index=False)

    # Players: prefer OE player parquet if present, else LP
    oe_pl = PARQUET_DIR / "oe_player_games.parquet"
    if oe_pl.exists():
        oe_players = pd.read_parquet(oe_pl)
        if not oe_players.empty:
            oe_players = oe_players.rename(columns={"gameid": "game_uid"})
            oe_players["game_uid"] = oe_players["game_uid"].astype(str)
            oe_players = canonicalize_competition_frame(oe_players)
            oe_players.to_parquet(PARQUET_DIR / "players.parquet", index=False)
            print(f"[join] wrote OE players n={len(oe_players)}")
    elif lp_players is not None and not lp_players.empty:
        lp_players.to_parquet(PARQUET_DIR / "players.parquet", index=False)

    schema = {
        "identity": {
            "taxonomy_version": "2026-07-26.1",
            "team_key": "canonical organization identity, independent of league/event",
            "league_source": "original source label retained for audit",
        },
        "maps": {
            "n_rows": int(len(maps)),
            "primary": "oracle_elixir",
            "description": "One row per OE map; LP enrichment when matched; early OE cols for rolling only",
            "columns": list(maps.columns) if len(maps) else [],
        },
        "keys": {"game_uid": "OE gameid", "lp_game_id": "Leaguepedia GameId when matched"},
        "labels": ["y_blue_win", "y_total_kills", "y_blue_firstblood", "y_blue_first_inhib"],
        "leakage_note": "Do not use same-game golddiffat10/15 as model features; rolling priors only",
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2))
    print(f"[join] wrote {maps_path} n={len(maps)} (OE-primary)")
    return maps
