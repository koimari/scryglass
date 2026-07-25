#!/usr/bin/env python3
"""
Build per-map feature matrix → data/lol/features/maps.parquet

Features: matchup strength (Elo), draft ridge scores, pace/CKPM,
rolling early-game (OE when present), context (league/side/patch).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from lol_kills.etl.aliases import normalize_champ, normalize_team
from lol_kills.etl.paths import FEATURES_DIR, PARQUET_DIR

ROOT = Path(__file__).resolve().parents[2]
DRAFT_MODEL = ROOT / "data" / "lol" / "draft_model.json"
MARKETS_MODEL = ROOT / "data" / "lol" / "markets_model.json"


def _elo_table(maps: pd.DataFrame, k: float = 24.0, base: float = 1500.0) -> dict[str, float]:
    """Sequential Elo by date; returns final ratings (also used for rolling history)."""
    ratings: dict[str, float] = defaultdict(lambda: base)
    # We'll compute pre-match elo into the feature frame separately
    return ratings


def attach_elo_features(maps: pd.DataFrame, k: float = 24.0, base: float = 1500.0) -> pd.DataFrame:
    df = maps.sort_values("date").copy()
    ratings: dict[str, float] = defaultdict(lambda: base)
    elo_b, elo_r, elo_diff = [], [], []
    for _, row in df.iterrows():
        bt = normalize_team(str(row.get("blue_team") or ""))
        rt = normalize_team(str(row.get("red_team") or ""))
        eb, er = ratings[bt], ratings[rt]
        elo_b.append(eb)
        elo_r.append(er)
        elo_diff.append(eb - er)
        # update
        y = row.get("y_blue_win")
        if pd.isna(y):
            continue
        exp_b = 1.0 / (1.0 + 10 ** ((er - eb) / 400.0))
        ratings[bt] = eb + k * (float(y) - exp_b)
        ratings[rt] = er + k * ((1.0 - float(y)) - (1.0 - exp_b))
    df["elo_blue"] = elo_b
    df["elo_red"] = elo_r
    df["elo_diff"] = elo_diff
    return df


def attach_form_features(maps: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Rolling win rate, kills for/against, ckpm prior to each game."""
    df = maps.sort_values("date").copy()
    hist: dict[str, list[dict]] = defaultdict(list)

    wr_b, wr_r = [], []
    kf_b, kf_r = [], []
    ka_b, ka_r = [], []
    ck_b, ck_r = [], []
    h2h = []

    for _, row in df.iterrows():
        bt = normalize_team(str(row.get("blue_team") or ""))
        rt = normalize_team(str(row.get("red_team") or ""))

        def stats(team: str) -> tuple[float, float, float, float]:
            h = hist[team][-window:]
            if not h:
                return 0.5, 14.0, 14.0, 1.0
            wr = sum(x["win"] for x in h) / len(h)
            kf = sum(x["kills"] for x in h) / len(h)
            ka = sum(x["opp_kills"] for x in h) / len(h)
            ck = sum(x["ckpm"] for x in h) / len(h)
            return wr, kf, ka, ck

        wb, kfb, kab, ckb = stats(bt)
        wr_, kfr, kar, ckr = stats(rt)
        wr_b.append(wb)
        wr_r.append(wr_)
        kf_b.append(kfb)
        kf_r.append(kfr)
        ka_b.append(kab)
        ka_r.append(kar)
        ck_b.append(ckb)
        ck_r.append(ckr)

        # H2H residual: blue win rate in prior meetings
        meetings = [
            x
            for x in hist[bt]
            if x.get("opp") == rt
        ][-8:]
        if meetings:
            h2h.append(sum(x["win"] for x in meetings) / len(meetings) - 0.5)
        else:
            h2h.append(0.0)

        # append results after features (no leakage)
        y = row.get("y_blue_win")
        bk = row.get("blue_kills")
        rk = row.get("red_kills")
        ck = row.get("ckpm")
        if pd.isna(ck) and row.get("length_min") and row.get("total_kills"):
            ck = float(row["total_kills"]) / float(row["length_min"])
        if not pd.isna(y):
            hist[bt].append(
                {
                    "win": float(y),
                    "kills": float(bk) if pd.notna(bk) else 14.0,
                    "opp_kills": float(rk) if pd.notna(rk) else 14.0,
                    "ckpm": float(ck) if pd.notna(ck) else 1.0,
                    "opp": rt,
                }
            )
            hist[rt].append(
                {
                    "win": 1.0 - float(y),
                    "kills": float(rk) if pd.notna(rk) else 14.0,
                    "opp_kills": float(bk) if pd.notna(bk) else 14.0,
                    "ckpm": float(ck) if pd.notna(ck) else 1.0,
                    "opp": bt,
                }
            )

    df["form_wr_blue"] = wr_b
    df["form_wr_red"] = wr_r
    df["form_wr_diff"] = np.array(wr_b) - np.array(wr_r)
    df["form_kills_blue"] = kf_b
    df["form_kills_red"] = kf_r
    df["form_kills_diff"] = np.array(kf_b) - np.array(kf_r)
    df["form_ka_blue"] = ka_b
    df["form_ka_red"] = ka_r
    df["form_ckpm_blue"] = ck_b
    df["form_ckpm_red"] = ck_r
    df["form_ckpm_avg"] = (np.array(ck_b) + np.array(ck_r)) / 2.0
    df["h2h_blue_edge"] = h2h
    return df


def _load_champ_betas() -> tuple[dict[str, float], dict[str, float], float]:
    """Kills betas from draft_model; win deltas from markets_model if present."""
    kill_beta: dict[str, float] = {}
    win_delta: dict[str, float] = {}
    mu = 28.0
    if DRAFT_MODEL.exists():
        dm = json.loads(DRAFT_MODEL.read_text())
        model = dm.get("model", dm)
        kill_beta = {
            k: float(v) for k, v in (model.get("champion_effects") or {}).items()
        }
        mu = float(model.get("intercept", dm.get("intercept", mu)))
    if MARKETS_MODEL.exists():
        mm = json.loads(MARKETS_MODEL.read_text())
        cwr = mm.get("champion_wr") or {}
        # champion_wr: {champ: {delta or wr effect}}
        for champ, val in cwr.items():
            if isinstance(val, dict):
                if "logit" in val:
                    win_delta[champ] = float(val["logit"])
                else:
                    # convert pp at 50 → approx logit
                    pp = float(val.get("delta_wr_pp_at_50", 0.0))
                    win_delta[champ] = pp / 25.0
            else:
                try:
                    win_delta[champ] = float(val)
                except (TypeError, ValueError):
                    pass
        model = mm.get("model") or {}
        if not win_delta and isinstance(model.get("champion_wr"), dict):
            for champ, val in model["champion_wr"].items():
                if isinstance(val, dict) and "logit" in val:
                    win_delta[champ] = float(val["logit"])
    return kill_beta, win_delta, mu


def attach_draft_features(maps: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    kill_beta, win_delta, mu = _load_champ_betas()
    df = maps.copy()
    if players is None or players.empty:
        df["draft_kills_shift"] = 0.0
        df["draft_win_logit_blue"] = 0.0
        df["draft_unknown_frac"] = 1.0
        df["draft_n_champs"] = 0
        df["draft_expected_kills"] = mu
        return df

    gcol = "game_uid" if "game_uid" in players.columns else "gameid"
    by_game: dict[str, list[dict]] = defaultdict(list)
    for rec in players.to_dict("records"):
        by_game[str(rec.get(gcol) or "")].append(rec)

    # map team→side from maps for OE players missing side alignment
    team_side = {}
    for _, row in df.iterrows():
        gid = str(row.get("game_uid") or "")
        team_side[(gid, normalize_team(str(row.get("blue_team") or "")))] = "Blue"
        team_side[(gid, normalize_team(str(row.get("red_team") or "")))] = "Red"

    shifts, win_logits, unk, nch = [], [], [], []
    for _, row in df.iterrows():
        gid = str(row.get("game_uid") or row.get("lp_game_id") or "")
        plist = by_game.get(gid, [])
        blue_champs, red_champs = [], []
        for p in plist:
            champ = p.get("champion")
            if not champ:
                continue
            side = str(p.get("side") or "").title()
            if side not in ("Blue", "Red"):
                team = normalize_team(str(p.get("teamname") or p.get("team") or ""))
                side = team_side.get((gid, team), "")
            c = normalize_champ(str(champ))
            if side == "Blue":
                blue_champs.append(c)
            elif side == "Red":
                red_champs.append(c)
        all_c = blue_champs + red_champs
        nch.append(len(all_c))
        if not all_c:
            shifts.append(0.0)
            win_logits.append(0.0)
            unk.append(1.0)
            continue
        known = [c for c in all_c if c in kill_beta or c in win_delta]
        unk.append(1.0 - len(known) / max(len(all_c), 1))
        shifts.append(sum(kill_beta.get(c, 0.0) for c in all_c))
        win_logits.append(
            sum(win_delta.get(c, 0.0) for c in blue_champs)
            - sum(win_delta.get(c, 0.0) for c in red_champs)
        )

    df["draft_kills_shift"] = shifts
    df["draft_win_logit_blue"] = win_logits
    df["draft_unknown_frac"] = unk
    df["draft_n_champs"] = nch
    df["draft_expected_kills"] = mu + np.array(shifts)
    return df

def attach_context(maps: pd.DataFrame) -> pd.DataFrame:
    df = maps.copy()
    df["league_code"] = df["league"].astype(str).str.upper()
    # simple league strength prior
    league_prior = {
        "LCK": 1.0,
        "LPL": 0.95,
        "LEC": 0.7,
        "LCS": 0.55,
        "CBLOL": 0.35,
        "PCS": 0.3,
        "VCS": 0.3,
        "INT": 0.85,
    }
    df["league_strength"] = df["league_code"].map(lambda x: league_prior.get(x, 0.4))
    # patch bucket
    def patch_bucket(p):
        if p is None or (isinstance(p, float) and math.isnan(p)):
            return "unk"
        s = str(p)
        parts = s.split(".")
        return ".".join(parts[:2]) if parts else "unk"

    df["patch_bucket"] = df["patch"].map(patch_bucket) if "patch" in df.columns else "unk"
    df["is_bo_map"] = 1  # placeholder; map index unknown in LP
    # OE early availability flag
    df["has_oe_early"] = (
        df["blue_golddiffat10"].notna() if "blue_golddiffat10" in df.columns else False
    )
    # For training early-game *as labels* only when present — as features use rolling team rates
    return df


def attach_rolling_early(maps: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Team rolling first-blood / gold@10 rates from OE-matched history (no leakage)."""
    df = maps.sort_values("date").copy()
    hist: dict[str, list[dict]] = defaultdict(list)
    fb_b, fb_r, g10_b, g10_r = [], [], [], []

    for _, row in df.iterrows():
        bt = normalize_team(str(row.get("blue_team") or ""))
        rt = normalize_team(str(row.get("red_team") or ""))

        def avg(team: str, key: str, default: float) -> float:
            h = [x[key] for x in hist[team][-window:] if x.get(key) is not None]
            return sum(h) / len(h) if h else default

        fb_b.append(avg(bt, "fb", 0.5))
        fb_r.append(avg(rt, "fb", 0.5))
        g10_b.append(avg(bt, "g10", 0.0))
        g10_r.append(avg(rt, "g10", 0.0))

        # store post-game
        fb = row.get("y_blue_firstblood")
        g10 = row.get("blue_golddiffat10")
        if pd.notna(fb):
            hist[bt].append({"fb": float(fb), "g10": float(g10) if pd.notna(g10) else None})
            hist[rt].append(
                {"fb": 1.0 - float(fb), "g10": float(-g10) if pd.notna(g10) else None}
            )
        elif pd.notna(g10):
            hist[bt].append({"fb": None, "g10": float(g10)})
            hist[rt].append({"fb": None, "g10": float(-g10)})

    df["roll_fb_blue"] = fb_b
    df["roll_fb_red"] = fb_r
    df["roll_fb_diff"] = np.array(fb_b) - np.array(fb_r)
    df["roll_g10_blue"] = g10_b
    df["roll_g10_red"] = g10_r
    df["roll_g10_diff"] = np.array(g10_b) - np.array(g10_r)
    return df


FEATURE_COLS = [
    "elo_blue",
    "elo_red",
    "elo_diff",
    "mu_blue",
    "mu_red",
    "mu_diff",
    "sigma_pair",
    "p_dual_elo",
    "form_wr_blue",
    "form_wr_red",
    "form_wr_diff",
    "form_kills_blue",
    "form_kills_red",
    "form_kills_diff",
    "form_ka_blue",
    "form_ka_red",
    "form_ckpm_blue",
    "form_ckpm_red",
    "form_ckpm_avg",
    "h2h_blue_edge",
    "draft_kills_shift",
    "draft_win_logit_blue",
    "draft_unknown_frac",
    "draft_n_champs",
    "draft_expected_kills",
    "league_strength",
    "roll_fb_blue",
    "roll_fb_red",
    "roll_fb_diff",
    "roll_g10_blue",
    "roll_g10_red",
    "roll_g10_diff",
    "roster_changed",
    "player_mu_diff",
    "p_player_elo",
    "p_strength_blend",
    "map_index",
]

LABEL_COLS = ["y_blue_win", "y_total_kills", "y_blue_firstblood", "y_blue_first_inhib"]


def attach_dual_rating_features(maps: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    from lol_kills.ratings.dual_elo import build_dual_ratings, lineup_hashes_from_players

    hashes = lineup_hashes_from_players(players)
    ratings = build_dual_ratings(maps, lineup_by_game=hashes)
    df = maps.copy()
    # align on game_uid
    r = ratings[
        [
            "game_uid",
            "mu_blue",
            "mu_red",
            "mu_diff",
            "sigma_pair",
            "p_dual_elo",
            "sigma_blue",
            "sigma_red",
        ]
    ]
    df["game_uid"] = df["game_uid"].astype(str)
    r = r.copy()
    r["game_uid"] = r["game_uid"].astype(str)
    df = df.merge(r, on="game_uid", how="left")
    # legacy elo aliases = dual regional totals for compatibility
    df["elo_blue"] = df["mu_blue"].fillna(1500.0)
    df["elo_red"] = df["mu_red"].fillna(1500.0)
    df["elo_diff"] = df["elo_blue"] - df["elo_red"]
    return df


def attach_roster_and_player_features(maps: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Roster-change flag + player Dual-Elo aggregated to team μ."""
    df = maps.sort_values("date").copy()
    from lol_kills.ratings.dual_elo import lineup_hashes_from_players
    from lol_kills.ratings.player_elo import build_player_ratings

    hashes = lineup_hashes_from_players(players)
    last_hash: dict[str, str] = {}
    changed = []
    for _, row in df.iterrows():
        gid = str(row.get("game_uid") or "")
        bt = normalize_team(str(row.get("blue_team") or ""))
        rt = normalize_team(str(row.get("red_team") or ""))
        hb, hr = hashes.get(f"{gid}|{bt}"), hashes.get(f"{gid}|{rt}")
        ch = 0.0
        if hb and last_hash.get(bt) and hb != last_hash[bt]:
            ch += 0.5
        if hr and last_hash.get(rt) and hr != last_hash[rt]:
            ch += 0.5
        changed.append(ch)
        if hb:
            last_hash[bt] = hb
        if hr:
            last_hash[rt] = hr
    df["roster_changed"] = changed

    # Fit on full OE player history so roster moves accumulate; merge onto these maps.
    from lol_kills.ratings.player_elo import build_maps_frame_from_players

    maps_all = build_maps_frame_from_players(players)
    pr_all = build_player_ratings(maps_all, players)
    keep = [
        "game_uid",
        "player_mu_blue",
        "player_mu_red",
        "player_mu_diff",
        "player_sigma_pair",
        "p_player_elo",
        "player_known_blue",
        "player_known_red",
    ]
    pr = pr_all[keep].copy()
    pr["game_uid"] = pr["game_uid"].astype(str)
    df["game_uid"] = df["game_uid"].astype(str)
    df = df.drop(columns=[c for c in keep if c != "game_uid" and c in df.columns], errors="ignore")
    df = df.merge(pr, on="game_uid", how="left")
    df["player_mu_diff"] = df["player_mu_diff"].fillna(0.0)
    df["p_player_elo"] = df["p_player_elo"].fillna(0.5)
    # Strength blend: team org Elo + traveling player aggregate (fit weights later in stack)
    if "p_dual_elo" in df.columns:
        df["p_strength_blend"] = 0.60 * df["p_dual_elo"].fillna(0.5) + 0.40 * df["p_player_elo"]
    else:
        df["p_strength_blend"] = df["p_player_elo"]
    return df


def attach_series_map_index(maps: pd.DataFrame) -> pd.DataFrame:
    """Infer map index within a series from date+teams clustering (1..5)."""
    df = maps.sort_values("date").copy()
    idx = []
    # key: date floor hour + sorted teams
    counters: dict[str, int] = {}
    for _, row in df.iterrows():
        d = row.get("date")
        day = pd.Timestamp(d).floor("h") if pd.notna(d) else "na"
        teams = tuple(sorted([
            normalize_team(str(row.get("blue_team") or "")),
            normalize_team(str(row.get("red_team") or "")),
        ]))
        key = f"{day}|{teams[0]}|{teams[1]}"
        counters[key] = counters.get(key, 0) + 1
        idx.append(float(counters[key]))
    df["map_index"] = idx
    return df


def build_feature_store(
    maps_path: Path | None = None,
    players_path: Path | None = None,
) -> pd.DataFrame:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    maps_path = maps_path or (PARQUET_DIR / "maps.parquet")
    players_path = players_path or (PARQUET_DIR / "players.parquet")
    maps = pd.read_parquet(maps_path)
    players = pd.read_parquet(players_path) if players_path.exists() else pd.DataFrame()

    # Dual ratings replace simple Elo (still keeps elo_* aliases)
    df = attach_dual_rating_features(maps, players)
    df = attach_form_features(df)
    df = attach_draft_features(df, players)
    df = attach_context(df)
    df = attach_rolling_early(df)
    df = attach_roster_and_player_features(df, players)
    df = attach_series_map_index(df)

    df["league_id"] = pd.Categorical(df["league_code"]).codes
    # Ensure all feature cols exist
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    feat_cols = FEATURE_COLS + ["league_id"]
    schema = {
        "feature_cols": feat_cols,
        "label_cols": LABEL_COLS,
        "n_rows": int(len(df)),
        "id_cols": ["game_uid", "date", "league", "blue_team", "red_team"],
        "leakage": "same-game golddiffat10/15 never used as features; rolling only",
    }
    out = FEATURES_DIR / "maps.parquet"
    df.to_parquet(out, index=False)
    (FEATURES_DIR / "schema.json").write_text(json.dumps(schema, indent=2))
    print(f"[features] wrote {out} n={len(df)} features={len(feat_cols)}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    build_feature_store()


if __name__ == "__main__":
    main()
