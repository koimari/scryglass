#!/usr/bin/env python3
"""
Unified pre-game research board.

  python -m lol_kills.board Gen.G T1 --league LCK \\
    --blue "Gwen,Jarvan IV,Mel,Ezreal,Karma" \\
    --red "K'Sante,Wukong,Ryze,Caitlyn,Bard" \\
    --lines "29.5:1.52/2.45,32.5:1.93/1.80" \\
    --ml "Gen.G:1.42,T1:2.70"
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from lol_kills.draft_phase_score import draft_score_composite
from lol_kills.draft_dynamics import analyze_draft_dynamics, format_dynamics_report
from lol_kills.draft_tierlist import (
    blend_win_with_tierlist,
    format_tierlist_report,
    score_draft_tierlist,
)
from lol_kills.econ import (
    append_odds_journal,
    disagreement_highlights,
    evaluate_odds_row,
    grade_combo,
    p_under_normal,
    rank_plus_ev,
)
from lol_kills.etl.aliases import normalize_champ, normalize_team
from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR
from lol_kills.features.build import FEATURE_COLS, _load_champ_betas
from lol_kills.ml.train import predict_row, race_to_k
from lol_kills.predict_draft import parse_draft
from lol_kills.recommend import parse_lines, resolve_team
from lol_kills.research.public_soft_signal import (
    analyze_live_public_edge,
    format_public_edge_report,
)
ROOT = Path(__file__).resolve().parents[1]
KILL_LINES = [22.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 32.5, 34.5, 36.5]
HIGH, MED = 0.70, 0.60

# Kill lines to quote as fair-odds shopping list (book compare)
FAIR_KILL_LINES = [26.5, 27.5, 28.5, 29.5, 30.5, 32.5, 34.5, 36.5]


def fair_odds_card(sheet: dict) -> list[dict]:
    """
    Core shopping list: selection → model p → fair decimal odds.
    Book decimal > fair ⇒ +EV (before vig nuance on the other side).
    """
    m = sheet["match"]
    t1, t2 = m["team1"], m["team2"]
    by = {(r["market"], r["selection"]): r for r in sheet.get("board") or []}
    card = []

    def add(market: str, selection: str, *, tag: str = "") -> None:
        r = by.get((market, selection))
        if not r:
            return
        p = float(r["p"])
        fair = r.get("fair_odds") or (round(1.0 / p, 3) if p > 0.01 else None)
        card.append(
            {
                "market": market,
                "selection": selection,
                "p": round(p, 4),
                "fair_odds": fair,
                "tag": tag,
                "lean": p >= MED,
                "note": r.get("note"),
            }
        )

    add("Winner", t1, tag="ML")
    add("Winner", t2, tag="ML")
    add("First Blood", t1, tag="FB")
    add("First Blood", t2, tag="FB")
    add("First Inhibitor", t1, tag="FI")
    add("First Inhibitor", t2, tag="FI")
    for line in FAIR_KILL_LINES:
        mkt = f"Total kills O/U {line}"
        add(mkt, f"Under {line}", tag="O/U")
        add(mkt, f"Over {line}", tag="O/U")
    for k in (10, 15, 20):
        mkt = f"Race to {k} kills"
        add(mkt, t1, tag=f"R{k}")
        add(mkt, t2, tag=f"R{k}")
    return card


def format_fair_odds(sheet: dict) -> str:
    """Chat-first block: fair odds to compare vs book (no book prices needed)."""
    m = sheet["match"]
    card = fair_odds_card(sheet)
    lines = []
    lines.append("--- FAIR ODDS (shop the book) ---")
    lines.append("  Rule: book decimal > fair ⇒ +EV on that side.")
    lines.append("")

    def block(title: str, rows: list[dict]) -> None:
        if not rows:
            return
        lines.append(f"  {title}")
        for r in rows:
            mark = "●" if r["lean"] else " "
            label = r["selection"]
            if r["tag"].startswith("R"):
                label = f"{r['market']}: {r['selection']}"
            lines.append(
                f"  {mark} {label:28}  p={r['p']:.1%}  fair {r['fair_odds']}"
            )
        lines.append("")

    block("Moneyline", [r for r in card if r["tag"] == "ML"])
    block("First Blood", [r for r in card if r["tag"] == "FB"])
    # First inhib — soft head (sparse labels); never shop extremes
    fi = [r for r in card if r["tag"] == "FI"]
    if fi:
        note = next((r.get("note") for r in fi if r.get("note")), None)
        soft = note or "wide / noisy — treat fair as soft"
        lines.append(f"  First Inhibitor ({soft})")
        for r in fi:
            mark = "●" if r["lean"] else " "
            lines.append(
                f"  {mark} {r['selection']:28}  p={r['p']:.1%}  fair {r['fair_odds']}"
            )
        lines.append("")

    ou = [r for r in card if r["tag"] == "O/U"]
    if ou:
        lines.append("  Total kills O/U")
        lines.append(f"  {'line':>6}  {'Under p':>8} {'fair U':>7}  {'Over p':>8} {'fair O':>7}")
        by_line: dict[float, dict] = {}
        for r in ou:
            parts = r["selection"].split()
            side, line = parts[0], float(parts[1])
            by_line.setdefault(line, {})[side] = r
        for line in sorted(by_line):
            u, o = by_line[line].get("Under"), by_line[line].get("Over")
            if not u or not o:
                continue
            lines.append(
                f"  {line:6}  {u['p']:8.1%} {u['fair_odds']:7}  "
                f"{o['p']:8.1%} {o['fair_odds']:7}"
            )
        lines.append("")

    races = [r for r in card if r["tag"].startswith("R")]
    if races:
        block("Race to kills", races)

    # Shop list: leans excluding first-inhib (known overconfident head)
    tops = sorted(
        [r for r in card if r["lean"] and r["tag"] != "FI"],
        key=lambda x: -x["p"],
    )[:8]
    if tops:
        lines.append("  Look here first (p≥60%, excl. first inhib)")
        for r in tops:
            label = r["selection"]
            if r["tag"].startswith("R"):
                label = f"{r['market']}: {r['selection']}"
            elif r["tag"] == "ML":
                label = f"ML {r['selection']}"
            elif r["tag"] == "FB":
                label = f"FB {r['selection']}"
            lines.append(
                f"  ● {label:28}  fair {r['fair_odds']}  "
                f"→ +EV if book > {r['fair_odds']}"
            )
    pe = sheet.get("public_edge")
    if pe:
        lines.append("")
        lines.append(format_public_edge_report(pe))
    tier = sheet.get("tierlist")
    if tier and float(tier.get("coverage") or 0) >= 0.3:
        lean = "blue" if float(tier.get("edge_pp_shrunk") or 0) > 0 else "red"
        if abs(float(tier.get("edge_pp_shrunk") or 0)) >= 0.5:
            lines.append("")
            lines.append(
                f"  Elo tierlist lean {lean}: "
                f"{float(tier['edge_pp_shrunk']):+.1f}pp "
                f"(cov {float(tier['coverage']):.0%}, "
                f"blend Δ{((tier.get('blend') or {}).get('delta_pp') or 0):+.1f}pp on ML)"
            )
    return "\n".join(lines)


def _latest_team_features(df: pd.DataFrame, team: str, as_blue: bool) -> dict:
    """Pull most recent rolling features for a team from feature store."""
    team = normalize_team(team)
    prefix = "blue" if as_blue else "red"
    # Prefer rows where team played that side; else any side
    mask = (df["blue_team"] == team) if as_blue else (df["red_team"] == team)
    sub = df[mask].sort_values("date")
    if sub.empty:
        mask = (df["blue_team"] == team) | (df["red_team"] == team)
        sub = df[mask].sort_values("date")
        if sub.empty:
            return {}
        row = sub.iloc[-1]
        # map whichever side they were
        if row["blue_team"] == team:
            return {
                "elo": float(row.get("elo_blue", 1500)),
                "form_wr": float(row.get("form_wr_blue", 0.5)),
                "form_kills": float(row.get("form_kills_blue", 14)),
                "form_ka": float(row.get("form_ka_blue", 14)),
                "form_ckpm": float(row.get("form_ckpm_blue", 1.0)),
                "roll_fb": float(row.get("roll_fb_blue", 0.5)),
                "roll_g10": float(row.get("roll_g10_blue", 0.0)),
            }
        return {
            "elo": float(row.get("elo_red", 1500)),
            "form_wr": float(row.get("form_wr_red", 0.5)),
            "form_kills": float(row.get("form_kills_red", 14)),
            "form_ka": float(row.get("form_ka_red", 14)),
            "form_ckpm": float(row.get("form_ckpm_red", 1.0)),
            "roll_fb": float(row.get("roll_fb_red", 0.5)),
            "roll_g10": float(row.get("roll_g10_red", 0.0)),
        }
    row = sub.iloc[-1]
    return {
        "elo": float(row.get(f"elo_{prefix}", 1500)),
        "form_wr": float(row.get(f"form_wr_{prefix}", 0.5)),
        "form_kills": float(row.get(f"form_kills_{prefix}", 14)),
        "form_ka": float(row.get(f"form_ka_{prefix}", 14)),
        "form_ckpm": float(row.get(f"form_ckpm_{prefix}", 1.0)),
        "roll_fb": float(row.get(f"roll_fb_{prefix}", 0.5)),
        "roll_g10": float(row.get(f"roll_g10_{prefix}", 0.0)),
    }


def _h2h_edge(df: pd.DataFrame, blue: str, red: str) -> float:
    blue, red = normalize_team(blue), normalize_team(red)
    m = df[
        ((df["blue_team"] == blue) & (df["red_team"] == red))
        | ((df["blue_team"] == red) & (df["red_team"] == blue))
    ].dropna(subset=["y_blue_win"])
    if m.empty:
        return 0.0
    wins = 0
    n = 0
    for _, r in m.iterrows():
        n += 1
        if r["blue_team"] == blue:
            wins += float(r["y_blue_win"])
        else:
            wins += 1.0 - float(r["y_blue_win"])
    return wins / n - 0.5


# Cached feature store — avoid re-reading maps.parquet every board
_FEATURES_DF: pd.DataFrame | None = None


def _load_features_df() -> pd.DataFrame:
    global _FEATURES_DF
    if _FEATURES_DF is None:
        feat_path = FEATURES_DIR / "maps.parquet"
        _FEATURES_DF = pd.read_parquet(feat_path) if feat_path.exists() else pd.DataFrame()
    return _FEATURES_DF


def build_match_features(
    team1: str,
    team2: str,
    *,
    league: str,
    blue_champs: list[str],
    red_champs: list[str],
    team1_is_blue: bool = True,
) -> tuple[dict, dict]:
    """team1/team2 are matchup names; blue_champs belong to blue side."""
    df = _load_features_df()

    blue_team = team1 if team1_is_blue else team2
    red_team = team2 if team1_is_blue else team1

    b = _latest_team_features(df, blue_team, as_blue=True) if len(df) else {}
    r = _latest_team_features(df, red_team, as_blue=False) if len(df) else {}
    kill_beta, win_delta, mu = _load_champ_betas()
    blue_champs = [normalize_champ(c) for c in blue_champs]
    red_champs = [normalize_champ(c) for c in red_champs]
    all_c = blue_champs + red_champs
    draft_shift = sum(kill_beta.get(c, 0.0) for c in all_c)
    win_logit = sum(win_delta.get(c, 0.0) for c in blue_champs) - sum(
        win_delta.get(c, 0.0) for c in red_champs
    )
    known = sum(1 for c in all_c if c in kill_beta or c in win_delta)
    unk = 1.0 - known / max(len(all_c), 1)

    league_prior = {"LCK": 1.0, "LPL": 0.95, "LEC": 0.7, "LCS": 0.55, "CBLOL": 0.35}
    lg = league.upper()
    league_id = 0
    if len(df) and "league_code" in df.columns:
        cats = sorted(df["league_code"].dropna().unique().tolist())
        league_id = cats.index(lg) if lg in cats else 0

    mu_b = float(b.get("mu", b.get("elo", 1500.0)))
    mu_r = float(r.get("mu", r.get("elo", 1500.0)))
    # Prefer last row dual ratings if columns exist
    if len(df):
        for side, team, dest in (("blue", blue_team, "b"), ("red", red_team, "r")):
            mask = (df["blue_team"] == team) | (df["red_team"] == team)
            sub = df[mask].sort_values("date")
            if sub.empty:
                continue
            row = sub.iloc[-1]
            if row["blue_team"] == team:
                if dest == "b":
                    mu_b = float(row.get("mu_blue", row.get("elo_blue", mu_b)))
                    sig_b = float(row.get("sigma_blue", 80.0)) if "sigma_blue" in row else 80.0
                else:
                    mu_r = float(row.get("mu_blue", row.get("elo_blue", mu_r)))
            else:
                if dest == "b":
                    mu_b = float(row.get("mu_red", row.get("elo_red", mu_b)))
                else:
                    mu_r = float(row.get("mu_red", row.get("elo_red", mu_r)))

    sig_pair = math.sqrt(80.0**2 + 80.0**2)
    if len(df) and "sigma_pair" in df.columns:
        mask = ((df["blue_team"] == blue_team) & (df["red_team"] == red_team)) | (
            (df["blue_team"] == red_team) & (df["red_team"] == blue_team)
        )
        sub = df[mask].sort_values("date")
        if not sub.empty:
            sig_pair = float(sub.iloc[-1].get("sigma_pair", sig_pair) or sig_pair)

    mu_diff = float(mu_b - mu_r)
    p_dual_classic = 1.0 / (1.0 + 10 ** (-mu_diff / 400.0))
    shrink = 1.0 / (1.0 + (sig_pair / 120.0) ** 2)
    p_dual_classic = 0.5 + (p_dual_classic - 0.5) * shrink

    # Player-aggregate Elo from latest known rosters (travels on team changes)
    p_player = 0.5
    player_mu_diff = 0.0
    try:
        from lol_kills.ratings.player_elo import latest_roster_cached, score_player_lineups

        blu = latest_roster_cached(blue_team)
        red = latest_roster_cached(red_team)
        if len(blu) >= 3 and len(red) >= 3:
            scored = score_player_lineups(
                [n for n, _ in blu],
                [n for n, _ in red],
                blue_roles=[r for _, r in blu],
                red_roles=[r for _, r in red],
            )
            p_player = float(scored["p_player_elo"])
            player_mu_diff = float(scored["player_mu_diff"])
    except Exception:
        pass

    # Time-safe Elo→WR calibration (player scale was hotter than classic 400)
    try:
        from lol_kills.ratings.calibrate_elo_wr import calibrated_strength_p, load_calibration

        cal = load_calibration()
        if cal.get("team") or cal.get("player"):
            scored = calibrated_strength_p(mu_diff, player_mu_diff, cal)
            p_dual = float(scored["p_team_cal"])
            p_player = float(scored["p_player_cal"])
            p_strength = float(scored["p_strength_blend"])
            # mild uncertainty shrink after calibration
            p_dual = 0.5 + (p_dual - 0.5) * shrink
            p_player = 0.5 + (p_player - 0.5) * shrink
            p_strength = 0.5 + (p_strength - 0.5) * shrink
        else:
            p_dual = p_dual_classic
            p_strength = 0.60 * p_dual + 0.40 * p_player
    except Exception:
        p_dual = p_dual_classic
        p_strength = 0.60 * p_dual + 0.40 * p_player

    features = {
        "elo_blue": mu_b,
        "elo_red": mu_r,
        "elo_diff": mu_diff,
        "mu_blue": mu_b,
        "mu_red": mu_r,
        "mu_diff": mu_diff,
        "sigma_pair": sig_pair,
        "p_dual_elo": p_dual,
        "p_player_elo": p_player,
        "p_strength_blend": p_strength,
        "player_mu_diff": player_mu_diff,
        "form_wr_blue": b.get("form_wr", 0.5),
        "form_wr_red": r.get("form_wr", 0.5),
        "form_wr_diff": b.get("form_wr", 0.5) - r.get("form_wr", 0.5),
        "form_kills_blue": b.get("form_kills", 14.0),
        "form_kills_red": r.get("form_kills", 14.0),
        "form_kills_diff": b.get("form_kills", 14.0) - r.get("form_kills", 14.0),
        "form_ka_blue": b.get("form_ka", 14.0),
        "form_ka_red": r.get("form_ka", 14.0),
        "form_ckpm_blue": b.get("form_ckpm", 1.0),
        "form_ckpm_red": r.get("form_ckpm", 1.0),
        "form_ckpm_avg": (b.get("form_ckpm", 1.0) + r.get("form_ckpm", 1.0)) / 2,
        "h2h_blue_edge": _h2h_edge(df, blue_team, red_team) if len(df) else 0.0,
        "draft_kills_shift": draft_shift,
        "draft_win_logit_blue": win_logit,
        "draft_unknown_frac": unk,
        "draft_n_champs": len(all_c),
        "draft_expected_kills": mu + draft_shift,
        "league_strength": league_prior.get(lg, 0.4),
        "roll_fb_blue": b.get("roll_fb", 0.5),
        "roll_fb_red": r.get("roll_fb", 0.5),
        "roll_fb_diff": b.get("roll_fb", 0.5) - r.get("roll_fb", 0.5),
        "roll_g10_blue": b.get("roll_g10", 0.0),
        "roll_g10_red": r.get("roll_g10", 0.0),
        "roll_g10_diff": b.get("roll_g10", 0.0) - r.get("roll_g10", 0.0),
        "roster_changed": 0.0,
        "map_index": 1.0,
        "league_id": league_id,
    }
    meta = {"blue_team": blue_team, "red_team": red_team, "league": lg}
    return features, meta


def _shap_why(features: dict, top_n: int = 6) -> list[str]:
    """TreeSHAP on win GBM when available; else gain-style heuristics."""
    try:
        import shap  # heavy — only when explicitly enabled
        import joblib
        from lol_kills.etl.paths import MODELS_DIR
        from lol_kills.features.build import FEATURE_COLS

        gbm_path = MODELS_DIR / "win_gbm.joblib"
        meta_path = MODELS_DIR / "win_meta.json"
        if not gbm_path.exists():
            raise RuntimeError("no gbm")
        gbm = joblib.load(gbm_path)
        meta = json.loads(meta_path.read_text())
        cols = meta.get("feature_cols") or (FEATURE_COLS + ["league_id"])
        x = np.array([[float(features.get(c, 0.0) or 0.0) for c in cols]])
        explainer = shap.TreeExplainer(gbm)
        sv = explainer.shap_values(x)
        if isinstance(sv, list):
            sv = sv[1]
        vals = sv[0]
        order = np.argsort(np.abs(vals))[::-1][:top_n]
        return [f"SHAP {cols[i]}: {vals[i]:+.3f}" for i in order]
    except Exception:
        return []


def why_features(features: dict, ds: dict, top_n: int = 8, *, use_shap: bool = False) -> list[str]:
    """Human-readable drivers — heuristics by default (SHAP is slow to import)."""
    shap_lines = _shap_why(features, top_n=top_n) if use_shap else []
    if shap_lines:
        return shap_lines
    lines = []
    if abs(features.get("mu_diff", features.get("elo_diff", 0))) > 20:
        favor = "blue" if features.get("mu_diff", features.get("elo_diff", 0)) > 0 else "red"
        lines.append(f"Dual-Elo edge {favor}: Δ={features.get('mu_diff', features.get('elo_diff')):.0f}")
    if features.get("sigma_pair"):
        lines.append(f"Rating σ_pair={features['sigma_pair']:.0f} (shrink toward 50)")
    if abs(features.get("form_wr_diff", 0)) > 0.08:
        favor = "blue" if features["form_wr_diff"] > 0 else "red"
        lines.append(f"Form WR edge {favor}: Δ={features['form_wr_diff']:+.2f}")
    if abs(features.get("draft_win_logit_blue", 0)) > 0.05:
        lines.append(f"Draft win logit (blue): {features['draft_win_logit_blue']:+.3f}")
    if abs(features.get("draft_kills_shift", 0)) > 0.5:
        lines.append(f"Draft pace shift: {features['draft_kills_shift']:+.2f} kills")
    if abs(features.get("tierlist_edge_pp", 0)) >= 0.4:
        lines.append(
            f"Elo tierlist edge (blue): {features['tierlist_edge_pp']:+.1f}pp "
            f"(cov {features.get('tierlist_coverage', 0):.0%})"
        )
    lines.append(
        f"Draft Score {ds['draft_score_blue']:.0f}–{ds['draft_score_red']:.0f} "
        f"(conf {ds['confidence']:.0%})"
    )
    return lines[:top_n]


def parse_ml(spec: str) -> dict[str, float]:
    """'Gen.G:1.42,T1:2.70' → {team: odds}"""
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, odds = part.rsplit(":", 1)
        out[normalize_team(name.strip())] = float(odds)
    return out


def build_board(
    team1: str,
    team2: str,
    *,
    league: str,
    blue: list[str],
    red: list[str],
    team1_is_blue: bool = True,
    lines: list[tuple[float, float, float]] | None = None,
    ml_odds: dict[str, float] | None = None,
    fb_odds: dict[str, float] | None = None,
    inhib_odds: dict[str, float] | None = None,
    kelly_frac: float = 0.5,
    journal: bool = True,
    best_of: int = 1,
) -> dict:
    features, meta = build_match_features(
        team1, team2, league=league, blue_champs=blue, red_champs=red, team1_is_blue=team1_is_blue
    )
    # Align draft logit with Composite Draft Score v4 (phase buckets + beatdown)
    elo_diff = float(features.get("mu_diff") or features.get("elo_diff") or 0.0)
    ds = draft_score_composite(
        blue,
        red,
        league=league,
        elo_diff=elo_diff,
    )
    features["draft_win_logit_blue"] = float(ds["components"]["win_edge"])
    features["draft_beatdown_diff"] = float(ds["components"].get("beatdown_diff") or 0.0)
    features["draft_inev_diff"] = float(ds["components"].get("inev_diff") or 0.0)
    features["p_dual_elo"] = features.get("p_dual_elo", 0.5)
    features["p_player_elo"] = features.get("p_player_elo", 0.5)
    features["p_strength_blend"] = features.get(
        "p_strength_blend",
        0.60 * float(features["p_dual_elo"]) + 0.40 * float(features["p_player_elo"]),
    )
    preds = predict_row(features)

    # Prefer learned stack; fallback to weighted blend
    if "p_blue_win" in preds and preds.get("stack_components"):
        p = float(preds["p_blue_win"])
        preds["stack_weights"] = preds.get("stack_components")
    else:
        p_elo = float(features.get("p_strength_blend") or features.get("p_dual_elo") or 0.5)
        p_draft = float(ds["p_blue_draft"])
        p_gbm = float(preds.get("p_blue_win_gbm") or preds.get("p_blue_win") or p_elo)
        w_draft = 0.18 * float(ds.get("confidence", 0.5))
        w_gbm = 0.32
        w_elo = 1.0 - w_gbm - w_draft
        p = float(np.clip(w_gbm * p_gbm + w_elo * p_elo + w_draft * p_draft, 0.05, 0.95))
        preds["stack_weights"] = {"gbm": w_gbm, "elo_form": w_elo, "draft_score": w_draft}

    # Elo / Blade-Chest tierlist — soft blend on top of stack (coverage-weighted)
    # Assume paste order top/jng/mid/bot/sup when both sides have 5 champs.
    roles5 = ["top", "jng", "mid", "bot", "sup"]
    tier_kw = {}
    if len(blue) == 5 and len(red) == 5:
        tier_kw = {"blue_roles": roles5, "red_roles": roles5}
    tier = score_draft_tierlist(blue, red, league=league, **tier_kw)
    p, tier_blend = blend_win_with_tierlist(p, tier)
    tier["blend"] = tier_blend
    features["tierlist_edge_pp"] = float(tier.get("edge_pp_shrunk") or 0.0)
    features["tierlist_coverage"] = float(tier.get("coverage") or 0.0)
    features["tierlist_p_blue"] = float(tier.get("p_blue_tier") or 0.5)
    if tier_blend.get("applied"):
        sw = dict(preds.get("stack_weights") or {})
        sw["tierlist"] = float(tier_blend.get("weight") or 0.0)
        preds["stack_weights"] = sw

    preds["p_blue_win"] = p
    preds["p_red_win"] = 1.0 - p

    dynamics = analyze_draft_dynamics(
        blue,
        red,
        league=league,
        elo_diff=elo_diff,
        p_blue_pre=float(p),
    )
    mu = preds.get("kills_mean", features["draft_expected_kills"])
    sd = preds.get("kills_sd", 6.5)
    mu = 0.65 * float(mu) + 0.35 * float(features["draft_expected_kills"])

    blue_team, red_team = meta["blue_team"], meta["red_team"]
    # Map probs to team1/team2
    if team1_is_blue:
        p_t1, p_t2 = preds["p_blue_win"], preds["p_red_win"]
        p_fb_t1 = preds.get("p_blue_fb", 0.5)
        p_in_t1 = preds.get("p_blue_inhib", preds["p_blue_win"] * 0.9 + 0.05)
    else:
        p_t1, p_t2 = preds["p_red_win"], preds["p_blue_win"]
        p_fb_t1 = preds.get("p_red_fb", 0.5)
        p_in_t1 = preds.get("p_red_inhib", preds["p_red_win"] * 0.9 + 0.05)

    share = features["form_kills_blue"] / max(
        features["form_kills_blue"] + features["form_kills_red"], 1e-6
    )
    races = race_to_k(mu, share if team1_is_blue else 1 - share)

    rows = []
    rows.append({"market": "Winner", "selection": team1, "p": round(p_t1, 4)})
    rows.append({"market": "Winner", "selection": team2, "p": round(p_t2, 4)})

    p_fb_blue = float(preds.get("p_blue_fb", 0.5))
    p_fb_t1 = p_fb_blue if team1_is_blue else 1.0 - p_fb_blue
    rows.append(
        {
            "market": "First Blood",
            "selection": team1,
            "p": round(p_fb_t1, 4),
            "note": preds.get("fb_note"),
        }
    )
    rows.append(
        {
            "market": "First Blood",
            "selection": team2,
            "p": round(1.0 - p_fb_t1, 4),
            "note": preds.get("fb_note"),
        }
    )
    p_in_blue = float(preds.get("p_blue_inhib", preds["p_blue_win"]))
    p_in_t1 = p_in_blue if team1_is_blue else 1.0 - p_in_blue
    rows.append(
        {
            "market": "First Inhibitor",
            "selection": team1,
            "p": round(p_in_t1, 4),
            "note": preds.get("inhib_note"),
        }
    )
    rows.append(
        {
            "market": "First Inhibitor",
            "selection": team2,
            "p": round(1.0 - p_in_t1, 4),
            "note": preds.get("inhib_note"),
        }
    )

    for line in KILL_LINES:
        pu = p_under_normal(mu, sd, line)
        rows.append({"market": f"Total kills O/U {line}", "selection": f"Under {line}", "p": round(pu, 4)})
        rows.append({"market": f"Total kills O/U {line}", "selection": f"Over {line}", "p": round(1 - pu, 4)})

    for k, v in races.items():
        pb = v["p_blue_first"] if team1_is_blue else v["p_red_first"]
        rows.append({"market": f"Race to {k} kills", "selection": team1, "p": pb})
        rows.append({"market": f"Race to {k} kills", "selection": team2, "p": round(1 - pb, 4)})

    # Attach fair odds
    for r in rows:
        p = r["p"]
        r["fair_odds"] = round(1 / p, 3) if p and p > 0.01 else None

    # EV layer with conformal bands when available
    win_lo = preds.get("p_blue_win_lo")
    win_hi = preds.get("p_blue_win_hi")
    kills_lo = preds.get("kills_lo")
    kills_hi = preds.get("kills_hi")
    ev_rows = []
    if ml_odds:
        for team, odds in ml_odds.items():
            sel = normalize_team(team)
            p = p_t1 if sel == normalize_team(team1) else p_t2 if sel == normalize_team(team2) else None
            if p is None:
                if sel == normalize_team(blue_team):
                    p = preds["p_blue_win"]
                elif sel == normalize_team(red_team):
                    p = preds["p_red_win"]
            if p is not None:
                # map interval to this selection
                if team1_is_blue and sel == normalize_team(team1):
                    lo, hi = win_lo, win_hi
                elif team1_is_blue and sel == normalize_team(team2):
                    lo = (1 - win_hi) if win_hi is not None else None
                    hi = (1 - win_lo) if win_lo is not None else None
                else:
                    lo = hi = None
                ev_rows.append(
                    evaluate_odds_row(
                        sel, "Winner", p, odds, kelly_frac=kelly_frac, p_lo=lo, p_hi=hi
                    )
                )
    if lines:
        for line, over_o, under_o in lines:
            pu = p_under_normal(mu, sd, line)
            # rough interval from kills conformal
            if kills_lo is not None and kills_hi is not None:
                pu_lo = p_under_normal(kills_hi, sd, line)  # higher mean → lower under
                pu_hi = p_under_normal(kills_lo, sd, line)
            else:
                pu_lo = pu_hi = None
            ev_rows.append(
                evaluate_odds_row(
                    f"Over {line}",
                    f"Total kills O/U {line}",
                    1 - pu,
                    over_o,
                    kelly_frac=kelly_frac,
                    p_lo=(1 - pu_hi) if pu_hi is not None else None,
                    p_hi=(1 - pu_lo) if pu_lo is not None else None,
                )
            )
            ev_rows.append(
                evaluate_odds_row(
                    f"Under {line}",
                    f"Total kills O/U {line}",
                    pu,
                    under_o,
                    kelly_frac=kelly_frac,
                    p_lo=pu_lo,
                    p_hi=pu_hi,
                )
            )

    if inhib_odds:
        for team, odds in inhib_odds.items():
            sel = normalize_team(team)
            if sel == normalize_team(team1):
                p = p_in_t1
            elif sel == normalize_team(team2):
                p = 1.0 - p_in_t1
            else:
                continue
            row = evaluate_odds_row(
                sel, "First Inhibitor", p, odds, kelly_frac=kelly_frac * 0.5
            )
            # Soft market: never above B, haircut one letter
            g = row.get("grade") or "D"
            if g.startswith("A"):
                row["grade"] = "B"
                row["grade_why"] = "soft inhib head — capped at B"
            elif g.startswith("B"):
                row["grade"] = "C+"
                row["grade_why"] = "soft inhib head — haircut"
            row["note"] = preds.get("inhib_note")
            ev_rows.append(row)

    if fb_odds:
        for team, odds in fb_odds.items():
            sel = normalize_team(team)
            if sel == normalize_team(team1):
                p = p_fb_t1
            elif sel == normalize_team(team2):
                p = 1.0 - p_fb_t1
            else:
                continue
            ev_rows.append(
                evaluate_odds_row(sel, "First Blood", p, odds, kelly_frac=kelly_frac)
            )

    leans = [r for r in rows if r["p"] >= MED]
    leans.sort(key=lambda x: -x["p"])

    plus_ev = rank_plus_ev(ev_rows) if ev_rows else []
    combo = grade_combo(plus_ev, same_map=True) if len(plus_ev) >= 2 else None

    # Soft-public vs model disagreement / fade flags (parlay & dog hunting)
    public_edge = analyze_live_public_edge(
        blue_team,
        red_team,
        p_blue_model=float(preds["p_blue_win"]),
        heat_blue=float(features.get("form_wr_blue") or 0.5),
        heat_red=float(features.get("form_wr_red") or 0.5),
        player_mu_diff=float(features.get("player_mu_diff") or 0.0),
        draft_win_logit_blue=(
            float(features.get("draft_win_logit_blue") or 0.0)
            + 0.02 * float(features.get("tierlist_edge_pp") or 0.0)
        ),
        form_wr_diff=float(features.get("form_wr_diff") or 0.0),
        league=league,
        best_of=int(best_of or 1),
    )

    sheet_match = {
        "team1": team1,
        "team2": team2,
        "blue": blue_team,
        "red": red_team,
        "league": league,
    }
    if journal and ev_rows:
        append_odds_journal(ev_rows, sheet_match)

    return {
        "match": sheet_match,
        "draft_score": ds,
        "tierlist": tier,
        "dynamics": dynamics,
        "public_edge": public_edge,
        "kills": {
            "mean": round(mu, 2),
            "sd": round(sd, 2),
            "lo": round(kills_lo, 2) if kills_lo is not None else None,
            "hi": round(kills_hi, 2) if kills_hi is not None else None,
            "model": preds.get("kills_model"),
        },
        "winner": {
            "p_team1": round(p_t1, 4),
            "p_team2": round(p_t2, 4),
            "lo_blue": preds.get("p_blue_win_lo"),
            "hi_blue": preds.get("p_blue_win_hi"),
        },
        "board": rows,
        "leans": leans,
        "fair_odds": fair_odds_card(
            {
                "match": sheet_match,
                "board": rows,
            }
        ),
        "ev": plus_ev,
        "all_ev": ev_rows,
        "combo_grade": combo,
        "disagreement": disagreement_highlights(ev_rows) if ev_rows else [],
        "why": why_features(features, ds),
        "stack": preds.get("stack_weights"),
        "features": {
            k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
            for k, v in features.items()
        },
    }

def format_report(sheet: dict) -> str:
    m = sheet["match"]
    ds = sheet["draft_score"]
    lines = []
    lines.append(f"=== BOARD  {m['team1']} vs {m['team2']}  ({m['league']}) ===")
    lines.append(f"Blue {m['blue']} | Red {m['red']}")
    lines.append("")
    lines.append("--- Draft Score (composite @10/15/20/25) ---")
    lines.append(
        f"  Blue {ds['draft_score_blue']:.1f}  |  Red {ds['draft_score_red']:.1f}  |  "
        f"edge {ds['draft_edge']:+.1f}  |  conf {ds['confidence']:.0%}  |  curve={ds.get('curve')}"
    )
    buckets = ds.get("buckets") or {}
    if buckets:
        curve_bits = "  ".join(
            f"@{t} {buckets[str(t)]['draft_score_blue']:.1f}" for t in (10, 15, 20, 25) if str(t) in buckets
        )
        lines.append(f"  Buckets: {curve_bits}")
    bd = ds.get("beatdown") or {}
    if bd:
        lines.append(
            f"  Beatdown={bd.get('beatdown_side')}  control={bd.get('control_side')}  "
            f"plan={bd.get('plan')}  misassign={bd.get('misassign_risk')}"
        )
        if bd.get("advice"):
            lines.append(f"  {bd['advice']}")
    cal = ds.get("calibration") or {}
    bump = ds.get("wr_bump_pp")
    classic = ds.get("classic") or {}
    if bump is not None:
        lines.append(
            f"  WR bump ≈ {bump:+.1f}pp (Elo-controlled)  ·  "
            f"temp={cal.get('temperature')} via {cal.get('source')}  ·  "
            f"p_draft={ds.get('p_blue_draft')}  ·  classic={classic.get('p_blue_draft')}"
        )
    lines.append(f"  (stack input only — {ds.get('note') or 'composite draft'})")
    lines.append("")
    tier = sheet.get("tierlist")
    if tier:
        lines.append(
            format_tierlist_report(tier, team_blue=str(m["blue"]), team_red=str(m["red"]))
        )
        lines.append("")
    dyn = sheet.get("dynamics")
    if dyn:
        lines.append(format_dynamics_report(dyn, team_blue=str(m["blue"]), team_red=str(m["red"])))
        lines.append("")
    pe = sheet.get("public_edge")
    if pe:
        lines.append(format_public_edge_report(pe))
        lines.append("")
    lines.append(f"--- Kills  μ={sheet['kills']['mean']}  σ={sheet['kills']['sd']} ---")
    if sheet["kills"].get("lo") is not None:
        lines[-1] = (
            f"--- Kills  μ={sheet['kills']['mean']}  σ={sheet['kills']['sd']}  "
            f"90%≈[{sheet['kills']['lo']}, {sheet['kills']['hi']}]  "
            f"model={sheet['kills'].get('model')} ---"
        )
    w = sheet["winner"]
    band = ""
    if w.get("lo_blue") is not None:
        band = f"  (blue 90% [{w['lo_blue']:.2f}, {w['hi_blue']:.2f}])"
    lines.append(
        f"--- Winner  {m['team1']} {w['p_team1']:.1%}  /  "
        f"{m['team2']} {w['p_team2']:.1%}{band} ---"
    )
    lines.append("")
    lines.append(format_fair_odds(sheet))
    lines.append("")
    lines.append("--- Leans (≥60%) ---")
    for r in sheet["leans"][:20]:
        star = "★" if r["p"] >= 0.90 else ("●" if r["p"] >= HIGH else "○")
        note = f"  [{r['note']}]" if r.get("note") else ""
        lines.append(f"  {star} {r['market']:28} {r['selection']:20} {r['p']:.1%}  fair {r.get('fair_odds')}")
        if note:
            lines[-1] += note
    lines.append("")
    lines.append("--- Total kills O/U (model) ---")
    lines.append(f"  {'line':>6}  {'Under':>8}  {'Over':>8}")
    for line in KILL_LINES:
        pu = next(
            r["p"]
            for r in sheet["board"]
            if r["market"] == f"Total kills O/U {line}" and r["selection"].startswith("Under")
        )
        lines.append(f"  {line:6}  {pu:8.1%}  {1-pu:8.1%}")
    if sheet.get("ev"):
        lines.append("")
        lines.append("--- +EV / GRADE / Kelly (half-Kelly, 5% cap) ---")
        for r in sheet["ev"][:12]:
            g = r.get("grade", "?")
            why = r.get("grade_why", "")
            fair = r.get("fair_odds")
            lines.append(
                f"  [{g:<3}] {r['selection']:18} book@{r['odds']}  fair@{fair}  "
                f"p={r['p']:.1%}  EV={r['ev']:+.3f}  edge={r['edge_pp']:+.1f}pp  "
                f"Kelly={r['kelly']:.2%}  ({why})"
            )
        # Also show clearly −EV traps when priced (from all_ev)
        traps = [
            r
            for r in (sheet.get("all_ev") or [])
            if r.get("grade") == "F" or (r.get("ev") is not None and r["ev"] <= -0.02)
        ]
        traps.sort(key=lambda x: x.get("ev", 0))
        if traps:
            lines.append("  traps:")
            for r in traps[:6]:
                lines.append(
                    f"  [F  ] {r['selection']:18} @{r['odds']}  p={r['p']:.1%}  "
                    f"EV={r['ev']:+.3f}  ({r.get('grade_why', '−EV')})"
                )
    combo = sheet.get("combo_grade")
    if combo:
        lines.append("")
        lines.append("--- Combo GRADE (equal 1u, same-map ρ mix) ---")
        lines.append(
            f"  [{combo['grade']:<3}]  ROI≈{combo['portfolio_roi_pct']:+.1f}%  "
            f"P(both)≈{combo['p_both']:.0%}  P(any)≈{combo['p_any']:.0%}  "
            f"wipe≈{combo['p_wipe']:.0%}  n={combo['n_legs']}"
        )
        lines.append(
            f"  ({combo['grade_why']}; EV-joint {combo.get('raw_grade_before_map_haircut')})"
        )
        for leg in combo.get("legs") or []:
            lines.append(
                f"    · {leg['selection']:18} [{leg['grade']}]  "
                f"p={leg['p']:.1%}  EV={leg['ev']:+.3f}"
            )
    if sheet.get("disagreement"):
        lines.append("")
        lines.append("--- Disagreement (feels ~50/50 but book soft) ---")
        for r in sheet["disagreement"][:8]:
            g = r.get("grade", "?")
            lines.append(
                f"  [{g:<3}] {r['selection']:18} p={r['p']:.1%} vs implied {r['implied']:.1%}  "
                f"edge {r['edge_pp']:+.1f}pp"
            )
    lines.append("")
    lines.append("--- Why ---")
    for w in sheet.get("why", []):
        lines.append(f"  • {w}")
    if not sheet.get("ev"):
        lines.append("")
        lines.append(
            "No book odds pasted — use FAIR ODDS above. "
            "If book > fair on a side, that side is +EV; paste prices only to lock GRADE/Kelly."
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("team1")
    ap.add_argument("team2")
    ap.add_argument("--league", default="LCK")
    ap.add_argument("--blue", required=True, help="Comma-separated blue champs (team on blue)")
    ap.add_argument("--red", required=True, help="Comma-separated red champs")
    ap.add_argument(
        "--team1-red",
        action="store_true",
        help="If set, team1 is on red (default: team1=blue)",
    )
    ap.add_argument("--lines", default=None, help="LINE:OVER/UNDER,... for kills O/U")
    ap.add_argument("--ml", default=None, help="Team:odds,... moneyline")
    ap.add_argument("--kelly", type=float, default=0.5, help="Kelly fraction (default half)")
    ap.add_argument("--best-of", type=int, default=1, help="Series format 1/3/5 for public-edge series P")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # Resolve names against feature store if possible
    feat = FEATURES_DIR / "maps.parquet"
    known = []
    if feat.exists():
        df = pd.read_parquet(feat, columns=["blue_team", "red_team"])
        known = sorted(set(df["blue_team"].dropna()) | set(df["red_team"].dropna()))
    team1 = resolve_team(args.team1, known) if known else normalize_team(args.team1)
    team2 = resolve_team(args.team2, known) if known else normalize_team(args.team2)

    # Champ parse — reuse predict_draft known list from draft model
    from lol_kills.predict_draft import CHAMP_ALIASES  # noqa: F401
    import json as _json

    dm = _json.loads((ROOT / "data/lol/draft_model.json").read_text())
    known_champs = list((dm.get("model") or {}).get("champion_effects", {}).keys())
    blue = parse_draft(args.blue, known_champs)
    red = parse_draft(args.red, known_champs)

    lines = parse_lines(args.lines) if args.lines else None
    ml_odds = parse_ml(args.ml) if args.ml else None

    sheet = build_board(
        team1,
        team2,
        league=args.league,
        blue=blue,
        red=red,
        team1_is_blue=not args.team1_red,
        lines=lines,
        ml_odds=ml_odds,
        kelly_frac=args.kelly,
        best_of=args.best_of,
    )
    if args.json:
        print(json.dumps(sheet, indent=2, default=str))
    else:
        print(format_report(sheet))


if __name__ == "__main__":
    main()
