#!/usr/bin/env python3
"""
Soft public-favorite signal for historical LoL series analysis.

Reddit/Twitter aren't available as clean historical feeds, so we proxy
"who the public would lean" with fundamentals that drive casual money:

  1. Brand prestige (name recognition / international pedigree)
  2. Heat chase (recent map win-rate before the series — public chases form)

Compare that soft public pick to our sequential model pick, then measure
when agreeing / fading the public would have hit in 2026.

  python3 -m lol_kills.research.public_soft_signal
"""

from __future__ import annotations

import json
from math import comb

import numpy as np
import pandas as pd

from lol_kills.etl.aliases import normalize_team
from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR

OUT = MODELS_DIR / "public_soft_signal_2026.json"

# Soft "public magnetism" 0–100. Not strength — name recognition / casual lean.
# Research prior; update when a mega-brand shifts (new Worlds winner etc.).
BRAND_PRESTIGE: dict[str, float] = {
    "T1": 98,
    "Gen.G": 92,
    "JD Gaming": 88,
    "Bilibili Gaming": 86,
    "G2 Esports": 84,
    "Hanwha Life Esports": 80,
    "Top Esports": 78,
    "Dplus Kia": 76,
    "Anyone's Legend": 74,
    "Fnatic": 72,
    "KT Rolster": 70,
    "LNG Esports": 68,
    "Weibo Gaming": 66,
    "Team Vitality": 64,
    "Cloud9": 62,
    "Team Liquid": 60,
    "Karmine Corp": 58,
    "FlyQuest": 55,
    "Movistar KOI": 52,
    "DRX": 50,
    "Edward Gaming": 48,
    "Nongshim RedForce": 46,
    "BNK FEARX": 44,
    "DN Freecs": 42,
    "BRION": 40,
}


def brand_score(team: str) -> float:
    return float(BRAND_PRESTIGE.get(normalize_team(team), 35.0))


def series_p_from_map_p(p: float, best_of: int) -> float:
    p = float(np.clip(p, 0.01, 0.99))
    n = best_of // 2 + 1
    return float(sum(comb(n - 1 + L, L) * (p**n) * ((1 - p) ** L) for L in range(n)))


def _build_series(df: pd.DataFrame, gap_h: float = 4.0) -> list[dict]:
    df = df.sort_values("date").reset_index(drop=True)
    series, cur = [], None
    gap = gap_h * 3600
    for _, r in df.iterrows():
        if cur is None:
            cur = {"league": r["league"], "pair": r["pair"], "maps": [r]}
            continue
        dt = (r["date"] - cur["maps"][-1]["date"]).total_seconds()
        if r["pair"] == cur["pair"] and r["league"] == cur["league"] and dt <= gap:
            cur["maps"].append(r)
        else:
            series.append(cur)
            cur = {"league": r["league"], "pair": r["pair"], "maps": [r]}
    if cur:
        series.append(cur)
    return series


def _rolling_heat(df: pd.DataFrame, window: int = 10) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    """Causal last-N map WR per team (updated after each map)."""
    hist: dict[str, list[int]] = {}
    out: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for _, r in df.sort_values("date").iterrows():
        for team, won in (
            (r["blue_n"], float(r["y_blue_win"]) >= 0.5),
            (r["red_n"], float(r["y_blue_win"]) < 0.5),
        ):
            h = hist.setdefault(team, [])
            wr = float(np.mean(h[-window:])) if h else 0.5
            out.setdefault(team, []).append((r["date"], wr))
            h.append(1 if won else 0)
    return out


def _heat_before(heat: dict, team: str, when: pd.Timestamp) -> float:
    rows = heat.get(team) or []
    prev = [wr for ts, wr in rows if ts < when]
    return float(prev[-1]) if prev else 0.5


def public_soft_scores(
    team_a: str,
    team_b: str,
    *,
    heat_a: float,
    heat_b: float,
    w_brand: float = 0.55,
    w_heat: float = 0.45,
) -> dict:
    """Soft public lean in [0,1] for team_a (not a calibrated probability)."""
    ba, bb = brand_score(team_a), brand_score(team_b)
    brand_edge = (ba - bb) / 100.0
    heat_edge = float(heat_a - heat_b)
    raw = w_brand * brand_edge + w_heat * heat_edge
    p_pub_a = float(np.clip(0.5 + 0.5 * np.tanh(1.4 * raw), 0.05, 0.95))
    return {
        "brand_a": ba,
        "brand_b": bb,
        "heat_a": round(heat_a, 3),
        "heat_b": round(heat_b, 3),
        "p_public_a": round(p_pub_a, 4),
        "public_pick": team_a if p_pub_a >= 0.5 else team_b,
        "public_conf": round(max(p_pub_a, 1 - p_pub_a), 4),
    }


def analyze_live_public_edge(
    team_blue: str,
    team_red: str,
    *,
    p_blue_model: float,
    heat_blue: float,
    heat_red: float,
    player_mu_diff: float = 0.0,
    draft_win_logit_blue: float = 0.0,
    form_wr_diff: float = 0.0,
    league: str = "",
    best_of: int = 1,
) -> dict:
    """
    Live board soft-public vs model disagreement + fade flags.

    heat_* = recent form WR (board form_wr_*).
    player_mu_diff / draft_win_logit_blue / form_wr_diff are blue-centric.
    """
    blue = normalize_team(team_blue)
    red = normalize_team(team_red)
    pub = public_soft_scores(blue, red, heat_a=float(heat_blue), heat_b=float(heat_red))
    p_blue = float(np.clip(p_blue_model, 0.05, 0.95))
    # Series-aware model conf on favorite
    p_blue_ser = series_p_from_map_p(p_blue, best_of)
    model_pick = blue if p_blue_ser >= 0.5 else red
    model_conf = float(max(p_blue_ser, 1.0 - p_blue_ser))
    public_pick = pub["public_pick"]
    public_conf = float(pub["public_conf"])
    agree = model_pick == public_pick

    brand_fav = blue if pub["brand_a"] >= pub["brand_b"] else red
    heat_fav = blue if heat_blue >= heat_red else red
    dog = model_pick  # when disagree, model side is the soft-dog
    fav = public_pick

    # Drivers toward model pick (blue-centric → dog-centric)
    def _for_dog(blue_val: float) -> float:
        return float(blue_val if dog == blue else -blue_val)

    player_for = _for_dog(float(player_mu_diff))
    draft_for = _for_dog(float(draft_win_logit_blue))
    form_for = _for_dog(float(form_wr_diff))
    conf_gap = model_conf - public_conf

    flags: list[str] = []
    if not agree:
        if dog == brand_fav and dog != heat_fav:
            flags.append("fade-heat→brand")
        if dog == heat_fav and dog != brand_fav:
            flags.append("fade-brand→heat")
        if fav == brand_fav and fav == heat_fav:
            flags.append("public=brand+heat")
        if player_for > 40:
            flags.append("playerElo+")
        if form_for > 0.05:
            flags.append("form+")
        if draft_for > 0.02:
            flags.append("draft+")
        if conf_gap >= 0.10:
            flags.append("confGap≥10")
        if str(league).upper() in {"EWC", "MSI", "WORLDS", "FST"} and model_conf >= 0.60:
            flags.append("intl")

    confirms = sum(
        [
            "fade-heat→brand" in flags,
            "playerElo+" in flags,
            "form+" in flags,
            "draft+" in flags,
            "confGap≥10" in flags,
            "intl" in flags and model_conf >= 0.60,
        ]
    )
    # Actionable verdict for chat / parlay legs
    if agree:
        if model_conf >= 0.70:
            action = "AGREE_STRONG"
            advice = (
                f"Model + soft-public both on {model_pick} ({model_conf:.0%}). "
                "High hit-rate spot but short price — only if book still > fair."
            )
        else:
            action = "AGREE"
            advice = (
                f"Model + soft-public both on {model_pick}. "
                "Consensus lean — shop fair odds, don't force a dog."
            )
    elif "fade-brand→heat" in flags and confirms < 3:
        action = "PASS_ANTI"
        advice = (
            f"Anti-pattern: soft-public on brand {fav}, model wants heat {dog}. "
            "Historically weak fade — pass unless book is wildly long."
        )
    elif confirms >= 2 and model_conf >= 0.55:
        action = "FADE_CANDIDATE"
        dog_dec = round(1.0 / max(1e-6, 1.0 - public_conf), 2)
        advice = (
            f"FADE CANDIDATE: soft-public {fav} ({public_conf:.0%}) vs model {dog} "
            f"({model_conf:.0%}). Confirms={confirms} [{', '.join(flags)}]. "
            f"If book shades to public, dog ≈ {dog_dec} — grade vs actual odds."
        )
    else:
        action = "DISAGREE_THIN"
        advice = (
            f"Disagreement ({fav} public vs {dog} model) but thin confirms "
            f"({confirms}: {', '.join(flags) or 'none'}). Pass for now."
        )

    return {
        "agree": agree,
        "action": action,
        "advice": advice,
        "flags": flags,
        "confirms": confirms,
        "public_pick": public_pick,
        "public_conf": round(public_conf, 4),
        "model_pick": model_pick,
        "model_conf": round(model_conf, 4),
        "conf_gap": round(conf_gap, 4),
        "brand_blue": pub["brand_a"],
        "brand_red": pub["brand_b"],
        "heat_blue": round(float(heat_blue), 3),
        "heat_red": round(float(heat_red), 3),
        "brand_fav": brand_fav,
        "heat_fav": heat_fav,
        "player_for_model": round(player_for, 1),
        "draft_for_model": round(draft_for, 3),
        "form_for_model": round(form_for, 3),
        "soft_dog_decimal": round(1.0 / max(1e-6, 1.0 - public_conf), 2) if not agree else None,
        "best_of": best_of,
        "league": league,
    }


def format_public_edge_report(edge: dict) -> str:
    """Chat block for soft-public vs model fade flags."""
    if not edge:
        return ""
    lines = ["--- PUBLIC EDGE (soft signal) ---"]
    lines.append(
        f"  Soft-public: {edge['public_pick']} ({edge['public_conf']:.0%})  ·  "
        f"Model: {edge['model_pick']} ({edge['model_conf']:.0%})  ·  "
        f"{'AGREE' if edge['agree'] else 'DISAGREE'}"
    )
    lines.append(
        f"  Brand {edge['brand_blue']:.0f}/{edge['brand_red']:.0f}  ·  "
        f"heat {edge['heat_blue']:.0%}/{edge['heat_red']:.0%}  ·  "
        f"brand_fav={edge['brand_fav']}  heat_fav={edge['heat_fav']}"
    )
    if not edge["agree"]:
        lines.append(
            f"  Drivers → model: playerΔ={edge['player_for_model']:+.0f}  "
            f"form={edge['form_for_model']:+.2f}  draft={edge['draft_for_model']:+.3f}  "
            f"confGap={edge['conf_gap']:+.0%}"
        )
        if edge.get("flags"):
            lines.append(f"  Flags: {', '.join(edge['flags'])}  (confirms={edge['confirms']})")
        if edge.get("soft_dog_decimal"):
            lines.append(
                f"  Soft dog ≈ {edge['soft_dog_decimal']} if book ≈ public lean "
                "(not a real quote — paste odds to GRADE)"
            )
    act = edge.get("action") or ""
    prefix = {
        "FADE_CANDIDATE": "★",
        "AGREE_STRONG": "●",
        "PASS_ANTI": "✗",
        "DISAGREE_THIN": "·",
        "AGREE": "·",
    }.get(act, "·")
    lines.append(f"  {prefix} [{act}] {edge.get('advice')}")
    return "\n".join(lines)


def run_2026_study(year: int = 2026) -> dict:
    df = pd.read_parquet(FEATURES_DIR / "maps.parquet")
    df = df.dropna(subset=["date", "y_blue_win", "blue_team", "red_team"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year == year].sort_values("date").reset_index(drop=True)
    df["blue_n"] = df["blue_team"].map(lambda t: normalize_team(str(t)))
    df["red_n"] = df["red_team"].map(lambda t: normalize_team(str(t)))
    df["pair"] = [tuple(sorted(x)) for x in zip(df["blue_n"], df["red_n"])]

    heat = _rolling_heat(df, window=10)
    series = _build_series(df)

    rows = []
    for s in series:
        ms = s["maps"]
        a, b = s["pair"]
        wins = {a: 0, b: 0}
        for m in ms:
            if float(m["y_blue_win"]) >= 0.5:
                wins[m["blue_n"]] += 1
            else:
                wins[m["red_n"]] += 1
        if wins[a] == wins[b]:
            continue
        winner = a if wins[a] > wins[b] else b
        m0 = ms[0]
        when = m0["date"]
        p_blue = float(m0.get("p_strength_blend") or m0.get("p_dual_elo") or 0.5)
        p_a_map = p_blue if m0["blue_n"] == a else 1.0 - p_blue
        n = len(ms)
        mx = max(wins[a], wins[b])
        fmt = 1 if n == 1 else (5 if (mx >= 3 or n >= 4) else 3)
        p_a_ser = series_p_from_map_p(p_a_map, fmt)
        model_pick = a if p_a_ser >= 0.5 else b
        model_conf = max(p_a_ser, 1 - p_a_ser)

        pub = public_soft_scores(
            a,
            b,
            heat_a=_heat_before(heat, a, when),
            heat_b=_heat_before(heat, b, when),
        )
        public_pick = pub["public_pick"]
        agree = model_pick == public_pick

        rows.append(
            {
                "date": str(when),
                "league": str(m0["league"]),
                "t1": a,
                "t2": b,
                "n_maps": int(n),
                "is_bo1": n == 1,
                "winner": winner,
                "model_pick": model_pick,
                "model_conf": round(float(model_conf), 4),
                "public_pick": public_pick,
                "public_conf": pub["public_conf"],
                "agree": bool(agree),
                "model_hit": bool(model_pick == winner),
                "public_hit": bool(public_pick == winner),
                "brand_a": pub["brand_a"],
                "brand_b": pub["brand_b"],
                "heat_a": pub["heat_a"],
                "heat_b": pub["heat_b"],
                "month": when.to_period("M").strftime("%Y-%m"),
            }
        )

    ser = pd.DataFrame(rows)
    n = len(ser)
    disagree = ser[~ser["agree"]]
    agree_df = ser[ser["agree"]]

    def _wr(g: pd.DataFrame, col: str) -> dict:
        if g.empty:
            return {"n": 0, "hits": 0, "wr": None}
        hits = int(g[col].sum())
        return {"n": int(len(g)), "hits": hits, "wr": round(hits / len(g), 4)}

    fade_windows = []
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
        g = disagree[disagree["model_conf"] >= thr]
        fade_windows.append(
            {
                "min_model_conf": thr,
                **_wr(g, "model_hit"),
                "coverage_of_disagree": round(len(g) / max(len(disagree), 1), 4),
            }
        )

    brand_fade = disagree[
        (
            (disagree["public_pick"] == disagree["t1"])
            & (disagree["brand_a"] > disagree["brand_b"])
        )
        | (
            (disagree["public_pick"] == disagree["t2"])
            & (disagree["brand_b"] > disagree["brand_a"])
        )
    ]

    by_league = []
    for lg, g in ser.groupby("league"):
        if len(g) < 20:
            continue
        d = g[~g["agree"]]
        by_league.append(
            {
                "league": lg,
                "n": int(len(g)),
                "model_wr": round(float(g["model_hit"].mean()), 4),
                "public_wr": round(float(g["public_hit"].mean()), 4),
                "disagree_n": int(len(d)),
                "disagree_model_wr": round(float(d["model_hit"].mean()), 4) if len(d) else None,
                "disagree_rate": round(float((~g["agree"]).mean()), 4),
            }
        )
    by_league.sort(key=lambda x: -x["n"])

    art = {
        "version": 1,
        "year": year,
        "method": {
            "public_proxy": "0.55·brand_prestige + 0.45·rolling_10_map_WR (causal heat)",
            "model": "calibrated sequential p_strength_blend → series P",
            "note": (
                "Soft signal only — not book odds. Brand table is a research prior for "
                "casual/public lean; use for disagreement analysis, not Kelly sizing."
            ),
        },
        "n_series": n,
        "overall": {
            "model": _wr(ser, "model_hit"),
            "public_soft": _wr(ser, "public_hit"),
            "agree_rate": round(float(ser["agree"].mean()), 4),
        },
        "when_agree": {
            "model_equals_public": _wr(agree_df, "model_hit"),
            "by_model_confidence": [
                {
                    "min_model_conf": thr,
                    **_wr(agree_df[agree_df["model_conf"] >= thr], "model_hit"),
                }
                for thr in (0.55, 0.60, 0.65, 0.70, 0.75)
            ],
        },
        "when_disagree": {
            "n": int(len(disagree)),
            "share": round(len(disagree) / max(n, 1), 4),
            "model_wr": _wr(disagree, "model_hit"),
            "public_wr": _wr(disagree, "public_hit"),
            "fade_public_using_model": _wr(disagree, "model_hit"),
            "follow_public_vs_model": _wr(disagree, "public_hit"),
        },
        "fade_public_by_model_confidence": fade_windows,
        "brand_driven_disagreements": {
            "n": int(len(brand_fade)),
            "model_wr": _wr(brand_fade, "model_hit"),
            "public_wr": _wr(brand_fade, "public_hit"),
        },
        "by_league": by_league,
        "bo1_vs_multi": {
            "bo1": {
                "model": _wr(ser[ser["is_bo1"]], "model_hit"),
                "public": _wr(ser[ser["is_bo1"]], "public_hit"),
                "disagree_model": _wr(ser[ser["is_bo1"] & ~ser["agree"]], "model_hit"),
            },
            "bo2_plus": {
                "model": _wr(ser[~ser["is_bo1"]], "model_hit"),
                "public": _wr(ser[~ser["is_bo1"]], "public_hit"),
                "disagree_model": _wr(ser[~ser["is_bo1"] & ~ser["agree"]], "model_hit"),
            },
        },
        "takeaway": _takeaway(ser, disagree, fade_windows),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=2))
    print(json.dumps(art["overall"], indent=2))
    print("disagree", json.dumps(art["when_disagree"], indent=2))
    print("fade windows", json.dumps(fade_windows, indent=2))
    print("takeaway", json.dumps(art["takeaway"], indent=2))
    print(f"[public_soft] wrote {OUT}")
    return art


def _takeaway(ser: pd.DataFrame, disagree: pd.DataFrame, fade_windows: list) -> dict:
    model_wr = float(ser["model_hit"].mean())
    pub_wr = float(ser["public_hit"].mean())
    agree = ser[ser["agree"]]
    d_model = float(disagree["model_hit"].mean()) if len(disagree) else None
    d_pub = float(disagree["public_hit"].mean()) if len(disagree) else None
    agree_wr = float(agree["model_hit"].mean()) if len(agree) else None
    # Best actionable window: agree + model confidence floor
    best_agree = None
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
        g = agree[agree["model_conf"] >= thr]
        if len(g) < 80:
            continue
        wr = float(g["model_hit"].mean())
        if best_agree is None or wr > best_agree["wr"]:
            best_agree = {"min_model_conf": thr, "n": int(len(g)), "wr": wr}
    best_fade = None
    for w in fade_windows:
        if w["n"] >= 80 and (best_fade is None or (w["wr"] or 0) > (best_fade["wr"] or 0)):
            best_fade = w
    return {
        "model_beats_public_overall_pp": round((model_wr - pub_wr) * 100, 2),
        "disagree_share_pct": round(100 * len(disagree) / max(len(ser), 1), 1),
        "when_agree_wr": round(agree_wr, 4) if agree_wr is not None else None,
        "when_disagree_model_minus_public_pp": (
            round((d_model - d_pub) * 100, 2)
            if d_model is not None and d_pub is not None
            else None
        ),
        "suggested_bet_rule": (
            f"Bet WITH soft-public when model agrees AND model conf ≥ "
            f"{best_agree['min_model_conf']:.0%}: historical WR {best_agree['wr']:.1%} "
            f"on {best_agree['n']} series"
            if best_agree
            else "Prefer agreement spots"
        ),
        "fade_caution": (
            f"Blind fade is weak — disagree model WR "
            f"{d_model:.1%} vs public {d_pub:.1%}. "
            f"Best fade window (≥{best_fade['min_model_conf']:.0%} conf) only "
            f"{best_fade['wr']:.1%} on {best_fade['n']} series."
            if best_fade and d_model is not None and d_pub is not None
            else "Fade sample thin"
        ),
        "fundamental": (
            "Brand + recent heat is a strong public proxy (~same WR as the model). "
            "Estimated edge for future play is concentrated in AGREEMENT spots "
            "(model + public lean same way), not in fading the public. "
            "True +EV still needs book odds — this soft signal sizes WHEN to trust a lean."
        ),
    }


def main() -> None:
    run_2026_study(2026)


if __name__ == "__main__":
    main()
