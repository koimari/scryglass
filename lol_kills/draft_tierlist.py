#!/usr/bin/env python3
"""
Elo-controlled Blade-Chest tierlist → draft betting edge.

Consumes research artifacts (not a re-fit):
  - champ_tierlist_16_13_blade_chest.json  (scope boards: msi/ewc/lec/lcs)
  - champ_tierlist_side_blind_counter.json (pooled majors, denser)
  - champ_tierlist_patch_window.json       (overall ΔWR fallback)

Waterfall per champ/side: scope → MSI → side_blind → patch_window.
Draft edge = mean(blue tier_pp) − mean(red tier_pp), with lane answerability tax
when opponent sits in answered_by. Soft-blended into board win p (coverage-weighted).
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import MODELS_DIR

BLADE_PATH = MODELS_DIR / "champ_tierlist_16_13_blade_chest.json"
SIDE_BLIND_PATH = MODELS_DIR / "champ_tierlist_side_blind_counter.json"
PATCH_WINDOW_PATH = MODELS_DIR / "champ_tierlist_patch_window.json"

# League → board scope (europe=LEC, americas=LCS, intl separated)
LEAGUE_SCOPE = {
    "LEC": "lec",
    "LCS": "lcs",
    "MSI": "msi",
    "EWC": "ewc",
    "LCK": "msi",
    "LPL": "msi",
    "WORLDS": "msi",
    "FST": "msi",
}

# Soft blend into win stack (avoid double-count vs classic draft_score betas)
MAX_TIER_WEIGHT = 0.14
PP_TO_P = 0.01  # 1pp tier edge → +1pp win before shrink
LANE_ANSWER_TAX_PP = 1.25  # tax when lane opp is in answered_by
# Cap single-champ sway so one Z-outlier (e.g. +20pp) can't dominate the mean
CHAMP_PP_CLIP = 8.0

# Brand-new champs with no OE Elo yet — kit prior only (not a fitted edge).
# answered_by = classic lane answers from kit/role (soft tax when present).
NEW_CHAMP_PRIORS: dict[str, dict] = {
    "Locke": {
        "role": "mid",
        "elo_pp": 0.0,  # neutral until OE sample
        "tier_score": 0.0,
        "n": 0,
        "archetype": "Counter",  # high answerability as new mid assassin
        "tier_label": "new_champ_prior",
        "source": "new_champ_prior",
        "counterable": {
            "score": 62.0,
            "label": "new AP assassin — answerable",
            "cvar_floor": 42.0,
            "gamma": 0.0,
            "answered_by": [
                ["Galio", 70.0],
                ["Lissandra", 65.0],
                ["Poppy", 60.0],
                ["Malzahar", 58.0],
                ["Vex", 55.0],
            ],
            "holds_vs": [["Azir", 55.0], ["Orianna", 52.0], ["Syndra", 50.0]],
        },
        "why": (
            "Locke (26.13 mid AP assassin) — no OE Elo; neutral prior. "
            "Pilot/variance seat; classic answers include Galio."
        ),
        "provisional": True,
    },
}


def _sigmoid(x: float) -> float:
    if x >= 30:
        return 1.0
    if x <= -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def league_to_scope(league: str | None) -> str:
    u = (league or "").strip().upper()
    return LEAGUE_SCOPE.get(u, "msi")


def _flatten_board(side_block: dict | None) -> dict[str, dict]:
    """Tier buckets → champ → entry."""
    out: dict[str, dict] = {}
    if not side_block:
        return out
    board = side_block.get("board") or {}
    for tier, rows in board.items():
        for row in rows or []:
            champ = normalize_champ(str(row.get("champ") or ""))
            if not champ:
                continue
            entry = dict(row)
            entry["champ"] = champ
            entry["tier_label"] = tier
            out[champ] = entry
    return out


@lru_cache(maxsize=1)
def _load_artifacts() -> dict:
    blade: dict = {}
    if BLADE_PATH.exists():
        try:
            blade = json.loads(BLADE_PATH.read_text())
        except Exception:
            blade = {}

    by_scope: dict[str, dict] = {}
    scopes = blade.get("by_scope") or {}
    for key, block in scopes.items():
        if not isinstance(block, dict) or not block.get("blue"):
            continue
        by_scope[key] = {
            "blue": _flatten_board(block.get("blue")),
            "red": _flatten_board(block.get("red")),
            "n_games": int(block.get("n_games") or 0),
            "patch": block.get("patch"),
            "label": block.get("label") or key,
        }
        # Prefer explicit lookup if research refresh wrote it
        for side in ("blue", "red"):
            lookup = (block.get(side) or {}).get("lookup") or {}
            for champ, row in lookup.items():
                c = normalize_champ(str(champ))
                if c and c not in by_scope[key][side]:
                    entry = dict(row)
                    entry["champ"] = c
                    entry["tier_label"] = entry.get("tier_label") or "lookup"
                    by_scope[key][side][c] = entry

    side_blind = {"blue": {}, "red": {}}
    if SIDE_BLIND_PATH.exists():
        try:
            sb = json.loads(SIDE_BLIND_PATH.read_text())
            side_blind["blue"] = _flatten_board(sb.get("blue"))
            side_blind["red"] = _flatten_board(sb.get("red"))
        except Exception:
            pass

    patch_window: dict[str, dict] = {}
    if PATCH_WINDOW_PATH.exists():
        try:
            pw = json.loads(PATCH_WINDOW_PATH.read_text())
            for row in pw.get("overall_ranked") or []:
                c = normalize_champ(str(row.get("champ") or ""))
                if c:
                    patch_window[c] = {
                        "champ": c,
                        "elo_pp": float(row.get("delta_wr_pp") or 0.0),
                        "tier_score": float(row.get("delta_wr_pp") or 0.0),
                        "n": int(row.get("n") or 0),
                        "source": "patch_window",
                        "tier_label": "overall",
                    }
        except Exception:
            pass

    return {
        "by_scope": by_scope,
        "side_blind": side_blind,
        "patch_window": patch_window,
        "prefer_patch": blade.get("prefer_patch"),
        "formula": blade.get("formula"),
    }


def _normalize_entry(entry: dict, *, source: str) -> dict:
    """Map any artifact row → common betting fields."""
    elo = entry.get("elo_pp")
    if elo is None:
        elo = entry.get("delta_wr_pp")
    if elo is None:
        elo = entry.get("resid_pp")
    elo_pp = float(elo or 0.0)
    tier = entry.get("tier_score")
    if tier is None:
        # side_blind: prefer blind/counter path pp when present
        arch = str(entry.get("archetype") or "").lower()
        if arch == "counter" and entry.get("counter_pp") is not None:
            tier = float(entry["counter_pp"])
        elif entry.get("blind_pp") is not None:
            tier = float(entry["blind_pp"])
        else:
            tier = elo_pp
    return {
        "champ": entry.get("champ"),
        "role": entry.get("role"),
        "elo_pp": round(elo_pp, 2),
        "tier_score": round(float(tier), 2),
        "n": int(entry.get("n") or 0),
        "archetype": entry.get("archetype"),
        "tier_label": entry.get("tier_label"),
        "counterability_tax": entry.get("counterability_tax"),
        "impact_pp": entry.get("impact_pp"),
        "counterable": entry.get("counterable"),
        "source": source,
        "why": entry.get("why"),
    }


def lookup_champ(
    champ: str,
    side: str,
    *,
    league: str | None = None,
    arts: dict | None = None,
) -> dict | None:
    """Waterfall lookup for one champ on blue/red."""
    arts = arts or _load_artifacts()
    c = normalize_champ(champ)
    side = "blue" if side.lower().startswith("b") else "red"
    scope = league_to_scope(league)
    by_scope = arts["by_scope"]

    chain: list[tuple[str, dict]] = []
    if scope in by_scope:
        chain.append((f"blade:{scope}", by_scope[scope][side]))
    if scope != "msi" and "msi" in by_scope:
        chain.append(("blade:msi", by_scope["msi"][side]))
    chain.append(("side_blind", arts["side_blind"][side]))

    for source, table in chain:
        if c in table:
            return _normalize_entry(table[c], source=source)

    pw = arts["patch_window"]
    if c in pw:
        return _normalize_entry(pw[c], source="patch_window")
    prior = NEW_CHAMP_PRIORS.get(c)
    if prior:
        entry = dict(prior)
        entry["champ"] = c
        return _normalize_entry(entry, source="new_champ_prior")
    return None


def _answered_by_names(entry: dict | None) -> set[str]:
    if not entry:
        return set()
    cb = entry.get("counterable") or {}
    names = set()
    for pair in cb.get("answered_by") or []:
        if isinstance(pair, (list, tuple)) and pair:
            names.add(normalize_champ(str(pair[0])))
        elif isinstance(pair, str):
            names.add(normalize_champ(pair))
    return names


def _lane_tax(blue_e: dict | None, red_e: dict | None) -> tuple[float, float, str | None]:
    """
    If lane opponent is a known answer, tax the answered side's tier_pp.
    Returns (blue_tax, red_tax, note).
    """
    note = None
    b_tax = r_tax = 0.0
    if not blue_e or not red_e:
        return b_tax, r_tax, note
    b_name = normalize_champ(str(blue_e.get("champ") or ""))
    r_name = normalize_champ(str(red_e.get("champ") or ""))
    if r_name and r_name in _answered_by_names(blue_e):
        b_tax = LANE_ANSWER_TAX_PP
        note = f"{b_name} answered by {r_name} (−{LANE_ANSWER_TAX_PP:.1f}pp blue)"
    if b_name and b_name in _answered_by_names(red_e):
        r_tax = LANE_ANSWER_TAX_PP
        if note:
            note += f"; {r_name} answered by {b_name} (−{LANE_ANSWER_TAX_PP:.1f}pp red)"
        else:
            note = f"{r_name} answered by {b_name} (−{LANE_ANSWER_TAX_PP:.1f}pp red)"
    return b_tax, r_tax, note


def score_draft_tierlist(
    blue: list[str],
    red: list[str],
    *,
    league: str | None = None,
    blue_roles: list[str] | None = None,
    red_roles: list[str] | None = None,
) -> dict:
    """
    Elo/Blade-Chest draft edge for betting.

    Returns edge_pp (blue−red), p_blue_tier, coverage, per-champ rows, blend weight hint.
    """
    arts = _load_artifacts()
    scope = league_to_scope(league)
    blue_n = [normalize_champ(c) for c in blue]
    red_n = [normalize_champ(c) for c in red]

    blue_rows = [lookup_champ(c, "blue", league=league, arts=arts) for c in blue_n]
    red_rows = [lookup_champ(c, "red", league=league, arts=arts) for c in red_n]

    # Role-pair answerability: match by role when available, else by index
    roles_b = blue_roles or [((r or {}).get("role") if r else None) for r in blue_rows]
    roles_r = red_roles or [((r or {}).get("role") if r else None) for r in red_rows]
    lane_notes: list[str] = []
    blue_adj: list[float] = []
    red_adj: list[float] = []
    used_b = used_r = 0
    n_weight_b = n_weight_r = 0.0

    red_by_role: dict[str, dict] = {}
    for r, role in zip(red_rows, roles_r):
        if r and role:
            red_by_role[str(role)] = r

    for i, (b, role) in enumerate(zip(blue_rows, roles_b)):
        if not b:
            continue
        used_b += 1
        opp = red_by_role.get(str(role)) if role else None
        if opp is None and i < len(red_rows):
            opp = red_rows[i]
        b_tax, _, note = _lane_tax(b, opp)
        if note:
            lane_notes.append(note)
        pp = max(-CHAMP_PP_CLIP, min(CHAMP_PP_CLIP, float(b["tier_score"]) - b_tax))
        blue_adj.append(pp)
        n_weight_b += min(1.0, math.sqrt(max(b["n"], 1) / 20.0))

    blue_by_role: dict[str, dict] = {}
    for b, role in zip(blue_rows, roles_b):
        if b and role:
            blue_by_role[str(role)] = b

    for i, (r, role) in enumerate(zip(red_rows, roles_r)):
        if not r:
            continue
        used_r += 1
        opp = blue_by_role.get(str(role)) if role else None
        if opp is None and i < len(blue_rows):
            opp = blue_rows[i]
        _, r_tax, note = _lane_tax(opp, r)
        if note and note not in lane_notes:
            lane_notes.append(note)
        pp = max(-CHAMP_PP_CLIP, min(CHAMP_PP_CLIP, float(r["tier_score"]) - r_tax))
        red_adj.append(pp)
        n_weight_r += min(1.0, math.sqrt(max(r["n"], 1) / 20.0))

    mean_b = sum(blue_adj) / len(blue_adj) if blue_adj else 0.0
    mean_r = sum(red_adj) / len(red_adj) if red_adj else 0.0
    edge_pp = mean_b - mean_r

    coverage = (used_b + used_r) / max(len(blue_n) + len(red_n), 1)
    sample_conf = 0.0
    if used_b + used_r:
        sample_conf = (n_weight_b + n_weight_r) / max(used_b + used_r, 1)
    conf = float(min(0.95, max(0.05, 0.55 * coverage + 0.45 * sample_conf)))

    # Shrink edge toward 0 by confidence; map to p
    edge_shrunk = edge_pp * conf
    p_blue = 0.5 + edge_shrunk * PP_TO_P
    p_blue = float(min(0.92, max(0.08, p_blue)))

    sources = sorted(
        {
            (r or {}).get("source")
            for r in (blue_rows + red_rows)
            if r and r.get("source")
        }
    )
    missing = [c for c, r in zip(blue_n + red_n, blue_rows + red_rows) if r is None]

    return {
        "scope": scope,
        "league": league,
        "edge_pp": round(edge_pp, 2),
        "edge_pp_shrunk": round(edge_shrunk, 2),
        "mean_blue_pp": round(mean_b, 2),
        "mean_red_pp": round(mean_r, 2),
        "p_blue_tier": round(p_blue, 4),
        "coverage": round(coverage, 3),
        "confidence": round(conf, 3),
        "blend_weight": round(MAX_TIER_WEIGHT * conf * coverage, 4),
        "sources": sources,
        "missing": missing,
        "lane_notes": lane_notes[:6],
        "blue": blue_rows,
        "red": red_rows,
        "prefer_patch": arts.get("prefer_patch"),
        "note": (
            "Blade-Chest Elo tier_score (elo_pp − cb tax + OE impact); "
            "waterfall scope→MSI→side_blind→patch_window; soft board blend only."
        ),
    }


def blend_win_with_tierlist(p_blue: float, tier: dict) -> tuple[float, dict]:
    """Coverage-weighted blend of board p with tierlist p. Returns (p, meta)."""
    w = float(tier.get("blend_weight") or 0.0)
    if w <= 1e-6 or not tier.get("coverage"):
        return float(p_blue), {"applied": False, "weight": 0.0, "p_before": p_blue}

    p_tier = float(tier["p_blue_tier"])
    # Logit blend keeps extremes better behaved than linear
    z = (1.0 - w) * _logit(float(p_blue)) + w * _logit(p_tier)
    p_out = float(min(0.95, max(0.05, _sigmoid(z))))
    return p_out, {
        "applied": True,
        "weight": round(w, 4),
        "p_before": round(float(p_blue), 4),
        "p_tier": round(p_tier, 4),
        "p_after": round(p_out, 4),
        "delta_pp": round(100.0 * (p_out - float(p_blue)), 2),
        "edge_pp": tier.get("edge_pp_shrunk"),
    }


def format_tierlist_report(tier: dict, *, team_blue: str = "Blue", team_red: str = "Red") -> str:
    lines = [
        "--- Elo / Blade-Chest tierlist ---",
        (
            f"  Scope={tier.get('scope')}  edge {tier.get('edge_pp_shrunk'):+.1f}pp "
            f"(raw {tier.get('edge_pp'):+.1f})  "
            f"p_tier={tier.get('p_blue_tier'):.1%}  "
            f"cov={tier.get('coverage'):.0%}  conf={tier.get('confidence'):.0%}  "
            f"blend_w={tier.get('blend_weight')}"
        ),
        (
            f"  {team_blue} mean {tier.get('mean_blue_pp'):+.1f}pp  |  "
            f"{team_red} mean {tier.get('mean_red_pp'):+.1f}pp  |  "
            f"sources={','.join(tier.get('sources') or []) or '—'}"
        ),
    ]
    blend = tier.get("blend") or {}
    if blend.get("applied"):
        lines.append(
            f"  Win blend: {blend['p_before']:.1%} → {blend['p_after']:.1%} "
            f"({blend['delta_pp']:+.1f}pp @ w={blend['weight']})"
        )

    def _side(label: str, rows: list) -> None:
        bits = []
        for r in rows or []:
            if not r:
                continue
            raw = float(r["tier_score"])
            shown = max(-CHAMP_PP_CLIP, min(CHAMP_PP_CLIP, raw))
            tag = f"{r.get('tier_label') or r.get('source')}"
            if abs(raw) > CHAMP_PP_CLIP + 1e-9:
                bits.append(f"{r['champ']} {shown:+.1f}*{tag}")
            else:
                bits.append(f"{r['champ']} {shown:+.1f}[{tag}]")
        lines.append(f"  {label}: " + (", ".join(bits) if bits else "—"))

    _side(str(team_blue), tier.get("blue") or [])
    _side(str(team_red), tier.get("red") or [])
    if tier.get("lane_notes"):
        lines.append("  Lane answers: " + "; ".join(tier["lane_notes"][:3]))
    if tier.get("missing"):
        lines.append(f"  Missing lookup: {', '.join(tier['missing'])}")
    return "\n".join(lines)


def clear_cache() -> None:
    _load_artifacts.cache_clear()
