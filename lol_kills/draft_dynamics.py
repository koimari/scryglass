#!/usr/bin/env python3
"""
Multi-angle draft dynamics for betting (beyond beatdown/control).

Each lens is an analogy with a LoL read + betting action line.
Beatdown/control (Flores) remains primary; these are complementary seats.

Lenses
  1. beatdown_control   — Flores: who must convert vs who must reach late
  2. tempo_inevitability — short-game vs long-game preference
  3. goldfish            — unanswered win speed (MTG goldfish)
  4. wincon_trichotomy   — siege / teamfight / split (LoL-native)
  5. pilot_carry         — hypercarry dependency (pilot vs committee)
  6. initiative          — who is allowed to start fights
  7. combo_fair          — must-land wombo vs fair continuous damage
  8. variance_seat       — chaos preference (dog wants variance)
  9. spike_calendar      — who owns grub→herald→soul→baron windows
 10. magriel             — ahead→safe / behind→bold (needs live gold)

  from lol_kills.draft_dynamics import analyze_draft_dynamics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lol_kills.draft_archetypes import champ_tags, draft_archetype_features, side_archetype_counts
from lol_kills.draft_phase_score import (
    BEATDOWN_WEIGHTS,
    INEVITABILITY_WEIGHTS,
    _axis,
    assign_roles,
    draft_score_composite,
)
from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import MODELS_DIR
from lol_kills.live_win import kill_conc_from_draft

VAL_PATH = MODELS_DIR / "draft_dynamics_validation.json"

# Wincon weights (trichotomy)
SIEGE_W = {"poke_siege": 1.0, "siege_specialist": 1.2, "control_mage": 0.35, "hypercarry_adc": 0.25}
TEAMFIGHT_W = {"teamfight_aoe": 1.0, "engage": 0.55, "control_mage": 0.4, "peel_enchanter": 0.25}
SPLIT_W = {"splitpush": 1.2, "skirmisher": 0.45, "assassin": 0.35, "scaling_late": 0.2}

# Combo (must-land) vs fair (continuous)
COMBO_W = {"engage": 0.7, "burst_mage": 0.9, "teamfight_aoe": 0.45, "pick": 0.35}
FAIR_W = {"poke_siege": 0.7, "hypercarry_adc": 0.6, "control_mage": 0.5, "scaling_late": 0.4, "skirmisher": 0.3}

# Initiative / catch
INIT_W = {"engage": 1.0, "pick": 0.85, "roam": 0.4, "assassin": 0.35}
REACT_W = {"peel_enchanter": 0.9, "control_mage": 0.45, "tank_frontline": 0.35}

# Spike calendar ownership (soft priors by archetype)
SPIKE_OWN = {
    "grubs_10": {"early_snowball": 0.5, "engage": 0.25, "roam": 0.35, "assassin": 0.3},
    "herald_14": {"early_snowball": 0.4, "poke_siege": 0.35, "engage": 0.3, "roam": 0.25},
    "soul_25": {"teamfight_aoe": 0.45, "control_mage": 0.35, "engage": 0.3, "hypercarry_adc": 0.25},
    "baron_25": {"teamfight_aoe": 0.4, "scaling_late": 0.35, "hypercarry_adc": 0.4, "poke_siege": 0.25},
}


def _side_w(counts: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights.get(a, 0.0) * float(counts.get(f"arch_{a}", 0.0)) for a in weights)


def _load_validation() -> dict:
    if not VAL_PATH.exists():
        return {}
    try:
        return json.loads(VAL_PATH.read_text())
    except Exception:
        return {}


def _norm_pair(a: float, b: float) -> tuple[float, float, float]:
    """Return (a, b, a-b)."""
    return round(a, 3), round(b, 3), round(a - b, 3)


def _winner(diff: float, blue_label: str, red_label: str, eps: float = 0.35) -> str:
    if diff >= eps:
        return f"blue_{blue_label}"
    if diff <= -eps:
        return f"red_{red_label}"
    return "contested"


def lens_beatdown(powers: dict, composite: dict) -> dict:
    roles = composite.get("beatdown") or assign_roles(powers)
    return {
        "id": "beatdown_control",
        "analogy": "Mike Flores — Who's the Beatdown?",
        "blue": roles.get("beatdown_side") == "blue" and "beatdown" or "control",
        "red": roles.get("beatdown_side") == "red" and "beatdown" or "control",
        "beatdown_side": roles.get("beatdown_side"),
        "control_side": roles.get("control_side"),
        "plan": roles.get("plan"),
        "misassign_risk": roles.get("misassign_risk"),
        "powers": roles.get("powers") or {
            "beatdown_diff": powers["beatdown_diff"],
            "inev_diff": powers["inev_diff"],
        },
        "curve": composite.get("curve"),
        "buckets": {
            t: {
                "p_blue": (composite.get("buckets") or {}).get(t, {}).get("p_blue"),
                "score_blue": (composite.get("buckets") or {}).get(t, {}).get("draft_score_blue"),
            }
            for t in ("10", "15", "20", "25")
        },
        "bet": (
            f"{str(roles.get('beatdown_side')).upper()} must convert by ~15–20 "
            f"(towers/drags/gold). {str(roles.get('control_side')).upper()} wants even/long. "
            "Live: if beatdown is ahead @15 → hold chalk; if behind → dog’s map."
        ),
        "confidence": "medium_high",
    }


def lens_tempo_inevitability(powers: dict, bc: dict, rc: dict) -> dict:
    # Tempo ≈ beatdown; inevitability ≈ late axis (already computed)
    tempo_b, tempo_r = powers["beatdown_blue"], powers["beatdown_red"]
    inev_b, inev_r = powers["inev_blue"], powers["inev_red"]
    # Preference: tempo_share = tempo / (tempo+inev)
    def share(t, i):
        s = t + i
        return t / s if s > 1e-6 else 0.5

    sb, sr = share(tempo_b, inev_b), share(tempo_r, inev_r)
    # Map length lean: higher combined inevitability → longer; higher tempo sum → shorter
    length_lean = (inev_b + inev_r) - (tempo_b + tempo_r)  # >0 → over/long bias
    return {
        "id": "tempo_inevitability",
        "analogy": "Tempo vs inevitability (extended Flores / control theory)",
        "tempo_blue": round(tempo_b, 3),
        "tempo_red": round(tempo_r, 3),
        "inev_blue": round(inev_b, 3),
        "inev_red": round(inev_r, 3),
        "tempo_share_blue": round(sb, 3),
        "tempo_share_red": round(sr, 3),
        "length_lean": round(length_lean, 3),
        "length_bias": "longer" if length_lean > 0.6 else ("shorter" if length_lean < -0.6 else "neutral"),
        "bet": (
            "Length lean "
            + ("UNDER / shorter maps" if length_lean < -0.6 else "OVER / longer maps" if length_lean > 0.6 else "no strong length lean")
            + f". Tempo share blue={sb:.2f} red={sr:.2f}."
        ),
        "confidence": "medium",
    }


def lens_goldfish(bc: dict, rc: dict, powers: dict) -> dict:
    """Unanswered win speed — who goldfishes faster if the other misfolds."""
    # Fast clock: early_snowball + assassin + engage; slow: scaling + hypercarry
    fast_w = {"early_snowball": 1.0, "assassin": 0.9, "engage": 0.4, "roam": 0.35, "burst_mage": 0.4}
    slow_w = {"scaling_late": 0.9, "hypercarry_adc": 0.7, "control_mage": 0.5}
    fb, fr = _side_w(bc, fast_w), _side_w(rc, fast_w)
    sb, sr = _side_w(bc, slow_w), _side_w(rc, slow_w)
    # goldfish score = fast - 0.6*slow (higher = faster clock)
    gb, gr = fb - 0.6 * sb, fr - 0.6 * sr
    diff = gb - gr
    faster = "blue" if diff > 0.25 else ("red" if diff < -0.25 else "even")
    # Convert to rough "must win by minute" prior
    # higher goldfish → earlier deadline for that side if they are beatdown
    deadline = 18 if max(gb, gr) > 2.0 else (22 if max(gb, gr) > 1.2 else 28)
    return {
        "id": "goldfish",
        "analogy": "MTG goldfish — unanswered win speed",
        "goldfish_blue": round(gb, 3),
        "goldfish_red": round(gr, 3),
        "faster_side": faster,
        "diff": round(diff, 3),
        "pressure_deadline_min": deadline,
        "bet": (
            f"{faster.upper()} goldfishes faster (diff {diff:+.2f}). "
            f"If they are also beatdown, treat ~{deadline} as conversion deadline; "
            "missed deadline → fade them live."
        ),
        "confidence": "medium",
    }


def lens_trichotomy(bc: dict, rc: dict) -> dict:
    def triple(c):
        return {
            "siege": _side_w(c, SIEGE_W),
            "teamfight": _side_w(c, TEAMFIGHT_W),
            "split": _side_w(c, SPLIT_W),
        }

    b, r = triple(bc), triple(rc)

    def primary(t: dict) -> str:
        return max(t, key=t.get)

    pb, pr = primary(b), primary(r)
    clash = pb != pr
    return {
        "id": "wincon_trichotomy",
        "analogy": "LoL-native — siege / teamfight / split",
        "blue": {k: round(v, 3) for k, v in b.items()},
        "red": {k: round(v, 3) for k, v in r.items()},
        "primary_blue": pb,
        "primary_red": pr,
        "clash": clash,
        "bet": (
            f"Blue wincon={pb}, red={pr}"
            + (". Clash — wrong-plan tax if the map state forces the other script." if clash else " (same lane — mirror wincon).")
            + " Siege likes towers/herald; TF likes dragon soul/baron setups; split likes side-lane KPIs."
        ),
        "confidence": "medium",
    }


def lens_pilot(blue: list[str], red: list[str]) -> dict:
    conc = kill_conc_from_draft(blue, red)
    # Identify top carry tag presence
    def top_carry(champs: list[str]) -> tuple[str | None, float]:
        best, name = 0.0, None
        table = {}
        try:
            raw = json.loads((MODELS_DIR / "champ_kill_concentration.json").read_text())
            table = raw.get("champs") or {}
        except Exception:
            pass
        for c in champs:
            cn = normalize_champ(c)
            share = float((table.get(cn) or {}).get("mean_share", 0.2))
            if share > best:
                best, name = share, cn
        return name, best

    bc, bshare = top_carry(blue)
    rc, rshare = top_carry(red)
    blue_pilot = bool(conc.get("blue_hypercarry"))
    red_pilot = bool(conc.get("red_hypercarry"))
    return {
        "id": "pilot_carry",
        "analogy": "Pilot vs committee — hypercarry dependency",
        "kill_conc_blue": conc.get("kill_conc_blue"),
        "kill_conc_red": conc.get("kill_conc_red"),
        "max_carry_blue": conc.get("max_carry_blue"),
        "max_carry_red": conc.get("max_carry_red"),
        "blue_hypercarry": blue_pilot,
        "red_hypercarry": red_pilot,
        "blue_carry_champ": bc,
        "red_carry_champ": rc,
        "bet": (
            ("Blue is pilot-dependent"
             if blue_pilot else "Blue is committee damage")
            + f" ({bc} share≈{bshare:.2f}); "
            + ("red is pilot-dependent"
               if red_pilot else "red is committee")
            + f" ({rc} share≈{rshare:.2f}). "
            "Pilot behind @15 needs shutdown gold or the map is chalk to the lead."
        ),
        "confidence": "medium_high",
    }


def lens_initiative(bc: dict, rc: dict) -> dict:
    ib, ir = _side_w(bc, INIT_W), _side_w(rc, INIT_W)
    rb, rr = _side_w(bc, REACT_W), _side_w(rc, REACT_W)
    diff = (ib - rb) - (ir - rr)
    who = _winner(diff, "initiative", "initiative", eps=0.4)
    return {
        "id": "initiative",
        "analogy": "Fighter-game initiative — who starts fights",
        "init_blue": round(ib, 3),
        "init_red": round(ir, 3),
        "react_blue": round(rb, 3),
        "react_red": round(rr, 3),
        "net_diff": round(diff, 3),
        "owner": who,
        "bet": (
            f"Initiative lean: {who.replace('_', ' ')} (net {diff:+.2f}). "
            "Initiative side should force; reactive side wants peel/zone and to punish over-engage. "
            "Live first-blood / first-engage success updates this hard."
        ),
        "confidence": "medium",
    }


def lens_combo_fair(bc: dict, rc: dict) -> dict:
    cb, cr = _side_w(bc, COMBO_W), _side_w(rc, COMBO_W)
    fb, fr = _side_w(bc, FAIR_W), _side_w(rc, FAIR_W)

    def seat(c, f):
        if c - f >= 0.5:
            return "combo"
        if f - c >= 0.5:
            return "fair"
        return "mixed"

    sb, sr = seat(cb, fb), seat(cr, fr)
    return {
        "id": "combo_fair",
        "analogy": "MTG combo vs fair — must-land spike vs continuous",
        "combo_blue": round(cb, 3),
        "combo_red": round(cr, 3),
        "fair_blue": round(fb, 3),
        "fair_red": round(fr, 3),
        "seat_blue": sb,
        "seat_red": sr,
        "bet": (
            f"Blue={sb}, red={sr}. "
            "Combo seats need a clean engage window (miss → fade). "
            "Fair seats grind plates/siege and punish failed all-ins."
        ),
        "confidence": "medium",
    }


def lens_variance(
    bc: dict,
    rc: dict,
    *,
    p_blue_pre: float | None,
) -> dict:
    """Chaos preference — assassin/early/skirmish sum. Dog wants high variance."""
    chaos_w = {"assassin": 1.0, "early_snowball": 0.7, "skirmisher": 0.5, "pick": 0.4, "burst_mage": 0.35}
    stable_w = {"control_mage": 0.6, "peel_enchanter": 0.5, "tank_frontline": 0.4, "poke_siege": 0.35}
    chaos = _side_w(bc, chaos_w) + _side_w(rc, chaos_w)
    stable = _side_w(bc, stable_w) + _side_w(rc, stable_w)
    map_chaos = chaos - 0.7 * stable
    # Per-side chaos (who brings the entropy)
    cb, cr = _side_w(bc, chaos_w), _side_w(rc, chaos_w)
    # Favorite wants low variance
    fav = None
    if p_blue_pre is not None:
        fav = "blue" if p_blue_pre >= 0.5 else "red"
    dog = ("red" if fav == "blue" else "blue") if fav else None
    dog_chaos = None
    if dog == "blue":
        dog_chaos = cb - cr
    elif dog == "red":
        dog_chaos = cr - cb
    align = None
    if dog is not None and dog_chaos is not None:
        align = "dog_has_chaos" if dog_chaos > 0.35 else ("fav_has_chaos" if dog_chaos < -0.35 else "even")
    return {
        "id": "variance_seat",
        "analogy": "Poker variance seat — dog wants chaos, chalk wants clean",
        "map_chaos": round(map_chaos, 3),
        "chaos_blue": round(cb, 3),
        "chaos_red": round(cr, 3),
        "favorite": fav,
        "dog": dog,
        "alignment": align,
        "kill_total_bias": "higher" if map_chaos > 0.8 else ("lower" if map_chaos < -0.8 else "neutral"),
        "bet": (
            f"Map chaos {map_chaos:+.2f} → kill totals bias {('OVER' if map_chaos > 0.8 else 'UNDER' if map_chaos < -0.8 else 'flat')}. "
            + (
                f"Dog={dog} alignment={align}: "
                + ("good dog script (chaos on the underdog)." if align == "dog_has_chaos"
                   else "chalk owns the chaos — dog needs a clean steal or live flip."
                   if align == "fav_has_chaos" else "variance even.")
                if dog
                else "Pass p_pre to tag dog/chalk variance seats."
            )
        ),
        "confidence": "medium",
    }


def lens_spike_calendar(bc: dict, rc: dict) -> dict:
    windows = {}
    for name, w in SPIKE_OWN.items():
        bb, rr = _side_w(bc, w), _side_w(rc, w)
        diff = bb - rr
        windows[name] = {
            "blue": round(bb, 3),
            "red": round(rr, 3),
            "diff": round(diff, 3),
            "owner": "blue" if diff > 0.3 else ("red" if diff < -0.3 else "even"),
        }
    # Narrative chain
    chain = [f"{k}:{v['owner']}" for k, v in windows.items()]
    return {
        "id": "spike_calendar",
        "analogy": "Objective clock — who’s the beatdown *this window*",
        "windows": windows,
        "chain": chain,
        "bet": (
            "Window owners: "
            + ", ".join(chain)
            + ". Bet/live: fade a side that loses its owned spike (e.g. early side losing Herald)."
        ),
        "confidence": "low_medium",
    }


def lens_magriel(gold_diff: float | None, beatdown_side: str | None) -> dict:
    """Ahead → safe; behind → bold. Needs live gold."""
    if gold_diff is None:
        return {
            "id": "magriel",
            "analogy": "Magriel (backgammon) — ahead play safe, behind play bold",
            "status": "pregame",
            "bet": "Live only: when ahead, take guaranteed towers/drags (safe); when behind, force volatile fights (bold). Mis-Magriel = throwing a won map.",
            "confidence": "medium_high_when_live",
        }
    ahead = "blue" if gold_diff > 500 else ("red" if gold_diff < -500 else "even")
    # If beatdown is ahead → correct (press but not int); if control ahead → safe siege
    advice = "even — reset to plan"
    if ahead == "blue":
        advice = "BLUE ahead → Magriel SAFE (towers, vision, deny). Bold only if needed to close."
    elif ahead == "red":
        advice = "RED ahead → Magriel SAFE. Trailing side should take bold variance."
    return {
        "id": "magriel",
        "analogy": "Magriel (backgammon) — ahead play safe, behind play bold",
        "status": "live",
        "gold_diff": gold_diff,
        "ahead_side": ahead,
        "beatdown_side": beatdown_side,
        "bet": advice,
        "confidence": "medium_high",
    }


def analyze_draft_dynamics(
    blue: list[str],
    red: list[str],
    *,
    league: str | None = None,
    elo_diff: float | None = None,
    p_blue_pre: float | None = None,
    gold_diff: float | None = None,
) -> dict[str, Any]:
    """
    Full multi-lens draft analysis for betting.

    Returns lenses + ranked betting implications + OE validation pointers.
    """
    blue = [normalize_champ(c) for c in blue]
    red = [normalize_champ(c) for c in red]
    feats = draft_archetype_features(blue, red)
    bc, rc = side_archetype_counts(blue), side_archetype_counts(red)
    powers = {
        "beatdown_blue": _axis(feats, BEATDOWN_WEIGHTS, "blue"),
        "beatdown_red": _axis(feats, BEATDOWN_WEIGHTS, "red"),
        "beatdown_diff": _axis(feats, BEATDOWN_WEIGHTS, "diff"),
        "inev_blue": _axis(feats, INEVITABILITY_WEIGHTS, "blue"),
        "inev_red": _axis(feats, INEVITABILITY_WEIGHTS, "red"),
        "inev_diff": _axis(feats, INEVITABILITY_WEIGHTS, "diff"),
    }
    composite = draft_score_composite(blue, red, league=league, elo_diff=elo_diff)
    if p_blue_pre is None:
        # Prefer board stack when caller passes; else Elo-tinged composite
        p_blue_pre = float(composite.get("p_blue_draft") or 0.5)
        if elo_diff is not None:
            # mild blend toward Elo for dog/chalk tagging only
            from lol_kills.draft_score import sigmoid

            p_elo = sigmoid(float(elo_diff) / 400.0 * 1.2)
            p_blue_pre = 0.55 * p_elo + 0.45 * p_blue_pre

    lenses = [
        lens_beatdown(powers, composite),
        lens_tempo_inevitability(powers, bc, rc),
        lens_goldfish(bc, rc, powers),
        lens_trichotomy(bc, rc),
        lens_pilot(blue, red),
        lens_initiative(bc, rc),
        lens_combo_fair(bc, rc),
        lens_variance(bc, rc, p_blue_pre=p_blue_pre),
        lens_spike_calendar(bc, rc),
        lens_magriel(gold_diff, (composite.get("beatdown") or {}).get("beatdown_side")),
    ]

    # Priority betting cards (scannable)
    cards = []
    for L in lenses:
        cards.append(
            {
                "lens": L["id"],
                "analogy": L.get("analogy"),
                "action": L.get("bet"),
                "confidence": L.get("confidence"),
            }
        )

    validation = _load_validation()
    return {
        "version": 1,
        "blue": blue,
        "red": red,
        "p_blue_pre_used": round(float(p_blue_pre), 4),
        "composite_draft": {
            "p_blue": composite.get("p_blue_draft"),
            "score_blue": composite.get("draft_score_blue"),
            "curve": composite.get("curve"),
        },
        "lenses": {L["id"]: L for L in lenses},
        "cards": cards,
        "validation": {
            "path": str(VAL_PATH) if VAL_PATH.exists() else None,
            "n_maps": validation.get("n_maps"),
            "summary": validation.get("summary"),
        },
        "note": (
            "Multi-angle draft dynamics for betting. Beatdown/control is primary; "
            "other lenses are complementary. Not a standalone odds model."
        ),
    }


def format_dynamics_report(dyn: dict, *, team_blue: str = "Blue", team_red: str = "Red") -> str:
    lines = [f"=== DRAFT DYNAMICS  {team_blue} (blue) vs {team_red} (red) ===", ""]
    for card in dyn.get("cards") or []:
        lines.append(f"• [{card['lens']}] ({card.get('confidence')})")
        lines.append(f"  {card['action']}")
        lines.append("")
    bd = (dyn.get("lenses") or {}).get("beatdown_control") or {}
    if bd.get("buckets"):
        bits = "  ".join(
            f"@{t} {bd['buckets'][t]['score_blue']}"
            for t in ("10", "15", "20", "25")
            if bd["buckets"].get(t, {}).get("score_blue") is not None
        )
        lines.append(f"Draft curve: {bits}  curve={bd.get('curve')}")
    val = dyn.get("validation") or {}
    if val.get("summary"):
        lines.append(f"OE validation: {val['summary']}")
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blue", required=True)
    ap.add_argument("--red", required=True)
    ap.add_argument("--league", default="EWC")
    ap.add_argument("--elo-diff", type=float, default=None)
    ap.add_argument("--p-blue", type=float, default=None)
    ap.add_argument("--gold-diff", type=float, default=None)
    args = ap.parse_args()
    blue = [c.strip() for c in args.blue.split(",") if c.strip()]
    red = [c.strip() for c in args.red.split(",") if c.strip()]
    dyn = analyze_draft_dynamics(
        blue,
        red,
        league=args.league,
        elo_diff=args.elo_diff,
        p_blue_pre=args.p_blue,
        gold_diff=args.gold_diff,
    )
    print(format_dynamics_report(dyn))
    print(json.dumps({"lenses": list(dyn["lenses"]), "validation": dyn["validation"]}, indent=2))


if __name__ == "__main__":
    main()
