"""Economic decision layer: EV, fractional Kelly, bet GRADE, disagreement, odds journal."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

# Letter ranks for combo downgrades (A+ best … F worst)
_GRADE_RANK = {
    "A+": 12,
    "A": 11,
    "A-": 10,
    "B+": 9,
    "B": 8,
    "B-": 7,
    "C+": 6,
    "C": 5,
    "C-": 4,
    "D": 3,
    "F": 1,
}
_RANK_GRADE = {v: k for k, v in _GRADE_RANK.items()}


def implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def ev_per_unit(p: float, decimal_odds: float) -> float:
    """Expected value of staking 1 unit at decimal odds."""
    return p * decimal_odds - 1.0


def edge_pp(p: float, decimal_odds: float) -> float:
    """Model prob minus implied prob, in percentage points."""
    return (p - implied_prob(decimal_odds)) * 100.0


def kelly_fraction(p: float, decimal_odds: float, fraction: float = 0.5, cap: float = 0.05) -> float:
    """
    Fractional Kelly for decimal odds.
    f* = (bp - q) / b  where b = odds-1, q = 1-p.
    Default half-Kelly with 5% bankroll cap.
    """
    if decimal_odds <= 1.0 or p <= 0:
        return 0.0
    b = decimal_odds - 1.0
    q = 1.0 - p
    full = (b * p - q) / b
    if full <= 0:
        return 0.0
    return float(min(full * fraction, cap))


def correlation_haircut(kelly: float, n_same_map_legs: int, rho: float = 0.35) -> float:
    """
    Shrink Kelly when parlays / multiple legs share the same map.
    Conservative: divide by 1 + (n-1)*rho.
    """
    if n_same_map_legs <= 1:
        return kelly
    return kelly / (1.0 + (n_same_map_legs - 1) * rho)


def _downgrade(grade: str, steps: int = 1) -> str:
    """Drop letter quality by `steps` notches (A → A- → B+ …)."""
    rank = _GRADE_RANK.get(grade, 3)
    # Prefer stepping through known ranks; skip gaps toward next lower letter
    target = rank - steps
    while target >= 1 and target not in _RANK_GRADE:
        target -= 1
    return _RANK_GRADE.get(max(target, 1), "F")


def grade_bet(
    p: float,
    decimal_odds: float,
    *,
    ev: float | None = None,
    edge: float | None = None,
    book_outside_interval: bool | None = None,
) -> dict:
    """
    Risk/ROI letter grade beyond raw confidence.

    A  punch hard — EV≥20% and p≥60%
    B  good spot  — EV≥10% and p≥55%, or EV≥25% with p≥45%
    C  thin       — EV>0 but softer hit rate / edge
    D  pass       — EV≤0 or book inside model band with thin edge
    F  trap       — clearly −EV

    Optional +/− when near band edges.
    """
    if decimal_odds <= 1.0 or p <= 0:
        return {"grade": "F", "grade_why": "invalid odds/p"}

    ev_v = float(ev if ev is not None else ev_per_unit(p, decimal_odds))
    edge_v = float(edge if edge is not None else edge_pp(p, decimal_odds))

    if ev_v <= -0.02:
        base, why = "F", "−EV trap"
    elif ev_v <= 0:
        base, why = "D", "no edge / pass"
    elif ev_v >= 0.20 and p >= 0.60:
        base, why = "A", "strong EV + hit rate"
    elif (ev_v >= 0.10 and p >= 0.55) or (ev_v >= 0.25 and p >= 0.45):
        base, why = "B", "good spot"
    else:
        base, why = "C", "thin / speculative"

    # Soft modifiers
    grade = base
    if base == "A":
        if ev_v >= 0.35 and p >= 0.70:
            grade = "A+"
        elif p < 0.65 or ev_v < 0.28:
            grade = "A-"
    elif base == "B":
        if ev_v >= 0.18 and p >= 0.58:
            grade = "B+"
        elif p < 0.52:
            grade = "B-"
    elif base == "C":
        if ev_v >= 0.08 and p >= 0.52:
            grade = "C+"
        elif p < 0.45:
            grade = "C-"

    # Conformal: inside a (often wide) band → uncertainty haircut, not auto-pass
    if book_outside_interval is False and ev_v > 0:
        if edge_v < 5.0:
            grade, why = "D", "book inside band, thin edge"
        else:
            grade = _downgrade(grade, 1)
            why = f"{why}; inside band −1"

    return {
        "grade": grade,
        "grade_why": why,
        "grade_ev": round(ev_v, 4),
        "grade_p": round(p, 4),
    }


def grade_combo(
    rows: Iterable[dict],
    *,
    rho: float = 0.35,
    same_map: bool = True,
) -> dict | None:
    """
    Portfolio grade for 2+ priced legs (equal 1u stakes assumed).

    Uses mean EV as portfolio ROI and a ρ-mixed joint P(all win).
    Same-map tickets take a one-notch haircut; strong low-wipe tickets
    can also be anchored off the best single-leg grade.
    """
    plus = [r for r in rows if (r.get("ev") or 0) > 0 and r.get("p") and r.get("odds")]
    if len(plus) < 2:
        return None

    n = len(plus)
    portfolio_ev = sum(float(r["ev"]) for r in plus) / n
    p_indep = 1.0
    q_indep = 1.0
    for r in plus:
        p_i = float(r["p"])
        p_indep *= p_i
        q_indep *= 1.0 - p_i
    p_min = min(float(r["p"]) for r in plus)
    # Mix independence toward shared-map dependence for joint win
    p_both = (1.0 - rho) * p_indep + rho * p_min
    # Wipe: start from independence, modest positive-corr inflation (not toward 1-p_min)
    p_wipe = min(1.0, q_indep * (1.0 + rho))
    p_any = 1.0 - p_wipe

    # Grade ROI vs joint hit rate (ρ already mixes dependence — no extra letter drop here)
    g = grade_bet(p_both, 1.0 + max(portfolio_ev, 1e-6), ev=portfolio_ev)
    ev_grade = g["grade"]

    leg_grades = []
    best_rank = 0
    for r in plus:
        if r.get("grade"):
            lg = r["grade"]
            why = r.get("grade_why")
        else:
            gg = grade_bet(
                float(r["p"]),
                float(r["odds"]),
                ev=float(r.get("ev") or 0),
                edge=r.get("edge_pp"),
                book_outside_interval=r.get("book_outside_interval"),
            )
            lg, why = gg["grade"], gg["grade_why"]
        best_rank = max(best_rank, _GRADE_RANK.get(lg, 0))
        leg_grades.append(
            {
                "selection": r.get("selection"),
                "grade": lg,
                "grade_why": why,
                "ev": r.get("ev"),
                "p": r.get("p"),
            }
        )

    # Low wipe / high survival + solid ROI: anchor off best leg (−1 multi)
    if (
        same_map
        and portfolio_ev >= 0.10
        and (p_wipe <= 0.22 or p_any >= 0.72)
        and best_rank >= _GRADE_RANK["B-"]
    ):
        anchor = _downgrade(_RANK_GRADE[best_rank], 1)
        grade = (
            anchor
            if _GRADE_RANK.get(anchor, 0) >= _GRADE_RANK.get(ev_grade, 0)
            else ev_grade
        )
        why = f"same-map ×{n} low-wipe anchor (ρ={rho})"
    else:
        grade = ev_grade
        why = f"same-map ×{n} equal-stake (ρ={rho})" if same_map else f"×{n} equal-stake"

    return {
        "grade": grade,
        "grade_why": why,
        "n_legs": n,
        "portfolio_ev": round(portfolio_ev, 4),
        "portfolio_roi_pct": round(portfolio_ev * 100, 1),
        "p_both_indep": round(p_indep, 4),
        "p_both": round(p_both, 4),
        "p_any": round(p_any, 4),
        "p_wipe": round(p_wipe, 4),
        "legs": leg_grades,
        "raw_grade_before_map_haircut": g["grade"],
    }


def evaluate_odds_row(
    selection: str,
    market: str,
    p: float,
    decimal_odds: float,
    *,
    kelly_frac: float = 0.5,
    kelly_cap: float = 0.05,
    p_lo: float | None = None,
    p_hi: float | None = None,
) -> dict:
    imp = implied_prob(decimal_odds)
    ev = ev_per_unit(p, decimal_odds)
    edge = edge_pp(p, decimal_odds)
    outside = None
    if p_lo is not None and p_hi is not None:
        outside = imp < p_lo or imp > p_hi
    g = grade_bet(p, decimal_odds, ev=ev, edge=edge, book_outside_interval=outside)
    return {
        "market": market,
        "selection": selection,
        "p": round(p, 4),
        "odds": decimal_odds,
        "implied": round(imp, 4),
        "edge_pp": round(edge, 2),
        "ev": round(ev, 4),
        "kelly": round(kelly_fraction(p, decimal_odds, kelly_frac, kelly_cap), 4),
        "fair_odds": round(1.0 / p, 3) if p > 0.01 else None,
        "p_lo": round(p_lo, 4) if p_lo is not None else None,
        "p_hi": round(p_hi, 4) if p_hi is not None else None,
        "book_outside_interval": outside,
        "grade": g["grade"],
        "grade_why": g["grade_why"],
    }


def disagreement_highlights(
    rows: Iterable[dict],
    *,
    near_coin_flip: float = 0.08,
    min_edge_pp: float = 5.0,
) -> list[dict]:
    """
    Spots that 'feel 50/50' (|p-0.5| small) but model edge vs book is large.
    Prefer rows where book implied is outside conformal interval when available.
    """
    hits = []
    for r in rows:
        p = r.get("p")
        edge = r.get("edge_pp")
        if p is None or edge is None:
            continue
        outside = r.get("book_outside_interval")
        if outside or (abs(p - 0.5) <= near_coin_flip and edge >= min_edge_pp):
            why = "book outside conformal band" if outside else "near-coin-flip model vs soft book"
            hits.append({**r, "why": why})
    return sorted(hits, key=lambda x: -x["edge_pp"])


def rank_plus_ev(rows: Iterable[dict], min_ev: float = 0.0) -> list[dict]:
    return sorted([r for r in rows if r.get("ev", -1) > min_ev], key=lambda x: -x["ev"])


def p_under_normal(mu: float, sd: float, line: float) -> float:
    thr = math.floor(line) + 0.5
    z = (thr - mu) / max(sd, 1e-6)
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def append_odds_journal(rows: list[dict], meta: dict) -> Path:
    """Append priced board rows to data/lol/odds_journal.parquet."""
    from datetime import datetime, timezone

    import pandas as pd

    from lol_kills.etl.paths import DATA

    path = DATA / "odds_journal.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    recs = []
    for r in rows:
        recs.append(
            {
                "ts": now,
                "team1": meta.get("team1"),
                "team2": meta.get("team2"),
                "league": meta.get("league"),
                "blue": meta.get("blue"),
                "red": meta.get("red"),
                "market": r.get("market"),
                "selection": r.get("selection"),
                "p": r.get("p"),
                "odds": r.get("odds"),
                "implied": r.get("implied"),
                "ev": r.get("ev"),
                "edge_pp": r.get("edge_pp"),
                "kelly": r.get("kelly"),
                "grade": r.get("grade"),
                "grade_why": r.get("grade_why"),
                "result": None,  # filled later when known
            }
        )
    if not recs:
        return path
    new = pd.DataFrame(recs)
    if path.exists():
        old = pd.read_parquet(path)
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_parquet(path, index=False)
    return path


def betting_report(bankroll: float = 100.0, kelly_frac: float = 0.5) -> dict:
    """Paper half/eighth-Kelly growth from odds journal (+EV only)."""
    import json
    from pathlib import Path

    import pandas as pd

    from lol_kills.etl.paths import DATA, MODELS_DIR

    path = DATA / "odds_journal.parquet"
    if not path.exists():
        report = {"n": 0, "note": "no odds journal yet"}
        (MODELS_DIR / "betting_report.json").write_text(json.dumps(report, indent=2))
        return report

    df = pd.read_parquet(path)
    priced = df.dropna(subset=["odds", "p", "ev"])
    plus = priced[priced["ev"] > 0].copy()
    wealth = bankroll
    path_w = [wealth]
    for _, r in plus.iterrows():
        f = kelly_fraction(float(r["p"]), float(r["odds"]), fraction=kelly_frac, cap=0.05)
        # if result known use it; else mark unrealized EV contribution
        if pd.notna(r.get("result")):
            won = float(r["result"]) >= 0.5
            wealth = wealth * (1 + f * (float(r["odds"]) - 1)) if won else wealth * (1 - f)
        else:
            # paper EV mark-to-model (not realized)
            wealth = wealth * (1 + f * float(r["ev"]))
        path_w.append(wealth)

    report = {
        "n_journal": int(len(df)),
        "n_plus_ev": int(len(plus)),
        "start_bankroll": bankroll,
        "end_bankroll_paper": round(wealth, 2),
        "growth": round(wealth / bankroll - 1.0, 4),
        "mean_edge_pp": float(plus["edge_pp"].mean()) if len(plus) else None,
        "kelly_frac": kelly_frac,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "betting_report.json").write_text(json.dumps(report, indent=2))
    print(f"[econ] betting_report growth={report['growth']:.2%} n+={report['n_plus_ev']}")
    return report
