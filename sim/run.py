"""CLI: run exact optimization and write results/allocation.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim.optimize import (
    ExactEvaluator,
    allocation_string,
    optimize_kelly,
    optimize_max_p_any,
    per_ticket_dict,
    run_pareto_frontier,
    sensitivity_edge,
    try_ladder_improvement,
    weights_dict,
)
from sim.probs import baseline_probs, legs_summary, load_legs, odds_array
from sim.tickets import BUDGET, TicketBook

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "allocation.json"

# Calibrated longshots for optional 27-fold ladder:
# SAW OCR 2.12 (id 13), Imperial (id 14), Cloud9 OCR 1.72 (id 30)
# Legs JSON uses 1-based ids; TicketBook uses 0-based indices.
LADDER_LEG_IDS = (13, 14, 30)  # 1-based


def _pack(x: np.ndarray) -> tuple[float, float, float]:
    return float(x[0]), float(x[1]), float(x[2])


def _bench(ev: ExactEvaluator, name: str, wa: float, wb: float, wc: float) -> dict:
    return {
        "weights": weights_dict(wa, wb, wc),
        "metrics": ev.metrics(wa, wb, wc),
    }


def main() -> None:
    data = load_legs()
    legs = data["legs"]
    odds = odds_array(legs)
    p = baseline_probs(odds)
    total_odds = float(np.prod(odds))

    # Map 1-based ids -> 0-based indices
    id_to_idx = {leg["id"]: i for i, leg in enumerate(legs)}
    ladder_idx = tuple(id_to_idx[i] for i in LADDER_LEG_IDS)

    book = TicketBook(odds, ladder_exclude=ladder_idx)
    ev = ExactEvaluator(book, p)

    # Keep at least R$1 on the full 30-fold (user: bet all games in a parlay)
    MIN_W30 = 1.0

    # Benchmarks
    all_in = np.array([BUDGET, 0.0, 0.0])
    equal = np.array([BUDGET / 3, BUDGET / 3, BUDGET / 3])
    x_kelly_free = optimize_kelly(ev, min_w30=0.0)
    x_kelly = optimize_kelly(ev, min_w30=MIN_W30)
    x_maxp = optimize_max_p_any(ev)

    wa_k, wb_k, wc_k = _pack(x_kelly)
    wa_p, wb_p, wc_p = _pack(x_maxp)
    wa_a, wb_a, wc_a = _pack(all_in)
    wa_e, wb_e, wc_e = _pack(equal)

    m_kelly = ev.metrics(wa_k, wb_k, wc_k)
    m_kelly_free = ev.metrics(*_pack(x_kelly_free))
    m_allin = ev.metrics(wa_a, wb_a, wc_a)

    # Pareto frontier
    frontier = run_pareto_frontier(ev, n_lambda=11)

    # Optional Class D ladder (respect min 30-fold floor)
    ladder = try_ladder_improvement(book, p, (wa_k, wb_k, wc_k), min_w30=MIN_W30)

    # Recommended: constrained Kelly (+ ladder if it still helps under the floor)
    if ladder["include_in_recommended"] and ladder["weights"]["w_30"] >= MIN_W30 - 1e-9:
        w = ladder["weights"]
        rec_name = "kelly_log_min_w30_with_ladder"
        rec_wa, rec_wb, rec_wc, rec_wd = (
            w["w_30"],
            w["w_29"],
            w["w_28"],
            w["w_ladder"],
        )
        rec_metrics = ladder["metrics"]
        ev_rec = ExactEvaluator(book, p, include_ladder_patterns=True)
    else:
        rec_name = "kelly_log_min_w30"
        rec_wa, rec_wb, rec_wc, rec_wd = wa_k, wb_k, wc_k, 0.0
        rec_metrics = m_kelly
        ev_rec = ev

    # Survival tables
    surv_allin = ev.survival_by_failures(wa_a, wb_a, wc_a, 0.0)
    surv_rec = ev_rec.survival_by_failures(rec_wa, rec_wb, rec_wc, rec_wd)

    # Sensitivity ±5%
    x_rec = np.array([rec_wa, rec_wb, rec_wc])
    sens_m = sensitivity_edge(book, p, -0.05, x_rec, min_w30=MIN_W30)
    sens_p = sensitivity_edge(book, p, 0.05, x_rec, min_w30=MIN_W30)

    # Longshot impact
    longshot_legs = []
    for lid in LADDER_LEG_IDS:
        i = id_to_idx[lid]
        longshot_legs.append(
            {
                "id": lid,
                "team": legs[i]["team"],
                "odds": float(odds[i]),
                "p": float(p[i]),
            }
        )

    lift_any = rec_metrics["P_W_gt_0"] - m_allin["P_W_gt_0"]
    lift_be = rec_metrics["P_W_ge_10"] - m_allin["P_W_ge_10"]
    ratio_any = (
        rec_metrics["P_W_gt_0"] / m_allin["P_W_gt_0"]
        if m_allin["P_W_gt_0"] > 0
        else float("inf")
    )
    ratio_be = (
        rec_metrics["P_W_ge_10"] / m_allin["P_W_ge_10"]
        if m_allin["P_W_ge_10"] > 0
        else float("inf")
    )

    jackpot_note = (
        "All-in 30-fold concentrates mass on the jackpot (full hit only); "
        "spreading into 29/28-folds trades peak O(1e5×) payout for survival when "
        "1–2 legs fail. With recommended mix, P(W≥1e4) is "
        f"{rec_metrics['P_W_ge_1e4']:.6g} vs all-in {m_allin['P_W_ge_1e4']:.6g}."
    )

    ladder_note = (
        f"Optional 27-fold excludes SAW/Imperial/Cloud9 (ids {list(LADDER_LEG_IDS)}). "
        f"Kelly improve={ladder['improves_kelly']}, "
        f"P(W>0) improve={ladder['improves_p_any']}; "
        f"included_in_recommended={ladder['include_in_recommended']}."
    )

    payload = {
        "meta": {
            "budget": BUDGET,
            "n_legs": len(legs),
            "total_odds": total_odds,
            "method": "exact_poisson_binomial_k0_1_2_SLSQP",
            "n_failure_regimes_exact": [0, 1, 2],
            "min_w30_constraint": MIN_W30,
            "unconstrained_kelly": {
                "weights": weights_dict(*_pack(x_kelly_free)),
                "metrics": m_kelly_free,
            },
            "ladder_comparison": {
                "odds_27": ladder["odds_27"],
                "improves_kelly": ladder["improves_kelly"],
                "improves_p_any": ladder["improves_p_any"],
                "include_in_recommended": ladder["include_in_recommended"],
                "weights": ladder["weights"],
                "metrics": ladder["metrics"],
            },
        },
        "legs_summary": legs_summary(legs, p),
        "recommended": {
            "name": rec_name,
            "weights": weights_dict(rec_wa, rec_wb, rec_wc, rec_wd),
            "per_ticket": per_ticket_dict(rec_wa, rec_wb, rec_wc),
            "metrics": rec_metrics,
        },
        "benchmarks": {
            "all_in_30": _bench(ev, "all_in_30", wa_a, wb_a, wc_a),
            "equal_split": _bench(ev, "equal_split", wa_e, wb_e, wc_e),
            "max_p_any_return": _bench(ev, "max_p_any_return", wa_p, wb_p, wc_p),
            "kelly_log": _bench(ev, "kelly_log", wa_k, wb_k, wc_k),
        },
        "pareto_frontier": frontier,
        "survival_by_failures": {
            "all_in_30": surv_allin,
            "recommended": surv_rec,
        },
        "sensitivity": {
            "edge_minus_5pct": {
                "recommended_weights": sens_m["recommended_weights"],
                "metrics": sens_m["metrics"],
            },
            "edge_plus_5pct": {
                "recommended_weights": sens_p["recommended_weights"],
                "metrics": sens_p["metrics"],
            },
        },
        "longshot_impact": {
            "legs": longshot_legs,
            "note": ladder_note,
        },
        "verdict": {
            "allocation_string": allocation_string(rec_wa, rec_wb, rec_wc, rec_wd),
            "lift_p_any_return_vs_all_in": float(lift_any),
            "lift_p_breakeven_vs_all_in": float(lift_be),
            "ratio_p_any_return_vs_all_in": float(ratio_any),
            "ratio_p_breakeven_vs_all_in": float(ratio_be),
            "jackpot_tail_note": jackpot_note,
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Stdout summary
    print("=== Parlay risk allocation (exact k=0,1,2) ===")
    print(f"Total odds product: {total_odds:.4f}")
    print(f"Recommended ({rec_name}): {payload['verdict']['allocation_string']}")
    print(
        f"  P(W>0)={rec_metrics['P_W_gt_0']:.6g}  "
        f"P(W>=10)={rec_metrics['P_W_ge_10']:.6g}  "
        f"E[W]={rec_metrics['E_W']:.4f}  "
        f"E[log(1+W)]={rec_metrics['E_log_1p_W']:.6g}"
    )
    print(
        f"All-in 30: P(W>0)={m_allin['P_W_gt_0']:.6g}  "
        f"P(W>=10)={m_allin['P_W_ge_10']:.6g}  E[W]={m_allin['E_W']:.4f}"
    )
    print(
        f"Lift vs all-in: ΔP(any)={lift_any:.6g}  ΔP(breakeven)={lift_be:.6g}"
    )
    print(
        f"Ladder D: include={ladder['include_in_recommended']}  "
        f"Kelly↑={ladder['improves_kelly']}  P↑={ladder['improves_p_any']}"
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
