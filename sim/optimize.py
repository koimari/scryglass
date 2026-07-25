"""Exact metrics and SLSQP allocation optimizer."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize

from sim.probs import (
    apply_edge,
    failure_probs,
    pattern_probability,
    poisson_binomial_pmf_fft,
)
from sim.tickets import BUDGET, N_28, N_29, TicketBook

CRRA_SIGMAS = (0.5, 1.0, 2.0)


def _crra(w: float, sigma: float) -> float:
    """
    CRRA utility on (1+W) so zero-payout outcomes stay finite and align
    with the Kelly objective E[log(1+W)].
    """
    x = 1.0 + max(float(w), 0.0)
    if abs(sigma - 1.0) < 1e-12:
        return float(np.log(x))
    return float((x ** (1.0 - sigma) - 1.0) / (1.0 - sigma))


def enumerate_nonzero_patterns(
    book: TicketBook, p_win: np.ndarray
) -> list[tuple[tuple[int, ...], float]]:
    """
    All failure patterns with possible nonzero A/B/C payout: k=0,1,2.
    Returns list of (failed_tuple, probability).
    """
    patterns: list[tuple[tuple[int, ...], float]] = []
    # k=0
    patterns.append(((), pattern_probability(p_win, ())))
    # k=1
    for i in range(book.n):
        patterns.append(((i,), pattern_probability(p_win, (i,))))
    # k=2
    for i, j in combinations(range(book.n), 2):
        patterns.append(((i, j), pattern_probability(p_win, (i, j))))
    return patterns


def enumerate_patterns_with_ladder(
    book: TicketBook, p_win: np.ndarray
) -> list[tuple[tuple[int, ...], float]]:
    """
    Patterns that can pay with A/B/C/D: k<=2, plus k=3 exactly equal to ladder trio.
    """
    patterns = enumerate_nonzero_patterns(book, p_win)
    if book.ladder_exclude is not None:
        trio = book.ladder_exclude
        patterns.append((trio, pattern_probability(p_win, trio)))
    return patterns


class ExactEvaluator:
    """Exact evaluation over k=0,1,2 (+ optional ladder k=3) failure patterns."""

    def __init__(
        self,
        book: TicketBook,
        p_win: np.ndarray,
        include_ladder_patterns: bool = False,
    ):
        self.book = book
        self.p_win = np.asarray(p_win, dtype=np.float64)
        self.pmf = poisson_binomial_pmf_fft(failure_probs(self.p_win))
        if include_ladder_patterns and book.ladder_exclude is not None:
            self.patterns = enumerate_patterns_with_ladder(book, self.p_win)
        else:
            self.patterns = enumerate_nonzero_patterns(book, self.p_win)
        # Precompute W for unit stakes is not useful; cache nothing until weights known
        self._include_ladder = include_ladder_patterns

    def metrics(
        self,
        w_a: float,
        w_b: float,
        w_c: float,
        w_d: float = 0.0,
    ) -> dict[str, float]:
        book = self.book
        # Accumulate over paying patterns
        e_w = 0.0
        e_log = 0.0
        e_w_pos = 0.0  # mass-weighted W for W>0
        p_pos = 0.0
        p_ge_10 = 0.0
        p_ge_100 = 0.0
        p_ge_1e4 = 0.0
        crra_acc = {s: 0.0 for s in CRRA_SIGMAS}

        # Probability mass covered by enumerated patterns
        mass_enum = 0.0
        for failed, prob in self.patterns:
            if prob <= 0:
                continue
            mass_enum += prob
            W = book.wealth(failed, w_a, w_b, w_c, w_d)
            e_w += prob * W
            e_log += prob * np.log1p(W)
            for s in CRRA_SIGMAS:
                crra_acc[s] += prob * _crra(W, s)
            if W > 0:
                p_pos += prob
                e_w_pos += prob * W
                if W >= 10:
                    p_ge_10 += prob
                if W >= 100:
                    p_ge_100 += prob
                if W >= 1e4:
                    p_ge_1e4 += prob

        # Remaining mass (k>=3 without ladder pay, or non-ladder k=3): W=0
        # E[log(1+0)]=0 contribution; CRRA(0) we treat as 0 contribution via convention
        # of not adding -1e6 * mass (that would dominate); use u(0)=0 for reporting
        p_full_hit = float(self.pmf[0]) if self.pmf.size else 0.0

        e_w_given_pos = (e_w_pos / p_pos) if p_pos > 0 else 0.0

        out: dict[str, float] = {
            "P_W_gt_0": float(p_pos),
            "P_W_ge_10": float(p_ge_10),
            "P_W_ge_100": float(p_ge_100),
            "P_W_ge_1e4": float(p_ge_1e4),
            "P_full_30_hit": p_full_hit,
            "E_W": float(e_w),
            "E_W_given_W_gt_0": float(e_w_given_pos),
            "E_log_1p_W": float(e_log),
        }
        for s in CRRA_SIGMAS:
            key = f"CRRA_sigma_{s:g}"
            out[key] = float(crra_acc[s])
        return out

    def survival_by_failures(
        self, w_a: float, w_b: float, w_c: float, w_d: float = 0.0, max_k: int = 5
    ) -> list[dict[str, float]]:
        """For each k, p_k and E[W|K=k] (exact for k<=2 / ladder; 0 for other k>=3)."""
        book = self.book
        rows: list[dict[str, float]] = []
        # Group pattern contributions by k
        sum_w_prob: dict[int, float] = {k: 0.0 for k in range(max_k + 1)}
        for failed, prob in self.patterns:
            k = len(failed)
            if k > max_k:
                continue
            W = book.wealth(failed, w_a, w_b, w_c, w_d)
            sum_w_prob[k] += prob * W

        for k in range(max_k + 1):
            p_k = float(self.pmf[k]) if k < self.pmf.size else 0.0
            e_w_gk = (sum_w_prob[k] / p_k) if p_k > 1e-30 else 0.0
            # For k=3 with ladder: only one of C(30,3) patterns may pay; E[W|K=3]
            # uses only that pattern's contribution / P(K=3) — correct.
            rows.append({"k": k, "p_k": p_k, "E_W_given_k": float(e_w_gk)})
        return rows


def _pack_weights(x: np.ndarray) -> tuple[float, float, float]:
    return float(x[0]), float(x[1]), float(x[2])


def optimize_slsqp(
    evaluator: ExactEvaluator,
    objective: Callable[[float, float, float], float],
    x0: np.ndarray | None = None,
    maximize: bool = True,
    min_w30: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Optimize (w_a, w_b, w_c) with sum=BUDGET, w>=0 via SLSQP.
    objective(w_a,w_b,w_c) returns scalar to maximize (if maximize).
    min_w30: floor on full 30-fold stake (user requires parlay remains in book).
    """
    min_w30 = float(max(0.0, min(min_w30, BUDGET)))
    if x0 is None:
        rest = BUDGET - min_w30
        x0 = np.array(
            [min_w30, rest / 2 if rest > 0 else 0.0, rest / 2 if rest > 0 else 0.0],
            dtype=np.float64,
        )
        if min_w30 <= 0:
            x0 = np.array([BUDGET / 3, BUDGET / 3, BUDGET / 3], dtype=np.float64)

    def fun(x: np.ndarray) -> float:
        wa, wb, wc = _pack_weights(x)
        val = objective(wa, wb, wc)
        return -val if maximize else val

    cons = [{"type": "eq", "fun": lambda x: float(np.sum(x) - BUDGET)}]
    bounds = [(min_w30, BUDGET), (0.0, BUDGET), (0.0, BUDGET)]
    res = minimize(
        fun,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    x = np.asarray(res.x, dtype=np.float64)
    x[0] = max(x[0], min_w30)
    x = np.clip(x, 0.0, BUDGET)
    # Renormalize to exact budget while preserving min_w30
    if x[0] < min_w30:
        x[0] = min_w30
    s = x.sum()
    if s > 0:
        if min_w30 > 0:
            rest = BUDGET - min_w30
            tail = x[1:].sum()
            x[0] = min_w30
            if tail > 0 and rest > 0:
                x[1:] = x[1:] * (rest / tail)
            elif rest > 0:
                x[1:] = rest / 2
                x[2] = rest / 2
            else:
                x[1:] = 0.0
        else:
            x = x * (BUDGET / s)
    else:
        x = np.array([BUDGET, 0.0, 0.0])
    wa, wb, wc = _pack_weights(x)
    return x, float(objective(wa, wb, wc))


def optimize_kelly(
    evaluator: ExactEvaluator,
    x0: np.ndarray | None = None,
    min_w30: float = 0.0,
) -> np.ndarray:
    def obj(wa: float, wb: float, wc: float) -> float:
        return evaluator.metrics(wa, wb, wc)["E_log_1p_W"]

    x, _ = optimize_slsqp(evaluator, obj, x0=x0, min_w30=min_w30)
    return x


def optimize_max_p_any(evaluator: ExactEvaluator, x0: np.ndarray | None = None) -> np.ndarray:
    """Maximize P(W>0). With full C cover any w_c>0 achieves P(K<=2); prefer slight C."""

    def obj(wa: float, wb: float, wc: float) -> float:
        m = evaluator.metrics(wa, wb, wc)
        # Primary P(W>0); tiny tie-break toward growth when flat
        return m["P_W_gt_0"] + 1e-12 * m["E_log_1p_W"]

    # Start with all on C (covers k<=2)
    if x0 is None:
        x0 = np.array([0.0, 0.0, BUDGET], dtype=np.float64)
    x, _ = optimize_slsqp(evaluator, obj, x0=x0)
    return x


def optimize_pareto(
    evaluator: ExactEvaluator,
    lam: float,
    scale_p: float,
    scale_e: float,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Maximize λ * P(W>=10)/scale_p + (1-λ) * E[W]/scale_e."""

    def obj(wa: float, wb: float, wc: float) -> float:
        m = evaluator.metrics(wa, wb, wc)
        p_term = m["P_W_ge_10"] / scale_p if scale_p > 0 else 0.0
        e_term = m["E_W"] / scale_e if scale_e > 0 else 0.0
        return lam * p_term + (1.0 - lam) * e_term

    x, _ = optimize_slsqp(evaluator, obj, x0=x0)
    return x


def weights_dict(wa: float, wb: float, wc: float, wd: float = 0.0) -> dict[str, float]:
    return {
        "w_30": float(wa),
        "w_29": float(wb),
        "w_28": float(wc),
        "w_ladder": float(wd),
    }


def per_ticket_dict(wa: float, wb: float, wc: float) -> dict[str, float]:
    return {
        "stake_30": float(wa),
        "stake_each_29": float(wb / N_29) if N_29 else 0.0,
        "stake_each_28": float(wc / N_28) if N_28 else 0.0,
    }


def allocation_string(wa: float, wb: float, wc: float, wd: float = 0.0) -> str:
    parts = [
        f"R${wa:.4f} on 30-fold",
        f"R${wb:.4f} on 29-folds (R${wb / N_29:.6f} each)",
        f"R${wc:.4f} on 28-folds (R${wc / N_28:.6f} each)",
    ]
    if wd > 1e-9:
        parts.append(f"R${wd:.4f} on 27-fold ladder")
    return ", ".join(parts)


def run_pareto_frontier(
    evaluator: ExactEvaluator, n_lambda: int = 11
) -> list[dict[str, Any]]:
    """Sweep λ in [0,1] for scaled P(W>=10) vs E[W]."""
    # Reference scales from corner solutions
    x_p = optimize_max_p_any(evaluator)
    x_kelly = optimize_kelly(evaluator)
    # Also try all-in and equal for scale
    candidates = [
        x_p,
        x_kelly,
        np.array([BUDGET, 0.0, 0.0]),
        np.array([BUDGET / 3] * 3),
        np.array([0.0, 0.0, BUDGET]),
        np.array([0.0, BUDGET, 0.0]),
    ]
    p_vals = []
    e_vals = []
    for x in candidates:
        m = evaluator.metrics(*_pack_weights(x))
        p_vals.append(m["P_W_ge_10"])
        e_vals.append(m["E_W"])
    scale_p = max(p_vals) if max(p_vals) > 0 else 1.0
    scale_e = max(e_vals) if max(e_vals) > 0 else 1.0

    frontier: list[dict[str, Any]] = []
    x_prev = x_kelly.copy()
    for lam in np.linspace(0.0, 1.0, n_lambda):
        x = optimize_pareto(evaluator, float(lam), scale_p, scale_e, x0=x_prev)
        x_prev = x.copy()
        wa, wb, wc = _pack_weights(x)
        m = evaluator.metrics(wa, wb, wc)
        frontier.append(
            {
                "lambda": float(lam),
                "weights": weights_dict(wa, wb, wc),
                "metrics": m,
            }
        )
    return frontier


def try_ladder_improvement(
    book: TicketBook,
    p_win: np.ndarray,
    base_weights: tuple[float, float, float],
    min_w30: float = 0.0,
) -> dict[str, Any]:
    """
    Compare adding Class D (27-fold) via 4-weight simplex optimization.
    Include in recommended only if Kelly or P(W>0) improves.
    """
    ev = ExactEvaluator(book, p_win, include_ladder_patterns=True)
    wa0, wb0, wc0 = base_weights
    base_m = ev.metrics(wa0, wb0, wc0, 0.0)
    min_w30 = float(max(0.0, min(min_w30, BUDGET)))

    def obj(x: np.ndarray) -> float:
        wa, wb, wc, wd = (float(v) for v in x)
        return -ev.metrics(wa, wb, wc, wd)["E_log_1p_W"]

    x0 = np.array([max(wa0, min_w30), wb0 * 0.9, wc0 * 0.9, BUDGET * 0.1], dtype=np.float64)
    # renormalize
    x0 = x0 * (BUDGET / x0.sum())
    if x0[0] < min_w30:
        x0[0] = min_w30
        tail = x0[1:].sum()
        rest = BUDGET - min_w30
        if tail > 0 and rest > 0:
            x0[1:] = x0[1:] * (rest / tail)
    cons = {"type": "eq", "fun": lambda x: float(np.sum(x) - BUDGET)}
    bounds = [(min_w30, BUDGET), (0.0, BUDGET), (0.0, BUDGET), (0.0, BUDGET)]
    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    x = np.clip(res.x, 0.0, BUDGET)
    x[0] = max(x[0], min_w30)
    if x.sum() > 0:
        if min_w30 > 0:
            rest = BUDGET - min_w30
            tail = x[1:].sum()
            x[0] = min_w30
            if tail > 0 and rest > 0:
                x[1:] = x[1:] * (rest / tail)
            elif rest > 0:
                x[1:] = rest / 3.0
        else:
            x = x * (BUDGET / x.sum())
    else:
        x = np.array([BUDGET, 0, 0, 0])
    wa, wb, wc, wd = (float(v) for v in x)
    m = ev.metrics(wa, wb, wc, wd)

    improves_kelly = m["E_log_1p_W"] > base_m["E_log_1p_W"] + 1e-15
    improves_p = m["P_W_gt_0"] > base_m["P_W_gt_0"] + 1e-15
    include = bool((improves_kelly or improves_p) and wd > 1e-6)

    return {
        "improves_kelly": improves_kelly,
        "improves_p_any": improves_p,
        "include_in_recommended": include,
        "weights": weights_dict(wa, wb, wc, wd),
        "metrics": m,
        "base_metrics": base_m,
        "odds_27": book.odds_27,
    }


def sensitivity_edge(
    book: TicketBook,
    p_base: np.ndarray,
    relative_edge: float,
    x_rec: np.ndarray,
    min_w30: float = 0.0,
) -> dict[str, Any]:
    """Re-optimize Kelly under edged probs; also report metrics at recommended weights."""
    p = apply_edge(p_base, relative_edge)
    ev = ExactEvaluator(book, p)
    x_new = optimize_kelly(ev, x0=x_rec.copy(), min_w30=min_w30)
    wa, wb, wc = _pack_weights(x_new)
    # Metrics at re-optimized weights under edged model
    m_opt = ev.metrics(wa, wb, wc)
    # Also metrics if keeping original recommended weights
    wa0, wb0, wc0 = _pack_weights(x_rec)
    m_fixed = ev.metrics(wa0, wb0, wc0)
    return {
        "recommended_weights": weights_dict(wa, wb, wc),
        "metrics": m_opt,
        "metrics_at_baseline_weights": m_fixed,
        "relative_edge": relative_edge,
    }
