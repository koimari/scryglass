"""Probability model and Poisson-binomial failure distribution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_LEGS_PATH = Path(__file__).resolve().parent.parent / "data" / "legs.json"


def load_legs(path: str | Path | None = None) -> dict[str, Any]:
    """Load legs JSON; returns full payload including legs list and meta."""
    p = Path(path) if path else DEFAULT_LEGS_PATH
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def odds_array(legs: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(leg["odds"]) for leg in legs], dtype=np.float64)


def baseline_probs(odds: np.ndarray) -> np.ndarray:
    """
    Fair-under-quoted-odds model: p_i = 1/o_i clipped to (0.01, 0.99).
    No cross-leg renormalization (independent matches).
    """
    p = 1.0 / odds
    return np.clip(p, 0.01, 0.99)


def apply_edge(p: np.ndarray, relative_edge: float) -> np.ndarray:
    """Sensitivity: scale all probs by (1 + relative_edge), re-clip."""
    return np.clip(p * (1.0 + relative_edge), 0.01, 0.99)


def poisson_binomial_pmf_fft(fail_probs: np.ndarray) -> np.ndarray:
    """
    PMF of K = sum of independent Bernoullis with success probs fail_probs.

    Uses the stable O(n^2) DP recursion (FFT left-tail cancels for this
    regime where E[K]≈10 and P(K≤2) is O(1e-3)). Name kept for API;
    still O(n^2) with n=30.
    Returns array pmf[0..n] with pmf[k] = P(K=k).
    """
    q = np.asarray(fail_probs, dtype=np.float64)
    n = q.size
    pmf = np.zeros(n + 1, dtype=np.float64)
    pmf[0] = 1.0
    for qi in q:
        # pmf_new[k] = pmf[k]*(1-qi) + pmf[k-1]*qi
        new = np.empty_like(pmf)
        new[0] = pmf[0] * (1.0 - qi)
        new[1:] = pmf[1:] * (1.0 - qi) + pmf[:-1] * qi
        pmf = new
    pmf = np.maximum(pmf, 0.0)
    s = pmf.sum()
    if s > 0:
        pmf /= s
    return pmf


def failure_probs(p_win: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(p_win, dtype=np.float64)


def p_all_win(p_win: np.ndarray) -> float:
    return float(np.prod(p_win))


def p_exactly_k_failures(p_win: np.ndarray, k: int, pmf: np.ndarray | None = None) -> float:
    if pmf is None:
        pmf = poisson_binomial_pmf_fft(failure_probs(p_win))
    if k < 0 or k >= pmf.size:
        return 0.0
    return float(pmf[k])


def pattern_probability(p_win: np.ndarray, failed_indices: tuple[int, ...] | list[int]) -> float:
    """P(exact failure set F and all others win)."""
    p = np.asarray(p_win, dtype=np.float64)
    q = 1.0 - p
    failed = set(int(i) for i in failed_indices)
    prob = 1.0
    for i in range(p.size):
        prob *= q[i] if i in failed else p[i]
    return float(prob)


def legs_summary(legs: list[dict[str, Any]], p: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for leg, pi in zip(legs, p):
        out.append(
            {
                "id": leg["id"],
                "team": leg["team"],
                "odds": float(leg["odds"]),
                "p_baseline": float(pi),
            }
        )
    return out
