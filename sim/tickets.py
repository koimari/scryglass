"""Structured ticket classes A/B/C/(D) and wealth evaluation."""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

N_LEGS = 30
N_29 = 30  # C(30,1) exclude-one tickets
N_28 = 435  # C(30,2)
BUDGET = 10.0


class TicketBook:
    """
    Class A: 1× 30-fold (all legs).
    Class B: all 29-folds (exclude one leg each).
    Class C: all 28-folds (exclude two legs each).
    Optional Class D: single 27-fold excluding a fixed trio (risk ladder).
    """

    def __init__(self, odds: np.ndarray, ladder_exclude: Sequence[int] | None = None):
        self.odds = np.asarray(odds, dtype=np.float64)
        self.n = self.odds.size
        assert self.n == N_LEGS
        self.O30 = float(np.prod(self.odds))
        # 29-fold excluding i: product of all odds except i
        self.odds_29 = self.O30 / self.odds  # shape (30,)
        # Precompute pair odds for 28-folds: dict and dense upper triangle list
        self.pair_index: dict[tuple[int, int], int] = {}
        self.pairs: list[tuple[int, int]] = []
        self.odds_28 = np.empty(N_28, dtype=np.float64)
        idx = 0
        for i, j in combinations(range(self.n), 2):
            self.pairs.append((i, j))
            self.pair_index[(i, j)] = idx
            self.odds_28[idx] = self.O30 / (self.odds[i] * self.odds[j])
            idx += 1

        self.ladder_exclude: tuple[int, ...] | None = None
        self.odds_27: float | None = None
        if ladder_exclude is not None:
            ex = tuple(sorted(int(x) for x in ladder_exclude))
            assert len(ex) == 3
            self.ladder_exclude = ex
            self.odds_27 = self.O30 / float(
                self.odds[ex[0]] * self.odds[ex[1]] * self.odds[ex[2]]
            )

        # Aggregates for k=0 closed form
        self.sum_odds_29 = float(self.odds_29.sum())
        self.sum_odds_28 = float(self.odds_28.sum())

    def wealth(
        self,
        failed: Sequence[int] | set[int] | tuple[int, ...],
        w_a: float,
        w_b: float,
        w_c: float,
        w_d: float = 0.0,
    ) -> float:
        """Total payout W given failed leg indices (0-based) and class stakes."""
        F = set(int(i) for i in failed)
        k = len(F)
        W = 0.0

        # Class A: pays only if k=0
        if k == 0 and w_a > 0:
            W += w_a * self.O30

        # Class B: 29-fold excluding i pays iff F ⊆ {i}
        if w_b > 0 and k <= 1:
            stake_each = w_b / N_29
            if k == 0:
                W += stake_each * self.sum_odds_29
            else:
                # only the ticket excluding the failed leg
                j = next(iter(F))
                W += stake_each * self.odds_29[j]

        # Class C: 28-fold excluding {i,j} pays iff F ⊆ {i,j}
        if w_c > 0 and k <= 2:
            stake_each = w_c / N_28
            if k == 0:
                W += stake_each * self.sum_odds_28
            elif k == 1:
                j = next(iter(F))
                # all pairs containing j
                s = 0.0
                for m in range(self.n):
                    if m == j:
                        continue
                    a, b = (j, m) if j < m else (m, j)
                    s += self.odds_28[self.pair_index[(a, b)]]
                W += stake_each * s
            else:  # k == 2
                a, b = sorted(F)
                W += stake_each * self.odds_28[self.pair_index[(a, b)]]

        # Class D: 27-fold excluding ladder trio — pays iff F ⊆ ladder_exclude
        if w_d > 0 and self.ladder_exclude is not None and self.odds_27 is not None:
            if F.issubset(self.ladder_exclude):
                W += w_d * self.odds_27

        return float(W)

    def wealth_by_k0(self, w_a: float, w_b: float, w_c: float, w_d: float = 0.0) -> float:
        return self.wealth((), w_a, w_b, w_c, w_d)
