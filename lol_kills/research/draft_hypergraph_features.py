"""Deterministic higher-order ally and cross-team draft features.

The feature generator is deliberately separate from model fitting.  It can
enumerate every supported ally hyperedge and cross-team hyperedge up to a
declared order, while the caller controls support gates and regularization.
Terms are canonicalized so replay does not depend on input ordering.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np
from scipy import sparse


ROLES = ("top", "jng", "mid", "bot", "sup")


def _canonical(values: tuple[str, ...]) -> str:
    return "|".join(sorted(values))


def _ally_keys(game: dict[str, Any], max_order: int) -> list[tuple[str, float]]:
    keys: list[tuple[str, float]] = []
    max_order = min(max_order, len(ROLES))
    for side, sign in (("blue", 1.0), ("red", -1.0)):
        champions = tuple(str(game[side][role]["champion"]) for role in ROLES)
        for order in range(3, max_order + 1):
            for subset in combinations(champions, order):
                keys.append((f"HA|{order}|{_canonical(subset)}", sign))
    return keys


def _counter_keys(game: dict[str, Any], max_order: int) -> list[str]:
    blue = tuple(str(game["blue"][role]["champion"]) for role in ROLES)
    red = tuple(str(game["red"][role]["champion"]) for role in ROLES)
    keys: list[str] = []
    for blue_order in range(1, len(ROLES) + 1):
        for red_order in range(1, len(ROLES) + 1):
            total_order = blue_order + red_order
            if total_order < 3 or total_order > max_order:
                continue
            for blue_subset in combinations(blue, blue_order):
                for red_subset in combinations(red, red_order):
                    keys.append(
                        "HC|{}x{}|{}|{}".format(
                            blue_order,
                            red_order,
                            _canonical(blue_subset),
                            _canonical(red_subset),
                        )
                    )
    return keys


def hypergraph_vocabulary(
    games: list[dict[str, Any]],
    *,
    max_ally_order: int = 5,
    max_counter_order: int = 4,
    min_support: int = 12,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return support-gated higher-order vocabulary and raw counts."""

    if max_ally_order < 3:
        raise ValueError("max_ally_order must be at least 3")
    if max_counter_order < 3:
        raise ValueError("max_counter_order must be at least 3")
    if min_support < 1:
        raise ValueError("min_support must be positive")
    counts: Counter[str] = Counter()
    for game in games:
        counts.update(key for key, _ in _ally_keys(game, max_ally_order))
        counts.update(_counter_keys(game, max_counter_order))
    selected = {
        key: int(count)
        for key, count in counts.items()
        if count >= min_support
    }
    ordered = {key: index for index, key in enumerate(sorted(selected))}
    return ordered, selected


def hypergraph_feature_rows(
    games: list[dict[str, Any]],
    vocabulary: dict[str, int],
) -> sparse.csr_matrix:
    """Build blue-perspective signed sparse rows for a frozen vocabulary."""

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row_index, game in enumerate(games):
        for key, sign in _ally_keys(game, max_order=5):
            column = vocabulary.get(key)
            if column is not None:
                rows.append(row_index)
                columns.append(column)
                values.append(sign)
        for key in _counter_keys(game, max_order=10):
            column = vocabulary.get(key)
            if column is not None:
                rows.append(row_index)
                columns.append(column)
                values.append(1.0)
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(games), len(vocabulary)),
        dtype=np.float64,
    )


def hypergraph_breakdown(
    game: dict[str, Any],
    vocabulary: dict[str, int],
    counts: dict[str, int],
) -> list[dict[str, Any]]:
    """Return supported higher-order terms present in one draft."""

    rows: list[dict[str, Any]] = []
    for key, sign in _ally_keys(game, max_order=5):
        if key in vocabulary:
            rows.append(
                {
                    "family": "ally_hyperedge",
                    "key": key,
                    "support": int(counts.get(key, 0)),
                    "blue_perspective_sign": sign,
                }
            )
    for key in _counter_keys(game, max_order=10):
        if key in vocabulary:
            rows.append(
                {
                    "family": "counter_hyperedge",
                    "key": key,
                    "support": int(counts.get(key, 0)),
                    "blue_perspective_sign": 1.0,
                }
            )
    return rows


def _anchored_keys(
    game: dict[str, Any],
    *,
    max_order: int = 9,
) -> list[tuple[str, float, int]]:
    """Enumerate an anchor against subsets of the other nine picks."""

    if max_order < 1 or max_order > 9:
        raise ValueError("anchored max_order must be between 1 and 9")
    keys: list[tuple[str, float, int]] = []
    for side, sign in (("blue", 1.0), ("red", -1.0)):
        other_side = "red" if side == "blue" else "blue"
        for role in ROLES:
            anchor = game[side][role]
            anchor_champion = str(anchor["champion"])
            contexts: list[str] = []
            for other_role in ROLES:
                if other_role == role:
                    continue
                contexts.append(
                    f"A:{str(game[side][other_role]['champion'])}"
                )
            for other_role in ROLES:
                contexts.append(
                    f"E:{str(game[other_side][other_role]['champion'])}"
                )
            for order in range(1, min(max_order, len(contexts)) + 1):
                for subset in combinations(tuple(contexts), order):
                    key = (
                        f"AH|{role}|{anchor_champion}|{order}|"
                        f"{'|'.join(sorted(subset))}"
                    )
                    keys.append((key, sign, order))
    return keys


def anchored_hypergraph_vocabulary(
    games: list[dict[str, Any]],
    *,
    max_order: int = 9,
    min_support_by_order: dict[int, int] | None = None,
) -> tuple[dict[str, int], dict[str, int], dict[int, int]]:
    """Build support-gated order-1..9 anchored context terms.

    The returned order counts are useful even when a high-order vocabulary is
    empty: an absent order is an evidence/coverage result, not an implicit
    zero-strength claim.
    """

    counts: Counter[str] = Counter()
    order_counts: Counter[int] = Counter()
    for game in games:
        for key, _, order in _anchored_keys(game, max_order=max_order):
            counts[key] += 1
            order_counts[order] += 1
    thresholds = min_support_by_order or {
        order: max(12, 2 ** (order - 1))
        for order in range(1, max_order + 1)
    }
    selected = {
        key: int(count)
        for key, count in counts.items()
        if count >= int(thresholds.get(int(key.split("|")[3]), 12))
    }
    vocabulary = {key: index for index, key in enumerate(sorted(selected))}
    return vocabulary, selected, dict(sorted(order_counts.items()))


def anchored_hypergraph_feature_rows(
    games: list[dict[str, Any]],
    vocabulary: dict[str, int],
    *,
    max_order: int = 9,
) -> sparse.csr_matrix:
    """Build signed sparse rows for an anchored hypergraph vocabulary."""

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row_index, game in enumerate(games):
        for key, sign, _ in _anchored_keys(game, max_order=max_order):
            column = vocabulary.get(key)
            if column is not None:
                rows.append(row_index)
                columns.append(column)
                values.append(sign)
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(games), len(vocabulary)),
        dtype=np.float64,
    )
