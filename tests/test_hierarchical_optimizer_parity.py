"""Parity and callback-count tests for the hierarchical BT optimizer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

import lol_kills.ratings.hierarchical_bt as hierarchical_bt


def _map(index: int, blue: str, red: str, blue_win: int) -> dict[str, object]:
    return {
        "date": f"2026-01-{index + 1:02d} 10:00:00",
        "league": "LEC",
        "blue_team": blue,
        "red_team": red,
        "y_blue_win": blue_win,
        "game_uid": f"optimizer-fixture-{index}",
    }


def _fixture() -> pd.DataFrame:
    rows = [
        _map(0, "Team A", "Team B", 1),
        _map(1, "Team B", "Team C", 0),
        _map(2, "Team C", "Team A", 1),
        _map(3, "Team A", "Team C", 1),
        _map(4, "Team B", "Team A", 0),
        _map(5, "Team C", "Team B", 1),
        _map(6, "Team A", "Team B", 0),
        _map(7, "Team B", "Team C", 1),
        _map(8, "Team C", "Team A", 0),
        _map(9, "Team A", "Team C", 0),
        _map(10, "Team B", "Team A", 1),
        _map(11, "Team C", "Team B", 0),
    ]
    return pd.DataFrame(rows)


def _fit_with_callback_mode(
    maps: pd.DataFrame,
    monkeypatch: Any,
    *,
    split: bool,
) -> tuple[pd.DataFrame, dict[str, Any], Any, int]:
    real_minimize = hierarchical_bt.minimize
    result_box: list[Any] = []
    objective_calls: list[None] = []

    def patched_minimize(
        objective: Callable[..., Any],
        x0: np.ndarray,
        *,
        jac: Any,
        **kwargs: Any,
    ) -> Any:
        if split:
            def value(beta: np.ndarray) -> float:
                objective_calls.append(None)
                return objective(beta)[0]

            def gradient(beta: np.ndarray) -> np.ndarray:
                objective_calls.append(None)
                return objective(beta)[1]

            result = real_minimize(value, x0, jac=gradient, **kwargs)
        else:
            def fused_objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
                objective_calls.append(None)
                return objective(beta)

            result = real_minimize(
                fused_objective,
                x0,
                jac=jac,
                **kwargs,
            )
        result_box.append(result)
        return result

    monkeypatch.setattr(hierarchical_bt, "minimize", patched_minimize)
    try:
        snapshot, meta = hierarchical_bt.fit_hierarchical_bt(maps, write=False)
    finally:
        monkeypatch.setattr(hierarchical_bt, "minimize", real_minimize)
    return snapshot, meta, result_box[0], len(objective_calls)


def test_fused_hierarchical_optimizer_preserves_fit_and_reduces_callbacks(monkeypatch: Any) -> None:
    maps = _fixture()
    fused_snapshot, fused_meta, fused_result, fused_calls = _fit_with_callback_mode(
        maps,
        monkeypatch,
        split=False,
    )
    split_snapshot, split_meta, split_result, split_calls = _fit_with_callback_mode(
        maps,
        monkeypatch,
        split=True,
    )

    assert fused_result.success is split_result.success
    assert fused_result.nit == split_result.nit
    assert fused_result.nfev == split_result.nfev
    assert fused_result.njev == split_result.njev
    assert fused_result.fun == split_result.fun
    np.testing.assert_array_equal(fused_result.x, split_result.x)
    assert fused_calls == fused_result.nfev
    assert split_calls == split_result.nfev + split_result.njev
    assert fused_calls < split_calls

    pd.testing.assert_frame_equal(fused_snapshot, split_snapshot, check_exact=True)
    assert fused_meta["optimizer_success"] == split_meta["optimizer_success"]
    assert fused_meta["optimizer_message"] == split_meta["optimizer_message"]
