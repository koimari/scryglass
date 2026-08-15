from __future__ import annotations

import pytest

from lol_kills.ml.eval import evaluate_gates


@pytest.mark.parametrize(
    ("model_brier", "mean_brier", "elo_brier", "expected"),
    (
        (0.20, 0.25, 0.20, True),
        (0.20, 0.25, 0.19, False),
        (0.25, 0.25, 0.19, False),
    ),
)
def test_win_gate_preserves_mean_and_elo_baseline_semantics(
    model_brier: float,
    mean_brier: float,
    elo_brier: float,
    expected: bool,
) -> None:
    result = evaluate_gates(
        {
            "win": {
                "status": "ok",
                "holdout": {"brier": model_brier},
                "baselines": {"mean_brier": mean_brier, "elo_brier": elo_brier},
            },
            "kills": {"status": "skipped"},
        }
    )
    assert result["details"]["win"]["pass"] is expected
