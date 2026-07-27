from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from lol_kills.export.player_performance_artifacts import (
    build_player_performance_public_artifacts,
    render_player_performance_public_artifacts,
    select_canonical_player_performance_rows,
)
from lol_kills.ratings.player_performance import (
    ESTIMAND,
    PairedRMSEContrast,
    PerformanceMetrics,
    PlayerPerformanceConfig,
    PlayerPerformanceDataError,
)


def _metric(rows: int, rmse: float) -> PerformanceMetrics:
    return PerformanceMetrics(
        rows=rows,
        rmse=rmse,
        mae=rmse * 0.8,
        r2=0.1,
        spearman=0.2,
        zero_baseline_rmse=1.1,
        relative_rmse_lift=(1.1 - rmse) / 1.1,
    )


def _tournament() -> SimpleNamespace:
    ratings = pd.DataFrame(
        [
            {
                "player_id": "p1",
                "player_name": "One",
                "role": "top",
                "last_team_key": "team-a",
                "last_date": pd.Timestamp("2026-04-20", tz="UTC"),
                "maps": 50,
                "performance_mean": 0.4,
                "performance_sd": 0.1,
                "conservative_performance": 0.2355,
                "estimand": ESTIMAND,
                "uncertainty_method": "exact_penalized_hessian_diagonal",
                "promotion_status": "research_candidate_not_production",
            },
            {
                "player_id": "p2",
                "player_name": "Two",
                "role": "top",
                "last_team_key": "team-b",
                "last_date": pd.Timestamp("2026-04-21", tz="UTC"),
                "maps": 48,
                "performance_mean": 0.4,
                "performance_sd": 0.1,
                "conservative_performance": 0.2355,
                "estimand": ESTIMAND,
                "uncertainty_method": "exact_penalized_hessian_diagonal",
                "promotion_status": "research_candidate_not_production",
            },
            {
                "player_id": "p3",
                "player_name": "Three",
                "role": "top",
                "last_team_key": "team-c",
                "last_date": pd.Timestamp("2026-04-19", tz="UTC"),
                "maps": 42,
                "performance_mean": 0.2,
                "performance_sd": 0.1,
                "conservative_performance": 0.0355,
                "estimand": ESTIMAND,
                "uncertainty_method": "exact_penalized_hessian_diagonal",
                "promotion_status": "research_candidate_not_production",
            },
        ]
    )
    contrast = PairedRMSEContrast(
        rows=120,
        calendar_day_blocks=20,
        candidate_rmse=1.0,
        baseline_rmse=1.04,
        relative_rmse_lift=0.038,
        ci_low=0.02,
        ci_high=0.05,
        confidence_level=0.95,
        bootstrap_replicates=5_000,
        resampling_unit="calendar_day",
    )
    return SimpleNamespace(
        audit=SimpleNamespace(
            ready=True,
            eligible_matchups=100,
            stable_identity_matchups=98,
        ),
        test_gate_passed=True,
        player_ratings=ratings,
        split_boundaries={
            "train_start": "2025-01-01 00:00:00+00:00",
            "train_end": "2026-02-01 00:00:00+00:00",
            "validation_start": "2026-02-02 00:00:00+00:00",
            "validation_end": "2026-04-21 00:00:00+00:00",
            "test_start": "2026-04-22 00:00:00+00:00",
            "test_end": "2026-07-01 00:00:00+00:00",
        },
        selected_base_penalty=8.0,
        selected_context_base_penalty=32.0,
        validation_candidates=pd.DataFrame(
            [{"base_penalty": 8.0, "validation_rmse": 0.99}]
        ),
        test_metrics=_metric(120, 1.0),
        test_context_baseline_metrics=_metric(120, 1.04),
        player_incremental_test_rmse_lift=0.038,
        player_incremental_test_contrast=contrast,
        future_patch_test_metrics=_metric(80, 1.01),
        roster_move_test_metrics=_metric(30, 1.02),
        limitations=("Narrow early-resource target.",),
    )


def _selected_rows() -> pd.DataFrame:
    rows = []
    for index, (player_id, name, team, league) in enumerate(
        [
            ("p1", "One", "team-a", "LCK"),
            ("p2", "Two", "team-b", "LPL"),
            ("p3", "Three", "team-c", "LEC"),
        ]
    ):
        rows.append(
            {
                "gameid": f"g{index}",
                "date": pd.Timestamp(
                    f"2026-04-{19 + index:02d}", tz="UTC"
                ),
                "year": 2026,
                "oe_year": 2026,
                "source": "oe",
                "datacompleteness": "complete",
                "side": "Blue",
                "position": "top",
                "playerid": player_id,
                "playername": name,
                "team_key": team,
                "league": league,
            }
        )
    return pd.DataFrame(rows)


class PlayerPerformanceArtifactTests(unittest.TestCase):
    def test_compact_artifacts_preserve_exact_ties_and_contract(self) -> None:
        artifacts = render_player_performance_public_artifacts(
            _tournament(),
            _selected_rows(),
            {"input_rows": 3, "selected_complete_oe_rows": 3},
            years=(2025, 2026),
            config=PlayerPerformanceConfig(),
        )
        snapshot = artifacts.snapshot.set_index("player_id")
        self.assertEqual(snapshot.loc["p1", "rank"], 1)
        self.assertEqual(snapshot.loc["p2", "rank"], 1)
        self.assertEqual(snapshot.loc["p3", "rank"], 3)
        self.assertEqual(
            artifacts.meta["display_name"],
            "15-minute resource performance",
        )
        self.assertIn("causal player skill", artifacts.meta["non_estimands"])
        self.assertEqual(
            artifacts.meta["model_hash"],
            artifacts.validation["model_hash"],
        )
        self.assertTrue(artifacts.validation["test_gate_passed"])
        self.assertFalse(
            artifacts.validation["large_prediction_ledger_exported"]
        )
        self.assertNotIn("prediction_ledger", artifacts.validation)
        self.assertTrue(
            (artifacts.snapshot["model_hash"] == artifacts.meta["model_hash"]).all()
        )

    def test_source_selection_is_exact_and_year_conflicts_fail(self) -> None:
        rows = _selected_rows()
        extra = rows.iloc[[0]].copy()
        extra["gameid"] = "partial"
        extra["datacompleteness"] = "partial"
        grid = rows.iloc[[1]].copy()
        grid["gameid"] = "grid"
        grid["source"] = "grid"
        selected, audit = select_canonical_player_performance_rows(
            pd.concat([rows, extra, grid], ignore_index=True),
            (2025, 2026),
        )
        self.assertEqual(len(selected), 3)
        self.assertEqual(audit["incomplete_rows"], 1)
        self.assertEqual(audit["non_oe_rows"], 1)

        conflict = rows.copy()
        conflict.loc[0, "oe_year"] = 2025
        with self.assertRaisesRegex(PlayerPerformanceDataError, "disagree"):
            select_canonical_player_performance_rows(
                conflict, (2025, 2026)
            )
        with self.assertRaisesRegex(PlayerPerformanceDataError, "locked"):
            select_canonical_player_performance_rows(rows, (2026,))

    def test_failed_tournament_is_never_published(self) -> None:
        tournament = _tournament()
        tournament.test_gate_passed = False
        with self.assertRaisesRegex(PlayerPerformanceDataError, "gate"):
            render_player_performance_public_artifacts(
                tournament,
                _selected_rows(),
                {"input_rows": 3, "selected_complete_oe_rows": 3},
                years=(2025, 2026),
                config=PlayerPerformanceConfig(),
            )

    def test_stale_400_replicate_interval_is_never_published(self) -> None:
        tournament = _tournament()
        tournament.player_incremental_test_contrast = PairedRMSEContrast(
            **{
                **tournament.player_incremental_test_contrast.__dict__,
                "bootstrap_replicates": 400,
            }
        )
        with self.assertRaisesRegex(PlayerPerformanceDataError, "5,000"):
            render_player_performance_public_artifacts(
                tournament,
                _selected_rows(),
                {"input_rows": 3, "selected_complete_oe_rows": 3},
                years=(2025, 2026),
                config=PlayerPerformanceConfig(),
            )

    def test_build_path_runs_the_governed_tournament(self) -> None:
        # This assertion is deliberately structural: the public helper imports
        # and calls the one governed tournament rather than maintaining a
        # second fit implementation.
        self.assertIn(
            "run_player_performance_tournament",
            build_player_performance_public_artifacts.__code__.co_names,
        )


if __name__ == "__main__":
    unittest.main()
