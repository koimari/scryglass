from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from lol_kills.ratings.dynamic_bt import (
    DynamicBTConfig,
    evaluate_binary_predictions,
    run_prequential_dynamic_bt,
)
from lol_kills.ratings.robust_dynamic_bt import (
    RESEARCH_STATUS,
    ROBUST_MODEL_ID,
    RobustDynamicBradleyTerry,
    RobustDynamicBTConfig,
    RobustHyperparameterCandidate,
    run_prequential_robust_dynamic_bt,
    run_robust_dynamic_bt_tournament,
)


def _maps(
    outcomes: list[int],
    *,
    start: str = "2026-01-01",
    blue: str = "org-a",
    red: str = "org-b",
    blue_context: str = "L1",
    red_context: str = "L1",
    competition: str = "domestic",
) -> pd.DataFrame:
    start_at = pd.Timestamp(start)
    return pd.DataFrame(
        [
            {
                "game_uid": f"g{index:04d}",
                "date": start_at + pd.Timedelta(days=index),
                "blue_team_key": blue,
                "red_team_key": red,
                "blue_league": blue_context,
                "red_league": red_context,
                "competition": competition,
                "y_blue_win": outcome,
            }
            for index, outcome in enumerate(outcomes)
        ]
    )


def _base_config() -> DynamicBTConfig:
    return DynamicBTConfig(
        blue_side_prior_logit=0.0,
        blue_side_prior_sd=0.10,
        team_prior_sd=0.80,
        team_variance_per_day=0.001,
        context_variance_per_day=0.0,
        side_variance_per_day=0.0,
        mean_reversion_half_life_days=500.0,
        enable_bridge_terms=False,
        max_team_variance=8.0,
    )


def _baseline_ledger(
    frame: pd.DataFrame,
    *,
    base_config: DynamicBTConfig,
    test_start: str,
    dual_probability: float = 0.5,
) -> pd.DataFrame:
    dynamic = run_prequential_dynamic_bt(frame, config=base_config).predictions
    test_at = pd.Timestamp(test_start, tz="UTC")
    dynamic = dynamic.loc[dynamic["timestamp"] >= test_at].reset_index(drop=True)
    rows = []
    for model_id, probability in (
        ("dynamic_bt_raw", dynamic["p_blue"].to_numpy(float)),
        (
            "dual_elo",
            np.full(len(dynamic), dual_probability, dtype=float),
        ),
    ):
        rows.append(
            pd.DataFrame(
                {
                    "model_id": model_id,
                    "row_id": dynamic["row_id"].astype(str),
                    "timestamp": dynamic["timestamp"],
                    "y_true": dynamic["y_true"].astype(float),
                    "probability": probability,
                    "prediction_before_outcome": True,
                    "uses_same_event_post_map_features": False,
                    "historical_update_contract": ("binary_map_outcome_only"),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _stationary_maps(seed: int = 0, n: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = np.array(["org-a", "org-b", "org-c", "org-d"])
    strength = {
        "org-a": 0.80,
        "org-b": 0.25,
        "org-c": -0.20,
        "org-d": -0.70,
    }
    start = pd.Timestamp("2025-01-01")
    rows = []
    for index in range(n):
        blue, red = rng.choice(teams, size=2, replace=False)
        logit = strength[str(blue)] - strength[str(red)] + 0.08
        probability = 1.0 / (1.0 + math.exp(-logit))
        rows.append(
            {
                "game_uid": f"stationary-{index:04d}",
                "date": start + pd.Timedelta(days=index),
                "blue_team_key": str(blue),
                "red_team_key": str(red),
                "blue_league": "L1",
                "red_league": "L1",
                "competition": "domestic",
                "y_blue_win": int(rng.random() < probability),
            }
        )
    return pd.DataFrame(rows)


class RobustDynamicBradleyTerryTests(unittest.TestCase):
    def test_no_shock_path_exactly_recovers_existing_raw_dynamic_bt(
        self,
    ) -> None:
        frame = _maps([1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
        base = _base_config()
        existing = run_prequential_dynamic_bt(frame, config=base)
        robust = run_prequential_robust_dynamic_bt(
            frame,
            config=RobustDynamicBTConfig(
                base_config=base,
                shock_probability=0.0,
                shock_variance=0.0,
            ),
        )

        self.assertTrue(
            np.array_equal(
                existing.predictions["p_blue"].to_numpy(float),
                robust.predictions["p_blue"].to_numpy(float),
            )
        )
        self.assertTrue(
            np.array_equal(
                existing.predictions["latent_logit"].to_numpy(float),
                robust.predictions["latent_logit"].to_numpy(float),
            )
        )
        self.assertTrue(robust.predictions["innovation_components"].eq(1).all())
        self.assertTrue(robust.update_diagnostics["posterior_any_shock"].eq(0.0).all())

    def test_same_timestamp_outcomes_cannot_change_same_timestamp_predictions(
        self,
    ) -> None:
        frame = _maps([1, 1, 0, 1, 0, 0])
        frame["date"] = pd.Timestamp("2026-01-01")
        changed = frame.copy()
        changed.loc[0, "y_blue_win"] = 0
        config = RobustDynamicBTConfig(
            base_config=_base_config(),
            shock_probability=0.20,
            shock_variance=2.0,
            minimum_team_observations=0,
        )

        original = run_prequential_robust_dynamic_bt(frame, config=config)
        counterfactual = run_prequential_robust_dynamic_bt(changed, config=config)

        pd.testing.assert_series_equal(
            original.predictions["p_blue"],
            counterfactual.predictions["p_blue"],
            check_exact=True,
        )
        self.assertTrue(original.predictions["prediction_before_outcome"].all())
        self.assertFalse(
            original.update_diagnostics["used_for_current_prediction"].any()
        )

    def test_side_swap_is_an_exact_complement_with_shock_mixture(
        self,
    ) -> None:
        frame = _maps([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
        config = RobustDynamicBTConfig(
            base_config=_base_config(),
            shock_probability=0.10,
            shock_variance=1.25,
            minimum_team_observations=2,
        )
        run = run_prequential_robust_dynamic_bt(frame, config=config)

        original = run.model.predict(
            "org-a",
            "org-b",
            timestamp="2026-03-01",
            side_indicator=1,
        )
        swapped = run.model.predict(
            "org-b",
            "org-a",
            timestamp="2026-03-01",
            side_indicator=-1,
        )

        self.assertEqual(swapped.latent_logit, -original.latent_logit)
        self.assertEqual(
            swapped.predictive_variance,
            original.predictive_variance,
        )
        self.assertEqual(swapped.probability, 1.0 - original.probability)
        self.assertEqual(swapped.blue_shock_prior, original.red_shock_prior)
        self.assertEqual(swapped.red_shock_prior, original.blue_shock_prior)

    def test_bridge_support_remains_fail_closed(self) -> None:
        l1 = _maps([1, 0, 1], blue="a", red="b")
        l2 = _maps(
            [0, 1, 0],
            start="2026-01-10",
            blue="c",
            red="d",
            blue_context="L2",
            red_context="L2",
        )
        config = RobustDynamicBTConfig(
            base_config=DynamicBTConfig(
                min_bridge_maps=2,
                min_bridge_teams_per_context=1,
                unsupported_bridge_variance=3.0,
            ),
            shock_probability=0.10,
            shock_variance=1.0,
            minimum_team_observations=0,
        )
        run = run_prequential_robust_dynamic_bt(
            pd.concat([l1, l2], ignore_index=True),
            config=config,
        )
        within = run.model.predict(
            "a",
            "b",
            timestamp="2026-02-01",
            blue_context="L1",
            red_context="L1",
        )
        disconnected = run.model.predict(
            "a",
            "b",
            timestamp="2026-02-01",
            blue_context="L1",
            red_context="L2",
        )

        self.assertEqual(within.bridge_status, "within_context")
        self.assertEqual(disconnected.bridge_status, "unsupported")
        self.assertEqual(disconnected.blue_context_mean, 0.0)
        self.assertEqual(disconnected.red_context_mean, 0.0)
        self.assertGreater(
            disconnected.predictive_variance,
            within.predictive_variance,
        )
        self.assertLessEqual(
            abs(disconnected.probability - 0.5),
            abs(within.probability - 0.5),
        )

    def test_display_alias_series_and_post_map_columns_are_ignored(
        self,
    ) -> None:
        frame = _maps([1, 0, 1, 1, 0, 1, 0, 0, 1])
        decorated = frame.copy()
        decorated["blue_team"] = [
            "Renamed Display A" if index % 2 else "Alias A"
            for index in range(len(frame))
        ]
        decorated["red_team"] = "Mixed Organization Label"
        decorated["series_score_before"] = np.arange(len(frame))
        decorated["series_winner"] = "forbidden"
        decorated["blue_gold_at_15"] = np.linspace(-5000.0, 5000.0, len(frame))
        decorated["kills"] = np.arange(len(frame)) * 7
        decorated["patch"] = "future-patch-label"
        decorated["roster_hash"] = "post-map-roster"
        config = RobustDynamicBTConfig(
            base_config=_base_config(),
            shock_probability=0.10,
            shock_variance=1.0,
            minimum_team_observations=2,
        )

        plain = run_prequential_robust_dynamic_bt(frame, config=config)
        extra = run_prequential_robust_dynamic_bt(decorated, config=config)

        pd.testing.assert_series_equal(
            plain.predictions["p_blue"],
            extra.predictions["p_blue"],
            check_exact=True,
        )
        self.assertEqual(
            extra.audit["permitted_predictors"],
            plain.audit["permitted_predictors"],
        )
        self.assertIn(
            "series score or scheduled format",
            extra.audit["forbidden_predictors"],
        )

    def test_abrupt_shock_gets_posterior_support_and_recovers_faster(
        self,
    ) -> None:
        frame = _maps([1] * 30 + [0] * 24)
        base = DynamicBTConfig(
            blue_side_prior_logit=0.0,
            blue_side_prior_sd=0.05,
            team_prior_sd=0.60,
            team_variance_per_day=0.0,
            context_variance_per_day=0.0,
            side_variance_per_day=0.0,
            mean_reversion_half_life_days=None,
            enable_bridge_terms=False,
            max_team_variance=8.0,
        )
        raw = run_prequential_dynamic_bt(frame, config=base)
        robust = run_prequential_robust_dynamic_bt(
            frame,
            config=RobustDynamicBTConfig(
                base_config=base,
                shock_probability=0.20,
                shock_variance=2.0,
                minimum_team_observations=5,
            ),
        )
        first_reversal = robust.update_diagnostics.iloc[30]
        raw_after = evaluate_binary_predictions(raw.predictions.iloc[30:])
        robust_after = evaluate_binary_predictions(robust.predictions.iloc[30:])

        self.assertGreater(
            first_reversal["posterior_any_shock"],
            first_reversal["prior_any_shock"],
        )
        self.assertLess(robust_after.log_loss, raw_after.log_loss)
        self.assertLess(robust_after.brier, raw_after.brier)
        self.assertLess(
            robust.predictions.iloc[-1]["p_blue"],
            raw.predictions.iloc[-1]["p_blue"],
        )

    def test_stationary_negative_control_rejects_unneeded_shock_complexity(
        self,
    ) -> None:
        frame = _stationary_maps(seed=0)
        base = _base_config()
        candidates = (
            RobustHyperparameterCandidate(
                "gaussian_no_shock",
                RobustDynamicBTConfig(
                    base_config=base,
                    shock_probability=0.0,
                    shock_variance=0.0,
                ),
                complexity_rank=0,
            ),
            RobustHyperparameterCandidate(
                "frequent_large_shock",
                RobustDynamicBTConfig(
                    base_config=base,
                    shock_probability=0.10,
                    shock_variance=1.50,
                    minimum_team_observations=4,
                ),
                complexity_rank=1,
            ),
        )
        baselines = _baseline_ledger(
            frame,
            base_config=base,
            test_start="2025-09-01",
        )

        result = run_robust_dynamic_bt_tournament(
            frame,
            validation_start="2025-06-01",
            test_start="2025-09-01",
            baseline_ledger=baselines,
            candidates=candidates,
            calibration_bins=5,
        )

        self.assertEqual(result.selected_candidate, "gaussian_no_shock")
        validation = result.validation_scores.set_index("candidate")
        self.assertLess(
            validation.loc["gaussian_no_shock", "validation_log_loss"],
            validation.loc["frequent_large_shock", "validation_log_loss"],
        )

    def test_complexity_loses_exact_validation_ties(self) -> None:
        frame = _maps([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
        base = _base_config()
        same = RobustDynamicBTConfig(
            base_config=base,
            shock_probability=0.0,
            shock_variance=0.0,
        )
        candidates = (
            RobustHyperparameterCandidate("complex_duplicate", same, complexity_rank=5),
            RobustHyperparameterCandidate("simple", same, complexity_rank=0),
        )
        baselines = _baseline_ledger(
            frame,
            base_config=base,
            test_start="2026-01-14",
        )

        result = run_robust_dynamic_bt_tournament(
            frame,
            validation_start="2026-01-09",
            test_start="2026-01-14",
            baseline_ledger=baselines,
            candidates=candidates,
            calibration_bins=5,
        )

        self.assertEqual(result.selected_candidate, "simple")
        scores = result.validation_scores.set_index("candidate")
        self.assertEqual(
            scores.loc["simple", "validation_log_loss"],
            scores.loc["complex_duplicate", "validation_log_loss"],
        )
        self.assertEqual(
            scores.loc["simple", "validation_brier"],
            scores.loc["complex_duplicate", "validation_brier"],
        )

    def test_tournament_selection_never_reads_final_test_labels(
        self,
    ) -> None:
        frame = _maps([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        base = _base_config()
        candidates = (
            RobustHyperparameterCandidate(
                "no_shock",
                RobustDynamicBTConfig(
                    base_config=base,
                    shock_probability=0.0,
                    shock_variance=0.0,
                ),
                complexity_rank=0,
            ),
            RobustHyperparameterCandidate(
                "shock",
                RobustDynamicBTConfig(
                    base_config=base,
                    shock_probability=0.10,
                    shock_variance=1.0,
                    minimum_team_observations=2,
                ),
                complexity_rank=1,
            ),
        )
        original_baselines = _baseline_ledger(
            frame,
            base_config=base,
            test_start="2026-01-14",
        )
        original = run_robust_dynamic_bt_tournament(
            frame,
            validation_start="2026-01-09",
            test_start="2026-01-14",
            baseline_ledger=original_baselines,
            candidates=candidates,
            calibration_bins=5,
        )

        changed = frame.copy()
        changed.loc[changed.index[-1], "y_blue_win"] = (
            1 - changed.loc[changed.index[-1], "y_blue_win"]
        )
        changed_baselines = _baseline_ledger(
            changed,
            base_config=base,
            test_start="2026-01-14",
        )
        counterfactual = run_robust_dynamic_bt_tournament(
            changed,
            validation_start="2026-01-09",
            test_start="2026-01-14",
            baseline_ledger=changed_baselines,
            candidates=candidates,
            calibration_bins=5,
        )

        self.assertEqual(
            original.selected_candidate,
            counterfactual.selected_candidate,
        )
        pd.testing.assert_frame_equal(
            original.validation_scores,
            counterfactual.validation_scores,
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            original.test_predictions["p_blue"],
            counterfactual.test_predictions["p_blue"],
            check_exact=True,
        )
        self.assertFalse(original.audit["selection"]["test_used_for_selection"])
        self.assertTrue(
            original.audit["selection"]["hyperparameters_frozen_before_test"]
        )

    def test_baseline_seam_compares_raw_dynamic_and_dual_elo(self) -> None:
        frame = _maps([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
        base = _base_config()
        baselines = _baseline_ledger(
            frame,
            base_config=base,
            test_start="2026-01-14",
            dual_probability=0.52,
        )
        baselines.loc[
            baselines["model_id"].eq("dual_elo"),
            "historical_update_contract",
        ] = "binary_result_plus_historical_gold_margin"
        result = run_robust_dynamic_bt_tournament(
            frame,
            validation_start="2026-01-09",
            test_start="2026-01-14",
            baseline_ledger=baselines,
            candidates=(
                RobustHyperparameterCandidate(
                    "fixed",
                    RobustDynamicBTConfig(
                        base_config=base,
                        shock_probability=0.05,
                        shock_variance=0.75,
                        minimum_team_observations=2,
                    ),
                    complexity_rank=1,
                ),
            ),
            calibration_bins=5,
        )

        self.assertEqual(
            result.test_scores["model_id"].tolist(),
            [ROBUST_MODEL_ID, "dynamic_bt_raw", "dual_elo"],
        )
        self.assertTrue(
            {
                "log_loss",
                "brier",
                "ece",
                "pav_in_sample_miscalibration",
            }.issubset(result.test_scores.columns)
        )
        self.assertFalse(result.audit["promotion_authorized"])
        self.assertFalse(result.audit["sota_claim_authorized"])
        self.assertEqual(result.audit["status"], RESEARCH_STATUS)
        self.assertEqual(
            result.audit["baseline_seam"]["historical_update_contracts"]["dual_elo"],
            ["binary_result_plus_historical_gold_margin"],
        )

    def test_baseline_seam_fails_closed_on_post_map_or_misaligned_rows(
        self,
    ) -> None:
        frame = _maps([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        base = _base_config()
        valid = _baseline_ledger(
            frame,
            base_config=base,
            test_start="2026-01-12",
        )
        candidate = (
            RobustHyperparameterCandidate(
                "fixed",
                RobustDynamicBTConfig(base_config=base),
                complexity_rank=1,
            ),
        )
        post_map = valid.copy()
        post_map.loc[
            post_map["model_id"].eq("dual_elo"),
            "uses_same_event_post_map_features",
        ] = True
        with self.assertRaisesRegex(ValueError, "post-map"):
            run_robust_dynamic_bt_tournament(
                frame,
                validation_start="2026-01-07",
                test_start="2026-01-12",
                baseline_ledger=post_map,
                candidates=candidate,
            )

        missing = valid.drop(valid.index[-1]).reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "exactly"):
            run_robust_dynamic_bt_tournament(
                frame,
                validation_start="2026-01-07",
                test_start="2026-01-12",
                baseline_ledger=missing,
                candidates=candidate,
            )

    def test_paired_cluster_block_uncertainty_requires_explicit_clusters(
        self,
    ) -> None:
        frame = _maps([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
        base = _base_config()
        result = run_robust_dynamic_bt_tournament(
            frame,
            validation_start="2026-01-09",
            test_start="2026-01-14",
            baseline_ledger=_baseline_ledger(
                frame,
                base_config=base,
                test_start="2026-01-14",
            ),
            candidates=(
                RobustHyperparameterCandidate(
                    "fixed",
                    RobustDynamicBTConfig(
                        base_config=base,
                        shock_probability=0.05,
                        shock_variance=0.75,
                        minimum_team_observations=2,
                    ),
                    complexity_rank=1,
                ),
            ),
            calibration_bins=5,
        )
        row_ids = result.test_predictions["row_id"].astype(str).tolist()
        clusters = {
            row_id: f"cluster-{index // 2}" for index, row_id in enumerate(row_ids)
        }
        report = result.paired_block_comparison(
            "dynamic_bt_raw",
            cluster_ids=clusters,
            score="log_loss",
            bootstrap_replicates=200,
            moving_block_size=2,
            random_seed=17,
        )

        self.assertEqual(report["events"], len(row_ids))
        self.assertEqual(report["clusters"], 3)
        self.assertEqual(report["bootstrap"]["block_size_clusters"], 2)
        self.assertEqual(len(report["confidence_interval"]), 2)
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["sota_claim_authorized"])

        incomplete = dict(clusters)
        incomplete.pop(row_ids[-1])
        with self.assertRaisesRegex(ValueError, "missing test rows"):
            result.paired_block_comparison(
                "dual_elo",
                cluster_ids=incomplete,
                bootstrap_replicates=100,
            )

    def test_config_validation_and_research_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "shock_probability"):
            RobustDynamicBTConfig(shock_probability=1.0)
        with self.assertRaisesRegex(ValueError, "shock_variance"):
            RobustDynamicBTConfig(shock_variance=-0.1)
        with self.assertRaisesRegex(ValueError, "minimum_team_observations"):
            RobustDynamicBTConfig(minimum_team_observations=-1)

        model = RobustDynamicBradleyTerry()
        audit = model.audit_snapshot()
        self.assertEqual(audit["status"], RESEARCH_STATUS)
        self.assertIn("not exact BOCPD", audit["limitations"][0])


if __name__ == "__main__":
    unittest.main()
