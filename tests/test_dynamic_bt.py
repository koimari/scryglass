from __future__ import annotations

import math
import unittest

import pandas as pd

from lol_kills.ratings.dynamic_bt import (
    DynamicBTConfig,
    HyperparameterCandidate,
    run_hyperparameter_tournament,
    run_prequential_dynamic_bt,
)


def _maps(
    outcomes: list[int],
    *,
    start: str = "2026-01-01",
    blue: str = "org-a",
    red: str = "org-b",
    blue_league: str = "L1",
    red_league: str = "L1",
    competition: str = "domestic",
) -> pd.DataFrame:
    start_at = pd.Timestamp(start)
    return pd.DataFrame(
        [
            {
                "game_uid": f"g{index:03d}",
                "date": start_at + pd.Timedelta(days=index),
                "blue_team_key": blue,
                "red_team_key": red,
                "blue_league": blue_league,
                "red_league": red_league,
                "competition": competition,
                "y_blue_win": outcome,
            }
            for index, outcome in enumerate(outcomes)
        ]
    )


class DynamicBradleyTerryTests(unittest.TestCase):
    def test_predictions_are_made_before_current_and_future_outcomes(self) -> None:
        config = DynamicBTConfig(
            blue_side_prior_logit=0.0,
            enable_bridge_terms=False,
            mean_reversion_half_life_days=None,
        )
        baseline = _maps([1, 1, 0, 0, 1])
        changed_current = baseline.copy()
        changed_current.loc[0, "y_blue_win"] = 0
        changed_future = baseline.copy()
        changed_future.loc[3:, "y_blue_win"] = [1, 0]

        baseline_run = run_prequential_dynamic_bt(baseline, config=config)
        current_run = run_prequential_dynamic_bt(changed_current, config=config)
        future_run = run_prequential_dynamic_bt(changed_future, config=config)

        self.assertEqual(
            baseline_run.predictions.loc[0, "p_blue"],
            current_run.predictions.loc[0, "p_blue"],
        )
        self.assertNotEqual(
            baseline_run.predictions.loc[1, "p_blue"],
            current_run.predictions.loc[1, "p_blue"],
        )
        pd.testing.assert_series_equal(
            baseline_run.predictions.loc[:2, "p_blue"],
            future_run.predictions.loc[:2, "p_blue"],
            check_names=False,
            check_exact=True,
        )
        self.assertTrue(
            baseline_run.predictions["prediction_before_outcome"].all()
        )

    def test_row_order_is_deterministic_after_chronological_sort(self) -> None:
        frame = _maps([1, 0, 1, 1, 0, 0])
        frame.loc[[0, 1, 2], "date"] = pd.Timestamp("2026-01-01")
        frame.loc[[3, 4], "date"] = pd.Timestamp("2026-01-02")
        shuffled = frame.sample(frac=1.0, random_state=91)
        config = DynamicBTConfig(enable_bridge_terms=False)

        ordered_run = run_prequential_dynamic_bt(frame, config=config)
        shuffled_run = run_prequential_dynamic_bt(shuffled, config=config)

        columns = [
            "row_id",
            "p_blue",
            "latent_logit",
            "predictive_variance",
            "blue_team_mean",
            "red_team_mean",
        ]
        pd.testing.assert_frame_equal(
            ordered_run.predictions[columns].reset_index(drop=True),
            shuffled_run.predictions[columns].reset_index(drop=True),
            check_exact=True,
        )

    def test_same_timestamp_rows_do_not_leak_into_each_other(self) -> None:
        frame = _maps([1, 1])
        frame["date"] = pd.Timestamp("2026-01-01")
        changed = frame.copy()
        changed.loc[0, "y_blue_win"] = 0
        config = DynamicBTConfig(
            blue_side_prior_logit=0.0, enable_bridge_terms=False
        )

        original = run_prequential_dynamic_bt(frame, config=config)
        counterfactual = run_prequential_dynamic_bt(changed, config=config)

        self.assertEqual(
            original.predictions.loc[1, "p_blue"],
            counterfactual.predictions.loc[1, "p_blue"],
        )

    def test_side_swap_is_exact_when_nuisance_context_is_swapped(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "game_uid": "swap-1",
                    "date": "2026-01-01",
                    "blue_team_key": "a1",
                    "red_team_key": "b1",
                    "blue_league": "L1",
                    "red_league": "L2",
                    "competition": "INTL",
                    "y_blue_win": 1,
                },
                {
                    "game_uid": "swap-2",
                    "date": "2026-01-02",
                    "blue_team_key": "a2",
                    "red_team_key": "b2",
                    "blue_league": "L1",
                    "red_league": "L2",
                    "competition": "INTL",
                    "y_blue_win": 1,
                },
                {
                    "game_uid": "swap-3",
                    "date": "2026-01-03",
                    "blue_team_key": "a1",
                    "red_team_key": "b2",
                    "blue_league": "L1",
                    "red_league": "L2",
                    "competition": "INTL",
                    "y_blue_win": 0,
                },
            ]
        )
        run = run_prequential_dynamic_bt(
            frame,
            config=DynamicBTConfig(
                min_bridge_maps=2,
                min_bridge_teams_per_context=2,
            ),
        )
        timestamp = "2026-02-01"
        original = run.model.predict(
            "a1",
            "b2",
            timestamp=timestamp,
            blue_context="L1",
            red_context="L2",
            side_indicator=1,
        )
        swapped = run.model.predict(
            "b2",
            "a1",
            timestamp=timestamp,
            blue_context="L2",
            red_context="L1",
            side_indicator=-1,
        )

        self.assertEqual(original.bridge_status, "supported")
        self.assertEqual(swapped.bridge_status, "supported")
        self.assertEqual(swapped.latent_logit, -original.latent_logit)
        self.assertEqual(swapped.predictive_variance, original.predictive_variance)
        self.assertEqual(swapped.probability, 1.0 - original.probability)
        self.assertGreater(original.probability, 0.0)
        self.assertLess(original.probability, 1.0)

    def test_disconnected_leagues_are_flagged_with_wider_uncertainty(self) -> None:
        l1 = _maps([1, 0, 1], blue="a", red="b")
        l2 = _maps(
            [0, 1, 0],
            start="2026-01-10",
            blue="c",
            red="d",
            blue_league="L2",
            red_league="L2",
        )
        frame = pd.concat([l1, l2], ignore_index=True)
        config = DynamicBTConfig(
            min_bridge_maps=2,
            min_bridge_teams_per_context=1,
            unsupported_bridge_variance=3.0,
        )
        run = run_prequential_dynamic_bt(frame, config=config)
        timestamp = "2026-02-01"
        within = run.model.predict(
            "a",
            "b",
            timestamp=timestamp,
            blue_context="L1",
            red_context="L1",
        )
        disconnected = run.model.predict(
            "a",
            "b",
            timestamp=timestamp,
            blue_context="L1",
            red_context="L2",
        )

        self.assertEqual(within.bridge_status, "within_context")
        self.assertEqual(disconnected.bridge_status, "unsupported")
        self.assertGreater(
            disconnected.predictive_variance, within.predictive_variance
        )
        self.assertLessEqual(
            abs(disconnected.probability - 0.5),
            abs(within.probability - 0.5),
        )

    def test_bridge_term_activates_only_after_identifiability_threshold(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "game_uid": "bridge-1",
                    "date": "2026-01-01",
                    "blue_team_key": "a1",
                    "red_team_key": "b1",
                    "blue_league": "L1",
                    "red_league": "L2",
                    "competition": "INTL-A",
                    "y_blue_win": 1,
                },
                {
                    "game_uid": "bridge-2",
                    "date": "2026-01-02",
                    "blue_team_key": "a2",
                    "red_team_key": "b2",
                    "blue_league": "L1",
                    "red_league": "L2",
                    "competition": "INTL-A",
                    "y_blue_win": 0,
                },
                {
                    "game_uid": "bridge-3",
                    "date": "2026-01-03",
                    "blue_team_key": "a1",
                    "red_team_key": "b2",
                    "blue_league": "L1",
                    "red_league": "L2",
                    "competition": "INTL-A",
                    "y_blue_win": 1,
                },
            ]
        )
        config = DynamicBTConfig(
            min_bridge_maps=2,
            min_bridge_teams_per_context=2,
            min_bridge_competitions=1,
        )
        run = run_prequential_dynamic_bt(frame, config=config)

        self.assertEqual(
            run.predictions["bridge_status"].tolist(),
            ["unsupported", "unsupported", "supported"],
        )
        bridge_audit = run.audit["model"]["bridge"]
        self.assertEqual(bridge_audit["active_edges"], 1)
        self.assertTrue(bridge_audit["edges"][0]["identifiable"])

    def test_inactivity_inflates_uncertainty(self) -> None:
        config = DynamicBTConfig(
            enable_bridge_terms=False,
            team_variance_per_day=0.02,
            mean_reversion_half_life_days=None,
        )
        run = run_prequential_dynamic_bt(_maps([1, 0, 1]), config=config)
        near = run.model.predict("org-a", "org-b", timestamp="2026-01-04")
        far = run.model.predict("org-a", "org-b", timestamp="2026-03-04")

        self.assertGreater(far.predictive_variance, near.predictive_variance)
        self.assertLessEqual(
            abs(far.probability - 0.5), abs(near.probability - 0.5)
        )

    def test_dynamic_state_adapts_after_a_result_shock(self) -> None:
        frame = _maps([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        config = DynamicBTConfig(
            blue_side_prior_logit=0.0,
            blue_side_prior_sd=0.10,
            team_prior_sd=1.0,
            team_variance_per_day=0.02,
            mean_reversion_half_life_days=None,
            enable_bridge_terms=False,
        )
        run = run_prequential_dynamic_bt(frame, config=config)
        shock_predictions = run.predictions.loc[6:, "p_blue"].reset_index(
            drop=True
        )

        self.assertGreater(shock_predictions.iloc[0], 0.5)
        self.assertLess(shock_predictions.iloc[-1], shock_predictions.iloc[0])
        self.assertLess(shock_predictions.iloc[-1], 0.5)

    def test_map_grain_audit_cutoff_and_exclusions(self) -> None:
        frame = _maps([1, 0, 1, 0])
        frame["scheduled_best_of"] = [1, 3, 5, 99]
        invalid = pd.DataFrame(
            [
                {
                    "game_uid": "bad-outcome",
                    "date": "2026-01-02",
                    "blue_team_key": "x",
                    "red_team_key": "y",
                    "blue_league": "L1",
                    "red_league": "L1",
                    "competition": "domestic",
                    "y_blue_win": 2,
                }
            ]
        )
        frame = pd.concat([frame, invalid], ignore_index=True)
        run = run_prequential_dynamic_bt(
            frame,
            config=DynamicBTConfig(enable_bridge_terms=False),
            data_cutoff="2026-01-03",
        )

        self.assertEqual(len(run.predictions), 3)
        self.assertEqual(run.audit["accepted_map_rows"], 3)
        self.assertEqual(run.audit["excluded_rows"], 2)
        self.assertEqual(
            run.audit["row_exclusion_counts"]["invalid_binary_outcome"], 1
        )
        self.assertEqual(
            run.audit["row_exclusion_counts"]["after_data_cutoff"], 1
        )
        self.assertEqual(run.audit["data_cutoff_source"], "supplied")
        self.assertIn(
            "no scheduled-series-format assumption",
            run.audit["observation_unit"],
        )

    def test_hyperparameter_tournament_selects_on_validation_only(self) -> None:
        frame = _maps(
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        )
        candidates = (
            HyperparameterCandidate(
                "slow",
                DynamicBTConfig(
                    team_variance_per_day=0.001,
                    mean_reversion_half_life_days=600.0,
                    enable_bridge_terms=False,
                ),
            ),
            HyperparameterCandidate(
                "fast",
                DynamicBTConfig(
                    team_variance_per_day=0.02,
                    mean_reversion_half_life_days=120.0,
                    enable_bridge_terms=False,
                ),
            ),
        )
        kwargs = {
            "validation_start": "2026-01-09",
            "test_start": "2026-01-14",
            "candidates": candidates,
            "calibration_bins": 5,
        }
        result = run_hyperparameter_tournament(frame, **kwargs)
        changed_test = frame.copy()
        changed_test.loc[13:, "y_blue_win"] = (
            1 - changed_test.loc[13:, "y_blue_win"]
        )
        counterfactual = run_hyperparameter_tournament(changed_test, **kwargs)

        self.assertEqual(
            result.selected_candidate, counterfactual.selected_candidate
        )
        pd.testing.assert_frame_equal(
            result.validation_scores,
            counterfactual.validation_scores,
            check_exact=True,
        )
        pd.testing.assert_frame_equal(
            result.calibration_scores,
            counterfactual.calibration_scores,
            check_exact=True,
        )
        self.assertEqual(
            result.calibration.method, counterfactual.calibration.method
        )
        self.assertEqual(
            result.calibration.intercept,
            counterfactual.calibration.intercept,
        )
        self.assertEqual(
            result.calibration.slope, counterfactual.calibration.slope
        )
        pd.testing.assert_frame_equal(
            result.validation_calibrated_ledger,
            counterfactual.validation_calibrated_ledger,
            check_exact=True,
        )
        self.assertEqual(result.test_evaluation.n, 5)
        self.assertEqual(
            len(result.test_evaluation.calibration_inputs), 5
        )
        self.assertTrue(math.isfinite(result.test_evaluation.log_loss))
        self.assertTrue(math.isfinite(result.test_evaluation.brier))
        self.assertFalse(
            result.audit["selection"]["test_used_for_selection"]
        )
        self.assertTrue(
            result.audit["selection"]["hyperparameters_frozen_before_test"]
        )
        self.assertEqual(
            result.validation_evaluation.n,
            result.audit["split"]["validation_rows"],
        )
        self.assertEqual(result.calibration.fit_split, "validation")
        self.assertEqual(
            result.calibration.fit_rows,
            result.audit["split"]["validation_rows"],
        )
        self.assertIn(
            "slope_status", result.calibration.diagnostics
        )
        self.assertFalse(
            result.calibration.diagnostics["test_labels_used"]
        )
        self.assertFalse(
            result.audit["calibration"]["test_labels_used"]
        )
        self.assertIn("probability", result.test_raw_ledger)
        self.assertIn("probability", result.test_calibrated_ledger)
        self.assertEqual(
            result.test_raw_evaluation.log_loss,
            result.test_evaluation.log_loss,
        )

    def test_final_test_label_cannot_change_any_probability_or_calibration(
        self,
    ) -> None:
        frame = _maps(
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        )
        candidate = HyperparameterCandidate(
            "fixed",
            DynamicBTConfig(enable_bridge_terms=False),
        )
        kwargs = {
            "validation_start": "2026-01-09",
            "test_start": "2026-01-14",
            "candidates": (candidate,),
            "calibration_bins": 5,
        }
        original = run_hyperparameter_tournament(frame, **kwargs)
        changed = frame.copy()
        changed.loc[changed.index[-1], "y_blue_win"] = (
            1 - changed.loc[changed.index[-1], "y_blue_win"]
        )
        counterfactual = run_hyperparameter_tournament(changed, **kwargs)

        self.assertEqual(original.calibration, counterfactual.calibration)
        pd.testing.assert_series_equal(
            original.test_raw_ledger["probability"],
            counterfactual.test_raw_ledger["probability"],
            check_exact=True,
        )
        pd.testing.assert_series_equal(
            original.test_calibrated_ledger["probability"],
            counterfactual.test_calibrated_ledger["probability"],
            check_exact=True,
        )
        self.assertNotEqual(
            original.test_raw_evaluation.log_loss,
            counterfactual.test_raw_evaluation.log_loss,
        )

    def test_promotion_evidence_requires_and_blocks_on_canonical_series(
        self,
    ) -> None:
        frame = _maps(
            [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        )
        result = run_hyperparameter_tournament(
            frame,
            validation_start="2026-01-09",
            test_start="2026-01-14",
            candidates=(
                HyperparameterCandidate(
                    "fixed",
                    DynamicBTConfig(enable_bridge_terms=False),
                ),
            ),
            calibration_bins=5,
        )
        baseline = result.test_raw_ledger[
            ["row_id", "y_true"]
        ].rename(columns={"y_true": "outcome"})
        baseline["probability"] = 0.5
        with self.assertRaisesRegex(ValueError, "canonical_series_ids"):
            result.promotion_evidence(
                baseline,
                candidate_variant="raw",
                bootstrap_replicates=200,
            )

        baseline["canonical_series_id"] = [
            "s1",
            "s1",
            "s2",
            "s2",
            "s3",
        ]
        evidence = result.promotion_evidence(
            baseline,
            candidate_variant="raw",
            minimum_test_events=5,
            bootstrap_replicates=200,
            moving_block_size=2,
            random_seed=17,
        )

        self.assertEqual(evidence["events"], 5)
        self.assertEqual(evidence["series_clusters"], 3)
        self.assertEqual(
            evidence["evidence_path"], "explicit canonical series ids"
        )
        self.assertFalse(evidence["promotion_authorized"])
        self.assertEqual(
            evidence["research_status"],
            "research_candidate_not_for_production_promotion",
        )
        self.assertIn("confidence_interval", evidence)

        canonical_rows = []
        series_by_row = dict(
            zip(baseline["row_id"], baseline["canonical_series_id"])
        )
        for _, row in result.test_raw_ledger.iterrows():
            event_time = pd.Timestamp(row["timestamp"])
            for model_id, probability in (
                ("candidate", float(row["probability"])),
                ("baseline", 0.5),
            ):
                canonical_rows.append(
                    {
                        "prediction_id": f"{model_id}:{row['row_id']}",
                        "model_id": model_id,
                        "model_version": "frozen-v1",
                        "event_id": row["row_id"],
                        "series_id": series_by_row[row["row_id"]],
                        "prediction_time": (
                            event_time - pd.Timedelta(hours=1)
                        ),
                        "event_time": event_time,
                        "data_as_of": event_time - pd.Timedelta(hours=2),
                        "outcome": int(row["y_true"]),
                        "probability": probability,
                        "split": "test",
                        "league": "L1",
                        "patch": "test-patch",
                        "roster_state_id": "test-roster",
                    }
                )
        canonical_evidence = result.promotion_evidence(
            pd.DataFrame(canonical_rows),
            candidate_variant="raw",
            candidate_model_id="candidate",
            baseline_model_id="baseline",
            minimum_test_events=5,
            bootstrap_replicates=100,
            moving_block_size=2,
            random_seed=17,
        )
        self.assertEqual(
            canonical_evidence["evidence_path"],
            "lol_kills.model_tournament canonical ledger",
        )
        self.assertFalse(canonical_evidence["promotion_authorized"])


if __name__ == "__main__":
    unittest.main()
