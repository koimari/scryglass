from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lol_kills.ratings.player_performance import (
    CANONICAL_ROLES,
    ESTIMAND,
    PlayerPerformanceConfig,
    PlayerPerformanceDataError,
    RobustContextStandardizer,
    audit_player_map_input,
    fit_player_performance_candidate,
    prepare_player_map_matchups,
    run_player_performance_tournament,
)


def _synthetic_player_maps(
    n_games: int = 90,
    *,
    seed: int = 19,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    player_effects: dict[str, float] = {}
    pools: dict[str, list[str]] = {}
    for role_index, role in enumerate(CANONICAL_ROLES):
        pools[role] = [f"{role}-id-{index}" for index in range(6)]
        role_effects = np.asarray([-1.5, -0.9, -0.3, 0.3, 0.9, 1.5])
        role_effects = np.roll(role_effects, role_index)
        for player_id, effect in zip(pools[role], role_effects):
            player_effects[player_id] = float(effect)

    champions = {
        role: [f"{role}-champ-{index}" for index in range(5)]
        for role in CANONICAL_ROLES
    }
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for game_index in range(n_games):
        date = start + pd.Timedelta(days=game_index)
        patch = (
            "25.01"
            if game_index < n_games * 0.6
            else "25.02"
            if game_index < n_games * 0.8
            else "25.03"
        )
        league = "LCK" if game_index % 3 else "LPL"
        for role_index, role in enumerate(CANONICAL_ROLES):
            selected = rng.choice(pools[role], size=2, replace=False)
            if rng.random() < 0.5:
                selected = selected[::-1]
            blue_id, red_id = map(str, selected)
            blue_team = f"team-{(game_index + role_index) % 4}"
            red_team = f"team-{(game_index + role_index + 1) % 4}"
            blue_champion = champions[role][
                (game_index + role_index) % len(champions[role])
            ]
            red_champion = champions[role][
                (game_index + role_index + 2) % len(champions[role])
            ]
            latent = (
                player_effects[blue_id]
                - player_effects[red_id]
                + 0.10 * (blue_team > red_team)
                - 0.10 * (blue_team < red_team)
                + rng.normal(0.0, 0.20)
            )
            metrics = {
                "golddiffat15": 340.0 * latent,
                "xpdiffat15": 260.0 * latent,
                "csdiffat15": 11.0 * latent,
            }
            for side, sign, player_id, team, champion, opponent_id in (
                (
                    "Blue",
                    1.0,
                    blue_id,
                    blue_team,
                    blue_champion,
                    red_id,
                ),
                (
                    "Red",
                    -1.0,
                    red_id,
                    red_team,
                    red_champion,
                    blue_id,
                ),
            ):
                rows.append(
                    {
                        "gameid": f"g{game_index:04d}",
                        "date": date,
                        "patch": patch,
                        "league": league,
                        "source": "oe",
                        "datacompleteness": "complete",
                        "side": side,
                        "position": role,
                        "playerid": player_id,
                        "playername": player_id.replace("-id-", "-name-"),
                        "team_key": team,
                        "champion": champion,
                        "result": int(
                            (latent > 0 and side == "Blue")
                            or (latent <= 0 and side == "Red")
                        ),
                        "kills": 99 if side == "Blue" else 0,
                        "opponent_id_for_test": opponent_id,
                        **{
                            metric: sign * value
                            for metric, value in metrics.items()
                        },
                    }
                )
    return pd.DataFrame(rows), player_effects


def _config(**overrides: object) -> PlayerPerformanceConfig:
    values: dict[str, object] = {
        "min_context_player_maps": 8,
        "min_stable_identity_matchup_coverage": 0.85,
        "ridge_grid": (0.25, 1.0, 4.0),
        "min_train_matchups_per_role": 12,
        "uncertainty_probes": 4,
        "exact_covariance_max_features": 512,
        "metric_bootstrap_replicates": 400,
    }
    values.update(overrides)
    return PlayerPerformanceConfig(**values)


class PlayerPerformanceAuditTests(unittest.TestCase):
    def test_strict_oe_identity_and_antisymmetry_audit(self) -> None:
        frame, _ = _synthetic_player_maps(12)
        partial = frame.iloc[:10].copy()
        partial["gameid"] = "partial-game"
        partial["datacompleteness"] = "partial"
        partial[["golddiffat15", "xpdiffat15", "csdiffat15"]] = np.nan
        audited = audit_player_map_input(
            pd.concat((frame, partial), ignore_index=True), _config()
        )
        self.assertTrue(audited.ready)
        self.assertEqual(audited.partial_rows_excluded, 10)
        self.assertEqual(audited.non_antisymmetric_matchups, 0)
        self.assertEqual(audited.eligible_matchups, 12 * 5)

        broken = frame.copy()
        selector = (
            broken["gameid"].eq("g0000")
            & broken["position"].eq("top")
            & broken["side"].eq("Red")
        )
        broken.loc[selector, "golddiffat15"] += 1.0
        audit = audit_player_map_input(broken, _config())
        self.assertFalse(audit.ready)
        self.assertEqual(audit.non_antisymmetric_matchups, 1)
        with self.assertRaisesRegex(
            PlayerPerformanceDataError, "antisymmetry"
        ):
            prepare_player_map_matchups(broken, _config())

    def test_names_never_replace_missing_stable_ids(self) -> None:
        frame, _ = _synthetic_player_maps(12)
        selector = (
            frame["gameid"].eq("g0000")
            & frame["position"].eq("mid")
            & frame["side"].eq("Blue")
        )
        frame.loc[selector, "playerid"] = ""
        prepared = prepare_player_map_matchups(frame, _config())
        self.assertEqual(prepared.audit.missing_stable_player_rows, 1)
        self.assertEqual(prepared.audit.eligible_matchups, 12 * 5 - 1)
        self.assertNotIn(
            "g0000",
            prepared.matchups.loc[
                prepared.matchups["role"].eq("mid"), "game_id"
            ].tolist(),
        )

    def test_complete_missing_target_and_mixed_source_fail_closed(self) -> None:
        frame, _ = _synthetic_player_maps(12)
        frame.loc[0, "golddiffat15"] = np.nan
        frame.loc[1, "source"] = "grid"
        audit = audit_player_map_input(frame, _config())
        self.assertFalse(audit.ready)
        self.assertEqual(audit.non_oe_rows, 1)
        self.assertEqual(audit.complete_target_missing_rows, 1)


class PlayerPerformanceModelTests(unittest.TestCase):
    def test_target_is_result_and_kda_invariant(self) -> None:
        frame, _ = _synthetic_player_maps(30)
        prepared = prepare_player_map_matchups(frame, _config()).matchups
        scaler = RobustContextStandardizer.fit(prepared, _config())
        original = scaler.transform(prepared)["observed_performance"]

        changed = frame.copy()
        changed["result"] = 1 - changed["result"]
        changed["kills"] = np.arange(len(changed)) * 1000
        changed_prepared = prepare_player_map_matchups(
            changed, _config()
        ).matchups
        changed_target = scaler.transform(changed_prepared)[
            "observed_performance"
        ]
        np.testing.assert_allclose(original, changed_target)
        self.assertIn("Descriptive", ESTIMAND)

    def test_sparse_signed_context_model_recovers_player_order(self) -> None:
        frame, true_effects = _synthetic_player_maps(120)
        prepared = prepare_player_map_matchups(frame, _config()).matchups
        candidate = fit_player_performance_candidate(
            prepared,
            base_penalty=0.25,
            config=_config(),
            compute_uncertainty=True,
        )
        ratings = candidate.player_ratings(prepared, _config())
        estimated: list[float] = []
        truth: list[float] = []
        for row in ratings.itertuples(index=False):
            estimated.append(row.performance_mean)
            truth.append(true_effects[row.player_id])
            self.assertGreaterEqual(row.performance_sd, 0.0)
        correlation = pd.Series(estimated).corr(
            pd.Series(truth), method="spearman"
        )
        self.assertGreater(float(correlation), 0.80)

        role = "top"
        model = candidate.role_models[role]
        feature_index = model.feature_index()
        self.assertIn("intercept", feature_index)
        self.assertTrue(
            any(name.startswith("player:") for name in feature_index)
        )
        self.assertTrue(
            any(name.startswith("champion:") for name in feature_index)
        )
        self.assertTrue(any(name.startswith("team:") for name in feature_index))

    def test_role_models_are_independent_and_row_order_is_invariant(self) -> None:
        frame, _ = _synthetic_player_maps(50)
        prepared = prepare_player_map_matchups(frame, _config()).matchups
        original = fit_player_performance_candidate(
            prepared,
            base_penalty=1.0,
            config=_config(),
            compute_uncertainty=False,
        )
        shuffled = fit_player_performance_candidate(
            prepared.sample(frac=1.0, random_state=9).reset_index(drop=True),
            base_penalty=1.0,
            config=_config(),
            compute_uncertainty=False,
        )
        for role in CANONICAL_ROLES:
            left = original.role_models[role]
            right = shuffled.role_models[role]
            self.assertEqual(left.feature_names, right.feature_names)
            np.testing.assert_allclose(
                left.coefficients, right.coefficients, atol=1e-8
            )

        changed = prepared.copy()
        changed.loc[changed["role"].eq("sup"), "golddiffat15"] *= 100.0
        changed.loc[changed["role"].eq("sup"), "xpdiffat15"] *= 100.0
        changed.loc[changed["role"].eq("sup"), "csdiffat15"] *= 100.0
        changed_candidate = fit_player_performance_candidate(
            changed,
            base_penalty=1.0,
            config=_config(),
            compute_uncertainty=False,
        )
        np.testing.assert_allclose(
            original.role_models["top"].coefficients,
            changed_candidate.role_models["top"].coefficients,
            atol=1e-8,
        )


class PlayerPerformanceTournamentTests(unittest.TestCase):
    def test_chronological_ledger_and_required_slices(self) -> None:
        frame, _ = _synthetic_player_maps(90)
        result = run_player_performance_tournament(frame, _config())
        ledger = result.prediction_ledger
        self.assertEqual(set(ledger["split"]), {"train", "validation", "test"})
        self.assertTrue((ledger["target_uses_match_result"] == False).all())  # noqa: E712
        train_end = pd.Timestamp(result.split_boundaries["train_end"])
        validation_start = pd.Timestamp(
            result.split_boundaries["validation_start"]
        )
        validation_end = pd.Timestamp(
            result.split_boundaries["validation_end"]
        )
        test_start = pd.Timestamp(result.split_boundaries["test_start"])
        self.assertLess(train_end, validation_start)
        self.assertLess(validation_end, test_start)
        self.assertGreater(result.test_metrics.rows, 0)
        self.assertGreater(
            result.validation_context_baseline_metrics.rows, 0
        )
        self.assertGreater(result.test_context_baseline_metrics.rows, 0)
        self.assertGreater(result.player_incremental_test_rmse_lift, 0.0)
        self.assertGreater(
            result.player_incremental_test_contrast.ci_low, 0.0
        )
        self.assertEqual(
            result.player_incremental_test_contrast.resampling_unit,
            "calendar_day",
        )
        self.assertGreater(result.future_patch_test_metrics.rows, 0)
        self.assertGreater(result.roster_move_test_metrics.rows, 0)
        self.assertTrue(
            (
                ledger.loc[
                    ledger["split"].eq("test"), "model_fit_through"
                ]
                < ledger.loc[ledger["split"].eq("test"), "date"]
            ).all()
        )
        self.assertEqual(
            result.promotion_status, "research_candidate_not_production"
        )

    def test_test_targets_cannot_change_validation_selection(self) -> None:
        frame, _ = _synthetic_player_maps(90)
        config = _config()
        original = run_player_performance_tournament(frame, config)
        changed = frame.copy()
        test_start = pd.Timestamp(original.split_boundaries["test_start"])
        test_mask = pd.to_datetime(changed["date"], utc=True) >= test_start
        for metric in ("golddiffat15", "xpdiffat15", "csdiffat15"):
            changed.loc[test_mask, metric] *= 3.0
        modified = run_player_performance_tournament(changed, config)
        self.assertEqual(
            original.selected_base_penalty,
            modified.selected_base_penalty,
        )
        pd.testing.assert_frame_equal(
            original.validation_candidates,
            modified.validation_candidates,
        )
        original_before_test = original.prediction_ledger[
            ~original.prediction_ledger["split"].eq("test")
        ].reset_index(drop=True)
        modified_before_test = modified.prediction_ledger[
            ~modified.prediction_ledger["split"].eq("test")
        ].reset_index(drop=True)
        pd.testing.assert_frame_equal(original_before_test, modified_before_test)


if __name__ == "__main__":
    unittest.main()
