from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

import lol_kills.ratings.player_apm as player_apm_module
from lol_kills.ratings.player_apm import (
    CANONICAL_ROLES,
    LINEUP_PLAYER_COLUMNS,
    LineupValidationError,
    PlayerAPMConfig,
    build_design_matrix,
    chronological_player_apm_evaluation,
    detect_identical_exposure_cohorts,
    fit_player_apm_candidate,
    select_player_apm_candidate,
    validate_lineups,
)


def _map_row(
    game_number: int,
    blue: dict[str, str],
    red: dict[str, str],
    blue_win: int,
    *,
    date: str | pd.Timestamp | None = None,
    league: str = "LCK",
) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": f"g{game_number:04d}",
        "date": date or (
            pd.Timestamp("2025-01-01", tz="UTC")
            + pd.Timedelta(days=game_number)
        ),
        "blue_win": blue_win,
        "league": league,
    }
    for role in CANONICAL_ROLES:
        row[f"blue_{role}"] = blue[role]
        row[f"red_{role}"] = red[role]
    return row


def _fixed_lineups(n_maps: int = 24) -> pd.DataFrame:
    blue = {role: f"Blue-{role}" for role in CANONICAL_ROLES}
    red = {role: f"Red-{role}" for role in CANONICAL_ROLES}
    return pd.DataFrame(
        [
            _map_row(
                game_number,
                blue,
                red,
                blue_win=int(game_number % 3 != 0),
            )
            for game_number in range(n_maps)
        ]
    )


def _role_pools() -> dict[str, tuple[str, ...]]:
    return {
        role: tuple(f"{role}-p{index}" for index in range(4))
        for role in CANONICAL_ROLES
    }


def _true_player_effects() -> dict[str, float]:
    pools = _role_pools()
    base = (-1.8, -0.6, 0.6, 1.8)
    effects: dict[str, float] = {}
    for role_index, role in enumerate(CANONICAL_ROLES):
        rotated = base[role_index % len(base) :] + base[: role_index % len(base)]
        effects.update(dict(zip(pools[role], rotated)))
    return effects


def _varying_lineups(
    n_maps: int,
    *,
    seed: int = 17,
    start: str = "2025-01-01",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pools = _role_pools()
    effects = _true_player_effects()
    rows: list[dict[str, object]] = []
    start_date = pd.Timestamp(start, tz="UTC")
    for game_number in range(n_maps):
        blue: dict[str, str] = {}
        red: dict[str, str] = {}
        for role in CANONICAL_ROLES:
            selected = rng.choice(pools[role], size=2, replace=False)
            if rng.random() < 0.5:
                selected = selected[::-1]
            blue[role] = str(selected[0])
            red[role] = str(selected[1])
        logit = sum(effects[player] for player in blue.values()) / 5.0
        logit -= sum(effects[player] for player in red.values()) / 5.0
        blue_win = int(rng.random() < expit(logit))
        rows.append(
            _map_row(
                game_number,
                blue,
                red,
                blue_win,
                date=start_date + pd.Timedelta(days=game_number),
            )
        )
    return pd.DataFrame(rows)


class PlayerAPMValidationTests(unittest.TestCase):
    def test_exact_canonical_role_complete_validation(self) -> None:
        fixed = _fixed_lineups(1)
        validated = validate_lineups(fixed)
        self.assertEqual(len(validated), 1)
        self.assertEqual(
            set(validated.loc[0, list(LINEUP_PLAYER_COLUMNS)]),
            {
                *(f"Blue-{role}" for role in CANONICAL_ROLES),
                *(f"Red-{role}" for role in CANONICAL_ROLES),
            },
        )

        missing_role = fixed.drop(columns=["red_sup"])
        with self.assertRaisesRegex(LineupValidationError, "missing columns"):
            validate_lineups(missing_role)

        duplicate_player = fixed.copy()
        duplicate_player.loc[0, "blue_sup"] = duplicate_player.loc[0, "blue_top"]
        with self.assertRaisesRegex(LineupValidationError, "10 distinct players"):
            validate_lineups(duplicate_player)

    def test_current_long_interface_is_supported_but_role_aliases_are_rejected(
        self,
    ) -> None:
        wide = _fixed_lineups(1).iloc[0]
        rows: list[dict[str, object]] = []
        for side in ("Blue", "Red"):
            for role in CANONICAL_ROLES:
                rows.append(
                    {
                        "game_uid": wide["game_id"],
                        "gameid": "source-provider-id",
                        "date": wide["date"],
                        "league": wide["league"],
                        "side": side,
                        "position": role,
                        "playername": wide[f"{side.casefold()}_{role}"],
                        "result": (
                            wide["blue_win"]
                            if side == "Blue"
                            else 1 - wide["blue_win"]
                        ),
                    }
                )
        long = pd.DataFrame(rows)
        canonical = validate_lineups(long)
        self.assertEqual(canonical.loc[0, "game_id"], wide["game_id"])
        self.assertEqual(canonical.loc[0, "blue_jng"], "name:blue-jng")

        long.loc[long["position"].eq("jng"), "position"] = "jungle"
        with self.assertRaisesRegex(LineupValidationError, "exact canonical roles"):
            validate_lineups(long)

    def test_long_interface_uses_stable_player_ids_across_handle_changes(
        self,
    ) -> None:
        wide = _fixed_lineups(2)
        rows: list[dict[str, object]] = []
        for map_index, map_row in wide.iterrows():
            for side in ("Blue", "Red"):
                for role in CANONICAL_ROLES:
                    handle = str(map_row[f"{side.casefold()}_{role}"])
                    player_id = f"id:{side.casefold()}:{role}"
                    if map_index == 1 and side == "Blue" and role == "top":
                        handle = "Renamed top"
                    rows.append(
                        {
                            "game_uid": map_row["game_id"],
                            "date": map_row["date"],
                            "league": map_row["league"],
                            "side": side,
                            "position": role,
                            "playername": handle,
                            "playerid": player_id,
                            "result": (
                                map_row["blue_win"]
                                if side == "Blue"
                                else 1 - map_row["blue_win"]
                            ),
                        }
                    )

        canonical = validate_lineups(pd.DataFrame(rows))
        self.assertEqual(canonical.loc[0, "blue_top"], "id:blue:top")
        self.assertEqual(canonical.loc[1, "blue_top"], "id:blue:top")

    def test_unique_handle_fills_missing_id_but_ambiguous_handle_fails(
        self,
    ) -> None:
        wide = _fixed_lineups(2)
        rows: list[dict[str, object]] = []
        for map_index, map_row in wide.iterrows():
            for side in ("Blue", "Red"):
                for role in CANONICAL_ROLES:
                    handle = str(map_row[f"{side.casefold()}_{role}"])
                    player_id: str | None = f"id:{side.casefold()}:{role}"
                    if side == "Blue" and role == "top" and map_index == 1:
                        player_id = None
                    rows.append(
                        {
                            "game_uid": map_row["game_id"],
                            "date": map_row["date"],
                            "league": map_row["league"],
                            "side": side,
                            "position": role,
                            "playername": handle,
                            "playerid": player_id,
                            "result": (
                                map_row["blue_win"]
                                if side == "Blue"
                                else 1 - map_row["blue_win"]
                            ),
                        }
                    )

        unique = pd.DataFrame(rows)
        canonical = validate_lineups(unique)
        self.assertEqual(canonical.loc[1, "blue_top"], "id:blue:top")

        ambiguous = pd.concat(
            [
                unique,
                unique.iloc[[0]].assign(
                    game_uid="identity-only-extra",
                    playerid="id:second:blue:top",
                ),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(
            LineupValidationError,
            "same handle maps to multiple identities",
        ):
            validate_lineups(ambiguous)

    def test_design_uses_one_fifth_signed_players_and_predeclared_nuisances(
        self,
    ) -> None:
        frame = _fixed_lineups(1)
        frame.loc[0, "league"] = "LPL"
        players = tuple(
            str(frame.loc[0, column]) for column in LINEUP_PLAYER_COLUMNS
        )
        config = PlayerAPMConfig(
            include_side_term=True,
            league_levels=("LCK", "LPL"),
        )
        design = build_design_matrix(
            frame, player_order=players, config=config
        )
        self.assertTrue(sparse.isspmatrix_csr(design.values))
        np.testing.assert_allclose(
            design.values[0, :5].toarray(), 0.2
        )
        np.testing.assert_allclose(
            design.values[0, 5:10].toarray(), -0.2
        )
        self.assertEqual(design.feature_names[-2:], (
            "nuisance:blue_side",
            "nuisance:league:LPL",
        ))
        np.testing.assert_allclose(
            design.values[0, -2:].toarray(), ((1.0, 1.0),)
        )

        future = frame.copy()
        future.loc[0, "blue_top"] = "Future-top"
        future_design = build_design_matrix(
            future, player_order=players, config=config
        )
        self.assertEqual(
            future_design.unknown_players_by_map,
            (("Future-top",),),
        )
        self.assertEqual(future_design.values.nnz, design.values.nnz - 1)


class PlayerAPMIdentifiabilityTests(unittest.TestCase):
    def test_fixed_lineups_cannot_identify_individual_teammates(self) -> None:
        frame = _fixed_lineups()
        config = PlayerAPMConfig(
            include_side_term=False,
            player_l2_grid=(0.1,),
            nuisance_l2_grid=(1.0,),
        )
        model = fit_player_apm_candidate(
            frame,
            player_l2=0.1,
            nuisance_l2=1.0,
            config=config,
        )
        self.assertEqual(model.diagnostics.player_rank, 1)
        self.assertEqual(model.diagnostics.player_nullity, 9)
        self.assertTrue(math.isinf(model.diagnostics.player_condition_number))

        cohorts = detect_identical_exposure_cohorts(frame, config)
        self.assertEqual(sorted(len(cohort.players) for cohort in cohorts), [5, 5])
        blue_effects = [
            model.player_effect(f"Blue-{role}") for role in CANONICAL_ROLES
        ]
        np.testing.assert_allclose(blue_effects, blue_effects[0], atol=1e-10)

        requested = model.covariance_for_players(("Blue-top", "Blue-jng"))
        self.assertEqual(requested.covariance.shape, (2, 2))
        self.assertFalse(any(requested.level_data_identified))
        self.assertNotEqual(float(requested.covariance[0, 1]), 0.0)

        contrast = model.contrast({"Blue-top": 1.0, "Blue-jng": -1.0})
        expected_variance = np.array([1.0, -1.0]) @ requested.covariance @ np.array(
            [1.0, -1.0]
        )
        self.assertAlmostEqual(
            contrast.standard_error**2, float(expected_variance), places=10
        )
        self.assertFalse(contrast.data_identified)
        self.assertTrue(contrast.identical_exposure_confounded)
        self.assertAlmostEqual(contrast.estimate, 0.0, places=10)

    def test_lineup_swap_is_antisymmetric_without_side_nuisance_and_bounds_hold(
        self,
    ) -> None:
        frame = _varying_lineups(180)
        config = PlayerAPMConfig(
            include_side_term=False,
            player_l2_grid=(0.1,),
            nuisance_l2_grid=(1.0,),
        )
        model = fit_player_apm_candidate(
            frame,
            player_l2=0.1,
            nuisance_l2=1.0,
            config=config,
        )
        original = frame.iloc[[0]].copy()
        swapped = original.copy()
        swapped["game_id"] = "swapped"
        swapped["date"] = original["date"] + pd.Timedelta(seconds=1)
        swapped["blue_win"] = 1.0 - original["blue_win"].to_numpy()
        for role in CANONICAL_ROLES:
            swapped[f"blue_{role}"] = original[
                f"red_{role}"
            ].to_numpy()
            swapped[f"red_{role}"] = original[
                f"blue_{role}"
            ].to_numpy()

        original_design = build_design_matrix(
            original, player_order=model.player_order, config=config
        )
        swapped_design = build_design_matrix(
            swapped, player_order=model.player_order, config=config
        )
        np.testing.assert_allclose(
            original_design.values.toarray(),
            -swapped_design.values.toarray(),
            atol=0.0,
        )
        p_original = float(model.predict_proba(original)[0])
        p_swapped = float(model.predict_proba(swapped)[0])
        self.assertGreater(p_original, 0.0)
        self.assertLess(p_original, 1.0)
        self.assertGreater(p_swapped, 0.0)
        self.assertLess(p_swapped, 1.0)
        self.assertAlmostEqual(p_original + p_swapped, 1.0, places=12)

    def test_player_recovery_requires_lineup_variation(self) -> None:
        varied = _varying_lineups(900, seed=29)
        config = PlayerAPMConfig(
            include_side_term=False,
            player_l2_grid=(0.05,),
            nuisance_l2_grid=(1.0,),
        )
        varied_model = fit_player_apm_candidate(
            varied,
            player_l2=0.05,
            nuisance_l2=1.0,
            config=config,
        )
        truth = _true_player_effects()
        estimated = np.asarray(
            [varied_model.player_effect(player) for player in varied_model.player_order]
        )
        expected = np.asarray(
            [truth[player] for player in varied_model.player_order]
        )
        correlation = float(np.corrcoef(estimated, expected)[0, 1])
        self.assertGreater(correlation, 0.85)
        self.assertGreater(varied_model.diagnostics.player_rank, 10)

        same_role = _role_pools()["top"]
        low = min(same_role, key=truth.__getitem__)
        high = max(same_role, key=truth.__getitem__)
        contrast = varied_model.contrast({high: 1.0, low: -1.0})
        self.assertTrue(contrast.data_identified)
        self.assertGreater(contrast.estimate, 0.0)

        fixed = _fixed_lineups(120)
        fixed_model = fit_player_apm_candidate(
            fixed,
            player_l2=0.05,
            nuisance_l2=1.0,
            config=config,
        )
        fixed_blue = [
            fixed_model.player_effect(f"Blue-{role}")
            for role in CANONICAL_ROLES
        ]
        np.testing.assert_allclose(fixed_blue, fixed_blue[0], atol=1e-10)


class PlayerAPMScalabilityTests(unittest.TestCase):
    def test_grid_selection_skips_postfit_diagnostics_and_full_covariance(
        self,
    ) -> None:
        frame = _varying_lineups(100, seed=211)
        train = frame.iloc[:70].copy()
        validation = frame.iloc[70:].copy()
        config = PlayerAPMConfig(
            include_side_term=False,
            player_l2_grid=(0.03, 0.3, 3.0),
            nuisance_l2_grid=(1.0,),
        )
        with (
            mock.patch.object(
                player_apm_module,
                "_matrix_diagnostics",
                wraps=player_apm_module._matrix_diagnostics,
            ) as diagnostics_spy,
            mock.patch.object(
                np.linalg,
                "inv",
                side_effect=AssertionError(
                    "grid selection formed a dense inverse"
                ),
            ),
            mock.patch.object(
                np.linalg,
                "pinv",
                side_effect=AssertionError(
                    "grid selection formed a dense pseudoinverse"
                ),
            ),
        ):
            selection = select_player_apm_candidate(
                train, validation, config
            )

        # One selected-model diagnostic pass, not one pass per grid point.
        self.assertEqual(diagnostics_spy.call_count, 1)
        self.assertTrue(
            (
                ~selection.candidate_ledger[
                    "postfit_diagnostics_computed"
                ]
            ).all()
        )
        self.assertTrue(
            (
                ~selection.candidate_ledger["full_covariance_formed"]
            ).all()
        )
        self.assertEqual(
            set(selection.candidate_ledger["design_format"]),
            {"scipy_sparse_csr"},
        )
        self.assertTrue(selection.model.postfit_computed)
        self.assertFalse(selection.model.full_covariance_formed)
        with self.assertRaisesRegex(RuntimeError, "intentionally not formed"):
            _ = selection.model.covariance

    def test_requested_hessian_solve_matches_small_dense_reference(
        self,
    ) -> None:
        frame = _varying_lineups(160, seed=307)
        config = PlayerAPMConfig(
            include_side_term=False,
            player_l2_grid=(0.2,),
            nuisance_l2_grid=(1.0,),
        )
        model = fit_player_apm_candidate(
            frame,
            player_l2=0.2,
            nuisance_l2=1.0,
            config=config,
        )
        design = build_design_matrix(
            frame, player_order=model.player_order, config=config
        )
        dense = design.values.toarray()
        fitted = expit(
            np.einsum(
                "ij,j->i", dense, model.coefficients, optimize=True
            )
        )
        hessian = (
            np.einsum(
                "i,ij,ik->jk",
                fitted * (1.0 - fitted),
                dense,
                dense,
                optimize=True,
            )
            + np.diag(model.penalty)
        )
        dense_reference = np.linalg.inv(hessian)

        players = model.player_order[:3]
        indices = [model.player_order.index(player) for player in players]
        requested = model.covariance_for_players(players)
        expected = dense_reference[np.ix_(indices, indices)]
        np.testing.assert_allclose(
            requested.covariance, expected, rtol=1e-7, atol=1e-9
        )

        weights = {players[0]: 1.0, players[1]: -0.5, players[2]: -0.5}
        contrast = model.contrast(weights)
        weight_vector = np.asarray(tuple(weights.values()), dtype=float)
        expected_variance = float(
            weight_vector @ expected @ weight_vector
        )
        self.assertAlmostEqual(
            contrast.standard_error**2, expected_variance, places=8
        )
        self.assertEqual(contrast.covariance.shape, (3, 3))
        self.assertIn("hessian_solves", contrast.uncertainty_kind)
        self.assertFalse(model.full_covariance_formed)

    def test_large_design_diagnostics_are_explicitly_approximate(self) -> None:
        frame = _varying_lineups(180, seed=401)
        config = PlayerAPMConfig(
            include_side_term=False,
            player_l2_grid=(0.1,),
            nuisance_l2_grid=(1.0,),
            diagnostic_exact_max_cells=10,
            diagnostic_exact_max_columns=5,
            diagnostic_svd_components=6,
        )
        model = fit_player_apm_candidate(
            frame,
            player_l2=0.1,
            nuisance_l2=1.0,
            config=config,
        )
        self.assertIsNotNone(model.diagnostics)
        diagnostics = model.diagnostics
        assert diagnostics is not None
        self.assertTrue(diagnostics.diagnostics_approximate)
        self.assertIn("truncated_svd", diagnostics.diagnostic_method)
        self.assertIn("lower bound", diagnostics.rank_interpretation)
        self.assertIn(
            "not estimated", diagnostics.condition_interpretation
        )
        self.assertTrue(math.isnan(diagnostics.condition_number))


class PlayerAPMChronologyTests(unittest.TestCase):
    def test_test_outcomes_cannot_change_selection_fit_or_predictions(self) -> None:
        frame = _varying_lineups(120, seed=101)
        config = PlayerAPMConfig(
            include_side_term=False,
            player_l2_grid=(0.03, 0.3, 3.0),
            nuisance_l2_grid=(1.0,),
            selection_metric="log_loss",
        )
        train_end = frame.loc[59, "date"]
        validation_end = frame.loc[89, "date"]
        first = chronological_player_apm_evaluation(
            frame,
            train_end=train_end,
            validation_end=validation_end,
            config=config,
        )

        changed = frame.copy()
        test_mask = pd.to_datetime(changed["date"], utc=True) > validation_end
        changed.loc[test_mask, "blue_win"] = (
            1 - changed.loc[test_mask, "blue_win"].astype(int)
        )
        second = chronological_player_apm_evaluation(
            changed,
            train_end=train_end,
            validation_end=validation_end,
            config=config,
        )

        self.assertEqual(first.selection.player_l2, second.selection.player_l2)
        self.assertEqual(
            first.selection.nuisance_l2, second.selection.nuisance_l2
        )
        pd.testing.assert_frame_equal(
            first.selection.candidate_ledger,
            second.selection.candidate_ledger,
        )
        np.testing.assert_allclose(
            first.selection.model.coefficients,
            second.selection.model.coefficients,
            atol=0.0,
        )
        np.testing.assert_allclose(
            first.test_model.coefficients,
            second.test_model.coefficients,
            atol=0.0,
        )
        np.testing.assert_allclose(
            first.ledger["predicted_blue_win"],
            second.ledger["predicted_blue_win"],
            atol=0.0,
        )
        self.assertEqual(
            set(first.ledger["split"]), {"train", "validation", "test"}
        )
        self.assertEqual(
            set(first.metrics), {"train", "validation", "test"}
        )
        self.assertIn("brier", first.metrics["test"])
        self.assertIn("log_loss", first.metrics["test"])
        self.assertLessEqual(
            first.test_model.fitted_through, pd.Timestamp(validation_end)
        )
        test_ids = set(
            first.ledger.loc[first.ledger["split"].eq("test"), "game_id"]
        )
        self.assertTrue(
            test_ids.isdisjoint(first.selection.train_game_ids)
        )
        self.assertTrue(
            test_ids.isdisjoint(first.selection.validation_game_ids)
        )


if __name__ == "__main__":
    unittest.main()
