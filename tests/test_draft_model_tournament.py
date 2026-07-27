from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from lol_kills.composition_model import ROLES, CompositionGame
from lol_kills.draft_model_tournament import (
    BASELINE_LEAGUE,
    BASELINE_OVERALL,
    CandidateEvaluation,
    CandidateSpec,
    DraftTournamentError,
    TournamentConfig,
    chronological_date_splits,
    composition_logit,
    run_draft_model_tournament,
    select_simplest_within_tolerance,
    validate_identity_free_feature_names,
)


TEAM_A = tuple(
    zip(
        ROLES,
        ("Atlas", "Bolt", "Cipher", "Drift", "Echo"),
    )
)
TEAM_B = tuple(
    zip(
        ROLES,
        ("Fable", "Grove", "Halo", "Ion", "Jade"),
    )
)


def _synthetic_games() -> tuple[CompositionGame, ...]:
    """Balanced side/league rates with a stable champion-composition signal."""

    games: list[CompositionGame] = []
    start = pd.Timestamp("2025-01-01")
    for day_index in range(40):
        if day_index < 20:
            patch = "25.01"
        elif day_index < 34:
            patch = "25.02"
        else:
            # The final six date groups are an unseen future patch.
            patch = "25.03"
        for variant in range(2):
            blue_is_a = variant == 0
            outcome = int(blue_is_a)
            # A small deterministic disturbance avoids perfect separation.
            if day_index in {11, 27}:
                outcome = 1 - outcome
            games.append(
                CompositionGame(
                    game_id=f"synthetic:{day_index:02d}:{variant}",
                    blue=TEAM_A if blue_is_a else TEAM_B,
                    red=TEAM_B if blue_is_a else TEAM_A,
                    y=outcome,
                    league="LCK" if (day_index + variant) % 2 == 0 else "LPL",
                    patch=patch,
                    date=start + pd.Timedelta(days=day_index),
                )
            )
    return tuple(games)


def _config() -> TournamentConfig:
    return TournamentConfig(
        primary_score="log_loss",
        split_fractions=(0.5, 0.2, 0.15, 0.15),
        minimum_events_per_split=8,
        league_prior_n=8.0,
        tie_tolerance=1e-8,
        invariant_check_games=8,
        candidate_specs=(
            CandidateSpec("additive", min_support=2, prior_n=10.0),
            CandidateSpec("synergy", min_support=2, prior_n=10.0),
            CandidateSpec("opposition", min_support=2, prior_n=10.0),
            CandidateSpec(
                "low_rank",
                min_support=2,
                prior_n=10.0,
                low_rank_rank=1,
            ),
        ),
    )


@pytest.fixture(scope="module")
def tournament_result():
    games = _synthetic_games()
    return games, run_draft_model_tournament(games, config=_config())


def test_four_splits_are_chronological_and_date_group_safe():
    splits = chronological_date_splits(
        _synthetic_games(),
        fractions=(0.5, 0.2, 0.15, 0.15),
        minimum_events_per_split=8,
    )
    seen_dates: set[pd.Timestamp] = set()
    previous_max: pd.Timestamp | None = None
    for name, partition in splits.items():
        dates = {
            pd.Timestamp(game.date).tz_localize("UTC").normalize()
            for game in partition
        }
        assert dates
        assert not seen_dates.intersection(dates), name
        if previous_max is not None:
            assert previous_max < min(dates)
        previous_max = max(dates)
        seen_dates.update(dates)
        # Both games from every date stay together.
        assert len(partition) == 2 * len(dates)


def test_tournament_keeps_mandatory_models_and_final_evidence(
    tournament_result,
):
    _games, result = tournament_result
    validation_families = {row.family for row in result.validation_scores}
    assert {
        "additive",
        "synergy",
        "opposition",
        "low_rank",
        "baseline_overall",
        "baseline_side_league",
    } <= validation_families
    assert len(result.family_winner_ids) == 4
    assert {BASELINE_OVERALL, BASELINE_LEAGUE} <= {
        row.candidate_id for row in result.selection_scores
    }

    final_rows = [
        row for row in result.prediction_rows if row.phase == "final"
    ]
    assert {row.model_id for row in final_rows} == {
        result.best_composition_id,
        BASELINE_OVERALL,
        BASELINE_LEAGUE,
    }
    final_events = next(
        summary.events
        for summary in result.split_summaries
        if summary.name == "final"
    )
    assert len(final_rows) == 3 * final_events

    for diagnostics in result.final_diagnostics:
        assert diagnostics.events == final_events
        assert diagnostics.log_loss >= 0.0
        assert 0.0 <= diagnostics.brier <= 1.0
        assert 0.0 <= diagnostics.ece_10 <= 1.0
        assert abs(diagnostics.calibration_decomposition_residual) < 1e-10

    assert result.composition_lift.baseline_model_id == BASELINE_LEAGUE
    assert result.composition_lift.composition_model_id == (
        result.best_composition_id
    )
    assert result.composition_lift.log_loss_lift > 0.0
    assert result.composition_lift.brier_lift > 0.0


def test_future_patch_and_all_required_invariants_are_reported(
    tournament_result,
):
    _games, result = tournament_result
    assert result.future_patch.status == "available"
    assert result.future_patch.patches == ("25.03",)
    assert result.future_patch.events > 0
    assert result.future_patch.composition is not None
    assert result.future_patch.side_league_baseline is not None
    assert result.future_patch.composition_lift is not None

    invariants = result.invariants
    assert invariants.checked_games == 8
    assert invariants.exact_side_swap_antisymmetry
    assert invariants.role_preserving_permutation
    assert invariants.probability_bounds
    assert invariants.identity_free_features
    assert invariants.outcome_blind_features
    assert invariants.no_temporal_leakage
    assert all(
        0.0 <= row.probability <= 1.0
        for row in result.prediction_rows
    )


def test_event_rows_are_frozen_and_final_events_never_enter_training(
    tournament_result,
):
    _games, result = tournament_result
    row = result.prediction_rows[0]
    with pytest.raises(FrozenInstanceError):
        row.probability = 0.5

    final_ids = set(
        next(
            summary.event_ids
            for summary in result.split_summaries
            if summary.name == "final"
        )
    )
    assert final_ids
    assert all(
        final_ids.isdisjoint(manifest.training_event_ids)
        for manifest in result.fit_manifests
    )
    assert all(
        row.trained_through < row.event_date
        for row in result.prediction_rows
    )


def test_changing_final_labels_cannot_change_validation_or_selection(
    tournament_result,
):
    games, result = tournament_result
    final_ids = set(
        next(
            summary.event_ids
            for summary in result.split_summaries
            if summary.name == "final"
        )
    )
    changed_final = tuple(
        replace(game, y=1 - game.y)
        if game.game_id in final_ids
        else game
        for game in games
    )
    changed = run_draft_model_tournament(changed_final, config=_config())

    assert changed.family_winner_ids == result.family_winner_ids
    assert changed.validation_scores == result.validation_scores
    assert changed.selection_scores == result.selection_scores
    assert changed.winner_id == result.winner_id
    assert changed.best_composition_id == result.best_composition_id


def test_composition_score_is_exactly_antisymmetric_and_role_order_invariant():
    model = {
        "intercept": 0.2,
        "components": ["main", "synergy", "opposition"],
        "feature_specs": {
            "main|top|Atlas": {"coef": 0.3},
            "synergy|Atlas|Bolt": {"coef": 0.2},
            "opposition|Atlas|Fable": {"coef": -0.1},
        },
        "low_rank": {
            "rank": 1,
            "champions": ["Atlas", "Fable"],
            "left": [[1.0], [2.0]],
            "right": [[3.0], [5.0]],
        },
    }
    game = _synthetic_games()[0]
    swapped = CompositionGame(
        game_id="swapped",
        blue=game.red,
        red=game.blue,
        y=1 - game.y,
        league=game.league,
        patch=game.patch,
        date=game.date,
    )
    permuted = CompositionGame(
        game_id="permuted",
        blue=tuple((*game.blue[2:], *game.blue[:2])),
        red=tuple((*game.red[1:], game.red[0])),
        y=game.y,
        league=game.league,
        patch=game.patch,
        date=game.date,
    )
    score = composition_logit(model, game)
    assert score == -composition_logit(model, swapped)
    assert score == composition_logit(model, permuted)


def test_complexity_loses_a_proper_score_tie():
    common = {
        "family": "additive",
        "stage": "validation",
        "events": 20,
        "primary_score_name": "log_loss",
        "log_loss": 0.6,
        "brier": 0.2,
        "model_version": "v1",
    }
    complex_model = CandidateEvaluation(
        candidate_id="complex",
        primary_score=0.6000000,
        complexity=100,
        **common,
    )
    simple_model = CandidateEvaluation(
        candidate_id="simple",
        primary_score=0.6000005,
        complexity=10,
        **common,
    )
    selected = select_simplest_within_tolerance(
        (complex_model, simple_model),
        tie_tolerance=1e-6,
    )
    assert selected.candidate_id == "simple"


def test_team_player_or_roster_feature_names_are_rejected():
    validate_identity_free_feature_names(
        (
            "main|top|Atlas",
            "league|LCK|top|Atlas",
            "patch|25.03|top|Atlas",
            "synergy|Atlas|Bolt",
            "opposition|Atlas|Fable",
        )
    )
    for forbidden in (
        "team|T1",
        "player|Faker",
        "roster|state-1",
    ):
        with pytest.raises(
            DraftTournamentError,
            match="team/player/roster",
        ):
            validate_identity_free_feature_names((forbidden,))
