from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.special import expit

import lol_kills.draft_residual_tournament as residual_tournament
from lol_kills.draft_residual_tournament import (
    LEAGUE_SIDE_BASELINE,
    OFFSET_ONLY_BASELINE,
    ROLES,
    CandidateSpec,
    DraftResidualTournamentError,
    PreEventTeamLogit,
    PreparedDraftMap,
    TournamentConfig,
    _calibration_objective,
    _penalized_offset_objective,
    chronological_date_splits,
    paired_circular_block_bootstrap,
    raw_composition_logit,
    run_draft_residual_tournament,
    score_neutral_composition,
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


def _synthetic_inputs() -> tuple[
    tuple[PreparedDraftMap, ...],
    dict[str, PreEventTeamLogit],
]:
    """Deterministic maps with team and residual composition signal."""

    rows: list[PreparedDraftMap] = []
    offsets: dict[str, PreEventTeamLogit] = {}
    start = pd.Timestamp("2025-01-01T12:00:00Z")
    for day_index in range(48):
        if day_index < 18:
            patch = "25.01"
        elif day_index < 41:
            patch = "25.02"
        else:
            patch = "25.03"
        for game_index in range(2):
            event_id = f"map:{day_index:02d}:{game_index}"
            event_time = start + pd.Timedelta(
                days=day_index,
                hours=game_index,
            )
            blue_is_a = (day_index + game_index) % 2 == 0
            blue = TEAM_A if blue_is_a else TEAM_B
            red = TEAM_B if blue_is_a else TEAM_A
            if day_index == 0 and game_index == 0:
                blue = tuple(
                    (role, "Lumen" if role == "top" else champion)
                    for role, champion in blue
                )
            if day_index == 47 and game_index == 1:
                blue = tuple(
                    (role, "Kestrel" if role == "top" else champion)
                    for role, champion in blue
                )

            team_logit = 0.95 if (2 * day_index + game_index) % 4 in (0, 1) else -0.95
            composition_logit = 1.10 if blue_is_a else -1.10
            league_side_logit = 0.12 if day_index % 2 == 0 else -0.08
            probability = float(
                expit(team_logit + composition_logit + league_side_logit)
            )
            pseudo_uniform = (((day_index * 2 + game_index) * 37 + 11) % 101) / 101.0
            outcome = int(pseudo_uniform < probability)
            row = PreparedDraftMap(
                event_id=event_id,
                dependence_id=f"series:{day_index:02d}",
                event_time=event_time,
                league="LCK" if day_index % 2 == 0 else "LPL",
                patch=patch,
                blue=blue,
                red=red,
                y_blue_win=outcome,
            )
            rows.append(row)
            offsets[event_id] = PreEventTeamLogit(
                event_id=event_id,
                logit=team_logit,
                as_of=event_time - pd.Timedelta(hours=1),
                model_version=f"dynamic-team-v{day_index // 8}",
                provenance="synthetic strictly rolling team model",
            )
    return tuple(rows), offsets


def _config() -> TournamentConfig:
    return TournamentConfig(
        primary_score="log_loss",
        split_fractions=(0.50, 0.20, 0.15, 0.15),
        minimum_events_per_split=8,
        tie_tolerance=1e-8,
        league_side_l2=8.0,
        invariant_check_events=8,
        candidate_specs=(
            CandidateSpec(
                "additive",
                l2=5.0,
                min_support=2,
                include_patch_main=True,
            ),
            CandidateSpec(
                "additive",
                l2=25.0,
                min_support=2,
                include_patch_main=True,
            ),
            CandidateSpec(
                "synergy",
                l2=8.0,
                min_support=2,
                include_patch_main=True,
            ),
            CandidateSpec(
                "opposition",
                l2=12.0,
                min_support=2,
                include_patch_main=True,
            ),
        ),
    )


@pytest.fixture(scope="module")
def tournament_result():
    rows, offsets = _synthetic_inputs()
    result = run_draft_residual_tournament(
        rows,
        offsets,
        config=_config(),
    )
    return rows, offsets, result


def test_four_splits_are_strictly_chronological_and_date_safe():
    rows, _offsets = _synthetic_inputs()
    splits = chronological_date_splits(
        rows,
        fractions=(0.50, 0.20, 0.15, 0.15),
        minimum_events_per_split=8,
    )
    seen_days: set[pd.Timestamp] = set()
    previous_max: pd.Timestamp | None = None
    for name, partition in splits.items():
        days = {
            pd.Timestamp(row.event_time).tz_convert("UTC").normalize()
            for row in partition
        }
        assert days
        assert seen_days.isdisjoint(days), name
        if previous_max is not None:
            assert previous_max < min(days)
        previous_max = max(days)
        seen_days.update(days)
        assert len(partition) == 2 * len(days)


def test_offset_logistic_analytic_gradient_matches_finite_difference():
    design = sparse.csr_matrix(
        np.asarray(
            [
                [1.0, -0.5, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.75, -1.0],
                [1.0, -1.25, 0.5],
            ]
        )
    )
    outcome = np.asarray([1.0, 0.0, 1.0, 0.0])
    offset = np.asarray([0.4, -0.3, 0.2, -0.1])
    penalty = np.asarray([0.0, 2.0, 3.0])
    coefficients = np.asarray([0.15, -0.25, 0.35])
    _loss, analytic = _penalized_offset_objective(
        coefficients,
        design,
        outcome,
        offset,
        penalty,
    )
    epsilon = 1e-6
    finite_difference = np.empty_like(coefficients)
    for index in range(len(coefficients)):
        direction = np.zeros_like(coefficients)
        direction[index] = epsilon
        upper, _ = _penalized_offset_objective(
            coefficients + direction,
            design,
            outcome,
            offset,
            penalty,
        )
        lower, _ = _penalized_offset_objective(
            coefficients - direction,
            design,
            outcome,
            offset,
            penalty,
        )
        finite_difference[index] = (upper - lower) / (2.0 * epsilon)
    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=1e-7,
        atol=1e-8,
    )


def test_convex_polish_enforces_gradient_acceptance_bound(monkeypatch):
    feature = np.linspace(-2.0, 2.0, 400)
    design = sparse.csr_matrix(
        np.column_stack(
            (
                np.ones(len(feature)),
                feature,
                np.square(feature),
            )
        )
    )
    outcome = (
        feature + 0.35 * np.sin(np.arange(len(feature))) > 0.0
    ).astype(float)
    offset = np.zeros(len(feature), dtype=float)
    penalty = np.ones(design.shape[1], dtype=float)
    config = TournamentConfig(
        optimizer_ftol=0.5,
        optimizer_gtol=1e-12,
        maximum_gradient_inf_norm=1e-6,
    )
    real_minimize = residual_tournament.minimize
    methods: list[str] = []

    def tracked_minimize(*args, **kwargs):
        methods.append(str(kwargs.get("method")))
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(
        residual_tournament,
        "minimize",
        tracked_minimize,
    )
    coefficients, iterations, gradient = (
        residual_tournament._fit_penalized_offset_logistic(
            design,
            outcome,
            offset,
            penalty,
            config=config,
        )
    )

    assert methods == ["L-BFGS-B", "Newton-CG"]
    assert np.isfinite(coefficients).all()
    assert iterations > 1
    assert gradient <= config.maximum_gradient_inf_norm


def test_bounded_platt_gradient_matches_finite_difference():
    raw_probability = np.asarray((0.12, 0.31, 0.54, 0.77, 0.91))
    raw_logit = np.log(raw_probability) - np.log1p(
        -raw_probability
    )
    design = np.column_stack(
        (np.ones(len(raw_probability)), raw_logit)
    )
    outcome = np.asarray((0.0, 1.0, 0.0, 1.0, 1.0))
    coefficients = np.asarray((0.17, 0.83))
    kwargs = {
        "l2_identity_centered": 0.25,
        "probability_floor": 1e-4,
    }
    _loss, analytic = _calibration_objective(
        coefficients,
        design,
        outcome,
        **kwargs,
    )
    epsilon = 1e-6
    numeric = np.empty_like(coefficients)
    for index in range(len(coefficients)):
        direction = np.zeros_like(coefficients)
        direction[index] = epsilon
        upper, _ = _calibration_objective(
            coefficients + direction,
            design,
            outcome,
            **kwargs,
        )
        lower, _ = _calibration_objective(
            coefficients - direction,
            design,
            outcome,
            **kwargs,
        )
        numeric[index] = (upper - lower) / (2.0 * epsilon)
    np.testing.assert_allclose(
        analytic,
        numeric,
        rtol=1e-7,
        atol=1e-8,
    )


def test_split_boundaries_never_divide_a_dependence_cluster():
    rows, _offsets = _synthetic_inputs()
    cross_date = tuple(
        replace(row, dependence_id="cross-date-series")
        if row.event_id.startswith(("map:23:", "map:24:"))
        else row
        for row in rows
    )
    splits = chronological_date_splits(
        cross_date,
        fractions=(0.50, 0.20, 0.15, 0.15),
        minimum_events_per_split=8,
    )
    containing_splits = {
        name
        for name, partition in splits.items()
        if any(row.dependence_id == "cross-date-series" for row in partition)
    }
    assert len(containing_splits) == 1


def test_predeclared_end_dates_override_fractional_boundaries():
    rows, _offsets = _synthetic_inputs()
    splits = chronological_date_splits(
        rows,
        fractions=(0.70, 0.10, 0.10, 0.10),
        end_dates=("2025-01-12", "2025-01-24", "2025-02-05"),
        minimum_events_per_split=8,
    )
    assert tuple(len(partition) for _name, partition in splits.items()) == (
        24,
        24,
        24,
        24,
    )
    assert max(row.event_time for row in splits.train).normalize() == pd.Timestamp(
        "2025-01-12T00:00:00Z"
    )
    assert min(row.event_time for row in splits.final).normalize() == pd.Timestamp(
        "2025-02-06T00:00:00Z"
    )


def test_tournament_preserves_all_three_untouched_gates(
    tournament_result,
):
    rows, offsets, result = tournament_result
    validation_families = {score.family for score in result.validation_scores}
    assert {
        "additive",
        "synergy",
        "opposition",
        "baseline_offset_only",
        "baseline_league_side",
    } <= validation_families
    assert len(result.family_winner_ids) == 3
    assert {
        OFFSET_ONLY_BASELINE,
        LEAGUE_SIDE_BASELINE,
    } <= {score.candidate_id for score in result.selection_scores}

    final_ids = {
        event_id
        for summary in result.split_summaries
        if summary.name == "final"
        for event_id in summary.event_ids
    }
    changed_rows = tuple(
        replace(row, y_blue_win=1 - row.y_blue_win)
        if row.event_id in final_ids
        else row
        for row in rows
    )
    changed = run_draft_residual_tournament(
        changed_rows,
        offsets,
        config=_config(),
    )
    assert changed.validation_scores == result.validation_scores
    assert changed.family_winner_ids == result.family_winner_ids
    assert changed.selection_scores == result.selection_scores
    assert changed.selected_candidate_id == result.selected_candidate_id
    assert changed.final_model == result.final_model
    assert (
        changed.calibration_transfer.selection_scores
        == result.calibration_transfer.selection_scores
    )
    assert (
        changed.calibration_transfer.frozen_model
        == result.calibration_transfer.frozen_model
    )
    assert tuple(
        row.raw_probability
        for row in changed.calibration_transfer.final_predictions
    ) == tuple(
        row.raw_probability
        for row in result.calibration_transfer.final_predictions
    )
    assert tuple(
        row.calibrated_probability
        for row in changed.calibration_transfer.final_predictions
    ) == tuple(
        row.calibrated_probability
        for row in result.calibration_transfer.final_predictions
    )


def test_selection_labels_cannot_change_validation_gate(
    tournament_result,
):
    rows, offsets, result = tournament_result
    selection_ids = {
        event_id
        for summary in result.split_summaries
        if summary.name == "selection"
        for event_id in summary.event_ids
    }
    changed_rows = tuple(
        replace(row, y_blue_win=1 - row.y_blue_win)
        if row.event_id in selection_ids
        else row
        for row in rows
    )
    changed = run_draft_residual_tournament(
        changed_rows,
        offsets,
        config=_config(),
    )
    assert changed.validation_scores == result.validation_scores
    assert changed.family_winner_ids == result.family_winner_ids


def test_offset_is_exact_and_candidate_decomposition_reconciles(
    tournament_result,
):
    _rows, offsets, result = tournament_result
    final_predictions = [row for row in result.prediction_rows if row.phase == "final"]
    offset_rows = [
        row for row in final_predictions if row.model_id == OFFSET_ONLY_BASELINE
    ]
    assert offset_rows
    for row in offset_rows:
        assert row.contextual_logit == offsets[row.event_id].logit
        assert row.probability == float(expit(offsets[row.event_id].logit))

    candidate_rows = [
        row for row in final_predictions if row.model_id == result.selected_candidate_id
    ]
    for row in candidate_rows:
        assert row.contextual_logit == pytest.approx(
            row.team_offset_logit + row.league_side_logit + row.raw_composition_logit,
            abs=1e-15,
        )
        assert row.probability == pytest.approx(
            float(expit(row.contextual_logit)),
            abs=1e-15,
        )


def test_raw_composition_is_exactly_antisymmetric_and_context_free(
    tournament_result,
):
    rows, _offsets, result = tournament_result
    for row in rows:
        swapped = replace(
            row,
            event_id=f"{row.event_id}:swapped",
            blue=row.red,
            red=row.blue,
            y_blue_win=1 - row.y_blue_win,
        )
        permuted = replace(
            row,
            event_id=f"{row.event_id}:permuted",
            blue=tuple(reversed(row.blue)),
            red=(*row.red[2:], *row.red[:2]),
        )
        score = raw_composition_logit(result.final_model, row)
        swapped_score = raw_composition_logit(result.final_model, swapped)
        assert score == -swapped_score
        assert score == raw_composition_logit(result.final_model, permuted)

    row = rows[-1]
    score = raw_composition_logit(result.final_model, row)
    estimate = score_neutral_composition(result.final_model, row)
    assert estimate.raw_composition_logit == score
    assert estimate.neutral_team_probability == pytest.approx(
        float(expit(score)),
        abs=1e-15,
    )
    assert not hasattr(estimate, "team_offset_logit")


def test_unknown_patch_and_champion_fallbacks_are_explicit(
    tournament_result,
):
    rows, _offsets, result = tournament_result
    final_candidate_rows = [
        row
        for row in result.prediction_rows
        if row.phase == "final" and row.model_id == result.selected_candidate_id
    ]
    assert final_candidate_rows
    assert {row.patch_status for row in final_candidate_rows} == {
        "unseen_global_fallback"
    }
    novel = next(row for row in final_candidate_rows if row.event_id == "map:47:1")
    assert novel.unknown_champion_roles == ("blue:top:Kestrel",)

    explicit_unknown = replace(rows[-1], patch="UNKNOWN")
    estimate = score_neutral_composition(
        result.final_model,
        explicit_unknown,
    )
    assert estimate.patch_status == "explicit_unknown_global_fallback"

    unsupported = score_neutral_composition(result.final_model, rows[0])
    assert unsupported.unknown_champion_roles == ()
    assert unsupported.unsupported_champion_roles == ("blue:top:Lumen",)


def test_metrics_and_paired_dependence_ledger_are_event_aligned(
    tournament_result,
):
    _rows, _offsets, result = tournament_result
    diagnostic_ids = {diagnostics.model_id for diagnostics in result.final_diagnostics}
    assert diagnostic_ids == {
        result.selected_candidate_id,
        OFFSET_ONLY_BASELINE,
        LEAGUE_SIDE_BASELINE,
    }
    for diagnostics in result.final_diagnostics:
        assert diagnostics.log_loss >= 0.0
        assert 0.0 <= diagnostics.brier <= 1.0
        assert 0.0 <= diagnostics.ece <= 1.0

    final_events = next(
        summary.events for summary in result.split_summaries if summary.name == "final"
    )
    ledger = result.final_paired_ledger
    assert len(ledger) == final_events
    assert len({row.event_id for row in ledger}) == final_events
    assert len({row.dependence_id for row in ledger}) < final_events
    assert {
        comparison.baseline_model_id for comparison in result.final_comparisons
    } == {OFFSET_ONLY_BASELINE, LEAGUE_SIDE_BASELINE}
    for comparison in result.final_comparisons:
        assert comparison.events == final_events
        assert comparison.dependence_clusters == len(
            {row.dependence_id for row in ledger}
        )
        assert math.isfinite(comparison.log_loss_delta)
        assert math.isfinite(comparison.brier_delta)
        assert math.isfinite(comparison.ece_delta)


def test_paired_circular_block_bootstrap_matches_independent_recomputation(
    tournament_result,
):
    _rows, _offsets, result = tournament_result
    ledger = result.final_paired_ledger
    inference = paired_circular_block_bootstrap(
        tuple(reversed(ledger)),
        replicates=257,
        block_size=3,
        random_seed=17,
        alpha=0.10,
    )
    repeated = paired_circular_block_bootstrap(
        ledger,
        replicates=257,
        block_size=3,
        random_seed=17,
        alpha=0.10,
    )
    assert inference == repeated

    target = next(
        row
        for row in inference
        if row.baseline_model_id == OFFSET_ONLY_BASELINE
        and row.metric == "log_loss"
    )
    grouped: dict[str, list[float]] = {}
    earliest: dict[str, pd.Timestamp] = {}
    for row in ledger:
        grouped.setdefault(row.dependence_id, []).append(
            row.candidate_minus_offset_log_loss
        )
        earliest[row.dependence_id] = min(
            earliest.get(row.dependence_id, row.event_time),
            row.event_time,
        )
    cluster_ids = sorted(
        grouped,
        key=lambda cluster_id: (
            earliest[cluster_id],
            cluster_id,
        ),
    )
    cluster_counts = np.asarray(
        [len(grouped[cluster_id]) for cluster_id in cluster_ids],
        dtype=int,
    )
    cluster_sums = np.asarray(
        [
            math.fsum(grouped[cluster_id])
            for cluster_id in cluster_ids
        ],
        dtype=float,
    )
    effective_block_size = min(3, len(cluster_ids))
    blocks_needed = (
        len(cluster_ids) + effective_block_size - 1
    ) // effective_block_size
    offsets = np.arange(effective_block_size)
    rng = np.random.default_rng(17)
    manual = np.empty(257, dtype=float)
    for replicate in range(257):
        starts = rng.integers(
            0,
            len(cluster_ids),
            size=blocks_needed,
        )
        sampled = (
            (starts[:, None] + offsets[None, :]) % len(cluster_ids)
        ).ravel()[: len(cluster_ids)]
        manual[replicate] = (
            cluster_sums[sampled].sum()
            / cluster_counts[sampled].sum()
        )
    lower, upper = np.quantile(
        manual,
        (0.05, 0.95),
        method="linear",
    )
    assert target.point_delta == pytest.approx(
        math.fsum(
            row.candidate_minus_offset_log_loss for row in ledger
        )
        / len(ledger),
        abs=1e-15,
    )
    assert target.bootstrap_mean_delta == pytest.approx(
        manual.mean(),
        abs=1e-15,
    )
    assert target.bootstrap_standard_error == pytest.approx(
        manual.std(ddof=1),
        abs=1e-15,
    )
    assert target.interval_lower == pytest.approx(lower, abs=1e-15)
    assert target.interval_upper == pytest.approx(upper, abs=1e-15)


def test_default_bootstrap_and_calibration_transfer_are_auditable(
    tournament_result,
):
    _rows, _offsets, result = tournament_result
    inference = result.final_bootstrap_inference
    assert {
        (row.baseline_model_id, row.metric) for row in inference
    } == {
        (OFFSET_ONLY_BASELINE, "log_loss"),
        (OFFSET_ONLY_BASELINE, "brier"),
        (LEAGUE_SIDE_BASELINE, "log_loss"),
        (LEAGUE_SIDE_BASELINE, "brier"),
    }
    for row in inference:
        assert row.events == len(result.final_paired_ledger)
        assert row.dependence_clusters == len(
            {
                ledger_row.dependence_id
                for ledger_row in result.final_paired_ledger
            }
        )
        assert row.replicates == _config().bootstrap_replicates
        assert row.requested_block_size == _config().bootstrap_block_size
        assert row.effective_block_size == min(
            _config().bootstrap_block_size,
            row.dependence_clusters,
        )
        assert row.interval_lower <= row.interval_upper
        assert len(row.bootstrap_distribution_sha256) == 64
        assert (
            row.dependence_status
            == "caller_supplied_not_verified_by_module"
        )

    transfer = result.calibration_transfer
    assert transfer.candidate_model_id == result.selected_candidate_id
    assert {row.method for row in transfer.selection_scores} == {
        "identity",
        "platt",
    }
    assert all(
        row.fit_split == "validation_oos"
        and row.score_split == "selection"
        for row in transfer.selection_scores
    )
    assert transfer.selected_method in {"identity", "platt"}
    assert transfer.frozen_model.fit_split == (
        "validation+selection_oos"
    )
    assert transfer.frozen_model.selected_on_split == "selection"
    assert transfer.frozen_model.fit_events == sum(
        summary.events
        for summary in result.split_summaries
        if summary.name in {"validation", "selection"}
    )
    final_start = next(
        summary.date_min
        for summary in result.split_summaries
        if summary.name == "final"
    )
    assert transfer.frozen_model.fit_through < final_start
    assert len(transfer.final_predictions) == next(
        summary.events
        for summary in result.split_summaries
        if summary.name == "final"
    )
    assert not transfer.final_labels_used_for_selection
    assert transfer.selected_before_final
    raw = np.asarray(
        [row.raw_probability for row in transfer.final_predictions]
    )
    calibrated = np.asarray(
        [
            row.calibrated_probability
            for row in transfer.final_predictions
        ]
    )
    np.testing.assert_allclose(
        calibrated,
        transfer.frozen_model.apply(raw),
        atol=0.0,
        rtol=0.0,
    )
    blue = transfer.frozen_model.apply(
        np.asarray((0.2, 0.4, 0.7)),
        side_indicator=1.0,
    )
    swapped = transfer.frozen_model.apply(
        np.asarray((0.8, 0.6, 0.3)),
        side_indicator=-1.0,
    )
    np.testing.assert_allclose(
        blue,
        1.0 - swapped,
        atol=1e-15,
        rtol=0.0,
    )
    assert result.invariants.paired_dependence_blocks_preserved
    assert result.invariants.calibration_frozen_before_final


def test_injected_residual_signal_is_not_silently_ignored(
    tournament_result,
):
    _rows, _offsets, result = tournament_result
    candidate = result.diagnostics_for(result.selected_candidate_id)
    offset = result.diagnostics_for(OFFSET_ONLY_BASELINE)
    league_side = result.diagnostics_for(LEAGUE_SIDE_BASELINE)
    assert candidate.log_loss < offset.log_loss
    assert candidate.log_loss < league_side.log_loss
    assert candidate.brier < offset.brier
    assert candidate.brier < league_side.brier


def test_no_future_leakage_and_sparse_deterministic_fit(
    tournament_result,
):
    rows, offsets, result = tournament_result
    assert result.config == _config()
    assert result.invariants.no_future_leakage
    assert result.invariants.sparse_design
    assert result.invariants.exact_raw_side_swap_antisymmetry
    assert result.invariants.raw_estimand_excludes_team_offset
    assert result.invariants.offset_baseline_exact
    assert (
        result.final_model.optimizer_gradient_inf_norm
        <= _config().maximum_gradient_inf_norm
    )

    manifests = {manifest.model_version: manifest for manifest in result.fit_manifests}
    for prediction in result.prediction_rows:
        assert prediction.team_offset_as_of < prediction.event_time
        assert prediction.team_offset_provenance
        if prediction.model_id == OFFSET_ONLY_BASELINE:
            continue
        manifest = manifests[prediction.model_version]
        assert len(manifest.offset_digest) == 64
        assert prediction.event_id not in manifest.training_event_ids
        assert manifest.trained_through < prediction.event_time

    repeated = run_draft_residual_tournament(
        rows,
        offsets,
        config=_config(),
    )
    assert repeated == result


def test_illegal_duplicate_champions_and_role_states_are_rejected():
    rows, _offsets = _synthetic_inputs()
    row = rows[0]
    repeated_across_sides = replace(
        row,
        red=tuple(
            (
                role,
                row.blue[0][1] if role == "top" else champion,
            )
            for role, champion in row.red
        ),
    )
    with pytest.raises(
        DraftResidualTournamentError,
        match="duplicates a champion",
    ):
        chronological_date_splits(
            (repeated_across_sides, *rows[1:]),
            minimum_events_per_split=1,
        )

    duplicate_role = replace(
        row,
        blue=(
            ("top", "Atlas"),
            ("top", "Bolt"),
            *row.blue[2:],
        ),
    )
    with pytest.raises(
        DraftResidualTournamentError,
        match="roles must be complete and unique",
    ):
        chronological_date_splits(
            (duplicate_role, *rows[1:]),
            minimum_events_per_split=1,
        )


def test_explicit_blind_pick_allows_only_cross_side_mirrors(
    tournament_result,
):
    rows, _offsets = _synthetic_inputs()
    _fixture_rows, _fixture_offsets, result = tournament_result
    row = rows[0]
    mirrored = replace(
        row,
        draft_mode="blind_pick",
        red=tuple(
            (
                role,
                row.blue[0][1] if role == "top" else champion,
            )
            for role, champion in row.red
        ),
    )
    chronological_date_splits(
        (mirrored, *rows[1:]),
        minimum_events_per_split=1,
    )
    swapped = replace(
        mirrored,
        event_id=f"{mirrored.event_id}:swapped",
        blue=mirrored.red,
        red=mirrored.blue,
        y_blue_win=1 - mirrored.y_blue_win,
    )
    score = raw_composition_logit(result.final_model, mirrored)
    swapped_score = raw_composition_logit(
        result.final_model,
        swapped,
    )
    assert score == -swapped_score

    same_side_duplicate = replace(
        mirrored,
        blue=(
            ("top", "Atlas"),
            ("jng", "Atlas"),
            *mirrored.blue[2:],
        ),
    )
    with pytest.raises(
        DraftResidualTournamentError,
        match="repeats a champion",
    ):
        chronological_date_splits(
            (same_side_duplicate, *rows[1:]),
            minimum_events_per_split=1,
        )


def test_offset_provenance_must_be_exact_and_strictly_pre_event():
    rows, offsets = _synthetic_inputs()
    missing = dict(offsets)
    missing.pop(rows[0].event_id)
    with pytest.raises(
        DraftResidualTournamentError,
        match="match prepared events exactly",
    ):
        run_draft_residual_tournament(
            rows,
            missing,
            config=_config(),
        )

    late = dict(offsets)
    event_id = rows[0].event_id
    late[event_id] = replace(
        late[event_id],
        as_of=rows[0].event_time,
    )
    with pytest.raises(
        DraftResidualTournamentError,
        match="not strictly pre-event",
    ):
        run_draft_residual_tournament(
            rows,
            late,
            config=_config(),
        )

    contaminated = dict(offsets)
    contaminated[event_id] = replace(
        contaminated[event_id],
        includes_draft=True,
    )
    with pytest.raises(
        DraftResidualTournamentError,
        match="excludes side and draft features",
    ):
        run_draft_residual_tournament(
            rows,
            contaminated,
            config=_config(),
        )

    wrong_orientation = dict(offsets)
    wrong_orientation[event_id] = replace(
        wrong_orientation[event_id],
        orientation="red_minus_blue",
    )
    with pytest.raises(
        DraftResidualTournamentError,
        match="blue-minus-red natural log-odds",
    ):
        run_draft_residual_tournament(
            rows,
            wrong_orientation,
            config=_config(),
        )


def test_result_and_ledger_are_immutable(tournament_result):
    _rows, _offsets, result = tournament_result
    with pytest.raises(FrozenInstanceError):
        result.selected_candidate_id = "changed"
    with pytest.raises(FrozenInstanceError):
        result.final_paired_ledger[0].candidate_probability = 0.5
    with pytest.raises(FrozenInstanceError):
        result.final_bootstrap_inference[0].interval_lower = 0.0
    with pytest.raises(FrozenInstanceError):
        result.calibration_transfer.frozen_model.slope = 1.0
