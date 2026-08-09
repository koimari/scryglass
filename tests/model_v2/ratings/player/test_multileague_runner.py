from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

import lol_kills.v2.ratings.player.multileague_runner as runner
from lol_kills.v2.ratings.player.multileague_development import (
    CLAIM_CEILING,
    DevelopmentMap,
    DevelopmentSeries,
    ObservedLineup,
    PlayerSlot,
    PrivateMultiLeagueRatingInput,
    SealedMapMetadata,
    SealedSeriesMetadata,
)
from lol_kills.v2.ratings.player.multileague_runner import (
    CANDIDATES,
    MultiLeagueRunnerError,
    build_multileague_development_artifact,
    validate_multileague_development_artifact,
    verify_multileague_development_artifact,
    write_multileague_development_artifact_no_clobber,
)


MAPS_PIN = "a" * 64
PLAYERS_PIN = "b" * 64
CURRENT_TEST_MAPS_SHA256 = "04c0cce1d86a4358d9eeb5937f61d5288358953e66c693a1ce88b0b650295d08"
CURRENT_TEST_PLAYERS_SHA256 = "12f1cca978d683a0df8ceec0772999aeb03c723b4465f98674247f327dea71fa"


def _lineup(side: str, team: str, suffix: str = "") -> ObservedLineup:
    return ObservedLineup(
        side=side,
        team_id=f"oe:team:{team}",
        team_key=team,
        team_name=team.title(),
        players=tuple(
            PlayerSlot(
                role=role,
                player_id=f"oe:player:{team}:{role}{suffix}",
                player_name=f"{team}-{role}{suffix}",
                team_id=f"oe:team:{team}",
            )
            for role in ("top", "jungle", "mid", "bot", "support")
        ),
    )


def _map(
    game_id: str,
    series_id: str,
    fold: str,
    at: str,
    game_number: int,
    outcome: int,
    *,
    league: str = "LCS",
    blue: str = "blue",
    red: str = "red",
) -> DevelopmentMap:
    return DevelopmentMap(
        game_id=game_id,
        series_id=series_id,
        series_identity_kind="DERIVED_DEPENDENCE_CLUSTER",
        fold_id=fold,
        league=league,
        source_local_start=at,
        game_number=game_number,
        patch_token="15.10",
        blue_lineup=_lineup("blue", blue),
        red_lineup=_lineup("red", red),
        blue_win=outcome,
    )


def _series(
    series_id: str,
    fold: str,
    starts: list[str],
    outcomes: list[int],
    *,
    league: str = "LCS",
    blue: str = "blue",
    red: str = "red",
) -> DevelopmentSeries:
    maps = tuple(
        _map(
            f"{series_id}-g{index}",
            series_id,
            fold,
            at,
            index,
            outcome,
            league=league,
            blue=blue,
            red=red,
        )
        for index, (at, outcome) in enumerate(zip(starts, outcomes), start=1)
    )
    return DevelopmentSeries(
        series_id=series_id,
        series_identity_kind="DERIVED_DEPENDENCE_CLUSTER",
        fold_id=fold,
        league=league,
        source_local_start=starts[0],
        source_local_end=starts[-1],
        maps=maps,
    )


def _sealed() -> SealedSeriesMetadata:
    blue = _lineup("blue", "blue")
    red = _lineup("red", "red")
    item = SealedMapMetadata(
        game_id="sealed-g1",
        series_id="sealed",
        series_identity_kind="DERIVED_DEPENDENCE_CLUSTER",
        league="LCS",
        source_local_start="2026-05-01T10:00:00",
        game_number=1,
        patch_token="16.08",
        blue_lineup=blue,
        red_lineup=red,
    )
    return SealedSeriesMetadata(
        series_id="sealed",
        series_identity_kind="DERIVED_DEPENDENCE_CLUSTER",
        league="LCS",
        source_local_start=item.source_local_start,
        source_local_end=item.source_local_start,
        maps=(item,),
    )


def _input(series: tuple[DevelopmentSeries, ...] | None = None) -> PrivateMultiLeagueRatingInput:
    series = series or (
        _series(
            "train",
            "TRAIN",
            ["2025-06-01T10:00:00", "2025-06-01T11:00:00"],
            [1, 0],
        ),
        _series("development", "DEVELOPMENT", ["2025-08-01T10:00:00"], [1]),
        _series("validation", "VALIDATION", ["2026-02-01T10:00:00"], [0]),
    )
    sealed = (_sealed(),)
    selected = sum(len(value.maps) for value in series) + 1
    value = PrivateMultiLeagueRatingInput(
        schema_version="scryglass:multileague-player-development-input:v1",
        maps_locator="maps.parquet",
        players_locator="players.parquet",
        maps_sha256=MAPS_PIN,
        players_sha256=PLAYERS_PIN,
        development_selected_rows_sha256="c" * 64,
        sealed_selected_metadata_sha256="d" * 64,
        player_selected_metadata_sha256="e" * 64,
        cluster_partition_sha256="0" * 64,
        development_series=series,
        sealed_series_metadata=sealed,
        quarantined_clusters=(),
        coverage={"selected_maps": selected},
        claim_ceiling=dict(CLAIM_CEILING),
    )
    return replace(
        value,
        cluster_partition_sha256=runner._canonical_sha256(runner._partition_payload(value)),
    )


def _loader(value: PrivateMultiLeagueRatingInput):
    def load(**_kwargs):
        return value

    return load


def test_rank_one_update_creates_and_preserves_full_covariance() -> None:
    state = runner._GaussianState(CANDIDATES[0])
    at = runner.source_local_datetime("2025-01-01T00:00:00")
    state.transition_players(["a", "b"], at)
    state.ensure_structural_keys([])
    weights = {
        runner._player_key("a"): 0.5,
        runner._player_key("b"): -0.5,
        runner.BLUE_SIDE_KEY: 1.0,
    }

    state.update(weights, 1)

    a = state.index[runner._player_key("a")]
    b = state.index[runner._player_key("b")]
    assert state.covariance[a, b] > 0.0
    assert state.covariance[a, state.index[runner.BLUE_SIDE_KEY]] < 0.0
    assert state.assert_psd()["minimum_eigenvalue"] >= -runner.PSD_TOLERANCE


def test_series_forecasts_are_frozen_and_exact_48_hours_is_not_eligible() -> None:
    series = (
        _series(
            "origin",
            "TRAIN",
            ["2025-06-01T10:00:00", "2025-06-01T11:00:00"],
            [1, 1],
        ),
        _series("exact-48h", "TRAIN", ["2025-06-03T11:00:00"], [0]),
        _series("after-48h", "TRAIN", ["2025-06-03T11:00:01"], [0]),
        _series("development", "DEVELOPMENT", ["2025-08-01T10:00:00"], [1]),
        _series("validation", "VALIDATION", ["2026-02-01T10:00:00"], [0]),
    )
    replay = runner._replay(_input(series), CANDIDATES[0])
    by_id = {row["game_id"]: row for row in replay.predictions}

    assert by_id["origin-g1"]["probability"] == pytest.approx(
        by_id["origin-g2"]["probability"], abs=1e-15
    )
    assert by_id["exact-48h-g1"]["probability"] == pytest.approx(0.5, abs=1e-15)
    assert by_id["after-48h-g1"]["probability"] > 0.5


def test_forged_pin_or_cluster_partition_is_rejected() -> None:
    value = _input()
    with pytest.raises(MultiLeagueRunnerError, match="independent warehouse pins"):
        build_multileague_development_artifact(
            expected_maps_sha256="f" * 64,
            expected_players_sha256=PLAYERS_PIN,
            input_loader=_loader(value),
        )

    forged = replace(value, cluster_partition_sha256="f" * 64)
    with pytest.raises(MultiLeagueRunnerError, match="partition digest"):
        build_multileague_development_artifact(
            expected_maps_sha256=MAPS_PIN,
            expected_players_sha256=PLAYERS_PIN,
            input_loader=_loader(forged),
        )


def test_tiny_artifact_keeps_unidentified_team_components_null() -> None:
    artifact = build_multileague_development_artifact(
        expected_maps_sha256=MAPS_PIN,
        expected_players_sha256=PLAYERS_PIN,
        input_loader=_loader(_input()),
    )
    validate_multileague_development_artifact(artifact)

    assert artifact["selection"]["sealed_final_opened"] is False
    assert set(artifact["decision_outputs"].values()) == {None}
    assert artifact["development_posterior"]["teams"]
    for team in artifact["development_posterior"]["teams"]:
        assert team["components"]["lineup_synergy"] == {
            "status": "UNAVAILABLE",
            "value": None,
            "reason": "not_identified_by_the_player_plus_league_outcome_estimand",
        }
        assert team["components"]["team_policy"]["status"] == "UNAVAILABLE"
        assert team["components"]["team_policy"]["value"] is None

    forged = json.loads(json.dumps(artifact))
    forged["decision_outputs"]["fair_odds"] = 1.9
    unsigned = dict(forged)
    unsigned.pop("artifact_sha256")
    forged["artifact_sha256"] = runner._canonical_sha256(unsigned)
    with pytest.raises(MultiLeagueRunnerError, match="actionable"):
        validate_multileague_development_artifact(forged)


def test_external_artifact_pin_and_no_clobber_writer(tmp_path: Path) -> None:
    artifact = build_multileague_development_artifact(
        expected_maps_sha256=MAPS_PIN,
        expected_players_sha256=PLAYERS_PIN,
        input_loader=_loader(_input()),
    )
    verify_multileague_development_artifact(
        artifact, expected_artifact_sha256=artifact["artifact_sha256"]
    )
    with pytest.raises(MultiLeagueRunnerError, match="independent pin"):
        verify_multileague_development_artifact(
            artifact, expected_artifact_sha256="0" * 64
        )

    path = tmp_path / "candidate.json"
    first_raw_sha = write_multileague_development_artifact_no_clobber(artifact, path)
    before = path.read_bytes()
    assert first_raw_sha == runner._sha256(before)
    with pytest.raises(MultiLeagueRunnerError, match="refusing to clobber"):
        write_multileague_development_artifact_no_clobber(artifact, path)
    assert path.read_bytes() == before


@pytest.mark.skipif(
    not (Path(__file__).parents[4] / "data/lol/warehouse/parquet/maps.parquet").exists(),
    reason="private warehouse snapshot is absent",
)
def test_current_snapshot_fails_closed_at_validation() -> None:
    artifact = build_multileague_development_artifact(
        expected_maps_sha256=CURRENT_TEST_MAPS_SHA256,
        expected_players_sha256=CURRENT_TEST_PLAYERS_SHA256,
    )
    validate_multileague_development_artifact(artifact)

    assert artifact["result_state"] == "DEVELOPMENT_CANDIDATE_VALIDATION_GATE_FAILED"
    assert artifact["selection"] == {
        "development_winner_candidate_id": "random_walk_no_reset",
        "validation_gate_passed": False,
        "validation_gate_failures": [
            "overall_validation_interval_does_not_dominate_static",
            "lcs_validation_interval_does_not_dominate_static",
            "lec_validation_interval_does_not_dominate_static",
            "lck_validation_interval_does_not_dominate_static",
            "lpl_validation_interval_does_not_dominate_static",
        ],
        "candidate_eligible_for_separately_authorized_sealed_evaluation": None,
        "sealed_final_opened": False,
    }
    result = next(
        item
        for item in artifact["candidate_results"]
        if item["candidate"]["candidate_id"] == "random_walk_no_reset"
    )
    assert result["development"]["overall"]["series_macro"]["log_loss"] == pytest.approx(
        0.6209425852603869, abs=1e-12
    )
    assert result["validation"]["overall"]["series_macro"]["log_loss"] == pytest.approx(
        0.6578377275941257, abs=1e-12
    )
    regions = {
        item["league"]: item["status"]
        for item in result["paired_against_static"]["validation"]["domestic_leagues"]
    }
    assert regions == {
        "LCS": "FAIL_UPPER_95_ABOVE_ZERO",
        "LEC": "FAIL_UPPER_95_ABOVE_ZERO",
        "LCK": "FAIL_UPPER_95_ABOVE_ZERO",
        "LPL": "FAIL_UPPER_95_ABOVE_ZERO",
    }
    assert len(artifact["development_posterior"]["players"]) == 359
    assert len(artifact["development_posterior"]["teams"]) == 56
    assert set(artifact["decision_outputs"].values()) == {None}
