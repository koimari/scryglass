from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import lol_kills.v2.ratings.player.multileague_benchmark as benchmark
from lol_kills.v2.ratings.player.multileague_development import (
    DevelopmentMap,
    ObservedLineup,
    PlayerSlot,
)


ROOT = Path(__file__).parents[4]
RATING_PATH = (
    ROOT
    / "data/lol/v2/models/player/multileague-v1/private-development-artifact-v1.json"
)
MAPS_SHA256 = "04c0cce1d86a4358d9eeb5937f61d5288358953e66c693a1ce88b0b650295d08"
PLAYERS_SHA256 = "12f1cca978d683a0df8ceec0772999aeb03c723b4465f98674247f327dea71fa"
RATING_ARTIFACT_SHA256 = "116a835e4e5c5a801a8b8faa1519e39171ca64b4c5e3edfb80b8a98201222287"


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


def _map(blue_suffix: str = "", red_suffix: str = "") -> DevelopmentMap:
    return DevelopmentMap(
        game_id="game",
        series_id="series",
        series_identity_kind="DERIVED_DEPENDENCE_CLUSTER",
        fold_id="VALIDATION",
        league="LCS",
        source_local_start="2026-02-01T10:00:00",
        game_number=1,
        patch_token="16.03",
        blue_lineup=_lineup("blue", "blue", blue_suffix),
        red_lineup=_lineup("red", "red", red_suffix),
        blue_win=1,
    )


def test_roster_change_stratum_uses_only_prior_available_exact_lineups() -> None:
    item = _map()
    known = {
        item.blue_lineup.team_id: benchmark._lineup_identity(item.blue_lineup),
        item.red_lineup.team_id: benchmark._lineup_identity(item.red_lineup),
    }
    assert benchmark._roster_status(item, known) == (
        "STABLE",
        "STABLE",
        "BOTH_ROSTERS_STABLE",
    )

    changed = _map(blue_suffix="-sub")
    assert benchmark._roster_status(changed, known) == (
        "CHANGED",
        "STABLE",
        "ONE_OR_BOTH_ROSTERS_CHANGED",
    )
    assert benchmark._roster_status(item, {}) == (
        "NO_PRIOR_EXACT_LINEUP",
        "NO_PRIOR_EXACT_LINEUP",
        "NO_PRIOR_EXACT_LINEUP",
    )


@pytest.fixture(scope="module")
def current_benchmark() -> dict:
    if not RATING_PATH.is_file():
        pytest.skip("private multi-league rating artifact is absent")
    rating_artifact = json.loads(RATING_PATH.read_text(encoding="utf-8"))
    return benchmark.build_strong_baseline_benchmark(
        expected_maps_sha256=MAPS_SHA256,
        expected_players_sha256=PLAYERS_SHA256,
        rating_artifact=rating_artifact,
        expected_rating_artifact_sha256=RATING_ARTIFACT_SHA256,
    )


def test_rating_artifact_requires_independent_pin() -> None:
    if not RATING_PATH.is_file():
        pytest.skip("private multi-league rating artifact is absent")
    rating_artifact = json.loads(RATING_PATH.read_text(encoding="utf-8"))
    with pytest.raises(benchmark.MultiLeagueBenchmarkError, match="independent pin"):
        benchmark.build_strong_baseline_benchmark(
            expected_maps_sha256=MAPS_SHA256,
            expected_players_sha256=PLAYERS_SHA256,
            rating_artifact=rating_artifact,
            expected_rating_artifact_sha256="0" * 64,
        )


def test_current_player_candidate_does_not_clear_strong_baseline_gate(
    current_benchmark: dict,
) -> None:
    artifact = benchmark.validate_strong_baseline_benchmark(current_benchmark)
    assert artifact["result_state"] == "PLAYER_DOES_NOT_BEAT_STRONG_BASELINE"
    assert artifact["selection"]["organization_candidate_id"] == (
        "organization_random_walk_no_reset"
    )
    assert artifact["selection"]["validation_gate_passed"] is False
    assert artifact["selection"]["sealed_final_opened"] is False
    assert set(artifact["decision_outputs"].values()) == {None}

    comparison = artifact["validation_player_minus_selected_organization"]
    assert comparison["overall"]["log_loss_player_minus_organization"][
        "upper_95"
    ] < 0.0
    assert comparison["overall"]["brier_player_minus_organization"][
        "upper_95"
    ] > 0.0
    regions = {item["league"]: item["status"] for item in comparison["by_domestic_league"]}
    assert regions == {
        "LCS": "FAIL_UPPER_95_ABOVE_ZERO",
        "LEC": "FAIL_UPPER_95_ABOVE_ZERO",
        "LCK": "FAIL_UPPER_95_ABOVE_ZERO",
        "LPL": "FAIL_UPPER_95_ABOVE_ZERO",
    }
    roster = {
        item["roster_change_stratum"]: item
        for item in comparison["by_roster_change_stratum"]
    }
    assert roster["BOTH_ROSTERS_STABLE"]["status"] == "PASS_NONPOSITIVE_UPPER_95"
    assert roster["ONE_OR_BOTH_ROSTERS_CHANGED"]["status"] == (
        "FAIL_UPPER_95_ABOVE_ZERO"
    )
    assert roster["ONE_OR_BOTH_ROSTERS_CHANGED"]["series"] == 40


def test_benchmark_no_clobber_and_actionable_output_rejection(
    current_benchmark: dict, tmp_path: Path
) -> None:
    output = tmp_path / "benchmark.json"
    raw_sha = benchmark.write_strong_baseline_benchmark_no_clobber(
        current_benchmark, output
    )
    assert raw_sha == hashlib.sha256(output.read_bytes()).hexdigest()
    before = output.read_bytes()
    with pytest.raises(benchmark.MultiLeagueBenchmarkError, match="refusing to clobber"):
        benchmark.write_strong_baseline_benchmark_no_clobber(current_benchmark, output)
    assert output.read_bytes() == before

    forged = json.loads(json.dumps(current_benchmark))
    forged["decision_outputs"]["expected_value"] = 0.1
    unsigned = dict(forged)
    unsigned.pop("artifact_sha256")
    forged["artifact_sha256"] = benchmark.rating._canonical_sha256(unsigned)
    with pytest.raises(benchmark.MultiLeagueBenchmarkError, match="actionable"):
        benchmark.validate_strong_baseline_benchmark(forged)
