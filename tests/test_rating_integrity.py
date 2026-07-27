from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from lol_kills.export.player_metadata import build_player_metadata
from lol_kills.ratings import (
    calibrate_elo_wr,
    dual_elo,
    hierarchical_bt,
    player_elo,
)
from lol_kills.ratings.calibrate_elo_wr import (
    CalibrationArtifactError,
    apply_scale,
    fit_elo_wr_calibration,
)
from lol_kills.ratings.hierarchical_bt import (
    HierarchicalRatingError,
    _observations,
    fit_hierarchical_bt,
)
from lol_kills.ratings.player_elo import (
    PlayerEloConfig,
    PlayerIdentityError,
    _run_player_elo,
    score_player_lineups,
)

ROLES = ("top", "jng", "mid", "bot", "sup")


def _map(
    game_uid: str,
    date: str,
    blue_team: str,
    red_team: str,
    outcome: float,
    gold_diff: float | None = None,
) -> dict[str, object]:
    return {
        "game_uid": game_uid,
        "date": date,
        "league": "LCS",
        "blue_team": blue_team,
        "red_team": red_team,
        "y_blue_win": outcome,
        "blue_golddiffat15": gold_diff,
        "length_min": 30.0,
    }


def _player_rows(
    game_uid: str,
    date: str,
    blue_team: str,
    red_team: str,
    *,
    names_by_id: dict[str, str] | None = None,
    missing_id: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    names_by_id = names_by_id or {}
    for side, team, prefix in (
        ("Blue", blue_team, blue_team.casefold()),
        ("Red", red_team, red_team.casefold()),
    ):
        for index, role in enumerate(ROLES):
            player_id = f"{prefix}-{index}"
            rows.append(
                {
                    "game_uid": game_uid,
                    "date": date,
                    "league": "LCS",
                    "side": side,
                    "teamname": team,
                    "position": role,
                    "playerid": (
                        None if player_id == missing_id else player_id
                    ),
                    "playername": names_by_id.get(
                        player_id, f"{team}{index}"
                    ),
                }
            )
    return rows


def _hierarchical_maps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_uid": "series-1-game-1",
                "date": "2026-01-01 10:00",
                "league": "LCS",
                "blue_team": "A",
                "red_team": "B",
                "y_blue_win": 1,
                "grid_series_id": "series-1",
                "grid_game_index": 1,
                "series_format": "Bo3",
            },
            {
                "game_uid": "series-1-game-2",
                "date": "2026-01-01 10:35",
                "league": "LCS",
                "blue_team": "A",
                "red_team": "B",
                "y_blue_win": 1,
                "grid_series_id": "series-1",
                "grid_game_index": 2,
                "series_format": "Bo3",
            },
        ]
    )


def _calibration_frame(n_rows: int = 800) -> pd.DataFrame:
    index = np.arange(n_rows, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n_rows, freq="D"),
            "y_blue_win": (index.astype(int) % 2).astype(float),
            "mu_diff": 160.0 * np.sin(index / 17.0),
            "player_mu_diff": 130.0 * np.cos(index / 19.0),
        }
    )


def test_dual_elo_margin_update_is_side_swap_symmetric(
    tmp_path: Path,
) -> None:
    direct = pd.DataFrame(
        [
            _map("g1", "2026-01-01", "A", "B", 1.0, 3000.0),
            _map("g2", "2026-01-02", "A", "B", np.nan),
        ]
    )
    swapped = pd.DataFrame(
        [
            _map("g1", "2026-01-01", "B", "A", 0.0, -3000.0),
            _map("g2", "2026-01-02", "A", "B", np.nan),
        ]
    )
    with patch.object(dual_elo, "FEATURES_DIR", tmp_path):
        direct_out = dual_elo.build_dual_ratings(direct)
        swapped_out = dual_elo.build_dual_ratings(swapped)

    direct_state = direct_out.iloc[1]
    swapped_state = swapped_out.iloc[1]
    assert direct_state["mu_blue"] == pytest.approx(
        swapped_state["mu_blue"]
    )
    assert direct_state["mu_red"] == pytest.approx(
        swapped_state["mu_red"]
    )
    assert direct_state["mu_diff"] == pytest.approx(
        swapped_state["mu_diff"]
    )


def test_player_elo_margin_update_is_side_swap_symmetric() -> None:
    direct_maps = pd.DataFrame(
        [
            _map("g1", "2026-01-01", "A", "B", 1.0, 3000.0),
            _map("g2", "2026-01-02", "A", "B", np.nan),
        ]
    )
    direct_players = pd.DataFrame(
        _player_rows("g1", "2026-01-01", "A", "B")
        + _player_rows("g2", "2026-01-02", "A", "B")
    )
    swapped_maps = pd.DataFrame(
        [
            _map("g1", "2026-01-01", "B", "A", 0.0, -3000.0),
            _map("g2", "2026-01-02", "A", "B", np.nan),
        ]
    )
    swapped_players = pd.DataFrame(
        _player_rows("g1", "2026-01-01", "B", "A")
        + _player_rows("g2", "2026-01-02", "A", "B")
    )

    direct_out, _, _, direct_resolution = _run_player_elo(
        direct_maps, direct_players, PlayerEloConfig()
    )
    swapped_out, _, _, swapped_resolution = _run_player_elo(
        swapped_maps, swapped_players, PlayerEloConfig()
    )
    assert direct_resolution.audit["n_quarantined_maps"] == 0
    assert swapped_resolution.audit["n_quarantined_maps"] == 0
    assert direct_out.iloc[1]["player_mu_blue"] == pytest.approx(
        swapped_out.iloc[1]["player_mu_blue"]
    )
    assert direct_out.iloc[1]["player_mu_red"] == pytest.approx(
        swapped_out.iloc[1]["player_mu_red"]
    )


def test_provider_id_is_the_player_state_key_across_handle_change() -> None:
    maps = pd.DataFrame(
        [
            _map("g1", "2026-01-01", "A", "B", 1.0),
            _map("g2", "2026-01-02", "A", "B", 0.0),
        ]
    )
    first = _player_rows("g1", "2026-01-01", "A", "B")
    second = _player_rows(
        "g2",
        "2026-01-02",
        "A",
        "B",
        names_by_id={"a-0": "RenamedA0"},
    )
    _, states, _, resolution = _run_player_elo(
        maps, pd.DataFrame(first + second), PlayerEloConfig()
    )

    assert resolution.audit["identity_mode"] == "provider_playerid"
    assert len(states) == 10
    assert states["provider:a-0"].n_maps == 2
    assert states["provider:a-0"].aliases == {"A0", "RenamedA0"}


def test_player_snapshot_persists_provider_identity_and_audit(
    tmp_path: Path,
) -> None:
    maps = pd.DataFrame(
        [_map("g1", "2026-01-01", "A", "B", 1.0)]
    )
    players = pd.DataFrame(
        _player_rows("g1", "2026-01-01", "A", "B")
    )
    with patch.object(player_elo, "FEATURES_DIR", tmp_path):
        player_elo.build_player_ratings(maps, players)

    snapshot = pd.read_parquet(
        tmp_path / "player_ratings_snapshot.parquet"
    )
    metadata = json.loads(
        (tmp_path / "player_ratings_meta.json").read_text(encoding="utf-8")
    )
    assert snapshot["player_id"].nunique() == 10
    assert snapshot["player_identity"].nunique() == 10
    assert metadata["identity_audit"]["stable_provider_ids"]
    assert metadata["identity_audit"]["n_quarantined_maps"] == 0
    assert "display_name_collisions" not in metadata["identity_audit"]
    assert metadata["identity_audit"]["n_display_name_collisions"] == 0
    assert metadata["identity_audit"]["display_name_collision_examples"] == []


def test_name_based_lineup_scoring_accepts_unique_snapshot_rows() -> None:
    rows = []
    for team in ("A", "B"):
        for index in range(5):
            rows.append(
                {
                    "player": f"{team}{index}",
                    "player_id": f"{team.casefold()}-{index}",
                    "mu_regional": 1500.0,
                    "mu_meta": 0.0,
                    "sigma": 40.0,
                    "n_maps": 10,
                    "last_team": team,
                    "identity_source": "provider_playerid",
                }
            )
    scored = score_player_lineups(
        [f"A{index}" for index in range(5)],
        [f"B{index}" for index in range(5)],
        snapshot=pd.DataFrame(rows),
    )
    assert scored["player_known_blue"] == 5
    assert scored["player_known_red"] == 5


def test_name_based_lineup_scoring_rejects_colliding_snapshot_rows() -> None:
    snapshot = pd.DataFrame(
        [
            {
                "player": "Shared",
                "player_id": player_id,
                "mu_regional": 1500.0,
                "mu_meta": 0.0,
                "sigma": 40.0,
            }
            for player_id in ("provider-1", "provider-2")
        ]
    )
    with pytest.raises(PlayerIdentityError, match="colliding display names"):
        score_player_lineups(
            ["Shared"], ["Opponent"], snapshot=snapshot
        )


@pytest.mark.parametrize(
    ("players", "reason"),
    [
        (
            pd.DataFrame(
                _player_rows(
                    "g1",
                    "2026-01-01",
                    "A",
                    "B",
                    names_by_id={"a-0": "Shared", "b-0": "Shared"},
                )
            ),
            "display_name_maps_to_multiple_player_ids",
        ),
        (
            pd.DataFrame(
                _player_rows(
                    "g1",
                    "2026-01-01",
                    "A",
                    "B",
                    missing_id="a-0",
                )
            ),
            "missing_player_id",
        ),
    ],
)
def test_identity_collision_or_missing_id_quarantines_whole_map(
    players: pd.DataFrame,
    reason: str,
) -> None:
    maps = pd.DataFrame(
        [_map("g1", "2026-01-01", "A", "B", 1.0)]
    )
    out, states, _, resolution = _run_player_elo(
        maps, players, PlayerEloConfig()
    )

    assert states == {}
    assert not bool(out.iloc[0]["player_identity_eligible"])
    assert pd.isna(out.iloc[0]["player_mu_diff"])
    assert resolution.audit["n_quarantined_maps"] == 1
    assert resolution.audit["quarantine_reasons"][reason] == 1


def test_player_metadata_omits_display_name_shared_by_multiple_ids(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "leaguepedia.json"
    cache.write_text(
        json.dumps(
            [
                {
                    "Player": "Shared",
                    "Country": "Brazil",
                    "NationalityPrimary": "Brazil",
                }
            ]
        ),
        encoding="utf-8",
    )

    metadata = build_player_metadata(
        ["Shared"],
        cache_path=cache,
        player_identities=[("Shared", "provider-1"), ("Shared", "provider-2")],
    )
    assert metadata == {}


def test_hierarchical_lower_quantile_schema_matches_its_algebra() -> None:
    snapshot, meta = fit_hierarchical_bt(
        _hierarchical_maps(), write=False
    )
    expected_z = NormalDist().inv_cdf(0.95)

    assert "rating_p05" in snapshot
    assert "rating_p10" not in snapshot
    assert np.allclose(
        snapshot["rating_p05"],
        snapshot["mu_total"] - expected_z * snapshot["sigma"],
    )
    assert meta["uncertainty"] == {
        "field": "rating_p05",
        "lower_tail_probability": 0.05,
        "one_sided_coverage": 0.95,
        "z": expected_z,
        "formula": "rating_p05 = mu_total - z * sigma",
        "interpretation": (
            "Uncertainty-adjusted display score. Sigma contains a floor "
            "and bridge inflation in addition to local Laplace variance; "
            "coverage is not claimed until validated."
        ),
    }


def test_series_rating_is_invariant_to_first_map_side_assignment() -> None:
    direct = _hierarchical_maps()
    swapped = direct.copy()
    swapped["blue_team"] = direct["red_team"]
    swapped["red_team"] = direct["blue_team"]
    swapped["y_blue_win"] = 1.0 - direct["y_blue_win"]

    direct_snapshot, _ = fit_hierarchical_bt(direct, write=False)
    swapped_snapshot, _ = fit_hierarchical_bt(swapped, write=False)

    direct_ratings = direct_snapshot.set_index("team_key")["mu_total"].sort_index()
    swapped_ratings = swapped_snapshot.set_index("team_key")["mu_total"].sort_index()
    assert np.allclose(direct_ratings, swapped_ratings)


def test_as_of_excludes_in_progress_series_until_completion() -> None:
    maps = _hierarchical_maps()

    during = _observations(
        maps,
        pd.Timestamp("2026-01-01 10:20", tz="UTC"),
        365.0,
    )
    complete = _observations(
        maps,
        pd.Timestamp("2026-01-01 10:35", tz="UTC"),
        365.0,
    )

    assert during.empty
    assert len(complete) == 1
    assert complete.iloc[0]["date"] == pd.Timestamp(
        "2026-01-01 10:35", tz="UTC"
    )
    assert complete.iloc[0]["n_maps"] == 2


def test_rating_half_life_uses_base_two_and_explicit_as_of() -> None:
    maps = pd.DataFrame(
        [
            {
                "game_uid": "old",
                "date": "2025-01-01",
                "league": "LCS",
                "blue_team": "A",
                "red_team": "B",
                "y_blue_win": 1,
                "grid_series_id": "old",
                "grid_game_index": 1,
                "series_format": "Bo1",
            },
            {
                "game_uid": "new",
                "date": "2026-01-01",
                "league": "LCS",
                "blue_team": "A",
                "red_team": "B",
                "y_blue_win": 0,
                "grid_series_id": "new",
                "grid_game_index": 1,
                "series_format": "Bo1",
            },
        ]
    )

    observations = _observations(
        maps,
        pd.Timestamp("2026-01-01", tz="UTC"),
        365.0,
    ).sort_values("date")

    assert observations.iloc[0]["weight"] == pytest.approx(0.5)
    assert observations.iloc[1]["weight"] == pytest.approx(1.0)


def test_domestic_affiliation_is_frozen_at_series_boundary() -> None:
    maps = pd.DataFrame(
        [
            {
                "game_uid": "lck",
                "date": "2026-01-01",
                "league": "LCK",
                "blue_team": "A",
                "red_team": "C",
                "y_blue_win": 1,
                "grid_series_id": "lck",
                "grid_game_index": 1,
                "series_format": "Bo1",
            },
            {
                "game_uid": "ldl-1",
                "date": "2026-02-01 10:00",
                "league": "LDL",
                "blue_team": "A",
                "red_team": "B",
                "y_blue_win": 1,
                "grid_series_id": "ldl",
                "grid_game_index": 1,
                "series_format": "Bo3",
            },
            {
                "game_uid": "ldl-2",
                "date": "2026-02-01 10:35",
                "league": "LDL",
                "blue_team": "B",
                "red_team": "A",
                "y_blue_win": 0,
                "grid_series_id": "ldl",
                "grid_game_index": 2,
                "series_format": "Bo3",
            },
        ]
    )

    observations = _observations(maps, None, 365.0)
    ldl = observations.loc[observations["league"].eq("LDL")].iloc[0]

    assert ldl["home_a"] == "LDL"
    assert ldl["home_b"] == "LDL"


def test_hierarchical_optimizer_failure_cannot_write_snapshot(
    tmp_path: Path,
) -> None:
    failed = SimpleNamespace(
        success=False,
        x=np.zeros(3, dtype=float),
        fun=1.0,
        message="forced optimizer failure",
    )
    with (
        patch.object(hierarchical_bt, "FEATURES_DIR", tmp_path),
        patch.object(hierarchical_bt, "minimize", return_value=failed),
        pytest.raises(HierarchicalRatingError, match="optimizer failed"),
    ):
        fit_hierarchical_bt(_hierarchical_maps(), write=True)
    assert list(tmp_path.iterdir()) == []


def test_hierarchical_input_audit_failure_cannot_write_snapshot(
    tmp_path: Path,
) -> None:
    maps = _hierarchical_maps()
    maps.loc[1, "game_uid"] = maps.loc[0, "game_uid"]
    with (
        patch.object(hierarchical_bt, "FEATURES_DIR", tmp_path),
        pytest.raises(HierarchicalRatingError, match="input audit failed"),
    ):
        fit_hierarchical_bt(maps, write=True)
    assert list(tmp_path.iterdir()) == []


def test_hierarchical_input_audit_failure_blocks_read_only_fit() -> None:
    maps = _hierarchical_maps()
    maps.loc[1, "game_uid"] = maps.loc[0, "game_uid"]

    with pytest.raises(HierarchicalRatingError, match="input audit failed"):
        fit_hierarchical_bt(maps, write=False)


def test_calibration_never_backfills_missing_coefficients() -> None:
    with pytest.raises(
        CalibrationArtifactError, match="validated_time_holdout"
    ):
        apply_scale(np.array([0.0, 40.0]), {}, "player")


def test_incomplete_calibration_population_cannot_write_artifact(
    tmp_path: Path,
) -> None:
    frame = _calibration_frame()
    frame["player_mu_diff"] = np.nan
    output = tmp_path / "elo_wr_calibration.json"
    with (
        patch.object(calibrate_elo_wr, "OUT", output),
        patch.object(calibrate_elo_wr, "MODELS_DIR", tmp_path),
        pytest.raises(
            CalibrationArtifactError,
            match="incomplete calibration population",
        ),
    ):
        fit_elo_wr_calibration(frame)
    assert not output.exists()


def test_complete_calibration_artifact_declares_future_holdout(
    tmp_path: Path,
) -> None:
    output = tmp_path / "elo_wr_calibration.json"
    with (
        patch.object(calibrate_elo_wr, "OUT", output),
        patch.object(calibrate_elo_wr, "MODELS_DIR", tmp_path),
    ):
        artifact = fit_elo_wr_calibration(_calibration_frame())

    assert output.exists()
    assert artifact["status"] == "validated_time_holdout"
    assert set(artifact) >= {"team", "player", "strength_blend"}
    assert (
        pd.Timestamp(artifact["time_split"]["holdout_start"])
        > pd.Timestamp(artifact["time_split"]["train_end"])
    )
