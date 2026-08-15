from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa
import pytest

from lol_kills.export.pack_spec import RATINGS_SNAPSHOT_COLS
from lol_kills.export.public_pack import export_public_pack, serialize_rating_snapshot_rows
from lol_kills.ratings.dual_elo import (
    DualEloConfig,
    apply_team_momentum_snapshot,
    build_dual_ratings,
)
from lol_kills.ratings.momentum_config import (
    CANDIDATE_MOMENTUM_SCALE,
    CANDIDATE_MOMENTUM_WINDOW_GAMES,
    DEFAULT_MOMENTUM_SCALE,
    DEFAULT_MOMENTUM_WINDOW_GAMES,
    PublicMomentumAuthorityError,
    registered_momentum_bundle,
    require_public_momentum_disabled,
)
from lol_kills.ratings.player_elo import (
    PlayerEloConfig,
    build_maps_frame_from_players,
    build_player_ratings,
)


def _maps() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_uid": "g2", "date": "2026-01-01", "league": "LCK", "blue_team": "A", "red_team": "B", "y_blue_win": 0},
            {"game_uid": "g1", "date": "2026-01-01", "league": "LCK", "blue_team": "A", "red_team": "B", "y_blue_win": 1},
            {"game_uid": "g3", "date": "2026-01-02", "league": "LCK", "blue_team": "A", "red_team": "B", "y_blue_win": 1},
        ]
    )


def _players() -> pd.DataFrame:
    rows = []
    roles = ["top", "jng", "mid", "bot", "sup"]
    for game_uid, date, blue_win in (
        ("g1", "2026-01-01", 1),
        ("g2", "2026-01-01", 0),
        ("g3", "2026-01-02", 1),
    ):
        for side, team, result in (("Blue", "A", blue_win), ("Red", "B", 1 - blue_win)):
            for index, role in enumerate(roles):
                rows.append(
                    {
                        "game_uid": game_uid,
                        "gameid": game_uid,
                        "date": date,
                        "league": "LCK",
                        "side": side,
                        "position": role,
                        "playername": f"{team}{index}",
                        "teamname": team,
                        "result": result,
                    }
                )
    return pd.DataFrame(rows)


def test_zero_momentum_is_the_active_default_and_preserves_baseline() -> None:
    assert DEFAULT_MOMENTUM_WINDOW_GAMES == 0
    assert DEFAULT_MOMENTUM_SCALE == 0.0
    default = build_dual_ratings(_maps(), output_dir=None)
    explicit = build_dual_ratings(
        _maps(), DualEloConfig(momentum_window_games=0, momentum_scale=0.0), output_dir=None
    )
    pd.testing.assert_frame_equal(
        default[["game_uid", "mu_blue", "mu_red", "mu_diff", "p_dual_elo", "p_dual_elo_raw"]],
        explicit[["game_uid", "mu_blue", "mu_red", "mu_diff", "p_dual_elo", "p_dual_elo_raw"]],
    )
    assert (default["momentum_diff"] == 0.0).all()


def test_candidate_is_pre_match_and_keeps_deterministic_order_and_sides() -> None:
    ratings = build_dual_ratings(
        _maps(),
        DualEloConfig(
            momentum_window_games=CANDIDATE_MOMENTUM_WINDOW_GAMES,
            momentum_scale=CANDIDATE_MOMENTUM_SCALE,
        ),
        output_dir=None,
    )
    assert ratings["game_uid"].tolist() == ["g1", "g2", "g3"]
    assert ratings.iloc[0]["momentum_diff"] == 0.0
    assert ratings.iloc[1]["momentum_blue"] > 0.0
    assert ratings.iloc[1]["momentum_red"] < 0.0
    assert ratings.iloc[1]["p_dual_elo_raw"] > ratings.iloc[1]["p_dual_elo_base_raw"]
    assert ratings.iloc[1]["blue_team"] == "A"
    assert ratings.iloc[1]["red_team"] == "B"


def test_player_candidate_preserves_base_and_effective_snapshot_values(monkeypatch) -> None:
    players = _players()
    maps = build_maps_frame_from_players(players)
    monkeypatch.setattr(
        "lol_kills.ratings.player_elo.fit_global_player_bt",
        lambda *args, **kwargs: (pd.DataFrame(), {}),
    )
    with TemporaryDirectory() as directory:
        build_player_ratings(
            maps,
            players,
            PlayerEloConfig(
                momentum_window_games=CANDIDATE_MOMENTUM_WINDOW_GAMES,
                momentum_scale=CANDIDATE_MOMENTUM_SCALE,
            ),
            output_dir=Path(directory),
        )
        snapshot = pd.read_parquet(Path(directory) / "player_ratings_snapshot.parquet")
    assert {"mu_base_total", "mu_total", "mu_effective", "momentum_residual"}.issubset(snapshot.columns)
    assert (snapshot["mu_total"] == snapshot["mu_effective"]).all()


def test_production_serializer_keeps_enabled_mu_total_and_mu_base_total() -> None:
    table = pa.Table.from_pandas(
        pd.DataFrame(
            [
                {
                    "team": "A",
                    "mu_base_total": 1500.0,
                    "mu_total": 1540.0,
                    "mu_effective": 1540.0,
                    "momentum_residual": 0.5,
                    "mu_regional": 1500.0,
                    "mu_meta": 0.0,
                    "sigma": 30.0,
                }
            ]
        ),
        preserve_index=False,
    )
    rows, columns = serialize_rating_snapshot_rows(table, RATINGS_SNAPSHOT_COLS)
    assert {"mu_base_total", "mu_total", "mu_effective", "momentum_residual"}.issubset(columns)
    assert rows[0]["mu_total"] == 1540.0
    assert rows[0]["mu_base_total"] == 1500.0


def test_enabled_team_momentum_reaches_production_snapshot_serializer() -> None:
    base = pd.DataFrame(
        [{"team": "A", "team_key": "a", "mu_total": 1600.0, "rating_p10": 1550.0}]
    )
    sequential = pd.DataFrame([{"team": "A", "momentum_residual": 0.5}])
    enabled = apply_team_momentum_snapshot(
        base,
        sequential,
        DualEloConfig(momentum_window_games=7, momentum_scale=80.0),
    )
    table = pa.Table.from_pandas(enabled, preserve_index=False)
    rows, columns = serialize_rating_snapshot_rows(table, RATINGS_SNAPSHOT_COLS)
    assert {"mu_base_total", "mu_total", "mu_effective", "momentum_residual"}.issubset(columns)
    assert rows[0]["mu_base_total"] == 1600.0
    assert rows[0]["mu_total"] == 1640.0


def test_candidate_bundle_is_research_only() -> None:
    bundle = registered_momentum_bundle()
    assert bundle["active"]["window_games"] == 0
    assert bundle["active"]["scale"] == 0.0
    assert bundle["candidate"]["window_games"] == 7
    assert bundle["candidate"]["scale"] == 80.0
    assert all(value is False for value in bundle["candidate"]["authority"].values())
    assert bundle["promotion"]["status"] == "unavailable"


def test_public_pack_rejects_research_momentum_before_touching_sources(tmp_path: Path) -> None:
    with pytest.raises(PublicMomentumAuthorityError, match="promotion contract"):
        export_public_pack(
            project_root=tmp_path,
            runtime_root=tmp_path,
            out_root=tmp_path / "output",
            momentum_window_games=CANDIDATE_MOMENTUM_WINDOW_GAMES,
            momentum_scale=CANDIDATE_MOMENTUM_SCALE,
        )


def test_scheduled_refresh_rejects_research_momentum_before_reading_sources(tmp_path: Path) -> None:
    from lol_kills.v2.tierlists.rating_refresh import refresh_ratings

    with pytest.raises(PublicMomentumAuthorityError, match="promotion contract"):
        refresh_ratings(
            tmp_path,
            momentum_window_games=CANDIDATE_MOMENTUM_WINDOW_GAMES,
            momentum_scale=CANDIDATE_MOMENTUM_SCALE,
        )


def test_public_entry_guard_accepts_only_active_zero_state() -> None:
    assert require_public_momentum_disabled(
        window_games=DEFAULT_MOMENTUM_WINDOW_GAMES,
        scale=DEFAULT_MOMENTUM_SCALE,
        entrypoint="test",
    ) is None


def test_public_pack_zero_momentum_reaches_source_validation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export_public_pack(
            project_root=tmp_path,
            runtime_root=tmp_path,
            out_root=tmp_path / "output",
            momentum_window_games=DEFAULT_MOMENTUM_WINDOW_GAMES,
            momentum_scale=DEFAULT_MOMENTUM_SCALE,
        )


def test_scheduled_refresh_zero_momentum_reaches_source_validation(tmp_path: Path) -> None:
    from lol_kills.v2.tierlists.rating_refresh import refresh_ratings

    with pytest.raises(FileNotFoundError):
        refresh_ratings(
            tmp_path,
            momentum_window_games=DEFAULT_MOMENTUM_WINDOW_GAMES,
            momentum_scale=DEFAULT_MOMENTUM_SCALE,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"momentum_window_games": -1, "momentum_scale": 80.0},
        {"momentum_window_games": 1.5, "momentum_scale": 80.0},
        {"momentum_window_games": 7, "momentum_scale": -1.0},
        {"momentum_window_games": 7, "momentum_scale": float("nan")},
    ],
)
def test_invalid_momentum_configuration_fails_closed(kwargs) -> None:
    with pytest.raises(ValueError):
        DualEloConfig(**kwargs)
    with pytest.raises(ValueError):
        PlayerEloConfig(**kwargs)
