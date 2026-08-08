"""Tests for the development champion-role Elo replay."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from lol_kills.v2.tierlists.champion_elo import (
    TIER_BUCKETS,
    ChampState,
    _assign_tier_buckets,
    _fit_joint_scope,
    build_candidate,
)


ROOT = Path(__file__).resolve().parents[3]


def _write_identity_sources(root: Path) -> None:
    crosswalk = {
        "entries": [
            {
                "oe_name": "Aatrox",
                "normalized_oe_name": "aatrox",
                "stable_champion_id": "riot:champion:266",
            }
        ]
    }
    metadata = {
        "version": "16.14.1",
        "data": {
            "Briar": {"name": "Briar", "key": "233"},
        },
    }
    crosswalk_path = root / "data/lol/v2/champions/champion-id-crosswalk-v1.json"
    metadata_path = root / "data/lol/v2/champions/sources/riot-champion-metadata-16.14.1.json"
    crosswalk_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    crosswalk_path.write_text(json.dumps(crosswalk), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    atom_path = root / "data/lol/v2/champions/lcc-atom-bridge-v1.json"
    shutil.copyfile(ROOT / "data/lol/v2/champions/lcc-atom-bridge-v1.json", atom_path)


def _source_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    roles = ["top", "jng", "mid", "bot", "sup"]
    for game_id, date, blue_champion, red_champion, blue_team, red_team in (
        ("g1", "2025-01-01T12:00:00Z", "Aatrox", "Briar", "Blue Team", "Red Team"),
        ("g2", "2025-01-02T12:00:00Z", "Briar", "Aatrox", "Blue Two", "Red Two"),
        ("g3", "2025-01-03T12:00:00Z", "Briar", "Aatrox", "Blue Three", "Red Three"),
    ):
        for side, champion, result in (
            ("Blue", blue_champion, 1),
            ("Red", red_champion, 0),
        ):
            for role in roles:
                rows.append(
                    {
                        "gameid": game_id,
                        "date": date,
                        "league": "LEC",
                        "competition_tier": "tier1",
                        "event_kind": "domestic",
                        "patch": "15.01",
                        "position": role,
                        "champion": champion,
                        "side": side,
                        "teamname": blue_team if side == "Blue" else red_team,
                        "result": result,
                    }
                )
    return pd.DataFrame(rows)


def test_replay_covers_all_roles_and_tracks_rank_movement(tmp_path: Path) -> None:
    _write_identity_sources(tmp_path)
    source_path = tmp_path / "data/lol/warehouse/parquet/oe_player_games.parquet"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    _source_rows().to_parquet(source_path, index=False)

    first = build_candidate(tmp_path, as_of=pd.Timestamp("2025-01-01T23:59:59Z"))
    current = build_candidate(tmp_path, previous=first)

    assert current["history_start"] == "2025-01-01T00:00:00Z"
    assert current["live_window_start"] == "2026-07-18T00:00:00Z"
    assert current["development_only"] is True
    assert current["publication_eligible"] is False
    assert current["source_mode"] == "oe_only"
    assert current["rating_method"]["name"].startswith("joint five-role")
    assert "full observed-Hessian" in current["rating_method"]["fit"]
    assert current["options"]["roles"] == ["top", "jungle", "mid", "bot", "support"]
    assert len(current["cells"]) == 5

    mid = next(cell for cell in current["cells"] if cell["role"] == "mid")
    rows = {row["champion"]: row for row in mid["rows"]}
    assert rows["Aatrox"]["movement"] == "down"
    assert rows["Aatrox"]["rank_delta"] == -1
    assert rows["Briar"]["movement"] == "up"
    assert rows["Briar"]["rank_delta"] == 1
    assert {row["champion_id"] for row in mid["rows"]} == {
        "riot:champion:266",
        "riot:champion:233",
    }
    assert all(row["tier_bucket"] in TIER_BUCKETS for row in mid["rows"])
    assert all(row["counterability_status"] == "unavailable" for row in mid["rows"])


def test_matchup_shape_assigns_distinct_blind_and_counter_tiers() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "champion": f"Champion {index}",
                "rating": 1600.0 - index,
                "counterability_status": "available",
                    "blind_score_pp": 12.0 - index,
                    "counter_score": 12.0 - index,
                "countered_opponent_count": 12 - index,
                "countered_opponent_share": (12 - index) / 12,
            }
        )

    _assign_tier_buckets(rows)

    assigned = {row["tier_bucket"] for row in rows}
    assert {"Z Blind", "Z Counter", "S Blind", "S Counter"}.issubset(assigned)
    assert all(row["tier_bucket"] in TIER_BUCKETS for row in rows)


def test_joint_fit_uses_identified_coordinates_and_legal_opponents() -> None:
    rng = np.random.default_rng(20260808)
    champions = [f"Champion {index}" for index in range(6)]
    states = {
        role: {champion: ChampState(appearances=480) for champion in champions}
        for role in ("top", "jungle", "mid", "bot", "support")
    }
    strength = np.linspace(0.35, -0.35, len(champions))
    observations = []
    reference_date = pd.Timestamp("2026-08-08T00:00:00Z")
    for map_index in range(480):
        roles = {}
        linear_predictor = 0.0
        for role_index, role in enumerate(states):
            blue_i, red_i = rng.choice(len(champions), size=2, replace=False)
            roles[role] = {
                "blue_champion": champions[int(blue_i)],
                "red_champion": champions[int(red_i)],
            }
            linear_predictor += strength[int(blue_i)] - strength[int(red_i)]
        observations.append(
            {
                "outcome": int(rng.random() < 1.0 / (1.0 + np.exp(-linear_predictor))),
                "team_logit": 0.0,
                "date": reference_date - pd.Timedelta(days=map_index // 12),
                "series_id": f"series-{map_index // 2}",
                "roles": roles,
            }
        )

    fit = _fit_joint_scope(
        states,
        observations,
        min_appearances=1,
        reference_date=reference_date,
        scope_id="synthetic:tier1",
    )

    for role in states:
        role_fit = fit[role]
        assert role_fit["design"]["fit_coordinates"] == "orthonormal reduced contrasts"
        assert role_fit["design"]["role_location_gauge"] == "sum_to_zero_per_connected_component"
        ratings = [row["rating"] for row in role_fit["rows"]]
        assert np.mean(ratings) == pytest.approx(1500.0, abs=1e-3)
        for row in role_fit["rows"]:
            assert len(row["legal_opponents"]) == 5
            assert row["champion"] not in {opponent["champion"] for opponent in row["legal_opponents"]}
            assert row["legal_opponent_distribution_sha256"]
