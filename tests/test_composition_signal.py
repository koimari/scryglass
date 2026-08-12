from __future__ import annotations

from copy import deepcopy

import pandas as pd

from lol_kills.research.composition_signal import (
    FittedCompositionModel,
    _feature_names,
    build_composition_games,
    evaluate_composition_signal,
    public_signal_for_game,
    score_games_temporally,
)


ROLES = ("top", "jng", "mid", "bot", "sup")


def _rows(game_id: str, date: str, result: int, *, suffix: str = "") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side, team, side_result in (("Blue", "Blue Team", result), ("Red", "Red Team", 1 - result)):
        for index, role in enumerate(ROLES):
            rows.append(
                {
                    "game_uid": game_id,
                    "date": date,
                    "league": "LCS",
                    "patch": "16.15",
                    "side": side,
                    "teamname": team,
                    "playername": f"{team}-{role}-{game_id}",
                    "position": role,
                    "champion": f"{side}-{role}-{suffix or 'champion'}-{index}",
                    "result": side_result,
                }
            )
    return rows


def _games(*rows: list[dict[str, object]]) -> list[dict[str, object]]:
    frame = pd.DataFrame([row for group in rows for row in group])
    strength = pd.DataFrame(
        [
            {"game_uid": game_id, "mu_diff": 20.0, "sigma_pair": 50.0}
            for game_id in frame["game_uid"].unique()
        ]
    )
    return build_composition_games(frame, strength_features=strength)


def test_target_result_does_not_change_its_composition_signal() -> None:
    training = [_rows("old-1", "2026-01-01", 1), _rows("old-2", "2026-01-02", 0)]
    target = _rows("target", "2026-01-03", 1)
    changed_target = _rows("target", "2026-01-03", 0)
    first = score_games_temporally(
        _games(*training, target),
        target_game_ids={"target"},
        min_training_games=2,
        min_support_games=1,
    )
    second = score_games_temporally(
        _games(*training, changed_target),
        target_game_ids={"target"},
        min_training_games=2,
        min_support_games=1,
    )
    assert first.signals["target"] == second.signals["target"]


def test_same_date_outcome_does_not_enter_target_fit() -> None:
    training = [_rows("old-1", "2026-01-01", 1), _rows("old-2", "2026-01-02", 0)]
    same_date = _rows("same-date", "2026-01-03T01:00:00Z", 1)
    target = _rows("target", "2026-01-03T20:00:00Z", 0)
    changed_same_date = _rows("same-date", "2026-01-03T01:00:00Z", 0)
    first = score_games_temporally(
        _games(*training, same_date, target),
        target_game_ids={"target"},
        min_training_games=2,
        min_support_games=1,
    )
    second = score_games_temporally(
        _games(*training, changed_same_date, target),
        target_game_ids={"target"},
        min_training_games=2,
        min_support_games=1,
    )
    assert first.signals["target"] == second.signals["target"]
    assert first.signals["target"]["fit_through"] < "2026-01-03T00:00:00Z"


def test_cache_reuses_unchanged_training_block_when_later_games_arrive(tmp_path) -> None:
    initial = _games(
        _rows("old-1", "2026-01-01", 1),
        _rows("old-2", "2026-01-02", 0),
        _rows("target", "2026-01-03", 1),
    )
    first = score_games_temporally(
        initial,
        target_game_ids={"target"},
        cache_dir=tmp_path,
        source_digest="source-a",
        worker_commit="worker-a",
        min_training_games=2,
        min_support_games=1,
    )
    appended = _games(
        _rows("old-1", "2026-01-01", 1),
        _rows("old-2", "2026-01-02", 0),
        _rows("target", "2026-01-03", 1),
        _rows("future", "2026-01-04", 0),
    )
    second = score_games_temporally(
        appended,
        target_game_ids={"target"},
        cache_dir=tmp_path,
        source_digest="source-b",
        worker_commit="worker-a",
        min_training_games=2,
        min_support_games=1,
    )

    assert second.audit["cache_hits"] == 1
    assert second.signals["target"] == first.signals["target"]

    corrected = _games(
        _rows("old-1", "2026-01-01", 0),
        _rows("old-2", "2026-01-02", 0),
        _rows("target", "2026-01-03", 1),
    )
    third = score_games_temporally(
        corrected,
        target_game_ids={"target"},
        cache_dir=tmp_path,
        source_digest="source-c",
        worker_commit="worker-a",
        min_training_games=2,
        min_support_games=1,
    )
    assert third.audit["cache_hits"] == 0


def test_incomplete_roles_and_duplicate_champions_are_unavailable() -> None:
    game = _games(_rows("game", "2026-01-01", 1))[0]
    invalid = deepcopy(game)
    invalid["red"].pop("sup")
    assert public_signal_for_game(invalid, None)["status"] == "unavailable"
    assert "five roles" in public_signal_for_game(invalid, None)["reason"]

    duplicate = deepcopy(game)
    duplicate["red"]["sup"]["champion"] = duplicate["blue"]["top"]["champion"]
    assert public_signal_for_game(duplicate, None)["status"] == "unavailable"
    assert "unique champions" in public_signal_for_game(duplicate, None)["reason"]


def test_duplicate_player_identity_is_rejected_from_training() -> None:
    rows = _rows("game", "2026-01-01", 1)
    rows[-1]["playername"] = rows[0]["playername"]
    assert _games(rows) == []


def test_missing_identity_values_are_rejected_from_training() -> None:
    rows = _rows("game", "2026-01-01", 1)
    rows[0]["playername"] = None
    rows[1]["teamname"] = None
    rows[2]["champion"] = None
    assert _games(rows) == []


def test_thin_role_champion_history_is_limited() -> None:
    games = _games(_rows("old-1", "2026-01-01", 1), _rows("old-2", "2026-01-02", 0), _rows("target", "2026-01-03", 1))
    result = score_games_temporally(
        games,
        target_game_ids={"target"},
        min_training_games=2,
        min_support_games=3,
    )
    signal = result.signals["target"]
    assert signal["status"] == "limited"
    assert signal["blue"]["signal"] is None
    assert any(pick["evidence_status"] == "limited" for pick in signal["picks"])


def test_public_contributions_are_relative_to_the_picking_side() -> None:
    game = _games(_rows("game", "2026-01-01", 1))[0]
    names = _feature_names([game])
    coefficients = [0.0] * len(names)
    coefficients[names.index("draft|top|Blue-top-champion-0")] = 0.5
    coefficients[names.index("draft|top|Red-top-champion-0")] = -0.25
    model = FittedCompositionModel(
        model_version="composition-signal-v1",
        fit_through="2025-12-31T00:00:00Z",
        feature_names=tuple(names),
        coefficients=tuple(coefficients),
        intercept=0.0,
        support={f"{role}|{champion}": 40 for side in ("blue", "red") for role, champion in ((role, game[side][role]["champion"]) for role in ROLES)},
        train_games=100,
    )
    signal = public_signal_for_game(game, model, min_support_games=40)
    blue_top = next(pick for pick in signal["picks"] if pick["side"] == "Blue" and pick["role"] == "top")
    red_top = next(pick for pick in signal["picks"] if pick["side"] == "Red" and pick["role"] == "top")
    assert blue_top["contribution"] == 0.5
    assert red_top["contribution"] == -0.25
    assert signal["blue"]["signal"] == 0.5
    assert signal["red"]["signal"] == -0.25
    assert "coefficients" not in signal


def test_evaluator_writes_four_windows_and_keeps_team_history_diagnostic() -> None:
    games = _games(*[_rows(f"game-{index}", f"2026-01-{index + 1:02d}", index % 2) for index in range(30)])
    report = evaluate_composition_signal(
        games,
        source_hash="source-hash",
        worker_commit="worker-commit",
        bootstrap_reps=10,
        min_training_games=4,
    )

    assert len(report["holdout_windows"]) == 4
    assert report["source_hash"] == "source-hash"
    assert report["worker_commit"] == "worker-commit"
    assert "brier" in report["holdout_windows"][0]["baseline"]
    assert "log_loss" in report["holdout_windows"][0]["draft_augmented"]
    assert report["team_history_diagnostic"]["included_in_team_rating"] is False
