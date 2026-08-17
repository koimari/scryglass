from __future__ import annotations

from copy import deepcopy

import pandas as pd

from lol_kills.research.composition_signal import (
    ATOM_CORPUS_PATH,
    DESCRIPTIVE_EXCLUDED_TERMS,
    DESCRIPTIVE_MODEL_VERSION,
    FittedCompositionModel,
    REGULARIZATION_C,
    _apply_oof_recalibration,
    _feature_names,
    _fit_model,
    _history_features,
    _match_delta_intervals,
    _matrix,
    _recalibrate_history_probabilities,
    _select_history_regularization,
    _select_regularization,
    build_composition_games,
    evaluate_composition_signal,
    public_signal_for_game,
    score_games_temporally,
    _composition_code_digest,
    _descriptive_feature_names,
)

import numpy as np
from sklearn.linear_model import LogisticRegression


ROLES = ("top", "jng", "mid", "bot", "sup")


def test_composition_digest_changes_when_atom_corpus_changes(monkeypatch) -> None:
    before = _composition_code_digest()
    original_read_bytes = type(ATOM_CORPUS_PATH).read_bytes

    def changed_read_bytes(path):
        content = original_read_bytes(path)
        if path == ATOM_CORPUS_PATH:
            return content + b"\nchanged for digest test"
        return content

    monkeypatch.setattr(type(ATOM_CORPUS_PATH), "read_bytes", changed_read_bytes)
    assert _composition_code_digest() != before


def _rows(game_id: str, date: str, result: int, *, suffix: str = "") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side, team, side_result in (("Blue", "Blue Team", result), ("Red", "Red Team", 1 - result)):
        for index, role in enumerate(ROLES):
            rows.append(
                {
                    "game_uid": game_id,
                    "date": date,
                    "league": "LCS",
                    "competition_tier": "tier1",
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


def test_composition_game_keeps_its_release_scope() -> None:
    game = build_composition_games(pd.DataFrame(_rows("scope", "2026-01-01", 1)))[0]

    assert game["league"] == "LCS"
    assert game["competition_tier"] == "tier1"


def _rows_with_extras(game_id: str, date: str, result: int) -> list[dict[str, object]]:
    """Rows with ban columns and per-player stats (round-5 frontier inputs)."""
    rows = _rows(game_id, date, result)
    for index, row in enumerate(rows):
        row["kills"] = 1 + index % 5
        row["deaths"] = index % 3
        row["damageshare"] = 0.1 + (index % 10) / 100
        row["cspm"] = 6.0 + (index % 20) / 10
        row["visionscore"] = 20.0 + (index % 40)
    for side in ("Blue", "Red"):
        first = next(row for row in rows if row["side"] == side)
        for slot in range(1, 6):
            first[f"ban{slot}"] = f"Ban{side}{slot}"
    return rows


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


def test_public_descriptive_scorer_has_no_context_controls() -> None:
    training = [
        *_rows("old-1", "2026-01-01", 1),
        *_rows("old-2", "2026-01-02", 0),
    ]
    games = build_composition_games(pd.DataFrame(training))
    names = _descriptive_feature_names(games)
    assert names
    assert all(name.startswith(("draft|", "atom|")) for name in names)
    assert not any(name.startswith(("control|", "league|", "patch|")) for name in names)
    assert {
        "pre_game_team_strength_gap",
        "rating_uncertainty",
        "momentum",
        "live_state",
        "outcome",
        "r9e_state_space",
    }.issubset(DESCRIPTIVE_EXCLUDED_TERMS)

    result = score_games_temporally(
        games,
        target_game_ids={"old-2"},
        min_training_games=1,
        min_support_games=1,
        composition_only=True,
    )
    signal = result.signals["old-2"]
    assert result.audit["estimand"] == "composition_only"
    assert result.audit["model_version"] == DESCRIPTIVE_MODEL_VERSION
    assert signal["estimand"] == "composition_only"
    assert "probability" not in signal


def test_public_descriptive_score_is_invariant_to_strength_and_patch_controls() -> None:
    games = build_composition_games(
        pd.DataFrame(
            [
                *_rows("old-1", "2026-01-01", 1),
                *_rows("old-2", "2026-01-02", 0),
                *_rows("target", "2026-01-03", 1),
            ]
        )
    )
    first = score_games_temporally(
        games,
        target_game_ids={"target"},
        min_training_games=2,
        min_support_games=1,
        composition_only=True,
    )
    changed = deepcopy(games)
    changed_target = next(game for game in changed if game["game_uid"] == "target")
    changed_target["mu_diff"] = 9999.0
    changed_target["sigma_pair"] = 0.0
    changed_target["league"] = "OTHER"
    changed_target["patch"] = "99.99"
    changed_target["y"] = 0
    second = score_games_temporally(
        changed,
        target_game_ids={"target"},
        min_training_games=2,
        min_support_games=1,
        composition_only=True,
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


def test_per_role_mastery_features_use_strictly_prior_experience() -> None:
    players: list[tuple[str, str, str, str]] = []
    for side, team in (("Blue", "Blue Team"), ("Red", "Red Team")):
        for role in ROLES:
            players.append((f"{team}-{role}", side, role, f"{side}-{role}-champ"))

    def rows(game_id: str, date: str, result: int, *, red_swaps: bool = False) -> list[dict[str, object]]:
        built: list[dict[str, object]] = []
        for playername, side, role, champion in players:
            champion = f"{side}-{role}-other" if red_swaps and side == "Red" else champion
            built.append(
                {
                    "game_uid": game_id,
                    "date": date,
                    "league": "LCS",
                    "patch": "16.15",
                    "side": side,
                    "teamname": "Blue Team" if side == "Blue" else "Red Team",
                    "playername": playername,
                    "position": role,
                    "champion": champion,
                    "result": result if side == "Blue" else 1 - result,
                }
            )
        return built

    frame = pd.DataFrame(
        [
            *rows("prior-1", "2025-12-01", 1),
            *rows("prior-2", "2025-12-02", 0),
            *rows("prior-3", "2025-12-03", 1, red_swaps=True),
            *rows("target", "2026-01-01", 1),
        ]
    )
    games = build_composition_games(frame, strength_features=None)
    target = next(game for game in games if game["game_uid"] == "target")
    names = _feature_names([target])
    for role in ROLES:
        assert f"draft|exp|{role}" in names
    # Blue players kept their champion across three prior games; red players
    # swapped in the last prior game, so blue mastery exceeds red by one game.
    for side, expected in (("blue", 3), ("red", 2)):
        for role in ROLES:
            assert target[side][role]["experience"] == expected
    matrix = _matrix([target], names, include_draft=True)
    dense = matrix.toarray()[0]
    index = {name: position for position, name in enumerate(names)}
    for role in ROLES:
        assert abs(dense[index[f"draft|exp|{role}"]] - 1.0 / 50.0) < 1e-9
    baseline = _matrix([target], names, include_draft=False)
    assert abs(baseline.toarray()[0, index[f"draft|exp|top"]] - 0.0) < 1e-12


def _dated_rows(count: int) -> list[list[dict[str, object]]]:
    start = pd.Timestamp("2026-01-01T12:00:00Z")
    return [
        _rows(
            f"game-{index}",
            (start + pd.Timedelta(days=index)).isoformat(),
            index % 2,
        )
        for index in range(count)
    ]


def test_select_history_regularization_picks_a_candidate_on_internal_split() -> None:
    games = _games(*_dated_rows(120))
    names = _feature_names(games)
    model = _fit_model(games, names=names, include_draft=True)
    assert model is not None
    history = _history_features(games, model)
    train_x = np.column_stack(
        [
            _matrix(games, model.feature_names, include_draft=True) @ np.asarray(model.coefficients),
            history,
        ]
    )
    chosen = _select_history_regularization(
        games,
        train_x,
        candidates=(0.03, 0.1, 0.3, 1.0, 3.0),
        internal_fraction=0.15,
        min_training_games=4,
    )
    assert chosen in (0.03, 0.1, 0.3, 1.0, 3.0)


def test_select_history_regularization_falls_back_on_thin_data() -> None:
    games = _games(*_dated_rows(100))
    names = _feature_names(games)
    model = _fit_model(games, names=names, include_draft=True)
    assert model is not None
    history = _history_features(games, model)
    train_x = np.column_stack(
        [
            _matrix(games, model.feature_names, include_draft=True) @ np.asarray(model.coefficients),
            history,
        ]
    )
    assert (
        _select_history_regularization(
            games,
            train_x,
            candidates=(0.1, 1.0),
            internal_fraction=0.15,
            min_training_games=100,
        )
        == 1.0
    )


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
    assert "probabilities" in report["holdout_windows"][0]["draft_augmented"]
    assert "outcomes" in report["holdout_windows"][0]["draft_augmented"]
    assert "brier_delta" in report["pooled_bootstrap"]
    assert "brier_delta" in report["team_history_bootstrap"]


def test_select_regularization_falls_back_on_thin_data() -> None:
    games = _games(*[_rows(f"game-{index}", f"2026-01-{index + 1:02d}", index % 2) for index in range(10)])
    selected = _select_regularization(
        games,
        names=_feature_names(games),
        candidates=(0.01, 0.1, 1.0),
        internal_fraction=0.15,
        min_training_games=4,
        worker_commit=None,
    )
    assert selected == REGULARIZATION_C


def test_apply_oof_recalibration_falls_back_on_tiny_training_fold() -> None:
    games = _games(*[_rows(f"game-{index}", f"2026-01-{index + 1:02d}", index % 2) for index in range(12)])
    train = games[:6]
    validation = games[6:8]
    names = _feature_names(train)
    baseline = _fit_model(train, names=names, include_draft=False, min_training_games=2)
    draft = _fit_model(train, names=names, include_draft=True, min_training_games=2)
    assert baseline is not None and draft is not None
    base_probs, draft_probs = _apply_oof_recalibration(train, validation, baseline, draft)
    assert len(base_probs) == len(validation) == len(draft_probs)
    assert all(0.0 < value < 1.0 for value in draft_probs)


def test_match_delta_intervals_support_a_consistently_better_candidate() -> None:
    # All-positive outcomes keep every bootstrap resample strictly negative
    # for both brier and log-loss deltas (the interval is built from per-match
    # paired deltas; the log-loss delta term is the positive-class
    # contribution, faithful to the registered evaluation).
    outcomes = [1, 1, 1, 1, 1, 1]
    baseline = [0.5] * len(outcomes)
    draft = [0.95] * len(outcomes)
    windows = [
        {
            "baseline": {"probabilities": baseline, "outcomes": outcomes},
            "draft_augmented": {"probabilities": draft, "outcomes": outcomes},
        }
    ]
    intervals = _match_delta_intervals(
        windows, reps=200, seed=7, label="draft_vs_baseline"
    )
    assert intervals["brier_delta"]["upper"] < 0
    assert intervals["log_loss_delta"]["upper"] < 0


def test_history_features_are_strictly_lagged_and_shrunk() -> None:
    games = _games(
        _rows("game-1", "2026-01-01", 1),
        _rows("game-2", "2026-01-02", 0),
        _rows("game-3", "2026-01-03", 1),
    )
    names = _feature_names(games)
    coefficients = [0.0] * len(names)
    coefficients[names.index("draft|top|Blue-top-champion-0")] = 0.5
    model = FittedCompositionModel(
        model_version="composition-signal-v1",
        fit_through="2025-12-31T00:00:00Z",
        feature_names=tuple(names),
        coefficients=tuple(coefficients),
        intercept=0.0,
        support={},
        train_games=len(games),
    )
    features = _history_features(games, model)
    assert features.shape == (3, 3)
    # Strictly prior: the first game has no history for either team.
    assert features[0][0] == 0.0 and features[0][1] == 0.0
    # The second game sees the count-shrunk prior draft signal of game 1.
    assert abs(features[1][0] - (0.5 / 6.0)) < 1e-9


def test_recalibrate_history_probabilities_falls_back_on_tiny_data() -> None:
    games = _games(*[_rows(f"game-{index}", f"2026-01-{index + 1:02d}", index % 2) for index in range(14)])
    train = games[:10]
    validation = games[10:12]
    names = _feature_names(train)
    draft = _fit_model(train, names=names, include_draft=True, min_training_games=2)
    assert draft is not None
    history = _history_features(train, draft)
    history_model = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=461)
    train_x = np.column_stack(
        [_matrix(train, draft.feature_names, include_draft=True) @ np.asarray(draft.coefficients), history]
    )
    history_model.fit(train_x, [int(game["y"]) for game in train])
    validation_history = _history_features(validation, draft)
    validation_x = np.column_stack(
        [_matrix(validation, draft.feature_names, include_draft=True) @ np.asarray(draft.coefficients), validation_history]
    )
    probabilities = _recalibrate_history_probabilities(
        train, validation, draft, train_x, validation_x, history_model, shrink=0.5
    )
    assert len(probabilities) == len(validation)
    assert all(0.0 < value < 1.0 for value in probabilities)


def test_games_carry_bans_and_player_stats() -> None:
    games = _games(*[_rows_with_extras(f"game-{index}", f"2026-01-{index + 1:02d}", index % 2) for index in range(3)])
    game = games[0]
    assert game["bans"]["blue"] == ["BanBlue1", "BanBlue2", "BanBlue3", "BanBlue4", "BanBlue5"]
    assert game["bans"]["red"] == ["BanRed1", "BanRed2", "BanRed3", "BanRed4", "BanRed5"]
    assert len(game["player_stats"]) == 10
    sample = next(iter(game["player_stats"].values()))
    assert {"kills", "deaths", "damageshare", "cspm", "visionscore"} <= set(sample)


def test_exp_rows_log1p_are_strictly_prior() -> None:
    from lol_kills.research.composition_signal import _exp_rows_log1p
    games = _games(*[_rows(f"game-{index}", f"2026-01-{index + 1:02d}", index % 2) for index in range(6)])
    rows = _exp_rows_log1p(games)
    assert rows.shape == (6, 5)
    assert np.allclose(rows[0], 0.0)


def test_ban_v2_builder_shape_and_strictly_prior() -> None:
    from lol_kills.research.composition_signal import _bans_v2_feature_builder
    games = _games(*[_rows_with_extras(f"game-{index}", f"2026-01-{index + 1:02d}", index % 2) for index in range(6)])
    builder = _bans_v2_feature_builder(games[:4], games[4:], None)
    rows = builder(games)
    assert rows.shape == (6, 8)
    assert np.allclose(rows[0], 0.0)


def test_frontier_is_strictly_prior_and_shaped() -> None:
    from lol_kills.research.composition_signal import (
        _build_frontier, _frontier_names, _frontier_rows, _depth2_keys, _depth3_keys, _depth4_keys, _SS_KEYS,
    )
    games = _games(*[_rows_with_extras(f"game-{index}", f"2026-01-{index + 1:02d}", index % 2) for index in range(6)])
    ordered = sorted(games, key=lambda game: (game["date"], game["game_uid"]))
    _build_frontier(ordered)
    names = _frontier_names()
    # 67 round-5+L5 + 23 depth-2 + 17 depth-3 + 12 depth-4 descriptors + 4 state-space + 6 L7 + 30 per-role L7
    assert len(names) == 159
    assert len(_depth2_keys()) == 23
    assert len(_depth3_keys()) == 17
    assert len(_depth4_keys()) == 12
    assert len(_SS_KEYS) == 4
    assert all(name.startswith("d4_") for name in names if name.startswith("d4_"))
    assert all(name.startswith("ss_") for name in names if name.startswith("ss_"))
    rows = _frontier_rows(ordered)
    assert rows.shape == (6, 159)
    # Fake champion names -> no corpus/depth2/L7 data; strictly-prior features zero on game 1
    assert np.allclose(rows[0], 0.0)


def test_depth4_corpus_has_no_cooldown_sentinels() -> None:
    import json
    from pathlib import Path
    payload = json.loads(Path("data/lol/v2/champions/atom-corpus-aggregate-v4.json").read_text(encoding="utf-8"))
    offenders = []
    for slug, entry in payload["champions"].items():
        for key, value in entry.items():
            if not isinstance(value, (int, float)) or value < 0 or value > 600:
                offenders.append((slug, key, value))
    assert not offenders, f"d4 corpus sentinels: {offenders[:5]}"


def test_production_model_uses_atomized_descriptors() -> None:
    from lol_kills.research.composition_signal import (
        _feature_names, _atom_term_keys, _atom_desc_value, FittedCompositionModel,
    )
    game = {
        "league": "LEC", "patch": "25.9",
        "blue": {role: {"champion": champ} for role, champ in
                 (("top", "Wukong"), ("jng", "Lee Sin"), ("mid", "Ahri"),
                  ("bot", "Ashe"), ("sup", "Renata Glasc"))},
        "red": {role: {"champion": champ} for role, champ in
                (("top", "Nunu"), ("jng", "Viego"), ("mid", "Orianna"),
                 ("bot", "Jinx"), ("sup", "Rell"))},
    }
    names = _feature_names([game])
    atom_names = [name for name in names if name.startswith("atom|")]
    assert len(atom_names) == 5 * len(_atom_term_keys())
    assert "atom|top|d2_ad_ratio" in names
    # alias-aware lookups: Wukong/Renata Glasc/Nunu & Willump resolve to their
    # corpus keys and must not silently zero every descriptor
    assert _atom_desc_value("Wukong", "d4_chain_len") != 0.0
    assert _atom_desc_value("Renata Glasc", "d4_cast_share") != 0.0
    assert _atom_desc_value("Nunu", "d3_tempo_burst") != 0.0
    # pick_contribution extends the plain champion coefficient with the
    # coefficient-weighted descriptor terms
    model = FittedCompositionModel(
        model_version="composition-signal-v3",
        fit_through="2026-08-01",
        feature_names=tuple(names),
        coefficients=tuple(0.0 for _ in names),
        intercept=0.0,
        support={},
        train_games=100,
    )
    assert model.pick_contribution("top", "Ahri") == 0.0
    idx = names.index("atom|top|d2_burst")
    rich = FittedCompositionModel(
        model_version="composition-signal-v3",
        fit_through="2026-08-01",
        feature_names=tuple(names),
        coefficients=tuple(0.5 if i == idx else 0.0 for i in range(len(names))),
        intercept=0.0,
        support={},
        train_games=100,
    )
    ahri_contribution = rich.pick_contribution("top", "Ahri")
    assert ahri_contribution != 0.0
    assert ahri_contribution == 0.5 * _atom_desc_value("Ahri", "d2_burst")


def test_corpus_game_features_shape_and_unknown_champion_zeros() -> None:
    from lol_kills.research.composition_signal import _corpus_game_features
    game = {
        "blue": {role: {"champion": f"UnknownBlue{role}"} for role in ROLES},
        "red": {role: {"champion": f"UnknownRed{role}"} for role in ROLES},
    }
    row = _corpus_game_features(game)
    assert row.shape == (49,)
    assert np.allclose(row, 0.0)


def test_production_style_recalibrate_falls_back_on_tiny_data() -> None:
    from lol_kills.research.composition_signal import _production_style_recalibrate
    games = _games(*[_rows(f"game-{index}", f"2026-01-{index + 1:02d}", index % 2) for index in range(8)])
    raw = np.asarray([0.4, 0.6, 0.5, 0.45, 0.55, 0.48, 0.52, 0.51])
    result = _production_style_recalibrate(games[:6], games[6:], None, lambda: LogisticRegression(), raw)
    assert np.allclose(result, np.clip(raw, 1e-5, 1 - 1e-5))


def test_evaluator_accepts_bans_and_stats_games() -> None:
    games = _games(*[_rows_with_extras(f"game-{index}", f"2026-01-{index + 1:02d}", index % 2) for index in range(30)])
    report = evaluate_composition_signal(
        games,
        source_hash="source-hash",
        worker_commit="worker-commit",
        bootstrap_reps=10,
        min_training_games=4,
    )
    assert len(report["holdout_windows"]) == 4
    assert "draft_plus_team_history" in report["holdout_windows"][0]
    assert "brier_delta" in report["team_history_bootstrap"]


def _real_rows(game_id: str, date: str, result: int, *, pool: list[str], rotation: int = 0) -> list[dict[str, object]]:
    """Rows with real champion names so the atom-corpus prior can fire."""
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
                    "champion": pool[(rotation + index) % 10 if side == "Blue" else (rotation + index + 5) % 10],
                    "result": side_result,
                }
            )
    return rows


def test_unsupported_pick_gets_atom_estimate() -> None:
    pool = ["Ahri", "Lux", "Zed", "Yasuo", "Jinx", "Leona", "LeeSin", "Thresh", "Azir", "Rakan"]
    training = [
        _real_rows(
            f"old-{index}",
            f"2026-{1 + index // 28:02d}-{1 + index % 28:02d}",
            index % 2,
            pool=pool,
            rotation=index,
        )
        for index in range(160)
    ]
    target = _real_rows("target", "2026-07-01", 1, pool=["Syndra", "Ahri", "Lux", "Zed", "Yasuo", "Jinx", "Leona", "LeeSin", "Thresh", "Azir"])
    result = score_games_temporally(
        _games(*training, target),
        target_game_ids={"target"},
        min_training_games=4,
        min_support_games=40,
    )
    signal = result.signals["target"]
    syndra = next(pick for pick in signal["picks"] if pick["champion"] == "Syndra")
    assert syndra["evidence_status"] == "atom_estimate"
    assert syndra["contribution"] is not None
    assert signal["status"] == "available"
    assert signal["blue"]["signal"] is not None and signal["red"]["signal"] is not None


def _fixed_players_rows(game_id: str, date: str, result: int, *, champs: list[str]) -> list[dict[str, object]]:
    """Rows with FIXED player/team names (L7 profiles persist across games) and real champions."""
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
                    "playername": f"pro-{side.lower()}-{role}",
                    "position": role,
                    "champion": champs[index if side == "Blue" else index + 5],
                    "result": side_result,
                }
            )
    return rows


def test_depth2_rows_are_static_blue_minus_red() -> None:
    from lol_kills.research.composition_signal import _build_frontier, _frontier_names, _frontier_rows, _champion_depth2, _depth2_keys
    pool = ["Ahri", "Lux", "Zed", "Yasuo", "Jinx", "Leona", "LeeSin", "Thresh", "Azir", "Rakan"]
    games = _games(*[_fixed_players_rows(f"g{i}", f"2026-01-{i+1:02d}", i % 2, champs=pool) for i in range(4)])
    ordered = sorted(games, key=lambda game: (game["date"], game["game_uid"]))
    _build_frontier(ordered)
    names = _frontier_names()
    rows = _frontier_rows(ordered)
    start = names.index("d2_" + sorted(_depth2_keys())[0])
    end = start + len(_depth2_keys())
    # blue-minus-red means of the 5 picks' depth2 descriptors; identical across games (static)
    first = rows[0][start:end]
    for game, row in zip(ordered, rows):
        blue = [_champion_depth2(game["blue"][r]["champion"]) for r in ROLES]
        red = [_champion_depth2(game["red"][r]["champion"]) for r in ROLES]
        for key_index, key in enumerate(sorted(_depth2_keys())):
            b = float(np.mean([e.get(key, 0.0) for e in blue]))
            r = float(np.mean([e.get(key, 0.0) for e in red]))
            assert abs(row[start + key_index] - (b - r)) < 1e-9, key
    assert np.allclose(rows[:, start:end], first)  # static per champion -> identical rows


def test_l7_is_strictly_prior_to_own_outcome() -> None:
    from lol_kills.research.composition_signal import _build_frontier, _frontier_names, _frontier_rows
    pool = ["Ahri", "Lux", "Zed", "Yasuo", "Jinx", "Leona", "LeeSin", "Thresh", "Azir", "Rakan"]
    champs_a = pool
    champs_b = ["Syndra", "Ahri", "Lux", "Zed", "Yasuo", "Jinx", "Leona", "LeeSin", "Thresh", "Azir"]
    # scenario A: g2 outcome 1; scenario B: g2 outcome 0 (g1 identical)
    games_a = _games(
        _fixed_players_rows("g1", "2026-01-01", 1, champs=champs_a),
        _fixed_players_rows("g2", "2026-01-02", 1, champs=champs_b),
        _fixed_players_rows("g3", "2026-01-03", 1, champs=champs_a),
    )
    games_b = _games(
        _fixed_players_rows("g1", "2026-01-01", 1, champs=champs_a),
        _fixed_players_rows("g2", "2026-01-02", 0, champs=champs_b),
        _fixed_players_rows("g3", "2026-01-03", 0, champs=champs_a),
    )
    _build_frontier(sorted(games_a, key=lambda g: (g["date"], g["game_uid"])))
    names_a = _frontier_names()
    rows_a = _frontier_rows(sorted(games_a, key=lambda g: (g["date"], g["game_uid"])))
    l7_start = names_a.index("l7_ccm")
    _build_frontier(sorted(games_b, key=lambda g: (g["date"], g["game_uid"])))
    rows_b = _frontier_rows(sorted(games_b, key=lambda g: (g["date"], g["game_uid"])))
    # g2's L7 feature must NOT depend on g2's own outcome (strictly prior)
    assert np.allclose(rows_a[1][l7_start:l7_start + 6], rows_b[1][l7_start:l7_start + 6])
    # g3's L7 feature SHOULD reflect g2's outcome (profile updated after g2)
    assert not np.allclose(rows_a[2][l7_start:l7_start + 6], rows_b[2][l7_start:l7_start + 6])


def test_atom_estimate_picks_validate_in_an_available_signal() -> None:
    from lol_kills.research.composition_signal import validate_public_signal
    signal = {
        "schema_version": "scryglass:composition-signal:v1",
        "status": "available",
        "model_version": "composition-signal-v2",
        "fit_through": "2026-04-23T23:12:11Z",
        "blue": {"signal": 0.1, "prior_role_games": 500},
        "red": {"signal": -0.1, "prior_role_games": 500},
        "note": "test",
        "picks": [],
    }
    players = []
    for side in ("Blue", "Red"):
        for role in ROLES:
            players.append({"side": side, "role": role, "champion": f"Champ{side}{role}"})
    for side in ("Blue", "Red"):
        for role in ROLES:
            signal["picks"].append({
                "side": side,
                "role": role,
                "champion": f"Champ{side}{role}",
                "contribution": 0.02 if side == "Blue" else -0.02,
                "prior_role_games": 0 if (side, role) == ("Blue", "top") else 100,
                "evidence_status": "atom_estimate" if (side, role) == ("Blue", "top") else "available",
            })
    game = {
        "game_id": "game-x",
        "date": "2026-05-01T00:00:00Z",
        "players": players,
    }
    validate_public_signal(signal, game)
